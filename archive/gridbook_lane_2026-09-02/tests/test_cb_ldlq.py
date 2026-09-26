"""Production-contract tests for flag-gated CB LDLQ assignment."""
from __future__ import annotations

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant.cb_ldlq import fill_empty_expert_activation_rows
from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_ldlq_atoms import CBLDLQError
from prismaquant.cb_warm_state import CBWarmStateStore, build_warm_record
from prismaquant.measure_quant_cost import (
    _cb_cost_quantize_dequantize,
    cost_payload_provenance,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_serialization_stamps,
    cb_fields_for_context,
    cb_serialization_context_from_env,
    cb_serialization_context_stamp,
    validate_cb_assignment_serialization_stamps,
    validate_cb_serialization_context_stamp,
)


def _case(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(8, 256, generator=generator) * 0.25
    latent = torch.randn(48, 64, generator=generator)
    activations = torch.cat(
        (
            latent,
            0.9 * latent + 0.1 * torch.randn(48, 64, generator=generator),
            -0.8 * latent + 0.2 * torch.randn(48, 64, generator=generator),
            0.7 * latent + 0.3 * torch.randn(48, 64, generator=generator),
        ),
        dim=1,
    )
    return weight, activations, activations.square().mean(dim=0)


def _set_context_env(monkeypatch, *, ldlq: bool) -> None:
    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")
    monkeypatch.setenv("CB_SCALE_SWEEP", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ", "1" if ldlq else "0")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_TIER", "fast")


def test_empty_expert_rows_use_same_layer_routed_pool():
    first = torch.tensor([[1.0, 2.0]])
    third = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    rows, missing = fill_empty_expert_activation_rows(
        (first, torch.empty(0, 2), third),
        qname="model.layers.0.mlp.experts.down_proj",
    )
    assert missing == (1,)
    assert torch.equal(rows[0], first)
    assert torch.equal(rows[2], third)
    assert torch.equal(rows[1], torch.cat((first, third)))


def test_flag_off_is_byte_identical_to_the_existing_encoder(monkeypatch):
    weight, _activations, col_weights = _case()
    spec = fr.get_format("NVFP4_CB_K12")
    monkeypatch.delenv("PRISMAQUANT_CB_LDLQ", raising=False)
    assert cb_serialization_context_from_env().ldlq is False
    _set_context_env(monkeypatch, ldlq=False)
    context = cb_serialization_context_from_env(require_explicit=True)

    baseline, _ = cb.nvfp4_cb_pack(
        weight,
        12,
        grid="fp4",
        mode="product",
        col_weights=col_weights,
        scale_coding="two_tier",
        encode_tier="fast",
    )
    fields = cb_fields_for_context(
        spec,
        weight,
        context=context,
        col_weights=col_weights,
    )
    flagged_off = cb.nvfp4_cb_assemble_bytes(
        fields, 12, grid="fp4", mode="product"
    )

    assert context.ldlq is False
    assert torch.equal(flagged_off, baseline)


def test_flag_on_cost_render_is_deterministic_and_preserves_fitted_scales(
    monkeypatch,
):
    weight, activations, col_weights = _case(3)
    spec = fr.get_format("NVFP4_CB_K12")
    _set_context_env(monkeypatch, ldlq=True)

    first = _cb_cost_quantize_dequantize(
        spec,
        weight,
        col_weights=col_weights,
        activation_rows=activations,
    )
    second = _cb_cost_quantize_dequantize(
        spec,
        weight,
        col_weights=col_weights,
        activation_rows=activations,
    )
    plain_fields = cb.nvfp4_cb_fields(
        weight,
        12,
        col_weights=col_weights,
        scale_coding="two_tier",
        encode_tier="fast",
    )
    ldlq_fields = cb_fields_for_context(
        spec,
        weight,
        context=cb_serialization_context_from_env(require_explicit=True),
        col_weights=col_weights,
        activation_rows=activations,
    )

    assert torch.equal(first, second)
    assert torch.equal(ldlq_fields["scales"], plain_fields["scales"])
    assert torch.equal(ldlq_fields["scale_super"], plain_fields["scale_super"])
    assert torch.equal(ldlq_fields["scale_sub"], plain_fields["scale_sub"])


def test_cost_provenance_carries_ldlq_and_export_refuses_mismatch(monkeypatch):
    spec = fr.get_format("NVFP4_CB_K12")
    plain = CBSerializationContext.production(encode_tier="fast", ldlq=False)
    feedback = CBSerializationContext.production(encode_tier="fast", ldlq=True)
    _set_context_env(monkeypatch, ldlq=True)
    provenance = cost_payload_provenance([spec])

    assert provenance["cb_serialized_payload"]["ldlq"] is True
    try:
        validate_cb_serialization_context_stamp(
            cb_serialization_context_stamp(plain),
            feedback,
            where="export_nvfp4_cb test",
        )
    except ValueError as exc:
        assert "differs from allocator recipe" in str(exc)
        assert "'ldlq': False" in str(exc)
    else:
        raise AssertionError("export accepted a plain/LDLQ context mismatch")

    assignment = {"layers.0.proj": spec.name}
    shapes = {"layers.0.proj": (8, 256)}
    plain_stamps = cb_assignment_serialization_stamps(
        assignment,
        shapes,
        context=plain,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_cb_assignment_serialization_stamps(
            assignment,
            shapes,
            context=feedback,
            stamps=plain_stamps,
            where="export_nvfp4_cb test",
        )


def test_non_ldlq_warm_record_cold_falls_back_under_ldlq(tmp_path):
    weight, _activations, col_weights = _case(5)
    fmt = "NVFP4_CB_K12"
    plain = CBSerializationContext.production(encode_tier="fast", ldlq=False)
    feedback = CBSerializationContext.production(encode_tier="fast", ldlq=True)
    fields = cb.nvfp4_cb_fields(
        weight,
        12,
        col_weights=col_weights,
        scale_coding="two_tier",
        encode_tier="fast",
    )
    store = CBWarmStateStore(tmp_path)
    record = build_warm_record(
        qname="layers.0.proj",
        format_name=fmt,
        source_weight=weight,
        col_weights=col_weights,
        context=plain,
        fields=fields,
    )
    store.write(record)

    loaded = store.load_matching(
        qname="layers.0.proj",
        format_name=fmt,
        source_shape=list(weight.shape),
        source_digest=str(record.metadata["source_digest"]),
        col_weights_shape=list(col_weights.shape),
        col_weights_digest=str(record.metadata["col_weights_digest"]),
        context=feedback,
    )

    assert loaded is None


def test_gated_ldlq_honest_holdout_no_harm_regression(monkeypatch):
    """Honest held-out no-harm regression for the former in-sample case.

    The pilot's synthetic "known better" asserted in-sample
    feedback_sse/plain_sse < 0.5, which the pre-2026-08-08 gate could not
    fail (measured anti-correlation 20x\u201348.5x, nvfp4_cb_formats.py:1777).
    The production gate splits activation rows deterministically by
    content (sha256 of row bytes, nvfp4_cb_formats.py:2311) into FIT/CERT,
    fits LDLQ on FIT only, scores the exact shipped candidate on CERT,
    and keeps LDLQ only when CERT strictly improves (ties \u2192 raw).
    This test asserts both branches with explicit holdout control.
    """
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_GATE", "holdout")
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(8, 256, generator=generator) * 0.25
    col_weights = torch.rand(256, generator=generator) + 0.05
    activations = torch.randn(32, 256, generator=generator)
    # Content-keyed split is deterministic and not a coin flip (MIN_ROWS 16).
    fit, cert = cb._ldlq_holdout_split(activations)
    assert fit is not None and cert is not None
    fit2, cert2 = cb._ldlq_holdout_split(activations)
    assert torch.equal(fit, fit2) and torch.equal(cert, cert2)
    # The same bytes with a one-bit flip must give a different split,
    # proving content-keying (not identity).
    flipped = activations.clone()
    flipped[0, 0] += 1e-6
    fit3, _ = cb._ldlq_holdout_split(flipped)
    assert not torch.equal(fit, fit3)

    spec = fr.get_format("NVFP4_CB_K12")
    raw_fields = cb.nvfp4_cb_fields(
        weight, 12, grid="fp4", mode="product",
        col_weights=col_weights, scale_sweep=False, encode_tier="max",
    )
    # Build a distinct candidate so "kept LDLQ" is distinguishable from raw.
    candidate_fields = dict(raw_fields)
    candidate_fields["indices"] = torch.ones_like(raw_fields["indices"])
    raw_recon = cb.nvfp4_cb_reconstruct(raw_fields, 12, grid="fp4", mode="product")
    cand_recon = cb.nvfp4_cb_reconstruct(candidate_fields, 12, grid="fp4", mode="product")
    raw_mse_cert = cb._ldlq_activation_mse(weight, raw_recon, cert)
    cand_mse_cert = cb._ldlq_activation_mse(weight, cand_recon, cert)
    assert raw_mse_cert is not None and cand_mse_cert is not None

    # Branch A \u2014 gate keeps raw when holdout says raw (LDLQ worse or equal).
    # Inject controlled MSEs: raw better.
    monkeypatch.setattr(cb, "ldlq_reassign_cb_fields", lambda *a, **k: candidate_fields)
    scores_a = iter([raw_mse_cert, cand_mse_cert if cand_mse_cert > raw_mse_cert else raw_mse_cert + 1.0])
    monkeypatch.setattr(cb, "_ldlq_activation_mse", lambda *a, **k: next(scores_a))
    gated_a, info_a = cb.ldlq_reassign_cb_fields_gated(
        weight, raw_fields, col_weights, activations,
        grid="fp4", mode="product", k=12, gate_mode=cb.LDLQ_GATE_MODE_HOLDOUT,
    )
    assert info_a["gate"] == "raw_kept"
    assert info_a["kept_ldlq"] is False
    assert torch.equal(gated_a["indices"], raw_fields["indices"])

    # Branch B \u2014 gate keeps LDLQ only when holdout strictly improves.
    # Fit rows count proves the shipped candidate is the FIT-half candidate,
    # not an unscored all-rows refit (nvfp4_cb_formats.py:1778 certify-ship).
    fit_row_counts: list[int] = []
    def fake_reassign(_weight, _fields, _cw, rows, **_kwargs):
        fit_row_counts.append(int(torch.as_tensor(rows).shape[0]))
        return candidate_fields
    monkeypatch.setattr(cb, "ldlq_reassign_cb_fields", fake_reassign)
    # Make candidate strictly better on CERT.
    better = raw_mse_cert - abs(raw_mse_cert) * 0.1 - 1e-6
    scores_b = iter([raw_mse_cert, better])
    monkeypatch.setattr(cb, "_ldlq_activation_mse", lambda *a, **k: next(scores_b))
    gated_b, info_b = cb.ldlq_reassign_cb_fields_gated(
        weight, raw_fields, col_weights, activations,
        grid="fp4", mode="product", k=12, gate_mode=cb.LDLQ_GATE_MODE_HOLDOUT,
    )
    assert fit_row_counts == [fit.shape[0]]
    assert info_b["gate"] == "ldlq_kept"
    assert info_b["kept_ldlq"] is True
    assert torch.equal(gated_b["indices"], candidate_fields["indices"])


# Canonical E16 is the only packed-LDLQ ABI. See
# tests/test_cb_ldlq_canonical_e16.py::test_route_env_refusals and
# ::test_serial_shared_hessian_and_nondivisible_routes_are_refused for
# the per-route unit tests \u2014 this parametrized matrix replaces the
# stale byte-identity assertions (serial==batched, chunked/threaded)
# with refusal contracts plus a positive batched-repeat check.
@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
@pytest.mark.parametrize("format_name", ["NVFP4_CB_K12", "FP8_CB_K28"])
@pytest.mark.parametrize("strategy", ["chunked", "threaded"])
def test_batched_expert_ldlq_refuses_noncanonical_routes_and_repeats(
    device, format_name, strategy, monkeypatch
):
    """Formerly test_batched_expert_ldlq_is_bit_identical_to_serial.

    The four CPU combos (NVFP4_CB_K12/FP8_CB_K28 \u00d7 chunked/threaded \u00d7 cpu)
    previously asserted serial==batched under EXPERT_BATCH=3 /
    FEEDER_THREADS 0/4 / BATCH_STREAMS 1/2. Those packed routes are now
    intentionally refused by the canonical E16 ABI
    (nvfp4_cb_formats.py:_validate_packed_ldlq_route_env, product atoms
    fp4=4/fp8=2, outer tile 64 buffering only).
    """
    for name in (
        "PRISMAQUANT_CB_LDLQ_BATCH_EXPERTS",
        "PRISMAQUANT_CB_LDLQ_EXPERT_BATCH",
        "PRISMAQUANT_CB_LDLQ_FEEDER_THREADS",
        "PRISMAQUANT_CB_LDLQ_BATCH_STREAMS",
    ):
        monkeypatch.delenv(name, raising=False)
    generator = torch.Generator(device="cpu").manual_seed(19)
    experts, rows, columns = 16, 2, 256
    weight = (
        torch.randn(experts, rows, columns, generator=generator) * 0.08
    ).to(device=device, dtype=torch.bfloat16)
    col_weights = (
        torch.rand(experts, 1, columns, generator=generator) + 0.05
    ).to(device)
    activation_rows = tuple(
        torch.randn(16, columns, generator=generator).to(device)
        for _ in range(experts)
    )
    spec = fr.get_format(format_name)
    grid = "fp4" if format_name.startswith("NVFP4") else "fp8"
    fields = cb_fields_for_context(
        spec,
        weight,
        context=CBSerializationContext.production(encode_tier="fast"),
        col_weights=col_weights,
    )
    # Serial route \u2014 refused (tested exhaustively per-env in canonical_e16).
    with pytest.raises(CBLDLQError, match="canonical E16 batching"):
        cb.ldlq_reassign_cb_fields(
            weight, fields, col_weights, activation_rows,
            grid=grid, mode="product", batch_experts=False,
        )
    # Stale EXPERT_BATCH=3 \u2014 refused (canonical is 16).
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_EXPERT_BATCH", "3")
    with pytest.raises(CBLDLQError, match="must be 16"):
        cb.ldlq_reassign_cb_fields(
            weight, fields, col_weights, activation_rows,
            grid=grid, mode="product", batch_experts=True,
        )
    monkeypatch.delenv("PRISMAQUANT_CB_LDLQ_EXPERT_BATCH", raising=False)
    # Stale FEEDER_THREADS \u2014 refused (canonical is 0).
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_FEEDER_THREADS", "4")
    with pytest.raises(CBLDLQError, match="must be 0"):
        cb.ldlq_reassign_cb_fields(
            weight, fields, col_weights, activation_rows,
            grid=grid, mode="product", batch_experts=True,
        )
    monkeypatch.delenv("PRISMAQUANT_CB_LDLQ_FEEDER_THREADS", raising=False)
    # Canonical E16 batched route repeats deterministically.
    first = cb.ldlq_reassign_cb_fields(
        weight, fields, col_weights, activation_rows,
        grid=grid, mode="product", batch_experts=True,
    )
    second = cb.ldlq_reassign_cb_fields(
        weight, fields, col_weights, activation_rows,
        grid=grid, mode="product", batch_experts=True,
    )
    assert torch.equal(first["indices"], second["indices"])
    assert torch.equal(first["scales"], second["scales"])
