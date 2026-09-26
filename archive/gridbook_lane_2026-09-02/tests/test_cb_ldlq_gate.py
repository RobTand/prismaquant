"""Gated LDLQ refinement: do-no-harm on col-weighted MSE."""

import os

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_ldlq_refinement import (
    REFINEMENT_SCHEMA,
    build_refinement_provenance,
    validate_refinement_provenance,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_fields_for_context,
)


def test_ldlq_gate_keeps_better_arm():
    torch.manual_seed(0)
    w = torch.randn(32, 2048)
    cw = torch.ones(1, 2048) * 2.0
    k = 12
    grid, mode = "fp4", "product"
    fields = cb.nvfp4_cb_fields(w, k, grid=grid, mode=mode, col_weights=cw)
    recon_raw = cb.nvfp4_cb_reconstruct(fields, k, grid=grid, mode=mode)
    # Create synthetic activation rows: identity proxy for Hessian
    act = torch.randn(128, 2048)
    raw_act_err = float((act @ (w - recon_raw).T).pow(2).mean().item())
    # Gated call should not be worse than raw on the activation metric
    gated_fields, info = cb.ldlq_reassign_cb_fields_gated(
        w, fields, cw, act, grid=grid, mode=mode, k=k
    )
    recon_gated = cb.nvfp4_cb_reconstruct(gated_fields, k, grid=grid, mode=mode)
    gated_act_err = float((act @ (w - recon_gated).T).pow(2).mean().item())
    assert gated_act_err <= raw_act_err + 1e-9, f"gated regressed {gated_act_err} > {raw_act_err}"
    assert info["gate"] in {"ldlq_kept", "raw_kept", "ldlq_kept_all", "raw_kept_all", "mixed_per_expert", "disabled"}
    # Byte count identical regardless of arm
    assert gated_fields["indices"].numel() == fields["indices"].numel()


def test_ldlq_gate_disabled_returns_ldlq():
    os.environ["PRISMAQUANT_CB_LDLQ_GATE"] = "0"
    try:
        torch.manual_seed(1)
        w = torch.randn(8, 2048)
        cw = torch.randn(1, 2048).abs()
        act = torch.randn(64, 2048)
        fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
        gated_fields, info = cb.ldlq_reassign_cb_fields_gated(
            w, fields, cw, act, grid="fp4", mode="product", k=12
        )
        assert info["gate"] == "disabled"
        # When gate disabled, result should equal ungated LDLQ
        ungated = cb.ldlq_reassign_cb_fields(w, fields, cw, act, grid="fp4", mode="product")
        assert torch.equal(gated_fields["indices"], ungated["indices"])
    finally:
        os.environ["PRISMAQUANT_CB_LDLQ_GATE"] = "1"


def test_pack_emits_same_bytes_as_gated_footprint(monkeypatch):
    """The byte producer must apply the same gate the allocator prices."""
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_GATE", "holdout")
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(8, 256, generator=generator) * 0.25
    col_weights = torch.rand(256, generator=generator) + 0.05
    # One row cannot provide a held-out certificate, so the declared gate
    # must keep raw.  The old packer bypassed the gate and changed 25 indices
    # for this construction, making this a direct regression for shipped
    # bytes rather than merely a call-count assertion.
    activation_rows = torch.randn(1, 256, generator=generator)
    spec = fr.get_format("NVFP4_CB_K12")
    context = CBSerializationContext.production(
        encode_tier="balanced",
        ldlq=True,
    )
    footprint_fields = cb_fields_for_context(
        spec,
        weight,
        context=context,
        col_weights=col_weights,
        activation_rows=activation_rows,
    )
    expected = cb.nvfp4_cb_assemble_bytes(
        footprint_fields,
        12,
        grid="fp4",
        mode="product",
    )

    packed, packed_fields = cb.nvfp4_cb_pack(
        weight,
        12,
        grid="fp4",
        mode="product",
        col_weights=col_weights,
        scale_coding="two_tier",
        encode_tier="balanced",
        ldlq=True,
        activation_rows=activation_rows,
    )

    assert torch.equal(packed, expected)
    assert torch.equal(packed_fields["indices"], footprint_fields["indices"])


def test_ldlq_gate_per_expert_mixed():
    torch.manual_seed(2)
    E, R, C = 16, 2, 256
    w = torch.randn(E, R, C)
    cw = torch.ones(E, 1, C)
    # Make one expert pathological: its activation rows are tiny, so LDLQ may not help
    # but gate should still decide per expert
    acts = [torch.randn(32, C) for _ in range(E)]
    # Use a fixed codebook so we can compare
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    gated_fields, info = cb.ldlq_reassign_cb_fields_gated(
        w, fields, cw, acts, grid="fp4", mode="product", k=12
    )
    assert "per_expert_kept" in info or "kept_ldlq" in info
    # Ensure indices shape preserved
    assert gated_fields["indices"].shape == fields["indices"].shape


def test_refinement_provenance_roundtrip():
    prov = build_refinement_provenance(cost_ldlq=False, export_ldlq=True, gate="col_weighted_mse", gate_enabled=True)
    assert prov["schema"] == REFINEMENT_SCHEMA
    out = validate_refinement_provenance(prov, where="test")
    assert out["cost_ldlq"] is False
    assert out["export_ldlq"] is True
    # Forged ldlq downgrade should fail
    try:
        build_refinement_provenance(cost_ldlq=True, export_ldlq=False, gate="x", gate_enabled=True)
        assert False, "should have raised"
    except ValueError:
        pass


def test_batch_vs_serial_ldlq_bit_identical():
    # Artifact production has one canonical E16 regime; serial is an oracle.
    os.environ["PRISMAQUANT_CB_LDLQ_GATE"] = "0"
    torch.manual_seed(3)
    E, R, C = 16, 2, 256
    w = torch.randn(E, R, C)
    cw = torch.ones(E, 1, C)
    acts = [torch.randn(32, C) for _ in range(E)]
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    with pytest.raises(RuntimeError, match="canonical E16 batching"):
        cb.ldlq_reassign_cb_fields(
            w, fields, cw, acts, grid="fp4", mode="product",
            batch_experts=False,
        )
    batched = cb.ldlq_reassign_cb_fields(w, fields, cw, acts, grid="fp4", mode="product", batch_experts=True)
    repeated = cb.ldlq_reassign_cb_fields(w, fields, cw, acts, grid="fp4", mode="product", batch_experts=True)
    assert torch.equal(repeated["indices"], batched["indices"])
    os.environ["PRISMAQUANT_CB_LDLQ_GATE"] = "1"


# --- held-out certificate (2026-08-08) -------------------------------------
# The legacy gate scored on the same rows that fitted the Hessian, so it could
# not fail; measured, its error was ANTI-correlated with the true benefit
# (20x overstatement at 64 activation rows, 48.5x at 1-3). These cover the
# replacement: the decision is made out of sample, and where no held-out row
# exists the tensor keeps raw.


def _small_case(rows, *, seed=7, C=2048, R=16):
    torch.manual_seed(seed)
    w = torch.randn(R, C)
    cw = torch.randn(1, C).abs() + 0.1
    act = torch.randn(rows, C)
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    return w, cw, act, fields


def test_holdout_split_is_content_keyed_and_deterministic():
    rows = torch.randn(16, 64)
    a = cb._ldlq_holdout_split(rows)
    b = cb._ldlq_holdout_split(rows.clone())
    assert a is not None and b is not None
    # Same content -> same split, across independent calls and tensor identities.
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # Partition: fit and hold are disjoint and together cover every row.
    assert a[0].shape[0] + a[1].shape[0] == rows.shape[0]
    assert a[0].shape[0] >= 1 and a[1].shape[0] >= 1
    # Different content -> the seed changes.
    other = cb._ldlq_holdout_split(torch.randn(16, 64))
    assert other is not None
    assert not torch.equal(a[0], other[0])


def test_holdout_split_refuses_below_min_rows():
    assert cb.LDLQ_GATE_MIN_ROWS == 16
    assert cb._ldlq_holdout_split(torch.randn(1, 64)) is None
    assert cb._ldlq_holdout_split(torch.randn(15, 64)) is None
    assert cb._ldlq_holdout_split(torch.randn(16, 64)) is not None


def test_single_row_tensor_is_uncertifiable_and_keeps_raw():
    """n=1: no held-out row exists, so LDLQ cannot be certified -> raw."""
    w, cw, act, fields = _small_case(1)
    gated, info = cb.ldlq_reassign_cb_fields_gated(
        w, fields, cw, act, grid="fp4", mode="product", k=12,
        gate_mode=cb.LDLQ_GATE_MODE_HOLDOUT,
    )
    assert info["gate"] == "raw_uncertifiable_too_few_rows"
    assert info["kept_ldlq"] is False
    assert info["metric"] == "holdout_activation_output_mse"
    # Raw fields are returned verbatim.
    assert torch.equal(gated["indices"], fields["indices"])


def test_holdout_gate_scores_out_of_sample_not_in_sample():
    """The two modes must be able to disagree; holdout must not score on the fit rows."""
    w, cw, act, fields = _small_case(24, seed=11)
    _, info_h = cb.ldlq_reassign_cb_fields_gated(
        w, fields, cw, act, grid="fp4", mode="product", k=12,
        gate_mode=cb.LDLQ_GATE_MODE_HOLDOUT,
    )
    _, info_i = cb.ldlq_reassign_cb_fields_gated(
        w, fields, cw, act, grid="fp4", mode="product", k=12,
        gate_mode=cb.LDLQ_GATE_MODE_IN_SAMPLE,
    )
    assert info_h["metric"] == "holdout_activation_output_mse"
    assert info_i["metric"] == "activation_output_mse"
    assert info_h["gate_mode"] == cb.LDLQ_GATE_MODE_HOLDOUT
    # The in-sample arm scores the all-rows LDLQ on all rows; the holdout arm
    # scores a half-fit LDLQ on the other half. Identical numbers would mean
    # the split never happened.
    assert info_h["ldlq_mse"] != info_i["ldlq_mse"]


def test_holdout_gate_ships_the_exact_candidate_it_scores(monkeypatch):
    """A half-fit certificate must never authorize a different all-row refit."""
    generator = torch.Generator().manual_seed(20260808)
    w = torch.randn(4, 256, generator=generator)
    cw = torch.ones(256)
    act = torch.randn(16, 256, generator=generator)
    fields = cb.nvfp4_cb_fields(
        w, 12, grid="fp4", mode="product", col_weights=cw
    )
    candidate = dict(fields)
    candidate["indices"] = torch.ones_like(fields["indices"])
    fit_row_counts = []

    def fake_reassign(_weight, _fields, _cw, rows, **_kwargs):
        fit_row_counts.append(int(torch.as_tensor(rows).shape[0]))
        return candidate

    scores = iter((2.0, 1.0))  # raw, then the exact candidate
    monkeypatch.setattr(cb, "ldlq_reassign_cb_fields", fake_reassign)
    monkeypatch.setattr(cb, "_ldlq_activation_mse", lambda *_args: next(scores))

    gated, info = cb.ldlq_reassign_cb_fields_gated(
        w, fields, cw, act, grid="fp4", mode="product", k=12,
        gate_mode=cb.LDLQ_GATE_MODE_HOLDOUT,
    )

    assert fit_row_counts == [8]
    assert info["gate"] == "ldlq_kept"
    assert torch.equal(gated["indices"], candidate["indices"])


def test_holdout_gate_emits_per_tensor_ratio_telemetry():
    """The certificate ratio IS the honest per-tensor s_output; it must surface."""
    E, R, C = 16, 2, 256
    torch.manual_seed(3)
    w = torch.randn(E, R, C)
    cw = (torch.randn(E, 1, C).abs() + 0.1)
    acts = tuple(torch.randn(12 + 4 * i, C) for i in range(E))
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    _, info = cb.ldlq_reassign_cb_fields_gated(
        w, fields, cw, acts, grid="fp4", mode="product", k=12,
        gate_mode=cb.LDLQ_GATE_MODE_HOLDOUT,
    )
    ratios = info["holdout_ratio_per_expert"]
    assert len(ratios) == E
    kept = info["per_expert_kept"]
    # A kept slice must have earned it: ratio strictly below 1.
    for keep, ratio in zip(kept, ratios):
        if keep:
            assert ratio is not None and ratio < 1.0, (keep, ratio)


def test_gate_mode_env_spellings():
    from prismaquant.nvfp4_cb_formats import _ldlq_gate_mode as m
    assert m({}) == cb.LDLQ_GATE_MODE_HOLDOUT                       # default
    assert m({"PRISMAQUANT_CB_LDLQ_GATE": "1"}) == cb.LDLQ_GATE_MODE_HOLDOUT
    assert m({"PRISMAQUANT_CB_LDLQ_GATE": "holdout"}) == cb.LDLQ_GATE_MODE_HOLDOUT
    assert m({"PRISMAQUANT_CB_LDLQ_GATE": "in_sample"}) == cb.LDLQ_GATE_MODE_IN_SAMPLE
    assert m({"PRISMAQUANT_CB_LDLQ_GATE": "0"}) == cb.LDLQ_GATE_MODE_DISABLED
    try:
        m({"PRISMAQUANT_CB_LDLQ_GATE": "maybe"})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown gate spelling must fail loudly")


def test_holdout_gate_name_is_accepted_by_refinement_contract():
    payload = build_refinement_provenance(
        cost_ldlq=False, export_ldlq=True,
        gate="holdout_activation_output_mse", gate_enabled=True,
    )
    assert validate_refinement_provenance(payload, where="test") is not None
