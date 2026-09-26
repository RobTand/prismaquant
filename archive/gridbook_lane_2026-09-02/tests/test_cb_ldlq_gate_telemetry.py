from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_ldlq_atoms import CBLDLQHessianError
from prismaquant.cb_ldlq_gate_telemetry import (
    KERNEL_STAMP_SCHEMA,
    LDLQGateTelemetryCollector,
    LDLQTelemetryError,
    SIDECAR_FILENAME,
    normalize_gate_record,
    verify_sidecar_reference,
)


def _kernel_stamp() -> dict:
    return {
        "schema": KERNEL_STAMP_SCHEMA,
        "abi": "prismaquant.cb_ldlq.product_atom_e16_route_split.v2",
        "implementation_sha256": hashlib.sha256(b"fixture").hexdigest(),
        "objective": "squared_l2((target-codeword)@inv(U_AA))",
        "candidate_solver": {
            "dense_2d": "explicit_forward_substitution_2d4d_v1",
            "packed_e16": "torch.linalg.solve_triangular",
        },
        "tie_break": "strict_argmin_lowest_codebook_index",
        "outer_tile_columns": 64,
        "atom_size_by_grid": {"fp4": 4, "fp8": 2},
        "packed_expert_kernel": {
            "route": "e16_batched_v1",
            "batch_size": 16,
            "streams": 1,
            "nondivisible_experts": "refuse",
        },
        "execution_environment": {
            "torch_version": "fixture",
            "cuda_version": "fixture",
            "gpu_arch": "sm_121",
            "gpu_name": "NVIDIA GB10",
            "producer_image_digest": "sha256:fixture",
        },
    }


def _gate_info() -> dict:
    return {
        "kernel_stamp": _kernel_stamp(),
        "gate": "mixed_per_expert",
        "metric": "holdout_activation_output_mse",
        "gate_mode": "holdout",
        "per_expert_kept": [True, False, False],
        "missing_experts": [1],
        "uncertifiable_experts": [],
        "hessian_failed_experts": [2],
        "raw_mse_per_expert": [4.0, float("inf"), 3.0],
        "ldlq_mse_per_expert": [2.0, float("inf"), 3.0],
        "holdout_ratio_per_expert": [0.5, None, None],
    }


def test_normalizes_missing_and_hessian_status_without_nonfinite_json():
    record = normalize_gate_record(
        qname="model.layers.7.mlp.experts.w13",
        shape=(3, 8, 64),
        grid="fp4",
        mode="product",
        k=18,
        gate_info=_gate_info(),
    )
    assert record["per_expert"] == {
        "kept_ldlq": [True, False, False],
        "missing_activation": [1],
        "uncertifiable": [],
        "hessian_failed": [2],
        "raw_mse": [4.0, None, 3.0],
        "ldlq_mse": [2.0, None, 3.0],
        "ldlq_over_raw_mse": [0.5, None, None],
    }


def test_excluded_expert_cannot_claim_a_ratio():
    info = _gate_info()
    info["holdout_ratio_per_expert"][2] = 1.0
    with pytest.raises(LDLQTelemetryError, match="must not claim"):
        normalize_gate_record(
            qname="x",
            shape=(3, 8, 64),
            grid="fp4",
            mode="product",
            k=18,
            gate_info=info,
        )


def test_collector_is_exact_coverage_content_addressed_and_verifiable(
    tmp_path: Path,
):
    qname = "model.layers.7.mlp.experts.w13"
    collector = LDLQGateTelemetryCollector(
        expected_qnames=[qname], kernel_stamp=_kernel_stamp()
    )
    kwargs = {
        "qname": qname,
        "shape": (3, 8, 64),
        "grid": "fp4",
        "mode": "product",
        "k": 18,
        "gate_info": _gate_info(),
    }
    collector.record(**kwargs)
    collector.record(**kwargs)
    config = {"provenance": {}}
    reference = collector.publish(tmp_path, config)
    assert reference["summary"]["hessian_failed"] == 1
    assert (
        hashlib.sha256((tmp_path / SIDECAR_FILENAME).read_bytes()).hexdigest()
        == reference["sha256"]
    )
    payload = verify_sidecar_reference(tmp_path, reference)
    assert payload["records"][0]["qname"] == qname


def test_missing_coverage_and_conflicting_duplicate_fail_closed(tmp_path: Path):
    collector = LDLQGateTelemetryCollector(
        expected_qnames=["x", "missing"], kernel_stamp=_kernel_stamp()
    )
    info = _gate_info()
    collector.record(
        qname="x",
        shape=(3, 8, 64),
        grid="fp4",
        mode="product",
        k=18,
        gate_info=info,
    )
    changed = _gate_info()
    changed["per_expert_kept"] = [False, False, False]
    with pytest.raises(LDLQTelemetryError, match="conflicting"):
        collector.record(
            qname="x",
            shape=(3, 8, 64),
            grid="fp4",
            mode="product",
            k=18,
            gate_info=changed,
        )
    with pytest.raises(LDLQTelemetryError, match="coverage mismatch"):
        collector.publish(tmp_path, {"provenance": {}})
    assert not (tmp_path / SIDECAR_FILENAME).exists()


def test_pack_out_channel_preserves_only_failed_expert_raw(
    monkeypatch: pytest.MonkeyPatch,
):
    for name in (
        "PRISMAQUANT_CB_LDLQ_BATCH_EXPERTS",
        "PRISMAQUANT_CB_LDLQ_EXPERT_BATCH",
        "PRISMAQUANT_CB_LDLQ_FEEDER_THREADS",
        "PRISMAQUANT_CB_LDLQ_BATCH_STREAMS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_GATE", "0")
    experts, rows, columns = 16, 2, 256
    generator = torch.Generator().manual_seed(20260808)
    weight = torch.randn(
        experts, rows, columns, generator=generator
    ) * 0.05
    col_weights = torch.ones(experts, 1, columns)
    activations = tuple(
        torch.full((16, columns), float(expert + 1))
        for expert in range(experts)
    )

    def fake_factor(x, *, device, damping_fraction):
        expert = int(torch.as_tensor(x)[0, 0].item()) - 1
        if expert == 5:
            raise CBLDLQHessianError("injected dead channel")
        return torch.eye(columns, device=device)

    def fake_reassign(weight_arg, scales, codebook, upper, **kwargs):
        return SimpleNamespace(
            indices=torch.zeros(
                experts, rows, columns // 8, 2, dtype=torch.long
            )
        )

    monkeypatch.setattr(cb, "_ldlq_inverse_factor_cached", fake_factor)
    monkeypatch.setattr(
        "prismaquant.cb_ldlq_atoms.reassign_product_3d_batched",
        fake_reassign,
    )
    gate_info: dict[str, object] = {}
    _packed, result = cb.nvfp4_cb_pack(
        weight,
        12,
        grid="fp4",
        mode="product",
        col_weights=col_weights,
        scale_sweep=False,
        ldlq=True,
        activation_rows=activations,
        ldlq_gate_info_out=gate_info,
    )
    raw = cb.nvfp4_cb_fields(
        weight,
        12,
        grid="fp4",
        mode="product",
        col_weights=col_weights,
        scale_sweep=False,
    )
    result_view = result["indices"].reshape(
        experts, rows, columns // 8, 2
    )
    raw_view = raw["indices"].reshape_as(result_view)
    assert gate_info["hessian_failed_experts"] == [5]
    assert torch.equal(result_view[5], raw_view[5])
    assert all(
        torch.count_nonzero(result_view[expert]).item() == 0
        for expert in range(experts)
        if expert != 5
    )
    assert gate_info["kernel_stamp"] == cb.canonical_ldlq_kernel_stamp()
