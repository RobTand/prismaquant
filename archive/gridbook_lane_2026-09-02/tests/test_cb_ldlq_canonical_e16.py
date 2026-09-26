"""Canonical packed-LDLQ routing, cache, and gate contracts."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_ldlq_atoms import CBLDLQError
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_serialization_context_from_stamp,
    cb_serialization_context_stamp,
)


_ROUTE_ENVS = (
    "PRISMAQUANT_CB_LDLQ_BATCH_EXPERTS",
    "PRISMAQUANT_CB_LDLQ_EXPERT_BATCH",
    "PRISMAQUANT_CB_LDLQ_FEEDER_THREADS",
    "PRISMAQUANT_CB_LDLQ_BATCH_STREAMS",
)


def _canonical_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ROUTE_ENVS:
        monkeypatch.delenv(name, raising=False)


def _manual_case(*, experts: int = 16, rows: int = 2, columns: int = 8):
    weight = torch.zeros(experts, rows, columns)
    fields = {
        "indices": torch.full(
            (experts * rows, columns // 8, 4), 99, dtype=torch.long
        ),
        "scales": torch.ones(experts * rows, 1),
        "codebook": tuple(torch.zeros(2, 2) for _ in range(4)),
        "shape": tuple(weight.shape),
    }
    col_weights = torch.ones(experts, 1, columns)
    activations = tuple(
        torch.full((16, columns), float(expert + 1))
        for expert in range(experts)
    )
    return weight, fields, col_weights, activations


def test_serialization_stamp_pins_route_solver_streams_and_roundtrips(
    monkeypatch: pytest.MonkeyPatch,
):
    _canonical_env(monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_BATCH_STREAMS", "4")
    context = CBSerializationContext.production(ldlq=True)
    stamp = cb_serialization_context_stamp(context)
    kernel = stamp["ldlq_packed_kernel"]
    assert kernel["route"] == "e16_batched_v1"
    assert kernel["expert_batch"] == 16
    assert kernel["batch_streams"] == 4
    assert kernel["nondivisible_experts"] == "refuse"
    assert kernel["dense_solver"] == "explicit_forward_substitution_2d4d_v1"
    assert kernel["packed_solver"] == "torch.linalg.solve_triangular"
    assert cb_serialization_context_from_stamp(
        stamp, where="canonical E16"
    ) == context

    missing = copy.deepcopy(stamp)
    del missing["ldlq_packed_kernel"]
    with pytest.raises(ValueError, match="missing its packed-kernel ABI"):
        cb_serialization_context_from_stamp(missing, where="missing")
    stale = copy.deepcopy(stamp)
    stale["ldlq_packed_kernel"]["expert_batch"] = 8
    with pytest.raises(ValueError, match="packed-kernel ABI mismatch"):
        cb_serialization_context_from_stamp(stale, where="stale")


@pytest.mark.parametrize(
    ("name", "value", "pattern"),
    (
        ("PRISMAQUANT_CB_LDLQ_BATCH_EXPERTS", "0", "canonical E16"),
        ("PRISMAQUANT_CB_LDLQ_BATCH_EXPERTS", "maybe", "must be 0 or 1"),
        ("PRISMAQUANT_CB_LDLQ_EXPERT_BATCH", "8", "must be 16"),
        ("PRISMAQUANT_CB_LDLQ_EXPERT_BATCH", "many", "must be an integer"),
        ("PRISMAQUANT_CB_LDLQ_FEEDER_THREADS", "2", "must be 0"),
        ("PRISMAQUANT_CB_LDLQ_FEEDER_THREADS", "-1", "greater than"),
        ("PRISMAQUANT_CB_LDLQ_BATCH_STREAMS", "0", "greater than"),
    ),
)
def test_route_env_refusals(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    pattern: str,
):
    _canonical_env(monkeypatch)
    monkeypatch.setenv(name, value)
    weight, fields, col_weights, activations = _manual_case()
    with pytest.raises(CBLDLQError, match=pattern):
        cb.ldlq_reassign_cb_fields(
            weight,
            fields,
            col_weights,
            activations,
            grid="fp8",
            mode="product",
        )


def test_serial_shared_hessian_and_nondivisible_routes_are_refused(
    monkeypatch: pytest.MonkeyPatch,
):
    _canonical_env(monkeypatch)
    weight, fields, col_weights, activations = _manual_case()
    with pytest.raises(CBLDLQError, match="canonical E16"):
        cb.ldlq_reassign_cb_fields(
            weight,
            fields,
            col_weights,
            activations,
            grid="fp8",
            mode="product",
            batch_experts=False,
        )
    with pytest.raises(CBLDLQError, match="one activation tensor per expert"):
        cb.ldlq_reassign_cb_fields(
            weight,
            fields,
            col_weights,
            torch.ones(16, 8),
            grid="fp8",
            mode="product",
        )
    bad = _manual_case(experts=17)
    with pytest.raises(CBLDLQError, match="not divisible"):
        cb.ldlq_reassign_cb_fields_gated(
            bad[0],
            bad[1],
            bad[2],
            bad[3],
            grid="fp8",
            mode="product",
            k=28,
        )


def test_shared_activation_gate_falls_back_raw_before_candidate_assertion(
    monkeypatch: pytest.MonkeyPatch,
):
    _canonical_env(monkeypatch)
    weight, fields, col_weights, _activations = _manual_case()
    result, info = cb.ldlq_reassign_cb_fields_gated(
        weight,
        fields,
        col_weights,
        torch.ones(16, 8),
        grid="fp8",
        mode="product",
        k=28,
    )
    assert result is fields
    assert info["gate"] == "raw_fallback_shared_activation_for_packed"


def test_empty_expert_stays_raw_without_blocking_e16_candidate(
    monkeypatch: pytest.MonkeyPatch,
):
    _canonical_env(monkeypatch)
    weight, fields, col_weights, _activations = _manual_case()
    weight.fill_(1.0)
    activations = tuple(
        torch.empty(0, 8)
        if expert == 3
        else torch.full((16, 8), float(expert + 1))
        for expert in range(16)
    )
    candidate = dict(fields)
    candidate["indices"] = torch.arange(16).reshape(
        16, 1, 1, 1
    ).expand(16, 2, 1, 4).reshape_as(fields["indices"]).clone()

    monkeypatch.setattr(
        cb,
        "ldlq_reassign_cb_fields",
        lambda *args, **kwargs: candidate,
    )

    def fake_reconstruct(source, _k, *, grid, mode):
        # The gate now scores expert-slice-by-expert-slice, so this fake
        # serves both the full (16*2)-row stack and a single-expert 2-row
        # slice of it (shape carried by the sliced fields dict).
        idx = source["indices"]
        experts = idx.shape[0] // 2
        view = idx.reshape(experts, 2, 1, 4)
        changed = view[:, 0, 0, 0] != 99
        result = torch.zeros(experts, 2, 8, dtype=weight.dtype)
        result[changed] = 1.0
        if len(source["shape"]) == 2:
            return result[0]
        return result

    monkeypatch.setattr(cb, "nvfp4_cb_reconstruct", fake_reconstruct)
    result, info = cb.ldlq_reassign_cb_fields_gated(
        weight,
        fields,
        col_weights,
        activations,
        grid="fp8",
        mode="product",
        k=28,
    )
    view = result["indices"].reshape(16, 2, 1, 4)
    assert info["gate"] == "mixed_per_expert"
    assert info["missing_experts"] == [3]
    assert torch.all(view[3] == 99)
    assert all(
        torch.all(view[expert] == expert)
        for expert in range(16)
        if expert != 3
    )


def test_repeated_cold_prior_split_builds_one_factor(
    monkeypatch: pytest.MonkeyPatch,
):
    source = torch.randn(16, 8)
    splits = cb._ldlq_holdout_splits((source,) * 16)
    assert all(split is not None for split in splits)
    fit = [split[0] for split in splits if split is not None]
    assert all(item is fit[0] for item in fit)

    with cb._LDLQ_FACTOR_CACHE_LOCK:
        cb._LDLQ_FACTOR_CACHE.clear()
    builds = 0

    def fake_prepare(rows, *, device, damping_fraction):
        nonlocal builds
        builds += 1
        return SimpleNamespace(upper_inverse_cholesky=torch.eye(8))

    monkeypatch.setattr(
        "prismaquant.cb_ldlq_atoms.prepare_upper_inverse_cholesky",
        fake_prepare,
    )
    for rows in fit:
        cb._ldlq_inverse_factor_cached(
            rows, device=torch.device("cpu"), damping_fraction=0.01
        )
    assert builds == 1


def test_e32_is_always_two_e16_chunks(monkeypatch: pytest.MonkeyPatch):
    _canonical_env(monkeypatch)
    weight, fields, col_weights, activations = _manual_case(experts=32)
    monkeypatch.setattr(
        cb,
        "_ldlq_inverse_factor_cached",
        lambda *args, **kwargs: torch.eye(8),
    )
    batches: list[int] = []

    def fake_batched(weight_arg, scales, codebook, upper, **kwargs):
        batches.append(int(weight_arg.shape[0]))
        return SimpleNamespace(
            indices=torch.zeros(16, 2, 1, 4, dtype=torch.long)
        )

    monkeypatch.setattr(
        "prismaquant.cb_ldlq_atoms.reassign_product_3d_batched",
        fake_batched,
    )
    result = cb._ldlq_reassign_fields_3d_batched(
        weight,
        fields,
        col_weights,
        activations,
        grid="fp8",
        mode="product",
        block_size=64,
        damping_fraction=0.01,
    )
    assert batches == [16, 16]
    assert result["indices"].shape == fields["indices"].shape


def test_gate_mode_typo_is_rejected_before_legacy_arm():
    weight = torch.zeros(1, 256)
    fields = cb.nvfp4_cb_fields(
        weight,
        12,
        grid="fp4",
        mode="product",
        col_weights=torch.ones(256),
        scale_sweep=False,
    )
    with pytest.raises(ValueError, match="gate_mode must be one of"):
        cb.ldlq_reassign_cb_fields_gated(
            weight,
            fields,
            torch.ones(256),
            torch.ones(16, 256),
            grid="fp4",
            mode="product",
            k=12,
            gate_mode="Holdout",
        )
