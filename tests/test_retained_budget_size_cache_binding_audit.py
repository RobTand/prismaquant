"""Retained-budget derivation must read the current prepared candidate sizes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import pickle

import pytest
import torch

from prismaquant.joint_retained_window_plan import EXECUTION_SCHEMA, RetainedWindowBudget
from prismaquant.production_weight_cache import ProductionWeightCache
from test_joint_operator_windows import policy as operator_policy
from test_layer_major_boundary_capture import policy as boundary_policy


UNIT = "model.layers.0.q_proj"
FORMAT = "FP8_E4M3"
MIB = 1024**2


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value))
    return path


def _prepared(directory, dtype):
    directory.mkdir()
    render = directory / "render.pt"
    torch.save(torch.zeros(32, 32, dtype=dtype), render)
    cache = ProductionWeightCache(
        weights={(UNIT, FORMAT): render.name}, levers={}, cache_dir=str(directory),
    )
    payload = directory / "production.pkl"
    payload.write_bytes(pickle.dumps(cache))
    prepared = _write(directory / "prepared.json", {
        "status": "complete", "formats_by_qname": {UNIT: [FORMAT, "BF16"]},
        "production_cache": {"path": str(payload), "sha256": _sha(payload)},
    })
    assert cache.estimate_nbytes([(UNIT, FORMAT)]) == render.stat().st_size
    return prepared, render


@pytest.mark.parametrize("case", [
    "same_preparation", "different_preparation", "changed_render_file",
])
def test_size_cache_does_not_change_current_preparation_derivation(tmp_path, case):
    repository = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "budget_size_cache_audit", repository / "tools/derive_retained_window_budget.py",
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    budget = RetainedWindowBudget(
        physical_limit_bytes=64*MIB, safety_margin_bytes=MIB,
        metadata_reserve_bytes=MIB, runtime_reserve_bytes=MIB,
        workspace_reserve_bytes=MIB, boundary_reserve_bytes=MIB,
        auxiliary_reserve_bytes=MIB, load_buffer_bytes=MIB,
        read_page_reserve_bytes=4096, candidate_delta_bytes=MIB,
        statistics_cap_bytes=MIB, retained_render_cap_bytes=MIB,
        max_windows_per_layer=8,
    )
    plan = _write(tmp_path / "plan.json", {
        "schema": "prismaquant.tessera_joint_aura.plan.v1", "max_gpu_bytes": 32*MIB,
        "execution": {
            "retained_operator_windows": {
                "schema": EXECUTION_SCHEMA, "budget": budget.as_dict(),
                "source_reserve_bytes": MIB, "source_loading_reserve_bytes": MIB,
            },
            "operator_windows": operator_policy(
                max_statistics_bytes=MIB, max_candidate_bytes=MIB, prefetch_workers=1,
            ),
            "boundary_storage": boundary_policy(tmp_path / "boundaries"),
        },
    })
    census = _write(tmp_path / "census.json", {
        "unit_shapes": {UNIT: [32, 32]}, "max_abs": {UNIT: 1.0},
    })
    first, original_render = _prepared(tmp_path / "first", torch.bfloat16)
    sizes = tmp_path / "candidate-sizes.json"

    def derive(prepared, output, *, cached):
        argv = [
            "--plan", str(plan), "--plan-sha256", _sha(plan),
            "--prepared", str(prepared), "--prepared-sha256", _sha(prepared),
            "--census", str(census), "--out", str(output),
        ]
        if cached:
            argv += ["--render-size-cache", str(sizes)]
        assert tool.main(argv) == 0
        return json.loads(output.read_text())

    derive(first, tmp_path / "warm.json", cached=True)
    selected, current_render = first, original_render
    original_size = original_render.stat().st_size
    if case == "different_preparation":
        selected, current_render = _prepared(tmp_path / "second", torch.float32)
    elif case == "changed_render_file":
        torch.save(torch.zeros(32, 32, dtype=torch.float32), original_render)
    if case != "same_preparation":
        assert current_render.stat().st_size != original_size

    reused = derive(selected, tmp_path / "reused.json", cached=True)
    fresh = derive(selected, tmp_path / "fresh.json", cached=False)
    expected_bytes = current_render.stat().st_size
    fresh_demand = fresh["retained_window_budget_derivation"]["demand"]["load_buffer_bytes"]
    assert fresh_demand["largest_serialized_bytes"] == expected_bytes
    reused_demand = reused["retained_window_budget_derivation"]["demand"]["load_buffer_bytes"]
    assert reused_demand["largest_serialized_bytes"] == expected_bytes
    assert reused["execution"]["retained_operator_windows"]["budget"] == (
        fresh["execution"]["retained_operator_windows"]["budget"]
    )
