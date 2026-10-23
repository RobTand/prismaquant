from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def allocator_fixture(tmp_path, monkeypatch):
    fixture_path = Path.cwd() / "tests" / "test_allocator_serve_constraints.py"
    assert (Path.cwd() / "prismaquant" / "allocator.py").is_file()
    monkeypatch.syspath_prepend(str(fixture_path.parent))
    spec = importlib.util.spec_from_file_location("_allocator_fixture_audit", fixture_path)
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    model, probe, costs, stats = fixture._fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6,
    )
    fp8_bytes, _floor = fixture._artifact_bytes(model, "FP8_E4M3", stats)
    nvfp4_bytes, _floor = fixture._artifact_bytes(model, "NVFP4", stats)
    table = fixture._dispatch_table(tmp_path, nvfp4_prefill=1.0, fp8_prefill=1.0)
    return fixture, probe, costs, fp8_bytes, nvfp4_bytes, table


def _run_allocator(fixture, tmp_path, monkeypatch, extra):
    module, probe, costs, fp8_bytes, _nvfp4_bytes, _table = fixture
    out = tmp_path / "allocation"
    out.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "allocator", "--probe", str(probe), "--costs", str(costs),
        "--formats", "NVFP4,FP8_E4M3", "--pareto-targets", "4.6,8.2",
        "--target-disk-gb", repr((fp8_bytes + 10_000) / module.fp.GB),
        "--layer-config", str(out / "layer_config.json"),
        "--pareto-csv", str(out / "pareto.csv"), "--allow-default-profile",
        "--artifact-overhead-reserve-bytes", str(module._OVERHEAD_RESERVE),
        "--threads", "1", *extra,
    ])
    module.alloc.main()
    return (
        json.loads((out / "selection.json").read_text()),
        json.loads((out / "layer_config.json").read_text()),
    )


@pytest.mark.parametrize("context", ["neither", "table_only", "mix_only"])
def test_explicit_unevaluable_device_budget_refuses_public_allocator(
    allocator_fixture, tmp_path, monkeypatch, context,
):
    _module, _probe, _costs, _fp8, _nvfp4, table = allocator_fixture
    extra = ["--serve-device-budget-bytes", "1"]
    if context == "table_only":
        extra += ["--serve-dispatch-table", str(table)]
    elif context == "mix_only":
        extra += ["--serve-workload-mix", "prefill:dense=1.0"]

    with pytest.raises(SystemExit) as refused:
        _run_allocator(allocator_fixture, tmp_path, monkeypatch, extra)

    assert refused.value.code != 0
    assert not (tmp_path / "allocation" / "layer_config.json").exists()


def test_absent_device_budget_keeps_real_unconstrained_selection(
    allocator_fixture, tmp_path, monkeypatch,
):
    selection, config = _run_allocator(allocator_fixture, tmp_path, monkeypatch, [])

    assert selection["serve_constraints"]["active"] is False
    assert {row["data_type"] for name, row in config.items()
            if name != "__prismaquant__"} == {"fp8_e4m3"}


def test_complete_memory_context_enforces_budget_with_real_solver(
    allocator_fixture, tmp_path, monkeypatch,
):
    _module, _probe, _costs, fp8, nvfp4, table = allocator_fixture
    kv, scratch = 1000, 2000
    budget = nvfp4 + kv + scratch + 10
    assert budget < fp8 + kv + scratch
    selection, config = _run_allocator(allocator_fixture, tmp_path, monkeypatch, [
        "--serve-dispatch-table", str(table),
        "--serve-workload-mix", "prefill:dense=1.0",
        "--serve-device-budget-bytes", str(budget),
        "--serve-kv-bytes", str(kv), "--serve-peak-scratch-bytes", str(scratch),
    ])

    verdict = selection["serve_constraints"]
    assert verdict["active"] is True and verdict["feasible"] is True
    assert verdict["predicted"]["device_memory_bytes"] <= budget
    assert verdict["coverage"]["memory"]["kv_bytes"] == kv
    assert verdict["coverage"]["memory"]["peak_scratch_bytes"] == scratch
    assert any(row["binding_constraint"] == "device_memory_bytes"
               for row in verdict["rejected_assignments"])
    assert {row["data_type"] for name, row in config.items()
            if name != "__prismaquant__"} == {"nv_fp"}
