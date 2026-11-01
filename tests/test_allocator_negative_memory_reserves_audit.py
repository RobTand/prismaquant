from __future__ import annotations

import pytest

import test_allocator_memory_budget_admission_audit as baseline

allocator_fixture = baseline.allocator_fixture


@pytest.mark.parametrize("reserve", ["--serve-kv-bytes", "--serve-peak-scratch-bytes"])
def test_negative_reserve_refuses_actual_nonmeasured_allocator(
    allocator_fixture, tmp_path, monkeypatch, reserve,
):
    _module, _probe, _costs, fp8, _nvfp4, table = allocator_fixture
    with pytest.raises(SystemExit) as refused:
        baseline._run_allocator(allocator_fixture, tmp_path, monkeypatch, [
            "--serve-dispatch-table", str(table),
            "--serve-workload-mix", "prefill:dense=1.0",
            "--serve-device-budget-bytes", "1", reserve, str(-fp8),
        ])

    assert refused.value.code != 0
    assert reserve in str(refused.value)
    assert not (tmp_path / "allocation" / "layer_config.json").exists()


@pytest.mark.parametrize("reserve", ["--serve-kv-bytes", "--serve-peak-scratch-bytes"])
def test_negative_reserve_refuses_inactive_public_allocator(
    allocator_fixture, tmp_path, monkeypatch, reserve,
):
    with pytest.raises(SystemExit, match=reserve) as refused:
        baseline._run_allocator(
            allocator_fixture, tmp_path, monkeypatch, [reserve, "-1"],
        )

    assert refused.value.code != 0
    assert not (tmp_path / "allocation" / "layer_config.json").exists()


@pytest.mark.parametrize("kv,scratch", [(0, 0), (1000, 2000)])
def test_nonnegative_reserves_keep_inactive_selection(
    allocator_fixture, tmp_path, monkeypatch, kv, scratch,
):
    selection, config = baseline._run_allocator(
        allocator_fixture, tmp_path, monkeypatch, [
            "--serve-kv-bytes", str(kv), "--serve-peak-scratch-bytes", str(scratch),
        ],
    )

    verdict = selection["serve_constraints"]
    assert verdict["active"] is False
    assert verdict["slos"]["kv_bytes"] == kv
    assert verdict["slos"]["peak_scratch_bytes"] == scratch
    assert {row["data_type"] for name, row in config.items()
            if name != "__prismaquant__"} == {"fp8_e4m3"}


def test_nonnegative_reserves_keep_physical_memory_sum(
    allocator_fixture, tmp_path, monkeypatch,
):
    _module, _probe, _costs, fp8, _nvfp4, table = allocator_fixture
    kv, scratch = 1000, 2000
    budget = fp8 + kv + scratch + 10
    selection, config = baseline._run_allocator(
        allocator_fixture, tmp_path, monkeypatch, [
            "--serve-dispatch-table", str(table),
            "--serve-workload-mix", "prefill:dense=1.0",
            "--serve-device-budget-bytes", str(budget),
            "--serve-kv-bytes", str(kv), "--serve-peak-scratch-bytes", str(scratch),
        ],
    )

    verdict = selection["serve_constraints"]
    assert verdict["active"] is True and verdict["feasible"] is True
    assert verdict["predicted"]["device_memory_bytes"] == fp8 + kv + scratch
    assert {row["data_type"] for name, row in config.items()
            if name != "__prismaquant__"} == {"fp8_e4m3"}
