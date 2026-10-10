"""PQ #2039: the paired copy measure stays honest and its CPU entry point runs.

The measure is the evidence for the staged pinned read, so its own guards are
tested here: it reads every unit through the real staged read, it refuses a run
the stage did not serve, it attributes ``aten::copy_`` to the line that made it,
and the CPU dry run (CEO D38) exercises the same fixture and map binding the
GPU action uses.
"""
import json

import pytest
import torch

from prismaquant.residency_map import (
    ENV_VAR, reset_residency_resolver_for_tests, residency_report,
)
from tools import measure_projected_copy_2039 as measure


@pytest.fixture(autouse=True)
def _forget_resolver(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    reset_residency_resolver_for_tests()
    yield
    reset_residency_resolver_for_tests()


def _arm(kind, wall, *, stage=None, expected=1024, pool=0, fallbacks=0,
         pageable=0, verdicts=("w1 (live)", "w9 (live)"), cpu=0.5, faults=10.0):
    return {"kind": kind, "wall_median_s": wall, "cpu_s_per_pass": cpu,
            "minor_faults_per_pass": faults, "pageable_reads": pageable,
            "stage_expected_bytes": expected,
            "stage_total": {"bytes_from_stage": expected if stage is None else stage,
                            "bytes_from_pool": pool, "fallback_count": fallbacks,
                            "range_hits": 1},
            "mismatch": {"ok": True, "verdicts": list(verdicts)}}


def test_stage_gate_passes_an_arm_the_stage_served_whole():
    arms = [_arm("before", 0.4), _arm("after", 0.2), _arm("after", 0.2), _arm("before", 0.4)]
    assert measure.stage_gate(arms) == []


@pytest.mark.parametrize("damage", [dict(pool=16), dict(fallbacks=1), dict(stage=1000)],
                         ids=["pool-read", "fallback", "short-stage"])
def test_stage_gate_refuses_a_run_the_stage_did_not_serve(damage):
    arms = [_arm("before", 0.4), _arm("after", 0.2, **damage)]
    problems = measure.stage_gate(arms)
    assert problems and problems[0]["arm"] == "after"


def test_stage_gate_refuses_arms_that_name_different_mismatches():
    arms = [_arm("before", 0.4), _arm("after", 0.2, verdicts=("w2 (live)",))]
    assert any("verdicts_differ" in problem for problem in measure.stage_gate(arms))


def test_paired_summary_pools_the_two_arms_of_each_kind():
    arms = [_arm("before", 0.40, pageable=100), _arm("after", 0.18, cpu=0.1),
            _arm("after", 0.22, cpu=0.1), _arm("before", 0.44, pageable=100)]
    summary = measure.paired_summary(arms)
    assert summary["wall_median_s"]["before"] == pytest.approx(0.42)
    assert summary["wall_median_s"]["after"] == pytest.approx(0.20)
    assert summary["wall_median_s"]["reduction_fraction"] == pytest.approx(1 - 0.20 / 0.42)
    assert summary["cpu_s_per_pass"]["reduction_fraction"] == pytest.approx(0.8)
    assert summary["private_staging_copies"] == {"before": 200, "after": 0}
    assert summary["within_kind_spread_fraction"]["before"] == pytest.approx(0.04 / 0.42)


def _frame(ident, parent, name, ts, dur, tid=1):
    return {"ph": "X", "cat": "python_function", "name": name, "tid": tid,
            "ts": ts, "dur": dur, "args": {"Python id": ident, "Python parent id": parent}}


def _event(category, name, ts, dur, tid=1):
    return {"ph": "X", "cat": category, "name": name, "tid": tid,
            "ts": ts, "dur": dur, "args": {}}


PREPARE = "/w/prismaquant/tessera_campaign.py(5451): _prepare_device_projected_check"
LAUNCH = "/w/prismaquant/tessera_campaign.py(5500): _launch_prepared_projected_check"


def _synthetic_trace():
    return [
        _frame(1, None, "/w/tools/run.py(10): outer", 0, 1000),
        _frame(2, 1, PREPARE, 100, 400),
        _frame(3, 2, "torch/_tensor.py(77): wrapper", 150, 100),
        _frame(4, 3, "<built-in method copy_ of Tensor object>", 160, 60),
        _frame(5, 1, LAUNCH, 600, 200),
        # The private staging copy: a plain host memcpy, no runtime call inside.
        _event("cpu_op", "aten::copy_", 170, 30),
        # The H2D copy: the enqueue is a cudaMemcpyAsync nested in the op.
        _event("cpu_op", "aten::copy_", 650, 50),
        _event("cuda_runtime", "cudaMemcpyAsync", 655, 20),
        # Another op, and a copy on a thread with no Python frame.
        _event("cpu_op", "aten::ne", 300, 10),
        _event("cpu_op", "aten::copy_", 650, 40, tid=2),
    ]


def test_trace_call_sites_charge_each_copy_to_the_first_frame_outside_torch():
    result = measure.trace_call_sites(_synthetic_trace())
    rows = {row["site"]: row for row in result["rows"]}
    assert result["python_frames"] == 5
    assert set(rows) == {"tessera_campaign.py:5451 _prepare_device_projected_check",
                         "tessera_campaign.py:5500 _launch_prepared_projected_check",
                         "<no python frame>"}
    staging = rows["tessera_campaign.py:5451 _prepare_device_projected_check"]
    assert (staging["count"], staging["self_cpu_us"], staging["runtime_us"]) == (1, 30, 0)
    launch = rows["tessera_campaign.py:5500 _launch_prepared_projected_check"]
    # 50 us in the op, 20 of them the nested runtime call: 30 us of its own.
    assert (launch["count"], launch["total_cpu_us"]) == (1, 50)
    assert (launch["self_cpu_us"], launch["runtime_us"]) == (30, 20)
    assert sum(row["count"] for row in result["rows"]) == 3


def test_trace_call_sites_rank_the_largest_self_cpu_first_and_other_ops_stay_out():
    rows = measure.trace_call_sites(_synthetic_trace(), op="aten::ne")["rows"]
    assert [row["count"] for row in rows] == [1]
    ranked = measure.trace_call_sites(_synthetic_trace())["rows"]
    assert [row["self_cpu_us"] for row in ranked] == sorted(
        (row["self_cpu_us"] for row in ranked), reverse=True)


def test_trace_call_sites_say_when_the_profiler_recorded_no_python_frames():
    events = [event for event in _synthetic_trace() if event["cat"] != "python_function"]
    result = measure.trace_call_sites(events)
    assert result["python_frames"] == 0
    assert [row["site"] for row in result["rows"]] == ["<no python frame>"]


def _staging_site(destination, source):
    destination.copy_(source)


def test_a_real_profile_attributes_copies_to_their_frames(tmp_path):
    source, target = torch.ones(64, 64), torch.empty(64, 64)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                with_stack=True) as profile:
        for _ in range(3):
            _staging_site(target, source)
    events = measure.trace_events(profile, tmp_path, keep=tmp_path / "kept" / "trace.json")
    result = measure.trace_call_sites(events)
    if not result["python_frames"]:
        pytest.skip("this interpreter's profiler records no Python frames in its trace")
    rows = {row["site"].split()[-1]: row for row in result["rows"]}
    assert rows["_staging_site"]["count"] == 3
    assert rows["_staging_site"]["site"].startswith("test_measure_projected_copy_2039.py:")
    totals = measure.profile_totals(profile, {"aten::copy_"})
    assert totals["aten::copy_"]["count"] == sum(row["count"] for row in result["rows"])
    assert (tmp_path / "kept" / "trace.json").is_file()
    # The scratch directory the trace went through is gone.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["kept"]


def _netdata_document():
    def series(labels, rows):
        return {"points": len(rows), "labels": ["time", *labels], "data": rows}

    def host(busy, power):
        # Netdata hides idle: user + system is the busy share, iowait is apart.
        return {"charts": [], "series": {
            "system.cpu": series(["user", "system", "iowait"],
                                 [[t, busy / 2, busy / 2, 4.0] for t in range(100, 110)]),
            "system.ram": series(["free", "used"], [[t, 5.0, 7.0] for t in range(100, 110)]),
            "system.io": series(["reads", "writes"], [[t, 1.0, -3.0] for t in range(100, 110)]),
            "nvidia_smi.gpu_x_power_draw": series(
                ["power_draw"], [[t, power if t < 105 else None] for t in range(100, 110)])}}

    return {"hosts": {"sparky": host(10.0, 12.0), "sparklina": host(1.0, 4.0)}}


def test_netdata_arm_means_give_each_arm_the_load_of_both_hosts():
    arms = [{"kind": "before", "epoch_start": 100, "epoch_end": 104},
            {"kind": "after", "epoch_start": 105, "epoch_end": 109}]
    rows = measure.netdata_arm_means(_netdata_document(), arms)
    before, after = rows
    assert before["kind"] == "before" and after["kind"] == "after"
    sparky, sparklina = before["hosts"]["sparky"], before["hosts"]["sparklina"]
    assert sparky["cpu_busy_percent"] == pytest.approx(10.0)
    assert sparky["cpu_iowait_percent"] == pytest.approx(4.0)
    assert sparklina["cpu_busy_percent"] == pytest.approx(1.0)
    assert sparky["ram_used_mib"] == 7.0
    assert sparky["io_writes_kib_s"] == 3.0  # Netdata plots writes negative
    assert sparky["gpu_power_w"] == 12.0 and sparklina["gpu_power_w"] == 4.0
    # A row the chart has no reading for does not count as a zero.
    assert sparky["gpu_power_w_rows"] == 5
    assert "gpu_power_w" not in after["hosts"]["sparky"]
    assert after["hosts"]["sparky"]["cpu_busy_percent"] == pytest.approx(10.0)


def test_the_fixture_stages_every_unit_and_the_read_comes_off_the_stage(
        tmp_path, source_bound_reader_sdk):
    fixture = measure.build_fixture(tmp_path / "fx", units=3, rows=8, cols=16)
    assert fixture.names == ["w0", "w1", "w2"] and fixture.unit_bytes == 8 * 16 * 2
    binding = measure.bind_fixture(fixture)
    assert binding["sdk_version"] == 5
    body = measure.run_cpu_dry(fixture)
    assert body["stage_gate_problems"] == []
    assert body["stage"]["bytes_from_stage"] == 3 * fixture.unit_bytes
    assert body["stage"]["bytes_from_pool"] == 0
    assert residency_report()["fallback_count"] == 0


def test_the_dry_run_refuses_staged_bytes_that_differ_from_the_source(
        tmp_path, source_bound_reader_sdk):
    fixture = measure.build_fixture(tmp_path / "fx", units=2, rows=8, cols=16)
    measure.bind_fixture(fixture)
    fixture.tensors["w1"] = fixture.tensors["w1"] + 1
    with pytest.raises(RuntimeError, match="w1: staged bytes differ"):
        measure.run_cpu_dry(fixture)


def test_the_cpu_entry_point_runs_the_same_fixture_and_prints_its_record(
        tmp_path, capsys, source_bound_reader_sdk):
    out = tmp_path / "result.json"
    code = measure.main(["--cpu-dry-run", "--fixture-dir", str(tmp_path),
                         "--out", str(out), "--declared-cpus", "4"])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    record = json.loads(out.read_text())
    assert printed["schema"] == record["schema"] == measure.SCHEMA
    assert printed["dry_run"] is True and printed["stage_gate_problems"] == []
    assert printed["reservations"]["declared"]["cpus"] == 4
    # Clocks and energy stay held; the record says so in its own words.
    assert any("clock alignment" in line for line in printed["hold"])
    assert any("work per joule" in line for line in printed["hold"])
    assert printed["result_file"]["bytes"] == out.stat().st_size
    # The fixture's scratch is removed; only the result stays.
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith("pq2039-")] == []


def test_without_cuda_the_measure_refuses_to_pretend(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert measure.main(["--fixture-dir", str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)["skipped"] is True
    assert list(tmp_path.iterdir()) == []

