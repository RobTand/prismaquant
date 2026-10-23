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


def test_the_control_summary_drops_the_staged_copy_count_and_none_means_it_did_not_run():
    arms = [_arm("before", 0.40, pageable=100), _arm("after", 0.40), _arm("after", 0.40),
            _arm("before", 0.40, pageable=100)]
    summary = measure.control_summary(arms)
    # A pool read pins after the read: it returns pinned, so zero would lie.
    assert "private_staging_copies" not in summary
    assert summary["wall_median_s"]["reduction_fraction"] == pytest.approx(0.0)
    assert measure.control_summary([]) is None


def test_retained_trace_collision_refuses_without_replacing_evidence(tmp_path):
    class Profile:
        def export_chrome_trace(self, path):
            from pathlib import Path
            Path(path).write_text(json.dumps({"traceEvents": []}))

    keep = tmp_path / "kept" / "trace.json"
    assert measure.trace_events(Profile(), tmp_path, keep=keep) == []
    original = keep.read_bytes()
    with pytest.raises(FileExistsError):
        measure.trace_events(Profile(), tmp_path, keep=keep)
    assert keep.read_bytes() == original
    assert list(tmp_path.iterdir()) == [keep.parent]


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
        tmp_path, installed_client_sdk):
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
        tmp_path, installed_client_sdk):
    fixture = measure.build_fixture(tmp_path / "fx", units=2, rows=8, cols=16)
    measure.bind_fixture(fixture)
    fixture.tensors["w1"] = fixture.tensors["w1"] + 1
    with pytest.raises(RuntimeError, match="w1: staged bytes differ"):
        measure.run_cpu_dry(fixture)


def test_the_cpu_entry_point_runs_the_same_fixture_and_prints_its_record(
        tmp_path, capsys, installed_client_sdk):
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


def test_fixture_scratch_does_not_reuse_or_remove_an_existing_pid_directory(
        tmp_path, capsys, installed_client_sdk):
    prior = tmp_path / f"pq2039-{measure.os.getpid()}"
    prior.mkdir()
    marker = prior / "existing-evidence"
    marker.write_bytes(b"preserve")
    assert measure.main(["--cpu-dry-run", "--fixture-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    assert marker.read_bytes() == b"preserve"
    assert list(tmp_path.iterdir()) == [prior]


def test_without_cuda_the_measure_refuses_to_pretend(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert measure.main(["--fixture-dir", str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)["skipped"] is True
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("units", [0, 1, 2, -1])
@pytest.mark.parametrize("dry", [False, True])
def test_small_unit_counts_reject_before_cuda_or_fixture(
        units, dry, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid units reached CUDA or fixture construction")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(measure, "build_fixture", forbidden)
    argv = ["--units", str(units), "--fixture-dir", str(tmp_path)]
    if dry:
        argv.append("--cpu-dry-run")
    with pytest.raises(SystemExit) as error:
        measure.main(argv)
    assert error.value.code == 2
    assert "--units must be at least 3" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("units", [3, 4])
def test_cpu_dry_run_never_probes_cuda(
        units, tmp_path, monkeypatch, capsys, installed_client_sdk):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU dry run queried CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert measure.main(["--cpu-dry-run", "--units", str(units),
                         "--fixture-dir", str(tmp_path)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["reads"] == units
    assert record["stage"]["bytes_from_pool"] == 0
    assert record["stage_gate_problems"] == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("picks", [(1, 1), (3,), (-1,), ()])
def test_invalid_mismatch_picks_do_not_mutate_live_tensors(picks):
    from types import SimpleNamespace

    fixture = SimpleNamespace(names=["w0", "w1", "w2"])
    live = {name: torch.zeros(2) for name in fixture.names}
    with pytest.raises(ValueError, match="distinct valid unit indices"):
        measure.mismatch_pass(None, fixture, live, measure.Phases(), picks)
    assert all(torch.equal(tensor, torch.zeros(2)) for tensor in live.values())


@pytest.mark.parametrize("fail", [False, True])
def test_mismatch_control_restores_distinct_picks_even_on_check_failure(monkeypatch, fail):
    from types import SimpleNamespace

    fixture = SimpleNamespace(names=["w0", "w1", "w2"])
    live = {name: torch.zeros(2) for name in fixture.names}

    def check(*args):
        assert live["w0"][0] == 0
        assert live["w1"][0] == live["w2"][0] == 1
        if fail:
            raise RuntimeError("check failed")
        return 0.0, [None, "w1 (live)", "w2 (live)"]

    monkeypatch.setattr(measure, "run_projected_check_pass", check)
    if fail:
        with pytest.raises(RuntimeError, match="check failed"):
            measure.mismatch_pass(None, fixture, live, measure.Phases(), (1, 2))
    else:
        result = measure.mismatch_pass(None, fixture, live, measure.Phases(), (1, 2))
        assert result == {"picks": ["w1", "w2"],
                          "verdicts": ["w1 (live)", "w2 (live)"], "ok": True}
    assert all(torch.equal(tensor, torch.zeros(2)) for tensor in live.values())


def test_sdk_binding_uses_current_guarded_owner(tmp_path, monkeypatch):
    from prismaquant import staged_lease

    fixture = measure.build_fixture(tmp_path / "fx", units=3, rows=8, cols=16)

    def unavailable():
        raise RuntimeError("reviewed SDK unavailable")

    monkeypatch.setattr(staged_lease, "client_sdk", unavailable)
    with pytest.raises(RuntimeError, match="reviewed SDK unavailable"):
        measure.bind_fixture(fixture)
    assert measure.stage_counters()["bytes_from_pool"] == 0


@pytest.mark.parametrize("kind,pinned", [("before", False), ("after", True)])
def test_instrument_changes_only_the_pin_request_and_restores_seams(kind, pinned):
    from types import SimpleNamespace

    seen = []
    weight = SimpleNamespace(is_pinned=lambda: pinned)
    release = object()

    def read(name, unit, **kwargs):
        seen.append((name, unit, kwargs))
        return weight, release

    def prepare(*args, **kwargs):
        return args, kwargs

    def launch(*args, **kwargs):
        raise RuntimeError("launch failed")

    campaign = SimpleNamespace(_read_projected_unit=read,
                               _prepare_device_projected_check=prepare,
                               _launch_prepared_projected_check=launch)
    phases = measure.Phases()
    restore = measure.instrument(campaign, kind, phases)
    try:
        assert campaign._read_projected_unit("unit", {}, pinned_host=True,
                                             source_authentication="guard") == (weight, release)
        assert seen == [("unit", {}, {"pinned_host": pinned,
                                       "source_authentication": "guard"})]
        assert campaign._prepare_device_projected_check("arg", flag=True) == (
            ("arg",), {"flag": True})
        with pytest.raises(RuntimeError, match="launch failed"):
            campaign._launch_prepared_projected_check()
    finally:
        restore()
    assert campaign._read_projected_unit is read
    assert campaign._prepare_device_projected_check is prepare
    assert campaign._launch_prepared_projected_check is launch
    assert phases.pinned_reads == int(pinned)
    assert phases.pageable_reads == int(not pinned)


def test_power_envelope_is_configurable_without_sampling():
    sampler = measure.PowerSampler(envelope_w=200.0)
    sampler.samples = [(100, 50.0), (101, 70.0)]
    assert sampler.window(100, 101)["envelope_fraction_mean"] == pytest.approx(0.3)


@pytest.mark.parametrize("option,value", [
    ("--rows", "0"), ("--cols", "-1"), ("--passes", "-1"),
    ("--control-passes", "-1"), ("--min-passes", "0"),
    ("--profile-passes", "0"), ("--arm-seconds", "nan"),
    ("--power-envelope-w", "0"),
])
def test_invalid_measure_arguments_refuse_before_cuda(option, value, monkeypatch):
    def forbidden():
        pytest.fail("invalid arguments reached CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    with pytest.raises(SystemExit) as error:
        measure.main([option, value])
    assert error.value.code == 2

