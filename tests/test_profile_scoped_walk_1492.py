"""The scoped head walk measures I/O width on a bounded, read-only sample (#1492, #1247)."""
from __future__ import annotations

import threading

import pytest

from prismaquant import io_spans, tessera_joint_aura as bridge
from tools import profile_stage_b_head as head


def _mountstats(monkeypatch, fields):
    rows = {"/mnt/shared": {"bytes": None, "ops": {} if fields is None else {"READ": fields}}}
    monkeypatch.setattr(io_spans, "read_mountstats", lambda _path=None: rows)


def test_read_rtt_reads_the_ops_and_round_trip_columns(monkeypatch):
    _mountstats(monkeypatch, [10, 10, 0, 100, 200, 3, 40, 50])
    assert head.read_rtt() == (10, 40)


def test_read_rtt_of_a_mount_without_reads_is_zero(monkeypatch):
    _mountstats(monkeypatch, None)
    assert head.read_rtt() == (0, 0)
    assert head.read_rtt("/absent") == (0, 0)


def test_interval_mean_is_round_trip_per_read():
    assert head.interval_read_rtt_ms((100, 1000), (220, 1600)) == 5.0


def test_interval_mean_ignores_a_quiet_interval():
    assert head.interval_read_rtt_ms((100, 1000), (110, 9000), min_ops=20) is None


def _guard(samples, limit_ms, stops):
    feed = iter(samples)
    last = [samples[-1]]

    def sample():
        last[0] = next(feed, last[0])
        return last[0]

    return head.start_read_guard(limit_ms, interval_s=0.01, min_ops=20,
                                 sample=sample, on_stop=stops.append)


def test_guard_stops_when_the_round_trip_rises_past_the_limit():
    stopped = []
    event = threading.Event()
    stops = type("S", (), {"append": lambda self, v: (stopped.append(v), event.set())})()
    _guard([(0, 0), (100, 500), (200, 2500)], 10.0, stops)
    assert event.wait(5), "the guard never fired"
    assert stopped == [20.0]


def test_guard_does_not_stop_at_a_normal_round_trip_and_records_what_it_saw():
    stopped = []
    stop, trace = _guard([(0, 0), (100, 500), (200, 1000), (300, 1500)], 10.0, stopped)
    deadline = threading.Event()
    deadline.wait(0.3)
    stop.set()
    assert stopped == []
    assert trace[:3] == [5.0, 5.0, 5.0]


def test_scope_over_the_cap_is_refused_before_any_read(monkeypatch):
    monkeypatch.setattr(bridge, "load_measured_anchor_input",
                        lambda *a, **k: pytest.fail("the loader must not run"))
    with pytest.raises(SystemExit, match="the limit is 2000"):
        head.scoped_walk_intake({"inputs": {}}, scope=(0, head.SCOPED_WALK_MAX_UNITS + 1),
                                workers=4)
    with pytest.raises(SystemExit, match="empty or reversed"):
        head.scoped_walk_intake({"inputs": {}}, scope=(5, 5), workers=4)


def test_scoped_walk_passes_the_scope_and_width_and_stays_read_only(monkeypatch):
    from prismaquant import tessera_reader
    seen = {}

    class Data:
        head_walk_workers = 4
        cells = [("a", "F"), ("a", "G"), ("b", "F")]

    def fake_load(inputs, **kwargs):
        seen.update(inputs=inputs, **kwargs)
        return Data()

    monkeypatch.setattr(bridge, "load_measured_anchor_input", fake_load)
    monkeypatch.setattr(tessera_reader, "load_declared_reader", lambda spec: "reader")
    result = head.scoped_walk_intake({"inputs": {"k": 1}, "reader": None},
                                     scope=(100, 600), workers=4)
    assert seen["unit_scope"] == (100, 600)
    assert seen["head_walk_workers"] == 4
    assert seen["head_checkpoint"] is None and seen["head_resume"] is False
    assert seen["verify_payloads"] is False
    assert seen["require_existing_renders"] is True
    assert seen["progress_phase"] is None
    assert seen["inputs"] == {"k": 1}
    assert result["scope"] == [100, 600] and result["scope_units"] == 500
    assert result["head_walk_workers"] == 4
    assert result["measured_cells"] == 3 and result["units_with_cells"] == 2


def _argv(tmp_path, *extra, mode="scoped-walk"):
    return ["--mode", mode, "--quantum", str(tmp_path / "q.json"), "--quantum-sha256", "0" * 64,
            "--plan", str(tmp_path / "p.json"), "--plan-sha256", "0" * 64,
            "--scratch", str(tmp_path / "scratch"), *extra]


@pytest.mark.parametrize("extra,message", [
    ((), "explicit end"),
    (("--unit-scope", "0:2001", "--head-walk-workers", "4"), "at most 2000"),
    (("--unit-scope", "0:", "--head-walk-workers", "4"), "explicit end"),
    (("--unit-scope", "5:5", "--head-walk-workers", "4"), "empty or reversed"),
    (("--unit-scope", "0:100"), "head-walk-workers"),
    (("--unit-scope", "0:100", "--head-walk-workers", "17"), "head-walk-workers"),
    (("--unit-scope", "0:100", "--head-walk-workers", "0"), "head-walk-workers"),
    (("--unit-scope", "0:100", "--head-walk-workers", "4", "--stop-read-rtt-ms", "0"),
     "guard limits"),
])
def test_main_refuses_a_bad_scoped_call_before_reading_anything(tmp_path, capsys, extra, message):
    with pytest.raises(SystemExit) as exit_info:
        head.main(_argv(tmp_path, *extra))
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "scratch").exists(), "a refused call creates nothing"


def test_the_scoped_options_belong_to_the_scoped_mode(tmp_path, capsys):
    with pytest.raises(SystemExit):
        head.main(_argv(tmp_path, "--unit-scope", "0:100", mode="walk"))
    assert "belong to --mode scoped-walk" in capsys.readouterr().err


def test_sweep_plan_puts_a_one_unit_baseline_first_and_keeps_the_worker_order():
    scopes = head.sweep_plan(1000, 250, [4, 1, 8, 2])
    assert scopes[0] == {"label": "baseline", "scope": (1000, 1001), "workers": 4}
    assert [item["workers"] for item in scopes[1:]] == [4, 1, 8, 2]
    assert [item["label"] for item in scopes] == ["baseline", "run0", "run1", "run2", "run3"]


def test_sweep_slices_are_disjoint_and_adjacent():
    scopes = head.sweep_plan(0, 250, [4, 1, 8, 2, 2, 8, 1, 4])
    edges = [item["scope"] for item in scopes]
    for (_, high), (low, _) in zip(edges, edges[1:]):
        assert low == high
    assert len({unit for low, high in edges for unit in range(low, high)}) == 1 + 8 * 250


def test_sweep_holds_one_budget_for_the_whole_sweep():
    assert len(head.sweep_plan(0, 250, [1, 2, 4, 8, 8, 4, 2])) == 8
    with pytest.raises(SystemExit, match="limit is 2000 for the whole sweep"):
        head.sweep_plan(0, 250, [1, 2, 4, 8, 8, 4, 2, 1])
    with pytest.raises(SystemExit, match="limit is 2000 for the whole sweep"):
        head.sweep_plan(0, 500, [4, 1, 8, 2])
    with pytest.raises(SystemExit, match="at least one worker count"):
        head.sweep_plan(0, 250, [])


def test_sweep_runs_each_scope_once_in_order_and_reports_read_load(monkeypatch):
    calls = []
    counters = iter([(0, 0), (10, 100), (10, 100), (30, 700)])
    monkeypatch.setattr(head, "mount_ops", lambda mount_point="/mnt/shared": {"READ": 1})
    monkeypatch.setattr(head, "read_rtt", lambda mount_point="/mnt/shared": next(counters))

    def fake_walk(config, *, scope, workers):
        calls.append((scope, workers))
        return {"units_with_cells": scope[1] - scope[0], "measured_cells": 7}

    report = head.scoped_walk_sweep({"inputs": {}}, start=0, slice_units=250,
                                    workers_order=[4], walk=fake_walk)
    assert calls == [((0, 1), 4), ((1, 251), 4)]
    assert [run["label"] for run in report["runs"]] == ["baseline", "run0"]
    assert report["runs"][0]["read_ops"] == 10
    assert report["runs"][0]["read_rtt_ms_mean"] == 10.0
    assert report["runs"][1]["read_ops"] == 20
    assert report["runs"][1]["read_rtt_ms_mean"] == 30.0
    assert report["sweep"]["units_total"] == 251


@pytest.mark.parametrize("extra,message", [
    (("--sweep-start", "0", "--slice-units", "500", "--sweep-workers", "4,1,8,2"),
     "limit is 2000 for the whole sweep"),
    (("--sweep-start", "0", "--slice-units", "250"), "needs --sweep-start, --slice-units and --sweep-workers"),
    (("--sweep-start", "0", "--slice-units", "250", "--sweep-workers", "4,x"), "integers"),
    (("--sweep-start", "0", "--slice-units", "250", "--sweep-workers", "4,17"), "counts in 1:16"),
    (("--sweep-start", "0", "--slice-units", "250", "--sweep-workers", "4",
      "--unit-scope", "0:10", "--head-walk-workers", "2"), "either"),
])
def test_main_refuses_a_bad_sweep_before_reading_anything(tmp_path, capsys, extra, message):
    with pytest.raises(SystemExit) as exit_info:
        head.main(_argv(tmp_path, *extra))
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "scratch").exists()
