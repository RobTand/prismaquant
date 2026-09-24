"""The Stage B capture workspace profile's ladder, admission and receipt (PQ #1151)."""
from __future__ import annotations

import contextlib
import json

import pytest

from prismaquant import stage_b_workspace_profile as profile


def test_ladder_admits_each_step_from_the_one_before():
    assert profile.ladder(4) == [
        ("window-backward-b1", False, 1, ("declared", 1)),
        ("window-backward-b4", False, 4, ("window-backward-b1", 4)),
        ("capture-b1", True, 1, ("declared", 1)),
        ("capture-b2", True, 2, ("capture-b1", 2)),
        ("capture-b4", True, 4, ("capture-b2", 2)),
    ]
    assert profile.ladder(1) == [
        ("window-backward-b1", False, 1, ("declared", 1)),
        ("capture-b1", True, 1, ("declared", 1)),
    ]
    assert [step[2] for step in profile.ladder(3)] == [1, 3, 1, 2, 3]


def test_request_is_opt_in_and_needs_enough_groups(tmp_path):
    assert profile.profile_request({}) is None
    out = tmp_path / "profile.json"
    assert profile.profile_request({profile.PROFILE_ENV: str(out)}) == {
        "path": out, "groups": profile.DEFAULT_GROUPS}
    with pytest.raises(ValueError, match="absolute"):
        profile.profile_request({profile.PROFILE_ENV: "relative.json"})
    with pytest.raises(ValueError, match="at least"):
        profile.profile_request({profile.PROFILE_ENV: str(out), profile.GROUPS_ENV: "4"})


def test_host_sampler_keeps_the_window_peak_with_its_stat(tmp_path):
    (tmp_path / "memory.current").write_text("100\n")
    (tmp_path / "memory.stat").write_text("anon 60\nfile 40\nshmem 0\n")
    sampler = profile.HostPeakSampler(tmp_path, interval_s=0.001)
    sampler.open()
    (tmp_path / "memory.current").write_text("300\n")
    (tmp_path / "memory.stat").write_text("anon 200\nfile 100\nfile_dirty 7\n")
    # One sample by hand: the thread is not needed to test the bookkeeping.
    current, stat = sampler._read()
    with sampler._lock:
        sampler._window["peak"] = {"unix": 1.0, "current": current, "stat": stat}
    (tmp_path / "memory.current").write_text("150\n")
    (tmp_path / "memory.stat").write_text("anon 100\nfile 50\n")
    closed = sampler.close()
    assert closed["start_current_bytes"] == 100
    assert closed["peak_current_bytes"] == 300
    assert closed["peak_stat"]["anon"] == 200
    assert closed["peak_stat"]["file_dirty"] == 7
    assert closed["peak_stat"]["shmem"] is None


class _Guard:
    device_bytes = 1000

    def __init__(self, refuse_label):
        self.refuse_label = refuse_label
        self.charged = []
        self.last = None

    def check(self, label, *, reserve_bytes=0, reserve_device_bytes=0):
        self.charged.append((label, reserve_bytes, reserve_device_bytes))
        self.last = {"label": label}
        if label == self.refuse_label:
            raise RuntimeError(f"refused {label}")
        return dict(self.last)


def _fake_measure(peaks, fail_at=None):
    calls = []

    def measure(run, group, *, device, host):
        run(group)
        batch = len(group)
        calls.append(batch)
        failure = ("OutOfMemoryError: pass stopped"
                   if fail_at is not None and (batch, calls.count(batch)) == fail_at else None)
        return {"indices": [item[0] for item in group], "allocated_delta_bytes": peaks[batch],
                "reserved_delta_bytes": peaks[batch] - 1,
                "host": {"peak_current_bytes": 5}, "failure": failure}
    return measure


def _run(monkeypatch, *, refuse_label, peaks, fail_at=None):
    monkeypatch.setattr(profile, "measure_group", _fake_measure(peaks, fail_at))
    monkeypatch.setattr("prismaquant.aura_cost._release_streamed_anchor_allocator_cache",
                        lambda device: None)
    runs, contexts = [], []

    @contextlib.contextmanager
    def observed():
        contexts.append("entered")
        try:
            yield "observer"
        except BaseException:
            contexts.append("left-by-exception")
            raise
        contexts.append("left-normally")

    guard = _Guard(refuse_label)
    items = [(index, None, None, None) for index in range(64)]
    settings, complete = profile.run_ladder(
        items, run_group=lambda group, observer: runs.append((len(group), observer)),
        observed=observed, guard=guard, declared_workspace_bytes=100,
        capture_reserve_bytes=3, capture_batch=4, groups=16, device="cpu", host=None)
    return settings, complete, guard, runs, contexts


def test_run_ladder_charges_the_rule_and_never_ends_the_probe(monkeypatch):
    peaks = {1: 10, 2: 18, 4: 30}
    settings, complete, guard, runs, contexts = _run(
        monkeypatch, refuse_label=None, peaks=peaks)
    assert complete
    assert [setting["name"] for setting in settings] == [
        "window-backward-b1", "window-backward-b4", "capture-b1", "capture-b2", "capture-b4"]
    # Declared reserve for both B=1 steps; the measured peak times the ratio after.
    assert [charge[2] for charge in guard.charged] == [100, 10 * 4, 100 + 3, 10 * 2 + 3,
                                                       18 * 2 + 3]
    assert all(charge[1] == 0 for charge in guard.charged)
    # 16 groups per setting, of the setting's own batch.
    assert [len([run for run in runs if run[0] == batch and run[1] is None])
            for batch in (1, 4)] == [16, 16]
    assert [len([run for run in runs if run[0] == batch and run[1] == "observer"])
            for batch in (1, 2, 4)] == [16, 16, 16]
    assert contexts == ["entered", "left-by-exception"]
    by_name = {setting["name"]: setting for setting in settings}
    assert by_name["capture-b4"]["peak_bytes"] == 30
    assert by_name["capture-b4"]["peak_per_batch_bytes"] == 8
    assert by_name["capture-b2"]["allocated_delta"]["spread"] == 0


def test_a_refused_step_ends_the_ladder_and_is_recorded(monkeypatch):
    peaks = {1: 10, 2: 18, 4: 30}
    settings, complete, guard, runs, contexts = _run(
        monkeypatch, refuse_label="stage_b_workspace_profile:capture-b4", peaks=peaks)
    assert not complete
    assert [setting["name"] for setting in settings][-1] == "capture-b4"
    refused = settings[-1]
    assert refused["admission"]["admitted"] is False
    assert "refused" in refused["admission"]["refusal"]
    assert "groups" not in refused
    assert not [run for run in runs if run[0] == 4 and run[1] == "observer"]
    assert contexts == ["entered", "left-by-exception"]


def test_a_pass_that_stops_prices_nothing_and_ends_the_ladder(monkeypatch):
    # The window backward at B=4 runs 16 passes; the capture at B=4 is the
    # fifth setting, and its third pass stops.
    peaks = {1: 10, 2: 18, 4: 30}
    settings, complete, guard, runs, contexts = _run(
        monkeypatch, refuse_label=None, peaks=peaks, fail_at=(4, 16 + 3))
    assert not complete
    stopped = settings[-1]
    assert stopped["name"] == "capture-b4"
    assert stopped["failure"].startswith("OutOfMemoryError")
    assert len(stopped["groups"]) == 3
    assert stopped["lower_bound_peak_bytes"] == 30
    assert "peak_bytes" not in stopped and "peak_per_batch_bytes" not in stopped
    assert contexts == ["entered", "left-by-exception"]


def test_the_stop_is_found_through_a_wrapping_exception(tmp_path):
    stop = profile.CaptureWorkspaceProfiled(tmp_path / "p.json", "ab" * 32)
    try:
        try:
            raise stop
        except profile.CaptureWorkspaceProfiled as exc:
            raise RuntimeError("wrapped") from exc
    except RuntimeError as wrapped:
        assert profile.CaptureWorkspaceProfiled.found_in(wrapped) is stop
    assert profile.CaptureWorkspaceProfiled.found_in(ValueError("other")) is None


def test_receipt_is_canonical_json_with_its_digest(tmp_path):
    path = tmp_path / "nested" / "profile.json"
    digest = profile.write_profile(path, {"b": 1, "a": {"x": tmp_path}})
    raw = path.read_bytes()
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == digest
    assert json.loads(raw) == {"a": {"x": str(tmp_path)}, "b": 1}


def test_tool_moves_only_the_output_space(tmp_path):
    from experiments.stage_b_capture_workspace_profile import moved_output_space

    record = {"layer": 44, "quantum_id": "layer-044", "output_space": {"root": "/campaign"},
              "adjoint": {"slice_sha256": "x"}}
    moved = moved_output_space(record, tmp_path)
    assert record["output_space"] == {"root": "/campaign"}
    assert moved["adjoint"] == record["adjoint"]
    assert moved["output_space"]["root"] == str(tmp_path / "layer-044")
    assert moved["output_space"]["counters"] == str(tmp_path / "layer-044" / "counters.json")
