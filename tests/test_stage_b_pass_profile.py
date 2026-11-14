"""PQ #1269: the Stage B pass profiler is off by default and bounded when on."""
import json
from pathlib import Path

import pytest
import torch

from prismaquant.stage_b_pass_profile import (
    PROFILE_ENV,
    SPEC_ENV,
    pass_profile_request,
)


def test_off_without_the_env():
    assert pass_profile_request({}) is None
    assert pass_profile_request({SPEC_ENV: "capture=1"}) is None


def test_spec_parses_probes_stack_and_schedules(tmp_path):
    request = pass_profile_request({
        PROFILE_ENV: str(tmp_path),
        SPEC_ENV: "capture=1+2:stack,windowed=none,wait=3,active=1,shadow_batches=5"})
    assert request.capture_probes == {1: False, 2: True}
    assert request.windowed_probe is None
    assert (request.wait, request.warmup, request.active) == (3, 1, 1)
    assert request.shadow_batches == 5
    assert request.profiles_capture(2) and not request.profiles_capture(0)


def test_defaults(tmp_path):
    request = pass_profile_request({PROFILE_ENV: str(tmp_path)})
    assert request.capture_probes == {1: False}
    assert request.windowed_probe == 1
    assert request.shadow_batches == 48
    assert request.render_window is None and not request.profiles_render(0)


def test_spec_parses_a_render_window_and_its_schedule(tmp_path):
    request = pass_profile_request({
        PROFILE_ENV: str(tmp_path),
        SPEC_ENV: "capture=,windowed=none,render=5:stack,render_wait=0,render_active=2"})
    assert request.capture_probes == {} and request.windowed_probe is None
    assert request.render_window == 5 and request.render_stack is True
    assert (request.render_wait, request.render_warmup, request.render_active) == (0, 1, 2)
    assert request.profiles_render(5) and not request.profiles_render(4)


def test_render_zero_refuses_nested_capture_sessions(tmp_path):
    # Window zero holds the spill captures; two profilers cannot nest.
    with pytest.raises(ValueError, match="render=0"):
        pass_profile_request({PROFILE_ENV: str(tmp_path), SPEC_ENV: "render=0"})
    request = pass_profile_request({
        PROFILE_ENV: str(tmp_path), SPEC_ENV: "capture=,windowed=none,render=0"})
    assert request.profiles_render(0)


def test_render_session_names_its_window_and_records_counter_deltas(tmp_path):
    request = pass_profile_request({
        PROFILE_ENV: str(tmp_path),
        SPEC_ENV: "capture=,windowed=none,render=3,render_wait=1,render_warmup=0"})
    counters = {"reader_wait_s": 0.0, "read_calls": 0}
    session = request.session(kind="render", window=3, counters=lambda: counters,
                              identity={"quantum_id": "layer-007"})
    with session:
        for probe in range(3):
            session.unit_begin()
            counters["reader_wait_s"] += 0.5
            counters["read_calls"] += probe + 1
            session.unit_end()
    timing = json.loads((tmp_path / "layer-007-w3-render.timing.json").read_text())
    assert (timing["kind"], timing["window"], timing["probe"]) == ("render", 3, None)
    assert [u["profiler"] for u in timing["units"]] == ["wait", "active", "after"]
    assert [u["counter_deltas"] for u in timing["units"]] == [
        {"reader_wait_s": 0.5, "read_calls": 1},
        {"reader_wait_s": 0.5, "read_calls": 2},
        {"reader_wait_s": 0.5, "read_calls": 3}]
    assert (tmp_path / "layer-007-w3-render.trace.json.gz").exists()


def test_a_render_session_names_its_window(tmp_path):
    request = pass_profile_request({PROFILE_ENV: str(tmp_path)})
    with pytest.raises(ValueError, match="window"):
        request.session(kind="render", identity={})


@pytest.mark.parametrize("spec", ["bogus=1", "capture=1:deep", "active=0", "nokey",
                                  "render=2:deep", "render=2,render_active=0"])
def test_malformed_spec_refuses(tmp_path, spec):
    with pytest.raises(ValueError):
        pass_profile_request({PROFILE_ENV: str(tmp_path), SPEC_ENV: spec})


def test_capture_session_times_every_unit_and_traces_the_active_ones(tmp_path):
    request = pass_profile_request({PROFILE_ENV: str(tmp_path),
                                    SPEC_ENV: "capture=0,wait=2,warmup=1,active=2"})
    session = request.session(kind="capture", probe=0,
                              identity={"quantum_id": "layer-007"})
    with session:
        for _ in range(7):
            session.unit_begin()
            a = torch.randn(64, 64)
            (a @ a).sum()
            session.unit_end()
    timing = json.loads((tmp_path / "layer-007-p0-capture.timing.json").read_text())
    assert timing["units_count"] == 7
    assert [u["profiler"] for u in timing["units"]] == [
        "wait", "wait", "warmup", "active", "active", "after", "after"]
    assert timing["units_wall_sum_s"] + timing["gaps_sum_s"] <= timing["pass_wall_s"]
    assert (tmp_path / "layer-007-p0-capture.trace.json.gz").exists()
    assert (tmp_path / "layer-007-p0-capture.key_averages.txt").exists()
    assert timing["errors"] == []


def test_shadow_session_stops_after_its_bound(tmp_path):
    request = pass_profile_request({
        PROFILE_ENV: str(tmp_path),
        SPEC_ENV: "shadow_batches=4,shadow_wait=1,shadow_warmup=1,shadow_active=1"})
    session = request.session(kind="windowed", probe=1,
                              identity={"quantum_id": "layer-007"}, stop_after=4)
    ran = 0
    with session:
        for _ in range(100):
            session.unit_begin()
            ran += 1
            session.unit_end()
    assert ran == 4
    timing = json.loads((tmp_path / "layer-007-p1-windowed.timing.json").read_text())
    assert timing["stopped_early"] is True and timing["units_count"] == 4


def test_other_errors_propagate(tmp_path):
    request = pass_profile_request({PROFILE_ENV: str(tmp_path)})
    session = request.session(kind="capture", probe=1, identity={})
    with pytest.raises(RuntimeError, match="boom"):
        with session:
            raise RuntimeError("boom")
    timing = json.loads((tmp_path / "quantum-p1-capture.timing.json").read_text())
    assert timing["ended_by"] == "RuntimeError"


def _memory_request(tmp_path, spec):
    """Parse a spec; a refusal gives None, so that a test asserts on it."""
    try:
        return pass_profile_request({PROFILE_ENV: str(tmp_path), SPEC_ENV: spec})
    except ValueError:
        return None


class _MemorySpy:
    """Stands in for the two CUDA allocator history calls."""

    def __init__(self):
        self.calls = []
        self.on = False
        self.fail_dump = False

    def record(self, enabled, **kwargs):
        self.calls.append(("record", enabled, kwargs))
        self.on = enabled is not None

    def dump(self, path, *args, **kwargs):
        self.calls.append(("dump", str(path)))
        if self.fail_dump:
            raise OSError("disk full")
        Path(path).write_bytes(b"spy")


@pytest.fixture
def memory_spy(monkeypatch):
    spy = _MemorySpy()
    monkeypatch.setattr(torch.cuda.memory, "_record_memory_history", spy.record)
    monkeypatch.setattr(torch.cuda.memory, "_dump_snapshot", spy.dump)
    return spy


def _run_memory_groups(session, monkeypatch, spy, *, cuda=True, groups=3, fail_in=None):
    """Run capture groups; return whether the history was on inside each."""
    seen = []
    with session:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        for index in range(groups):
            session.unit_begin()
            seen.append(spy.on)
            if index == fail_in:
                raise RuntimeError("capture failed")
            session.unit_end()
    return seen


def _memory_timing(tmp_path):
    return json.loads((tmp_path / "quantum-p1-capture.timing.json").read_text())


def test_memory_keys_default_to_off(tmp_path):
    request = _memory_request(tmp_path, "")
    assert request is not None
    assert getattr(request, "memory_group", "missing") is None
    entries = getattr(request, "memory_entries", None)
    assert isinstance(entries, int) and entries >= 1


def test_memory_keys_parse(tmp_path):
    request = _memory_request(tmp_path, "memory_group=2,memory_entries=50")
    assert request is not None, "memory_group and memory_entries are spec keys"
    assert request.memory_group == 2 and request.memory_entries == 50
    request = _memory_request(tmp_path, "memory_group=none,memory_entries=1000000")
    assert request is not None
    assert request.memory_group is None and request.memory_entries == 1000000
    request = _memory_request(tmp_path, "memory_group=0")
    assert request is not None and request.memory_group == 0


@pytest.mark.parametrize("spec", ["memory_group=-1", "memory_group=x",
                                  "memory_group=1.5", "memory_entries=0",
                                  "memory_entries=abc", "memory_entries=1000001"])
def test_memory_keys_refuse_bad_values(tmp_path, spec):
    assert _memory_request(tmp_path, "memory_group=1,memory_entries=5") is not None
    with pytest.raises(ValueError):
        pass_profile_request({PROFILE_ENV: str(tmp_path), SPEC_ENV: spec})


def test_memory_group_outside_the_pass_refuses(tmp_path):
    # The quantum puts its group count in the identity. A session without it
    # cannot know the length, so it reports not_reached (see the test below).
    request = _memory_request(tmp_path, "memory_group=3")
    assert request is not None
    with pytest.raises(ValueError, match="memory_group"):
        request.session(kind="capture", probe=1, identity={"capture_groups": 3})
    request.session(kind="capture", probe=1, identity={"capture_groups": 4})


def test_memory_history_is_not_called_when_unset(tmp_path, monkeypatch, memory_spy):
    request = pass_profile_request({PROFILE_ENV: str(tmp_path)})
    session = request.session(kind="capture", probe=1, identity={})
    seen = _run_memory_groups(session, monkeypatch, memory_spy)
    assert seen == [False, False, False]
    assert memory_spy.calls == []
    assert "memory" not in _memory_timing(tmp_path)


def test_memory_history_runs_only_in_the_selected_group(tmp_path, monkeypatch, memory_spy):
    request = _memory_request(tmp_path, "memory_group=1,memory_entries=50")
    assert request is not None
    session = request.session(kind="capture", probe=1, identity={"capture_groups": 3})
    seen = _run_memory_groups(session, monkeypatch, memory_spy)
    assert seen == [False, True, False]
    kind, enabled, kwargs = memory_spy.calls[0]
    assert (kind, enabled) == ("record", "all")
    assert kwargs["max_entries"] == 50
    assert kwargs["context"] == "alloc" and kwargs["stacks"] == "python"
    stops = [call for call in memory_spy.calls if call[:2] == ("record", None)]
    assert len(stops) == 1


def test_memory_snapshot_and_complete_status(tmp_path, monkeypatch, memory_spy):
    request = _memory_request(tmp_path, "memory_group=1,memory_entries=50")
    assert request is not None
    session = request.session(kind="capture", probe=1, identity={"capture_groups": 3})
    _run_memory_groups(session, monkeypatch, memory_spy)
    snapshot = tmp_path / "quantum-p1-capture.memory_snapshot.pickle"
    assert snapshot.exists()
    memory = _memory_timing(tmp_path)["memory"]
    assert memory["status"] == "complete"
    assert memory["group"] == 1 and memory["entries"] == 50


def test_memory_history_stops_after_a_good_group(tmp_path, monkeypatch, memory_spy):
    request = _memory_request(tmp_path, "memory_group=0")
    assert request is not None
    session = request.session(kind="capture", probe=1, identity={})
    with session:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        session.unit_begin()
        assert memory_spy.on is True
        session.unit_end()
        assert memory_spy.on is False
        assert memory_spy.calls[-2][0] == "dump"


def test_memory_history_stops_when_the_capture_raises(tmp_path, monkeypatch, memory_spy):
    request = _memory_request(tmp_path, "memory_group=1")
    assert request is not None
    session = request.session(kind="capture", probe=1, identity={})
    with pytest.raises(RuntimeError, match="capture failed"):
        _run_memory_groups(session, monkeypatch, memory_spy, fail_in=1)
    assert memory_spy.on is False
    assert _memory_timing(tmp_path)["memory"]["status"] == "incomplete"


def test_memory_history_stops_when_the_export_fails(tmp_path, monkeypatch, memory_spy):
    request = _memory_request(tmp_path, "memory_group=1")
    assert request is not None
    memory_spy.fail_dump = True
    session = request.session(kind="capture", probe=1, identity={})
    seen = _run_memory_groups(session, monkeypatch, memory_spy)
    assert seen == [False, True, False]
    assert memory_spy.on is False
    assert _memory_timing(tmp_path)["memory"]["status"] == "export_failed"
    assert _memory_timing(tmp_path)["units_count"] == 3


def test_memory_without_cuda_is_unsupported_and_the_run_continues(
        tmp_path, monkeypatch, memory_spy):
    request = _memory_request(tmp_path, "memory_group=1")
    assert request is not None
    session = request.session(kind="capture", probe=1, identity={"capture_groups": 3})
    seen = _run_memory_groups(session, monkeypatch, memory_spy, cuda=False)
    assert seen == [False, False, False]
    assert memory_spy.calls == []
    timing = _memory_timing(tmp_path)
    assert timing["memory"]["status"] == "unsupported"
    assert timing["units_count"] == 3 and timing["errors"] == []


def test_memory_group_that_the_pass_never_reaches_is_not_reached(
        tmp_path, monkeypatch, memory_spy):
    request = _memory_request(tmp_path, "memory_group=5")
    assert request is not None
    session = request.session(kind="capture", probe=1, identity={})
    _run_memory_groups(session, monkeypatch, memory_spy, groups=2)
    assert memory_spy.calls == []
    assert _memory_timing(tmp_path)["memory"]["status"] == "not_reached"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (allocator)")
def test_cuda_memory_snapshot_of_the_selected_group(tmp_path):
    import pickle

    request = _memory_request(
        tmp_path, "capture=0,wait=0,warmup=0,active=1,memory_group=1,memory_entries=8")
    assert request is not None
    session = request.session(kind="capture", probe=0,
                              identity={"quantum_id": "mem", "capture_groups": 3})
    with session:
        for index in range(3):
            session.unit_begin()
            if index == 1:
                kept = [torch.empty(1 << 16, device="cuda") for _ in range(20)]
                del kept
            session.unit_end()
    timing = json.loads((tmp_path / "mem-p0-capture.timing.json").read_text())
    assert timing["memory"]["status"] == "complete"
    with open(tmp_path / "mem-p0-capture.memory_snapshot.pickle", "rb") as handle:
        snapshot = pickle.load(handle)
    traces = [trace for trace in snapshot["device_traces"] if trace]
    assert traces and all(len(trace) <= 8 for trace in traces)
    assert any(event.get("frames") for trace in traces for event in trace)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (CUPTI trace)")
def test_cuda_trace_carries_kernels_and_the_device_to_host_copy(tmp_path):
    import gzip

    request = pass_profile_request({PROFILE_ENV: str(tmp_path),
                                    SPEC_ENV: "capture=0,wait=1,warmup=1,active=2"})
    session = request.session(kind="capture", probe=0, identity={"quantum_id": "cuda"})
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    with session:
        for _ in range(5):
            session.unit_begin()
            (a @ a).float().sum().to("cpu")
            session.unit_end()
    events = json.loads(gzip.open(tmp_path / "cuda-p0-capture.trace.json.gz").read())
    events = events["traceEvents"] if isinstance(events, dict) else events
    categories = {event.get("cat") for event in events}
    assert "kernel" in categories
    assert any(event.get("cat") == "gpu_memcpy" for event in events)
