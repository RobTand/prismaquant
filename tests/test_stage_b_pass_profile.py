"""PQ #1269: the Stage B pass profiler is off by default and bounded when on."""
import json

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
