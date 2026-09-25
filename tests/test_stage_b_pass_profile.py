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


@pytest.mark.parametrize("spec", ["bogus=1", "capture=1:deep", "active=0", "nokey"])
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
