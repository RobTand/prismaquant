"""Public sampler summaries keep percentile order on short valid windows."""
from __future__ import annotations

import subprocess
import threading

import pytest

from prismaquant.io_spans import GpuPowerSampler


class _PowerProcess:
    """Only the external process boundary is replaced; no GPU is launched."""

    def __init__(self, readings):
        self.stdout = self
        self._lines = iter(f"{value}\n" for value in readings)
        self.waiting = threading.Event()
        self.terminated = threading.Event()
        self.returncode = None

    def readline(self):
        try:
            return next(self._lines)
        except StopIteration:
            self.waiting.set()
            if not self.terminated.wait(5):
                raise OSError("fixture process was not terminated")
            return ""

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self.terminated.set()

    def wait(self, timeout):
        assert self.terminated.wait(timeout)
        return self.returncode


@pytest.mark.parametrize("readings", [
    [30.25, 12.5],
    [70.0],
    [130.0, 10.0, 70.0],
], ids=["two-sample-order", "singleton-control", "three-sample-control"])
def test_public_stop_preserves_power_percentile_order(monkeypatch, readings):
    process = _PowerProcess(readings)
    launches = []

    def launch(argv, **kwargs):
        launches.append(argv)
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.DEVNULL
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    sampler = GpuPowerSampler().start()
    try:
        assert process.waiting.wait(5), "sampler did not consume the valid readings"
    finally:
        record = sampler.stop()
    assert len(launches) == 1
    assert not sampler._sampler.is_alive()
    assert "sampler_error" not in record
    assert sampler.samples == readings
    assert record["sample_count"] == len(readings)
    assert record["gpu_joules"] == sum(readings)
    assert min(readings) <= record["gpu_power_w_p50"]
    assert record["gpu_power_w_p50"] <= record["gpu_power_w_p95"], record
    assert record["gpu_power_w_p95"] <= record["gpu_power_w_max"] == max(readings)


@pytest.mark.parametrize(("readings", "p50", "p95"), [
    ([30.25, 12.5], 12.5, 30.25),
    ([130.0, 10.0, 70.0], 70.0, 130.0),
    ([40.0, 10.0, 30.0, 20.0], 20.0, 40.0),
    (list(range(20, 0, -1)), 10, 19),
], ids=["two-observations", "three-observations", "even-median", "exact-p95-rank"])
def test_public_stop_reports_nearest_rank_quantiles(monkeypatch, readings, p50, p95):
    process = _PowerProcess(readings)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    sampler = GpuPowerSampler().start()
    try:
        assert process.waiting.wait(5), "sampler did not consume all readings"
    finally:
        record = sampler.stop()
    assert not sampler._sampler.is_alive()
    assert "sampler_error" not in record
    assert record["sample_count"] == len(readings)
    assert record["gpu_power_w_p50"] == p50
    assert record["gpu_power_w_p95"] == p95
    assert record["gpu_power_w_max"] == max(readings)
    assert record["gpu_joules"] == sum(readings)
