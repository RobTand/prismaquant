"""A power-reader exit before requested stop must invalidate retained samples."""
from __future__ import annotations

import io
import subprocess
import threading

import pytest

from prismaquant.io_spans import GpuPowerSampler


class _EndedProcess:
    def __init__(self, code):
        self.stdout = io.StringIO("70\n")
        self.returncode = code

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def wait(self, timeout):
        return self.returncode


class _LiveProcess:
    def __init__(self):
        self.stdout = self
        self.waiting = threading.Event()
        self.stopped = threading.Event()
        self._first = True
        self.returncode = None

    def readline(self):
        if self._first:
            self._first = False
            return "70\n"
        self.waiting.set()
        if not self.stopped.wait(5):
            raise OSError("fixture did not receive process termination")
        return ""

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self.stopped.set()

    def wait(self, timeout):
        assert self.stopped.wait(timeout)
        return self.returncode


@pytest.mark.parametrize("returncode", [0, 7], ids=["clean_early_eof", "failed_child"])
def test_public_sampler_marks_unexpected_eof_even_with_retained_samples(monkeypatch, returncode):
    process = _EndedProcess(returncode)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    sampler = GpuPowerSampler().start()
    try:
        sampler._sampler.join(timeout=5)
        assert not sampler._sampler.is_alive(), "real power-reader thread did not finish"
    finally:
        record = sampler.stop()
    assert record["sample_count"] == 1
    assert record.get("sampler_error"), (
        "power child ended before the row requested stop, but retained samples "
        f"remain indistinguishable from a complete reading: {record}"
    )


def test_public_sampler_requested_stop_preserves_successful_sample(monkeypatch):
    process = _LiveProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    sampler = GpuPowerSampler().start()
    try:
        assert process.waiting.wait(5), "real sampler did not consume its first sample"
    finally:
        record = sampler.stop()
    assert not sampler._sampler.is_alive()
    assert "sampler_error" not in record
    assert record["sample_count"] == 1
    assert record["gpu_power_w_p50"] == 70.0
    assert record["gpu_joules"] == 70.0


def test_public_stop_reports_known_child_death_before_reader_sees_eof(monkeypatch):
    process = _LiveProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    sampler = GpuPowerSampler().start()
    try:
        assert process.waiting.wait(5)
        process.returncode = 7
    finally:
        record = sampler.stop()
    assert not sampler._sampler.is_alive()
    assert record.get("sampler_error")


def test_public_stop_reports_a_reader_still_alive_after_join(monkeypatch):
    class StuckProcess(_LiveProcess):
        def terminate(self):
            self.returncode = -15

        def wait(self, timeout):
            return self.returncode

    process = StuckProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    sampler = GpuPowerSampler().start()
    try:
        assert process.waiting.wait(5)
        record = sampler.stop()
        assert "reader did not stop" in record.get("sampler_error", "")
    finally:
        process.stopped.set()
        sampler._sampler.join(timeout=5)
    assert not sampler._sampler.is_alive()


def test_sampler_teardown_failure_does_not_replace_workload_failure(monkeypatch):
    class FailedWait(_LiveProcess):
        def wait(self, timeout):
            raise OSError("controlled process wait failure")

    class WorkloadFailure(RuntimeError):
        pass

    process = FailedWait()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    sampler = GpuPowerSampler().start()
    with pytest.raises(WorkloadFailure, match="original workload failure"):
        try:
            assert process.waiting.wait(5)
            raise WorkloadFailure("original workload failure")
        finally:
            record = sampler.stop()
    assert not sampler._sampler.is_alive()
    assert "teardown failed" in record.get("sampler_error", "")
