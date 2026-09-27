"""Shard-commit progress reporting for child-run exports (PrismaQuant #1516).

The A8 v39 body export attempt 3 (PB d008b42a9c74..., 2026-09-27) was killed
at 3316 s with ``no_progress`` 30 s from completion because the exporter, a
pinned producer child, never reported a unit while its campaign row declared
an ``export`` phase.  These tests pin the parent-side contract the fix adds:
one progress unit per durable shard commit, never ahead of the work.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from prismaquant import shard_commit_progress as scp


class Channel:
    """A fake PrismaBuild progress channel on a real filesystem."""

    def __init__(self, path: Path, phases=("export",)):
        self.path = path
        self.record_path = path / "progress.json"
        path.mkdir(parents=True, exist_ok=True)
        self.environ = {
            "PRISMABUILD_ACTION_PROGRESS_PATH": str(self.record_path),
            "PRISMABUILD_ACTION_PROGRESS_TOKEN": "tok-1516",
            "PRISMABUILD_ACTION_PROGRESS_PHASES": ",".join(phases),
        }

    def records(self):
        if not self.record_path.exists():
            return []
        raw = self.record_path.read_text().strip()
        return [json.loads(line) for line in raw.splitlines() if line] if raw else []

    def last(self):
        return self.records()[-1] if self.records() else None


def write_shard(out: Path, name: str, payload: bytes) -> Path:
    path = out / name
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def steady_writer(out: Path, names, *, interval: float, stop: threading.Event):
    def run():
        for index, name in enumerate(names):
            if stop.is_set():
                return
            payload = bytes([index % 251]) * (1024 + index)
            write_shard(out, name, payload)
            time.sleep(interval)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


@pytest.fixture
def out(tmp_path):
    out = tmp_path / "exported"
    out.mkdir()
    return out


@pytest.fixture
def channel(tmp_path):
    return Channel(tmp_path / "channel")


def test_steady_shard_writes_report_monotone_units(out, channel, monkeypatch):
    monkeypatch.setattr(scp.os, "environ", channel.environ, raising=False)
    names = [f"model-{i:05d}-of-00003.safetensors" for i in (1, 2, 3)]
    stop = threading.Event()
    steady_writer(out, names, interval=0.05, stop=stop)
    watcher = scp.ShardCommitProgress(out, phase="export",
                                      interval_seconds=0.05)
    deadline = time.monotonic() + 10.0
    while watcher.poll() < len(names) and time.monotonic() < deadline:
        time.sleep(0.02)
    stop.set()
    assert watcher.committed == tuple(names)
    records = channel.records()
    assert records, "a declared phase must commit records"
    units = [record["units_completed"] for record in records]
    assert units == sorted(units), "units_completed must be monotone"
    assert units[-1] == 3
    assert records[-1]["unit"] == names[-1]
    assert records[-1]["phase"] == "export"
    assert records[-1]["schema"] == "prismabuild.action_progress.v1"


def test_growing_file_is_not_committed_until_quiescent(out, channel, monkeypatch):
    monkeypatch.setattr(scp.os, "environ", channel.environ, raising=False)
    # a quiescent shard commits on its second identical observation
    write_shard(out, "model-00001-of-00002.safetensors", b"one")
    watcher = scp.ShardCommitProgress(out, phase="export",
                                      interval_seconds=0.05)
    watcher.poll()
    assert watcher.poll() == 1
    # a growing file: the first observation pends it, an append between
    # polls resets its fingerprint, and only quiescence commits it
    write_shard(out, "model-00002-of-00002.safetensors", b"two")
    assert watcher.poll() == 1
    with (out / "model-00002-of-00002.safetensors").open("ab") as handle:
        handle.write(b"-grow")
        handle.flush()
        os.fsync(handle.fileno())
    assert watcher.poll() == 1, "a changed fingerprint must restart quiescence"
    assert watcher.poll() == 2


def test_stalled_writer_reports_no_new_units(out, channel, monkeypatch):
    monkeypatch.setattr(scp.os, "environ", channel.environ, raising=False)
    write_shard(out, "model-00001-of-00002.safetensors", b"one")
    watcher = scp.ShardCommitProgress(out, phase="export",
                                      interval_seconds=0.05)
    deadline = time.monotonic() + 5.0
    while watcher.poll() < 1 and time.monotonic() < deadline:
        time.sleep(0.02)
    before = channel.last()
    assert before is not None and before["units_completed"] == 1
    for _ in range(4):
        time.sleep(0.05)
        assert watcher.poll() == 1, "a stall must not invent units"
    assert channel.last() == before, "a stall must not rewrite the record"


def test_watch_reports_while_child_runs_and_settles_after_exit(out, channel,
                                                               monkeypatch):
    monkeypatch.setattr(scp.os, "environ", channel.environ, raising=False)
    child = subprocess.Popen([sys.executable, "-c",
                              "import time; time.sleep(0.5)"])
    names = [f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)]
    stop = threading.Event()
    steady_writer(out, names, interval=0.05, stop=stop)

    def drive():
        watcher = scp.ShardCommitProgress(out, phase="export",
                                          interval_seconds=0.05, expected=4)
        count = watcher.watch(child, poll_seconds=0.05)
        assert count == 4
        assert watcher.expected_complete()

    drive()
    stop.set()
    assert channel.last()["units_completed"] == 4


def test_undeclared_phase_is_logged_and_not_committed(out, channel, monkeypatch):
    logged = []
    monkeypatch.setattr(scp.os, "environ", channel.environ, raising=False)
    write_shard(out, "model-00001-of-00001.safetensors", b"one")
    watcher = scp.ShardCommitProgress(out, phase="not-a-phase", log=logged.append,
                                      interval_seconds=0.01)
    watcher.poll()
    watcher.poll()
    assert channel.records() == [], "an undeclared phase must not commit"
    assert any("not declared" in line for line in logged)


def test_without_channel_the_watcher_is_inert(out, monkeypatch):
    monkeypatch.delenv("PRISMABUILD_ACTION_PROGRESS_PATH", raising=False)
    monkeypatch.delenv("PRISMABUILD_ACTION_PROGRESS_TOKEN", raising=False)
    write_shard(out, "model-00001-of-00001.safetensors", b"one")
    watcher = scp.ShardCommitProgress(out, phase="export",
                                      interval_seconds=0.01)
    watcher.poll()
    assert watcher.poll() == 1
    assert watcher.poll() == 1


def test_stall_allowance_derives_from_measured_rate():
    assert scp.stall_allowance(3241.0 / 120) == pytest.approx(300.0)
    assert scp.stall_allowance(3241.0 / 120,
                               minimum=200.0) == pytest.approx(10 * 3241.0 / 120)
    with pytest.raises(ValueError):
        scp.stall_allowance(0)
