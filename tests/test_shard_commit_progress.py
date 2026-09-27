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
            # The worker publishes the phase list as a JSON array
            # (prismabuild pool.py mints json.dumps([phase.name, ...])), never
            # as a bare comma list: the incident this file grew from was a
            # comma parse reading '["export"]' as one bogus phase name and
            # silently refusing every commit (PB #1516, action 0a8bdf67).
            "PRISMABUILD_ACTION_PROGRESS_PHASES": json.dumps(list(phases)),
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
                                      poll_seconds=0.05)
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


def test_partial_staging_never_counts_only_renamed_names_do(out, channel,
                                                             monkeypatch):
    monkeypatch.setattr(scp.os, "environ", channel.environ, raising=False)
    # The producer publishes each shard with one rename (save_serving_shard,
    # producer #366): bytes land in a dot-prefixed .partial staging name and
    # Path.replace moves the complete payload onto its final name.  The
    # watcher must count final names on first appearance and never the
    # staging file, however long it sits there growing.
    staging = out / ".model-00001-of-00002.safetensors.hj3k4n.partial"
    staging.write_bytes(b"growing-payload")
    watcher = scp.ShardCommitProgress(out, phase="export",
                                      poll_seconds=0.05)
    assert watcher.poll() == 0, "a .partial staging file must not count"
    with staging.open("ab") as handle:
        handle.write(b"-more")
    assert watcher.poll() == 0, "a growing staging file must not count"
    os.replace(staging, out / "model-00001-of-00002.safetensors")
    assert watcher.poll() == 1, "a renamed final name commits at once"
    assert channel.last()["units_completed"] == 1
    assert channel.last()["unit"] == "model-00001-of-00002.safetensors"


def test_stalled_writer_reports_no_new_units(out, channel, monkeypatch):
    monkeypatch.setattr(scp.os, "environ", channel.environ, raising=False)
    write_shard(out, "model-00001-of-00002.safetensors", b"one")
    watcher = scp.ShardCommitProgress(out, phase="export",
                                      poll_seconds=0.05)
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
                                          poll_seconds=0.05, expected=4)
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
                                      poll_seconds=0.01)
    watcher.poll()
    watcher.poll()
    assert channel.records() == [], "an undeclared phase must not commit"
    assert any("not declared" in line for line in logged)


def test_without_channel_the_watcher_is_inert(out, monkeypatch):
    monkeypatch.delenv("PRISMABUILD_ACTION_PROGRESS_PATH", raising=False)
    monkeypatch.delenv("PRISMABUILD_ACTION_PROGRESS_TOKEN", raising=False)
    write_shard(out, "model-00001-of-00001.safetensors", b"one")
    watcher = scp.ShardCommitProgress(out, phase="export",
                                      poll_seconds=0.01)
    watcher.poll()
    assert watcher.poll() == 1
    assert watcher.poll() == 1


def test_stall_allowance_derives_from_measured_rate():
    assert scp.stall_allowance(3241.0 / 120) == pytest.approx(300.0)
    assert scp.stall_allowance(3241.0 / 120,
                               minimum=200.0) == pytest.approx(10 * 3241.0 / 120)
    with pytest.raises(ValueError):
        scp.stall_allowance(0)


def test_declared_phases_reads_the_worker_json_list():
    # The pool launcher publishes the phase list as a JSON array
    # (prismabuild pool.py: json.dumps([phase.name for phase in policy])).
    # A8 r6 (PB 0a8bdf67, 2026-09-27) died because a comma parse read
    # '["export"]' as one bogus phase name, refused the real one, and
    # turned the contract into a wall clock.  The reading is imported from
    # joint_run_progress: one implementation, no copy to drift (#1517).
    from prismaquant import joint_run_progress as jrp
    assert scp.declared_phases is jrp.declared_phases, (
        "declared_phases must be imported, not copied")
    assert scp.declared_phases(
        {"PRISMABUILD_ACTION_PROGRESS_PHASES": '["export", "publish"]'}
    ) == ("export", "publish")
    assert scp.declared_phases(
        {"PRISMABUILD_ACTION_PROGRESS_PHASES": '["export"]'}
    ) == ("export",)
    # A bare comma list is not the worker's spelling: unknown, not guessed.
    assert scp.declared_phases(
        {"PRISMABUILD_ACTION_PROGRESS_PHASES": "export,publish"}) is None
    assert scp.declared_phases(
        {"PRISMABUILD_ACTION_PROGRESS_PHASES": "[]"}) is None
    assert scp.declared_phases({}) is None


def test_worker_json_phase_list_commits_records(out, channel, monkeypatch):
    # The incident regression: with the channel's phases published exactly as
    # the worker publishes them, a watcher for a legitimately declared phase
    # must commit progress records, not refuse the phase.
    monkeypatch.setattr(scp.os, "environ", channel.environ, raising=False)
    write_shard(out, "model-00001-of-00002.safetensors", b"one")
    watcher = scp.ShardCommitProgress(out, phase="export",
                                      poll_seconds=0.01)
    assert not watcher._phase_refused, (
        "a phase the worker declared must not be refused")
    watcher.poll()
    watcher.poll()
    assert channel.records(), "a declared phase must commit records"
    assert channel.last()["phase"] == "export"
    assert channel.last()["units_completed"] == 1


def test_default_log_prints_flushed_diagnostics(capsys):
    # The refusal that hid the r6 failure reached the attempt log only when
    # the process exited cleanly: the parent's block-buffered stdout was
    # discarded on SIGKILL.  The module's own log must flush.
    scp._default_log("diagnostic line")
    assert "diagnostic line" in capsys.readouterr().out


def test_entry_whose_stat_raises_is_skipped_and_picked_up_later(
        out, channel, monkeypatch):
    monkeypatch.setattr(scp.os, "environ", channel.environ, raising=False)
    write_shard(out, "model-00001-of-00002.safetensors", b"payload")
    watcher = scp.ShardCommitProgress(out, phase="export",
                                      poll_seconds=0.01)
    # An ESTALE hiccup on one entry must not escape poll(): the entry is
    # skipped this scan and picked up on a later one (review r2 -- the
    # output root is on NFS; stat/is_file can raise OSError).
    real_stat = os.stat

    def estale(path, *args, **kwargs):
        if "model-00001" in str(path):
            raise OSError("NFSv4 server %s not responding", "ESTALE")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(scp.os, "stat", estale)
    assert watcher.poll() == 0, "an entry whose stat raises must be skipped"
    # Restore only os.stat (undo() would also drop the channel-environ patch
    # and silence the reports).
    monkeypatch.setattr(scp.os, "stat", real_stat)
    assert watcher.poll() == 1, "the same entry counts once the hiccup ends"
    assert channel.last()["units_completed"] == 1


def test_glob_error_is_logged_once_and_reports_nothing_new(
        out, channel, monkeypatch):
    monkeypatch.setattr(scp.os, "environ", channel.environ, raising=False)
    write_shard(out, "model-00001-of-00002.safetensors", b"payload")
    watcher = scp.ShardCommitProgress(out, phase="export",
                                      poll_seconds=0.01)
    assert watcher.poll() == 1

    class broken_glob:
        def __init__(self, real):
            self._real = real

        def glob(self, pattern):
            raise OSError("NFSv4 directory has been unmounted")

    watcher.out_dir = broken_glob(watcher.out_dir)
    logged = []
    watcher._log = logged.append
    # A glob that raises must not escape poll(): the poll reports nothing
    # new, the committed set stands, and the error is logged once, flushed.
    assert watcher.poll() == 1, "a glob error must not change the count"
    assert len(logged) == 1
    assert "unmounted" in logged[0]
    assert watcher.poll() == 1
    assert len(logged) == 2, "each occurrence is logged once"
