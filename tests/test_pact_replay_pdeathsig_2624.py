"""A SIGKILLed parent must not leave a replay worker alive (prismaquant#2624)."""
import os
import signal
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import pytest

REPLAY_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "pact_replay"
sys.path.insert(0, str(REPLAY_ROOT))

from replay_progress import supervise  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="needs Linux prctl and fork")

PID_WAIT_SECONDS = 30.0
DEATH_WAIT_SECONDS = 20.0


def _wait_for(condition, timeout, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return condition()


def _pid_gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _parent_with_native_block(pid_path):
    """Supervise a worker that blocks in a native read, like the #2520 stall."""

    def blocked(send):
        with open(pid_path, "w") as handle:
            handle.write(str(os.getpid()))
        reader, _writer = os.pipe()
        # A native read runs no Python code and sends no progress record.
        os.read(reader, 1)

    supervise(blocked, 1800)


def _parent_with_steady_work(pid_path):
    """Supervise a worker that reports steady progress."""

    def steady(send):
        with open(pid_path, "w") as handle:
            handle.write(str(os.getpid()))
        while True:
            send({"units_completed": 1, "monotonic_seconds": time.monotonic()})
            time.sleep(0.05)

    supervise(steady, 1800)


def _run_parent(entry, tmp_path):
    pid_path = tmp_path / "worker.pid"
    worker = None
    parent = get_context("fork").Process(target=entry, args=(str(pid_path),))
    parent.start()
    try:
        assert _wait_for(pid_path.exists, PID_WAIT_SECONDS), "the worker never started"
        worker = int(pid_path.read_text().strip())
        assert worker != parent.pid
        os.kill(parent.pid, signal.SIGKILL)
        parent.join(timeout=DEATH_WAIT_SECONDS)
        assert not parent.is_alive(), "the SIGKILLed parent is still alive"
        assert _wait_for(lambda: _pid_gone(worker), DEATH_WAIT_SECONDS), (
            "the worker survived its parent's SIGKILL")
        with pytest.raises(ProcessLookupError):
            os.killpg(worker, 0)
    finally:
        if parent.is_alive():
            os.kill(parent.pid, signal.SIGKILL)
            parent.join(timeout=10)
        # Never leave a stalled orphan behind on failure.
        if worker is not None:
            try:
                os.killpg(worker, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def test_sigkill_parent_kills_native_blocked_worker(tmp_path):
    """The #2520 stall case: no Python code runs, so only the kernel stops it."""
    _run_parent(_parent_with_native_block, tmp_path)


def test_sigkill_parent_kills_steady_worker(tmp_path):
    """A worker with live progress output must also leave no orphan behind."""
    _run_parent(_parent_with_steady_work, tmp_path)
