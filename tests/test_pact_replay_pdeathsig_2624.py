"""A SIGKILLed parent must not leave a replay worker alive (prismaquant#2624)."""
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import pytest

REPLAY_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "pact_replay"
sys.path.insert(0, str(REPLAY_ROOT))

import replay_progress  # noqa: E402

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


def _announce(path, **record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": os.getpid(), **record}))
    temporary.replace(path)


def _parent(scenario, ready_path):
    if scenario == "startup":
        original_worker = replay_progress._worker

        def paused_worker(*args):
            # Stop before the original worker's first instruction/getppid.
            _announce(ready_path, phase="startup")
            os.kill(os.getpid(), signal.SIGSTOP)
            original_worker(*args)

        replay_progress._worker = paused_worker

    def work(send):
        if scenario == "gpu":
            # Only the numerical child imports Torch, after supervise forks.
            import torch

            if not torch.cuda.is_available():
                _announce(ready_path, skip="requires live CUDA; CPU is not GPU evidence")
                return
            allocation = torch.ones(1024 * 1024, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            assert allocation.is_cuda and torch.cuda.memory_allocated() >= allocation.nbytes
            _announce(ready_path, phase="work", cuda_bytes=allocation.nbytes)
        else:
            _announce(ready_path, phase="work")
        if scenario == "steady":
            while True:
                send({"units_completed": 1, "monotonic_seconds": time.monotonic()})
                time.sleep(0.05)
        else:
            reader, writer = os.pipe()
            # Both ends stay open here: parent death cannot unblock this read
            # with EOF. No Python code or progress sends run while blocked.
            try:
                os.read(reader, 1)
            finally:
                os.close(reader)
                os.close(writer)
            if scenario == "gpu":
                # Keep the CUDA allocation live throughout the native read.
                assert allocation.is_cuda

    replay_progress.supervise(work, 1800)


def _run_parent(scenario, tmp_path):
    # Run only in a fresh driver interpreter, not in the shared pytest process.
    # Adopt/reap the orphan ourselves: PID 1 need not reap promptly in containers.
    PR_SET_CHILD_SUBREAPER = 36
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl PR_SET_CHILD_SUBREAPER failed")
    ready_path = tmp_path / "worker.json"
    worker = None
    reaped = False
    parent = get_context("fork").Process(target=_parent, args=(scenario, ready_path))
    parent.start()
    try:
        assert _wait_for(ready_path.exists, PID_WAIT_SECONDS), "the worker never started"
        record = json.loads(ready_path.read_text())
        worker = record["pid"]
        if "skip" in record:
            parent.join(timeout=DEATH_WAIT_SECONDS)
            assert parent.exitcode == 0
            print(record["skip"], flush=True)
            return 77
        assert worker != parent.pid
        if scenario == "startup":
            state_path = Path(f"/proc/{worker}/status")
            assert _wait_for(lambda: "State:\tT" in state_path.read_text(), PID_WAIT_SECONDS)
        elif scenario != "steady":
            # Observe the actual native pipe read, not just a ready message
            # before it. wchan avoids architecture-specific syscall numbers.
            wchan = Path(f"/proc/{worker}/wchan")
            try:
                blocked = _wait_for(lambda: "pipe_read" in wchan.read_text(), PID_WAIT_SECONDS)
                hidden = wchan.read_text().strip() == "0"
            except PermissionError:
                blocked, hidden = False, True
            if not blocked and hidden:
                print("requires visible Linux wait channels to prove native blocking", flush=True)
                return 77
            assert blocked, "worker did not enter the native read"
        if scenario == "gpu":
            assert record["cuda_bytes"] > 0, "worker held no CUDA allocation"
        os.kill(parent.pid, signal.SIGKILL)
        parent.join(timeout=DEATH_WAIT_SECONDS)
        assert parent.exitcode == -signal.SIGKILL
        if scenario == "startup":
            assert f"PPid:\t{os.getpid()}\n" in state_path.read_text(), (
                "worker was not reparented to the test subreaper")
            os.kill(worker, signal.SIGCONT)

        status = None

        def reap_worker():
            nonlocal status, reaped
            pid, status = os.waitpid(worker, os.WNOHANG)
            reaped = pid == worker
            return reaped

        assert _wait_for(reap_worker, DEATH_WAIT_SECONDS), (
            "the worker survived its parent's SIGKILL; "
            f"last readiness record: {ready_path.read_text()}")
        if scenario == "startup":
            assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 128 + signal.SIGKILL
            assert json.loads(ready_path.read_text())["phase"] == "startup", (
                "orphan entered work after its original supervisor died")
        else:
            assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL
        assert _pid_gone(worker)
        with pytest.raises(ProcessLookupError):
            os.killpg(worker, 0)
    finally:
        if parent.is_alive():
            parent.kill()
        parent.join(timeout=10)
        # Never leave a stopped, native-blocked, or GPU-holding orphan on failure.
        if worker is None and ready_path.exists():
            worker = json.loads(ready_path.read_text())["pid"]
        if worker is not None and not reaped:
            try:
                os.kill(worker, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(worker, 0)
            except ChildProcessError:
                pass  # A normally exiting supervisor may already have reaped it.
    return 0


def _check_parent_death(scenario, tmp_path):
    # exec, not fork, isolates CUDA initialization and subreaper state from pytest.
    with subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), scenario, str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    ) as driver:
        try:
            stdout, stderr = driver.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            # The worker left the driver's group with setsid. Kill both the
            # supervisor's group and the isolated worker on this emergency path.
            try:
                os.killpg(driver.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            ready_path = tmp_path / "worker.json"
            if ready_path.exists():
                try:
                    os.kill(json.loads(ready_path.read_text())["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            driver.communicate(timeout=10)
            raise
    if driver.returncode == 77:
        pytest.skip(stdout.strip())
    assert driver.returncode == 0, stdout + stderr


def test_sigkill_parent_kills_native_blocked_worker(tmp_path):
    """The #2520 stall case: no Python code runs, so only the kernel stops it."""
    _check_parent_death("native", tmp_path)


def test_sigkill_parent_kills_steady_worker(tmp_path):
    """A worker with live progress output must also leave no orphan behind."""
    _check_parent_death("steady", tmp_path)


def test_sigkill_parent_before_worker_startup(tmp_path):
    """Reparenting before the child's first getppid must prevent work starting."""
    _check_parent_death("startup", tmp_path)


@pytest.mark.cuda
def test_sigkill_parent_leaves_no_gpu_holding_worker(tmp_path):
    """Hold a real CUDA tensor across a native block, then require PID/group removal."""
    _check_parent_death("gpu", tmp_path)


if __name__ == "__main__":
    raise SystemExit(_run_parent(sys.argv[1], Path(sys.argv[2])))
