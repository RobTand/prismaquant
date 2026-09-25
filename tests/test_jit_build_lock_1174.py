"""PQ #1174: a builder killed while it holds torch's build lock must not
leave every later load of the extension waiting.

CPU-only. ``torch.utils.cpp_extension.load`` is replaced by a stand-in that
serializes exactly as torch's ``FileBaton`` does (``O_EXCL`` create, poll for
the file to vanish, delete on release), with a deadline so a wedge fails the
test instead of hanging it. The builder is a real child process, killed with
SIGKILL while it holds the baton.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time

import pytest
import torch.utils.cpp_extension as cpp_extension

from prismaquant.kernels import jit_build_lock as jbl
from prismaquant.kernels import joint_projection_reduce as kernel

REPO = Path(__file__).resolve().parents[1]
WAIT_DEADLINE_S = 5.0


class Wedged(RuntimeError):
    pass


def _baton_load(build_directory, calls):
    """A stand-in for ``load`` with torch's ``FileBaton`` serialization."""
    def load(name, **_):
        baton = Path(build_directory) / jbl.TORCH_BATON_NAME
        deadline = time.monotonic() + WAIT_DEADLINE_S
        while True:
            try:
                fd = os.open(baton, os.O_CREAT | os.O_EXCL)
                break
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise Wedged(f'waited {WAIT_DEADLINE_S}s on {baton}')
                time.sleep(0.02)
        try:
            calls.append(name)
            return object()
        finally:
            os.close(fd)
            os.remove(baton)
    return load


def _killed_builder(build_directory):
    """Run a child that takes the guard and the baton, then SIGKILL it."""
    ready = Path(build_directory).parent / 'builder.ready'
    script = textwrap.dedent(f'''
        import os, time
        from pathlib import Path
        from prismaquant.kernels.jit_build_lock import jit_build_lock, TORCH_BATON_NAME
        d = Path({str(build_directory)!r})
        with jit_build_lock(d):
            os.open(d / TORCH_BATON_NAME, os.O_CREAT | os.O_EXCL)
            Path({str(ready)!r}).write_text('ready')
            time.sleep(600)
    ''')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(REPO), os.environ.get('PYTHONPATH', '')]))
    child = subprocess.Popen([sys.executable, '-c', script], env=env)
    try:
        deadline = time.monotonic() + 60
        while not ready.exists():
            assert child.poll() is None, 'builder exited before taking the baton'
            assert time.monotonic() < deadline, 'builder never took the baton'
            time.sleep(0.02)
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait()
    assert child.returncode == -signal.SIGKILL
    assert (Path(build_directory) / jbl.TORCH_BATON_NAME).exists()


@pytest.fixture
def build_directory(tmp_path, monkeypatch):
    directory = tmp_path / 'torch_extensions' / 'pq_joint_projection_reduce_x'
    directory.mkdir(parents=True)
    monkeypatch.setattr(cpp_extension, '_get_build_directory',
                        lambda name, verbose: str(directory))
    kernel.load_backend.cache_clear()
    yield directory
    kernel.load_backend.cache_clear()


def test_load_after_a_killed_builder_finishes(build_directory, monkeypatch):
    _killed_builder(build_directory)
    calls = []
    monkeypatch.setattr(cpp_extension, 'load', _baton_load(build_directory, calls))
    kernel.load_backend()
    assert calls and calls[0].startswith('pq_joint_projection_reduce_')
    assert not (build_directory / jbl.TORCH_BATON_NAME).exists()


def test_a_stale_baton_without_any_builder_is_cleared(build_directory, monkeypatch):
    os.close(os.open(build_directory / jbl.TORCH_BATON_NAME, os.O_CREAT | os.O_EXCL))
    calls = []
    monkeypatch.setattr(cpp_extension, 'load', _baton_load(build_directory, calls))
    kernel.load_backend()
    assert len(calls) == 1


def test_a_live_flock_holder_keeps_its_lock(tmp_path):
    """A torch that flocks ``lock`` itself: a live holder is never robbed."""
    baton = tmp_path / jbl.TORCH_BATON_NAME
    holder = os.open(baton, os.O_RDWR | os.O_CREAT)
    try:
        # A separate open file description conflicts with this one's flock,
        # as another process's would.
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert jbl.clear_stale_baton(tmp_path) is False
        assert baton.exists()
    finally:
        os.close(holder)
    assert jbl.clear_stale_baton(tmp_path) is True
    assert not baton.exists()


def test_the_guard_serializes_builders_and_is_released_by_a_kill(tmp_path):
    directory = tmp_path / 'build'
    ready = tmp_path / 'builder.ready'
    script = textwrap.dedent(f'''
        import time
        from pathlib import Path
        from prismaquant.kernels.jit_build_lock import jit_build_lock
        with jit_build_lock({str(directory)!r}):
            Path({str(ready)!r}).write_text('ready')
            time.sleep(600)
    ''')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(REPO), os.environ.get('PYTHONPATH', '')]))
    child = subprocess.Popen([sys.executable, '-c', script], env=env)
    try:
        deadline = time.monotonic() + 60
        while not ready.exists():
            assert child.poll() is None and time.monotonic() < deadline
            time.sleep(0.02)
        guard = os.open(directory / jbl.GUARD_NAME, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(guard)
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait()
    with jbl.jit_build_lock(directory) as removed:
        assert removed is False
