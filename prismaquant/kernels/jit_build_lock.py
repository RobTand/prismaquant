"""Serialize a JIT extension build with a lock the kernel releases (PQ #1174).

``torch.utils.cpp_extension.load`` serializes builds of one extension with a
``lock`` file in its build directory. In the releases the fleet runs that
file is a ``FileBaton``: an ``O_EXCL`` file the builder deletes when it
finishes, and waiters poll for it to disappear with no timeout and no owner
check. A builder killed mid-build (an OOM kill, a withdrawn run) never deletes
it, and every later load of the extension then waits forever. Later releases
hold an advisory ``flock`` on the same path instead (pytorch#189245), which
the kernel releases when its holder dies; a leftover file there blocks nobody.

:func:`jit_build_lock` holds an ``fcntl.flock`` on a separate file,
``prismaquant.build.flock``, in the build directory around ``load``. The
kernel releases it when its holder dies, so a killed builder cannot wedge it.
While a process holds it no other PrismaQuant builder of that extension is
inside ``load``, so a ``lock`` file present then is a dead builder's baton and
is removed without any age threshold. One guard remains for a torch that
flocks ``lock`` itself: the file is removed only while a non-blocking
``flock`` on it succeeds, so a live holder's lock is never taken from it.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import os
from pathlib import Path

#: The file this module flocks; distinct from torch's own ``lock``.
GUARD_NAME = 'prismaquant.build.flock'
#: The file ``torch.utils.cpp_extension`` serializes a build with.
TORCH_BATON_NAME = 'lock'


def clear_stale_baton(build_directory) -> bool:
    """Remove torch's ``lock`` file when no live process holds it.

    Call only while holding :func:`jit_build_lock` for ``build_directory``:
    that is what makes a ``FileBaton`` found here a dead builder's. Returns
    True when a file was removed.
    """
    path = Path(build_directory) / TORCH_BATON_NAME
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                return False  # a live torch flock holder owns it
            raise
        # Only a baton this process just flocked is removed; the flock is
        # released with the descriptor below.
        path.unlink(missing_ok=True)
        return True
    finally:
        os.close(fd)


@contextmanager
def jit_build_lock(build_directory):
    """Hold the build directory's kernel-released lock; clear a dead baton.

    Yields whether a stale ``lock`` file was removed.
    """
    directory = Path(build_directory)
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / GUARD_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        removed = clear_stale_baton(directory)
        if removed:
            print(f'[jit-build-lock] removed a stale torch build lock left by a '
                  f'killed builder: {directory / TORCH_BATON_NAME}', flush=True)
        yield removed
    finally:
        os.close(fd)  # closing the only descriptor releases the flock


def torch_build_directory(name: str) -> str:
    """The directory ``cpp_extension.load(name, ...)`` builds in by default."""
    from torch.utils.cpp_extension import _get_build_directory

    return _get_build_directory(name, verbose=False)
