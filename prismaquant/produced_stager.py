"""Synchronization for streamed lifecycle bookkeeping; IO owns asynchronous work."""
from __future__ import annotations

import threading

class OwnerLock:
    """The owner's bookkeeping lock, which a thread gives up around a call.

    ``held`` nests on the owning thread. ``yielded`` temporarily releases
    bookkeeping while an IO completion barrier runs.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._owner = None

    def held(self):
        return _Held(self)

    def yielded(self):
        return _Yielded(self)


class _Held:
    __slots__ = ("_outer", "_took")

    def __init__(self, outer):
        self._outer = outer
        self._took = False

    def __enter__(self):
        outer = self._outer
        me = threading.get_ident()
        if outer._owner != me:
            outer._lock.acquire()
            outer._owner = me
            self._took = True
        return self

    def __exit__(self, *_exc):
        if self._took:
            self._outer._owner = None
            self._outer._lock.release()
        return False


class _Yielded:
    __slots__ = ("_outer", "_gave")

    def __init__(self, outer):
        self._outer = outer
        self._gave = False

    def __enter__(self):
        outer = self._outer
        if outer._owner == threading.get_ident():
            outer._owner = None
            outer._lock.release()
            self._gave = True
        return self

    def __exit__(self, *_exc):
        if self._gave:
            self._outer._lock.acquire()
            self._outer._owner = threading.get_ident()
        return False

