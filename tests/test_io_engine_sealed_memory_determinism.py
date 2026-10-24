"""Deterministic reproductions for PQ #3051, not a cgroup memory benchmark.

A worker may be descheduled at any lock release. Reclamation must not
report completion while that worker still owns the decoded value. Separately,
a test of this process's sealed mappings must tolerate unrelated processes
changing the cgroup's totals. Neither reproduction waits for host load.
"""
from __future__ import annotations

import itertools
import os
import threading
import weakref

import pytest

from prismaquant import io_engine
import test_io_engine_sealed_memory as sealed_memory


class _DecodedValue:
    pass


class _CompletionBoundary:
    """Release the real lock, then park the reader at completion publication.

    This only schedules an otherwise legal interleaving. It does not change
    engine state, accounting, or references to any decoded value.
    """

    def __init__(self, condition, stream, reader_ident, published, proceed):
        self._condition = condition
        self._stream = stream
        self._reader_ident = reader_ident
        self._published = published
        self._proceed = proceed

    def __getattr__(self, name):
        return getattr(self._condition, name)

    def __enter__(self):
        self._condition.__enter__()
        return self

    def __exit__(self, *exc):
        completion = (
            self._reader_ident == [threading.get_ident()]
            and self._stream._active == 0
            and not self._published.is_set()
        )
        result = self._condition.__exit__(*exc)
        if completion:
            self._published.set()
            assert self._proceed.wait(60), "consumer never released the reader"
        return result


@pytest.mark.parametrize("disposal", ["reclaim", "take-and-drop"])
def test_published_read_does_not_retain_a_worker_owned_value(disposal, monkeypatch):
    """Both disposal paths must release a value before the reader runs again."""
    refs, reader_ident, futures = [], [], []
    published, proceed = threading.Event(), threading.Event()

    def reader():
        value = _DecodedValue()
        refs.append(weakref.ref(value))
        reader_ident.append(threading.get_ident())
        return value, ()

    entry = io_engine.ReadEntry(
        key="one", path=None, size=4096, limit=4096, held_bytes=4096,
        expected_sha256=None, decoder=None, group=0, reader=reader,
    )
    engine = io_engine.IOEngine()
    engine.width = 1  # one bounded worker, regardless of host affinity
    submit = engine.submit

    def record_future(fn, *args):
        future = submit(fn, *args)
        futures.append(future)
        return future

    monkeypatch.setattr(engine, "submit", record_future)
    stream = io_engine.ReadStream(
        engine, [entry], io_engine.FixedBudget(buffer_bytes=4096, headroom=8192), None,
    )
    stream._cond = _CompletionBoundary(
        stream._cond, stream, reader_ident, published, proceed,
    )
    try:
        with stream._cond:
            stream._pump()
        assert published.wait(60), "reader never published its completed read"
        assert stream.held_bytes() == 4096
        assert len(refs) == 1 and refs[0]() is not None
        with stream.paused():
            if disposal == "reclaim":
                assert stream.reclaim(4096) == 4096
                assert stream.counters["evictions"] == 1
            else:
                delivered = stream.take(0)
                assert len(delivered) == 1
                assert isinstance(delivered[0].value, _DecodedValue)
                # Taking transfers ownership, rather than freeing the value.
                assert refs[0]() is delivered[0].value
                delivered = None
                stream.release()
            assert stream.held_bytes() == 0
            assert stream.held_actual_bytes() == 0
            assert refs[0]() is None, (
                f"{disposal} returned, but the completed reader still owns the value"
            )
    finally:
        # Unblock even on assertion failure; never leave a parked pool thread.
        proceed.set()
        stream.close()
        if engine._pool is not None:
            engine._pool.shutdown(wait=True)
        for future in futures:
            future.result(timeout=60)


def test_named_regression_ignores_unrelated_cgroup_allocations(tmp_path, monkeypatch):
    """Run the actual named regression with deterministic ambient interference.

    Real archives, the PWC decoder, sealed buffers, reclaim and take/drop are
    untouched. Only the old ambient observation boundary is injected: another
    process allocates 64 MiB of shmem between each cgroup reading. That must
    not turn a successful disposal of our own mappings into a test failure.
    Keep the fixture small enough for two workers in the admitted CPU budget.
    """
    monkeypatch.delenv("PRISMABUILD_RESIDENCY_MAP", raising=False)
    scope = tmp_path / "ambient-cgroup"
    scope.mkdir()
    (scope / "cgroup.procs").write_text(str(os.getpid()))
    samples = itertools.count(1)

    def unrelated_allocations(_scope):
        shmem = next(samples) * (64 << 20)
        anon = 32 << 20
        return {"committed": anon + shmem, "anon_shmem": anon + shmem,
                "shmem": shmem, "anon": anon}

    # These are optional legacy observation hooks. A deterministic rewrite
    # can delete them; this harness must not require retaining obsolete code.
    monkeypatch.setattr(sealed_memory, "_own_cgroup", lambda: scope, raising=False)
    monkeypatch.setattr(sealed_memory, "_committed", unrelated_allocations, raising=False)
    monkeypatch.setattr(sealed_memory, "READING_GRAIN", 4096, raising=False)
    monkeypatch.setattr(sealed_memory, "RENDER_SHAPE", (128, 128), raising=False)
    monkeypatch.setattr(sealed_memory, "PER_GROUP", 2, raising=False)
    sealed_memory.test_freed_read_ahead_leaves_the_committed_reading(tmp_path)
