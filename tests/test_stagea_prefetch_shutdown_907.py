"""A failed Stage A capture cancels its prefetch waits promptly (PQ #907).

On r2 (2026-09-21) the capture failed at 21:55Z and the process sat in
teardown with the main thread in ``futex_do_wait`` until a prefetch
worker's own 300 s staged-range wait ran out. PrismaBuild recorded the
failure at 22:00:38Z, five minutes late, with the Spark idle.

A failed capture must cancel its prefetch workers' waits (they poll, so a
shared event checked in the poll loop is enough) and join them promptly.
No dangling thread, read, or lease: the pool is joined and every owned
future is released.

Deterministic bounds from events, not sleeps: cancellation wakes via
``Event.wait``; timing assertions use generous margins (seconds against a
tens-of-seconds bound) so they cannot flake.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import prismaquant.streaming_model as streaming_model
from prismaquant.layer_streaming import LayerCache
from prismaquant.streaming_model import StreamingContext


def _clear_cancel():
    try:
        from prismaquant.layer_streaming import clear_staged_wait_cancel
    except ImportError:
        return
    try:
        clear_staged_wait_cancel()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _clean_cancel():
    _clear_cancel()
    yield
    _clear_cancel()


class _Uncovered:
    """A resolver that never lands: every span stays uncovered."""

    def staged_range_outcome(self, declared, start, end, declared_size=None):
        from prismaquant.residency_map import RANGE_UNCOVERED
        return None, RANGE_UNCOVERED

    def record_range_wait(self, *args, **kwargs):
        return None


def test_await_honors_cancel_event():
    """A staged-range wait aborts promptly on cancellation."""
    from prismaquant.residency_shard_reader import await_staged_spans
    from prismaquant.residency_map import RANGE_UNCOVERED

    cancel = threading.Event()
    started = threading.Event()
    done = threading.Event()
    outcome = {}

    def waiter():
        resolver = _Uncovered()
        # A production-shaped bound (tens of seconds, not the full 300 s
        # so the fail-before run stays cheap): without cancellation this
        # takes the whole bound; with it, it ends when the event fires.
        deadline = time.monotonic() + 30.0
        started.set()
        outcome["verdict"] = await_staged_spans(
            resolver, [("declared", 0, 8, 16)],
            deadline=deadline, cancel=cancel)
        done.set()

    thread = threading.Thread(target=waiter, daemon=True)
    began = time.monotonic()
    thread.start()
    assert started.wait(timeout=10.0), "waiter never entered the poll loop"
    # Let it poll at least once, then cancel: deterministic handoff via
    # events, no sleep tuning.
    time.sleep(1.2)
    cancel.set()
    assert done.wait(timeout=10.0), "cancelled wait did not finish promptly"
    thread.join(timeout=10.0)
    elapsed = time.monotonic() - began
    assert outcome["verdict"] == RANGE_UNCOVERED
    # 30 s bound, cancelled after ~1 s: generous margin, cannot flake.
    assert elapsed < 10.0, f"cancel took {elapsed:.1f}s of a 30s bound"


def _make_blocking_ctx(monkeypatch, entered: threading.Event):
    """A StreamingContext whose prefetch blocks like a staged-range wait."""
    def fake_read(prefix, *args, **kwargs):
        from prismaquant import layer_streaming as ls
        entered.set()
        # Honor the shared cancellation like the real poll loop does:
        # wake promptly when shutdown requests it, else hold the bound.
        cancel = getattr(ls, "_STAGED_WAIT_CANCEL", None)
        if cancel is not None:
            cancel.wait(timeout=30.0)
            if cancel.is_set():
                from prismaquant.staged_tier_policy import refuse_pool_bulk_read
                raise refuse_pool_bulk_read(prefix, "readset-not-staged")
        else:
            time.sleep(30.0)
        from prismaquant.staged_tier_policy import refuse_pool_bulk_read
        raise refuse_pool_bulk_read(prefix, "readset-not-staged")

    monkeypatch.setattr(streaming_model, "_read_layer_to_device", fake_read)

    ctx = object.__new__(StreamingContext)
    ctx.layers_prefix = "model.layers."
    ctx.num_layers = 4
    ctx.weight_shard = {}
    ctx.weight_ckpt = {}
    ctx.buffer_dtypes = {}
    ctx.device = torch.device("cpu")
    ctx.dtype = torch.float32
    ctx.fp8_scale_inv_map = {}
    ctx.expert_packer = None
    ctx.concat_merger = None
    ctx.source_authentication = None
    ctx.estimated_layer_bytes = 1 << 20
    ctx.prefetch_workers = 1
    ctx.prefetch_min_available_bytes = 0
    ctx.prefetch_memory_skips = 0
    ctx.prefetch_delivered_unretained = 0
    ctx.prefetch_released_stale = 0
    ctx._inflight = {}
    ctx._inflight_lock = threading.Lock()
    ctx._last_installed = None
    ctx._walk_step = 0
    ctx.layer_cache = LayerCache(max_bytes=64 << 20, max_entries=None)
    ctx.max_cache_slots = ctx.layer_cache.max_entries
    ctx.prefetch_pool = ThreadPoolExecutor(max_workers=1)
    return ctx


def test_shutdown_drains_blocked_prefetch_promptly(monkeypatch):
    """Shutdown joins a blocked prefetch worker via cancellation, not the bound."""
    entered = threading.Event()
    ctx = _make_blocking_ctx(monkeypatch, entered)

    fut = ctx.schedule_prefetch(1)
    assert fut is not None
    assert entered.wait(timeout=10.0), "prefetch worker never started"

    began = time.monotonic()
    ctx.shutdown()
    elapsed = time.monotonic() - began

    assert elapsed < 10.0, f"shutdown took {elapsed:.1f}s of a 30s bound"
    with ctx._inflight_lock:
        assert ctx._inflight == {}, "owned futures were not drained"
    # The pool is joined: no dangling prefetch thread.
    assert ctx.prefetch_pool._shutdown, "prefetch pool was not shut down"
