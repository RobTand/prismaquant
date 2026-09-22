"""A speculative availability timeout must not poison the demand read.

Handover r2: a speculative prefetch looks 4 layers ahead with a 300 s
staged-range bound; when it runs the bound out, its future holds the
timeout, and the compute thread fails at once when it later needs that
layer even though the bytes have since landed (it will not need the
layer for ~10 minutes at ~2.7 min/layer).

Production-shaped: drive StreamingContext scheduling over a stub layer
reader. The first read (speculation) fails with availability
(readset-not-staged: the mover has not landed yet); the second read
(demand, after landing) succeeds. Demand must retry availability once
through the existing prefetch machinery with the bounded declared wait,
preserve integrity failures at once, never read the pool, never swallow,
never retry unboundedly.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor

import pytest
import torch

import prismaquant.streaming_model as streaming_model
from prismaquant.layer_streaming import LayerCache
from prismaquant.streaming_model import StreamingContext

LAYER_BYTES = 1 << 20


def _tensors(L: int):
    return {f"model.layers.{L}.weight": torch.full((LAYER_BYTES // 4,), float(L), dtype=torch.float32)}


def _make_ctx(monkeypatch, reader):
    monkeypatch.setattr(streaming_model, "_read_layer_to_device", reader)
    ctx = object.__new__(StreamingContext)
    ctx.layers_prefix = "model.layers."
    ctx.num_layers = 8
    ctx.weight_shard = {}
    ctx.weight_ckpt = {}
    ctx.buffer_dtypes = {}
    ctx.device = torch.device("cpu")
    ctx.dtype = torch.float32
    ctx.fp8_scale_inv_map = {}
    ctx.expert_packer = None
    ctx.concat_merger = None
    ctx.source_authentication = None
    ctx.estimated_layer_bytes = LAYER_BYTES
    ctx.prefetch_workers = 2
    ctx.prefetch_min_available_bytes = 0
    ctx.prefetch_memory_skips = 0
    ctx.prefetch_delivered_unretained = 0
    ctx.prefetch_released_stale = 0
    ctx._inflight = {}
    ctx._inflight_lock = threading.Lock()
    ctx._last_installed = None
    ctx._walk_step = 0
    ctx.layer_cache = LayerCache(max_bytes=64 * LAYER_BYTES, max_entries=None)
    ctx.max_cache_slots = ctx.layer_cache.max_entries
    ctx.prefetch_pool = ThreadPoolExecutor(max_workers=2)
    return ctx


def test_speculative_availability_timeout_does_not_poison_demand(monkeypatch):
    """Speculation times out, bytes land, demand retries once and serves."""
    reads: list[int] = []

    def reader(prefix, *args, **kwargs):
        from prismaquant.staged_tier_policy import StagedRangeNotLanded
        L = int(prefix.rstrip(".").rsplit(".", 1)[-1])
        reads.append(L)
        if len(reads) == 1:
            # Speculative prefetch: the declared range has not landed yet.
            raise StagedRangeNotLanded(prefix, 0, 1)
        return _tensors(L)

    ctx = _make_ctx(monkeypatch, reader)
    try:
        fut = ctx.schedule_prefetch(3)
        assert fut is not None
        # Speculation runs out its wait: the future holds availability.
        with pytest.raises(Exception):
            fut.result()
        # Bytes land. Demand must retry once through the prefetch
        # machinery and serve, not poison with the stale timeout.
        tensors, src = ctx.ensure_loaded(3)
        assert torch.equal(tensors["model.layers.3.weight"][0], torch.tensor(3.0))
        assert src in ("wait", "cold")
        # Exactly one bounded retry: speculation + demand, never unbounded.
        assert reads == [3, 3], f"expected one retry, got {reads}"
    finally:
        ctx.shutdown()


def test_speculative_integrity_failure_is_preserved(monkeypatch):
    """Integrity still fails at once: no retry, no swallow, no pool read."""
    from prismaquant.staged_lease import LeaseRefused
    reads: list[int] = []

    def reader(prefix, *args, **kwargs):
        L = int(prefix.rstrip(".").rsplit(".", 1)[-1])
        reads.append(L)
        raise LeaseRefused("lease-open-size-changed", kind="integrity")

    ctx = _make_ctx(monkeypatch, reader)
    try:
        fut = ctx.schedule_prefetch(3)
        assert fut is not None
        with pytest.raises(Exception):
            fut.result()
        with pytest.raises(LeaseRefused):
            ctx.ensure_loaded(3)
        # No retry on integrity: speculation only, never a second read.
        assert reads == [3], f"integrity must not retry, got {reads}"
    finally:
        ctx.shutdown()


def test_unknown_error_with_availability_words_is_not_retried(monkeypatch):
    """Incidental wording is not proof of a transient declared-range cause."""
    reads: list[int] = []

    def reader(prefix, *args, **kwargs):
        L = int(prefix.rstrip(".").rsplit(".", 1)[-1])
        reads.append(L)
        if len(reads) == 1:
            # Corrupt fixture whose message happens to contain availability
            # words: no typed cause, so demand must refuse, not retry.
            raise RuntimeError(
                "stage unpublished then file-missing at readset-not-staged")
        return _tensors(L)

    ctx = _make_ctx(monkeypatch, reader)
    try:
        fut = ctx.schedule_prefetch(3)
        assert fut is not None
        with pytest.raises(Exception):
            fut.result()
        with pytest.raises(RuntimeError, match="file-missing"):
            ctx.ensure_loaded(3)
        assert reads == [3], f"unknown failure must not retry, got {reads}"
    finally:
        ctx.shutdown()


def test_second_availability_failure_propagates_after_single_retry(monkeypatch):
    """Exactly one bounded retry: a still-unlanded range refuses loudly."""
    from prismaquant.staged_tier_policy import StagedRangeNotLanded
    reads: list[int] = []

    def reader(prefix, *args, **kwargs):
        L = int(prefix.rstrip(".").rsplit(".", 1)[-1])
        reads.append(L)
        raise StagedRangeNotLanded(prefix, 0, 1)

    ctx = _make_ctx(monkeypatch, reader)
    try:
        fut = ctx.schedule_prefetch(3)
        assert fut is not None
        with pytest.raises(Exception):
            fut.result()
        with pytest.raises(StagedRangeNotLanded):
            ctx.ensure_loaded(3)
        assert reads == [3, 3], f"expected exactly one retry, got {reads}"
    finally:
        ctx.shutdown()


def test_successful_prefetch_served_with_single_read(monkeypatch):
    """Already-queued success delivery is untouched by retry logic."""
    reads: list[int] = []

    def reader(prefix, *args, **kwargs):
        L = int(prefix.rstrip(".").rsplit(".", 1)[-1])
        reads.append(L)
        return _tensors(L)

    ctx = _make_ctx(monkeypatch, reader)
    try:
        fut = ctx.schedule_prefetch(3)
        assert fut is not None
        fut.result()
        tensors, src = ctx.ensure_loaded(3)
        assert torch.equal(tensors["model.layers.3.weight"][0], torch.tensor(3.0))
        assert src in ("hot", "wait")
        assert reads == [3], f"success must cost one read, got {reads}"
    finally:
        ctx.shutdown()


def test_cancelled_context_performs_no_demand_read(monkeypatch):
    """A shut-down context reads, installs, and retries nothing on demand."""
    reads: list[int] = []

    def reader(prefix, *args, **kwargs):
        L = int(prefix.rstrip(".").rsplit(".", 1)[-1])
        reads.append(L)
        return _tensors(L)

    ctx = _make_ctx(monkeypatch, reader)
    ctx._staged_wait_cancel = threading.Event()
    try:
        # Poison the future first: cancellation must still win over retry.
        from prismaquant.staged_tier_policy import StagedRangeNotLanded

        def poison(prefix, *args, **kwargs):
            reads.append(3)
            raise StagedRangeNotLanded(prefix, 0, 1)

        monkeypatch.setattr(streaming_model, "_read_layer_to_device", poison)
        fut = ctx.schedule_prefetch(3)
        assert fut is not None
        with pytest.raises(Exception):
            fut.result()
        monkeypatch.setattr(streaming_model, "_read_layer_to_device", reader)
        ctx._staged_wait_cancel.set()
        with pytest.raises(CancelledError):
            ctx.ensure_loaded(3)
        assert reads == [3], f"cancelled demand must add no read, got {reads}"
    finally:
        ctx.shutdown()


def test_settle_fails_closed_on_failed_future(monkeypatch):
    """Certification never retries, cold-reads, or ignores a failed future."""
    from prismaquant.staged_lease import LeaseRefused
    reads: list[int] = []

    def reader(prefix, *args, **kwargs):
        L = int(prefix.rstrip(".").rsplit(".", 1)[-1])
        reads.append(L)
        raise LeaseRefused("lease-open-size-changed", kind="integrity")

    ctx = _make_ctx(monkeypatch, reader)
    try:
        fut = ctx.schedule_prefetch(3)
        assert fut is not None
        with pytest.raises(Exception):
            fut.result()
        with pytest.raises(LeaseRefused):
            ctx.settle_prefetched_layers([3])
        assert reads == [3], f"settle must add no read, got {reads}"
    finally:
        ctx.shutdown()


def test_snapshot_never_waits_nor_loads(monkeypatch):
    """Inspection describes a pending future without touching it."""
    entered = threading.Event()

    def reader(prefix, *args, **kwargs):
        entered.set()
        time.sleep(30.0)
        return _tensors(3)

    ctx = _make_ctx(monkeypatch, reader)
    ctx.layers = [torch.nn.Linear(4, 4) for _ in range(8)]
    try:
        fut = ctx.schedule_prefetch(3)
        assert fut is not None
        assert entered.wait(timeout=10.0), "worker never started"
        began = time.monotonic()
        snapshot = ctx.source_residency_snapshot([3])
        elapsed = time.monotonic() - began
        assert elapsed < 10.0, f"snapshot waited {elapsed:.1f}s on a pending future"
        states = [o.get("state") for o in snapshot["owners"]
                  if o.get("owner") == "prefetch_future"]
        assert states == ["pending"], f"expected one pending future, got {states}"
        fut.cancel()
    finally:
        ctx.shutdown()
