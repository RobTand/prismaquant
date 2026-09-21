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
from concurrent.futures import ThreadPoolExecutor

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
    from prismaquant.staged_tier_policy import refuse_pool_bulk_read
    reads: list[int] = []

    def reader(prefix, *args, **kwargs):
        L = int(prefix.rstrip(".").rsplit(".", 1)[-1])
        reads.append(L)
        if len(reads) == 1:
            # Speculative prefetch: the mover has not landed yet.
            raise refuse_pool_bulk_read(prefix, "readset-not-staged")
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
