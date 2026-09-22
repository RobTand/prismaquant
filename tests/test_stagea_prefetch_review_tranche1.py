"""Review tranche 1 regressions: per-context cancel, no read after cancel.

Covers prefetch-review blockers 1-4 at the seams, against the CURRENT
implementation first (red), then the fix (green):

1. Cancellation isolation: shutting down one StreamingContext must not
   abort another context's staged wait. The production wait honors
   whatever event the implementation threads through; the fake below
   honors the same source (the context's own event when the
   implementation provides one, else the legacy module-global event),
   with per-context semantics for the former (CancelledError, no payload
   read) and legacy semantics for the latter (generic refusal after a
   payload attempt). Either way the assertions are identical.
2. A cancelled wait raises CancelledError promptly; it never resolves as
   UNCOVERED and never falls through to payload reading. The owning
   future therefore holds CancelledError, and teardown drains without
   masking the primary capture failure.
3. A declared-but-unlanded span raises a typed wait-expiry error; an
   undeclared span and a refused covering entry keep the existing generic
   refusal. Unknown, integrity, and cancellation never become the typed
   cause (predicate narrowing itself is the next tranche).
4. Handoffs use poll-entry/resolver events, not sleeps; timing margins
   are stated as generous bounds against a 30 s production-shaped bound,
   not as determinism claims.
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
BOUND_S = 30.0


def _tensors(L: int):
    return {f"model.layers.{L}.weight": torch.full(
        (LAYER_BYTES // 4,), float(L), dtype=torch.float32)}


def _make_ctx(monkeypatch, reader, *, num_layers=8, workers=1):
    monkeypatch.setattr(streaming_model, "_read_layer_to_device", reader)
    ctx = object.__new__(StreamingContext)
    ctx.layers_prefix = "model.layers."
    ctx.num_layers = num_layers
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
    ctx.prefetch_workers = workers
    ctx.prefetch_min_available_bytes = 0
    ctx.prefetch_memory_skips = 0
    ctx.prefetch_delivered_unretained = 0
    ctx.prefetch_released_stale = 0
    ctx._inflight = {}
    ctx._inflight_lock = threading.Lock()
    ctx._last_installed = None
    ctx._walk_step = 0
    # Mirrors production __init__: each context owns its wait-cancellation
    # event. Implementations predating per-context ownership ignore it.
    ctx._staged_wait_cancel = threading.Event()
    ctx.layer_cache = LayerCache(max_bytes=64 * LAYER_BYTES, max_entries=None)
    ctx.max_cache_slots = ctx.layer_cache.max_entries
    ctx.prefetch_pool = ThreadPoolExecutor(max_workers=workers)
    return ctx


def _blocking_reader(ctx, entered, landed, L_wanted):
    """A staged-wait-shaped read honoring the implementation's event source.

    Mirrors production on both sides of the revision: when the legacy
    module-global event exists and fires, the read behaves the legacy way
    (a payload attempt that refuses generically); when the context's own
    event fires, the cancelled owner reads no payload and raises
    CancelledError. A hard production-shaped bound always applies, and
    landing always serves, so no path blocks the suite's teardown.
    """
    from prismaquant import layer_streaming as ls

    def reader(prefix, *args, **kwargs):
        L = int(prefix.rstrip(".").rsplit(".", 1)[-1])
        assert L == L_wanted
        entered.set()
        own = getattr(ctx, "_staged_wait_cancel", None)
        legacy = getattr(ls, "_STAGED_WAIT_CANCEL", None)
        deadline = time.monotonic() + BOUND_S
        while not landed.is_set():
            if time.monotonic() >= deadline:
                break
            if legacy is not None and legacy.is_set():
                break
            if own is not None:
                if own.wait(0.2):
                    break
            elif legacy is not None:
                if legacy.wait(0.2):
                    break
            else:
                time.sleep(0.2)
        if landed.is_set():
            return _tensors(L)
        if own is not None and own.is_set():
            raise CancelledError(f"layer {L} staged wait cancelled")
        from prismaquant.staged_tier_policy import refuse_pool_bulk_read
        raise refuse_pool_bulk_read(prefix, "readset-not-staged")

    return reader


def test_context_shutdown_does_not_abort_other_context(monkeypatch):
    entered_a, entered_b = threading.Event(), threading.Event()
    landed_a, landed_b = threading.Event(), threading.Event()
    # Placeholders replaced once each context exists (the reader closes
    # over its own context).
    refs = {}

    def make_reader(tag, entered, landed):
        def reader(prefix, *args, **kwargs):
            return _blocking_reader(
                refs[tag], entered, landed,
                1 if tag == "a" else 2)(prefix, *args, **kwargs)
        return reader

    ctx_a = _make_ctx(monkeypatch, None, workers=1)
    ctx_b = _make_ctx(monkeypatch, None, workers=1)
    refs["a"], refs["b"] = ctx_a, ctx_b
    # Bind each context to its own blocking reader (monkeypatch is
    # module-global, so serialize: A first, then rebind for B after A is
    # shut down and joined).
    monkeypatch.setattr(
        streaming_model, "_read_layer_to_device",
        make_reader("a", entered_a, landed_a))
    fut_a = ctx_a.schedule_prefetch(1)
    assert fut_a is not None
    assert entered_a.wait(timeout=10.0), "context A worker never started"
    monkeypatch.setattr(
        streaming_model, "_read_layer_to_device",
        make_reader("b", entered_b, landed_b))
    fut_b = ctx_b.schedule_prefetch(2)
    assert fut_b is not None
    assert entered_b.wait(timeout=10.0), "context B worker never started"

    try:
        began = time.monotonic()
        ctx_a.shutdown()
        elapsed_a = time.monotonic() - began
        assert elapsed_a < 10.0, f"A shutdown took {elapsed_a:.1f}s of 30s bound"
        with pytest.raises(CancelledError):
            fut_a.result()
        with ctx_a._inflight_lock:
            assert ctx_a._inflight == {}
        assert ctx_a.prefetch_pool._shutdown
        # The live coexisting context is untouched by A's teardown.
        assert not fut_b.done(), "shutting down A aborted B's staged wait"
        landed_b.set()
        tensors, src = ctx_b.ensure_loaded(2)
        assert torch.equal(tensors["model.layers.2.weight"][0], torch.tensor(2.0))
        assert src in ("hot", "wait")
    finally:
        # Bound teardown on every path: idle pool threads are non-daemon
        # and would hold the pytest worker past the suite otherwise.
        landed_a.set()
        landed_b.set()
        ctx_a.shutdown()
        ctx_b.shutdown()
    with ctx_b._inflight_lock:
        assert ctx_b._inflight == {}
    assert ctx_b.prefetch_pool._shutdown


def test_shutdown_does_not_mask_primary_capture_failure(monkeypatch):
    entered, landed = threading.Event(), threading.Event()
    refs = {}
    ctx = _make_ctx(monkeypatch, None, workers=1)
    refs["c"] = ctx
    monkeypatch.setattr(
        streaming_model, "_read_layer_to_device",
        _blocking_reader(ctx, entered, landed, 1))
    fut = ctx.schedule_prefetch(1)
    assert fut is not None
    assert entered.wait(timeout=10.0), "worker never started"
    with pytest.raises(RuntimeError, match="primary"):
        try:
            raise RuntimeError("primary capture failure")
        finally:
            ctx.shutdown()
    landed.set()


def test_cancelled_staged_wait_raises_cancelled_error():
    """The wait seam aborts with CancelledError, never UNCOVERED."""
    from prismaquant.residency_shard_reader import await_staged_spans

    class _Uncovered:
        def __init__(self):
            self.polls = 0
            self.first_poll = threading.Event()

        def staged_range_outcome(self, declared, start, end, declared_size=None):
            from prismaquant.residency_map import RANGE_UNCOVERED
            self.polls += 1
            if self.polls == 1:
                self.first_poll.set()
            return None, RANGE_UNCOVERED

        def record_range_wait(self, *args, **kwargs):
            return None

    resolver = _Uncovered()
    cancel = threading.Event()
    started, done = threading.Event(), threading.Event()
    outcome = {}

    def waiter():
        started.set()
        try:
            outcome["verdict"] = await_staged_spans(
                resolver, [("declared", 0, 8, 16)],
                deadline=time.monotonic() + BOUND_S, cancel=cancel)
        except CancelledError as exc:
            outcome["cancelled"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    assert started.wait(timeout=10.0), "waiter never started"
    assert resolver.first_poll.wait(timeout=10.0), "wait never polled"
    cancel.set()
    assert done.wait(timeout=10.0), "cancelled wait did not finish promptly"
    thread.join(timeout=10.0)
    assert "cancelled" in outcome, (
        f"cancel resolved as {outcome.get('verdict')!r}, not CancelledError")
    assert isinstance(outcome["cancelled"], CancelledError)


def test_declared_unlanded_span_raises_typed_expiry():
    """UNCOVERED at the read seam is a typed transient cause."""
    from prismaquant import staged_tier_policy as policy
    from prismaquant.residency_map import RANGE_UNCOVERED
    from prismaquant.residency_shard_reader import StagedShardReader

    cls = getattr(policy, "StagedRangeNotLanded", None)
    assert cls is not None, "no typed wait-expiry cause produced"

    class _Stub:
        def staged_range(self, *args, **kwargs):
            return None

        def staged_range_outcome(self, *args, **kwargs):
            return None, RANGE_UNCOVERED

    reader = object.__new__(StagedShardReader)
    reader._bound = []
    reader._strict = True
    reader._resolver = _Stub()
    reader._declared = "declared"
    reader._declared_size = 64
    with pytest.raises(cls) as info:
        reader._range_for(0, 8)
    assert info.value.start == 0 and info.value.end == 8


def test_undeclared_and_refused_spans_keep_generic_refusal():
    """Neither unknown nor integrity evidence becomes the typed cause."""
    from prismaquant import staged_tier_policy as policy
    from prismaquant.residency_map import RANGE_REFUSED, RANGE_UNDECLARED
    from prismaquant.residency_shard_reader import StagedShardReader
    from prismaquant.staged_tier_policy import TierPolicyRefused

    typed = getattr(policy, "StagedRangeNotLanded", None)

    class _Stub:
        def __init__(self, outcome):
            self._outcome = outcome

        def staged_range(self, *args, **kwargs):
            return None

        def staged_range_outcome(self, *args, **kwargs):
            return None, self._outcome

    for outcome in (RANGE_UNDECLARED, RANGE_REFUSED):
        reader = object.__new__(StagedShardReader)
        reader._bound = []
        reader._strict = True
        reader._resolver = _Stub(outcome)
        reader._declared = "declared"
        reader._declared_size = 64
        with pytest.raises(TierPolicyRefused) as info:
            reader._range_for(0, 8)
        assert type(info.value) is TierPolicyRefused, (
            f"{outcome} must keep the generic refusal, got {type(info.value)}")
        if typed is not None:
            assert not isinstance(info.value, typed)
