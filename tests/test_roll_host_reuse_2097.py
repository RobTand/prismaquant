"""CPU control-flow qualification; simulated CUDA is not device evidence."""
from __future__ import annotations

import pytest
import torch

from prismaquant.joint_adjoint_checkpoints import _RollPipeline


@pytest.fixture
def cuda_copy(monkeypatch):
    """Defer copies until their event fence, retaining real CPU tensor bytes."""
    allocate, copy = torch.empty, torch.Tensor.copy_
    owned, pending, busy, allocations, events = set(), [], set(), [], []

    def empty(*args, **kwargs):
        pinned = kwargs.pop("pin_memory", False)
        tensor = allocate(*args, **kwargs)
        if pinned:
            owned.add(id(tensor))
            allocations.append(tensor)
        return tensor

    def copy_(target, source, non_blocking=False):
        if id(target) in owned and non_blocking:
            assert id(target) not in busy, "buffer reused before its copy landed"
            busy.add(id(target))
            pending.append((target, source, torch.cuda.current_stream("cuda")))
            return target
        return copy(target, source, non_blocking=non_blocking)

    class Event:
        def __init__(self):
            self.copies = []
            self.waits = 0
            events.append(self)

        def record(self, stream):
            assert stream == "compute"
            self.copies = [item for item in pending if item[2] is stream]
            pending[:] = [item for item in pending if item[2] is not stream]

        def synchronize(self):
            self.waits += 1
            for target, source, _stream in self.copies:
                copy(target, source)
                busy.remove(id(target))
            self.copies = []

    class Stream:
        blocked = False
        fences = 0

        def __eq__(self, value):
            return value == "compute" if isinstance(value, str) else self is value

        def synchronize(self):
            self.fences += 1
            if self.blocked:
                raise RuntimeError("original copy stream still blocked")
            # A real stream fence proves completion even when event record
            # or synchronization failed. Do not call those broken methods.
            for event in events:
                for target, source, owner in event.copies:
                    if owner is self:
                        copy(target, source)
                        busy.remove(id(target))
                event.copies = [item for item in event.copies if item[2] is not self]
            for target, source, owner in pending:
                if owner is self:
                    copy(target, source)
                    busy.remove(id(target))
            pending[:] = [item for item in pending if item[2] is not self]

    stream = Stream()

    class State:
        def __iter__(self):
            return iter((allocations, events, busy))

    state = State()
    state.stream = stream

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    return state


def _gradient(step, count=2, *, width=8, dtype=torch.float32):
    return torch.arange(count * width, dtype=dtype).reshape(count, 1, width) + step * 100


@pytest.mark.parametrize("reuse", [False, True])
def test_exact_rows_order_and_two_bank_bound(cuda_copy, reuse):
    allocations, events, busy = cuda_copy
    received, expected, durable, launched, identities = [], [], [], [0], []

    def roll(row, batch, probe):
        assert not busy.intersection({id(row)})
        assert row.storage_offset() == 0
        assert row.untyped_storage().nbytes() == row.numel() * row.element_size()
        received.append((probe, batch, row.clone(), launched[0]))
        identities.append(row.data_ptr())

    pipeline = _RollPipeline(roll, device="cuda", roll_may_keep=False,
                             reuse_host_buffers=reuse,
                             on_durable=lambda batch, probe: durable.append((probe, batch)))
    for step in range(8):
        count = 1 if step == 7 else 2
        gradient = _gradient(step, count)
        indices = list(range(step * 2, step * 2 + count))
        expected.extend((step % 3, index, gradient[row:row + 1].clone())
                        for row, index in enumerate(indices))
        launched[0] += 1
        pipeline.submit(gradient, indices, step % 3)
        if step == 3:
            pipeline.drain()  # Read-window barrier must not discard reusable banks.
    pipeline.drain()
    assert [(p, b) for p, b, _r, _n in received] == durable
    assert len(received) == len(expected) == 15
    for (probe, batch, row, launch), (want_probe, want_batch, want) in zip(received, expected):
        assert (probe, batch) == (want_probe, want_batch)
        assert torch.equal(row.view(torch.uint8), want.view(torch.uint8))
        expected_launch = 4 if batch // 2 == 3 else min(batch // 2 + 2, 8)
        assert launch == expected_launch
    assert not busy
    assert all(event.waits == 1 for event in events)
    assert len(allocations) == (4 if reuse else 15)
    assert len(set(identities)) == (4 if reuse else 15)


def test_default_still_allocates_and_retained_rows_are_independent(cuda_copy):
    allocations, _events, _busy = cuda_copy
    kept = []
    pipeline = _RollPipeline(lambda row, _batch, _probe: kept.append(row), device="cuda")
    for step in range(4):
        pipeline.submit(_gradient(step, 1), [step], 0)
    pipeline.drain()
    assert len(allocations) == 4
    for step, row in enumerate(kept):
        assert torch.equal(row, _gradient(step, 1))
        assert row.data_ptr() != allocations[step].data_ptr()


@pytest.mark.parametrize("keep", [True, 1])
def test_reuse_refuses_a_consumer_that_may_keep_rows(keep):
    with pytest.raises(ValueError, match="keep"):
        _RollPipeline(lambda *args: None, device="cuda", roll_may_keep=keep,
                      reuse_host_buffers=True)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_reuse_requires_an_explicit_bool(value):
    with pytest.raises(ValueError, match="bool"):
        _RollPipeline(lambda *args: None, device="cuda", roll_may_keep=False,
                      reuse_host_buffers=value)


@pytest.mark.parametrize("change", ["width", "dtype", "count"])
def test_layout_drift_refuses_before_another_copy(cuda_copy, change):
    allocations, events, busy = cuda_copy
    rolled = []
    pipeline = _RollPipeline(lambda *args: rolled.append(args), device="cuda",
                             roll_may_keep=False, reuse_host_buffers=True)
    pipeline.submit(_gradient(0), [0, 1], 0)
    changed = _gradient(1, 3 if change == "count" else 2,
                        width=9 if change == "width" else 8,
                        dtype=torch.float64 if change == "dtype" else torch.float32)
    with pytest.raises(RuntimeError, match="layout"):
        pipeline.submit(changed, list(range(changed.shape[0])), 0)
    assert len(allocations) == 2
    assert len(events) == 1
    pipeline.abandon()
    assert not busy and not rolled


def test_failed_roll_abandons_current_copy_without_delivering_it(cuda_copy):
    _allocations, events, busy = cuda_copy
    rolled = []

    def fail(row, batch, probe):
        rolled.append(batch)
        raise RuntimeError("writer failed")

    pipeline = _RollPipeline(fail, device="cuda", roll_may_keep=False,
                             reuse_host_buffers=True)
    pipeline.submit(_gradient(0, 1), [0], 0)
    with pytest.raises(RuntimeError, match="writer failed"):
        pipeline.submit(_gradient(1, 1), [1], 0)
    pipeline.abandon()
    assert rolled == [0]
    assert not busy
    assert all(event.waits == 1 for event in events)


def test_cpu_path_remains_owned_by_gradient():
    got = []
    pipeline = _RollPipeline(lambda row, *args: got.append(row), device="cpu",
                             roll_may_keep=False, reuse_host_buffers=True)
    gradient = _gradient(0, 1)
    pipeline.submit(gradient, [0], 0)
    pipeline.drain()
    assert got[0].data_ptr() == gradient.data_ptr()


def test_reused_rows_write_the_exact_allocating_paths_entry_files(cuda_copy, tmp_path):
    from pathlib import Path
    from prismaquant.perturbed_x_cache import write_exact_activation_cache_entry

    files = []
    for reuse in (False, True):
        written = []

        def roll(row, batch, probe):
            name = f"cotangent-{probe}-{batch}"
            nbytes = row.numel() * row.element_size()
            ref = write_exact_activation_cache_entry(
                tmp_path / str(reuse), name, row, identity={"slot": name},
                max_tensor_bytes=nbytes, max_file_bytes=nbytes + 65536)
            written.append((name, Path(ref.path).read_bytes(), ref.sha256))

        pipeline = _RollPipeline(roll, device="cuda", roll_may_keep=False,
                                 reuse_host_buffers=reuse)
        for step in range(6):
            pipeline.submit(_gradient(step), [2 * step, 2 * step + 1], step % 2)
        pipeline.drain()
        files.append(written)
    assert files[0] == files[1]


def test_partial_copy_failure_is_fenced_before_abandon(cuda_copy, monkeypatch):
    allocations, events, busy = cuda_copy
    original = torch.Tensor.copy_
    calls = [0]

    def failing(target, source, non_blocking=False):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("copy failed")
        return original(target, source, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "copy_", failing)
    rolled = []
    pipeline = _RollPipeline(lambda *args: rolled.append(args), device="cuda",
                             roll_may_keep=False, reuse_host_buffers=True)
    with pytest.raises(RuntimeError, match="copy failed"):
        pipeline.submit(_gradient(0), [0, 1], 0)
    assert len(allocations) == 2 and len(events) == 1
    assert events[0].waits == 1 and not busy
    pipeline.abandon()
    assert not rolled


@pytest.mark.parametrize("operation", ["submit", "drain", "abandon"])
def test_reentrant_callbacks_refuse_before_reusing_or_delivering_a_bank(cuda_copy, operation):
    _allocations, events, busy = cuda_copy

    def roll(row, batch, probe):
        if operation == "submit":
            pipeline.submit(_gradient(2, 1), [2], 0)
        else:
            getattr(pipeline, operation)()

    pipeline = _RollPipeline(roll, device="cuda", roll_may_keep=False,
                             reuse_host_buffers=True)
    pipeline.submit(_gradient(0, 1), [0], 0)
    with pytest.raises(RuntimeError, match="reenter"):
        pipeline.submit(_gradient(1, 1), [1], 0)
    pipeline.abandon()
    assert not busy and len(events) == 2


def test_durable_callback_cannot_overwrite_undelivered_rows(cuda_copy):
    _allocations, _events, busy = cuda_copy
    delivered = []

    def durable(batch, probe):
        # Row 1 of bank 0 is still awaiting delivery here. A nested third
        # submit would reuse bank 0 and overwrite it without this fence.
        pipeline.submit(_gradient(2), [4, 5], 0)

    pipeline = _RollPipeline(lambda row, batch, probe: delivered.append(batch),
                             device="cuda", roll_may_keep=False,
                             reuse_host_buffers=True, on_durable=durable)
    pipeline.submit(_gradient(0), [0, 1], 0)
    with pytest.raises(RuntimeError, match="reenter"):
        pipeline.submit(_gradient(1), [2, 3], 0)
    pipeline.abandon()
    assert delivered == [0] and not busy


@pytest.mark.parametrize("failure", ["create", "record", "sync"])
def test_unfenced_partial_copy_keeps_banks_until_original_stream_fences(
        cuda_copy, monkeypatch, failure):
    import gc
    import weakref

    allocations, events, busy = cuda_copy
    original_event = torch.cuda.Event
    original_copy = torch.Tensor.copy_
    calls = [0]
    rolled = []
    pipeline = _RollPipeline(lambda *args: rolled.append(args), device="cuda",
                             roll_may_keep=False, reuse_host_buffers=True)
    pipeline.submit(_gradient(0), [0, 1], 0)
    previous = pipeline._waiting

    def failed_copy(target, source, non_blocking=False):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("copy failed")
        return original_copy(target, source, non_blocking=non_blocking)

    def failed_fence():
        if failure == "create":
            raise RuntimeError("event creation failed")
        event = original_event()
        def fail(*args):
            raise RuntimeError("event fence failed")
        if failure == "record":
            event.record = fail
        else:
            event.synchronize = fail
        return event

    monkeypatch.setattr(torch.Tensor, "copy_", failed_copy)
    monkeypatch.setattr(torch.cuda, "Event", failed_fence)
    with pytest.raises(RuntimeError, match="copy failed") as primary:
        pipeline.submit(_gradient(1), [2, 3], 0)
    assert primary.value.__notes__
    del primary
    cuda_copy.stream.blocked = True
    # Cleanup must not use a newly current stream. Its success proves
    # nothing about the stream that actually submitted these copies.
    decoy = type(cuda_copy.stream)()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: decoy)
    with pytest.raises(RuntimeError, match="original copy stream still blocked"):
        pipeline.abandon()
    assert any(bank is not None for bank in pipeline._host_banks)
    assert pipeline._waiting is previous
    assert pipeline.held_host_bytes == sum(t.numel() * t.element_size() for t in allocations)
    assert not rolled and decoy.fences == 0
    with pytest.raises(RuntimeError, match="unfenced"):
        pipeline.submit(_gradient(2), [4, 5], 0)
    with pytest.raises(RuntimeError, match="unfenced"):
        _RollPipeline(lambda *args: None, device="cuda")
    weak = weakref.ref(pipeline)
    del pipeline, previous
    gc.collect()
    assert weak() is not None, "failed owner must survive lost traceback/caller"
    cuda_copy.stream.blocked = False
    weak().abandon()
    assert not busy and decoy.fences == 0
    assert weak() is None or weak().held_host_bytes == 0
    _RollPipeline(lambda *args: None, device="cuda")


@pytest.mark.parametrize("failure", ["create", "record"])
def test_completed_copy_with_failed_event_is_retained_until_stream_fences(
        cuda_copy, monkeypatch, failure):
    allocations, _events, busy = cuda_copy
    original_event = torch.cuda.Event

    def event():
        if failure == "create":
            raise RuntimeError("event creation failed")
        got = original_event()
        def refuse(stream):
            raise RuntimeError("event record failed")
        got.record = refuse
        return got

    monkeypatch.setattr(torch.cuda, "Event", event)
    pipeline = _RollPipeline(lambda *args: None, device="cuda", roll_may_keep=False,
                             reuse_host_buffers=True)
    with pytest.raises(RuntimeError, match="event"):
        pipeline.submit(_gradient(0), [0, 1], 0)
    cuda_copy.stream.blocked = True
    with pytest.raises(RuntimeError, match="original copy stream"):
        pipeline.abandon()
    assert pipeline.held_host_bytes == sum(t.numel() * t.element_size() for t in allocations)
    cuda_copy.stream.blocked = False
    pipeline.abandon()
    assert not busy and pipeline.held_host_bytes == 0


@pytest.mark.parametrize("operation", ["drain", "abandon"])
def test_previous_steps_failed_event_keeps_credit_until_original_stream_fences(
        cuda_copy, monkeypatch, operation):
    _allocations, events, busy = cuda_copy
    rolled = []
    pipeline = _RollPipeline(lambda *args: rolled.append(args), device="cuda",
                             roll_may_keep=False, reuse_host_buffers=True)
    pipeline.submit(_gradient(0), [0, 1], 0)
    previous = pipeline._waiting
    def failed_event():
        raise RuntimeError("previous event failed")
    events[0].synchronize = failed_event
    with pytest.raises(RuntimeError, match="previous event failed"):
        getattr(pipeline, operation)()
    assert pipeline._waiting is (None if operation == "drain" else previous)
    cuda_copy.stream.blocked = True
    with pytest.raises(RuntimeError, match="original copy stream"):
        pipeline.abandon()
    assert pipeline.held_host_bytes > 0 and not rolled
    cuda_copy.stream.blocked = False
    pipeline.abandon()
    assert not busy and pipeline.held_host_bytes == 0 and not rolled


def test_both_banks_keep_credit_until_both_original_streams_complete(cuda_copy, monkeypatch):
    allocations, _events, busy = cuda_copy
    old_copy, old_event = torch.Tensor.copy_, torch.cuda.Event
    pipeline = _RollPipeline(lambda *args: None, device="cuda", roll_may_keep=False,
                             reuse_host_buffers=True)
    pipeline.submit(_gradient(0), [0, 1], 0)
    second = type(cuda_copy.stream)()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: second)
    calls = [0]
    def failed_copy(target, source, non_blocking=False):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("copy failed")
        return old_copy(target, source, non_blocking=non_blocking)
    def broken_event():
        raise RuntimeError("event failed")
    monkeypatch.setattr(torch.Tensor, "copy_", failed_copy)
    monkeypatch.setattr(torch.cuda, "Event", broken_event)
    with pytest.raises(RuntimeError, match="copy failed"):
        pipeline.submit(_gradient(1), [2, 3], 0)
    second.blocked = True
    with pytest.raises(RuntimeError, match="original copy stream"):
        pipeline.abandon()
    assert cuda_copy.stream.fences == 1 and second.fences == 1
    assert busy and pipeline.held_host_bytes == sum(t.numel() * t.element_size() for t in allocations)
    second.blocked = False
    pipeline.abandon()
    assert not busy and pipeline.held_host_bytes == 0


def test_missing_stream_fence_refuses_before_any_copy(cuda_copy, monkeypatch):
    _allocations, events, busy = cuda_copy
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    pipeline = _RollPipeline(lambda *args: None, device="cuda", roll_may_keep=False,
                             reuse_host_buffers=True)
    with pytest.raises(RuntimeError, match="original copy stream fence"):
        pipeline.submit(_gradient(0), [0, 1], 0)
    assert not events and not busy
    pipeline.abandon()
    assert pipeline.held_host_bytes == 0


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("callback", ["roll", "durable"])
def test_failed_drain_never_redelivers_a_durable_prefix_or_none_rows(cuda_copy, reuse, callback):
    _allocations, _events, busy = cuda_copy
    rolled, durable = [], []

    def roll(row, batch, probe):
        assert isinstance(row, torch.Tensor), "drain redelivered a consumed None row"
        rolled.append(batch)
        if callback == "roll" and batch == 1:
            raise RuntimeError("second row callback failed")

    def on_durable(batch, probe):
        durable.append(batch)
        if callback == "durable" and batch == 1:
            raise RuntimeError("second row callback failed")

    pipeline = _RollPipeline(roll, device="cuda", roll_may_keep=False,
                             reuse_host_buffers=reuse, on_durable=on_durable)
    pipeline.submit(_gradient(0, 3), [0, 1, 2], 0)
    with pytest.raises(RuntimeError, match="second row callback failed"):
        pipeline.drain()
    expected = [0] if callback == "roll" else [0, 1]
    assert rolled == [0, 1] and durable == expected
    pipeline.drain()
    pipeline.abandon()
    pipeline.drain()
    assert rolled == [0, 1] and durable == expected and not busy
    assert pipeline.held_host_bytes == 0
