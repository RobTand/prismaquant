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
            pending.append((target, source))
            return target
        return copy(target, source, non_blocking=non_blocking)

    class Event:
        def __init__(self):
            self.copies = []
            self.waits = 0
            events.append(self)

        def record(self, stream):
            assert stream == "compute"
            self.copies, pending[:] = pending[:], []

        def synchronize(self):
            self.waits += 1
            for target, source in self.copies:
                copy(target, source)
                busy.remove(id(target))
            self.copies = []

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "compute")
    return allocations, events, busy


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
