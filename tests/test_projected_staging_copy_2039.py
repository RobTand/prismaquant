"""PQ #2039: the device check stages source bytes into pinned memory once.

Call-site attribution for the remaining host-copy cost: the legacy device
preparation reads a pageable source tensor and performs one private
``pinned.copy_(weight)`` per unit. The pinned-direct path fills a pinned
buffer at the staged reader and the check adopts it with no second copy.
Both paths preserve bytes, dtype, shape, first-mismatch order, release-once
and cancellation behavior.

The pinned reads run on every host. Where torch has CUDA the requests reach the
real page-locked allocator and the tests also prove the pages are locked.
Elsewhere a recording stand-in answers with pageable memory, so every byte of
the read, the fence and the refusal still runs, and only the CUDA run says the
memory is page-locked (``_PinnedMemory.real``).
"""
import json
import os

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import layer_streaming, residency_shard_reader
from prismaquant import tessera_campaign as campaign
from prismaquant.residency_map import (
    ENV_VAR, reset_residency_resolver_for_tests, residency_report,
)
from prismaquant.tessera_expert_projection import source_unit_weight
from tests.test_residency_shard_reader import (
    _bind, _header, _mount_table, _stage_range, _stage_root, _use_table, _write_map,
)

TENSOR = 'model.layers.0.mlp.experts.0.gate_proj.weight'
ROWS, COLS = 16, 32


@pytest.fixture(autouse=True)
def _forget_resolver(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    reset_residency_resolver_for_tests()
    residency_shard_reader.reset_read_shape_cache_for_tests()
    yield
    reset_residency_resolver_for_tests()
    residency_shard_reader.reset_read_shape_cache_for_tests()


def _declared_and_staged(tmp_path, *, rows=ROWS, cols=COLS):
    """One declared shard (bytes 1.0) and a staged range of it holding 2.0."""
    model = tmp_path / 'model'
    model.mkdir()
    declared = model / 'model-00001-of-00001.safetensors'
    save_file({TENSOR: torch.ones(rows, cols, dtype=torch.bfloat16)}, str(declared))
    start, end = _header(declared)[TENSOR]
    staged_bytes = torch.full((rows, cols), 2.0, dtype=torch.bfloat16)
    blob = staged_bytes.view(torch.uint8).numpy().tobytes()
    assert len(blob) == end - start
    root = _stage_root(tmp_path)
    staged = _stage_range(root, declared, start, end - start, blob=blob)
    map_path = _write_map(tmp_path, root, [(declared, start, end - start, staged)])
    source = {'tensors': {TENSOR: declared.name}}
    unit = {'source_tensor': TENSOR, 'rows': rows, 'cols': cols}
    return model, source, unit, map_path, end - start


_NEEDS_PINNED = pytest.mark.skipif(not torch.cuda.is_available(),
    reason="pinned staging needs the CUDA backend allocator")


class _CopyCounter:
    """Counts private staging copies by call site, never the H2D launch."""

    def __init__(self, monkeypatch):
        self.staging = 0
        original, counter = torch.Tensor.copy_, self

        # A plain function, so the class binds it as a method: a bound method
        # stored on the class would never receive the tensor it was called on.
        def counting(target, source, *args, **kwargs):
            counter.staging += 1
            return original(target, source, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, 'copy_', counting)


class _PinnedMemory:
    """What the reads asked of the pinned allocator, and whether it was real.

    ``allocations`` counts ``torch.empty(..., pin_memory=True)``, the staged
    reader's own pinned buffer. ``after_read`` counts ``Tensor.pin_memory()``,
    the one copy a read that cannot fill pinned memory still makes. With CUDA
    the requests pass through, so ``is_pinned()`` is the real page-lock. Without
    it the stand-in answers pageable memory and remembers which storages it
    handed out as locked, so ``is_pinned()`` still tells the code under test
    what the allocator would have.
    """

    def __init__(self, monkeypatch):
        self.real = torch.cuda.is_available()
        self.allocations = self.after_read = 0
        self._locked, self._kept = set(), []
        real_empty, real_pin = torch.empty, torch.Tensor.pin_memory
        real_is_pinned = torch.Tensor.is_pinned

        def lock(tensor):
            self._kept.append(tensor)   # a live storage keeps its address unique
            self._locked.add(tensor.untyped_storage().data_ptr())
            return tensor

        def empty(*args, pin_memory=False, **kwargs):
            self.allocations += bool(pin_memory)
            if self.real:
                return real_empty(*args, pin_memory=pin_memory, **kwargs)
            tensor = real_empty(*args, **kwargs)
            return lock(tensor) if pin_memory else tensor

        def pin(tensor):
            self.after_read += 1
            return real_pin(tensor) if self.real else lock(tensor.clone())

        def is_pinned(tensor, *args, **kwargs):
            if self.real:
                return real_is_pinned(tensor, *args, **kwargs)
            return tensor.untyped_storage().data_ptr() in self._locked

        monkeypatch.setattr(torch, 'empty', empty)
        monkeypatch.setattr(torch.Tensor, 'pin_memory', pin)
        monkeypatch.setattr(torch.Tensor, 'is_pinned', is_pinned)


@pytest.fixture
def pinned(monkeypatch):
    return _PinnedMemory(monkeypatch)


def _nonqualified_owner(model):
    """A sealed-roster owner over the declared shard, never original material."""
    import hashlib
    from prismaquant.tessera_calibration_cache import CaptureSourceAuthentication
    declared = model / 'model-00001-of-00001.safetensors'
    digest = hashlib.sha256(declared.read_bytes()).hexdigest()
    owner = CaptureSourceAuthentication(model,
        {'source_files': {declared.name: digest}, 'census_sha256': 'a' * 64}, {},
        manifest_sha256='b' * 64)
    assert not owner.is_qualified_original_material
    return owner


# -- the read: no map, then a staged map --------------------------------------

def test_pin_request_without_map_never_reaches_raw_opener(tmp_path, monkeypatch, pinned):
    """Unmapped path: raw ``safe_open`` takes no pin flag, so the pin follows the read."""
    model, source, unit, _map_path, _nbytes = _declared_and_staged(tmp_path)
    seen = []
    real_open = layer_streaming.safe_open

    def spy(path, *args, **kwargs):
        seen.append(dict(kwargs))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(layer_streaming, 'safe_open', spy)
    weight = source_unit_weight(model, source, unit, pin_memory=True)
    assert torch.equal(weight, torch.ones((ROWS, COLS), dtype=torch.bfloat16))
    assert seen, "expected at least one shard open"
    # The first attempt offers the staged fast path; the open that succeeds
    # on the raw opener carries no pin flag (the first raises TypeError).
    assert seen[0].get('pinned_host') is True
    assert 'pinned_host' not in seen[-1]
    assert (pinned.allocations, pinned.after_read) == (0, 1)
    assert weight.is_pinned()
    assert residency_report() is None


def test_pin_request_with_owner_without_map_pins_after_read(tmp_path, pinned):
    """The owner branch with no map: the flag stops at the retry."""
    model, source, unit, _map_path, _nbytes = _declared_and_staged(tmp_path)
    owner = _nonqualified_owner(model)
    try:
        weight = source_unit_weight(model, source, unit,
            source_authentication=owner, pin_memory=True)
        assert torch.equal(weight, torch.ones((ROWS, COLS), dtype=torch.bfloat16))
        assert (pinned.allocations, pinned.after_read) == (0, 1)
        assert weight.is_pinned()
        assert residency_report() is None
    finally:
        owner.close()


def test_staged_pin_request_fills_the_pinned_buffer_from_the_stage(
        tmp_path, monkeypatch, source_bound_reader_sdk, pinned):
    model, source, unit, map_path, nbytes = _declared_and_staged(tmp_path)
    _bind(monkeypatch, map_path)
    counter = _CopyCounter(monkeypatch)
    weight = source_unit_weight(model, source, unit, pin_memory=True)
    assert weight.dtype == torch.bfloat16
    assert tuple(weight.shape) == (ROWS, COLS)
    assert torch.equal(weight, torch.full((ROWS, COLS), 2.0, dtype=torch.bfloat16))
    # One pinned buffer, filled by the reader: no pin-after-read, no private copy.
    assert (pinned.allocations, pinned.after_read, counter.staging) == (1, 0, 0)
    assert weight.is_pinned()
    report = residency_report()
    assert report['bytes_from_pool'] == 0
    assert report['bytes_from_stage'] == nbytes


def test_staged_pageable_request_never_asks_for_pinned_memory(
        tmp_path, monkeypatch, source_bound_reader_sdk, pinned):
    model, source, unit, map_path, nbytes = _declared_and_staged(tmp_path)
    _bind(monkeypatch, map_path)
    weight = source_unit_weight(model, source, unit)
    assert torch.equal(weight, torch.full((ROWS, COLS), 2.0, dtype=torch.bfloat16))
    assert (pinned.allocations, pinned.after_read) == (0, 0)
    assert not weight.is_pinned()
    assert residency_report()['bytes_from_stage'] == nbytes


def test_staged_pin_request_with_owner_fills_the_pinned_buffer(
        tmp_path, monkeypatch, source_bound_reader_sdk, pinned):
    """The owner branch keeps the staged fast path under a map."""
    model, source, unit, map_path, nbytes = _declared_and_staged(tmp_path)
    _bind(monkeypatch, map_path)
    owner = _nonqualified_owner(model)
    try:
        counter = _CopyCounter(monkeypatch)
        weight = source_unit_weight(model, source, unit,
            source_authentication=owner, pin_memory=True)
        assert torch.equal(weight, torch.full((ROWS, COLS), 2.0, dtype=torch.bfloat16))
        assert (pinned.allocations, pinned.after_read, counter.staging) == (1, 0, 0)
        assert weight.is_pinned()
        report = residency_report()
        assert report['bytes_from_pool'] == 0
        assert report['bytes_from_stage'] == nbytes
    finally:
        owner.close()


def _nfs_stage(monkeypatch, tmp_path):
    """Make the stage root an NFS mount of four streams and 4 KiB reads."""
    _use_table(monkeypatch, _mount_table(tmp_path, [
        ('10.100.99.3:/prewarm', str(_stage_root(tmp_path)), 'nfs4',
         'rw,rsize=4096,nconnect=4')]))


def test_staged_pin_request_on_several_streams_is_the_one_stream_bytes(
        tmp_path, monkeypatch, source_bound_reader_sdk, pinned):
    """The cut pieces land in disjoint windows of the one pinned buffer."""
    model, source, unit, map_path, nbytes = _declared_and_staged(
        tmp_path, rows=256, cols=256)
    _nfs_stage(monkeypatch, tmp_path)
    _bind(monkeypatch, map_path)
    calls, real = [], os.preadv
    monkeypatch.setattr(os, 'preadv', lambda fd, bufs, off: (calls.append(off), real(fd, bufs, off))[1])
    weight = source_unit_weight(model, source, unit, pin_memory=True)
    assert torch.equal(weight, torch.full((256, 256), 2.0, dtype=torch.bfloat16))
    assert len(calls) > 4, "the span was never cut"
    assert (pinned.allocations, pinned.after_read) == (1, 0)
    assert weight.is_pinned()
    assert residency_report()['bytes_from_stage'] == nbytes


def test_failed_chunk_hands_out_the_pool_bytes_and_never_the_partial_buffer(
        tmp_path, monkeypatch, source_bound_reader_sdk, pinned):
    """A pinned buffer a failed read half filled is dropped, not returned."""
    model, source, unit, map_path, nbytes = _declared_and_staged(
        tmp_path, rows=256, cols=256)
    _nfs_stage(monkeypatch, tmp_path)
    _bind(monkeypatch, map_path)
    real, seen = os.preadv, []

    def flaky(fd, bufs, off):
        seen.append(off)
        if len(seen) > 3:
            raise OSError(5, 'Input/output error')
        return real(fd, bufs, off)

    monkeypatch.setattr(os, 'preadv', flaky)
    weight = source_unit_weight(model, source, unit, pin_memory=True)
    # The pool's bytes (1.0), never the stage's (2.0) and never half of each.
    assert torch.equal(weight, torch.ones((256, 256), dtype=torch.bfloat16))
    report = residency_report()
    assert report['fallback_count'] == 1
    assert report['bytes_from_stage'] == 0
    assert report['bytes_from_pool'] == nbytes
    # The staged attempt allocated its buffer and lost it; the pool read took
    # the one pin-after-read copy a read that cannot fill pinned memory makes.
    assert (pinned.allocations, pinned.after_read) == (1, 1)


@pytest.mark.parametrize('pin_memory', [False, True], ids=['pageable', 'pinned'])
def test_a_stage_file_that_changes_during_the_read_falls_back_to_the_pool(
        tmp_path, monkeypatch, source_bound_reader_sdk, pinned, pin_memory):
    """One stat fence guards both buffers: it runs after the read, before use."""
    model, source, unit, map_path, nbytes = _declared_and_staged(tmp_path)
    _bind(monkeypatch, map_path)
    staged = next(iter(json.loads(map_path.read_text())['entries'].values()))['stage_path']
    real = residency_shard_reader._read_span_into

    def read_then_touch(fd, view, offset, shape):
        real(fd, view, offset, shape)
        os.utime(staged, ns=(1, 1))

    monkeypatch.setattr(residency_shard_reader, '_read_span_into', read_then_touch)
    weight = source_unit_weight(model, source, unit, pin_memory=pin_memory)
    # The pool's bytes (1.0), never the stage's (2.0) the changed file held.
    assert torch.equal(weight, torch.ones((ROWS, COLS), dtype=torch.bfloat16))
    report = residency_report()
    assert report['fallback_count'] == 1
    assert 'changed during its content read' in report['fallbacks'][0]['reason']
    assert report['bytes_from_stage'] == 0
    assert report['bytes_from_pool'] == nbytes


# -- the check: adopt a pinned source, or make the one private copy ------------

def _prepare_kwargs(**overrides):
    kwargs = dict(live_shape=(2, 3), live_dtype=torch.bfloat16,
                  model_path='unused', source={}, release_source_pages=False,
                  source_authentication=None)
    kwargs.update(overrides)
    return kwargs

@_NEEDS_PINNED
def test_device_prepare_adopts_pinned_source_without_second_copy(monkeypatch):
    pinned = torch.full((2, 3), 2.0, dtype=torch.bfloat16).pin_memory()
    assert pinned.is_pinned()
    released = []

    def read(name, unit, **kwargs):
        assert kwargs.get('pinned_host') is True
        return pinned, lambda: released.append(name)

    # Setup copies must not count: the spy starts after the source exists.
    monkeypatch.setattr(campaign, '_read_projected_unit', read)
    counter = _CopyCounter(monkeypatch)
    unit = dict(source_tensor='w', rows=2, cols=3)
    check = campaign._prepare_device_projected_check('u', unit, **_prepare_kwargs())
    assert counter.staging == 0
    assert check.pinned is not None and check.pinned.is_pinned()
    assert check.pinned.data_ptr() == pinned.data_ptr()
    assert torch.equal(check.pinned, torch.full((2, 3), 2.0, dtype=torch.bfloat16))
    assert released == ['u']

@_NEEDS_PINNED
def test_device_prepare_keeps_one_copy_for_pageable_source(monkeypatch):
    source = torch.full((2, 3), 3.0, dtype=torch.bfloat16)
    assert not source.is_pinned()
    released = []

    def read(name, unit, **kwargs):
        return source, lambda: released.append(name)

    monkeypatch.setattr(campaign, '_read_projected_unit', read)
    counter = _CopyCounter(monkeypatch)
    unit = dict(source_tensor='w', rows=2, cols=3)
    check = campaign._prepare_device_projected_check('u', unit, **_prepare_kwargs())
    assert counter.staging == 1
    assert check.pinned is not None and check.pinned.is_pinned()
    assert check.pinned.data_ptr() != source.data_ptr()
    assert torch.equal(check.pinned, source)
    assert released == ['u']


def test_device_prepare_reports_shape_mismatch_without_staging(monkeypatch):
    released = []

    def read(name, unit, **kwargs):
        return torch.zeros(4, 5, dtype=torch.bfloat16), lambda: released.append(name)

    monkeypatch.setattr(campaign, '_read_projected_unit', read)
    counter = _CopyCounter(monkeypatch)
    unit = dict(source_tensor='w', rows=2, cols=3)
    check = campaign._prepare_device_projected_check('u', unit, **_prepare_kwargs())
    assert check.differs is True
    assert check.pinned is None
    assert counter.staging == 0
    assert released == ['u']


def test_device_prepare_requests_pinned_read_and_adopts_without_copy(monkeypatch):
    """CPU attribution: the device path asks for pinned bytes and keeps one copy."""
    class _StubWeight:
        dtype = torch.bfloat16
        shape = (2, 3)
        def is_pinned(self):
            return True
        def is_contiguous(self):
            return True
    stub = _StubWeight()
    released = []
    def read(name, unit, **kwargs):
        assert kwargs.get('pinned_host') is True
        return stub, lambda: released.append(name)
    monkeypatch.setattr(campaign, '_read_projected_unit', read)
    counter = _CopyCounter(monkeypatch)
    unit = dict(source_tensor='w', rows=2, cols=3)
    check = campaign._prepare_device_projected_check('u', unit, **_prepare_kwargs())
    assert counter.staging == 0
    assert check.pinned is stub
    assert check.differs is None
    assert released == ['u']
