"""PQ #2039: read staged payload into the private CUDA-preparation owner once.

These CPU checks emulate only page locking, not its performance or CUDA DMA.
The staged-reader check uses real safetensors headers, descriptors and preadv;
the resolver double supplies one already-resolved local range. Authentication
and strict-tier policy are deliberately not mocked into a claim of GPU proof.
"""
import ctypes
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import residency_shard_reader as reader_module
from prismaquant import tessera_campaign as campaign


@pytest.fixture
def private_pin_allocator(monkeypatch):
    """Keep the allocation/copy boundary observable on the CPU-only test arms."""
    owners, copies, read_addresses = [], [], []
    empty, empty_like = torch.empty, torch.empty_like
    copy, preadv = torch.Tensor.copy_, reader_module.os.preadv
    is_pinned = torch.Tensor.is_pinned

    def allocate(fn, *args, **kwargs):
        pinned = kwargs.pop('pin_memory', False)
        tensor = fn(*args, **kwargs)
        if pinned:
            owners.append(tensor)
        return tensor

    def pinned(tensor, *args, **kwargs):
        return any(tensor.untyped_storage().data_ptr() ==
                   owner.untyped_storage().data_ptr() for owner in owners) or is_pinned(
                       tensor, *args, **kwargs)

    def host_copy(target, source, *args, **kwargs):
        copies.append((target.data_ptr(), source.data_ptr()))
        return copy(target, source, *args, **kwargs)

    def pin(tensor, *args, **kwargs):
        if pinned(tensor):
            return tensor
        target = allocate(empty_like, tensor, pin_memory=True)
        target.copy_(tensor)
        return target

    def read(fd, buffers, offset):
        read_addresses.extend(ctypes.addressof(ctypes.c_char.from_buffer(view))
                              for view in buffers if len(view))
        return preadv(fd, buffers, offset)

    monkeypatch.setattr(torch, 'empty', lambda *a, **k: allocate(empty, *a, **k))
    monkeypatch.setattr(torch, 'empty_like', lambda *a, **k: allocate(empty_like, *a, **k))
    monkeypatch.setattr(torch.Tensor, 'is_pinned', pinned)
    monkeypatch.setattr(torch.Tensor, 'pin_memory', pin)
    monkeypatch.setattr(torch.Tensor, 'copy_', host_copy)
    monkeypatch.setattr(reader_module.os, 'preadv', read)
    return SimpleNamespace(owners=owners, copies=copies, read_addresses=read_addresses)


@pytest.mark.parametrize('cancel_after_preparation', [False, True])
def test_preparation_adopts_an_already_private_pin_without_copying(
        monkeypatch, private_pin_allocator, cancel_after_preparation):
    from concurrent.futures import CancelledError

    private = torch.empty((2, 3), dtype=torch.bfloat16, pin_memory=True)
    private.fill_(2)
    retired = []
    monkeypatch.setattr(campaign, '_read_projected_unit',
                        lambda *args, **kwargs: (private, lambda: retired.append('released')))
    calls = []

    def cancelled():
        calls.append(None)
        return cancel_after_preparation and len(calls) == 3

    kwargs = dict(live_shape=(2, 3), live_dtype=torch.bfloat16,
                  model_path='unused', source={}, release_source_pages=True,
                  source_authentication=None, cancelled=cancelled)
    if cancel_after_preparation:
        with pytest.raises(CancelledError):
            campaign._prepare_device_projected_check('u', {'source_tensor': 'w'}, **kwargs)
    else:
        check = campaign._prepare_device_projected_check('u', {'source_tensor': 'w'}, **kwargs)
        assert check.pinned is private, 'preparation replaced an already-private pinned owner'
        assert check.differs is None
    assert retired == ['released']
    assert private_pin_allocator.copies == [], 'preparation made a redundant host copy'
    assert len(private_pin_allocator.owners) == 1


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_staged_projected_payload_lands_in_its_final_private_buffer(
        tmp_path, monkeypatch, private_pin_allocator, dtype):
    # A whole-file stage entry exercises the real header/offset calculation.
    # Resolver validation/leases have their own tests; this locator double
    # intentionally provides no authentication or retirement exemption.
    expected = torch.tensor([[0., -0., 2.], [float('nan'), -3., 4.]], dtype=dtype)
    declared = tmp_path / 'source.safetensors'
    staged = tmp_path / 'staged.safetensors'
    save_file({'w': expected}, str(declared))
    staged.write_bytes(declared.read_bytes())
    entry = {'stage_path': str(staged), 'offset': 0, 'bytes': staged.stat().st_size}
    served, fallbacks = [], []

    class ResolvedRange:
        def stages(self, path):
            return str(path) == str(declared)

        def staged_range_outcome(self, path, start, end, **kwargs):
            assert 0 <= start < end <= entry['bytes']
            return entry, None

        def record_stage_range_read(self, path, count):
            served.append(count)

        def record_pool_read(self, *args):
            pytest.fail('projected payload fell back to the pool')

        def record_fallback(self, *args):
            fallbacks.append(args)

    resolver = ResolvedRange()
    monkeypatch.setattr(reader_module, 'residency_resolver', lambda: resolver)
    # Scheduling/mount qualification is not under test: one real local preadv
    # is sufficient to identify whether the destination is the final pin.
    monkeypatch.setattr(reader_module, '_read_shape', lambda path: None)
    source = {'tensors': {'w': declared.name}}
    unit = {'source_tensor': 'w', 'rows': 2, 'cols': 3}
    # A smaller live shape/dtype must still refuse before allocating a pin:
    # the preparation byte credit is live-sized, not source-sized. Keep this
    # in the same transaction as the new matching-read optimization.
    for live_shape, live_dtype in [((1, 3), dtype), ((2, 3), torch.float16)]:
        mismatch = campaign._prepare_device_projected_check(
            'u', unit, live_shape=live_shape, live_dtype=live_dtype,
            model_path=tmp_path, source=source, release_source_pages=False,
            source_authentication=None)
        assert mismatch.differs is True and mismatch.pinned is None
        assert mismatch.description == (
            f'u (live {live_shape} {live_dtype} vs source w (2, 3) {dtype})')
        assert private_pin_allocator.owners == []
        assert private_pin_allocator.copies == []
    served.clear()
    private_pin_allocator.read_addresses.clear()
    check = campaign._prepare_device_projected_check(
        'u', unit, live_shape=(2, 3), live_dtype=dtype,
        model_path=tmp_path, source=source, release_source_pages=False,
        source_authentication=None)

    assert check.differs is None
    assert check.pinned.is_pinned() and check.pinned.is_contiguous()
    # Byte comparison deliberately preserves NaNs and signed zero semantics.
    assert torch.equal(check.pinned.view(torch.uint8), expected.view(torch.uint8))
    assert served == [expected.nbytes] and not fallbacks
    assert len(private_pin_allocator.owners) == 1
    assert private_pin_allocator.copies == [], (
        'staged bytes were copied again instead of being read into the private pin')
    assert check.pinned.data_ptr() in private_pin_allocator.read_addresses
    # The returned storage must remain private after the reader has closed.
    staged.write_bytes(b'\0' * entry['bytes'])
    assert torch.equal(check.pinned.view(torch.uint8), expected.view(torch.uint8))
