"""Bounded non-stream PWC loads use the existing sealed/mmap decoder."""
import hashlib

import pytest
import torch

from prismaquant.io_engine import SealedBuffer
from test_pwc_file_load_receipts import make_cache


@pytest.mark.parametrize('load_style', ['receipt-prefetch', 'digest-get'])
def test_bounded_pwc_decode_is_sealed_and_mapping_outlives_buffer(
    tmp_path, monkeypatch, load_style
):
    cache, paths, tensors = make_cache(tmp_path)
    key, path = next(iter(paths.items()))
    original_bytes = path.read_bytes()
    expected_sha = hashlib.sha256(original_bytes).hexdigest()
    decoded = []
    original_decode = cache._decode_file_tensor

    def observe_decode(window_entry, raw, receipt, staged):
        decoded.append(raw)
        return original_decode(window_entry, raw, receipt, staged)

    monkeypatch.setattr(cache, '_decode_file_tensor', observe_decode)
    if load_style == 'receipt-prefetch':
        cache.enable_file_load_receipts(max_file_bytes=len(original_bytes))
        assert cache.prefetch([key], max_workers=1) == 1
    else:
        cache.require_file_load_sha256({key: expected_sha},
                                      max_file_bytes=len(original_bytes))
    tensor = cache.get(*key)
    assert isinstance(tensor, torch.Tensor)
    assert torch.equal(tensor, tensors[key])
    assert len(decoded) == 1
    assert isinstance(decoded[0], SealedBuffer), 'bounded PWC load copied verified bytes'
    with pytest.raises(RuntimeError, match='sealed io buffer is closed'):
        _ = decoded[0].path
    receipt = cache.file_load_receipt(key, tensor)
    assert receipt['sha256'] == expected_sha
    assert receipt['bytes'] == len(original_bytes)
    # The mapping must be private, and the existing mutation guard remains live.
    tensor[0, 0] += 1
    assert path.read_bytes() == original_bytes
    with pytest.raises(RuntimeError, match='receipt|changed'):
        cache.file_load_receipt(key, tensor)
