"""PQ #2039: unmapped pinned reads against the real CUDA allocator.

No residency map is bound, so no PrismaBuild SDK is needed. These tests run
only where torch ships a CUDA pinned allocator. They cover the fallback the
review of PR #2591 demanded: the raw opener never sees ``pinned_host`` and
the tensor is pinned after the read, with equal bytes.
"""
import pytest
import torch

from prismaquant.tessera_expert_projection import source_unit_weight
from tests.test_projected_staging_copy_2039 import (
    _declared_and_staged,
    _nonqualified_owner,
    COLS,
    ROWS,
)

_NEEDS_CUDA = pytest.mark.skipif(not torch.cuda.is_available(),
    reason="unmapped pin-after-read needs the CUDA pinned allocator")


@_NEEDS_CUDA
def test_unmapped_pin_read_is_pinned_and_equal(tmp_path):
    model, source, unit, _map_path, _nbytes = _declared_and_staged(tmp_path)
    weight = source_unit_weight(model, source, unit, pin_memory=True)
    assert weight.is_pinned()
    assert weight.is_contiguous() and weight.device.type == 'cpu'
    assert tuple(weight.shape) == (ROWS, COLS)
    assert torch.equal(weight, torch.ones((ROWS, COLS), dtype=torch.bfloat16))


@_NEEDS_CUDA
def test_owner_unmapped_pin_read_is_pinned_and_equal(tmp_path):
    model, source, unit, _map_path, _nbytes = _declared_and_staged(tmp_path)
    owner = _nonqualified_owner(model)
    try:
        weight = source_unit_weight(model, source, unit,
            source_authentication=owner, pin_memory=True)
        assert weight.is_pinned()
        assert tuple(weight.shape) == (ROWS, COLS)
        assert torch.equal(weight, torch.ones((ROWS, COLS), dtype=torch.bfloat16))
    finally:
        owner.close()
