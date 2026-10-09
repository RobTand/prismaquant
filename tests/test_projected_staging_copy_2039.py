"""PQ #2039: the device check stages source bytes into pinned memory once.

Call-site attribution for the remaining host-copy cost: the legacy device
preparation reads a pageable source tensor and performs one private
``pinned.copy_(weight)`` per unit. The pinned-direct path fills a pinned
buffer at the staged reader and the check adopts it with no second copy.
Both paths preserve bytes, dtype, shape, first-mismatch order, release-once
and cancellation behavior.
"""
import pytest
import torch
from safetensors.torch import save_file

from prismaquant import tessera_campaign as campaign
from prismaquant.residency_map import (
    ENV_VAR, reset_residency_resolver_for_tests, residency_report,
)
from prismaquant.tessera_expert_projection import source_unit_weight
from tests.test_residency_shard_reader import (
    _bind, _header, _stage_range, _stage_root, _write_map,
)

#: The resolver validates maps with PrismaBuild's own validator, through the
#: client SDK (PB #1254); bind the reviewed installed one for every test.
pytestmark = pytest.mark.usefixtures("installed_client_sdk")

TENSOR = 'model.layers.0.mlp.experts.0.gate_proj.weight'
ROWS, COLS = 16, 32


@pytest.fixture(autouse=True)
def _forget_resolver(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    reset_residency_resolver_for_tests()
    yield
    reset_residency_resolver_for_tests()


def _declared_and_staged(tmp_path):
    """One declared shard (bytes 1.0) and a staged range of it holding 2.0."""
    model = tmp_path / 'model'
    model.mkdir()
    declared = model / 'model-00001-of-00001.safetensors'
    save_file({TENSOR: torch.ones(ROWS, COLS, dtype=torch.bfloat16)}, str(declared))
    start, end = _header(declared)[TENSOR]
    staged_bytes = torch.full((ROWS, COLS), 2.0, dtype=torch.bfloat16)
    blob = staged_bytes.view(torch.uint8).numpy().tobytes()
    assert len(blob) == end - start
    root = _stage_root(tmp_path)
    staged = _stage_range(root, declared, start, end - start, blob=blob)
    map_path = _write_map(tmp_path, root, [(declared, start, end - start, staged)])
    source = {'tensors': {TENSOR: declared.name}}
    unit = {'source_tensor': TENSOR, 'rows': ROWS, 'cols': COLS}
    return model, source, unit, map_path, end - start

_NEEDS_PINNED = pytest.mark.skipif(not torch.cuda.is_available(),
    reason="pinned staging needs the CUDA backend allocator")


class _CopyCounter:
    """Counts private staging copies by call site, never the H2D launch."""

    def __init__(self, monkeypatch):
        self.staging = 0
        self._original = torch.Tensor.copy_
        monkeypatch.setattr(torch.Tensor, 'copy_', self._count)

    def _count(self, target, source, *args, **kwargs):
        self.staging += 1
        return self._original(target, source, *args, **kwargs)


@_NEEDS_PINNED
def test_pin_request_reads_staged_bytes_without_private_copy(
        tmp_path, monkeypatch):
    model, source, unit, map_path, nbytes = _declared_and_staged(tmp_path)
    _bind(monkeypatch, map_path)
    counter = _CopyCounter(monkeypatch)
    weight = source_unit_weight(model, source, unit, pin_memory=True)
    assert weight.is_pinned()
    assert weight.dtype == torch.bfloat16
    assert tuple(weight.shape) == (ROWS, COLS)
    assert torch.equal(weight, torch.full((ROWS, COLS), 2.0, dtype=torch.bfloat16))
    assert counter.staging == 0
    report = residency_report()
    assert report is not None
    assert report['bytes_from_pool'] == 0
    assert report['bytes_from_stage'] == nbytes

@_NEEDS_PINNED
def test_pin_request_pins_pool_bytes_with_equal_content(tmp_path, monkeypatch):
    model, source, unit, _map_path, _nbytes = _declared_and_staged(tmp_path)
    weight = source_unit_weight(model, source, unit, pin_memory=True)
    assert weight.is_pinned()
    assert weight.is_contiguous() and weight.device.type == 'cpu'
    assert torch.equal(weight, torch.ones((ROWS, COLS), dtype=torch.bfloat16))
    assert residency_report() is None


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
