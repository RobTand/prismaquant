"""The source-projection check reads its tensor through the residency map (PQ #1529).

``tessera_expert_projection.source_unit_weight`` is the read behind the Tessera
campaign's byte-for-byte source-projection check. It opened the declared shard
with a raw ``safe_open``, so under a PrismaBuild residency map every other
source read came off the stage while this one still read the pool.

The driver is mutated, not the fixture: the staged copy holds different tensor
bytes from the declared file, so the returned tensor names the file it came
from. A reader that silently fell back to the pool would return the declared
bytes and fail here.
"""
import pytest
import torch
from safetensors.torch import save_file

from prismaquant.residency_map import (
    ENV_VAR, reset_residency_resolver_for_tests, residency_report,
)
from prismaquant.tessera_expert_projection import source_unit_weight
from tests.test_residency_shard_reader import (
    _bind, _header, _stage_range, _stage_root, _write_map,
)

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


def test_projection_check_reads_the_staged_copy_under_a_map(tmp_path, monkeypatch):
    model, source, unit, map_path, nbytes = _declared_and_staged(tmp_path)
    _bind(monkeypatch, map_path)
    weight = source_unit_weight(model, source, unit)
    assert torch.equal(weight, torch.full((ROWS, COLS), 2.0, dtype=torch.bfloat16))
    report = residency_report()
    assert report is not None
    assert report['bytes_from_pool'] == 0
    assert report['bytes_from_stage'] == nbytes
    assert report['range_hits'] == 1


def test_projection_check_reads_the_declared_file_with_no_map(tmp_path):
    model, source, unit, _map_path, _nbytes = _declared_and_staged(tmp_path)
    weight = source_unit_weight(model, source, unit)
    assert torch.equal(weight, torch.ones(ROWS, COLS, dtype=torch.bfloat16))
    assert residency_report() is None
