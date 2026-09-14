"""Retained execution joins the real streamed probe and existing source owners."""
from pathlib import Path
import hashlib
import threading
from types import SimpleNamespace

import pytest
import torch

import prismaquant.aura_cost as aura
from prismaquant.cost_streaming import StreamedBoundaryArtifacts
from prismaquant.joint_retained_window_plan import RetainedWindowBudget, EXECUTION_SCHEMA
from prismaquant.streaming_model import StreamingContext
from prismaquant.production_weight_cache import _cb_cache_tensor_identity
from test_joint_operator_windows import policy as operator_policy
from test_layer_major_boundary_capture import fixture, policy as boundary_policy, draw
from test_streamed_cost_checkpoints import _model_identity


def _case(tmp_path, monkeypatch, retained, *, checkpoint=None, resume=False, proof_change=None):
    monkeypatch.setattr(aura, '_checkpoint_git_commit', lambda: '1' * 40)
    model, context, runner, cache = fixture()
    context.settle_prefetched_layers = lambda layers: None
    context.source_residency_snapshot = lambda layers, include_head=False: {
        'owners': [], 'unique_storage_bytes': sum(p.numel() * p.element_size() for p in model.parameters())}
    files, proofs, file_shas = {}, {}, {}
    assets = tmp_path / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    for index, (key, tensor) in enumerate(cache.weights.items()):
        path = assets / f'{index}.pt'
        if not path.exists():
            torch.save(tensor, path)
        files[key] = str(path)
        proofs[key] = _cb_cache_tensor_identity(tensor)
        file_shas[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    cache.weights = files
    cache.enable_lru(1 << 20)
    cache.metadata = {}
    cache.metadata["verified_cells"] = {key: {"rendered_weight": value,
        "render_file_sha256": file_shas[key]} for key, value in proofs.items()}
    cache.require_file_load_sha256(file_shas, max_file_bytes=1 << 20)
    if proof_change is not None:
        proof_change(proofs, files)
    b = RetainedWindowBudget(50 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20,
                            1 << 20, 1 << 20, 1 << 20, 1 << 20, 1024,
                            2048, 4 << 20, 4)
    execution = {'schema': EXECUTION_SCHEMA, 'budget': b.as_dict(),
                 'source_reserve_bytes': 1 << 20, 'source_loading_reserve_bytes': 2 << 20}
    paths = []
    load = cache._load_file_tensor
    def recorded(path):
        paths.append(path)
        return load(path)
    cache._load_file_tensor = recorded
    result = aura.compute_aura_cost_streamed(runner, draw(),
        ['FP8_DYNAMIC', 'NVFP4A16', 'BF16'], n_probes=4, probe_microbatch=1,
        min_free_gib=0, production_cache=cache, joint_activation=True,
        prepared_render_identities=proofs,
        model_identity=_model_identity('joint-source'), operator_windows=operator_policy(),
        boundary_storage=boundary_policy(tmp_path / 'boundaries'), checkpoint_dir=checkpoint, resume=resume,
        **({'retained_operator_windows': execution} if retained else {}))
    return result, paths, context


def test_full_streamed_retained_path_matches_each_probe_and_reads_once(tmp_path, monkeypatch):
    expected, original_reads, _ = _case(tmp_path / 'old', monkeypatch, False)
    actual, new_reads, context = _case(tmp_path / 'new', monkeypatch, True,
                                     checkpoint=tmp_path / 'checkpoints')
    assert len(original_reads) == 4 * len(new_reads)
    assert len(new_reads) == len(set(new_reads))
    assert not context.active
    for name, rows in expected['costs'].items():
        for fmt, row in rows.items():
            assert actual['costs'][name][fmt]['signed_components_per_probe'] == row['signed_components_per_probe']
    assert len(list((tmp_path / 'checkpoints' / 'units').glob('*.pkl'))) == len(actual['stats'])


def test_streamed_retained_resume_preserves_completed_costs(tmp_path, monkeypatch):
    first, _, _ = _case(tmp_path / 'same', monkeypatch, True, checkpoint=tmp_path / 'checkpoint')
    resumed, reads, context = _case(tmp_path / 'same', monkeypatch, True,
                                   checkpoint=tmp_path / 'checkpoint', resume=True)
    assert first['costs'] == resumed['costs']
    assert not reads
    assert context.install_calls == 0


def test_existing_source_owner_counts_head_and_aliases_once():
    model, _, _, _ = fixture()
    context = StreamingContext.__new__(StreamingContext)
    context.model = model
    context.layers = model.model.layers
    context.num_layers = len(context.layers)
    context.layers_prefix = 'model.layers.'
    context.layer_cache = SimpleNamespace(_cache={})
    context._inflight = {}
    context._inflight_lock = threading.Lock()
    body = context.source_residency_snapshot(range(context.num_layers))
    full = context.source_residency_snapshot(range(context.num_layers), include_head=True)
    expected = {(str(p.device), p.untyped_storage().data_ptr()): p.untyped_storage().nbytes()
                for p in model.parameters()}
    assert full['unique_storage_bytes'] == sum(expected.values())
    assert full['unique_storage_bytes'] > body['unique_storage_bytes']
    assert sum(row['owner'] == 'always_resident_head' for row in full['owners']) == 1
    with pytest.raises(ValueError, match='boolean'):
        context.source_residency_snapshot([], include_head=1)


def test_partial_streamed_resume_reads_only_pending_renders(tmp_path, monkeypatch):
    first, all_reads, _ = _case(tmp_path / 'same', monkeypatch, True,
                                checkpoint=tmp_path / 'checkpoint')
    pending_name = sorted(first['stats'])[0]
    aura._aura_unit_checkpoint_path(tmp_path / 'checkpoint', pending_name).unlink()
    resumed, reads, context = _case(tmp_path / 'same', monkeypatch, True,
                                   checkpoint=tmp_path / 'checkpoint', resume=True)
    assert resumed['costs'] == first['costs']
    assert len(reads) * 2 == len(all_reads)
    assert len(reads) == len(set(reads))
    assert context.install_calls > 0  # completed upper layer still propagates cotangents


@pytest.mark.parametrize('mutation,match', [('shape', 'tensor proof'), ('hash', 'actual render'),
                                         ('file', 'prepared PWC render checksum changed')])
def test_prepared_identity_reuse_refuses_changed_proof_or_file(tmp_path, monkeypatch, mutation, match):
    def change(proofs, files):
        key = next(iter(proofs))
        if mutation == 'shape':
            proofs[key]['shape'] = [1, 1]
        elif mutation == 'hash':
            proofs[key]['content_sha256'] = '0' * 64
        else:
            path = Path(files[key])
            path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(RuntimeError, match=match):
        _case(tmp_path, monkeypatch, True, checkpoint=tmp_path / 'checkpoint',
              proof_change=change)
