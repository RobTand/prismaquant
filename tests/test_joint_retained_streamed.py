"""Retained execution joins the real streamed probe and existing source owners."""
from pathlib import Path
import hashlib
import threading
from types import SimpleNamespace

import pytest
import torch

import prismaquant.aura_cost as aura
from prismaquant import io_engine
from prismaquant.cost_streaming import StreamedBoundaryArtifacts
from prismaquant.joint_retained_window_plan import RetainedWindowBudget, EXECUTION_SCHEMA
from prismaquant.streaming_model import StreamingContext
from prismaquant.production_weight_cache import _cb_cache_tensor_identity
from test_joint_operator_windows import policy as operator_policy
from test_layer_major_boundary_capture import fixture, policy as boundary_policy, draw
from test_streamed_cost_checkpoints import _model_identity


def _case(tmp_path, monkeypatch, retained, *, checkpoint=None, resume=False, proof_change=None,
          skeleton=None, budget=None, observed=None):
    monkeypatch.setattr(aura, '_checkpoint_git_commit', lambda: '1' * 40)
    model, context, runner, cache = fixture()
    if observed is not None:
        observed['context'] = context
    if skeleton is not None:
        skeleton(model, context)
    context.settle_prefetched_layers = lambda layers, *, retry_availability=False: None
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
    b = budget if budget is not None else RetainedWindowBudget(
        50 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20,
        1 << 20, 1 << 20, 1 << 20, 1 << 20, 1024,
        2048, 4 << 20, 4)
    execution = {'schema': EXECUTION_SCHEMA, 'budget': b.as_dict(),
                 'source_reserve_bytes': 1 << 20, 'source_loading_reserve_bytes': 2 << 20}
    paths = []
    load = io_engine.load_file
    def recorded(path, limit, **kwargs):
        # Every render read goes through the IO engine (PQ #1294), on
        # whichever thread reads it.
        paths.append(str(path))
        return load(path, limit, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(io_engine, 'load_file', recorded)
        result = aura.compute_aura_cost_streamed(runner, draw(),
            ['FP8_DYNAMIC', 'NVFP4A16', 'BF16'], n_probes=4, probe_microbatch=1,
            min_free_gib=0, production_cache=cache, joint_activation=True,
            prepared_render_identities=proofs,
            model_identity=_model_identity('joint-source'), operator_windows=operator_policy(),
            boundary_storage=boundary_policy(tmp_path / 'boundaries'), checkpoint_dir=checkpoint, resume=resume,
            **({'retained_operator_windows': execution} if retained else {}))
    return result, paths, context


def _meta_skeleton_until_install(model, context, skeleton_dtype):
    """Model the streamed source: every decoder Linear is a meta parameter until install.

    ``build_streaming_skeleton`` instantiates the model on meta with no dtype,
    so the skeleton's parameters carry torch's default dtype, not the
    checkpoint's; ``_fast_install`` then replaces each meta slot with a fresh
    Parameter of the loaded tensor's dtype. The property modelled here is only
    that the skeleton dtype differs from the installed dtype (production: a
    float32 skeleton over a bf16 checkpoint; this fp32 fixture inverts the pair).
    """
    installed = {}
    for index, layer in enumerate(model.model.layers):
        installed[index] = layer.proj.weight
        layer.proj.weight = torch.nn.Parameter(
            torch.empty(installed[index].shape, device='meta', dtype=skeleton_dtype),
            requires_grad=False)
    install = context.install
    def install_from_checkpoint(layer, *, require_prefetched=False, prefetch_following=True):
        model.model.layers[int(layer)].proj.weight = installed[int(layer)]
        return install(layer, require_prefetched=require_prefetched,
                       prefetch_following=prefetch_following)
    context.install = install_from_checkpoint


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
                                         # The IO engine holds each read to its bound digest
                                         # before decoding it (PQ #1294).
                                         ('file', 'checksum changed')])
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


def test_prepared_proof_is_compared_with_the_installed_source_not_the_meta_skeleton(tmp_path, monkeypatch):
    """The prepared render identity was verified against the INSTALLED source (prepare's
    ``verify_anchor_render``). Before install the live slot is the meta skeleton, whose dtype
    is not the checkpoint's, so the dtype/byte comparison belongs to install time, per layer,
    before that layer's first render is consumed."""
    expected, _, _ = _case(tmp_path / 'resident', monkeypatch, True, checkpoint=tmp_path / 'resident-ckpt')
    actual, _, context = _case(tmp_path / 'skeleton', monkeypatch, True, checkpoint=tmp_path / 'skeleton-ckpt',
                               skeleton=lambda model, context: _meta_skeleton_until_install(model, context, torch.bfloat16))
    assert context.install_calls > 0
    assert actual['costs'] == expected['costs']


def test_installed_source_that_differs_from_the_prepared_proof_is_refused_before_any_render(tmp_path, monkeypatch):
    """A proof the skeleton agrees with but the installed tensor does not is refused at install,
    before that layer's first render is consumed (the up-front check cannot see it)."""
    def claim_bf16(proofs, files):
        for value in proofs.values():
            value['dtype'] = str(torch.bfloat16)
            value['logical_bytes'] = value['logical_bytes'] // 2
    with pytest.raises(RuntimeError, match='installed source'):
        _case(tmp_path, monkeypatch, True, checkpoint=tmp_path / 'ckpt', proof_change=claim_bf16,
              skeleton=lambda model, context: _meta_skeleton_until_install(model, context, torch.bfloat16))


def test_an_inadmissible_retained_budget_is_refused_before_any_capture(tmp_path, monkeypatch):
    """Issue #743: this refusal reads declared bytes only, so it owes t=0.

    A ``candidate_delta_bytes`` one byte under a single target's fp32 delta is
    the GLM-5.3-Flash refusal in miniature. Before the preflight it cost a full
    boundary capture to discover; the capture is what this asserts never ran.
    """
    boundaries = []
    write = StreamedBoundaryArtifacts.write
    def recorded(self, *args, **kwargs):
        boundaries.append(kwargs.get('boundary_index'))
        return write(self, *args, **kwargs)
    monkeypatch.setattr(StreamedBoundaryArtifacts, 'write', recorded)
    observed = {}
    inadmissible = RetainedWindowBudget(50 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20,
                                        1 << 20, 1 << 20, 1 << 20, 1 << 20, 1023,
                                        2048, 4 << 20, 4)
    with pytest.raises(RuntimeError, match='indivisible target does not fit'):
        _case(tmp_path, monkeypatch, True, budget=inadmissible, observed=observed,
              checkpoint=tmp_path / 'checkpoints')
    assert boundaries == []
    assert observed['context'].install_calls == 0
    assert not list((tmp_path / 'boundaries').rglob('*.pt'))


def test_an_admissible_retained_budget_still_captures_and_measures(tmp_path, monkeypatch):
    """The preflight admits what the per-layer planner admits, and nothing else."""
    boundaries = []
    write = StreamedBoundaryArtifacts.write
    def recorded(self, *args, **kwargs):
        boundaries.append(kwargs.get('boundary_index'))
        return write(self, *args, **kwargs)
    monkeypatch.setattr(StreamedBoundaryArtifacts, 'write', recorded)
    result, _, context = _case(tmp_path, monkeypatch, True,
                               checkpoint=tmp_path / 'checkpoints')
    assert boundaries and context.install_calls > 0 and result['costs']


def test_a_missing_pwc_candidate_is_refused_before_any_capture_without_a_retained_budget(
        tmp_path, monkeypatch):
    """Issue #743: the hoist covers the windowed path's coverage check too.

    ``resident_candidates`` used to be the first thing to notice a candidate
    the production cache never rendered, one layer into the reverse replay.
    The roster, the menu and the cache index are all declared, so the preflight
    answers it for every layer before the boundary capture starts -- and this
    asserts the capture is what does not run.
    """
    boundaries = []
    write = StreamedBoundaryArtifacts.write
    def recorded(self, *args, **kwargs):
        boundaries.append(kwargs.get('boundary_index'))
        return write(self, *args, **kwargs)
    monkeypatch.setattr(StreamedBoundaryArtifacts, 'write', recorded)
    dropped = []
    def drop_one_candidate(proofs, files):
        dropped.append(sorted(files)[0])
        files.pop(dropped[0])
        proofs.pop(dropped[0])
    observed = {}
    with pytest.raises(RuntimeError, match='PWC candidate entry missing'):
        _case(tmp_path, monkeypatch, False, proof_change=drop_one_candidate, observed=observed)
    assert boundaries == []
    assert observed['context'].install_calls == 0
    assert not list((tmp_path / 'boundaries').rglob('*.pt'))
