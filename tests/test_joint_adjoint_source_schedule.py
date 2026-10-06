"""Compare complete real source bytes with independently owned loader schedules."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import joint_cost_stage_a as stage_a
from prismaquant.layer_streaming import _get_layer_list, _head_prefixes, _resolve_base_prefix
from prismaquant.model_profiles import detect_profile
from prismaquant.source_read_plan import read_safetensors_header, tensor_span
from test_streamed_metadata_staged_reads import _activate, _deny_pool_opens
from test_strict_reader_tier_enforcement import _forget_state  # noqa: F401

pytestmark = pytest.mark.own_process


def _source(tmp_path, *, kind='llama', indexed=True):
    root = torch.nn.Module()
    root.config = SimpleNamespace(model_type=kind, architectures=[])
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for layer in root.model.layers:
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(8, 8, bias=False, device='meta')
    root.model.embed_tokens = torch.nn.Embedding(8, 8, device='meta')
    root.model.norm = torch.nn.LayerNorm(8, device='meta')
    root.lm_head = torch.nn.Linear(8, 8, bias=False, device='meta')
    if kind == 'lfm2_moe':
        root.model.embedding_norm = torch.nn.LayerNorm(8, device='meta')
        root.model.pos_emb = torch.nn.Linear(8, 8, bias=False, device='meta')
    elif kind == 'deepseek_v4':
        root.model.hc_head = torch.nn.Module()
        for name in ('hc_base', 'hc_fn', 'hc_scale'):
            root.model.hc_head.register_parameter(name,
                torch.nn.Parameter(torch.empty(8, device='meta')))
    source = tmp_path / 'source'
    source.mkdir()
    config_path = source / 'config.json'
    config_path.write_text(json.dumps({'model_type': kind, 'architectures': []}))
    profile = detect_profile(str(source))
    head_checkpoint_names = ({profile.checkpoint_to_live_name(name): name
        for name in ('hc_head_base', 'hc_head_fn', 'hc_head_scale')}
        if kind == 'deepseek_v4' else {})
    def checkpoint_name(live):
        return head_checkpoint_names.get(live, profile.source_tensor_name(live))
    checkpoint = {checkpoint_name(name): torch.zeros(
        tuple(value.shape), dtype=torch.bfloat16)
        for name, value in root.state_dict().items()}
    shard = source / 'model.safetensors'
    save_file(checkpoint, str(shard))
    if indexed:
        (source / 'model.safetensors.index.json').write_text(json.dumps({
            'weight_map': {name: shard.name for name in checkpoint}}))
    entries = [{'path': str(shard), 'offset': 0, 'bytes': shard.stat().st_size,
                'sha256': hashlib.sha256(shard.read_bytes()).hexdigest()}]
    header, base, size = read_safetensors_header(str(shard))
    extras = profile.head_resident_extra_prefixes(root)
    extra_entries = []
    head_entries = []
    base_model, layers = _get_layer_list(root)
    base_prefix = _resolve_base_prefix(root, base_model)
    prefixes = _head_prefixes(root, base_prefix)
    for live_name in root.state_dict():
        if not live_name.startswith(tuple(prefixes)):
            continue
        path, start, stop = tensor_span(header, base, size, str(shard),
                                        checkpoint_name(live_name))
        entries.append({'path': path, 'offset': start, 'bytes': stop-start, 'sha256': None})
        head_entries.append(len(entries)-1)
        if live_name.startswith(tuple(extras)):
            extra_entries.append(len(entries)-1)
    metadata_entries = []
    for path in (config_path, source / 'model.safetensors.index.json'):
        if path.exists():
            entries.append({'path': str(path), 'offset': 0, 'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            metadata_entries.append(len(entries)-1)
    phases = [{'name': 'head', 'entry_indices': head_entries+metadata_entries}]
    phases += [{'name': f'{walk}-{layer:03d}', 'entry_indices': [0]}
               for walk in ('forward', 'chain') for layer in range(len(layers))]
    return {'model': str(source)}, root, {'entries': entries, 'read_plan': {'phases': phases}}, extra_entries


@pytest.mark.parametrize('kind', ['lfm2_moe', 'deepseek_v4'])
def test_complete_supported_head_extra_profile_is_not_refused(tmp_path, kind):
    config, root, manifest, extras = _source(tmp_path, kind=kind)
    assert extras, 'the real profile must select architecture-specific head tensors'
    report = stage_a.audit_stage_a_source_spans(config, manifest, source_model=root)
    assert report['uncovered_spans'] == 0


@pytest.mark.parametrize('kind', ['lfm2_moe', 'deepseek_v4'])
def test_head_extra_tensor_is_never_skipped(tmp_path, kind):
    config, root, manifest, extras = _source(tmp_path, kind=kind)
    assert extras
    manifest['read_plan']['phases'][0]['entry_indices'].remove(extras[0])
    with pytest.raises(stage_a.AdjointIdentityRefused, match='head'):
        stage_a.audit_stage_a_source_spans(config, manifest, source_model=root)


def test_complete_unindexed_source_uses_the_loaders_real_header_keys(tmp_path, monkeypatch):
    config, root, manifest, _extras = _source(tmp_path, indexed=False)
    _activate(tmp_path, monkeypatch, manifest)
    _deny_pool_opens(monkeypatch, tmp_path / 'source')
    report = stage_a.audit_stage_a_source_spans(config, manifest, source_model=root)
    assert report['uncovered_spans'] == 0
    assert report['source_phases_checked'] == 5


@pytest.mark.parametrize('missing', ['head', 'forward-001', 'chain-001', 'all-forward'])
def test_the_actual_loader_schedule_catches_an_entire_omitted_phase(tmp_path, missing):
    config, root, manifest, _extras = _source(tmp_path)
    manifest['read_plan']['phases'] = [phase for phase in manifest['read_plan']['phases']
        if phase['name'] != missing and not (
            missing == 'all-forward' and phase['name'].startswith('forward-'))]
    with pytest.raises(stage_a.AdjointIdentityRefused, match='missing.*phase'):
        stage_a.audit_stage_a_source_spans(config, manifest, source_model=root)


@pytest.mark.parametrize('with_recovery', [False, True])
def test_resume_variant_uses_genuine_checkpoint_marker_and_never_recaptures(
        tmp_path, monkeypatch, with_recovery):
    from test_stage_a_chain_resume import _at, _interrupted, _resume
    from prismaquant import stage_a_chain_resume as resume_owner
    root = tmp_path / 'run'
    _interrupted(root, monkeypatch, interrupt=_at(1, 0, 4))
    resume = _resume(root, resume_from=2)
    recovery = None
    if with_recovery:
        # Isolate the second finding from the first; the real checkpoint
        # reader still verifies every byte and session. No check is mocked.
        reader = resume_owner._sealed_checkpoints
        def correct_marker(space, marker, boundaries):
            return reader(space, {**marker, 'kind': 'adjoint_checkpoint'}, boundaries)
        monkeypatch.setattr(resume_owner, '_sealed_checkpoints', correct_marker)
        from test_joint_forward_resume import identity_document
        capsule, _identity = identity_document()
        capsule['groups'] = []
        path = tmp_path / 'recovery.json'
        path.write_text(json.dumps(capsule))
        recovery = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    phases = stage_a._preflight_job_source_phases(5, output_root=root,
        chain_resume=resume, forward_recovery=recovery)
    assert phases == {'head': None, 'chain-001': 1, 'chain-000': 0}
