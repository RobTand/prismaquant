"""Selected reuse keeps canonical identity while authenticating one buffer."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_tessera_calibration_cache import capture
from test_verified_capture_load import policy


def test_selected_cli_accepts_explicit_load_policy_before_authentication(monkeypatch, tmp_path):
    from test_tessera_campaign_resume import _main_fixture
    from prismaquant import tessera_calibration_cache as cc
    from prismaquant.autoscale import BOUNDED_CAPTURE_ENV
    campaign, _, argv, _, _ = _main_fixture(monkeypatch, tmp_path)
    argv[argv.index('--hessian')+1] = 'require'
    for key, value in BOUNDED_CAPTURE_ENV.items():
        monkeypatch.setenv(key, value)
    def reached(*args, **kwargs):
        raise RuntimeError('actual complete-capture authentication reached')
    monkeypatch.setattr(cc, 'authenticate_selected_capture_source', reached)
    with pytest.raises(RuntimeError, match='complete-capture authentication reached'):
        campaign.main([*argv, '--streaming', '--units', '/units',
            '--calibration-census', '/census', '--calibration-cache', '/capture',
            '--calibration-cache-sha256', 'a'*64, '--attention-implementation', 'eager',
            '--capture-load-policy', json.dumps(policy())])


@pytest.mark.parametrize('extra', [[], ['--streaming'], ['--units', '/units'],
    ['--streaming', '--units', '/units', '--calibration-cache', '/capture']])
def test_selected_policy_requires_hash_bound_selected_streaming_scope(tmp_path, extra):
    from prismaquant.tessera_campaign import main
    with pytest.raises(SystemExit) as error:
        main(['--model', '/missing', '--out', str(tmp_path/'out'),
            '--cache-dir', str(tmp_path/'cache'), '--capture-load-policy',
            json.dumps(policy()), *extra])
    assert error.value.code == 2
    assert not (tmp_path/'cache').exists()


def test_selected_admission_prices_private_buffer_source_pages_and_scratch(monkeypatch):
    from prismaquant import autoscale
    source = dict(live_layer_prefix='layers.',
        terms=dict(nonbody_source_bytes=100, declared_headroom_bytes=200),
        body_layer_bytes={'0': 1000}, body_loader_transient_bytes={'0': 100},
        body_source_file_bytes={'0': 900}, unit_source_weight_bytes={'layers.0.proj': 24},
        full_hessian_bytes=64, full_prefix_bytes=32, source_header_sha256='a'*64)
    monkeypatch.setattr(autoscale, 'streamed_calibration_resources', lambda *a, **k: source)
    kwargs = dict(unit_shapes={'layers.0.proj': [3, 4]}, counts={'layers.0.proj': 9},
        max_act_rows=2, cache_slots=2, prefetch_workers=1, headroom_gb=0)
    legacy = autoscale.selected_anchor_resources('/source', **kwargs)
    verified = autoscale.selected_anchor_resources('/source', **kwargs, capture_load_policy=policy())
    assert {name: verified['phases'][name] for name in legacy['phases']} == legacy['phases']
    phase = verified['phases']['capture_prefetch']
    assert phase['capture_serialized_buffer_bytes'] == policy()['max_buffer_bytes']
    assert phase['capture_source_page_cache_bytes'] == policy()['max_buffer_bytes']
    assert phase['capture_load_scratch_bytes'] == policy()['max_scratch_bytes']
    assert phase['capture_decode_storage_bytes'] == 96
    assert phase['selected_hessian_bytes'] == 64 and phase['selected_prefix_bytes'] == 32
    assert verified['capture_load_policy'] == policy()
    assert verified['memory_bytes'] == max(sum(p.values()) for p in verified['phases'].values())


def test_dispatch_seals_same_selected_load_demand(monkeypatch):
    from tools import dispatch_tessera_campaign as dispatch
    from prismaquant import autoscale
    seen = {}
    def selected(model, **kwargs):
        seen.update(kwargs)
        return {'memory_bytes': 100}
    monkeypatch.setattr(autoscale, 'selected_anchor_resources', selected)
    dispatch._streamed_resource_plan(dict(model='/source', campaign_argv=[
        '--streaming', '--capture-load-policy', json.dumps(policy())]),
        dict(unit_shapes={'a': [3, 4]}, counts={'a': 9}), ['a'], selected_source=True)
    assert seen['capture_load_policy'] == policy()


def _selected_args(tmp_path, record, load_policy):
    return SimpleNamespace(calibration_cache=record['path'],
        calibration_cache_sha256=record['sha256'], capture_load_policy=load_policy,
        cache_dir=str(tmp_path/'selected-cache'))


def test_selected_prefetch_preserves_original_capture_and_records_actual_load(capture, tmp_path, monkeypatch):
    import hashlib
    from prismaquant import tessera_campaign as campaign, tessera_calibration_cache as cc
    root, _, census, identity, acts, hs, record = capture
    original_manifest = Path(record['path']).read_bytes()
    original_files = {p: p.read_bytes() for p in (root/'inputs').glob('*.pt')}
    reads = []
    original = cc._verified_capture_entry
    def tracked(path, name, **kwargs):
        reads.append((name, kwargs['expected_sha256']))
        return original(path, name, **kwargs)
    monkeypatch.setattr(cc, '_verified_capture_entry', tracked)
    args = _selected_args(tmp_path, record, policy())
    resources = {'capture_load_policy': policy(), 'memory_bytes': 8*1024**2}
    values, returned_capture, execution = campaign._prefetch_selected_capture(args,
        expected_identity=identity, census=census, names=['a'], device='cpu', resources=resources)
    assert returned_capture == record
    assert torch.equal(values[0]['a'], acts['a']) and torch.equal(values[1]['a'], hs['a'])
    assert values[2] == {'a': census['counts']['a']} and values[3] == {'a': census['max_abs']['a']}
    raw = Path(execution['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == execution['sha256']
    assert execution['sha256'] in Path(execution['path']).name
    run = json.loads(raw)
    assert run['capture'] == record and run['resources'] == resources
    assert run['prefetch']['policy'] == policy() and run['prefetch']['loaded_entries'] == 1
    assert run['prefetch']['source_read_bytes'] == (root/'inputs/a.pt').stat().st_size
    assert run['prefetch']['live_buffer_bytes'] == 0
    assert reads == [('a', json.loads(original_manifest)['entries']['a']['sha256'])]
    assert Path(record['path']).read_bytes() == original_manifest
    assert all(p.read_bytes() == original for p, original in original_files.items())
    # A different load policy gets a different execution receipt, preserving
    # the old binding and exactly the same canonical capture/tensor identity.
    args.capture_load_policy = policy(buffer=2*1024**2)
    _, same_capture, other = campaign._prefetch_selected_capture(args,
        expected_identity=identity, census=census, names=['a'], device='cpu', resources=resources)
    assert same_capture == record and other != execution
    assert Path(execution['path']).read_bytes() == raw


@pytest.mark.parametrize('failure', ['capture_sha', 'artifact_sha', 'oversize', 'guard'])
def test_selected_prefetch_refuses_without_publishing_load_authority(capture, tmp_path, failure):
    from prismaquant import tessera_campaign as campaign
    root, _, census, identity, _, _, record = capture
    args = _selected_args(tmp_path, record, policy())
    guard = None
    if failure == 'capture_sha':
        args.calibration_cache_sha256 = '0'*64
    elif failure == 'artifact_sha':
        p = root/'inputs/a.pt'
        p.write_bytes(p.read_bytes()+b'changed')
    elif failure == 'oversize':
        args.capture_load_policy = policy(buffer=1)
    else:
        def refuse(*args, **kwargs):
            raise RuntimeError('physical guard refusal')
        guard = SimpleNamespace(check=refuse)
    with pytest.raises(RuntimeError):
        campaign._prefetch_selected_capture(args, expected_identity=identity, census=census,
            names=['a'], device='cpu', resources={}, guard=guard)
    assert not list(Path(args.cache_dir).glob('capture-load-execution-*.json'))


def test_selected_legacy_reuse_keeps_its_path_and_has_no_load_receipt(capture, tmp_path, monkeypatch):
    from prismaquant import tessera_campaign as campaign, tessera_calibration_cache as cc
    _, _, census, identity, acts, _, record = capture
    monkeypatch.setattr(cc, '_verified_capture_entry', lambda *a, **k: pytest.fail('legacy used verified loader'))
    values, same_capture, execution = campaign._prefetch_selected_capture(
        _selected_args(tmp_path, record, None), expected_identity=identity,
        census=census, names=['a'], device='cpu', resources={})
    assert torch.equal(values[0]['a'], acts['a']) and same_capture == record and execution is None
    assert not list((tmp_path/'selected-cache').glob('capture-load-execution-*.json'))


@pytest.mark.parametrize('change', ['load_policy', 'source'])
def test_selected_load_policy_and_source_keep_existing_resume_boundaries(tmp_path, monkeypatch, change):
    from prismaquant import tessera_campaign as campaign, production_weight_cache as pwc
    from prismaquant.cost_stage_checkpoint import prepare_journal
    source = {'value': 'a'*64}
    monkeypatch.setattr(pwc, '_production_cache_source_sha256', lambda: source['value'])
    args = SimpleNamespace(capture_load_policy=policy(), cache_dir='/first-cache')
    def identity():
        return campaign._campaign_checkpoint_identity(weights={'a': torch.ones(2, 2)},
            acts={'a': torch.ones(2, 2)}, hessians={'a': torch.eye(2)}, menus={'a': []},
            args=args, calibration_identity={'fit_ids_sha256': 'original-draw'},
            serving_scope=None, static_scales={}, static_scale_policy='fixture')
    before = identity()
    journal = tmp_path/'journal'
    prepare_journal(journal, stage='Tessera campaign', resume=True, identity=before, qnames=['a'])
    if change == 'load_policy':
        args.capture_load_policy = policy(buffer=2*1024**2)
    else:
        source['value'] = 'b'*64
    after = identity()
    assert before['calibration'] == after['calibration'] and before['units'] == after['units']
    with pytest.raises(RuntimeError, match='checkpoint identity mismatch'):
        prepare_journal(journal, stage='Tessera campaign', resume=True, identity=after, qnames=['a'])
