"""Native recovery launcher intent and finite descriptor-bound capsule reads."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
from pathlib import Path

import pytest


def _campaign():
    return {'artifact_budget_bytes': '4096', 'campaign_bindings': {'plan_sha256': 'a' * 64},
        'campaign_name': 'CPU fixture', 'campaign_scope': {'fixture': True},
        'original_manifest_sha256': 'b' * 64, 'panel': '/fixture/panel',
        'records': 'records', 'run': 'run', 'scope_refusal': 'campaign scope differs: '}


R11 = {'label': 'r11', 'source_head': 'c' * 40, 'manifest_sha256': 'd' * 64,
       'capsule': {'path': '/fixture/capsule.json', 'sha256': 'e' * 64}}


def test_native_launcher_preserves_scientific_bindings_and_uses_existing_row_owner():
    from tools.build_stagea_forward_recovery_package import render_launcher
    text = render_launcher(_campaign(), **R11)
    code = compile(text, 'launch-r11.py', 'exec')
    namespace = {'__file__': '/fixture/package/launch-r11.py', '__name__': 'fixture'}
    exec(code, namespace)
    assert namespace['HEAD'] == 'c' * 40
    assert namespace['MANIFEST_SHA'] == 'd' * 64
    assert namespace['CAPSULE'] == {'path': '/fixture/capsule.json', 'sha256': 'e' * 64}
    argv = ['python', '-s', '-m', 'payload', '--forward-recovery', '/fixture/capsule.json']
    namespace['require_argument'](argv, '--forward-recovery', '/fixture/capsule.json')
    for changed in (argv[:-1] + ['/foreign'], argv + ['--forward-recovery', '/fixture/capsule.json']):
        with pytest.raises(RuntimeError, match='argv differs'):
            namespace['require_argument'](changed, '--forward-recovery', '/fixture/capsule.json')
    tree = ast.parse(text)
    assert any(isinstance(node, ast.ImportFrom) and node.module == 'prismaquant.tessera_campaign_rows'
               and any(name.name == 'submit_rows' for name in node.names) for node in ast.walk(tree))
    assert 'pbrun' not in text and 'PRISMABUILD' not in text


@pytest.mark.parametrize('mutate', [lambda fields: fields.pop('panel'),
    lambda fields: fields.update(unreviewed='x'), lambda fields: fields.update(bound_demand='cpu=2'),
    lambda fields: fields.update(campaign_name='injected\ncode')])
def test_launcher_fields_are_exact(mutate):
    from tools.build_stagea_forward_recovery_package import render_launcher
    fields = _campaign(); mutate(fields)
    with pytest.raises(ValueError, match='launcher fields'):
        render_launcher(fields, **R11)


def _write(path, value):
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('pinned', [False, True])
def test_capsule_read_obeys_same_finite_bound_with_or_without_digest(tmp_path, monkeypatch, pinned):
    from prismaquant import joint_forward_resume as recovery
    from prismaquant.staged_manifest import ManifestRefused
    monkeypatch.setattr(recovery, 'PROOF_READ_MAX_BYTES', 1024)
    path = tmp_path / 'capsule.json'
    digest = _write(path, {'known': [1, 2, 3]})
    assert recovery._read(path, digest if pinned else None)[0] == {'known': [1, 2, 3]}
    oversized = tmp_path / 'oversized.json'
    digest = _write(oversized, {'padding': 'z' * 2048})
    with pytest.raises(ManifestRefused, match='exceeds bound'):
        recovery._read(oversized, digest if pinned else None)


def test_pinned_read_refuses_digest_mismatch_after_real_positive(tmp_path):
    from prismaquant import joint_forward_resume as recovery
    path = tmp_path / 'capsule.json'
    digest = _write(path, {'known': [1, 2, 3]})
    assert recovery._read(path, digest)[0] == {'known': [1, 2, 3]}
    with pytest.raises(recovery.ForwardRecoveryRefused, match='SHA256'):
        recovery._read(path, '0' * 64)


# ------------------------------------------------------ head walk reads (PQ #1051)

WALK_INPUTS = ('campaign_plan', 'census', 'campaign_receipts', 'merged_cost',
               'merged_checkpoint')


def _walk_source(tmp_path, *, annotated_plan=None):
    """A source manifest whose head holds two of the walk's reads, and its campaign."""
    walk = tmp_path / 'walk'
    inputs = {key: {'path': str(walk / f'{key}.json'), 'sha256': 'a' * 64}
              for key in WALK_INPUTS}
    raw = json.dumps({'inputs': inputs}).encode()
    (tmp_path / 'plan.json').write_bytes(raw)
    plan = {'path': str(tmp_path / 'plan.json'), 'sha256': hashlib.sha256(raw).hexdigest()}
    entries = [{'path': '/input/prepared.json', 'offset': 0, 'bytes': 1, 'sha256': None},
               {'path': str(walk / 'census.json'), 'offset': 0, 'bytes': 10, 'sha256': None},
               {'path': str(walk / 'merged_checkpoint.json.parts' / 'units' / '0.pkl'),
                'offset': 0, 'bytes': 100, 'sha256': None},
               {'path': '/input/layer0', 'offset': 0, 'bytes': 3, 'sha256': None}]
    phases = [{'name': name, 'entry_indices': indices, 'bytes': 0, 'cumulative_bytes': 0}
              for name, indices in (('head', [0, 1, 2]), ('forward-000', [3]),
                                    ('chain-000', [3]))]
    original = {'entries': entries, 'entry_count': 4, 'total_bytes': 114,
                'annotations': {'plan_sha256': annotated_plan or plan['sha256']},
                'read_plan': {'phases': phases, 'read_bytes': 0}}
    wire = gzip.compress(json.dumps(original).encode(), mtime=0)
    (tmp_path / 'original.json.gz').write_bytes(wire)
    campaign = {'original_manifest_sha256': hashlib.sha256(wire).hexdigest(),
                'campaign_bindings': {'plan_sha256': plan['sha256']}}
    return tmp_path / 'original.json.gz', campaign, plan


def test_the_recovery_source_leaves_out_the_head_walk_reads(tmp_path):
    from tools.build_stagea_forward_recovery_package import load_original
    path, campaign, plan = _walk_source(tmp_path)
    original, dropped = load_original(path, campaign, plan)
    assert dropped == {'entries': 2, 'bytes': 110}
    head, forward, chain = original['read_plan']['phases']
    assert [original['entries'][i]['path'] for i in head['entry_indices']] == [
        '/input/prepared.json']
    assert [original['entries'][i]['path'] for i in forward['entry_indices']] == [
        '/input/layer0']
    assert chain['entry_indices'] == forward['entry_indices']
    assert [phase['bytes'] for phase in (head, forward, chain)] == [1, 3, 3]
    assert [phase['cumulative_bytes'] for phase in (head, forward, chain)] == [1, 4, 7]
    assert original['read_plan']['read_bytes'] == 7
    assert (original['entry_count'], original['total_bytes']) == (2, 4)


def _another_campaign_plan(path, campaign, plan):
    campaign['campaign_bindings']['plan_sha256'] = 'f' * 64


def _a_moved_plan(path, campaign, plan):
    Path(plan['path']).write_text('{"inputs": {}}')


def _a_moved_manifest(path, campaign, plan):
    campaign['original_manifest_sha256'] = 'f' * 64


@pytest.mark.parametrize('mutate, match', [
    (_another_campaign_plan, 'not the one the campaign binds'),
    (_a_moved_plan, 'does not have the pinned digest'),
    (_a_moved_manifest, 'original scientific read manifest changed'),
])
def test_the_recovery_source_refuses(tmp_path, mutate, match):
    from tools.build_stagea_forward_recovery_package import load_original
    path, campaign, plan = _walk_source(tmp_path)
    mutate(path, campaign, plan)
    with pytest.raises(ValueError, match=match):
        load_original(path, campaign, plan)


def test_the_recovery_source_refuses_a_manifest_built_for_another_plan(tmp_path):
    from tools.build_stagea_forward_recovery_package import load_original
    path, campaign, plan = _walk_source(tmp_path, annotated_plan='e' * 64)
    with pytest.raises(ValueError, match='built for another plan'):
        load_original(path, campaign, plan)
