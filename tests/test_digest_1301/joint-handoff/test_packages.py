"""Package compatibility against verbatim research-base consumer excerpts."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types

import pytest

from tools import build_stagea_forward_recovery_package as recovery
from tools import build_stagea_seed_package as seed
from tools import build_stagea_split_package as split

FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures/digest_1301/joint-handoff'


def original(module, kind):
    namespace = dict(vars(module))
    namespace['hashlib'] = hashlib
    exec(compile((FIXTURES / f'original_packages_{kind}.py').read_text(),
                 f'original_packages_{kind}.py', 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


def patch(monkeypatch, module, oracle, name, value):
    monkeypatch.setattr(module, name, value)
    # The excerpt functions share this independent globals dictionary.
    for item in vars(oracle).values():
        if isinstance(item, types.FunctionType) and item.__code__.co_filename.startswith('original_packages_'):
            item.__globals__[name] = value


def manifest():
    entries = [{'path': 'weights.safetensors', 'offset': 0, 'bytes': 7, 'sha256': 'a' * 64}]
    phases = [{'name': name, 'entry_indices': [0], 'bytes': 7, 'cumulative_bytes': 7 * (i + 1)}
              for i, name in enumerate(['head', 'forward-000', 'forward-001', 'chain-001', 'chain-000'])]
    return {'entries': entries, 'entry_count': 1, 'total_bytes': 7,
            'read_plan': {'phases': phases, 'read_bytes': 35},
            'annotations': {'forward_recovery': {'path': 'capsule', 'sha256': 'b' * 64}, 'unicode': '雪'}}


def files(root):
    return {p.name: p.read_bytes() for p in root.iterdir()}


def test_manifest_wire_and_refusals():
    old = original(seed, 'seed')
    for value in [manifest(), {'é': [float('nan'), float('inf'), -0.0], 'a': '\n'}, {10: 'x', 9: 'y'}]:
        actual = seed.manifest_wire(value)
        assert actual == old.manifest_wire(value)
        assert actual[4:8] == b'\0' * 4
        assert gzip.decompress(actual).endswith(b'\n')
    for value in [{'x': object()}, {'x': '\ud800'}, {1: 0, 'a': 1}]:
        # ASCII escaping accepts lone surrogates; other failures stay stdlib failures.
        try:
            expected = old.manifest_wire(value)
        except Exception as error:
            with pytest.raises(type(error)) as actual:
                seed.manifest_wire(value)
            assert str(actual.value) == str(error)
        else:
            assert seed.manifest_wire(value) == expected


def test_pins_and_source_refusals(tmp_path, monkeypatch):
    old_seed = original(seed, 'seed')
    old_recovery = original(recovery, 'forward_recovery')
    source = tmp_path / 'source.gz'
    wire = seed.manifest_wire(manifest())
    source.write_bytes(wire)
    digest = hashlib.sha256(wire).hexdigest()
    assert seed._pin(source, digest, 'checkpoint') == old_seed._pin(source, digest, 'checkpoint')
    patch(monkeypatch, recovery, old_recovery, 'drop_source_head_walk_reads', lambda obj, plan: (obj, []))
    campaign = {'original_manifest_sha256': digest, 'campaign_bindings': {'plan_sha256': 'plan'}}
    assert recovery.load_original(source, campaign, {'sha256': 'plan'}) == old_recovery.load_original(source, campaign, {'sha256': 'plan'})
    for module in [seed, old_seed]:
        with pytest.raises(seed.SeedPackageRefused, match='does not have the pinned digest'):
            module._pin(source, 'bad', 'checkpoint')
        with pytest.raises(FileNotFoundError):
            module._pin(tmp_path / 'missing', digest, 'checkpoint')
    for module in [recovery, old_recovery]:
        with pytest.raises(ValueError, match='the plan is not the one'):
            module.load_original(source, campaign, {'sha256': 'other'})
        with pytest.raises(ValueError, match='original scientific read manifest changed'):
            module.load_original(source, {}, {})


@pytest.mark.parametrize('nonfinite', [False, True], ids=['finite', 'nonfinite'])
def test_seed_package_publication(tmp_path, monkeypatch, nonfinite):
    old = original(seed, 'seed')
    source = tmp_path / 'source'
    source.write_bytes(seed.manifest_wire(manifest()))
    values = [float('nan'), float('inf'), -float('inf')] if nonfinite else []
    patch(monkeypatch, seed, old, 'drop_source_head_walk_reads', lambda obj, plan: (obj, {'entries': 0, 'diagnostics': values}))
    patch(monkeypatch, seed, old, 'normalize_seed_spec', lambda obj: obj)
    patch(monkeypatch, seed, old, 'load_pinned_checkpoint', lambda *args: {'boundary': 2})
    patch(monkeypatch, seed, old, '_checkpoint_files', lambda record: ([], [], []))
    root = tmp_path / 'output'
    kwargs = dict(original_manifest=source, original_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  plan={}, checkpoint={'diagnostics': values}, capsule=manifest()['annotations']['forward_recovery'], through=0,
                  implementation_from=None, implementation_to=None, compare=None, output=root)
    expected = old.build(**kwargs)
    expected_files = files(root)
    shutil.rmtree(root)
    seen = []
    assert seed.build(**kwargs, validate=seen.append) == expected
    assert files(root) == expected_files
    if nonfinite:
        for name in [seed.SPEC_NAME, seed.PACKAGE_NAME]:
            for token in [b'NaN', b'Infinity', b'-Infinity']:
                assert token in expected_files[name]
            assert expected_files[name].endswith(b'\n')
    assert len(seen) == 1
    with pytest.raises(FileExistsError):
        seed.build(**kwargs)
    shutil.rmtree(root)
    def refuse(_):
        raise ValueError('validation refused')
    with pytest.raises(ValueError, match='validation refused'):
        seed.build(**kwargs, validate=refuse)
    assert not root.exists()
    kwargs['original_manifest_sha256'] = 'bad'
    with pytest.raises(seed.SeedPackageRefused, match='source manifest does not have the pinned digest'):
        seed.build(**kwargs)
    assert not root.exists()


@pytest.mark.parametrize('nonfinite', [False, True], ids=['finite', 'nonfinite'])
def test_split_describe_and_chain_publication(tmp_path, monkeypatch, nonfinite):
    old = original(split, 'split')
    obj = manifest()
    wire = seed.manifest_wire(obj)
    assert split._describe('é.gz', wire, obj, {'checked': 0}) == old._describe('é.gz', wire, obj, {'checked': 0})
    values = [float('nan'), float('inf'), -float('inf')] if nonfinite else []
    patch(monkeypatch, split, old, '_load_original', lambda *args: (obj, values))
    loaded = {'boundary': 2, 'state_entry': {'path': 'state'},
              'state': {'chain_state_sha256': 'state', 'n_batches': 1},
              'sealed': {2: {'cotangent_sha256': 'plane'}}}
    patch(monkeypatch, split, old, 'load_round', lambda *args: loaded)
    prep = json.loads(json.dumps(obj))
    prep['annotations']['chain_split'] = {'boundaries': [2, 0]}
    patch(monkeypatch, split, old, 'round_manifests', lambda *args, **kwargs: (prep, {(0, 1): obj}))
    root = tmp_path / 'output'
    kwargs = dict(original_manifest='source', original_manifest_sha256='pin', plan={}, output_root='run',
                  chain_state_sha256='state', checkpoint_sha256='ckpt', through=0, ranges=[(0, 1)], output=root)
    expected = old.build(**kwargs)
    expected_files = files(root)
    shutil.rmtree(root)
    assert split.build(**kwargs) == expected
    assert files(root) == expected_files
    if nonfinite:
        for token in [b'NaN', b'Infinity', b'-Infinity']:
            assert token in expected_files[split.PACKAGE_NAME]
        assert expected_files[split.PACKAGE_NAME].endswith(b'\n')
    with pytest.raises(FileExistsError):
        split.build(**kwargs)


def test_load_round_checkpoint_digest(tmp_path, monkeypatch):
    from prismaquant import joint_adjoint_checkpoints as checkpoints
    from prismaquant import stage_a_chain_resume as resume
    old = original(split, 'split')
    directory = tmp_path / 'checkpoint'
    directory.mkdir()
    checkpoint = directory / 'checkpoint.json'
    checkpoint.write_bytes(b'{"z": "\\u96ea"}\n')
    state_path = tmp_path / 'chain-state.json'
    state_path.write_bytes(b'{}\n')
    state = {'boundary_storage': {'session': {'generation': 'g', 'run_identity_sha256': 'id'}},
             'stride': {'boundaries': [2]}}
    monkeypatch.setattr(checkpoints, 'adjoint_space', lambda path: path)
    monkeypatch.setattr(resume, 'load_chain_state', lambda *args: state)
    monkeypatch.setattr(resume, 'chain_state_path', lambda space: state_path)
    monkeypatch.setattr(resume, '_sealed_checkpoints', lambda *args: ({2: ({'boundary': 2}, directory)}, []))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert split.load_round('run', 'state', digest) == old.load_round('run', 'state', digest)
    for module in [split, old]:
        with pytest.raises(split.SplitPackageRefused, match='lowest sealed checkpoint'):
            module.load_round('run', 'state', 'bad')


@pytest.mark.parametrize('nonfinite', [False, True], ids=['finite', 'nonfinite'])
def test_recovery_main_publication(tmp_path, monkeypatch, capsys, nonfinite):
    from prismaquant import staged_lease
    old = original(recovery, 'forward_recovery')
    obj = manifest()
    patch(monkeypatch, recovery, old, '_read', lambda *args: ({'frontier': 1}, None))
    patch(monkeypatch, recovery, old, 'load_original', lambda *args: (obj, []))
    patch(monkeypatch, recovery, old, 'recovery_manifest', lambda *args: obj)
    monkeypatch.setattr(staged_lease, 'sdk_submodule', lambda name: types.SimpleNamespace(validate_data_manifest=lambda obj: None))
    campaign = tmp_path / 'campaign.json'
    campaign.write_text(json.dumps({name: ('雪' if name != 'records' else 'records') for name in recovery.CAMPAIGN_FIELDS}))
    prior = tmp_path / 'prior'
    prior.mkdir()
    for name in ['prefetch-override-r7.json', 'stage-a-spec-r7.json', 'stage-a-dispatch-config.json']:
        (prior / name).write_bytes(b'{"keep": "exact bytes"}\n')
    template = {'template_id': 'snow-r9', 'é': 1}
    if nonfinite:
        template['diagnostics'] = [float('nan'), float('inf'), -float('inf')]
    (prior / 'stagea-512-template-r7.json').write_text(json.dumps(template))
    root = tmp_path / 'output'
    argv = ['builder', '--capsule', 'capsule', '--capsule-sha256', 'pin', '--original-manifest', 'original',
            '--plan', 'plan', '--plan-sha256', 'plan-pin', '--campaign-fields', str(campaign),
            '--r9-package', str(prior), '--output', str(root), '--source-head', 'head']
    monkeypatch.setattr(sys, 'argv', argv)
    old.main()
    expected_stdout = capsys.readouterr().out
    expected_files = files(root)
    shutil.rmtree(root)
    recovery.main()
    assert capsys.readouterr().out == expected_stdout
    assert files(root) == expected_files
    if nonfinite:
        for token in [b'NaN', b'Infinity', b'-Infinity']:
            assert token in expected_files['stagea-512-template-r7.json']
        assert expected_files['stagea-512-template-r7.json'].endswith(b'\n')
    with pytest.raises(FileExistsError):
        recovery.main()
    shutil.rmtree(root)
    monkeypatch.setattr(sys, 'argv', argv + ['--label', 'r9'])
    with pytest.raises(ValueError, match='post-R9 attempt'):
        recovery.main()
    assert not root.exists()
    # Unsorted launch-fields publication is intentionally not a sorted profile.
    fields = expected_files['launch-fields.json']
    assert fields.startswith(b'{\n  "campaign": ')
    assert fields != (json.dumps(json.loads(fields), sort_keys=True, indent=2) + '\n').encode()


def test_unresolved_recovery_roundtrip(monkeypatch, tmp_path):
    old = original(recovery, 'forward_recovery')
    capsule_path = tmp_path / 'capsule'
    capsule_path.write_bytes(b'capsule')
    capsule = {'frontier': 2, 'n_batches': 1, 'groups': [{'manifest': {'entries': [
        {'destination_path': f'boundary-0-{layer}-at-{layer}.pt', 'bytes': 3, 'sha256': 'c' * 64}
        for layer in range(3)]}}]}
    patch(monkeypatch, recovery, old, 'chain_documents', lambda obj: [obj])
    obj = manifest()
    obj['annotations']['key_order'] = {10: 'ten', 9: 'nine'}
    bound = {'path': str(capsule_path), 'sha256': 'd' * 64}
    assert recovery.recovery_manifest(obj, capsule, bound) == old.recovery_manifest(obj, capsule, bound)
    assert list(obj['annotations']['key_order']) == [10, 9]  # no input mutation


def test_ledger_provenance():
    ledger = json.loads((FIXTURES / 'ledger_packages.json').read_text())
    registry = json.loads((FIXTURES.parents[1] / 'digest_parent_completion_1301.json').read_text())
    calls = {row['id']: row for row in registry['calls']}
    assert ledger['counts'] == {'eligible': 21, 'replaced': 18,
                                'unresolved_owner_api_gaps': 3, 'justified_retentions': 0}
    assert len(ledger['replacements']) == 18
    indent_routes = [row for row in ledger['replacements']
                     if row['owner_route'] == 'prismaquant.digests.DIRECT_ASCII_INDENT2_LAX.text']
    assert len(indent_routes) == 5
    assert all(row['owner_route'] != 'prismaquant.digests.indent2_json_file_bytes'
               for row in ledger['replacements'])
    assert ledger['retentions'] == []
    assert len(ledger['unresolved_owner_api_gaps']) == 3
    for gap in ledger['unresolved_owner_api_gaps']:
        assert gap['status'] == 'eligible_unresolved_owner_api_gap'
        assert gap['owner_route'] is None
        assert gap['blocker']
    for row in ledger['replacements'] + ledger['unresolved_owner_api_gaps']:
        base = calls[row['original_id']]
        assert row['source_commit'] == registry['source_commit']
        assert row['source_blob'] == base['source_blob']
        excerpt = FIXTURES / Path(row['fixtures'][0]).name
        assert base['scope'] in {n.name for n in ast.parse(excerpt.read_text()).body if isinstance(n, ast.FunctionDef)}
        for node in row['tests']:
            path, name = node.split('::')
            tree = ast.parse((FIXTURES.parents[3] / path).read_text())
            assert name in {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
