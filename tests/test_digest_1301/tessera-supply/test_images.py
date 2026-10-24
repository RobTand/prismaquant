"""Behavioral comparisons against independent, extracted pre-routing source.

Only domain boundaries (Docker, selection, and publication) are injected. The
functions under test are compiled verbatim from the actual consumer AST, avoiding
GPU/package bootstrap side effects. The ledger names non-parametrized test nodes.
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import subprocess
from unittest.mock import patch
from types import SimpleNamespace

from prismaquant import digests

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/tessera-supply/images'


def load(name, original=False, **dependencies):
    path = (FIXTURES if original else ROOT / 'tools') / (name + '.py')
    tree = ast.parse(path.read_text())
    # Keep real function/class bodies, not import-time deployment dependencies.
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes]
    scope = dict(vars(digests), argparse=argparse, json=json, hashlib=hashlib,
                 Path=Path, sys=sys, base64=base64, io=io, tarfile=tarfile,
                 tempfile=tempfile, subprocess=subprocess)
    scope.update(dependencies)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), scope)
    scope.update(dependencies)
    return SimpleNamespace(**scope)


def outcome(fn, *args):
    try:
        return ('return', fn(*args))
    except Exception as exc:
        return ('raise', type(exc).__name__, str(exc),
                type(exc.__cause__).__name__ if exc.__cause__ else None)


def test_campaign_serializers_and_file_policy(tmp_path):
    name = 'build_prismasnap_27b_campaign'
    # The original wrappers use this unchanged domain error class.
    actual = load(name)
    old = load(name, True, CampaignBuildError=actual.CampaignBuildError)
    values = [{'z': 'é雪', 'a': [1, {}, [], -0.0]}, {10: 'ten', 9: 'nine'},
              {}, [], {'bad': float('nan')}, {'bad': float('inf')},
              {'bad': object()}, {'bad': '\ud800'}, {1: 'one', 'x': 'mixed'}]
    for value in values:
        for method in ('_canonical_bytes', '_pretty_bytes'):
            assert outcome(getattr(actual, method), value) == outcome(getattr(old, method), value)
    for layers in ([], ['sha256:é', 'sha256:b']):
        assert actual._rootfs_layers_sha256(layers) == old._rootfs_layers_sha256(layers)
    for content in (b'', b'a', b'\x00\xff' * (9 * 1024 * 1024)):
        path = tmp_path / 'input'
        path.write_bytes(content)
        assert actual._sha256_file(path) == old._sha256_file(path)
    # Both implementations open once and retain the original 16 MiB read size,
    # including the final EOF read. Refusal checks still precede opening.
    builtin_open = open
    traces = []
    for module in (old, actual):
        events = []
        @contextlib.contextmanager
        def traced_open(file, mode='r', *args, **kwargs):
            events.append(('open', str(file), mode))
            with builtin_open(file, mode, *args, **kwargs) as handle:
                def read(size):
                    block = handle.read(size)
                    events.append(('read', size, len(block)))
                    return block
                yield SimpleNamespace(read=read)
            events.append(('close',))
        with patch('builtins.open', traced_open), patch.object(Path, 'open', traced_open):
            assert module._sha256_file(path) == hashlib.sha256(content).hexdigest()
        traces.append(events)
    assert traces[0] == traces[1]
    assert [e[1] for e in traces[0] if e[0] == 'read'] == [16 * 1024 * 1024] * 3
    link = tmp_path / 'link'
    link.symlink_to(path)
    for refused in (link, tmp_path, tmp_path / 'missing'):
        assert outcome(actual._sha256_file, refused) == outcome(old._sha256_file, refused)
    # Exercise the real CLI serializer without building an actual campaign.
    for payload in ({'z': 'é', 'a': 1}, {'bad': float('nan')}):
        outputs = []
        for original in (True, False):
            module = load(name, _tree_manifest=lambda _: payload,
                          _parser=lambda: SimpleNamespace(parse_args=lambda _: SimpleNamespace(command='hash-tree', path=tmp_path)))
            # The fixture main print is a separate original expression below.
            if original:
                outputs.append(outcome(lambda: old.stdout_json(payload)))
            else:
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    result = outcome(module.main, ['hash-tree'])
                outputs.append(('return', stream.getvalue().rstrip('\n')) if result[0] == 'return' else result)
        assert outputs[0] == outputs[1]


def publish_run(name, original, root, manifest, *, conflict=False, bad_sha=False, refuse=False):
    """Run real main/publication with deterministic domain adapters and capture order."""
    events = []
    root.mkdir(exist_ok=True)
    assignment = root / 'assignment.json'
    assignment.write_bytes('{"selected":"é"}\n'.encode())
    output = root / 'child.json'
    if output.exists():
        output.unlink()
    if conflict:
        output.write_bytes(b'keep')
    def publish(path, encoded):
        events.append(('publish', str(path), encoded))
        if refuse or Path(path).exists():
            return False
        Path(path).write_bytes(encoded)
        return True
    deps = dict(publish_new_bytes=publish)
    if name == 'build_tessera_hessian_collection':
        manifest = dict(manifest, units=['u'], capture_sha256='capture-é')
        deps.update(collection_document=lambda paths: manifest, CANONICAL_CAPTURE='capture')
        reader = SimpleNamespace(__file__=str(assignment))
        class Checked:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self): return iter(manifest['units'])
            def __len__(self): return len(manifest['units'])
            def binding(self): return {'é': 'binding'}
        reader.ReferenceHessianCollection = Checked
        args = ([], output)
        method = 'publish_collection'
    elif name == 'build_tessera_mtp_cached_units':
        def selected(*args, **kwargs):
            events.append(('select',))
            return manifest
        deps['selected_mtp_child'] = selected
        args = (['--assignment', str(assignment), '--assignment-sha256',
                 '0' * 64 if bad_sha else hashlib.sha256(assignment.read_bytes()).hexdigest(),
                 '--mtp-joint-cost', str(root / 'cost'), '--out', str(output)],)
        method = 'main'
    else:
        def compose(*args, **kwargs):
            events.append(('compose',))
            return manifest, SimpleNamespace(child_manifests=['child'], units=['u'],
                       encoder_source_proof_mode='fixture', warnings=[])
        deps['compose'] = compose
        args = (['--child-v2', str(assignment), 'child-sha', '--out', str(output)],)
        method = 'main'
    module = load(name, original, **deps)
    stream = io.StringIO()
    modules = {'tessera': SimpleNamespace(hessian_capture=reader), 'tessera.hessian_capture': reader} if name == 'build_tessera_hessian_collection' else {}
    with patch.dict(sys.modules, modules), contextlib.redirect_stdout(stream):
        result = outcome(getattr(module, method), *args)
    return result, stream.getvalue(), output.read_bytes() if output.exists() else None, events


def compare_publication(name, tmp_path):
    manifest = {'z': 'é雪', 'units': {'u': {'identity': {'encoder_source_sha256': 'a' * 64}}}, 'a': []}
    for kwargs in ({}, {'conflict': True}, {'refuse': True}):
        expected = publish_run(name, True, tmp_path, manifest, **kwargs)
        actual = publish_run(name, False, tmp_path, manifest, **kwargs)
        assert expected == actual
        if not kwargs:
            assert actual[0][0] == 'return'
            assert actual[2].endswith(b'\n') and not actual[2].endswith(b'\n\n')
            assert b'\\u00e9' in actual[2]
        else:
            assert actual[0][0] == 'raise'
    for invalid in (float('nan'), object()):
        broken = dict(manifest, bad=invalid)
        assert publish_run(name, True, tmp_path, broken) == publish_run(name, False, tmp_path, broken)
    if name == 'build_tessera_mtp_cached_units':
        assert publish_run(name, True, tmp_path, manifest, bad_sha=True) == publish_run(name, False, tmp_path, manifest, bad_sha=True)


def test_hessian_publication(tmp_path):
    compare_publication('build_tessera_hessian_collection', tmp_path)


def test_mtp_publication(tmp_path):
    compare_publication('build_tessera_mtp_cached_units', tmp_path)


def test_composed_publication(tmp_path):
    compare_publication('compose_tessera_cached_units', tmp_path)


def add_tar(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def test_glm_image_bytes_and_readback(tmp_path):
    name = 'build_glm_derivative_image'
    old = load(name, True)
    base = tmp_path / 'base.tar'
    config = {'architecture': 'arm64', 'os': 'linux', 'config': {'Env': ['X=é']}, 'rootfs': {'diff_ids': ['sha256:base']},
              'history': [], 'lax': float('nan')}
    with tarfile.open(base, 'w:') as archive:
        add_tar(archive, 'original.json', json.dumps(config).encode())
        add_tar(archive, 'manifest.json', json.dumps([{'Config': 'original.json', 'Layers': ['base/layer.tar']}]).encode())
        add_tar(archive, 'base/layer.tar', b'base-layer')
    source = b'original-model'
    patched = b'corrected-model'
    path = '/site/transformers/models/glm5_next/modeling_glm5_next.py'
    before = {'Id': 'base-id', 'Config': {'Env': ['X=é']}, 'Architecture': 'arm64', 'Os': 'linux',
              'RootFS': {'Layers': ['sha256:base']}}
    after = copy.deepcopy(before)
    def identity(image):
        return 'original' if image == before else 'corrected'
    def run(argv, **kwargs):
        if argv[:2] == ['docker', 'run']:
            return SimpleNamespace(stdout=json.dumps({'source': base64.b64encode(source).decode(),
                        'path': path, 'hub': base64.b64encode('hub-é'.encode()).decode()}))
        if argv[:2] == ['docker', 'load']:
            with tarfile.open(argv[-1]) as archive:
                manifest = json.load(archive.extractfile('manifest.json'))
                layer = archive.extractfile(manifest[0]['Layers'][-1]).read()
                after['RootFS']['Layers'].append('sha256:' + old.layer_digest(layer))
        return SimpleNamespace(stdout='')
    module = load(name, inspect_image=lambda image: before if image == 'prismaquant-glm-producer:content-qualified-20260908' else after,
                  _runtime_identity={'image_content_sha256': identity},
                  ORIGINAL_IMAGE_CONTENT_SHA256='original', CORRECTED_IMAGE_CONTENT_SHA256='corrected',
                  ORIGINAL_MODELING_SHA256=hashlib.sha256(source).hexdigest(),
                  CORRECTED_MODELING_SHA256=hashlib.sha256(patched).hexdigest(),
                  corrected_source=lambda s: patched,
                  subprocess=SimpleNamespace(run=run, check_output=lambda *a, **kw: run(['docker', 'run']).stdout),
                  shutil=SimpleNamespace(disk_usage=lambda p: SimpleNamespace(free=100 * 1024**3)),
                  sha256=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    out = tmp_path / 'out'
    stream = io.StringIO()
    with patch.object(sys, 'argv', ['glm', '--out', str(out), '--base-archive', str(base)]), contextlib.redirect_stdout(stream):
        module.main()
    result = json.loads(stream.getvalue())
    assert stream.getvalue() == old.result_stdout(result) + '\n'
    assert (out / 'result.json').read_text() == old.result_file(result) + '\n'
    assert result['hub_kernels_sha256'] == old.hub_digest({'hub': base64.b64encode('hub-é'.encode()).decode()})
    with tarfile.open(result['archive']['path']) as archive:
        manifest = json.load(archive.extractfile('manifest.json'))[0]
        encoded = archive.extractfile(manifest['Config']).read()
        assert encoded == old.config_bytes(json.loads(encoded))
        assert result['corrected_config_sha256'] == old.config_digest(encoded)
        assert result['corrected_config_sha256'] == old.verified_config_digest(archive, manifest['Config'])
        layer = archive.extractfile(manifest['Layers'][-1]).read()
        assert result['added_layer_sha256'] == old.layer_digest(layer) == old.verified_layer_digest(layer)
    assert base.exists()  # Caller-owned archive was not removed.


def test_embedded_query_witness():
    path = ROOT / 'tools/build_glm_derivative_image.py'
    fixture = FIXTURES / path.name
    def query(source):
        tree = ast.parse(source)
        return next(ast.literal_eval(n.value) for n in ast.walk(tree)
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'query' for t in n.targets))
    assert query(path.read_text()) == query(fixture.read_text())
    # No owner module is assumed to exist inside the original base image.
    assert 'json.dumps' in query(path.read_text())


def test_ledger_fragments_resolve():
    ledger = json.loads((FIXTURES / 'ledger.json').read_text())
    registry = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
    consumers = {f'tools/{path.name}' for path in FIXTURES.glob('*.py')}
    original = {row['id']: row for row in registry['calls'] if row['file'] in consumers}
    ids = [row['original_id'] for row in ledger['calls']]
    assert len(ids) == len(set(ids)) == 22
    assert set(ids) == set(original), 'ledger must join exactly to the frozen registry'
    assert sum(row['status'] == 'routed' for row in ledger['calls']) == 20
    for row in ledger['calls']:
        frozen = original[row['original_id']]
        assert row['source_commit'] == registry['source_commit']
        assert row['source_blob'] == frozen['source_blob']
        assert row['census_id'] == row['original_id']
    witnesses = ledger['auxiliary_witnesses']
    assert len(witnesses) == 1
    assert witnesses[0]['witness_id'] == 'embedded-query:tools/build_glm_derivative_image.py::main.query'
    assert 'original_id' not in witnesses[0]  # Not a completed registry call.
    for row in ledger['calls'] + witnesses:
        assert row['source_commit'] == registry['source_commit']
        assert len(row['source_blob']) == 40
        assert row['fixtures'] and row['tests']
        for fixture in row['fixtures']:
            path = ROOT / fixture
            assert '::' not in fixture and not Path(fixture).is_absolute()
            assert path.resolve().is_relative_to(ROOT) and path.is_file()
            assert fixture.startswith('tests/')
        for reference in row['fixture_references']:
            file, function = reference.split('::')
            assert file in row['fixtures']
            tree = ast.parse((ROOT / file).read_text())
            assert any(isinstance(n, ast.FunctionDef) and n.name == function for n in tree.body)
        for node in row['tests']:
            file, function = node.split('::')
            tree = ast.parse((ROOT / file).read_text())
            assert any(isinstance(n, ast.FunctionDef) and n.name == function for n in tree.body)
        if row['status'] == 'gap':
            assert row['owner_route'] is None and row['reason']
