"""CPU consumer witnesses; decoder adapters are NOT pilot GPU qualification.

Original mains are independent verbatim blocks pinned by the per-call registry.
No fixture is generated from the migrated implementation at test runtime.
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/t4-qualification'
NAMES = ('audit_existing_t4', 'audit_t4_render_paths',
         'audit_tessera_selected_cache_slice', 'inspect_t4_binding', 'pilot_existing_t4')
Q = 'model.language_model.layers.10.mlp.experts.0.down_proj'
FMT = 'TESSERA_E2M1_K2_R896'


class Tensor:
    shape = (2, 3)
    dtype = 'adapter.bfloat16'

    def to(self, dtype):
        return self

    def cpu(self):
        return self


def load_consumer(name, monkeypatch):
    """Import the actual file; replace only unavailable CPU/GPU dependencies."""
    torch = types.ModuleType('torch')
    torch.Tensor = Tensor
    torch.bfloat16 = Tensor.dtype
    torch.load = lambda *a, **k: {'weight': Tensor()}
    torch.equal = lambda a, b: isinstance(a, Tensor) and isinstance(b, Tensor)
    torch.cuda = types.SimpleNamespace(get_device_name=lambda: 'CPU decoder adapter, not GPU qualification',
                                      max_memory_allocated=lambda: 0)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    stubs = {
        'tessera.cached_unit': {'verify_cached_unit': lambda blob, record, identity: types.SimpleNamespace(wire_bytes=len(blob))},
        'prismaquant.tessera_joint_aura': {'_decode_wire': lambda *a, **k: Tensor()},
        'prismaquant.production_weight_cache': {'_canonical_rendered_weight_tensor': lambda x: x},
        'prismaquant.cost_stage_checkpoint': {
            '_load_unit': lambda path, **kw: pickle.loads(path.read_bytes())['state'],
            'unit_path': lambda parts, name: parts / (name + '.pkl')},
        'prismaquant.tessera_campaign_table': {'STAGE': 'adapter-stage'},
    }
    for key, attrs in stubs.items():
        mod = types.ModuleType(key)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, key, mod)
    spec = importlib.util.spec_from_file_location('consumer_' + name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    # Both script-style consumers temporarily modify sys.path during import.
    with monkeypatch.context() as mp:
        mp.setattr(sys, 'path', list(sys.path))
        spec.loader.exec_module(module)
    return module


def setup_inputs(name, tmp_path):
    """Map historical campaign paths into a disposable local tree."""
    def path(value):
        p = Path(value)
        return tmp_path / str(p).lstrip('/') if str(p).startswith('/mnt/') else p

    original = (FIXTURES / ('audits_original_' + name + '.py')).read_text()
    paths = [n.value for n in ast.walk(ast.parse(original))
             if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith('/mnt/')]
    def write(p, data):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    identity = dict(source='source-λ', projection='projection', calibration='cal', encoder_fixture_id='encoder')
    wire = b'wire\x00\xff'
    record = dict(identity=identity, file='sample.wire', blob_sha256=hashlib.sha256(wire).hexdigest())
    unit = dict(wire_records={FMT: record}, candidates=[dict(format=FMT, provenance={'unicode': 'λ'})],
                qname=Q, identity={'source': 'λ'}, anchor={'x': 1})
    payload = pickle.dumps(unit)
    envelope = pickle.dumps(dict(payload=payload, payload_sha256=hashlib.sha256(payload).hexdigest()))
    outputs = []
    if name in ('audit_existing_t4', 'inspect_t4_binding', 'pilot_existing_t4'):
        base = path(paths[0])
        if name == 'audit_existing_t4':
            write(base / 'cost.anchors.json.parts/units/sample.pkl', envelope)
            write(base / 'cost.pkl', pickle.dumps({'costs': {Q: {FMT: 1}}, 'provenance': {'label': 'λ'},
                                                'tessera_expert_wires': {Q: record}, 'formats': [FMT]}))
            write(path(paths[1]), json.dumps({'formats_by_qname': {Q: [FMT], 'other.experts.1': []},
                                             'calibration_input': {'provenance': 'cal'}}).encode())
            write(base / 'cost.anchors.json', json.dumps({'identity': {'calibration': 'cal', 'encoder_source_sha256': 'encoder'},
                                                        'units': [{'qname': Q, 'file': 'units/sample.pkl'}]}).encode())
        elif name == 'inspect_t4_binding':
            write(base / 'production.pkl', pickle.dumps(types.SimpleNamespace(
                metadata={'verified_cells': {(Q, FMT): {'label': 'λ'}}},
                activation_max_abs={Q: 2.0}, activation_scales={Q: 0.5})))
            write(path(paths[1]) / (hashlib.sha256(Q.encode()).hexdigest()+'.pkl'), envelope)
        else:
            for folder in ('extension-e2m1-01/workspace/merged-c92826fa4', 'extension-r1024-02/workspace/rows/row-0045'):
                write(base / folder / 'cost.anchors.json.parts/units' / (hashlib.sha256(Q.encode()).hexdigest()+'.pkl'), envelope)
            write(base / 'extension-e2m1-01/workspace/merged-c92826fa4/cache/wire/sample.wire', wire)
            write(base / 'extension-e2m1-01/workspace/rows/row-0045/cache' / (Q.replace('.', '_')+'__'+FMT+'.pt'), b'render')
    elif name == 'audit_t4_render_paths':
        write(path(paths[1]), json.dumps({'ok': True, 'cell_count': 2, 'pb_actions': ['λ']}).encode())
        row_dir = tmp_path / 'row'
        rendered = row_dir / 'cache' / (Q.replace('.', '_')+'__'+FMT+'.pt')
        write(rendered, b'render\x00\xff')
        rows = [{'dir': str(row_dir), 'members': [Q, 'missing.experts.1']}]
        write(path(paths[0]) / 'plan.json', json.dumps({'rows': rows}).encode())
    else:
        for q in ('dense', 'expert'):
            write(tmp_path / 'parts' / (q+'.pkl'), pickle.dumps({'identity_sha256': 'same-seal', 'state': unit}))
        write(tmp_path / 'wires/sample.wire', wire)
    if name not in ('inspect_t4_binding',):
        outputs = [tmp_path / 'out.json']
    return path, outputs


def invoke(module, name, path, outputs, tmp_path, monkeypatch, original=False):
    ns = dict(module.__dict__)
    ns.update(Path=path, hashlib=hashlib)
    # Deterministic CPU timing only, not an accelerator latency measurement.
    ns['time'] = types.SimpleNamespace(monotonic=lambda: 10.0)
    if name == 'audit_t4_render_paths':
        # The historical proof digest cannot be reproduced. Adapt only this
        # external gate; the rendered-file SHA still uses the real owner.
        proof = json.dumps({'ok': True, 'cell_count': 2, 'pb_actions': ['λ']}).encode()
        # Read the frozen pin rather than assuming another campaign's digest.
        tree = ast.parse((FIXTURES / ('audits_original_'+name+'.py')).read_text())
        pin = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                   and isinstance(n.value, str) and len(n.value) == 64)
        real_sha = hashlib.sha256
        def sha(data):
            return types.SimpleNamespace(hexdigest=lambda: pin) if data == proof else real_sha(data)
        ns['hashlib'] = types.SimpleNamespace(sha256=sha)
        ns['bytes_sha256hex'] = lambda data: sha(data).hexdigest()
        # The consumer imports its stdlib owner inside main. Preserve that
        # route and its real JSON serializer; adapt only the proof digest gate.
        from prismaquant.digests import DIRECT_ASCII_SPACED_LAX
        digests = types.ModuleType('digests')
        digests.DIRECT_ASCII_SPACED_LAX = DIRECT_ASCII_SPACED_LAX
        digests.bytes_sha256hex = lambda data: sha(data).hexdigest()
        monkeypatch.setitem(sys.modules, 'digests', digests)
    if original:
        exec(compile((FIXTURES / ('audits_original_'+name+'.py')).read_text(), '<frozen original main>', 'exec'), ns)
        main = ns['main']
    else:
        # Call the actual imported consumer main, not a rewritten digest lambda.
        for key in ('Path', 'hashlib', 'time', 'bytes_sha256hex'):
            if key in ns:
                monkeypatch.setitem(module.__dict__, key, ns[key])
        main = module.main
    argv = [name]
    if name == 'audit_tessera_selected_cache_slice':
        argv += ['--checkpoint-parts', str(tmp_path/'parts'), '--wire-dir', str(tmp_path/'wires'),
                 '--dense', 'dense', '--expert', 'expert', '--dense-format', FMT, '--expert-format', FMT]
    if outputs:
        argv += ['--out', str(outputs[0])]
    monkeypatch.setattr(sys, 'argv', argv)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        main()
    return capture.getvalue(), [p.read_bytes() for p in outputs]


@pytest.mark.parametrize('name', NAMES)
def test_consumer_main(name, tmp_path, monkeypatch):
    module = load_consumer(name, monkeypatch)
    path, outputs = setup_inputs(name, tmp_path)
    expected = invoke(module, name, path, outputs, tmp_path, monkeypatch, original=True)
    for p in outputs:
        p.unlink()
    actual = invoke(module, name, path, outputs, tmp_path, monkeypatch)
    assert actual == expected  # Exact stdout and publication bytes, including newline/indentation.
    assert actual[0]
    for blob in actual[1]:
        assert blob.endswith(b'\n')
        json.loads(blob)


@pytest.mark.parametrize('name', ['audit_existing_t4', 'pilot_existing_t4'])
def test_malformed_payload_refused_before_unpickle(name, tmp_path, monkeypatch):
    module = load_consumer(name, monkeypatch)
    path, outputs = setup_inputs(name, tmp_path)
    bad_payload = b'not a pickle: must never reach decoder'
    for p in tmp_path.rglob('units/*.pkl'):
        p.write_bytes(pickle.dumps({'payload': bad_payload, 'payload_sha256': '0'*64}))
    real_loads = pickle.loads
    seen = []
    def loads(data):
        seen.append(data)
        assert data != bad_payload, 'malformed payload reached unpickler'
        return real_loads(data)
    monkeypatch.setattr(module, 'pickle', types.SimpleNamespace(loads=loads))
    for original in (True, False):
        with pytest.raises(AssertionError):
            invoke(module, name, path, outputs, tmp_path, monkeypatch, original=original)
        assert bad_payload not in seen
        assert not any(p.exists() for p in outputs)


def test_render_proof_refuses_without_publication(tmp_path, monkeypatch):
    module = load_consumer('audit_t4_render_paths', monkeypatch)
    path, outputs = setup_inputs('audit_t4_render_paths', tmp_path)
    monkeypatch.setattr(module, 'Path', path)
    monkeypatch.setattr(sys, 'argv', ['audit', '--out', str(outputs[0])])
    # No proof digest adapter here: arbitrary bytes must fail the real gate.
    monkeypatch.delitem(sys.modules, 'digests', raising=False)
    with pytest.raises(AssertionError):
        module.main()
    assert not outputs[0].exists()


def test_frozen_registry_and_ast_routes():
    ledger = json.loads((FIXTURES/'audits_ledger.json').read_text())
    deferred = json.loads((FIXTURES/'audits_deferred_json.json').read_text())['eligible_gaps']
    assert len(ledger['calls']) == 6
    assert deferred and all(c['status'] == 'deferred_missing_owner' for c in deferred)
    for name in NAMES:
        source = (ROOT/'tools'/f'{name}.py').read_text()
        tree = ast.parse(source)
        original_main = (FIXTURES/f'audits_original_{name}.py').read_text()
        records = [c for c in ledger['calls'] + deferred if c.get('original_location', c['original_id']).startswith(f'tools/{name}.py:')]
        assert records
        frozen_tree = ast.parse(original_main)
        for c in records:
            # Git objects need not be shipped in test/deployment snapshots.
            # The registry pins the independently extracted original block.
            assert hashlib.sha256(original_main.encode()).hexdigest() == c['fixture_sha256']
            assert len(c['source_commit']) == 40 and len(c['source_blob']) == 40
            _, line, col = c.get('original_location', c['original_id']).rsplit(':', 2)
            node = next(n for n in ast.walk(frozen_tree) if isinstance(n, ast.Call)
                        and n.lineno + c['fixture_main_start_line'] - 1 == int(line)
                        and n.col_offset == int(col)
                        and ast.get_source_segment(original_main, n) == c['expression'])
            assert node is not None
            assert c['fixture'] == f'audits_original_{name}.py'
            assert c['test_node'].split('[')[0].endswith('::test_consumer_main')
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in ('sha256', 'hexdigest') for n in calls)
        migrated = [n.func.id for n in calls if isinstance(n.func, ast.Name)
                    and n.func.id in ('bytes_sha256hex', 'text_sha256hex')]
        assert sorted(migrated) == sorted(c['owner_route'].rsplit('.', 1)[1] for c in ledger['calls']
                                         if c.get('original_location', c['original_id']).startswith(f'tools/{name}.py:'))
        # Every unsorted JSON call stays byte-for-byte unchanged, not a retention.
        dumps = [ast.get_source_segment(source, n) for n in calls
                 if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == 'json' and n.func.attr == 'dumps']
        assert sorted(dumps) == sorted(c['expression'] for c in deferred
                                      if c.get('original_location', c['original_id']).startswith(f'tools/{name}.py:'))


def test_stdlib_t4_tools_load_without_package(tmp_path):
    """Same-tree owner loading never requires an installed PrismaQuant package."""
    import subprocess
    (tmp_path / 'tools').mkdir()
    (tmp_path / 'prismaquant').mkdir()
    (tmp_path / 'prismaquant/digests.py').write_bytes((ROOT / 'prismaquant/digests.py').read_bytes())
    names = ('audit_existing_t4', 'inspect_t4_binding', 'build_t4_recovery_request')
    for name in names:
        (tmp_path / 'tools' / (name + '.py')).write_bytes((ROOT / 'tools' / (name + '.py')).read_bytes())
    code = r"""
import hashlib, importlib.abc, runpy, sys
from pathlib import Path
class NoPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('prismaquant', 'torch', 'tessera'):
            raise AssertionError('unexpected package import: ' + fullname)
sys.meta_path.insert(0, NoPackage())
root = Path(sys.argv[1])
for name in ('audit_existing_t4', 'inspect_t4_binding', 'build_t4_recovery_request'):
    before = list(sys.path)
    ns = runpy.run_path(str(root / 'tools' / (name + '.py')))
    assert sys.path == before if name != 'build_t4_recovery_request' else sys.path[1:] == before
    if name == 'build_t4_recovery_request':
        assert ns['sha'](b'\x00example') == hashlib.sha256(b'\x00example').hexdigest()
    if name == 'audit_existing_t4':
        sys.argv = [name, '--help']
        try:
            ns['main']()
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError('help did not exit')
        assert sys.path == before
"""
    result = subprocess.run([sys.executable, '-I', '-S', '-c', code, str(tmp_path)],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
