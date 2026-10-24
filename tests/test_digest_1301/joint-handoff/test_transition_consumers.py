"""Execute live consumers against independently frozen pre-migration excerpts.

Dependency doubles isolate expensive intake, not the consumer or hash sites.
Main tests execute whole live AST transaction blocks, including their refusals.
"""
import ast
import copy
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / 'tests/fixtures/digest_1301/joint-handoff'
COMMIT = '6d8288adc94b3c4b624c143ed372fb6513f32784'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def consumer_pair(stem, name):
    live = importlib.import_module('tools.' + stem)
    ns = dict(vars(live), hashlib=hashlib)
    source = (FIX / (stem + '.consumers.original.txt')).read_text()
    exec(compile('from __future__ import annotations\n' + source, '<frozen consumers>', 'exec'), ns)
    return live, (ns[name], getattr(live, name))


def test_consumer_excerpt_provenance():
    """Every frozen function/block is verbatim original, not reconstructed live code."""
    provenance = json.loads((FIX / 'consumer_excerpts.json').read_text())
    registry = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
    assert provenance['source_commit'] == registry['source_commit'] == COMMIT
    for filename, evidence in provenance['excerpts'].items():
        assert sha((FIX / filename).read_bytes()) == evidence['sha256']
        assert evidence['source_blob'] in {c['source_blob'] for c in registry['calls'] if c['file'] == evidence['source_file']}
        assert all(start <= end for start, end in evidence['line_ranges'])


def test_load_json_wire_verification_before_decode(tmp_path, monkeypatch):
    import prismaquant.stage_b_prep_io as io
    live, pair = consumer_pair('regenerate_joint_quanta', '_load_json')
    path = tmp_path / 'not-a-gzip-suffix'
    # Bypass the shared IO verifier so the consumer's own check is exercised.
    monkeypatch.setattr(io, 'read_input', lambda path, **kw: path.read_bytes())
    for fn in pair:
        for raw in (b'\x1f\x8bnot gzip', b'\xffnot utf8', b'not json'):
            path.write_bytes(raw)
            with pytest.raises(ValueError, match='digest mismatch'):
                fn(path, digest=sha(b'other'), where='fixture')
        raw = gzip.compress(b'{"name":"\\u00e9"}', mtime=0)
        path.write_bytes(raw)
        assert fn(path, digest=sha(raw), where='fixture') == {'name': 'é'}
        with pytest.raises(ValueError, match='digest mismatch'):
            fn(path, digest=sha(gzip.decompress(raw)), where='fixture')
        path.write_bytes(b'\xff')
        with pytest.raises(ValueError, match='is not JSON'):
            fn(path, digest=sha(b'\xff'), where='fixture')


def test_prior_slice_refuses_unverified_bytes(tmp_path, monkeypatch):
    live, pair = consumer_pair('regenerate_joint_quanta', '_prior_slice_manifest')
    path = tmp_path / 'slice'
    path.write_bytes(b'not JSON')
    record = {'read_set': {'manifest_path': str(path), 'manifest_sha256': sha(b'other')}}
    for fn in pair:
        calls = []
        with monkeypatch.context() as m:
            m.setitem(fn.__globals__, '_load_verified_json', lambda *a, **kw: calls.append((a, kw)) or {'verified': True})
            with pytest.raises(ValueError, match='refusing to compare unverified bytes'):
                fn(record, where='prior')
            assert calls == []
            record['read_set']['manifest_sha256'] = sha(path.read_bytes())
            assert fn(record, where='prior') == {'verified': True}
            assert calls[0][0] == (path, sha(path.read_bytes()))
            record['read_set']['manifest_sha256'] = sha(b'other')


def test_cache_mismatch_before_unpickle(tmp_path, monkeypatch):
    import prismaquant.stage_b_prep_io as io
    import prismaquant.production_weight_cache as cachemod
    _, pair = consumer_pair('regenerate_joint_quanta', '_load_production_cache')
    path = tmp_path / 'cache'
    path.write_bytes(b'not a pickle')
    monkeypatch.setattr(io, 'read_input', lambda path, **kw: path.read_bytes())
    cache = object.__new__(cachemod.ProductionWeightCache)
    for fn in pair:
        calls = []
        with monkeypatch.context() as m:
            import pickle
            m.setattr(pickle, 'loads', lambda raw: calls.append(raw) or cache)
            with pytest.raises(ValueError, match='digest mismatch'):
                fn({'production_cache': {'path': str(path), 'sha256': sha(b'other')}})
            assert calls == []
            assert fn({'production_cache': {'path': str(path), 'sha256': sha(path.read_bytes())}}) is cache
            assert calls == [path.read_bytes()]


def test_head_slices_bind_exact_emitted_bytes(tmp_path, monkeypatch):
    import prismaquant.aura_provenance as provenance
    import prismaquant.joint_stage_b_head as head
    import prismaquant.tessera_joint_aura as aura
    import prismaquant.tessera_reader as reader
    _, pair = consumer_pair('regenerate_joint_quanta', '_build_head_slices')
    wire = b'\x1f\x8bwire head\x00\xff'
    calls = []
    monkeypatch.setattr(provenance, 'producer_package_sha256', lambda: 'implementation')
    monkeypatch.setattr(reader, 'load_declared_reader', lambda value: 'reader')
    monkeypatch.setattr(aura, 'resolve_file_hash_workers', lambda plan: 1)
    monkeypatch.setattr(aura, 'load_measured_anchor_input', lambda *a, **kw: calls.append(('intake', kw)) or types.SimpleNamespace(formats_by_qname={}, cells=[]))
    monkeypatch.setattr(head, 'build_head_slices', lambda **kw: {2: {'head_files': [{'path': 'input'}]}})
    monkeypatch.setattr(head, 'head_slice_bytes', lambda value: wire)
    monkeypatch.setattr(head, 'head_slice_path', lambda *a, **kw: str(tmp_path / 'head'))
    results = [fn({'inputs': {}}, plan_sha256='plan', prepared={}, prepared_binding={}, production_cache=None, layers=[2], output_root=str(tmp_path), metadata_root=None) for fn in pair]
    assert results[0] == results[1]
    sealed = results[1][2]
    assert sealed['bytes'] == wire
    assert sealed['binding']['sha256'] == sha(wire)
    assert sealed['binding']['bytes'] == len(wire)
    assert sealed['binding']['head_files'] == [{'path': 'input'}]
    assert len(calls) == 2  # exactly one intake per consumer call
    assert all(c[1]['require_existing_renders'] and not c[1]['verify_payloads'] for c in calls)


def test_prepare_reads_extension_wire_digest(tmp_path, monkeypatch):
    import tools.prepare_extended_joint_quanta as prepare
    _, pair = consumer_pair('stage_b_preparation_submission', 'prepare_reads')
    extension = tmp_path / 'catalog-extension.json'
    wire = b'{ "z": 1, "name": "\\u00e9" }\n'
    extension.write_bytes(wire)
    inputs = {'extended_plan': {'path': 'plan', 'sha256': 'p'}, 'extended_prepared': {'path': 'prepared', 'sha256': 'r'}}
    docs = {'pair': inputs, 'plan': {}, 'prepared': {'production_cache': {'path': 'cache', 'sha256': 'c'}}, 'parent': {'annotations': {'layers': [0, 1]}}, 'derivation': {}, 'spec': {}}
    args = types.SimpleNamespace(pair_inputs='pair', pair_inputs_sha256='i', parent_manifest='parent', parent_manifest_sha256='m', derivation='derivation', derivation_sha256='d', spec='spec', spec_sha256='s', metadata_root=tmp_path, campaign_identity=None)
    seen = []
    monkeypatch.setattr(prepare, 'control_digests', lambda inputs, plan, prepared, ext: seen.append(ext) or {})
    results = []
    for fn in pair:
        with monkeypatch.context() as m:
            for name, value in {'_load': lambda path, *a: docs[str(path)], '_proof_files': lambda args: ([], 1), 'head_phase_entries': lambda parent: [], '_source_reads': lambda *a: [], 'preparation_read_entries': lambda **kw: kw['files']}.items():
                m.setitem(fn.__globals__, name, value)
            results.append(fn(args))
    assert results[0] == results[1]
    assert seen == [{'path': str(extension.resolve()), 'sha256': sha(wire)}] * 2
    assert (extension, sha(wire)) in results[1][0]
    assert results[1][2:] == (2, 1)


def test_submission_published_wire_and_strict_flags(tmp_path, monkeypatch):
    _, pair = consumer_pair('stage_b_preparation_submission', 'write_submission')
    args = types.SimpleNamespace(out=tmp_path / 'submission', metadata_root=tmp_path / 'metadata', tool='prepare', mount_prefix='/fixture', tier='ssd')
    manifest = {'entry_count': 1, 'total_bytes': 7}
    template = {'template_id': 'fixture', 'name': 'é'}
    wire = gzip.compress(b'{ "wire": true }\n', mtime=0)
    results = []
    for fn in pair:
        with monkeypatch.context() as m:
            for name, value in {'preparation_read_manifest': lambda *a, **kw: manifest, 'build_preparation_template': lambda **kw: template, 'preparation_payload_ceiling': lambda *a, **kw: 100, 'seal_manifest_bytes': lambda value: wire}.items():
                m.setitem(fn.__globals__, name, value)
            result = fn(args, [], {}, 1, 1)
            results.append(result)
            assert (args.out / 'read-manifest.json.gz').read_bytes() == wire
            template_wire = (json.dumps(template, sort_keys=True, indent=1) + '\n').encode()
            assert (args.out / 'template.json').read_bytes() == template_wire
            assert result['data_manifest']['sha256'] == sha(wire)
            assert result['produced_output_template']['sha256'] == sha(template_wire)
            assert result['strict_read_flags'] == ['--data-manifest-sha256', sha(wire), '--allowed-tiers', 'ram,ssd']
            assert json.loads((args.out / 'submission.json').read_bytes()) == result
            (args.out / 'template.json').write_bytes(b'conflict')
            with pytest.raises(ValueError, match='already holds other bytes'):
                fn(args, [], {}, 1, 1)
            (args.out / 'template.json').write_bytes(template_wire)
    assert results[0] == results[1]


def live_block(label):
    module = ast.parse((ROOT / 'tools/regenerate_joint_quanta.py').read_text())
    tree = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    needle = '_check_authorized_metadata_diff' if label == 'relocation' else 'publish_files'
    candidates = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and any(isinstance(c, ast.Call) and ast.unparse(c.func) == needle for c in ast.walk(n))]
    node = min(candidates, key=lambda n: n.end_lineno - n.lineno)
    assert sum(isinstance(c, ast.Call) and ast.unparse(c.func) == 'bytes_sha256hex' for c in ast.walk(node)) == (1 if label == 'relocation' else 4)
    return node


def run_block(node, ns):
    # Only add a function boundary to preserve original return/refusal semantics.
    wrapper = ast.FunctionDef(name='_transaction', args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[copy.deepcopy(node)], decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), '<transaction>', 'exec'), ns)
    return ns['_transaction']()


@pytest.mark.parametrize('damage', [None, 'head', 'adjoint', 'slice', 'bound', 'missing-slice', 'missing-bound'])
@pytest.mark.parametrize('produced_output', [False, True])
def test_main_publication_transaction(tmp_path, monkeypatch, damage, produced_output):
    import tools.regenerate_joint_quanta as live
    import prismaquant.stage_b_prep_io as io
    frozen = ast.parse((FIX / 'publication.block.original.txt').read_text()).body[0]
    traces = []
    for node in (frozen, live_block('publication')):
        trace = []
        for path in tmp_path.iterdir():
            if path.is_file():
                path.unlink()
        paths = {k: tmp_path / k for k in ('head', 'adjoint', 'slice', 'bound')}
        payload = {k: ('wire ' + k).encode() for k in paths}
        record = {'quantum_id': 'q', 'read_set': {'manifest_path': str(paths['slice']), 'manifest_sha256': sha(payload['slice'])}, 'adjoint': {'slice_path': str(paths['adjoint']), 'slice_sha256': sha(payload['adjoint'])}}
        def publish(publication, phase, files):
            trace.append((phase, [kind for _, _, kind in files]))
            for path, raw, _ in files:
                path.write_bytes(raw)
            if phase == 'manifests' and damage:
                key = damage.removeprefix('missing-')
                if damage.startswith('missing-'):
                    paths[key].unlink()
                else:
                    paths[key].write_bytes(b'tampered')
        with monkeypatch.context() as m:
            m.setattr(io, 'bind_preparation_publication', lambda *a, **kw: trace.append(('bind', kw)) or None)
            m.setattr(io, 'publish_files', publish)
            ns = dict(vars(live), hashlib=hashlib, Path=Path, bind_preparation_publication=io.bind_preparation_publication, publish_files=io.publish_files, PreparationPublicationRefused=io.PreparationPublicationRefused, args=types.SimpleNamespace(produced_output=produced_output, adjoint_band=[], adjoint_receipt=None), metadata_root=tmp_path, out=tmp_path, produced={'records': [record], 'slice_manifests': {'q': payload['slice']}, 'derivation': {}}, head_slices={0: {'binding': {'path': str(paths['head']), 'sha256': sha(payload['head'])}, 'bytes': payload['head']}}, adjoint_slices={'q': payload['adjoint']}, bound_manifests=[(str(paths['bound']), payload['bound'], sha(payload['bound']))], seal_manifest_bytes=lambda value: value, adjoint_slice_bytes=lambda value: value, _fail=lambda message: trace.append(('refused', message)) or 2)
            result = run_block(node, ns)
        assert result == (None if damage is None else 2)
        assert any(t[0] == 'records' for t in trace) == (damage is None)
        assert trace[0] == ('bind', {'required': produced_output})
        assert trace[1][0] == 'manifests'
        if damage is None:
            assert json.loads((tmp_path / 'records.json').read_bytes()) == [record]
        traces.append(trace)
    assert traces[0] == traces[1]


@pytest.mark.parametrize('refusal', [None, 'prior', 'authorization'])
def test_main_relocation_authorization_transaction(tmp_path, monkeypatch, refusal):
    import tools.regenerate_joint_quanta as live
    frozen = ast.parse((FIX / 'relocation.block.original.txt').read_text()).body[0]
    traces = []
    for node in (frozen, live_block('relocation')):
        trace = []
        wire = b'wire relocated manifest'
        old, new = {'quantum_id': 'q', 'generation': 'old'}, {'quantum_id': 'q', 'generation': 'new'}
        def prior(record, **kw):
            trace.append(('prior', record, kw))
            if refusal == 'prior':
                raise ValueError('prior unverified')
        def authorize(before, after, **kw):
            trace.append(('authorize', before, after, kw))
            assert kw['slice_sha256'] == sha(wire)
            assert kw['bound'] is True
            if refusal == 'authorization':
                raise ValueError('unauthorized relocation')
        ns = dict(vars(live), hashlib=hashlib, old_by_id={'q': old}, moved={'records': [new], 'slice_manifests': {'q': wire}}, metadata_root=str(tmp_path), args=types.SimpleNamespace(original_root=tmp_path / 'old'), receipt={}, seal_manifest_bytes=lambda value: wire, _prior_slice_manifest=prior, _check_authorized_metadata_diff=authorize, _fail=lambda msg: trace.append(('refused', msg)) or 2)
        result = run_block(node, ns)
        assert result == (None if refusal is None else 2)
        assert [t[0] for t in trace] == ({None: ['prior', 'authorize'], 'prior': ['prior', 'refused'], 'authorization': ['prior', 'authorize', 'refused']}[refusal])
        traces.append(trace)
    assert traces[0] == traces[1]


def test_package_unicode_and_surrogate_parity():
    from tools.generate_transition_rewrites import _package_digest
    ns = {'hashlib': hashlib, 'Path': Path}
    exec((FIX / 'generate_transition_rewrites.original.txt').read_text(), ns)
    original = ns['_package_digest']
    for files in ({'é.py': b'one', 'e\u0301.py': b'two', '雪/😀.py': b'\x00\xff'}, {'😀': b'x', 'a': b'y'}):
        assert _package_digest(files) == original(files)
        assert _package_digest(dict(reversed(list(files.items())))) == original(files)
    for name in ('\ud800.py', 'x\udfff'):
        for fn in (original, _package_digest):
            with pytest.raises(UnicodeEncodeError):
                fn({name: b'x'})
