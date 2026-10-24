"""Executable custody of unrouteable manifest consumers, not replacement encoders.

Original consumer ASTs are compiled without rewriting their bodies; current
consumers are imported and invoked directly. Campaign input dependencies are
controlled; serialization and filesystem publication execute the real consumer
code. Frozen expected bytes were captured separately
from the research-base sources, never from digests.py.
"""
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Mapping
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/stagea-custody'
BASE = '6d8288adc94b3c4b624c143ed372fb6513f32784'


def consumer(module, name, original=False, **bindings):
    if not original:
        from importlib import import_module
        fn = import_module(f'prismaquant.{module}')
        for part in name.split('.'):
            fn = getattr(fn, part)
        def invoke(*args, **kwargs):
            with patch.dict(fn.__globals__, bindings):
                return fn(*args, **kwargs)
        return invoke
    path = FIXTURES / f'manifest_original_{module}.py'
    tree = ast.parse(path.read_bytes())
    if '.' in name:
        cls, method = name.split('.')
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    else:
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(__name__='prismaquant._manifest_custody', __package__='prismaquant',
                 json=json, gzip=gzip, os=os, Mapping=Mapping, Path=Path,
                 bytes_sha256hex=lambda b: hashlib.sha256(b).hexdigest(),
                 file_sha256hex=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    scope.update(bindings)
    # Fixtures retain exact selected source, including the method's original
    # class indentation/header. Compile only that method with an explicit self
    # namespace rather than copying the unexercised class implementation.
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(ast.Module(body=[*imports, node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[node.name]


def seal_observation(original=False):
    fn = consumer('joint_layer_quanta', 'seal_manifest_bytes', original)
    value = {'z': '\u00e9\n', 'a': [float('nan'), -0.0, {'b': 2, 'a': 1}]}
    blob = fn(value)
    errors = []
    for invalid in ([], {'bad': object()}, {'bad': {(1, 2): 1}}):
        try:
            fn(invalid)
        except Exception as exc:
            errors.append([type(exc).__name__, str(exc), type(exc.__cause__).__name__])
    assert fn(value) == blob
    assert fn({'a': 1, 'z': 2}) != fn({'z': 2, 'a': 1})
    return {'gzip_hex': blob.hex(), 'decoded_hex': gzip.decompress(blob).hex(), 'errors': errors}


def row_observation(original=False, dry_run=False, bad_second=False, fail_publish=False):
    class Refused(RuntimeError):
        pass
    rows = [{'argv': ['row-0001'], 'z': '\u00e9'}, {'argv': ['row-0002'], 'z': 'two'}]
    Path('input.json').write_text(json.dumps(rows), encoding='utf-8')
    manifest = {'z': '\u00e9', 'entry_count': 1, 'total_bytes': 7,
                'annotations': {'counts': {'captures': 1, 'weight_extents': 0, 'seeds': 0},
                                'bytes': {'captures': 7, 'weight_extents': 0, 'seeds': 0}}}
    def build(*args):
        if bad_second and args[1] == 'row-0002':
            raise Refused('second row refused')
        return manifest
    fn = consumer('campaign_data_manifests', 'publish_row_manifests', original,
                  ReadsetRefused=Refused, load_sizes=lambda _: {},
                  CachedSizeCampaign=lambda *_: SimpleNamespace(rows={'row-0001': {}, 'row-0002': {}}),
                  deterministic_provenance=lambda *_: {'z': '\u00e9', 'a': 1},
                  row_id_of=lambda r: r['argv'][0], build_campaign_row_manifest=build)
    events = []
    replace = os.replace
    def publish(src, dst):
        # Observe all bytes at the actual atomic publication boundary.
        events.append([str(src), str(dst), Path(src).read_bytes().hex()])
        if fail_publish and str(dst) == 'rows.json':
            raise OSError('injected publication failure')
        replace(src, dst)
    Path('rows.json').write_bytes(b'previous rows\n')
    error = None
    summary = None
    with patch.object(os, 'replace', publish):
        try:
            summary = fn('input.json', 'workspace', out_dir='manifests',
                         out_manifest='rows.json', dry_run=dry_run)
        except Exception as exc:
            error = [type(exc).__name__, str(exc)]
    files = {str(p): p.read_bytes().hex() for p in sorted(Path('.').rglob('*')) if p.is_file()}
    return {'events': events, 'files': files, 'summary': summary, 'error': error}


def status_observation(original=False, readonly=False, no_directory=False, invalid=False, owner=True):
    from prismaquant import cost_stage_checkpoint as checkpoint
    fn = consumer('streamed_boundary_artifacts', 'StreamedBoundaryArtifacts._publish_status', original)
    policy = {'schema': 'prismaquant.aura.boundary_storage.v2', 'capture_order': 'layer_major',
              'directory': 'artifacts', 'prefetch_batches': 2, 'max_resident_bytes': 1024,
              'max_auxiliary_bytes': 2048, 'max_artifact_bytes': 4096}
    obj = SimpleNamespace(directory=None if no_directory else Path('artifacts'), _readonly=readonly,
                          config=policy, session='s\u00e9', identity=policy,
                          _status='working', telemetry={'z': float('nan') if invalid else 1, 'a': '\u00e9'},
                          _owner_label='split' if owner else None, _owner_fields={'boundary': 3},
                          _retained_debt=lambda: {'bytes': 11, 'entries': ['embedded-boundary']},
                          status_path=lambda: Path('artifacts/owners/split.json' if owner else 'artifacts/generation.json'))
    Path('artifacts/owners').mkdir(parents=True)
    obj.status_path().write_bytes(b'previous status\n')
    events = []
    replace = os.replace
    def publish(src, dst):
        events.append(['replace', Path(dst).as_posix(), Path(src).read_bytes().hex(), Path(dst).read_bytes().hex()])
        replace(src, dst)
    fsync = os.fsync
    def sync(fd):
        events.append(['fsync'])
        fsync(fd)
    error = None
    with patch.object(checkpoint, 'unique_temp_suffix', lambda: '.tmpcustody'), patch.object(os, 'replace', publish), patch.object(os, 'fsync', sync):
        try:
            fn(obj)
        except Exception as exc:
            error = [type(exc).__name__, str(exc)]
    return {'events': events, 'bytes_hex': obj.status_path().read_bytes().hex(), 'error': error,
            'temporary_exists': obj.status_path().with_name(obj.status_path().name + '.tmpcustody').exists()}


CASES = {
    'seal': (seal_observation, {}),
    'rows_publish': (row_observation, {}),
    'rows_dryrun': (row_observation, {'dry_run': True}),
    'rows_refusal': (row_observation, {'bad_second': True}),
    'rows_publish_failure': (row_observation, {'fail_publish': True}),
    'status_owner': (status_observation, {}),
    'status_generation': (status_observation, {'owner': False}),
    'status_readonly': (status_observation, {'readonly': True}),
    'status_no_directory': (status_observation, {'no_directory': True}),
    'status_refusal': (status_observation, {'invalid': True}),
}


@pytest.mark.parametrize('case', list(CASES))
def test_manifest_original_outcomes(case, tmp_path, monkeypatch):
    expected = json.loads((FIXTURES / 'manifest_original_outcomes.json').read_text())[case]
    fn, kwargs = CASES[case]
    # Both executions use fresh filesystem state and the same relative paths.
    for original in (True, False):
        directory = tmp_path / str(original)
        directory.mkdir()
        monkeypatch.chdir(directory)
        assert fn(original=original, **kwargs) == expected


def test_manifest_retention_witnesses():
    ledger = json.loads((FIXTURES / 'ledger_manifests.json').read_text())
    registry = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
    selected = {c['id']: c for c in registry['calls'] if c['file'] in ledger['source_witnesses']}
    assert ledger['source_commit'] == registry['source_commit'] == BASE
    assert len(ledger['completed_calls']) == 1
    assert {r['original_id'] for r in ledger['retentions'] + ledger['completed_calls']} == set(selected)
    for row in ledger['retentions'] + ledger['completed_calls']:
        assert row['source_blob'] == selected[row['original_id']]['source_blob']
        assert row['source_commit'] == BASE
        assert row['current_source_witness']['ast_node_id'] == row['original_id']
    assert all(row['owner_route'] is None for row in ledger['retentions'])
    assert ledger['completed_calls'][0]['owner_route'] == 'prismaquant.digests.indent2_json_file_bytes'
    for row in ledger['retentions']:
        witness = row['current_source_witness']
        line, col = map(int, row['original_id'].split('@')[1].split(':'))
        node = next(n for n in ast.walk(ast.parse((ROOT / witness['file']).read_bytes()))
                    if isinstance(n, ast.Call) and (n.lineno, n.col_offset) == (line, col))
        assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == witness['syntax_sha256']
    for filename, witness in ledger['source_witnesses'].items():
        current = (ROOT / filename).read_bytes()
        original = (FIXTURES / witness['fixture']).read_bytes()
        assert hashlib.sha256(current).hexdigest() == witness['current_sha256']
        # Whole-source commit/blob/SHA-256 remain provenance, not slice hashes.
        assert witness['source_commit'] == BASE
        assert witness['source_blob'] == next(
            row['source_blob'] for row in selected.values() if row['file'] == filename)
        assert len(witness['original_sha256']) == 64
        assert len(original) == witness['fixture_size_bytes'] < witness['original_size_bytes']
        assert hashlib.sha256(original).hexdigest() == witness['fixture_sha256']
        lines = original.splitlines(keepends=True)
        covered = set()
        for segment in witness['selected_original_source']:
            start, end = segment['fixture_start_line'], segment['fixture_end_line']
            chunk = b''.join(lines[start - 1:end])
            assert hashlib.sha256(chunk).hexdigest() == segment['sha256']
            assert len(chunk) == segment['source_end_byte'] - segment['source_start_byte']
            assert end - start == segment['source_end_line'] - segment['source_start_line']
            assert not covered.intersection(range(start, end + 1))
            covered.update(range(start, end + 1))
        assert all(not line.strip() for i, line in enumerate(lines, 1) if i not in covered)
        tree = ast.parse(original)
        for row in ledger['retentions'] + ledger['completed_calls']:
            if row['current_source_witness']['file'] != filename:
                continue
            source_line, col = map(int, row['original_id'].split('@')[1].split(':'))
            segment = next(s for s in witness['selected_original_source']
                           if s['source_start_line'] <= source_line <= s['source_end_line'])
            fixture_line = source_line - segment['source_start_line'] + segment['fixture_start_line']
            node = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and (n.lineno, n.col_offset) == (fixture_line, col))
            current_witness = row['current_source_witness']
            expected_syntax = current_witness.get('original_syntax_sha256', current_witness['syntax_sha256'])
            assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == expected_syntax
