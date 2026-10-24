"""Small independent native-JSON witnesses; no model/GPU qualification claim."""
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
LEDGER = json.loads((ROOT / 'tests/fixtures/digest_1301/quality-calibration/remaining/ledger.json').read_text())
ROWS = [row for row in LEDGER['calls'] if row['operation'].startswith('json.')]


@pytest.fixture(scope='module')
def owner():
    # Direct stdlib owner import: no package startup, torch, model or native ops.
    name = '_remaining_json_gap_owner'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'prismaquant/digests.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def native(row, value, stream=None):
    """Execute only the original call's options with an independent test value."""
    call = ast.parse(row['source'], mode='eval').body
    call.args = [ast.Name(id='value', ctx=ast.Load())]
    if row['operation'] == 'json.dump':
        call.args.append(ast.Name(id='stream', ctx=ast.Load()))
    ast.fix_missing_locations(call)
    return eval(compile(ast.Expression(call), row['id'], 'eval'),
                {'json': json, 'value': value, 'stream': stream})


@pytest.mark.parametrize('row', ROWS, ids=lambda row: row['id'])
def test_native_call_witness(row, owner):
    # Literal oracle, not the replacement owner or a generated reference copy.
    expected = '{"z": 1, "a": 2}'
    sorted_expected = '{"a": 2, "z": 1}'
    if row['effective_options']['indent'] == 2:
        expected = '{\n  "z": 1,\n  "a": 2\n}'
        sorted_expected = '{\n  "a": 2,\n  "z": 1\n}'
    if row['effective_options']['sort_keys']:
        expected = sorted_expected
    sink = io.StringIO()
    result = native(row, {'z': 1, 'a': 2}, sink)
    actual = sink.getvalue() if row['operation'] == 'json.dump' else result
    assert actual == expected == row['independent_reference']['native_literal']
    profile = getattr(owner, row['owner_candidate'])
    assert profile.text({'z': 1, 'a': 2}) == sorted_expected
    assert (actual != sorted_expected) == (row['disposition'] == 'eligible-owner-gap')


def test_original_source_coordinates_and_complete_population():
    actual = []
    for file in LEDGER['files']:
        if 'original_fixture' in file:
            # vLLM migrated independently. Its frozen main block retains the
            # original calls; never pin the migrated producer to a current hash.
            raw = (ROOT / file['original_fixture']).read_bytes()
            vllm = json.loads((ROOT / file['migration_ledger']).read_text())
            assert hashlib.sha256(raw).hexdigest() == vllm[0]['fixture_sha256']
            assert all(r['source_blob'] == file['original_blob'] for r in vllm)
            for row in LEDGER['calls']:
                if row['path'] == file['path'] and row['disposition'] == 'migrated':
                    migration = next(r for r in vllm
                                     if r['original_id'] == row['migration_original_id'])
                    assert migration['status'] == 'migrated'
            blob = file['original_blob']  # provenance, not a hash of the block
            line_offset = file['original_fixture_line_offset']
        else:
            # Other assigned sources remain unchanged, including in snapshots
            # that do not carry the historical Git DAG.
            raw = (ROOT / file['path']).read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            assert blob == file['original_blob']
            line_offset = 0
        source = raw.decode()
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in {'json', 'pickle'}
                    and node.func.attr in {'dump', 'dumps'}):
                continue
            actual.append((file['path'], node.lineno + line_offset, node.col_offset,
                           ast.get_source_segment(source, node), blob))
    recorded = [(r['path'], r['line'], r['column'], r['source'], r['source_blob']) for r in LEDGER['calls']]
    assert sorted(actual) == sorted(recorded)
    assert len(ROWS) == 15
    assert LEDGER['summary']['missing_unsorted_recipe'] == 13
    assert len(LEDGER['changed_file_source_bound_audit']['files']) == 12
    assert all(not row['scientific_retention'] for row in LEDGER['calls'])


def test_mixed_keys_default_str_and_diagnostic_prefix(owner):
    # Sorting changes acceptance, not just aesthetics, for mixed JSON keys.
    value = {'z': 1, 2: 'two'}
    assert json.dumps(value) == '{"z": 1, "2": "two"}'
    with pytest.raises(TypeError):
        owner.DIRECT_ASCII_SPACED_LAX.text(value)
    assert json.dumps({'z': Path('sample'), 'a': 'é'}, default=str) == '{"z": "sample", "a": "\\u00e9"}'
    assert json.dumps({'z': 'x' * 420, 'a': 2})[:400].startswith('{"z":')
    assert owner.DIRECT_ASCII_SPACED_LAX.text({'z': 'x' * 420, 'a': 2})[:400].startswith('{"a":')


@pytest.mark.parametrize('row', [r for r in ROWS if r['operation'] == 'json.dump'], ids=lambda row: row['id'])
def test_dump_partial_encoder_failure_is_not_text_then_write(row, owner):
    sink = io.StringIO()
    with pytest.raises(TypeError, match='not JSON serializable'):
        native(row, {'z': 1, 'bad': object()}, sink)
    assert sink.getvalue() == '{\n  "z": 1,\n  "bad": '
    replacement = io.StringIO()
    with pytest.raises(TypeError, match='not JSON serializable'):
        replacement.write(getattr(owner, row['owner_candidate']).text({'z': 1, 'bad': object()}))
    assert replacement.getvalue() == ''


@pytest.mark.parametrize('row', [r for r in ROWS if r['operation'] == 'json.dump'], ids=lambda row: row['id'])
def test_dump_sink_failure_precedes_later_encoding_error(row):
    class Sink:
        def __init__(self):
            self.calls = []

        def write(self, chunk):
            self.calls.append(chunk)
            if len(self.calls) == 2:
                raise OSError('injected sink failure')

    sink = Sink()
    with pytest.raises(OSError, match='injected sink failure'):
        native(row, {'z': 1, 'bad': object()}, sink)
    assert sink.calls == ['{', '\n  ']
