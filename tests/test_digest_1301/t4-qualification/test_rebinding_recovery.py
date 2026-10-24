"""Behavioral parity against committed, independent research-base functions."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/t4-qualification'


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modules(kind):
    tool = 'rebind_t4_qualified_results' if kind == 'rebind' else 'build_t4_recovery_request'
    original = load(FIXTURES / f'rebinding_recovery_original_{kind}.py')
    # This pre-existing identity helper is outside the migrated call sites.
    original.cell_task_id = lambda cell: digest(compact([cell['qname'], cell['format']]))
    return original, load(ROOT / 'tools' / f'{tool}.py')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compact(value))
    return {'path': str(path), 'sha256': digest(path.read_bytes())}


def invoke(module, args, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', [module.__name__, *args])
    module.main()
    return capsys.readouterr().out


def test_aliases_and_lax_recipes():
    from prismaquant.digests import DIRECT_ASCII_LAX, bytes_sha256hex
    for kind in ('rebind', 'recovery'):
        old, new = modules(kind)
        assert new.sha is bytes_sha256hex if kind == 'rebind' else (new.sha.__module__ == 'digests' and new.sha.__name__ == 'bytes_sha256hex')
        for raw in (b'', b'\x00\xff\r\n', 'café'.encode()):
            assert new.sha(raw) == old.sha(raw)
    old, new = modules('rebind')
    assert new.cell_sha256 == DIRECT_ASCII_LAX.sha256
    for value in ({'z': '雪', 'a': [None, True, -0.0]}, {'x': float('nan')}, {'x': float('inf')}):
        assert new.cell_sha256(value) == old.cell_sha256(value)


def test_rebind_row_and_refusals(tmp_path):
    old, new = modules('rebind')
    previous = {'qname': '雪', 'format': 'q4', 'anchor': 'before', 'extra': [1, None]}
    current = {**previous, 'anchor': 'after'}
    verified = {'unicode': 'é', 'nested': [False]}
    value = {'qname': '雪', 'format': 'q4', 'cell_sha256': digest(compact(previous)),
             'verified_cell': verified, 'verified_cell_sha256': digest(compact(verified))}
    raw = compact(value)
    expected = old.rebind_cell(current, previous, raw)
    assert new.rebind_cell(current, previous, raw) == expected
    for module in (old, new):
        assert module.require_rebound(current, raw, expected) == value
        pilot = {k: v for k, v in value.items() if k != 'verified_cell_sha256'}
        assert module.rebind_cell(current, previous, compact(pilot))['result_sha256'] == digest(compact(pilot))
        for bad_cell in ({**current, 'extra': []}, {**current, 'new': 1}, {**current, 'qname': 'other'}):
            with pytest.raises(AssertionError):
                module.rebind_cell(bad_cell, previous, raw)
        for bad_value in ({**value, 'cell_sha256': 'bad'}, {**value, 'verified_cell_sha256': 'bad'}, {**value, 'format': 'other'}):
            with pytest.raises(AssertionError):
                module.rebind_cell(current, previous, compact(bad_value))
        with pytest.raises(AssertionError):
            module.require_rebound(current, raw + b' ', expected)
        with pytest.raises(AssertionError):
            module.require_rebound({**current, 'anchor': 'tampered'}, raw, expected)
        path = module.result_path(tmp_path, '雪', 'q4', key='pair')
        path.write_bytes(raw)
        legacy = module.result_path(tmp_path, '雪')
        legacy.write_bytes(raw)
        with pytest.raises(ValueError, match='ambiguous'):
            module.result_path(tmp_path, '雪', 'q4')
        with pytest.raises(ValueError, match='unknown'):
            module.result_path(tmp_path, '雪', 'q4', key='invalid')
        assert module.result_path(tmp_path, '雪', 'q4', key='qname') == legacy
        assert module.bound_json(path, digest(raw)) == value
        with pytest.raises(AssertionError):
            module.bound_json(path, 'bad')
        path.unlink()
        legacy.unlink()


def test_rebind_publication(tmp_path, monkeypatch, capsys):
    old, new = modules('rebind')
    prepared = put(tmp_path / 'prepared-old.json', {'plan_sha256': 'old', 'cache': 'é'})
    prepared_new = put(tmp_path / 'prepared-new.json', {'plan_sha256': 'new', 'cache': 'é'})
    previous_cell = {'qname': '雪', 'format': 'q4', 'anchor': 'before'}
    current_cell = {**previous_cell, 'anchor': 'after'}
    previous = put(tmp_path / 'previous.json', {'cells': [previous_cell], 'old_prepared': prepared, 'anchor_journal': 'old'})
    current = put(tmp_path / 'current.json', {'cells': [current_cell], 'old_prepared': prepared_new, 'anchor_journal': 'new'})
    qualified = tmp_path / 'qualified'
    qualified.mkdir()
    raw_result = compact({'qname': '雪', 'format': 'q4', 'cell_sha256': digest(compact(previous_cell))})
    (qualified / (digest(compact(['雪', 'q4'])) + '.json')).write_bytes(raw_result)
    out = tmp_path / 'nested' / 'rebound.json'
    args = ['--catalog', current['path'], '--catalog-sha256', current['sha256'],
            '--previous-catalog', previous['path'], '--previous-catalog-sha256', previous['sha256'],
            '--qualified-dir', str(qualified), '--out', str(out)]
    expected_stdout = invoke(old, args, monkeypatch, capsys)
    expected = out.read_bytes()
    out.unlink()
    assert invoke(new, args, monkeypatch, capsys) == expected_stdout
    assert out.read_bytes() == expected
    assert expected.endswith(b'\n') and b'\\u96ea' in expected
    assert json.loads(expected)['old_prepared_rebound']['differs_only_in'] == ['plan_sha256']
    for module in (old, new):
        with pytest.raises(AssertionError):
            invoke(module, args, monkeypatch, capsys)
        bad = put(tmp_path / 'bad-prepared.json', {'plan_sha256': 'new', 'cache': 'changed'})
        with pytest.raises(AssertionError, match='beyond'):
            module.rebound_prepared(prepared, bad)
    assert (qualified / (digest(compact(['雪', 'q4'])) + '.json')).read_bytes() == raw_result


def recovery_setup(tmp_path):
    pb = tmp_path / 'pb'
    parent = 'ab-parent'
    dep = pb / 'decompositions' / parent[:2] / parent
    put(dep / 'publication.json', {'child_action_keys': ['failed-child', 'healthy-child']})
    put(dep / 'plan.json', {'partitions': [['fresh', 'bound', 'missing'], ['healthy']]})
    failed = pb / 'queue' / 'failed'
    failed.mkdir(parents=True)
    (failed / 'failed-child.json').write_text('{}')
    (failed / 'ignored.txt').write_text('not a child')
    cell = {'qname': '雪', 'format': 'q4'}
    for kind in ('wire', 'render'):
        path = tmp_path / kind
        path.write_bytes(b'tiny PB input')
        stat = path.stat()
        cell[kind] = str(path)
        cell[kind + '_stat'] = dict(inode=stat.st_ino, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)
    tasks = []
    for name in ('fresh', 'bound', 'missing', 'healthy'):
        output = tmp_path / f'{name}.json'
        payload = {'cell': copy.deepcopy(cell), 'output': str(output), 'reads': ['wire', 'render']}
        if name in ('fresh', 'bound'):
            value = {'qname': cell['qname'], 'format': cell['format'], 'cell_sha256': digest(compact(cell)),
                     'verified_cell': {'é': [None, True]}, 'verified_cell_sha256': digest(compact({'é': [None, True]}))}
            binding = put(output, value)
            if name == 'bound':
                payload['existing_result_sha256'] = binding['sha256']
        tasks.append({'id': name, 'payload': payload, 'estimated_seconds': 10})
    request = put(tmp_path / 'request.json', {'roster': {'tasks': tasks}, 'common': {'cwd': 'old', 'timeout_s': 1}, 'extra': 'é'})
    return pb, parent, request


def adapt_pb(module, pb, monkeypatch):
    # Production's existing fixed PB roots are redirected only in the test.
    def path_adapter(value):
        if str(value) == '/mnt/shared/prismabuild-fleet/cas/decompositions':
            return pb / 'decompositions'
        return Path(value)
    monkeypatch.setattr(module, 'Path', path_adapter)
    monkeypatch.setattr(module, 'QUEUE', pb / 'queue')


def test_recovery_main_publications(tmp_path, monkeypatch, capsys):
    pb, parent, request = recovery_setup(tmp_path)
    old, new = modules('recovery')
    out = tmp_path / 'published' / 'request.json'
    args = ['--out', str(out), '--qualifier-checkout', str(tmp_path / 'checkout-é'),
            '--parent', parent, '--request', request['path'], '--request-sha256', request['sha256']]
    source_before = Path(request['path']).read_bytes()
    for all_members in (False, True):
        flags = args + (['--all-members'] if all_members else [])
        outputs = []
        for module in (old, new):
            adapt_pb(module, pb, monkeypatch)
            stdout = invoke(module, flags, monkeypatch, capsys)
            outputs.append((out.read_bytes(), out.with_suffix('.recovery.json').read_bytes(), stdout))
            result = json.loads(outputs[-1][0])
            assert len(result['roster']['tasks']) == (4 if all_members else 3)
            assert result['common']['timeout_s'] == 600
            for task in result['roster']['tasks'][:2]:
                assert task['payload']['reads'] == []
                assert task['estimated_seconds'] == .001
            assert result['roster']['tasks'][2]['payload']['reads'] == ['wire', 'render']
            proof = json.loads(outputs[-1][1])
            assert proof['reused_results'] == 2
            assert proof['request']['sha256'] == digest(outputs[-1][0])
            assert proof['publication']['sha256'] == digest((pb / 'decompositions' / parent[:2] / parent / 'publication.json').read_bytes())
            assert outputs[-1][0].endswith(b'\n') and outputs[-1][1].endswith(b'\n')
            with pytest.raises(AssertionError):
                invoke(module, flags, monkeypatch, capsys)
            out.unlink()
            out.with_suffix('.recovery.json').unlink()
        assert outputs[0] == outputs[1]
    assert Path(request['path']).read_bytes() == source_before


def test_recovery_main_refusals(tmp_path, monkeypatch, capsys):
    pb, parent, request = recovery_setup(tmp_path)
    original = Path(request['path']).read_bytes()
    out = tmp_path / 'refused.json'
    args = ['--out', str(out), '--qualifier-checkout', str(tmp_path / 'checkout'), '--parent', parent,
            '--request', request['path'], '--request-sha256', request['sha256']]
    for module in modules('recovery'):
        adapt_pb(module, pb, monkeypatch)
        with pytest.raises(AssertionError):
            invoke(module, args[:-1] + ['wrong'], monkeypatch, capsys)
        result_path = tmp_path / 'fresh.json'
        raw = result_path.read_bytes()
        for field in ('cell_sha256', 'verified_cell_sha256', 'qname', 'format'):
            result_path.write_bytes(compact({**json.loads(raw), field: 'bad'}))
            with pytest.raises(AssertionError):
                invoke(module, args, monkeypatch, capsys)
        result_path.write_bytes(raw)
        bound = tmp_path / 'bound.json'
        bound_raw = bound.read_bytes()
        bound.write_bytes(bound_raw + b' ')
        with pytest.raises(AssertionError):
            invoke(module, args, monkeypatch, capsys)
        bound.write_bytes(bound_raw)
        value = json.loads(original)
        value['roster']['tasks'][0]['payload']['cell']['wire_stat']['bytes'] += 1
        # Rebind the result too, to reach the physical-input stat check.
        result_path.write_bytes(compact({**json.loads(raw), 'cell_sha256': digest(compact(value['roster']['tasks'][0]['payload']['cell']))}))
        changed = put(Path(request['path']), value)
        with pytest.raises(AssertionError):
            invoke(module, args[:-1] + [changed['sha256']], monkeypatch, capsys)
        Path(request['path']).write_bytes(original)
        result_path.write_bytes(raw)
        assert not out.exists()
        assert not out.with_suffix('.recovery.json').exists()


def test_ledger_registry_and_test_nodes():
    registry = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
    ledger = json.loads((FIXTURES / 'rebinding_recovery_ledger.json').read_text())
    scope = {row['id']: row for row in registry['calls'] if row['file'] in (
        'tools/rebind_t4_qualified_results.py', 'tools/build_t4_recovery_request.py')}
    assert {row['original_id'] for row in ledger['calls']} == set(scope)
    nodes = {node.name for node in ast.parse(Path(__file__).read_text()).body if isinstance(node, ast.FunctionDef)}
    for row in ledger['calls']:
        frozen_row = scope[row['original_id']]
        assert row['source_commit'] == registry['source_commit']
        assert row['source_blob'] == frozen_row['source_blob']
        for reference in row['tests']:
            path, node = reference.split('::')
            assert ROOT / path == Path(__file__)
            assert node in nodes
        for path in row['fixtures']:
            assert (ROOT / path).is_file()
