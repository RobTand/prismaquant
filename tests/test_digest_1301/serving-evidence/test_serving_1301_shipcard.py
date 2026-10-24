"""Inspection-only contracts against independently frozen original consumers."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from prismaquant import shipcard

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/serving-evidence'
spec = importlib.util.spec_from_file_location('shipcard_original', FIXTURES / 'shipcard_original.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def test_executable_call_accounting():
    """Count Call nodes, never identifiers in parameters, comments or strings."""
    tree = ast.parse((ROOT / 'prismaquant/shipcard.py').read_text())
    calls = []
    for fn in tree.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Function arguments and annotations are deliberately outside the walk.
        for stmt in fn.body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and (node.func.value.id, node.func.attr) in {('hashlib', 'sha256'), ('json', 'dumps')}:
                        calls.append((fn.name, ast.unparse(node)))
    assert sorted(calls) == sorted([
        ('_verify_open_safetensors_fd', 'hashlib.sha256()'),
        ('_verify_open_safetensors_fd', 'hashlib.sha256()'),
        ('_publish_shipcard', 'json.dumps(card, indent=2, default=str)'),
        ('_publish_shipcard', "json.dumps(card, separators=(',', ':'), default=str)"),
    ])
    ledger = json.loads((FIXTURES / 'ledger_shipcard.json').read_text())
    assert len(ledger['calls']) == len(calls) == 4
    assert len({r['id'] for r in ledger['calls']}) == 4
    for row in ledger['calls']:
        assert row['eligible'] and row['disposition'] == 'blocked-eligible'
        assert row['retention_justified'] is False and row['retention_evidence'] is None
        assert row['owner']['route'] is None and row['owner']['gap']


def publish_pair(tmp_path, card):
    results = []
    for name, fn in [('actual', shipcard._publish_shipcard), ('original', original.write_shipcard)]:
        path = tmp_path / name
        path.write_bytes(b'previous publication')
        try:
            result = fn(path, card)
        except Exception as exc:
            results.append((type(exc), str(exc), path.read_bytes()))
        else:
            assert result == path
            results.append((None, None, path.read_bytes()))
    assert results[0] == results[1]
    return results[0]


@pytest.mark.parametrize('mode', ['unreserved', 'pretty-padded', 'compact-exact', 'compact-padded'])
def test_publication_bytes(tmp_path, mode):
    # Unsorted keys, nested order, non-ASCII, lax nonfinite and default=str.
    card = {'z': [float('nan'), float('inf'), -float('inf')], 'a': {'y': 'é', 'b': Path('relative')}}
    if mode != 'unreserved':
        card['reserved_file_bytes'] = 1000
        if mode.startswith('compact'):
            for _ in range(5):
                compact = (json.dumps(card, separators=(',', ':'), default=str) + '\n').encode()
                card['reserved_file_bytes'] = len(compact) + (3 if mode.endswith('padded') else 0)
    error, _, data = publish_pair(tmp_path, card)
    assert error is None
    pretty = (json.dumps(card, indent=2, default=str) + '\n').encode()
    compact = (json.dumps(card, separators=(',', ':'), default=str) + '\n').encode()
    reserved = card.get('reserved_file_bytes')
    chosen = compact if mode.startswith('compact') else pretty
    assert data == chosen + (b' ' * (reserved - len(chosen)) if reserved else b'')
    assert data.startswith(b'{"z"' if mode.startswith('compact') else b'{\n  "z"')
    assert b'\\u00e9' in data and b'NaN' in data and b'relative' in data
    if reserved:
        assert len(data) == reserved


@pytest.mark.parametrize('reserved', [True, False, 0, -1, 2.5, '100', 1])
def test_publication_reservation_refusals(tmp_path, reserved):
    error, message, data = publish_pair(tmp_path, {'z': 'payload', 'reserved_file_bytes': reserved})
    assert error is ValueError
    assert ('positive integer' in message) if reserved != 1 or isinstance(reserved, bool) else ('fixed reservation' in message)
    assert data == b'previous publication'


@pytest.mark.parametrize('case', ['cycle', 'key', 'fallback'])
def test_publication_encoder_refusals(tmp_path, case):
    class BadString:
        def __str__(self):
            raise RuntimeError('fallback refused')
    card = {}
    if case == 'cycle':
        card['cycle'] = card
    elif case == 'key':
        card[('tuple',)] = 1
    else:
        card['object'] = BadString()
    error, _, data = publish_pair(tmp_path, card)
    assert error is {'cycle': ValueError, 'key': TypeError, 'fallback': RuntimeError}[case]
    assert data == b'previous publication'


def container(header, payload=b'abcde'):
    raw = header if isinstance(header, bytes) else json.dumps(header, separators=(',', ':')).encode()
    return struct.pack('<Q', len(raw)) + raw + payload


def valid_header():
    # Physical ordering differs from names; includes a zero-length tensor.
    return {'z': {'dtype': 'U8', 'shape': [3], 'data_offsets': [0, 3]},
            'a': {'dtype': 'U8', 'shape': [2], 'data_offsets': [3, 5]},
            'empty': {'dtype': 'U8', 'shape': [0], 'data_offsets': [5, 5]}}


def verify_pair(tmp_path, monkeypatch, raw, *, short=False, drift=False, truncate=False):
    path = tmp_path / 'weights.safetensors'
    path.write_bytes(raw)
    read, fstat, lseek = os.read, os.fstat, os.lseek
    outcomes, traces = [], []
    for module in [shipcard, original]:
        fd = os.open(path, os.O_RDONLY)
        initial = fstat(fd)
        trace = []
        def traced_read(got_fd, requested):
            assert got_fd == fd
            before = lseek(fd, 0, os.SEEK_CUR)
            data = read(fd, min(requested, 2) if short else requested)
            if truncate and before >= 8:
                data = b''
            trace.append((before, requested, data))
            return data
        def final_stat(got_fd):
            st = fstat(got_fd)
            if not drift:
                return st
            fields = {key: getattr(st, key) for key in ['st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns']}
            fields['st_mtime_ns'] += 1
            return SimpleNamespace(**fields)
        try:
            with monkeypatch.context() as m:
                m.setattr(module.os, 'read', traced_read)
                m.setattr(module.os, 'fstat', final_stat)
                # Any second traversal/seek by the implementation must fail.
                m.setattr(module.os, 'lseek', lambda *a: pytest.fail('implementation sought descriptor'))
                try:
                    result = module._verify_open_safetensors_fd(fd, name='weights', initial_stat=initial)
                except Exception as exc:
                    outcomes.append((type(exc), str(exc)))
                else:
                    outcomes.append(result)
        finally:
            os.close(fd)
        traces.append(trace)
    assert outcomes[0] == outcomes[1]
    assert traces[0] == traces[1]
    return outcomes[0], traces[0]


@pytest.mark.parametrize('short', [False, True])
def test_single_forward_full_and_tensor_hashes(tmp_path, monkeypatch, short):
    raw = container(valid_header())
    (record, count, calls), trace = verify_pair(tmp_path, monkeypatch, raw, short=short)
    assert record['sha256'] == hashlib.sha256(raw).hexdigest()
    assert record['tensor_sha256'] == {k: hashlib.sha256(v).hexdigest() for k, v in [('a', b'de'), ('empty', b''), ('z', b'abc')]}
    assert list(record['tensor_sha256']) == ['a', 'empty', 'z']
    assert count == len(raw) and calls == len(trace)
    assert b''.join(t[2] for t in trace) == raw
    cursor = 0
    for offset, request, data in trace:
        assert offset == cursor and 0 < request <= 1024 * 1024
        cursor += len(data)
    assert cursor == len(raw)


@pytest.mark.parametrize('case', ['duplicate', 'nonfinite', 'not-object', 'bad-metadata', 'unknown-dtype', 'bool-shape', 'span-size', 'gap', 'overlap', 'trailing', 'no-tensors', 'out-of-range', 'zero-prefix', 'huge-prefix', 'short-prefix', 'truncated'])
def test_header_span_refusals(tmp_path, monkeypatch, case):
    header = valid_header()
    if case == 'duplicate':
        header = b'{"x":1,"x":2}'
    elif case == 'nonfinite':
        header = b'{"x":NaN}'
    elif case == 'not-object':
        header = b'[]'
    elif case == 'bad-metadata':
        header['__metadata__'] = {'x': 1}
    elif case == 'unknown-dtype':
        header['z']['dtype'] = 'unknown'
    elif case == 'bool-shape':
        header['z']['shape'] = [True]
    elif case == 'span-size':
        header['z']['shape'] = [4]
    elif case == 'gap':
        header['a']['data_offsets'] = [4, 5]
        header['a']['shape'] = [1]
    elif case == 'overlap':
        header['a']['data_offsets'] = [2, 4]
    elif case == 'trailing':
        header.pop('a')
        header.pop('empty')
    elif case == 'no-tensors':
        header = {}
    elif case == 'out-of-range':
        header['a']['data_offsets'] = [3, 6]
    raw = container(header)
    if case == 'zero-prefix':
        raw = struct.pack('<Q', 0) + b'x'
    elif case == 'huge-prefix':
        raw = struct.pack('<Q', 2**63) + b'x'
    elif case == 'short-prefix':
        raw = b'123'
    outcome, trace = verify_pair(tmp_path, monkeypatch, raw, truncate=case == 'truncated')
    assert outcome[0] is ValueError
    assert trace


def test_megabyte_read_boundary(tmp_path, monkeypatch):
    payload = b'x' * (1024 * 1024 + 7)
    raw = container({'large': {'dtype': 'U8', 'shape': [len(payload)],
                               'data_offsets': [0, len(payload)]}}, payload)
    (record, count, calls), trace = verify_pair(tmp_path, monkeypatch, raw)
    assert record['sha256'] == hashlib.sha256(raw).hexdigest()
    assert record['tensor_sha256'] == {'large': hashlib.sha256(payload).hexdigest()}
    assert [t[1] for t in trace][-2:] == [1024 * 1024, 7]
    assert b''.join(t[2] for t in trace) == raw
    assert count == len(raw) and calls == 4


def test_stat_drift_refuses(tmp_path, monkeypatch):
    outcome, trace = verify_pair(tmp_path, monkeypatch, container(valid_header()), drift=True)
    assert outcome == (ValueError, 'weights: file stat changed during content verification')
    assert b''.join(t[2] for t in trace) == container(valid_header())
