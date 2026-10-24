"""Consumer-level compatibility against small, frozen pre-edit source excerpts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace

import pytest

from prismaquant import dev_mode, io_engine, io_spans

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/orchestration-runtime/runtime_io'


def original(filename, name, module):
    namespace = dict(vars(module))
    namespace.update(hashlib=hashlib, bytes_sha256hex=lambda b: hashlib.sha256(b).hexdigest())
    exec(compile((FIXTURES / filename).read_text(), filename, 'exec'), namespace)
    return namespace[name]


class Odd:
    def __repr__(self):
        return 'representation-é' * 30

    def __str__(self):
        return 'string-only'


@pytest.mark.parametrize('case', ['short', 'unicode', 'object', 'mixed', 'cycle', 'surrogate'])
def test_shown_consumer(case):
    cycle = []; cycle.append(cycle)
    value = {'short': 'ok', 'unicode': 'é' * 200, 'object': Odd(),
             'mixed': {1: 'a' * 200, 'b': 'c'}, 'cycle': [cycle, 'x' * 200],
             'surrogate': '\ud800' * 200}[case]
    old = original('dev_mode__shown.py', '_shown', dev_mode)
    assert dev_mode._shown(value) == old(value)
    if case == 'object':
        expected = hashlib.sha256(json.dumps(value, sort_keys=True, default=repr).encode()).hexdigest()[:16]
        assert expected in dev_mode._shown(value)
        assert expected != hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def test_seal_warning_and_refusal(monkeypatch, capsys):
    left, right = 'é' * 200, 'x' * 200
    assert dev_mode.seal_check('identity', left, right, where='runtime', environ={}) is False
    current = capsys.readouterr().out
    monkeypatch.setattr(dev_mode, '_shown', original('dev_mode__shown.py', '_shown', dev_mode))
    assert dev_mode.seal_check('identity', left, right, where='runtime', environ={}) is False
    assert capsys.readouterr().out == current
    assert current.startswith('[DEV-MODE] seal identity differs')
    with pytest.raises(RuntimeError, match='runtime: identity differs'):
        dev_mode.seal_check('identity', left, right, where='runtime', environ={'PRISMAQUANT_DEV_MODE': '0'})
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('payload', [b'', b'\x00\xffbinary', '雪é'.encode()])
def test_sealed_buffer_consumer(payload):
    old = original('io_engine_SealedBuffer_seal.py', 'seal', io_engine)
    digests = []
    for seal in (old, io_engine.SealedBuffer.seal):
        buffer = io_engine.SealedBuffer(len(payload))
        fd = buffer._fd
        if payload:
            buffer._map[:] = payload
        try:
            digests.append(seal(buffer))
            assert buffer._map is None
            assert os.pread(fd, len(payload), 0) == payload
            assert io_engine.fcntl.fcntl(fd, io_engine._F_GET_SEALS) & io_engine._SEALS == io_engine._SEALS
            with pytest.raises(OSError):
                os.ftruncate(fd, len(payload) + 1)
        finally:
            buffer.close()
        with pytest.raises(OSError):
            os.fstat(fd)
    assert digests == [hashlib.sha256(payload).hexdigest()] * 2


@pytest.mark.parametrize('payload', [b'', b'\x00\xffbinary', '雪é'.encode()])
def test_seal_lifecycle_order_and_owner_feed(monkeypatch, payload):
    """Independent trace: feed, release, map close, kernel seals, return."""
    old = original('io_engine_SealedBuffer_seal.py', 'seal', io_engine)
    traces = []
    for seal in (old, io_engine.SealedBuffer.seal):
        events = []
        feeds = []

        class Mapping(bytearray):
            def close(self):
                # A live exported view would make this resize raise BufferError.
                self.append(0)
                self.pop()
                events.extend(['release', 'map.close'])

        mapping = Mapping(payload) if payload else None
        buffer = SimpleNamespace(_map=mapping, _fd=123)

        def add_seals(fd, operation, seals):
            assert (fd, operation, seals) == (123, io_engine._F_ADD_SEALS, io_engine._SEALS)
            assert buffer._map is None
            assert mapping is None or bytes(mapping) == payload
            events.append('kernel.seals')

        def owner(value):
            assert isinstance(value, memoryview) if payload else value == b''
            feeds.append(bytes(value))
            events.append('hash')
            return hashlib.sha256(value).hexdigest()

        class OriginalHash:
            def __init__(self):
                self.digest = hashlib.sha256()
                events.append('hash')

            def update(self, value):
                assert isinstance(value, memoryview)
                assert bytes(value) == payload
                self.digest.update(value)

            def hexdigest(self):
                return self.digest.hexdigest()

        with monkeypatch.context() as patch:
            patch.setattr(io_engine.fcntl, 'fcntl', add_seals)
            patch.setattr(io_engine, 'bytes_sha256hex', owner)
            # Only the frozen method uses this private execution namespace.
            old.__globals__['hashlib'] = SimpleNamespace(sha256=OriginalHash)
            result = seal(buffer)
        events.append('return')
        assert result == hashlib.sha256(payload).hexdigest()
        assert bytes(mapping) == payload if mapping is not None else payload == b''
        if seal is io_engine.SealedBuffer.seal:
            assert feeds == [payload]  # Exactly one owner call, including empty input.
        traces.append(events)
    expected = ['hash'] + (['release', 'map.close'] if payload else []) + ['kernel.seals', 'return']
    assert traces == [expected, expected]


def test_seal_releases_view_on_failure(monkeypatch):
    buffer = io_engine.SealedBuffer(3)
    buffer._map[:] = b'abc'
    seen = []
    def fail(value):
        if isinstance(value, memoryview):
            seen.append(value)
            raise RuntimeError('digest failure')
        return hashlib.sha256(value).hexdigest()
    monkeypatch.setattr(io_engine, 'bytes_sha256hex', fail)
    try:
        with pytest.raises(RuntimeError, match='digest failure'):
            buffer.seal()
        assert buffer._map is not None
        with pytest.raises(ValueError, match='released memoryview'):
            seen[0].tobytes()
        # Closing must not fail with an exported-pointer BufferError.
        buffer.close()
    finally:
        buffer.close()


def rate_trace(emit, mode):
    lines = []
    attempts = []
    def log(line):
        attempts.append(line)
        if mode == 'logger_failure' and line.startswith(io_spans.READ_RATE_MARKER + ' {'):
            raise OSError('log down')
        lines.append(line)
    reporter = io_spans.ReadRateReporter('雪', total_entries=10, total_bytes=100,
                                        proc_io=lambda: {}, log=log, clock=lambda: 0.0)
    reporter._emit = MethodType(emit, reporter)
    reporter._last_clock = 2.0
    reporter.bytes = 50
    reporter.entries = 5
    now = float('nan') if mode == 'nan' else 3.0
    reporter._emit(now, final=False)
    return lines, attempts, reporter.lines, reporter._last_bytes, reporter._last_clock


@pytest.mark.parametrize('mode', ['normal', 'nan', 'logger_failure'])
def test_rate_emit_consumer(mode):
    old = original('io_spans_ReadRateReporter__emit.py', '_emit', io_spans)
    current, previous = rate_trace(io_spans.ReadRateReporter._emit, mode), rate_trace(old, mode)
    assert current[:4] == previous[:4]
    assert (current[4] != current[4]) if mode == 'nan' else current[4] == previous[4]
    lines, attempts, count, last, _ = current
    assert last == 50
    if mode == 'normal':
        assert count == 1
        assert lines[0].startswith(io_spans.READ_RATE_MARKER + ' {')
        assert '\\u96ea' in lines[0]
        assert lines[0].encode() == previous[0][0].encode()
        record = json.loads(lines[0].split(' ', 1)[1])
        assert record['final'] is False
        from prismaquant.digests import DIRECT_ASCII_SPACED_STRICT
        assert lines[0].encode() == (io_spans.READ_RATE_MARKER + ' ').encode() + DIRECT_ASCII_SPACED_STRICT.encoded(record)
    else:
        assert count == 0
        assert lines[-1].startswith(io_spans.READ_RATE_MARKER + ' 雪: rate line failed: ')
        assert ('ValueError' if mode == 'nan' else 'OSError') in lines[-1]


def span_trace(close, mode):
    lines = []
    def log(line):
        if mode == 'logger_failure' and line.startswith(io_spans.IO_SPAN_MARKER + ' {'):
            raise OSError('log down')
        lines.append(line)
    spans = io_spans.IoSpanLog(scope='雪', proc_io=lambda: {}, log=log,
                              clock=lambda: 3.0, wall=lambda: 9.0)
    span = spans.open('operation', detail=Odd(), bad=float('nan') if mode == 'nan' else 1)
    record = close(spans, span, error=ValueError('workload') if mode == 'error' else None)
    assert close(spans, span) is None
    # Normalize the custom attr only after encoding has exercised default=str.
    if record is not None:
        record['detail'] = str(record['detail'])
        if mode == 'nan':
            assert record['bad'] != record['bad']
            record['bad'] = 'nan'
    return lines, record, spans.open_names


@pytest.mark.parametrize('mode', ['normal', 'nan', 'logger_failure', 'error'])
def test_span_consumer(mode):
    old = original('io_spans_IoSpanLog_close.py', 'close', io_spans)
    current, previous = span_trace(io_spans.IoSpanLog.close, mode), span_trace(old, mode)
    assert current == previous
    lines, record, opened = current
    assert opened == []
    if mode in ('normal', 'error'):
        assert lines[0].startswith(io_spans.IO_SPAN_MARKER + ' {')
        assert 'string-only' in lines[0] and 'representation' not in lines[0]
        assert '\\u96ea' in lines[0]
        assert lines[0].encode() == previous[0][0].encode()
    else:
        assert 'accounting for span' in lines[-1] and 'failed:' in lines[-1]
        assert ('ValueError' if mode == 'nan' else 'OSError') in lines[-1]
    if mode == 'error':
        assert record['outcome'] == 'error'


def test_file_spec_deployment():
    # Exercise the actual bootstrap while making all package imports impossible.
    code = '''import importlib.abc, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
class NoPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'prismaquant' or fullname.startswith('prismaquant.'):
            raise RuntimeError('package import forbidden: ' + fullname)
sys.meta_path.insert(0, NoPackage())
from pq_profile_source import profile_source_owner
m = profile_source_owner('io_spans')
assert m is profile_source_owner('io_spans')
lines = []
r = m.ReadRateReporter('standalone', total_entries=1, total_bytes=1, proc_io=lambda: {}, log=lines.append, clock=lambda: 0.0)
r._emit(1.0, final=False)
s = m.IoSpanLog(scope='standalone', proc_io=lambda: {}, log=lines.append)
s.close(s.open('operation'))
assert any(x.startswith(m.READ_RATE_MARKER + ' {') for x in lines)
assert any(x.startswith(m.IO_SPAN_MARKER + ' {') for x in lines)
assert not any(x == 'prismaquant' or x.startswith('prismaquant.') for x in sys.modules)
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(ROOT / 'tools')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_ledger_witnesses():
    ledger = json.loads((FIXTURES / 'ledger.json').read_text())
    frozen = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
    files = {'prismaquant/dev_mode.py', 'prismaquant/io_engine.py', 'prismaquant/io_spans.py'}
    originals = {call['id']: call for call in frozen['calls'] if call['file'] in files}
    assert len(originals) == 4
    assert len(ledger['calls']) == 4
    assert {call['original_id'] for call in ledger['calls']} == set(originals)
    assert len(ledger['raw_original_membership']) == len(originals)
    assert set(ledger['raw_original_membership']) == set(originals)
    assert ledger['source_commit'] == frozen['source_commit'] == '6d8288adc94b3c4b624c143ed372fb6513f32784'
    for call in ledger['calls']:
        source = originals[call['original_id']]
        assert call['file'] == source['file']
        assert call['source_commit'] == frozen['source_commit']
        assert call['source_blob'] == source['source_blob']
        assert call['syntax_sha256'] == source['syntax_sha256']
        assert call['tests'] and call['witnesses']
        for node in call['tests']:
            path, name = node.split('::')
            assert path == 'tests/test_digest_1301/orchestration-runtime/test_runtime_io.py'
            assert name.startswith('test_') and '[' not in name
            assert callable(globals()[name])
        assert len(call['fixtures']) == 1
        fixture = call['fixtures'][0]
        assert not Path(fixture).is_absolute()
        assert hashlib.sha256((ROOT / fixture).read_bytes()).hexdigest() == call['fixture_sha256']
        data = (ROOT / call['file']).read_bytes()
        assert call['current_blob'] == hashlib.sha1(
            b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()
    dispositions = {call['original_id']: call['status'] for call in ledger['calls']}
    assert dispositions['sorted-json:prismaquant/dev_mode.py::_shown@130:18'] == 'blocked'
    assert dispositions['hashlib:prismaquant/io_engine.py::SealedBuffer.seal@261:17'] == 'replaced'
    assert all(dispositions[key] == 'retained' for key in originals if 'io_spans.py' in key)
