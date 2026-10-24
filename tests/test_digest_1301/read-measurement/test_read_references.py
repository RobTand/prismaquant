"""CPU operation witnesses, not throughput evidence, for retained benchmark arms."""
import ast
import hashlib
import importlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / 'tests/fixtures/digest_1301/read-measurement/read_references_source.json'
ORIGINAL = json.loads(FIXTURE.read_text())


def old_function(module, name):
    record = ORIGINAL['files'][f'tools/{module.__name__.split(".")[-1]}.py']
    namespace = dict(vars(module))
    exec(record['functions'][name], namespace)
    return namespace[name]


def test_wire_hash_caller_owned_readinto_trace(monkeypatch):
    module = importlib.import_module('tools.measure_wire_rehash_readers')
    original = old_function(module, '_hash')
    real_sha256 = hashlib.sha256
    traces = []
    for fn in (original, module._hash):
        trace = []
        traces.append(trace)
        class Reader(io.BytesIO):
            def __enter__(self):
                trace.append('enter')
                return self
            def __exit__(self, *args):
                trace.append('close')
                return super().__exit__(*args)
            def readinto(self, view):
                assert isinstance(view, memoryview) and not view.readonly
                # One caller-owned buffer throughout, including the EOF read.
                trace.append(('readinto', len(view), id(view.obj)))
                return super().readinto(view[:min(len(view), 3)])
        class Digest:
            def __init__(self):
                self.real = real_sha256()
                trace.append('construct')
            def update(self, view):
                assert isinstance(view, memoryview)
                trace.append(('update', bytes(view), id(view.obj)))
                self.real.update(view)
            def hexdigest(self):
                pytest.fail('The measured arm intentionally returns bytes read, not a digest')
        def opened(path, mode, **kwargs):
            trace.append(('open', str(path), mode, kwargs))
            return Reader(b'abcdefg')
        with monkeypatch.context() as patch:
            patch.setattr(module, 'open', opened, raising=False)
            original.__globals__['open'] = opened
            patch.setattr(hashlib, 'sha256', Digest)
            assert fn(Path('wire')) == 7
        ids = {event[-1] for event in trace if isinstance(event, tuple) and event[0] in ('update', 'readinto')}
        assert len(ids) == 1
        traces[-1] = [event[:-1] if isinstance(event, tuple) and event[0] in ('update', 'readinto') else event for event in trace]
    assert traces[0] == traces[1]
    assert [e[1] for e in traces[1] if isinstance(e, tuple) and e[0] == 'update'] == [b'abc', b'def', b'g']
    assert [e[1] for e in traces[1] if isinstance(e, tuple) and e[0] == 'readinto'] == [16 << 20] * 4


def test_fence_read_digest_independent_reference(monkeypatch):
    module = importlib.import_module('tools.bench_fence_rehash')
    original = old_function(module, '_read_digest')
    real = hashlib.file_digest
    for fn in (original, module._read_digest):
        seen = []
        handle = io.BytesIO(b'\x00wire\xff')
        def opened(path, mode):
            assert (path, mode) == ('wire', 'rb')
            return handle
        def digest(stream, algorithm):
            assert stream is handle and not handle.closed
            seen.append((algorithm, stream.tell()))
            return real(stream, algorithm)
        with monkeypatch.context() as patch:
            patch.setattr(module, 'open', opened, raising=False)
            original.__globals__['open'] = opened
            patch.setattr(hashlib, 'file_digest', digest)
            assert fn('wire') == (hashlib.sha256(b'\x00wire\xff').hexdigest(), None)
        assert handle.closed
        assert seen == [('sha256', 0)]


def test_busy_consumer_work_and_release_order(monkeypatch):
    module = importlib.import_module('tools.bench_fence_rehash')
    original = old_function(module, '_busy_consumer')
    from prismaquant import io_engine
    real_sha256 = hashlib.sha256
    outcomes = []
    for fn in (original, module._busy_consumer):
        trace = []
        class Digest:
            def __init__(self, raw):
                trace.append(('cpu-sha256', raw))
                self.real = real_sha256(raw)
            def digest(self):
                trace.append('binary-result')
                return self.real.digest()
        class Stream:
            counters = dict(pool_width=2, peak_workers=2, consumer_work_s=0.2)
            def __enter__(self):
                trace.append('enter')
                return self
            def take(self, index):
                trace.append(('take', index))
                return (SimpleNamespace(value=f'digest-{index}'),)
            def release(self):
                trace.append('release')
            def __exit__(self, *args):
                trace.append('exit')
        def read_stream(entries, budget):
            assert budget.buffer_bytes == 1
            assert [e.size for e in entries] == [7, 11]
            assert [e.held_bytes for e in entries] == [0, 0]
            assert all(e.expected_sha256 is None and e.path is None for e in entries)
            trace.append('stream')
            return Stream()
        clock = iter((10.0, 12.0))
        wires = [('one', {'bytes': 7}, 'digest-0'), ('two', {'bytes': 11}, 'digest-1')]
        report = {'runs': []}
        with monkeypatch.context() as patch:
            patch.setattr(io_engine, 'read_stream', read_stream)
            patch.setattr(hashlib, 'sha256', Digest)
            patch.setattr(module.time, 'perf_counter', lambda: next(clock))
            patch.setattr(module, '_fadvise_drop', lambda path: trace.append(('drop', path)))
            original.__globals__['_fadvise_drop'] = module._fadvise_drop
            fn(report, wires, 18, SimpleNamespace(repeats=1, cold=True))
        outcomes.append((report, trace))
    assert outcomes[0] == outcomes[1]
    assert outcomes[1][1] == [('drop', 'one'), ('drop', 'two'), 'stream', 'enter',
        ('take', 0), ('cpu-sha256', bytes(7)), 'binary-result',
        ('take', 1), ('cpu-sha256', bytes(7)), 'binary-result', 'release', 'exit']


def test_wire_report_actual_main_matches_original(tmp_path, monkeypatch, capsys):
    module = importlib.import_module('tools.measure_wire_rehash_readers')
    original = old_function(module, 'main')
    files = [tmp_path / 'é-first', tmp_path / 'next']
    for index, path in enumerate(files):
        path.write_bytes(bytes(index + 1))
    output = tmp_path / 'report.json'
    results = []
    for fn in (original, module.main):
        # Distinct slices per arm and a deterministic observation clock.
        clock = iter((10.0, 12.0, 15.0, 19.0))
        monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock))
        for namespace in (vars(module), original.__globals__):
            monkeypatch.setitem(namespace, '_candidates', lambda *a: iter(files))
            monkeypatch.setitem(namespace, '_proc_io', lambda: {'rchar': 100, 'read_bytes': 80})
            monkeypatch.setitem(namespace, '_nfs_read', lambda *a: [2, 0, 0, 0, 0, 3, 4, 5])
        assert fn(['--wire-dir', str(tmp_path), '--first-layer', '0', '--readers', '1,2',
                   '--bytes-per-arm', '1', '--out', str(output)]) == 0
        results.append((output.read_bytes(), capsys.readouterr().out))
    assert results[0] == results[1]
    assert len(results[1][1].splitlines()) == 2
    assert b'\\u00e9-first' in results[1][0]
    assert results[1][0].endswith(b'\n')


def test_in_memory_reference_executes_four_primitive_hashes():
    source = (ROOT / 'tools/bench_fence_rehash.py').read_text()
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.For)
                and ast.unparse(n.iter) == 'range(4)')
    original = ORIGINAL['files']['tools/bench_fence_rehash.py']['functions']['in_memory_reference']
    outcomes = []
    for block in (ast.parse(original), ast.Module(body=[loop], type_ignores=[])):
        trace = []
        class Digest:
            def hexdigest(self):
                trace.append('hex-result')
                return 'unused'
        def sha256(raw):
            trace.append(('hash', raw))
            return Digest()
        namespace = {'hashlib': SimpleNamespace(sha256=sha256), 'blob': b'read-once'}
        exec(compile(block, '<reference-loop>', 'exec'), namespace)
        outcomes.append(trace)
    assert outcomes[0] == outcomes[1] == [('hash', b'read-once'), 'hex-result'] * 4


def test_pair_file_digest_preserves_handle_contract(tmp_path, monkeypatch):
    module = importlib.import_module('tools.benchmarks.joint_validation_pair')
    path = tmp_path / 'payload'
    path.write_bytes(b'content')
    seen = []
    real = module.file_digest_sha256hex
    def observe(handle):
        seen.append(handle)
        assert handle.tell() == 0 and not handle.closed
        return real(handle)
    monkeypatch.setattr(module, 'file_digest_sha256hex', observe)
    assert module.sha(path) == hashlib.sha256(b'content').hexdigest()
    assert len(seen) == 1 and seen[0].closed
