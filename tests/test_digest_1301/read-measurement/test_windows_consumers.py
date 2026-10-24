"""Execute original SOURCE functions and current consumer ASTs with narrow adapters.

No campaign, GPU, or replacement expression is executed in lieu of a consumer.
The frozen fixture contains only original functions extracted from the base git blob.
"""
import ast
import contextlib
import hashlib
import io
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest

from test_windows import LEDGER, ROOT, load

FIXTURE = ROOT / 'tests/fixtures/digest_1301/read-measurement/windows_consumers.json'
ORIGINAL = json.loads(FIXTURE.read_text())


def test_ledger_consumer_mapping():
    assert ORIGINAL['source_commit'] == '6d8288adc94b3c4b624c143ed372fb6513f32784'
    assert len({r['original_id'] for r in LEDGER}) == len(LEDGER)
    for row in LEDGER:
        assert row['original_id'].endswith(f"@{row['line']}:{row['col']}")
        if row['status'] == 'blocked':
            assert row['blocker'] and row['owner_route'] is None
            continue
        assert row['status'] == 'migrated'
        assert row['owner_route'].startswith('prismaquant.digests.')
        original = next(r for r in ORIGINAL['functions']
                        if r['file'] == row['file'] and r['scope'] == row['scope'])
        tree = ast.parse(original['SOURCE'])
        assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
        for reference in row['tests']:
            file, name = reference.split('::')
            tests = ast.parse((ROOT / file).read_text())
            assert any(isinstance(n, ast.FunctionDef) and n.name == name for n in tests.body)
            assert name != 'test_original_call_behavior'


def consumer(file, name, old, trace, fail_hash=None, **adapters):
    module = load(file)
    if old:
        source = next(r['SOURCE'] for r in ORIGINAL['functions']
                      if r['file'] == file and r['scope'] == name)
        tree = ast.parse(source)
    else:
        node = next(n for n in ast.parse((ROOT / file).read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        tree = ast.Module(body=[node], type_ignores=[])
    count = 0

    def mark(data):
        nonlocal count
        count += 1
        trace.append(('hash', bytes(data)))
        if count == fail_hash:
            raise ValueError('hash adapter failure')
        return hashlib.sha256(data).hexdigest()

    class Sha:
        def __init__(self, data=b''):
            self.data = data
        def hexdigest(self):
            return mark(self.data)
        def update(self, data):
            trace.append(('update', bytes(data)))
            self.data += data

    namespace = dict(vars(module), hashlib=NS(sha256=Sha), bytes_sha256hex=mark)
    namespace.update(adapters)
    exec(compile(tree, file, 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('fail_read', [False, True])
def test_file_consumer_original(monkeypatch, fail_read):
    import builtins
    results = []
    for old in (True, False):
        trace = []
        class Reader(io.BytesIO):
            def read(self, size=-1):
                trace.append(('read', size))
                if fail_read:
                    raise OSError('reader failed')
                return super().read(size)
            def close(self):
                trace.append(('close',))
                super().close()
        def open_file(*args, **kw):
            trace.append(('open', args, kw))
            return Reader(b'file\\x00\\xff' * (1 << 19))
        with monkeypatch.context() as patch:
            patch.setattr(builtins, 'open', open_file)
            fn = consumer('tools/chain_roll_bench.py', '_sha256_file', old, [],
                          hashlib=hashlib)
            result = outcome(fn, 'unused')
        results.append((result, trace))
    assert results[0] == results[1]
    assert results[1][1][-1] == ('close',)


def install(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def outcome(fn, *args):
    try:
        return ('ok', fn(*args))
    except Exception as exc:
        return (type(exc).__name__, str(exc))


def clock():
    ticks = iter(range(1000))
    return NS(perf_counter=lambda: next(ticks), monotonic=lambda: next(ticks),
              time=lambda: next(ticks), thread_time=lambda: next(ticks))


@pytest.mark.parametrize('fail_hash', [None, 1])
def test_setup_write_hash_publish(monkeypatch, tmp_path, fail_hash):
    """The staged bytes exist at hash time; no publication follows hash failure."""
    root = tmp_path / 'fixture'
    source = tmp_path / 'one.pt'
    source.write_bytes(b'original\x00\xff')
    results = []
    for old in (True, False):
        shutil.rmtree(root, ignore_errors=True)
        trace = []
        stage = root / 'stage'
        def queue(*args):
            stage.mkdir()
            return None, stage
        def publish(*args):
            trace.append(('publish', (stage / source.name).read_bytes()))
        def write_map(*args, **kw):
            trace.append(('map',))
            return root / 'map'
        strict = install(monkeypatch, 'test_strict_reader_tier_enforcement',
                         _pb=lambda: (None, None, None), _hex64=lambda x: x,
                         _pb_queue=queue, _pb_publish=publish, _write_map=write_map,
                         MANIFEST={})
        install(monkeypatch, 'prismaquant.residency_map', residency_map_key=lambda *a: a)
        args = NS(fixture=str(root), record='unused', files=1,
                  fixture_json=str(root / 'fixture.json'))
        original_write = Path.write_bytes
        with monkeypatch.context() as patch:
            def write(path, blob):
                trace.append(('write', path.name, blob))
                return original_write(path, blob)
            patch.setattr(Path, 'write_bytes', write)
            fn = consumer('tools/pwc_window_load_bench.py', 'run_setup', old, trace,
                          fail_hash, time=clock(), _row_041_renders=lambda *a: [str(source)],
                          _log=lambda text: trace.append(('log', text)))
            result = outcome(fn, args)
        output = json.loads(Path(args.fixture_json).read_text()) if Path(args.fixture_json).exists() else None
        results.append((result, trace, output))
    assert results[0] == results[1]
    events = [e[0] for e in results[1][1]]
    assert events[:2] == ['write', 'hash']
    assert ('publish' in events) == (fail_hash is None)
    if fail_hash is None:
        assert events.index('hash') < events.index('publish') < events.index('map')


@pytest.mark.parametrize('bad_metadata', [False, True])
def test_window_metadata_before_tensor(monkeypatch, tmp_path, bad_metadata):
    results = []
    for old in (True, False):
        trace = []
        class Tensor:
            shape = (2, 3)
            dtype = 'uint8'
            def contiguous(self):
                trace.append(('tensor',))
                return self
            def view(self, dtype): return self
            def numpy(self): return self
            def tobytes(self): return b'\x00tensor\xff'
        tensor = Tensor()
        class Cache:
            def __init__(self, **kw): pass
            def enable_lru(self, total): pass
            def require_file_load_sha256(self, *a, **kw): pass
            @contextlib.contextmanager
            def retained_window(self, *a, **kw):
                trace.append(('enter',))
                try: yield {'load_quanta': [1]}
                finally: trace.append(('exit',))
            def get_resident(self, *a): return tensor
            def file_load_receipt(self, *a):
                trace.append(('receipt',))
                return {'path': 'omit', 'size': object() if bad_metadata else 9}
            def resident_render_identity(self, *a):
                trace.append(('identity',))
                return {'identity': 'fixed'}
        install(monkeypatch, 'torch', uint8='uint8')
        cache_module = install(monkeypatch, 'prismaquant.production_weight_cache', ProductionWeightCache=Cache)
        import prismaquant
        monkeypatch.setattr(prismaquant, 'production_weight_cache', cache_module, raising=False)
        install(monkeypatch, 'prismaquant.residency_map', residency_resolver=lambda: NS(report=lambda: {}))
        fixture = tmp_path / 'fixture.json'
        fixture.write_text(json.dumps({'declared': ['module__fmt.pt'], 'staged': [],
                                       'file_bytes': {'module__fmt.pt': 9},
                                       'sha256': {'module__fmt.pt': 'digest'}}))
        output = tmp_path / 'window.json'
        output.unlink(missing_ok=True)
        args = NS(fixture_json=str(fixture), consumer='cpu', windows=2, workers=1,
                  load_buffer_bytes=100, compute_s=0, out_json=str(output))
        fn = consumer('tools/pwc_window_load_bench.py', 'run_window', old, trace,
                      time=clock(), _attach_strict=lambda f: None,
                      _drop_client_pages=lambda p: None, _consume_sleep=lambda s: trace.append(('consume',)),
                      _log=lambda s: None)
        result = outcome(fn, args)
        results.append((result, trace, output.read_bytes() if output.exists() else None))
    assert results[0] == results[1]
    trace = results[1][1]
    events = [e[0] for e in trace]
    if bad_metadata:
        assert results[1][0][0] == 'TypeError'
        assert 'tensor' not in events and events[-1] == 'exit'
    else:
        assert events[:5] == ['enter', 'receipt', 'identity', 'update', 'tensor']
        assert json.loads(trace[3][1])[3] == {'size': 9}
        assert events.count('tensor') == 1
        assert trace[5] == ('update', b'\x00tensor\xff')


@pytest.mark.parametrize('fail_hash', [None, 1, 2, 3, 4, 5, 6])
def test_quantum_pinned_files_summary(monkeypatch, tmp_path, fail_hash):
    results = []
    for old in (True, False):
        trace = []
        scratch = tmp_path / 'scratch'
        out = tmp_path / 'summary.json'
        out.unlink(missing_ok=True)
        package = sys.modules.get('prismaquant')
        if package is None:
            import prismaquant as package
        module = install(monkeypatch, 'prismaquant.windows_test_producer', producer_package_sha256=lambda: 'unpatched')
        def campaign(path, patch):
            trace.append(('campaign', module.producer_package_sha256()))
            root = path / 'outputs'
            root.mkdir()
            (root / 'a.json').write_text('{"z": 3}')
            (root / 'b.pkl').write_bytes(pickle.dumps({'shared': [1, 2]}))
            (root / 'c.bin').write_bytes(b'\x00\xffbinary')
            (root / 'directory').mkdir()
            return 'single', 'receipt', root
        def quantum(scratch, patch, **kw):
            trace.append(('quantum', kw['single'], kw['receipt'], kw['plan_sha'], kw['prepared_sha']))
            return {'costs': [1, 2], 'value': 'é'}, {'output_space': {'root': str(kw['output_root'])}}, {}
        cache_module = install(monkeypatch, 'prismaquant.production_weight_cache')
        monkeypatch.setattr(package, 'production_weight_cache', cache_module, raising=False)
        install(monkeypatch, 'test_quantum_probe_identity_once_1183', _campaign=campaign)
        install(monkeypatch, 'test_joint_cost_quantum_runtime', _run_quantum=quantum, _hex=lambda x: x * 64)
        def leaves(obj, prefix, target=None):
            trace.append(('leaves', prefix, obj))
            target = {} if target is None else target
            target[prefix] = repr(obj)
            return target
        fn = consumer('tools/pwc_window_load_bench.py', 'run_quantum', old, trace,
                      fail_hash, _leaves=leaves)
        result = outcome(fn, NS(scratch=str(scratch), out_json=str(out)))
        results.append((result, trace, out.read_bytes() if out.exists() else None))
        assert module.producer_package_sha256() == 'unpatched'
    assert results[0] == results[1]
    if fail_hash is None:
        summary = json.loads(results[1][2])
        assert summary['files'] == {name: hashlib.sha256((scratch / 'outputs' / name).read_bytes()).hexdigest()
                                    for name in ('a.json', 'b.pkl', 'c.bin')}
        assert summary['payload_sha256'] == hashlib.sha256(pickle.dumps({'costs': [1, 2], 'value': 'é'})).hexdigest()
        assert summary['leaves']['file:c.bin'] == summary['files']['c.bin']
        assert results[1][1][1] == ('campaign', hashlib.sha256(b'pwc-window-load-bench source').hexdigest())
    else:
        assert results[1][0] == ('ValueError', 'hash adapter failure')
        assert results[1][2] is None


def test_renders_real_files(tmp_path):
    pytest.importorskip('torch')
    root = tmp_path / 'renders'
    results = []
    for old in (True, False):
        shutil.rmtree(root, ignore_errors=True)
        trace = []
        fn = consumer('tools/render_window_bench.py', '_renders', old, trace)
        manifest = fn(NS(units=2, dim=3), root)
        files = {str(p.relative_to(root)): p.read_bytes() for p in root.iterdir()}
        for row in manifest.values():
            assert row['sha256'] == hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
        assert len(list(root.glob('*.source.pt'))) == 2
        results.append((manifest, files, trace))
    assert results[0] == results[1]


@pytest.mark.parametrize('timed,fail_hash', [(False, None), (True, None), (True, 1)])
def test_commit_unit_pickle_framing(monkeypatch, tmp_path, timed, fail_hash):
    results = []
    for old in (True, False):
        trace = []
        shared = ['é', b'\x00\xff']
        state = {'first': shared, 'second': shared, 'large': b'x' * 70000}
        class Measurements:
            def __init__(self, **kw): pass
        class Price:
            def __init__(self, *a, **kw): pass
            def prepare_rows(self, names): trace.append(('prepare', names))
            def captured_state(self, name):
                trace.append(('state', name))
                return state
        class Writer:
            def __init__(self, root, identity): pass
            def publish(self, **kw): trace.append(('publish', kw))
        install(monkeypatch, 'prismaquant.aura_checkpoint_journal', UnitCheckpointWriter=Writer,
                AURA_CHECKPOINT_UNIT_SCHEMA='schema')
        install(monkeypatch, 'prismaquant.joint_price_step', AuraUnitMeasurements=Measurements,
                JointPriceStep=Price, JointResumeContract=lambda p: p,
                ProbeResumePolicy=NS(RUN_SEALS='seals'))
        written = []
        install(monkeypatch, 'prismaquant.cost_stage_checkpoint', unit_path=lambda r,n: r / (n + '.pkl'),
                atomic_write_bytes=lambda p,b: (trace.append(('write', p.name)), written.append(b)))
        inputs = NS(probe_identity={'n_probes': 2}, render_formats=['fmt'], joint_probe={}, formats=['fmt'],
                    operators={}, components={}, **{k: {} for k in ('s2', 's4', 'x2_probe', 'dw_src', 'g_trace')})
        timings = {} if timed else None
        fn = consumer('tools/unit_journal_bench.py', '_commit_unit', old, trace, fail_hash, time=clock())
        result = outcome(fn, inputs, 'unit', tmp_path, 'identity', timings)
        results.append((result, trace, written, timings))
        if written:
            import pickletools
            envelope = pickle.loads(written[0])
            raw = envelope['payload']
            assert raw == pickle.dumps(dict(state), protocol=pickle.HIGHEST_PROTOCOL)
            restored = pickle.loads(raw)
            assert restored['first'] is restored['second']
            assert any(op.name == 'FRAME' for op, arg, pos in pickletools.genops(raw))
            assert any(op.name.endswith('GET') for op, arg, pos in pickletools.genops(raw))
            assert envelope['payload_sha256'] == hashlib.sha256(raw).hexdigest()
    assert results[0] == results[1]
    events = [e[0] for e in results[1][1]]
    assert events == (['prepare', 'state', 'publish'] if not timed else
                      ['prepare', 'state', 'hash'] + ([] if fail_hash else ['write']))


@pytest.mark.parametrize('fail_hash', [None, 1, 2])
def test_unit_child_setup_receipt(monkeypatch, tmp_path, fail_hash):
    results = []
    for old in (True, False):
        trace = []
        output = tmp_path / 'result.json'
        output.unlink(missing_ok=True)
        inputs = NS(formats=['fmt'], render_formats=['fmt'], components={'unit': [1]}, names=['unit'])
        def commit(inputs, name, root, identity, timings):
            trace.append(('commit', name, identity, timings is not None))
            (root / 'units' / (name + '.pkl')).write_bytes(b'checkpoint\x00\xff')
        fn = consumer('tools/unit_journal_bench.py', 'cmd_child', old, trace, fail_hash,
                      time=clock(), _cgroup_dir=lambda: None, _load_sample=lambda p: {},
                      _unit_inputs=lambda *a: inputs, _memory_sample=lambda g: {},
                      _delta=lambda *a: {}, _commit_unit=commit,
                      _drop_cache=lambda r: trace.append(('drop', r.name)))
        args = NS(idle_threads=0, sample_checkpoint='unused', units=1, windows=1,
                  scratch=str(tmp_path), arm='baseline', repeat=0, result=str(output))
        result = outcome(fn, args)
        results.append((result, trace, output.read_bytes() if output.exists() else None))
    assert results[0] == results[1]
    if fail_hash is None:
        report = json.loads(results[1][2])
        assert report['checkpoint_sha256_first'] == hashlib.sha256(b'checkpoint\x00\xff').hexdigest()
        assert results[1][1][1] == ('commit', 'unit', hashlib.sha256(b'unit-journal-bench').hexdigest(), True)
        assert not (tmp_path / 'checkpoints-baseline-0').exists()
    else:
        assert results[1][0] == ('ValueError', 'hash adapter failure')
        assert results[1][2] is None


@pytest.mark.parametrize('bad_result', [False, True])
def test_staged_child_stdout(tmp_path, bad_result):
    results = []
    for old in (True, False):
        trace = []
        def child(args, doc):
            trace.append(('child', doc))
            return {'value': object() if bad_result else ['é', None, 1.5]}
        fn = consumer('tools/staged_exact_read_bench.py', 'child_main', old, trace,
                      load_slice=lambda *a: {'slice': 1}, CHILDREN={'test': child})
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = outcome(fn, NS(slice='unused', slice_sha256='sha', child='test'))
        results.append((result, trace, stdout.getvalue()))
    assert results[0] == results[1]
    if bad_result:
        assert results[1][0][0] == 'TypeError' and results[1][2] == ''
    else:
        marker = load('tools/staged_exact_read_bench.py').MARKER
        assert results[1][2].startswith(marker)
        assert json.loads(results[1][2][len(marker):])['value'] == ['é', None, 1.5]
        assert '\\u00e9' in results[1][2]
