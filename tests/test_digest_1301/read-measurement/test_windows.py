"""Original research-base call snippets are independent byte-level references."""
import ast
import hashlib
import importlib.util
import io
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / 'tests/fixtures/digest_1301/read-measurement/windows_source.json'
SOURCE = json.loads(FIXTURE.read_text())
LEDGER = json.loads(Path(__file__).with_name('windows_ledger.json').read_text())['calls']


def load(file):
    spec = importlib.util.spec_from_file_location(Path(file).stem, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scoped_calls(file, scope):
    tree = ast.parse((ROOT / file).read_text())
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        current, names = node, []
        while current in parents:
            current = parents[current]
            if isinstance(current, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                names.append(current.name)
        if '.'.join(reversed(names)) == scope:
            result.append(node)
    return result


class TensorView:
    shape = (2, 3)
    dtype = 'torch.uint8'

    def reshape(self, *shape):
        return self

    def view(self, dtype):
        return self

    def numpy(self):
        return bytearray(b'\x00\xffpayload\x00')


@pytest.mark.parametrize('index', [i for i, r in enumerate(LEDGER) if r['status'] == 'migrated'])
def test_original_call_behavior(index, tmp_path):
    row, reference = LEDGER[index], SOURCE['snippets'][index]
    assert row['source_commit'] == SOURCE['source_commit'] == '6d8288adc94b3c4b624c143ed372fb6513f32784'
    ast.parse(reference['SOURCE'], mode='eval')
    module = load(row['file'])
    route = row['owner_route'].removeprefix('prismaquant.digests.')
    if route == 'file_sha256hex':
        path = tmp_path / 'file'
        path.write_bytes(b'abc\x00\xff' * 300000)
        original = next(f for f in SOURCE['functions'] if f['name'] == '_sha256_file')
        namespace = {'hashlib': hashlib}
        exec(original['SOURCE'], namespace)
        assert module._sha256_file(path) == namespace['_sha256_file'](path)
        return
    candidates = [n for n in scoped_calls(row['file'], row['scope']) if ast.unparse(n.func) == route]
    earlier = sum(r['file'] == row['file'] and r['scope'] == row['scope'] and r['owner_route'] == row['owner_route'] for r in LEDGER[:index])
    # ast.walk order is retained from the original AST, not sorted line order.
    actual = candidates[earlier]
    root = tmp_path
    (root / 'units').mkdir()
    path = root / 'units' / 'one.pkl'
    path.write_bytes(b'file\x00\xffdata')
    document = {'z': ['é', None, True, 1.5], 'a': {'unordered': 2}, 'nan': float('nan')}
    env = dict(vars(module), hashlib=hashlib, json=json, pickle=pickle,
               raw=b'raw\x00\xff', blob=b'blob\x00\xff', obj=bytearray(b'bytes'),
               state_bytes=pickle.dumps(document), payload=document, text='é\x00text',
               root=root, path=path, summary=document, result=document,
               tensor=TensorView(), torch=SimpleNamespace(uint8='uint8'),
               key=('unit', 'fmt'), file_receipt={'path': 'ignored', 'size': 6},
               identity={'sha256': 'a' * 64})
    expected = eval(reference['expression'], env)
    assert eval(compile(ast.Expression(actual), row['file'], 'eval'), env) == expected


@pytest.mark.parametrize('index', [i for i, r in enumerate(LEDGER) if r['status'] == 'blocked'])
def test_blocked_recipes_remain_literal(index):
    row, reference = LEDGER[index], SOURCE['snippets'][index]
    assert row['owner_route'] is None and row['blocker']
    class NormalizeNestedBytes(ast.NodeTransformer):
        def visit_Call(self, node):
            if (isinstance(node.func, ast.Attribute) and node.func.attr == 'hexdigest'
                    and isinstance(node.func.value, ast.Call)
                    and ast.unparse(node.func.value.func) == 'hashlib.sha256'
                    and node.func.value.args):
                return ast.Call(func=ast.Name(id='bytes_sha256hex', ctx=ast.Load()),
                                args=node.func.value.args, keywords=[])
            return self.generic_visit(node)

    expected = ast.dump(NormalizeNestedBytes().visit(ast.parse(reference['SOURCE'], mode='eval').body))
    assert expected in [ast.dump(n) for n in scoped_calls(row['file'], row['scope'])]


@pytest.mark.parametrize('file,name', [('tools/chain_roll_bench.py', 'load_verified_json'),
                                     ('tools/staged_exact_read_bench.py', 'load_slice')])
def test_verified_json_errors_and_parsing(file, name, tmp_path):
    module = load(file)
    namespace = {'Path': Path, 'json': json, 'hashlib': hashlib}
    original = next(f for f in SOURCE['functions'] if f['name'] == name)
    exec(original['SOURCE'], namespace)
    path = tmp_path / 'slice'
    for raw in [b'{"a":1,"a":2}', b'not JSON', b'\xff']:
        path.write_bytes(raw)
        for expected in [hashlib.sha256(raw).hexdigest(), '0' * 64]:
            outcomes = []
            for fn in [namespace[name], getattr(module, name)]:
                try:
                    outcomes.append(('ok', fn(path, expected)))
                except (SystemExit, ValueError, UnicodeError) as exc:
                    outcomes.append((type(exc), str(exc)))
            assert outcomes[0] == outcomes[1]


def test_file_retention_read_witness(monkeypatch):
    module = load('tools/chain_roll_bench.py')
    import builtins
    reads = []
    payload = b'abcd' * (1 << 19)

    class Reader(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            assert size == 1 << 20
            return super().read(size)

    monkeypatch.setattr(builtins, 'open', lambda *a, **kw: Reader(payload))
    assert module._sha256_file('unused') == hashlib.sha256(payload).hexdigest()
    assert reads == [1 << 20] * 3


def test_tensor_view_not_copied(monkeypatch):
    module = load('tools/staged_read_stream_ab.py')
    seen = []
    original = module.bytes_sha256hex

    def capture(data):
        seen.append(data)
        assert isinstance(data, memoryview)
        return original(data)

    monkeypatch.setattr(module, 'bytes_sha256hex', capture)
    tensor = TensorView()
    assert module.digest_tensor(tensor, SimpleNamespace(uint8='uint8')) == hashlib.sha256(tensor.numpy()).hexdigest()
    assert len(seen) == 1 and isinstance(seen[0].obj, bytearray)


def test_leaf_walker_original_behavior():
    import numpy as np
    import torch

    module = load('tools/pwc_window_load_bench.py')
    original = next(f for f in SOURCE['functions'] if f['name'] == '_leaves')
    namespace = {'hashlib': hashlib, 'pickle': pickle}
    exec(original['SOURCE'], namespace)
    obj = {'tensor': torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
           'array': np.arange(12, dtype=np.int16).reshape(3, 4).T,
           'bytes': b'not a pickle\x00', 'bytearray': bytearray(b'\xff'),
           'nested': pickle.dumps({'long': 'é' * 200, 'float': -0.0}),
           'tuple': (None, True, 1.0)}
    assert module._leaves(obj, 'root') == namespace['_leaves'](obj, 'root')


def test_real_tensor_view_digest():
    import torch

    module = load('tools/staged_read_stream_ab.py')
    original = next(f for f in SOURCE['functions'] if f['name'] == 'digest_tensor')
    namespace = {'hashlib': hashlib}
    exec(original['SOURCE'], namespace)
    for tensor in [torch.arange(12, dtype=torch.float32),
                   torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
                   torch.empty(0, dtype=torch.uint8)]:
        assert module.digest_tensor(tensor, torch) == namespace['digest_tensor'](tensor, torch)


def test_timing_and_retention_boundaries():
    # Narrow lexical witnesses: preserve buffer-producing expressions and do
    # not move the measured hash across its adjacent timer calls.
    unit = (ROOT / 'tools/unit_journal_bench.py').read_text()
    assert 'digest = bytes_sha256hex(state_bytes)\n    hashed = time.perf_counter()' in unit
    pwc = (ROOT / 'tools/pwc_window_load_bench.py').read_text()
    assert 'renders = tensor = None' in pwc
    assert 'digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())' in pwc
    render = (ROOT / 'tools/render_window_bench.py').read_text()
    assert 'digest.update(repr((probe_index, key, sorted(components.items()))).encode())' in render
