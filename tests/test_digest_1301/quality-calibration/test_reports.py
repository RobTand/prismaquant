"""CPU execution of extracted publication/Counter blocks, not GPU qualification."""
import ast
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import digests

ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / 'tests/fixtures/digest_1301/quality-calibration/reports'


def actual(path, select):
    source = (ROOT / 'tools' / path).read_text()
    nodes = [n for n in ast.parse(source).body if select(n)]
    assert nodes
    return '\n'.join(ast.get_source_segment(source, n) for n in nodes)


def run(code, values):
    ns = dict(json=json, hashlib=hashlib, Counter=Counter,
              DIRECT_ASCII_SPACED_LAX=digests.DIRECT_ASCII_SPACED_LAX,
              DIRECT_ASCII_SPACED_STRICT=digests.DIRECT_ASCII_SPACED_STRICT,
              bytes_sha256hex=digests.bytes_sha256hex)
    ns.update(values)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, ns)
    return out.getvalue(), ns


def print_node(n):
    return isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'print'


@pytest.mark.parametrize('value', ['héllo', float('nan'), float('inf'), -0.0])
def test_group_report(value):
    values = dict(binding={'z': value, 'a': 2}, identity={'z': value}, observations=[{'z': value, 'a': 1}])
    old, _ = run((FIX / 'group.py').read_text(), values)
    new, _ = run(actual('check_stageb_a4_group_quantizer.py', print_node), values)
    assert new == old
    assert new.endswith('\n') and new.startswith('{"a8_qdq_and_identity_unchanged": true, ')
    if isinstance(value, str):
        assert 'h\\u00e9llo' in new


@pytest.mark.parametrize('value', ['héllo', float('nan'), -0.0])
def test_quantizer_report(value):
    class Tensor:
        shape = (4, 1024)
        def __ne__(self, other):
            return self
        def sum(self):
            return self
        def item(self):
            return 17
    values = dict(identity={'z': value, 'a': 1}, x=Tensor(), y=Tensor(),
                  torch=SimpleNamespace(version=SimpleNamespace(cuda='CPU adapter'),
                                        cuda=SimpleNamespace(get_device_name=lambda: 'CPU adapter')))
    old, _ = run((FIX / 'quantizer.py').read_text(), values)
    new, _ = run(actual('check_stageb_a4_quantizer.py', print_node), values)
    assert old == new
    assert json.loads(new)['changed_elements'] == 17


@pytest.mark.parametrize('case', ['reordered', 'multiplicity', 'content'])
def test_refusal_multiset(case):
    a = {'z': 'héllo', 'a': -0.0}
    b = {'kind': 'refused', 'values': [1, 2]}
    left = [a, b, a]
    right = [b, a, a] if case == 'reordered' else ([b, a] if case == 'multiplicity' else [b, a, {'changed': 1}])
    values = dict(payloads={'monolith': {'non_interpolable': left},
                           'historical_merge': {'non_interpolable': right}})
    code = actual('pq282_equality/recheck_refusal_order.py',
                  lambda n: (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'counts' for t in n.targets)) or
                  (isinstance(n, ast.If) and 'counts' in ast.unparse(n.test)))
    outcomes = []
    for source in [(FIX / 'refusals.py').read_text(), code]:
        try:
            _, ns = run(source, values)
            outcomes.append((ns['counts'], None))
        except RuntimeError as exc:
            outcomes.append((None, str(exc)))
    assert outcomes[0] == outcomes[1]
    assert bool(outcomes[1][1]) == (case != 'reordered')
    if case == 'reordered':
        assert sorted(outcomes[1][0]['monolith'].values()) == [1, 2]


def test_recheck_report(tmp_path):
    code = actual('pq282_equality/recheck_refusal_order.py',
                  lambda n: isinstance(n, ast.Expr) and 'recheck.json' in ast.unparse(n))
    outputs = []
    raw = {'monolith': b'\x00\xff\n', 'historical_merge': b'different pickle bytes'}
    for source in [(FIX / 'recheck_report.py').read_text(), code]:
        run(source, dict(args=SimpleNamespace(out=tmp_path), paths={k: Path(k) for k in raw},
                         raw=raw, result=SimpleNamespace(returncode=3), command=['python', 'diff']))
        outputs.append((tmp_path / 'recheck.json').read_bytes())
    assert outputs[0] == outputs[1]
    assert outputs[1].startswith(b'{\n  "inputs":') and outputs[1].endswith(b'\n')
    report = json.loads(outputs[1])
    assert report['comparison_returncode'] == 3
    for k, data in raw.items():
        assert report['inputs'][k]['sha256'] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize('mode', ['ok', 'nan', 'error'])
def test_waiter_report(mode):
    code = actual('wait_dsv4_aura_campaign.py', lambda n: isinstance(n, ast.FunctionDef) and n.name == 'main')
    outcomes = []
    for source in [(FIX / 'waiter.py').read_text(), code]:
        err = io.StringIO()
        def inner(args):
            if mode == 'error':
                raise RuntimeError('refused')
            return {'z': 'héllo' if mode == 'ok' else float('nan'), 'a': 1}
        values = dict(sys=SimpleNamespace(argv=['waiter'], stderr=err),
                      os=SimpleNamespace(environ={'internal': '1'}), _INTERNAL='internal',
                      Sequence=list, argparse=SimpleNamespace(), WaiterError=RuntimeError,
                      _parser=lambda: SimpleNamespace(parse_args=lambda argv: argv), _inner_main=inner)
        _, ns = run(source, values)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                result = ns['main'](['verify'])
                error = None
            except ValueError as exc:
                result, error = None, type(exc)
        outcomes.append((result, error, out.getvalue(), err.getvalue()))
    assert outcomes[0] == outcomes[1]
    if mode == 'ok':
        assert outcomes[1] == (0, None, '{"a": 1, "z": "h\\u00e9llo"}\n', '')
    elif mode == 'error':
        assert outcomes[1][0] == 2 and not outcomes[1][2] and 'refused' in outcomes[1][3]
    else:
        assert outcomes[1][1] is ValueError and not outcomes[1][2]
