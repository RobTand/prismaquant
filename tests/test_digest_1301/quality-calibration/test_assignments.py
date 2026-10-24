"""CPU publication-block differentials against verbatim research-base consumers.

AST isolation avoids model loading; serialization, filesystem writes, reports and
refusals execute the actual consumer blocks, not a reimplemented encoder.
"""
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import textwrap
from typing import Sequence
import unittest
from unittest.mock import Mock

from prismaquant.digests import DIRECT_ASCII_INDENT2_LAX

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/quality-calibration/assignments'


def consumer(kind, original):
    namespace = dict(Path=Path, Sequence=Sequence, json=json,
                     DIRECT_ASCII_INDENT2_LAX=DIRECT_ASCII_INDENT2_LAX)
    if original:
        source = (FIXTURES / (kind + '.py')).read_text()
    else:
        tool = ('make_uniform_assignment' if kind == 'uniform'
                else 'build_sensitivity_ablation_assignments')
        source = (ROOT / 'tools' / (tool + '.py')).read_text()
        tree = ast.parse(source)
        name = '_write_overlay' if kind == 'overlay' else 'main'
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        start = fn.lineno - 1
        if kind != 'overlay':
            target = 'payload' if kind == 'uniform' else 'manifest'
            node = next(n for n in fn.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == target for t in n.targets))
            start = node.lineno - 1
        source = ''.join(source.splitlines(keepends=True)[start:fn.end_lineno])
    if kind == 'overlay':
        exec(source, namespace)
        return namespace['_write_overlay'], namespace
    exec('def publish():\n' + textwrap.indent(textwrap.dedent(source), '    '), namespace)
    return namespace['publish'], namespace


def inputs(kind, directory):
    if kind == 'manifest':
        return dict(args=SimpleNamespace(base_assignment='base', probe='pröbe', costs='costs',
                                        top_groups_per_category=2),
                    base_bpp=float('nan'), categories=['é', 'z'], output_dir=directory,
                    variants=[dict(label='一', non_bf16_count=2, bpp_delta=-0.25,
                                   predicted_dloss_saved=float('inf'))])
    result = SimpleNamespace(assignment={'z': 'BF16', 'é': 'NVFP4'},
                             excluded={'一': 'illegal (source)'}, unit_count=3,
                             demoted_units=('é',), params_by_format={'NVFP4': 32, 'BF16': 16})
    return dict(args=SimpleNamespace(out=directory / 'nested/assignment.json',
                                    report=directory / 'reports/report.json'),
                result=result, meta={'note': 'é', 'value': float('nan')}, achieved=8.0,
                layer_config_payload=lambda result, meta: dict(result.assignment, __meta__=meta))


def compare_publication(kind):
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        observations = []
        for original in (True, False):
            fn, ns = consumer(kind, original)
            route = Mock(wraps=DIRECT_ASCII_INDENT2_LAX.text)
            ns['DIRECT_ASCII_INDENT2_LAX'] = SimpleNamespace(text=route)
            if kind != 'overlay':
                ns.update(inputs(kind, directory))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                if kind == 'overlay':
                    value = fn(directory / 'overlay.json', ['z', 'é', '一', 'z'])
                else:
                    value = fn()
            files = {str(p.relative_to(directory)): p.read_bytes()
                     for p in directory.rglob('*.json')}
            observations.append((value, stdout.getvalue(), files))
            if not original:
                assert route.call_count == (2 if kind == 'uniform' else 1)
            assert files
            for data in files.values():
                assert data.endswith(b'\n') and not data.endswith(b'\n\n')
                assert b'\\u' in data
            for p in directory.rglob('*.json'):
                p.unlink()
        assert observations[0] == observations[1]


def compare_refusals(kind):
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        exceptions = []
        for original in (True, False):
            fn, ns = consumer(kind, original)
            if kind == 'overlay':
                target = directory / 'missing/overlay.json'
                call = lambda: fn(target, ['é'])
                expected = FileNotFoundError
            else:
                ns.update(inputs(kind, directory))
                if kind == 'manifest':
                    target = directory / 'manifest.json'
                    ns['variants'] = [object()]
                    call = fn
                else:
                    target = Path(ns['args'].out)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    ns['layer_config_payload'] = lambda *a, **kw: {'invalid': object()}
                    call = fn
                target.write_bytes(b'previous publication\n')
                expected = TypeError
            try:
                call()
            except expected as exc:
                exceptions.append((type(exc), str(exc)))
            else:
                raise AssertionError('publication unexpectedly succeeded')
            if kind == 'overlay':
                assert not target.exists()
            else:
                assert target.read_bytes() == b'previous publication\n'
            assert not (directory / 'reports/report.json').exists()
        assert exceptions[0] == exceptions[1]


def test_ledger_nodes():
    # No Git history dependency: snapshot/packaged checkouts can run this too.
    ledger = json.loads((FIXTURES / 'ledger.json').read_text())
    assert len(ledger) == 4
    assert len({row['original_id'] for row in ledger}) == 4
    for row in ledger:
        assert row['source_commit'] == '6d8288adc94b3c4b624c143ed372fb6513f32784'
        assert row['owner_route'] == 'prismaquant.digests.DIRECT_ASCII_INDENT2_LAX.text'
        for fixture in row['fixtures']:
            block = (ROOT / fixture).read_text()
            assert len(block.splitlines()) < 60
            assert 'write_text' in block
        for node in row['tests']:
            file, name = node.split('::')
            assert (ROOT / file).resolve() == Path(__file__).resolve()
            assert callable(globals()[name])


def test_overlay_publication():
    compare_publication('overlay')


def test_overlay_refusals():
    compare_refusals('overlay')


def test_manifest_publication():
    compare_publication('manifest')


def test_manifest_refusals():
    compare_refusals('manifest')


def test_uniform_publication():
    compare_publication('uniform')


def test_uniform_refusals():
    compare_refusals('uniform')


def test_report_refusals():
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        observations = []
        for original in (True, False):
            fn, ns = consumer('uniform', original)
            ns.update(inputs('uniform', directory))
            # The assignment is valid, but report-only metadata is not encodable.
            ns['meta'] = {'unsupported': object()}
            ns['layer_config_payload'] = lambda *a, **kw: {'é': 'BF16'}
            report = Path(ns['args'].report)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_bytes(b'previous report\n')
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                try:
                    fn()
                except TypeError as exc:
                    error = (type(exc), str(exc))
                else:
                    raise AssertionError('report unexpectedly succeeded')
            assignment = Path(ns['args'].out).read_bytes()
            assert assignment.endswith(b'\n')
            assert report.read_bytes() == b'previous report\n'
            observations.append((error, stdout.getvalue(), assignment))
        assert observations[0] == observations[1]


def test_uniform_without_report():
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        observations = []
        for original in (True, False):
            fn, ns = consumer('uniform', original)
            ns.update(inputs('uniform', directory))
            ns['args'].report = None
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                value = fn()
            assert not (directory / 'reports').exists()
            observations.append((value, stdout.getvalue(), Path(ns['args'].out).read_bytes()))
        assert observations[0] == observations[1]


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in
                               sorted(globals().copy().items()) if name.startswith('test_'))
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not outcome.wasSuccessful())
