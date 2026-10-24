"""Original consumer versus actual main on CPU; not served vLLM qualification."""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'tools/vllm_prompt_smoke.py'
FIX = ROOT / 'tests/fixtures/digest_1301/quality-calibration/vllm'


def load(original):
    if original:
        ns = dict(argparse=argparse, json=json, time=SimpleNamespace(), Path=Path,
                  __doc__='Original consumer fixture')
        exec((FIX / 'main.py').read_text(), ns)
        return SimpleNamespace(**ns)
    spec = importlib.util.spec_from_file_location('vllm_smoke_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(monkeypatch, original, argv, text='héllo\n世界', failure=None):
    module = load(original)
    ticks = iter([10.0, 11.25, 12.0, 14.5])
    monkeypatch.setattr(module.time, 'time', lambda: next(ticks), raising=False)
    calls = []

    class LLM:
        def __init__(self, **kwargs):
            calls.append(('init', kwargs))
            if failure == 'init':
                raise RuntimeError('initialization refused')

        def generate(self, prompts, sampling):
            calls.append(('generate', prompts, vars(sampling)))
            if failure == 'generate':
                raise RuntimeError('generation refused')
            return [SimpleNamespace(outputs=[SimpleNamespace(text=text)])]

    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=LLM, SamplingParams=SimpleNamespace))
    monkeypatch.setattr(sys, 'argv', ['vllm_prompt_smoke.py', '--model', 'model-é', *argv])
    stdout, stderr = io.StringIO(), io.StringIO()
    error = None
    result = None
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            result = module.main()
        except (SystemExit, RuntimeError, OSError, TypeError) as exc:
            error = (type(exc).__name__, str(exc))
    return result, stdout.getvalue(), stderr.getvalue(), error, calls


@pytest.mark.parametrize('text', ['héllo\n世界', float('nan'), float('inf'), -0.0])
@pytest.mark.parametrize('extra', [[], ['--no-enforce-eager', '--dtype', 'bfloat16',
    '--quantization', 'compressed-tensors', '--tensor-parallel-size', '2',
    '--distributed-executor-backend', 'mp', '--prompt', 'é?']])
def test_main_equivalence(monkeypatch, tmp_path, text, extra):
    path = tmp_path / 'nested' / 'result.json'
    args = [*extra, '--output-json', str(path)]
    old = invoke(monkeypatch, True, args, text)
    before = path.read_bytes()
    path.write_text('prior content to replace')
    new = invoke(monkeypatch, False, args, text)
    assert old == new
    assert new[0] == 0 and new[3] is None
    assert path.read_bytes() == before == new[1].encode()
    assert before.endswith(b'\n') and not before.endswith(b'\n\n')
    assert before.startswith(b'{\n  "distributed_executor_backend":')
    if isinstance(text, str):
        assert b'h\\u00e9llo' in before
    assert json.loads(before)['mode'] == ('graph' if extra else 'eager')


@pytest.mark.parametrize('args', [
    ['--tensor-parallel-size', '0'], ['--max-model-len', '0'],
    ['--max-new-tokens', '-1'], ['--gpu-memory-utilization', '0'],
    ['--gpu-memory-utilization', '1'], ['--gpu-memory-utilization', 'nan'],
    ['--distributed-executor-backend', 'invalid'],
])
def test_refusals(monkeypatch, args):
    old = invoke(monkeypatch, True, args)
    assert old == invoke(monkeypatch, False, args)
    assert old[3] == ('SystemExit', '2') and old[4] == []
    assert not old[1]


@pytest.mark.parametrize('failure', ['init', 'generate'])
def test_runtime_refusal(monkeypatch, tmp_path, failure):
    path = tmp_path / 'result.json'
    path.write_bytes(b'existing')
    args = ['--output-json', str(path)]
    old = invoke(monkeypatch, True, args, failure=failure)
    assert old == invoke(monkeypatch, False, args, failure=failure)
    assert old[3][0] == 'RuntimeError' and not old[1]
    assert path.read_bytes() == b'existing'


def test_publication_refusal(monkeypatch, tmp_path):
    parent = tmp_path / 'not-a-directory'
    parent.write_bytes(b'keep')
    args = ['--output-json', str(parent / 'result.json')]
    old = invoke(monkeypatch, True, args)
    assert old == invoke(monkeypatch, False, args)
    assert old[3][0] == 'FileExistsError'
    assert old[1].endswith('\n')  # stdout precedes failed publication
    assert parent.read_bytes() == b'keep'


def test_serialization_refusal(monkeypatch, tmp_path):
    path = tmp_path / 'result.json'
    path.write_bytes(b'keep')
    args = ['--output-json', str(path)]
    old = invoke(monkeypatch, True, args, text={'not-json'})
    assert old == invoke(monkeypatch, False, args, text={'not-json'})
    assert old[3][0] == 'TypeError' and not old[1]
    assert path.read_bytes() == b'keep'


def test_no_publication(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    old = invoke(monkeypatch, True, [])
    assert old == invoke(monkeypatch, False, [])
    assert list(tmp_path.iterdir()) == []


def test_preflight_gap(monkeypatch):
    old = invoke(monkeypatch, True, ['--cpu-preflight'])
    assert old == invoke(monkeypatch, False, ['--cpu-preflight'])
    assert old[4] == []
    assert old[1].startswith('{\n  "status":')  # deliberately unsorted


def test_standalone_launch(tmp_path):
    # -I discards PYTHONPATH and cwd imports; the tools script must still launch.
    env = dict(os.environ, PYTHONPATH='')
    proc = subprocess.run([sys.executable, '-I', str(SCRIPT), '--model', 'portable',
                           '--cpu-preflight'], cwd=tmp_path, env=env,
                          text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ''
    assert json.loads(proc.stdout)['status'] == 'cpu_preflight_only'


def test_provenance():
    ledger = json.loads((FIX / 'ledger.json').read_text())
    assert len(ledger) == 3
    fixture = (FIX / 'main.py').read_bytes()
    for row in ledger:
        assert row['fixture_sha256'] == hashlib.sha256(fixture).hexdigest()
        for path in row['fixtures']:
            assert (ROOT / path).is_file()
        for test in row['tests']:
            path, name = test.split('::')
            tree = ast.parse((ROOT / path).read_text())
            assert any(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body)
    assert [r['status'] for r in ledger] == ['migrated', 'migrated', 'GAP']
    assert ledger[-1]['owner_route'] is None
