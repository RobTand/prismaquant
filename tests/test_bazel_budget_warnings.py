"""Bazel records time-budget warnings without a second pytest execution (#3216)."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('budget_warning_runner', ROOT / 'tools/bazel/python_runner.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def report(root, elapsed=2.0, failed=False):
    (root / 'tools/ci').mkdir(parents=True)
    (root / 'tools/ci/gate_budget.json').write_text(json.dumps({'seconds': 1, 'slow': {}}))
    junit = root / 'junit.xml'
    failure = '<failure message="test failed"/>' if failed else ''
    junit.write_text('<testsuites><testsuite><testcase file="tests/test_example.py" '
                     'classname="tests.test_example" name="test_one" '
                     f'time="{elapsed}">{failure}</testcase></testsuite></testsuites>')
    return junit


def test_every_overrun_warns_without_the_deleted_tolerance_api(tmp_path, monkeypatch):
    junit = report(tmp_path, elapsed=1.2)
    monkeypatch.syspath_prepend(str(ROOT / 'tools/ci'))
    import gate_run
    monkeypatch.delattr(gate_run, 'BUDGET_ENFORCEMENT_TOLERANCE', raising=False)
    assert runner.budget_overruns(tmp_path, junit, 'tests/test_example.py') == [
        {'nodeid': 'tests/test_example.py::test_one', 'seconds': 1.2, 'budget_seconds': 1}]


@pytest.mark.parametrize('record_mode', [None, 'record'])
@pytest.mark.parametrize('exit_status', [0, 1, 124])
def test_budget_policy_preserves_real_verdict_and_executes_once(tmp_path, monkeypatch, capsys,
                                                              record_mode, exit_status):
    junit = report(tmp_path, failed=exit_status == 1)
    outputs = tmp_path / 'outputs'
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('PQ_BAZEL_LANE', raising=False)
    monkeypatch.delenv('TEST_TOTAL_SHARDS', raising=False)
    monkeypatch.delenv('TEST_SHARD_INDEX', raising=False)
    monkeypatch.setenv('TEST_TMPDIR', str(tmp_path))
    monkeypatch.setenv('TEST_UNDECLARED_OUTPUTS_DIR', str(outputs))
    monkeypatch.setenv('XML_OUTPUT_FILE', str(junit))
    if record_mode is None:
        monkeypatch.delenv('PQ_BAZEL_BUDGETS', raising=False)
    else:
        monkeypatch.setenv('PQ_BAZEL_BUDGETS', record_mode)
    monkeypatch.setattr(sys, 'argv', ['python_runner.py', 'tests/test_example.py'])
    monkeypatch.setattr(runner, 'prepare_remote_test', lambda root: (scratch, {'PYTHONPATH': ''}))
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=exit_status)

    monkeypatch.setattr(runner.subprocess, 'run', execute)
    assert runner.main() == exit_status
    assert len(calls) == 1
    if exit_status == 0:
        inventory = json.loads((outputs / 'inventory.json').read_text())
        assert inventory['budget_warnings'] == [
            {'nodeid': 'tests/test_example.py::test_one', 'seconds': 2.0, 'budget_seconds': 1}]
        assert 'BUDGET_WARNING:' in capsys.readouterr().err
    else:
        assert not (outputs / 'inventory.json').exists()
