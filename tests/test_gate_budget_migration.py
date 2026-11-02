"""A reviewed migration record is the one way a budget grows (#3216, #3223).

``docs/ci_gate.md``: after bootstrap a budget cannot be raised by editing the
JSON. A verified dependency change that invalidates the baseline needs an
explicit policy migration that the readiness lead reviews, plus hosted
evidence. ``tools/ci/gate_budget_migrations.json`` is that record. A
migration names the exact base budget policy (the Git blob of
``tools/ci/gate_budget.json`` it was measured against), the reason, and per
node the hosted evidence from both architectures. Its limit must be
``ceil(2 * largest hosted time)``. Growth is accepted only for the listed
nodes, only up to their limits, and only against that base policy.
Everything else is refused as before.
"""
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from test_gate_budget import budget_run, case, commit_target, policy, runner

NODE = 'tests/test_sample.py::test_cost'
MIGRATIONS = 'tools/ci/gate_budget_migrations.json'
X64, ARM = 'ubuntu-24.04', 'ubuntu-24.04-arm'


@pytest.fixture(autouse=True)
def isolate_job_summary(monkeypatch):
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)


def evidence(x64=40.2, arm=31.5):
    return [{'run': 101, 'attempt': 1, 'runner': X64, 'seconds': x64},
            {'run': 102, 'attempt': 1, 'runner': ARM, 'seconds': arm}]


def base_policy_blob(root):
    return subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD:tools/ci/gate_budget.json'], text=True).strip()


def record(root, blob, nodes, *, reason='dependency change: PrismaBuild SDK installed, #3223'):
    (root / MIGRATIONS).write_text(json.dumps({
        'schema': 'prismaquant.gate_budget_migration.v1',
        'migrations': [{'base_policy': blob, 'base_sha': 'a' * 40, 'reason': reason,
                        'slow': nodes}]}))


def entry(seconds=None, rows=None):
    rows = evidence() if rows is None else rows
    limit = math.ceil(2 * max(row['seconds'] for row in rows)) if seconds is None else seconds
    return {'seconds': limit, 'evidence': rows}


def based(tmp_path):
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40})
    base = commit_target(tmp_path)
    return base, base_policy_blob(tmp_path)


def test_a_listed_node_may_be_added_at_its_measured_limit(tmp_path):
    base, blob = based(tmp_path)
    record(tmp_path, blob, {NODE: entry()})
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, NODE: 81})
    code, result = budget_run(tmp_path, case('45'), base=base)
    assert code == 0, result
    assert result['budget']['status'] == 'pass'


def test_without_a_record_the_same_addition_is_refused(tmp_path):
    base, _ = based(tmp_path)
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, NODE: 81})
    code, result = budget_run(tmp_path, case('45'), base=base)
    assert code == 1 and 'grow' in result['budget']['reason']


def test_a_record_for_another_base_policy_grants_nothing(tmp_path):
    base, _ = based(tmp_path)
    record(tmp_path, 'b' * 40, {NODE: entry()})
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, NODE: 81})
    code, result = budget_run(tmp_path, case('45'), base=base)
    assert code == 1 and 'grow' in result['budget']['reason']


@pytest.mark.parametrize('slow', [
    {'tests/test_sample.py::test_known': 40, NODE: 82},            # above its limit
    {'tests/test_sample.py::test_known': 41, NODE: 81},            # an unlisted node grows
    {'tests/test_sample.py::test_known': 40, NODE: 81, 'tests/test_sample.py::test_other': 30},
])
def test_growth_beyond_the_record_is_refused(tmp_path, slow):
    base, blob = based(tmp_path)
    record(tmp_path, blob, {NODE: entry()})
    policy(tmp_path, slow=slow)
    code, result = budget_run(tmp_path, case('0'), base=base)
    assert code == 1 and 'grow' in result['budget']['reason']


def test_a_record_cannot_raise_the_default_or_file_budgets(tmp_path):
    base, blob = based(tmp_path)
    record(tmp_path, blob, {NODE: entry()})
    policy(tmp_path, seconds=21, slow={'tests/test_sample.py::test_known': 40, NODE: 81})
    code, result = budget_run(tmp_path, case('0'), base=base)
    assert code == 1 and 'grow' in result['budget']['reason']
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, NODE: 81},
           file_seconds={'tests/test_other.py': 50})
    code, result = budget_run(tmp_path, case('0'), base=base)
    assert code == 1 and 'grow' in result['budget']['reason']


@pytest.mark.parametrize('nodes', [
    {NODE: entry(seconds=90)},                                      # not ceil(2 * max)
    {NODE: entry(rows=evidence()[:1])},                             # one architecture only
    {NODE: entry(81, [{**evidence()[0], 'seconds': float('nan')}, evidence()[1]])},
    {NODE: entry(81, [{**evidence()[0], 'run': 'x'}, evidence()[1]])},
    {NODE: {'seconds': 81}},                                        # no evidence
])
def test_a_record_without_two_architecture_evidence_fails_closed(tmp_path, nodes):
    base, blob = based(tmp_path)
    record(tmp_path, blob, nodes)
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, NODE: 81})
    code, result = budget_run(tmp_path, case('0'), base=base)
    assert code == 1 and result['budget']['status'] == 'fail'
    assert 'migration' in result['budget']['reason']


@pytest.mark.parametrize('reason', ['', None])
def test_a_record_must_state_its_reason(tmp_path, reason):
    base, blob = based(tmp_path)
    record(tmp_path, blob, {NODE: entry()}, reason=reason)
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, NODE: 81})
    code, result = budget_run(tmp_path, case('0'), base=base)
    assert code == 1 and 'migration' in result['budget']['reason']


@pytest.mark.parametrize('again_seconds,expected,outcome', [(120, 0, 'pass'), (123, 1, 'fail')])
def test_budget_rerun_uses_the_reviewed_migration_limit(tmp_path, again_seconds, expected, outcome):
    base, blob = based(tmp_path)
    record(tmp_path, blob, {NODE: entry()})
    policy(tmp_path, slow={'tests/test_sample.py::test_known': 40, NODE: 81})
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        report = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--junitxml=')))
        report.write_text(case(str(123 if len(calls) == 1 else again_seconds)))
        return SimpleNamespace(returncode=0)

    code = runner.run_plan({'selected': [{'path': 'tests/test_sample.py'}],
                            'deselect': [], 'base_sha': base},
                           root=tmp_path, output=tmp_path / 'result.json',
                           junit=tmp_path / 'junit.xml', execute=execute, affinity={1})
    result = json.loads((tmp_path / 'result.json').read_text())
    assert code == expected
    assert len(calls) == 2
    assert calls[1][-1] == str(tmp_path / 'tests/test_sample.py') + '::test_cost'
    assert result['budget_reruns'] == [{'nodeid': NODE, 'first_seconds': 123.0,
        'rerun_seconds': float(again_seconds), 'limit': 121.5, 'outcome': outcome}]
    assert result['budget']['status'] == outcome
    nested = json.loads((tmp_path / 'junit.budget-rerun-0.result.json').read_text())
    assert nested['budget']['baseline_sha'] == base
