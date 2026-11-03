"""Hosted CI must cover integrated batches and preserve the post-merge suite."""
from pathlib import Path
import os
import re

import yaml
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gate_uses_authenticated_batch_heads_and_no_label_events():
    workflow = yaml.safe_load((ROOT / '.github/workflows/gate.yml').read_text())
    assert workflow['on']['pull_request']['types'] == ['opened', 'synchronize', 'reopened']
    mode = workflow['env']['GATE_MODE']
    for required in ['head.repo.full_name == github.repository', 'mergify/merge-queue/',
                     "user.login == 'mergify[bot]'", 'user.id == 37929162', 'gate-canary']:
        assert required in mode
    assert {'gate', 'pr-gate'} <= workflow['jobs'].keys()
    assert int(workflow['env']['GATE_SHARDS']) in (12, 16)
    assert 'ubuntu-24.04-arm' in workflow['env']['GATE_RUNNERS']


def test_actions_use_commit_pins_and_all_jobs_use_hosted_runners():
    for filename in ['gate.yml', 'ci.yml']:
        workflow = yaml.safe_load((ROOT / '.github/workflows' / filename).read_text())
        for job in workflow['jobs'].values():
            assert 'self-hosted' not in str(job['runs-on'])
            for step in job.get('steps', []):
                if uses := step.get('uses'):
                    assert re.fullmatch(r'[^@]+@[0-9a-f]{40}', uses), uses


def test_cpu_full_suite_only_runs_after_merge_and_keeps_its_name():
    workflow = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
    assert workflow['jobs']['tests']['if'] == "github.event_name != 'pull_request'"
    assert workflow['jobs']['tests']['name'] == 'tests (py${{ matrix.python }}, cpu)'
    assert workflow['jobs']['imports']['name'] == 'import surface'
    assert workflow['jobs']['allowlists']['name'] == 'boundary allowlists only shrink'
    assert not (ROOT / '.github/hosted-known-failures.txt').exists()


def test_gate_pin_resolution_refuses_a_failed_resolver(tmp_path):
    import os
    import subprocess
    workflow = yaml.safe_load((ROOT / '.github/workflows/gate.yml').read_text())
    for job in ('shards', 'retry-shards'):
        command = next(step['run'] for step in workflow['jobs'][job]['steps']
                       if step.get('name') == 'Resolve Tessera development pin')
        (tmp_path / 'tools').mkdir(exist_ok=True)
        (tmp_path / 'tools/resolve_tessera_dev_pin.py').write_text('raise SystemExit(9)\n')
        output = tmp_path / 'github-output'
        result = subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', command], cwd=tmp_path,
            env={**os.environ, 'GITHUB_OUTPUT': str(output)}, capture_output=True, text=True)
        assert result.returncode != 0
        assert not output.exists() or not output.read_text()


def test_gate_uses_owned_temporary_space_for_both_attempts(tmp_path):
    import os
    import subprocess
    workflow = yaml.safe_load((ROOT / '.github/workflows/gate.yml').read_text())
    for job in ('shards', 'retry-shards'):
        definition = workflow['jobs'][job]
        assert 'runner.temp' not in str(definition.get('env', {}))
        steps = definition['steps']
        configured = next(index for index, step in enumerate(steps)
                          if step.get('name') == 'Configure owned temporary space')
        assert configured < next(index for index, step in enumerate(steps)
                                 if step.get('name') == 'Run assigned gate tests')
        exported = tmp_path / (job + '-env')
        subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', steps[configured]['run']],
            env={**os.environ, 'GITHUB_ENV': str(exported), 'RUNNER_TEMP': str(tmp_path)}, check=True)
        assert exported.read_text() == 'TMPDIR=' + str(tmp_path) + '\n'


@pytest.mark.parametrize('mode,attempt,first', [
    ('selection', 1, 'ubuntu-24.04-arm'),
    ('selection', 2, 'ubuntu-24.04-arm'),
    ('full', 1, 'ubuntu-24.04-arm'),
    ('full', 2, 'ubuntu-24.04'),
])
def test_initial_placement_prefers_arm_and_full_reruns_swap(mode, attempt, first):
    import json
    import os
    import subprocess
    import sys
    workflow = yaml.safe_load((ROOT / '.github/workflows/gate.yml').read_text())
    command = next(step['run'] for step in workflow['jobs']['plan']['steps']
                   if 'runners=$(' in step.get('run', ''))
    script = command.split("runners=$(python - <<'PYTHON'\n", 1)[1].split('\nPYTHON\n', 1)[0]
    result = subprocess.check_output([sys.executable, '-c', script], text=True,
        env={**os.environ, 'GATE_MODE': mode, 'GITHUB_RUN_ATTEMPT': str(attempt),
             'GATE_RUNNERS': workflow['env']['GATE_RUNNERS']})
    assert result.strip().split(',')[0] == first


def test_control_jobs_include_the_shared_digest_owner_in_sparse_checkout():
    workflow = yaml.safe_load((ROOT / '.github/workflows/gate.yml').read_text())
    for job in ('retry-plan', 'pr-gate', 'gate'):
        checkout = next(step for step in workflow['jobs'][job]['steps']
                        if step.get('uses', '').startswith('actions/checkout@'))
        paths = checkout['with']['sparse-checkout'].splitlines()
        assert 'tools/ci' in paths
        assert 'prismaquant' in paths, 'node ownership uses prismaquant.digests before dependency setup'


def _post_merge_plugin_runs():
    workflow = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
    steps = workflow['jobs']['tests']['steps']
    run_tests = next(step for step in steps if step.get('name') == 'Run tests')
    scoped = next(step for step in steps if step.get('name', '').startswith('Run cgroup-reading tests'))
    yield run_tests['env']['PYTHONPATH'], run_tests['run']
    path = re.search(r'PYTHONPATH=(\S+)', scoped['run']).group(1)
    yield path, scoped['run']


@pytest.mark.parametrize('pythonpath,command', list(_post_merge_plugin_runs()),
                         ids=['suite', 'own-cgroup-scope'])
def test_post_merge_suite_can_load_the_gate_collection_plugin(tmp_path, pythonpath, command):
    # Both post-merge pytest commands load the gate's collection plugin with
    # the step's own PYTHONPATH. A plugin that can't import its siblings fails
    # every push to main before any test runs.
    import subprocess
    import sys
    plugins = re.findall(r'-p (\S+)', command)
    plugin = next(name for name in plugins if 'gate_collect' in name)
    (tmp_path / 'test_probe.py').write_text('def test_probe():\n    pass\n')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        str(ROOT / part) if part != '.' else str(ROOT) for part in pythonpath.split(':')))
    env.pop('GATE_ASSIGNED_FILES', None)
    env.pop('GATE_DESELECT_NODEIDS', None)
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-c', str(ROOT / 'pytest.ini'), '--rootdir', str(ROOT),
         '-p', 'no:cacheprovider', '-p', plugin, '-q', str(tmp_path / 'test_probe.py')],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_both_shard_attempts_upload_verdict_and_warning_evidence():
    from fnmatch import fnmatch
    workflow = yaml.safe_load((ROOT / '.github/workflows/gate.yml').read_text())
    for job in ('shards', 'retry-shards'):
        upload = next(step for step in workflow['jobs'][job]['steps']
                      if step.get('name') == 'Upload verdict and JUnit')
        patterns = upload['with']['path'].splitlines()
        for filename in ('gate-artifacts/result.json', 'gate-artifacts/junit.xml',
                         'gate-artifacts/junit.cgroup.xml', 'gate-artifacts/junit.collection.json'):
            assert any(fnmatch(filename, pattern) or filename.startswith(pattern.rstrip('/') + '/')
                       for pattern in patterns), (job, filename)


def test_graph_check_is_a_required_parallel_job_with_full_history():
    jobs = yaml.safe_load((ROOT / '.github/workflows/gate.yml').read_text())['jobs']
    job = jobs['graph-impact']
    assert not set(job.get('needs', [])) & {'shards', 'retry-shards'}
    checkout = next(s for s in job['steps'] if s.get('uses', '').startswith('actions/checkout@'))
    assert checkout['with']['fetch-depth'] == 0
    assert checkout['with']['ref'] == '${{ github.event.pull_request.head.sha }}'
    run = next(s for s in job['steps'] if s.get('id') == 'check')
    assert 'tests/test_graph_impact.py::test_graph_impact_gate' in run['run']
    assert '--noconftest' in run['run']
    assert 'gate_run.py' not in run['run']
    assert job['outputs']['verdict'] == '${{ steps.check.outputs.verdict }}'
    for name in ('pr-gate', 'gate'):
        assert 'graph-impact' in jobs[name]['needs']
    for name in ('plan', 'shards', 'retry-shards'):
        command = next(step['run'] for step in jobs[name]['steps']
                       if any('gate_partition.py ' + command in step.get('run', '')
                              for command in ('matrix', 'shard')))
        assert '--plan gate-artifacts/plan.json' in command


@pytest.mark.parametrize('result,verdict,passes', [
    ('success', 'pass', True), ('success', '', False), ('failure', 'pass', False),
    ('skipped', '', False), ('cancelled', '', False), ('', '', False),
])
def test_graph_aggregate_refuses_failed_or_missing_verdict(result, verdict, passes, tmp_path):
    import subprocess
    jobs = yaml.safe_load((ROOT / '.github/workflows/gate.yml').read_text())['jobs']
    for name in ('pr-gate', 'gate'):
        step = next(s for s in jobs[name]['steps'] if s.get('name') == 'Require graph impact verdict')
        assert step['env']['GRAPH_RESULT'] == '${{ needs.graph-impact.result }}'
        assert step['env']['GRAPH_VERDICT'] == '${{ needs.graph-impact.outputs.verdict }}'
        run = subprocess.run(['bash', '-e', '-c', step['run']], capture_output=True, text=True,
            env={**os.environ, 'GRAPH_RESULT': result, 'GRAPH_VERDICT': verdict}, cwd=tmp_path)
        assert (run.returncode == 0) is passes, run.stdout + run.stderr


@pytest.mark.parametrize('cases,passes', [
    ('<testcase name="test_graph_impact_gate"/>', True), ('', False),
    ('<testcase name="other"/>', False),
    ('<testcase name="test_graph_impact_gate"><skipped/></testcase>', False), ('<testcase name="test_graph_impact_gate"><failure/></testcase>', False),
    ('<testcase name="test_graph_impact_gate"><error/></testcase>', False), ('<testcase/><testcase/>', False),
])
def test_graph_job_publishes_only_one_passing_test(cases, passes, tmp_path):
    import subprocess
    import sys
    jobs = yaml.safe_load((ROOT / '.github/workflows/gate.yml').read_text())['jobs']
    command = next(s['run'] for s in jobs['graph-impact']['steps'] if s.get('id') == 'check')
    script = command.split("python - <<'GRAPH_VERDICT'\n", 1)[1].split('\nGRAPH_VERDICT', 1)[0]
    (tmp_path / 'graph-impact').mkdir()
    (tmp_path / 'graph-impact/junit.xml').write_text('<testsuites><testsuite>' + cases + '</testsuite></testsuites>')
    output = tmp_path / 'output'
    run = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, capture_output=True, text=True,
                          env={**os.environ, 'GITHUB_OUTPUT': str(output)})
    assert (run.returncode == 0) is passes, run.stdout + run.stderr
    assert (output.read_text() if output.exists() else '') == ('verdict=pass\n' if passes else '')
