"""Hosted CI must cover integrated batches and preserve the post-merge suite."""
from pathlib import Path
import os
import re

import yaml
import pytest

ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_SOURCE = "(github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)"
FORK_SOURCE = "github.event_name == 'pull_request' && github.event.pull_request.head.repo.full_name != github.repository"
FORK_NOTICE = 'artifact uploads skipped: fork PR has no secrets to scan for'


def test_artifact_uploads_require_available_secrets_and_repository_source():
    import shlex
    for filename in ('gate.yml', 'ci.yml'):
        workflow = yaml.safe_load((ROOT / '.github/workflows' / filename).read_text())
        for job in workflow['jobs'].values():
            steps = job.get('steps', [])
            for index, upload in enumerate(steps):
                if not upload.get('uses', '').startswith('actions/upload-artifact@'):
                    continue
                scanner = steps[index - 1]
                assert scanner['if'] == '${{ !cancelled() && ' + ARTIFACT_SOURCE + ' }}'
                assert upload['if'] == ('${{ !cancelled() && ' + ARTIFACT_SOURCE
                                         + ' && steps.' + scanner['id'] + ".outcome == 'success' }}")
                command = shlex.split(scanner['run'])
                assert command[:2] == ['python3', 'tools/ci/check_no_secret_leak.py']
                for name in ('BUILDBUDDY_API_KEY', 'MERGIFY_TOKEN'):
                    assert any(command[i:i + 2] == ['--require', name]
                               for i in range(len(command) - 1))
                    assert scanner['env'][name] == '${{ secrets.' + name + ' }}'


def test_forks_skip_artifact_consumers_and_report_no_verdict():
    import subprocess
    workflow = yaml.safe_load((ROOT / '.github/workflows/gate.yml').read_text())
    jobs = workflow['jobs']
    for name in ('shards', 'retry-plan'):
        assert ARTIFACT_SOURCE in jobs[name]['if']
    assert "needs.retry-plan.result == 'success'" in jobs['retry-shards']['if']
    assert 'head.repo.full_name == github.repository' in jobs['gate']['if']
    notice = jobs['plan']['steps'][0]
    assert notice['if'] == '${{ !cancelled() && ' + FORK_SOURCE + ' }}'
    assert subprocess.check_output(['bash', '-e', '-c', notice['run']], text=True).strip() == FORK_NOTICE
    refusal = jobs['pr-gate']['steps'][0]
    assert refusal['if'] == '${{ !cancelled() && ' + FORK_SOURCE + ' }}'
    result = subprocess.run(['bash', '-e', '-c', refusal['run']], capture_output=True, text=True)
    assert result.returncode == 75
    assert FORK_NOTICE in result.stdout
    assert 'NO_VERDICT: fork PR gate requires plan and shard artifacts; uploads are disabled' in result.stdout


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
