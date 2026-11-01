"""Hosted CI must cover integrated batches and preserve the post-merge suite."""
from pathlib import Path
import re

import yaml

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
