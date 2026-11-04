"""The cutover's required checks join their own remote, hosted and graph outcomes."""
import os
from pathlib import Path
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def required_gate_steps():
    jobs = yaml.safe_load((ROOT / '.github/workflows/gate.yml').read_text())['jobs']
    return [next(step for step in jobs[name]['steps']
                 if step.get('name') == 'Require complete cutover evidence')
            for name in ('pr-gate', 'gate')]


@pytest.mark.parametrize('trusted,remote_result,remote_verdict,passes', [
    ('true', 'success', 'pass', True),
    ('true', 'success', 'skipped', True),
    ('true', 'success', 'fail', False),
    ('true', 'success', '', False),
    ('true', 'failure', 'pass', False),
    ('true', 'cancelled', 'pass', False),
    ('true', 'skipped', '', False),
    ('true', '', '', False),
    ('false', 'skipped', '', True),
    ('false', 'failure', '', False),
    ('false', 'success', 'pass', False),
])
def test_required_checks_accept_only_the_expected_remote_or_fork_verdict(
        tmp_path, trusted, remote_result, remote_verdict, passes):
    for step in required_gate_steps():
        result = subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', step['run']],
            cwd=tmp_path, capture_output=True, text=True, env={
                'PATH': os.defpath, 'TRUSTED': trusted,
                'REMOTE_RESULT': remote_result, 'REMOTE_VERDICT': remote_verdict,
                'HOSTED_RESULT': 'success', 'HOSTED_VERDICT': 'pass',
                'GRAPH_RESULT': 'success', 'GRAPH_VERDICT': 'pass'})
        assert (result.returncode == 0) is passes, result.stdout + result.stderr


@pytest.mark.parametrize('changed,value', [
    ('HOSTED_RESULT', 'failure'), ('HOSTED_RESULT', 'cancelled'),
    ('HOSTED_RESULT', 'skipped'), ('HOSTED_RESULT', ''),
    ('HOSTED_VERDICT', 'fail'), ('HOSTED_VERDICT', ''),
    ('GRAPH_RESULT', 'failure'), ('GRAPH_RESULT', 'skipped'),
    ('GRAPH_RESULT', ''), ('GRAPH_VERDICT', 'fail'), ('GRAPH_VERDICT', ''),
    ('TRUSTED', ''), ('TRUSTED', 'unexpected'),
])
def test_required_checks_refuse_missing_or_failed_sibling_evidence(tmp_path, changed, value):
    environment = {'PATH': os.defpath, 'TRUSTED': 'true',
                   'REMOTE_RESULT': 'success', 'REMOTE_VERDICT': 'pass',
                   'HOSTED_RESULT': 'success', 'HOSTED_VERDICT': 'pass',
                   'GRAPH_RESULT': 'success', 'GRAPH_VERDICT': 'pass'}
    environment[changed] = value
    for step in required_gate_steps():
        result = subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', step['run']],
            cwd=tmp_path, capture_output=True, text=True, env=environment)
        assert result.returncode != 0, result.stdout + result.stderr


def test_receipt_downloads_have_exact_names_and_explicit_destinations():
    found = []
    for name in ('gate.yml', 'ci.yml', 'bazel-hosted.yml'):
        workflow = yaml.safe_load((ROOT / '.github/workflows' / name).read_text())
        for job in workflow['jobs'].values():
            for step in job.get('steps', []):
                if step.get('uses', '').startswith('actions/download-artifact@'):
                    settings = step['with']
                    assert settings.get('name') and settings.get('path')
                    assert 'pattern' not in settings and not settings.get('merge-multiple')
                    found.append((name, settings['name'], settings['path']))
    assert found, 'hosted lanes must consume the exact scanned controller receipt'


def test_shared_hosted_lane_is_used_by_pr_and_both_postmerge_architectures():
    gate = yaml.safe_load((ROOT / '.github/workflows/gate.yml').read_text())
    ci = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
    hosted = './.github/workflows/bazel-hosted.yml'
    pr = [job for job in gate['jobs'].values() if job.get('uses') == hosted]
    main = [job for job in ci['jobs'].values() if job.get('uses') == hosted]
    assert len(pr) == 1 and pr[0]['with']['architecture'] == 'amd64'
    assert {job['with']['architecture'] for job in main} == {'amd64', 'arm64'}
    assert all(job['with']['mode'] == 'main' for job in main)
    assert all(job['with']['mode'] == 'pr' for job in pr)
