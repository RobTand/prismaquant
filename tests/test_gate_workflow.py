"""The cutover preserves required verdicts, hosted exceptions and policy checks."""
from pathlib import Path
import re

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflows():
    return {name: yaml.safe_load((ROOT / '.github/workflows' / name).read_text())
            for name in ('gate.yml', 'ci.yml', 'bazel-hosted.yml')}


def test_gate_uses_authenticated_batch_heads_and_no_label_events():
    gate = workflows()['gate.yml']
    assert gate['on']['pull_request']['types'] == ['opened', 'synchronize', 'reopened']
    for required in ('head.repo.full_name == github.repository', 'mergify/merge-queue/',
                     "user.login == 'mergify[bot]'", 'user.id == 37929162', 'gate-canary'):
        assert required in gate['env']['GATE_MODE']
    assert {'gate', 'graph-impact', 'remote', 'hosted'} <= gate['jobs'].keys()
    assert not {'retry-plan', 'retry-shards', 'shards'} & gate['jobs'].keys()


def test_actions_are_pinned_and_execution_jobs_stay_on_github_hosted_runners():
    for workflow in workflows().values():
        for job in workflow['jobs'].values():
            if 'uses' in job:
                assert job['uses'] == './.github/workflows/bazel-hosted.yml'
                continue
            assert 'self-hosted' not in str(job['runs-on'])
            for step in job.get('steps', []):
                if uses := step.get('uses'):
                    assert re.fullmatch(r'[^@]+@[0-9a-f]{40}', uses), uses


def test_job_environment_does_not_reference_runtime_only_contexts():
    for name, workflow in workflows().items():
        for job_name, job in workflow['jobs'].items():
            for variable, value in job.get('env', {}).items():
                for expression in re.findall(r'\$\{\{(.*?)\}\}', str(value)):
                    assert not re.search(r'\b(?:runner|steps|job|env)\s*\.', expression), (
                        f'{name}: jobs.{job_name}.env.{variable} references a context '
                        'available only after the job starts; set it in a step instead')


def test_remote_planner_uses_queue_aware_refs_and_requires_scope_upload():
    jobs = workflows()['gate.yml']['jobs']
    remote = jobs['remote']
    # A stop job cancels the run at the first red job; gate still runs and reports red.
    assert jobs['gate']['if'] == remote['if'].replace('!cancelled()', 'always()')
    # Only PR and queue gates stop early; records (ci.yml, the hosted lane workflow) run every lane to the end.
    assert set(workflows()['gate.yml']['on']) == {'pull_request'}
    assert not any('gh run cancel' in (ROOT / '.github/workflows' / name).read_text()
                   for name in ('ci.yml', 'bazel-hosted.yml'))
    for name in ('remote', 'hosted', 'graph-impact'):
        stop = jobs['stop-after-' + name]
        assert (stop['needs'], stop['if'], stop['permissions']) == (name, '${{ failure() }}', {'actions': 'write'})
        assert stop['steps'][0]['run'] == 'gh run cancel "$GITHUB_RUN_ID" -R "$GITHUB_REPOSITORY"'
    assert jobs['hosted']['if'] == remote['if'] and 'needs' not in jobs['hosted']
    steps = {step.get('id'): step for step in remote['steps']}
    assert steps['refs']['with']['action'] == 'scopes-git-refs'
    assert steps['scopes']['with']['action'] == 'scopes-upload'
    assert steps['scopes']['with']['token'] == '${{ secrets.MERGIFY_TOKEN }}'
    assert steps['controller']['env']['BASE'] == '${{ steps.refs.outputs.base }}'
    assert steps['controller']['env']['HEAD'] == '${{ steps.refs.outputs.head }}'
    assert 'remote_gate.py pr' in steps['controller']['run']
    # A cancelled run kills the controller; a cancelled() step cancels the shards it recorded.
    cancel = next(step for step in remote['steps'] if 'CANCEL_SHARDS' in step.get('run', ''))
    assert cancel['if'] == '${{ cancelled() }}' and 'cancel_runs' in cancel['run']
    assert cancel['env']['REMOTE_SHARDS'] == steps['controller']['env']['REMOTE_SHARDS']
    assert 'MERGIFY_TOKEN' in steps['token']['env']
    assert 'scopes_ready' in steps['token']['if']
    assert 'steps.scopes.outcome' in steps['scope-verdict']['env']['OUTCOME']
    assert 'github.event.pull_request.head.repo.full_name == github.repository' in remote['if']


def test_postmerge_has_one_serial_controller_for_both_architectures():
    ci = workflows()['ci.yml']
    controller = ci['jobs']['tests']
    assert controller['if'] == "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
    assert ci['on']['schedule'] == [{'cron': '17 */6 * * *'}]  # Not at every push (#3763).
    assert {ci['jobs'][job]['if'] for job in ('hosted-amd64', 'hosted-arm64')} == {"github.event_name == 'push'"}
    checkout = next(step for step in controller['steps']
                    if step.get('uses', '').startswith('actions/checkout@'))
    assert checkout.get('with', {}).get('fetch-depth') == 0, 'historical-main graph contract needs full history'
    assert 'strategy' not in controller
    command = next(s['run'] for s in controller['steps'] if s.get('id') == 'controller')
    assert 'remote_gate.py main' in command
    assert ci['jobs']['imports']['name'] == 'import surface'
    assert ci['jobs']['allowlists']['name'] == 'boundary allowlists only shrink'
    assert not (ROOT / '.github/hosted-known-failures.txt').exists()


def test_hosted_lane_preserves_python_floor_policy_and_own_cgroup_routing():
    job = workflows()['bazel-hosted.yml']['jobs']['tests']
    steps = job['steps']
    setup = next(s for s in steps if s.get('id') == 'test-python')
    assert setup['with']['python-version'] == '3.12'
    execute = next(s for s in steps if s.get('id') == 'tests')
    assert 'gate_run.py --plan hosted-private/plan.json' in execute['run']
    assert '--assignment' not in execute['run']
    assert 'gate_partition.py' not in next(s for s in steps if s.get('id') == 'plan')['run']
    assert not any(key in execute.get('env', {}) for key in ('BUILDBUDDY_API_KEY', 'MERGIFY_TOKEN', 'GITHUB_TOKEN'))
    plan = next(s for s in steps if s.get('id') == 'plan')
    assert '--mode full' in plan['run'] and 'gate_select.py' in plan['run']
    assert 'bazel_gate.py hosted' in plan['run']
    assert '--architecture "$ARCHITECTURE"' in plan['run']
    scratch = next(s for s in steps if s.get('name') == 'Configure owned temporary space')
    assert 'printf \'TMPDIR=%s\\n\' "$RUNNER_TEMP" >> "$GITHUB_ENV"' in scratch['run']
    assert steps.index(scratch) < steps.index(plan)
    assert job['runs-on'] == "${{ inputs.architecture == 'arm64' && 'ubuntu-24.04-arm' || 'ubuntu-24.04' }}"
    assert '--timeout' not in execute['run'], 'per-test hard timeout remains gate_run policy'


def test_uploads_have_python314_scan_and_no_fork_uploads():
    for workflow in workflows().values():
        for job in workflow['jobs'].values():
            steps=job.get('steps', [])
            for index, step in enumerate(steps):
                if step.get('uses', '').startswith('actions/upload-artifact@'):
                    assert "steps.scan.outcome == 'success'" in step['if']
                    scan=next(s for s in steps[:index] if s.get('id')=='scan')
                    assert 'scan_upload' in scan['run'] or 'publish_json' in scan['run']
                    assert 'python tests/bazel_scan_runtime.py' in scan['run']
                    assert 'BUILDBUDDY_API_KEY' in scan['env']
                    assert 'MERGIFY_TOKEN' in scan['env']
                    py=next(s for s in steps[:index] if s.get('id')=='scan-python')
                    assert py['with']['python-version']=='3.14'
                    assert 'trusted' in scan.get('if', '') or 'TRUSTED' in scan.get('if', '')
                    assert step['with']['path'].endswith('upload/')


def test_graph_check_remains_parallel_and_requires_exactly_one_pass():
    jobs=workflows()['gate.yml']['jobs']
    graph=jobs['graph-impact']
    assert not graph.get('needs')
    checkout=next(s for s in graph['steps'] if s.get('uses', '').startswith('actions/checkout@'))
    assert checkout['with']['fetch-depth']==0
    check=next(s for s in graph['steps'] if s.get('id')=='check')
    assert 'tools/ci/static_checks.sh' in check['run']
    # One list of the checks that execute no test: the gate's job and a lead's pre-queue run read the same script.
    script = (ROOT / 'tools/ci/static_checks.sh').read_text()
    for entry in ('tools/ci/bazel_generated.py', 'tools/ci/gate_select.py', 'tools/ci/graph_impact_pq.py', '--merge-main'):
        assert entry in script
    assert 'bazel_generated.py' not in (ROOT / '.github/workflows/bazel-hosted.yml').read_text()
    assert 'graph-impact' in jobs['gate']['needs']
