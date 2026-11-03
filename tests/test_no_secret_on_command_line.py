"""Keep BuildBuddy authentication out of tracked commands and runner inputs."""
from fnmatch import fnmatch
from pathlib import Path
import re
import shlex
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SECRET_REFERENCE = re.compile(r'\$\{\{\s*secrets\.BUILDBUDDY_API_KEY\s*\}\}', re.I)
FORBIDDEN = [
    re.compile(r'--remote_header[\s\\\x27\x22,=]+[^\n]*(?:api[-_]key|BUILDBUDDY)', re.I),
    re.compile(r'x-buildbuddy-api-key\s*=\s*[^\s\x27\x22]+', re.I),
    re.compile(r'--(?:test_env|action_env|repo_env|host_action_env)[\s\\\x27\x22,=]+'
               r'[^\s\x27\x22\\,]*(?:BUILDBUDDY|API_KEY)', re.I),
]


def mapping(node, key):
    if isinstance(node, yaml.MappingNode):
        return next((value for name, value in node.value if name.value == key), None)
    return None


def violations(path, text):
    allowed = []
    if path.endswith(('.yml', '.yaml')):
        jobs = mapping(yaml.compose(text), 'jobs')
        if isinstance(jobs, yaml.MappingNode):
            for _, job in jobs.value:
                steps = mapping(job, 'steps')
                if not isinstance(steps, yaml.SequenceNode):
                    continue
                for step in steps.value:
                    env = mapping(step, 'env')
                    if (isinstance(env, yaml.MappingNode)
                            and step.start_mark.index <= env.start_mark.index < step.end_mark.index):
                        allowed.extend((value.start_mark.index, value.end_mark.index)
                                       for _, value in env.value if isinstance(value, yaml.ScalarNode)
                                       and env.start_mark.index <= value.start_mark.index < env.end_mark.index)
    errors = [match.start() for pattern in FORBIDDEN for match in pattern.finditer(text)]
    errors.extend(match.start() for match in SECRET_REFERENCE.finditer(text)
                  if not any(start <= match.start() < end for start, end in allowed))
    # Do not include matching source, which could itself contain a secret.
    return [f'{path}:{text.count(chr(10), 0, offset) + 1}: forbidden secret transport'
            for offset in errors]


def test_tracked_files_do_not_transport_buildbuddy_secrets():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    errors = []
    for path in tracked:
        if (path.startswith(('.github/', 'tools/')) or path in ('.bazelrc', 'MODULE.bazel')
                or fnmatch(path, '*.bazelrc')):
            errors.extend(violations(path, (ROOT / path).read_text(errors='replace')))
    assert not errors, '\n'.join(errors)


@pytest.mark.parametrize('text', [
    'bazel --remote_header=x-buildbuddy-api-key=fake',
    'bazel --remote_header "x-buildbuddy-api-key=$TOKEN"',
    '["--remote_header", "Authorization=$API_KEY"]',
    'x-buildbuddy-api-key=fake',
    *[f'bazel --{flag}={name}' for flag in
      ('test_env', 'action_env', 'repo_env', 'host_action_env')
      for name in ('BUILDBUDDY_API_KEY', 'OTHER_API_KEY')],
    '["--test_env", "BUILDBUDDY_API_KEY"]',
    '--action_env=\\"BUILDBUDDY_API_KEY\\"',
    '--repo_env \\\n        BUILDBUDDY_API_KEY',
    'echo "${{ secrets.BUILDBUDDY_API_KEY }}"',
])
def test_forbidden_transports_are_recognized(text):
    assert violations('tools/example.sh', text)


@pytest.mark.parametrize('location', ['workflow', 'job', 'run', 'with'])
def test_secret_expression_requires_step_env(location):
    reference = '${{ secrets.BUILDBUDDY_API_KEY }}'
    snippets = {
        'workflow': f'env: {{KEY: "{reference}"}}\njobs: {{}}\n',
        'job': f'jobs:\n  test:\n    env: {{KEY: "{reference}"}}\n',
        'run': f'jobs:\n  test:\n    steps:\n      - run: echo "{reference}"\n',
        'with': f'jobs:\n  test:\n    steps:\n      - with: {{key: "{reference}"}}\n',
    }
    assert violations('.github/workflows/example.yml', snippets[location])


def test_step_env_and_python_header_are_sanctioned():
    workflow = ('jobs:\n  test:\n    steps:\n      - env:\n'
                '          BUILDBUDDY_API_KEY: ${{ secrets.BUILDBUDDY_API_KEY }}\n'
                '        run: python scan.py\n')
    assert not violations('.github/workflows/example.yml', workflow)
    assert not violations('tools/client.py',
                          'headers = {"x-buildbuddy-api-key": os.environ["BUILDBUDDY_API_KEY"]}')


def test_every_upload_is_immediately_guarded_with_step_scoped_secrets():
    for filename in ('gate.yml', 'ci.yml'):
        workflow = yaml.safe_load((ROOT / '.github/workflows' / filename).read_text())
        for job in workflow['jobs'].values():
            steps = job.get('steps', [])
            for index, upload in enumerate(steps):
                if not upload.get('uses', '').startswith('actions/upload-artifact@'):
                    continue
                assert index > 0
                guard = steps[index - 1]
                arguments = shlex.split(guard.get('run', ''))
                assert arguments[:2] == ['python3', 'tools/ci/check_no_secret_leak.py']
                assert guard.get('env') == {
                    'BUILDBUDDY_API_KEY': '${{ secrets.BUILDBUDDY_API_KEY }}',
                    'MERGIFY_TOKEN': '${{ secrets.MERGIFY_TOKEN }}',
                }
                assert guard.get('if') == '${{ !cancelled() }}'
                assert upload.get('if') == (
                    "${{ !cancelled() && steps." + guard['id'] + ".outcome == 'success' }}")
                assert not guard.get('continue-on-error', False)
                assert upload.get('if', '') != 'always()'
                for path in upload['with']['path'].splitlines():
                    assert path in arguments
                assert arguments.count('--env') == 2


def _reusable_secret_workflow(*, target='./.github/workflows/bazel-hosted.yml',
                              binding='${{ secrets.BUILDBUDDY_API_KEY }}',
                              key='BUILDBUDDY_API_KEY'):
    return (f'jobs:\n  call:\n    uses: {target}\n'
            f'    secrets:\n      {key}: "{binding}"\n')


def test_exact_reusable_workflow_secret_binding_is_sanctioned():
    assert not violations('.github/workflows/gate.yml', _reusable_secret_workflow())


@pytest.mark.parametrize('target', [
    './.github/workflows/unreviewed.yml',
    'other/repo/.github/workflows/bazel-hosted.yml@main',
])
def test_reusable_secret_binding_requires_the_reviewed_local_workflow(target):
    assert violations('.github/workflows/gate.yml', _reusable_secret_workflow(target=target))


@pytest.mark.parametrize('key,binding', [
    ('OTHER_KEY', '${{ secrets.BUILDBUDDY_API_KEY }}'),
    ('BUILDBUDDY_API_KEY', 'prefix ${{ secrets.BUILDBUDDY_API_KEY }}'),
])
def test_reusable_secret_binding_is_an_exact_same_name_reference(key, binding):
    assert violations('.github/workflows/gate.yml',
                      _reusable_secret_workflow(key=key, binding=binding))


def test_a_reusable_binding_cannot_also_transport_the_secret_on_argv():
    text = _reusable_secret_workflow() + (
        '    steps:\n      - run: echo "${{ secrets.BUILDBUDDY_API_KEY }}"\n')
    assert violations('.github/workflows/gate.yml', text)


@pytest.mark.parametrize('location', ['run', 'with'])
def test_a_step_secret_yaml_alias_cannot_escape_to_a_command_or_action_input(location):
    text = ('jobs:\n  test:\n    steps:\n      - env:\n'
            '          BUILDBUDDY_API_KEY: &key "${{ secrets.BUILDBUDDY_API_KEY }}"\n')
    text += '        run: *key\n' if location == 'run' else '        with: {key: *key}\n'
    assert violations('.github/workflows/example.yml', text)


def test_a_reusable_secret_yaml_alias_cannot_escape_to_with():
    text = ('jobs:\n  call:\n    uses: ./.github/workflows/bazel-hosted.yml\n'
            '    secrets:\n'
            '      BUILDBUDDY_API_KEY: &key "${{ secrets.BUILDBUDDY_API_KEY }}"\n'
            '    with: {command: *key}\n')
    assert violations('.github/workflows/gate.yml', text)


def test_reusable_workflows_cannot_inherit_undeclared_secrets():
    text = ('jobs:\n  call:\n    uses: ./.github/workflows/bazel-hosted.yml\n'
            '    secrets: inherit\n')
    assert violations('.github/workflows/gate.yml', text)
