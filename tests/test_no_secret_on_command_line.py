"""Keep BuildBuddy authentication out of tracked commands and runner inputs."""
from fnmatch import fnmatch
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType

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
    allowed, errors = [], [match.start() for pattern in FORBIDDEN for match in pattern.finditer(text)]
    if path.endswith(('.yml', '.yaml')):
        document = yaml.compose(text)
        jobs = mapping(document, 'jobs')

        def sanctioned(node, context):
            if (len(context) == 6 and context[0] == 'jobs' and context[2] == 'steps'
                    and isinstance(context[3], int) and context[4] == 'env'):
                return True
            if (len(context) != 4 or context[0] != 'jobs'
                    or context[2:] != ('secrets', 'BUILDBUDDY_API_KEY')):
                return False
            job = mapping(jobs, context[1])
            target = mapping(job, 'uses')
            return (isinstance(target, yaml.ScalarNode)
                    and target.value == './.github/workflows/bazel-hosted.yml'
                    and all(mapping(job, name) is None for name in ('steps', 'run', 'env'))
                    and node.value == '${{ secrets.BUILDBUDDY_API_KEY }}')

        def visit(node, context=(), active=frozenset()):
            if node is None:
                return
            if id(node) in active:
                errors.append(node.start_mark.index)
                return
            active = active | {id(node)}
            if isinstance(node, yaml.MappingNode):
                keys = [key.value for key, _ in node.value if isinstance(key, yaml.ScalarNode)]
                if len(keys) != len(node.value) or len(keys) != len(set(keys)):
                    errors.append(node.start_mark.index)
                for key, value in node.value:
                    visit(key, context + ('<key>',), active)
                    visit(value, context + (key.value,), active)
            elif isinstance(node, yaml.SequenceNode):
                for index, value in enumerate(node.value):
                    visit(value, context + (index,), active)
            elif isinstance(node, yaml.ScalarNode):
                if (len(context) == 3 and context[0] == 'jobs'
                        and context[2] == 'secrets' and node.value == 'inherit'):
                    errors.append(node.start_mark.index)
                if SECRET_REFERENCE.search(node.value):
                    if sanctioned(node, context):
                        allowed.append((node.start_mark.index, node.end_mark.index))
                    else:
                        errors.append(node.start_mark.index)

        # Visit aliases in every context: an env anchor reused as run/with is unsafe.
        visit(document)
    errors.extend(match.start() for match in SECRET_REFERENCE.finditer(text)
                  if not any(start <= match.start() < end for start, end in allowed))
    # Do not include matching source, which could itself contain a secret.
    return [f'{path}:{text.count(chr(10), 0, offset) + 1}: forbidden secret transport'
            for offset in sorted(set(errors))]


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


UPLOADS = {
    ('gate.yml', 'remote'): ('bazel-upload', None),
    ('gate.yml', 'graph-impact'): ('graph-upload', 'graph-impact/result.json'),
    ('ci.yml', 'tests'): ('bazel-main-upload', None),
    ('bazel-hosted.yml', 'tests'): ('hosted-upload', 'hosted-private/result.json'),
}
SCAN_KEYS = ['BUILDBUDDY_API_KEY', 'MERGIFY_TOKEN']
SCAN_CONDITIONS = {
    'bazel-upload': "${{ !cancelled() && env.TRUSTED == 'true' && steps.controller.outputs.scopes_ready == 'true' }}",
    'bazel-main-upload': "${{ !cancelled() && env.TRUSTED == 'true' && steps.controller.outputs.scopes_ready == 'true' }}",
    'graph-upload': "${{ !cancelled() && env.TRUSTED == 'true' && steps.check.outcome == 'success' }}",
    'hosted-upload': "${{ !cancelled() && inputs.trusted && steps.tests.outcome == 'success' }}",
}


def _assert_upload_guard(steps, index, directory, source, monkeypatch, tmp_path):
    assert index > 0
    upload, guard = steps[index], steps[index - 1]
    assert guard.get('env') == {key: '${{ secrets.' + key + ' }}' for key in SCAN_KEYS}
    assert guard.get('if') == SCAN_CONDITIONS[directory]
    assert not guard.get('continue-on-error', False)
    assert not upload.get('continue-on-error', False)
    assert guard.get('id') == 'scan'
    assert upload.get('if') == "${{ !cancelled() && steps.scan.outcome == 'success' }}"
    assert upload['with']['path'] == directory + '/'
    assert upload['with']['if-no-files-found'] == 'error'
    python = [step for step in steps[:index - 1]
              if step.get('uses', '').startswith('actions/setup-python@')][-1]
    assert python['id'] == 'scan-python'
    assert python['with']['python-version'] == '3.14'
    lines = guard['run'].strip().splitlines()
    assert lines[:2] == ['python tests/bazel_scan_runtime.py', "python - <<'SCAN_JSON'"]
    assert lines[-1] == 'SCAN_JSON'
    script = '\n'.join(lines[2:-1])
    module = ModuleType('tools.ci.bazel_artifacts')
    called = []

    def scan_upload(path, keys):
        assert source is None
        assert path == Path(directory)
        assert keys == SCAN_KEYS
        called.append('scan')
        return 93

    class Refused(RuntimeError):
        pass

    def publish_json(path, documents, keys):
        assert source is not None
        assert path == Path(directory)
        assert documents == {'result.json': b'{"verdict":"pass"}\n'}
        assert keys == SCAN_KEYS
        called.append('publish')
        raise Refused('synthetic scanner refusal')

    module.scan_upload, module.publish_json = scan_upload, publish_json
    monkeypatch.setitem(sys.modules, 'tools.ci.bazel_artifacts', module)
    monkeypatch.chdir(tmp_path)
    if source is not None:
        input_path = tmp_path / source
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(b'{"verdict":"pass"}\n')
        with pytest.raises(Refused):
            exec(compile(script, '<workflow-scan>', 'exec'), {})
        assert called == ['publish']
    else:
        with pytest.raises(SystemExit) as refused:
            exec(compile(script, '<workflow-scan>', 'exec'), {})
        assert refused.value.code == 93
        assert called == ['scan']


def test_every_upload_is_immediately_guarded_with_step_scoped_secrets(monkeypatch, tmp_path):
    seen = set()
    for filename in ('gate.yml', 'ci.yml', 'bazel-hosted.yml'):
        workflow = yaml.safe_load((ROOT / '.github/workflows' / filename).read_text())
        for name, job in workflow['jobs'].items():
            steps = job.get('steps', [])
            for index, upload in enumerate(steps):
                if upload.get('uses', '').startswith('actions/upload-artifact@'):
                    seen.add((filename, name))
                    directory, source = UPLOADS[filename, name]
                    _assert_upload_guard(steps, index, directory, source, monkeypatch, tmp_path)
    assert seen == set(UPLOADS)


@pytest.mark.parametrize('filename,job', [('gate.yml', 'remote'), ('bazel-hosted.yml', 'tests')])
@pytest.mark.parametrize('defect', [
    'wrong-directory', 'missing-scan-key', 'swallowed-refusal', 'missing-runtime',
    'wrong-python', 'unconditional-upload', 'missing-step-secret', 'unguarded-scan',
])
def test_upload_guard_rejects_weakened_scan_contract(filename, job, defect, monkeypatch, tmp_path):
    workflow = yaml.safe_load((ROOT / '.github/workflows' / filename).read_text())
    steps = workflow['jobs'][job]['steps']
    index = next(index for index, step in enumerate(steps)
                 if step.get('uses', '').startswith('actions/upload-artifact@'))
    upload, guard = steps[index], steps[index - 1]
    directory, source = UPLOADS[filename, job]
    if defect == 'wrong-directory':
        guard['run'] = guard['run'].replace(directory, 'other-upload')
    elif defect == 'missing-scan-key':
        guard['run'] = guard['run'].replace(", 'MERGIFY_TOKEN'", '')
    elif defect == 'swallowed-refusal':
        lines = guard['run'].strip().splitlines()
        guard['run'] = '\n'.join(lines[:2] + ['try:']
                                 + ['    ' + line for line in lines[2:-1]]
                                 + ['except BaseException:', '    pass', lines[-1]])
    elif defect == 'missing-runtime':
        guard['run'] = guard['run'].replace('python tests/bazel_scan_runtime.py\n', '')
    elif defect == 'wrong-python':
        next(step for step in steps if step.get('id') == 'scan-python')['with']['python-version'] = '3.12'
    elif defect == 'unconditional-upload':
        upload['if'] = '${{ always() }}'
    elif defect == 'missing-step-secret':
        del guard['env']['MERGIFY_TOKEN']
    elif defect == 'unguarded-scan':
        guard['if'] = '${{ !cancelled() }}'
    # Both assertion failures and the explicit missing-refusal failure reject a mutation.
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        _assert_upload_guard(steps, index, directory, source, monkeypatch, tmp_path)


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
