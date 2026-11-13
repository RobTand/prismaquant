"""Derived Bazel inputs fail CI when stale, and the runner reads hosted-style JUnit (#3216)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bazel_generated = load('bazel_generated', 'tools/ci/bazel_generated.py')
requirements = load('bazel_requirements', 'tools/bazel/requirements.py')
python_runner = load('bazel_python_runner', 'tools/bazel/python_runner.py')
runtime = load('bazel_runtime', 'tools/bazel/runtime.py')


def test_the_gate_environment_is_private_scratch_and_names_no_box_state(tmp_path):
    environ = runtime.gate_environment({'PATH': '/usr/bin', 'TEST_TMPDIR': '/bazel/tmp'}, ROOT, tmp_path)
    assert environ['TMPDIR'] == environ['HOME'] == str(tmp_path)
    assert 'TEST_TMPDIR' not in environ
    assert not [name for name in environ if name.startswith('PRISMABUILD_')]


def test_every_generated_input_has_a_check_and_one_fix():
    commands = [' '.join(map(str, command)) for command in bazel_generated.commands()]
    assert any('tools/bazel/requirements.py --check' in command for command in commands)
    assert any('tools/bazel/build_files.py --check' in command for command in commands)


def test_requirements_follow_what_the_hosted_gate_installs():
    for platform in requirements.PLATFORMS:
        export = (ROOT / requirements.EXPORT.format(platform)).read_text()
        assert requirements.problems(ROOT, platform, export) == []


def test_a_changed_constraint_or_tessera_pin_is_drift(tmp_path):
    for name in requirements.SOURCES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    (tmp_path / 'tools').mkdir(exist_ok=True)
    (tmp_path / 'tools/resolve_tessera_dev_pin.py').write_bytes((ROOT / 'tools/resolve_tessera_dev_pin.py').read_bytes())
    export = (ROOT / requirements.EXPORT.format('linux_x86_64')).read_text()
    assert requirements.problems(tmp_path, 'linux_x86_64', export) == []
    constraints = tmp_path / '.github/ci-constraints.txt'
    constraints.write_text(constraints.read_text().replace('pytest==9.1.1', 'pytest==9.2.0'))
    found = requirements.problems(tmp_path, 'linux_x86_64', export)
    assert any('was not generated from the current' in problem for problem in found)
    assert any('pins pytest==9.2.0' in problem for problem in found)


def test_every_imported_distribution_has_a_name_in_the_hub():
    modules = requirements.json.loads((ROOT / requirements.MODULES).read_text())
    assert modules['torch'] == ['torch'] and modules['yaml'] == ['pyyaml'] and modules['PIL'] == ['pillow']
    assert modules['tessera'] == ['tessera-quant']


def test_outcomes_read_hosted_style_junit(tmp_path):
    report = tmp_path / 'junit.xml'
    report.write_text(
        '<testsuites><testsuite>'
        '<testcase file="tests/test_a.py" classname="tests.test_a.TestX" name="test_one" time="1"/>'
        '<testcase file="tests/test_a.py" classname="tests.test_a" name="test_two" time="1"><skipped/></testcase>'
        '<testcase file="tests/test_a.py" classname="tests.test_a" name="test_three" time="1"><failure/></testcase>'
        '</testsuite></testsuites>')
    assert python_runner.outcomes(report) == {
        'tests/test_a.py::TestX::test_one': 'passed',
        'tests/test_a.py::test_two': 'skipped',
        'tests/test_a.py::test_three': 'failed'}


def test_a_fully_deselected_file_passes_and_a_failed_node_fails():
    # The gate accepts pytest's exit 5 when the whole assignment collected.
    assert python_runner.file_verdict(5, {}) == 0
    assert python_runner.file_verdict(0, {'tests/test_a.py::test_one': 'passed'}) == 0
    assert python_runner.file_verdict(0, {'tests/test_a.py::test_one': 'failed'}) == 1
    assert python_runner.file_verdict(2, {}) == 2


def test_shards_partition_the_collected_tests(tmp_path):
    module = tmp_path / 'test_many.py'
    module.write_text('import sys\nassert sys.flags.no_user_site\n' +
                      ''.join(f'def test_{index}():\n    pass\n' for index in range(7)))
    (tmp_path / 'bazel_shard.py').write_text(python_runner.SHARD_PLUGIN)
    seen = []
    for index in range(3):
        environment = dict(os.environ, PQ_BAZEL_SHARD_INDEX=str(index), PQ_BAZEL_TOTAL_SHARDS='3',
                           PYTHONPATH=str(tmp_path))
        result = subprocess.run([sys.executable, '-s', '-m', 'pytest', str(module), '-p', 'bazel_shard',
                                 '-q', '--collect-only', '--rootdir=' + str(tmp_path), '-p', 'no:cacheprovider'],
                                cwd=tmp_path, env=environment, capture_output=True, text=True, check=True)
        seen.append({line for line in result.stdout.splitlines() if '::test_' in line})
    assert sum(map(len, seen)) == 7
    assert set().union(*seen) == {f'test_many.py::test_{index}' for index in range(7)}


def test_the_runner_keeps_the_gate_environment():
    gate = (ROOT / '.github/workflows/gate.yml').read_text()
    gate_run = (ROOT / 'tools/ci/gate_run.py').read_text()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'TORCHINDUCTOR_COMPILE_THREADS', 'MAX_JOBS'):
        assert f"{name}='1'" in gate_run and runtime.GATE_ENVIRONMENT[name] == '1'
    assert "CUDA_VISIBLE_DEVICES=''" in gate_run and runtime.GATE_ENVIRONMENT['CUDA_VISIBLE_DEVICES'] == ''
    for name in ('GIT_AUTHOR_NAME', 'GIT_AUTHOR_EMAIL', 'GIT_COMMITTER_NAME', 'GIT_COMMITTER_EMAIL'):
        assert f'{name}: {runtime.GATE_ENVIRONMENT[name]}' in gate


def test_pytest_sees_the_gate_environment_without_bazels(tmp_path):
    environ = {'PATH': '/usr/bin', 'USER': 'ci', 'TEST_TMPDIR': '/x', 'RUNFILES_DIR': '/r',
               'BASH_FUNC_rlocation%%': '() { :; }', 'XML_OUTPUT_FILE': '/t.xml', 'PYTHONSAFEPATH': '1',
               'LD_PRELOAD': '/evil.so', 'PQ_BAZEL_LANE': 'runner'}
    environment = runtime.gate_environment(environ, tmp_path, tmp_path / 'scratch')
    assert environment['USER'] == 'ci' and environment['PATH'] == '/usr/bin'
    for name in ('TEST_TMPDIR', 'RUNFILES_DIR', 'BASH_FUNC_rlocation%%', 'XML_OUTPUT_FILE', 'PYTHONSAFEPATH',
                 'LD_PRELOAD', 'PQ_BAZEL_LANE'):
        assert name not in environment
    assert environment['TMPDIR'] == environment['HOME'] == str(tmp_path / 'scratch')
    assert environment['PYTHONPATH'].split(os.pathsep) == [str(tmp_path / 'tools/ci'), str(tmp_path)]


def test_short_scratch_climbs_to_the_top_writable_directory_of_the_same_filesystem(tmp_path):
    private = tmp_path / 'a/very/deep/test/tmpdir'
    private.mkdir(parents=True)
    scratch = runtime.short_scratch(private)
    try:
        assert scratch.stat().st_dev == private.stat().st_dev
        assert len(str(scratch)) < len(str(private))
        assert scratch.parent in private.parents
    finally:
        scratch.rmdir()


def test_runner_lane_tests_need_a_git_checkout(tmp_path):
    checkout = tmp_path / 'checkout'
    (checkout / '.git').mkdir(parents=True)
    (checkout / 'pytest.ini').write_text('')
    runfiles = tmp_path / 'runfiles'
    runfiles.mkdir()
    (runfiles / 'pytest.ini').symlink_to(checkout / 'pytest.ini')
    assert runtime.checkout_of(runfiles) == checkout.resolve()
    (checkout / '.git').rmdir()
    with pytest.raises(RuntimeError, match='needs a git checkout'):
        runtime.checkout_of(runfiles)


def test_no_credential_reaches_a_test_on_any_lane(tmp_path):
    # The Remote Bazel runner VM exports BUILDBUDDY_API_KEY; runner-lane tests run there.
    environ = {'PATH': '/usr/bin', 'BUILDBUDDY_API_KEY': 'k', 'GITHUB_TOKEN': 't', 'gh_token': 't',
               'OTHER_API_KEY_FILE': 'f', 'PQ_BAZEL_LANE': 'runner', 'TOKENIZERS_PARALLELISM': 'false'}
    environment = runtime.gate_environment(environ, tmp_path, tmp_path / 'scratch')
    assert not [name for name in environment if 'API_KEY' in name.upper() or 'TOKEN' in name.upper()]
    assert environment['PATH'] == '/usr/bin'


def test_the_export_tracks_the_project_lock(tmp_path):
    for name in requirements.SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    before = requirements.source_digest(tmp_path)
    (tmp_path / 'uv.lock').write_text((ROOT / 'uv.lock').read_text() + '\n')
    assert requirements.source_digest(tmp_path) != before
    assert not hasattr(requirements, 'SDK_PROVENANCE')


def test_the_registry_export_pins_no_retired_client():
    for platform in requirements.PLATFORMS:
        export = (ROOT / requirements.EXPORT.format(platform)).read_text()
        assert 'prismabuild' not in {name.replace('_', '-') for name in requirements.pins(export)}
