"""Remote cutover children never receive credentials; uploads require a full scan."""
from __future__ import annotations

import importlib.util
import os

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('pq_bazel_artifacts', ROOT / 'tools/ci/bazel_artifacts.py')
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)


@pytest.fixture
def controlled_scan_runtime(monkeypatch):
    """Isolate runtime admission; native 3.14 behavior has its own explicit step."""
    monkeypatch.setattr(artifacts, 'scan_runtime_available', lambda: True)


def test_children_get_only_the_declared_nonsecret_environment():
    source = {'PATH': '/usr/bin', 'HOME': '/private/home', 'TMPDIR': '/private/tmp',
              'LANG': 'C.UTF-8', 'BUILDBUDDY_API_KEY': 'bb-sentinel',
              'MERGIFY_TOKEN': 'mergify-sentinel', 'OTHER_API_KEY': 'other-sentinel',
              'GITHUB_TOKEN': 'github-sentinel', 'UNRELATED_SECRET': 'unrelated-sentinel',
              'LD_PRELOAD': '/untrusted/loader', 'PYTHONPATH': '/untrusted/modules'}
    assert artifacts.child_environment(source) == {
        'PATH': '/usr/bin', 'HOME': '/private/home', 'TMPDIR': '/private/tmp', 'LANG': 'C.UTF-8'}
    assert source['BUILDBUDDY_API_KEY'] == 'bb-sentinel'


def test_an_absent_required_key_refuses_an_upload(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('PQ_TEST_SCAN_KEY', raising=False)
    (tmp_path / 'result.json').write_text('{}')
    assert artifacts.scan_upload(tmp_path, ['PQ_TEST_SCAN_KEY']) == 75
    assert 'PQ_TEST_SCAN_KEY' in capsys.readouterr().out


def test_an_empty_required_key_refuses_an_upload(tmp_path, monkeypatch):
    monkeypatch.setenv('PQ_TEST_SCAN_KEY', '')
    assert artifacts.scan_upload(tmp_path, ['PQ_TEST_SCAN_KEY']) == 75


def test_the_exact_upload_directory_is_scanned(tmp_path, monkeypatch, controlled_scan_runtime):
    monkeypatch.setenv('PQ_TEST_SCAN_KEY', 'only-the-upload-sentinel')
    private = tmp_path / 'private'
    upload = tmp_path / 'upload'
    private.mkdir()
    upload.mkdir()
    (private / 'raw-diagnostics').write_text(os.environ['PQ_TEST_SCAN_KEY'])
    (upload / 'result.json').write_text('{}')
    assert artifacts.scan_upload(upload, ['PQ_TEST_SCAN_KEY']) == 0
    (upload / 'nested').mkdir()
    (upload / 'nested' / 'leak.txt').write_text(os.environ['PQ_TEST_SCAN_KEY'])
    assert artifacts.scan_upload(upload, ['PQ_TEST_SCAN_KEY']) == 1


def test_encoded_leaks_fail_without_printing_the_sentinel(tmp_path, monkeypatch, capsys, controlled_scan_runtime):
    import base64
    monkeypatch.setenv('PQ_TEST_SCAN_KEY', 'encoded-upload-sentinel')
    encoded = base64.b64encode(os.environ['PQ_TEST_SCAN_KEY'].encode()).decode()
    (tmp_path / 'result.json').write_text(encoded)
    assert artifacts.scan_upload(tmp_path, ['PQ_TEST_SCAN_KEY']) == 1
    output = capsys.readouterr().out
    assert 'secret detected' in output
    assert os.environ['PQ_TEST_SCAN_KEY'] not in output
    assert encoded not in output


def test_missing_upload_directory_cannot_pass_scan(tmp_path, monkeypatch):
    monkeypatch.setenv('PQ_TEST_SCAN_KEY', 'scan-sentinel')
    assert artifacts.scan_upload(tmp_path / 'absent', ['PQ_TEST_SCAN_KEY']) == 75


@pytest.mark.parametrize('name,payload', [
    ('execution_log.binpb.zst', b'\x28\xb5\x2f\xfd'),
    ('diagnostics.zip', b'PK\x03\x04'),
    ('diagnostics.xz', b'\xfd7zXZ\x00'),
    ('diagnostics.bz2', b'BZh9'),
    ('renamed.json', b'PK\x03\x04'),
])
def test_upload_set_refuses_diagnostics_and_disguised_binary(tmp_path, monkeypatch, name, payload, controlled_scan_runtime):
    monkeypatch.setenv('PQ_TEST_SCAN_KEY', 'upload-format-sentinel')
    (tmp_path / 'gate-result.json').write_text('{}')
    (tmp_path / name).write_bytes(payload)
    assert artifacts.scan_upload(tmp_path, ['PQ_TEST_SCAN_KEY']) == 75
    # Rejected evidence remains private; callers must not publish this directory.
    assert (tmp_path / name).read_bytes() == payload


def test_zero_exit_with_not_scanned_diagnostics_refuses_upload(tmp_path, monkeypatch, capsys, controlled_scan_runtime):
    monkeypatch.setenv('PQ_TEST_SCAN_KEY', 'incomplete-scan-sentinel')
    (tmp_path / 'gate-result.json').write_text('{}')
    def partial_scan(argv):
        print('private/path: future format: not scanned')
        return 0
    monkeypatch.setattr(artifacts.scanner, 'main', partial_scan)
    assert artifacts.scan_upload(tmp_path, ['PQ_TEST_SCAN_KEY']) == 75
    assert 'private/path' not in capsys.readouterr().out


def test_scan_requires_python_314_even_for_plain_receipts(tmp_path, monkeypatch):
    import sys
    monkeypatch.setenv('PQ_TEST_SCAN_KEY', 'runtime-scan-sentinel')
    (tmp_path / 'gate-result.json').write_text('{}')
    monkeypatch.setattr(sys, 'version_info', (3, 12, 0))
    assert artifacts.scan_upload(tmp_path, ['PQ_TEST_SCAN_KEY']) == 75


def test_scanner_core_remains_byte_identical_to_pr_a():
    import hashlib
    assert hashlib.sha256((ROOT / 'tools/ci/check_no_secret_leak.py').read_bytes()).hexdigest() == 'a7c88079f0b2558d24d68f7eef2d0824a9b32def42db49c733dc7a905f136eb2'


@pytest.mark.parametrize('machine', ['x86_64', 'aarch64'])
def test_scanner_bootstrap_uses_verified_python_without_pip(tmp_path, monkeypatch, capsys, machine):
    import io
    import platform
    import tarfile
    monkeypatch.setattr(artifacts.sys, 'version_info', (3, 12, 0))
    monkeypatch.setattr(platform, 'machine', lambda: machine)
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    monkeypatch.setenv('BUILDBUDDY_API_KEY', 'bootstrap-private-sentinel')
    content = b'verified-python-fixture'
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as bundle:
        member = tarfile.TarInfo('python/bin/python3.14')
        member.size = len(content)
        member.mode = 0o755
        bundle.addfile(member, io.BytesIO(content))
    payload = archive.getvalue()
    monkeypatch.setattr(artifacts, 'PYTHON_RUNTIME_SHA256', {
        machine: artifacts.bytes_sha256hex(payload)})
    calls = []
    def fetch(url):
        calls.append(url)
        assert f'{machine}-unknown-linux-gnu-install_only_stripped.tar.gz' in url
        assert 'bootstrap-private-sentinel' not in url
        return payload
    def forbidden_child(*args, **kwargs):
        pytest.fail('runtime bootstrap must not launch a pip or installer child')
    monkeypatch.setattr(artifacts.subprocess, 'run', forbidden_child)
    def replace_interpreter(executable, argv):
        assert 'Scanner runtime verified: pinned Python 3.14.0' in capsys.readouterr().out
        assert Path(executable).read_bytes() == content
        assert argv == [executable, *artifacts.sys.argv]
        assert os.environ['BUILDBUDDY_API_KEY'] == 'bootstrap-private-sentinel'
        raise SystemExit(0)
    with pytest.raises(SystemExit, match='0'):
        artifacts.require_scan_python(fetch=fetch, execute=replace_interpreter)
    assert len(calls) == 1


def test_scanner_bootstrap_is_a_noop_on_python314(monkeypatch):
    monkeypatch.setattr(artifacts.sys, 'version_info', (3, 14, 0))
    def forbidden_bootstrap(*args, **kwargs):
        pytest.fail('Python 3.14 must not bootstrap again')
    artifacts.require_scan_python(fetch=forbidden_bootstrap, execute=forbidden_bootstrap)


@pytest.mark.parametrize('failure', ['download', 'integrity', 'extract', 'executable', 'platform'])
def test_scanner_bootstrap_refuses_unverified_runtime_without_private_diagnostics(
        monkeypatch, capsys, failure):
    import io
    import platform
    import tarfile
    monkeypatch.setattr(artifacts.sys, 'version_info', (3, 12, 0))
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(platform, 'machine', lambda: 'unknown' if failure == 'platform' else 'x86_64')
    payload = b'not-a-runtime-archive'
    if failure == 'executable':
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode='w:gz'):
            pass
        payload = archive.getvalue()
    if failure in ('extract', 'executable'):
        monkeypatch.setattr(artifacts, 'PYTHON_RUNTIME_SHA256', {
            'x86_64': artifacts.bytes_sha256hex(payload)})
    def fetch(url):
        if failure == 'download':
            raise OSError('private-path-and-query')
        return payload
    def forbidden_exec(*args):
        pytest.fail('unverified runtime must never execute')
    with pytest.raises(RuntimeError, match='Python 3.14') as error:
        artifacts.require_scan_python(fetch=fetch, execute=forbidden_exec)
    assert 'private-path-and-query' not in str(error.value)
    output = capsys.readouterr()
    assert failure in output.err
    assert 'private-path-and-query' not in output.err + output.out


@pytest.mark.parametrize('role', ['planner', 'shard', 'merger', 'post-merge'])
def test_cli_refusal_names_a_fixed_phase_without_exception_values(tmp_path, monkeypatch, capsys, role):
    from tools.ci import bazel_gate, bazel_selection, remote_gate
    upload = tmp_path / 'upload'
    monkeypatch.setenv('BUILDBUDDY_ARTIFACTS_DIRECTORY', str(upload))
    def unavailable_runtime():
        raise RuntimeError('private-path-token-and-query')
    monkeypatch.setattr(bazel_gate.artifacts, 'require_scan_python', unavailable_runtime)
    mode = 'main' if role == 'post-merge' else 'pr'
    if role == 'planner':
        code = bazel_selection.main(['plan', '--mode', mode, '--base', 'b' * 40,
            '--head', 'a' * 40, '--branch', 'runtime-proof'])
    elif role == 'shard':
        code = bazel_gate.main(['shard'])
    else:
        code = remote_gate.main([mode, '--base', 'b' * 40, '--head', 'a' * 40,
            '--branch', 'runtime-proof', '--output', str(upload)])
    assert code == 75 and list(upload.iterdir()) == []
    output = capsys.readouterr()
    category = ' category=runtime;' if role == 'shard' else ''
    assert 'refused at scanner-runtime;' + category + ' upload refused' in output.err
    assert 'private-path-token-and-query' not in output.err + output.out


@pytest.mark.parametrize('wrong_head', [False, True])
def test_runner_history_hydrates_a_shallow_checkout_for_archived_main(tmp_path, monkeypatch, wrong_head):
    import subprocess
    source, checkout = tmp_path / 'origin', tmp_path / 'checkout'
    source.mkdir()
    def git(directory, *arguments):
        return subprocess.run(['git', *arguments], cwd=directory, check=True,
            text=True, capture_output=True, env=artifacts.child_environment()).stdout.strip()
    git(source, 'init', '-b', 'main')
    git(source, 'config', 'user.name', 'History fixture')
    git(source, 'config', 'user.email', 'fixture@example.invalid')
    (source / 'tracked').write_text('archived main')
    git(source, 'add', 'tracked')
    git(source, 'commit', '-m', 'base')
    base = git(source, 'rev-parse', 'HEAD')
    (source / 'tracked').write_text('head')
    git(source, 'commit', '-am', 'head')
    head = git(source, 'rev-parse', 'HEAD')
    git(tmp_path, 'clone', '--depth=1', source.as_uri(), str(checkout))
    assert git(checkout, 'rev-parse', '--is-shallow-repository') == 'true'
    monkeypatch.setenv('BUILDBUDDY_API_KEY', 'history-private-sentinel')
    calls = []
    def run(command, **kwargs):
        assert 'BUILDBUDDY_API_KEY' not in kwargs['env']
        assert 'history-private-sentinel' not in repr(command)
        assert command[:5] == ['git', '-c', 'gc.auto=0', '-c', 'maintenance.auto=false']
        assert kwargs['capture_output'] is True
        calls.append(command)
        return subprocess.run(command, **kwargs)
    if wrong_head:
        with pytest.raises(RuntimeError, match='history'):
            artifacts.prepare_history(base, root=checkout, runner=run)
    else:
        artifacts.prepare_history(head, root=checkout, runner=run)
        assert git(checkout, 'rev-parse', '--is-shallow-repository') == 'false'
        assert git(checkout, 'show', base + ':tracked') == 'archived main'
        assert any('--unshallow' in command and '--no-filter' in command for command in calls)


def test_runner_history_failure_redacts_git_diagnostics(tmp_path, capsys):
    import subprocess
    def broken_git(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, output='private-history-credential')
    with pytest.raises(RuntimeError, match='history') as error:
        artifacts.prepare_history('a' * 40, root=tmp_path, runner=broken_git)
    assert 'private-history-credential' not in str(error.value)
    assert 'private-history-credential' not in capsys.readouterr().out
