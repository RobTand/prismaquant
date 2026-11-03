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


def test_the_exact_upload_directory_is_scanned(tmp_path, monkeypatch):
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


def test_encoded_leaks_fail_without_printing_the_sentinel(tmp_path, monkeypatch, capsys):
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
def test_upload_set_refuses_diagnostics_and_disguised_binary(tmp_path, monkeypatch, name, payload):
    monkeypatch.setenv('PQ_TEST_SCAN_KEY', 'upload-format-sentinel')
    (tmp_path / 'gate-result.json').write_text('{}')
    (tmp_path / name).write_bytes(payload)
    assert artifacts.scan_upload(tmp_path, ['PQ_TEST_SCAN_KEY']) == 75
    # Rejected evidence remains private; callers must not publish this directory.
    assert (tmp_path / name).read_bytes() == payload


def test_zero_exit_with_not_scanned_diagnostics_refuses_upload(tmp_path, monkeypatch, capsys):
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


def test_python_314_scanner_really_decodes_zstd(tmp_path, monkeypatch, capsys):
    from compression import zstd
    monkeypatch.setenv('PQ_TEST_SCAN_KEY', 'compressed-scan-sentinel')
    path = tmp_path / 'execution_log.binpb.zst'
    path.write_bytes(zstd.compress(b'prefix compressed-scan-sentinel suffix'))
    assert artifacts.scanner.main(['--env', 'PQ_TEST_SCAN_KEY', str(path)]) == 1
    output = capsys.readouterr().out
    assert 'secret detected' in output
    assert 'not scanned' not in output
    assert 'compressed-scan-sentinel' not in output


def test_scanner_core_remains_byte_identical_to_pr_a():
    import hashlib
    assert hashlib.sha256((ROOT / 'tools/ci/check_no_secret_leak.py').read_bytes()).hexdigest() == 'a7c88079f0b2558d24d68f7eef2d0824a9b32def42db49c733dc7a905f136eb2'


def test_scanner_bootstrap_installs_pinned_python_without_secrets_in_children(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(artifacts.sys, 'version_info', (3, 12, 0))
    monkeypatch.setenv('BUILDBUDDY_API_KEY', 'bootstrap-private-sentinel')
    monkeypatch.setenv('OTHER_API_KEY', 'bootstrap-other-sentinel')
    calls = []
    python = str(tmp_path / 'python3.14')
    def bootstrap_run(command, **kwargs):
        calls.append(command)
        assert 'BUILDBUDDY_API_KEY' not in kwargs['env']
        assert 'OTHER_API_KEY' not in kwargs['env']
        assert 'bootstrap-private-sentinel' not in repr(command)
        assert kwargs['check'] is True and kwargs['capture_output'] is True
        return SimpleNamespace(stdout=python+'\n' if 'find' in command else '')
    def replace_interpreter(executable, argv):
        assert executable == python and argv == [python, *artifacts.sys.argv]
        assert os.environ['BUILDBUDDY_API_KEY'] == 'bootstrap-private-sentinel'
        raise SystemExit(0)
    with pytest.raises(SystemExit, match='0'):
        artifacts.require_scan_python(runner=bootstrap_run, execute=replace_interpreter)
    assert calls[0][-1] == 'uv==0.9.5'
    assert calls[1][-3:] == ['python', 'install', '3.14.0']
    assert calls[2][-4:] == ['python', 'find', '--managed-python', '3.14.0']


def test_scanner_bootstrap_is_a_noop_on_python314(monkeypatch):
    monkeypatch.setattr(artifacts.sys, 'version_info', (3, 14, 0))
    def forbidden_bootstrap(*args, **kwargs):
        pytest.fail('Python 3.14 must not bootstrap again')
    artifacts.require_scan_python(runner=forbidden_bootstrap, execute=forbidden_bootstrap)


def test_scanner_bootstrap_failure_never_exposes_child_diagnostics(monkeypatch, capsys):
    import subprocess
    monkeypatch.setattr(artifacts.sys, 'version_info', (3, 12, 0))
    def broken_bootstrap(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, output='private-bootstrap-output')
    with pytest.raises(RuntimeError, match='Python 3.14') as error:
        artifacts.require_scan_python(runner=broken_bootstrap)
    assert 'private-bootstrap-output' not in str(error.value)
    assert 'private-bootstrap-output' not in capsys.readouterr().out


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
