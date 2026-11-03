"""Remote cutover children never receive credentials; uploads require a full scan."""
from __future__ import annotations

import importlib.util
import os
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
