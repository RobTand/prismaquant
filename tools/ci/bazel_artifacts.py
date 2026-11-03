"""Credential isolation and required scans for the remote gate's upload boundary."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

CHILD_ENVIRONMENT = {'PATH', 'HOME', 'TMPDIR', 'TMP', 'TEMP', 'LANG', 'LC_ALL',
                     'BAZELISK_HOME', 'JAVA_HOME'}
SCANNER_SPEC = importlib.util.spec_from_file_location(
    'pq_cutover_secret_scanner', Path(__file__).with_name('check_no_secret_leak.py'))
scanner = importlib.util.module_from_spec(SCANNER_SPEC)
SCANNER_SPEC.loader.exec_module(scanner)


API = 'https://app.buildbuddy.io/api/v1/'


class RemoteError(RuntimeError):
    """Remote evidence is unavailable or does not establish success."""


class BuildBuddy:
    def __init__(self, key, api=API):
        if not key:
            raise RemoteError('BUILDBUDDY_API_KEY is unavailable')
        self.key, self.api = key, api

    def call(self, method, body, *, raw=False, timeout=120):
        request = urllib.request.Request(
            self.api + method, data=json.dumps(body).encode(),
            headers={'x-buildbuddy-api-key': self.key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read() if raw else json.load(response)
        except (urllib.error.URLError, TimeoutError, ValueError):
            raise RemoteError(f'BuildBuddy {method} request failed') from None


def child_environment(environ=None):
    environ = os.environ if environ is None else environ
    return {name: value for name, value in environ.items() if name in CHILD_ENVIRONMENT}



def prepare_history(expected_head, *, root=None, runner=None):
    """Hydrate the runner checkout before tests archive historical main inputs."""
    runner = subprocess.run if runner is None else runner
    def git(arguments, accepted=(0,)):
        result = runner(['git', '-c', 'gc.auto=0', '-c', 'maintenance.auto=false',
                         *arguments], cwd=root, env=child_environment(),
                        capture_output=True, text=True, timeout=300, check=False)
        if result.returncode not in accepted:
            raise RuntimeError('runner history preparation failed')
        return result
    try:
        if not re.fullmatch('[0-9a-f]{40}', expected_head):
            raise ValueError('invalid history head')
        if git(['rev-parse', '--verify', 'HEAD']).stdout.strip() != expected_head:
            raise ValueError('runner history head mismatch')
        shallow = git(['rev-parse', '--is-shallow-repository']).stdout.strip()
        if shallow not in ('true', 'false'):
            raise ValueError('invalid shallow status')
        fetch = ['fetch', '--refetch', '--no-filter', '--tags']
        if shallow == 'true':
            fetch.append('--unshallow')
        git([*fetch, 'origin', '+refs/heads/*:refs/remotes/origin/*'])
        objects = git(['rev-list', '--objects', '--missing=print', '--all', 'HEAD']).stdout.splitlines()
        if not objects or any(not re.fullmatch('[0-9a-f]{40}', line.split(' ', 1)[0])
                              for line in objects):
            raise ValueError('missing history objects')
        # A verified full refetch can leave stale partial-clone metadata.
        pack = Path(git(['rev-parse', '--git-path', 'objects/pack']).stdout.strip())
        if not pack.is_absolute():
            pack = Path(root or Path.cwd()) / pack
        for marker in pack.glob('*.promisor'):
            marker.unlink(missing_ok=True)
        for name in ('remote.origin.promisor', 'remote.origin.partialclonefilter',
                     'extensions.partialclone'):
            git(['config', '--local', '--unset-all', name], accepted=(0, 5))
        if (git(['rev-parse', '--verify', 'HEAD']).stdout.strip() != expected_head
                or git(['rev-parse', '--is-shallow-repository']).stdout.strip() != 'false'):
            raise ValueError('incomplete runner history')
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        raise RuntimeError('runner history preparation failed; tests refused') from None


def require_scan_python(*, runner=None, execute=None):
    """Bootstrap through key-free children, then replace the keyed parent."""
    if sys.version_info[:2] == (3, 14):
        return
    runner = subprocess.run if runner is None else runner
    execute = os.execv if execute is None else execute
    bootstrap = Path(tempfile.gettempdir()) / 'pq-scanner-bootstrap'
    environment = child_environment()
    def prepare(command):
        return runner(command, env=environment, text=True, capture_output=True,
                      check=True, timeout=300)
    try:
        prepare([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps',
                 '--target', str(bootstrap), 'uv==0.9.5'])
        uv = str(bootstrap / 'bin/uv')
        prepare([uv, 'python', 'install', '3.14.0'])
        python = prepare([uv, 'python', 'find', '--managed-python', '3.14.0']).stdout.strip()
        if not Path(python).is_absolute():
            raise ValueError('invalid managed Python path')
        execute(python, [python, *sys.argv])
    except (OSError, ValueError, subprocess.SubprocessError):
        raise RuntimeError('Python 3.14 scanner bootstrap failed; upload refused') from None


def scan_upload(directory, required):
    """The credential stays in this process; the scanner never receives its value in argv."""
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        for name in missing:
            print(f'{name}: required key unavailable; upload refused')
        return 75
    if not required or not Path(directory).is_dir():
        print('Upload directory or required scan keys unavailable; upload refused')
        return 75
    # The caller publishes JSON receipts only. All raw diagnostics stay private.
    if sys.version_info[:2] != (3, 14) or scanner.zstd is None:
        print('Python 3.14 with compression.zstd required; upload refused')
        return 75
    arguments = [word for name in required for word in ('--env', name)]
    diagnostics = io.StringIO()
    try:
        with contextlib.redirect_stdout(diagnostics), contextlib.redirect_stderr(diagnostics):
            code = scanner.main([*arguments, str(directory)])
        if 'not scanned' in diagnostics.getvalue().lower():
            print('Incomplete artifact scan; upload refused')
            return 75
        if code:
            print('Artifact secret detected; upload refused' if code == 1
                  else 'Artifact scan failed; upload refused')
            return 1 if code == 1 else 75
        for path in Path(directory).rglob('*'):
            if path.is_symlink():
                raise ValueError('linked artifact')
            if path.is_dir():
                continue
            if path.suffix != '.json':
                raise ValueError('unsupported artifact format')
            json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, RuntimeError):
        print('Artifact unavailable or unsupported; upload refused')
        return 75
    return 0


def publish_json(upload, documents, required):
    """Scan private JSON bytes before publishing the exact upload set."""
    upload = Path(upload)
    if upload.is_symlink() or list(upload.iterdir()):
        raise RuntimeError('upload directory is not empty')
    with tempfile.TemporaryDirectory(prefix='pq-private-upload-') as directory:
        staged = Path(directory)
        for name, content in documents.items():
            (staged / name).write_bytes(content)
        if scan_upload(staged, required):
            raise RuntimeError('artifact scan refused upload')
        if list(upload.iterdir()):
            raise RuntimeError('unexpected file in upload directory')
        for name in documents:
            (staged / name).replace(upload / name)
        if scan_upload(upload, required):
            raise RuntimeError('final artifact scan refused upload')
