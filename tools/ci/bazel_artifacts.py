"""Credential isolation and required scans for the remote gate's upload boundary."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import http.client
import json
import os
import platform
import re
import sys
import subprocess
import shutil
import socket
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from prismaquant.digests import bytes_sha256hex

CHILD_ENVIRONMENT = {'PATH', 'HOME', 'TMPDIR', 'TMP', 'TEMP', 'LANG', 'LC_ALL',
                     'BAZELISK_HOME', 'JAVA_HOME'}
SCANNER_SPEC = importlib.util.spec_from_file_location(
    'pq_cutover_secret_scanner', Path(__file__).with_name('check_no_secret_leak.py'))
scanner = importlib.util.module_from_spec(SCANNER_SPEC)
SCANNER_SPEC.loader.exec_module(scanner)


API = 'https://app.buildbuddy.io/api/v1/'
REFUSAL_DETAIL = 'plan-refusal.json'
PYTHON_RUNTIME_RELEASE = '20251014'
PYTHON_RUNTIME_SHA256 = {
    'x86_64': '493c477b4a88bb1ea2f6c6f57fa0e88ffbe55d9e7b1405c4699f2d41c04eb154',
    'aarch64': '7dbb43b742c040835a277318355fb359b41e509dbf4fbb614da38005a9290e16',
}


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



class RuntimeDownloadError(RuntimeError):
    """Fixed bootstrap diagnostics, never a partial response or exception text."""

    def __init__(self, category, attempts):
        super().__init__('scanner runtime download unavailable')
        self.category, self.attempts = category, attempts



def runtime_remaining(deadline, sock=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('runtime download deadline exhausted')
    if sock is not None:
        sock.settimeout(remaining)
    return remaining


class RuntimeDeadlineReader(io.RawIOBase):
    """Bound every socket read, including headers and slow-drip body chunks."""

    def __init__(self, raw, sock, deadline):
        self.raw, self.sock, self.deadline = raw, sock, deadline

    def readable(self):
        return True

    def readinto(self, buffer):
        runtime_remaining(self.deadline, self.sock)
        return self.raw.readinto(buffer)

    def close(self):
        try:
            self.raw.close()
        finally:
            super().close()


class RuntimeResponseSocket:
    def __init__(self, sock, deadline):
        self.sock, self.deadline = sock, deadline

    def makefile(self, mode):
        raw = self.sock.makefile(mode, buffering=0)
        return io.BufferedReader(RuntimeDeadlineReader(raw, self.sock, self.deadline))


def open_runtime_url(url, deadline):
    """Keep urllib redirects/TLS while sharing one deadline across operations."""
    def connect(address, timeout=None, source_address=None):
        # getaddrinfo is synchronous; a hung resolver remains a documented risk.
        addresses = socket.getaddrinfo(*address, 0, socket.SOCK_STREAM)
        error = None
        for family, kind, protocol, _, destination in addresses:
            sock = socket.socket(family, kind, protocol)
            try:
                runtime_remaining(deadline, sock)
                if source_address:
                    sock.bind(source_address)
                sock.connect(destination)
                runtime_remaining(deadline, sock)
                return sock
            except OSError as failure:
                sock.close()
                error = failure
        if error is not None:
            raise error
        raise OSError('runtime download has no address')

    def connection_type(base):
        class Connection(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._create_connection = connect
                self.response_class = lambda sock, *a, **kw: http.client.HTTPResponse(
                    RuntimeResponseSocket(sock, deadline), *a, **kw)

            def connect(self):
                super().connect()
                runtime_remaining(deadline, self.sock)

            def _tunnel(self):
                super()._tunnel()
                runtime_remaining(deadline, self.sock)

            def send(self, data):
                if self.sock is None:
                    self.connect()
                runtime_remaining(deadline, self.sock)
                super().send(data)
        return Connection

    class HTTP(urllib.request.HTTPHandler):
        def http_open(self, request):
            return self.do_open(connection_type(http.client.HTTPConnection), request)

    class HTTPS(urllib.request.HTTPSHandler):
        def https_open(self, request):
            return self.do_open(connection_type(http.client.HTTPSConnection), request,
                                context=self._context)

    opener = urllib.request.build_opener(HTTP(), HTTPS())
    return opener.open(url, timeout=runtime_remaining(deadline))



def download_runtime_archive(url, fetch):
    deadline = time.monotonic() + 120
    for attempt in range(1, 4):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeDownloadError('deadline', attempt - 1)
        try:
            if fetch is None:
                with open_runtime_url(url, deadline) as response:
                    length = response.headers.get('Content-Length')
                    expected = int(length) if length is not None else None
                    chunks, received = [], 0
                    while True:
                        runtime_remaining(deadline)
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        received += len(chunk)
                    if expected is not None and received < expected:
                        raise http.client.IncompleteRead(b'', expected - received)
                    content = b''.join(chunks)
            else:
                content = fetch(url)
        except TimeoutError:
            raise RuntimeDownloadError('deadline', attempt) from None
        except http.client.IncompleteRead:
            backoff = attempt * 0.1
            if attempt < 3 and deadline - time.monotonic() > backoff:
                time.sleep(backoff)
                if time.monotonic() < deadline:
                    continue
            raise RuntimeDownloadError('incomplete-read', attempt) from None
        if time.monotonic() >= deadline:
            raise RuntimeDownloadError('deadline', attempt)
        return content


def runtime_archive(url, digest, fetch):
    """Reuse only digest-verified archives, with atomic concurrent publication."""
    cache = Path.home() / '.cache/prismaquant/scan-runtime'
    if directory := os.environ.get('BUILDBUDDY_ARTIFACTS_DIRECTORY'):
        if cache.resolve().is_relative_to(Path(directory).resolve()):
            raise RuntimeError('runtime cache overlaps uploads')
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / (digest + '.tar.gz')
    try:
        with os.fdopen(os.open(archive, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
            content = stream.read()
    except FileNotFoundError:
        content = None
    if content is not None and bytes_sha256hex(content) == digest:
        return content
    content = download_runtime_archive(url, fetch)
    if bytes_sha256hex(content) != digest:
        raise ValueError('runtime checksum mismatch')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, prefix='.runtime-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, archive)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return content


def require_scan_python(*, fetch=None, execute=None):
    """Verify a standalone runtime without pip, then replace the keyed parent."""
    if sys.version_info[:2] == (3, 14):
        print('Scanner runtime active: Python 3.14', flush=True)
        return
    execute = os.execv if execute is None else execute
    bootstrap = None
    phase = 'platform'
    try:
        machine = platform.machine()
        if platform.system() != 'Linux' or machine not in PYTHON_RUNTIME_SHA256:
            raise ValueError('unsupported bootstrap platform')
        url = ('https://github.com/astral-sh/python-build-standalone/releases/download/'
               + PYTHON_RUNTIME_RELEASE + '/cpython-3.14.0%2B' + PYTHON_RUNTIME_RELEASE
               + '-' + machine + '-unknown-linux-gnu-install_only_stripped.tar.gz')
        phase = 'download'
        content = runtime_archive(url, PYTHON_RUNTIME_SHA256[machine], fetch)
        phase = 'integrity'
        if bytes_sha256hex(content) != PYTHON_RUNTIME_SHA256[machine]:
            raise ValueError('runtime checksum mismatch')
        phase = 'extract'
        bootstrap = Path(tempfile.mkdtemp(prefix='pq-scanner-bootstrap-'))
        with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as archive:
            archive.extractall(bootstrap, filter='data')
        phase = 'executable'
        python = str(bootstrap / 'python/bin/python3.14')
        if not Path(python).is_file():
            raise ValueError('runtime executable unavailable')
        phase = 'exec'
        print('Scanner runtime verified: pinned Python 3.14.0', flush=True)
        execute(python, [python, *sys.argv])
        raise RuntimeError('runtime replacement returned')
    except (OSError, ValueError, TypeError, tarfile.TarError, RuntimeError,
            http.client.HTTPException) as error:
        if bootstrap is not None:
            shutil.rmtree(bootstrap, ignore_errors=True)
        detail = ''
        if isinstance(error, RuntimeDownloadError):
            detail = f'; category={error.category}; attempts={error.attempts}'
        elif isinstance(error, ValueError) and phase == 'download':
            phase = 'integrity'
        print('Scanner runtime bootstrap refused at ' + phase + detail, file=sys.stderr)
        raise RuntimeError('Python 3.14 scanner bootstrap failed; upload refused') from None


def scan_runtime_available():
    """The upload boundary requires the real 3.14 stdlib decoder."""
    return sys.version_info[:2] == (3, 14) and scanner.zstd is not None


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
    if not scan_runtime_available():
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
