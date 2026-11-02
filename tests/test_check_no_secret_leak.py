"""The portable artifact scanner reports identities without disclosing secrets."""
import base64
import gzip
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote, quote_plus

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / 'tools/ci/check_no_secret_leak.py'
SECRET = 'fake-token/+ with?punctuation=42'


def scan(*paths, variables=('SCAN_FAKE_SECRET',), values=None):
    # Never inherit credentials from the pytest worker.
    env = {'PATH': os.defpath, 'SCAN_FAKE_SECRET': SECRET}
    env.update(values or {})
    return subprocess.run(
        [sys.executable, str(SCANNER),
         *(argument for name in variables for argument in ('--env', name)),
         *(str(path) for path in paths)],
        env=env, capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize('payload', [
    SECRET.encode(), base64.b64encode(SECRET.encode()),
    base64.urlsafe_b64encode(SECRET.encode()),
    quote(SECRET, safe='').encode(), quote(SECRET).encode(), quote_plus(SECRET).encode(),
    gzip.compress(SECRET.encode(), mtime=0),
    gzip.compress(b'clean', mtime=0) + gzip.compress(SECRET.encode(), mtime=0),
    gzip.compress(gzip.compress(SECRET.encode(), mtime=0), mtime=0),
])
def test_secret_forms_in_nested_directory_are_refused(tmp_path, payload):
    nested = tmp_path / 'nested'
    nested.mkdir()
    artifact = nested / 'diagnostic.bin'
    artifact.write_bytes(b'prefix\n' + payload)
    result = scan(tmp_path)
    assert result.returncode == 1, result.stderr
    output = result.stdout + result.stderr
    assert str(artifact) in output and 'SCAN_FAKE_SECRET' in output
    for value in (SECRET, base64.b64encode(SECRET.encode()).decode(), quote(SECRET, safe='')):
        assert value not in output


def test_secret_payload_collection_is_independent_of_clock():
    script = '''
import gzip
import sys
gzip.time.time = lambda: int(sys.argv[1])
import pytest
raise SystemExit(pytest.main(sys.argv[2:]))
'''
    collections = []
    for timestamp in (1_700_000_000, 1_800_000_000):
        result = subprocess.run(
            [sys.executable, '-c', script, str(timestamp), '--collect-only', '-q',
             '--noconftest', '-c', str(ROOT / 'pytest.ini'), '--rootdir', str(ROOT),
             '-o', 'addopts=', str(Path(__file__).resolve()),
             '-k', 'test_secret_forms_in_nested_directory_are_refused'],
            cwd=ROOT, env={'PATH': os.defpath, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'},
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        nodes = [line for line in result.stdout.splitlines()
                 if '::test_secret_forms_in_nested_directory_are_refused[' in line]
        assert len(nodes) == 9, result.stdout
        collections.append(nodes)
    assert collections[0] == collections[1]


def test_clean_files_and_multiple_paths_and_variables(tmp_path):
    clean = tmp_path / 'clean.txt'
    clean.write_text('ordinary diagnostic')
    leaked = tmp_path / 'leaked.txt'
    leaked.write_text('another-fake-secret')
    assert scan(clean).returncode == 0
    result = scan(clean, leaked, variables=('SCAN_FAKE_SECRET', 'SECOND_FAKE'),
                  values={'SECOND_FAKE': 'another-fake-secret'})
    assert result.returncode == 1
    assert 'SECOND_FAKE' in result.stdout + result.stderr
    assert 'another-fake-secret' not in result.stdout + result.stderr


@pytest.mark.parametrize('values', [{}, {'ABSENT_FAKE': ''}])
def test_unavailable_variable_is_explicit_and_not_a_failure(tmp_path, values):
    artifact = tmp_path / 'clean'
    artifact.write_text('clean')
    result = scan(artifact, variables=('ABSENT_FAKE',), values=values)
    assert result.returncode == 0
    assert 'ABSENT_FAKE: not available, not scanned' in result.stdout + result.stderr


def test_missing_path_is_infrastructure_failure(tmp_path):
    result = scan(tmp_path / 'missing')
    assert result.returncode == 75
    assert 'not scanned' in result.stdout + result.stderr


def load_scanner():
    spec = importlib.util.spec_from_file_location('secret_scanner', SCANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unreadable_file_never_reports_exception_contents(tmp_path, monkeypatch, capsys):
    scanner = load_scanner()
    artifact = tmp_path / 'denied'
    artifact.write_text('clean')
    monkeypatch.setenv('SCAN_FAKE_SECRET', SECRET)

    def denied(*args, **kwargs):
        raise PermissionError(SECRET)

    monkeypatch.setattr(Path, 'read_bytes', denied)
    assert scanner.main(['--env', 'SCAN_FAKE_SECRET', str(artifact)]) == 75
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err
    assert str(artifact) in output.out + output.err


def test_secret_in_filename_is_redacted(tmp_path):
    secret = 'fake-filename-token'
    artifact = tmp_path / secret
    artifact.write_text(secret)
    result = scan(artifact, values={'SCAN_FAKE_SECRET': secret})
    assert result.returncode == 1
    assert secret not in result.stdout + result.stderr


def test_argument_errors_do_not_echo_secret_values():
    secret = 'fake-option-token'
    result = scan(SCANNER, '--' + secret, values={'SCAN_FAKE_SECRET': secret})
    assert secret not in result.stdout + result.stderr
    assert result.returncode == 75


def test_path_redaction_cannot_reassemble_a_secret(tmp_path):
    secret = 'fake-filename-token'
    artifact = tmp_path / (secret[:5] + secret + secret[5:])
    artifact.write_text(secret)
    result = scan(artifact, values={'SCAN_FAKE_SECRET': secret})
    assert result.returncode == 1
    assert secret not in result.stdout + result.stderr


def test_corrupt_gzip_is_not_silently_passed(tmp_path):
    artifact = tmp_path / 'broken.gz'
    artifact.write_bytes(b'\x1f\x8b\x08broken')
    result = scan(artifact)
    assert result.returncode == 75
    assert 'not scanned' in result.stdout + result.stderr


def test_directory_traversal_errors_fail_closed(tmp_path, monkeypatch, capsys):
    scanner = load_scanner()
    monkeypatch.setenv('SCAN_FAKE_SECRET', SECRET)

    def denied(path, *, onerror, **kwargs):
        onerror(PermissionError(13, SECRET, str(tmp_path)))
        return iter(())

    monkeypatch.setattr(os, 'walk', denied)
    assert scanner.main(['--env', 'SCAN_FAKE_SECRET', str(tmp_path)]) == 75
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err


def test_zstd_scanned_or_explicitly_unavailable(tmp_path):
    secret = 'Q' * 64
    # Single-segment zstd frame with a final RLE block of 64 Q bytes.
    frame = bytes.fromhex('28b52ffd204003020051')
    artifact = tmp_path / 'diagnostic.zst'
    artifact.write_bytes(frame)
    result = scan(artifact, values={'SCAN_FAKE_SECRET': secret})
    try:
        from compression import zstd
    except ImportError:
        assert result.returncode == 0
        assert 'zstd: not scanned' in result.stdout + result.stderr
    else:
        assert zstd.decompress(frame) == secret.encode()
        assert result.returncode == 1
    assert secret not in result.stdout + result.stderr


def zstd_raw_frame(payload):
    """Build a single-segment frame with one final raw block, without a decoder."""
    assert len(payload) < 256
    return (bytes.fromhex('28b52ffd20') + bytes([len(payload)])
            + ((len(payload) << 3) | 1).to_bytes(3, 'little') + payload)


def test_execution_log_zstd_with_stray_gzip_magic_is_clean(tmp_path):
    # Synthetic length-delimited protobuf record, like a Bazel execution log.
    # Its raw zstd block deliberately contains gzip magic with no gzip member.
    record = b'\x0a\x0a/bin/clang\x12\x04\x1f\x8b\x00\xff'
    frame = zstd_raw_frame(bytes([len(record)]) + record)
    scanner = load_scanner()
    if scanner.zstd is not None:
        assert scanner.zstd.decompress(frame) == bytes([len(record)]) + record
    artifact = tmp_path / 'execution_log.binpb.zst'
    artifact.write_bytes(frame)
    result = scan(artifact)
    assert result.returncode == 0, result.stdout + result.stderr
    if scanner.zstd is None:
        assert 'zstd: not scanned (compression.zstd unavailable)' in result.stdout


@pytest.mark.parametrize('planted', [False, True])
def test_unknown_binary_magic_does_not_prevent_raw_scan(tmp_path, planted):
    artifact = tmp_path / 'unknown.bin'
    artifact.write_bytes(b'\x00binary\x1f\x8b\x00\xff\x28\xb5\x2f\xfd\xff'
                         + (SECRET.encode() if planted else b'clean'))
    result = scan(artifact)
    assert result.returncode == (1 if planted else 0), result.stdout + result.stderr
    assert SECRET not in result.stdout + result.stderr


@pytest.mark.parametrize('wrapper', ['plain', 'embedded', 'gzip', 'zstd'])
def test_concatenated_zstd_frames_scan_compressed_secret(tmp_path, wrapper):
    secret = 'Q' * 64
    # RLE hides the planted value from the raw byte scan.
    frames = zstd_raw_frame(b'clean\x1f\x8b\x00\xff') + bytes.fromhex('28b52ffd204003020051')
    if wrapper == 'embedded':
        frames = b'prefix\x00' + frames
    elif wrapper == 'gzip':
        frames = gzip.compress(frames)
    elif wrapper == 'zstd':
        frames = zstd_raw_frame(frames)
    artifact = tmp_path / 'nested.zst'
    artifact.write_bytes(frames)
    result = scan(artifact, values={'SCAN_FAKE_SECRET': secret})
    if load_scanner().zstd is None:
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'zstd: not scanned (compression.zstd unavailable)' in result.stdout
    else:
        assert result.returncode == 1, result.stdout + result.stderr
    assert secret not in result.stdout + result.stderr


def test_corrupt_concatenated_gzip_is_not_silently_passed(tmp_path):
    artifact = tmp_path / 'broken-concatenated.gz'
    artifact.write_bytes(gzip.compress(b'clean') + b'\x1f\x8b\x08broken')
    result = scan(artifact)
    assert result.returncode == 75
    assert 'gzip member unreadable, not scanned' in result.stdout


def test_corrupt_zstd_is_not_silently_passed_when_supported(tmp_path):
    artifact = tmp_path / 'broken.zst'
    artifact.write_bytes(bytes.fromhex('28b52ffdff'))
    result = scan(artifact)
    if load_scanner().zstd is None:
        assert result.returncode == 0
        assert 'zstd: not scanned (compression.zstd unavailable)' in result.stdout
    else:
        assert result.returncode == 75
        assert 'zstd member unreadable, not scanned' in result.stdout


def test_portable_core_matches_cross_repo_pin():
    expected = (SCANNER.with_suffix('.sha256')).read_text().split()[0]
    assert hashlib.sha256(SCANNER.read_bytes()).hexdigest() == expected, (
        'Change the shared scanner core only in step across PrismaQuant, TL3, '
        'PrismaBuild and Tessera; vendor byte-identical files and update each pin.'
    )
