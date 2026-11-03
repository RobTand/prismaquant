"""Credential isolation and required scans for the remote gate's upload boundary."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

CHILD_ENVIRONMENT = {'PATH', 'HOME', 'TMPDIR', 'TMP', 'TEMP', 'LANG', 'LC_ALL',
                     'BAZELISK_HOME', 'JAVA_HOME'}
SCANNER_SPEC = importlib.util.spec_from_file_location(
    'pq_cutover_secret_scanner', Path(__file__).with_name('check_no_secret_leak.py'))
scanner = importlib.util.module_from_spec(SCANNER_SPEC)
SCANNER_SPEC.loader.exec_module(scanner)


def child_environment(environ=None):
    environ = os.environ if environ is None else environ
    return {name: value for name, value in environ.items() if name in CHILD_ENVIRONMENT}


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
