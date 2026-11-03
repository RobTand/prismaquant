"""Credential isolation and required scans for the remote gate's upload boundary."""
from __future__ import annotations

import importlib.util
import os
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
    arguments = [word for name in required for word in ('--env', name)]
    return scanner.main([*arguments, str(directory)])
