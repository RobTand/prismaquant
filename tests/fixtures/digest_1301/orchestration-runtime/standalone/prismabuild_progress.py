# Frozen executable excerpts from git blob 8d2b281527a61910f2550857fadd3e88d5eed85e; docstrings removed only.
from __future__ import annotations
import json
import os
import time
from pathlib import Path
RECORD_SCHEMA = 'prismabuild.action_progress.v1'
PATH_ENV = 'PRISMABUILD_ACTION_PROGRESS_PATH'
TOKEN_ENV = 'PRISMABUILD_ACTION_PROGRESS_TOKEN'

def report(phase: str, units_completed: int, *, unit: str='anchors') -> bool:
    destination = os.environ.get(PATH_ENV) or ''
    token = os.environ.get(TOKEN_ENV) or ''
    if not destination or not token:
        return False
    record = {'schema': RECORD_SCHEMA, 'token': token, 'phase': str(phase), 'units_completed': int(units_completed), 'unit': str(unit), 'reported_unix': time.time()}
    path = Path(destination)
    try:
        temporary = path.parent / f'.{path.name}.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(record, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    except OSError:
        return False
    return True
_DEV_SOURCE_SHA256_MEMO: str | None = None

def commit(units, phase, unit=None):
    path = os.environ.get(PATH_ENV)
    token = os.environ.get(TOKEN_ENV)
    if not path or not token:
        return False
    record = {'schema': RECORD_SCHEMA, 'token': token, 'phase': phase, 'units_completed': units, 'unit': unit, 'reported_unix': time.time()}
    if os.environ.get('PRISMAQUANT_DEV_PROGRESS_STAMP') == '1':
        from . import dev_mode
        if dev_mode.dev_mode_enabled():
            record.update(dev_mode.dev_stamp(_progress_dev_source_sha256()))
    temporary = f'{path}.{os.getpid()}.tmp'
    with open(temporary, 'w') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')
    os.replace(temporary, path)
    return True
STAGED_WAIT_SCHEMA = 'prismabuild.staged_wait.v1'
STAGED_WAIT_SUFFIX = '.staged-wait'

def declare_staged_wait(movers, *, since_unix: float) -> bool:
    destination = os.environ.get(PATH_ENV) or ''
    token = os.environ.get(TOKEN_ENV) or ''
    names = sorted({str(mover) for mover in movers})
    if not destination or not token or (not names):
        return False
    record = {'schema': STAGED_WAIT_SCHEMA, 'token': token, 'since_unix': float(since_unix), 'movers': names}
    path = Path(destination + STAGED_WAIT_SUFFIX)
    try:
        temporary = path.parent / f'.{path.name}.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(record, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    except OSError:
        return False
    return True
EXPORT_WAIT_SCHEMA = 'prismabuild.export_wait.v1'
EXPORT_WAIT_SUFFIX = '.export-wait'

def declare_export_wait(exports, *, since_unix: float) -> bool:
    destination = os.environ.get(PATH_ENV) or ''
    token = os.environ.get(TOKEN_ENV) or ''
    names = sorted({str(export) for export in exports})
    if not destination or not token or (not names):
        return False
    record = {'schema': EXPORT_WAIT_SCHEMA, 'token': token, 'since_unix': float(since_unix), 'exports': names}
    path = Path(destination + EXPORT_WAIT_SUFFIX)
    try:
        temporary = path.parent / f'.{path.name}.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(record, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    except OSError:
        return False
    return True
