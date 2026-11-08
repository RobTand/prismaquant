"""Finite prefill documents; native execution remains with the shared Flux owner."""
from __future__ import annotations

import os
from pathlib import Path
import re

from . import quality_prefill_contract as scientific
from .cost_stage_checkpoint import publish_new_bytes
from .digests import bytes_sha256hex
from .flux_jobs import FluxResources
from .native_results import validate_runtime
from .schemas import Contract
from .staged_launch import validate_binding as validate_staged_binding
from .staged_manifest import digest, integer, open_directory, path as _path, read_document


ROSTER_SCHEMA = 'prismaquant.quality_prefill_roster.v1'
MAX_ROSTER_BYTES = 16 << 20
MAX_TASKS = 1024
MAX_CHILDREN = 256
MAX_CHILD_TASKS = 64
MAX_INPUTS = 32
_MODULE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z')
_check = Contract(scientific.QualityPrefillContractError, 'prefill roster: ')


def _identifier(value, where):
    return scientific.prefill_identifier(value, where=where)


def _sequence(value, maximum, where):
    _check.require(isinstance(value, list) and 0 < len(value) <= maximum,
                   f'{where} must be a finite nonempty list of at most {maximum}')
    return value


def _binding(value, where):
    _check.exact_mapping(value, keys=frozenset({'path', 'sha256', 'bytes'}), where=where)
    _path(value['path']); digest(value['sha256'])
    integer(value['bytes'])


def validate_phase_roster(value: object) -> dict:
    """Validate the finite ordered cover; staged files use their existing owner."""
    raw = scientific.canonical_json_bytes(value)
    _check.require(len(raw) + 1 <= MAX_ROSTER_BYTES, 'document exceeds byte limit')
    body = scientific.decode_strict_json(raw.decode('utf-8'), where='prefill roster')
    _check.exact_mapping(body, keys=frozenset({
        'schema', 'phase_id', 'plan_context_sha256', 'command', 'runtime',
        'inputs', 'tasks', 'children',
    }), where='roster')
    _check.require(body['schema'] == ROSTER_SCHEMA, 'schema differs')
    _identifier(body['phase_id'], 'phase_id'); digest(body['plan_context_sha256'])
    command = _check.exact_mapping(body['command'],
                                  keys=frozenset({'interpreter', 'module'}), where='command')
    _path(command['interpreter'])
    _check.string(command['module'], where='command.module', pattern=_MODULE)
    inputs = _check.mapping(body['inputs'], where='inputs')
    _check.require(len(inputs) <= MAX_INPUTS, 'input count exceeds limit')
    for role, binding in inputs.items():
        _identifier(role, 'input role'); _binding(binding, f'input {role}')
    tasks = _sequence(body['tasks'], MAX_TASKS, 'tasks')
    children = _sequence(body['children'], MAX_CHILDREN, 'children')
    task_ids, output_ids, child_ids, cover, staged = [], set(), set(), [], []
    for task in tasks:
        _check.exact_mapping(task, keys=frozenset({'task_id', 'output_id', 'payload'}), where='task')
        task_ids.append(_identifier(task['task_id'], 'task_id'))
        output = _identifier(task['output_id'], 'output_id')
        _check.require(output not in output_ids, 'duplicate output_id')
        output_ids.add(output)
        scientific.validate_task_payload(task['payload'], where=f"task {task['task_id']}")
    _check.require(len(task_ids) == len(set(task_ids)), 'duplicate task_id')
    for child in children:
        _check.exact_mapping(child, keys=frozenset({
            'child_id', 'task_ids', 'resources', 'staged_input',
        }), where='child')
        identity = _identifier(child['child_id'], 'child_id')
        _check.require(identity not in child_ids, 'duplicate child_id')
        child_ids.add(identity)
        cover.extend(_identifier(task, 'child task_id') for task in
                     _sequence(child['task_ids'], MAX_CHILD_TASKS, 'child tasks'))
        resources = _check.exact_mapping(child['resources'], keys=frozenset({
            'cores', 'memory_gib', 'gpus', 'walltime_s', 'placement',
        }), where='resources')
        admitted = FluxResources(**resources)
        if child['staged_input'] is not None:
            staged.append((child['staged_input'], admitted.placement))
    _check.require(cover == task_ids, 'children must cover every task exactly once in order')
    validate_runtime(body['runtime'])
    for binding, placement in staged:
        validate_staged_binding(binding, placement=placement)
    return body


def write_phase_roster(value: object, path: Path) -> dict:
    """Publish one new roster, preserving the held parent and exact writer inode."""
    body = validate_phase_roster(value)
    filename = Path(_path(os.fspath(path)))
    raw = scientific.canonical_json_bytes(body) + b'\n'
    directory = open_directory(str(filename.parent))
    written = {}
    try:
        observed = os.fstat(directory)
        identity = (observed.st_dev, observed.st_ino)
        if not publish_new_bytes(filename, raw, nofollow=True, writer_identities=written,
                                 writer_directories={str(filename.parent): identity}):
            raise FileExistsError(filename)
        witness = open_directory(str(filename.parent), identity=identity)
        os.close(witness)
        actual, signature = read_document(filename, MAX_ROSTER_BYTES)
        _check.require(signature == written[str(filename)] and actual == raw,
                       'published roster differs from owned writer')
        witness = open_directory(str(filename.parent), identity=identity)
        os.close(witness)
    finally:
        os.close(directory)
    return {'path': str(filename), 'sha256': bytes_sha256hex(raw), 'bytes': len(raw)}
