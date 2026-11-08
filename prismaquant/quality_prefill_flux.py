"""Finite prefill documents; native execution remains with the shared Flux owner."""
from __future__ import annotations

import os
from pathlib import Path
import re
from fractions import Fraction

from . import quality_prefill_contract as scientific
from .cost_stage_checkpoint import publish_new_bytes
from .digests import bytes_sha256hex
from .flux_jobs import FluxResources
from .native_results import _owned, validate_runtime
from .schemas import Contract
from .staged_launch import validate_binding as validate_staged_binding
from .staged_manifest import digest, open_directory, path as _path, read_document


ROSTER_SCHEMA = 'prismaquant.quality_prefill_roster.v1'
MAX_ROSTER_BYTES = 16 << 20
MAX_TASKS = scientific.MAX_PREFILL_TASKS
MAX_CHILDREN = scientific.MAX_PREFILL_CHILDREN
MAX_CHILD_TASKS = scientific.MAX_PREFILL_CHILD_TASKS
MAX_INPUTS = scientific.MAX_PREFILL_INPUTS
_MODULE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z')
_check = Contract(scientific.QualityPrefillContractError, 'prefill roster: ')


def _identifier(value, where):
    return scientific.prefill_identifier(value, where=where)


def _sequence(value, maximum, where):
    _check.require(isinstance(value, list) and 0 < len(value) <= maximum,
                   f'{where} must be a finite nonempty list of at most {maximum}')
    return value


def _binding(value, where):
    return scientific.prefill_file_binding(value, where=where)


def _validate_admission(manifest: dict, phase: str, roster: dict) -> None:
    """Authenticate the declared measured work before creating native records."""
    reference = manifest['execution']['setup_cost_reference']
    raw, _ = read_document(reference['path'], scientific.MAX_WORK_ESTIMATES_BYTES)
    _check.require(bytes_sha256hex(raw) == reference['sha256'], 'work estimates digest differs')
    estimates = scientific.validate_work_estimates(
        scientific.decode_strict_json(raw.decode('utf-8'), where='work estimates'))
    _check.require(phase in estimates['phases'], 'selected phase has no work estimates')
    declared = estimates['phases'][phase]
    for key in ('command', 'runtime'):
        _check.require(declared[key + '_sha256'] == scientific.canonical_sha256(roster[key]),
                       f'estimated {key} differs from roster')
    children = roster['children']
    _check.require([c['child_id'] for c in declared['children']] ==
                   [c['child_id'] for c in children], 'estimated child cover differs')
    tasks = {task['task_id']: task for task in roster['tasks']}
    evidence, total = {}, Fraction(0)
    for child, estimate in zip(children, declared['children']):
        _check.require(estimate['resources_sha256'] == scientific.canonical_sha256(child['resources']),
                       'estimated resources differ')
        _check.require([t['task_id'] for t in estimate['tasks']] == child['task_ids'],
                       'estimated task cover differs')
        seconds = Fraction(str(estimate['setup_seconds']))
        bindings = [estimate['setup_evidence']]
        for task in estimate['tasks']:
            _check.require(task['payload_sha256'] ==
                           scientific.canonical_sha256(tasks[task['task_id']]['payload']),
                           'estimated task payload differs')
            seconds += Fraction(str(task['estimated_seconds']))
            bindings.append(task['estimate_evidence'])
        _check.require(seconds <= child['resources']['walltime_s'], 'estimate exceeds child walltime')
        total += seconds
        for binding in bindings:
            evidence[(binding['path'], binding['sha256'], binding['bytes'])] = binding
    limits = manifest['execution']['phases'][phase]
    _check.require(len(tasks) <= limits['work_limit'], 'phase work_limit exceeded')
    _check.require(total <= limits['cost_limit'], 'phase cost_limit exceeded')
    _check.require(sum(b['bytes'] for b in evidence.values()) <= scientific.MAX_WORK_ESTIMATES_BYTES,
                   'aggregate estimate evidence exceeds limit')
    for binding in evidence.values():
        _owned(binding, scientific.MAX_ESTIMATE_EVIDENCE_BYTES)


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
    return _publish_document(validate_phase_roster(value), path, MAX_ROSTER_BYTES)


def _read_bound_document(binding: dict, limit: int) -> object:
    """Authenticate bounded contents before strict document decoding."""
    raw = _owned(binding, limit)
    return scientific.decode_strict_json(raw.decode('utf-8'), where='prefill document')


def _publish_document(value: object, path: Path, limit: int) -> dict:
    """Publish checked document bytes once through the existing writer owner."""
    filename = Path(_path(os.fspath(path)))
    raw = scientific.canonical_json_bytes(value) + b'\n'
    _check.require(len(raw) <= limit, 'document exceeds byte limit')
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
        actual, signature = read_document(filename, limit)
        _check.require(signature == written[str(filename)] and actual == raw,
                       'published roster differs from owned writer')
        witness = open_directory(str(filename.parent), identity=identity)
        os.close(witness)
    finally:
        os.close(directory)
    return {'path': str(filename), 'sha256': bytes_sha256hex(raw), 'bytes': len(raw)}
