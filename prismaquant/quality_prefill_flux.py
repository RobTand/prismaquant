"""Finite prefill documents; native execution remains with the shared Flux owner."""
from __future__ import annotations

import os
from pathlib import Path
import re
from fractions import Fraction

from . import flux_jobs, native_results
from . import quality_prefill_contract as scientific
from .cost_stage_checkpoint import publish_new_bytes
from .digests import bytes_sha256hex
from .flux_jobs import FluxResources
from .native_results import _owned, validate_runtime
from .schemas import Contract
from .staged_launch import validate_binding as validate_staged_binding
from .staged_manifest import digest, open_directory, path as _path, read_document


ROSTER_SCHEMA = 'prismaquant.quality_prefill_roster.v1'
ENVELOPE_SCHEMA = 'prismaquant.quality_prefill_envelope.v1'
RESULTS_SCHEMA = 'prismaquant.quality_prefill_results.v1'
MAX_ROSTER_BYTES = 16 << 20
MAX_ENVELOPE_BYTES = 1 << 20
MAX_RESULT_BYTES = 1 << 20
_NATIVE_ARTIFACT_BYTES = 34 << 20
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


def _plan_context_sha256(manifest: dict) -> str:
    body = scientific.decode_strict_json(scientific.canonical_json_bytes(manifest).decode('utf-8'),
                                         where='manifest context')
    body.pop('identity_sha256', None)
    for entry in body['execution']['phases'].values():
        entry.pop('task_roster', None)
    return scientific.canonical_sha256({
        'schema': 'prismaquant.quality_prefill_plan_context.v1', 'manifest': body})


def _read_phase(manifest_binding: dict, phase: str) -> tuple:
    _check.require(phase in scientific.PHASE_BY_NAME, 'unknown phase')
    manifest = scientific.validate_manifest(_read_bound_document(manifest_binding, MAX_ROSTER_BYTES))
    roster_binding = manifest['execution']['phases'][phase]['task_roster']
    roster = validate_phase_roster(_read_bound_document(roster_binding, MAX_ROSTER_BYTES))
    _check.require(roster['phase_id'] == phase, 'roster phase differs')
    _check.require(roster['plan_context_sha256'] == _plan_context_sha256(manifest),
                   'roster plan context differs')
    return manifest, roster_binding, roster


def read_task_envelope(binding: dict) -> dict:
    """Read the exact frozen child population without trusting its own claims."""
    body = _read_bound_document(binding, MAX_ENVELOPE_BYTES)
    _check.exact_mapping(body, keys=frozenset({
        'schema', 'phase_id', 'plan_identity_sha256', 'manifest', 'roster',
        'phase_record_dir', 'child_id', 'tasks', 'result_path',
    }), where='task envelope')
    _check.require(body['schema'] == ENVELOPE_SCHEMA, 'envelope schema differs')
    _identifier(body['phase_id'], 'envelope phase'); _identifier(body['child_id'], 'envelope child')
    digest(body['plan_identity_sha256'])
    _binding(body['manifest'], 'manifest'); _binding(body['roster'], 'roster')
    _sequence(body['tasks'], MAX_CHILD_TASKS, 'envelope tasks')
    _path(body['result_path'])
    root = Path(_path(body['phase_record_dir']))
    directory = open_directory(str(root))
    os.close(directory)
    _check.require(Path(binding['path']).parent == root and
                   Path(body['result_path']).parent == root,
                   'envelope and result must be in the explicit phase directory')
    manifest, declared_roster, roster = _read_phase(body['manifest'], body['phase_id'])
    _check.require(body['plan_identity_sha256'] == manifest['identity_sha256'],
                   'envelope plan identity differs')
    _check.require(body['roster'] == declared_roster, 'envelope roster differs from manifest')
    children = [child for child in roster['children'] if child['child_id'] == body['child_id']]
    _check.require(len(children) == 1, 'envelope child is not declared')
    tasks = {task['task_id']: task for task in roster['tasks']}
    _check.require(body['tasks'] == [tasks[task] for task in children[0]['task_ids']],
                   'envelope does not match frozen ordered child tasks')
    protected = {binding['path'], body['manifest']['path'], body['roster']['path'],
                 str(root / 'submission.json'),
                 *(row['path'] for row in roster['inputs'].values())}
    _check.require(body['result_path'] not in protected, 'result path aliases an input')
    return body


def _result_document(envelope: dict, results: list) -> dict:
    document = dict(schema=RESULTS_SCHEMA, phase_id=envelope['phase_id'],
        plan_identity_sha256=envelope['plan_identity_sha256'],
        roster_sha256=envelope['roster']['sha256'], child_id=envelope['child_id'], results=results)
    raw = scientific.canonical_json_bytes(document)
    _check.require(len(raw) + 1 <= MAX_RESULT_BYTES, 'result document exceeds1MiB')
    document = scientific.decode_strict_json(raw.decode('utf-8'), where='prefill results')
    rows = _sequence(document['results'], MAX_CHILD_TASKS, 'task results')
    _check.require(len(rows) == len(envelope['tasks']), 'result task count differs')
    artifact_bytes = 0
    for row, task in zip(rows, envelope['tasks']):
        _check.exact_mapping(row, keys=frozenset({
            'task_id', 'output_id', 'value', 'value_sha256',
        }), where='task result')
        _check.require((row['task_id'], row['output_id']) == (task['task_id'], task['output_id']),
                       'result ordered task/output cover differs')
        measurement = scientific.validate_task_measurement(row['value'], payload=task['payload'])
        digest(row['value_sha256'])
        _check.require(row['value_sha256'] == scientific.canonical_sha256(measurement),
                       'measurement value digest differs')
        artifact_bytes += sum(binding['bytes'] for binding in measurement['artifacts'])
    _check.require(artifact_bytes <= 32 << 20, 'child measurement artifacts exceed32MiB')
    return document


def _authenticate_measurements(document: dict) -> None:
    artifacts = {}
    for row in document['results']:
        for binding in row['value']['artifacts']:
            previous = artifacts.setdefault(binding['path'], binding)
            _check.require(previous == binding, 'measurement artifact has conflicting bindings')
    for binding in artifacts.values():
        _owned(binding, 1 << 20)


def _prefill_source(manifest: dict) -> dict:
    commits = [entry['commit'] for entry in manifest['source']['dependency_commits']
               if entry['name'] == 'prismaquant']
    _check.require(len(commits) == 1, 'live prefill requires one prismaquant dependency commit')
    source = flux_jobs._source_identity(Path(__file__).resolve().parents[1])
    _check.require(source['commit'] == commits[0], 'prismaquant source differs from frozen commit')
    return source


def _child_argv(envelope_binding: dict, roster: dict) -> list:
    return [roster['command']['interpreter'], '-s', '-m', roster['command']['module'],
            '--envelope', envelope_binding['path'], '--envelope-sha256', envelope_binding['sha256']]


def publish_task_results(envelope_binding: dict, *, results: list[dict],
                         dependencies: tuple[str, ...] = ()) -> dict:
    """Publish actual admitted task bytes; selection still requires terminal success."""
    envelope = read_task_envelope(envelope_binding)
    manifest, _, roster = _read_phase(envelope['manifest'], envelope['phase_id'])
    _validate_admission(manifest, envelope['phase_id'], roster)
    document = _result_document(envelope, results)
    _authenticate_measurements(document)
    source = _prefill_source(manifest)
    child = next(child for child in roster['children'] if child['child_id'] == envelope['child_id'])
    context = native_results.execution_context()
    _check.require(context['source'] == source and context['argv'] == _child_argv(envelope_binding, roster)
        and context['resources'] == child['resources'] and context['staged_input'] == child['staged_input'],
        'current native launch differs from frozen prefill child')
    _check.require(isinstance(dependencies, tuple) and
                   tuple(sorted(dependencies)) == tuple(sorted(roster['runtime']['dependencies'])),
                   'runtime dependency roster differs')
    _check.require(native_results.measure_runtime(dependencies=dependencies) == roster['runtime'],
                   'current measured runtime differs from frozen roster')
    result_path = Path(envelope['result_path'])
    _publish_document(document, result_path, MAX_RESULT_BYTES)
    return native_results.publish_result(
        result_path.with_name(result_path.name + '.publication.json'), producer_id=envelope['child_id'],
        inputs={key: envelope[key] for key in ('manifest', 'roster')} | {'envelope': envelope_binding},
        artifacts={'prefill_results': result_path}, dependencies=dependencies,
        max_artifact_bytes=_NATIVE_ARTIFACT_BYTES)


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
