"""Finite prefill documents; native execution remains with the shared Flux owner."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import re
from fractions import Fraction
import subprocess

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
SUBMISSION_SCHEMA = 'prismaquant.quality_prefill_submission.v1'
COLLECTION_SCHEMA = 'prismaquant.quality_prefill_collection.v1'
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


def _phase_layout(root: Path, manifest_binding: dict, manifest: dict,
                  roster_binding: dict, roster: dict) -> tuple:
    root = Path(_path(os.fspath(root)))
    tasks = {task['task_id']: task for task in roster['tasks']}
    envelopes, children = [], []
    for index, child in enumerate(roster['children']):
        envelope = dict(schema=ENVELOPE_SCHEMA, phase_id=roster['phase_id'],
            plan_identity_sha256=manifest['identity_sha256'], manifest=manifest_binding,
            roster=roster_binding, phase_record_dir=str(root), child_id=child['child_id'],
            tasks=[tasks[task] for task in child['task_ids']],
            result_path=str(root / f'results-{index:04d}.json'))
        raw = scientific.canonical_json_bytes(envelope) + b'\n'
        _check.require(len(raw) <= MAX_ENVELOPE_BYTES, 'child envelope exceeds1MiB')
        binding = dict(path=str(root / f'envelope-{index:04d}.json'),
                       sha256=bytes_sha256hex(raw), bytes=len(raw))
        envelopes.append(envelope)
        children.append(dict(child_id=child['child_id'], task_ids=child['task_ids'],
                             envelope=binding, record_dir=str(root / f'native-{index:04d}')))
    return envelopes, children


def _phase_records(manifest_binding: dict, phase: str, root: Path) -> dict:
    root = Path(_path(os.fspath(root)))
    directory = open_directory(str(root))
    os.close(directory)
    manifest, roster_binding, roster = _read_phase(manifest_binding, phase)
    envelopes, children = _phase_layout(root, manifest_binding, manifest, roster_binding, roster)
    raw, _ = read_document(root / 'submission.json', 1 << 20)
    submission = scientific.decode_strict_json(raw.decode('utf-8'), where='phase submission')
    _check.exact_mapping(submission, keys=frozenset({
        'schema', 'phase_id', 'plan_identity_sha256', 'roster_sha256', 'afterok', 'children',
    }), where='phase submission')
    expected = dict(schema=SUBMISSION_SCHEMA, phase_id=phase,
        plan_identity_sha256=manifest['identity_sha256'], roster_sha256=roster_binding['sha256'],
        afterok=submission['afterok'], children=children)
    _check.require(submission == expected, 'phase submission differs from frozen layout')
    parents = submission['afterok']
    _check.require(isinstance(parents, list) and len(parents) <= 4096,
                   'afterok must be a finite list of at most4096')
    ids, coordinates = set(), set()
    for parent in parents:
        _check.exact_mapping(parent, keys=frozenset({
            'phase_id', 'child_id', 'job_id', 'record_dir', 'launch_sha256',
        }), where='parent coordinate')
        _identifier(parent['phase_id'], 'parent phase'); _identifier(parent['child_id'], 'parent child')
        _path(parent['record_dir']); digest(parent['launch_sha256'])
        number = flux_jobs.job_id_number(parent['job_id'])
        key = (parent['phase_id'], parent['child_id'])
        _check.require(number not in ids and key not in coordinates, 'duplicate parent coordinate')
        ids.add(number); coordinates.add(key)
    for expected_envelope, child in zip(envelopes, children):
        _check.require(read_task_envelope(child['envelope']) == expected_envelope,
                       'phase envelope differs from frozen layout')
    return dict(root=root, manifest_binding=manifest_binding, manifest=manifest,
                roster_binding=roster_binding, roster=roster, envelopes=envelopes,
                submission=submission)


def _parent_from_native(submission: flux_jobs.FluxSubmission, manifest_binding: dict) -> dict:
    launch = flux_jobs._saved_launch(submission)
    argv = launch['argv']
    _check.require(isinstance(argv, list) and len(argv) == 8 and
                   argv[1:3] == ['-s', '-m'] and argv[4] == '--envelope' and
                   argv[6] == '--envelope-sha256', 'parent has no exact prefill envelope argv')
    _path(argv[5]); digest(argv[7])
    raw, _ = read_document(argv[5], MAX_ENVELOPE_BYTES)
    _check.require(bytes_sha256hex(raw) == argv[7], 'parent envelope digest differs from launch')
    binding = dict(path=argv[5], sha256=argv[7], bytes=len(raw))
    envelope = read_task_envelope(binding)
    _check.require(envelope['manifest'] == manifest_binding, 'parent belongs to a different manifest')
    _, _, roster = _read_phase(manifest_binding, envelope['phase_id'])
    _check.require(argv == _child_argv(binding, roster), 'parent command differs from declared roster')
    return envelope


def _expected_child(records: dict, index: int) -> dict:
    child = records['roster']['children'][index]
    return dict(argv=_child_argv(records['submission']['children'][index]['envelope'], records['roster']),
        source=_prefill_source(records['manifest']), resources=FluxResources(**child['resources']),
        afterok=[parent['job_id'] for parent in records['submission']['afterok']],
        environment=None, staged_input=child['staged_input'])


def _authenticate_graph(manifest_binding: dict, phase: str, root: Path, *, prepared=None) -> dict:
    cached, active = {}, set()
    def visit(selected_phase, selected_root):
        selected_root = Path(selected_root)
        _check.require(selected_phase not in active, 'dependency cycle')
        if selected_phase in cached:
            _check.require(cached[selected_phase]['root'] == selected_root,
                           'phase has conflicting record roots')
            return cached[selected_phase]
        active.add(selected_phase)
        records = (prepared if selected_phase == phase and prepared is not None else
                   _phase_records(manifest_binding, selected_phase, selected_root))
        dependencies = records['manifest']['execution']['phases'][selected_phase]['dependencies']
        expected_cover, observed_cover, parents = [], [], []
        for dependency in dependencies:
            _, _, parent_roster = _read_phase(manifest_binding, dependency)
            expected_cover.extend((dependency, child['child_id']) for child in parent_roster['children'])
        for coordinate in records['submission']['afterok']:
            parent_phase = coordinate['phase_id']
            _check.require(parent_phase in dependencies, 'foreign parent phase')
            saved = flux_jobs.load_submission(Path(coordinate['record_dir']))
            _check.require(saved.launch_sha256 == coordinate['launch_sha256'] and
                flux_jobs.job_id_number(saved.job_id) == flux_jobs.job_id_number(coordinate['job_id']),
                'parent native coordinates differ')
            envelope = _parent_from_native(saved, manifest_binding)
            _check.require((envelope['phase_id'], envelope['child_id']) ==
                           (parent_phase, coordinate['child_id']), 'parent envelope coordinate differs')
            parent_records = visit(parent_phase, Path(envelope['phase_record_dir']))
            index = next((i for i, child in enumerate(parent_records['roster']['children'])
                          if child['child_id'] == coordinate['child_id']), None)
            _check.require(index is not None and
                parent_records['submission']['children'][index]['record_dir'] == str(saved.record_dir),
                'parent record is outside its frozen population')
            flux_jobs.verify_native_launch(saved, max_evidence_bytes=16 << 20,
                                          **_expected_child(parent_records, index))
            observed_cover.append((parent_phase, coordinate['child_id']))
            parents.append((parent_records, index, saved))
        _check.require(observed_cover == expected_cover, 'parent child cover differs from declared DAG')
        records['parents'] = parents
        cached[selected_phase] = records
        active.remove(selected_phase)
        return records
    return visit(phase, root)


def submit_phase(manifest_binding: dict, *, phase: str, checkout: Path,
                 record_dir: Path, afterok: tuple[flux_jobs.FluxSubmission, ...] = ()) -> dict:
    """Submit one explicit finite roster once; partial/unknown launches never retry."""
    manifest, roster_binding, roster = _read_phase(manifest_binding, phase)
    _validate_admission(manifest, phase, roster)
    source = _prefill_source(manifest)
    _check.require(Path(checkout) == Path(source['checkout']), 'prefill checkout differs from frozen source')
    root = Path(_path(os.fspath(record_dir)))
    _check.require(root.is_relative_to('/mnt/shared') and not root.is_relative_to(checkout),
                   'prefill records require a separate shared phase directory')
    _check.require(isinstance(afterok, tuple) and len(afterok) <= 4096 and
                   all(isinstance(parent, flux_jobs.FluxSubmission) for parent in afterok),
                   'afterok requires a finite explicit FluxSubmission tuple')
    parents, ids, coordinates = [], set(), set()
    for saved in afterok:
        envelope = _parent_from_native(saved, manifest_binding)
        number = flux_jobs.job_id_number(saved.job_id)
        key = (envelope['phase_id'], envelope['child_id'])
        _check.require(number not in ids and key not in coordinates, 'duplicate parent submission')
        ids.add(number); coordinates.add(key)
        parents.append(dict(phase_id=key[0], child_id=key[1], job_id=saved.job_id,
                            record_dir=str(saved.record_dir), launch_sha256=saved.launch_sha256))
    envelopes, children = _phase_layout(root, manifest_binding, manifest, roster_binding, roster)
    submission = dict(schema=SUBMISSION_SCHEMA, phase_id=phase,
        plan_identity_sha256=manifest['identity_sha256'], roster_sha256=roster_binding['sha256'],
        afterok=parents, children=children)
    _check.require(len(scientific.canonical_json_bytes(submission)) + 1 <= 1 << 20,
                   'phase submission exceeds1MiB')
    prepared = dict(root=root, manifest_binding=manifest_binding, manifest=manifest,
        roster_binding=roster_binding, roster=roster, envelopes=envelopes, submission=submission)
    _authenticate_graph(manifest_binding, phase, root, prepared=prepared)
    flux_jobs._profile_identity()
    for child in roster['children']:
        measured = flux_jobs.validate_job(checkout=checkout, resources=FluxResources(**child['resources']),
                                          staged_input=child['staged_input'])
        _check.require(measured == source, 'source changed during phase preflight')
    parent_fd = open_directory(str(root.parent))
    try:
        os.mkdir(root.name, mode=0o700, dir_fd=parent_fd)
        directory = open_directory(str(root))
    finally:
        os.close(parent_fd)
    failure = None
    try:
        witness = os.fstat(directory)
        identity = (witness.st_dev, witness.st_ino)
        for envelope, child in zip(envelopes, children):
            _check.require(_publish_document(envelope, Path(child['envelope']['path']),
                MAX_ENVELOPE_BYTES, directory_identity=identity) == child['envelope'],
                'published envelope differs from frozen bytes')
        _publish_document(submission, root / 'submission.json', 1 << 20, directory_identity=identity)
        for index, child in enumerate(children):
            held = open_directory(str(root), identity=identity)
            os.close(held)
            try:
                intent = _expected_child(prepared, index)
                flux_jobs.submit_job(intent['argv'], checkout=checkout, resources=intent['resources'],
                    record_dir=Path(child['record_dir']), afterok=intent['afterok'],
                    staged_input=intent['staged_input'])
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                failure = f"submit {child['child_id']}: {type(exc).__name__}: {str(exc)[:512]}"
                break
    finally:
        os.close(directory)
    result = collect_phase(manifest_binding, phase=phase, record_dir=root, wait=False)
    if failure is not None:
        result['errors'].append(failure)
        if result['state'] != 'failed':
            result['state'] = 'unknown'
    return result


def _selected_child(records: dict, index: int, saved: flux_jobs.FluxSubmission) -> tuple:
    child = records['submission']['children'][index]
    envelope = records['envelopes'][index]
    stdout, _ = read_document(saved.record_dir / 'stdout', 64 << 20)
    prefix = native_results.ANNOUNCEMENT_PREFIX
    announcements = [line[len(prefix):] for line in stdout.splitlines() if line.startswith(prefix)]
    _check.require(len(announcements) == 1, 'missing or ambiguous native publication announcement')
    announcement = scientific.decode_strict_json(announcements[0].decode('utf-8'), where='announcement')
    _check.exact_mapping(announcement, keys=frozenset({'schema', 'path', 'sha256', 'bytes'}),
                         where='publication announcement')
    _check.require(announcement['schema'] == native_results.ANNOUNCEMENT_SCHEMA and
                   announcement['path'] == envelope['result_path'] + '.publication.json',
                   'publication announcement differs from declared path')
    publication = {key: announcement[key] for key in ('path', 'sha256', 'bytes')}
    expected = _expected_child(records, index)
    expected['resources'] = asdict(expected['resources'])
    selection = dict(schema=native_results.SELECTION_SCHEMA, job_id=saved.job_id,
        record_dir=str(saved.record_dir), launch_sha256=saved.launch_sha256,
        publication=publication, expected_launch=expected, producer_id=envelope['child_id'],
        inputs={key: envelope[key] for key in ('manifest', 'roster')} | {'envelope': child['envelope']},
        runtime=records['roster']['runtime'], max_publication_bytes=16 << 20,
        max_evidence_bytes=64 << 20, max_artifact_bytes=_NATIVE_ARTIFACT_BYTES)
    selected = native_results.read_selected_result(selection, artifact_limits={'prefill_results': MAX_RESULT_BYTES})
    document = scientific.decode_strict_json(selected.artifact_payloads['prefill_results'].decode('utf-8'),
                                             where='selected task results')
    _check.require(isinstance(document, dict) and
                   document == _result_document(envelope, document.get('results')),
                   'selected result metadata differs from frozen child')
    _authenticate_measurements(document)
    return selection, document['results']


def _execution_value(result):
    return dict(status=result.status, job_id=result.job_id, record_dir=str(result.record_dir),
                payload_exit=result.payload_exit, query_error=result.query_error)


def collect_phase(manifest_binding: dict, *, phase: str, record_dir: Path, wait: bool = True) -> dict:
    """Collect every known child, preserving native failures apart from missing evidence."""
    _check.require(type(wait) is bool, 'wait must be boolean')
    records = _phase_records(manifest_binding, phase, record_dir)
    errors, outcomes, children, task_results = [], [], [], []
    parents_ok = True
    try:
        records = _authenticate_graph(manifest_binding, phase, record_dir)
        pending = list(records['parents'])
        seen, expanded = set(), set()
        while pending:
            parent_records, index, saved = pending.pop()
            parent_phase = parent_records['roster']['phase_id']
            if parent_phase not in expanded:
                pending.extend(parent_records['parents'])
                expanded.add(parent_phase)
            number = flux_jobs.job_id_number(saved.job_id)
            if number in seen:
                continue
            seen.add(number)
            try:
                execution = flux_jobs.collect_job(saved, wait=wait)
                outcomes.append(execution)
                _check.require(execution.status == 'succeeded' and not execution.query_error,
                               'parent execution is not successful and available')
                _selected_child(parent_records, index, saved)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                if isinstance(exc, native_results.ProducerFailed):
                    outcomes.append(exc.execution)
                parents_ok = False
                errors.append(f'parent {saved.job_id}: {type(exc).__name__}: {str(exc)[:512]}')
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parents_ok = False
        errors.append(f'parent authority: {type(exc).__name__}: {str(exc)[:512]}')
    for index, child in enumerate(records['submission']['children']):
        row = dict(child_id=child['child_id'], execution=None, selection=None)
        children.append(row)
        try:
            saved = flux_jobs.load_submission(Path(child['record_dir']))
            execution = flux_jobs.collect_job(saved, wait=wait)
            outcomes.append(execution)
            row['execution'] = _execution_value(execution)
            flux_jobs.verify_native_launch(saved, max_evidence_bytes=16 << 20,
                                          **_expected_child(records, index))
            if execution.status == 'succeeded' and not execution.query_error and parents_ok:
                selection, results = _selected_child(records, index, saved)
                row['selection'] = selection
                task_results.extend(dict(task_id=result['task_id'], output_id=result['output_id'],
                    child_id=child['child_id'], value=result['value'], value_sha256=result['value_sha256'],
                    selection=selection) for result in results)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            if isinstance(exc, native_results.ProducerFailed):
                outcomes.append(exc.execution)
                row['execution'] = _execution_value(exc.execution)
            errors.append(f"child {child['child_id']}: {type(exc).__name__}: {str(exc)[:512]}")
    if any(result.status in {'failed', 'canceled', 'dependency-failed'} for result in outcomes):
        state = 'failed'
    elif not errors and parents_ok and len(task_results) == len(records['roster']['tasks']):
        state = 'complete'
    elif any(result.status == 'running' for result in outcomes):
        state = 'running'
    elif any(row['execution'] is None for row in children) or any(
            result.status == 'unknown' or result.query_error for result in outcomes):
        state = 'unknown'
    else:
        state = 'incomplete'
    return dict(schema=COLLECTION_SCHEMA, phase_id=phase,
        plan_identity_sha256=records['manifest']['identity_sha256'],
        roster_sha256=records['roster_binding']['sha256'], state=state,
        children=children, task_results=task_results, errors=errors)


def cancel_phase(manifest_binding: dict, *, phase: str, record_dir: Path) -> dict:
    """Cancel only authenticated acknowledged children; never parents or discovered work."""
    records = _phase_records(manifest_binding, phase, record_dir)
    errors = []
    try:
        records = _authenticate_graph(manifest_binding, phase, record_dir)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        result = collect_phase(manifest_binding, phase=phase, record_dir=record_dir, wait=False)
        result['errors'].append(f'cancel authority: {type(exc).__name__}: {str(exc)[:512]}')
        if result['state'] == 'complete':
            result['state'] = 'unknown'
        return result
    for index, child in enumerate(records['submission']['children']):
        try:
            saved = flux_jobs.load_submission(Path(child['record_dir']))
            flux_jobs.verify_native_launch(saved, max_evidence_bytes=16 << 20,
                                          **_expected_child(records, index))
            outcome = flux_jobs.cancel_job(saved)
            if outcome.query_error:
                errors.append(f"cancel {child['child_id']}: {outcome.query_error[:512]}")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            errors.append(f"cancel {child['child_id']}: {type(exc).__name__}: {str(exc)[:512]}")
    result = collect_phase(manifest_binding, phase=phase, record_dir=record_dir, wait=not errors)
    result['errors'].extend(errors)
    if errors and result['state'] == 'complete':
        result['state'] = 'unknown'
    return result


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
    records = _authenticate_graph(envelope['manifest'], envelope['phase_id'],
                                   Path(envelope['phase_record_dir']))
    index = next(index for index, row in enumerate(records['submission']['children'])
                 if row['child_id'] == envelope['child_id'])
    _check.require(records['submission']['children'][index]['envelope'] == envelope_binding and
        context['record_dir'] == records['submission']['children'][index]['record_dir'] and
        [flux_jobs.job_id_number(value) for value in context['afterok']] ==
        [flux_jobs.job_id_number(parent['job_id']) for parent in records['submission']['afterok']],
        'current phase record or dependencies differ from frozen submission')
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


def _publish_document(value: object, path: Path, limit: int, *, directory_identity=None) -> dict:
    """Publish checked document bytes once through the existing writer owner."""
    filename = Path(_path(os.fspath(path)))
    raw = scientific.canonical_json_bytes(value) + b'\n'
    _check.require(len(raw) <= limit, 'document exceeds byte limit')
    directory = open_directory(str(filename.parent), identity=directory_identity)
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
