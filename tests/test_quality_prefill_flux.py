"""Finite application documents, without native execution or runtime measurement."""
import copy
import errno
import hashlib
import json
import os

import pytest

from prismaquant import quality_prefill_contract as scientific
from prismaquant import quality_prefill_flux as prefill


def payload():
    return dict(schema=scientific.TASK_PAYLOAD_SCHEMA, serving_unit='layer.0/up+gate',
                members=['layer.0/up', 'layer.0/gate'], family='e4m3', rate=832,
                context_id='prefill-2048', currency=scientific.JOINT_CURRENCY)


def admission_fixture(tmp_path, *, size=3, width=2):
    """Invented timing evidence for schema tests, never a measured live estimate."""
    from test_quality_prefill_contract import resolved_manifest
    phase = scientific.PHASE_DAG[0].phase
    value = roster(size, width=width)
    value['phase_id'] = phase
    evidence_path = tmp_path / 'synthetic-timing-evidence.txt'
    evidence_path.write_bytes(b'synthetic test evidence, not a scientific measurement')
    binding = dict(path=str(evidence_path), bytes=evidence_path.stat().st_size,
                   sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest())
    entry = dict(command_sha256=scientific.canonical_sha256(value['command']),
                 runtime_sha256=scientific.canonical_sha256(value['runtime']), children=[])
    tasks = {t['task_id']: t for t in value['tasks']}
    for child in value['children']:
        entry['children'].append(dict(child_id=child['child_id'],
            resources_sha256=scientific.canonical_sha256(child['resources']),
            setup_seconds=0.25, setup_evidence=copy.deepcopy(binding), tasks=[dict(
                task_id=task, payload_sha256=scientific.canonical_sha256(tasks[task]['payload']),
                estimated_seconds=0.5, estimate_evidence=copy.deepcopy(binding))
                for task in child['task_ids']]))
    estimates = dict(schema=scientific.WORK_ESTIMATES_SCHEMA, phases={phase: entry})
    manifest = resolved_manifest()
    manifest['execution']['phases'][phase].update(work_limit=size, cost_limit=2)
    estimates_path = tmp_path / 'work-estimates.json'
    def publish_estimates():
        estimates_path.write_text(json.dumps(estimates, allow_nan=False))
        manifest['execution']['setup_cost_reference'] = dict(path=str(estimates_path),
            sha256=hashlib.sha256(estimates_path.read_bytes()).hexdigest())
    publish_estimates()
    return manifest, phase, value, estimates, publish_estimates


def test_admission_accepts_known_exact_budget_and_zero_setup(tmp_path):
    manifest, phase, value, estimates, publish = admission_fixture(tmp_path)
    # Two setups at .25 and three tasks at .5 exactly reach the budget of 2s.
    prefill._validate_admission(scientific.freeze_manifest(manifest), phase,
                               prefill.validate_phase_roster(value))
    for child in estimates['phases'][phase]['children']:
        child['setup_seconds'] = 0
    publish()
    prefill._validate_admission(manifest, phase, value)


@pytest.mark.parametrize('kind', ['work', 'cost', 'wall', 'tiny-positive'])
def test_admission_enforces_each_independent_budget_without_rounding(tmp_path, kind):
    manifest, phase, value, estimates, publish = admission_fixture(tmp_path)
    prefill._validate_admission(manifest, phase, value)
    if kind == 'work':
        manifest['execution']['phases'][phase]['work_limit'] = 2
    elif kind == 'cost':
        manifest['execution']['phases'][phase]['cost_limit'] = 1
    elif kind == 'wall':
        value['children'][0]['resources']['walltime_s'] = 1
        estimates['phases'][phase]['children'][0]['resources_sha256'] = scientific.canonical_sha256(
            value['children'][0]['resources'])
    else:
        # Exactly 2 + 1e-100 must refuse, even under the default Decimal precision.
        estimates['phases'][phase]['children'][0]['tasks'][0]['estimated_seconds'] = 1e-100
        estimates['phases'][phase]['children'][0]['setup_seconds'] = 0.75
    publish()
    with pytest.raises(scientific.QualityPrefillContractError, match='limit|walltime'):
        prefill._validate_admission(manifest, phase, value)


@pytest.mark.parametrize('kind', ['phase', 'command', 'runtime', 'resources', 'payload',
    'child-order', 'task-order', 'missing-child', 'missing-task'])
def test_admission_authenticates_the_declared_order_and_computation(tmp_path, kind):
    manifest, phase, value, estimates, publish = admission_fixture(tmp_path)
    prefill._validate_admission(manifest, phase, value)
    entry = estimates['phases'][phase]
    if kind == 'phase': estimates['phases'] = {scientific.PHASE_DAG[1].phase: entry}
    elif kind in ('command', 'runtime'): entry[kind + '_sha256'] = '0' * 64
    elif kind == 'resources': entry['children'][0]['resources_sha256'] = '0' * 64
    elif kind == 'payload': entry['children'][0]['tasks'][0]['payload_sha256'] = '0' * 64
    elif kind == 'child-order': entry['children'].reverse()
    elif kind == 'task-order': entry['children'][0]['tasks'].reverse()
    elif kind == 'missing-child': entry['children'].pop()
    else: entry['children'][0]['tasks'].pop()
    publish()
    with pytest.raises(scientific.QualityPrefillContractError):
        prefill._validate_admission(manifest, phase, value)


@pytest.mark.parametrize('kind', ['missing', 'changed', 'symlink', 'size', 'digest',
                                 'estimate-digest'])
def test_admission_reads_and_authenticates_actual_evidence(tmp_path, kind):
    manifest, phase, value, estimates, publish = admission_fixture(tmp_path)
    prefill._validate_admission(manifest, phase, value)
    entry = estimates['phases'][phase]['children'][0]
    binding = entry['setup_evidence']
    filename = tmp_path / 'synthetic-timing-evidence.txt'
    original = filename.read_bytes()
    if kind == 'missing': filename.unlink()
    elif kind == 'changed': filename.write_bytes(b'x' * len(original))
    elif kind == 'symlink':
        target = tmp_path / 'target'; filename.rename(target); filename.symlink_to(target)
    elif kind == 'size': binding['bytes'] += 1
    elif kind == 'digest': binding['sha256'] = '0' * 64
    else: manifest['execution']['setup_cost_reference']['sha256'] = '0' * 64
    if kind != 'estimate-digest': publish()
    with pytest.raises((ValueError, OSError)):
        prefill._validate_admission(manifest, phase, value)


@pytest.mark.parametrize('kind', ['schema', 'unknown-phase', 'duplicate-child',
    'duplicate-task', 'empty-tasks', 'unknown-child-key', 'unknown-task-key',
    'setup-negative', 'setup-bool', 'task-zero', 'task-bool', 'bad-evidence-size',
    'bad-payload-digest'])
def test_work_estimate_structure_rejects_malformed_inputs(tmp_path, kind):
    _, phase, _, estimates, _ = admission_fixture(tmp_path)
    assert scientific.validate_work_estimates(estimates) == estimates
    child = estimates['phases'][phase]['children'][0]
    task = child['tasks'][0]
    if kind == 'schema': estimates['schema'] = 'prismabuild.logical_task_roster.v1'
    elif kind == 'unknown-phase': estimates['phases']['unknown-phase'] = copy.deepcopy(estimates['phases'][phase])
    elif kind == 'duplicate-child': estimates['phases'][phase]['children'].append(copy.deepcopy(child))
    elif kind == 'duplicate-task': child['tasks'].append(copy.deepcopy(task))
    elif kind == 'empty-tasks': child['tasks'] = []
    elif kind == 'unknown-child-key': child['ignored'] = 1
    elif kind == 'unknown-task-key': task['ignored'] = 1
    elif kind == 'setup-negative': child['setup_seconds'] = -1
    elif kind == 'setup-bool': child['setup_seconds'] = True
    elif kind == 'task-zero': task['estimated_seconds'] = 0
    elif kind == 'task-bool': task['estimated_seconds'] = True
    elif kind == 'bad-evidence-size': task['estimate_evidence']['bytes'] = (1 << 20) + 1
    else: task['payload_sha256'] = 'invalid'
    with pytest.raises(scientific.QualityPrefillContractError):
        scientific.validate_work_estimates(estimates)


@pytest.mark.parametrize('kind', ['children', 'tasks', 'child-tasks', 'document'])
def test_work_estimate_limits_apply_at_their_exact_boundary(tmp_path, monkeypatch, kind):
    _, _, _, estimates, _ = admission_fixture(tmp_path)
    limits = dict(children=('MAX_PREFILL_CHILDREN', 2), tasks=('MAX_PREFILL_TASKS', 3),
                  **{'child-tasks': ('MAX_PREFILL_CHILD_TASKS', 2),
                     'document': ('MAX_WORK_ESTIMATES_BYTES',
                                  len(scientific.canonical_json_bytes(estimates)) + 1)})
    name, boundary = limits[kind]
    monkeypatch.setattr(scientific, name, boundary)
    scientific.validate_work_estimates(estimates)
    monkeypatch.setattr(scientific, name, boundary - 1)
    with pytest.raises(scientific.QualityPrefillContractError):
        scientific.validate_work_estimates(estimates)


def test_estimate_evidence_aggregate_refuses_before_any_evidence_open(tmp_path, monkeypatch):
    manifest, phase, value, estimates, publish = admission_fixture(tmp_path, size=20, width=10)
    manifest['execution']['phases'][phase]['cost_limit'] = 11
    number = 0
    for child in estimates['phases'][phase]['children']:
        for binding in [child['setup_evidence'], *[t['estimate_evidence'] for t in child['tasks']]]:
            binding.update(path=str(tmp_path / f'absent-{number}'), bytes=1 << 20)
            number += 1
    publish()
    def forbidden(*args, **kwargs):
        raise AssertionError('aggregate bound must precede evidence opens')
    monkeypatch.setattr(prefill, '_owned', forbidden)
    with pytest.raises(scientific.QualityPrefillContractError, match='aggregate estimate evidence'):
        prefill._validate_admission(manifest, phase, value)


def roster(size=3, *, width=2):
    # Structurally complete invented evidence. It is deliberately NOT a claim
    # about an installed interpreter; that is the native boundary's obligation.
    content = {'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}
    runtime = dict(schema='prismaquant.flux_result_runtime.v5',
        python=dict(path='/runtime/python', **content, version='fixture',
                    implementation='cpython', cache_tag='fixture', flags=[1]),
        dependencies={}, stdlib={'/runtime/stdlib': ['os.py']},
        imports={'sys': dict(origin='built-in', locations=[], loader='fixture', cached=None)},
        libraries=['/runtime/lib.so'], files={name: dict(content) for name in
        ['/runtime/python', '/runtime/stdlib/os.py', '/runtime/lib.so']})
    tasks = [dict(task_id=f'task/{i}', output_id=f'out/{i}', payload=payload())
             for i in range(size)]
    return dict(schema=prefill.ROSTER_SCHEMA, phase_id='phase/A',
        plan_context_sha256='a' * 64,
        command=dict(interpreter='/runtime/python', module='experiments.producer'),
        runtime=runtime,
        inputs={'calibration': dict(path='/inputs/calibration', bytes=3,
                                   sha256=hashlib.sha256(b'abc').hexdigest())},
        tasks=tasks, children=[dict(child_id=f'child/{i}',
            task_ids=[t['task_id'] for t in tasks[i:i + width]],
            resources=dict(cores=1, memory_gib=2, gpus=0, walltime_s=60, placement='portable'),
            staged_input=None) for i in range(0, size, width)])


def measurement(task=None):
    return dict(schema=scientific.MEASUREMENT_SCHEMA, payload=task or payload(),
                units='dloss', measurement_status='measured', value=0.125,
                artifacts=[dict(path='/owned/value.json', bytes=2,
                                sha256=hashlib.sha256(b'42').hexdigest())])


@pytest.mark.parametrize('status', scientific.COVERAGE_TYPES)
def test_scalar_measurement_retains_each_declared_coverage_status(status):
    task = payload()
    task['currency'] = scientific.SCALAR_CURRENCY
    value = measurement(task)
    value['measurement_status'] = status
    assert scientific.validate_task_measurement(value, payload=task) == value


def test_measurement_is_detached_and_retains_zero_and_exact_frozen_payload():
    value = measurement()
    value['value'] = 0
    checked = scientific.validate_task_measurement(value, payload=payload())
    value['payload']['members'].clear()
    value['artifacts'][0]['bytes'] = 0
    assert checked['payload'] == payload() and checked['artifacts'][0]['bytes'] == 2
    assert checked['value'] == 0


@pytest.mark.parametrize('kind', ['schema', 'extra', 'missing', 'foreign-payload',
    'joint-predicted', 'joint-pending', 'status', 'units', 'negative', 'bool',
    'nan', 'infinite', 'overflow', 'artifacts-empty', 'artifacts-many', 'artifact-large',
    'artifact-duplicate', 'artifact-relative', 'artifact-extra'])
def test_measurement_refuses_invalid_science_and_unbounded_artifacts(kind):
    value = measurement()
    scientific.validate_task_measurement(value, payload=payload())
    if kind == 'schema': value['schema'] = 'legacy.v1'
    elif kind == 'extra': value['job_id'] = 'self-asserted'
    elif kind == 'missing': del value['units']
    elif kind == 'foreign-payload': value['payload']['context_id'] = 'other-context'
    elif kind == 'joint-predicted': value['measurement_status'] = 'predicted_screen'
    elif kind == 'joint-pending': value['measurement_status'] = 'retained_pending_measurement'
    elif kind == 'status': value['measurement_status'] = 'success'
    elif kind == 'units': value['units'] = 'bad units'
    elif kind == 'negative': value['value'] = -0.125
    elif kind == 'bool': value['value'] = True
    elif kind == 'nan': value['value'] = float('nan')
    elif kind == 'infinite': value['value'] = float('inf')
    elif kind == 'overflow': value['value'] = 1 << 4096
    elif kind == 'artifacts-empty': value['artifacts'] = []
    elif kind == 'artifacts-many': value['artifacts'] *= 33
    elif kind == 'artifact-large': value['artifacts'][0]['bytes'] = (1 << 20) + 1
    elif kind == 'artifact-duplicate': value['artifacts'] *= 2
    elif kind == 'artifact-relative': value['artifacts'][0]['path'] = 'relative'
    else: value['artifacts'][0]['claim'] = 'extra'
    with pytest.raises((scientific.QualityPrefillContractError, ValueError)):
        scientific.validate_task_measurement(value, payload=payload())


def envelope_fixture(tmp_path, *, staged=None):
    manifest, phase, value, _, _ = admission_fixture(tmp_path)
    value['children'][0]['staged_input'] = staged
    value['plan_context_sha256'] = prefill._plan_context_sha256(manifest)
    roster_binding = prefill.write_phase_roster(value, tmp_path / 'roster.json')
    manifest['execution']['phases'][phase]['task_roster'] = roster_binding
    manifest = scientific.freeze_manifest(manifest)
    manifest_binding = prefill._publish_document(manifest, tmp_path / 'manifest.json', 16 << 20)
    envelope = dict(schema=prefill.ENVELOPE_SCHEMA, phase_id=phase,
        plan_identity_sha256=manifest['identity_sha256'], manifest=manifest_binding,
        roster=roster_binding, child_id='child/0', tasks=value['tasks'][:2],
        phase_record_dir=str(tmp_path),
        result_path=str(tmp_path / 'results.json'))
    return manifest, value, envelope


@pytest.mark.parametrize('online', ['0', '1'])
def test_phase_preflight_enforces_child_host_health_before_native_validation(tmp_path, monkeypatch, online):
    manifest, phase, value, estimates, publish = admission_fixture(tmp_path)
    for child, estimate in zip(value['children'], estimates['phases'][phase]['children']):
        child['resources']['placement'] = 'host:dl380g10'
        estimate['resources_sha256'] = scientific.canonical_sha256(child['resources'])
    publish()
    value['plan_context_sha256'] = prefill._plan_context_sha256(manifest)
    roster_binding = prefill.write_phase_roster(value, tmp_path / 'roster.json')
    manifest['execution']['phases'][phase]['task_roster'] = roster_binding
    manifest = scientific.freeze_manifest(manifest)
    binding = prefill._publish_document(manifest, tmp_path / 'manifest.json', 16 << 20)
    checkout = tmp_path / 'source'
    source = dict(checkout=str(checkout), commit='a' * 40, tree='b' * 40)
    monkeypatch.setattr(prefill, '_prefill_source', lambda _: source)
    monkeypatch.setattr(prefill, '_authenticate_graph', lambda *args, **kwargs: None)
    monkeypatch.setenv('FLUX_URI', 'local:///fixture/run/local-0')
    model = dict(version=1, execution=dict(R_lite=[dict(rank='0-1')],
                                          nodelist=['dl380g10,sparky']))
    def query(args, **kwargs):
        if args[0] == 'getattr': text = '900.5\n'
        elif args[0] == 'config': text = '{}\n'
        elif args[0] == 'kvs': text = json.dumps(model)
        elif args[0] == 'hostlist': text = 'dl380g10 sparky\n'
        elif args[0] == 'resource': text = online + '\n'
        elif args[0] == 'overlay': text = '0 coordinator: full\n1 worker: offline\n'
        else: raise AssertionError('unexpected native query')
        return prefill.flux_jobs.subprocess.CompletedProcess([], 0, text, '')
    monkeypatch.setattr(prefill.flux_jobs, '_flux_run', query)
    def native_validation(**kwargs):
        raise RuntimeError('native validation reached after health admission')
    monkeypatch.setattr(prefill.flux_jobs, 'validate_job', native_validation)
    root = prefill.Path('/mnt/shared') / tmp_path.name / 'health-phase'
    error, message = ((RuntimeError, 'native validation reached') if online == '0'
                      else (ValueError, 'every eligible rank.*online'))
    with pytest.raises(error, match=message):
        prefill.submit_phase(binding, phase=phase, checkout=checkout, record_dir=root)


def test_task_envelope_binds_complete_child_and_sealed_plan(tmp_path):
    manifest, value, envelope = envelope_fixture(tmp_path)
    binding = prefill._publish_document(envelope, tmp_path / 'envelope.json', 1 << 20)
    checked = prefill.read_task_envelope(binding)
    assert checked == envelope
    assert checked['tasks'] == value['tasks'][:2]
    assert checked['plan_identity_sha256'] == manifest['identity_sha256']
    checked['tasks'][0]['payload']['members'].clear()
    assert prefill.read_task_envelope(binding)['tasks'][0]['payload'] == payload()


@pytest.mark.parametrize('kind', ['identity', 'schema', 'phase', 'child', 'extra',
    'task-missing', 'task-extra', 'task-order', 'payload', 'output', 'roster-path',
    'roster-digest', 'manifest-digest', 'result-relative', 'result-roster',
    'result-manifest', 'result-envelope', 'result-input', 'result-submission',
    'root-relative', 'root-foreign', 'root-symlink'])
def test_resealed_envelope_cannot_change_plan_or_shrink_population(tmp_path, kind):
    _, value, envelope = envelope_fixture(tmp_path)
    positive = prefill._publish_document(envelope, tmp_path / 'positive.json', 1 << 20)
    assert prefill.read_task_envelope(positive)['tasks'] == value['tasks'][:2]
    filename = tmp_path / 'mutated.json'
    if kind == 'identity': envelope['plan_identity_sha256'] = '0' * 64
    elif kind == 'schema': envelope['schema'] = 'legacy.v1'
    elif kind == 'phase': envelope['phase_id'] = scientific.PHASE_DAG[1].phase
    elif kind == 'child': envelope['child_id'] = 'undeclared'
    elif kind == 'extra': envelope['claim'] = 'trusted'
    elif kind == 'task-missing': envelope['tasks'].pop()
    elif kind == 'task-extra': envelope['tasks'].append(value['tasks'][2])
    elif kind == 'task-order': envelope['tasks'].reverse()
    elif kind == 'payload': envelope['tasks'][0]['payload']['rate'] += 1
    elif kind == 'output': envelope['tasks'][0]['output_id'] = 'foreign-output'
    elif kind == 'roster-path': envelope['roster']['path'] = str(tmp_path / 'other-roster.json')
    elif kind == 'roster-digest': envelope['roster']['sha256'] = '0' * 64
    elif kind == 'manifest-digest': envelope['manifest']['sha256'] = '0' * 64
    elif kind == 'result-relative': envelope['result_path'] = 'relative'
    elif kind == 'result-roster': envelope['result_path'] = envelope['roster']['path']
    elif kind == 'result-manifest': envelope['result_path'] = envelope['manifest']['path']
    elif kind == 'result-envelope': envelope['result_path'] = str(filename)
    elif kind == 'result-input': envelope['result_path'] = value['inputs']['calibration']['path']
    elif kind == 'result-submission': envelope['result_path'] = str(tmp_path / 'submission.json')
    elif kind == 'root-relative': envelope['phase_record_dir'] = 'relative'
    elif kind == 'root-foreign': envelope['phase_record_dir'] = str(tmp_path.parent)
    else:
        link = tmp_path / 'root-link'
        link.symlink_to(tmp_path, target_is_directory=True)
        envelope['phase_record_dir'] = str(link)
    binding = prefill._publish_document(envelope, filename, 1 << 20)
    with pytest.raises((ValueError, OSError)):
        prefill.read_task_envelope(binding)


def test_context_omits_only_cycle_fields_and_refuses_changed_science(tmp_path):
    manifest, value, envelope = envelope_fixture(tmp_path)
    expected = value['plan_context_sha256']
    changed = copy.deepcopy(manifest)
    changed['identity_sha256'] = '0' * 64
    for entry in changed['execution']['phases'].values():
        entry['task_roster'] = dict(path='/future/roster.json', sha256='f' * 64, bytes=1)
    assert prefill._plan_context_sha256(changed) == expected
    changed = copy.deepcopy(manifest)
    del changed['identity_sha256']
    changed['experiment_id'] = 'other-experiment'
    assert prefill._plan_context_sha256(changed) != expected
    envelope['plan_identity_sha256'] = scientific.freeze_manifest(changed)['identity_sha256']
    envelope['manifest'] = prefill._publish_document(scientific.freeze_manifest(changed),
        tmp_path / 'other-manifest.json', 16 << 20)
    binding = prefill._publish_document(envelope, tmp_path / 'changed.json', 1 << 20)
    with pytest.raises(scientific.QualityPrefillContractError, match='plan context differs'):
        prefill.read_task_envelope(binding)


@pytest.mark.parametrize('kind', ['content', 'symlink', 'missing', 'size'])
def test_envelope_binding_itself_requires_original_bytes(tmp_path, kind):
    _, _, envelope = envelope_fixture(tmp_path)
    filename = tmp_path / 'envelope.json'
    binding = prefill._publish_document(envelope, filename, 1 << 20)
    assert prefill.read_task_envelope(binding) == envelope
    if kind == 'content': filename.write_bytes(b'{}')
    elif kind == 'symlink':
        held = tmp_path / 'held.json'
        filename.rename(held)
        filename.symlink_to(held)
    elif kind == 'missing': filename.unlink()
    else: binding['bytes'] += 1
    if kind == 'symlink':
        with pytest.raises(OSError) as refused:
            prefill.read_task_envelope(binding)
        assert refused.value.errno == errno.ELOOP
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            prefill.read_task_envelope(binding)


def task_results(envelope):
    rows = []
    for task in envelope['tasks']:
        value = measurement(copy.deepcopy(task['payload']))
        rows.append(dict(task_id=task['task_id'], output_id=task['output_id'], value=value,
                         value_sha256=scientific.canonical_sha256(value)))
    return rows


def test_result_document_preserves_exact_population_and_independent_value_hashes(tmp_path):
    _, _, envelope = envelope_fixture(tmp_path)
    rows = task_results(envelope)
    document = prefill._result_document(envelope, rows)
    assert document['schema'] == prefill.RESULTS_SCHEMA
    assert document['phase_id'] == envelope['phase_id']
    assert document['plan_identity_sha256'] == envelope['plan_identity_sha256']
    assert document['roster_sha256'] == envelope['roster']['sha256']
    assert [(row['task_id'], row['output_id']) for row in document['results']] == [
        ('task/0', 'out/0'), ('task/1', 'out/1')]
    for row in document['results']:
        raw = json.dumps(row['value'], sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
        assert row['value_sha256'] == hashlib.sha256(raw).hexdigest()
    rows[0]['value']['value'] = 7
    assert document['results'][0]['value']['value'] == 0.125


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'order', 'task', 'output',
    'value-hash', 'extra', 'payload', 'aggregate-artifacts', 'result-bytes'])
def test_results_refuse_wrong_cover_and_finite_limits_before_artifact_reads(tmp_path, monkeypatch, kind):
    _, _, envelope = envelope_fixture(tmp_path)
    rows = task_results(envelope)
    prefill._result_document(envelope, rows)
    if kind == 'missing': rows.pop()
    elif kind == 'duplicate': rows[1] = copy.deepcopy(rows[0])
    elif kind == 'order': rows.reverse()
    elif kind == 'task': rows[0]['task_id'] = 'foreign'
    elif kind == 'output': rows[0]['output_id'] = 'foreign'
    elif kind == 'value-hash': rows[0]['value_sha256'] = '0' * 64
    elif kind == 'extra': rows[0]['claim'] = 'extra'
    elif kind == 'payload':
        rows[0]['value']['payload']['rate'] += 1
        rows[0]['value_sha256'] = scientific.canonical_sha256(rows[0]['value'])
    elif kind == 'aggregate-artifacts':
        for number, row in enumerate(rows):
            row['value']['artifacts'] = [dict(path=f'/absent/{number}/{i}',bytes=1 << 20,
                                             sha256='a'*64) for i in range(17)]
            row['value_sha256'] = scientific.canonical_sha256(row['value'])
    else:
        for number, row in enumerate(rows):
            row['value']['artifacts'] = [dict(path='/'+'a'*(1 << 19)+str(number),bytes=1,
                                             sha256='a'*64)]
            row['value_sha256'] = scientific.canonical_sha256(row['value'])
    def forbidden(*args, **kwargs):
        raise AssertionError('result bounds and cover must precede artifact opens')
    monkeypatch.setattr(prefill, '_owned', forbidden)
    with pytest.raises(ValueError):
        prefill._result_document(envelope, rows)


@pytest.mark.parametrize('kind', ['changed', 'missing', 'symlink', 'conflict'])
def test_result_measurement_artifacts_use_actual_known_bytes(tmp_path, kind):
    _, _, envelope = envelope_fixture(tmp_path)
    rows = task_results(envelope)
    artifact = tmp_path / 'actual-measurement.bin'
    raw = b'\x00known measurement\xff\n'
    artifact.write_bytes(raw)
    for row in rows:
        row['value']['artifacts'] = [dict(path=str(artifact), bytes=len(raw),
                                         sha256=hashlib.sha256(raw).hexdigest())]
        row['value_sha256'] = scientific.canonical_sha256(row['value'])
    document = prefill._result_document(envelope, rows)
    prefill._authenticate_measurements(document)
    if kind == 'changed': artifact.write_bytes(b'\x00wrong measurement\xff\n')
    elif kind == 'missing': artifact.unlink()
    elif kind == 'symlink':
        held = tmp_path / 'held.bin'
        artifact.rename(held)
        artifact.symlink_to(held)
    else:
        document['results'][1]['value']['artifacts'][0]['sha256'] = '0' * 64
    with pytest.raises((ValueError, OSError)):
        prefill._authenticate_measurements(document)


def phase_records_fixture(tmp_path, *, staged=None):
    manifest, roster_value, original = envelope_fixture(tmp_path, staged=staged)
    root = tmp_path / 'phase'
    root.mkdir()
    children = []
    for index, (child, tasks) in enumerate(zip(roster_value['children'],
                                             [roster_value['tasks'][:2], roster_value['tasks'][2:]])):
        envelope = dict(original, phase_record_dir=str(root), child_id=child['child_id'],
                        tasks=tasks, result_path=str(root / f'results-{index:04d}.json'))
        binding = prefill._publish_document(envelope, root / f'envelope-{index:04d}.json', 1 << 20)
        children.append(dict(child_id=child['child_id'], task_ids=child['task_ids'],
                             envelope=binding, record_dir=str(root / f'native-{index:04d}')))
    submission = dict(schema=prefill.SUBMISSION_SCHEMA, phase_id=original['phase_id'],
        plan_identity_sha256=manifest['identity_sha256'], roster_sha256=original['roster']['sha256'],
        afterok=[], children=children)
    filename = root / 'submission.json'
    filename.write_bytes(scientific.canonical_json_bytes(submission) + b'\n')
    return original['manifest'], original['phase_id'], root, submission


def test_phase_records_match_independently_written_finite_layout(tmp_path):
    binding, phase, root, submission = phase_records_fixture(tmp_path)
    records = prefill._phase_records(binding, phase, root)
    assert records['submission'] == submission
    assert [row['child_id'] for row in records['submission']['children']] == ['child/0', 'child/2']
    assert [row['record_dir'] for row in records['submission']['children']] == [
        str(root / 'native-0000'), str(root / 'native-0001')]
    assert [task['task_id'] for envelope in records['envelopes'] for task in envelope['tasks']] == [
        'task/0', 'task/1', 'task/2']
    authenticated = prefill._authenticate_graph(binding, phase, root)
    assert authenticated['parents'] == []
    assert not list(root.glob('native-*'))


@pytest.mark.parametrize('kind', ['schema', 'phase', 'identity', 'roster', 'missing-child',
    'duplicate-child', 'order', 'task', 'record-dir', 'envelope-path', 'envelope-digest',
    'extra', 'afterok-type', 'afterok-large', 'parent-extra', 'parent-duplicate',
    'parent-duplicate-id', 'parent-digest', 'parent-id', 'parent-relative'])
def test_phase_records_refuse_resealed_foreign_or_partial_metadata(tmp_path, kind):
    binding, phase, root, submission = phase_records_fixture(tmp_path)
    prefill._phase_records(binding, phase, root)
    parent = dict(phase_id=phase, child_id='child/0', job_id='f2',
                  record_dir=str(root / 'parent-native'), launch_sha256='a' * 64)
    if kind == 'schema': submission['schema'] = 'legacy.v1'
    elif kind == 'phase': submission['phase_id'] = 'foreign'
    elif kind == 'identity': submission['plan_identity_sha256'] = '0' * 64
    elif kind == 'roster': submission['roster_sha256'] = '0' * 64
    elif kind == 'missing-child': submission['children'].pop()
    elif kind == 'duplicate-child': submission['children'].append(copy.deepcopy(submission['children'][0]))
    elif kind == 'order': submission['children'].reverse()
    elif kind == 'task': submission['children'][0]['task_ids'].pop()
    elif kind == 'record-dir': submission['children'][0]['record_dir'] = str(tmp_path / 'foreign')
    elif kind == 'envelope-path': submission['children'][0]['envelope']['path'] = str(tmp_path / 'foreign')
    elif kind == 'envelope-digest': submission['children'][0]['envelope']['sha256'] = '0' * 64
    elif kind == 'extra': submission['claim'] = 'extra'
    elif kind == 'afterok-type': submission['afterok'] = {}
    elif kind == 'afterok-large': submission['afterok'] = [parent] * 4097
    else:
        submission['afterok'] = [parent]
        if kind == 'parent-extra': parent['claim'] = 'extra'
        elif kind == 'parent-duplicate': submission['afterok'].append(copy.deepcopy(parent))
        elif kind == 'parent-duplicate-id':
            submission['afterok'].append(dict(parent, child_id='child/2', job_id='1'))
        elif kind == 'parent-digest': parent['launch_sha256'] = 'not-a-digest'
        elif kind == 'parent-id': parent['job_id'] = 'not-an-id'
        else: parent['record_dir'] = 'relative'
    (root / 'submission.json').write_bytes(scientific.canonical_json_bytes(submission) + b'\n')
    with pytest.raises((ValueError, OSError)):
        prefill._phase_records(binding, phase, root)


def test_dependency_phase_must_belong_to_declared_dag_before_native_lookup(tmp_path, monkeypatch):
    binding, phase, root, submission = phase_records_fixture(tmp_path)
    submission['afterok'] = [dict(phase_id=scientific.PHASE_DAG[1].phase, child_id='child/0',
        job_id='f2', record_dir=str(root / 'foreign'), launch_sha256='a' * 64)]
    (root / 'submission.json').write_bytes(scientific.canonical_json_bytes(submission) + b'\n')
    def forbidden(*args, **kwargs):
        raise AssertionError('foreign dependency must refuse before native lookup')
    monkeypatch.setattr(prefill.flux_jobs, 'load_submission', forbidden)
    with pytest.raises(scientific.QualityPrefillContractError, match='foreign parent phase'):
        prefill._authenticate_graph(binding, phase, root)


@pytest.mark.parametrize('kind', ['changed-envelope', 'missing-envelope', 'duplicate-json'])
def test_child_authority_requires_original_envelope_and_common_strict_document(tmp_path, kind):
    binding, phase, root, _ = phase_records_fixture(tmp_path)
    prefill._phase_records(binding, phase, root)
    if kind == 'changed-envelope': (root / 'envelope-0000.json').write_bytes(b'{}')
    elif kind == 'missing-envelope': (root / 'envelope-0001.json').unlink()
    else: (root / 'submission.json').write_bytes(b'{"schema":"x","schema":"y"}')
    with pytest.raises((ValueError, OSError)):
        records = prefill._phase_records(binding, phase, root)
        prefill._authenticate_child(records, 1 if kind == 'missing-envelope' else 0)


@pytest.mark.parametrize('kind', ['missing', 'changed'])
def test_common_population_survives_child_local_envelope_refusal(tmp_path, kind):
    binding, phase, root, submission = phase_records_fixture(tmp_path)
    target = root / 'envelope-0000.json'
    if kind == 'missing':
        target.unlink()
    else:
        target.write_bytes(b'{}')
    records = prefill._phase_records(binding, phase, root)
    assert records['submission'] == submission
    assert [child['child_id'] for child in records['submission']['children']] == ['child/0', 'child/2']
    prefill._authenticate_child(records, 1)
    with pytest.raises((OSError, ValueError)):
        prefill._authenticate_child(records, 0)


def test_child_local_staged_refusal_keeps_healthy_envelope_and_common_population(tmp_path):
    from staged_native_fixtures import publish_inputs
    stage = tmp_path / 'stage'
    stage.mkdir()
    source = stage / 'known.bin'
    source.write_bytes(b'independent staged bytes')
    staged = publish_inputs(stage, [source])
    binding, phase, root, submission = phase_records_fixture(tmp_path, staged=staged)
    prefill.read_task_envelope(submission['children'][0]['envelope'])
    prefill.read_task_envelope(submission['children'][1]['envelope'])
    (stage / 'native-read-contract.json').write_bytes(b'changed owned read contract')
    records = prefill._phase_records(binding, phase, root)
    assert records['submission'] == submission
    prefill._authenticate_child(records, 1)
    with pytest.raises(ValueError):
        prefill._authenticate_child(records, 0)
    with pytest.raises(ValueError):
        prefill.validate_phase_roster(records['roster'])


def test_phase_writer_refuses_replacement_directory_before_writing(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    witness = root.stat()
    identity = (witness.st_dev, witness.st_ino)
    held = tmp_path / 'held-root'
    root.rename(held)
    root.mkdir()
    control = root / 'foreign-control'
    control.write_bytes(b'leave unchanged')
    with pytest.raises(ValueError, match='directory identity changed'):
        prefill._publish_document({'value': 42}, root / 'new.json', 1 << 20,
                                   directory_identity=identity)
    assert list(held.iterdir()) == []
    assert sorted(path.name for path in root.iterdir()) == ['foreign-control']
    assert control.read_bytes() == b'leave unchanged'


def test_ordered_cover_is_detached_and_preserves_scientific_meaning():
    value = roster()
    checked = prefill.validate_phase_roster(value)
    assert [c['task_ids'] for c in checked['children']] == [['task/0', 'task/1'], ['task/2']]
    assert checked['tasks'][0]['payload'] == payload()
    value['tasks'][0]['payload']['members'].clear()
    assert checked['tasks'][0]['payload']['members'] == ['layer.0/up', 'layer.0/gate']


@pytest.mark.parametrize('currency', sorted(scientific.CURRENCIES))
def test_scientific_task_retains_each_independent_currency(currency):
    task = payload()
    task['currency'] = currency
    assert scientific.validate_task_payload(task, where='task') == task


@pytest.mark.parametrize('field,bad', [
    ('schema', 'prismabuild.logical_task.v1'), ('members', []),
    ('members', ['one', 'one']), ('members', ['bad space']),
    ('rate', True), ('rate', 1.0), ('rate', 0), ('currency', 'mixed'),
    ('serving_unit', ''), ('family', 'a' * 257), ('context_id', 'non-ascii-\N{SNOWMAN}'),
])
def test_invalid_scientific_task_is_refused(field, bad):
    task = payload()
    scientific.validate_task_payload(task, where='positive')
    task[field] = bad
    with pytest.raises(scientific.QualityPrefillContractError):
        scientific.validate_task_payload(task, where='negative')


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unknown', 'reordered',
                                   'output', 'child', 'task', 'empty-child'])
def test_cover_refuses_without_silently_repairing_it(change):
    value = roster()
    prefill.validate_phase_roster(value)
    if change == 'missing': value['children'][-1]['task_ids'] = ['task/1']
    elif change == 'duplicate': value['children'][0]['task_ids'].append('task/1')
    elif change == 'unknown': value['children'][-1]['task_ids'] = ['unknown']
    elif change == 'reordered': value['children'].reverse()
    elif change == 'output': value['tasks'][1]['output_id'] = 'out/0'
    elif change == 'child': value['children'][1]['child_id'] = 'child/0'
    elif change == 'task': value['tasks'][1]['task_id'] = 'task/0'
    else: value['children'][0]['task_ids'].clear()
    with pytest.raises(ValueError): prefill.validate_phase_roster(value)


@pytest.mark.parametrize('field,bad', [('cores', 0), ('cores', True),
    ('memory_gib', 0), ('gpus', -1), ('walltime_s', 0), ('placement', 'auto-discover')])
def test_resource_refusal_precedes_staged_file_access(monkeypatch, field, bad):
    value = roster()
    value['children'][0]['staged_input'] = {'unread': 'marker'}
    value['children'][-1]['resources'][field] = bad
    reads = []
    def forbidden(*args, **kwargs):
        reads.append(args)
        raise AssertionError('staged file access before complete admission')
    monkeypatch.setattr(prefill, 'validate_staged_binding', forbidden)
    with pytest.raises(ValueError): prefill.validate_phase_roster(value)
    assert reads == []


@pytest.mark.parametrize('kind', ['tasks', 'children', 'child-tasks', 'inputs', 'bytes'])
def test_finite_limits_and_their_positive_boundaries(monkeypatch, kind):
    if kind == 'tasks':
        value = roster(1024, width=64)
        prefill.validate_phase_roster(value)
        value['tasks'].append(dict(task_id='task/1024', output_id='out/1024', payload=payload()))
    elif kind == 'children':
        value = roster(256, width=1)
        prefill.validate_phase_roster(value)
        value['children'].append(copy.deepcopy(value['children'][-1]))
    elif kind == 'child-tasks':
        value = roster(64, width=64)
        prefill.validate_phase_roster(value)
        value['children'][0]['task_ids'].append('extra')
    elif kind == 'inputs':
        value = roster()
        value['inputs'] = {f'role/{i}': dict(path=f'/input/{i}', bytes=1, sha256='a' * 64)
                           for i in range(32)}
        prefill.validate_phase_roster(value)
        value['inputs']['extra'] = dict(path='/extra', bytes=1, sha256='a' * 64)
    else:
        value = roster()
        size = len(scientific.canonical_json_bytes(value)) + 1
        monkeypatch.setattr(prefill, 'MAX_ROSTER_BYTES', size)
        prefill.validate_phase_roster(value)
        monkeypatch.setattr(prefill, 'MAX_ROSTER_BYTES', size - 1)
    with pytest.raises(ValueError): prefill.validate_phase_roster(value)


@pytest.mark.parametrize('key', ['root', 'command', 'input', 'task', 'payload', 'child', 'resource', 'runtime'])
def test_every_document_boundary_refuses_unknown_fields(key):
    value = roster()
    bodies = dict(root=value, command=value['command'], input=value['inputs']['calibration'],
        task=value['tasks'][0], payload=value['tasks'][0]['payload'], child=value['children'][0],
        resource=value['children'][0]['resources'], runtime=value['runtime'])
    bodies[key]['undeclared'] = 1
    with pytest.raises(ValueError): prefill.validate_phase_roster(value)


@pytest.mark.parametrize('field,bad', [('path', 'relative'), ('path', '/input/../escape'),
    ('sha256', 'A' * 64), ('bytes', True), ('bytes', -1), ('bytes', 1 << 63)])
def test_input_bindings_are_exact_normalized_and_bounded(field, bad):
    value = roster()
    value['inputs']['calibration'][field] = bad
    with pytest.raises(ValueError): prefill.validate_phase_roster(value)


def test_complete_admission_delegates_staged_authentication_with_placement(tmp_path):
    from staged_native_fixtures import publish_inputs
    source = tmp_path / 'known'
    source.write_bytes(b'known staged input')
    value = roster()
    value['children'][0]['staged_input'] = publish_inputs(tmp_path, [source])
    assert prefill.validate_phase_roster(value)['children'][0]['staged_input']['schema'] == 'prismaquant.staged_launch.v1'
    value['children'][0]['staged_input']['sha256'] = '0' * 64
    with pytest.raises(ValueError): prefill.validate_phase_roster(value)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_values_never_reach_publication(tmp_path, bad):
    value = roster()
    value['tasks'][0]['payload']['rate'] = bad
    with pytest.raises(ValueError): prefill.write_phase_roster(value, tmp_path / 'never')
    assert list(tmp_path.iterdir()) == []


def test_publication_is_canonical_known_bytes_and_never_overwrites(tmp_path):
    value = roster()
    expected = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    target = tmp_path / 'roster.json'
    binding = prefill.write_phase_roster(value, path=target)
    assert target.read_bytes() == expected
    assert binding == dict(path=str(target), sha256=hashlib.sha256(expected).hexdigest(), bytes=len(expected))
    with pytest.raises(FileExistsError): prefill.write_phase_roster(value, path=target)
    assert target.read_bytes() == expected
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize('kind', ['parent-symlink', 'target-symlink', 'directory', 'missing-parent'])
def test_publication_refuses_foreign_targets_without_modifying_them(tmp_path, kind):
    owned = tmp_path / 'owned'; owned.mkdir()
    foreign = tmp_path / 'foreign'; foreign.mkdir()
    sentinel = foreign / 'keep'; sentinel.write_bytes(b'unchanged')
    target = owned / 'roster.json'
    if kind == 'parent-symlink':
        owned.rmdir(); owned.symlink_to(foreign, target_is_directory=True)
    elif kind == 'target-symlink': target.symlink_to(sentinel)
    elif kind == 'directory': target.mkdir()
    else: owned.rmdir()
    with pytest.raises((ValueError, OSError)): prefill.write_phase_roster(roster(), target)
    assert sentinel.read_bytes() == b'unchanged'
    assert list(foreign.iterdir()) == [sentinel]


@pytest.mark.parametrize('when', ['before-write', 'before-read', 'after-read'])
def test_replaced_parent_never_grants_write_or_return_authority(tmp_path, monkeypatch, when):
    parent = tmp_path / 'parent'; parent.mkdir()
    moved = tmp_path / 'original'
    target = parent / 'roster.json'
    def replace():
        parent.rename(moved)
        parent.mkdir()
        (parent / 'foreign').write_bytes(b'leave')
    original_write, original_read = prefill.publish_new_bytes, prefill.read_document
    def write(*args, **kwargs):
        if when == 'before-write': replace()
        result = original_write(*args, **kwargs)
        if when == 'before-read': replace()
        return result
    def read(*args, **kwargs):
        result = original_read(*args, **kwargs)
        if when == 'after-read': replace()
        return result
    monkeypatch.setattr(prefill, 'publish_new_bytes', write)
    monkeypatch.setattr(prefill, 'read_document', read)
    with pytest.raises((ValueError, OSError)): prefill.write_phase_roster(roster(), target)
    assert sorted(p.name for p in parent.iterdir()) == ['foreign']
    assert (parent / 'foreign').read_bytes() == b'leave'


def test_same_byte_replacement_cannot_be_adopted_as_the_writer(tmp_path, monkeypatch):
    original = prefill.publish_new_bytes
    def substitute(target, raw, **kwargs):
        result = original(target, raw, **kwargs)
        replacement = tmp_path / 'replacement'
        replacement.write_bytes(raw)
        assert replacement.stat().st_ino != target.stat().st_ino
        os.replace(replacement, target)
        return result
    monkeypatch.setattr(prefill, 'publish_new_bytes', substitute)
    with pytest.raises(ValueError, match='owned writer'):
        prefill.write_phase_roster(roster(), tmp_path / 'roster.json')


@pytest.mark.parametrize('mutation', ['missing', 'changed', 'length', 'digest', 'oversize', 'symlink'])
def test_bound_document_authenticates_known_bytes_before_json(tmp_path, mutation):
    filename = tmp_path / 'document.json'
    raw = b'{"known":42}\n'
    filename.write_bytes(raw)
    binding = dict(path=str(filename), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    assert prefill._read_bound_document(binding, len(raw)) == {'known': 42}
    if mutation == 'missing': filename.unlink()
    elif mutation == 'changed': filename.write_bytes(b'{"known":43}\n')
    elif mutation == 'length': binding['bytes'] -= 1
    elif mutation == 'digest': binding['sha256'] = '0' * 64
    elif mutation == 'oversize': binding['bytes'] += 1
    else:
        target = tmp_path / 'target'; filename.rename(target); filename.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        prefill._read_bound_document(binding, len(raw))


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}'])
def test_authenticated_document_still_refuses_invalid_json(tmp_path, raw):
    filename = tmp_path / 'invalid.json'
    filename.write_bytes(raw)
    binding = dict(path=str(filename), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    with pytest.raises(ValueError): prefill._read_bound_document(binding, len(raw))


def test_private_publisher_enforces_document_bound_before_writes(tmp_path):
    filename = tmp_path / 'never.json'
    with pytest.raises(ValueError): prefill._publish_document({'known': 42}, filename, 1)
    assert list(tmp_path.iterdir()) == []
