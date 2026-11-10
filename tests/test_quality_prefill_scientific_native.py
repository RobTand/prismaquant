"""Scientific joins at the collector seam; real native proof is recorded separately."""
import copy
import hashlib

import pytest

from prismaquant import quality_prefill_contract as sc, quality_prefill_flux as pf
from test_quality_prefill_flux import phase_records_fixture, measurement


@pytest.fixture
def population(tmp_path, monkeypatch):
    binding, phase, root, submission = phase_records_fixture(tmp_path)
    _, _, roster = pf._read_phase(binding, phase)
    artifact = tmp_path / 'known.bin'
    artifact.write_bytes(b'42')
    results = []
    for child in roster['children']:
        for task_id in child['task_ids']:
            task = next(task for task in roster['tasks'] if task['task_id'] == task_id)
            value = measurement(task['payload'])
            value['artifacts'] = [dict(path=str(artifact), bytes=2,
                                      sha256=hashlib.sha256(b'42').hexdigest())]
            results.append(dict(task_id=task_id, output_id=task['output_id'],
                child_id=child['child_id'], value=value, value_sha256=sc.canonical_sha256(value),
                selection=dict(schema='prismaquant.flux_result_selection.v1', producer_id=child['child_id'])))
    collection = dict(state='complete', errors=[], task_results=results, children=[])
    # An explicit offline seam, not synthetic evidence presented to a native
    # reader. No native execution or selected publication is claimed here.
    monkeypatch.setattr(pf, 'collect_phase', lambda *args, **kwargs: copy.deepcopy(collection))
    return binding, phase, root, collection


def metadata():
    return dict(observation_id='obs-1', candidate_id='cand-1', context_sha256='a' * 64,
                inner_schema='unit_kl_v1', attempt=1, validity='valid', unknown_reason='')


def test_independent_frozen_population_and_known_scientific_values(population):
    binding, phase, root, collection = population
    native, result = pf.collect_phase_result(binding, phase=phase, record_dir=root)
    assert native == collection
    assert result['expected_task_ids'] == ['task/0', 'task/1', 'task/2']
    assert [row['task_id'] for row in result['child_receipts']] == ['task/0', 'task/1', 'task/2']
    assert result['coverage'] == dict.fromkeys(sc.COVERAGE_TYPES, 0) | {'measured': 3}
    assert result['terminal_state'] == 'complete'
    assert sc.validate_phase_result(result) == result


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'foreign', 'digest', 'payload', 'errors'])
def test_complete_refuses_changed_or_shrunken_population(population, kind):
    binding, phase, root, collection = population
    if kind == 'missing': collection['task_results'].pop()
    elif kind == 'duplicate': collection['task_results'].append(copy.deepcopy(collection['task_results'][0]))
    elif kind == 'foreign': collection['task_results'][0]['task_id'] = 'foreign'
    elif kind == 'digest': collection['task_results'][0]['value_sha256'] = '0' * 64
    elif kind == 'payload': collection['task_results'][0]['value']['payload']['rate'] += 1
    else: collection['errors'].append('publication unavailable')
    with pytest.raises(sc.QualityPrefillContractError):
        pf.collect_phase_result(binding, phase=phase, record_dir=root)


@pytest.mark.parametrize('state', ['running', 'unknown', 'incomplete'])
def test_nonterminal_or_unavailable_population_never_gets_a_terminal_result(population, state):
    binding, phase, root, collection = population
    collection['state'] = state
    _, result = pf.collect_phase_result(binding, phase=phase, record_dir=root, wait=False)
    assert result is None


def test_native_failure_retains_partial_selected_coverage_without_completing(population):
    binding, phase, root, collection = population
    collection.update(state='failed', task_results=collection['task_results'][:1],
                      children=[dict(child_id='child/2', execution=dict(status='failed'))])
    _, result = pf.collect_phase_result(binding, phase=phase, record_dir=root)
    assert result['terminal_state'] == 'failed'
    assert result['expected_task_ids'] == ['task/0', 'task/1', 'task/2']
    assert result['coverage']['measured'] == 1
    assert result['refusal_reasons'] == ['child child/2: failed']


def test_observation_preserves_labels_and_joins_only_selected_value(population):
    binding, phase, root, _ = population
    observation = pf.observe_task(binding, phase=phase, record_dir=root,
                                  task_id='task/0', metadata=metadata())
    assert observation['schema'] == sc.OBSERVATION_SCHEMA
    assert {key: observation[key] for key in metadata()} == metadata()
    assert observation['value'] == 0.125
    assert observation['currency'] == sc.JOINT_CURRENCY
    assert observation['task_id'] == 'task/0' and observation['child_id'] == 'child/0'
    assert pf.read_observation(observation, manifest_binding=binding, record_dir=root) == observation


@pytest.mark.parametrize('kind', ['value', 'selection', 'digest', 'currency', 'artifacts'])
def test_resealed_observation_cannot_replace_selected_science(population, kind):
    binding, phase, root, _ = population
    observation = pf.observe_task(binding, phase=phase, record_dir=root,
                                  task_id='task/0', metadata=metadata())
    observation.pop('identity_sha256')
    if kind == 'value': observation['value'] = 100
    elif kind == 'selection': observation['selection']['job_id'] = 'foreign'
    elif kind == 'digest': observation['value_sha256'] = 'b' * 64
    elif kind == 'currency': observation['currency'] = sc.SCALAR_CURRENCY
    else: observation['artifacts'][0]['sha256'] = '0' * 64
    observation = sc.seal_envelope('observation', observation)
    with pytest.raises(sc.QualityPrefillContractError, match='differs from authenticated'):
        pf.read_observation(observation, manifest_binding=binding, record_dir=root)


@pytest.mark.parametrize('kind', ['foreign', 'failed', 'unknown', 'retry', 'incomplete', 'extra-label'])
def test_observation_cannot_bypass_native_cover_or_attempt_policy(population, kind):
    binding, phase, root, collection = population
    labels, task_id = metadata(), 'task/0'
    if kind == 'foreign': task_id = 'foreign'
    elif kind == 'retry': labels['attempt'] = 2
    elif kind == 'extra-label': labels['value'] = 100
    else: collection['state'] = kind
    with pytest.raises(sc.QualityPrefillContractError):
        pf.observe_task(binding, phase=phase, record_dir=root, task_id=task_id, metadata=labels)
