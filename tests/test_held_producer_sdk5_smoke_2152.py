"""Native reader-domain joins over actual CPU material observations.

The historical filename remains selectable. These tests grant no CUDA/source
qualification; the full native boundary has its separate producer/consumer proof.
"""
import copy
from dataclasses import replace
import gc
import hashlib
from pathlib import Path

import pytest
from safetensors import safe_open

from prismaquant import source_generation as sg, native_results as nr
from prismaquant.flux_jobs import ExecutionResult
from test_capture_original_material import _owner
from test_original_source_admission import (  # noqa: F401
    authority_case, original_runner, original_model, material, _forget_state, _bound,
    runtime_layout, nonqualifying_runtime,
)

pytestmark = pytest.mark.own_process


def _running_control(directory):
    """Nonterminal join input, with no native execution or source authority."""
    return ExecutionResult('running', '1', Path(directory))


def reader_domain_case(case):
    authority = case['authority']
    runtime = case['packet']['runtime']
    expected = _bound(case['tmp'] / 'independent-reader-runtime.json', runtime)
    inputs = {role: {**binding, 'bytes': Path(binding['path']).stat().st_size}
              for role, binding in (('observed-runtime', authority['runtime']),
                                    ('selected-runtime', expected))}
    # A join-only input cannot satisfy read_selected_result or issue source authority.
    execution = _running_control(case['tmp'])
    witness = case['owner'].receipt()['deliveries'][0]['native_delivery']
    staged = {'sha256': witness['staged_manifest_sha256'],
              'read_contract_sha256': witness['read_contract_sha256']}
    result = nr.SelectedResult({'runtime': runtime['native_runtime'], 'inputs': inputs}, {}, {},
        {'resources': case['packet']['resources']['claim_demand'], 'staged_input': staged}, execution)
    return case['owner'].receipt(), authority, result


@pytest.mark.parametrize('mode', ['0', '1'])
@pytest.mark.parametrize('damage', ['runtime', 'resources', 'manifest', 'readset', 'missing-input'])
def test_held_producer_context_authenticates_and_joins(authority_case, monkeypatch, mode, damage):
    receipt, authority, selected = reader_domain_case(authority_case)
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    assert sg._require_original_reader_producer(receipt, authority, selected, authority['runtime']) is None
    publication = copy.deepcopy(selected.publication)
    launch = copy.deepcopy(selected.launch)
    if damage == 'runtime':
        publication['runtime']['python']['sha256'] = '0' * 64
    elif damage == 'resources':
        launch['resources']['memory_gib'] += 1
    elif damage == 'missing-input':
        publication['inputs'] = {}
    elif damage == 'manifest':
        launch['staged_input']['sha256'] = '0' * 64
    else:
        launch['staged_input']['read_contract_sha256'] = '0' * 64
    changed = nr.SelectedResult(publication, {}, {}, launch, selected.execution)
    refusal = {'runtime': 'selected reader measured native runtime',
               'resources': 'selected reader actual producer reservation',
               'manifest': 'actual reader delivery selected staged contract',
               'readset': 'actual reader delivery selected staged contract',
               'missing-input': 'reader runtime must be an independently declared native input'}[damage]
    with pytest.raises(RuntimeError, match=refusal):
        sg._require_original_reader_producer(receipt, authority, changed, authority['runtime'])


def _selected_runtime_input(selected):
    return {key: selected.publication['inputs']['selected-runtime'][key] for key in ('path', 'sha256')}


@pytest.mark.parametrize('mode', ['0', '1'])
@pytest.mark.parametrize('axis,value', [('cores', 2), ('memory_gib', 2), ('gpus', 1),
                                      ('walltime_s', 61), ('placement', 'aarch64')])
def test_reader_reservation_checks_each_native_resource_axis(authority_case, monkeypatch, mode, axis, value):
    receipt, authority, selected = reader_domain_case(authority_case)
    expected = _selected_runtime_input(selected)
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    assert sg._require_original_reader_producer(receipt, authority, selected, expected) is None
    launch = copy.deepcopy(selected.launch)
    assert launch['resources'][axis] != value
    launch['resources'][axis] = value
    with pytest.raises(RuntimeError, match='selected reader actual producer reservation'):
        sg._require_original_reader_producer(receipt, authority, replace(selected, launch=launch), expected)


@pytest.mark.parametrize('mode', ['0', '1'])
@pytest.mark.parametrize('role', ['observed-runtime', 'selected-runtime'])
@pytest.mark.parametrize('damage', ['missing', 'path', 'sha256', 'changed-bytes'])
def test_each_reader_runtime_input_is_independently_bound(authority_case, monkeypatch, mode, role, damage):
    receipt, authority, selected = reader_domain_case(authority_case)
    expected = _selected_runtime_input(selected)
    assert authority['runtime']['path'] != expected['path']
    assert Path(authority['runtime']['path']).read_bytes() == Path(expected['path']).read_bytes()
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    assert sg._require_original_reader_producer(receipt, authority, selected, expected) is None
    publication = copy.deepcopy(selected.publication)
    if damage == 'missing':
        del publication['inputs'][role]
    elif damage in ('path', 'sha256'):
        publication['inputs'][role][damage] = '/independent/foreign-runtime' if damage == 'path' else '0' * 64
    else:
        Path(publication['inputs'][role]['path']).write_bytes(b'{"changed":true}')
    changed = replace(selected, publication=publication)
    refusal = 'owned bytes' if damage == 'changed-bytes' else 'independently declared native input'
    with pytest.raises(ValueError if damage == 'changed-bytes' else RuntimeError, match=refusal):
        sg._require_original_reader_producer(receipt, authority, changed, expected)


@pytest.mark.parametrize('mode', ['0', '1'])
@pytest.mark.parametrize('damage', ['observed-runtime', 'selected-runtime', 'both-foreign-config'])
def test_reader_runtime_selection_and_source_config_are_separate_joins(authority_case, monkeypatch, mode, damage):
    receipt, authority, selected = reader_domain_case(authority_case)
    authority = copy.deepcopy(authority)
    expected = _selected_runtime_input(selected)
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    assert sg._require_original_reader_producer(receipt, authority, selected, expected) is None
    runtime = copy.deepcopy(authority_case['packet']['runtime'])
    if damage == 'both-foreign-config':
        runtime['config']['hidden_size'] += 1
        authority['runtime'] = _bound(authority_case['tmp'] / 'foreign-observed.json', runtime)
        expected = _bound(authority_case['tmp'] / 'foreign-selected.json', runtime)
        publication = copy.deepcopy(selected.publication)
        publication['inputs'] = {role: {**binding, 'bytes': Path(binding['path']).stat().st_size}
            for role, binding in (('observed-runtime', authority['runtime']), ('selected-runtime', expected))}
        selected = replace(selected, publication=publication)
        refusal = 'selected reader runtime source config'
    else:
        runtime['versions']['python'] += '.independent-change'
        binding = _bound(authority_case['tmp'] / 'changed-runtime.json', runtime)
        if damage == 'observed-runtime':
            authority['runtime'] = binding
        else:
            expected = binding
        refusal = 'observed original source runtime'
    with pytest.raises(RuntimeError, match=refusal):
        sg._require_original_reader_producer(receipt, authority, selected, expected)


@pytest.mark.parametrize('mode', ['0', '1'])
@pytest.mark.parametrize('damage', [None, []])
def test_reader_requires_explicit_staged_launch(authority_case, monkeypatch, mode, damage):
    receipt, authority, selected = reader_domain_case(authority_case)
    expected = _selected_runtime_input(selected)
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    assert sg._require_original_reader_producer(receipt, authority, selected, expected) is None
    changed = replace(selected, launch={**selected.launch, 'staged_input': damage})
    with pytest.raises(RuntimeError, match='requires explicit immutable staged launch'):
        sg._require_original_reader_producer(receipt, authority, changed, expected)


@pytest.mark.parametrize('mode', ['0', '1'])
@pytest.mark.parametrize('delivery', ['nonfirst', 'reacquired'])
@pytest.mark.parametrize('axis', ['read_contract_sha256', 'staged_manifest_sha256'])
def test_every_actual_reader_delivery_joins_selected_staging(authority_case, monkeypatch, mode, delivery, axis):
    receipt, authority, selected = reader_domain_case(authority_case)
    expected = _selected_runtime_input(selected)
    if delivery == 'reacquired':
        # authority_case advances staging while preparing its controls. A new
        # real owner isolates these two deliveries to one actual manifest.
        with _owner(authority_case['material']) as owner:
            path = owner.root / 'two.safetensors'
            for _ in range(2):
                with owner.material_window([path]):
                    with owner.safe_open(safe_open, path, framework='pt') as reader:
                        tensor = reader.get_tensor(reader.keys()[0])
                        assert tensor.device.type == 'cpu'
                    del tensor
                gc.collect()
            receipt = owner.receipt()
        targets = [i for i, row in enumerate(receipt['deliveries']) if row['name'] == path.name]
        assert len(targets) == 2
        target = targets[-1]
        assert receipt['deliveries'][target]['delivery_index'] == 2
        witness = receipt['deliveries'][0]['native_delivery']
        staged = {'sha256': witness['staged_manifest_sha256'],
                  'read_contract_sha256': witness['read_contract_sha256']}
        selected = replace(selected, launch={**selected.launch, 'staged_input': staged})
    else:
        assert len(receipt['deliveries']) > 1
        target = 1
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    assert sg._require_original_reader_producer(receipt, authority, selected, expected) is None
    changed = copy.deepcopy(receipt)
    changed['deliveries'][target]['native_delivery'][axis] = '0' * 64
    with pytest.raises(RuntimeError, match='actual reader delivery selected staged contract'):
        sg._require_original_reader_producer(changed, authority, selected, expected)


@pytest.mark.parametrize('mode', ['0', '1'])
@pytest.mark.parametrize('damage', ['node', 'missing-role', 'extra-role', 'rebound'])
def test_scientific_artifact_consumer_checks_selected_node_roles_and_binding(tmp_path, monkeypatch, mode, damage):
    path = tmp_path / 'control.json'
    raw = b'{"answer":42}\n'
    path.write_bytes(raw)
    binding = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    measured = nr.bind_result_artifact(path, max_bytes=128)
    execution = _running_control(tmp_path)
    selected = nr.SelectedResult({'producer_id': 'nonqualifying-artifact-control',
        'artifacts': {'control': measured}}, {'control': raw}, {}, {}, execution)
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    assert sg._original_artifact_publication(selected, node_id='nonqualifying-artifact-control',
                                              roles={'control'}) is selected
    assert sg._published_original_artifact(selected, 'control', binding,
        resource_check=None, max_bytes=128) == {'answer': 42}
    publication = copy.deepcopy(selected.publication)
    if damage == 'node':
        publication['producer_id'] = 'other-node'
        refusal = 'actual artifact publication selected node'
    elif damage in ('missing-role', 'extra-role'):
        if damage == 'missing-role':
            publication['artifacts'] = {}
        else:
            publication['artifacts']['extra'] = measured
        refusal = 'actual published artifact roles'
    else:
        replacement = b'{"answer":43}\n'
        path.write_bytes(replacement)
        publication['artifacts']['control'] = nr.bind_result_artifact(path, max_bytes=128)
        rebound = replace(selected, publication=publication, artifact_payloads={'control': replacement})
        new_binding = {'path': str(path), 'sha256': hashlib.sha256(replacement).hexdigest()}
        assert sg._published_original_artifact(rebound, 'control', new_binding,
            resource_check=None, max_bytes=128) == {'answer': 43}
        with pytest.raises(RuntimeError, match='selected published control binding'):
            sg._published_original_artifact(rebound, 'control', binding, resource_check=None, max_bytes=128)
        return
    with pytest.raises(RuntimeError, match=refusal):
        sg._original_artifact_publication(replace(selected, publication=publication),
            node_id='nonqualifying-artifact-control', roles={'control'})
