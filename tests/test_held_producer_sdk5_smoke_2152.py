"""Native reader-domain joins over actual CPU material observations.

The historical filename remains selectable. These tests grant no CUDA/source
qualification; the full native boundary has its separate producer/consumer proof.
"""
import copy
import os
from pathlib import Path

import pytest

from prismaquant import source_generation as sg, native_results as nr
from prismaquant.flux_jobs import ExecutionResult
from test_original_source_admission import (  # noqa: F401
    authority_case, original_runner, original_model, material, _forget_state, _bound,
)

pytestmark = pytest.mark.own_process


def reader_domain_case(case):
    authority = case['authority']
    runtime = case['packet']['runtime']
    inputs = {'runtime': {**authority['runtime'], 'bytes': Path(authority['runtime']['path']).stat().st_size}}
    # Join-only fixture. Execution is truthfully still running in this admitted
    # test job, so this object cannot satisfy read_selected_result.
    execution = ExecutionResult('running', os.environ['FLUX_JOB_ID'], Path(os.environ['OUT']))
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
    with pytest.raises(RuntimeError):
        sg._require_original_reader_producer(receipt, authority, changed, authority['runtime'])
