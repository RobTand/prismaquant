"""The native reader smoke remains a control join, never CUDA admission."""
import pytest

from prismaquant import source_generation as sg, tessera_calibration_cache as cc
from test_held_producer_sdk5_smoke_2152 import reader_domain_case
from test_original_source_admission import (  # noqa: F401
    authority_case, original_runner, original_model, material, _forget_state,
    runtime_layout, nonqualifying_runtime,
)

pytestmark = pytest.mark.own_process


def test_native_reader_smoke_cannot_issue_original_source_authority(authority_case):
    receipt, authority, selected = reader_domain_case(authority_case)
    assert sg._require_original_reader_producer(receipt, authority, selected, authority['runtime']) is None
    assert selected.execution.status == 'running'
    assert authority['qualification'] is None and authority['root_admission'] is None
    for owner in (None, authority_case['owner']):
        with pytest.raises(RuntimeError, match='full64'):
            cc.require_original_source_authority(owner, authority_case['authority_input'],
                authority_case['plan_input'], authority_case['packet'])
