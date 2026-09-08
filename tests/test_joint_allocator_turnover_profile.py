"""Admission and sampling checks; actual planned-size turnover is native only."""
import json

import pytest
import torch

from experiments.joint_allocator_turnover_profile import GIB, live_identity, main, plan_geometry


def ledger():
    return dict(schema='prismaquant.glm_joint_resource_ledger.v2',
        status='BLOCKED_NOT_RUNNABLE', physical_admission_bytes=104 * GIB,
        gpu_subset_admission_bytes=92 * GIB,
        terms_bytes=dict(source_settled=33315763888, statistics_cap=32 * GIB,
                         workspace_allowance=16 * GIB))


def test_proxy_uses_exact_ledger_bytes_without_claiming_the_ledger_is_runnable():
    original = ledger()
    plan = plan_geometry(original)
    assert original == ledger()
    assert plan['source_bytes'] == 33315763888
    assert plan['future_reserve_bytes'] == 48 * GIB
    assert plan['source_bytes'] + plan['future_reserve_bytes'] < plan['gpu_subset_admission_bytes']


@pytest.mark.parametrize('bad', [True, 1.5, 0, -1])
def test_proxy_refuses_noninteger_or_nonpositive_source_counts(bad):
    item = ledger()
    item['terms_bytes']['source_settled'] = bad
    with pytest.raises(ValueError, match='positive integers'):
        plan_geometry(item)


@pytest.mark.parametrize('source', [GIB, 80 * GIB])
def test_proxy_refuses_geometry_that_cannot_test_turnover_within_its_caps(source):
    item = ledger()
    item['terms_bytes']['source_settled'] = source
    with pytest.raises(ValueError, match='does not distinguish'):
        plan_geometry(item)


@pytest.mark.parametrize('field', ['physical_admission_bytes', 'gpu_subset_admission_bytes'])
def test_proxy_does_not_expand_declared_admission(field):
    item = ledger()
    item[field] += GIB
    with pytest.raises(ValueError, match='declared'):
        plan_geometry(item)


def test_live_identity_samples_endpoints_and_preserves_storage_identity():
    live = torch.arange(200, dtype=torch.uint8)
    result = live_identity(live)
    assert result['data_ptr'] == live.data_ptr()
    assert result['storage_bytes'] == live.untyped_storage().nbytes()
    assert len(result['samples']) == 64
    assert result['positions'][0] == 0 and result['positions'][-1] == 199
    assert result['samples'] == result['positions']


def test_frozen_ledger_refusal_precedes_any_native_allocation(tmp_path):
    path = tmp_path / 'ledger.json'
    path.write_text(json.dumps(ledger()))
    with pytest.raises(SystemExit) as caught:
        main(['--ledger', str(path), '--ledger-sha256', '0' * 64,
              '--out', str(tmp_path / 'must-not-exist')])
    assert caught.value.code == 2
    assert not (tmp_path / 'must-not-exist').exists()
