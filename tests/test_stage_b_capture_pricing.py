"""The retained budget plans the Stage B capture pass the guard charges (PQ #1151).

The R13 layer-044 v6 gate (PB ``d39b475a…``) planned a 93.1 GB peak, and the
guard then refused its first capture pass: 4 stored batches times the declared
16 GiB workspace, a charge no plan had priced. These cases carry the R13
policy's declared owners, so a derivation that admits that capture fails here
in milliseconds instead of after a checkpoint load.
"""
import pytest

from prismaquant.joint_retained_window_plan import (
    DECLARED_BUDGET_FIELDS, MEASURED_BUDGET_FIELDS, capture_workspace_bytes,
    derive_retained_window_budget, plan_retained_targets,
)
from test_retained_window_budget_derivation import PREFETCH_WORKERS, SCOPE, _roster

GIB = 1 << 30
#: ``derivation.declared`` of stageb-resource-policy.r13.json (sha 004c8cfa…).
R13_DECLARED = {
    'auxiliary_reserve_bytes': 2147483648, 'boundary_reserve_bytes': 2281701376,
    'metadata_reserve_bytes': 21474836480, 'physical_limit_bytes': 100931731456,
    'read_page_reserve_bytes': 4096, 'runtime_reserve_bytes': 4294967296,
    'safety_margin_bytes': 2147483648, 'workspace_reserve_bytes': 17179869184,
}
R13_SOURCE_BYTES = 33285996544
R13_HOST_CAP_BYTES = 30064771072
RECEIPT = {'action_key': 'a' * 64, 'path': '/receipts/profile.json', 'sha256': 'b' * 64}


def _derive(**kwargs):
    declared = dict(R13_DECLARED)
    if kwargs.get('measured') is not None:
        for name in MEASURED_BUDGET_FIELDS:
            declared.pop(name)
    return derive_retained_window_budget(
        _roster(), declared=declared, source_bytes=R13_SOURCE_BYTES,
        prefetch_workers=PREFETCH_WORKERS, host_cap_bytes=R13_HOST_CAP_BYTES,
        footprint_scope=SCOPE, **kwargs)


def _measured(nbytes):
    return {'workspace_reserve_bytes': {'bytes': nbytes, 'receipt': dict(RECEIPT),
                                        'basis': 'test'}}


def test_the_v6_capture_refuses_at_derivation():
    """16 GiB declared, 4 stored batches per capture: the v6 gate's charge."""
    assert set(R13_DECLARED) == set(DECLARED_BUDGET_FIELDS)
    with pytest.raises(RuntimeError, match='capture pass at capture_batch 4 plans'):
        _derive(capture_batch=4)
    # One stored batch per pass is what the fixed owners already hold.
    budget, record = _derive(capture_batch=1)
    assert record['capture']['workspace_bytes'] == budget.workspace_reserve_bytes


def test_a_measured_workspace_plans_the_capture_and_names_its_receipt():
    budget, record = _derive(measured=_measured(4 * GIB), capture_batch=4)
    assert budget.workspace_reserve_bytes == 4 * GIB
    assert 'workspace_reserve_bytes' not in record['declared']
    assert record['measured'] == {'workspace_reserve_bytes': {
        'bytes': 4 * GIB, 'receipt': RECEIPT, 'basis': 'test'}}
    capture = record['capture']
    assert capture['capture_batch'] == 4
    assert capture['workspace_bytes'] == capture_workspace_bytes(4 * GIB, 4) == 16 * GIB
    assert capture['peak_planned_bytes'] == budget.capture_peak_bytes(
        R13_SOURCE_BYTES, capture_batch=4, render_bytes=budget.retained_render_cap_bytes)
    assert capture['peak_planned_bytes'] == (
        budget.fixed_bytes(R13_SOURCE_BYTES) + 3 * 4 * GIB + budget.retained_render_cap_bytes)
    assert capture['peak_planned_bytes'] <= (budget.physical_limit_bytes
                                             - budget.safety_margin_bytes)
    windows = [window.peak_planned_bytes for targets in _roster().values()
               for window in plan_retained_targets(
                   targets, budget=budget, source_bytes=R13_SOURCE_BYTES,
                   footprint_scope=SCOPE).windows]
    assert record['peak_planned_bytes'] == max(max(windows), capture['peak_planned_bytes'])


def test_without_a_capture_the_record_is_the_one_before():
    budget, record = _derive()
    assert 'capture' not in record and 'measured' not in record
    assert record['declared'] == R13_DECLARED
    assert budget.workspace_reserve_bytes == R13_DECLARED['workspace_reserve_bytes']


@pytest.mark.parametrize('owner, message', [
    ({'bytes': 4 * GIB, 'basis': 'test'}, 'bytes, receipt and basis'),
    ({'bytes': 4 * GIB, 'receipt': {**RECEIPT, 'sha256': 'B' * 64}, 'basis': 'test'},
     'sha256 must be 64 lowercase hex'),
    ({'bytes': 4 * GIB, 'receipt': {**RECEIPT, 'path': 'relative.json'}, 'basis': 'test'},
     'path must be absolute'),
    ({'bytes': 0, 'receipt': dict(RECEIPT), 'basis': 'test'}, 'positive'),
])
def test_a_measured_owner_carries_its_receipt(owner, message):
    with pytest.raises(ValueError, match=message):
        _derive(measured={'workspace_reserve_bytes': owner}, capture_batch=4)


def test_an_owner_is_declared_or_measured_never_both():
    with pytest.raises(ValueError, match='exactly the declared owners'):
        derive_retained_window_budget(
            _roster(), declared=dict(R13_DECLARED), source_bytes=R13_SOURCE_BYTES,
            prefetch_workers=PREFETCH_WORKERS, host_cap_bytes=R13_HOST_CAP_BYTES,
            footprint_scope=SCOPE, measured=_measured(4 * GIB))
