"""The retained budget plans the Stage B chain phase the guard charges (PQ #1163).

A Stage B quantum in chain mode rolls every chain layer from its checkpoint
down to ``layer + 1`` before its retained reverse step. Each roll is a
batched backward at Stage A's chain regime, and under probe fusion it keeps
the forward graph across the probes' backwards. Before #1163 the derivation
had no chain term, so ``peak_planned_bytes`` and the row's ``mem_gb`` left it
out.

A roll is planned by what is resident while it rolls, split the way the guard
reads it: the CUDA reservation at the roll's admission plus the roll's
workspace against the device envelope, and that plus the cgroup's committed
bytes against the physical budget less its margin. The owners below are the
R13 rolls PB ``1b6bc9e5d2be`` measured in-process (``--stop-after-chain`` on
layer 42, 2026-09-24): layer 44 (KDA attention, routed MoE) and layer 43
(sparse MLA attention, routed MoE), at batch 4 with probe fusion, under the
28/96/68 GiB limits of policy ``c03fc00e``.
"""
import inspect
import json

import pytest

from prismaquant.joint_retained_window_plan import (
    EXECUTION_SCHEMA, RetainedWindowBudget, derive_retained_window_budget,
    normalize_retained_execution, plan_retained_targets,
)
from test_retained_window_budget_derivation import PREFETCH_WORKERS, SCOPE, _roster
from test_stage_b_capture_pricing import (
    R13_DECLARED, R13_HOST_CAP_BYTES, R13_SOURCE_BYTES, RECEIPT, _measured,
)

GIB = 1 << 30
#: R13's chain regime (``run/layer-quanta/adjoint/chain-state.json``).
REGIME = {'batch_size': 4, 'probe_fusion': True}
#: Policy c03fc00e's physical bound and device ceiling (28/96/68 GiB).
PHYSICAL = 96 * GIB
DEVICE = 68 * GIB
BOUND = PHYSICAL - R13_DECLARED['safety_margin_bytes']
#: ``execution.retained_operator_windows.source_loading_reserve_bytes`` of R13.
LOADING = 536870912
#: The measured rolls: (workspace, CUDA reservation at admission, host committed).
KDA_44 = (54425288704, 17913872384, 9892261888)
DSA_43 = (31092375552, 17985175552, 14533726208)


def _owner(layers, measured, *, source='measured', regime=REGIME):
    workspace, device, host = measured
    owner = {'layers': list(layers), 'bytes': workspace, 'device_resident_bytes': device,
             'host_committed_bytes': host, 'regime': dict(regime), 'source': source,
             'basis': 'test'}
    if source == 'measured':
        owner['receipt'] = dict(RECEIPT)
    return owner


def _kda(measured=KDA_44, **kwargs):
    return _owner([4, 5, 6, 44], measured, **kwargs)


def _dsa(measured=DSA_43, **kwargs):
    return _owner([3, 7, 43], measured, **kwargs)


def _derive(**kwargs):
    """The R13 declared owners at policy c03fc00e's 96 GiB physical bound."""
    declared = dict(R13_DECLARED, physical_limit_bytes=PHYSICAL)
    declared.pop('workspace_reserve_bytes')
    return derive_retained_window_budget(
        _roster(), declared=declared, source_bytes=R13_SOURCE_BYTES,
        prefetch_workers=PREFETCH_WORKERS, host_cap_bytes=R13_HOST_CAP_BYTES,
        footprint_scope=SCOPE, measured=_measured(4 * GIB), capture_batch=4, **kwargs)


def _chain(owners, device=DEVICE):
    return dict(chain_regime=dict(REGIME), chain_workspace=owners,
                chain_device_limit_bytes=device)


def _require_chain_term():
    params = inspect.signature(derive_retained_window_budget).parameters
    assert {'chain_regime', 'chain_workspace', 'chain_device_limit_bytes'} <= set(params), (
        'the retained derivation has no chain term (PQ #1163)')


def test_the_derivation_plans_the_chain_phase_per_layer_shape():
    _require_chain_term()
    _unpriced, before = _derive()
    budget, record = _derive(**_chain({'kda': _kda(), 'dsa': _dsa()}))
    chain = record['chain']
    assert chain['regime'] == REGIME and chain['device_limit_bytes'] == DEVICE
    kda, dsa = chain['shapes']['kda'], chain['shapes']['dsa']
    assert kda['source'] == 'measured' and kda['receipt'] == RECEIPT
    # Each shape as the guard checks it: device, then aggregate.
    assert kda['device_peak_bytes'] == 72339161088
    assert kda['device_margin_bytes'] == 675282944
    assert kda['peak_planned_bytes'] == 82231422976
    assert dsa['device_peak_bytes'] == 49077551104
    assert dsa['peak_planned_bytes'] == 63611277312
    for shape in (kda, dsa):
        assert shape['slack_bytes'] == BOUND - shape['peak_planned_bytes']
    # The budget charges the largest of each owner over the shapes.
    assert (budget.chain_workspace_reserve_bytes, budget.chain_device_resident_bytes,
            budget.chain_host_committed_bytes) == (KDA_44[0], DSA_43[1], DSA_43[2])
    assert chain['device_peak_bytes'] == budget.chain_device_peak_bytes() == 72410464256
    assert chain['device_margin_bytes'] == DEVICE - 72410464256 == 603979776
    assert chain['peak_planned_bytes'] == budget.chain_peak_bytes() == 86944190464
    windows = [window.peak_planned_bytes for targets in _roster().values()
               for window in plan_retained_targets(
                   targets, budget=budget, source_bytes=R13_SOURCE_BYTES,
                   footprint_scope=SCOPE).windows]
    assert record['peak_planned_bytes'] == max(
        max(windows), record['capture']['peak_planned_bytes'], chain['peak_planned_bytes'])
    assert record['peak_planned_bytes'] >= before['peak_planned_bytes']
    # The guard charges the largest workspace before every priced roll.
    assert budget.chain_workspace_bytes(4, fused=True) == KDA_44[0]
    for batch_size, fused in ((4, False), (2, True), (8, True)):
        with pytest.raises(RuntimeError, match='chain regime'):
            budget.chain_workspace_bytes(batch_size, fused=fused)
    assert RetainedWindowBudget.from_dict(budget.as_dict()) == budget


def test_a_roll_past_the_device_margin_or_the_physical_bound_refuses():
    """Both directions at the measured R13 rolls (PQ #1163).

    The measured rolls admit, with a 0.60 GB device margin. The device
    envelope is the tight side: a workspace one byte past it refuses, even
    though the aggregate would still fit, and a host that pushes the
    aggregate past the physical bound refuses while the device fits.
    """
    _require_chain_term()
    _derive(**_chain({'kda': _kda(), 'dsa': _dsa()}))
    workspace, device, host = KDA_44
    exact = (DEVICE - device, device, host)
    _budget, record = _derive(**_chain({'kda': _kda(exact)}))
    assert record['chain']['device_margin_bytes'] == 0
    assert record['chain']['peak_planned_bytes'] < BOUND
    with pytest.raises(RuntimeError, match=r"'kda' at batch 4 \(probe_fusion True\) plans "
                                           rf"{DEVICE + 1} device bytes"):
        _derive(**_chain({'kda': _kda((DEVICE - device + 1, device, host))}))
    # Past the physical bound on the host side, with the device inside its envelope.
    with pytest.raises(RuntimeError, match=rf"plans {BOUND + 1} bytes .* physical budget"):
        _derive(**_chain({'kda': _kda((workspace, device,
                                       BOUND + 1 - device - workspace))}))
    # Each shape fits alone, but the budget charges the largest resident and
    # the largest workspace together: DSA's larger reservation beside KDA's
    # exact-fit workspace refuses.
    with pytest.raises(RuntimeError, match='chain phase cannot fit the device envelope'):
        _derive(**_chain({'kda': _kda(exact), 'dsa': _dsa()}))
