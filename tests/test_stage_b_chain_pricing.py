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


def test_a_chain_owner_is_measured_with_its_receipt_or_declared_as_such():
    measured_without_receipt = _kda()
    measured_without_receipt.pop('receipt')
    declared_with_receipt = dict(_dsa(source='declared'), receipt=dict(RECEIPT))
    no_resident = _kda()
    no_resident.pop('device_resident_bytes')
    other_regime = _kda(regime={'batch_size': 4, 'probe_fusion': False})
    for owners, match in (
            ({'kda': measured_without_receipt}, 'measured chain owner'),
            ({'kda': _kda(), 'dsa': declared_with_receipt}, 'declared chain owner'),
            ({'kda': no_resident}, 'measured chain owner'),
            ({'kda': other_regime}, 'another chain regime'),
            ({'kda': _kda(), 'dsa': dict(_dsa(), layers=[43, 44])},
             'more than one chain layer shape'),
            ({}, 'at least one chain layer shape')):
        with pytest.raises(ValueError, match=match):
            _derive(**_chain(owners))
    # A declared owner states itself and carries no receipt.
    _budget, record = _derive(**_chain({'kda': _kda(), 'dsa': _dsa(source='declared')}))
    assert record['chain']['shapes']['dsa']['source'] == 'declared'
    assert 'receipt' not in record['chain']['shapes']['dsa']
    # The regime, the owners and the device envelope go together.
    with pytest.raises(ValueError, match='together'):
        _derive(chain_regime=dict(REGIME), chain_workspace={'kda': _kda()})


def test_without_a_chain_the_budget_and_record_are_the_ones_before():
    budget, record = _derive()
    assert 'chain' not in record
    assert not any(key.startswith('chain_') for key in budget.as_dict())
    with pytest.raises(RuntimeError, match='prices no chain phase'):
        budget.chain_workspace_bytes(4, fused=True)


def test_the_budget_carries_the_priced_chain_layers_and_refuses_any_other():
    from prismaquant.joint_retained_window_plan import ChainRetainedWindowBudget

    budget, record = _derive(**_chain({'kda': _kda(), 'dsa': _dsa()}))
    assert budget.chain_layers == (3, 4, 5, 6, 7, 43, 44)
    assert record['chain']['shapes']['kda']['layers'] == [4, 5, 6, 44]
    # The largest shape's workspace is what every priced roll is charged.
    for layer in budget.chain_layers:
        assert budget.chain_workspace_bytes(4, fused=True, layer=layer) == KDA_44[0]
    with pytest.raises(RuntimeError, match='chain layer 2 has no priced chain layer shape'):
        budget.chain_workspace_bytes(4, fused=True, layer=2)
    # A JSON round trip loads the same budget.
    loaded = RetainedWindowBudget.from_dict(json.loads(json.dumps(budget.as_dict())))
    assert loaded == budget and isinstance(loaded, ChainRetainedWindowBudget)
    # All seven chain fields or none.
    partial = {k: v for k, v in budget.as_dict().items() if k != 'chain_device_limit_bytes'}
    with pytest.raises(ValueError, match='complete versioned'):
        RetainedWindowBudget.from_dict(partial)


def test_the_retained_execution_refuses_a_chain_that_does_not_fit():
    budget, _record = _derive(**_chain({'kda': _kda(), 'dsa': _dsa()}))
    execution = {'schema': EXECUTION_SCHEMA, 'budget': budget.as_dict(),
                 'source_reserve_bytes': R13_SOURCE_BYTES,
                 'source_loading_reserve_bytes': LOADING}
    operator = {'max_statistics_bytes': budget.statistics_cap_bytes,
                'max_candidate_bytes': budget.candidate_delta_bytes,
                'max_load_buffer_bytes': budget.load_buffer_bytes}
    boundary = {'capture_order': 'layer_major',
                'max_resident_bytes': budget.boundary_reserve_bytes,
                'max_auxiliary_bytes': budget.auxiliary_reserve_bytes}
    normalized = normalize_retained_execution(
        execution, operator_windows=operator, boundary_storage=boundary)
    assert normalized['budget'] == budget.as_dict()
    margin = DEVICE - budget.chain_device_peak_bytes()
    wide = dict(execution, budget=dict(
        budget.as_dict(), chain_workspace_reserve_bytes=budget.chain_workspace_reserve_bytes
        + margin + 1))
    with pytest.raises(RuntimeError, match='chain phase cannot fit the device envelope'):
        normalize_retained_execution(wide, operator_windows=operator, boundary_storage=boundary)
    heavy = dict(execution, budget=dict(
        budget.as_dict(), chain_host_committed_bytes=BOUND - budget.chain_device_peak_bytes()
        + 1))
    with pytest.raises(RuntimeError, match='chain phase cannot fit the retained COST physical'):
        normalize_retained_execution(heavy, operator_windows=operator, boundary_storage=boundary)


def test_the_measurement_tool_declares_the_headroom_owner_when_the_plan_prices_none(
        monkeypatch):
    """``--stop-after-chain`` on an unpriced plan admits under a declared owner.

    The owner is the largest workspace the chain phase's resident owners, at
    their declared caps, leave under the device envelope and the physical
    bound, recorded as declared; a plan that already prices the chain keeps
    its own owner.
    """
    import os

    import prismaquant.joint_adjoint_slices as slices
    from experiments.stage_b_capture_workspace_profile import declare_chain_owner
    from prismaquant.joint_retained_window_plan import ChainRetainedWindowBudget
    from prismaquant.stage_b_workspace_profile import CHAIN_OWNER_ENV

    monkeypatch.setattr(slices, 'chain_regime_of', lambda _identity: dict(REGIME))
    monkeypatch.setenv(CHAIN_OWNER_ENV, '')
    budget, _record = _derive()
    retained = {'budget': budget.as_dict(), 'source_reserve_bytes': R13_SOURCE_BYTES,
                'source_loading_reserve_bytes': LOADING}
    config = {'execution': {'retained_operator_windows': retained}, 'max_gpu_bytes': DEVICE}
    owner = declare_chain_owner(config, {'run_identity': {}}, [44, 43])
    residents = budget.declared_chain_residents(R13_SOURCE_BYTES, loading_bytes=LOADING)
    assert residents == {
        'device_resident_bytes': (budget.runtime_reserve_bytes + R13_SOURCE_BYTES + LOADING
                                  + budget.auxiliary_reserve_bytes
                                  + budget.boundary_reserve_bytes),
        'host_committed_bytes': budget.metadata_reserve_bytes}
    headroom = DEVICE - residents['device_resident_bytes']
    assert headroom < BOUND - sum(residents.values())
    assert owner['source'] == 'declared'
    assert owner['chain_workspace_reserve_bytes'] == headroom
    assert owner['chain_layers'] == [43, 44] and owner['regime'] == REGIME
    assert json.loads(os.environ[CHAIN_OWNER_ENV]) == owner
    planned = RetainedWindowBudget.from_dict(retained['budget'])
    assert isinstance(planned, ChainRetainedWindowBudget)
    assert planned.chain_layers == (43, 44)
    assert planned.chain_workspace_bytes(4, fused=True, layer=43) == headroom
    assert planned.chain_device_peak_bytes() == DEVICE
    assert declare_chain_owner(config, {'run_identity': {}}, [44, 43]) == {
        'source': 'plan', 'chain_workspace_reserve_bytes': headroom, 'chain_layers': [43, 44]}
