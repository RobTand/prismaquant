"""Preflight prices exact geometry from a launch-bound native declaration."""
import pytest
N_PROBES, STRIDE = 2, 8


def _preflight(tmp_path, budget, *, probe_microbatch=0):
    from test_layer_major_boundary_capture import draw, fixture
    from prismaquant.joint_adjoint_checkpoints import adjoint_space
    from prismaquant.joint_cost_stage_a import _run_artifact_preflight
    _, _, runner, _ = fixture()
    return _run_artifact_preflight(runner, draw(), {'n_probes': N_PROBES,
        'probe_microbatch': probe_microbatch,
        'boundary_storage': {'max_auxiliary_bytes': 1 << 20}},
        STRIDE, adjoint_space(tmp_path / 'out'), {'run_used': int(budget), 'override': None})


def _numbers(tmp_path, monkeypatch, *, probe_microbatch=0):
    monkeypatch.delenv('FLUX_JOB_ID', raising=False)
    result = _preflight(tmp_path, 1 << 40, probe_microbatch=probe_microbatch)
    floor, allowance = result['required_floor_bytes'], result['planning_estimate_bytes']
    assert floor < allowance
    return floor, allowance, result


def _bind_spool(tmp_path, monkeypatch, *, spool=True, write_only=False):
    from produced_native_fixtures import publication
    from prismaquant import native_results
    pub = publication(tmp_path / 'launch', spool_bytes=(1 << 24) if spool else None,
        write_only=write_only)
    monkeypatch.setenv('FLUX_JOB_ID', pub.producer['job_id'])
    monkeypatch.setattr(native_results, 'execution_context', lambda: pub.producer)
    return pub
def test_a_spool_bound_owner_below_its_allowance_refuses_at_the_preflight(
        tmp_path, monkeypatch):
    from prismaquant.joint_cost_stage_a import AdjointIdentityRefused

    floor, allowance, _ = _numbers(tmp_path, monkeypatch)
    between = floor + (allowance - floor) // 2
    _bind_spool(tmp_path, monkeypatch)
    with pytest.raises(AdjointIdentityRefused) as refused:
        _preflight(tmp_path, between)
    message = str(refused.value)
    assert str(floor) in message and str(allowance) in message, message
    assert str(between) in message, message


def test_a_spool_bound_owner_at_its_allowance_passes_as_before(
        tmp_path, monkeypatch):
    floor, allowance, _ = _numbers(tmp_path, monkeypatch)
    _bind_spool(tmp_path, monkeypatch)
    for budget in (allowance, allowance + 1):
        result = _preflight(tmp_path, budget)
        assert result["declared_bytes"] == budget
        assert result["required_floor_bytes"] == floor
        assert result["planning_estimate_bytes"] == allowance



@pytest.mark.parametrize('mode', ['unadmitted', 'no-spool', 'write-only'])
def test_without_a_spool_bound_owner_the_floor_still_gates(tmp_path, monkeypatch, mode):
    from prismaquant.joint_cost_stage_a import AdjointIdentityRefused
    floor, allowance, _ = _numbers(tmp_path, monkeypatch)
    _bind_spool(tmp_path, monkeypatch, spool=mode != 'no-spool', write_only=mode == 'write-only')
    if mode == 'unadmitted':
        monkeypatch.delenv('FLUX_JOB_ID')
    between = floor + (allowance - floor) // 2
    assert _preflight(tmp_path, between)['declared_bytes'] == between
    with pytest.raises(AdjointIdentityRefused, match='hard geometry floor'):
        _preflight(tmp_path, floor - 1)
def test_a_partial_last_batch_is_charged_at_the_full_bound(
        tmp_path, monkeypatch):
    """Five rows at four per batch: a 4-row entry and a 1-row entry.

    The declaration bounds each entry of a group at the bound entry size, the
    full batch's, so a read-back owner's 1-row entries are charged as 4-row
    ones: one per retained boundary group, live probe plane and checkpoint
    copy. The planning allowance prices them at their own size, so a budget
    at that allowance still falls short.
    """

    from prismaquant.joint_cost_stage_a import AdjointIdentityRefused

    _, allowance, result = _numbers(tmp_path, monkeypatch, probe_microbatch=4)
    demand = result["demand"]
    assert demand["remainder_rows"] == 1 and demand["n_full_batches"] == 1
    unit_sets = (demand["n_retained_boundary_groups"] + demand["n_probes"]
                 + demand["n_checkpoints"] * demand["n_probes"])
    charged = allowance + unit_sets * (
        demand["per_full_tensor_nbytes"] - demand["per_remainder_tensor_nbytes"])
    assert charged > allowance
    _bind_spool(tmp_path, monkeypatch)
    with pytest.raises(AdjointIdentityRefused) as refused:
        _preflight(tmp_path, allowance, probe_microbatch=4)
    assert str(charged) in str(refused.value), str(refused.value)
    assert _preflight(tmp_path, charged,
                      probe_microbatch=4)["declared_bytes"] == charged
