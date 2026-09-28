"""``run_layer_quantum_core`` refuses an unholdable cotangent plane before the load (PQ #1141).

The dispatcher checks the plane against the sealed host bound at admission; a
quantum launched some other way (a resume, a hand launch) still reaches
``run_layer_quantum_core``, whose own check runs before the checkpoint load.
The tests drive the real quantum on the runtime fixture with a policy whose
host bound cannot hold the plane and no scratch pair in the environment. The
driver is what is mutated, not the fixture: with the guard neutralised the
same launch reaches the load, so the guard is what stops it.
"""
from __future__ import annotations

import pytest

from prismaquant import joint_adjoint_checkpoints, joint_stageb_resources
from prismaquant.joint_stageb_resources import COTANGENT_SCRATCH_ENV
from prismaquant.production_weight_cache import ProductionWeightCache

import test_joint_cost_quantum_runtime as runtime
from test_quantum_failure_counters import _Stop
from test_quantum_probe_identity_once_1183 import _campaign

#: Owners at zero and a one-byte host bound: any plane is over it.
_POLICY = {
    "budget": {"safety_margin_bytes": 0, "metadata_reserve_bytes": 0,
               "load_buffer_bytes": 0, "read_page_reserve_bytes": 0,
               "retained_render_cap_bytes": 0},
    "limits": {"host_bytes": 1},
}


def test_a_plane_the_host_cannot_hold_refuses_before_the_checkpoint_load(tmp_path, monkeypatch):
    loads = []
    monkeypatch.setattr(joint_adjoint_checkpoints, "load_adjoint_checkpoint",
                        lambda *a, **k: loads.append(1))
    for name in COTANGENT_SCRATCH_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ProductionWeightCache, "_joint_stage_b_resource_policy",
                        _POLICY, raising=False)
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="cotangent plane is held on the host"):
        runtime._run_quantum(
            tmp_path, monkeypatch, single=single, layer=1, receipt=receipt,
            output_root=output_root, plan_sha=runtime._hex("d"),
            prepared_sha=runtime._hex("e"))
    assert loads == [], "the guard must refuse before the checkpoint is read"


def test_with_the_guard_neutralised_the_same_launch_reaches_the_load(tmp_path, monkeypatch):
    """Mutate the driver: the refusal above is the guard's, not a later failure."""
    monkeypatch.setattr(joint_stageb_resources, "verify_cotangent_plane_fits",
                        lambda *args, **kwargs: None)
    for name in COTANGENT_SCRATCH_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ProductionWeightCache, "_joint_stage_b_resource_policy",
                        _POLICY, raising=False)
    loads = []

    def load(*args, **kwargs):
        loads.append(1)
        raise _Stop("checkpoint load reached")

    monkeypatch.setattr(joint_adjoint_checkpoints, "load_adjoint_checkpoint", load)
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    with pytest.raises(_Stop):
        runtime._run_quantum(
            tmp_path, monkeypatch, single=single, layer=1, receipt=receipt,
            output_root=output_root, plan_sha=runtime._hex("d"),
            prepared_sha=runtime._hex("e"))
    assert loads == [1]


def test_a_scratch_pair_that_covers_the_plane_passes_the_guard(tmp_path, monkeypatch):
    root, ceiling = COTANGENT_SCRATCH_ENV
    (tmp_path / "scratch").mkdir()
    for name, value in ((root, str(tmp_path / "scratch")), (ceiling, str(1 << 40))):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(ProductionWeightCache, "_joint_stage_b_resource_policy",
                        _POLICY, raising=False)
    loads = []

    def load(*args, **kwargs):
        loads.append(1)
        raise _Stop("checkpoint load reached")

    monkeypatch.setattr(joint_adjoint_checkpoints, "load_adjoint_checkpoint", load)
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    with pytest.raises(_Stop):
        runtime._run_quantum(
            tmp_path, monkeypatch, single=single, layer=1, receipt=receipt,
            output_root=output_root, plan_sha=runtime._hex("d"),
            prepared_sha=runtime._hex("e"))
    assert loads == [1]
