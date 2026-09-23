"""A producer quantum's handoff through a real PrismaBuild produced output (PQ #996).

Inside an admitted action, quantum ``L`` writes its handoff entries into the
output prefix its row declared (``--produced-output-template``). This drives
that path end to end on tiny tensors with the pinned produced-output
candidate that ``test_stage_a_produced_boundary_chain`` qualifies: the
template is the one the dispatcher writes for the row, the owner is a real
sealed request published, claimed and bound on a private queue, and the
emitter writes through the bound publication. The consumer quantum then
binds the handoff with its own checks.

The entries are written with ``read_back=False``, as Stage A writes its last
roll: no read follows in the producer's action, so no stage copy is
published for them. The consumer stages them as ordinary inputs of its
derived readset. ``test_band_serial_handoff_spool_real_pb`` drives the same
emitter through PrismaBuild's local output spool.

Run it in its own pytest invocation: resolving the pinned candidate leaves
``prismabuild`` in ``sys.modules`` (the harness defect that module records).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import test_stage_a_produced_boundary_chain as chain  # noqa: E402
from test_quantum_executable_readset import (  # noqa: E402
    N_BATCHES, N_PROBES, _bind_slice, _tiny_receipt, _tiny_records)
from test_stage_a_produced_boundary_chain import (  # noqa: E402,F401
    _isolated_launch_context)

PREFETCH = 2
ARTIFACT_MAX = 1 << 20


class _Owner:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return dict(self.state)


def _adjoint_slice(record):
    return json.loads(Path(record["adjoint"]["slice_path"]).read_text())


def band_campaign(tmp_path):
    """Layers 3 (producer) and 2 (consumer) of band 4, bound to Stage A."""
    from prismaquant.cost_streaming import BOUNDARY_STORAGE_SCHEMA

    records, _parent = _tiny_records(tmp_path)
    receipt = _tiny_receipt(tmp_path, records[0]["campaign"])
    bound = {record["layer"]: _bind_slice(
        record, receipt, tmp_path / "adjoint-slices")[0]
        for record in records if record["layer"] in (2, 3)}
    storage = {"schema": BOUNDARY_STORAGE_SCHEMA,
               "directory": str(tmp_path / "exact"),
               "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
               "max_artifact_bytes": ARTIFACT_MAX, "prefetch_batches": PREFETCH}
    return bound[3], bound[2], storage


def producer_owner(tmp_path, pb_repo, producer, storage, *,
                   producer_environment=None, claim_capacity=None):
    """The dispatcher's template for ``producer``, and a real admitted owner
    that declared it on a private queue: ``(publication, queue, env)``."""
    from prismabuild import produced_output as po

    import dispatch_joint_quanta as dispatch
    from prismaquant.joint_quantum_handoff import handoff_root
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication

    template_path = dispatch.handoff_template_path(
        producer, plan={"execution": {"boundary_storage": storage}},
        adjoint_slice=_adjoint_slice(producer), tier=chain.TIER,
        output_root=tmp_path / "out")
    template = json.loads(template_path.read_text())
    assert template["output_prefix"] == str(
        handoff_root(producer["output_space"]["root"]).resolve())
    assert template["durable_maxima"]["payload_max_bytes"] == ARTIFACT_MAX

    cas_root = tmp_path / "cas"
    q = chain._queue(tmp_path, gib=4)
    owner = chain._sealed_producer_request(
        tmp_path, cas_root, pb_repo, template,
        producer_environment=producer_environment)
    q.publish(action_key=owner, cas_root=str(cas_root),
              worker_script=str(pb_repo / "tools" / "prismabuild_worker.py"),
              checkout_root=str(tmp_path / "mover-checkout"),
              resources={"cpu": 1, "mem_gb": 1, **po.owner_demand_terms(template)},
              produced_output_template=template)
    claimed = q.claim(owner="w-owner", capacity=claim_capacity)
    assert claimed is not None and claimed["action_key"] == owner
    control = chain._broker_control(q, owner)
    env = {"PRISMABUILD_ACTION_KEY": owner,
           "PRISMABUILD_ACTION_NONCE": control["nonce"],
           "PRISMABUILD_ACTION_SCOPE": control["scope_id"],
           **(producer_environment or {})}
    po.declare_template(q.root, template)
    publication = BoundaryProducedPublication.bind_from_admitted_owner(
        queue_root=q.root, tier=chain.TIER, env=env, command_extra=("--unpaced",))
    chain._announce_tier(q, tmp_path / "stage", pb_repo)
    return publication, q, env


def emit_handoff(producer, storage, publication):
    from prismaquant.joint_quantum_handoff import HandoffEmitter

    plane = {(p, b): torch.full((2, 4), 10.0 * p + b)
             for p in range(N_PROBES) for b in range(N_BATCHES)}
    owners = [[_Owner({"scale": float(p + b)}) for b in range(N_BATCHES)]
              for p in range(N_PROBES)]
    emitter = HandoffEmitter(record=producer, adjoint_slice=_adjoint_slice(producer),
                             boundary_storage=storage, publication=publication)
    published = emitter.emit(grad_plane=plane, cotangent_owners=owners,
                             n_probes=N_PROBES, n_batches=N_BATCHES)
    return published, plane


def check_consumer_binds(published, plane, consumer, publication):
    """The consumer's own checks accept the handoff, and it is the plane."""
    from prismaquant.joint_adjoint_checkpoints import read_exact_entry_tensors
    from prismaquant.joint_quantum_handoff import load_quantum_handoff

    handoff = load_quantum_handoff(published["path"], published["sha256"],
                                   record=consumer,
                                   adjoint_slice=_adjoint_slice(consumer))
    for entry in handoff["activation_entries"]:
        assert publication.contains(Path(entry["path"]))
        coordinates = entry["metadata"]["identity"]["coordinates"]
        tensor = read_exact_entry_tensors(
            [entry], expected_session=handoff["session"])[entry["name"]]
        assert torch.equal(tensor, plane[(coordinates["probe"],
                                          coordinates["batch"])])
    assert publication.contains(Path(handoff["owner_states"]["path"]))
    assert publication.contains(Path(published["path"]))
    return handoff


def test_a_producer_writes_its_handoff_inside_its_declared_output(
        tmp_path, monkeypatch):
    _src, pb_repo = chain._pb_source()
    from prismaquant.staged_lease import set_lease_helper_root
    set_lease_helper_root(str(pb_repo))
    from prismaquant.joint_quantum_handoff import (
        QuantumHandoffRefused, bind_handoff_publication)
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication

    producer, consumer, storage = band_campaign(tmp_path)
    publication, _q, env = producer_owner(tmp_path, pb_repo, producer, storage)

    # The quantum binds exactly as Stage A does in production (queue and
    # tier from the launch context); here that answer is the private queue's.
    calls = []

    def bind(cls, **kwargs):
        calls.append(kwargs)
        return publication

    monkeypatch.setattr(BoundaryProducedPublication, "bind_from_admitted_owner",
                        classmethod(bind))
    assert bind_handoff_publication(boundary_storage=storage, env={}) is None
    assert bind_handoff_publication(boundary_storage=storage, env=env) is publication
    assert calls == [{"queue_root": None, "tier": None, "env": env,
                      "command_extra": ()}]
    with pytest.raises(QuantumHandoffRefused, match="durable payload maximum"):
        bind_handoff_publication(
            boundary_storage={**storage, "max_artifact_bytes": ARTIFACT_MAX * 2},
            env=env)

    published, plane = emit_handoff(producer, storage, publication)
    check_consumer_binds(published, plane, consumer, publication)
