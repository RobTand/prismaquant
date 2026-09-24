"""A resumed chain reads its own write-behind cotangents on its own box.

R13's resume from checkpoint 45 (PrismaBuild action 7c9744ea823b,
2026-09-24) rolled layer 44 from the sealed checkpoint's borrowed plane and
then refused the first window of layer 43 with "exact boundary reference is
stale or belongs to another generation". Layer 43 is the first roll that
reads cotangents the resumed owner wrote itself, and the write-behind spool
(RobTand/prismaquant#1110) still held them, so the window read them from the
local copy.

A resumed owner holds borrowed inputs, so its windows check each entry's
session through the owner (``session_for_reference``). The local read ran
that check on a copy of the reference re-pointed at the spool file, which
the owner never recorded, so the owner refused its own entry. A fresh run
passes no such callback and never showed it.

The owner, the rebind, the resume inputs, the writer, the spool and the
queue are the real ones, bound in the order ``joint_cost_stage_a`` binds a
resumed run: ``rebind``, then ``bind_produced_output``, then
``authorize_resume_inputs``. Only PrismaBuild's asynchronous exporter is
replaced, as in ``test_stage_a_same_box_readback``. The strict tier policy
stays off: the refusal is in the session check every local read runs, under
either policy, and a strict read of the borrowed inputs would need the
action's own input map.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import test_stage_a_produced_boundary_chain as chain
import test_stage_a_same_box_readback as readback
# Autouse, and it must apply HERE too (RobTand/prismaquant#889).
from test_stage_a_produced_boundary_chain import (  # noqa: F401
    _isolated_launch_context)

# PQ #1008: one pinned prismabuild per process (tests/conftest.py).
pytestmark = pytest.mark.own_process

GROUP_SIZE = chain.GROUP_SIZE
N_BATCHES = readback.N_BATCHES
N_PROBES = readback.N_PROBES
TOP = readback.TOP
IDENTITY = {"source_model": "fixture"}


def _storage_config(prefix):
    from prismaquant.cost_streaming import BOUNDARY_STORAGE_SCHEMA
    return {"schema": BOUNDARY_STORAGE_SCHEMA, "directory": str(prefix / "exact"),
            "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
            "max_artifact_bytes": 1 << 24, "prefetch_batches": GROUP_SIZE}


def _interrupted_run(prefix):
    """The run that died below its tail checkpoint.

    It wrote every forward boundary and the tail cotangent plane at ``TOP``,
    which a sealed checkpoint names in place (PQ #1036), then stopped. Its
    generation stays ``running``, as a killed run's does.
    """

    from prismaquant.cost_streaming import StreamedBoundaryArtifacts

    storage = StreamedBoundaryArtifacts(_storage_config(prefix))
    storage.bind(IDENTITY, n_probes=N_PROBES, published=True)
    boundaries = {layer: [storage.write(readback._boundary(layer, batch),
                                        batch_index=batch, boundary_index=layer)
                          for batch in range(N_BATCHES)]
                  for layer in range(TOP)}
    plane = {probe: [storage.write(readback._tail(probe, batch), batch_index=batch,
                                   boundary_index=TOP, probe_index=probe)
                     for batch in range(N_BATCHES)]
             for probe in range(N_PROBES)}
    return dict(storage.session), boundaries, plane


def _rebound_owner(tmp_path, session, *, read_order):
    """``chain._bound_owner``, rebinding ``session`` instead of binding anew."""

    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    from prismaquant.produced_output_spool import ProducedOutputSpool
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication
    from prismaquant.staged_lease import set_lease_helper_root

    _src, pb_repo = chain._pb_source()
    set_lease_helper_root(str(pb_repo))
    from prismabuild import produced_output as po

    cas_root = tmp_path / "cas"
    q = chain._queue(tmp_path, gib=16)
    prefix = tmp_path / "outputs"
    template = chain._template(str(prefix), payload_max_bytes=readback.PAYLOAD_MAX_BYTES,
                               window_gib=8)
    owner = chain._sealed_producer_request(tmp_path, cas_root, pb_repo, template)
    q.publish(action_key=owner, cas_root=str(cas_root),
              worker_script=str(pb_repo / "tools" / "prismabuild_worker.py"),
              checkout_root=str(tmp_path / "mover-checkout"),
              resources={"cpu": 1, "mem_gb": 1, **po.owner_demand_terms(template)},
              produced_output_template=template)
    claimed = q.claim(owner="w-owner")
    assert claimed is not None and claimed["action_key"] == owner
    control = chain._broker_control(q, owner)
    env = {"PRISMABUILD_ACTION_KEY": owner,
           "PRISMABUILD_ACTION_NONCE": control["nonce"],
           "PRISMABUILD_ACTION_SCOPE": control["scope_id"]}
    po.declare_template(q.root, template)
    publication = BoundaryProducedPublication.bind_from_admitted_owner(
        queue_root=q.root, tier=chain.TIER, env=env, command_extra=("--unpaced",))
    assert publication.admit_window().get("ok") is True
    chain._announce_tier(q, tmp_path / "stage", pb_repo)

    storage = StreamedBoundaryArtifacts(_storage_config(prefix))
    storage.rebind(session, identity=IDENTITY, n_probes=N_PROBES)
    storage.bind_produced_output(
        publication, group_size=GROUP_SIZE, n_batches=N_BATCHES,
        max_entry_tensor_bytes=1 << 14, staging_timeout_s=60.0,
        read_order=read_order)
    backend = readback.LandingExport(tmp_path / "local", capacity=1 << 30)
    storage._local_output_spool = ProducedOutputSpool(
        backend, capacity_deferred=readback.spool_tests.CapacityDeferred)
    return storage, q


def _roll_probe_major(storage, batches, grad_outs):
    """``test_stage_a_same_box_readback._run_chain``'s roll, from ``TOP``."""

    from prismaquant.cost_streaming import prefetched_boundary_batches

    final = {probe: [None] * N_BATCHES for probe in range(N_PROBES)}
    for layer in range(TOP - 1, -1, -1):
        for probe in range(N_PROBES):
            following = ((layer, grad_outs[probe + 1]) if probe + 1 < N_PROBES
                         else ((layer - 1, grad_outs[0]) if layer > 0 else None))
            with prefetched_boundary_batches(
                    storage, batches, layer, incoming=grad_outs[probe],
                    then=following) as windows:
                for index, _batch, boundary, incoming in windows:
                    rolled = readback._roll(incoming, boundary)
                    if layer == 0:
                        final[probe][index] = rolled.clone()
                    grad_outs[probe][index] = storage.write(
                        rolled, batch_index=index, boundary_index=layer,
                        probe_index=probe, previous=grad_outs[probe][index],
                        **({} if layer > 0 else {"read_back": False}))
    return final


def _roll_sample_major(storage, batches, grad_outs):
    """The fused roll R13 runs: one window serves every probe (PQ #997)."""

    from prismaquant.cost_streaming import prefetched_fused_boundary_windows

    final = {probe: [None] * N_BATCHES for probe in range(N_PROBES)}
    for layer in range(TOP - 1, -1, -1):
        incoming = [list(grad_outs[probe]) for probe in range(N_PROBES)]
        with prefetched_fused_boundary_windows(
                storage, batches, layer, incoming=incoming,
                window_batches=GROUP_SIZE) as windows:
            for indices, boundary_of, incoming_of in windows:
                for index in indices:
                    for probe in range(N_PROBES):
                        rolled = readback._roll(incoming_of(probe, index),
                                                boundary_of(index))
                        if layer == 0:
                            final[probe][index] = rolled.clone()
                        grad_outs[probe][index] = storage.write(
                            rolled, batch_index=index, boundary_index=layer,
                            probe_index=probe, previous=incoming[probe][index],
                            **({} if layer > 0 else {"read_back": False}))
    return final


@pytest.mark.parametrize("read_order, roll", [
    ("probe_major", _roll_probe_major),
    ("sample_major", _roll_sample_major),
], ids=["probe_major", "sample_major"])
def test_a_resumed_chain_reads_its_own_cotangents_from_the_spool(
        tmp_path, readback_closing, read_order, roll):
    """RED on 3f471d7a6135: the second roll below the checkpoint refuses.

    The first roll below ``TOP`` reads only borrowed entries: the forward
    boundaries and the checkpoint plane. It writes this owner's first
    cotangent plane, which the spool keeps for the next read. The second roll
    reads that plane from the local copy. The resumed chain must produce the
    in-memory chain's exact bytes, and must have read its own entries locally.
    """

    prefix = tmp_path / "outputs"
    prefix.mkdir(parents=True)
    session, boundaries, plane = _interrupted_run(prefix)
    storage, q = _rebound_owner(tmp_path, session, read_order=read_order)
    readback_closing(storage)
    borrowed = [reference for layer in range(TOP) for reference in boundaries[layer]]
    checkpoint = [reference for probe in range(N_PROBES) for reference in plane[probe]]
    storage.authorize_resume_inputs(borrowed, checkpoint, boundary=TOP)

    batches = [SimpleNamespace(activations_cpu=[boundaries[layer][batch]
                                                for layer in range(TOP)])
               for batch in range(N_BATCHES)]
    grad_outs = {probe: list(plane[probe]) for probe in range(N_PROBES)}
    with chain._fleet(q, tmp_path):
        final = roll(storage, batches, grad_outs)
        assert storage.drain_produced_stager(60.0)
        storage.settle_local_output()
        storage.settle_produced_releases()

    assert storage.telemetry.get("produced_local_reads", 0) > 0, (
        "the resumed chain never read its own entries from the spool, so this "
        "test did not reach the path it guards")
    assert readback._payload_digests(final) == readback._payload_digests(
        readback._reference_chain())


@pytest.fixture
def readback_closing():
    """Close every owner a test made, so no stager thread outlives it."""

    owners = []
    yield owners.append
    for storage in owners:
        storage._produced_stop_stager()
