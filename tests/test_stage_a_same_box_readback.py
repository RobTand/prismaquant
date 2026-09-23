"""Same-box cotangent readback and write-behind export (RobTand/prismaquant#1110).

The reverse chain reads each cotangent plane on the box that wrote it one
layer earlier. Before #1110 every such read went through PrismaBuild: the
group's local export had to land, the local copy was released, and the group
was published and staged back from the pool to the same box. These tests pin
the fix:

* a chain of at least three steps on one host asks PrismaBuild to stage none
  of its own cotangent groups, and its output is bitwise identical to the
  staged path's;
* an export that failed refuses at the barrier that waits for it at once,
  naming the export and its state, and leaves a record;
* a slow but live export is waited on at a barrier with no clock.

The writer, the owner, the strict reader, the queue and the fleet are the
real ones. Only PrismaBuild's asynchronous exporter is replaced by a
transport whose receipt the test controls (``test_produced_output_spool``).
"""
from __future__ import annotations

import threading
import time

import pytest
import torch

import test_produced_output_spool as spool_tests
import test_stage_a_produced_boundary_chain as chain
# Autouse, and it must apply HERE too (RobTand/prismaquant#889).
from test_stage_a_produced_boundary_chain import (  # noqa: F401
    _isolated_launch_context)

# PQ #1008: one pinned prismabuild per process (tests/conftest.py).
pytestmark = pytest.mark.own_process

GROUP_SIZE = chain.GROUP_SIZE
N_BATCHES = 2 * GROUP_SIZE
N_PROBES = 2
#: Boundaries 0..TOP-1 are the forward's; TOP is the tail cotangent plane.
TOP = 4
#: The owner's durable payload maxima, priced as production prices its
#: artifact budget (``joint_cost_stage_a`` planning allowance): every file at
#: its tensor bound plus the 64 KiB envelope, which is also PrismaBuild's
#: prewrite ceiling. A group read only on this box is never committed, so its
#: prewrite stays charged at that ceiling until its last file goes (PQ #1110).
#: The forward's boundary groups are never retired here, so they stay; the
#: roll needs two cotangent planes at once. Without the group-final prewrite
#: release the chain would hold every plane it wrote (TOP + 1 of them) and
#: this refuses the first write past two.
GROUP_CEILING_BYTES = GROUP_SIZE * ((1 << 14) + 65536)
PAYLOAD_MAX_BYTES = GROUP_CEILING_BYTES * (
    TOP * N_BATCHES // GROUP_SIZE + 2 * N_PROBES * N_BATCHES // GROUP_SIZE)


class LandingExport(spool_tests.ControlledExport):
    """Acknowledges an export the first time it is polled.

    PrismaBuild's exporter runs on the producing host beside the owner; here
    the copy to the canonical destination happens at the first poll, so every
    path that waits for a landing (the staged read included) reaches the step
    after it.
    """

    def poll_group(self, batch_id):
        group = self.groups[batch_id]
        if group["entries"] and not group["complete"] and not group["failure"]:
            self.acknowledge(batch_id)
        return super().poll_group(batch_id)


@pytest.fixture
def closing():
    """Close every owner a test made, so no stager thread outlives it."""

    owners = []
    yield owners.append
    for storage in owners:
        storage._produced_stop_stager()


def _spool_owner(tmp_path, *, backend_type=LandingExport, capacity=1 << 30,
                 staging_timeout_s=60.0):
    from prismaquant.produced_output_spool import ProducedOutputSpool

    storage, publication, q, env, pb_repo = chain._bound_owner(
        tmp_path, n_batches=N_BATCHES, window_gib=8, gib=16,
        payload_max_bytes=PAYLOAD_MAX_BYTES, n_probes=N_PROBES,
        staging_timeout_s=staging_timeout_s)
    storage._published = True
    backend = backend_type(tmp_path / "local", capacity=capacity)
    storage._local_output_spool = ProducedOutputSpool(
        backend, capacity_deferred=spool_tests.CapacityDeferred)
    return storage, publication, q, env, pb_repo, backend


def _count_staging_requests(publication):
    """Count PrismaBuild staging calls per group kind.

    ``publish`` seals a group's first mover and ``ensure_batch_materialized``
    a repeat one; ``await_materialized`` waits for a mover's receipt and
    ``reader_context`` composes a staged read. Each is a request to stage a
    group through PrismaBuild, so an own group read on its own box must make
    none of them.
    """

    counts = {}

    def wrap(name):
        real = getattr(publication, name)

        def call(*args, **kwargs):
            batch_id = str(kwargs.get("batch_id", args[0] if args else ""))
            kind = batch_id.split("-")[1] if batch_id.startswith("stagea-") else "?"
            counts.setdefault(kind, {}).setdefault(name, 0)
            counts[kind][name] += 1
            return real(*args, **kwargs)

        setattr(publication, name, call)

    for name in ("publish", "ensure_batch_materialized", "await_materialized",
                 "reader_context"):
        wrap(name)
    return counts


def _boundary(layer, batch):
    return torch.full((2, 8), float(layer)) + torch.arange(8.0) * (batch + 1)


def _tail(probe, batch):
    return torch.linspace(-1.0, 1.0, 16).reshape(2, 8) * (probe + 1) + batch


def _roll(incoming, boundary):
    """One chain step. Float32, deterministic: the bytes are the claim."""

    return (incoming * 0.75 + boundary.sin()).contiguous()


def _reference_chain():
    """The chain computed in memory, with no storage at all."""

    planes = {p: [_tail(p, b) for b in range(N_BATCHES)] for p in range(N_PROBES)}
    for layer in range(TOP - 1, -1, -1):
        planes = {p: [_roll(planes[p][b], _boundary(layer, b))
                      for b in range(N_BATCHES)] for p in range(N_PROBES)}
    return planes


def _run_chain(storage):
    """Forward, tail, then TOP steps of the probe-major roll, as Stage A runs.

    Returns the final plane's tensors, read back from the storage's own
    windows (the last roll is written with ``read_back=False`` as in
    production, so the final plane is taken from the roll itself).
    """

    from prismaquant.cost_streaming import prefetched_boundary_batches

    batches = []
    for batch in range(N_BATCHES):
        batches.append(type("Batch", (), {})())
        batches[batch].activations_cpu = [
            storage.write(_boundary(layer, batch), batch_index=batch,
                          boundary_index=layer)
            for layer in range(TOP)]
    grad_outs = {p: [storage.write(_tail(p, b), batch_index=b,
                                   boundary_index=TOP, probe_index=p)
                     for b in range(N_BATCHES)]
                 for p in range(N_PROBES)}
    final = {p: [None] * N_BATCHES for p in range(N_PROBES)}
    for layer in range(TOP - 1, -1, -1):
        for probe in range(N_PROBES):
            following = ((layer, grad_outs[probe + 1]) if probe + 1 < N_PROBES
                         else ((layer - 1, grad_outs[0]) if layer > 0 else None))
            with prefetched_boundary_batches(
                    storage, batches, layer, incoming=grad_outs[probe],
                    then=following) as windows:
                for index, _batch, boundary, incoming in windows:
                    rolled = _roll(incoming, boundary)
                    if layer == 0:
                        final[probe][index] = rolled.clone()
                    grad_outs[probe][index] = storage.write(
                        rolled, batch_index=index, boundary_index=layer,
                        probe_index=probe, previous=grad_outs[probe][index],
                        **({} if layer > 0 else {"read_back": False}))
    return final, grad_outs


def _payload_digests(planes):
    from prismaquant.stage_a_chain_seed import tensor_payload_sha256
    return {p: [tensor_payload_sha256(t) for t in planes[p]] for p in sorted(planes)}


def test_a_chain_on_one_host_stages_none_of_its_own_cotangent_groups(
        tmp_path, monkeypatch, closing):
    """RED on the parent: every cotangent plane was staged back to its own box.

    ``TOP`` (4) chain steps over a two-probe plane of two groups each. The
    local run reads every own cotangent group from the spool that wrote it,
    so PrismaBuild is asked to stage none of them. The staged run is the same
    chain with the local read switched off, which is the path every read took
    before #1110; both produce the in-memory chain's exact bytes.
    """

    local = _spool_owner(tmp_path / "local-run")
    storage, publication, q, env, pb_repo, _backend = local
    closing(storage)
    counts = _count_staging_requests(publication)
    with chain._fleet(q, tmp_path / "local-run"):
        chain._strict(monkeypatch, env, pb_repo, q)
        local_final, _ = _run_chain(storage)
        assert storage.drain_produced_stager(60.0)
        storage.settle_local_output()
        storage.settle_produced_releases()
        # The last plane's prewrite releases run on the stager.
        assert storage.drain_produced_stager(60.0)
    assert counts.get("cotangent", {}) == {}, (
        "an own cotangent group read on its own box was staged through "
        "PrismaBuild", counts)
    assert storage.telemetry.get("produced_local_reads", 0) > 0
    # Every rolled-away plane gave its prewrite back (the tail and planes
    # TOP-1..1; plane 0 is written with no read to follow and settles).
    assert storage.telemetry["produced_groups_prewrite_released"] >= (
        TOP * N_PROBES * N_BATCHES // GROUP_SIZE)

    staged = _spool_owner(tmp_path / "staged-run")
    s_storage, s_publication, s_q, s_env, s_pb_repo, _s_backend = staged
    closing(s_storage)
    # The path every read took before #1110: no entry is read locally, so
    # every group a read needs is published and staged through PrismaBuild.
    from contextlib import nullcontext
    monkeypatch.setattr(s_storage._local_output_spool, "local_reads",
                        lambda references: nullcontext({}))
    s_counts = _count_staging_requests(s_publication)
    with chain._fleet(s_q, tmp_path / "staged-run"):
        chain._strict(monkeypatch, s_env, s_pb_repo, s_q)
        staged_final, _ = _run_chain(s_storage)
        assert s_storage.drain_produced_stager(60.0)
        s_storage.settle_local_output()
        s_storage.settle_produced_releases()
    assert s_counts["cotangent"]["publish"] > 0, s_counts
    assert s_storage.telemetry.get("produced_local_reads", 0) == 0

    expected = _payload_digests(_reference_chain())
    assert _payload_digests(staged_final) == expected
    assert _payload_digests(local_final) == expected, (
        "the local read changed the chain's bytes")


class FailingExport(spool_tests.ControlledExport):
    """An export that PrismaBuild reports failed, as ``poll_group`` does."""

    def submit_group(self, batch_id, *, entries):
        answer = super().submit_group(batch_id, entries=entries)
        self.groups[batch_id]["export_key"] = answer["export_key"]
        return answer

    def poll_group(self, batch_id):
        self.polls += 1
        group = self.groups[batch_id]
        if group["failure"]:
            return {"ok": False, "complete": False,
                    "export_key": group["export_key"],
                    "refusal": group["failure"]}
        return {"ok": True, "complete": group["complete"],
                "export_key": group["export_key"]}


def _write_cotangent_group(storage, *, probe=0, layer=TOP):
    return [storage.write(_tail(probe, b), batch_index=b, boundary_index=layer,
                          probe_index=probe)
            for b in range(GROUP_SIZE)]


def _barrier(storage, which, references):
    if which == "settle":
        storage.settle_local_output()
    else:
        storage.await_checkpoint_references(references)


@pytest.mark.parametrize("barrier", ["settle", "checkpoint"])
def test_a_failed_export_refuses_at_its_barrier_at_once_with_a_record(
        tmp_path, closing, barrier):
    """The barrier asks once, reads a failed export, and refuses.

    The refusal names the export action and PrismaBuild's state for it, and
    the owner's report carries the same record; a line on stdout is not a
    record.
    """

    from prismaquant.produced_output_spool import ProducedOutputSpoolRefused

    storage, _pub, _q, _env, _pb, backend = _spool_owner(
        tmp_path, backend_type=FailingExport, staging_timeout_s=2.0)
    closing(storage)
    references = _write_cotangent_group(storage)
    assert storage.drain_produced_stager(60.0)
    (batch_id,) = [batch for batch, group in backend.groups.items()
                   if group["entries"]]
    backend.groups[batch_id]["failure"] = "export-failed-without-ack"
    export_key = backend.groups[batch_id]["export_key"]
    started = time.monotonic()
    with pytest.raises(ProducedOutputSpoolRefused) as refused:
        _barrier(storage, barrier, references)
    assert time.monotonic() - started < 1.0, "a failed export is not waited on"
    message = str(refused.value)
    assert export_key in message and "export-failed-without-ack" in message
    records = storage.produced_output_report()["local_spool"]["refusals"]
    assert [(r["batch_id"], r["export_key"], r["state"]) for r in records] == [
        (batch_id, export_key, "export-failed-without-ack")], records


@pytest.mark.parametrize("barrier", ["settle", "checkpoint"])
def test_a_slow_live_export_is_waited_on_without_a_clock(
        tmp_path, closing, barrier):
    """A live export that outlasts every configured budget is still waited on.

    The owner's staging budget is 0.2 s and the export lands after ten times
    that. Before #1110 the barrier raised ``TimeoutError`` at its own clock
    while PrismaBuild still had the export in hand; now nothing but the
    export's state ends the wait.
    """

    storage, _pub, _q, _env, _pb, backend = _spool_owner(
        tmp_path, backend_type=FailingExport, staging_timeout_s=0.2)
    closing(storage)
    references = _write_cotangent_group(storage)
    assert storage.drain_produced_stager(60.0)
    (batch_id,) = [batch for batch, group in backend.groups.items()
                   if group["entries"]]
    landed = threading.Event()

    def land_later():
        time.sleep(2.0)
        backend.acknowledge(batch_id)
        landed.set()

    thread = threading.Thread(target=land_later)
    thread.start()
    try:
        _barrier(storage, barrier, references)
    finally:
        thread.join(timeout=10)
    assert landed.is_set()
    assert storage._local_output_spool.landed(batch_id)
    # The wait is a record: which export, where, and for how long.
    waits = [w for w in storage.produced_output_report()["local_spool"]["waits"]
             if w["kind"] == "export" and w["batch_id"] == batch_id]
    assert waits and waits[0]["export_key"] == backend.groups[batch_id]["export_key"]
    assert sum(w["seconds"] for w in waits) >= 1.5, waits
