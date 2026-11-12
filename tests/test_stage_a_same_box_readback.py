"""Exact four-step/two-probe arithmetic through bounded same-box IO owners."""
from __future__ import annotations
import threading
from pathlib import Path
import pytest
import torch
from produced_native_fixtures import owner
from prismaquant import io_engine
from prismaquant.produced_output_spool import ProducedExportRefused

GROUP_SIZE = 4
N_BATCHES = 2 * GROUP_SIZE
N_PROBES = 2
TOP = 4
GROUP_CEILING_BYTES = GROUP_SIZE * ((1 << 14) + 65536)
PAYLOAD_MAX_BYTES = GROUP_CEILING_BYTES * (TOP * 2 + 2 * N_PROBES * 2)

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

    from prismaquant.streamed_boundary_artifacts import prefetched_boundary_batches

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


def _spool_owner(tmp_path, monkeypatch):
    storage, pub = owner(tmp_path, n_batches=N_BATCHES, n_probes=N_PROBES,
        group_size=GROUP_SIZE, tensor_bytes=1 << 14, payload_max=PAYLOAD_MAX_BYTES,
        spool_bytes=2 * 2 * N_BATCHES * ((1 << 14) + 65536))
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    return storage, pub


def test_a_chain_on_one_host_reads_exact_local_cotangents(tmp_path, monkeypatch):
    storage, _ = _spool_owner(tmp_path, monkeypatch)
    with storage:
        actual, _ = _run_chain(storage)
    assert _payload_digests(actual) == _payload_digests(_reference_chain())
    assert storage.telemetry['produced_local_reads'] > 0
    assert not storage._deferred_unlinks


@pytest.mark.parametrize('barrier', ['settle', 'checkpoint'])
def test_failed_copy_refuses_at_its_barrier_and_retains_evidence(tmp_path, monkeypatch, barrier):
    storage, pub = _spool_owner(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def fail(*args, **kwargs):
        entered.set()
        assert release.wait(30)
        raise OSError('owned copy failed')
    monkeypatch.setattr(io_engine, 'publish_copy', fail)
    refs = [storage.write(_tail(0, b), batch_index=b, boundary_index=TOP,
        probe_index=0) for b in range(GROUP_SIZE)]
    assert entered.wait(30)
    release.set()
    with pytest.raises(ProducedExportRefused, match='owned copy failed'):
        if barrier == 'settle':
            storage.settle_local_output()
        else:
            storage.await_checkpoint_references(refs)
    storage._local_output_spool.join_copies()
    records = storage.produced_output_report()['local_spool']['refusals']
    assert len(records) == 1 and 'owned copy failed' in records[0]['publication_error']
    assert not pub._groups and all(not Path(ref.path).exists() for ref in refs)
    assert all(Path(row[0].path).is_file() for group in storage._local_output_spool._groups.values()
               for row in group['references'])
    import json
    from prismaquant.joint_cost_stage_a import FAILED_PRODUCED_OUTPUT_RECORDS, _failed_produced_output_record
    space = tmp_path / 'space'; space.mkdir()
    error = ProducedExportRefused('owned copy failed')
    with pytest.raises(ProducedExportRefused) as caught:
        with _failed_produced_output_record(space, storage):
            raise error
    assert caught.value is error
    kept = json.loads((space / FAILED_PRODUCED_OUTPUT_RECORDS).read_text())
    assert kept['produced_output']['local_spool']['refusals'] == records


@pytest.mark.parametrize('barrier', ['settle', 'checkpoint'])
def test_live_copy_finishes_before_the_barrier_publishes_completion(tmp_path, monkeypatch, barrier):
    storage, pub = _spool_owner(tmp_path, monkeypatch)
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    waiting = threading.Event()
    real_copy = io_engine.publish_copy
    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(30)
        return real_copy(*args, **kwargs)
    monkeypatch.setattr(io_engine, 'publish_copy', gated)
    refs = [storage.write(_tail(0, b), batch_index=b, boundary_index=TOP,
        probe_index=0) for b in range(GROUP_SIZE)]
    assert entered.wait(30)
    condition = storage._local_output_spool._cond
    actual_wait = condition.wait
    def observed_wait(*args, **kwargs):
        waiting.set()
        return actual_wait(*args, **kwargs)
    monkeypatch.setattr(condition, 'wait', observed_wait)
    failures = []
    def wait():
        try:
            if barrier == 'settle':
                storage.settle_local_output()
            else:
                storage.await_checkpoint_references(refs)
        except BaseException as error:
            failures.append(error)
        finally:
            completed.set()
    waiter = threading.Thread(target=wait)
    waiter.start()
    try:
        assert waiting.wait(30)
        # Completion cannot precede the copy's explicit release event.
        assert not completed.is_set() and not pub._groups
    finally:
        release.set()
        waiter.join(30)
    assert completed.is_set() and not failures and not waiter.is_alive()
    assert all(Path(ref.path).is_file() for ref in refs)
    with storage.prefetch(refs) as window:
        for b, ref in enumerate(refs):
            assert torch.equal(storage.get(window, ref), _tail(0, b))
