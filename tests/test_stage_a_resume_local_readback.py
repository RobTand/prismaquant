"""Resumed probe-major and fused rolls read new cotangents from local aliases."""
from types import SimpleNamespace
import pytest
from produced_native_fixtures import publication
from prismaquant.streamed_boundary_artifacts import BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts
import test_stage_a_same_box_readback as readback
GROUP_SIZE, N_BATCHES, N_PROBES, TOP = (readback.GROUP_SIZE, readback.N_BATCHES,
    readback.N_PROBES, readback.TOP)
IDENTITY = {'source_model': 'fixture'}


def _storage_config(prefix):
    return {'schema': BOUNDARY_STORAGE_SCHEMA, 'directory': str(prefix / 'exact'),
        'max_resident_bytes': 1 << 24, 'max_auxiliary_bytes': 1 << 24,
        'max_artifact_bytes': 1 << 24, 'prefetch_batches': GROUP_SIZE}


def _interrupted_run(prefix):
    storage = StreamedBoundaryArtifacts(_storage_config(prefix))
    storage.bind(IDENTITY, n_probes=N_PROBES, published=True)
    boundaries = {layer: [storage.write(readback._boundary(layer, batch),
        batch_index=batch, boundary_index=layer) for batch in range(N_BATCHES)]
        for layer in range(TOP)}
    plane = {probe: [storage.write(readback._tail(probe, batch), batch_index=batch,
        boundary_index=TOP, probe_index=probe) for batch in range(N_BATCHES)]
        for probe in range(N_PROBES)}
    return dict(storage.session), boundaries, plane


def _rebound_owner(tmp_path, session, monkeypatch, *, read_order):
    pub = publication(tmp_path, group_size=GROUP_SIZE, tensor_bytes=1 << 14,
        payload_max=readback.PAYLOAD_MAX_BYTES,
        spool_bytes=2 * 2 * N_BATCHES * ((1 << 14) + 65536))
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    storage = StreamedBoundaryArtifacts(_storage_config(tmp_path / 'output'))
    storage.rebind(session, identity=IDENTITY, n_probes=N_PROBES)
    storage.bind_produced_output(pub, group_size=GROUP_SIZE, n_batches=N_BATCHES,
        max_entry_tensor_bytes=1 << 14, read_order=read_order)
    return storage
def _roll_probe_major(storage, batches, grad_outs):
    """``test_stage_a_same_box_readback._run_chain``'s roll, from ``TOP``."""

    from prismaquant.streamed_boundary_artifacts import prefetched_boundary_batches

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

    from prismaquant.streamed_boundary_artifacts import prefetched_fused_boundary_windows

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



@pytest.mark.parametrize('read_order,roll', [
    ('probe_major', _roll_probe_major), ('sample_major', _roll_sample_major)])
def test_a_resumed_chain_reads_its_own_cotangents_from_the_spool(tmp_path, monkeypatch, read_order, roll):
    session, boundaries, plane = _interrupted_run(tmp_path / 'output')
    storage = _rebound_owner(tmp_path, session, monkeypatch, read_order=read_order)
    borrowed = [ref for layer in range(TOP) for ref in boundaries[layer]]
    checkpoint = [ref for probe in range(N_PROBES) for ref in plane[probe]]
    storage.authorize_resume_inputs(borrowed, checkpoint, boundary=TOP)
    batches = [SimpleNamespace(activations_cpu=[boundaries[layer][batch]
        for layer in range(TOP)]) for batch in range(N_BATCHES)]
    grad_outs = {probe: list(plane[probe]) for probe in range(N_PROBES)}
    with storage:
        final = roll(storage, batches, grad_outs)
    assert storage.telemetry['produced_local_reads'] > 0
    assert readback._payload_digests(final) == readback._payload_digests(readback._reference_chain())
    assert not storage._deferred_unlinks
