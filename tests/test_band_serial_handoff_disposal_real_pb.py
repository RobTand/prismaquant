"""Failed write-only handoffs dispose only their own fenced uncommitted files.

The historical module name identifies retained coverage. Transport is native
application publication; terminal execution is covered separately.
"""
from pathlib import Path
import pytest
import torch
from test_band_serial_handoff_produced import (
    ENTRY_GROUPS, N_BATCHES, N_PROBES, PREFETCH, _adjoint_slice, _Owner,
    band_campaign, producer_owner)


def _group_ids(publication, producer):
    return [publication.batch_id_for(kind='cotangent', boundary_index=producer['layer'],
        probe_index=probe, group_index=group) for probe in range(N_PROBES)
        for group in range(-(-N_BATCHES // PREFETCH))]


def _emitter(producer, storage, publication, emitters):
    from prismaquant.joint_quantum_handoff import HandoffEmitter
    emitter = HandoffEmitter(record=producer, adjoint_slice=_adjoint_slice(producer),
        boundary_storage=storage, capture_batch=1, publication=publication)
    emitters.append(emitter)
    return emitter


def _plane():
    return {(p, b): torch.full((2, 4), 10.0 * p + b)
        for p in range(N_PROBES) for b in range(N_BATCHES)}


def _owners():
    return [[_Owner({'scale': float(p + b)}) for b in range(N_BATCHES)]
        for p in range(N_PROBES)]


def _entries(producer):
    from prismaquant.joint_quantum_handoff import handoff_root
    root = handoff_root(producer['output_space']['root'])
    return sorted(root.glob('*/entries/*')), sorted(root.glob('*/handoff.json'))


@pytest.mark.parametrize('spool', [False, True])
def test_a_failed_handoff_disposes_its_incomplete_group(tmp_path, monkeypatch, spool):
    producer, _, storage = band_campaign(tmp_path)
    pub = producer_owner(tmp_path, producer, storage, monkeypatch, spool=spool)
    source = _plane()
    class Plane(dict):
        def __getitem__(self, key):
            if key == (0, 1):
                raise RuntimeError('fixture: plane read failed')
            return source[key]
    emitters = []
    with pytest.raises(RuntimeError, match='plane read failed'):
        _emitter(producer, storage, pub, emitters).emit(grad_plane=Plane(),
            cotangent_owners=_owners(), n_probes=N_PROBES, n_batches=N_BATCHES,
            kda_capture_kernel=None)
    assert _entries(producer) == ([], [])
    assert set(pub._prewrites) == pub._released
    assert not emitters[0].export_report['retained_uncommitted']
    assert not list((tmp_path / 'local').rglob('*.pt'))


def test_a_failed_handoff_disposes_its_landed_uncommitted_groups(tmp_path, monkeypatch):
    from prismaquant.streamed_boundary_artifacts import StreamedBoundaryArtifacts
    producer, _, storage = band_campaign(tmp_path)
    pub = producer_owner(tmp_path, producer, storage, monkeypatch, spool=True)
    landed = []
    def failing_settle(self):
        self._local_output_spool.drain()
        landed.extend(path for path in Path(pub.output_prefix).rglob('*.pt'))
        assert len(landed) == N_PROBES * N_BATCHES
        assert len(pub._groups) == ENTRY_GROUPS
        raise RuntimeError('fixture: settle failed after every copy landed')
    monkeypatch.setattr(StreamedBoundaryArtifacts, 'settle_local_output', failing_settle)
    emitters = []
    with pytest.raises(RuntimeError, match='settle failed'):
        _emitter(producer, storage, pub, emitters).emit(grad_plane=_plane(),
            cotangent_owners=_owners(), n_probes=N_PROBES, n_batches=N_BATCHES,
            kda_capture_kernel=None)
    assert landed and all(not path.exists() for path in landed)
    assert _entries(producer) == ([], [])
    assert set(pub._prewrites) == pub._released
    assert all(row['released'] for row in emitters[0].export_report['local_spool']['exports'])
    assert not list((tmp_path / 'local').rglob('*.pt'))


@pytest.mark.parametrize('mutation', ['replace', 'foreign'])
def test_failed_disposal_retains_changed_or_foreign_files(tmp_path, monkeypatch, mutation):
    from prismaquant.streamed_boundary_artifacts import StreamedBoundaryArtifacts
    producer, _, storage = band_campaign(tmp_path)
    pub = producer_owner(tmp_path, producer, storage, monkeypatch, spool=mutation == 'foreign')
    original = StreamedBoundaryArtifacts._dispose_uncommitted_group
    changed = []
    def mutate(self, group):
        if not changed:
            if mutation == 'replace':
                target = Path(group['references'][0].path)
                target.unlink()
                target.write_bytes(b'independently known replacement')
            else:
                target = self._local_output_spool.directory(group['batch_id']) / 'foreign'
                target.write_bytes(b'independently known foreign file')
            changed.append(target)
        return original(self, group)
    monkeypatch.setattr(StreamedBoundaryArtifacts, '_dispose_uncommitted_group', mutate)
    class Plane(dict):
        def __getitem__(self, key):
            if key == (0, 1):
                raise RuntimeError('fixture: plane read failed')
            return _plane()[key]
    emitters = []
    with pytest.raises(RuntimeError, match='plane read failed') as caught:
        _emitter(producer, storage, pub, emitters).emit(grad_plane=Plane(),
            cotangent_owners=_owners(), n_probes=N_PROBES, n_batches=N_BATCHES,
            kda_capture_kernel=None)
    assert changed[0].read_bytes() == (b'independently known replacement' if mutation == 'replace'
                                     else b'independently known foreign file')
    assert pub._prewrites and not _entries(producer)[1]
    assert emitters[0].export_report['release_errors']
    assert any('cleanup also failed' in note for note in caught.value.__notes__)
