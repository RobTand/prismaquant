"""Historical pacing fixture now proves real native application copies and release."""
from pathlib import Path
from prismaquant.stage_a_produced_output import read_bound_document
from test_band_serial_handoff_produced import band_campaign, producer_owner
from prismaquant.streamed_boundary_artifacts import StreamedBoundaryArtifacts


def test_actual_export_commit_release_lifecycle(tmp_path, monkeypatch):
    producer, _, storage = band_campaign(tmp_path)
    pub = producer_owner(tmp_path, producer, storage, monkeypatch, spool=True)
    storage = {**storage, 'directory': str(Path(pub.output_prefix) / 'exact')}
    owner = StreamedBoundaryArtifacts(storage)
    owner.bind({'component': 'native exact copy lifecycle'}, n_probes=1, published=True)
    owner.bind_produced_output(pub, group_size=2, n_batches=4,
        max_entry_tensor_bytes=32, origin_lifetime='retain')
    payloads = [b'independently known first body', b'independently known second body']
    with owner:
        refs = [owner.write_produced_files([(f'group-{i}.bin', payload)],
            kind='component', boundary_index=i)[0] for i, payload in enumerate(payloads)]
    assert [Path(ref.path).read_bytes() for ref in refs] == payloads
    groups = [read_bound_document(binding) for binding in pub._groups.values()]
    assert len(groups) == 2 and all(group['lifetime'] == 'retain' for group in groups)
    assert sorted(row['path'] for group in groups for row in group['entries']) == sorted(ref.path for ref in refs)
    report = owner.produced_output_report()['local_spool']
    assert len(report['exports']) == 2 and report['pending_groups'] == 0
    assert all(row['durable'] and row['released'] and row['publication_error'] is None for row in report['exports'])
    assert not list((tmp_path / 'local').rglob('*.bin'))
