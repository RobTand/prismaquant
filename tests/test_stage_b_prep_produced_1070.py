"""Actual generator metadata publication under an immutable native declaration.

The execution context is an offline fixture. Every generated file, descriptor,
prewrite and application manifest is real; native execution is proved separately.
"""
import hashlib
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prismaquant import stage_b_prep_io as io, native_results
from prismaquant.stage_a_produced_output import publish_document, read_bound_document
from test_stage_b_prep_io_1070 import generator_args, prep_fixture, write_submission
from produced_native_fixtures import selected_publications


def admitted_owner(tmp_path, declaration, monkeypatch):
    binding = publish_document(tmp_path / 'declaration.json', declaration)
    record = tmp_path / 'record'; record.mkdir()
    producer = {'schema': 'prismaquant.flux_result_producer.v1', 'job_id': '123',
        'record_dir': str(record), 'launch_sha256': 'a' * 64,
        'argv': ['fixture', '--produced-declaration', binding['path'],
            '--produced-declaration-sha256', binding['sha256']]}
    monkeypatch.setattr(native_results, 'execution_context', lambda: producer)
    register = selected_publications(monkeypatch)
    # The only terminal publication seam. Scientific metadata below is real.
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication
    monkeypatch.setattr(BoundaryProducedPublication, 'finalize', lambda pub, *a, **k: register(pub))
    return producer


def _files_under(root):
    return {str(path): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def test_the_generator_commits_every_file_it_writes(tmp_path, monkeypatch):
    import regenerate_joint_quanta as regen
    campaign, receipt_path, _ = prep_fixture(tmp_path)
    metadata = tmp_path / 'metadata-generation'
    _, _, declaration = write_submission(tmp_path, campaign, receipt_path, metadata)
    producer = admitted_owner(tmp_path, declaration, monkeypatch)
    io.reset_preparation_publications_for_tests()
    try:
        argv = generator_args(tmp_path, campaign, receipt_path, metadata) + ['--produced-output']
        assert regen.main(argv) == 0
        publication = io._BOUND[producer['launch_sha256']]
        assert [batch['kind'] for batch in publication.batches] == ['manifests', 'records']
        index_path = Path(producer['record_dir']) / 'produced' / 'index.json'
        assert index_path.is_file()
        entries = [entry for batch in publication.batches
            for entry in read_bound_document(batch['ref'])['entries']]
        written = _files_under(metadata)
        assert written and {entry['path'] for entry in entries} == set(written)
        for entry in entries:
            raw = written[entry['path']]
            assert entry['bytes'] == len(raw)
            assert entry['sha256'] == hashlib.sha256(raw).hexdigest()
        before = list(publication.batches)
        assert regen.main(argv) == 0
        assert publication.batches == before and _files_under(metadata) == written
    finally:
        io.reset_preparation_publications_for_tests()


def test_composed_generator_defers_finalization_until_launch_recipe(tmp_path, monkeypatch):
    import regenerate_joint_quanta as regen
    from prismaquant.stage_a_produced_output import decode_produced_document
    campaign, receipt_path, _ = prep_fixture(tmp_path)
    metadata = tmp_path / 'metadata-generation'
    _, _, declaration = write_submission(tmp_path, campaign, receipt_path, metadata)
    producer = admitted_owner(tmp_path, declaration, monkeypatch)
    io.reset_preparation_publications_for_tests()
    try:
        argv = generator_args(tmp_path, campaign, receipt_path, metadata) + ['--produced-output']
        assert regen.main(argv, finalize_publication=False) == 0
        publication = io._BOUND[producer['launch_sha256']]
        index_path = Path(producer['record_dir']) / 'produced' / 'index.json'
        assert not index_path.exists()
        generated = _files_under(metadata)
        assert generated and (metadata / 'records' / 'records.json').is_file()
        launch = metadata / 'launch.json'
        payload = b'{"known": "enclosing preparation launch"}\n'
        assert io.publish_files(publication, 'launch', [(launch, payload, 'launch')]) == [launch]
        publication.finalize(Path(producer['record_dir']) / 'produced-result.json')
        index = decode_produced_document(index_path.read_bytes())
        assert index['groups'] == [publication.publication._groups[key]
                                  for key in sorted(publication.publication._groups)]
        entries = [entry for binding in index['groups']
                   for entry in read_bound_document(binding)['entries']]
        expected = {**generated, str(launch): payload}
        assert {entry['path'] for entry in entries} == set(expected)
        for entry in entries:
            raw = expected[entry['path']]
            assert Path(entry['path']).read_bytes() == raw
            assert entry['sha256'] == hashlib.sha256(raw).hexdigest()
            assert entry['bytes'] == len(raw)
    finally:
        io.reset_preparation_publications_for_tests()


def test_a_bound_publication_refuses_before_writing(tmp_path, monkeypatch):
    metadata = tmp_path / 'metadata'
    declaration = io.build_preparation_template(metadata_root=metadata,
        payload_max_bytes=io.preparation_payload_ceiling(1))
    admitted_owner(tmp_path, declaration, monkeypatch)
    io.reset_preparation_publications_for_tests()
    try:
        with pytest.raises(io.PreparationPublicationRefused, match='metadata root differs'):
            io.bind_preparation_publication(tmp_path / 'elsewhere', required=True)
        publication = io.bind_preparation_publication(metadata, required=True)
        assert io.bind_preparation_publication(metadata, required=True) is publication
        outside = tmp_path / 'outside.json'
        with pytest.raises(io.PreparationPublicationRefused, match="outside the template"):
            io.publish_files(publication, 'stray', [(outside, b'known bytes', 'stray')])
        assert not outside.exists()
        metadata.mkdir()
        inside = metadata / 'a.json'
        assert io.publish_files(publication, 'one', [(inside, b'independent A', 'a')]) == [inside]
        assert inside.read_bytes() == b'independent A'
        with pytest.raises(io.PreparationPublicationRefused, match='once per launch'):
            io.publish_files(publication, 'one', [(metadata / 'b.json', b'independent B', 'b')])
        assert not (metadata / 'b.json').exists()
    finally:
        io.reset_preparation_publications_for_tests()
