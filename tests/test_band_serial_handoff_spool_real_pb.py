"""Exact tensor handoff publication through the joined local copy owner."""
import json
from pathlib import Path
import threading
import pytest
from prismaquant import io_engine
from prismaquant.stage_a_produced_output import read_bound_document
from test_band_serial_handoff_produced import (ENTRY_GROUPS, band_campaign,
    check_consumer_binds, check_origin_batches, emit_handoff, producer_owner)


def test_the_handoff_copies_entries_before_its_record(tmp_path, monkeypatch):
    producer, consumer, storage = band_campaign(tmp_path)
    pub = producer_owner(tmp_path, producer, storage, monkeypatch, spool=True)
    published_groups = []
    original = pub.publish
    def publish(**kwargs):
        rows = kwargs['descriptors']
        for row in rows:
            assert Path(row['path']).is_file()
        # The record names only already durable entries, never in-flight copies.
        if any(Path(row['path']).name == 'handoff.json' for row in rows):
            assert len(published_groups) == ENTRY_GROUPS
            for ref in published_groups:
                assert all(Path(row['path']).is_file()
                    for row in read_bound_document(ref)['entries'])
        result = original(**kwargs)
        published_groups.append(result['ref'])
        return result
    monkeypatch.setattr(pub, 'publish', publish)
    emitters = []
    published, plane = emit_handoff(producer, storage, pub, emitters=emitters)
    handoff = check_consumer_binds(published, plane, consumer, pub)
    origin = check_origin_batches(pub, producer, published, handoff)
    assert {row['path'] for row in origin} == {row['path'] for row in published_groups}
    assert origin[-1] == published_groups[-1]
    report = emitters[0].export_report
    rows = report['local_spool']['exports']
    assert len(rows) == ENTRY_GROUPS + 1
    assert all(row['durable'] and row['released'] and row['publication']
        and not row['publication_error'] for row in rows)
    assert not list((tmp_path / 'local').rglob('*.pt'))
    json.dumps(report)


def test_failed_entry_copy_never_publishes_a_handoff_record(tmp_path, monkeypatch):
    producer, _, storage = band_campaign(tmp_path)
    pub = producer_owner(tmp_path, producer, storage, monkeypatch, spool=True)
    entered, release = threading.Event(), threading.Event()
    original = io_engine.publish_copy
    def copy(source, *args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(30)
            raise OSError('fixture: entry copy failed')
        return original(source, *args, **kwargs)
    monkeypatch.setattr(io_engine, 'publish_copy', copy)
    errors, emitters = [], []
    def emit():
        try:
            emit_handoff(producer, storage, pub, emitters=emitters)
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=emit)
    worker.start()
    try:
        assert entered.wait(30)
        assert not list(Path(pub.output_prefix).rglob('handoff.json'))
    finally:
        release.set(); worker.join(30)
    assert not worker.is_alive() and len(errors) == 1
    assert 'entry copy failed' in str(errors[0])
    assert not list(Path(pub.output_prefix).rglob('handoff.json'))
    assert list((tmp_path / 'local').rglob('*.pt'))
    assert emitters[0].export_report['local_spool']['refusals']
