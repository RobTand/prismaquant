"""Selected allocations sharing original wires must retain distinct manifests."""
import hashlib
import json

import pytest

from prismaquant import tessera_export_lane as lane


def projection(tmp_path, label):
    wire = tmp_path / f'{label}.wire'
    raw = label.encode()
    wire.write_bytes(raw)
    return {'source': {'fixture': 'one original source'}, 'wire_dir': str(tmp_path),
            'units': {label: {'file': wire.name, 'blob_sha256': hashlib.sha256(raw).hexdigest(),
                              'blob_bytes': len(raw), 'identity': {'fixture': label}}}}


def test_distinct_allocations_share_wires_without_replacing_selected_manifest(tmp_path):
    first = lane.write_cached_expert_units(projection(tmp_path, 'first'))
    first_bytes = first.read_bytes()
    second = lane.write_cached_expert_units(projection(tmp_path, 'second'))
    assert first != second
    assert first.read_bytes() == first_bytes
    assert hashlib.sha256(first_bytes).hexdigest() in first.name
    assert set(json.loads(first_bytes)['units']) == {'first'}
    assert set(json.loads(second.read_bytes())['units']) == {'second'}
    assert (tmp_path / 'first.wire').read_bytes() == b'first'
    assert (tmp_path / 'second.wire').read_bytes() == b'second'


def test_identical_manifest_reuse_keeps_published_inode(tmp_path):
    selected = projection(tmp_path, 'first')
    first = lane.write_cached_expert_units(selected)
    inode = first.stat().st_ino
    assert lane.write_cached_expert_units(selected) == first
    assert first.stat().st_ino == inode


def test_existing_content_address_with_foreign_bytes_refuses(tmp_path):
    selected = projection(tmp_path, 'first')
    first = lane.write_cached_expert_units(selected)
    first.write_bytes(b'foreign manifest')
    with pytest.raises(lane.TesseraExportLaneError, match='manifest'):
        lane.write_cached_expert_units(selected)
    assert first.read_bytes() == b'foreign manifest'


@pytest.mark.parametrize('matching', [False, True])
def test_concurrent_publication_reuses_only_identical_bytes(tmp_path, monkeypatch, matching):
    from prismaquant import cluster_campaign
    selected = projection(tmp_path, 'first')
    link = cluster_campaign.os.link
    raced = []
    def publish_competitor(source, destination):
        raw = source.read_bytes() if matching else b'competing manifest'
        destination.write_bytes(raw)
        raced.append((destination, raw))
        return link(source, destination)
    monkeypatch.setattr(cluster_campaign.os, 'link', publish_competitor)
    if matching:
        path = lane.write_cached_expert_units(selected)
        assert path == raced[0][0]
    else:
        with pytest.raises(lane.TesseraExportLaneError, match='manifest'):
            lane.write_cached_expert_units(selected)
    assert len(raced) == 1
    assert raced[0][0].read_bytes() == raced[0][1]
    assert not list(tmp_path.glob('*.tmp'))


def test_existing_manifest_symlink_is_refused(tmp_path):
    selected = projection(tmp_path, 'first')
    path = lane.write_cached_expert_units(selected)
    original = path.read_bytes()
    target = tmp_path / 'other.json'
    target.write_bytes(original)
    path.unlink()
    path.symlink_to(target.name)
    with pytest.raises(lane.TesseraExportLaneError, match='manifest'):
        lane.write_cached_expert_units(selected)
    assert path.is_symlink()
    assert target.read_bytes() == original
