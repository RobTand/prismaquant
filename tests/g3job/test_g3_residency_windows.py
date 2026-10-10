"""Phase lifetime and atomic map refresh regressions for the G3 reader."""
import hashlib
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import g3_residency as R


def reader_fixture(tmp_path):
    paths = {}
    entries = {}
    for name, raw in (("source", b"source"), ("wire", b"wire"), ("teacher", b"teacher")):
        path = tmp_path / name
        path.write_bytes(raw)
        key = f"0:/host/{name}"
        paths[key] = path
        entries[key] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    mapping = {"tier_id": "stage", "manifest_sha256": "a" * 64,
               "generation": 1, "entries": entries}
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps(mapping))
    events, live, descriptors = [], {}, []
    parses = []

    def read_residency_map(path):
        parses.append(path)
        return json.loads(Path(path).read_bytes())

    def covers(_root, _owner, keys, **kwargs):
        events.append(("covers", set(keys)))
        return {"ok": True, "covers": []}

    def acquire(_ctx, **kwargs):
        token = kwargs["acquire_token"]
        live[token] = kwargs["expected"]
        events.append(("acquire", set(kwargs["expected"])))
        return {"ok": True, "pin": token, "pin_id": token, "ref_id": token}

    def pinned(_queue, pin, ref, key, **kwargs):
        assert key in live[pin]
        fd = os.open(paths[key], os.O_RDONLY)
        descriptors.append(fd)
        return fd, {"tier_id": "stage"}

    def release(_queue, pin, ref, **kwargs):
        for fd in descriptors:
            with pytest.raises(OSError):
                os.fstat(fd)
        del live[pin]
        events.append(("release", pin))
        return True

    reader = R.StagedReader.__new__(R.StagedReader)
    reader.maps = SimpleNamespace(residency_map_key=lambda p, o: f"{o}:{p}", read_residency_map=read_residency_map)
    reader.lease = SimpleNamespace(covers_for_keys=covers, acquire_for=acquire,
                                   open_pinned=pinned, release=release)
    reader.ctx = {"map_path": str(map_path), "action_key": "a" * 64}
    reader.queue, reader.root, reader.lock = None, tmp_path, threading.RLock()
    reader.stats = {"staged_bytes": 0, "staged_reads": 0, "read_s": 0., "tiers": {}}
    reader.mapping = None
    reader.map_identity = None
    reader.windows = {}
    reader.closed_phases = set()
    reader.phase_keys = {"layer-00": ["0:/host/source", "0:/host/wire"],
                         "teachers": ["0:/host/teacher"]}
    reader.key_phases = {key: phase for phase, keys in reader.phase_keys.items() for key in keys}
    return reader, mapping, map_path, parses, events, live


def test_phase_holds_source_and_wire_until_phase_completion(tmp_path):
    reader, _, _, parses, events, live = reader_fixture(tmp_path)
    assert reader.read("/host/source", 0, 6) == b"source"
    assert len(live) == 1  # PB cannot reclaim the remaining wire between tensor reads.
    assert reader.read("/host/wire", 0, 4) == b"wire"
    assert [keys for kind, keys in events if kind == "acquire"] == [{"0:/host/source", "0:/host/wire"}]
    assert len(parses) == 1
    reader.finish_phase("layer-00")
    assert not live
    with pytest.raises(RuntimeError, match="completed phase"):
        reader.read("/host/wire", 0, 4)
    assert reader.read("/host/teacher", 0, 7) == b"teacher"
    reader.close()
    assert not live


def test_atomic_map_refresh_preserves_live_phase_and_sees_new_ram_epoch(tmp_path):
    reader, mapping, map_path, parses, _, live = reader_fixture(tmp_path)
    assert reader.read("/host/source", 0, 6) == b"source"
    reader.finish_phase("layer-00")
    # RAM overlays can change without an increase in the fragment count.
    mapping.update(ram_tier_id="ram:stage", ram_epoch="new")
    mapping["entries"]["0:/host/teacher"]["ram_path"] = "/ram/teacher"
    replacement = tmp_path / "new-map"
    replacement.write_text(json.dumps(mapping))
    replacement.replace(map_path)
    assert reader.read("/host/teacher", 0, 7) == b"teacher"
    assert reader.mapping["ram_epoch"] == "new"
    assert len(parses) == 2
    reader.close()
    assert not live


def test_phase_integrity_error_closes_descriptor_without_origin_fallback(tmp_path):
    reader, mapping, map_path, _, _, live = reader_fixture(tmp_path)
    mapping["entries"]["0:/host/wire"]["sha256"] = "0" * 64
    map_path.write_text(json.dumps(mapping))
    with pytest.raises(RuntimeError, match="own digest"):
        reader.read("/host/wire", 0, 4)
    reader.close()
    assert not live
    assert reader.stats["staged_reads"] == 0


def test_parallel_source_and_wire_share_one_phase_pin(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    reader, _, _, parses, events, live = reader_fixture(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(reader.read, path, 0, size)
                   for path, size in (("/host/source", 6), ("/host/wire", 4))]
        assert [f.result() for f in futures] == [b"source", b"wire"]
    assert len(live) == len(parses) == 1
    assert len([event for event in events if event[0] == "covers"]) == 1
    reader.finish_phase("layer-00")
    assert not live


def test_real_comparison_progress_requires_both_complete_passes():
    from g3_progress_launch import ComparisonProgress, comparison_phases
    progress = ComparisonProgress()
    for phase in comparison_phases():
        progress.observe("G3_COMPARE_PROGRESS " + json.dumps({"phase": phase}))
        if phase.endswith("-teachers"):
            assert not progress.complete
            progress.observe("G3_COMPARE_PROGRESS " + json.dumps({"phase": phase, "scored": True}))
    assert progress.complete
    with pytest.raises(ValueError, match="repeated window"):
        progress.observe('G3_COMPARE_PROGRESS {"phase": "omission-teachers", "scored": true}')


def test_real_comparison_progress_refuses_missing_layers_and_early_score():
    from g3_progress_launch import ComparisonProgress
    progress = ComparisonProgress()
    with pytest.raises(ValueError, match="phase sequence"):
        progress.observe('G3_COMPARE_PROGRESS {"phase": "source-read-layer-01"}')
    with pytest.raises(ValueError, match="unrequested"):
        progress.observe('G3_COMPARE_PROGRESS {"phase": "source-read-teachers", "scored": true}')


def test_unpublished_ram_phase_uses_one_verified_stage_window(tmp_path):
    reader, mapping, map_path, _, events, live = reader_fixture(tmp_path)
    mapping.update(ram_tier_id="ram:stage", ram_epoch="current")
    for key in reader.phase_keys["layer-00"]:
        mapping["entries"][key]["ram_path"] = "/ram/" + key.rsplit("/", 1)[-1]
    map_path.write_text(json.dumps(mapping))
    stage_covers = reader.lease.covers_for_keys
    tiers = []

    def covers(root, owner, keys, **kwargs):
        tiers.append(kwargs["tier_id"])
        if kwargs["tier_id"] == "ram:stage":
            return {"ok": False, "refusal": "unpublished"}
        return stage_covers(root, owner, keys, **kwargs)

    reader.lease.covers_for_keys = covers
    assert reader.read("/host/source", 0, 6) == b"source"
    assert reader.read("/host/wire", 0, 4) == b"wire"
    assert tiers == ["ram:stage", "stage"]
    assert [keys for kind, keys in events if kind == "acquire"] == [{"0:/host/source", "0:/host/wire"}]
    assert reader.stats["tiers"] == {"stage": {"reads": 2, "bytes": 10}}
    reader.finish_phase("layer-00")
    assert not live


@pytest.mark.parametrize("refusal", ["source-coverage-gap", "ownership-uncertain", "unknown"])
def test_ram_integrity_or_unknown_refusal_never_selects_stage(tmp_path, refusal):
    reader, mapping, map_path, _, _, live = reader_fixture(tmp_path)
    mapping.update(ram_tier_id="ram:stage", ram_epoch="current")
    mapping["entries"]["0:/host/source"]["ram_path"] = "/ram/source"
    map_path.write_text(json.dumps(mapping))
    tiers = []

    def covers(_root, _owner, _keys, **kwargs):
        tiers.append(kwargs["tier_id"])
        return {"ok": False, "refusal": refusal}

    reader.lease.covers_for_keys = covers
    with pytest.raises(RuntimeError, match=refusal):
        reader.read("/host/source", 0, 6)
    assert tiers == ["ram:stage"]
    assert not live and reader.stats["staged_reads"] == 0


def test_unmapped_path_after_setup_close_reads_origin(tmp_path):
    reader, _, _, _, _, live = reader_fixture(tmp_path)
    reader.finish_phase("setup")
    assert reader.read("/host/other", 0, 9) is None
    assert not live and reader.stats["staged_reads"] == 0


def test_mapped_key_in_closed_phase_still_raises(tmp_path):
    reader, _, _, _, _, live = reader_fixture(tmp_path)
    reader.finish_phase("layer-00")
    with pytest.raises(RuntimeError, match="0:/host/source.*completed phase"):
        reader.read("/host/source", 0, 6)
    assert not live and reader.stats["staged_reads"] == 0
