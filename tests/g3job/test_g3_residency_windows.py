"""Phase lifetime and immutable-manifest reads of the G3 reader, over a real materialized manifest."""
import json
import threading

import pytest

import g3_residency as R
from staged_native_fixtures import bind, publish_inputs

NAMES = {"source": b"source", "wire": b"wire", "teacher": b"teacher"}
OPEN = []


def new_reader():
    """A reader the fixture releases: every window must be released before the resolver closes."""
    reader = R.StagedReader()
    OPEN.append(reader)
    return reader


@pytest.fixture
def staged(tmp_path, monkeypatch):
    from g3_pq_policy import residency_map as G3_MAPS
    files = {}
    for name, raw in NAMES.items():
        files[name] = tmp_path / name
        files[name].write_bytes(raw)
    binding = publish_inputs(tmp_path, list(files.values()))
    entries = [{"path": str(path), "offset": 0, "bytes": len(path.read_bytes()), "sha256": None}
               for path in files.values()]
    plan = {"schema": R.PLAN_SCHEMA, "mount_prefix": str(tmp_path), "produced_by": {"tool": "test"},
            "annotations": {}, "entries": entries, "entry_count": 3, "total_bytes": 17,
            "read_plan": {"phases": [
                {"name": "layer-00", "entry_indices": [0, 1], "bytes": 10, "cumulative_bytes": 10},
                {"name": "teachers", "entry_indices": [2], "bytes": 7, "cumulative_bytes": 17}],
                "read_bytes": 17}}
    plan_path = tmp_path / "read-plan.json"
    plan_path.write_text(json.dumps(plan))
    bind(monkeypatch, binding)
    monkeypatch.setenv("G3_READ_PLAN", str(plan_path))
    G3_MAPS.reset_residency_resolver_for_tests()
    R._READER = None
    yield files, binding
    try:
        R.close_reader()
        for reader in OPEN:
            reader.close()
    finally:
        OPEN.clear()
        R._READER = None
        G3_MAPS.reset_residency_resolver_for_tests()


def object_of(reader, path):
    return next(e.object_path for e in reader.resolver.manifest.extents if e.source_path == str(path))


def test_the_reader_requires_a_bound_manifest(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_STAGED_MANIFEST", raising=False)
    from g3_pq_policy import residency_map as G3_MAPS
    G3_MAPS.reset_residency_resolver_for_tests()
    with pytest.raises(RuntimeError, match="explicit staged manifest"):
        R.StagedReader()
    assert R.staged_range("/any", 0, 1) is None


def test_phase_holds_source_and_wire_until_phase_completion(staged):
    files, _ = staged
    reader = new_reader()
    assert reader.read(str(files["source"]), 0, 6) == b"source"
    assert len(reader.windows["layer-00"]) == 1 and reader.resolver._windows == 1
    assert reader.read(str(files["wire"]), 0, 4) == b"wire"
    assert len(reader.windows["layer-00"]) == 1 and reader.stats["phase_acquires"] == 1
    assert reader.stats["staged_bytes"] == 10 and reader.stats["tiers"]["stage"]["reads"] == 2
    reader.finish_phase("layer-00")
    assert reader.resolver._windows == 0 and not reader.windows["layer-00"]


def test_parallel_source_and_wire_share_one_phase_window(staged):
    files, _ = staged
    reader = new_reader()
    seen = []
    threads = [threading.Thread(target=lambda p=p, n=n: seen.append(reader.read(str(p), 0, n)))
               for p, n in ((files["source"], 6), (files["wire"], 4))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(seen) == [b"source", b"wire"]
    assert len(reader.windows["layer-00"]) == 1
    reader.finish_phase("layer-00")


def test_changed_object_bytes_raise_close_the_descriptor_and_never_read_origin(staged):
    files, _ = staged
    reader = new_reader()
    target = object_of(reader, files["source"])
    import os
    os.chmod(target, 0o644)
    with open(target, "r+b") as stream:
        stream.write(b"X")
    with pytest.raises(RuntimeError, match="differ from their own digest"):
        reader.read(str(files["source"]), 0, 6)
    window, = reader.windows["layer-00"]
    assert window["active"] == 0 and window["lease"]._fds == {}
    assert files["source"].read_bytes() == b"source"  # origin is intact and was not used
    assert reader.stats["staged_reads"] == 0


def test_a_range_that_is_not_a_declared_extent_raises(staged):
    files, _ = staged
    reader = new_reader()
    with pytest.raises(RuntimeError, match="differs from requested"):
        reader.read(str(files["source"]), 1, 2)


def test_mapped_key_in_closed_phase_still_raises(staged):
    files, _ = staged
    reader = new_reader()
    reader.finish_phase("layer-00")
    with pytest.raises(RuntimeError, match="completed phase layer-00"):
        reader.read(str(files["source"]), 0, 6)
    assert reader.stats["staged_reads"] == 0


def test_an_undeclared_path_reads_its_declared_origin(staged, tmp_path):
    reader = new_reader()
    other = tmp_path / "other"
    other.write_bytes(b"other-bytes")
    assert reader.read(str(other), 0, 11) is None
    assert R.read_g3_input(str(other)) == b"other-bytes"
    assert reader.stats["staged_reads"] == 0


def test_repeated_phase_use_acquires_each_phase_once_and_releases_it(staged):
    files, _ = staged
    reader = new_reader()
    assert reader.read(str(files["source"]), 0, 6) == b"source"
    reader.finish_phase("layer-00")
    assert reader.read(str(files["teacher"]), 0, 7) == b"teacher"
    reader.finish_phase("teachers")
    assert reader.stats["phase_acquires"] == 2 and reader.resolver._windows == 0


def test_finish_phase_refuses_open_descriptors(staged):
    files, _ = staged
    reader = new_reader()
    reader.read(str(files["source"]), 0, 6)
    window, = reader.windows["layer-00"]
    window["active"] = 1
    with pytest.raises(RuntimeError, match="still has open descriptors"):
        reader.finish_phase("layer-00")
    window["active"] = 0
    reader.finish_phase("layer-00")


def test_the_receipt_names_the_manifest(staged):
    _, binding = staged
    reader = R.staged_range(str(staged[0]["source"]), 0, 6)
    assert reader == b"source"
    receipt = R.receipt()
    assert receipt["staged_manifest"] == binding["path"] and receipt["manifest_sha256"] == binding["sha256"]
    assert receipt["staged_bytes"] == 6 and "map" not in receipt


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
