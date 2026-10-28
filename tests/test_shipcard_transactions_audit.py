from __future__ import annotations

import fcntl
import importlib.util
import multiprocessing
from pathlib import Path

import pytest


@pytest.fixture
def shipcard_fixture(tmp_path):
    fixture_path = Path.cwd() / "tests" / "test_shipcard.py"
    assert (Path.cwd() / "tools").is_dir(), "shipcard fixture requires the full checkout"
    spec = importlib.util.spec_from_file_location("_shipcard_fixture_audit", fixture_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module._artifact(tmp_path)


def _slot_record(fixture, card, slot):
    arm = slot.split(".")[-1]
    return fixture.make_record(
        slot=slot, tool="shipcard persistence audit", passed=False,
        metrics={"arm": arm, "enforce_eager": arm == "eager"},
        model_sha=card["model_sha"], detail="Recorded refusal; no serve performed",
    )


def _writer(path, operation, role, record, first_ready, second_boundary,
            release_first, release_second, results):
    from prismaquant import shipcard

    target = Path(path)
    publication_name = "_publish_shipcard" if hasattr(shipcard, "_publish_shipcard") else "write_shipcard"
    real_write, real_load, real_flock = (
        getattr(shipcard, publication_name), shipcard.load_shipcard, fcntl.flock,
    )

    def write_card(destination, card, *args):
        if Path(destination) == target:
            if role == "first":
                first_ready.set()
                if not release_first.wait(timeout=30):
                    raise RuntimeError("first writer was not released")
            else:
                second_boundary.set()
                if not release_second.wait(timeout=30):
                    raise RuntimeError("second writer was not released")
        return real_write(destination, card, *args)

    def load_card(source):
        card = real_load(source)
        if role == "second" and Path(source) == target:
            second_boundary.set()
        return card

    def acquire_lock(descriptor, operation):
        if role == "second" and operation & fcntl.LOCK_EX:
            second_boundary.set()
        return real_flock(descriptor, operation)

    setattr(shipcard, publication_name, write_card)
    shipcard.load_shipcard = load_card
    fcntl.flock = acquire_lock
    try:
        if operation == "fill":
            slot = "native_export.eager" if role == "first" else "native_export.graph"
            value = shipcard.fill_slot(target, slot, record)
        else:
            value = shipcard.open_shipcard(target.parent, build={"writer": role})
        results.put((role, "accepted", value))
    except ValueError as exc:
        results.put((role, "refused", str(exc)))
    except BaseException as exc:
        results.put((role, "error", repr(exc)))
        raise


def _race(path, operation, records):
    context = multiprocessing.get_context("spawn")
    first_ready, second_boundary = context.Event(), context.Event()
    release_first, release_second = context.Event(), context.Event()
    results = {role: context.Queue() for role in ("first", "second")}
    workers = [context.Process(
        target=_writer,
        args=(path, operation, role, records.get(role), first_ready, second_boundary,
              release_first, release_second, results[role]),
    ) for role in ("first", "second")]
    try:
        workers[0].start()
        assert first_ready.wait(timeout=30), "first writer did not reach publication"
        workers[1].start()
        assert second_boundary.wait(timeout=30), "second writer reached no read/lock boundary"
        release_first.set()
        first = results["first"].get(timeout=30)
        assert first[:2] == ("first", "accepted"), first
        release_second.set()
        second = results["second"].get(timeout=30)
        assert second[0] == "second", second
        for worker in workers:
            worker.join(timeout=10)
            assert not worker.is_alive(), "shipcard writer did not finish"
            assert worker.exitcode == 0
        return first, second
    finally:
        release_first.set()
        release_second.set()
        for worker in workers:
            if worker.pid is None:
                continue
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
            worker.close()
        for channel in results.values():
            channel.close()
            channel.join_thread()


def test_concurrent_slot_fills_preserve_both_acknowledged_records(shipcard_fixture):
    from prismaquant import shipcard

    fixture, artifact = shipcard_fixture
    card = shipcard.open_shipcard(artifact)
    path = artifact / shipcard.SHIPCARD_FILENAME
    records = {role: _slot_record(fixture, card, slot) for role, slot in (
        ("first", "native_export.eager"), ("second", "native_export.graph"),
    )}

    first, second = _race(path, "fill", records)

    assert first[1] == second[1] == "accepted"
    stored = shipcard.load_shipcard(path)
    assert stored["slots"]["native_export.eager"] == records["first"]
    assert stored["slots"]["native_export.graph"] == records["second"]


def test_concurrent_initial_open_refuses_a_second_default_writer(shipcard_fixture):
    from prismaquant import shipcard

    _fixture, artifact = shipcard_fixture
    path = artifact / shipcard.SHIPCARD_FILENAME

    first, second = _race(path, "open", {})

    assert first[1] == "accepted"
    assert second[1] == "refused", second
    assert "already exists" in second[2]
    assert shipcard.load_shipcard(path)["build"]["writer"] == "first"


def test_sequential_fills_preserve_identity_and_reserved_bytes(shipcard_fixture):
    from prismaquant import shipcard

    fixture, artifact = shipcard_fixture
    card = shipcard.open_shipcard(artifact)
    path = artifact / shipcard.SHIPCARD_FILENAME
    original_files = {entry.name for entry in artifact.iterdir()}
    original_sha = shipcard.compute_model_sha(artifact)
    original_bytes = shipcard.artifact_bytes(artifact)
    for slot in ("native_export.eager", "native_export.graph"):
        record = _slot_record(fixture, card, slot)
        shipcard.fill_slot(path, slot, record)
        assert shipcard.load_shipcard(path)["slots"][slot] == record
    assert {entry.name for entry in artifact.iterdir()} == original_files
    assert shipcard.compute_model_sha(artifact) == original_sha
    assert shipcard.artifact_bytes(artifact) == original_bytes
    assert path.stat().st_size == shipcard.SHIPCARD_RESERVED_BYTES


def test_existing_card_refuses_default_open_and_allows_explicit_overwrite(shipcard_fixture):
    from prismaquant import shipcard

    _fixture, artifact = shipcard_fixture
    shipcard.open_shipcard(artifact, build={"writer": "original"})
    path = artifact / shipcard.SHIPCARD_FILENAME
    before = path.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        shipcard.open_shipcard(artifact, build={"writer": "replacement"})
    assert path.read_bytes() == before
    shipcard.open_shipcard(artifact, build={"writer": "replacement"}, overwrite=True)
    assert shipcard.load_shipcard(path)["build"]["writer"] == "replacement"


@pytest.mark.parametrize("failure", ["file_fsync", "replace"])
def test_failed_publication_preserves_card_and_file_roster(shipcard_fixture, monkeypatch, failure):
    from prismaquant import cost_stage_checkpoint, shipcard

    fixture, artifact = shipcard_fixture
    card = shipcard.open_shipcard(artifact)
    path = artifact / shipcard.SHIPCARD_FILENAME
    before = path.read_bytes()
    before_files = {entry.name for entry in artifact.iterdir()}
    before_sha = shipcard.compute_model_sha(artifact)
    record = _slot_record(fixture, card, "native_export.eager")

    def fail(*args, **kwargs):
        raise OSError("injected publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(cost_stage_checkpoint.os,
                      "fsync" if failure == "file_fsync" else "replace", fail)
        with pytest.raises(OSError, match="injected publication failure"):
            shipcard.fill_slot(path, "native_export.eager", record)

    assert path.read_bytes() == before
    assert {entry.name for entry in artifact.iterdir()} == before_files
    assert shipcard.compute_model_sha(artifact) == before_sha
    shipcard.fill_slot(path, "native_export.eager", record)
    assert shipcard.load_shipcard(path)["slots"]["native_export.eager"] == record


def test_readers_observe_complete_old_or_new_card_at_publication(shipcard_fixture, monkeypatch):
    import os
    import stat

    from prismaquant import cost_stage_checkpoint, shipcard

    fixture, artifact = shipcard_fixture
    old = shipcard.open_shipcard(artifact)
    path = artifact / shipcard.SHIPCARD_FILENAME
    record = _slot_record(fixture, old, "native_export.eager")
    real_replace, real_fsync = os.replace, os.fsync
    sequence = []

    def fsync(descriptor):
        sequence.append("directory_fsync" if stat.S_ISDIR(os.fstat(descriptor).st_mode)
                        else "file_fsync")
        return real_fsync(descriptor)

    def replace(source, destination):
        assert Path(destination) == path
        assert shipcard.load_shipcard(path) == old
        assert sequence == ["file_fsync"]
        real_replace(source, destination)
        assert shipcard.load_shipcard(path)["slots"]["native_export.eager"] == record
        sequence.append("replace")

    monkeypatch.setattr(cost_stage_checkpoint.os, "fsync", fsync)
    monkeypatch.setattr(cost_stage_checkpoint.os, "replace", replace)
    acknowledged = shipcard.fill_slot(path, "native_export.eager", record)

    assert sequence == ["file_fsync", "replace", "directory_fsync"]
    assert shipcard.load_shipcard(path) == acknowledged
    assert path.stat().st_size == shipcard.SHIPCARD_RESERVED_BYTES


def test_failed_mutation_does_not_publish_partial_state(shipcard_fixture):
    from prismaquant import shipcard

    _fixture, artifact = shipcard_fixture
    shipcard.open_shipcard(artifact)
    path = artifact / shipcard.SHIPCARD_FILENAME
    before = path.read_bytes()

    def refuse(card):
        card["build"]["unacknowledged"] = True
        raise ValueError("refused mutation")

    with pytest.raises(ValueError, match="refused mutation"):
        shipcard.update_shipcard(path, refuse)
    assert path.read_bytes() == before


def test_optional_slot_noop_preserves_existing_bytes_and_inode(shipcard_fixture):
    from prismaquant import shipcard

    _fixture, artifact = shipcard_fixture
    shipcard.open_shipcard(artifact)
    path = artifact / shipcard.SHIPCARD_FILENAME
    slot = next(iter(shipcard.OPTIONAL_SLOTS))
    shipcard.ensure_optional_slot(path, slot)
    before, inode = path.read_bytes(), path.stat().st_ino

    assert slot in shipcard.ensure_optional_slot(path, slot)["slots"]
    assert path.read_bytes() == before
    assert path.stat().st_ino == inode


def test_directory_fsync_failure_reports_error_after_complete_replacement(shipcard_fixture, monkeypatch):
    import os
    import stat

    from prismaquant import cost_stage_checkpoint, shipcard

    fixture, artifact = shipcard_fixture
    card = shipcard.open_shipcard(artifact)
    path = artifact / shipcard.SHIPCARD_FILENAME
    before_files = {entry.name for entry in artifact.iterdir()}
    record = _slot_record(fixture, card, "native_export.eager")
    real_fsync = os.fsync

    def fsync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory durability not acknowledged")
        return real_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(cost_stage_checkpoint.os, "fsync", fsync)
        with pytest.raises(OSError, match="durability not acknowledged"):
            shipcard.fill_slot(path, "native_export.eager", record)

    assert shipcard.load_shipcard(path)["slots"]["native_export.eager"] == record
    assert {entry.name for entry in artifact.iterdir()} == before_files
    shipcard.fill_slot(path, "native_export.eager", record)
    assert path.stat().st_size == shipcard.SHIPCARD_RESERVED_BYTES


def _pause_staged_writer(path, record, staged, release):
    from prismaquant import cost_stage_checkpoint, shipcard

    real_replace = cost_stage_checkpoint.os.replace

    def replace(source, destination):
        if Path(destination) == Path(path):
            staged.set()
            if not release.wait(timeout=30):
                raise RuntimeError("staged writer was not released")
        return real_replace(source, destination)

    cost_stage_checkpoint.os.replace = replace
    shipcard.fill_slot(path, "native_export.eager", record)


def test_staging_and_killed_writer_refuse_identity_until_explicit_repair(shipcard_fixture):
    from prismaquant import shipcard

    fixture, artifact = shipcard_fixture
    old = shipcard.open_shipcard(artifact)
    path = artifact / shipcard.SHIPCARD_FILENAME
    expected_sha = shipcard.compute_model_sha(artifact)
    record = _slot_record(fixture, old, "native_export.eager")
    context = multiprocessing.get_context("spawn")
    staged, release = context.Event(), context.Event()
    worker = context.Process(target=_pause_staged_writer,
                             args=(path, record, staged, release))
    try:
        worker.start()
        assert staged.wait(timeout=30), "writer did not stage complete card bytes"
        stages = list(artifact.glob("shipcard.json.tmp*"))
        assert len(stages) == 1
        assert shipcard.load_shipcard(path) == old
        with pytest.raises(ValueError, match="shipcard publication incomplete") as active:
            shipcard.compute_model_sha(artifact)
        assert str(stages[0]) in str(active.value)

        worker.kill()
        worker.join(timeout=10)
        assert not worker.is_alive() and worker.exitcode != 0
        assert shipcard.load_shipcard(path) == old
        with pytest.raises(ValueError, match="shipcard publication incomplete"):
            shipcard.compute_model_sha(artifact)
        with pytest.raises(ValueError, match="shipcard publication incomplete"):
            shipcard.fill_slot(path, "native_export.eager", record)
        assert stages[0].is_file(), "ambiguous residue must not be deleted automatically"

        stages[0].unlink()
        assert shipcard.compute_model_sha(artifact) == expected_sha
        shipcard.fill_slot(path, "native_export.eager", record)
        assert shipcard.load_shipcard(path)["slots"]["native_export.eager"] == record
        assert shipcard.compute_model_sha(artifact) == expected_sha
        assert not list(artifact.glob("shipcard.json.tmp*"))
    finally:
        if worker.pid is not None:
            if worker.is_alive():
                worker.kill()
            worker.join(timeout=10)
            worker.close()


@pytest.mark.parametrize("relative", [
    "runtime.tmp", "shipcard.json.tmp-model-input", "nested/shipcard.json.tmphost123",
])
def test_other_auxiliary_paths_remain_content_bound(shipcard_fixture, relative):
    from prismaquant import shipcard

    _fixture, artifact = shipcard_fixture
    shipcard.open_shipcard(artifact)
    before = shipcard.compute_model_sha(artifact)
    auxiliary = artifact / relative
    auxiliary.parent.mkdir(parents=True, exist_ok=True)
    auxiliary.write_bytes(b"runtime input remains bound")
    assert shipcard.compute_model_sha(artifact) != before
