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
    real_write, real_load, real_flock = (
        shipcard.write_shipcard, shipcard.load_shipcard, fcntl.flock,
    )

    def write_card(destination, card):
        if Path(destination) == target:
            if role == "first":
                first_ready.set()
                if not release_first.wait(timeout=30):
                    raise RuntimeError("first writer was not released")
            else:
                second_boundary.set()
                if not release_second.wait(timeout=30):
                    raise RuntimeError("second writer was not released")
        return real_write(destination, card)

    def load_card(source):
        card = real_load(source)
        if role == "second" and Path(source) == target:
            second_boundary.set()
        return card

    def acquire_lock(descriptor, operation):
        if role == "second" and operation & fcntl.LOCK_EX:
            second_boundary.set()
        return real_flock(descriptor, operation)

    shipcard.write_shipcard = write_card
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
