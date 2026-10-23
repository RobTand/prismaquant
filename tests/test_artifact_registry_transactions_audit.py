"""Successful registrations survive other writers and failed publications."""
from __future__ import annotations

import multiprocessing

import pytest

from prismaquant import artifact_registry
from prismaquant.artifact_registry import ArtifactRecord, ArtifactRegistry


def _record(record_id):
    return ArtifactRecord(
        record_id=record_id, model_path="/source", artifact_path=None,
        layer_config_sha="a" * 64, layer_config_path=None,
        target_bpp=8.0, achieved_bpp=8.0, format_histogram={"FP8_E4M3": 1},
        ppl_wikitext=8.0, ppl_mmlu_acc=0.5, end_kl=0.1, eval_meta={},
        created_at="2026-10-17T00:00:00+00:00",
    )


def _ids(registry):
    return [record.record_id for record in registry.all()]


def test_preloaded_registries_preserve_both_acknowledged_adds(tmp_path):
    path = tmp_path / "registry.json"
    first, second = ArtifactRegistry(path), ArtifactRegistry(path)
    first.add(_record("first"))
    second.add(_record("second"))
    assert _ids(ArtifactRegistry(path)) == ["first", "second"]
    assert _ids(second) == ["first", "second"]


def test_duplicate_committed_by_another_registry_is_refused(tmp_path):
    path = tmp_path / "registry.json"
    first, stale = ArtifactRegistry(path), ArtifactRegistry(path)
    first.add(_record("existing"))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="duplicate artifact record_id"):
        stale.add(_record("existing"))
    assert path.read_bytes() == before
    assert _ids(ArtifactRegistry(path)) == ["existing"]


def test_failed_publication_preserves_disk_and_acknowledged_memory(tmp_path, monkeypatch):
    path = tmp_path / "registry.json"
    registry = ArtifactRegistry(path)
    registry.add(_record("existing"))
    before = path.read_bytes()

    def refuse_publication(_source, _destination):
        raise OSError("publication refused")

    with monkeypatch.context() as patch:
        patch.setattr(artifact_registry.os, "replace", refuse_publication)
        with pytest.raises(OSError, match="publication refused"):
            registry.add(_record("retry"))
    assert path.read_bytes() == before
    assert _ids(registry) == ["existing"]
    assert _ids(ArtifactRegistry(path)) == ["existing"]
    registry.add(_record("retry"))
    assert _ids(ArtifactRegistry(path)) == ["existing", "retry"]


def _register_from_preloaded_process(path, record_id, barrier):
    registry = ArtifactRegistry(path)
    barrier.wait(timeout=30)
    registry.add(_record(record_id))


def test_process_writers_preserve_both_preloaded_registrations(tmp_path):
    path = tmp_path / "registry.json"
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    workers = [context.Process(target=_register_from_preloaded_process,
                               args=(path, record_id, barrier))
               for record_id in ("first", "second")]
    try:
        for worker in workers:
            worker.start()
        barrier.wait(timeout=30)
        for worker in workers:
            worker.join(timeout=30)
            assert not worker.is_alive(), "registry writer did not finish"
            assert worker.exitcode == 0
        assert set(_ids(ArtifactRegistry(path))) == {"first", "second"}
    finally:
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
