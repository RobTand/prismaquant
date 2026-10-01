"""Capture identity hashes its source files together on the IO engine (PQ #1887).

One reader at a time hashed the 599 GB GLM-5.3-Flash source at ~140 MB/s over
NFS-RDMA, about 71 min with the row's GPU idle. The files now go out together
on ``io_engine.ENGINE``. These tests pin that they overlap, and that nothing
else moved: the digests, the per-file guard and its refusals, and which error
the serial loop raised first.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import threading
import time
from pathlib import Path

import pytest
import torch

from prismaquant import memory_management as mm
from prismaquant import tessera_calibration_cache as cc
from prismaquant import io_engine

SHARDS = ("model-00001.safetensors", "model-00002.safetensors",
          "model-00003.safetensors", "model-00004.safetensors")
GiB = 1024**3


def _canonical():
    version = importlib.metadata.version("transformers")
    return dict(model_load_contract=dict(schema="prismaquant.pretrained_initialization.v1",
                                         scope="checkpoint_missing_state", status="completed",
                                         transformers_version=version),
                attention_implementation="eager",
                capture_runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                     transformers=version))


def _source(tmp_path, *, producer_source=None, shard_bytes=64 * 1024):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    for index, name in enumerate(SHARDS):
        (source / name).write_bytes(bytes([index + 1]) * shard_bytes)
    census = dict(model=str(source), unit_shapes={"a": [3, 2]}, **_canonical())
    if producer_source is not None:
        census["expert_projection"] = {"producer": {"source": producer_source}}
    path = tmp_path / "census.json"
    path.write_text(json.dumps(census))
    return source, path


def _identity(census_path, **kwargs):
    fields = _canonical()
    return cc.capture_identity(census_path, calibration={"fit_ids_sha256": "draw"},
                               max_act_rows=2,
                               model_load_contract=fields["model_load_contract"],
                               attention_implementation="eager", **kwargs)


def _plain(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def engine(monkeypatch):
    """A fresh IO engine one worker per source file, whatever this runner's CPU set.

    Width is the engine's to decide in production (the CPU affinity). A fixed
    width here keeps the overlap proof the same on a 2-CPU row and a 20-CPU
    one; ``capture_identity`` reads ``io_engine.ENGINE`` when it is called.
    """
    fresh = io_engine.IOEngine()
    fresh.width = len(SHARDS) + 1
    monkeypatch.setattr(io_engine, "ENGINE", fresh)
    yield fresh
    if fresh._pool is not None:
        fresh._pool.shutdown(wait=True)


def test_capture_identity_hashes_every_source_file_together_on_the_io_engine(
        tmp_path, monkeypatch, engine):
    """All five files are in flight at once, on engine threads, with the guard's arguments."""
    source, census = _source(tmp_path)
    parties = len(SHARDS) + 1
    assert engine.width == parties
    barrier = threading.Barrier(parties, timeout=20)
    seen = []
    real = cc.sha256
    check = lambda label: None  # noqa: E731

    def overlapping(path, **kwargs):
        seen.append((Path(path).name, threading.current_thread().name, kwargs))
        barrier.wait()  # a serial loop breaks here: the second file never starts
        return real(path, **kwargs)

    monkeypatch.setattr(cc, "sha256", overlapping)
    identity = _identity(census, resource_check=check, release_read_pages=True)
    assert sorted(name for name, _, _ in seen) == sorted(["config.json", *SHARDS])
    assert all(thread.startswith("pq-io") for _, thread, _ in seen)
    assert all(kwargs == dict(resource_check=check, release_read_pages=True)
               for _, _, kwargs in seen)
    assert identity["source_files"] == {p.name: _plain(p) for p in sorted(source.iterdir())}


def test_capture_identity_digests_are_the_serial_full_file_digests(tmp_path):
    """Same digest per file as the serial guarded hash and as hashlib over every byte."""
    source, census = _source(tmp_path, shard_bytes=17 * 1024**2 + 3)
    events = []
    identity = _identity(census, resource_check=events.append)
    names = ["config.json", *SHARDS]
    assert identity["source_files"] == {
        name: cc.sha256(source / name) for name in names}
    assert identity["source_files"] == {name: _plain(source / name) for name in names}
    # The per-file guard still ran on every block of every file.
    for name in SHARDS:
        assert events.count(f"before_capture_hash:{name}") == 3
        assert events.count(f"after_capture_hash:{name}") == 2


def test_the_first_error_raised_is_the_serial_loops_and_no_hash_outlives_it(tmp_path, monkeypatch):
    """Two files fail; the later-named one fails first. The earlier name is raised."""
    _, census = _source(tmp_path)
    real = cc.sha256
    active = []
    lock = threading.Lock()

    def failing(path, **kwargs):
        name = Path(path).name
        with lock:
            active.append(name)
        try:
            if name == SHARDS[3]:
                raise RuntimeError(f"bad {name}")
            if name == SHARDS[1]:
                time.sleep(0.3)
                raise RuntimeError(f"bad {name}")
            time.sleep(0.1)
            return real(path, **kwargs)
        finally:
            with lock:
                active.remove(name)

    monkeypatch.setattr(cc, "sha256", failing)
    with pytest.raises(RuntimeError, match=f"^bad {SHARDS[1]}$"):
        _identity(census)
    assert active == []


def test_a_file_changed_while_it_is_hashed_still_refuses(tmp_path):
    """The identity fence holds on an engine thread exactly as on the main thread."""
    source, census = _source(tmp_path)
    target = source / SHARDS[2]
    changed = []

    def change(label):
        if label == f"after_capture_hash:{SHARDS[2]}" and not changed:
            with target.open("r+b") as handle:
                handle.write(b"x")
            changed.append(threading.current_thread().name)

    with pytest.raises(RuntimeError, match="source changed during guarded capture hashing"):
        _identity(census, resource_check=change)
    assert changed and changed[0].startswith("pq-io")


def test_a_resource_refusal_on_an_engine_thread_propagates(tmp_path):
    _, census = _source(tmp_path)

    def refuse(label):
        if label.startswith("after_capture_hash:"):
            raise RuntimeError("physical hash refusal")

    with pytest.raises(RuntimeError, match="physical hash refusal"):
        _identity(census, resource_check=refuse)


def test_producer_files_outside_the_glob_keep_the_serial_check_order(tmp_path):
    """A mismatch earlier in producer order is raised before a later unreadable file."""
    template = b"{{ chat }}"
    good = hashlib.sha256(template).hexdigest()
    producer = {"files": {SHARDS[0]: "0" * 64},
                "auxiliary_sha256": {"chat_template.jinja": good, "absent.jinja": "1" * 64}}
    source, census = _source(tmp_path, producer_source=producer)
    (source / "chat_template.jinja").write_bytes(template)
    with pytest.raises(RuntimeError, match=f"differs from census producer: {SHARDS[0]}$"):
        _identity(census)

    # With the shard right, the out-of-glob template is hashed and accepted,
    # and the missing file then refuses as the serial loop's open did.
    producer["files"][SHARDS[0]] = _plain(source / SHARDS[0])
    census.write_text(json.dumps({**json.loads(census.read_text()),
                                  "expert_projection": {"producer": {"source": producer}}}))
    with pytest.raises(FileNotFoundError, match="absent.jinja"):
        _identity(census)
    del producer["auxiliary_sha256"]["absent.jinja"]
    census.write_text(json.dumps({**json.loads(census.read_text()),
                                  "expert_projection": {"producer": {"source": producer}}}))
    identity = _identity(census)
    assert "chat_template.jinja" not in identity["source_files"]  # glob unchanged


def _cgroup(tmp_path):
    root = tmp_path / "cgroup"
    scope = root / "scope"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text(str(21 * GiB))
    (scope / "memory.current").write_text(str(5 * GiB))
    (scope / "memory.stat").write_text(
        "anon 0\nfile 0\nshmem 0\nfile_dirty 0\nfile_writeback 0\n")
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/scope\n")
    return {"cgroup_root": root, "membership": membership}


def test_the_capture_guard_check_is_one_at_a_time_across_threads(tmp_path, monkeypatch):
    """The guard is the hashes' ``resource_check``; its readings must not interleave."""
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 0)
    monkeypatch.setattr(mm, "_host_memory_info", lambda: (100 * GiB, 121 * GiB))
    guard = mm.CaptureMemoryGuard("cuda", **_cgroup(tmp_path))
    observe = guard._observe
    inside, overlap = [0], []
    lock = threading.Lock()

    def slow_observe():
        with lock:
            inside[0] += 1
            overlap.append(inside[0])
        time.sleep(0.02)
        try:
            return observe()
        finally:
            with lock:
                inside[0] -= 1

    guard._observe = slow_observe
    threads = [threading.Thread(target=guard.check, args=(f"after_capture_hash:f{i}",))
               for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max(overlap) == 1
    assert guard.failure is None and guard.baseline is not None
    assert guard.peak_by_checkpoint_prefix == {"after_capture_hash": guard.peak_bytes}
