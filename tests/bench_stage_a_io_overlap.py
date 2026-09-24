"""The Stage A chain's main-thread I/O profile and digest manifest (PQ #1128).

The suite does not collect this file (it is ``bench_*``). PrismaBuild runs it
by name, once at the pull request's base and once at its head:

    PQ_IO_BENCH_OUT=<directory> python -m pytest -q tests/bench_stage_a_io_overlap.py

Two fixtures, each written to ``<directory>``:

* ``chain.json``: ``test_stage_a_same_box_readback``'s reverse roll on the
  real owner, writer, strict reader, queue and fleet, at entries of
  ``ENTRY_SHAPE`` and with a fixed stand-in for the GPU step: ``COMPUTE_S``
  of sleep per roll, which releases the GIL as a CUDA synchronize does.
  PrismaBuild's local exporter is a thread that copies and lands each group
  it is given, so no export copy runs on the thread that polls. The profile
  is cProfile on the calling thread only, over the roll. The file carries
  the wall seconds, per-function cumulative seconds and calls on that
  thread, the export polls made on each thread, and the digest manifest:
  every written reference's name, sha256 and file size in write order, the
  final plane's payload digests, and the sha256 of every entry file left
  after settle, by path relative to the run root.
* ``capture.json``: ``test_stage_a_chain_resume``'s five-layer Stage A
  capture at stride 2 through ``run_adjoint_capture_core``: the sha256 of
  every file under its root by relative path, with JSON documents hashed
  after dropping their ``telemetry`` (timings, not bytes the run decides)
  and with the run root written as ``<run>``, so a path is compared relative
  to the run root and not to pytest's temporary directory.
  That covers every entry, checkpoint manifest, the chain state and the
  receipt. The capture runs twice into one path and says whether the two
  agree, so a difference between base and head is not run-to-run noise.
  The first run's tree is kept as ``capture-run/``, so ``compare`` can name
  the fields in which a JSON document differs. A checkpoint manifest's
  ``cotangent_sha256`` and the chain state's seal hash records that hold
  absolute entry paths, so run the base and the head with the same
  ``PQ_IO_BENCH_RUN_ROOT``: the capture then runs at that one path and those
  digests compare byte for byte.

The session id is pinned in both, so a file's bytes are comparable across
runs. Compare the two runs' manifests with ``compare`` below.
"""
from __future__ import annotations

import cProfile
import collections
import hashlib
import json
import os
from pathlib import Path
import pstats
import queue
import shutil
import threading
import time
import uuid

import pytest
import torch

import test_produced_output_spool as spool_tests
import test_stage_a_produced_boundary_chain as chain
# Autouse, and it must apply HERE too (RobTand/prismaquant#889).
from test_stage_a_produced_boundary_chain import (  # noqa: F401
    _isolated_launch_context)

pytestmark = pytest.mark.own_process

OUT_ENV = "PQ_IO_BENCH_OUT"
#: Where the capture runs; the same path at the base and the head.
RUN_ROOT_ENV = "PQ_IO_BENCH_RUN_ROOT"
GROUP_SIZE = chain.GROUP_SIZE
N_BATCHES = 4 * GROUP_SIZE
N_PROBES = 2
TOP = 4
#: 4 MiB float32 entries: I/O large enough to measure, at a quarter of R13's.
ENTRY_SHAPE = (512, 2048)
ENTRY_BYTES = ENTRY_SHAPE[0] * ENTRY_SHAPE[1] * 4
ENVELOPE = 65536
COMPUTE_S = float(os.environ.get("PQ_IO_BENCH_COMPUTE_S", "0.02"))
GROUP_CEILING_BYTES = GROUP_SIZE * (ENTRY_BYTES + ENVELOPE)
PAYLOAD_MAX_BYTES = GROUP_CEILING_BYTES * (
    TOP * N_BATCHES // GROUP_SIZE + 2 * N_PROBES * N_BATCHES // GROUP_SIZE)
#: One window (a boundary and an incoming entry per batch), the next one,
#: and one group of writes in flight.
MAX_RESIDENT_BYTES = (2 + 2 + 1) * GROUP_SIZE * ENTRY_BYTES
MAX_ARTIFACT_BYTES = 4 << 30

#: The functions each row of the profile is read from (file, name).
ROWS = {
    "owner.write": ("cost_streaming.py", "write"),
    "owner.prefetch": ("cost_streaming.py", "prefetch"),
    "owner.retire": ("cost_streaming.py", "_retire"),
    "owner.progress": ("cost_streaming.py", "_commit_local_output_progress"),
    "owner.flush_unlinks": ("cost_streaming.py", "_produced_flush_deferred_unlinks"),
    "entry.write": ("perturbed_x_cache.py", "write_exact_activation_cache_entry"),
    "entry.read": ("perturbed_x_cache.py", "_read_exact_entry"),
    "spool.durable_entries": ("produced_output_spool.py", "durable_entries"),
    "spool.submit": ("produced_output_spool.py", "submit"),
    "fsync": ("~", "<built-in method posix.fsync>"),
    "unlink": ("~", "<built-in method posix.unlink>"),
    "compute.sleep": ("~", "<built-in method time.sleep>"),
    "compute.roll": ("bench_stage_a_io_overlap.py", "_roll"),
}


def _out() -> Path:
    value = os.environ.get(OUT_ENV)
    if not value:
        pytest.skip(f"{OUT_ENV} names no output directory")
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


class ThreadedExport(spool_tests.ControlledExport):
    """PrismaBuild's local exporter: its own thread copies and lands a group."""

    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)
        self.poll_threads = collections.Counter()
        self._jobs = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="bench-exporter",
                                        daemon=True)
        self._thread.start()

    def submit_group(self, batch_id, *, entries):
        answer = super().submit_group(batch_id, entries=entries)
        self._jobs.put(batch_id)
        return answer

    def poll_group(self, batch_id):
        self.poll_threads[threading.current_thread().name] += 1
        return super().poll_group(batch_id)

    def _run(self):
        while True:
            batch_id = self._jobs.get()
            if batch_id is None:
                return
            self.acknowledge(batch_id)

    def close(self):
        self._jobs.put(None)
        self._thread.join(timeout=60)


def _pinned_session(monkeypatch):
    """Pin the generation id the owner draws at bind, and nothing else."""
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts

    bind = StreamedBoundaryArtifacts.bind

    def pinned(self, *args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=1128))
            return bind(self, *args, **kwargs)

    monkeypatch.setattr(StreamedBoundaryArtifacts, "bind", pinned)


def _boundary(layer, batch):
    base = torch.arange(ENTRY_SHAPE[1], dtype=torch.float32) * (batch + 1)
    return (torch.full(ENTRY_SHAPE, float(layer)) + base).contiguous()


def _tail(probe, batch):
    count = ENTRY_SHAPE[0] * ENTRY_SHAPE[1]
    return (torch.linspace(-1.0, 1.0, count).reshape(ENTRY_SHAPE) * (probe + 1)
            + batch).contiguous()


def _roll(incoming, boundary):
    return (incoming * 0.75 + boundary.sin()).contiguous()


def _chain(storage, written, profiler):
    """Forward and tail writes, then the profiled reverse roll."""

    def write(tensor, **kwargs):
        reference = storage.write(tensor, **kwargs)
        written.append([reference.name, reference.sha256, reference.file_bytes])
        return reference

    batches = []
    for batch in range(N_BATCHES):
        batches.append(type("Batch", (), {})())
        batches[batch].activations_cpu = [
            write(_boundary(layer, batch), batch_index=batch, boundary_index=layer)
            for layer in range(TOP)]
    grad_outs = {p: [write(_tail(p, b), batch_index=b, boundary_index=TOP,
                           probe_index=p)
                     for b in range(N_BATCHES)]
                 for p in range(N_PROBES)}
    final = {p: [None] * N_BATCHES for p in range(N_PROBES)}
    profiler.enable()
    started = time.perf_counter()
    try:
        _roll_chain(storage, batches, grad_outs, final, write)
    finally:
        wall = time.perf_counter() - started
        profiler.disable()
    return final, wall


def _roll_chain(storage, batches, grad_outs, final, write):
    from prismaquant.cost_streaming import prefetched_boundary_batches

    for layer in range(TOP - 1, -1, -1):
        for probe in range(N_PROBES):
            following = ((layer, grad_outs[probe + 1]) if probe + 1 < N_PROBES
                         else ((layer - 1, grad_outs[0]) if layer > 0 else None))
            with prefetched_boundary_batches(
                    storage, batches, layer, incoming=grad_outs[probe],
                    then=following) as windows:
                for index, _batch, boundary, incoming in windows:
                    rolled = _roll(incoming, boundary)
                    time.sleep(COMPUTE_S)
                    if layer == 0:
                        final[probe][index] = rolled.clone()
                    grad_outs[probe][index] = write(
                        rolled, batch_index=index, boundary_index=layer,
                        probe_index=probe, previous=grad_outs[probe][index],
                        **({} if layer > 0 else {"read_back": False}))


def _rows(profiler):
    stats = pstats.Stats(profiler)
    rows = {}
    for label, (suffix, name) in ROWS.items():
        cumulative, calls = 0.0, 0
        for (filename, _line, function), (_cc, nc, _tt, ct, _callers) in stats.stats.items():
            if function == name and (suffix == "~" and filename == "~"
                                     or filename.endswith(suffix)):
                cumulative += ct
                calls += nc
        rows[label] = {"cumulative_s": round(cumulative, 6), "calls": calls}
    top = sorted(stats.stats.items(), key=lambda item: -item[1][2])[:40]
    rows["_top_tottime"] = [
        {"function": f"{Path(filename).name}:{line}:{function}",
         "tottime_s": round(tt, 6), "cumulative_s": round(ct, 6), "calls": nc}
        for (filename, line, function), (_cc, nc, tt, ct, _callers) in top]
    return rows


def _files(root: Path, pattern: str):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob(pattern)) if path.is_file()}


def test_chain_profile_and_manifest(tmp_path, monkeypatch):
    from prismaquant.produced_output_spool import ProducedOutputSpool
    from prismaquant.stage_a_chain_seed import tensor_payload_sha256

    out = _out()
    _pinned_session(monkeypatch)
    storage, publication, q, env, pb_repo = chain._bound_owner(
        tmp_path, n_batches=N_BATCHES, window_gib=8, gib=16,
        payload_max_bytes=PAYLOAD_MAX_BYTES, n_probes=N_PROBES,
        staging_timeout_s=120.0, max_entry_tensor_bytes=ENTRY_BYTES,
        storage_config={"max_resident_bytes": MAX_RESIDENT_BYTES,
                        "max_artifact_bytes": MAX_ARTIFACT_BYTES})
    storage._published = True
    backend = ThreadedExport(tmp_path / "local", capacity=8 << 30)
    storage._local_output_spool = ProducedOutputSpool(
        backend, capacity_deferred=spool_tests.CapacityDeferred)
    enable = getattr(storage, "enable_io_overlap", None)
    written = []
    try:
        with chain._fleet(q, tmp_path):
            chain._strict(monkeypatch, env, pb_repo, q)
            if enable is not None:
                enable()
            profiler = cProfile.Profile()
            final, roll_wall_s = _chain(storage, written, profiler)
            settle_started = time.perf_counter()
            assert storage.drain_produced_stager(120.0)
            storage.settle_local_output()
            storage.settle_produced_releases()
            assert storage.drain_produced_stager(120.0)
            settle_s = time.perf_counter() - settle_started
    finally:
        storage._produced_stop_stager()
        backend.close()
    rows = _rows(profiler)
    compute = (rows["compute.sleep"]["cumulative_s"]
               + rows["compute.roll"]["cumulative_s"])
    record = {
        "schema": "pq1128.bench.chain.v1",
        "fixture": {"group_size": GROUP_SIZE, "n_batches": N_BATCHES,
                    "n_probes": N_PROBES, "top": TOP,
                    "entry_shape": list(ENTRY_SHAPE), "entry_bytes": ENTRY_BYTES,
                    "compute_s_per_roll": COMPUTE_S,
                    "max_resident_bytes": MAX_RESIDENT_BYTES,
                    "io_overlap": enable is not None},
        "roll_wall_s": round(roll_wall_s, 6),
        "settle_s": round(settle_s, 6),
        "main_thread_compute_s": round(compute, 6),
        "main_thread_other_share": round(1.0 - compute / roll_wall_s, 6),
        "rows": rows,
        "export_polls_by_thread": dict(backend.poll_threads),
        "telemetry": {key: value for key, value in storage.telemetry.items()
                      if isinstance(value, (int, float))},
        "manifest": {
            "written": written,
            "final_payload_sha256": {
                str(p): [tensor_payload_sha256(t) for t in final[p]]
                for p in sorted(final)},
            "entry_files": _files(tmp_path / "outputs", "*.pt"),
        },
    }
    (out / "chain.json").write_text(json.dumps(record, indent=1, sort_keys=True))


def _without_telemetry(value):
    if isinstance(value, dict):
        return {k: _without_telemetry(v) for k, v in value.items() if k != "telemetry"}
    if isinstance(value, list):
        return [_without_telemetry(v) for v in value]
    return value


def _capture_tree(root: Path):
    tree = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if path.suffix == ".json":
            try:
                document = json.loads(raw)
            except ValueError:
                pass
            else:
                raw = _relative_json(_without_telemetry(document), root).encode()
        tree[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
    return tree


def _relative_json(document, root) -> str:
    """A JSON document's text with the run root written as ``<run>``."""
    return json.dumps(document, sort_keys=True).replace(str(root), "<run>")


def test_capture_manifest(tmp_path, monkeypatch):
    import test_stage_a_chain_resume as capture

    out = _out()
    from prismaquant.staged_tier_policy import deactivate_staged_tier_policy_for_tests
    deactivate_staged_tier_policy_for_tests()
    root = Path(os.environ.get(RUN_ROOT_ENV) or tmp_path / "run")
    trees = []
    for _attempt in range(2):
        if root.exists():
            shutil.rmtree(root)
        capture._run(root, monkeypatch)
        trees.append(_capture_tree(root))
        if not (out / "capture-run").exists():
            shutil.copytree(root, out / "capture-run")
    record = {"schema": "pq1128.bench.capture.v1",
              "repeat_identical": trees[0] == trees[1],
              "root": str(root),
              "files": trees[0]}
    (out / "capture.json").write_text(json.dumps(record, indent=1, sort_keys=True))


def compare(base: Path, head: Path) -> dict:
    """Compare two runs' manifests; returns the differences, empty when equal."""
    differences = {}
    for name, keys in (("chain.json", ("written", "final_payload_sha256", "entry_files")),
                       ("capture.json", ("files",))):
        left = json.loads((base / name).read_text())
        right = json.loads((head / name).read_text())
        for key in keys:
            a = left.get("manifest", left)[key]
            b = right.get("manifest", right)[key]
            if a != b:
                differences[f"{name}:{key}"] = {"base": len(a), "head": len(b)}
    left = json.loads((base / "capture.json").read_text())
    right = json.loads((head / "capture.json").read_text())

    def document(run, record, relative):
        value = _without_telemetry(json.loads((run / "capture-run" / relative).read_text()))
        return json.loads(_relative_json(value, record["root"]))

    for relative in sorted(set(left["files"]) | set(right["files"])):
        if left["files"].get(relative) == right["files"].get(relative):
            continue
        row = differences.setdefault("capture.json:files:" + relative, {})
        if (relative.endswith(".json") and relative in left["files"]
                and relative in right["files"]):
            row["fields"] = _json_differences(document(base, left, relative),
                                              document(head, right, relative))
    return differences


def _json_differences(left, right, where=""):
    """The dotted paths at which two JSON values differ, with both values."""
    if isinstance(left, dict) and isinstance(right, dict):
        found = {}
        for key in sorted(set(left) | set(right)):
            found.update(_json_differences(left.get(key), right.get(key),
                                           f"{where}.{key}" if where else str(key)))
        return found
    if (isinstance(left, list) and isinstance(right, list)
            and len(left) == len(right)):
        found = {}
        for index, (a, b) in enumerate(zip(left, right)):
            found.update(_json_differences(a, b, f"{where}[{index}]"))
        return found
    return {} if left == right else {where: {"base": left, "head": right}}


if __name__ == "__main__":
    import sys

    print(json.dumps(compare(Path(sys.argv[1]), Path(sys.argv[2])), indent=1,
                     sort_keys=True))
