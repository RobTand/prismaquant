"""Opt-in CPU profile and exact digest manifests through existing native IO owners.

Run only inside admitted work with PQ_IO_BENCH_OUT naming an owned output
directory. Profiles describe this fixture; no fleet or production speed claim
follows. Capture manifests retain the scientific repeatability comparison.
"""
from __future__ import annotations

import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import shutil
import time
import uuid

import pytest
import torch

from prismaquant.cost_stage_checkpoint import publish_new_bytes

import test_stage_a_produced_boundary_chain as chain

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
GROUP_CEILING_BYTES = GROUP_SIZE * (ENTRY_BYTES + ENVELOPE)
PAYLOAD_MAX_BYTES = GROUP_CEILING_BYTES * (
    TOP * N_BATCHES // GROUP_SIZE + 2 * N_PROBES * N_BATCHES // GROUP_SIZE)
#: One window (a boundary and an incoming entry per batch), the next one,
#: and one group of writes in flight.
MAX_RESIDENT_BYTES = (2 + 2 + 1) * GROUP_SIZE * ENTRY_BYTES
MAX_ARTIFACT_BYTES = 4 << 30

#: The functions each row of the profile is read from (file, name).
ROWS = {
    "owner.write": ("streamed_boundary_artifacts.py", "write"),
    "owner.prefetch": ("streamed_boundary_artifacts.py", "prefetch"),
    "owner.retire": ("streamed_boundary_artifacts.py", "_retire"),
    "owner.progress": ("streamed_boundary_artifacts.py", "_commit_local_output_progress"),
    "owner.flush_unlinks": ("streamed_boundary_artifacts.py", "_produced_flush_deferred_unlinks"),
    "entry.write": ("perturbed_x_cache.py", "write_exact_activation_cache_entry"),
    "entry.read": ("perturbed_x_cache.py", "_read_exact_entry"),
    "spool.durable_entries": ("produced_output_spool.py", "durable_entries"),
    "spool.submit": ("produced_output_spool.py", "submit"),
    "fsync": ("~", "<built-in method posix.fsync>"),
    "unlink": ("~", "<built-in method posix.unlink>"),
    "compute.roll": ("bench_stage_a_io_overlap.py", "_roll"),
}


def _out(*, run_root: Path | None = None) -> Path:
    value = os.environ.get(OUT_ENV)
    if not value:
        raise ValueError(f"{OUT_ENV} names no output directory")
    path = Path(value)
    if run_root is not None and path.resolve().is_relative_to(run_root.resolve()):
        raise ValueError(f"{OUT_ENV} must be outside {RUN_ROOT_ENV}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pinned_session(monkeypatch):
    """Pin the generation id the owner draws at bind, and nothing else."""
    from prismaquant.streamed_boundary_artifacts import StreamedBoundaryArtifacts

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
    from prismaquant.streamed_boundary_artifacts import prefetched_boundary_batches

    for layer in range(TOP - 1, -1, -1):
        for probe in range(N_PROBES):
            following = ((layer, grad_outs[probe + 1]) if probe + 1 < N_PROBES
                         else ((layer - 1, grad_outs[0]) if layer > 0 else None))
            with prefetched_boundary_batches(
                    storage, batches, layer, incoming=grad_outs[probe],
                    then=following) as windows:
                for index, _batch, boundary, incoming in windows:
                    rolled = _roll(incoming, boundary)
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
    from produced_native_fixtures import publication, selected_publications
    from prismaquant.streamed_boundary_artifacts import StreamedBoundaryArtifacts, BOUNDARY_STORAGE_SCHEMA
    from prismaquant.stage_a_chain_seed import tensor_payload_sha256

    out = _out()
    _pinned_session(monkeypatch)
    pub = publication(tmp_path, payload_max=PAYLOAD_MAX_BYTES, group_size=GROUP_SIZE,
        tensor_bytes=ENTRY_BYTES, spool_bytes=1 << 30)
    register = selected_publications(monkeypatch)
    monkeypatch.setattr(pub, "finalize", lambda *a, **k: register(pub))
    storage = StreamedBoundaryArtifacts({"schema": BOUNDARY_STORAGE_SCHEMA,
        "directory": str(Path(pub.output_prefix) / "boundaries"),
        "max_resident_bytes": MAX_RESIDENT_BYTES, "max_auxiliary_bytes": 1 << 20,
        "max_artifact_bytes": MAX_ARTIFACT_BYTES, "prefetch_batches": GROUP_SIZE})
    storage.bind({"fixture": "native-io-profile"}, n_probes=N_PROBES, published=True)
    storage.bind_produced_output(pub, group_size=GROUP_SIZE, n_batches=N_BATCHES,
        max_entry_tensor_bytes=ENTRY_BYTES)
    written = []
    with storage:
        profiler = cProfile.Profile()
        final, roll_wall_s = _chain(storage, written, profiler)
        settle_started = time.perf_counter()
        storage.settle_local_output()
        settle_s = time.perf_counter() - settle_started
    rows = _rows(profiler)
    compute = rows["compute.roll"]["cumulative_s"]
    record = {
        "schema": "pq1128.bench.chain.v1",
        "fixture": {"group_size": GROUP_SIZE, "n_batches": N_BATCHES,
                    "n_probes": N_PROBES, "top": TOP,
                    "entry_shape": list(ENTRY_SHAPE), "entry_bytes": ENTRY_BYTES,
                    "max_resident_bytes": MAX_RESIDENT_BYTES,
                    "io_overlap": True},
        "roll_wall_s": round(roll_wall_s, 6),
        "settle_s": round(settle_s, 6),
        "main_thread_compute_s": round(compute, 6),
        "main_thread_other_share": round(1.0 - compute / roll_wall_s, 6),
        "rows": rows,
        "telemetry": {key: value for key, value in storage.telemetry.items()
                      if isinstance(value, (int, float))},
        "manifest": {
            "written": written,
            "final_payload_sha256": {
                str(p): [tensor_payload_sha256(t) for t in final[p]]
                for p in sorted(final)},
            "entry_files": _files(Path(pub.output_prefix), "*.pt"),
        },
    }
    (out / "chain.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    # This chain exercises every owner operation in the table. A stale
    # filename must fail instead of publishing zero calls for real work.
    for label in ("owner.write", "owner.prefetch", "entry.write", "entry.read"):
        assert rows[label]["calls"] > 0, f"profile missed {label}"


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

    root = Path(os.environ.get(RUN_ROOT_ENV) or tmp_path / "run")
    out = _out(run_root=root)
    report = out / "capture.json"
    if os.path.lexists(report):
        raise FileExistsError(f"capture report already exists: {report}")
    retained = out / "capture-run"
    # The output product is also the exclusive claim, including when captures
    # use different run roots. An interrupted claim remains refused evidence.
    retained.mkdir()
    created_output = retained.lstat()
    output_identity = (created_output.st_dev, created_output.st_ino)

    def require_owned_output():
        try:
            current = retained.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError("benchmark capture output changed after creation") from exc
        if (current.st_dev, current.st_ino) != output_identity:
            raise RuntimeError("benchmark capture output changed after creation")

    from prismaquant.staged_tier_policy import deactivate_staged_tier_policy_for_tests
    deactivate_staged_tier_policy_for_tests()
    root.mkdir(parents=True, exist_ok=False)
    created = root.lstat()
    root_identity = (created.st_dev, created.st_ino)

    def require_owned_root():
        try:
            current = root.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError("benchmark run root changed after creation") from exc
        if (current.st_dev, current.st_ino) != root_identity:
            raise RuntimeError("benchmark run root changed after creation")

    try:
        trees = []
        for attempt in range(2):
            require_owned_root()
            if attempt:
                for child in root.iterdir():
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            capture._run(root, monkeypatch)
            require_owned_root()
            require_owned_output()
            trees.append(_capture_tree(root))
            if not attempt:
                shutil.copytree(root, retained, dirs_exist_ok=True)
        record = {"schema": "pq1128.bench.capture.v1",
                  "repeat_identical": trees[0] == trees[1],
                  "root": str(root),
                  "files": trees[0]}
        require_owned_output()
        if not publish_new_bytes(report, json.dumps(record, indent=1, sort_keys=True).encode()):
            raise FileExistsError(f"capture report already exists: {report}")
    finally:
        require_owned_root()
        shutil.rmtree(root)


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
