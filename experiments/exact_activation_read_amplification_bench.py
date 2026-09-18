"""Read-amplification bench for exact boundary activation entries (#735).

Writes and then prefetches a realistic batch of 16 MiB entries through the
importable tree's own ``perturbed_x_cache``, with production's
``release_file_pages=True`` on both legs, and reports per-entry syscall read
volume (``rchar``), storage read volume (``read_bytes``), wall clock, and the
Python-heap peak of one prefetch. Run it once per arm, never in one process:
the two arms are two different checkouts.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import resource
import shutil
import sys
import time
import tracemalloc
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prismaquant.perturbed_x_cache import (  # noqa: E402
    prefetch_exact_activation_cache_entries, write_exact_activation_cache_entry,
)

SESSION = "pq735-bench"

try:  # the branch lets the window owner keep one read buffer; origin/main has none
    from prismaquant.perturbed_x_cache import EntryReadScratch
    SCRATCH = {"scratch": EntryReadScratch()}
except ImportError:
    SCRATCH = {}


def _io():
    fields = {}
    with open("/proc/self/io", "rb") as handle:
        for line in handle:
            key, _, value = line.decode().partition(":")
            fields[key.strip()] = int(value)
    return fields


def _delta(before, after):
    return {key: after[key] - before[key] for key in
            ("rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes")}


def _mount(path):
    target = os.path.realpath(path)
    best = None
    with open("/proc/mounts") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 3 and (target == parts[1] or target.startswith(parts[1].rstrip("/") + "/")):
                if best is None or len(parts[1]) > len(best[1]):
                    best = parts
    return {"device": best[0], "mountpoint": best[1], "fstype": best[2]} if best else {}


def _identity(slot):
    return {"session": SESSION, "slot": slot, "kind": "boundary",
            "coordinates": {"batch": 0, "boundary": 0, "probe": None}}


def _write_one(directory, index, tensor):
    nbytes = tensor.numel() * tensor.element_size()
    return write_exact_activation_cache_entry(
        directory, f"boundary-{index}", tensor, identity=_identity(f"slot-{index}"),
        max_tensor_bytes=nbytes, max_file_bytes=nbytes + 65536)


def _prefetch_one(ref):
    return _prefetch_window([ref])


def _prefetch_window(refs):
    budget = sum(ref.tensor_bytes for ref in refs)
    with prefetch_exact_activation_cache_entries(
            refs, max_tensor_bytes=budget, expected_session=SESSION, **SCRATCH) as window:
        return sum(int(window.get(ref).numel()) for ref in refs)


def run(entries, rows, base):
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(root) / f"run-{os.getpid()}"
    directory.mkdir()
    result = {"entries": entries, "filesystem": _mount(root), "shared_scratch": bool(SCRATCH),
              "python": sys.version.split()[0], "torch": torch.__version__,
              "host": os.uname().nodename}
    try:
        tensors = [torch.randn(rows, 1024) for _ in range(entries)]
        result["entry_tensor_bytes"] = tensors[0].numel() * tensors[0].element_size()
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # --- write leg -------------------------------------------------
        before, start = _io(), time.perf_counter()
        refs = [_write_one(directory, index, tensor) for index, tensor in enumerate(tensors)]
        result["write"] = _delta(before, _io())
        result["write"]["wall_s"] = time.perf_counter() - start
        result["entry_file_bytes"] = refs[0].file_bytes
        del tensors
        # --- prefetch leg ----------------------------------------------
        before, start = _io(), time.perf_counter()
        for ref in refs:
            _prefetch_one(ref)
        result["prefetch"] = _delta(before, _io())
        result["prefetch"]["wall_s"] = time.perf_counter() - start
        result["ru_maxrss_kb_delta"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0
        result["shared_scratch_bytes"] = (
            len(SCRATCH["scratch"].buffer(0)) if SCRATCH else 0)
        # --- prefetch leg again, in production-shaped multi-entry windows ---
        before, start = _io(), time.perf_counter()
        for index in range(0, len(refs), 4):
            _prefetch_window(refs[index:index + 4])
        result["prefetch_window4"] = _delta(before, _io())
        result["prefetch_window4"]["wall_s"] = time.perf_counter() - start
        # --- single-entry windows again, to separate order from window size --
        before, start = _io(), time.perf_counter()
        for ref in refs:
            _prefetch_one(ref)
        result["prefetch_repeat"] = _delta(before, _io())
        result["prefetch_repeat"]["wall_s"] = time.perf_counter() - start
        # --- one prefetch under tracemalloc (python-heap peak) ---------
        tracemalloc.start()
        _prefetch_one(refs[0])
        result["prefetch_python_heap_peak_bytes"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        # --- attribution pass ------------------------------------------
        profiler = cProfile.Profile()
        profiler.enable()
        for ref in refs[: min(8, len(refs))]:
            _prefetch_one(ref)
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(12)
        result["prefetch_cprofile_tottime"] = stream.getvalue()
        profiler = cProfile.Profile()
        extra = [torch.randn(rows, 1024) for _ in range(min(8, entries))]
        profiler.enable()
        for index, tensor in enumerate(extra):
            _write_one(directory, 10_000 + index, tensor)
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(12)
        result["write_cprofile_tottime"] = stream.getvalue()
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    for leg in ("write", "prefetch", "prefetch_window4", "prefetch_repeat"):
        for key in ("rchar", "read_bytes", "write_bytes"):
            result[leg][key + "_per_entry"] = result[leg][key] / entries
        result[leg]["ms_per_entry"] = 1000 * result[leg]["wall_s"] / entries
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", type=int, default=64)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--base", default="/home/rob/tmp/pq735-bench")
    parser.add_argument("--label", default="arm")
    args = parser.parse_args()
    result = run(args.entries, args.rows, args.base)
    result["label"] = args.label
    result["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print("BENCH_JSON_BEGIN")
    print(json.dumps(result, indent=1))
    print("BENCH_JSON_END")


if __name__ == "__main__":
    main()
