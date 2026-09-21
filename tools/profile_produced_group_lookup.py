"""Profile the produced-group reference lookup at Stage A scale.

Evidence tool, not a test. ``cost_streaming._produced_group_for`` walks
every bound group and asks ``reference in group["references"]``, so a late
window is O(groups x entries_per_group). At the production panel that is
~100k rotated cotangent entries, and the question this answers is whether
that walk is actually on the hot path or merely looks quadratic.

Profiles the SAME call twice -- once with the linear walk, once with a
reference->group index -- and prints both, so the delta is the claim
rather than the shape of the loop. Prints JSON on the last line.
"""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Ref:
    """A stand-in with the identity semantics the real reference has."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other) -> bool:
        return isinstance(other, _Ref) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


def _build(storage, *, groups: int, per_group: int):
    for index in range(groups):
        refs = [_Ref(f"g{index:05d}-e{slot:03d}") for slot in range(per_group)]
        storage._produced_groups[("boundary", index, -1, 0)] = {
            "batch_id": f"b{index:05d}", "planned": [], "references": refs,
            "published": None, "context": None, "manifest_digest": None,
            "retired": False, "origin_reclaimed": False}
        lookup = getattr(storage, "_produced_index", None)
        if lookup is not None:
            for reference in refs:
                lookup[reference] = ("boundary", index, -1, 0)
    last = storage._produced_groups[("boundary", groups - 1, -1, 0)]
    return list(last["references"])


def _indexed_lookup(storage, index, reference):
    key = index.get(reference)
    if key is None:
        return None, None
    return key, storage._produced_groups[key]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=1563)
    ap.add_argument("--per-group", type=int, default=64)
    ap.add_argument("--windows", type=int, default=20)
    args = ap.parse_args()

    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)

    storage = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA, "directory": "/dev/null/unused",
        "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
        "max_artifact_bytes": 1 << 24,
        "prefetch_batches": args.per_group})
    window = _build(storage, groups=args.groups, per_group=args.per_group)
    index = {reference: key
             for key, group in storage._produced_groups.items()
             for reference in group["references"]}

    def walk():
        for _ in range(args.windows):
            for reference in window:
                storage._produced_group_for(reference)

    def indexed():
        for _ in range(args.windows):
            for reference in window:
                _indexed_lookup(storage, index, reference)

    out = {"groups": args.groups, "per_group": args.per_group,
           "entries": args.groups * args.per_group, "windows": args.windows}
    for name, fn in (("produced_group_for", walk), ("reference_index_control", indexed)):
        fn()                                   # warm
        start = time.perf_counter()
        fn()
        wall = time.perf_counter() - start
        profiler = cProfile.Profile()
        profiler.enable()
        fn()
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(6)
        out[name] = {"wall_s_total": wall,
                     "wall_s_per_window": wall / args.windows,
                     "profile_head": stream.getvalue().splitlines()[:12]}
    out["speedup"] = (out["produced_group_for"]["wall_s_total"]
                      / out["reference_index_control"]["wall_s_total"])
    # The window this lookup sits in front of reads 64 x 16 MiB from a
    # staged tier. That is the number the lookup has to be compared with.
    out["window_payload_bytes"] = args.per_group * (16 << 20)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
