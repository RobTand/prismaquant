#!/usr/bin/env python3
"""Submit the quantum cache-peak row to PrismaBuild (PQ #2463).

Seals a container spec with a dedicated cache root and a separate
workspace root, both on writable identity mounts, then submits
``tools.measure_container_cache_quantum_row`` through
``tools.tessera_campaign_container`` on a gb10 worker.

Usage: ``python3 tools/submit_quantum_cache_row.py --row r1 --submit``.
Without ``--submit`` it prints the sealed spec and the pbrun argv
(the D38 dry run).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CACHE_GIB = {"r1": 2, "r2": 1}
WORKSPACE_GIB = 8
IMAGE = "sha256:c0e532d28a78b3bf425bbbc0d862e2840ba624249162aedfadd09748a6c68c37"
CONTENT_SHA256 = (
    "d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026")


def sealed_spec(row: str) -> dict:
    cache_gib = CACHE_GIB[row]
    return {
        "container": {
            "image": IMAGE,
            "content_sha256": CONTENT_SHA256,
            "mounts": [
                {"source": "/mnt/shared", "target": "/mnt/shared"},
                {"source": f"/home/rob/pb-scratch-pq2463q/{row}/cache",
                 "target": f"/home/rob/pb-scratch-pq2463q/{row}/cache"},
                {"source": f"/home/rob/pb-scratch-pq2463q/{row}/spill",
                 "target": f"/home/rob/pb-scratch-pq2463q/{row}/spill"},
            ],
        },
        "cpu_memory_gb": 32,
        "env": {
            "OMP_NUM_THREADS": "1",
            "MIMALLOC_PURGE_DELAY": "0",
            "PRISMAQUANT_RELEASE_SOURCE_PAGES": "1",
            "PRISMAQUANT_DEV_MODE": "1",
            "PRISMAQUANT_DETERMINISTIC": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "/workspace",
            "PRISMAQUANT_CONTAINER_CACHE_ROOT":
                f"/home/rob/pb-scratch-pq2463q/{row}/cache",
            "PRISMAQUANT_CONTAINER_CACHE_MAX_BYTES":
                str(cache_gib * (1 << 30)),
            "PRISMAQUANT_TMPDIR":
                f"/home/rob/pb-scratch-pq2463q/{row}/spill/row-tmp",
            "PRISMAQUANT_STAGE_B_SPILL_ROOT":
                f"/home/rob/pb-scratch-pq2463q/{row}/spill",
            "PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES":
                str(WORKSPACE_GIB * (1 << 30)),
        },
    }


def pbrun_argv(row: str, spec: dict) -> list[str]:
    from prismaquant.digests import DIRECT_ASCII_SPACED_LAX

    cache = f"/home/rob/pb-scratch-pq2463q/{row}/cache"
    spill = f"/home/rob/pb-scratch-pq2463q/{row}/spill/row-tmp"
    payload = [
        "python3", "-m", "tools.run_with_scratch_dirs",
        "--dir", f"/home/rob/pb-scratch-pq2463q/{row}/cache",
        "--dir", f"/home/rob/pb-scratch-pq2463q/{row}/spill",
        "--",
        "python3", "-m", "tools.tessera_campaign_container",
        "--spec", DIRECT_ASCII_SPACED_LAX.text(spec),
        "--",
        "python3", "-m", "tools.measure_container_cache_quantum_row",
        "--cache-root", cache,
        "--workspace", spill,
        "--out", spill + "/receipt.json",
        "--profile-out", spill + "/profile.txt",
        "--device", "cuda",
    ]
    env = {
        "PRISMAQUANT_CONTAINER_CACHE_ROOT": cache,
        "PRISMAQUANT_CONTAINER_CACHE_MAX_BYTES":
            str(CACHE_GIB[row] * (1 << 30)),
        "PRISMAQUANT_STAGE_B_SPILL_ROOT":
            f"/home/rob/pb-scratch-pq2463q/{row}/spill",
        "PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES":
            str(WORKSPACE_GIB * (1 << 30)),
        "PRISMABUILD_LOCAL_SCRATCH_PAIRS": (
            "PRISMAQUANT_CONTAINER_CACHE_ROOT:"
            "PRISMAQUANT_CONTAINER_CACHE_MAX_BYTES,"
            "PRISMAQUANT_STAGE_B_SPILL_ROOT:"
            "PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES"),
    }
    argv = [
        sys.executable, "/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py",
        "--cwd", str(Path(__file__).resolve().parents[1]),
        "--tag", "gb10",
        "--demand", "gpu=1,mem_gb=32", "--gpu-memory-gb", "20",
        "--cpus", "4",
        "--priority", "0",
        "--timeout-s", "1500",
        "--container-image", IMAGE,
    ]
    for key, value in env.items():
        argv += ["--env", f"{key}={value}"]
    return argv + ["--detach", "--", *payload]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", choices=("r1", "r2"), required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    spec = sealed_spec(args.row)
    from tools.tessera_campaign_container import (
        local_scratch_environment,
        validate_container,
    )

    validate_container(spec)
    scratch = local_scratch_environment(spec, spec["env"])
    print(json.dumps({"spec": spec, "scratch": scratch,
                      "pairs": scratch.get("PRISMABUILD_LOCAL_SCRATCH_PAIRS")},
                     indent=1, sort_keys=True))
    print("CACHE_GIB:", CACHE_GIB[args.row], "WORKSPACE_GIB:", WORKSPACE_GIB)
    if not args.submit:
        return 0
    import subprocess

    done = subprocess.run(pbrun_argv(args.row, spec), capture_output=True,
                          text=True, timeout=120)
    print(done.stdout)
    print(done.stderr, file=sys.stderr)
    return done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
