#!/usr/bin/env python3
"""Read one campaign row's capture files over NFS from a Spark and time it.

This is the consumer side of the ARC prewarm: it answers whether a row whose
bytes the server has already pulled into ARC is delivered at link speed rather
than at HDD speed. It reads the same files, in the same order
``prefetch_capture`` uses, with a chosen number of concurrent readers, and does
nothing else -- no hashing, no tensor decode -- so the number is transport and
cache, not the loader.

Run it through PrismaBuild on a Spark. Pair every run with dl380g10's nfsd and
diskstats counters over the same window: high MB/s alone cannot distinguish an
ARC hit from the client's own page cache, but nfsd read bytes ~ file bytes with
HDD reads ~ 0 can.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time

CAMPAIGN_BASE = "/mnt/shared/tessera-measurements/glm-canonical-census-20260908"


def _json(path):
    with open(path) as fh:
        return json.load(fh)


def row_files(workspace: str, row_id: str) -> list[tuple[str, int]]:
    plan = _json(os.path.join(workspace, "plan.json"))
    rows = {r["row_id"]: r for r in plan["rows"]}
    units = _json(rows[row_id]["units"])
    names = sorted(m for g in units["groups"] for m in g["members"])
    cap = plan["calibration_cache"]["path"]
    entries = _json(cap)["entries"]
    root = os.path.dirname(cap)
    out = []
    for name in names:
        rec = entries.get(name)
        if rec is None:
            continue
        path = os.path.join(root, rec["path"])
        try:
            out.append((path, os.stat(path).st_size))
        except OSError:
            out.append((path, 0))
    return out


def meminfo() -> dict:
    out = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                if k in ("MemTotal", "MemAvailable", "Cached", "Buffers"):
                    out[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return out


def nfs_client_stats() -> dict:
    """Client-side NFS byte counters, so the transport is evidenced locally too."""
    out = {}
    try:
        with open("/proc/self/mountstats") as fh:
            cur = None
            for line in fh:
                if line.startswith("device ") and " mounted on " in line:
                    cur = line.split(" mounted on ")[1].split()[0]
                elif cur == "/mnt/shared" and line.strip().startswith("bytes:"):
                    vals = [int(x) for x in line.split()[1:]]
                    out = {"normal_read": vals[0], "normal_write": vals[1],
                           "direct_read": vals[2], "direct_write": vals[3],
                           "server_read": vals[4], "server_write": vals[5]}
                    break
    except OSError:
        pass
    return out


def run(files: list[tuple[str, int]], readers: int, block: int) -> dict:
    work: "queue.Queue[str | None]" = queue.Queue()
    for p, _ in files:
        work.put(p)
    for _ in range(readers):
        work.put(None)
    totals = [0] * readers
    errors: list[str] = []
    lock = threading.Lock()

    def worker(idx: int) -> None:
        buf = bytearray(block)
        view = memoryview(buf)
        got = 0
        while True:
            path = work.get()
            if path is None:
                break
            try:
                with open(path, "rb", buffering=0) as fh:
                    while True:
                        n = fh.readinto(view)
                        if not n:
                            break
                        got += n
            except OSError as exc:
                with lock:
                    if len(errors) < 20:
                        errors.append(f"{os.path.basename(path)}: {exc}")
        totals[idx] = got

    before_nfs, before_mem = nfs_client_stats(), meminfo()
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(readers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    after_nfs, after_mem = nfs_client_stats(), meminfo()
    got = sum(totals)
    return {
        "readers": readers, "n_files": len(files), "bytes": got,
        "wall_s": wall, "mb_per_s": got / wall / 1e6 if wall else None,
        "gbit_per_s": got * 8 / wall / 1e9 if wall else None,
        "nfs_client_delta": {k: after_nfs.get(k, 0) - before_nfs.get(k, 0)
                             for k in after_nfs},
        "meminfo_before": before_mem, "meminfo_after": after_mem,
        "started_unix": t0, "finished_unix": t0 + wall, "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row", required=True)
    ap.add_argument("--workspace",
                    default=CAMPAIGN_BASE + "/first-proof-anchor-preparation-05/workspace")
    ap.add_argument("--readers", type=int, default=8)
    ap.add_argument("--block-bytes", type=int, default=1024 * 1024)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = row_files(args.workspace, args.row)
    print(f"[nfs-read] row={args.row} files={len(files)} "
          f"bytes={sum(s for _, s in files)} readers={args.readers}", flush=True)
    res = run(files, args.readers, args.block_bytes)
    res.update({"schema": "prismaquant.w3.nfs_row_read.v1", "row": args.row,
                "label": args.label, "host": os.uname().nodename})
    print(f"[nfs-read] {res['mb_per_s']:.1f} MB/s "
          f"({res['gbit_per_s']:.2f} Gbit/s) wall={res['wall_s']:.1f}s", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
    print("=== RESULT_JSON_BEGIN ===", flush=True)
    print(json.dumps(res))
    print("=== RESULT_JSON_END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
