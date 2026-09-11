#!/usr/bin/env python3
"""Measure the parallel cold-read ceiling of the dl380g10 raidz1 pool.

Runs on dl380g10 through PrismaBuild. Each arm reads a disjoint prefix of a
distinct already-priced expert row's capture files, straight off the local pool
path (``/storage_pool/shared/...``), never the NFS loopback, so the number is a
property of the pool and not of the NFS client.

Coldness is established after the fact, not asserted. The primary signal is
this process's own ``/proc/self/io`` ``read_bytes``, which counts only storage
reads this action caused and is therefore immune to whatever else the box is
doing; ``/proc/diskstats`` for the four raidz1 members is recorded alongside as
box-wide context. A warm arm shows near-zero physical reads on both. The box is
shared with other PrismaBuild work, so arms are repeated in a forward-and-back
order and the spread between the two runs of one reader count is the honest
error bar.

Nothing is written to the pool. Results go to stdout and, optionally, to
``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time

HDDS = ("sdb", "sdc", "sdd", "sde")
NVMES = ("nvme0n1", "nvme1n1")
SECTOR = 512

ARC_FIELDS = (
    "size", "c", "c_max", "hits", "misses",
    "demand_data_hits", "demand_data_misses",
    "prefetch_data_hits", "prefetch_data_misses",
    "l2_hits", "l2_misses", "l2_size", "l2_read_bytes",
    "mru_size", "mfu_size", "evict_l2_cached", "evict_l2_eligible",
)

ZFS_PARAMS = (
    "zfs_vdev_async_read_max_active", "zfs_vdev_async_read_min_active",
    "zfs_vdev_sync_read_max_active", "zfs_vdev_sync_read_min_active",
    "zfs_vdev_max_active", "zfetch_max_distance", "zfetch_max_streams",
    "zfs_prefetch_disable", "l2arc_write_max", "l2arc_write_boost",
    "l2arc_noprefetch", "zfs_arc_max", "zfs_arc_min",
)


def read_arcstats() -> dict:
    out = {}
    try:
        with open("/proc/spl/kstat/zfs/arcstats") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 3 and parts[0] in ARC_FIELDS:
                    out[parts[0]] = int(parts[2])
    except OSError as exc:
        out["error"] = str(exc)
    return out


def read_diskstats() -> dict:
    out = {}
    try:
        with open("/proc/diskstats") as fh:
            for line in fh:
                f = line.split()
                if len(f) < 14:
                    continue
                name = f[2]
                if name in HDDS or name in NVMES:
                    out[name] = {
                        "reads_completed": int(f[3]),
                        "read_bytes": int(f[5]) * SECTOR,
                        "writes_completed": int(f[7]),
                        "write_bytes": int(f[9]) * SECTOR,
                        "io_ms": int(f[12]),
                    }
    except OSError as exc:
        out["error"] = str(exc)
    return out


def loadavg() -> list[float]:
    try:
        with open("/proc/loadavg") as fh:
            return [float(x) for x in fh.read().split()[:3]]
    except (OSError, ValueError):
        return []


def concurrent_claims(host: str = "dl380g10") -> int:
    """How many other PrismaBuild actions hold this box right now.

    Only the count is kept: raw queue records are broker control material and
    are never logged.
    """
    d = "/mnt/shared/prismabuild-fleet/pb-queue/claimed"
    n = 0
    try:
        for fn in os.listdir(d):
            try:
                with open(os.path.join(d, fn)) as fh:
                    if json.load(fh).get("claimed_host") == host:
                        n += 1
            except (OSError, ValueError):
                continue
    except OSError:
        return -1
    return n


def read_self_io() -> dict:
    out = {}
    try:
        with open("/proc/self/io") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                out[k.strip()] = int(v)
    except OSError as exc:
        out["error"] = str(exc)
    return out


def zfs_params() -> dict:
    out = {}
    for name in ZFS_PARAMS:
        try:
            with open(f"/sys/module/zfs/parameters/{name}") as fh:
                out[name] = fh.read().strip()
        except OSError:
            out[name] = None
    return out


def disk_delta(before: dict, after: dict) -> dict:
    out = {"hdd_read_bytes": 0, "nvme_read_bytes": 0, "per_device": {}}
    for name in HDDS + NVMES:
        b, a = before.get(name), after.get(name)
        if not isinstance(b, dict) or not isinstance(a, dict):
            continue
        d = {k: a[k] - b[k] for k in a}
        out["per_device"][name] = d
        if name in HDDS:
            out["hdd_read_bytes"] += d["read_bytes"]
        else:
            out["nvme_read_bytes"] += d["read_bytes"]
    return out


class Sampler(threading.Thread):
    """Samples arcstats + diskstats once a second for the whole run."""

    def __init__(self, interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[dict] = []
        self._stopping = threading.Event()

    def run(self) -> None:
        while not self._stopping.is_set():
            self.samples.append({
                "t": time.time(),
                "arc": read_arcstats(),
                "disk": read_diskstats(),
            })
            self._stopping.wait(self.interval)

    def halt(self) -> None:
        self._stopping.set()


def run_arm(root: str, files: list[str], readers: int, block: int) -> dict:
    work: "queue.Queue[str | None]" = queue.Queue()
    for rel in files:
        work.put(rel)
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
            rel = work.get()
            if rel is None:
                break
            path = os.path.join(root, rel)
            try:
                with open(path, "rb", buffering=0) as fh:
                    while True:
                        n = fh.readinto(view)
                        if not n:
                            break
                        got += n
            except OSError as exc:
                with lock:
                    errors.append(f"{rel}: {exc}")
        totals[idx] = got

    io_before = read_self_io()
    arc_before = read_arcstats()
    disk_before = read_diskstats()
    load_before = loadavg()
    claims_before = concurrent_claims()
    t0 = time.time()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(readers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wall = time.time() - t0
    disk_after = read_diskstats()
    arc_after = read_arcstats()
    io_after = read_self_io()
    load_after = loadavg()
    claims_after = concurrent_claims()

    read_bytes = sum(totals)
    self_phys = io_after.get("read_bytes", 0) - io_before.get("read_bytes", 0)
    dd = disk_delta(disk_before, disk_after)
    arc_d = {k: arc_after.get(k, 0) - arc_before.get(k, 0)
             for k in ARC_FIELDS if k in arc_after and k in arc_before}
    return {
        "readers": readers,
        "n_files": len(files),
        "logical_bytes": read_bytes,
        "wall_s": wall,
        "mb_per_s": read_bytes / wall / 1e6 if wall else None,
        "self_physical_read_bytes": self_phys,
        "self_physical_over_logical": self_phys / read_bytes if read_bytes else None,
        "cold": (self_phys / read_bytes) > 0.8 if read_bytes else None,
        "box_hdd_read_bytes": dd["hdd_read_bytes"],
        "box_nvme_read_bytes": dd["nvme_read_bytes"],
        "box_hdd_over_self": (dd["hdd_read_bytes"] / self_phys) if self_phys else None,
        "per_device_delta": dd["per_device"],
        "arc_delta": arc_d,
        "arc_size_before": arc_before.get("size"),
        "arc_size_after": arc_after.get("size"),
        "proc_io_delta": {k: io_after.get(k, 0) - io_before.get(k, 0)
                          for k in ("rchar", "read_bytes") if k in io_after},
        "loadavg_before": load_before,
        "loadavg_after": load_after,
        "concurrent_claims_before": claims_before,
        "concurrent_claims_after": claims_after,
        "errors": errors,
        "started_unix": t0,
        "finished_unix": t0 + wall,
    }


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default=os.path.join(here, "glm_pool_read_ceiling_arms.json"))
    ap.add_argument("--block-bytes", type=int, default=1024 * 1024)
    ap.add_argument("--settle-s", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma-separated arm_ids to run")
    args = ap.parse_args()

    spec = json.load(open(args.arms))
    root = spec["root"]
    arms = spec["arms"]
    if args.only:
        want = set(args.only.split(","))
        arms = [a for a in arms if a["arm_id"] in want]

    if not os.path.isdir(root):
        print(f"pool root not present: {root}", file=sys.stderr)
        return 2

    result = {
        "schema": "prismaquant.w3.pool_read_ceiling_result.v1",
        "host": os.uname().nodename,
        "root": root,
        "block_bytes": args.block_bytes,
        "zfs_parameters_before": zfs_params(),
        "arcstats_before": read_arcstats(),
        "diskstats_before": read_diskstats(),
        "started_unix": time.time(),
        "arms": [],
    }

    sampler = Sampler()
    sampler.start()
    try:
        for i, arm in enumerate(arms):
            if i:
                time.sleep(args.settle_s)
            print(f"[arm] {arm['arm_id']} readers={arm['readers']} "
                  f"files={arm['n_files']} bytes={arm['bytes']}", flush=True)
            out = run_arm(root, arm["files"], arm["readers"], args.block_bytes)
            out["arm_id"] = arm["arm_id"]
            out["source_row"] = arm["source_row"]
            result["arms"].append(out)
            print(f"[arm] {arm['arm_id']} -> {out['mb_per_s']:.1f} MB/s "
                  f"wall={out['wall_s']:.1f}s self_phys={out['self_physical_read_bytes']} "
                  f"self/log={out['self_physical_over_logical']:.3f} "
                  f"box_hdd/self={out['box_hdd_over_self']:.2f} "
                  f"claims={out['concurrent_claims_before']}->"
                  f"{out['concurrent_claims_after']} "
                  f"cold={out['cold']}", flush=True)
    finally:
        sampler.halt()
        sampler.join(timeout=5)

    result["finished_unix"] = time.time()
    result["zfs_parameters_after"] = zfs_params()
    result["arcstats_after"] = read_arcstats()
    result["diskstats_after"] = read_diskstats()
    result["samples"] = sampler.samples

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
        print(f"[out] {args.out}", flush=True)

    print("=== TABLE ===", flush=True)
    print(f"{'arm':24} {'readers':>7} {'GiB':>7} {'wall_s':>8} {'MB/s':>8} "
          f"{'self/log':>9} {'boxhdd/self':>12} {'claims':>7} {'cold':>5}")
    for a in result["arms"]:
        print(f"{a['arm_id']:24} {a['readers']:7d} "
              f"{a['logical_bytes']/1024**3:7.2f} {a['wall_s']:8.1f} "
              f"{a['mb_per_s']:8.1f} {a['self_physical_over_logical']:9.3f} "
              f"{a['box_hdd_over_self']:12.2f} "
              f"{a['concurrent_claims_before']:7d} {str(a['cold']):>5}")

    compact = dict(result)
    compact.pop("samples", None)
    print("=== RESULT_JSON_BEGIN ===", flush=True)
    print(json.dumps(compact))
    print("=== RESULT_JSON_END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
