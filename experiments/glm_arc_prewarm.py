#!/usr/bin/env python3
"""Warm the dl380g10 ARC with the capture and weight bytes the next campaign
row will read, while the current row is still on the GPU.

Status: SUPERSEDED as a daemon
------------------------------
The warming *loop* now lives in PrismaBuild as the fleet's storage role,
``tools/fleet/prewarm_loop.py`` (issue #487, PR #494), which reads the same
data manifests this file's ``glm_data_manifests.py`` sibling emits, out of the
queue's own claim order, on the box that holds the pool.  It belongs there:
prewarm is a property of the fleet's storage, not of one campaign, and a
second dispatcher predicting the same claim order is exactly the duplication
the fleet's own rules forbid.

What stays useful here is everything the loop does not do: the pool-ceiling
and NFS read rigs that measured the tiers and the dry-run queue rig. Campaign
readset construction belongs to ``prismaquant.campaign_data_manifests``.
``--daemon`` is kept so the measurements remain
reproducible and so the running dry-run unit on dl380g10 can be read back; it
is not the thing to deploy.  Retirement of that unit is step 1 of the W3
switchover runbook (``SWITCHOVER.md``, kept with the W3 evidence outside this
repo).

Why this exists
---------------
Every row of the GLM-5.3-Flash Tessera census opens its 864 capture files and
its layer's weight tensors exactly once, at row start, and the campaign's whole
working set (1.894 TiB of captures) is far larger than the server's ARC. Caching
after the fact therefore never helps: a byte is read once and never asked for
again. The only read that can be made fast is one that was already resident
before the row started, so the lever is to spend the current row's ~1.8 hours of
GPU time pulling the *next* row's bytes off the HDDs.

What it does
------------
1. Reads PrismaBuild's ready queue and reproduces its claim order
   (``-priority``, ``-passes``, ``published_unix``; see
   ``prismabuild/pool.py:2895-2901``), keeping only items that resolve to a row of
   this campaign.
2. Resolves each action key to a row id from the sealed CAS request's argv
   (``--units .../units/row-XXXX.json``), falling back to a roster file. It
   never writes to the queue and never logs raw queue or CAS records.
3. Derives the row's read set: the 864 capture files named by the campaign's
   capture manifest, in the order ``prefetch_capture`` consumes them (sorted
   member name), plus the byte ranges of the layer's weight tensors inside the
   safetensors shards, coalesced to record boundaries.
4. Reads them off the local pool path with bounded parallelism, stopping when
   ARC headroom runs out and holding off while a claimed row is still in its
   own load phase.

Placement stays PrismaBuild's: this predicts *which rows run next*, not which
box gets them, so the warmed set is correct whichever Spark claims first.

Run it on dl380g10 as rob::

    systemd-run --user --unit glm-arc-prewarm \\
      /home/rob/venvs/pq-cpu312/bin/python experiments/glm_arc_prewarm.py --daemon
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import stat as statmod
import sys
import threading
import time

QUEUE = "/mnt/shared/prismabuild-fleet/pb-queue"
CAS_REQUESTS = "/mnt/shared/prismabuild-fleet/cas/requests"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prismaquant.campaign_data_manifests import (
    UNITS_RE, RECORD_SIZE, Campaign, ReadsetRefused,
    read_optional_campaign_json as _json,
)

#: This rig's default campaign, and the pool it measures reads from. Host
#: facts of the dl380g10 storage server, owned here rather than by the
#: package's readset producer, which reads the paths a manifest declares.
CAMPAIGN_BASE = "/mnt/shared/tessera-measurements/glm-canonical-census-20260908"
SHARED_ROOT = "/mnt/shared/"
POOL_ROOT = "/storage_pool/shared/"
_POOL_LOCAL = os.path.isdir(POOL_ROOT)


def to_pool(path: str) -> str:
    """Read a shared path off the storage server's local pool when this host is it."""
    if _POOL_LOCAL and path.startswith(SHARED_ROOT):
        return POOL_ROOT + path[len(SHARED_ROOT):]
    return path

ARC_PATH = "/proc/spl/kstat/zfs/arcstats"


# ---------------------------------------------------------------- utilities




def arcstats() -> dict:
    out = {}
    try:
        with open(ARC_PATH) as fh:
            for line in fh:
                f = line.split()
                if len(f) == 3:
                    try:
                        out[f[0]] = int(f[2])
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def arc_headroom_bytes(reserve_fraction: float) -> tuple[int, int, int, int]:
    """Usable prewarm budget, plus the two headroom numbers it sits between.

    ``c_max - size`` is the optimistic bound: the room ARC is *allowed* to grow
    into. ``c - size`` is the instantaneous one: room ARC currently *intends* to
    hold, and it is the number that decides whether a warm evicts something.
    ZFS raises ``c`` toward ``c_max`` under read demand, so budgeting on the
    optimistic bound is right, but ``c`` is volatile on a shared box -- it fell
    99 GB inside one 5-minute measurement window on 2026-09-11 -- so both are
    reported and the caller logs them.
    """
    a = arcstats()
    size, c, c_max = a.get("size", 0), a.get("c", 0), a.get("c_max", 0)
    return max(0, int((c_max - size) * reserve_fraction)), size, c, c_max






#: The two campaign flags that name bytes a seeded row reads before it prices
#: anything.  They are read off the row's argv because that is the only record
#: that always carries them; see ``Campaign.seed_reads``.






# ------------------------------------------------------------ campaign model




# ----------------------------------------------------------- queue reading


def _passes(action_key: str) -> int:
    rec = _json(os.path.join(QUEUE, "passes", f"{action_key}.json"))
    if isinstance(rec, dict):
        v = rec.get("passes", 0)
        return int(v) if isinstance(v, (int, float)) else 0
    return 0


def _row_of_key(action_key: str, roster: dict | None) -> str | None:
    """Resolve one action key to a campaign row id.

    The sealed CAS request carries the row in its argv, which survives a
    replanned campaign; the roster is only a fallback.
    """
    req = _json(os.path.join(CAS_REQUESTS, action_key[:2], f"{action_key}.json"))
    if isinstance(req, dict):
        found = UNITS_RE.findall(json.dumps(req.get("task", {})))
        if len(set(found)) == 1:
            return found[0]
    if roster:
        rec = roster.get(action_key)
        if isinstance(rec, dict):
            return rec.get("row_id")
    return None


def ready_rows(campaign: Campaign, roster: dict | None) -> list[dict]:
    """Ready items in PrismaBuild's own claim order, filtered to campaign rows."""
    items = []
    ready = os.path.join(QUEUE, "ready")
    try:
        names = os.listdir(ready)
    except OSError:
        return []
    for fn in names:
        if not fn.endswith(".json"):
            continue
        rec = _json(os.path.join(ready, fn))
        if not isinstance(rec, dict):
            continue
        key = str(rec.get("action_key", ""))
        row = _row_of_key(key, roster)
        if row is None or row not in campaign.rows:
            continue
        items.append({
            "action_key": key,
            "row_id": row,
            "priority": int(rec.get("priority", 0)),
            "passes": _passes(key),
            "published_unix": float(rec.get("published_unix", 0.0)),
        })
    items.sort(key=lambda r: (-r["priority"], -r["passes"], r["published_unix"]))
    return items


def claimed_rows(campaign: Campaign, roster: dict | None) -> list[dict]:
    out = []
    claimed = os.path.join(QUEUE, "claimed")
    try:
        names = os.listdir(claimed)
    except OSError:
        return []
    for fn in names:
        if not fn.endswith(".json"):
            continue
        rec = _json(os.path.join(claimed, fn))
        if not isinstance(rec, dict):
            continue
        key = str(rec.get("action_key", ""))
        row = _row_of_key(key, roster)
        if row is None or row not in campaign.rows:
            continue
        out.append({
            "action_key": key,
            "row_id": row,
            "host": rec.get("claimed_host"),
            "claimed_unix": float(rec.get("claimed_unix", 0.0)),
        })
    return out


def gpu_hosts() -> list[str]:
    out = []
    wd = os.path.join(QUEUE, "workers")
    try:
        names = sorted(os.listdir(wd))
    except OSError:
        return []
    for fn in names:
        rec = _json(os.path.join(wd, fn))
        if isinstance(rec, dict) and rec.get("has_gpu"):
            out.append(str(rec.get("host") or fn[:-5]))
    return out


def load_phase_done(campaign: Campaign, row_id: str) -> bool:
    """Has the claimed row finished reading its captures?

    The row writes an immutable ``capture-load-execution-<sha>.json`` beside its
    cache when the verified load completes; until then its capture files are
    still being consumed and must not be evicted for the next row.
    """
    cache = to_pool(os.path.join(campaign.rows[row_id]["dir"], "cache"))
    try:
        for entry in os.listdir(cache):
            if entry.startswith("capture-load-execution-"):
                return True
    except OSError:
        pass
    return False


# ------------------------------------------------------------------ reading


class Reader:
    def __init__(self, readers: int, block: int = RECORD_SIZE) -> None:
        self.readers = readers
        self.block = block

    def read(self, jobs: list[tuple[str, int, int]], stop: threading.Event,
             budget_bytes: int) -> dict:
        """Read ``(path, offset, length)`` jobs; length 0 means whole file."""
        work: "queue.Queue[tuple[str, int, int] | None]" = queue.Queue()
        for j in jobs:
            work.put(j)
        for _ in range(self.readers):
            work.put(None)
        totals = [0] * self.readers
        errors: list[str] = []
        lock = threading.Lock()
        spent = [0]

        def worker(idx: int) -> None:
            buf = bytearray(self.block)
            view = memoryview(buf)
            got = 0
            while not stop.is_set():
                job = work.get()
                if job is None:
                    break
                path, off, length = job
                with lock:
                    if spent[0] >= budget_bytes:
                        break
                try:
                    with open(to_pool(path), "rb", buffering=0) as fh:
                        if off:
                            fh.seek(off)
                        remaining = length if length else None
                        while not stop.is_set():
                            want = self.block if remaining is None \
                                else min(self.block, remaining)
                            if want <= 0:
                                break
                            n = fh.readinto(view[:want])
                            if not n:
                                break
                            got += n
                            if remaining is not None:
                                remaining -= n
                            with lock:
                                spent[0] += n
                except OSError as exc:
                    with lock:
                        if len(errors) < 20:
                            errors.append(f"{os.path.basename(path)}: {exc}")
            totals[idx] = got

        t0 = time.time()
        threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(self.readers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
        done = sum(totals)
        return {"bytes": done, "wall_s": wall,
                "mb_per_s": done / wall / 1e6 if wall else None,
                "errors": errors}


# -------------------------------------------------------------------- main


def jobs_for(plan: dict, include_weights: bool) -> list[tuple[str, int, int]]:
    jobs: list[tuple[str, int, int]] = [(record["path"], 0, 0) for record in plan.get("_acquisition", [])]
    jobs += [(p, 0, 0) for p, _ in plan["_captures"]]
    jobs += [(p, 0, 0) for p, _ in plan["_seeds"]]
    if include_weights:
        jobs += list(plan["_extents"])
    return jobs


def predict(campaign: Campaign, roster, window: int, simulate: list[str] | None):
    hosts = gpu_hosts()
    if simulate is not None:
        ready = [{"action_key": "(simulated)", "row_id": r, "priority": 0,
                  "passes": 0, "published_unix": float(i)}
                 for i, r in enumerate(simulate)]
    else:
        ready = ready_rows(campaign, roster)
    claimed = claimed_rows(campaign, roster)
    busy = {c["row_id"] for c in claimed}
    horizon = max(1, window) * max(1, len(hosts))
    upcoming = [r for r in ready if r["row_id"] not in busy][:horizon]
    return hosts, claimed, ready, upcoming


def log_event(log_path: str | None, event: dict) -> None:
    event["t"] = time.time()
    line = json.dumps(event)
    print(line, flush=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace",
                    default=CAMPAIGN_BASE + "/first-proof-anchor-preparation-05/workspace")
    ap.add_argument("--roster", default=None, help="optional action-key -> row_id map")
    ap.add_argument("--window", type=int, default=1,
                    help="upcoming rows to hold warm per GPU host")
    ap.add_argument("--readers", type=int, default=8,
                    help="concurrent pool readers (from the Part A ceiling)")
    ap.add_argument("--arc-reserve-fraction", type=float, default=0.85,
                    help="fraction of measured ARC headroom this may consume")
    ap.add_argument("--load-grace-s", type=float, default=900.0,
                    help="assume a claimed row's load finished after this long "
                         "when no capture-load receipt has appeared")
    ap.add_argument("--no-weights", action="store_true",
                    help="warm captures only, not the layer's weight ranges")
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--log", default="/home/rob/tmp/glm-perf-20260910/w3/arc-prewarm.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--simulate-ready", default=None,
                    help="comma-separated row ids to treat as the ready queue")
    ap.add_argument("--warm-row", default=None,
                    help="warm exactly this row once and exit (acceptance warm)")
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    campaign = Campaign(args.workspace)
    roster = _json(args.roster) if args.roster else None
    simulate = args.simulate_ready.split(",") if args.simulate_ready else None

    if args.warm_row:
        plan = campaign.row_plan(args.warm_row)
        jobs = jobs_for(plan, not args.no_weights)
        planned = sum(n for _, _, n in plan["_extents"]) if not args.no_weights else 0
        planned += plan["capture_bytes"] + plan["seed_bytes"] + plan.get("acquisition_bytes", 0)
        head, size, c, c_max = arc_headroom_bytes(args.arc_reserve_fraction)
        log_event(args.log, {"event": "warm_start", "row": args.warm_row,
                             "bytes": planned, "row_total_bytes": plan["total_bytes"],
                             "weights": not args.no_weights, "readers": args.readers,
                             "arc_size": size, "arc_c": c, "arc_c_max": c_max,
                             "arc_headroom_nominal": c_max - size,
                             "arc_headroom_effective": c - size,
                             "arc_headroom_usable": head})
        if args.dry_run:
            print(json.dumps({k: v for k, v in plan.items()
                              if not k.startswith("_")}, indent=1))
            return 0
        stop = threading.Event()
        res = Reader(args.readers).read(jobs, stop, planned + (1 << 30))
        head2, size2, c2, _ = arc_headroom_bytes(args.arc_reserve_fraction)
        log_event(args.log, {"event": "warm_done", "row": args.warm_row,
                             "arc_size_after": size2, "arc_c_after": c2,
                             "arc_size_delta": size2 - size, **res})
        return 0

    def cycle() -> None:
        hosts, claimed, ready, upcoming = predict(campaign, roster, args.window, simulate)
        head, size, c, c_max = arc_headroom_bytes(args.arc_reserve_fraction)
        plans = [campaign.row_plan(r["row_id"]) for r in upcoming]
        # A dry run prices the whole predicted order, not just the horizon, so
        # the cost of every row the queue would hand us is visible at once.
        horizon_ids = {r["row_id"] for r in upcoming}
        all_plans = ([campaign.row_plan(r["row_id"]) for r in ready]
                     if args.dry_run else plans)
        summary = {
            "event": "predict",
            "gpu_hosts": hosts,
            "claimed": [{"row": c["row_id"], "host": c["host"],
                         "load_done": load_phase_done(campaign, c["row_id"])}
                        for c in claimed],
            "ready_campaign_rows": [r["row_id"] for r in ready],
            "upcoming": [{k: v for k, v in p.items() if not k.startswith("_")}
                         for p in plans],
            "upcoming_total_bytes": sum(p["total_bytes"] for p in plans),
            "predicted_order": [
                dict({k: v for k, v in p.items() if not k.startswith("_")},
                     rank=i, in_horizon=p["row_id"] in horizon_ids)
                for i, p in enumerate(all_plans)],
            "predicted_order_total_bytes": sum(p["total_bytes"] for p in all_plans),
            "arc_size": size, "arc_c": c, "arc_c_max": c_max,
            "arc_headroom_nominal": c_max - size,
            "arc_headroom_effective": c - size,
            "arc_headroom_usable": head,
        }
        log_event(args.log, summary)
        if args.dry_run or not plans:
            return
        # Hold off while any claimed row is still reading its own captures.
        now = time.time()
        for c in claimed:
            if (not load_phase_done(campaign, c["row_id"])
                    and now - c["claimed_unix"] < args.load_grace_s):
                log_event(args.log, {"event": "hold",
                                     "reason": "claimed row still loading",
                                     "row": c["row_id"], "host": c["host"]})
                return
        if head <= 0:
            log_event(args.log, {"event": "hold", "reason": "no ARC headroom",
                                 "arc_size": size, "arc_c": c, "arc_c_max": c_max})
            return
        stop = threading.Event()
        budget = head
        for plan in plans:
            if budget <= 0:
                log_event(args.log, {"event": "budget_exhausted",
                                     "row": plan["row_id"]})
                break
            res = Reader(args.readers).read(jobs_for(plan, not args.no_weights),
                                            stop, budget)
            budget -= res["bytes"]
            log_event(args.log, {"event": "warmed", "row": plan["row_id"],
                                 "planned_bytes": plan["total_bytes"], **res})

    if args.daemon:
        log_event(args.log, {"event": "start", "pid": os.getpid(),
                             "dry_run": args.dry_run, "window": args.window,
                             "readers": args.readers})
        try:
            while True:
                cycle()
                time.sleep(args.poll_s)
        except KeyboardInterrupt:
            log_event(args.log, {"event": "stop"})
        return 0

    cycle()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ReadsetRefused as refusal:
        sys.exit(str(refusal))
