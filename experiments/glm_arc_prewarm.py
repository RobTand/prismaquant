#!/usr/bin/env python3
"""Warm the dl380g10 ARC with the capture and weight bytes the next campaign
row will read, while the current row is still on the GPU.

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
import sys
import threading
import time

CAMPAIGN_BASE = "/mnt/shared/tessera-measurements/glm-canonical-census-20260908"
POOL_BASE = "/storage_pool/shared/tessera-measurements/glm-canonical-census-20260908"
QUEUE = "/mnt/shared/prismabuild-fleet/pb-queue"
CAS_REQUESTS = "/mnt/shared/prismabuild-fleet/cas/requests"

UNITS_RE = re.compile(r"units/(row-\d{4})\.json")
ARC_PATH = "/proc/spl/kstat/zfs/arcstats"
RECORD_SIZE = 1 << 20


# ---------------------------------------------------------------- utilities


def _json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


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


_POOL_LOCAL = os.path.isdir("/storage_pool/shared")


def to_pool(path: str) -> str:
    """Map a /mnt/shared path onto the identical local ZFS mount.

    On dl380g10 both are mounts of ``storage_pool/shared``; using the pool path
    keeps the reader off the NFS loopback so the warm is a pool read. Off that
    box there is no pool mount, so the path is returned unchanged and only the
    read-free modes (``--dry-run``) are meaningful.
    """
    if not _POOL_LOCAL:
        return path
    if path.startswith(CAMPAIGN_BASE):
        return POOL_BASE + path[len(CAMPAIGN_BASE):]
    if path.startswith("/mnt/shared/"):
        return "/storage_pool/shared/" + path[len("/mnt/shared/"):]
    return path


# ------------------------------------------------------------ campaign model


class Campaign:
    """Read-only view of the campaign's plan, units and capture manifest."""

    def __init__(self, workspace: str, manifest_sha_check: bool = False) -> None:
        self.workspace = workspace
        plan = _json(os.path.join(workspace, "plan.json"))
        if plan is None:
            raise SystemExit(f"unreadable plan: {workspace}/plan.json")
        self.plan = plan
        self.rows = {r["row_id"]: r for r in plan["rows"]}
        self.model_dir = plan["model"]
        cap = plan["calibration_cache"]["path"]
        self.capture_manifest_path = cap
        manifest = _json(cap)
        if manifest is None:
            raise SystemExit(f"unreadable capture manifest: {cap}")
        self.capture_root = os.path.dirname(cap)
        self.entries = manifest["entries"]
        self._units_cache: dict[str, list[str]] = {}
        self._header_cache: dict[str, dict] = {}
        self._weight_map = None

    # -- rows ------------------------------------------------------------

    def members(self, row_id: str) -> list[str]:
        if row_id not in self._units_cache:
            row = self.rows[row_id]
            units = _json(row["units"])
            if units is None:
                raise SystemExit(f"unreadable units: {row['units']}")
            names = [m for g in units["groups"] for m in g["members"]]
            # prefetch_capture sorts before it reads; warm in the same order.
            self._units_cache[row_id] = sorted(names)
        return self._units_cache[row_id]

    def capture_files(self, row_id: str) -> list[tuple[str, int]]:
        out = []
        for name in self.members(row_id):
            rec = self.entries.get(name)
            if rec is None:
                continue
            path = os.path.join(self.capture_root, rec["path"])
            try:
                size = os.stat(to_pool(path)).st_size
            except OSError:
                size = 0
            out.append((path, size))
        return out

    # -- weights ---------------------------------------------------------

    @property
    def weight_map(self) -> dict:
        if self._weight_map is None:
            idx = _json(os.path.join(self.model_dir, "model.safetensors.index.json"))
            self._weight_map = idx["weight_map"] if idx else {}
        return self._weight_map

    def _header(self, shard: str) -> dict:
        if shard not in self._header_cache:
            path = to_pool(os.path.join(self.model_dir, shard))
            try:
                with open(path, "rb") as fh:
                    n = int.from_bytes(fh.read(8), "little")
                    head = json.loads(fh.read(n))
                head["__data_start__"] = 8 + n
            except (OSError, ValueError):
                head = {"__data_start__": 0}
            self._header_cache[shard] = head
        return self._header_cache[shard]

    def weight_extents(self, row_id: str) -> list[tuple[str, int, int]]:
        """(shard path, offset, length) covering this row's selected tensors.

        ``--source-snapshot-policy selected-tensors-v1`` reads only the named
        tensors' byte ranges via ``safe_open``
        (``layer_streaming.py`` ``_read_chunk``), so whole-shard warming would
        pull several times the bytes the row actually touches. Ranges are
        coalesced and rounded out to record boundaries because ZFS reads a
        whole 1 MiB record either way.
        """
        wm = self.weight_map
        by_shard: dict[str, list[tuple[int, int]]] = {}
        for name in self.members(row_id):
            key = name + ".weight"
            shard = wm.get(key)
            if shard is None:
                continue
            head = self._header(shard)
            meta = head.get(key)
            if not isinstance(meta, dict) or "data_offsets" not in meta:
                continue
            a, b = meta["data_offsets"]
            base = head["__data_start__"]
            by_shard.setdefault(shard, []).append((base + a, base + b))
        out = []
        for shard in sorted(by_shard):
            spans = sorted(by_shard[shard])
            merged: list[list[int]] = []
            for a, b in spans:
                a -= a % RECORD_SIZE
                b += (-b) % RECORD_SIZE
                if merged and a <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            path = os.path.join(self.model_dir, shard)
            for a, b in merged:
                out.append((path, a, b - a))
        return out

    def seed_reads(self, row_id: str) -> list[tuple[str, int]]:
        """Seed-checkpoint reads, when the row's argv declares one.

        The 16 remaining rows carry no ``--seed-checkpoint``, so this is empty
        for them; it is kept because a re-planned campaign may reinstate it.
        """
        seed = self.rows[row_id].get("seed") or {}
        ckpt = seed.get("checkpoint")
        if not ckpt:
            return []
        out = []
        for path in (ckpt, ckpt + ".parts"):
            p = to_pool(path)
            if os.path.isfile(p):
                out.append((path, os.stat(p).st_size))
            elif os.path.isdir(p):
                for entry in sorted(os.listdir(p)):
                    f = os.path.join(p, entry)
                    if os.path.isfile(f):
                        out.append((os.path.join(path, entry), os.stat(f).st_size))
        return out

    def row_plan(self, row_id: str) -> dict:
        caps = self.capture_files(row_id)
        ext = self.weight_extents(row_id)
        seeds = self.seed_reads(row_id)
        return {
            "row_id": row_id,
            "group": self.rows[row_id]["groups"][0],
            "capture_files": len(caps),
            "capture_bytes": sum(s for _, s in caps),
            "weight_extents": len(ext),
            "weight_bytes": sum(n for _, _, n in ext),
            "seed_files": len(seeds),
            "seed_bytes": sum(s for _, s in seeds),
            "total_bytes": sum(s for _, s in caps) + sum(n for _, _, n in ext)
                           + sum(s for _, s in seeds),
            "_captures": caps,
            "_extents": ext,
            "_seeds": seeds,
        }


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
    jobs: list[tuple[str, int, int]] = [(p, 0, 0) for p, _ in plan["_captures"]]
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
        planned += plan["capture_bytes"] + plan["seed_bytes"]
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
    sys.exit(main())
