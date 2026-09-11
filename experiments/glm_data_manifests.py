#!/usr/bin/env python3
"""Compute a per-row data manifest for each row of a pbcampaign manifest.

Why this exists
---------------
PrismaBuild issue #487 gives an action a second content-addressed input, a
*data manifest*: the exact files and byte ranges the action will read off the
shared mount, so a storage-role fleet loop can pull them into the dl380g10 ARC
before the row is claimed. W3 measured the payoff -- an ARC-resident row reads
at 3298.7 MB/s against 306.9 MB/s cold, 10.75x -- but only if the *right* bytes
are resident, and only the producer knows which those are.

This tool is the producer side. It reuses the read-set expansion that the W3
prewarm daemon already validated (``glm_arc_prewarm.Campaign``): the 864
capture files a row's members name, in the order ``prefetch_capture`` consumes
them (sorted member name), plus the coalesced byte ranges of those members'
weight tensors inside the safetensors shards. It writes one manifest per row
and a NEW pbcampaign manifest carrying ``data_manifest`` per row. The input
manifest is never rewritten.

Exact identity, not globs
-------------------------
Every entry names one file and, for a weight shard, one byte range. Nothing is
expanded at consume time, so the prewarm loop reads exactly what the producer
priced and a reviewer can diff the manifest against the row.

Per-file ``sha256`` is deliberately ``null``. Hashing this campaign's 1023 GB of
capture bytes would cost more than the prewarm saves, and the contract does not
need it: the *manifest file* is content-addressed in the PrismaBuild CAS, which
is what binds the action key to this byte list. The manifest is a residency
hint, not an integrity claim about the data, and it says so in its own schema.

Capture sizes come from a cached inventory (``--sizes-cache``, one
``<basename> <bytes>`` line per file) so that generating 16 manifests does not
stat 36,423 files over NFS. Pass ``--stat`` to stat instead.

Usage::

    python3 experiments/glm_data_manifests.py \
      --campaign-manifest BASE/transition-optimization-20260910/resume-overlap-manifest.json \
      --out-dir          BASE/transition-optimization-20260910/data-manifests \
      --out-manifest     BASE/transition-optimization-20260910/resume-overlap-manifest.with-data.json \
      --sizes-cache      /home/rob/tmp/glm-perf-20260910/w3/evidence/capture-file-sizes.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from glm_arc_prewarm import CAMPAIGN_BASE, Campaign, UNITS_RE  # noqa: E402

SCHEMA = "prismaquant.prismabuild.data_manifest.v1"
SHARED_MOUNT = "/mnt/shared"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(tree: str) -> str:
    try:
        out = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class CachedSizeCampaign(Campaign):
    """``Campaign`` whose capture sizes come from a cached inventory.

    The capture files are immutable once written -- the campaign wrote them
    during its load phase and only ever reads them back -- so a basename->bytes
    inventory taken from the same tree is as good as a stat, and 36,423 stats
    over NFS are not.
    """

    def __init__(self, workspace: str, sizes: dict[str, int] | None) -> None:
        super().__init__(workspace)
        self._sizes = sizes

    def capture_files(self, row_id):
        if self._sizes is None:
            return super().capture_files(row_id)
        out = []
        missing = []
        for name in self.members(row_id):
            rec = self.entries.get(name)
            if rec is None:
                continue
            path = os.path.join(self.capture_root, rec["path"])
            size = self._sizes.get(os.path.basename(path))
            if size is None:
                missing.append(path)
                size = 0
            out.append((path, size))
        if missing:
            raise SystemExit(
                f"{row_id}: {len(missing)} capture files absent from the size "
                f"cache, first {missing[0]}; rerun with --stat")
        return out


def load_sizes(path: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with open(path) as fh:
        for line in fh:
            f = line.rsplit(None, 1)
            if len(f) == 2:
                try:
                    sizes[f[0]] = int(f[1])
                except ValueError:
                    pass
    return sizes


def row_id_of(row: dict) -> str | None:
    found = set(UNITS_RE.findall(" ".join(str(a) for a in row.get("argv", []))))
    return found.pop() if len(found) == 1 else None


def build_manifest(campaign: Campaign, row_id: str, produced_by: dict) -> dict:
    """One row's read set, in the order the row consumes it.

    Captures come first because ``prefetch_capture`` runs before the layer's
    weights are touched; within each group the order is the consumer's own.
    """
    plan = campaign.row_plan(row_id)
    entries = []
    for path, size in plan["_captures"]:
        entries.append({"path": path, "offset": 0, "bytes": int(size), "sha256": None})
    for path, offset, length in plan["_extents"]:
        entries.append({"path": path, "offset": int(offset), "bytes": int(length),
                        "sha256": None})
    for path, size in plan["_seeds"]:
        entries.append({"path": path, "offset": 0, "bytes": int(size), "sha256": None})
    for e in entries:
        if not e["path"].startswith(SHARED_MOUNT + "/"):
            raise SystemExit(f"{row_id}: entry outside the shared mount: {e['path']}")
        if e["bytes"] <= 0:
            raise SystemExit(f"{row_id}: zero-length entry: {e['path']}")
    # The key set is the one ``prismabuild.core.validate_data_manifest``
    # accepts exactly; everything this campaign knows and PrismaBuild does not
    # goes under ``annotations``, which the contract carries but never reads.
    return {
        "schema": SCHEMA,
        "produced_by": produced_by,
        "mount_prefix": SHARED_MOUNT,
        "annotations": {
            "row_id": row_id,
            "group": plan["group"],
            "sha256_present": False,
            "sha256_absent_reason": (
                "hashing 1023 GB of capture bytes costs more than the prewarm "
                "saves; the manifest file itself is content-addressed in the "
                "CAS, which is what binds it to the action key"),
            "counts": {
                "captures": plan["capture_files"],
                "weight_extents": plan["weight_extents"],
                "seeds": plan["seed_files"],
            },
            "bytes": {
                "captures": plan["capture_bytes"],
                "weight_extents": plan["weight_bytes"],
                "seeds": plan["seed_bytes"],
            },
        },
        "entry_count": len(entries),
        "total_bytes": plan["total_bytes"],
        "entries": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    base = CAMPAIGN_BASE + "/transition-optimization-20260910"
    ap.add_argument("--workspace",
                    default=CAMPAIGN_BASE + "/first-proof-anchor-preparation-05/workspace")
    ap.add_argument("--campaign-manifest", default=base + "/resume-overlap-manifest.json")
    ap.add_argument("--out-dir", default=base + "/data-manifests")
    ap.add_argument("--out-manifest", default=base + "/resume-overlap-manifest.with-data.json")
    ap.add_argument("--sizes-cache",
                    default="/home/rob/tmp/glm-perf-20260910/w3/evidence/capture-file-sizes.txt")
    ap.add_argument("--stat", action="store_true",
                    help="stat every capture file instead of using the size cache")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and summarise; write nothing")
    ap.add_argument("--json", help="write the run summary here")
    args = ap.parse_args()

    src = args.campaign_manifest
    rows = json.load(open(src))
    if not isinstance(rows, list):
        raise SystemExit(f"{src}: expected a list of pbcampaign rows")
    src_sha = sha256_file(src)

    sizes = None if args.stat else load_sizes(args.sizes_cache)
    campaign = CachedSizeCampaign(args.workspace, sizes)
    produced_by = {
        "tool": "prismaquant/experiments/glm_data_manifests.py",
        "commit": git_commit(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "workspace": args.workspace,
        "capture_manifest": campaign.capture_manifest_path,
        "size_source": "stat" if args.stat else args.sizes_cache,
        "host": socket.gethostname(),
        "unix": int(time.time()),
    }

    out_rows = []
    report = []
    for row in rows:
        rid = row_id_of(row)
        if rid is None:
            raise SystemExit(f"{src}: a row's argv names no single units/row-XXXX.json")
        if rid not in campaign.rows:
            raise SystemExit(f"{src}: {rid} is not a row of {args.workspace}/plan.json")
        man = build_manifest(campaign, rid, produced_by)
        path = os.path.join(args.out_dir, f"{rid}.data-manifest.json")
        blob = json.dumps(man, indent=1, sort_keys=False).encode() + b"\n"
        if not args.dry_run:
            os.makedirs(args.out_dir, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, path)
        new_row = dict(row)
        new_row["data_manifest"] = path
        out_rows.append(new_row)
        report.append({
            "row_id": rid, "path": path,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "manifest_bytes": len(blob),
            "entry_count": man["entry_count"],
            "total_bytes": man["total_bytes"],
            "captures": man["annotations"]["counts"]["captures"],
            "weight_extents": man["annotations"]["counts"]["weight_extents"],
            "capture_bytes": man["annotations"]["bytes"]["captures"],
            "weight_bytes": man["annotations"]["bytes"]["weight_extents"],
        })

    out_blob = json.dumps(out_rows, indent=1).encode() + b"\n"
    if not args.dry_run:
        tmp = args.out_manifest + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(out_blob)
        os.replace(tmp, args.out_manifest)

    summary = {
        "source_manifest": src,
        "source_manifest_sha256": src_sha,
        "out_manifest": args.out_manifest,
        "out_manifest_sha256": hashlib.sha256(out_blob).hexdigest(),
        "out_dir": args.out_dir,
        "rows": len(report),
        "total_bytes": sum(r["total_bytes"] for r in report),
        "dry_run": args.dry_run,
        "produced_by": produced_by,
        "manifests": report,
    }
    print(json.dumps(summary, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=1)
    sys.stderr.write(
        f"\n{len(report)} rows, {summary['total_bytes'] / 1e9:.2f} GB planned\n"
        f"source   {src_sha}  {src}\n"
        f"with-data {summary['out_manifest_sha256']}  {args.out_manifest}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
