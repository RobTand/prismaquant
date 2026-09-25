#!/usr/bin/env python3
"""Compare PQ #1246 measurement runs: identity of outputs, then per-group costs.

Usage: pq1246_host_staging_compare.py RUN.json [RUN.json ...]

The first run is the reference for identity. Every run's payload digests
(per key and per metadata entry, per repeat), plane digests (per pass)
and Stage A receipt leaves are compared with it, and each difference is
listed. Differences that bind a run's own Stage A session are listed apart:
three metadata entries, and the receipt's session generations, paths,
entry file digests and wall clocks. They differ between any two runs of one
checkout. Then per-group host allocation counts, cgroup and vmstat fault
deltas, site times, and the pass head are summarised per run, with the
non-last and last groups of a pass apart: the last group has no next group
whose boundaries prefetch into it.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path


def stat(values):
    values = list(values)
    if not values:
        return "n=0"
    return (f"n={len(values)} mean={statistics.fmean(values):.1f} "
            f"median={statistics.median(values):.1f} min={min(values):.1f} "
            f"max={max(values):.1f}")


# Metadata entries that bind the run's own Stage A session: they differ
# between any two runs of one checkout.
SESSION_BOUND = {"adjoint_slice_sha256", "checkpoint_identity_sha256", "distributed_quantum"}
# Stage A receipt leaves that bind the session, a path or a wall clock.
RECEIPT_SESSION = re.compile(
    r"(\.session\.generation$|\.path$|\.generation_manifest$|^\.telemetry\.|^\.dev_mode\."
    r"|_entries(\.\w+)?\[\d+\]\.sha256$|\.cotangent_sha256$)")


def _flat(value, path="", out=None):
    out = {} if out is None else out
    if isinstance(value, dict):
        for key, item in value.items():
            _flat(item, f"{path}.{key}", out)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _flat(item, f"{path}[{index}]", out)
    else:
        out[path] = value
    return out


def identity(runs):
    reference = runs[0]
    ref_payload = reference["repeats"][0]["payload"]
    ref_planes = [p["plane"] for p in reference["repeats"][0]["passes"]]
    ref_receipt = _flat(reference["stage_a"]["receipt"])
    print("== identity (reference: %s)" % reference["label"])
    for run in runs:
        diffs, session = [], set()
        source = run.get("producer_source_sha256") or {}
        receipt = _flat(run["stage_a"]["receipt"])
        leaves = [k for k in sorted(set(receipt) | set(ref_receipt))
                  if receipt.get(k) != ref_receipt.get(k)]
        bound = [k for k in leaves if RECEIPT_SESSION.search(k)]
        diffs += [f"stage_a{k}" for k in leaves if not RECEIPT_SESSION.search(k)]
        for index, repeat in enumerate(run["repeats"]):
            payload = repeat["payload"]
            for key in sorted(set(ref_payload) | set(payload)):
                if key == "provenance":
                    mine, ref = payload.get(key, {}), ref_payload.get(key, {})
                    for sub in sorted(set(mine) | set(ref)):
                        if mine.get(sub) == ref.get(sub):
                            continue
                        if sub in SESSION_BOUND:
                            session.add(sub)
                        else:
                            diffs.append(f"repeat{index} metadata[{sub}]")
                elif payload.get(key) != ref_payload.get(key):
                    diffs.append(f"repeat{index} payload[{key}]")
            planes = [p["plane"] for p in repeat["passes"]]
            if planes != ref_planes:
                diffs.append(f"repeat{index} planes "
                             f"({sum(a != b for a, b in zip(planes, ref_planes))} of "
                             f"{len(ref_planes)} passes differ)")
            if repeat["costs_rows"] != reference["repeats"][0]["costs_rows"]:
                diffs.append(f"repeat{index} costs_rows")
        n_sub = len(ref_payload.get("provenance", {}))
        print(f"  {run['label']:12s} fixed={run['fixed']} host={run['host']} "
              f"defrag={run['thp']['defrag'].split('[')[1].split(']')[0]} "
              f"enabled={run['thp']['enabled'].split('[')[1].split(']')[0]} "
              f"repeats={len(run['repeats'])} passes/repeat={len(run['repeats'][0]['passes'])}")
        print(f"      producer source: real {str(source.get('real'))[:16]} "
              f"pinned {str(source.get('pinned'))[:16]}")
        print(f"      payload: {len(ref_payload) - 1} keys and {n_sub} metadata entries; "
              f"planes: {len(ref_planes)} passes x {len(ref_planes[0]) if ref_planes else 0} slots; "
              f"stage A receipt: {len(ref_receipt)} leaves, {len(bound)} session-bound differ")
        print(f"      session-bound metadata entries that differ: {sorted(session) or 'none'}")
        print("      -> " + ("IDENTICAL outside the session-bound entries" if not diffs
                            else "DIFFERS: " + ", ".join(diffs)))


def per_group(runs):
    print("== per group (all repeats, all passes)")
    for run in runs:
        rows = {"non-last": [], "last": []}
        for repeat in run["repeats"]:
            for record in repeat["passes"]:
                groups = record["groups"]
                for position, group in enumerate(groups):
                    rows["last" if position == len(groups) - 1 else "non-last"].append(group)
        print(f"-- {run['label']} (fixed={run['fixed']}, {run['host']}, "
              f"started {run['stage_a']['wall'][0]:.0f})")
        for name, groups in rows.items():
            if not groups:
                continue
            print(f"   [{name}] groups={len(groups)}")
            print(f"     large host allocations at the sites : {stat(g['large_at_sites'] for g in groups)}")
            print(f"     large allocations elsewhere         : prefetch {stat(g['large_in_prefetch'] for g in groups)} | other {stat(g['large_other'] for g in groups)}")
            print(f"     cgroup thp_fault_alloc              : {stat(g['counters']['cgroup'].get('thp_fault_alloc', 0) for g in groups)}")
            print(f"     cgroup pgfault                      : {stat(g['counters']['cgroup'].get('pgfault', 0) for g in groups)}")
            print(f"     cgroup pgscan_direct                : {stat(g['counters']['cgroup'].get('pgscan_direct', 0) for g in groups)}")
            print(f"     vmstat thp_fault_fallback           : {stat(g['counters']['vmstat'].get('thp_fault_fallback', 0) for g in groups)}")
            print(f"     vmstat compact_stall                : {stat(g['counters']['vmstat'].get('compact_stall', 0) for g in groups)}")
            print(f"     site read / stack / store ms        : "
                  f"{statistics.fmean(g['site_ns']['read'] for g in groups) / 1e6:.2f} / "
                  f"{statistics.fmean(g['site_ns']['stack'] for g in groups) / 1e6:.2f} / "
                  f"{statistics.fmean(g['site_ns']['store'] for g in groups) / 1e6:.2f} "
                  f"(sum {statistics.fmean(sum(g['site_ns'].values()) for g in groups) / 1e6:.2f})")
            print(f"     group wall ms                       : {stat(g['ns'] / 1e6 for g in groups)}")


def per_pass(runs):
    print("== per pass")
    for run in runs:
        passes = [p for r in run["repeats"] for p in r["passes"]]
        heads = [p["head"] for p in passes]
        print(f"-- {run['label']}")
        print(f"   head cgroup thp_fault_alloc : {stat(h['cgroup'].get('thp_fault_alloc', 0) for h in heads)}")
        print(f"   head cgroup pgfault         : {stat(h['cgroup'].get('pgfault', 0) for h in heads)}")
        print(f"   pass cgroup thp_fault_alloc : {stat(p['pass']['cgroup'].get('thp_fault_alloc', 0) for p in passes)}")
        print(f"   pass cgroup pgfault         : {stat(p['pass']['cgroup'].get('pgfault', 0) for p in passes)}")
        print(f"   pass cgroup pgscan_direct   : {stat(p['pass']['cgroup'].get('pgscan_direct', 0) for p in passes)}")
        print(f"   pass vmstat thp_fallback    : {stat(p['pass']['vmstat'].get('thp_fault_fallback', 0) for p in passes)}")
        print(f"   pass wall s                 : {stat(p['wall'][1] - p['wall'][0] for p in passes)}")
        print(f"   peak memory.current GiB     : {run['summary']['peak_current_bytes'] / 2**30:.2f}")
        print(f"   direct I/O grid (memory, offset): {run['summary']['direct_io_grid']}; "
              f"buffers per repeat: {run['summary']['direct_io_buffers']}")
        print(f"   direct I/O calls per repeat : {[r['direct_io_calls'] for r in run['repeats']]}")


def main(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    identity(runs)
    per_group(runs)
    per_pass(runs)


if __name__ == "__main__":
    main(sys.argv[1:])
