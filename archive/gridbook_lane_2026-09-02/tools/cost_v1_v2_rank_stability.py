#!/usr/bin/env python3
"""v1-vs-v2 cost-table cross-check: per-layer K14-excess rank stability.

v1 measured every expert Linear in layers 3-42 on ONE activation row (the
chunk-minimum truncation); v2 measures each on all of its own rows. The
allocator's decision is driven by how Linears RANK against each other within a
layer, not by the absolute MSE, so the cross-check that matters is whether the
ranking survived the fix.

"K14 excess" per Linear = rel_output_mse(NVFP4_CB_K14) - rel_output_mse(FP8_CB_K36):
how much extra relative output error the cheap rung costs over the expensive
one. That is the quantity the knapsack trades against bytes.

Emits, per layer: Spearman rho and Kendall tau between the v1 and v2 orderings,
the top-k set overlap (the Linears that actually get promoted), and the count of
Linears whose n_activation_rows is 1 in V2 (v1 pickles lack the field). NOTE
this UNDERCOUNTS v1's single-row damage: v1 truncated every chunk member to the
CHUNK minimum, so a Linear with 64 own rows was still measured on 1 row whenever
a chunk-mate had 1 — the v2 count only reflects genuinely sparse experts.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

LAYER = re.compile(r"\blayers\.(\d+)\.")


def load_costs(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh).get("costs", {})


def _rank(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def kendall(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            sa = (a[i] > a[j]) - (a[i] < a[j])
            sb = (b[i] > b[j]) - (b[i] < b[j])
            if sa * sb > 0:
                con += 1
            elif sa * sb < 0:
                dis += 1
    tot = con + dis
    return (con - dis) / tot if tot else float("nan")


def excess(row: dict, cheap: str, dear: str, key: str) -> float | None:
    if cheap not in row or dear not in row:
        return None
    a, b = row[cheap], row[dear]
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    if key not in a or key not in b:
        return None
    return float(a[key]) - float(b[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--cheap", default="NVFP4_CB_K14")
    ap.add_argument("--dear", default="FP8_CB_K36")
    ap.add_argument("--key", default="rel_output_mse")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    c1, c2 = load_costs(args.v1), load_costs(args.v2)
    shared = sorted(set(c1) & set(c2))
    by_layer: dict[int, list[str]] = defaultdict(list)
    for n in shared:
        m = LAYER.search(n)
        if m:
            by_layer[int(m.group(1))].append(n)

    report = {}
    print(f"{'layer':>5s} {'n':>5s} {'spearman':>9s} {'kendall':>8s} "
          f"{'top%d_overlap' % args.topk:>13s} {'v2_1row':>8s}")
    for layer in sorted(by_layer):
        names = by_layer[layer]
        pairs = []
        one_row = 0
        for n in names:
            e1 = excess(c1[n], args.cheap, args.dear, args.key)
            e2 = excess(c2[n], args.cheap, args.dear, args.key)
            if e1 is None or e2 is None:
                continue
            pairs.append((n, e1, e2))
            r = c2[n].get(args.cheap, {}).get("n_activation_rows")
            if r == 1:
                one_row += 1
        if len(pairs) < 3:
            continue
        a = [p[1] for p in pairs]
        b = [p[2] for p in pairs]
        rho, tau = spearman(a, b), kendall(a, b)
        k = min(args.topk, len(pairs))
        t1 = {p[0] for p in sorted(pairs, key=lambda p: -p[1])[:k]}
        t2 = {p[0] for p in sorted(pairs, key=lambda p: -p[2])[:k]}
        ov = len(t1 & t2) / k
        report[layer] = {"n": len(pairs), "spearman": rho, "kendall": tau,
                         f"top{k}_overlap": ov, "v1_single_row": one_row}
        print(f"{layer:5d} {len(pairs):5d} {rho:9.4f} {tau:8.4f} "
              f"{ov:13.3f} {one_row:8d}")

    if report:
        rhos = [v["spearman"] for v in report.values()]
        print(f"\nspearman across {len(rhos)} layers: "
              f"min={min(rhos):.4f} median={sorted(rhos)[len(rhos) // 2]:.4f} "
              f"max={max(rhos):.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
