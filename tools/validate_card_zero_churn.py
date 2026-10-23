#!/usr/bin/env python3
"""Compare the card's SCALAR pricing with the allocator's scalar loss formula.

For matching units with a ``weight_mse`` record, compare
``weight_dloss_scalar(unit, weight_mse)`` with
``allocator_solver.predicted_dloss(unit.h_trace, weight_mse)``. Other cost
records are skipped. Equality here does not establish equal candidate menus,
allocations, or exported artifacts.

It is a tool rather than a test because it needs a real WORK_DIR: a `probe.pkl`
and the `cost.pkl` produced from it. Run it against any completed run.

    python3 tools/validate_card_zero_churn.py --artifacts WORK/artifacts

Measured 2026-08-14 on `prod-27b-nvfp4cb-5p5` (Qwen3.6-27B, 505 units, the
NVFP4-CB menu): 3528/3528 (unit, format) pairs bit-identical, 3024 of them with
a nonzero predicted loss.

The ``ZERO_CHURN`` verdict covers only the compared scalar prices. MARGINAL and
AQUA are separate estimators and are not compared here.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifacts",
        required=True,
        help="a run's artifacts/ dir, containing probe.pkl and cost.pkl",
    )
    ap.add_argument("--cost", default="cost.pkl", help="cost file name")
    ap.add_argument("--json", default=None, help="write a JSON report here")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from prismaquant.sensitivity_card_build import card_from_probe
    from prismaquant.format_cost_protocol import weight_dloss_scalar
    from prismaquant.allocator_solver import predicted_dloss

    probe_path = os.path.join(args.artifacts, "probe.pkl")
    cost_path = os.path.join(args.artifacts, args.cost)
    for p in (probe_path, cost_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    with open(cost_path, "rb") as fh:
        cost = pickle.load(fh)
    card = card_from_probe(probe_path)

    table = cost.get("costs")
    if not isinstance(table, dict):
        raise SystemExit(
            f"{cost_path} has no 'costs' mapping (keys: {sorted(cost)[:12]})"
        )

    n_cmp = n_exact = n_nonzero = 0
    worst = 0.0
    worst_at = None
    absent: list[str] = []
    formats: set[str] = set()

    for name, per_fmt in table.items():
        if name not in card:
            absent.append(name)
            continue
        unit = card[name]
        for fmt, rec in per_fmt.items():
            if not isinstance(rec, dict) or "weight_mse" not in rec:
                continue
            formats.add(fmt)
            wmse = float(rec["weight_mse"])
            got = weight_dloss_scalar(unit, wmse)      # card SCALAR tier
            want = predicted_dloss(unit.h_trace, wmse)  # shipping allocator
            n_cmp += 1
            if got == want:
                n_exact += 1
            else:
                rel = abs(got - want) / max(abs(want), 1e-300)
                if rel > worst:
                    worst, worst_at = rel, {
                        "unit": name, "format": fmt, "card": got, "allocator": want,
                    }
            if want > 0.0:
                n_nonzero += 1

    report = {
        "artifacts": args.artifacts,
        "units_in_card": len(card),
        "units_in_cost": len(table),
        "units_in_cost_absent_from_card": len(absent),
        "formats": sorted(formats),
        "pairs_compared": n_cmp,
        "pairs_nonzero_dloss": n_nonzero,
        "pairs_bit_identical": n_exact,
        "worst_relative_deviation": worst,
        "worst_at": worst_at,
        "verdict": "ZERO_CHURN" if (n_cmp and n_exact == n_cmp) else "CHURN",
    }
    if absent[:5]:
        report["examples_absent_from_card"] = absent[:5]

    print(json.dumps(report, indent=2))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)

    if not n_cmp:
        print("\nFAIL: nothing was compared -- no unit matched between card and cost.")
        return 2
    if absent:
        print(
            f"\nNOTE: {len(absent)} unit(s) priced in cost.pkl are absent from the "
            "card. The card is built from probe.pkl, so this means the cost run "
            "saw units the probe did not."
        )
    return 0 if n_exact == n_cmp else 1


if __name__ == "__main__":
    sys.exit(main())
