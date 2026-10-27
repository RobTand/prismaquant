#!/usr/bin/env python3
"""Layer-0 identity gate for the encoder + activation-row work.

Layer 0's Linears every one have 64 cached activation rows, so the row fix is a
no-op there (one row bucket == the old code path's shape) and the encoder work
is byte-identical by construction. The new arm must therefore reproduce the
SHIPPED v1 shard on every invariant field. Layers 3+ differ by design.

The gate distinguishes its failure modes instead of collapsing them:

  NO_OUTPUT_NEW / NO_OUTPUT_BASE   the run did not produce a shard at all
  SHORT_OUTPUT_*                   shard exists but is missing rows
  UNEXPECTED_FIELDS                new arm grew a field this gate does not know
  MISMATCH                         invariant field differs -> encoder regression
  PASS

Additive fields are asserted EXPLICITLY (present, and equal to the expected
value), never silently excluded.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

# Fields the row fix is expected to ADD, with the value layer 0 must carry
# (every layer-0 Linear has 64 cached rows).
EXPECTED_ADDITIVE = {"n_activation_rows": 64}


def load(path: str):
    p = Path(path)
    if not p.exists():
        return None, f"missing file {path}"
    try:
        with open(p, "rb") as fh:
            blob = pickle.load(fh)
    except Exception as e:  # noqa: BLE001
        return None, f"unreadable ({type(e).__name__}: {e})"
    if not isinstance(blob, dict):
        return None, "cost envelope must be a mapping"
    costs = blob.get("costs")
    if not isinstance(costs, dict) or not costs:
        return None, "no 'costs' table"
    for name, row in costs.items():
        if not isinstance(name, str) or not isinstance(row, dict) or not row:
            return None, f"invalid cost row {name!r}: expected a nonempty format mapping"
        for fmt, entry in row.items():
            if not isinstance(fmt, str) or not isinstance(entry, dict) or not entry:
                return None, f"invalid cost entry {name!r}/{fmt!r}: expected a nonempty mapping"
    return costs, None


def compare(ref: dict, got: dict, ref_tag: str, got_tag: str):
    """Invariant fields must match exactly; report every difference."""
    diffs = []
    only_ref = sorted(set(ref) - set(got))
    only_got = sorted(set(got) - set(ref))
    if only_ref or only_got:
        diffs.append(("KEYSET", f"only in {ref_tag}: {len(only_ref)}, "
                                f"only in {got_tag}: {len(only_got)}",
                      (only_ref[:3], only_got[:3]), None))
    for name in sorted(set(ref) & set(got)):
        rrow, growth = ref[name], got[name]
        for fmt in sorted(set(rrow) | set(growth)):
            re_, ge = rrow.get(fmt), growth.get(fmt)
            if re_ is None or ge is None:
                diffs.append((name, fmt, "format present in only one arm",
                              None))
                continue
            for k, v in re_.items():
                if k not in ge:
                    diffs.append((name, fmt, f"{k}: MISSING in {got_tag}",
                                  (v, None)))
                elif ge[k] != v:
                    diffs.append((name, fmt, k, (v, ge[k])))
    return diffs


def check_additive(got: dict, ref: dict):
    """Require the expected values on every entry and refuse unknown additions."""
    problems = []
    for name, row in got.items():
        rrow = ref.get(name, {})
        for fmt, entry in row.items():
            for k, expected in EXPECTED_ADDITIVE.items():
                if k not in entry:
                    problems.append((name, fmt, f"{k}: MISSING", None))
                elif entry[k] != expected:
                    problems.append((name, fmt, f"{k} != {expected}", entry[k]))
            base_keys = set(rrow.get(fmt, {}))
            for k in set(entry) - base_keys:
                if k not in EXPECTED_ADDITIVE:
                    problems.append((name, fmt, f"unexpected new field {k!r}",
                                     entry[k]))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--base", default="")
    ap.add_argument("--min-rows", type=int, default=775)
    ap.add_argument("--verdict", required=True)
    args = ap.parse_args()

    def verdict(v, *msg):
        for m in msg:
            print(m)
        Path(args.verdict).write_text(v)
        print(f"GATE VERDICT: {v}")
        return 0 if v == "PASS" else 1

    ship, err_s = load(args.shipped)
    if ship is None:
        return verdict("NO_SHIPPED_REFERENCE", f"shipped: {err_s}")
    new, err_n = load(args.new)
    if new is None:
        return verdict("NO_OUTPUT_NEW",
                       f"new arm produced no usable shard: {err_n}",
                       "-> the RUN failed, this is NOT an encoder mismatch; "
                       "check the driver log and its exit code")
    if len(new) < args.min_rows:
        return verdict("SHORT_OUTPUT_NEW",
                       f"new shard has {len(new)} rows, expected "
                       f">= {args.min_rows}")

    print(f"shipped rows={len(ship)}  new rows={len(new)}")
    diffs = compare(ship, new, "shipped", "new")
    extra = check_additive(new, ship)

    if args.base:
        base, err_b = load(args.base)
        if base is None:
            print(f"NOTE: base arm shard unusable ({err_b}); "
                  "skipping base-vs-new cross-check")
        else:
            bd = compare(base, new, "base", "new")
            print(f"base-vs-new invariant diffs: {len(bd)}")
            for d in bd[:5]:
                print("   ", d)

    if extra:
        for p in extra[:10]:
            print("  ADDITIVE:", p)
        return verdict("UNEXPECTED_FIELDS",
                       f"{len(extra)} additive-field problems")
    n_add = sum(len(row) for row in new.values())
    print(f"n_activation_rows present and == 64 on {n_add} entries")

    if diffs:
        for d in diffs[:15]:
            print("  DIFF:", d)
        return verdict("MISMATCH",
                       f"{len(diffs)} invariant-field differences vs the "
                       f"shipped v1 shard -> ENCODER REGRESSION, do not merge")
    return verdict("PASS", "every invariant field matches the shipped v1 "
                           "layer-0 shard exactly")


if __name__ == "__main__":
    raise SystemExit(main())
