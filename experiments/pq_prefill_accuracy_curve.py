#!/usr/bin/env python3
"""The measured prefill price of each served artifact, against its served KL.

The table's own composition is ``sequential_operator_sum``, so an artifact's
prefill price over the priced units is the sum of those units' measured
``prefill_ms``. Two scopes are stated separately and never mixed:

* the **operator sum over priced units** -- exactly the units this table prices,
  at the format each artifact assigns them. This is a measurement.
* the **fixed whole-model charge** -- NOT included. ``admit_fixed_resources``
  refuses every v2 table today (``runtime_provenance.py:676``: the native-row
  and full-engine transient charge boundary is not versioned), so this tool
  carries no fixed term rather than inventing one. A price that omits a charge
  says so; it does not silently declare it zero.

The accuracy axis is not computed here. It is read from a column the caller
supplies, with its own metric identity, and copied through unchanged.

Each row's ``prefill_ms`` is the median of that operator's own CUDA-event
samples, so the operator sum is a sum of medians. The spread between two
artifacts is only a fact about the runtime if it is larger than what those
samples resolve, so every point also carries a paired bootstrap interval over
the rows' own samples -- resampling each row's samples with replacement,
re-taking each median, re-summing. No threshold is applied and no verdict is
declared: the intervals are published so a reader cannot mistake an unresolved
ordering for a measured one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.measured_runtime_prices import (  # noqa: E402
    RuntimePriceError, parse_measured_runtime_table, parse_runtime_context)
from prismaquant.runtime_provenance import admit_fixed_resources, admit_native_rows, load_runtime_relation  # noqa: E402

SCHEMA = "prismaquant.prefill_accuracy_curve.v1"


def sha256(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bootstrap_sum(samples_per_row, *, draws, seed):
    """The distribution of the operator sum under each row's own samples.

    Every row is resampled with replacement from its OWN measured samples and
    re-reduced by the same median the table row was reduced by, so this
    describes only the dispersion the measurement itself carries.
    """
    rng = random.Random(seed)
    totals = []
    for _ in range(draws):
        totals.append(sum(statistics.median(rng.choices(samples, k=len(samples)))
                          for samples in samples_per_row))
    totals.sort()
    return {"draws": draws, "seed": seed,
            "p2.5": totals[int(0.025 * draws)], "p50": totals[draws // 2],
            "p97.5": totals[min(draws - 1, int(0.975 * draws))],
            "samples_per_row": [len(samples) for samples in samples_per_row]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True,
                        help="JSON: artifact -> {unit: format} over the priced units")
    parser.add_argument("--accuracy", type=Path, required=True,
                        help="JSON: {metric: {...}, artifacts: {artifact: {...}}}")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=237)
    args = parser.parse_args(argv)

    payload = json.loads(args.table.read_text())
    table = parse_measured_runtime_table(
        payload, source_path=str(args.table.resolve()),
        expected_context=parse_runtime_context(payload["context"]),
        expected_cost_sha256=sha256(args.costs))
    relation = load_runtime_relation(table.runtime_provenance, context=table.context,
                                     root=args.table.resolve().parent)
    admit_native_rows(table, relation)
    try:
        admit_fixed_resources(table, relation)
        fixed = {"status": "admitted", "prefill_ms": table.fixed_resources.prefill_ms}
    except RuntimePriceError as exc:
        head = str(exc).split(";")[0]
        fixed = {"status": "refused", "prefill_ms": None, "first_refusal": head,
                 "scope": "no whole-model fixed charge is carried; every price below is the "
                          "operator sum over the priced units alone"}

    prices = {(row.unit, row.fmt): row for row in table.rows}
    # The parsed row keeps only the reduced price; the raw samples that
    # reduction came from live on the table document itself.
    payload_rows = {(row["unit"], row["format"]): row for row in payload["rows"]}
    priced_units = sorted({row.unit for row in table.rows})
    assignments = json.loads(args.assignments.read_text())
    accuracy = json.loads(args.accuracy.read_text())

    points, refused = [], {}
    for artifact, assignment in sorted(assignments.items()):
        # Only the priced units are in scope. A unit this table does not price
        # is recorded as out of scope, not as a refusal -- the refusal is a
        # priced unit whose ASSIGNED format this table never measured, because
        # then the sum would silently price the artifact at another rate.
        missing = [f"{unit}@{assignment[unit]}" for unit in priced_units
                   if unit in assignment and (unit, assignment[unit]) not in prices]
        outside = sorted(set(assignment) - set(priced_units))
        unpriced = sorted(set(priced_units) - set(assignment))
        measured = accuracy["artifacts"].get(artifact)
        if missing or measured is None or unpriced:
            refused[artifact] = {"unpriced_cells": missing, "units_without_an_assignment": unpriced,
                                 "assigned_units_this_table_does_not_price": outside,
                                 "accuracy": None if measured is None else "present"}
            continue
        rows = [prices[(unit, assignment[unit])] for unit in priced_units]
        samples = {phase: [list(payload_rows[(row.unit, row.fmt)][phase]["samples_ms"]) for row in rows]
                   for phase in ("prefill", "decode")}
        points.append({
            "prefill_ms_operator_sum_bootstrap": bootstrap_sum(
                samples["prefill"], draws=args.bootstrap_draws, seed=args.bootstrap_seed),
            "decode_ms_operator_sum_bootstrap": bootstrap_sum(
                samples["decode"], draws=args.bootstrap_draws, seed=args.bootstrap_seed),
            "artifact": artifact, "units": priced_units,
            "assignment": {unit: assignment[unit] for unit in priced_units},
            "prefill_ms_operator_sum": sum(row.resources.prefill_ms for row in rows),
            "decode_ms_operator_sum": sum(row.resources.decode_ms for row in rows),
            "serialized_bytes_sum": sum(row.resources.serialized_bytes for row in rows),
            "resident_bytes_sum": sum(row.resources.resident_bytes for row in rows),
            "per_unit_prefill_ms": {row.unit: row.resources.prefill_ms for row in rows},
            "operator_routes": {row.unit: row.binding.operator_route for row in rows},
            "accuracy": measured,
        })
    points.sort(key=lambda point: point["prefill_ms_operator_sum"])
    document = {
        "schema": SCHEMA,
        "scope": ("operator-sum prefill price over the units this table prices, at each served "
                  "artifact's own layer-0 assignment; no fixed whole-model charge"),
        "table": {"path": str(args.table.resolve()), "sha256": sha256(args.table),
                  "table_id": table.table_id, "rows": len(table.rows)},
        "cost_payload_sha256": sha256(args.costs),
        "runtime_context": payload["context"],
        "admission": {"load_runtime_relation": "loaded", "admit_native_rows": "admitted",
                      "rows_admitted": len(table.rows), "admit_fixed_resources": fixed},
        "accuracy_metric": accuracy["metric"],
        "priced_units": priced_units,
        "points": points, "refused_artifacts": refused,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "points": len(points),
                      "refused": sorted(refused)}, indent=1, sort_keys=True), flush=True)
    for point in points:
        interval = point["prefill_ms_operator_sum_bootstrap"]
        print(f"  {point['artifact']:30s} prefill_sum_ms {point['prefill_ms_operator_sum']:.6f} "
              f"[{interval['p2.5']:.6f}, {interval['p97.5']:.6f}]  "
              f"accuracy {point['accuracy']}", flush=True)
    return 0 if points else 2


if __name__ == "__main__":
    sys.exit(main())
