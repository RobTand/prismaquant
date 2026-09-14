"""Consume one actual Tessera native dense receipt and report where admission stops.

`prismaquant/native_operator_panel.py` is exercised by synthetic boundary
fixtures. This driver runs the same consumer over a receipt a producer actually
wrote on GPU, so the boundary is stated against real bytes: which phases yield
`prismaquant.native_dense_observation.v1` evidence, and which fields stay
unknown.

It imports no serving runtime, opens no model and measures nothing. It reads
three files -- the receipt, the panel the producer froze, and the raw CUPTI
trace -- and writes the observation the consumer returns, together with the
obligations a `prismaquant.measured_runtime_prices.v2` native row still owes
beyond that observation.

Those obligations are the ones
`docs/design/joint_aura_runtime_allocation.md` ("Frontier sweep") names and
`prismaquant.runtime_provenance.admit_native_rows` enforces. They are read off
the admitted observation and its panel here, never asserted: a caller that
wants the joint-cost obligation resolved passes the joint AURA cost row the
panel was frozen against.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path


def _row_obligations(observation, panel, joint_cost_row, identity_sha256):
    """What a v2 native row still owes, read off this observation and its panel.

    `runtime_provenance.admit_native_rows` refuses a row whose phases are not
    both measured, whose per-phase scratch is unknown, whose panel prefill `m`
    differs from the table context's `prompt_tokens`, or whose panel digests
    differ from the table's. Only the first two are settled by the receipt; the
    rest are stated here as the values a table must match, so a table author
    compares rather than guesses.
    """
    phases = observation["phases"]
    satisfied, owed = {}, []
    satisfied["both_phases_measured"] = all(
        phases.get(phase, {}).get("measurement") is not None for phase in ("prefill", "decode"))
    if not satisfied["both_phases_measured"]:
        owed.append("a v2 row needs both measured phases; a null decode refuses as "
                    "'native v2 row lacks a complete measured phase'")
    satisfied["peak_scratch_bytes"] = {
        phase: value.get("peak_scratch_bytes") for phase, value in phases.items()}
    if any(value is None for value in satisfied["peak_scratch_bytes"].values()):
        owed.append("a v2 row refuses an incomplete resource ledger; every phase needs a "
                    "non-null peak_scratch_bytes")
    satisfied["context_must_declare"] = {
        "prompt_tokens": panel["phases"]["prefill"]["m"], "batch_size": 1,
        "source_sha256": panel["source_sha256"], "calibration_sha256": panel["calibration_sha256"],
        "runtime_sha256": identity_sha256(panel["runtime"])}
    if joint_cost_row is None:
        satisfied["joint_cost_row"] = None
        owed.append("the panel's cost_sha256 is not shown to be a prismaquant.joint_aura."
                    "operator.v1 row; without that row no v2 table row may cite this panel")
    else:
        row = json.loads(Path(joint_cost_row).read_text())
        satisfied["joint_cost_row"] = {"path": str(joint_cost_row),
                                       "identity_sha256": identity_sha256(row)}
        if row.get("probe_identity_sha256") != panel["probe_identity_sha256"]:
            owed.append("the supplied joint cost row's probe identity differs from the panel's")
    satisfied["fixed_resources"] = None
    owed.append("the table itself still needs fixed_resources_receipt_path naming a recomputable "
                "tessera.full_engine_resource_report.v1; admit_fixed_resources runs after "
                "admit_native_rows and refuses a report with no timing partition (#420, #237)")
    return {"satisfied": satisfied, "owed": owed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True,
                        help="the panel frozen before timing; the receipt must carry it verbatim")
    parser.add_argument("--memory-trace", type=Path, default=None,
                        help="raw CUPTI trace; required by a complete operator bound")
    parser.add_argument("--joint-cost-row", type=Path, default=None,
                        help="the prismaquant.joint_aura.operator.v1 row the panel was frozen "
                             "against, when one exists; without it the panel's cost digest is "
                             "not joint currency and no v2 row may cite it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from prismaquant.native_operator_panel import consume_native_receipt
    from prismaquant.joint_aura import identity_sha256

    panel = json.loads(args.panel.read_text())
    receipt_sha256 = hashlib.sha256(args.receipt.read_bytes()).hexdigest()
    observation = consume_native_receipt(args.receipt, expected_sha256=receipt_sha256,
                                         expected_panel=panel,
                                         memory_trace_path=args.memory_trace)
    obligations = _row_obligations(observation, panel, args.joint_cost_row, identity_sha256)
    report = {"schema": "prismaquant.native_receipt_qualification.v1",
              "receipt_path": str(args.receipt), "receipt_sha256": receipt_sha256,
              "panel_path": str(args.panel),
              "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
              "memory_trace_path": None if args.memory_trace is None else str(args.memory_trace),
              "memory_trace_sha256": None if args.memory_trace is None else
                  hashlib.sha256(args.memory_trace.read_bytes()).hexdigest(),
              "observation": observation,
              "runtime_table_admissible": observation["runtime_table_admissible"],
              "unknown": observation["unknown"],
              "runtime_row_obligations": obligations}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    summary = {"receipt_sha256": receipt_sha256, "unit": observation["unit"],
               "format": observation["format"], "status": observation["status"],
               "runtime_table_admissible": observation["runtime_table_admissible"],
               "unknown": observation["unknown"],
               "phases": {phase: {"median_ms": value["median_ms"],
                                  "peak_scratch_bytes": value["peak_scratch_bytes"],
                                  "input_bytes": value["input_bytes"],
                                  "output_bytes": value["output_bytes"]}
                          for phase, value in observation["phases"].items()},
               "resident_bytes": observation["resident_bytes"],
               "serialized_unit_bytes": observation["serialized_unit_bytes"],
               "runtime_row_obligations": obligations,
               "report_path": str(args.out)}
    print(json.dumps(summary, sort_keys=True), flush=True)
    # Operator evidence is not a runtime row. Say so in the exit status too.
    return 0 if observation["status"] == "operator_evidence" else 2


if __name__ == "__main__":
    sys.exit(main())
