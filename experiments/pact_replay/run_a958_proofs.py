"""Run named proof steps as separate processes and print one JSON line per step.

--expect green exits 0 only if every step passes.
--expect red exits 0 only if every step fails with an AssertionError, which
proves that the case reaches its assertion and does not fail on setup.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

DRAW = "/mnt/shared/tessera-measurements/codec-decomp-20260930/indomain-h/corpus/draw/calibration_tokens.safetensors"
FINDINGS = ("panel_expert_mismatch", "panel_anchor_role", "teacher_identity", "bounded_readback",
            "window_telemetry", "teacher_content", "teacher_path_resume",
            "panel_reference_second_table", "panel_reference_full_stack", "energy_menu_chord_only",
            "packed_backend_taps")


def steps(root, pq_root):
    table = {"a958:" + case: ["test_a958_findings.py", "--case", case, "--output", str(root / (case + ".json"))]
             for case in FINDINGS}
    table["band_continuation"] = ["test_band_continuation.py", "--pq-root", pq_root,
                                  "--output", str(root / "band_continuation.json")]
    table["batched_execution"] = ["test_batched_execution.py", "--output", str(root / "batched.json")]
    for case in ("render", "identity", "restored", "restored_dtype", "forward", "replacement", "publication",
                 "publication_failure"):
        table["checkpoint_protocol:" + case] = ["test_checkpoint_protocol.py", "--case", case,
                                                "--output", str(root / ("protocol-" + case + ".json"))]
    for case in ("null", "receipt_identity", "t16_unit", "t16_panel"):
        table["review_corrections:" + case] = ["test_review_corrections.py", "--case", case,
                                               "--output", str(root / ("review-" + case + ".json"))]
    table["gain_price_contract"] = ["test_gain_price_contract.py", "--consumer-root", ".", "--draw", DRAW,
                                    "--output", str(root / "contracts.json")]
    table["price_calibration_panel"] = ["test_price_calibration_panel.py", "--output", str(root / "panels.json")]
    table["parent_numerical_contracts"] = ["test_parent_numerical_contracts.py", "--output",
                                           str(root / "numerical.json")]
    table["rendered_residency"] = ["test_rendered_residency.py", "--output", str(root / "residency.json")]
    table["t8_rate_anchor"] = ["test_t8_rate_anchor.py", "--consumer-root", ".", "--output",
                               str(root / "t8_rate.json")]
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pq-root", required=True)
    parser.add_argument("--expect", choices=("green", "red"), required=True)
    parser.add_argument("steps", nargs="+")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    table = steps(args.root, args.pq_root)
    outcomes = []
    for name in args.steps:
        if name not in table:
            raise SystemExit("unknown step " + name)
        result = subprocess.run([sys.executable] + table[name], capture_output=True, text=True)
        asserted = "AssertionError" in result.stderr
        outcomes.append({"step": name, "returncode": result.returncode, "assertion_failure": asserted})
        print(json.dumps({"step": name, "returncode": result.returncode, "assertion_failure": asserted,
                          "stdout_tail": result.stdout[-1500:], "stderr_tail": result.stderr[-2500:]}),
              flush=True)
    if args.expect == "green":
        ok = all(item["returncode"] == 0 for item in outcomes)
    else:
        ok = all(item["returncode"] != 0 and item["assertion_failure"] for item in outcomes)
    print(json.dumps({"schema": "pact.a958_proof_run.v1", "expect": args.expect, "satisfied": ok,
                      "steps": outcomes}), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
