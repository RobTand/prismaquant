"""CPU acceptance of the retained singleton checker through the public SDK."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant import shape_runtime_prices as prices, staged_lease
from prismaquant.digests import bytes_sha256hex


def publish(path, value):
    raw = prices.canonical_strict(value).encode() + b"\n"
    path.write_bytes(raw)
    return {"path": str(path), "bytes": len(raw), "sha256": bytes_sha256hex(raw)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--action-key", required=True)
    parser.add_argument("--published-unix", type=float, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    staged_lease.set_lease_helper_root(args.sdk_root)
    sdk = staged_lease.client_sdk()
    result = sdk.read_verified_action_result(
        sdk.PoolQueue(), args.action_key, published_unix=args.published_unix,
        attempt=args.attempt, max_result_bytes=prices.CHECKER_RESULT_MAX_BYTES,
        max_evidence_bytes=prices.CHECKER_EVIDENCE_MAX_BYTES)
    sdk.bind_standard_capture_command(result["request"])
    request = result["request"]
    args.output.mkdir(parents=True, exist_ok=True)
    observation = args.observation.read_bytes()
    binding = {"path": str(args.observation), "bytes": len(observation),
               "sha256": bytes_sha256hex(observation)}
    selector = {"action_key": args.action_key, "published_unix": args.published_unix,
                "attempt": args.attempt}
    receipt_path = args.output / "checker-receipt.json"
    proof = {"schema": prices.CHECKER_RECEIPT_SCHEMA, "selector": selector,
             "observation": binding}
    receipt = publish(receipt_path, proof)
    if args.inspect:
        config = {"schema": prices.CHECKER_CONFIG_SCHEMA, "checkers": [{
            "snapshot": request["params"]["checkout_snapshot"],
            "cwd": request["params"]["cwd"],
            "working_directory": request["task"]["working_directory"],
            "command": request["params"]["command"], "environment": request["environment"],
            "observation_output": str(args.observation)}]}
        config_binding = publish(args.output / "checker-config-candidate.json", config)
        print(json.dumps({"selector": selector, "receipt": receipt,
                          "candidate_config": config_binding,
                          "pb_receipt_sha256": result["receipt"]["receipt_sha256"]}, sort_keys=True))
        return 0
    scope = prices.ShapeTableScope(
        contract_sha256="0869f326543374dbd26b75e1d736befed378280d9a5724c4f170bf398aefdbaa",
        tessera_commit="b40c93cb73745097e57a1ba4cf5b9eee166c759a",
        runtime_image_digest="localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5",
        tensor_parallel=1, platform="sm_121", residency="resident", execution_mode="eager")
    eligibility = prices._convert_eligibility(scope)
    table = prices.consume_shape_time_observation(
        [args.observation], checker_receipts=[receipt_path], table_id="retained-singleton",
        expected_scope=scope, eligibility=eligibility)
    table_path, table_sha = prices.write_shape_table(table, args.output / "table.json")
    loaded = prices.load_shape_table(table_path)
    admitted = prices.admit_shape_table(loaded, scope=scope, eligibility=eligibility)
    refusals = []
    try:
        prices.admit_shape_table(loaded, scope=replace(scope, tensor_parallel=2), eligibility=eligibility)
    except prices.ShapeRuntimeError:
        refusals.append("tp2")
    forged_observation = json.loads(observation)
    samples = json.loads(Path(forged_observation["evidence"]["samples"]["path"]).read_bytes())
    samples["samples_ms"][0] *= 2
    samples_binding = publish(args.output / "forged-samples.json", samples)
    forged_panel = json.loads(Path(forged_observation["panel"]["path"]).read_bytes())
    forged_panel["evidence"]["samples"] = samples_binding
    forged_timing = prices._timing_summary(samples["samples_ms"])
    for panel_row in forged_panel["rows"]:
        panel_row["timing"] = forged_timing
    forged_panel_binding = publish(args.output / "forged-panel.json", forged_panel)
    forged_observation["panel"] = forged_panel_binding
    forged_observation["expected_panel_sha256"] = forged_panel_binding["sha256"]
    forged_observation["evidence"]["samples"] = samples_binding
    forged_observation["sampling"]["samples_ms"] = samples["samples_ms"]
    forged_observation["timing"] = forged_timing
    forged_observation_path = args.output / "forged-observation.json"
    forged_observation_binding = publish(forged_observation_path, forged_observation)
    forged = {**proof, "observation": forged_observation_binding,
              "selector": {**selector, "action_key": "0" * 64}}
    forged_path = args.output / "forged-checker-receipt.json"
    forged_binding = publish(forged_path, forged)
    try:
        prices.consume_shape_time_observation([forged_observation_path], checker_receipts=[forged_path], table_id="forged")
    except prices.ShapeRuntimeError:
        refusals.append("unexecuted-checker-convert")
    forged_table = copy.deepcopy(loaded.as_dict())
    for row in forged_table["rows"]:
        row["measurement"] = prices.OperatorMeasurement(
            method="cuda_events", samples_ms=tuple(samples["samples_ms"]),
            warmup_iterations=samples["warmup_iterations"], receipt_path=str(forged_path),
            receipt_sha256=forged_binding["sha256"]).as_dict()
    forged_table_path = args.output / "forged-table.json"
    publish(forged_table_path, forged_table)
    try:
        prices.load_shape_table(forged_table_path)
    except prices.ShapeRuntimeError:
        refusals.append("unexecuted-checker-load")
    if refusals != ["tp2", "unexecuted-checker-convert", "unexecuted-checker-load"]:
        raise ValueError("a required negative control was accepted")
    row = admitted.rows[0]
    trace = {"selector": selector, "pb_receipt_sha256": result["receipt"]["receipt_sha256"],
             "table_sha256": table_sha, "key": asdict(row.key), "rows": len(admitted.rows),
             "samples": len(row.measurement.samples_ms), "median_ms": row.measurement.median_ms,
             "admitted": admitted.admitted, "negative_controls": refusals,
             "forged_control": {"panel": forged_panel_binding, "samples": samples_binding,
                                "observation": forged_observation_binding, "proof": forged_binding},
             "gpu_executed": False, "rate_pools": len(admitted.rate_pools)}
    publish(args.output / "acceptance.json", trace)
    print(json.dumps(trace, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
