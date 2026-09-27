"""Add a constrained, measured MTP choice to an unchanged body allocation.

The existing MTP group-product selector owns the numerical choice. This
handoff only binds its explicit fixed group formats to a fresh body-only layer
config and then uses the existing M3 receipt backfill for exact cached wires.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import selection_serving_lane_provenance
from prismaquant.allocator import _mtp_rung_attestation
from prismaquant.cost_stage_checkpoint import publish_new_bytes
from prismaquant.footprint import (
    recursive_regular_file_bytes, whole_artifact_budget_from_assignment_payload,
    whole_artifact_budget_stamp,
)
from prismaquant.glm_mtp_selection import (
    backfill_mtp_selection_wires, load_mtp_cost, select_mtp_rungs,
)
from prismaquant.model_profiles import detect_profile
from prismaquant.layer_config import canonicalize_format
from prismaquant.serving_profiles import load_serving_profile
from prismaquant.tessera_serving_scope import (
    add_serving_scope_arguments, scope_provenance, serving_target_from_args,
    unit_structure_from_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-layer-config", required=True)
    parser.add_argument("--expect-body-sha256", required=True)
    parser.add_argument("--mtp-joint-cost", required=True)
    parser.add_argument("--expect-cost-sha256", required=True)
    parser.add_argument("--mtp-byte-budget", type=int, required=True)
    parser.add_argument("--whole-artifact-budget-bytes", type=int, required=True)
    parser.add_argument("--observed-body-export", required=True,
                        help="immutable completed body export used for reserve calibration")
    parser.add_argument("--new-metadata-allowance-bytes", type=int, required=True)
    parser.add_argument("--mtp-serve-constants", required=True)
    parser.add_argument("--fixed-formats", required=True,
                        help="JSON mapping of exact MTP group names to chosen formats")
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-profile", required=True)
    parser.add_argument("--output", required=True)
    add_serving_scope_arguments(parser)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; historical inputs are immutable")
    body_path = Path(args.body_layer_config)
    cost_path = Path(args.mtp_joint_cost)
    body_raw = body_path.read_bytes()
    if hashlib.sha256(body_raw).hexdigest() != args.expect_body_sha256:
        parser.error("body layer config differs from expected bytes")
    cost_raw = cost_path.read_bytes()
    if hashlib.sha256(cost_raw).hexdigest() != args.expect_cost_sha256:
        parser.error("MTP cost differs from expected bytes")
    del cost_raw
    body = json.loads(body_raw)
    if (not isinstance(body, dict) or
            not isinstance(body.get("__prismaquant__"), dict) or
            "mtp_selection" in body["__prismaquant__"] or
            any(".layers.45." in name for name in body if name != "__prismaquant__")):
        parser.error("base must be a body-only layer config without MTP selection")
    if body["__prismaquant__"].get("target_profile") != args.target_profile:
        parser.error("body target profile differs from MTP serving target")
    fixed = json.loads(Path(args.fixed_formats).read_bytes())
    constants = json.loads(Path(args.mtp_serve_constants).read_bytes())
    payload = load_mtp_cost(cost_path)
    if not isinstance(fixed, dict) or set(fixed) != set(payload["groups"]):
        parser.error("fixed formats must cover every MTP group exactly")
    profile = detect_profile(args.model)
    target = serving_target_from_args(
        args, target_platform=load_serving_profile(args.target_profile).target_platform)
    if target is None:
        parser.error("MTP serving target must be explicit")
    record = select_mtp_rungs(
        payload, byte_budget=args.mtp_byte_budget, constants=constants,
        fixed_formats=fixed, eligible=_mtp_rung_attestation(target, profile))
    assignment = record.pop("assignment")
    if set(assignment) & set(body):
        parser.error("MTP selection overlaps the body assignment")
    result = dict(body)
    for name, fmt in sorted(assignment.items()):
        result[name] = fr.get_format(fmt).autoround_config()
    body_meta = body["__prismaquant__"]
    body_assignment = {name: canonicalize_format(value) for name, value
                       in body.items() if name != "__prismaquant__"}
    original_budget = whole_artifact_budget_from_assignment_payload(
        body, where="MTP reselect body input", assignment=body_assignment)
    if original_budget is None:
        parser.error("base has no complete whole-artifact selection budget")
    observed_bytes = recursive_regular_file_bytes(args.observed_body_export)
    observed_overhead = (observed_bytes
                         - original_budget["selection_tensor_payload_bytes"])
    if observed_overhead < 0 or args.new_metadata_allowance_bytes < 0:
        parser.error("observed export overhead and new metadata allowance must be nonnegative")
    reserve_bytes = observed_overhead + args.new_metadata_allowance_bytes
    mtp_source_bytes = sum(2 * int(value) for value in payload["params"].values())
    selected_bytes = int(record["resident_bytes"])
    payload_bytes = (original_budget["selection_tensor_payload_bytes"]
                     - mtp_source_bytes + selected_bytes)
    complete_assignment = {name: canonicalize_format(value) for name, value
                           in result.items() if name != "__prismaquant__"}
    body_scope = body_meta.get("tessera_serving_scope")
    body_lane = body_meta.get("serving_lane_provenance")
    if (not isinstance(body_scope, dict) or
            not isinstance(body_scope.get("by_unit"), dict) or
            not isinstance(body_lane, dict) or
            not isinstance(body_lane.get("by_unit"), dict)):
        parser.error("body has no complete serving scope and route provenance")
    body_scope_names = set(body_scope["by_unit"])
    contexts = {name: target.context(unit_structure_from_profile(name, profile))
                for name in body_scope_names | set(assignment)}
    combined_scope = scope_provenance(target, contexts)
    if (combined_scope["target"] != body_scope["target"] or
            any(combined_scope["by_unit"][name] != body_scope["by_unit"][name]
                for name in body_scope_names)):
        parser.error("current serving target changes the body serving scope")
    combined_lane = selection_serving_lane_provenance(
        complete_assignment, target_profile=args.target_profile,
        context_by_unit=contexts)
    if any(combined_lane["by_unit"][name] != body_lane["by_unit"][name]
           for name in body_assignment):
        parser.error("current pinned runtime changes a body route")
    if any(payload["costs"][name][fmt].get("cost_source") != "joint_aura"
           for name, fmt in assignment.items() if fmt != "BF16"):
        parser.error("selected MTP activation-pricing branch is not joint AURA")
    old_branches = body_lane.get("activation_pricing_branches", {})
    if sum(old_branches.values()) != len(body_assignment):
        parser.error("body activation-pricing branch census is incomplete")
    combined_lane["activation_pricing_branches"] = {
        **old_branches,
        "joint_aura": old_branches.get("joint_aura", 0)
                      + sum(fmt != "BF16" for fmt in assignment.values()),
        "unrecorded": old_branches.get("unrecorded", 0)
                      + sum(fmt == "BF16" for fmt in assignment.values()),
    }
    result["__prismaquant__"] = {**body_meta,
        "tessera_serving_scope": combined_scope,
        "serving_lane_provenance": combined_lane,
        "assignment_payload_bits_total": (
            float(body_meta["assignment_payload_bits_total"])
            + 8.0 * selected_bytes),
        "whole_artifact_budget": whole_artifact_budget_stamp(
            budget_bytes=args.whole_artifact_budget_bytes,
            selection_tensor_payload_bytes=payload_bytes,
            selection_non_tensor_reserve_bytes=reserve_bytes,
            selection_assignment=complete_assignment,
            excluded_source_prefixes=(
                original_budget.get("excluded_source_prefixes") or ())),
        "mtp_selection": {**record, "cost_path": str(cost_path),
                          "units": len(assignment),
                          "budget_rebase": {
                              "schema": "prismaquant.mtp_budget_rebase.v1",
                              "body_export": str(Path(args.observed_body_export).resolve()),
                              "body_export_all_file_bytes": observed_bytes,
                              "body_selection_payload_bytes": original_budget[
                                  "selection_tensor_payload_bytes"],
                              "observed_overhead_bytes": observed_overhead,
                              "new_metadata_allowance_bytes": args.new_metadata_allowance_bytes,
                              "mtp_source_bf16_bytes": mtp_source_bytes,
                              "mtp_selected_wire_and_source_bytes": selected_bytes,
                          }}}
    result = backfill_mtp_selection_wires(result, cost_path)
    if any(result[name] != value for name, value in body.items()
           if name != "__prismaquant__"):
        raise ValueError("MTP selection changed a body format")
    raw = (json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if not publish_new_bytes(output, raw):
        parser.error("output already exists; refusing overwrite")
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "body_sha256": args.expect_body_sha256,
                      "mtp_cost_sha256": args.expect_cost_sha256,
                      "selected_mtp_wires": len(result["__prismaquant__"]["mtp_selection"][
                          "mtp_expert_wires"]),
                      "rung": record["rung"], "resident_bytes": record["resident_bytes"],
                      "E": record["E"],
                      "whole_artifact_payload_bytes": payload_bytes,
                      "whole_artifact_reserve_bytes": reserve_bytes,
                      "whole_artifact_budget_bytes": args.whole_artifact_budget_bytes},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
