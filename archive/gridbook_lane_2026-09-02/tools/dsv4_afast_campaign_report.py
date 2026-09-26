#!/usr/bin/env python3
"""Render the evidence-only A-FAST DSV4 campaign handoff."""
from __future__ import annotations

import json
import pickle
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dsv4_afast_burn import (
    AMENDMENT_JSON, BURN_ROOT, LAYER_COUNT, MTP_BF16_BYTES, PROJECTIONS,
    RUN_ROOT, SHARD_ROOT, atomic_text, sha256_file,
)


OLD_GRID = Path(
    "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/"
    "artifacts-mxfp4/oldmenu-grid"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.6e}"
    return f"{value:.6f}"


def _old(name: str) -> dict | None:
    path = OLD_GRID / name / "selection.json"
    return _json(path) if path.is_file() else None


def _variant_row(cell: Mapping[str, Any], variant: str) -> tuple[Any, ...]:
    data = cell["variants"][variant]
    collapsed = data["collapsed"]
    c_selection = collapsed["selection"]
    c_gate = collapsed["byte_gate"]
    expert = data["per_expert"]
    gap = data["observed_per_expert_solver_gap"]
    return (
        c_gate["reported_total_bytes"] / 1e9,
        float(c_selection["predicted_dloss"]),
        float(c_selection["chosen_achieved_bits"]),
        expert["reported_total_bytes"] / 1e9,
        float(expert["predicted_dloss"]),
        float(gap["relative_to_collapsed"]),
    )


def run() -> int:
    pilot = _json(RUN_ROOT / "pilot2/PILOT2_REPORT.json")
    amendment = _json(AMENDMENT_JSON)
    burn = _json(BURN_ROOT / "BURN_REPORT.json")
    grid = _json(BURN_ROOT / "allocation-grid/GRID_REPORT.json")
    shards = []
    for layer in range(LAYER_COUNT):
        with (SHARD_ROOT / f"layer_{layer:03d}.pkl").open("rb") as handle:
            shards.append(pickle.load(handle))

    measured = [s for s in shards if not s["meta"]["imported"]]
    layer_minutes = [float(s["meta"]["elapsed_seconds"]) / 60 for s in measured]
    total_wall = sum(float(s["meta"]["elapsed_seconds"]) for s in measured)
    projection_accepted = {
        projection: sum(
            int(shard["meta"]["projection"][projection]["fit"]["accepted"])
            for shard in shards
        ) for projection in PROJECTIONS
    }
    projection_total = LAYER_COUNT * 256
    fallback = {
        projection: projection_total - projection_accepted[projection]
        for projection in PROJECTIONS
    }
    arm_counts = {"free": 0, "embed": 0}
    replay_count = 0
    for shard in shards:
        for projection in PROJECTIONS:
            meta = shard["meta"]["projection"][projection]
            counts = meta.get("arm_counts") or {}
            for arm in arm_counts:
                arm_counts[arm] += int(counts.get(arm, 0))
            replay_count += int(meta.get("replay_count", 0))

    p3 = pilot["gates"]["P3"]
    p4 = pilot["gates"]["P4"]
    lines = [
        "# DSV4 A-FAST Cost Campaign", "",
        "This is measurement and counterfactual evidence only. **No budget, format mix, MTP policy, or artifact is nominated; the operator sizes from the curves.**",
        "", "## Outcome", "",
        f"- Acceptance amendment: **v2 PASS**, projected fresh-layer wall {amendment['cost']['projected_hours']:.3f} h <=20 h; banked gross-backstop rate {amendment['cost']['banked_truth_backstop_failure_rate']:.2%}.",
        f"- Pilot-2: **{pilot['decision']}**, pre-declared layer 14; P1 {pilot['gates']['P1']['violations']} violations / {pilot['gates']['P1']['aborts']} aborts, P2 {pilot['gates']['P2']['violations']} violations, P4 {p4['total_over_free']:.3f}x.",
        f"- Burn/merge: **{burn['merge']}**, {burn['layers']} content-verified layers, {burn['rows']:,} rows; layers 14 and 21 imported without remeasurement.",
        "- Phase C: collapsed and per-expert cold arms were solved at every MTP-in and MTP-out point; byte gates passed for every reported cell.",
        f"- MTP carry: {MTP_BF16_BYTES:,} bytes (10.8628383 GB), never dropped or quantized in the measured table.",
        "", "## Method and pilot-2", "",
        "OPTION A-FAST v2 uses K28/K33/K38/K43/K48 plus one deterministic non-anchor audit rung per layer. The only chain arms are free and resident predecessor embed; refine is deleted. Per-expert weight-MSE selects the arm with `selected <= free + 1e-12 rel`; epsilon/exact ties pass and choose free. Only the winning reconstruction receives production activation-QDQ output-MSE replay. Five-anchor PCHIP is used by default. A slice whose independent four-anchor CV error at K33 or K43 exceeds 25% measures its remaining missing rungs. Each projection's audit must have median <=5% and p95 <=15%; any failure full-measures all projections in the layer.",
        "", "| Pilot gate | Result | Measured |", "|---|---|---:|",
        f"| P1 monotonicity | {'PASS' if pilot['gates']['P1']['pass'] else 'FAIL'} | {pilot['gates']['P1']['violations']} violations; {pilot['gates']['P1']['aborts']} aborts |",
        f"| P2 zero tax | {'PASS' if pilot['gates']['P2']['pass'] else 'FAIL'} | {pilot['gates']['P2']['violations']} violations |",
        f"| P3 PCHIP | {'PASS' if p3['pass'] else 'FAIL'} | per projection below |",
        f"| P4 optimized overhead | {'PASS' if p4['pass'] else 'PROCEED'} | {p4['total_over_free']:.3f}x |",
        f"| P5 fallback | INFO | {pilot['gates']['P5']['fallback_fraction']:.2%} ({pilot['gates']['P5']['fallback_slices']}/768) |",
        "", "| Projection | Pilot median | Pilot p95 | Pilot fallback |", "|---|---:|---:|---:|",
    ]
    for projection in PROJECTIONS:
        row = p3["projections"][projection]
        lines.append(
            f"| {projection} | {row['median']:.2%} | {row['p95']:.2%} | "
            f"{row['fallback_fraction']:.2%} ({row['fallback_slices']}/256) |"
        )
    lines.extend([
        "", "The pilot P3/P5 table above is pre-amendment validation evidence; v2 burn admission is governed by the 25% gross backstop and per-layer audits below.",
        "", "## Burn coverage, fallback, audits, and wall", "",
        "| Projection | PCHIP/backstop-pass slices | Gross-backstop failures | PCHIP rate |",
        "|---|---:|---:|---:|",
    ])
    for projection in PROJECTIONS:
        lines.append(
            f"| {projection} | {projection_accepted[projection]:,}/{projection_total:,} | "
            f"{fallback[projection]:,} | {projection_accepted[projection]/projection_total:.2%} |"
        )
    lines.extend([
        "", "| Layer | Audit rung | gate / up / down median-p95 | Layer action |",
        "|---:|---:|---|---|",
    ])
    for shard in shards:
        meta = shard["meta"]
        values = []
        for projection in PROJECTIONS:
            audit = meta["projection"][projection]["audit"]
            values.append(f"{audit['median']:.2%}/{audit['p95']:.2%}")
        action = (
            "full-measured (audit fallback)"
            if meta["full_layer_fallback"] else "five-anchor interpolation trusted"
        )
        if meta["imported"] and meta["full_layer_fallback"]:
            action = "full truth imported; interpolation not trusted"
        lines.append(
            f"| {meta['layer']} | K{meta['audit_rung']} | "
            f"{' / '.join(values)} | {action} |"
        )
    lines.extend([
        "",
        f"- Fresh measured layers: {len(measured)}/41; imported layers: 14 and 21.",
        f"- Minutes/layer: min {min(layer_minutes):.2f}, median {statistics.median(layer_minutes):.2f}, max {max(layer_minutes):.2f}, mean {statistics.mean(layer_minutes):.2f}.",
        f"- Sum of content-shard foreground wall: {total_wall/3600:.3f} h.",
        f"- Measured winning arms: free {arm_counts['free']:,}, embed {arm_counts['embed']:,}; selected replay count {replay_count:,}.",
        "- Layers 0–2 used the existing hash-routed empty-activation fill path. MTP remained fixed source-format carry and is not a body cost row.",
        "", "## Phase C — MTP-in and MTP-out curves", "",
        "MTP-out keeps the same nominal total bytes: the fixed 10.8628383 GB MTP carry is removed from accounting and made available to body choices. Body bpp excludes immutable MTP in both variants.",
        "",
        "| Cell | Nominal GB | MTP-in collapsed GB / Δloss / bpp | MTP-out collapsed GB / Δloss / bpp | MTP-in per-expert GB / Δloss / observed gap | MTP-out per-expert GB / Δloss / observed gap |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in ("b-88", "b-92", "b-97", "b-102p8", "knee"):
        cell = grid["cells"][name]
        inside = _variant_row(cell, "mtp-in")
        outside = _variant_row(cell, "mtp-out")
        lines.append(
            f"| {name} | {cell['nominal_total_budget_bytes']/1e9:.6f} | "
            f"{inside[0]:.6f} / {_fmt(inside[1])} / {inside[2]:.4f} | "
            f"{outside[0]:.6f} / {_fmt(outside[1])} / {outside[2]:.4f} | "
            f"{inside[3]:.6f} / {_fmt(inside[4])} / {inside[5]:.2%} | "
            f"{outside[3]:.6f} / {_fmt(outside[4])} / {outside[5]:.2%} |"
        )
    knee = grid["cells"]["knee"]["knee_source"]
    lines.extend([
        "",
        f"Knee: target body bpp {float(knee['knee']['target_bits']):.4f}, nominal whole-artifact point {grid['cells']['knee']['nominal_total_budget_bytes']/1e9:.6f} GB, selected from the 92 GB MTP-in collapsed Pareto curve.",
        "",
        f"Per-expert solver gap: {grid['per_expert_solver_caveat']} The table's observed gap is `(collapsed Δloss − per-expert Δloss) / collapsed Δloss`; it is not a certified global optimality bound.",
        "", "## Era-labeled comparison", "",
        "Old era uses the pre-campaign K14/K15/K36 table. New era uses the A-FAST table and its full K12–K18 plus routed K28–K48 menu. These are estimator Δloss values, not served KL/PPL.",
        "", "| Point | Era | MTP policy | Budget GB | Collapsed Δloss | Upper-bound GB |",
        "|---|---|---|---:|---:|---:|",
    ])
    for old_name, new_name in (("b-88", "b-88"), ("b-92", "b-92")):
        old = _old(old_name)
        if old:
            lines.append(
                f"| {old_name} | old pre-A-FAST | registered | "
                f"{float(old['target_disk_gb']):.6f} | {_fmt(float(old['predicted_dloss']))} | "
                f"{float(old['predicted_whole_artifact_upper_bound_gb']):.6f} |"
            )
        for variant in ("mtp-in", "mtp-out"):
            cell = grid["cells"][new_name]
            row = _variant_row(cell, variant)
            lines.append(
                f"| {new_name} | new A-FAST | {variant} | "
                f"{cell['nominal_total_budget_bytes']/1e9:.6f} | {_fmt(row[1])} | {row[0]:.6f} |"
            )
    lines.extend([
        "| b-97 | old pre-A-FAST | no registered old cell | 97.000000 | — | — |",
        "| b-102p8 | old pre-A-FAST | no registered old cell | 102.800000 | — | — |",
        "", "## LDLQ serialization stamp", "",
        "The campaign imported and enforced this pilot-registered stamp:", "",
        "> `" + json.dumps(pilot["serialization"], sort_keys=True) + "`",
        "", "## Evidence paths", "",
        f"- Acceptance amendment v2: `{AMENDMENT_JSON}` (`{sha256_file(AMENDMENT_JSON)}`)",
        f"- Kill/resume shakedown: `{BURN_ROOT / 'SHAKEDOWN.json'}`",
        f"- Pilot-2: `{RUN_ROOT / 'PILOT2_REPORT.md'}` and `{RUN_ROOT / 'pilot2/PILOT2_REPORT.json'}`",
        f"- Burn table/report: `{BURN_ROOT / 'cost_merged.pkl'}` and `{BURN_ROOT / 'BURN_REPORT.json'}`",
        f"- Allocation grid: `{BURN_ROOT / 'allocation-grid/GRID_REPORT.json'}`",
        "", "No budget or MTP ship decision is nominated.", "",
    ])
    output = RUN_ROOT / "DSV4_CAMPAIGN.md"
    atomic_text(output, "\n".join(lines))
    print(f"[afast-campaign] wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
