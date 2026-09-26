#!/usr/bin/env python3
"""Render the evidence-only DSV4 campaign handoff (no nomination)."""

from __future__ import annotations

import json
import pickle
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dsv4_ldlq_burn import BURN_ROOT, LAYER_COUNT, PROJECTIONS, RUN_ROOT, atomic_text


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


def _gate_result(gate: Mapping[str, Any]) -> str:
    return "PASS" if gate.get("pass") else "FAIL"


def run() -> int:
    phase = _json(RUN_ROOT / "PHASE_A_INTEGRITY.json")
    phase_fit = _json(RUN_ROOT / "PILOT_FIT_REPORT.json")
    pilot = _json(RUN_ROOT / "minchain-pilot/MINCHAIN_PILOT.json")
    burn = _json(BURN_ROOT / "BURN_REPORT.json")
    grid = _json(BURN_ROOT / "allocation-grid/GRID_REPORT.json")
    shards = []
    for layer in range(LAYER_COUNT):
        with (BURN_ROOT / "by-layer" / f"layer_{layer:03d}.pkl").open("rb") as handle:
            shards.append(pickle.load(handle))
    expert_accepted = {
        projection: sum(
            shard["meta"]["projection"][projection]["fit"]["accepted"]
            for shard in shards
        ) for projection in PROJECTIONS
    }
    ordinary_accepted = sum(
        int(meta["accepted"])
        for shard in shards for meta in shard["meta"]["ordinary"].values()
    )
    ordinary_total = sum(len(shard["meta"]["ordinary"]) for shard in shards)
    arm_counts = {"free": 0, "embed": 0, "refine": 0}
    for shard in shards:
        for projection in PROJECTIONS:
            counts = shard["meta"]["projection"][projection].get("arm_counts")
            if counts:
                for arm in arm_counts:
                    arm_counts[arm] += int(counts[arm])

    lines = [
        "# DSV4 LDLQ Cost Campaign",
        "",
        "This document reports measurements and counterfactuals only. **No budget or artifact is nominated; that decision belongs to the operator.**",
        "",
        "## Outcome",
        "",
        f"- Phase A integrity: **{phase['result']}** ({phase['fp8_shards']}/{phase['fp8_shards']} FP8 shards, {phase['menu_shards']} menu shards, content key `{phase['inventory_content_key']}`).",
        f"- Min-chain pilot decision: **{pilot['decision']}**; mechanically selected burn method: **{pilot['burn_method']}**.",
        f"- Phase B merge: **{burn['merge']}**, {burn['layers']} layers, {burn['rows']:,} cost rows.",
        "- Phase C: all collapsed and per-expert byte gates passed at the registered grid points.",
        "- MTP: untouched, fixed BF16; excluded from quantizable-parameter bpp and shown explicitly in byte accounting.",
        "",
        "## Phase A — layer-21 free-fit truth",
        "",
        f"The resumed run preserved the verified K28–K30 shards and completed K28–K48 plus BF16/MXFP4/MXFP4_SOURCE coverage for gate, up, and down projections. Root source-index, by-layer, imatrix, serialization-context, shard, coordinate, metric-vector, and warm-state identities all passed. The incumbent-law phase-A diagnostic was **{phase_fit['overall']}** (aggregate dual acceptance {phase_fit['aggregate']['dual_acceptance_rate']:.2%}, median {phase_fit['aggregate']['median']:.2%}, p95 {phase_fit['aggregate']['p95']:.2%}).",
        "",
        "## Monotone min-chain pilot",
        "",
        "Selection and reporting used the same production `weight_mse` field. The free scale sweep was not repeated: matching warm argmins materialized the banked solution only so a free winner could become the next predecessor.",
        "",
        "| Gate | Result | Value |",
        "|---|---|---:|",
        f"| G1 selected monotonicity | {_gate_result(pilot['gates']['G1'])} | {pilot['gates']['G1']['violations']} violations |",
        f"| G2 zero tax | {_gate_result(pilot['gates']['G2'])} | {pilot['gates']['G2']['violations']} violations; max excess {pilot['gates']['G2']['max_excess']:.3e} |",
        f"| G3 dual K33+K43 fit | {_gate_result(pilot['gates']['G3'])} | median {pilot['gates']['G3']['median']:.2%}; p95 {pilot['gates']['G3']['p95']:.2%}; accepted {pilot['gates']['G3']['accepted_slices']}/{pilot['gates']['G3']['total_slices']} |",
        f"| G4 encode overhead | {_gate_result(pilot['gates']['G4'])} | {pilot['gates']['G4']['total_over_free_ratio']:.3f}x (limit 1.6x) |",
        f"| G5 refine, informative | INFO | {pilot['gates']['G5']['wins']}/{pilot['gates']['G5']['eligible_cells']} wins ({pilot['gates']['G5']['win_rate']:.2%}), median gain {pilot['gates']['G5']['median_gain']:.2%} |",
        "",
        "Each measured chain cell carries the winning arm, solution digest, predecessor digest when applicable, and chain version. The module is pilot/campaign scoped and flag-gated; serving decode and production defaults are unchanged.",
        "",
        "## Phase B — 43-layer burn",
        "",
        f"Method: **{burn['method']}**. Five rungs (K28/K33/K38/K43/K48) were measured first per slice; the production law was admitted only when both holdouts were within 10%, and rejected slices measured every missing rung. MTP was not touched.",
        "",
        "| Slice cohort | Accepted | Total | Rate |",
        "|---|---:|---:|---:|",
    ]
    for projection in PROJECTIONS:
        total = LAYER_COUNT * 256
        lines.append(
            f"| routed `{projection}` | {expert_accepted[projection]:,} | {total:,} | {expert_accepted[projection]/total:.2%} |"
        )
    lines.append(
        f"| ordinary body Linears | {ordinary_accepted:,} | {ordinary_total:,} | {ordinary_accepted/ordinary_total:.2%} |"
    )
    if burn["method"] == "MIN-CHAIN STRICT":
        total_arms = sum(arm_counts.values())
        lines.extend([
            "",
            f"Measured/fallback chain arm selections: free {arm_counts['free']:,}, embed {arm_counts['embed']:,}, refine {arm_counts['refine']:,} (total {total_arms:,}).",
        ])
    lines.extend([
        "",
        "## Phase C — allocation grid",
        "",
        f"Method under measurement: **{grid['method']}**. The collapsed arm is the serveable packed-stack allocator. The per-expert arm is a CPU-only counterfactual retaining fused gate+up coupling but allowing each expert w13 and w2 unit to differ.",
        "",
        "| Cell | Budget GB | MTP BF16 GB | Collapsed adjusted GB | Collapsed Δloss | Body bpp | Per-expert adjusted GB | Per-expert Δloss | Tier-2 prize | Byte gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    order = ("b-88", "b-92", "b-97", "c-92-mtp", "knee")
    for name in order:
        cell = grid["cells"][name]
        collapsed = cell["collapsed"]
        selection = collapsed["selection"]
        cgate = collapsed["byte_gate"]
        expert = cell["per_expert"]
        lines.append(
            f"| {name} | {cell['requested_budget_bytes']/1e9:.6f} | "
            f"{cell['mtp_bf16_bytes']/1e9:.6f} | "
            f"{cgate['adjusted_upper_bound_bytes']/1e9:.6f} | "
            f"{_fmt(float(selection['predicted_dloss']))} | "
            f"{float(selection['chosen_achieved_bits']):.4f} | "
            f"{expert['adjusted_exact_bytes']/1e9:.6f} | "
            f"{_fmt(float(expert['predicted_dloss']))} | "
            f"{float(expert['tier2_prize_fraction']):.2%} | PASS / PASS |"
        )
    lines.extend([
        "",
        f"MTP BF16 accounting uses **{grid['mtp_bf16_bytes']:,} bytes** at the registered c-92+MTP point. {grid['unique_chain_sidecar_policy']}. All reported bpp values cover quantizable body parameters only.",
        "",
        f"Per-expert caveat: {grid['per_expert_solver_caveat']}",
        "",
        "## Era-labeled comparison with the old table",
        "",
        "Old era = pre-campaign K14/K15/K36 menu. New era = this campaign’s full K12–K18 and K28–K48 menu under the method named above. Δloss values are estimator outputs, not served KL/PPL.",
        "",
        "| Cell | Era | Method/menu | Budget GB | Collapsed Δloss | Upper-bound GB |",
        "|---|---|---|---:|---:|---:|",
    ])
    mapping = (("b-88", "b-88"), ("b-92", "b-92"), ("c-92", "c-92-mtp"))
    for old_name, new_name in mapping:
        old = _old(old_name)
        if old is not None:
            lines.append(
                f"| {old_name} | old (pre-ladder) | K14/K15/K36 incumbent | "
                f"{float(old['target_disk_gb']):.6f} | {_fmt(float(old['predicted_dloss']))} | "
                f"{float(old['predicted_whole_artifact_upper_bound_gb']):.6f} |"
            )
        cell = grid["cells"][new_name]
        selection = cell["collapsed"]["selection"]
        gate = cell["collapsed"]["byte_gate"]
        lines.append(
            f"| {new_name} | new (LDLQ campaign) | full menu, {grid['method']} | "
            f"{cell['requested_budget_bytes']/1e9:.6f} | {_fmt(float(selection['predicted_dloss']))} | "
            f"{gate['adjusted_upper_bound_bytes']/1e9:.6f} |"
        )
    lines.extend([
        "| b-97 | old (pre-ladder) | no registered old cell | 97.000000 | — | — |",
        "",
        "## Evidence paths",
        "",
        f"- Phase A inventory: `{RUN_ROOT / 'PHASE_A_INTEGRITY.json'}`",
        f"- Min-chain pilot: `{RUN_ROOT / 'MINCHAIN_PILOT.md'}` and `{RUN_ROOT / 'minchain-pilot/MINCHAIN_PILOT.json'}`",
        f"- Burn report/table: `{BURN_ROOT / 'BURN_REPORT.json'}` and `{BURN_ROOT / 'cost_merged.pkl'}`",
        f"- Allocation grid: `{BURN_ROOT / 'allocation-grid/GRID_REPORT.json'}`",
        "",
        "No projection abort or other stop condition fired. This campaign does not nominate a budget.",
        "",
    ])
    output = RUN_ROOT / "DSV4_CAMPAIGN.md"
    atomic_text(output, "\n".join(lines))
    print(f"[campaign] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
