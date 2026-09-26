#!/usr/bin/env python3
"""CPU-only PCHIP-consistent acceptance amendment for the DSV4 A-FAST burn."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from prismaquant.cb_minchain import select_arm
from tools.dsv4_afast_campaign import ANCHORS, REL_EPSILON, pchip_monotone
from tools.dsv4_ldlq_cost_campaign import (
    PROJECTIONS,
    RUNGS,
    RUN_ROOT,
    atomic_json,
    atomic_text,
    percentile,
    sha256_file,
)


LAYER14_TRUTH = RUN_ROOT / "pilot2/shards/layer_014.pkl"
LAYER21_TRUTH = RUN_ROOT / "pilot-shards"
PILOT2_REPORT = RUN_ROOT / "pilot2/PILOT2_REPORT.json"
OUTPUT_JSON = RUN_ROOT / "ACCEPTANCE_AMENDMENT.json"
OUTPUT_MD = RUN_ROOT / "ACCEPTANCE_AMENDMENT.md"
LAYERS = (14, 21)
HOLDOUTS = (33, 43)
BACKSTOP_TOLERANCE = 0.25
AUDIT_MEDIAN_TOLERANCE = 0.05
AUDIT_P95_TOLERANCE = 0.15
NONANCHORS = tuple(k for k in RUNGS if k not in ANCHORS)
FRESH_LAYERS = 41
REGISTERED_FIVE_FREE_RUNG_HOURS_43_LAYERS = 9.0


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _layer14_truth() -> dict[str, dict[int, list[float]]]:
    payload = _load(LAYER14_TRUTH)
    if int(payload.get("layer", -1)) != 14:
        raise AssertionError("layer-14 truth shard has the wrong layer")
    out: dict[str, dict[int, list[float]]] = {}
    for projection in PROJECTIONS:
        cells = payload["projections"][projection]["cells"]
        out[projection] = {
            rung: list(map(float, cells[rung]["selected_weight_mse"]))
            for rung in RUNGS
        }
    return out


def _layer21_truth() -> dict[str, dict[int, list[float]]]:
    """Reproduce the banked A-FAST free/embed winner without GPU work."""
    out: dict[str, dict[int, list[float]]] = {}
    for projection in PROJECTIONS:
        predecessor = [math.inf] * 256
        curves: dict[int, list[float]] = {}
        for rung in RUNGS:
            path = LAYER21_TRUTH / f"layer_021_{projection}_K{rung}.pkl"
            payload = _load(path)
            if (
                int(payload.get("layer", -1)) != 21
                or payload.get("projection") != projection
                or int(payload.get("rung", -1)) != rung
                or len(payload.get("weight_mse_per_expert", ())) != 256
            ):
                raise AssertionError(f"invalid layer-21 truth shard {path}")
            free = list(map(float, payload["weight_mse_per_expert"]))
            selected = []
            for expert, free_value in enumerate(free):
                if rung == RUNGS[0]:
                    value = free_value
                else:
                    _, value = select_arm(
                        {"free": free_value, "embed": predecessor[expert]},
                        rtol=REL_EPSILON,
                    )
                selected.append(float(value))
            curves[rung] = selected
            predecessor = selected
        out[projection] = curves
    return out


def _relative(prediction: float, truth: float) -> float:
    return abs(float(prediction) - float(truth)) / max(abs(float(truth)), 1e-30)


def _old_acceptance(curve: Mapping[int, Sequence[float]], expert: int) -> bool:
    fit = (28, 38, 48)
    predictions = pchip_monotone(
        fit, [curve[k][expert] for k in fit], HOLDOUTS,
    )
    return all(
        _relative(prediction, curve[holdout][expert]) <= 0.10
        for prediction, holdout in zip(predictions, HOLDOUTS)
    )


def _amended_acceptance(
    curve: Mapping[int, Sequence[float]], expert: int,
) -> tuple[bool, dict[str, float]]:
    errors: dict[str, float] = {}
    for holdout in HOLDOUTS:
        fit = tuple(k for k in ANCHORS if k != holdout)
        prediction = pchip_monotone(
            fit, [curve[k][expert] for k in fit], (holdout,),
        )[0]
        errors[f"K{holdout}"] = _relative(
            float(prediction), float(curve[holdout][expert]),
        )
    return all(value <= BACKSTOP_TOLERANCE for value in errors.values()), errors


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def _analyze_projection(
    curve: Mapping[int, Sequence[float]],
) -> dict[str, Any]:
    accepted_ids: list[int] = []
    rejected_ids: list[int] = []
    old_accepted = 0
    holdout_errors = {f"K{k}": [] for k in HOLDOUTS}
    accepted_quality: list[float] = []
    rejected_quality: list[float] = []
    for expert in range(256):
        old_accepted += int(_old_acceptance(curve, expert))
        accepted, local_holdouts = _amended_acceptance(curve, expert)
        for name, value in local_holdouts.items():
            holdout_errors[name].append(value)
        (accepted_ids if accepted else rejected_ids).append(expert)
        prediction = pchip_monotone(
            ANCHORS, [curve[k][expert] for k in ANCHORS], NONANCHORS,
        )
        destination = accepted_quality if accepted else rejected_quality
        destination.extend(
            _relative(value, curve[rung][expert])
            for value, rung in zip(prediction, NONANCHORS)
        )
    accepted_stats = _stats(accepted_quality)
    rejected_stats = _stats(rejected_quality)
    return {
        "accepted": len(accepted_ids),
        "rejected": len(rejected_ids),
        "acceptance_rate": len(accepted_ids) / 256,
        "accepted_expert_ids": accepted_ids,
        "rejected_expert_ids": rejected_ids,
        "old_accepted": old_accepted,
        "old_acceptance_rate": old_accepted / 256,
        "holdout_prediction_error": {
            name: _stats(values) for name, values in holdout_errors.items()
        },
        "accepted_nonanchor_quality": accepted_stats,
        "rejected_nonanchor_quality": rejected_stats,
        "rejected_is_more_aberrant": bool(
            accepted_stats["median"] is not None
            and accepted_stats["p95"] is not None
            and rejected_stats["median"] is not None
            and rejected_stats["p95"] is not None
            and float(rejected_stats["median"]) > float(accepted_stats["median"])
            and float(rejected_stats["p95"]) > float(accepted_stats["p95"])
        ),
        "quality_gate_pass": bool(
            accepted_stats["median"] is not None
            and accepted_stats["p95"] is not None
            and float(accepted_stats["median"]) <= 0.05
            and float(accepted_stats["p95"]) <= 0.15
        ),
    }


def _cost_hours(backstop_rate: float, chain_overhead: float) -> float:
    base = (
        REGISTERED_FIVE_FREE_RUNG_HOURS_43_LAYERS
        * FRESH_LAYERS / 43
        * chain_overhead
    )
    # Six rungs are measured for every slice (five anchors plus the audit).
    # A backstop failure measures the remaining 15 non-anchor rungs.
    return base * (
        6.0 / len(ANCHORS)
        + backstop_rate * (len(NONANCHORS) - 1) / len(ANCHORS)
    )


def _audit_rung(layer: int) -> int:
    return random.Random(42 + int(layer)).choice(NONANCHORS)


def _audit_projection(
    curve: Mapping[int, Sequence[float]], rung: int,
) -> dict[str, Any]:
    errors = []
    for expert in range(256):
        prediction = pchip_monotone(
            ANCHORS, [curve[k][expert] for k in ANCHORS], (rung,),
        )[0]
        errors.append(_relative(float(prediction), curve[rung][expert]))
    stats = _stats(errors)
    stats["gate_pass"] = bool(
        float(stats["median"]) <= AUDIT_MEDIAN_TOLERANCE
        and float(stats["p95"]) <= AUDIT_P95_TOLERANCE
    )
    return stats


def analyze() -> dict[str, Any]:
    truths = {14: _layer14_truth(), 21: _layer21_truth()}
    pilot = json.loads(PILOT2_REPORT.read_text())
    chain_overhead = float(pilot["gates"]["P4"]["total_over_free"])
    layers: dict[str, Any] = {}
    backstop_failures = old_rejected = total = 0
    for layer in LAYERS:
        layer_rows = {}
        audit_rung = _audit_rung(layer)
        audit_rows = {}
        for projection in PROJECTIONS:
            row = _analyze_projection(truths[layer][projection])
            layer_rows[projection] = row
            total += 256
            backstop_failures += int(row["rejected"])
            old_rejected += 256 - int(row["old_accepted"])
            audit_rows[projection] = _audit_projection(
                truths[layer][projection], audit_rung,
            )
        layers[str(layer)] = {
            "audit_rung": audit_rung,
            "projections": layer_rows,
            "audit": audit_rows,
            "audit_gate_pass": all(
                audit_rows[p]["gate_pass"] for p in PROJECTIONS
            ),
            "full_measurement_available": True,
        }
    backstop_rate = backstop_failures / total
    old_rejection_rate = old_rejected / total
    new_hours = _cost_hours(backstop_rate, chain_overhead)
    cost_pass = new_hours <= 20.0
    return {
        "schema": "prismaquant.dsv4_acceptance_amendment.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule": {
            "anchors": list(ANCHORS),
            "holdouts": list(HOLDOUTS),
            "method": (
                "accept all except gross outliers: independently predict K33 "
                "and K43 from the other four anchors and full-measure a slice's "
                "remaining missing rungs if either relative error exceeds 25%"
            ),
            "interpolation": "five-anchor monotone PCHIP",
            "audit": (
                "one deterministic non-anchor rung per layer, seed 42+layer; "
                "median<=5% and p95<=15% per projection, otherwise full-measure layer"
            ),
            "scored_nonanchor_rungs": list(NONANCHORS),
        },
        "inputs": {
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
            str(LAYER14_TRUTH): sha256_file(LAYER14_TRUTH),
            str(PILOT2_REPORT): sha256_file(PILOT2_REPORT),
            "layer21_truth_shards_digest": hashlib.sha256("".join(
                sha256_file(
                    LAYER21_TRUTH / f"layer_021_{projection}_K{rung}.pkl"
                )
                for projection in PROJECTIONS for rung in RUNGS
            ).encode()).hexdigest(),
        },
        "layers": layers,
        "cost": {
            "basis": "registered 9h/43-layer five-free-rung campaign basis",
            "fresh_layers": FRESH_LAYERS,
            "pilot2_chain_overhead": chain_overhead,
            "base_five_anchor_hours": (
                REGISTERED_FIVE_FREE_RUNG_HOURS_43_LAYERS
                * FRESH_LAYERS / 43 * chain_overhead
            ),
            "always_measured_rungs_per_slice": 6,
            "fallback_missing_rungs": len(NONANCHORS) - 1,
            "banked_truth_backstop_failure_rate": backstop_rate,
            "banked_truth_backstop_failures": backstop_failures,
            "banked_truth_slices": total,
            "projected_hours": new_hours,
        },
        "gate": {
            "cost_pass": cost_pass,
            "pass": cost_pass,
            "audit_failure_action": "full_measure_entire_layer",
            "audit_thresholds": {
                "median": AUDIT_MEDIAN_TOLERANCE,
                "p95": AUDIT_P95_TOLERANCE,
            },
            "cost_threshold_hours": 20.0,
        },
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "## Amendment v2 — accept-all plus per-layer audit", "",
        f"Generated: {report['created_at']}. CPU-only; no new GPU measurements.", "",
        "### Operator decision and justification", "",
        "Per-slice acceptance gating is retired. The v1 banked-truth table showed "
        "that every rejected group was itself under the 5%/15% quality limits; "
        "quoted unchanged:", "",
        "| Layer | Projection | Rejected median / p95 |",
        "|---:|---|---:|",
        "| 14 | `gate_proj` | 2.88% / 14.73% |",
        "| 14 | `up_proj` | 2.63% / 13.98% |",
        "| 14 | `down_proj` | 2.93% / 12.93% |",
        "| 21 | `gate_proj` | 2.80% / 12.07% |",
        "| 21 | `up_proj` | 2.98% / 14.24% |",
        "| 21 | `down_proj` | 2.53% / 11.88% |", "",
        "The min-chain envelope therefore produced no aberrant subpopulation for "
        "a fine-grained admission gate to isolate. V2 accepts all slices for "
        "five-anchor interpolation except a gross-outlier backstop: independent "
        "four-anchor CV errors at K33 and K43 must each be <=25%. A failure "
        "full-measures that slice's remaining missing rungs.", "",
        "Every layer also fully measures one deterministic non-anchor audit rung "
        "drawn with seed `42 + layer`. Five-anchor PCHIP is scored against that "
        "truth per projection. Median <=5% and p95 <=15% passes; any projection "
        "failure forces full measurement of the entire layer.", "",
        "### Banked-truth backstop and audit", "",
        "| Layer | Projection | Backstop failures | Audit rung | Audit median / p95 | Audit action |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for layer in LAYERS:
        layer_row = report["layers"][str(layer)]
        for projection in PROJECTIONS:
            row = layer_row["projections"][projection]
            audit = layer_row["audit"][projection]
            lines.append(
                f"| {layer} | `{projection}` | {row['rejected']}/256 "
                f"({row['rejected']/256:.2%}) | K{layer_row['audit_rung']} | "
                f"{_pct(audit['median'])} / {_pct(audit['p95'])} | "
                f"{'interpolate' if audit['gate_pass'] else 'full layer (truth already banked)'} |"
            )
    cost = report["cost"]
    lines.extend([
        "", "Layer 21 fails the audit median on all three projections, so v2 "
        "marks it full-measure. Its 21-rung truth is already banked and is imported "
        "content-keyed; no interpolation is trusted for that layer.", "",
        "### Projected 41-layer burn cost", "",
        f"- Cost basis: {cost['basis']}; pilot-2 measured chain overhead "
        f"{cost['pilot2_chain_overhead']:.6f}x.",
        f"- Five-anchor base over 41 fresh layers: {cost['base_five_anchor_hours']:.3f} h.",
        f"- Banked-truth gross-backstop rate: {cost['banked_truth_backstop_failure_rate']:.2%} "
        f"({cost['banked_truth_backstop_failures']}/{cost['banked_truth_slices']} slices).",
        f"- Six always-measured rungs plus 15-rung fallback at that observed rate: "
        f"**{cost['projected_hours']:.3f} h**.", "",
        "Projection formula: `five-anchor base * (6/5 + backstop_rate * 15/5)`. "
        "The pilot-2 free/embed chain overhead is already applied. Audit-triggered "
        "full-layer fallback is a correctness action and is reported when observed; "
        "the mandatory three-shard wall projection remains the runtime stop gate.", "",
        "### Gate", "",
        f"- Projected fresh-layer burn <=20 h: **{'PASS' if report['gate']['cost_pass'] else 'FAIL'}**.",
        f"- Overall: **{'PASS' if report['gate']['pass'] else 'FAIL'}**.", "",
        "V2 changes only interpolation safeguards. Free/embed min-chain construction, "
        "weight-MSE arm selection, winner-only activation-QDQ replay, content-keyed "
        "persistence, RSS/cache hygiene, layer 14/21 imports, MTP treatment, phase-C "
        "grid, the 20-hour/three-shard gates, and no-nomination policy are unchanged.", "",
        "Machine-readable v2 evidence: `ACCEPTANCE_AMENDMENT.json`.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    report = analyze()
    atomic_json(OUTPUT_JSON, report)
    prior = OUTPUT_MD.read_text() if OUTPUT_MD.is_file() else "# DSV4 acceptance amendment\n"
    marker = "## Amendment v2 — accept-all plus per-layer audit"
    prior = prior.split(marker, 1)[0].rstrip()
    atomic_text(OUTPUT_MD, prior + "\n\n" + markdown(report))
    print(json.dumps(report["gate"], sort_keys=True))
    print(OUTPUT_MD)
    return 0 if report["gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
