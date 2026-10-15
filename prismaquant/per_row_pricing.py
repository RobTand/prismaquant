"""Per-row repricing for recorded allocator selections.

The allocator prices a raw ``(Linear, format)`` cost row through
``priced_cell.cost_entry_predicted_dloss``.  That precedence is
load-bearing: measured ``output_mse`` already includes the activation path,
whereas a weight-only row receives P5a's per-family multiplicative correction.
Aggregated serving-unit rows carry ``activation_pricing_applied=True`` because
their member prices already contain that correction; applying it again is the
historical double-counting bug this module exists to prevent.

This module deliberately does not aggregate, collapse, or promote an
assignment.  It reconstructs the price of exactly the layer-config rows it is
given, which makes it suitable both for reconciliation and for research tools
that need per-expert counterfactual prices.
"""
from __future__ import annotations

import json
import math
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import format_registry as fr
from .activation_fair_pricing import (
    BRANCH_ACTIVATION_IDENTITY,
    BRANCH_CALIBRATED,
    BRANCH_KILL_SWITCH,
    BRANCH_UNCALIBRATED,
    REASON_KILL_SWITCH,
)
from prismaquant.source_format_contracts import SOURCE_PASSTHROUGH_FORMATS
from prismaquant.priced_cell import (
    cost_entry_predicted_dloss,
    synthesized_source_passthrough_cost_entry,
)
from .layer_config import canonicalize_assignment


@dataclass(frozen=True)
class RecordedActivationFairPricing:
    """The P5a factors recorded in a selection artifact.

    Reconciliation must use the factors the allocator actually used, not
    silently refit them from a cost table that may have moved or gained rows
    since the selection was written.  Only ``penalty_for`` is needed by the
    authoritative cost-entry precedence.
    """

    enabled: bool
    reason: str
    penalties: Mapping[str, float]

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "RecordedActivationFairPricing":
        """Rehydrate the recorded run-level verdict from ``selection.json``."""
        raw = payload if isinstance(payload, Mapping) else {}
        families = raw.get("families")
        penalties = {
            str(family): float(entry["penalty"])
            for family, entry in (
                families.items() if isinstance(families, Mapping) else ()
            )
            if isinstance(entry, Mapping) and "penalty" in entry
        }
        return cls(
            enabled=bool(raw.get("enabled", False)),
            reason=str(raw.get("reason", "missing_recorded_pricing")),
            penalties=penalties,
        )

    def penalty_for(
        self,
        format_name: str | None,
        act_quant_changes_input: bool,
    ) -> tuple[float, str]:
        """Return the recorded multiplier and allocator branch label."""
        if not act_quant_changes_input:
            return 1.0, BRANCH_ACTIVATION_IDENTITY
        if self.reason == REASON_KILL_SWITCH:
            return 1.0, BRANCH_KILL_SWITCH
        try:
            family = str(fr.get_format(str(format_name)).family)
        except KeyError:
            return 1.0, BRANCH_UNCALIBRATED
        if family not in self.penalties:
            return 1.0, BRANCH_UNCALIBRATED
        return float(self.penalties[family]), BRANCH_CALIBRATED


def _cost_rows(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = payload.get("costs")
    return rows if isinstance(rows, Mapping) else payload


def _stats_rows(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = payload.get("stats")
    return rows if isinstance(rows, Mapping) else payload


def _resolve_cost_entry(
    cost_rows: Mapping[str, Any],
    format_name: str,
) -> tuple[Mapping[str, Any] | None, str]:
    for candidate_name in fr.aliases_for(format_name):
        entry = cost_rows.get(candidate_name)
        if isinstance(entry, Mapping):
            return entry, candidate_name
    if format_name in SOURCE_PASSTHROUGH_FORMATS:
        return synthesized_source_passthrough_cost_entry(format_name), format_name
    return None, format_name


def reprice_assignment(
    assignment: Mapping[str, Any],
    stats: Mapping[str, Any],
    costs: Mapping[str, Any],
    *,
    activation_pricing: Mapping[str, Any] | RecordedActivationFairPricing | None,
    calibrated_gains: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Price every layer-config row with the allocator's exact precedence.

    ``assignment`` may contain shorthand strings or full layer-config scheme
    dictionaries.  The result is keyed by canonical qname so callers can
    decompose the total without rerunning the pricing logic.  P5a is applied
    exactly once: raw weight-only rows receive the recorded family factor,
    while entries carrying ``activation_pricing_applied=True`` pass through
    unchanged via ``cost_entry_predicted_dloss``.
    """
    canonical = canonicalize_assignment(assignment)
    stat_rows = _stats_rows(stats)
    cost_rows = _cost_rows(costs)
    pricing = (
        RecordedActivationFairPricing.from_dict(activation_pricing)
        if isinstance(activation_pricing, Mapping)
        else activation_pricing
    )
    prices: dict[str, float] = {}
    missing: list[str] = []
    for qname, format_name in canonical.items():
        stat_entry = stat_rows.get(qname)
        if not isinstance(stat_entry, Mapping):
            missing.append(f"{qname}: missing stats")
            continue
        per_format = cost_rows.get(qname)
        if not isinstance(per_format, Mapping):
            missing.append(f"{qname}: missing cost row")
            continue
        try:
            prices[qname] = price_row(
                qname,
                format_name,
                stat_entry,
                per_format,
                activation_pricing=pricing,
                calibrated_gains=calibrated_gains,
            )
        except KeyError:
            missing.append(f"{qname}: missing cost for {format_name}")
    if missing:
        sample = "\n".join(f"    {item}" for item in missing[:12])
        raise KeyError(
            f"cannot reprice {len(missing)} assignment row(s):\n{sample}"
        )
    return prices


def price_row(
    qname: str,
    format_name: str,
    stats_entry: Mapping[str, Any],
    cost_rows: Mapping[str, Any],
    *,
    activation_pricing: Mapping[str, Any] | RecordedActivationFairPricing | None,
    calibrated_gains: Mapping[str, float] | None = None,
) -> float:
    """Price one canonical ``(qname, format)`` row.

    This is the scalar companion to :func:`reprice_assignment` for tools that
    build a candidate matrix.  It shares the same alias, source-passthrough,
    gain, applied-marker, and P5a semantics.
    """
    pricing = (
        RecordedActivationFairPricing.from_dict(activation_pricing)
        if isinstance(activation_pricing, Mapping)
        else activation_pricing
    )
    entry, entry_format = _resolve_cost_entry(cost_rows, format_name)
    if entry is None or "error" in entry:
        raise KeyError(f"{qname}: missing cost for {format_name}")
    gains = calibrated_gains or {}
    gain = float(gains.get(format_name, gains.get(entry_format, 1.0)))
    return cost_entry_predicted_dloss(
        dict(stats_entry),
        dict(entry),
        gain=gain,
        format_name=format_name,
        activation_pricing=pricing,
    )


def _recorded_artifact_path(
    selection_dir: Path,
    manifest: Mapping[str, Any],
    key: str,
    fallback_name: str,
) -> Path:
    recorded = manifest.get(key)
    if isinstance(recorded, str) and recorded:
        path = Path(recorded)
        if path.exists():
            return path
    fallback = selection_dir / fallback_name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"selection artifact has no readable {key!r} path and {fallback} "
        "does not exist"
    )


def reconcile_selection(selection_dir: str | Path) -> tuple[float, float, float]:
    """Return ``(reconstructed_dloss, recorded_dloss, ratio)`` for a cell.

    The selection's applicability manifest is the provenance authority for
    its probe and cost-table paths.  Source-passthrough rows are synthesized
    exactly as candidate construction synthesizes them, and the selection's
    own P5a factors are replayed without refitting.
    """
    root = Path(selection_dir)
    selection = json.loads((root / "selection.json").read_text())
    applicability = json.loads(
        (root / "format_applicability.json").read_text()
    )
    assignment = json.loads((root / "layer_config.json").read_text())
    probe_path = _recorded_artifact_path(root, applicability, "probe", "probe.pkl")
    cost_path = _recorded_artifact_path(root, applicability, "costs", "cost.pkl")
    with probe_path.open("rb") as handle:
        stats = pickle.load(handle)
    with cost_path.open("rb") as handle:
        costs = pickle.load(handle)
    prices = reprice_assignment(
        assignment,
        stats,
        costs,
        activation_pricing=selection.get("activation_fair_pricing"),
    )
    reconstructed = math.fsum(prices.values())
    recorded = float(selection["predicted_dloss"])
    if recorded == 0.0:
        ratio = 1.0 if reconstructed == 0.0 else math.inf
    else:
        ratio = reconstructed / recorded
    return reconstructed, recorded, ratio
