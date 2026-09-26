"""Tier-2 per-expert allocation counterfactual (CPU-only research tool).

The production allocator correctly collapses each packed MoE serving group
and later verifies that serving-unit promotion is a no-op.  This module asks a
different, explicitly counterfactual question: how much predicted quality is
left on the table if a runtime could serve mixed formats within each packed
stack?  It therefore never calls the packed collapse or promotion passes.

Decisions remain physically meaningful at the expert level.  Each routed
expert's gate/up pair (DeepSeek ``gate_proj`` + ``up_proj``, conventionally
``w1`` + ``w3``) is one fused ``w13`` decision; ``down_proj``/``w2`` is a
separate decision.  Ordinary body Linears remain singleton decisions.

The solver is the requested Lagrangian generator: per-unit
``argmin(dloss + lambda * payload_bytes)``, lambda bisection to the byte
boundary, then an exact-payload tidy that accounts for shared CB sidecars only
once.  This is not a claim of global discrete-knapsack optimality; unsupported
non-convex frontier pockets are one reason the production allocator uses its
DP.  The output is proposal data and an upper bound on a runtime-achievable
prize because no mixed-stack launch-overhead penalty is priced.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import pickle
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import format_registry as fr
from .allocator_candidates import serialized_candidate_payload
from .allocator_solver import _shape_from_stats
from .nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_payload_breakdown,
    cb_serialization_context_from_stamp,
    is_cb_format,
)
from .per_row_pricing import RecordedActivationFairPricing, price_row

SCHEMA = "prismaquant.tier2_per_expert_counterfactual.v1"

_EXPERT_RE = re.compile(
    r"^(?P<stem>.*\.experts\.\d+\.)(?P<role>"
    r"gate_proj|up_proj|down_proj|w1|w2|w3)$"
)


class CounterfactualError(RuntimeError):
    """The recorded cell cannot support the requested counterfactual."""


@dataclass(frozen=True)
class RowChoice:
    """One legal format for one layer-config row."""

    fmt: str
    dloss: float
    payload_bytes: int
    sidecars: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class UnitChoice:
    """One common-format choice for a singleton or fused w13 unit."""

    fmt: str
    dloss: float
    payload_bytes: int
    sidecars: tuple[tuple[str, int], ...]
    row_dloss: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class DecisionUnit:
    """A lambda-argmin unit: body row, expert w13 pair, or expert w2 row."""

    name: str
    members: tuple[str, ...]
    choices: tuple[UnitChoice, ...]


@dataclass(frozen=True)
class LambdaSolution:
    """A feasible selected choice per decision unit."""

    selected: tuple[int, ...]
    dloss: float
    exact_bytes: int
    lambda_value: float
    tidy_changes: int = 0


def _format_below_k14(format_name: str) -> bool:
    match = re.fullmatch(r"NVFP4_CB_K(\d+)", str(format_name).upper())
    return bool(match and int(match.group(1)) < 14)


def expand_packed_expert_rows(
    stats: Mapping[str, Mapping[str, Any]],
    costs: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]],
           dict[str, tuple[str, ...]]]:
    """Expand packed ``[E, M, N]`` rows into per-expert Linear rows.

    The probe records an exact Fisher trace per expert and the packed cost
    writer records reconstruction MSE per expert from the same rendered
    stack. Together those are sufficient for the established weight-space
    surrogate without re-encoding each expert. Fused gate+up is split into
    two equal surrogate members, which is loss-preserving because the tier-2
    solver couples them as one ``w13`` decision.
    """
    cost_rows = (
        costs.get("costs")
        if isinstance(costs.get("costs"), Mapping)
        else costs
    )
    expanded_stats: dict[str, dict[str, Any]] = {}
    expanded_costs: dict[str, dict[str, Any]] = {}
    children_by_parent: dict[str, tuple[str, ...]] = {}

    for qname, raw_stat in stats.items():
        stat = dict(raw_stat)
        num_experts = int(stat.get("num_experts", 0) or 0)
        packed_param = str(stat.get("_packed_param", ""))
        traces = stat.get("h_trace_per_expert")
        per_format = cost_rows.get(qname)
        if not num_experts or packed_param not in {"gate_up_proj", "down_proj"}:
            expanded_stats[qname] = stat
            if isinstance(per_format, Mapping):
                expanded_costs[qname] = {
                    str(fmt): dict(entry)
                    for fmt, entry in per_format.items()
                    if isinstance(entry, Mapping)
                }
            continue
        if not isinstance(traces, Sequence) or len(traces) != num_experts:
            raise CounterfactualError(
                f"{qname}: packed tier-2 expansion requires "
                f"h_trace_per_expert[{num_experts}]"
            )
        if not isinstance(per_format, Mapping):
            raise CounterfactualError(f"{qname}: missing packed cost row")

        stem = qname[: -len(packed_param)].rstrip(".")
        roles = ("gate_proj", "up_proj") if packed_param == "gate_up_proj" \
            else ("down_proj",)
        split = len(roles)
        out_features = int(stat["out_features"]) // split
        n_params = int(stat["n_params"]) // num_experts // split
        children: list[str] = []
        for expert in range(num_experts):
            for role in roles:
                child = f"{stem}.{expert}.{role}"
                children.append(child)
                child_stat = dict(stat)
                child_stat.update({
                    "h_trace": float(traces[expert]) / split,
                    "n_params": n_params,
                    "out_features": out_features,
                    "num_experts": 0,
                    "expert_id": expert,
                    "_packed_parent": qname,
                })
                child_stat.pop("h_trace_per_expert", None)
                child_stat.pop("h_trace_per_expert_raw", None)
                child_stat.pop("_packed_experts_module", None)
                child_stat.pop("_packed_param", None)
                expanded_stats[child] = child_stat

                child_formats: dict[str, dict[str, Any]] = {}
                for fmt, raw_entry in per_format.items():
                    if not isinstance(raw_entry, Mapping):
                        continue
                    vector = raw_entry.get("weight_mse_per_expert")
                    if not isinstance(vector, Sequence) or len(vector) != num_experts:
                        raise CounterfactualError(
                            f"{qname}/{fmt}: packed tier-2 expansion requires "
                            f"weight_mse_per_expert[{num_experts}]"
                        )
                    entry = dict(raw_entry)
                    entry["weight_mse"] = float(vector[expert])
                    entry["output_mse_measured"] = False
                    entry["packed_per_expert_surrogate"] = True
                    for field in (
                        "weight_mse_per_expert", "output_mse",
                        "rel_output_mse", "fisher_output_mse",
                        "predicted_dloss",
                    ):
                        entry.pop(field, None)
                    child_formats[str(fmt)] = entry
                expanded_costs[child] = child_formats
        children_by_parent[qname] = tuple(children)
    return expanded_stats, expanded_costs, children_by_parent


def _is_expert_row(qname: str) -> bool:
    return _EXPERT_RE.fullmatch(qname) is not None


def apply_unmeasured_floor(
    row_choices: Mapping[str, Sequence[RowChoice]],
    stats: Mapping[str, Mapping[str, Any]],
    *,
    enabled: bool,
) -> dict[str, tuple[RowChoice, ...]]:
    """Optionally remove sub-K14 formats from zero-evidence expert rows."""
    out: dict[str, tuple[RowChoice, ...]] = {}
    for qname, choices in row_choices.items():
        floor = (
            enabled
            and _is_expert_row(qname)
            and float(stats[qname].get("h_trace", 0.0) or 0.0) == 0.0
        )
        kept = tuple(
            choice for choice in choices
            if not (floor and _format_below_k14(choice.fmt))
        )
        if not kept:
            raise CounterfactualError(
                f"{qname}: --floor-unmeasured removed every legal format"
            )
        out[qname] = kept
    return out


def _merge_sidecars(choices: Sequence[RowChoice]) -> tuple[tuple[str, int], ...]:
    merged: dict[str, int] = {}
    for choice in choices:
        for identity, size in choice.sidecars:
            previous = merged.setdefault(identity, size)
            if previous != size:
                raise CounterfactualError(
                    f"sidecar {identity!r} has conflicting sizes "
                    f"{previous} and {size}"
                )
    return tuple(sorted(merged.items()))


def _unit_from_members(
    name: str,
    members: Sequence[str],
    row_choices: Mapping[str, Sequence[RowChoice]],
) -> DecisionUnit:
    by_member = {
        member: {choice.fmt: choice for choice in row_choices[member]}
        for member in members
    }
    common = set.intersection(*(set(items) for items in by_member.values()))
    if not common:
        raise CounterfactualError(
            f"{name}: coupled members {list(members)} share no legal format"
        )
    choices: list[UnitChoice] = []
    for fmt in sorted(common):
        rows = [by_member[member][fmt] for member in members]
        choices.append(UnitChoice(
            fmt=fmt,
            dloss=math.fsum(row.dloss for row in rows),
            payload_bytes=sum(row.payload_bytes for row in rows),
            sidecars=_merge_sidecars(rows),
            row_dloss=tuple(
                (member, by_member[member][fmt].dloss) for member in members
            ),
        ))
    return DecisionUnit(
        name=name,
        members=tuple(members),
        choices=tuple(choices),
    )


def build_decision_units(
    row_choices: Mapping[str, Sequence[RowChoice]],
) -> tuple[DecisionUnit, ...]:
    """Build singleton rows plus explicit expert w1+w3/w13 coupling."""
    expert_roles: dict[str, dict[str, str]] = {}
    for qname in row_choices:
        match = _EXPERT_RE.fullmatch(qname)
        if match:
            expert_roles.setdefault(match.group("stem"), {})[
                match.group("role")
            ] = qname

    consumed: set[str] = set()
    units: list[DecisionUnit] = []
    for stem, roles in sorted(expert_roles.items()):
        pair = (
            ("gate_proj", "up_proj")
            if {"gate_proj", "up_proj"}.issubset(roles)
            else ("w1", "w3") if {"w1", "w3"}.issubset(roles) else None
        )
        if pair is not None:
            members = [roles[pair[0]], roles[pair[1]]]
            units.append(_unit_from_members(stem + "w13", members, row_choices))
            consumed.update(members)
        for role in ("down_proj", "w2"):
            qname = roles.get(role)
            if qname is not None:
                units.append(_unit_from_members(stem + "w2", [qname], row_choices))
                consumed.add(qname)

    for qname in sorted(set(row_choices) - consumed):
        units.append(_unit_from_members(qname, [qname], row_choices))
    return tuple(sorted(units, key=lambda unit: unit.name))


def _argmin_indices(
    units: Sequence[DecisionUnit],
    lambda_value: float,
) -> tuple[int, ...]:
    selected = []
    for unit in units:
        index = min(
            range(len(unit.choices)),
            key=lambda i: (
                unit.choices[i].dloss
                + lambda_value * unit.choices[i].payload_bytes,
                unit.choices[i].payload_bytes,
                unit.choices[i].dloss,
                unit.choices[i].fmt,
            ),
        )
        selected.append(index)
    return tuple(selected)


def _selection_totals(
    units: Sequence[DecisionUnit],
    selected: Sequence[int],
    fixed_bytes: int,
) -> tuple[float, int]:
    dloss = 0.0
    payload = int(fixed_bytes)
    sidecars: dict[str, int] = {}
    for unit, index in zip(units, selected):
        choice = unit.choices[index]
        dloss += choice.dloss
        payload += choice.payload_bytes
        for identity, size in choice.sidecars:
            previous = sidecars.setdefault(identity, size)
            if previous != size:
                raise CounterfactualError(
                    f"sidecar {identity!r} has inconsistent exact sizes"
                )
    payload += sum(sidecars.values())
    return dloss, payload


def _tidy_feasible_solution(
    units: Sequence[DecisionUnit],
    selected: Sequence[int],
    *,
    budget_bytes: int,
    fixed_bytes: int,
) -> tuple[tuple[int, ...], int]:
    """Spend exact payload slack on greedy loss-reducing boundary moves."""
    current = list(selected)
    current_dloss = 0.0
    current_bytes = int(fixed_bytes)
    sidecar_counts: Counter[str] = Counter()
    sidecar_sizes: dict[str, int] = {}
    for unit, index in zip(units, current):
        choice = unit.choices[index]
        current_dloss += choice.dloss
        current_bytes += choice.payload_bytes
        for identity, size in choice.sidecars:
            sidecar_counts[identity] += 1
            previous = sidecar_sizes.setdefault(identity, size)
            if previous != size:
                raise CounterfactualError(
                    f"sidecar {identity!r} has inconsistent exact sizes"
                )
    current_bytes += sum(
        sidecar_sizes[identity] for identity in sidecar_counts
    )

    def transition_totals(
        unit_index: int,
        candidate_index: int,
    ) -> tuple[float, int, Counter[str]]:
        old = units[unit_index].choices[current[unit_index]]
        new = units[unit_index].choices[candidate_index]
        counts_delta: Counter[str] = Counter()
        for identity, _size in old.sidecars:
            counts_delta[identity] -= 1
        for identity, size in new.sidecars:
            counts_delta[identity] += 1
            previous = sidecar_sizes.setdefault(identity, size)
            if previous != size:
                raise CounterfactualError(
                    f"sidecar {identity!r} has inconsistent exact sizes"
                )
        sidecar_delta = 0
        for identity, delta in counts_delta.items():
            before = sidecar_counts.get(identity, 0)
            after = before + delta
            if after < 0:
                raise CounterfactualError(
                    f"negative sidecar reference count for {identity!r}"
                )
            if before == 0 and after > 0:
                sidecar_delta += sidecar_sizes[identity]
            elif before > 0 and after == 0:
                sidecar_delta -= sidecar_sizes[identity]
        return (
            current_dloss - old.dloss + new.dloss,
            current_bytes - old.payload_bytes + new.payload_bytes
            + sidecar_delta,
            counts_delta,
        )
    heap: list[tuple[float, float, int, int]] = []
    for unit_index, unit in enumerate(units):
        old = unit.choices[current[unit_index]]
        for candidate_index, candidate in enumerate(unit.choices):
            saving = old.dloss - candidate.dloss
            if saving <= 0.0:
                continue
            added = max(candidate.payload_bytes - old.payload_bytes, 1)
            heapq.heappush(
                heap,
                (-saving / added, -saving, unit_index, candidate_index),
            )
    changed: set[int] = set()
    tidy_changes = 0
    while heap:
        _density, _saving, unit_index, candidate_index = heapq.heappop(heap)
        if unit_index in changed:
            continue
        proposed_dloss, proposed_bytes, counts_delta = transition_totals(
            unit_index, candidate_index)
        if proposed_dloss >= current_dloss or proposed_bytes > budget_bytes:
            continue
        current[unit_index] = candidate_index
        current_dloss = proposed_dloss
        current_bytes = proposed_bytes
        for identity, delta in counts_delta.items():
            sidecar_counts[identity] += delta
            if sidecar_counts[identity] == 0:
                del sidecar_counts[identity]
        changed.add(unit_index)
        tidy_changes += 1
    assert current_bytes <= budget_bytes
    return tuple(current), tidy_changes


def solve_lambda_bisection(
    units: Sequence[DecisionUnit],
    *,
    budget_bytes: int,
    fixed_bytes: int = 0,
    iterations: int = 96,
) -> LambdaSolution:
    """Solve per-unit lambda argmins, bisect, then tidy exact payload slack."""
    if not units:
        raise CounterfactualError("counterfactual has no decision units")
    cheapest = tuple(
        min(
            range(len(unit.choices)),
            key=lambda i: (
                unit.choices[i].payload_bytes,
                unit.choices[i].dloss,
                unit.choices[i].fmt,
            ),
        )
        for unit in units
    )
    _minimum_dloss, minimum_bytes = _selection_totals(
        units, cheapest, fixed_bytes)
    if minimum_bytes > budget_bytes:
        raise CounterfactualError(
            f"minimum exact payload {minimum_bytes} exceeds budget "
            f"{budget_bytes} by {minimum_bytes - budget_bytes} bytes"
        )

    zero = _argmin_indices(units, 0.0)
    zero_dloss, zero_bytes = _selection_totals(units, zero, fixed_bytes)
    if zero_bytes <= budget_bytes:
        return LambdaSolution(zero, zero_dloss, zero_bytes, 0.0, 0)

    low = 0.0
    high = 1.0e-15
    best_selected = cheapest
    best_dloss, best_bytes = _selection_totals(units, cheapest, fixed_bytes)
    while high < 1.0e15:
        selected = _argmin_indices(units, high)
        dloss, exact_bytes = _selection_totals(units, selected, fixed_bytes)
        if exact_bytes <= budget_bytes:
            best_selected, best_dloss, best_bytes = selected, dloss, exact_bytes
            break
        low = high
        high *= 2.0
    else:
        raise CounterfactualError("could not bracket a byte-feasible lambda")

    for _ in range(iterations):
        mid = (low + high) / 2.0
        selected = _argmin_indices(units, mid)
        dloss, exact_bytes = _selection_totals(units, selected, fixed_bytes)
        if exact_bytes <= budget_bytes:
            high = mid
            if dloss < best_dloss or (
                dloss == best_dloss and exact_bytes > best_bytes
            ):
                best_selected, best_dloss, best_bytes = (
                    selected, dloss, exact_bytes)
        else:
            low = mid

    tidied, tidy_changes = _tidy_feasible_solution(
        units,
        best_selected,
        budget_bytes=budget_bytes,
        fixed_bytes=fixed_bytes,
    )
    final_dloss, final_bytes = _selection_totals(units, tidied, fixed_bytes)
    return LambdaSolution(
        selected=tidied,
        dloss=final_dloss,
        exact_bytes=final_bytes,
        lambda_value=high,
        tidy_changes=tidy_changes,
    )


def expand_solution(
    units: Sequence[DecisionUnit],
    solution: LambdaSolution,
) -> tuple[dict[str, str], dict[str, float]]:
    """Expand unit choices into per-row formats and per-row Δloss."""
    assignment: dict[str, str] = {}
    prices: dict[str, float] = {}
    for unit, index in zip(units, solution.selected):
        choice = unit.choices[index]
        for member, row_price in choice.row_dloss:
            assignment[member] = choice.fmt
            prices[member] = row_price
    return assignment, prices


def formats_per_stack_distribution(assignment: Mapping[str, str]) -> dict:
    """Histogram distinct formats required by every layer's w13/w2 stack."""
    stacks: dict[str, set[str]] = {}
    for qname, fmt in assignment.items():
        match = _EXPERT_RE.fullmatch(qname)
        if not match:
            continue
        role = match.group("role")
        stack = "w13" if role in {"gate_proj", "up_proj", "w1", "w3"} else "w2"
        layer_match = re.search(r"\.layers\.(\d+)\.", qname)
        layer = layer_match.group(1) if layer_match else match.group("stem")
        stacks.setdefault(f"layer.{layer}.{stack}", set()).add(fmt)
    histogram = Counter(len(formats) for formats in stacks.values())
    return {
        "histogram": {str(k): v for k, v in sorted(histogram.items())},
        "stacks": {
            stack: sorted(formats) for stack, formats in sorted(stacks.items())
        },
    }


def _sidecar_for_single_row(
    qname: str,
    fmt: str,
    shape: tuple[int, ...],
    context: CBSerializationContext,
    row_bytes: int,
    identity: str | None,
) -> tuple[tuple[str, int], ...]:
    if identity is None:
        return ()
    breakdown = cb_assignment_payload_breakdown(
        {qname: fmt}, {qname: shape}, context=context)
    size = int(breakdown["codebook_sidecar_bytes"])
    assert int(breakdown["tensor_payload_bytes"]) == row_bytes
    return ((identity, size),)


def build_row_choices(
    stats: Mapping[str, Mapping[str, Any]],
    costs: Mapping[str, Any],
    *,
    menu: Sequence[str],
    denied_pairs: set[tuple[str, str]],
    activation_pricing: Mapping[str, Any] | RecordedActivationFairPricing,
    cb_context: CBSerializationContext,
) -> dict[str, tuple[RowChoice, ...]]:
    """Build per-row legal choices using a baseline cell's recorded masks."""
    cost_rows = (
        costs.get("costs")
        if isinstance(costs.get("costs"), Mapping)
        else costs
    )
    sidecar_cache: dict[str, int] = {}
    out: dict[str, tuple[RowChoice, ...]] = {}
    for qname, stat_entry in stats.items():
        per_format = cost_rows.get(qname)
        if not isinstance(per_format, Mapping):
            continue
        shape = _shape_from_stats(dict(stat_entry))
        choices: list[RowChoice] = []
        for fmt in menu:
            if (qname, fmt) in denied_pairs:
                continue
            try:
                dloss = price_row(
                    qname,
                    fmt,
                    stat_entry,
                    per_format,
                    activation_pricing=activation_pricing,
                )
            except KeyError:
                continue
            spec = fr.get_format(fmt)
            row_bytes, _identity, sidecar_identity = serialized_candidate_payload(
                spec,
                shape,
                qname=qname,
                cb_serialization_context=cb_context,
            )
            sidecars: tuple[tuple[str, int], ...] = ()
            if sidecar_identity is not None:
                if sidecar_identity not in sidecar_cache:
                    single = _sidecar_for_single_row(
                        qname,
                        fmt,
                        shape,
                        cb_context,
                        row_bytes,
                        sidecar_identity,
                    )
                    sidecar_cache[sidecar_identity] = single[0][1]
                sidecars = ((sidecar_identity, sidecar_cache[sidecar_identity]),)
            choices.append(RowChoice(
                fmt=fmt,
                dloss=max(float(dloss), 0.0),
                payload_bytes=int(row_bytes),
                sidecars=sidecars,
            ))
        if not choices:
            raise CounterfactualError(
                f"{qname}: no legal/priced formats remain from menu {list(menu)}"
            )
        out[qname] = tuple(choices)
    return out


def _load_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise CounterfactualError(f"{path}: expected a mapping payload")
    return payload


def run_counterfactual(
    *,
    baseline_dir: Path,
    probe_path: Path,
    cost_path: Path,
    budget_bytes: int,
    menu: Sequence[str],
    floor_unmeasured: bool,
) -> dict:
    """Load one recorded cell, solve it, and return a report-ready payload."""
    selection = json.loads((baseline_dir / "selection.json").read_text())
    applicability = json.loads(
        (baseline_dir / "format_applicability.json").read_text()
    )
    stats_payload = _load_pickle(probe_path)
    stats = stats_payload.get("stats")
    if not isinstance(stats, Mapping):
        raise CounterfactualError(f"{probe_path}: missing stats mapping")
    costs = _load_pickle(cost_path)
    stats, expanded_cost_rows, children_by_parent = expand_packed_expert_rows(
        stats, costs)
    costs = dict(costs)
    costs["costs"] = expanded_cost_rows
    recorded_menu = [str(item) for item in applicability.get("formats", ())]
    unknown = sorted(set(menu) - set(recorded_menu))
    if unknown:
        raise CounterfactualError(
            f"menu contains formats absent from the baseline legality record: "
            f"{unknown}; recorded={recorded_menu}"
        )
    denied_pairs = {
        (str(record["qname"]), str(record["format"]))
        for record in applicability.get("records", ())
        if isinstance(record, Mapping)
        and "qname" in record
        and "format" in record
    }
    denied_pairs.update(
        (child, fmt)
        for parent, fmt in tuple(denied_pairs)
        for child in children_by_parent.get(parent, ())
    )
    provenance = costs.get("provenance")
    stamp = (
        provenance.get("cb_serialized_payload")
        if isinstance(provenance, Mapping) else None
    )
    if not isinstance(stamp, Mapping):
        raise CounterfactualError("baseline has no CB serialization stamp")
    cb_context = cb_serialization_context_from_stamp(
        stamp, where=str(baseline_dir / "layer_config.json"))
    pricing = RecordedActivationFairPricing.from_dict(
        selection.get("activation_fair_pricing"))
    rows = build_row_choices(
        stats,
        costs,
        menu=menu,
        denied_pairs=denied_pairs,
        activation_pricing=pricing,
        cb_context=cb_context,
    )
    rows = apply_unmeasured_floor(
        rows, stats, enabled=floor_unmeasured)
    units = build_decision_units(rows)
    floor_bytes = int(round(float(selection["predicted_floor_gb"]) * 1.0e9))
    reserve_bytes = int(selection.get("artifact_overhead_reserve_bytes", 0) or 0)
    fixed_bytes = floor_bytes + reserve_bytes
    solution = solve_lambda_bisection(
        units,
        budget_bytes=int(budget_bytes),
        fixed_bytes=fixed_bytes,
    )
    assignment, row_prices = expand_solution(units, solution)
    if set(assignment) != set(stats):
        missing = sorted(set(stats) - set(assignment))
        raise CounterfactualError(
            f"expanded solution missed {len(missing)} row(s): {missing[:8]}"
        )

    # Independent final exact-payload verification through the producer's
    # assignment accountant, not the solver's maintained sidecar counters.
    cb_assignment = {
        qname: fmt for qname, fmt in assignment.items() if is_cb_format(fmt)
    }
    cb_shapes = {
        qname: _shape_from_stats(dict(stats[qname])) for qname in cb_assignment
    }
    cb_breakdown = cb_assignment_payload_breakdown(
        cb_assignment, cb_shapes, context=cb_context)
    non_cb_bytes = sum(
        fr.get_format(fmt).memory_bytes_for_shape(
            _shape_from_stats(dict(stats[qname])))
        for qname, fmt in assignment.items()
        if not is_cb_format(fmt)
    )
    independently_verified_bytes = (
        fixed_bytes
        + int(cb_breakdown["tensor_payload_bytes"])
        + int(cb_breakdown["codebook_sidecar_bytes"])
        + int(non_cb_bytes)
    )
    if independently_verified_bytes != solution.exact_bytes:
        raise CounterfactualError(
            "exact-payload tidy disagrees with producer accounting: "
            f"solver={solution.exact_bytes}, producer={independently_verified_bytes}"
        )
    if independently_verified_bytes > budget_bytes:
        raise CounterfactualError(
            f"final exact bytes {independently_verified_bytes} exceed budget "
            f"{budget_bytes}"
        )

    zero_rows = {
        qname for qname, entry in stats.items()
        if _is_expert_row(qname)
        and float(entry.get("h_trace", 0.0) or 0.0) == 0.0
    }
    baseline_dloss = float(selection["predicted_dloss"])
    histogram = Counter(assignment.values())
    return {
        "schema": SCHEMA,
        "arm": "floored" if floor_unmeasured else "free",
        "baseline_dir": str(baseline_dir),
        "probe": str(probe_path),
        "cost_table": str(cost_path),
        "menu": list(menu),
        "budget_bytes": int(budget_bytes),
        "fixed_floor_bytes": floor_bytes,
        "non_tensor_reserve_bytes": reserve_bytes,
        "exact_bytes": independently_verified_bytes,
        "headroom_bytes": int(budget_bytes) - independently_verified_bytes,
        "bytes_gate": independently_verified_bytes <= int(budget_bytes),
        "lambda": solution.lambda_value,
        "lambda_bisection_iterations": 96,
        "exact_payload_tidy_changes": solution.tidy_changes,
        "decision_units": len(units),
        "rows": len(assignment),
        "packed_rows_expanded": len(children_by_parent),
        "packed_per_expert_surrogate": bool(children_by_parent),
        "expert_zero_evidence_rows": len(zero_rows),
        "expert_zero_evidence_experts": len({
            _EXPERT_RE.fullmatch(qname).group("stem") for qname in zero_rows
        }),
        "protection_floor": {
            "enabled": bool(floor_unmeasured),
            "minimum": "NVFP4_CB_K14",
            "sub_k14_formats_in_menu": sorted(
                fmt for fmt in menu if _format_below_k14(fmt)
            ),
            "effective_on_this_menu": bool(
                floor_unmeasured
                and any(_format_below_k14(fmt) for fmt in menu)
            ),
        },
        "predicted_dloss": solution.dloss,
        "collapsed_baseline_predicted_dloss": baseline_dloss,
        "tier2_prize_dloss": baseline_dloss - solution.dloss,
        "tier2_prize_fraction": (
            (baseline_dloss - solution.dloss) / baseline_dloss
            if baseline_dloss else 0.0
        ),
        "format_histogram_rows": dict(sorted(histogram.items())),
        "formats_per_stack": formats_per_stack_distribution(assignment),
        "assignment": dict(sorted(assignment.items())),
        "per_row_dloss": dict(sorted(row_prices.items())),
        "zero_evidence_rows": sorted(zero_rows),
        "activation_fair_pricing": selection.get("activation_fair_pricing"),
        "claim_scope": (
            "UPPER BOUND on the mixed-stack tier-2 prize: predicted per-row "
            "dloss and exact producer payload bytes, with no launch-overhead "
            "penalty and no serving-stack implementation."
        ),
        "baseline_assignment_rows": len(stats),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--cost-table", type=Path, required=True)
    parser.add_argument("--budget", "--budget-bytes", dest="budget_bytes",
                        type=int, required=True)
    parser.add_argument("--menu", required=True,
                        help="comma-separated format menu")
    parser.add_argument("--floor-unmeasured", action="store_true",
                        help="forbid sub-K14 formats on h_trace == 0 experts")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    menu = tuple(item.strip() for item in args.menu.split(",") if item.strip())
    result = run_counterfactual(
        baseline_dir=args.baseline_dir,
        probe_path=args.probe,
        cost_path=args.cost_table,
        budget_bytes=args.budget_bytes,
        menu=menu,
        floor_unmeasured=args.floor_unmeasured,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "arm": result["arm"],
        "predicted_dloss": result["predicted_dloss"],
        "tier2_prize_dloss": result["tier2_prize_dloss"],
        "exact_bytes": result["exact_bytes"],
        "budget_bytes": result["budget_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
