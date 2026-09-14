"""Put a packed-expert Fisher probe on a Tessera census's per-expert rows.

A Tessera census prices every routed expert projection as its own unit
(``...experts.7.gate_proj``) under the producer's carried expert projection.
A packed probe measures the same Fisher on the packed parameter
(``...experts.gate_up_proj``) and keeps one trace per expert.  The allocator
joins stats to costs by name, so without this expansion every routed census
row has no sensitivity and is silently skipped.

The expansion is exact bookkeeping, not an estimate:

* the child trace is the parent's per-expert trace divided by the number of
  projections the packed parameter holds (``gate_up_proj`` -> gate and up).
  The Fisher trace of a packed row is a sum over its elements, and the gate
  and up halves are not measured apart, so the even split is the only value
  the probe supports.  Every routed row this touches is priced by a measured
  census ``output_mse``, so the split enters only as the h_trace factor;
* geometry comes from the census's carried projection, not from the probe,
  and must multiply back to the parent's parameter count;
* the stack-level aggregates (per-expert activation marginals, weight norms,
  ``h_w2_sum``) have no per-expert value on this row and are dropped, never
  divided.  The expanded probe therefore carries no marginals and says so;
* dense census rows pass through unchanged; probe rows the census does not
  price are dropped by name and recorded.

The existing ``expand_packed_expert_rows`` is not used: it removes the cost
side's ``output_mse`` and stamps ``output_mse_measured=False``, which would
turn every measured census cell into an unmeasured one.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import numpy as np

from .tessera_expert_projection import PROJECTION_KEY, carried_units

SCHEMA = "prismaquant.tessera_census_stats.v1"
META_KEY = "tessera_census_stats"

#: The probe accumulates per-expert Fisher in float32; summing E float32
#: values can move the total by at most E * eps relative to the stored parent.
_FLOAT32_EPS = float(np.finfo(np.float32).eps)

#: Row fields copied to every child unchanged.
_CARRIED_FIELDS = ("h_trace_norm_tokens", "route_prob", "router_path")
#: Row fields the child rewrites from the census geometry or the per-expert trace.
_REWRITTEN_FIELDS = ("h_trace", "h_trace_raw", "n_params", "in_features",
                     "out_features", "num_experts", "expert_id", "n_tokens_seen")
#: Stack-level fields with no per-expert value on this row.
DROPPED_FIELDS = ("_packed_experts_module", "_packed_param", "h_trace_per_expert",
                  "h_trace_per_expert_raw", "expert_g_sq_sum", "expert_act_sq_sum",
                  "expert_act_absmax", "expert_tokens", "h_w2_sum", "h_w2_sum_raw",
                  "w_max_abs", "w_norm_sq")


class CensusStatsError(ValueError):
    """The probe cannot be placed on the census rows without guessing."""


def _is_packed(row: Mapping[str, Any]) -> bool:
    return ("_packed_param" in row or "_packed_experts_module" in row
            or int(row.get("num_experts") or 0) > 0)


def _sum_within_float32(total: float, parts: list[float], *, where: str) -> None:
    measured = math.fsum(parts)
    bound = len(parts) * _FLOAT32_EPS * abs(total)
    if abs(measured - total) > bound:
        raise CensusStatsError(
            f"{where}: per-expert traces sum to {measured!r}, parent is {total!r} "
            f"(bound {bound!r} = {len(parts)} x float32 eps x parent)")


def expand_probe_onto_census(probe: Mapping[str, Any], cost: Mapping[str, Any], *,
                             structure: Any) -> dict:
    """Return a probe whose ``stats`` are keyed exactly to ``cost["costs"]``.

    ``structure`` is the model's structure spec; its
    ``packed_expert_projection_names`` says which census projections one
    packed parameter holds.  Refuses by name on any row it cannot place.
    """
    stats = probe.get("stats")
    costs = cost.get("costs")
    if not isinstance(stats, Mapping) or not isinstance(costs, Mapping):
        raise CensusStatsError("probe needs stats and cost table needs costs")
    carried = (cost.get("provenance") or {}).get(PROJECTION_KEY)
    _source, units, stack_of = carried_units(carried)
    census = set(costs)
    stray = sorted(set(units) - census)
    if stray:
        raise CensusStatsError(
            f"{len(stray)} projected units have no census row (first: {stray[0]})")
    dense = census - set(units)

    # (stack, projection) -> {expert: unit name}
    slots: dict[tuple[str, str], dict[int, str]] = {}
    for name, unit in units.items():
        slot = slots.setdefault((stack_of[name], unit["projection"]), {})
        if unit["expert"] in slot:
            raise CensusStatsError(f"{name}: duplicate expert {unit['expert']} in its stack")
        slot[int(unit["expert"])] = name

    out: dict[str, dict] = {}
    consumed: set[tuple[str, str]] = set()
    dropped_rows: list[str] = []
    splits: dict[str, int] = {}
    for row_name, row in sorted(stats.items()):
        if not _is_packed(row):
            if row_name in units:
                raise CensusStatsError(
                    f"{row_name}: probe has an unpacked row for a projected routed unit")
            if row_name in dense:
                out[row_name] = copy.deepcopy(dict(row))
            else:
                dropped_rows.append(row_name)
            continue
        unknown = sorted(set(row) - set(_CARRIED_FIELDS) - set(_REWRITTEN_FIELDS)
                         - set(DROPPED_FIELDS))
        if unknown:
            raise CensusStatsError(
                f"{row_name}: packed row carries fields this expansion does not place: {unknown}")
        stack = row.get("_packed_experts_module")
        param = row.get("_packed_param")
        if not isinstance(stack, str) or not isinstance(param, str):
            raise CensusStatsError(f"{row_name}: packed row does not name its stack and parameter")
        projections = tuple(structure.packed_expert_projection_names(param))
        split = len(projections)
        experts = int(row["num_experts"])
        per = [float(v) for v in row["h_trace_per_expert"]]
        per_raw = [float(v) for v in row["h_trace_per_expert_raw"]]
        tokens = row.get("expert_tokens")
        if len(per) != experts or len(per_raw) != experts:
            raise CensusStatsError(
                f"{row_name}: {len(per)} per-expert traces for {experts} experts")
        _sum_within_float32(float(row["h_trace"]), per, where=f"{row_name}.h_trace")
        _sum_within_float32(float(row["h_trace_raw"]), per_raw, where=f"{row_name}.h_trace_raw")
        if tokens is not None:
            tokens = np.asarray(tokens)
            if tokens.shape != (experts,) or not np.array_equal(tokens, np.round(tokens)):
                raise CensusStatsError(f"{row_name}: expert_tokens is not one count per expert")
        geometry = []
        for projection in projections:
            key = (stack, projection)
            slot = slots.get(key)
            if slot is None:
                raise CensusStatsError(
                    f"{row_name}: census projects no {projection} units under {stack}")
            if key in consumed:
                raise CensusStatsError(f"{row_name}: {stack}.{projection} placed twice")
            if set(slot) != set(range(experts)):
                raise CensusStatsError(
                    f"{row_name}: census {stack}.*.{projection} covers experts "
                    f"{len(slot)}, probe row has {experts}")
            shapes = {(units[name]["rows"], units[name]["cols"]) for name in slot.values()}
            if len(shapes) != 1:
                raise CensusStatsError(f"{stack}.*.{projection}: experts differ in geometry {shapes}")
            geometry.append(shapes.pop())
            consumed.add(key)
        cols = {c for _r, c in geometry}
        if (cols != {int(row["in_features"])}
                or sum(r for r, _c in geometry) != int(row["out_features"])
                or experts * sum(r * c for r, c in geometry) != int(row["n_params"])):
            raise CensusStatsError(
                f"{row_name}: census geometry {geometry} x {experts} experts does not "
                f"rebuild the packed row ({row['out_features']}x{row['in_features']}, "
                f"{row['n_params']} params)")
        splits[param] = split
        for projection, (rows, cols_) in zip(projections, geometry):
            for expert in range(experts):
                child = {field: copy.deepcopy(row[field]) for field in _CARRIED_FIELDS
                         if field in row}
                child.update({
                    "h_trace": per[expert] / split,
                    "h_trace_raw": per_raw[expert] / split,
                    "n_params": rows * cols_,
                    "in_features": cols_,
                    "out_features": rows,
                    "num_experts": 0,
                    "expert_id": expert,
                })
                if tokens is not None:
                    child["n_tokens_seen"] = int(tokens[expert])
                out[slots[(stack, projection)][expert]] = child

    unplaced = sorted(set(slots) - consumed)
    if unplaced:
        raise CensusStatsError(
            f"{len(unplaced)} census expert projections have no packed probe row "
            f"(first: {unplaced[0][0]}.*.{unplaced[0][1]})")
    missing = sorted(dense - set(out))
    if missing:
        raise CensusStatsError(
            f"{len(missing)} dense census rows have no probe row (first: {missing[0]})")
    if set(out) != census:
        raise CensusStatsError("expanded stats do not match the census roster")

    expanded = {key: value for key, value in probe.items() if key != "stats"}
    expanded["stats"] = {name: out[name] for name in sorted(out)}
    meta = copy.deepcopy(dict(probe.get("meta") or {}))
    meta[META_KEY] = {
        "schema": SCHEMA,
        "routed_units": len(units),
        "dense_units": len(dense),
        "packed_rows": sum(1 for row in stats.values() if _is_packed(row)),
        "h_trace_split": dict(sorted(splits.items())),
        "dropped_rows": sorted(dropped_rows),
        "dropped_fields": list(DROPPED_FIELDS),
        "marginals_dropped": True,
    }
    expanded["meta"] = meta
    return expanded
