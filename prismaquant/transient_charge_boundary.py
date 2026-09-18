"""The versioned native/full-engine transient charge boundary (debt D37).

Two producers measure transient bytes over different intervals with different
bounds: a Tessera native operator receipt prices one ``(unit, format)`` from
one ``apply`` call, and a full-engine capture partitions every allocation of
one complete assignment. Nobody had written down what each side owns, so
``runtime_provenance._fixed_resource_refusals`` refused every v2 table
unconditionally. This module is that writing-down, as data a gate compares
verbatim: a *boundary* is a registry entry that pairs the native bound's own
``composition`` string with the partition schema name and maps every transient
term to one owner (``docs/design/pact_transient_charge_boundary.md``).

PrismaQuant asserts nothing here about how the serving runtime classified its
bytes (AGENTS.md principle 14). Both producers freeze their own rule as data --
the native bound's ``composition`` field and the partition's schema name -- and
a boundary version states which pair of published rules its ownership map is
defined over. A table or report that names another pair, or none, is refused
by name.

"Equal" under v1 is an identity of ownership on the one measured assignment,
checked from the partition's membership rows and the native observation, and
never a numeric equality between a native peak and a partition term: the two
are differently bounded on four counts (design §1.3), so an equality would
refuse every honest report or pass on a coincidence.

The device-budget half (design §0, decided 2026-09-18): the search charges the
allocated-block composition -- the only quantity both producers measure per
row -- and the *reserved* extent the box actually holds is compared on the
exported point at the ship gate, where it is a measurement rather than a
model. The reservation slack is published beside both and added to nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .measured_runtime_prices import RuntimePriceError

#: The one registered boundary.
BOUNDARY_V1 = "prismaquant.transient_charge_boundary.v1"

#: The native bound's own composition string, exactly as the producer freezes
#: it and ``native_operator_panel.native_operator_scratch`` compares it.
NATIVE_BOUND_COMPOSITION = "sum_of_independent_peaks_including_output"
#: The partition schema the ownership map is defined over.
FULL_ENGINE_PARTITION_SCHEMA = "tessera.full_engine_resource_partition.v1"

#: The KV capacity pin, read off the KV observation's ``capacity_policy.values``
#: (the runtime's own resolved cache configuration, carried verbatim). A
#: configuration pins capacity when either field is a positive integer; with
#: neither, the pool is sized from free memory after load and ``fixed_kv`` is
#: a function of the assignment, so no single fixed charge is invariant.
KV_CAPACITY_PIN_FIELDS = ("kv_cache_memory_bytes", "num_gpu_blocks_override")

def _frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


def _thawed(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thawed(item) for item in value]
    return value


@dataclass(frozen=True)
class BoundarySpec:
    """One registered boundary: the pair of producer rules and the owner map."""

    name: str
    native_row: Mapping[str, str]
    full_engine: Mapping[str, str]
    ownership: Mapping[str, str]
    row_terms_charged: tuple[str, ...]
    row_terms_witnessed: tuple[str, ...]
    fixed_terms_charged: tuple[str, ...]
    invariance: str
    device_budget: Mapping[str, str]
    kv_capacity_pin: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("native_row", "full_engine", "ownership", "device_budget", "kv_capacity_pin"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        for name in ("row_terms_charged", "row_terms_witnessed", "fixed_terms_charged"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if set(self.row_terms_charged) & set(self.row_terms_witnessed):
            raise ValueError("a row term is either charged or witnessed, never both")

    @property
    def charges_row_activation(self) -> bool:
        """Whether the DP adds a row's ``activation_bytes`` into the device sum."""
        return "activation_bytes" in self.row_terms_charged

    def as_dict(self) -> dict:
        return {"schema": self.name, "native_row": _thawed(self.native_row),
                "full_engine": _thawed(self.full_engine), "ownership": _thawed(self.ownership),
                "row_terms_charged": list(self.row_terms_charged),
                "row_terms_witnessed": list(self.row_terms_witnessed),
                "fixed_terms_charged": list(self.fixed_terms_charged),
                "invariance": self.invariance, "device_budget": _thawed(self.device_budget),
                "kv_capacity_pin": _thawed(self.kv_capacity_pin)}


BOUNDARIES: Mapping[str, BoundarySpec] = MappingProxyType({
    BOUNDARY_V1: BoundarySpec(
        name=BOUNDARY_V1,
        native_row={
            "bound_composition": NATIVE_BOUND_COMPOSITION,
            "byte_extent": "allocated_bytes_including_allocator_rounding",
            "interval": "single_apply",
            "activation_bytes_is": "logical_input_shape_witness",
        },
        full_engine={
            "partition_schema": FULL_ENGINE_PARTITION_SCHEMA,
            "byte_extent": "requested_allocation_bytes_excluding_allocator_rounding",
            "returned_output_lifetime_class": "activation",
            "step_coverage_required": "complete",
        },
        ownership={
            "unit_scratch": "native_row",
            "returned_output": "native_row",
            "unit_input": "full_engine",
            "fixed_activation": "full_engine",
            "fixed_scratch": "full_engine",
            "kv": "full_engine",
            "non_step_peak": "full_engine",
            "runtime_workspace": "row_identity_once_per_rank",
            "reservation_slack": "witnessed_at_ship_gate",
        },
        row_terms_charged=("resident_bytes", "peak_scratch_bytes"),
        row_terms_witnessed=("activation_bytes",),
        fixed_terms_charged=("fixed_resident", "fixed_activation", "fixed_scratch", "fixed_kv",
                             "non_step_transient_peak_bytes"),
        invariance="route_class_set_of_one_full_engine_run",
        device_budget={
            # What the search subtracts from --serve-device-budget-bytes: the
            # allocated-block composition, the one extent both producers
            # measure per row.
            "search_charges": "allocated_block_composition",
            # What the export gate compares to the same budget on the exported
            # point, where the reserved extent is a measurement.
            "ship_gate_compares": "reserved_peak_bytes",
            # Published beside both; never a term.
            "witness": "reservation_slack_bytes",
        },
        kv_capacity_pin={"observation_field": "capacity_policy",
                         "pinned_when_any_of": list(KV_CAPACITY_PIN_FIELDS)},
    ),
})


def require_boundary(table_boundary: str | None, *, partition_schema: Any,
                     report_boundary: str | None = None) -> BoundarySpec:
    """The one boundary both sides name, or the named reason there is none.

    The table names the boundary in its context. The report's side of the pair
    is the rule its producer froze on the partition itself, ``partition.schema``
    -- the name the boundary is defined over -- so a report that predates the
    optional ``reference.transient_charge_boundary`` stamp still names its
    side; when the stamp is present it must be the table's name. Each
    condition is its own refusal, and nothing here defaults a side to the
    other's value.
    """
    if table_boundary is None:
        raise RuntimePriceError(
            "the table declares no transient charge boundary, so no candidate activation or "
            "scratch term may be compared to a priced row")
    spec = BOUNDARIES.get(table_boundary)
    if spec is None:
        raise RuntimePriceError(
            f"transient charge boundary {table_boundary!r} is not a registered boundary "
            f"({sorted(BOUNDARIES)})")
    if report_boundary is not None and table_boundary != report_boundary:
        raise RuntimePriceError(
            f"the table declares transient charge boundary {table_boundary!r} and the report "
            f"{report_boundary!r} -- one assignment is priced under one owner map")
    if partition_schema != spec.full_engine["partition_schema"]:
        raise RuntimePriceError(
            f"the partition declares schema {partition_schema!r} where boundary "
            f"{table_boundary} is defined over {spec.full_engine['partition_schema']!r}")
    return spec


def _stack_unit(allocation: Mapping) -> str | None:
    """The one unit whose interval this allocation was made inside, or ``None``.

    A v1 scope stack names one unit; a stack naming two distinct units is not a
    shape this boundary defines and is refused by the caller, not collapsed.
    """
    stack = list(allocation.get("scope_stack") or ())
    distinct = sorted({name for name in stack if isinstance(name, str)})
    if not distinct:
        return None
    if len(distinct) > 1:
        raise RuntimePriceError(
            f"allocation {allocation['allocation_id']} carries a scope stack naming "
            f"{distinct} -- a v1 unit interval names one unit")
    return distinct[0]


def _block_extent(allocation: Mapping) -> int | None:
    observed = allocation.get("allocator_block_bytes_observed")
    if not isinstance(observed, (list, tuple)) or not observed:
        return None
    if not all(type(item) is int and item >= 0 for item in observed):
        return None
    return max(observed)


def _simultaneous_peak(rows: Sequence[tuple[int, int | None, int]]) -> int:
    """Peak simultaneous sum over ``(allocate_index, free_index, bytes)``;
    frees settle before allocations at the same index."""
    events = []
    for begin, end, size in rows:
        events.append((begin, 1, size))
        if end is not None:
            events.append((end, 0, -size))
    live = peak = 0
    for _, _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        live += delta
        peak = max(peak, live)
    return peak


def boundary_identity_refusals(spec: BoundarySpec, report: Mapping, *,
                               selected_rows: Mapping[str, Mapping],
                               menu: Mapping[tuple[str, str], Any]) -> list[str]:
    """Checks 1 to 5 of design §2.3, each refusal naming what it is about.

    1. Containment: every allocation made inside unit ``u``'s interval is
       candidate-owned and attributed to ``u``.
    2. Attribution: every candidate-owned transient allocation attributed to
       ``u`` was made inside ``u``'s interval. Resident candidate rows are the
       load-time weights the census attributes; they are check 5's.
    3. Escape: per unit invocation, exactly one allocation escapes the
       interval and its requested bytes equal the row's returned output for one
       of the row's measured phases.
    4. Cover: the simultaneous peak of the unit's transient allocations inside
       one invocation, over rounded block extents, does not exceed the row's
       ``peak_scratch_bytes``.
    5. Residency: a resident candidate allocation made inside a unit interval
       is a lazily allocated runtime workspace no row prices.

    None of these compares a native peak to a partition term by value.
    """
    refusals: list[str] = []
    partition = report["partition"]
    observations = report["observations"]
    if partition.get("schema") != spec.full_engine["partition_schema"]:
        refusals.append(
            f"the partition declares schema {partition.get('schema')!r} where boundary "
            f"{spec.name} is defined over {spec.full_engine['partition_schema']!r}")
        return refusals
    allocations = {row["allocation_id"]: row for row in observations["torch_allocations"]}
    membership = {row["allocation_id"]: row for row in partition["membership"]}
    try:
        inside = {ident: _stack_unit(row) for ident, row in allocations.items()}
    except RuntimePriceError as exc:
        return [str(exc)]

    # 1. Containment.
    for ident, unit in sorted(inside.items()):
        if unit is None:
            continue
        member = membership.get(ident)
        if member is None:
            continue  # unclassified: the consumer already blocks it by name
        if member["owner_class"] != "candidate":
            refusals.append(
                f"{member['owner_class']}-owned allocation {ident} ({member['bytes']} B) was "
                f"made inside the interval of unit {unit!r} -- under {spec.name} a byte belongs "
                "to the operator that allocated it")
        elif member["unit"] != unit:
            refusals.append(
                f"candidate-owned allocation {ident} was made inside the interval of unit "
                f"{unit!r} and is attributed to {member['unit']!r}")

    # 2. Attribution (transient rows only).
    for ident, member in sorted(membership.items()):
        if member["owner_class"] != "candidate" or member["lifetime_class"] not in ("scratch", "activation"):
            continue
        if inside.get(ident) != member["unit"]:
            refusals.append(
                f"candidate-owned {member['lifetime_class']} allocation {ident} is attributed to "
                f"unit {member['unit']!r} and was allocated outside its interval")

    # 3, 4, 5 per selected unit.
    for unit, selected in sorted(selected_rows.items()):
        priced = menu.get((unit, selected.get("format")))
        if priced is None:
            continue
        resources = getattr(priced, "resources", None)
        output_bytes = getattr(resources, "output_bytes", None)
        by_invocation: dict[str, list[Mapping]] = {}
        for ident, owner in inside.items():
            if owner != unit:
                continue
            allocation = allocations[ident]
            invocation = allocation.get("unit_invocation") or f"{unit}:?"
            by_invocation.setdefault(str(invocation), []).append(allocation)
        for invocation, rows in sorted(by_invocation.items()):
            escaping = [row for row in rows if row["lifetime_scope"] == "escapes_unit"]
            if output_bytes is None:
                refusals.append(
                    f"the priced row {(unit, selected.get('format'))} carries no output_bytes, so "
                    f"the allocation escaping invocation {invocation} cannot be identified as "
                    "its returned output")
            elif len(escaping) != 1:
                refusals.append(
                    f"invocation {invocation} escapes {len(escaping)} allocations "
                    f"({[row['allocation_id'] for row in escaping]}) -- under {spec.name} exactly "
                    "one escapes, the returned output")
            elif escaping[0]["bytes"] not in set(output_bytes.values()):
                refusals.append(
                    f"invocation {invocation} escapes allocation {escaping[0]['allocation_id']} of "
                    f"{escaping[0]['bytes']} B where the row's returned output is "
                    f"{dict(output_bytes)} B")
            transient, unobserved = [], 0
            for row in rows:
                member = membership.get(row["allocation_id"])
                if member is None or member["lifetime_class"] == "resident":
                    continue  # resident rows are check 5's, below
                extent = _block_extent(row)
                if extent is None:
                    unobserved += 1
                    continue
                transient.append((row["allocate_index"], row["free_completed_index"], extent))
            bound = getattr(resources, "peak_scratch_bytes", None)
            if unobserved:
                # The capture sampled no block extent for these rows (the
                # 2026-09-18 report carries an empty list on every in-unit
                # scratch allocation), so the cover check has no rounded
                # extent to sum; requested bytes are a floor and a floor
                # cannot establish cover.
                refusals.append(
                    f"{unobserved} transient allocations of unit {unit!r} inside invocation "
                    f"{invocation} observed no allocator block extent, so the cover check has "
                    "no rounded extent to sum")
            elif transient and isinstance(bound, int):
                peak = _simultaneous_peak(transient)
                if peak > bound:
                    refusals.append(
                        f"the engine held {peak} B of the allocations of unit {unit!r} at once "
                        f"in invocation {invocation} where the priced row bounds {bound} B -- the "
                        "native bound does not cover what the engine did")
        # 5. Residency. The partition's candidate_resident is compared to the
        # row's resident_bytes by the gate already; what v1 adds is the name
        # for the excess: resident candidate bytes above what the row prices
        # are a lazily allocated runtime workspace, which needs a row identity
        # (design §3.4) and is never folded into the row's resident charge.
        resident_rows = [member for member in membership.values()
                         if member["owner_class"] == "candidate"
                         and member["lifetime_class"] == "resident" and member["unit"] == unit]
        priced_resident = getattr(resources, "resident_bytes", None)
        if isinstance(priced_resident, int):
            excess = sum(member["bytes"] for member in resident_rows) - priced_resident
            if excess > 0:
                suspects = [member["allocation_id"] for member in resident_rows
                            if inside.get(member["allocation_id"]) == unit]
                refusals.append(
                    f"unit {unit!r} holds {excess} B of resident candidate bytes the priced row "
                    f"does not price (allocations inside its interval: {suspects}) -- a lazily "
                    "allocated runtime workspace needs a row identity (workspace_resident_bytes, "
                    "workspace_sha256)")
    return refusals


def _pinned(value: Any) -> bool:
    return type(value) is int and value > 0


def route_class_coverage_refusals(spec: BoundarySpec, table: Any, *, report: Mapping,
                                  selected_rows: Mapping[str, Mapping],
                                  menu: Mapping[tuple[str, str], Any]) -> list[str]:
    """Design §4.3: the fixed charge transfers to every assignment whose formats
    lie in the route classes the one full-engine run exercised, under a pinned
    KV capacity. Each unpriced route class is named with its units; an unpinned
    KV policy is named; nothing here is a constant."""
    refusals: list[str] = []
    exercised = set()
    for unit, selected in selected_rows.items():
        priced = menu.get((unit, selected.get("format")))
        binding = getattr(priced, "binding", None) if priced is not None else None
        route = getattr(binding, "operator_route", None)
        if isinstance(route, str):
            exercised.add(route)
    unpriced: dict[str, list[str]] = {}
    for row in table.rows:
        binding = getattr(row, "binding", None)
        route = getattr(binding, "operator_route", None)
        if not isinstance(route, str):
            refusals.append(
                f"row {(row.unit, row.fmt)} carries no operator route binding, so its route "
                "class cannot be checked against the full-engine run")
            continue
        if route not in exercised:
            unpriced.setdefault(route, []).append(f"{row.unit}@{row.fmt}")
    for route, units in sorted(unpriced.items()):
        refusals.append(
            f"route class {route!r} is priced for {units} and the full-engine run exercised no "
            "row of that class, so the fixed charge measured there does not transfer to it")

    records = report["observations"].get("kv_observations")
    if not records:
        refusals.append(
            "the capture carries no KV observation, so whether the configuration pins KV "
            "capacity cannot be read")
        return refusals
    field_name = spec.kv_capacity_pin["observation_field"]
    for index, record in enumerate(records):
        policy = record.get(field_name) if isinstance(record, Mapping) else None
        values = policy.get("values") if isinstance(policy, Mapping) else None
        if not isinstance(values, Mapping):
            refusals.append(
                f"KV observation {index} carries no {field_name}.values, so the capacity pin "
                "cannot be read")
            continue
        if not any(_pinned(values.get(key)) for key in KV_CAPACITY_PIN_FIELDS):
            refusals.append(
                f"KV observation {index}'s capacity policy pins no capacity "
                f"({', '.join(KV_CAPACITY_PIN_FIELDS)} are all unset), so fixed_kv is sized from "
                "free memory after load and is a function of the assignment")
    return refusals


def reservation_slack_refusals(spec: BoundarySpec, report: Mapping) -> list[str]:
    """Design §3.7: the report must observe the reserved extent beside the
    allocated one at one instant, so the witness the ship gate reads exists."""
    observation = report["observations"].get("reservation_slack")
    if observation is None:
        return ["the capture observes no reservation_slack, so the reserved-extent witness "
                f"{spec.device_budget['witness']} has no evidence"]
    return []


def reservation_slack_bytes(report: Mapping) -> int | None:
    """The witness, recomputed from the two readings; ``None`` when unobserved."""
    observation = report["observations"].get("reservation_slack")
    if observation is None:
        return None
    return int(observation["reserved_bytes"]) - int(observation["allocated_bytes"])
