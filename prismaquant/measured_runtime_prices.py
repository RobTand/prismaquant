"""Strict, opt-in measured operator prices for discrete allocation proposals.

Extends the dispatch-table input boundary, not its served-SLO evidence. Prices
are medians of repeated GPU operator timings for a whole serving unit. They
are never inferred from activation width, encoder time, or family speed hints.
The sequential operator-sum model can propose assignments under a declared
budget; it cannot certify end-to-end p95 TTFT/ITL. Producer admission and final
served validation remain independent gates. No measured table ships here.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .lane_eligibility import ServingContext
from .serve_dispatch_table import DispatchTableError
from .schemas import strict_json_loads

SCHEMA = "prismaquant.measured_runtime_prices.v1"
CONTEXT_SCHEMA = "prismaquant.measured_runtime_context.v1"
PROVENANCE_CONTEXT_SCHEMA = "prismaquant.measured_runtime_context.v2"
COHORT_CONTEXT_SCHEMA = "prismaquant.measured_runtime_context.v3"
MIXED_NATIVE_STRUCTURE = "mixed_native.v1"
PROVENANCE_TABLE_SCHEMA = "prismaquant.measured_runtime_prices.v2"
PROVENANCE_IDENTITY_KIND = "prismaquant.runtime_provenance_relation.v1"
RESOURCE_FIELDS = ("prefill_ms", "decode_ms", "serialized_bytes", "resident_bytes",
                   "peak_scratch_bytes", "activation_bytes", "kv_bytes")
#: The off-step half of the placement obligation
#: `max(scalar_budget_bytes, non_step_transient_peak_bytes)`, the producer's
#: frozen `PLACEMENT_OBLIGATION` contract string. The seven fields above
#: price one engine step; this prices what the engine still holds while no step
#: is running. The admission gate demands the obligation be recomputable and
#: nothing consumed it, so the DP pruned against the smaller of two numbers
#: whenever the off-step peak was the larger. Absent means "not priced", never
#: zero: a maximum taken against a default would read as the other side having
#: been checked. Optional on the wire so every table emitted before this field
#: existed keeps its digest.
OFF_STEP_FIELD = "non_step_transient_peak_bytes"
RESOURCE_FIELDS_WITH_OFF_STEP = RESOURCE_FIELDS + (OFF_STEP_FIELD,)
#: A native row's returned output per measured phase, in logical bytes -- the
#: observation's ``output_bytes`` (``native_operator_panel.consume_native_receipt``).
#: Under ``transient_charge_boundary`` v1 the returned output is native-row-owned
#: and already inside ``peak_scratch_bytes``; this field is the witness the
#: escape check (design §2.3 item 3) identifies it by, never a charge. Optional
#: on the wire so every table emitted before it existed keeps its digest.
OUTPUT_BYTES_FIELD = "output_bytes"
#: The transient charge boundary a context names (``RuntimeContext``).
CONTEXT_BOUNDARY_FIELD = "transient_charge_boundary"

#: The per-rank spelling of a row's resources, for a serving unit measured
#: under tensor parallelism. One ranked MoE owner has no single scalar answer
#: to "how many device bytes does this row cost": the ranks hold different
#: halves of the same stack, so a rank sum and a rank maximum are both numbers
#: no device ever held. The scalar fields above keep their meaning exactly, and
#: a row whose price is per-rank says so on its face rather than shipping one
#: of those two reductions under an existing name.
#:
#: ``prefill_ms``/``decode_ms`` stay ONE whole-owner measurement in both
#: spellings. The tensor-parallel world's collective cost is inside that
#: number; a rank record carries no timing, because a per-rank timing would
#: invite a sum of ranks where a whole-owner apply was measured once.
RANK_RESOURCES_SCHEMA = "prismaquant.runtime_rank_resources.v1"
#: One rank's own record. Every axis is required: an undefined axis is an
#: absence of evidence and never an implied zero, which is the whole reason
#: this spelling exists next to the scalar one.
RANK_FIELDS = ("rank", "resident_bytes", "peak_scratch_bytes",
               "activation_bytes", "workspace_resident_bytes", "workspace_sha256",
               "bound_sha256")
RANK_VECTOR_FIELDS = ("schema", "world_size", "timing_rule", "prefill_ms", "decode_ms",
                      "rank_medians_ms", "wire_bytes", "wire_sha256", "ranks")
#: The one timing rule this spelling carries. A routed owner apply at a world
#: above one was timed once on every rank, each including the runtime's own
#: final all-reduce, so the world's own step cannot be faster than its slowest
#: rank: the priced number is that rank's median, and every rank's median is
#: retained beside it. It is never a sum of leaf timings and never a mean.
RANK_TIMING_RULE = "slowest_rank_median_of_one_whole_owner_apply"
#: The versioned input a per-rank device budget arrives under. Two things are
#: declared together because either alone is a number nobody can check: the
#: budget each rank has, and the fixed whole-engine charge that has to be added
#: to each rank before a comparison means anything.
RANK_DEVICE_BOUNDS_SCHEMA = "prismaquant.runtime_rank_device_bounds.v1"
RANK_DEVICE_BOUNDS_FIELDS = ("schema", "world_size", "provenance", "budgets_per_rank",
                             "charge_per_rank", "evidence")
#: ``pending_measurement``: the charge is not known, so no device total may be
#: published and no rank may be admitted -- but the campaign's rank dimensions
#: still price, because a common unknown constant cannot reorder them.
#: ``recomputed_full_engine_partition``: the charge came from a per-rank
#: partition this consumer recomputed, and the axis admits.
RANK_DEVICE_PROVENANCE = ("pending_measurement", "recomputed_full_engine_partition")
_RANK_DEVICE_EVIDENCE_FIELDS = ("full_engine_report", "per_rank_partition")
#: Terms that add across sequentially priced units, per rank. The wire extent
#: is deliberately absent: one artifact is one charge, counted once by
#: ``RuntimeRankResources.wire_bytes``, not once per rank that holds a copy of
#: the same container.
RANK_ADDITIVE_TERMS = ("resident_bytes", "workspace_resident_bytes")
#: Terms that are independent per-rank maxima across those same units.
RANK_PEAK_TERMS = ("peak_scratch_bytes", "activation_bytes")
#: Terms a ranked consumer may not publish a device total without. The fixed
#: whole-engine terms come from the full-engine gate, which today refuses every
#: v2 table; the workspace is no longer one of them, because it has a versioned
#: composition rule (``RANK_WORKSPACE_RULE``).
RANK_WITHHELD_TERMS = ("fixed_resident_bytes", "fixed_activation_bytes",
                       "fixed_scratch_bytes", "fixed_kv_bytes", OFF_STEP_FIELD)
#: How one process-global workspace becomes a per-rank charge, versioned because
#: it is a rule rather than a number. The ``vllm.WorkspaceManager`` allocation is
#: persistent runtime state shared by every operator in the process, not
#: per-apply scratch and not an artifact: two priced rows that carry the same
#: frozen workspace identity are viewing the SAME allocation, so it is charged
#: once per rank for that identity rather than once per row. Two records that
#: disagree about the bytes behind one identity are refused -- one allocation
#: cannot be two allocations -- and distinct identities add, because they are
#: distinct allocations.
RANK_WORKSPACE_RULE = "sum_of_distinct_frozen_workspace_identities_per_rank"


class RuntimePriceError(DispatchTableError):
    """Missing, malformed, stale, or mismatched measured proposal evidence."""


def _object(value: Any, fields: tuple[str, ...], where: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise RuntimePriceError(f"{where}: expected exactly fields {sorted(fields)}")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RuntimePriceError(f"{where}: expected a nonempty trimmed string")
    return value


def _sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimePriceError(f"{where}: expected lowercase SHA-256")
    return value


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimePriceError(f"{where}: expected integer >= {minimum}")
    return value


def _number(value: Any, where: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise RuntimePriceError(f"{where}: expected finite {'positive' if positive else 'nonnegative'} number")
    return float(value)


def _timestamp(value: Any, where: str) -> datetime:
    try:
        result = datetime.fromisoformat(_string(value, where).replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimePriceError(f"{where}: invalid ISO-8601 timestamp") from exc
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise RuntimePriceError(f"{where}: timestamp must explicitly use UTC")
    return result


def _json(path: str | Path) -> dict:
    try:
        return strict_json_loads(Path(path).read_text(encoding="utf-8"), duplicate=lambda key:
                                 RuntimePriceError(f"{path}: duplicate JSON key {key!r}"))
    except (OSError, ValueError) as exc:
        raise RuntimePriceError(f"cannot load {path}: {exc}") from exc


def identity_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class MixedNativeServingContext:
    """Measured-table composition only; never a lane-eligibility cell target."""
    platform: str
    structure: str
    residency: str
    runtime_image: str
    execution_mode: str

    def __post_init__(self):
        if self.structure != MIXED_NATIVE_STRUCTURE:
            raise RuntimePriceError("mixed native context requires its explicit versioned structure")
        # Both real structures share these target coordinates. Do not expand
        # ServingContext's lane grammar or invent a mixed contract cell.
        for structure in ("dense", "routed_moe"):
            ServingContext(self.platform, structure, self.residency, self.runtime_image, self.execution_mode)

    def as_dict(self):
        return {name:getattr(self,name) for name in
            ("platform","structure","residency","runtime_image","execution_mode")}


@dataclass(frozen=True)
class RuntimeContext:
    """Workload/source identity extending the existing lane ServingContext.

    ``runtime_sha256`` is the digest of the full measured runtime manifest:
    packages/plugins/kernel builds, launch arguments, scheduler/chunking,
    graph capture settings, and parallel topology. An image digest alone is
    insufficient because these can change inside one image. ``gpu_identity``
    identifies the actual device/configuration used by that manifest. The
    explicit workload fields are additional checked coordinates, not a licence
    to omit unlisted runtime settings from the manifest identity. Version 2
    instead names a provenance-relation identity that retains every original
    runtime manifest and explicitly verifies their common coordinates.
    """

    serving_context: ServingContext
    gpu_identity: str
    runtime_sha256: str
    source_sha256: str
    calibration_sha256: str
    prompt_tokens: int
    batch_size: int
    tensor_parallel: int
    graph_mode: str
    operator_routes: Mapping[str, Mapping[str, str]]
    runtime_identity_kind: str | None = None
    #: The native/full-engine transient charge boundary every row of the table
    #: was priced under (``transient_charge_boundary.BOUNDARIES``). Appears in
    #: ``as_dict`` only when set, so a context emitted before the field existed
    #: re-emits byte-identically and keeps its digest. A table that names none
    #: prices operators and nothing else: its fixed charge refuses by name.
    transient_charge_boundary: str | None = None
    native_cohort: Mapping | None = None

    def __post_init__(self):
        if not isinstance(self.serving_context, (ServingContext, MixedNativeServingContext)):
            raise RuntimePriceError("serving_context must be a ServingContext")
        _string(self.gpu_identity, "gpu_identity")
        if self.runtime_identity_kind not in (None, PROVENANCE_IDENTITY_KIND):
            raise RuntimePriceError("unknown measured runtime identity kind")
        if self.transient_charge_boundary is not None:
            _string(self.transient_charge_boundary, CONTEXT_BOUNDARY_FIELD)
        for name in ("runtime_sha256", "source_sha256", "calibration_sha256"):
            _sha(getattr(self, name), name)
        for name in ("prompt_tokens", "batch_size", "tensor_parallel"):
            _integer(getattr(self, name), name, 1)
        _string(self.graph_mode, "graph_mode")
        if not isinstance(self.operator_routes, Mapping) or not self.operator_routes:
            raise RuntimePriceError("operator_routes must be a nonempty unit/format/route mapping")
        routes = {}
        for unit, formats in sorted(self.operator_routes.items()):
            _string(unit, "operator_routes unit")
            if not isinstance(formats, Mapping) or not formats:
                raise RuntimePriceError("operator_routes unit must have format routes")
            routes[unit] = MappingProxyType({_string(fmt, "route format"): _string(route, "operator route")
                                            for fmt, route in sorted(formats.items())})
        object.__setattr__(self, "operator_routes", MappingProxyType(routes))
        if self.native_cohort is not None:
            from .native_runtime_cohort import validate_cohort
            validate_cohort(self.native_cohort)
            if self.runtime_identity_kind != PROVENANCE_IDENTITY_KIND:
                raise RuntimePriceError("native cohort requires an explicit provenance relation")
            contexts=self.native_cohort["operator_contexts"]
            expected={unit+"@"+fmt for unit,formats in routes.items() for fmt in formats}
            if set(contexts)!=expected:
                raise RuntimePriceError("native cohort operator coverage differs from routes")
            structures={value["structure"] for value in contexts.values()}
            expected_structure=MIXED_NATIVE_STRUCTURE if len(structures)>1 else next(iter(structures))
            if self.serving_context.structure!=expected_structure:
                raise RuntimePriceError("native cohort structure differs from retained operators")
            for value in contexts.values():
                for route in value["routes"].values():
                    if identity_sha256(route)!=identity_sha256(json.loads(routes[value["unit"]][value["format"]])):
                        raise RuntimePriceError("native cohort route differs from retained operator")
            common=self.native_cohort["common"]
            if (common["gpu"]["uuid"]!=self.gpu_identity or common["image"]!=self.serving_context.runtime_image
                    or common["execution"]!={"mode":self.serving_context.residency,
                        "execution_mode":self.graph_mode,"tensor_parallel":self.tensor_parallel}):
                raise RuntimePriceError("native cohort shared runtime differs from context")
        elif isinstance(self.serving_context, MixedNativeServingContext):
            raise RuntimePriceError("mixed native context requires retained operator contexts")

    def operator_route(self, unit: str, fmt: str) -> str:
        try:
            return self.operator_routes[unit][fmt]
        except KeyError as exc:
            raise RuntimePriceError(f"missing expected operator route for {(unit, fmt)}") from exc

    def as_dict(self) -> dict:
        return {"schema": COHORT_CONTEXT_SCHEMA if self.native_cohort is not None else (PROVENANCE_CONTEXT_SCHEMA if self.runtime_identity_kind else CONTEXT_SCHEMA),
                **({"native_cohort":self.native_cohort} if self.native_cohort is not None else {}),
                **({"runtime_identity_kind": self.runtime_identity_kind} if self.runtime_identity_kind else {}),
                "serving_context": self.serving_context.as_dict(),
                **{name: getattr(self, name) for name in (
                    "gpu_identity", "runtime_sha256", "source_sha256", "calibration_sha256",
                    "prompt_tokens", "batch_size", "tensor_parallel", "graph_mode")},
                "operator_routes": {unit: dict(formats) for unit, formats in self.operator_routes.items()},
                **({CONTEXT_BOUNDARY_FIELD: self.transient_charge_boundary}
                   if self.transient_charge_boundary is not None else {})}


def parse_runtime_context(payload: Mapping) -> RuntimeContext:
    fields = ("schema", "serving_context", "gpu_identity", "runtime_sha256", "source_sha256",
              "calibration_sha256", "prompt_tokens", "batch_size", "tensor_parallel", "graph_mode", "operator_routes")
    if not isinstance(payload, Mapping):
        raise RuntimePriceError("runtime context: expected an object")
    if payload.get("schema") in (PROVENANCE_CONTEXT_SCHEMA, COHORT_CONTEXT_SCHEMA):
        fields += ("runtime_identity_kind",)
        if payload.get("runtime_identity_kind") != PROVENANCE_IDENTITY_KIND:
            raise RuntimePriceError("v2 runtime context requires an explicit provenance relation identity")
    if payload.get("schema")==COHORT_CONTEXT_SCHEMA:
        fields += ("native_cohort",)
    if CONTEXT_BOUNDARY_FIELD in payload:
        # Optional on the wire, never defaulted: a context that omits it names
        # no boundary, and a null value is refused rather than read as none.
        fields += (CONTEXT_BOUNDARY_FIELD,)
        _string(payload[CONTEXT_BOUNDARY_FIELD], "runtime context " + CONTEXT_BOUNDARY_FIELD)
    _object(payload, fields, "runtime context")
    if payload["schema"] not in (CONTEXT_SCHEMA, PROVENANCE_CONTEXT_SCHEMA, COHORT_CONTEXT_SCHEMA):
        raise RuntimePriceError(f"runtime context schema must be {CONTEXT_SCHEMA}")
    serving = _object(payload["serving_context"], ("platform", "structure", "residency", "runtime_image", "execution_mode"), "serving_context")
    try:
        factory=MixedNativeServingContext if serving.get("structure")==MIXED_NATIVE_STRUCTURE else ServingContext
        return RuntimeContext(serving_context=factory(**serving),
                              **{field: payload[field] for field in fields if field not in ("schema", "serving_context")})
    except (ValueError, TypeError) as exc:
        raise RuntimePriceError(f"runtime context: {exc}") from exc


def load_runtime_context(path: str | Path) -> RuntimeContext:
    return parse_runtime_context(_json(path))


@dataclass(frozen=True)
class RuntimeResources:
    """Separate wire bytes, resident terminal weights, and transient allocations.

    Timings/permanently resident weights add across sequential serving units. Scratch and
    activation use independent maxima; global KV and fixed overhead are added
    once. This conservative composition assumes no overlapping unit execution.
    """

    prefill_ms: float
    decode_ms: float | None
    serialized_bytes: int
    resident_bytes: int
    peak_scratch_bytes: int
    activation_bytes: int
    kv_bytes: int = 0
    non_step_transient_peak_bytes: int | None = None
    output_bytes: Mapping[str, int] | None = None

    def __post_init__(self):
        _number(self.prefill_ms, "prefill_ms")
        if self.decode_ms is not None:
            _number(self.decode_ms, "decode_ms")
        for name in RESOURCE_FIELDS[2:]:
            _integer(getattr(self, name), name)
        if self.non_step_transient_peak_bytes is not None:
            _integer(self.non_step_transient_peak_bytes, OFF_STEP_FIELD)
        if self.output_bytes is not None:
            if not isinstance(self.output_bytes, Mapping) or not self.output_bytes:
                raise RuntimePriceError(f"{OUTPUT_BYTES_FIELD} must be a nonempty phase -> bytes mapping")
            phases = {}
            for phase, size in sorted(self.output_bytes.items()):
                phases[_string(phase, OUTPUT_BYTES_FIELD + " phase")] = _integer(size, f"{OUTPUT_BYTES_FIELD} {phase}")
            object.__setattr__(self, "output_bytes", MappingProxyType(phases))

    def as_dict(self) -> dict:
        # The off-step and output fields appear only when priced, so a table
        # emitted before either existed re-emits byte-identically and keeps
        # its digest.
        payload = {field: getattr(self, field) for field in RESOURCE_FIELDS}
        if self.non_step_transient_peak_bytes is not None:
            payload[OFF_STEP_FIELD] = self.non_step_transient_peak_bytes
        if self.output_bytes is not None:
            payload[OUTPUT_BYTES_FIELD] = dict(self.output_bytes)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> RuntimeResources:
        fields = RESOURCE_FIELDS
        if isinstance(payload, Mapping):
            fields += (OFF_STEP_FIELD,) if OFF_STEP_FIELD in payload else ()
            fields += (OUTPUT_BYTES_FIELD,) if OUTPUT_BYTES_FIELD in payload else ()
        return cls(**_object(payload, fields, "resources"))


@dataclass(frozen=True)
class RankResources:
    """One rank's own share of a priced row; nothing here is a reduction.

    ``resident_bytes`` is what that rank holds; the module's *wire extent* is
    one canonical owner charge on the vector (``wire_bytes``), because the
    producer frames one whole-module container per rank and shards it locally --
    the same bytes, not one artifact per rank;
    ``peak_scratch_bytes`` and ``activation_bytes`` are that rank's own peaks
    over the row's measured phases, not a maximum taken over its peers.
    ``workspace_resident_bytes``/``workspace_sha256`` are the rank's *runtime
    workspace* -- persistent engine state the row did not allocate itself --
    carried with its frozen identity so no consumer has to guess whether two
    rows are describing the same allocation.
    """

    rank: int
    resident_bytes: int
    peak_scratch_bytes: int
    activation_bytes: int
    workspace_resident_bytes: int
    workspace_sha256: str
    #: The digest this rank's own resource record carried before its roster was
    #: attached (``runtime_provenance.routed_rank_bound``). Carried so two
    #: ranks' records can be checked against each other's observation of them.
    bound_sha256: str

    def __post_init__(self):
        _integer(self.rank, "per-rank resource rank")
        for name in RANK_ADDITIVE_TERMS + RANK_PEAK_TERMS:
            _integer(getattr(self, name), name)
        _sha(self.workspace_sha256, "workspace_sha256")
        _sha(self.bound_sha256, "bound_sha256")

    def as_dict(self) -> dict:
        return {name: getattr(self, name)
                for name in ("rank",) + RANK_ADDITIVE_TERMS + RANK_PEAK_TERMS
                + ("workspace_sha256", "bound_sha256")}

    @classmethod
    def from_dict(cls, payload: Mapping, *, where: str = "per-rank resources") -> "RankResources":
        return cls(**_object(payload, RANK_FIELDS, where))


@dataclass(frozen=True)
class RuntimeRankResources:
    """A row's per-rank resource vector, with one whole-owner timing pair.

    ``world_size`` must equal the table context's ``tensor_parallel``: a vector
    covering fewer ranks than the priced world is exactly the imbalance this
    spelling exists to expose, and a vector covering more is pricing a box the
    row was not measured on. The rank roster must be ``0..world_size-1`` once
    each and in order, so a missing rank is a refusal rather than a hole a
    later sum would quietly close.
    """

    prefill_ms: float
    decode_ms: float | None
    world_size: int
    #: ``{"prefill": [...], "decode": [...]}``, one median per rank in rank
    #: order. ``prefill_ms``/``decode_ms`` are the slowest rank's entry.
    rank_medians_ms: Mapping[str, tuple[float, ...]]
    #: The module's wire extent, counted ONCE for the whole owner: the producer
    #: frames one canonical whole-module container and shards it locally, so
    #: every rank holds a view of the same artifact rather than an artifact of
    #: its own. ``wire_sha256`` binds the ordered member wire identities the
    #: extent was summed from, so two ranks cannot disagree about the bytes.
    wire_bytes: int
    wire_sha256: str
    ranks: tuple[RankResources, ...]

    def __post_init__(self):
        _number(self.prefill_ms, "prefill_ms")
        if self.decode_ms is not None:
            _number(self.decode_ms, "decode_ms")
        _integer(self.wire_bytes, "wire_bytes")
        _sha(self.wire_sha256, "wire_sha256")
        _integer(self.world_size, "world_size", 1)
        if not isinstance(self.ranks, tuple) or len(self.ranks) != self.world_size:
            size = len(self.ranks) if isinstance(self.ranks, (tuple, list)) else "no"
            raise RuntimePriceError(
                f"per-rank resources must carry exactly one record per rank: world_size "
                f"{self.world_size} against {size} records")
        for entry in self.ranks:
            if not isinstance(entry, RankResources):
                raise RuntimePriceError("per-rank resources must be RankResources records")
        covered = tuple(entry.rank for entry in self.ranks)
        if covered != tuple(range(self.world_size)):
            raise RuntimePriceError(
                f"per-rank resources must cover ranks 0..{self.world_size - 1} once each and "
                f"in order, and they cover {list(covered)}")
        if (not isinstance(self.rank_medians_ms, Mapping)
                or set(self.rank_medians_ms) != {"prefill", "decode"}):
            raise RuntimePriceError(
                "per-rank resources carry one prefill and one decode median for every rank")
        medians = {}
        for phase in ("prefill", "decode"):
            values = self.rank_medians_ms[phase]
            if (not isinstance(values, (tuple, list)) or len(values) != self.world_size):
                raise RuntimePriceError(
                    f"per-rank {phase} medians must name every rank of this world")
            if phase == "decode" and self.decode_ms is None:
                # A row that measured no decode carries no rank median either:
                # an explicit absence, never a zero a reader could price.
                if any(value is not None for value in values):
                    raise RuntimePriceError(
                        "a per-rank row with no priced decode carries no rank decode median")
                medians[phase] = tuple(None for _ in values)
            else:
                medians[phase] = tuple(_number(value, f"rank {phase} median") for value in values)
        object.__setattr__(self, "rank_medians_ms", MappingProxyType(medians))
        for phase, priced in (("prefill", self.prefill_ms), ("decode", self.decode_ms)):
            if priced is not None and priced != max(medians[phase]):
                raise RuntimePriceError(
                    f"a per-rank row prices the slowest rank's {phase} median, and this row "
                    f"prices {priced} against {max(medians[phase])}")

    def rank(self, index: int) -> RankResources:
        if type(index) is not int or not 0 <= index < self.world_size:
            raise RuntimePriceError(f"rank {index!r} is outside this row's world of {self.world_size}")
        return self.ranks[index]

    def as_dict(self) -> dict:
        return {"schema": RANK_RESOURCES_SCHEMA, "world_size": self.world_size,
                "timing_rule": RANK_TIMING_RULE,
                "prefill_ms": self.prefill_ms, "decode_ms": self.decode_ms,
                "rank_medians_ms": {phase: list(values)
                                    for phase, values in self.rank_medians_ms.items()},
                "wire_bytes": self.wire_bytes, "wire_sha256": self.wire_sha256,
                "ranks": [entry.as_dict() for entry in self.ranks]}

    @classmethod
    def from_dict(cls, payload: Mapping, *, tensor_parallel: int,
                  where: str = "row resources") -> "RuntimeRankResources":
        fields = _object(payload, RANK_VECTOR_FIELDS, where)
        if fields["schema"] != RANK_RESOURCES_SCHEMA:
            raise RuntimePriceError(
                f"{where}: unknown per-rank resource schema {fields['schema']!r}")
        if fields["timing_rule"] != RANK_TIMING_RULE:
            raise RuntimePriceError(
                f"{where}: unknown per-rank timing rule {fields['timing_rule']!r}; a rank whose "
                "timing rule is not this one priced something else")
        if fields["world_size"] != tensor_parallel:
            raise RuntimePriceError(
                f"{where}: per-rank resources cover a world of {fields['world_size']!r} where "
                f"this table's context declares tensor_parallel {tensor_parallel}")
        ranks = fields["ranks"]
        if not isinstance(ranks, (list, tuple)):
            raise RuntimePriceError(f"{where}: per-rank resources must be an explicit ordered list")
        medians = fields["rank_medians_ms"]
        if not isinstance(medians, Mapping):
            raise RuntimePriceError(f"{where}: per-rank medians must be an explicit mapping")
        return cls(fields["prefill_ms"], fields["decode_ms"], fields["world_size"],
                   {phase: medians[phase] for phase in ("prefill", "decode") if phase in medians},
                   fields["wire_bytes"], fields["wire_sha256"],
                   tuple(RankResources.from_dict(entry, where=f"{where} rank record")
                         for entry in ranks))


def parse_row_resources(payload: Any, *, tensor_parallel: int,
                        where: str = "row resources") -> RuntimeResources | RuntimeRankResources:
    """A row's resources in the one spelling its world size licenses.

    A tensor-parallel table prices every row per rank, so a scalar row under
    such a context is refused rather than reinterpreted: whichever reduction a
    producer applied to get one number per axis, the consumer did not agree to
    it. Conversely the scalar spelling keeps its v2 meaning exactly, so a table
    emitted before this field existed re-reads byte-identically.
    """
    if isinstance(payload, Mapping) and "schema" in payload:
        return RuntimeRankResources.from_dict(payload, tensor_parallel=tensor_parallel, where=where)
    if tensor_parallel > 1:
        raise RuntimePriceError(
            f"{where}: a tensor-parallel table prices every row with "
            f"{RANK_RESOURCES_SCHEMA} per-rank resources and this row carries the scalar "
            "spelling; a rank sum or rank maximum written into the scalar byte fields is a "
            "number no device held")
    resources = RuntimeResources.from_dict(payload)
    if resources.kv_bytes:
        raise RuntimePriceError("KV belongs to fixed_resources, not per-unit rows")
    if resources.non_step_transient_peak_bytes is not None:
        raise RuntimePriceError("the off-step transient peak is one whole-engine obligation, "
                                "not a per-unit row price")
    return resources


@dataclass(frozen=True)
class RuntimeBinding:
    """Bind whole-unit timings to exact candidate members and joint operators.

    Joint operator identity covers source/rendered tensor content, full
    activation/scales contract, arithmetic, and aligned calibration probes.
    A fused group's measured row must identify every member; leaf medians
    cannot be silently summed into a fused operator measurement.
    """

    member_formats: Mapping[str, str]
    member_operator_identity_sha256: Mapping[str, str]
    member_shapes: Mapping[str, tuple[int, ...]]
    operator_route: str

    def __post_init__(self):
        _string(self.operator_route, "operator_route")
        for name in ("member_formats", "member_operator_identity_sha256", "member_shapes"):
            if not isinstance(getattr(self, name), Mapping) or not getattr(self, name):
                raise RuntimePriceError(f"{name} must be a nonempty mapping")
        if not set(self.member_formats) == set(self.member_operator_identity_sha256) == set(self.member_shapes):
            raise RuntimePriceError("binding member maps must have identical keys")
        formats, digests, shapes = {}, {}, {}
        for unit in sorted(self.member_formats):
            _string(unit, "member unit")
            formats[unit] = _string(self.member_formats[unit], "member format")
            digests[unit] = _sha(self.member_operator_identity_sha256[unit], "joint operator identity")
            shape = self.member_shapes[unit]
            if not isinstance(shape, (tuple, list)) or not shape:
                raise RuntimePriceError("member shape must be a nonempty dimension sequence")
            shapes[unit] = tuple(_integer(dim, "shape dimension", 1) for dim in shape)
        object.__setattr__(self, "member_formats", MappingProxyType(formats))
        object.__setattr__(self, "member_operator_identity_sha256", MappingProxyType(digests))
        object.__setattr__(self, "member_shapes", MappingProxyType(shapes))

    def as_dict(self) -> dict:
        return {"member_formats": dict(self.member_formats),
                "member_operator_identity_sha256": dict(self.member_operator_identity_sha256),
                "member_shapes": {unit: list(shape) for unit, shape in self.member_shapes.items()},
                "operator_route": self.operator_route}

    @classmethod
    def from_dict(cls, payload: Mapping) -> RuntimeBinding:
        return cls(**_object(payload, ("member_formats", "member_operator_identity_sha256", "member_shapes", "operator_route"), "binding"))


@dataclass(frozen=True)
class OperatorMeasurement:
    """Raw repeated per-invocation GPU durations with attributable evidence."""

    method: str
    samples_ms: tuple[float, ...]
    warmup_iterations: int
    receipt_path: str
    receipt_sha256: str

    @property
    def median_ms(self) -> float:
        return float(statistics.median(self.samples_ms))

    @classmethod
    def from_dict(cls, payload: Mapping) -> OperatorMeasurement:
        _object(payload, ("method", "samples_ms", "warmup_iterations", "receipt_path", "receipt_sha256"), "operator measurement")
        if payload["method"] not in ("cuda_events", "gpu_profiler", "synchronized_gpu_wall_clock"):
            raise RuntimePriceError("measurement method must time actual GPU operator execution")
        samples = payload["samples_ms"]
        if not isinstance(samples, (tuple, list)) or len(samples) < 3:
            raise RuntimePriceError("operator measurement requires at least three repeated samples")
        return cls(payload["method"], tuple(_number(v, "samples_ms", positive=True) for v in samples),
                   _integer(payload["warmup_iterations"], "warmup_iterations", 1),
                   _string(payload["receipt_path"], "receipt_path"), _sha(payload["receipt_sha256"], "receipt_sha256"))

    def as_dict(self) -> dict:
        return {"method": self.method, "samples_ms": list(self.samples_ms),
                "warmup_iterations": self.warmup_iterations, "receipt_path": self.receipt_path,
                "receipt_sha256": self.receipt_sha256}


def bootstrap_sum(samples_per_row, *, draws: int, seed: int, offset_ms: float = 0.0) -> dict:
    """The distribution of an operator sum under each row's own samples.

    Every row is resampled with replacement from its OWN measured samples and
    re-reduced by the same median :meth:`OperatorMeasurement.median_ms` reduced
    the priced row by, so this describes only the dispersion the measurement
    itself carries -- not run-to-run serving variance, and not a model.

    ``offset_ms`` is a constant the caller adds to every draw (the fixed
    whole-engine ``prefill_ms``, when a caller carries one). It shifts the
    distribution and contributes no width: the report schema observes no
    samples for the fixed term, so there is no dispersion to draw from and
    inventing one would be a number with no measurement under it.
    """
    if draws < 1:
        raise RuntimePriceError("bootstrap draws must be at least 1")
    rng = random.Random(seed)
    totals = []
    for _ in range(draws):
        totals.append(offset_ms + sum(statistics.median(rng.choices(samples, k=len(samples)))
                                      for samples in samples_per_row))
    totals.sort()
    return {"draws": draws, "seed": seed,
            "p2.5": totals[int(0.025 * draws)], "p50": totals[draws // 2],
            "p97.5": totals[min(draws - 1, int(0.975 * draws))],
            "offset_ms": float(offset_ms),
            "samples_per_row": [len(samples) for samples in samples_per_row]}


@dataclass(frozen=True)
class MeasuredRuntimeRow:
    unit: str
    fmt: str
    binding: RuntimeBinding
    #: One row's price in the spelling its world size licenses: the scalar v2
    #: fields, or the per-rank vector for a unit measured under tensor
    #: parallelism (``parse_row_resources``).
    resources: RuntimeResources | RuntimeRankResources
    prefill: OperatorMeasurement
    decode: OperatorMeasurement | None

    @property
    def key(self) -> tuple[str, str]:
        return self.unit, self.fmt

    def as_dict(self) -> dict:
        return {"unit": self.unit, "format": self.fmt, "binding": self.binding.as_dict(),
                "resources": self.resources.as_dict(), "prefill": self.prefill.as_dict(),
                "decode": self.decode.as_dict() if self.decode else None}


@dataclass(frozen=True)
class MeasuredRuntimeTable:
    table_id: str
    context: RuntimeContext
    cost_sha256: str
    measured_at: str
    valid_until: str
    fixed_assignment: Mapping[str, str]
    fixed_resources: RuntimeResources
    fixed_resources_receipt_path: str
    fixed_resources_receipt_sha256: str
    rows: tuple[MeasuredRuntimeRow, ...]
    source_path: str = ""
    runtime_provenance: Mapping | None = None
    native_receipt_bindings: tuple[Mapping, ...] = ()
    #: Two gates answer two questions, so they get two answers. `admit_native_rows`
    #: attests the per-row prices `build_runtime_resources` hands the DP;
    #: `admit_fixed_resources` attests the whole-engine `fixed_resources` the
    #: allocator adds once. A refusal on the second says nothing about the first,
    #: and folding them into one flag threw a passing native attestation away.
    native_rows_admitted: bool = False
    fixed_resources_admitted: bool = False
    #: Why `fixed_resources` is not admitted, verbatim from the gate, so the
    #: consumer that needs it can say what is owed rather than that something is.
    fixed_resources_refusal: str | None = None

    def as_dict(self) -> dict:
        return {"schema": PROVENANCE_TABLE_SCHEMA if self.runtime_provenance is not None else SCHEMA,
                **({"runtime_provenance": dict(self.runtime_provenance),
                    "native_receipt_bindings": [{key: dict(value) if isinstance(value, Mapping) else value
                                                  for key, value in binding.items()}
                                                 for binding in self.native_receipt_bindings]}
                   if self.runtime_provenance is not None else {}),
                "table_id": self.table_id, "status": "proposal_data",
                "composition": "sequential_operator_sum", "context": self.context.as_dict(),
                "cost_sha256": self.cost_sha256, "measured_at": self.measured_at,
                "valid_until": self.valid_until, "fixed_assignment": dict(self.fixed_assignment),
                "fixed_resources": self.fixed_resources.as_dict(),
                "fixed_resources_receipt_path": self.fixed_resources_receipt_path,
                "fixed_resources_receipt_sha256": self.fixed_resources_receipt_sha256,
                "rows": [row.as_dict() for row in self.rows]}

    def identity(self) -> dict:
        return {"schema": self.as_dict()["schema"], "table_id": self.table_id, "sha256": identity_sha256(self.as_dict()),
                "cost_sha256": self.cost_sha256, "context": self.context.as_dict(),
                "source_path": self.source_path, "status": "proposal_data", "slo_eligible": False,
                "composition": "sequential_operator_sum", "measured_at": self.measured_at,
                "valid_until": self.valid_until, "n_rows": len(self.rows)}


def parse_measured_runtime_table(payload: Mapping, *, expected_context: RuntimeContext,
                                 expected_cost_sha256: str, now: datetime | None = None,
                                 source_path: str = "") -> MeasuredRuntimeTable:
    """Validate an explicit table; caller supplies independent workload/cost identity.

    Parsing validates the evidence declarations. Loading additionally verifies
    local raw receipt content hashes. Version 2 additionally requires producer
    admission; parsing alone cannot supply allocation resources.
    """
    fields = ("schema", "table_id", "status", "composition", "context", "cost_sha256",
                      "measured_at", "valid_until", "fixed_assignment", "fixed_resources",
                      "fixed_resources_receipt_path", "fixed_resources_receipt_sha256", "rows")
    if not isinstance(payload, Mapping):
        raise RuntimePriceError("runtime table: expected an object")
    is_provenance = payload.get("schema") == PROVENANCE_TABLE_SCHEMA
    if is_provenance:
        fields += ("runtime_provenance", "native_receipt_bindings")
    _object(payload, fields, "runtime table")
    if payload["schema"] not in (SCHEMA, PROVENANCE_TABLE_SCHEMA) or payload["status"] != "proposal_data":
        raise RuntimePriceError("runtime table requires current schema and proposal_data status")
    if payload["composition"] != "sequential_operator_sum":
        raise RuntimePriceError("only explicit sequential_operator_sum composition is supported")
    context = parse_runtime_context(payload["context"])
    if is_provenance != (context.runtime_identity_kind is not None):
        raise RuntimePriceError("runtime table/context provenance version mismatch")
    if context != expected_context:
        raise RuntimePriceError("runtime context mismatch against independently supplied expected context")
    cost_sha256 = _sha(payload["cost_sha256"], "cost_sha256")
    if cost_sha256 != _sha(expected_cost_sha256, "expected_cost_sha256"):
        raise RuntimePriceError("stale runtime table: cost payload SHA-256 mismatch")
    measured = _timestamp(payload["measured_at"], "measured_at")
    expires = _timestamp(payload["valid_until"], "valid_until")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise RuntimePriceError("now must have a timezone")
    if not measured <= current < expires:
        raise RuntimePriceError("stale or future runtime table measurement window")
    fixed = payload["fixed_assignment"]
    if not isinstance(fixed, Mapping):
        raise RuntimePriceError("fixed_assignment must be an explicit mapping")
    fixed = MappingProxyType({_string(k, "fixed unit"): _string(v, "fixed format") for k, v in sorted(fixed.items())})
    if not isinstance(payload["rows"], list) or not payload["rows"]:
        raise RuntimePriceError("rows must be a nonempty list")
    rows = []
    seen = set()
    for raw in payload["rows"]:
        _object(raw, ("unit", "format", "binding", "resources", "prefill", "decode"), "runtime row")
        unit, fmt = _string(raw["unit"], "unit"), _string(raw["format"], "format")
        key = unit, fmt
        if key in seen:
            raise RuntimePriceError(f"duplicate runtime row {key}")
        seen.add(key)
        binding = RuntimeBinding.from_dict(raw["binding"])
        if binding.operator_route != context.operator_route(unit, fmt):
            raise RuntimePriceError(f"operator route mismatch for {key}")
        resources = parse_row_resources(raw["resources"], tensor_parallel=context.tensor_parallel,
                                        where=f"runtime row {key} resources")
        prefill = OperatorMeasurement.from_dict(raw["prefill"])
        decode = OperatorMeasurement.from_dict(raw["decode"]) if raw["decode"] is not None else None
        if resources.prefill_ms != prefill.median_ms or resources.decode_ms != (decode.median_ms if decode else None):
            raise RuntimePriceError(f"{key}: resource times must equal medians of measured operator samples")
        rows.append(MeasuredRuntimeRow(unit, fmt, binding, resources, prefill, decode))
    provenance, receipt_bindings = None, ()
    if is_provenance:
        reference = _object(payload["runtime_provenance"], ("path", "sha256"), "runtime provenance artifact")
        provenance = MappingProxyType({"path": _string(reference["path"], "runtime provenance path"),
                                       "sha256": _sha(reference["sha256"], "runtime provenance digest")})
        if not isinstance(payload["native_receipt_bindings"], list):
            raise RuntimePriceError("native receipt bindings must be a list")
        frozen = []
        for item in payload["native_receipt_bindings"]:
            fields = ["unit", "format", "run_id", "panel", "receipt", "memory_trace"]
            if "peer_receipts" in item:
                fields.append("peer_receipts")
            _object(item, tuple(fields), "native receipt binding")
            binding = {key: _string(item[key], "native receipt " + key) for key in ("unit", "format", "run_id")}
            for key in ("panel", "receipt", "memory_trace"):
                ref = _object(item[key], ("path", "sha256"), "native " + key + " artifact")
                binding[key] = MappingProxyType({"path": _string(ref["path"], key + " path"),
                                                 "sha256": _sha(ref["sha256"], key + " digest")})
            if "peer_receipts" in item:
                if not isinstance(item["peer_receipts"], list):
                    raise RuntimePriceError("native peer receipts must be a list")
                peers = []
                for peer in item["peer_receipts"]:
                    _object(peer, ("rank", "receipt", "memory_trace"), "native peer receipt binding")
                    entry = {"rank": _integer(peer["rank"], "native peer receipt rank")}
                    for key in ("receipt", "memory_trace"):
                        ref = _object(peer[key], ("path", "sha256"), "native peer " + key + " artifact")
                        entry[key] = MappingProxyType({"path": _string(ref["path"], key + " path"),
                                                       "sha256": _sha(ref["sha256"], key + " digest")})
                    peers.append(MappingProxyType(entry))
                binding["peer_receipts"] = tuple(peers)
            frozen.append(MappingProxyType(binding))
        receipt_bindings = tuple(frozen)
    return MeasuredRuntimeTable(_string(payload["table_id"], "table_id"), context, cost_sha256,
                                payload["measured_at"], payload["valid_until"], fixed,
                                RuntimeResources.from_dict(payload["fixed_resources"]),
                                _string(payload["fixed_resources_receipt_path"], "fixed_resources_receipt_path"),
                                _sha(payload["fixed_resources_receipt_sha256"], "fixed_resources_receipt_sha256"),
                                tuple(sorted(rows, key=lambda row: row.key)), source_path,
                                provenance, receipt_bindings)


def load_measured_runtime_table(path: str | Path, *, expected_context: RuntimeContext,
                                expected_cost_sha256: str, now: datetime | None = None) -> MeasuredRuntimeTable:
    table = parse_measured_runtime_table(_json(path), expected_context=expected_context,
                                        expected_cost_sha256=expected_cost_sha256, now=now, source_path=str(path))
    receipts = {(table.fixed_resources_receipt_path, table.fixed_resources_receipt_sha256)}
    for row in table.rows:
        for measurement in (row.prefill, row.decode):
            if measurement:
                receipts.add((measurement.receipt_path, measurement.receipt_sha256))
    for receipt, expected in sorted(receipts):
        receipt_path = Path(receipt)
        if not receipt_path.is_absolute():
            receipt_path = Path(path).parent / receipt_path
        try:
            with receipt_path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as exc:
            raise RuntimePriceError(f"cannot read measurement receipt {receipt_path}: {exc}") from exc
        if actual != expected:
            raise RuntimePriceError(f"measurement receipt SHA-256 mismatch: {receipt_path}")
    if table.runtime_provenance is not None:
        from .runtime_provenance import admit_runtime_provenance
        refusal = admit_runtime_provenance(table)
        table = replace(table, native_rows_admitted=True,
                        fixed_resources_admitted=refusal is None,
                        fixed_resources_refusal=refusal)
    return table


def admitted_fixed_resources(table: MeasuredRuntimeTable) -> RuntimeResources:
    """The whole-engine fixed resources, or the reason they have no evidence.

    The per-row prices and the fixed charge are attested by different gates.
    Anything that adds `fixed_resources` to a device budget reads it through
    here, so a fixed-resource refusal is spent where the fixed resources are
    used rather than where the priced rows are.

    A v1 table carries no `runtime_provenance` and calls no gate, so it is
    refused here rather than lent an unattested charge (PQ #560 defect 3: the
    v1 schema stays parseable so history remains readable, but it prices
    nothing and budgets nothing; re-emit through
    `native_receipt_table.emit_native_receipt_table` as v2).
    """
    if table.runtime_provenance is None:
        raise RuntimePriceError(
            "v1 fixed runtime resources carry no producer admission: this table names no "
            "runtime provenance, so no gate attested its fixed charge")
    if not table.fixed_resources_admitted:
        raise RuntimePriceError(
            "v2 fixed runtime resources require full-engine producer admission: "
            + (table.fixed_resources_refusal or "the loader performed no admission"))
    return table.fixed_resources


def admitted_charge_boundary(table: MeasuredRuntimeTable):
    """The boundary the admitted fixed charge composes under, or the refusal.

    A fixed charge has no composition without a boundary: which row terms the
    DP adds and which it reads as witnesses is the boundary's
    ``row_terms_charged``, so the solver takes the spec beside the charge
    (``allocator_solver.solve_runtime_frontier``). It is read through the same
    gate as the charge, and a table whose context names no boundary cannot
    have been admitted, so this never returns a default.
    """
    from .transient_charge_boundary import BOUNDARIES
    admitted_fixed_resources(table)
    name = table.context.transient_charge_boundary
    if name is None:
        raise RuntimePriceError(
            "the table declares no transient charge boundary, so its fixed charge has no "
            "composition")
    spec = BOUNDARIES.get(name)
    if spec is None:
        raise RuntimePriceError(f"transient charge boundary {name!r} is not a registered boundary")
    return spec


#: The one scope under which a table whose fixed charge is refused may still be
#: read, and the only value ``--measured-runtime-fixed-scope`` accepts besides
#: the default. It is asked for by name on the command line and stamped on the
#: document it produces. A scope that switched itself on when the gate refused
#: would be the silent default policy S1 forbids, so this one never does.
SHAPE_ONLY_SCOPE = "shape-only"

#: The fixed-resource scopes ``--measured-runtime-fixed-scope`` accepts.
FIXED_RESOURCE_SCOPES = ("admitted", SHAPE_ONLY_SCOPE)


def shape_only_fixed_resources(table: MeasuredRuntimeTable) -> tuple[RuntimeResources, dict]:
    """The fixed charge for a consumer that reads none of the refused terms.

    ``admit_fixed_resources`` refuses every v2 table at the producer's current
    schema version, and the refusal names its own subject: *"the native-row and
    full-engine transient charge boundary is not versioned, so no candidate
    activation or scratch term may be compared to a priced row"* (PQ debt D37).
    That is about the four device terms in
    ``runtime_provenance.FIXED_TERM_FIELDS`` and about the off-step transient
    peak. It is not about ``runtime_provenance.UNOBSERVED_FIXED_FIELDS`` -- the
    three fields the report schema carries no observation for at all, which the
    same gate refuses only when a table declares one nonzero.

    So this splits the table's declared charge along the gate's own line:

    * the four device terms and the off-step peak are **withheld**. They are
      zeroed in the returned object and named in the stamp, and the caller has
      already refused every path that could compare one to a budget. Nothing
      may publish a device number built from them:
      ``serve_constraints.evaluate_measured_assignment`` withholds
      ``device_memory_bytes`` under this scope rather than publish a sum with a
      term missing from it.
    * ``UNOBSERVED_FIXED_FIELDS`` are **checked, not trusted**. This re-runs the
      gate's own ``if value:`` rule here, because a prefill sweep does read
      ``prefill_ms`` as the offset of its SLO axis.

    Nothing here relaxes a gate. ``admit_fixed_resources`` still refuses, this
    table is still not admitted, ``fixed_resources_admitted`` stays ``False``
    everywhere it is read, and three refusals are *added* on paths that would
    otherwise read what the gate refused.

    Returns ``(resources, stamp)``. The stamp carries the gate's refusal
    verbatim and is written onto whatever document the caller emits.
    """
    from .runtime_provenance import FIXED_TERM_FIELDS, UNOBSERVED_FIXED_FIELDS

    if table.runtime_provenance is None:
        raise RuntimePriceError(
            f"the {SHAPE_ONLY_SCOPE} fixed-resource scope narrows a refusal this table never "
            "received: a v1 table carries no runtime provenance and calls no admission gate")
    if table.fixed_resources_admitted:
        raise RuntimePriceError(
            f"the {SHAPE_ONLY_SCOPE} fixed-resource scope withholds terms this table has "
            "evidence for: its fixed resources are admitted, so read them")
    fixed = table.fixed_resources
    for field in UNOBSERVED_FIXED_FIELDS:
        value = getattr(fixed, field)
        if value:
            raise RuntimePriceError(
                f"the {SHAPE_ONLY_SCOPE} fixed-resource scope reads {field}, and the report "
                f"carries no timing or serialized partition, so this table's fixed {field} "
                f"({value}) has no evidence")
    withheld = sorted([*FIXED_TERM_FIELDS.values(), OFF_STEP_FIELD])
    # Rows that price themselves per rank withhold a second set of terms, on
    # their own axis: the rank vector is published per rank, and no scalar
    # device number may be built from it while the fixed charge and the
    # runtime-global workspace have no versioned value.
    ranked_rows = [row for row in getattr(table, "rows", ())
                   if isinstance(getattr(row, "resources", None), RuntimeRankResources)]
    stamp = {
        "scope": SHAPE_ONLY_SCOPE,
        "fixed_resources_admitted": False,
        "fixed_resources_refusal": table.fixed_resources_refusal,
        "withheld_terms": withheld,
        "withheld_reason": ("the charge boundary between a native row and the full-engine "
                            "partition is not versioned, so these terms have no admitted "
                            "value; every consumer that would compare one to a budget is "
                            "refused instead of being handed a number"),
        "read_terms": {field: getattr(fixed, field) for field in UNOBSERVED_FIXED_FIELDS},
        "read_terms_reason": ("the report schema carries no observation for these fields at "
                              "all; the gate refuses them only when a table declares one "
                              "nonzero, and that check is re-run here"),
        "per_rank_rows": len(ranked_rows),
        "withheld_row_terms": [] if not ranked_rows else list(RANK_WITHHELD_TERMS),
        "withheld_row_reason": (None if not ranked_rows else
                                "these rows carry one value per rank, so no scalar device "
                                "total is published for them; see "
                                "measured_runtime_prices.compose_rank_totals"),
        "certifies_placement": False,
    }
    return replace(fixed, **{field: 0 for field in FIXED_TERM_FIELDS.values()},
                   **{OFF_STEP_FIELD: None}), stamp


def reconcile_serving_unit_rows(
        table: "MeasuredRuntimeTable", *,
        menu_members: Mapping[tuple[str, str], Mapping[str, str]],
) -> "MeasuredRuntimeTable":
    """Key a whole-owner row onto the aggregated serving unit the DP prices.

    A packed-MoE serving unit is one DP item whose name the allocator's own
    aggregation produces (``allocator_candidates``'s ``.__packed_serving__.``
    spelling). A producer that measured the whole owner writes the unit it
    observed -- the owner's own module name, which is also what its members are
    named under -- because that is what it measured. The two names are one
    serving unit seen from two sides, and they are reconciled here: by the
    row's own member roster, never by name similarity.

    ``menu_members`` maps each DP option ``(unit, format)`` to the member
    ``{name: format}`` map that option expands to. A row already keyed by a DP
    option is left exactly as it was. A row whose roster is exactly one
    option's roster is re-keyed to that option, after its declared operator
    route is checked against the route this table's own context names for the
    row's own unit -- so the route stays an independently supplied coordinate
    rather than a label the row supplies about itself. A roster that matches
    more than one option refuses by name, because no single DP unit may read
    that row. A roster that matches none is left alone, and the existing
    missing-row refusal fires where it is spent.
    """
    if not menu_members:
        return table
    by_roster: dict[tuple, list[tuple[str, str]]] = {}
    for key, members in menu_members.items():
        by_roster.setdefault((tuple(sorted(dict(members).items())), key[1]), []).append(key)
    rows = []
    for row in table.rows:
        if row.key in menu_members:
            rows.append(row)
            continue
        roster = (tuple(sorted(dict(row.binding.member_formats).items())), row.fmt)
        matches = by_roster.get(roster, ())
        if len(matches) > 1:
            raise RuntimePriceError(
                f"measured runtime row {row.key!r} prices a member roster that is exactly the "
                f"roster of {sorted(matches)}, so it is not one serving unit's row and no "
                "single DP unit may read it")
        if not matches:
            rows.append(row)
            continue
        (unit, _fmt), = matches
        declared_route = table.context.operator_route(row.unit, row.fmt)
        if declared_route != row.binding.operator_route:
            raise RuntimePriceError(
                f"measured runtime row {row.key!r} declares operator route "
                f"{row.binding.operator_route!r} where this table's context names "
                f"{declared_route!r} for its own unit")
        rows.append(replace(row, unit=unit))
    return replace(table, rows=tuple(rows))


#: The expert-role spellings a routed serving unit's members carry, mapped to
#: which axis of that member's own 2-D geometry is its intermediate one. The
#: tensor-parallel cut divides the intermediate axis, so gate/up (column
#: parallel: the output features are the container's rows) are cut on their
#: first axis and down (row parallel: the input features are its columns) on
#: its second. Both vocabularies are listed because the producer keeps the
#: source's own spelling -- `prismaquant.native_moe_panel.ROLE_PROJECTIONS`
#: holds the same table for the native panel, and a member whose name is in
#: neither is not a routed expert projection at all.
MOE_MEMBER_ROLE_AXES = {"w1": 0, "w3": 0, "w2": 1,
                        "gate_proj": 0, "up_proj": 0, "down_proj": 1}


def rank_local_member_shapes(source_shapes: Mapping[str, Sequence[int]], *,
                             tensor_parallel: int,
                             where: str = "runtime binding") -> dict[str, tuple[int, ...]]:
    """Each routed member's OWN rank-local geometry, from the trusted context.

    ``source_shapes`` is the module geometry the joint operator identity binds
    -- the probe's full source, which is what a quality identity must be taken
    on. The geometry the runtime binding carries is the same tensor CUT on the
    intermediate axis by the table context's ``tensor_parallel``, and the two
    are different fields that agree only at a world of one: comparing them as
    one field refuses every real TP2 row.

    The cut is validated rather than assumed: the extent must divide by the
    world, and every rank's own cut times the world must be exactly the source,
    so the canonical combination of all ranks is the container the wire
    identity names. A member that carries no routed role spelling is a dense
    Linear: its own geometry IS the binding, and it has no cut to take. An
    option that mixes the two is refused, because no single rule then says
    which members are cut.
    """
    world = _integer(tensor_parallel, "tensor_parallel", 1)
    routed = [unit for unit in source_shapes
              if str(unit).rsplit(".", 1)[-1] in MOE_MEMBER_ROLE_AXES]
    shapes = {unit: tuple(_integer(dim, f"{unit} extent", 1) for dim in shape)
              for unit, shape in source_shapes.items()}
    if not routed:
        return shapes
    if len(routed) != len(shapes):
        raise RuntimePriceError(
            f"{where}: this serving unit mixes routed expert projections with members that carry "
            "no expert role spelling, so no single rule says which of them is cut")
    local = {}
    for unit, shape in shapes.items():
        if len(shape) != 2:
            raise RuntimePriceError(
                f"{where}: routed member {unit!r} carries a {len(shape)}-D geometry, and the "
                "intermediate axis of a 2-D Linear is what the cut divides")
        axis = MOE_MEMBER_ROLE_AXES[str(unit).rsplit(".", 1)[-1]]
        if shape[axis] % world:
            raise RuntimePriceError(
                f"{where}: {unit}'s intermediate extent {shape[axis]} is not divisible by this "
                f"table's tensor_parallel {world}")
        cut = list(shape)
        cut[axis] = shape[axis] // world
        local[unit] = tuple(cut)
    return local


def build_runtime_resources(table: MeasuredRuntimeTable, candidates: Mapping[str, list], *,
                            expected_bindings: Mapping[tuple[str, str], RuntimeBinding]) -> dict[tuple[str, str], RuntimeResources]:
    """Price every candidate exactly; no family fallback or unmeasured group sums."""
    # The native-row gate, not the fixed-resource one: this function reads
    # `row.resources` for priced candidates and never touches
    # `table.fixed_resources`, and `admit_native_rows` is what attests those
    # rows against their receipts. The fixed charge is gated at its own
    # consumer, `admitted_fixed_resources`.
    #
    # A v1 table carries no `runtime_provenance` and calls no gate, so it is
    # refused here rather than priced unattested (PQ #560 defect 3: the v1
    # schema stays parseable so history remains readable, but it prices
    # nothing and budgets nothing).
    if table.runtime_provenance is None:
        raise RuntimePriceError(
            "v1 runtime prices carry no producer admission: this table names no runtime "
            "provenance, so no gate attested its rows")
    if not table.native_rows_admitted:
        raise RuntimePriceError(
            "v2 runtime prices require native-row producer admission through the loader")
    rows = {row.key: row for row in table.rows}
    result = {}
    for unit, options in sorted(candidates.items()):
        for candidate in sorted(options, key=lambda c: c.fmt):
            key = unit, candidate.fmt
            if key in result:
                raise RuntimePriceError(f"duplicate candidate {key}")
            if key not in rows or key not in expected_bindings:
                raise RuntimePriceError(f"missing measured runtime row or independent binding for {key}")
            row = rows[key]
            binding = expected_bindings[key]
            if not isinstance(binding, RuntimeBinding) or binding != row.binding:
                raise RuntimePriceError(f"runtime operator binding mismatch for {key}")
            members = candidate.member_formats if getattr(candidate, "member_formats", None) is not None else {unit: candidate.fmt}
            if dict(binding.member_formats) != members:
                raise RuntimePriceError(f"whole serving-unit member formats mismatch for {key}")
            if type(candidate.memory_bytes) is not int:
                raise RuntimePriceError(f"serialized byte mismatch for {key}")
            if isinstance(row.resources, RuntimeRankResources):
                # One artifact, one charge: the vector carries the module's
                # canonical wire extent, counted once. Summing a per-rank view
                # of the same container would double count the artifact.
                if row.resources.wire_bytes != candidate.memory_bytes:
                    raise RuntimePriceError(
                        f"whole-unit wire byte mismatch for {key}: the candidate prices "
                        f"{candidate.memory_bytes} bytes where the owner's canonical wire "
                        f"extent is {row.resources.wire_bytes}")
            elif row.resources.serialized_bytes != candidate.memory_bytes:
                raise RuntimePriceError(f"serialized byte mismatch for {key}")
            result[key] = row.resources
    return result


#: The per-rank totals one expanded assignment composes to. Its own schema
#: because it is its own object: an operator-side vector, indexed by rank,
#: carrying no scalar reduction and no device total.
RANK_TOTALS_SCHEMA = "prismaquant.runtime_rank_totals.v1"
RANK_TOTALS_WITHHELD_REASON = (
    "a device total needs the fixed whole-engine charge, and that charge has no admitted value "
    "at this schema version: the fixed terms come from "
    "runtime_provenance.admit_fixed_resources, which refuses every v2 table today. The runtime-"
    "global workspace is no longer withheld -- it is charged per rank by "
    f"RANK_WORKSPACE_RULE={RANK_WORKSPACE_RULE!r}")


@dataclass(frozen=True)
class RankTotals:
    """One expanded assignment's operator terms, indexed by rank.

    ``serialized_bytes``/``resident_bytes``/``workspace_resident_bytes`` add
    across the assignment's priced rows; ``peak_scratch_bytes`` and
    ``activation_bytes`` are independent per-rank maxima over those rows. Every
    tuple is published whole: a consumer that wants one number still has to
    choose the reduction, and this object never chooses it for them.
    """

    world_size: int
    #: One canonical wire extent for the whole assignment: each priced unit's
    #: artifact counted once, never once per rank.
    wire_bytes: int
    resident_bytes: tuple[int, ...]
    peak_scratch_bytes: tuple[int, ...]
    activation_bytes: tuple[int, ...]
    workspace_resident_bytes: tuple[int, ...]

    def as_dict(self) -> dict:
        return {"schema": RANK_TOTALS_SCHEMA, "world_size": self.world_size,
                "wire_bytes": self.wire_bytes,
                "ranks": [{"rank": rank,
                           "resident_bytes": self.resident_bytes[rank],
                           "peak_scratch_bytes": self.peak_scratch_bytes[rank],
                           "activation_bytes": self.activation_bytes[rank],
                           "workspace_resident_bytes": self.workspace_resident_bytes[rank]}
                          for rank in range(self.world_size)],
                "withheld_terms": list(RANK_WITHHELD_TERMS),
                "withheld_reason": RANK_TOTALS_WITHHELD_REASON}


def compose_rank_totals(resources) -> RankTotals:
    """Compose priced rows into per-rank terms without reducing over ranks.

    Accepts the scalar spelling only where it means what it says: a table whose
    world is one has one rank, and its scalar row *is* that rank's record. A
    scalar row in a world of more than one is refused by
    :func:`parse_row_resources` before it ever reaches here, so the two
    spellings cannot be mixed into a reduction by this function.
    """
    rows = list(resources)
    if not rows:
        raise RuntimePriceError("per-rank totals require at least one priced row")
    ranked = [row for row in rows if isinstance(row, RuntimeRankResources)]
    if ranked:
        world = ranked[0].world_size
        if any(row.world_size != world for row in ranked):
            raise RuntimePriceError(
                "priced rows disagree about their world size, so no per-rank total covers one box")
        if world != 1 and any(not isinstance(row, RuntimeRankResources) for row in rows):
            raise RuntimePriceError(
                f"a world of {world} prices every row per rank, and this assignment mixes a "
                "scalar row into it")
    else:
        world = 1
    wire_bytes = 0
    resident = [0] * world
    scratch = [0] * world
    activation = [0] * world
    workspace = [dict() for _ in range(world)]
    for row in rows:
        if isinstance(row, RuntimeRankResources):
            wire_bytes += row.wire_bytes
            for rank, entry in enumerate(row.ranks):
                resident[rank] += entry.resident_bytes
                scratch[rank] = max(scratch[rank], entry.peak_scratch_bytes)
                activation[rank] = max(activation[rank], entry.activation_bytes)
                identity = entry.workspace_sha256
                prior = workspace[rank].get(identity)
                if prior is not None and prior != entry.workspace_resident_bytes:
                    raise RuntimePriceError(
                        f"two priced rows carry workspace identity {identity} at different "
                        f"sizes on rank {rank} ({prior} and {entry.workspace_resident_bytes}); "
                        "one frozen allocation cannot be two allocations")
                workspace[rank][identity] = entry.workspace_resident_bytes
        else:
            wire_bytes += row.serialized_bytes
            resident[0] += row.resident_bytes
            scratch[0] = max(scratch[0], row.peak_scratch_bytes)
            activation[0] = max(activation[0], row.activation_bytes)
    return RankTotals(world, wire_bytes, tuple(resident), tuple(scratch),
                      tuple(activation), tuple(sum(ids.values()) for ids in workspace))


def _rank_vector(value, world: int, where: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != world:
        size = len(value) if isinstance(value, (list, tuple)) else "no"
        raise RuntimePriceError(
            f"{where}: a per-rank charge must carry exactly one record per rank: world size "
            f"{world} against {size} records")
    return tuple(_integer(entry, f"{where} rank record") for entry in value)


def admit_rank_budgets(totals: RankTotals, *, budgets_per_rank, charge_per_rank,
                       charge_refusal: str | None,
                       where: str = "per-rank device budget") -> tuple[int, ...]:
    """Admit every rank against its own budget, or name the ranks that fail.

    The check is per rank and the numbers it compares are that rank's own: a
    world where one rank fits and its peer does not is refused, and it is
    refused whether or not the two ranks' *sum* or *mean* would have passed --
    those two reductions are exactly what this function exists to keep out of a
    placement decision.

    A device total also needs the fixed whole-engine charge per rank, so a
    caller with no admitted charge is refused by name (``charge_refusal`` is
    the gate's own refusal text, passed through verbatim) rather than handed a
    total that silently omits it. Returns the admitted per-rank totals.
    """
    world = totals.world_size
    budgets = _rank_vector(budgets_per_rank, world, where)
    refuses = []
    if charge_per_rank is None:
        refuses.append(
            "no admitted per-rank fixed charge prices the fixed source parameters, router, KV "
            "or activation transient, and a device total without it is a number with a charge "
            f"missing from it: {charge_refusal or 'the loader performed no admission'}")
        charge = (0,) * world
    else:
        charge = _rank_vector(charge_per_rank, world, where)
    # The runtime-global workspace is charged per rank by RANK_WORKSPACE_RULE,
    # which compose_rank_totals already applied: one frozen identity counts once
    # per rank no matter how many rows view it.
    totals_per_rank = tuple(totals.resident_bytes[rank] + totals.activation_bytes[rank]
                            + totals.peak_scratch_bytes[rank]
                            + totals.workspace_resident_bytes[rank] + charge[rank]
                            for rank in range(world))
    over = [(rank, totals_per_rank[rank], budgets[rank]) for rank in range(world)
            if totals_per_rank[rank] > budgets[rank]]
    for rank, total, budget in over:
        refuses.append(
            f"rank {rank} needs {total} device bytes where its own budget is {budget}")
    if refuses:
        raise RuntimePriceError(f"{where}: " + "; ".join(refuses))
    return totals_per_rank


@dataclass(frozen=True)
class RankDeviceBounds:
    """One per-rank device budget, with the fixed charge it may be read against.

    A rank budget is only a constraint once the fixed whole-engine charge per
    rank is known: without it, an assignment could be admitted against the
    operator terms alone and then OOM on the fixed source parameters, router,
    KV or activation transient nobody charged. So the two arrive together, and
    the provenance decides what may be done with them:

    * ``pending_measurement`` -- the charge has no value yet. The rank
      *dimensions* still price (a common unknown constant added to every
      candidate cannot reorder them), but no device total is published and no
      rank is admitted. This is the state of every table today.
    * ``recomputed_full_engine_partition`` -- the charge came from a per-rank
      full-engine partition this consumer recomputed, and a rank may be
      admitted against ``budgets_per_rank``.

    A declared zero charge is refused outright: it is indistinguishable from
    forgetting a term, and ``None`` is how this contract spells "not measured".
    An admitted charge is not declarable either: a number written into a
    document is a claim, and ``recomputed`` is the one constructor that only
    accepts a value a recomputation produced. ``from_dict`` therefore loads
    pending bounds and refuses an admitted spelling by name rather than reading
    its charge, so a forged report reference or a hand-edited charge cannot
    reach a placement decision.
    """

    world_size: int
    provenance: str
    budgets_per_rank: tuple[int, ...]
    charge_per_rank: tuple[int, ...] | None
    evidence: Mapping | None

    def __post_init__(self):
        _integer(self.world_size, "world_size", 1)
        if self.provenance not in RANK_DEVICE_PROVENANCE:
            raise RuntimePriceError(
                f"rank device bounds provenance {self.provenance!r} is not one of "
                f"{list(RANK_DEVICE_PROVENANCE)}")
        budgets = _rank_vector(self.budgets_per_rank, self.world_size, "rank device budgets")
        for index, budget in enumerate(budgets):
            if budget < 1:
                raise RuntimePriceError(
                    f"rank device budgets must be positive, and rank {index}'s is {budget}")
        if self.provenance == "pending_measurement":
            if self.charge_per_rank is not None or self.evidence is not None:
                raise RuntimePriceError(
                    "a pending per-rank charge carries no value and no evidence; declaring one "
                    "is claiming a measurement this contract says does not exist")
            charge = None
        else:
            # Only a recomputed full-engine partition may carry a charge, and the
            # recomputation is not something a document can assert about itself:
            # `recomputed` is the constructor for that verdict. Reading a charge
            # out of a payload would let a caller declare "recomputed" over an
            # arbitrary number and have every downstream gate believe it.
            raise RuntimePriceError(
                f"a declared per-rank charge is not evidence: only a recomputed full-engine "
                f"partition may carry one, and this value was constructed with provenance "
                f"{self.provenance!r} and no recomputation behind it. Load a sealed per-rank "
                f"partition with runtime_provenance.recompute_rank_fixed_charge and build the "
                f"bounds with RankDeviceBounds.recomputed")
        object.__setattr__(self, "budgets_per_rank", budgets)
        object.__setattr__(self, "charge_per_rank", charge)
        if self.evidence is not None:
            object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    @classmethod
    def recomputed(cls, *, world_size: int, budgets_per_rank, charge_per_rank,
                   recomputation) -> "RankDeviceBounds":
        """The one construction path an admitted per-rank charge may take.

        ``recomputation`` must be an actual
        ``runtime_provenance.RankFixedCharge``: the verdict of reading the
        sealed per-rank partition and recomputing each rank's own four fixed
        terms from that rank's own sealed capture. Every value below is read
        from that object rather than accepted from the caller, so a charge the
        recomputation did not produce -- or a budget vector it did not cover --
        cannot become bounds.

        The type is checked, not duck-typed. ``getattr`` on an arbitrary object
        accepts a ``SimpleNamespace`` carrying the two attribute names a
        verdict happens to have, which is a *claim* to be a recomputation
        rather than one, and this constructor is the one place that decides
        whether a per-rank charge may reach a placement.
        """
        from .runtime_provenance import RankFixedCharge

        if not isinstance(recomputation, RankFixedCharge):
            raise RuntimePriceError(
                "an admitted per-rank charge is read from a "
                "runtime_provenance.RankFixedCharge recomputation, and this value is a "
                f"{type(recomputation).__name__}: carrying the attribute names a verdict has "
                "is not the verdict. Load the sealed per-rank partition with "
                "runtime_provenance.recompute_rank_fixed_charge")
        world = _integer(world_size, "world_size", 1)
        if recomputation.world_size != world:
            raise RuntimePriceError(
                f"rank device bounds cover a world of {world} where the recomputation covers "
                f"{recomputation.world_size!r}")
        budgets = _rank_vector(budgets_per_rank, world, "rank device budgets")
        for index, budget in enumerate(budgets):
            if budget < 1:
                raise RuntimePriceError(
                    f"rank device budgets must be positive, and rank {index}'s is {budget}")
        charge = _rank_vector(charge_per_rank, world, "rank fixed charge")
        recomputed = tuple(recomputation.charge_per_rank)
        if recomputed != charge:
            raise RuntimePriceError(
                f"the declared per-rank charge {charge} is not the recomputed one "
                f"{recomputed}; a charge is read from the recomputation, never supplied beside it")
        for index, value in enumerate(charge):
            if value < 1:
                raise RuntimePriceError(
                    "a zero per-rank fixed charge is not evidence; the recomputation must show "
                    f"the fixed terms this rank actually holds (rank {index} recomputed {value})")
        evidence = recomputation.evidence
        if (not isinstance(evidence, Mapping)
                or set(evidence) != set(_RANK_DEVICE_EVIDENCE_FIELDS)
                or evidence["per_rank_partition"] is not True):
            raise RuntimePriceError(
                "an admitted per-rank charge references the recomputed full-engine partition "
                "it came from, and this recomputation carries no such reference")
        instance = object.__new__(cls)
        for name, value in (("world_size", world),
                            ("provenance", "recomputed_full_engine_partition"),
                            ("budgets_per_rank", budgets), ("charge_per_rank", charge),
                            ("evidence", MappingProxyType(dict(evidence)))):
            object.__setattr__(instance, name, value)
        return instance

    @property
    def admits_ranks(self) -> bool:
        return self.charge_per_rank is not None

    def per_rank_headroom(self) -> tuple[int, ...]:
        """Each rank's own remaining device budget, or a refusal while pending."""
        if not self.admits_ranks:
            raise RuntimePriceError(
                "per-rank headroom is unknown: the fixed whole-engine charge per rank is "
                "pending measurement, and subtracting nothing is not subtracting zero")
        return tuple(budget - charge for budget, charge
                     in zip(self.budgets_per_rank, self.charge_per_rank))

    def as_dict(self) -> dict:
        return {"schema": RANK_DEVICE_BOUNDS_SCHEMA, "world_size": self.world_size,
                "provenance": self.provenance,
                "budgets_per_rank": list(self.budgets_per_rank),
                "charge_per_rank": (None if self.charge_per_rank is None
                                    else list(self.charge_per_rank)),
                "evidence": None if self.evidence is None else dict(self.evidence)}

    @classmethod
    def from_dict(cls, payload: Mapping, *, where: str = "rank device bounds") -> "RankDeviceBounds":
        fields = _object(payload, RANK_DEVICE_BOUNDS_FIELDS, where)
        if fields["schema"] != RANK_DEVICE_BOUNDS_SCHEMA:
            raise RuntimePriceError(f"{where}: unknown rank device bounds schema "
                                    f"{fields['schema']!r}")
        return cls(fields["world_size"], fields["provenance"], tuple(fields["budgets_per_rank"]),
                   None if fields["charge_per_rank"] is None else tuple(fields["charge_per_rank"]),
                   fields["evidence"])


def pending_rank_device_bounds(world_size: int, budgets_per_rank=None) -> RankDeviceBounds:
    """The bounds a campaign has today: budgets declared, fixed charge pending.

    Exists so a caller states the pending state once, in one object, instead of
    passing ``None`` around where a zero could be read as a charge.
    """
    budgets = budgets_per_rank if budgets_per_rank is not None else (1,) * world_size
    return RankDeviceBounds(world_size, "pending_measurement", tuple(budgets), None, None)
