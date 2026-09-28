"""Shape-time price table: kernel time keyed by served shape, not by unit.

``prismaquant.shape_runtime_prices.v1`` is PACT's time input (PQ #1583,
design ``PACT-APPLICATION-DESIGN-2026-09-28`` §1). It exists beside
``measured_runtime_prices.v2``, not inside it. A v2 row is keyed
``(unit, format)`` and binds the unit's weights through
``member_operator_identity_sha256``. A shape row is keyed

    (structure, rank_local_shape, family, rate_q256, M)

and carries no weight identity, because kernel time is a function of the
served shape and of the kernel lane that dispatches it, not of the bytes
inside the operator (Rob, 2026-09-28). The rate stays in the key because the
rate can decide the lane: after Tessera #685 a routed E4M3 stack at R1024
takes the fused lane (17.5 ms at M=2048) and the same stack at R896 takes the
compact adapter (109 ms).

What one row says, and what the table never says
------------------------------------------------
* A row is one operator's repeated GPU time at batch 1 and ``M`` prompt rows,
  measured on the image, contract, tensor-parallel world, platform, residency
  and execution mode the context names, with the ``(symbol, decoder)`` launch
  the route telemetry recorded for the timed forward (``kernel_lane``).
* ``claims`` is fixed: ``time_claim: operator_sum_proposal``,
  ``certifies_placement: false``, ``served_p95: not_claimed``. A sum of shape
  rows proposes an ordering of assignments. It certifies no p95 TTFT, no
  placement and no residency, and the table prices no bytes on the device.
* No per-unit receipt and no weight identity are claimed.

Admission (principle 14)
------------------------
:func:`admit_shape_table` checks the context against the independently
supplied scope (the pinned contract digest and Tessera commit, the export's
runtime image, the serve world, platform, residency and execution mode) and
then checks every row, and every rate a pool prices, against the pinned
contract itself: a lane cell for ``(platform, family, structure, regime,
residency, image, execution mode)`` must cover the rate and name the row's
launch in ``executes``, and the published predicate of the lane that serves
that launch must admit the rate (``lane_eligibility.cell_lane_admits``, the
one decision path, asked about that one launch). So a row timed on
``native_routed_fused_window`` at R1024 admits, and one at R896 is refused
because the fused lane's ``requires`` does not read that wire.

Unit time
---------
A DP serving unit is ONE served operator: fused siblings are one merged GEMM
(which is why they must share a format) and a packed expert group is one
grouped kernel over the rank-local expert stack. So a unit's time is the row
of its operator, keyed by the operator's rank-local shape, which
:func:`served_operator` derives from the members' own shapes through
``measured_runtime_prices.rank_local_member_shapes``. (The design's
"sum over members" is this sum over served operators; each unit has one.) A
member set that no rule composes into one operator, a member with no
tensor-parallel cut rule at a world above one, a mixed-rate operator, and an
option with no time row are each an UNPRICED option: absent from the
time-aware candidate set and reported as a measurement gap, never priced at
zero (design §3.3).

Rate pools
----------
Pooling across rates is never a default. A table may declare a
``rate_pools`` entry: these rates of one ``(structure, shape, family)``
execute on one kernel lane, and their samples are one distribution. Admission
refuses a pool whose source rows disagree about the lane, and checks every
pooled rate against the contract as it would a row. The pooled samples are
the concatenation of the source rows' own samples, so the spread across rates
flows into :func:`operator_sum_bootstrap` instead of a tolerance deciding
anything. A pool whose rows sit at one rate measured no cross-rate spread,
and its record says so.

The Tessera receipt (``tessera.shape_time_panel.v1``, RobTand/tessera#688)
has no published schema yet; :func:`consume_shape_time_panel` is the named
stub its consumer will replace.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, replace
from statistics import median
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .digests import DIRECT_ASCII_STRICT, file_sha256hex
from .lane_eligibility import (
    STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE, STRUCTURES, EligibilityTable,
    LaneEligibilityError, ServingContext, cell_lane_admits,
    cell_matches_serving_context, resolve_payload_rung,
)
from .measured_runtime_prices import (
    MOE_MEMBER_ROLE_AXES, OperatorMeasurement, RuntimePriceError, RuntimeResources,
    _integer, _json, _object, _string, bootstrap_sum, rank_local_member_shapes,
)

SCHEMA = "prismaquant.shape_runtime_prices.v1"
ADMISSION_SCHEMA = "prismaquant.shape_runtime_admission.v1"
TIME_CLAIM = "operator_sum_proposal"
#: The table's whole claim, fixed. A table that states anything else about
#: itself is refused rather than read.
CLAIMS = MappingProxyType({"time_claim": TIME_CLAIM, "certifies_placement": False,
                           "served_p95": "not_claimed"})
#: A shape row times actual GPU operator execution; a synchronized wall clock
#: around a launch is not accepted here (design §1.3).
TIMING_METHODS = ("cuda_events", "gpu_profiler")
#: Tessera's ``contract.CENSUS_PHASE_REGIMES``: its lane cells call a one-row
#: forward ``decode`` and every M > 1 forward ``batch``. Mapped once, here, and
#: checked against the pinned table's own declared regimes at admission.
DECODE_M = 1
REGIME_DECODE = "decode"
REGIME_BATCH = "batch"

_DENSE_SHAPE = re.compile(r"([1-9][0-9]*)x([1-9][0-9]*)")
_ROUTED_SHAPE = re.compile(
    r"E([1-9][0-9]*):w13=([1-9][0-9]*)x([1-9][0-9]*):w2=([1-9][0-9]*)x([1-9][0-9]*)")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA = re.compile(r"[0-9a-f]{64}")

identity_sha256 = DIRECT_ASCII_STRICT.sha256


class ShapeRuntimeError(RuntimePriceError):
    """A shape table, its admission, or a unit derivation that is refused."""


def regime_for_m(m: int) -> str:
    """The contract regime a forward of ``m`` prompt rows exercises."""
    return REGIME_DECODE if _integer(m, "M", 1) == DECODE_M else REGIME_BATCH


def validate_shape(structure: str, shape: str, where: str = "rank_local_shape") -> str:
    """Check the operator-shape grammar for its structure; return it unchanged.

    ``dense``: ``NxK`` -- the served GEMM's rank-local output and input
    extents (a fused group's N is its members' concatenated outputs).
    ``routed_moe``: ``E<n>:w13=<2I>x<H>:w2=<H>x<I>`` -- the rank-local expert
    stack one grouped kernel runs over.
    """
    if structure == STRUCTURE_DENSE:
        if not _DENSE_SHAPE.fullmatch(shape):
            raise ShapeRuntimeError(f"{where}: dense shape {shape!r} is not 'NxK'")
        return shape
    if structure == STRUCTURE_ROUTED_MOE:
        match = _ROUTED_SHAPE.fullmatch(shape)
        if not match:
            raise ShapeRuntimeError(
                f"{where}: routed shape {shape!r} is not 'E<n>:w13=<2I>x<H>:w2=<H>x<I>'")
        _n, w13_n, w13_k, w2_n, w2_k = (int(v) for v in match.groups())
        if w13_n != 2 * w2_k or w13_k != w2_n:
            raise ShapeRuntimeError(
                f"{where}: routed shape {shape!r} is not one expert stack: w13 must be "
                "[2I, H] and w2 [H, I]")
        return shape
    raise ShapeRuntimeError(f"{where}: structure must be one of {sorted(STRUCTURES)}")


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ShapeRuntimeContext:
    """Where every row was timed. No field has a default."""

    runtime_image_digest: str
    tessera_commit: str
    contract_sha256: str
    tensor_parallel: int
    platform: str
    execution_mode: str
    residency: str
    batch_size: int
    regimes: tuple[int, ...]

    FIELDS = ("runtime_image_digest", "tessera_commit", "contract_sha256", "tensor_parallel",
              "platform", "execution_mode", "residency", "batch_size", "regimes")

    def __post_init__(self):
        try:
            # The serving-context grammar owns image, residency and mode spelling.
            ServingContext(self.platform, STRUCTURE_DENSE, self.residency,
                           self.runtime_image_digest, self.execution_mode)
        except LaneEligibilityError as exc:
            raise ShapeRuntimeError(f"shape table context: {exc}") from exc
        if not isinstance(self.tessera_commit, str) or not _COMMIT.fullmatch(self.tessera_commit):
            raise ShapeRuntimeError("shape table context: tessera_commit must be a 40-hex commit")
        if not isinstance(self.contract_sha256, str) or not _SHA.fullmatch(self.contract_sha256):
            raise ShapeRuntimeError("shape table context: contract_sha256 must be lowercase SHA-256")
        _integer(self.tensor_parallel, "tensor_parallel", 1)
        if self.batch_size != 1 or type(self.batch_size) is not int:
            raise ShapeRuntimeError(
                "shape table context: batch_size must be 1; a row prices one sequence of M rows")
        regimes = tuple(self.regimes) if isinstance(self.regimes, (list, tuple)) else None
        if (not regimes or any(type(m) is not int or m < 1 for m in regimes)
                or list(regimes) != sorted(set(regimes))):
            raise ShapeRuntimeError(
                "shape table context: regimes must be a nonempty ascending list of distinct M >= 1")
        object.__setattr__(self, "regimes", regimes)

    def as_dict(self) -> dict:
        payload = {name: getattr(self, name) for name in self.FIELDS}
        payload["regimes"] = list(self.regimes)
        return payload


@dataclass(frozen=True)
class KernelLane:
    symbol: str
    decoder: str

    def as_pair(self) -> tuple[str, str]:
        return self.symbol, self.decoder

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "decoder": self.decoder}

    @classmethod
    def from_dict(cls, payload: Any, where: str) -> "KernelLane":
        body = _object(payload, ("symbol", "decoder"), where)
        return cls(_string(body["symbol"], where + ".symbol"), _string(body["decoder"], where + ".decoder"))


@dataclass(frozen=True, order=True)
class ShapeKey:
    structure: str
    rank_local_shape: str
    family: str
    rate_q256: int
    m: int

    def label(self) -> str:
        return f"{self.structure}|{self.rank_local_shape}|{self.family}|R{self.rate_q256}|M{self.m}"


@dataclass(frozen=True)
class ShapeRow:
    key: ShapeKey
    kernel_lane: KernelLane
    measurement: OperatorMeasurement

    def as_dict(self) -> dict:
        return {"structure": self.key.structure, "rank_local_shape": self.key.rank_local_shape,
                "family": self.key.family, "rate_q256": self.key.rate_q256, "m": self.key.m,
                "kernel_lane": self.kernel_lane.as_dict(), "measurement": self.measurement.as_dict()}


@dataclass(frozen=True)
class RatePool:
    """An explicit statement that several rates share one time distribution."""

    structure: str
    rank_local_shape: str
    family: str
    kernel_lane: KernelLane
    rates_q256: tuple[int, ...]

    def covers(self, structure: str, shape: str, family: str, rate: int) -> bool:
        return ((structure, shape, family) == (self.structure, self.rank_local_shape, self.family)
                and rate in self.rates_q256)

    def as_dict(self) -> dict:
        return {"structure": self.structure, "rank_local_shape": self.rank_local_shape,
                "family": self.family, "kernel_lane": self.kernel_lane.as_dict(),
                "rates_q256": list(self.rates_q256)}


@dataclass(frozen=True)
class PricedTime:
    """One looked-up time: the median, its samples, and what produced them."""

    key: ShapeKey
    median_ms: float
    samples_ms: tuple[float, ...]
    kernel_lane: KernelLane
    #: The identity the bootstrap resamples: a row's own key label, or the
    #: pool's, so every unit reading one measurement shares one draw.
    source_id: str
    source: str  # "row" | "rate_pool"
    pool: Mapping | None = None


@dataclass(frozen=True)
class ShapeRuntimeTable:
    table_id: str
    context: ShapeRuntimeContext
    rows: tuple[ShapeRow, ...]
    rate_pools: tuple[RatePool, ...] = ()
    source_path: str = ""
    admission: Mapping | None = None
    _index: Mapping = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_index", MappingProxyType({row.key: row for row in self.rows}))

    @property
    def admitted(self) -> bool:
        return self.admission is not None

    def as_dict(self) -> dict:
        return {"schema": SCHEMA, "table_id": self.table_id, "status": "proposal_data",
                "composition": "sequential_operator_sum", "claims": dict(CLAIMS),
                "context": self.context.as_dict(), "rows": [row.as_dict() for row in self.rows],
                "rate_pools": [pool.as_dict() for pool in self.rate_pools]}

    def identity(self) -> dict:
        return {"schema": SCHEMA, "table_id": self.table_id, "sha256": identity_sha256(self.as_dict()),
                "context": self.context.as_dict(), "claims": dict(CLAIMS),
                "source_path": self.source_path, "status": "proposal_data", "slo_eligible": False,
                "composition": "sequential_operator_sum", "n_rows": len(self.rows),
                "n_rate_pools": len(self.rate_pools),
                "admission": None if self.admission is None else dict(self.admission)}

    def pool_for(self, structure: str, shape: str, family: str, rate: int) -> RatePool | None:
        for pool in self.rate_pools:
            if pool.covers(structure, shape, family, rate):
                return pool
        return None

    def lookup(self, key: ShapeKey) -> PricedTime | None:
        """The time for one key, or ``None`` when the table has no measurement.

        A pool that covers the key's rate answers for it (the pool is the
        table's explicit statement); otherwise the exact row does.
        """
        pool = self.pool_for(key.structure, key.rank_local_shape, key.family, key.rate_q256)
        rows = self._index
        if pool is None:
            row = rows.get(key)
            if row is None:
                return None
            return PricedTime(key, row.measurement.median_ms, row.measurement.samples_ms,
                              row.kernel_lane, key.label(), "row")
        sources = [rows[source] for source in (replace(key, rate_q256=rate) for rate in pool.rates_q256)
                   if source in rows]
        if not sources:
            return None
        samples = tuple(v for row in sources for v in row.measurement.samples_ms)
        medians = {row.key.rate_q256: row.measurement.median_ms for row in sources}
        record = {"rates_q256": list(pool.rates_q256), "source_rates_q256": sorted(medians),
                  "source_medians_ms": {str(rate): value for rate, value in sorted(medians.items())},
                  "cross_rate_spread_ms": (max(medians.values()) - min(medians.values())
                                           if len(medians) > 1 else None),
                  "cross_rate_spread_measured": len(medians) > 1}
        pool_id = (f"pool|{pool.structure}|{pool.rank_local_shape}|{pool.family}|"
                   f"R{'+'.join(str(r) for r in pool.rates_q256)}|M{key.m}")
        return PricedTime(key, float(median(samples)), samples, pool.kernel_lane, pool_id,
                          "rate_pool", MappingProxyType(record))


def parse_shape_table(payload: Mapping, *, source_path: str = "") -> ShapeRuntimeTable:
    """Validate the table's own declarations. Admission is a separate step."""
    try:
        return _parse_shape_table(payload, source_path)
    except ShapeRuntimeError:
        raise
    except RuntimePriceError as exc:
        raise ShapeRuntimeError(str(exc)) from exc


def _parse_shape_table(payload: Mapping, source_path: str) -> ShapeRuntimeTable:
    body = _object(payload, ("schema", "table_id", "status", "composition", "claims", "context",
                             "rows", "rate_pools"), "shape table")
    if body["schema"] != SCHEMA:
        raise ShapeRuntimeError(f"shape table schema must be {SCHEMA}")
    if body["status"] != "proposal_data" or body["composition"] != "sequential_operator_sum":
        raise ShapeRuntimeError("shape table requires proposal_data status and "
                                "sequential_operator_sum composition")
    if body["claims"] != dict(CLAIMS):
        raise ShapeRuntimeError(f"shape table claims must be exactly {dict(CLAIMS)}")
    raw_context = _object(body["context"], ShapeRuntimeContext.FIELDS, "shape table context")
    context = ShapeRuntimeContext(**raw_context)
    if not isinstance(body["rows"], list) or not body["rows"]:
        raise ShapeRuntimeError("shape table rows must be a nonempty list")
    rows: dict[ShapeKey, ShapeRow] = {}
    for index, raw in enumerate(body["rows"]):
        where = f"shape row {index}"
        item = _object(raw, ("structure", "rank_local_shape", "family", "rate_q256", "m",
                             "kernel_lane", "measurement"), where)
        structure = _string(item["structure"], where + ".structure")
        key = ShapeKey(structure, validate_shape(structure, _string(item["rank_local_shape"], where),
                                                 where + ".rank_local_shape"),
                       _string(item["family"], where + ".family"),
                       _integer(item["rate_q256"], where + ".rate_q256", 1),
                       _integer(item["m"], where + ".m", 1))
        if key.m not in context.regimes:
            raise ShapeRuntimeError(f"{where}: M={key.m} is not a regime the context declares")
        try:
            measurement = OperatorMeasurement.from_dict(item["measurement"])
        except RuntimePriceError as exc:
            raise ShapeRuntimeError(f"{where}: {exc}") from exc
        if measurement.method not in TIMING_METHODS:
            raise ShapeRuntimeError(f"{where}: method must be one of {list(TIMING_METHODS)}")
        if key in rows:
            raise ShapeRuntimeError(f"duplicate shape row {key.label()}")
        rows[key] = ShapeRow(key, KernelLane.from_dict(item["kernel_lane"], where + ".kernel_lane"),
                             measurement)
    regimes_seen = {key.m for key in rows}
    if regimes_seen != set(context.regimes):
        raise ShapeRuntimeError(
            f"context declares regimes {list(context.regimes)} but rows carry {sorted(regimes_seen)}")
    if not isinstance(body["rate_pools"], list):
        raise ShapeRuntimeError("rate_pools must be a list (empty when nothing is pooled)")
    pools: list[RatePool] = []
    for index, raw in enumerate(body["rate_pools"]):
        where = f"rate pool {index}"
        item = _object(raw, ("structure", "rank_local_shape", "family", "kernel_lane", "rates_q256"),
                       where)
        structure = _string(item["structure"], where + ".structure")
        rates = item["rates_q256"]
        if (not isinstance(rates, list) or len(rates) < 2
                or any(type(r) is not int or r < 1 for r in rates) or rates != sorted(set(rates))):
            raise ShapeRuntimeError(f"{where}: rates_q256 must list at least two ascending distinct rates")
        pool = RatePool(structure, validate_shape(structure, _string(item["rank_local_shape"], where)),
                        _string(item["family"], where + ".family"),
                        KernelLane.from_dict(item["kernel_lane"], where + ".kernel_lane"), tuple(rates))
        for other in pools:
            if (other.structure, other.rank_local_shape, other.family) == (
                    pool.structure, pool.rank_local_shape, pool.family) and (
                    set(other.rates_q256) & set(pool.rates_q256)):
                raise ShapeRuntimeError(f"{where}: a rate may belong to one pool only")
        sources = [row for key, row in rows.items()
                   if pool.covers(key.structure, key.rank_local_shape, key.family, key.rate_q256)]
        if not sources:
            raise ShapeRuntimeError(f"{where}: no row measures any pooled rate")
        disagreeing = sorted(row.key.label() for row in sources if row.kernel_lane != pool.kernel_lane)
        if disagreeing:
            raise ShapeRuntimeError(
                f"{where}: pooling is allowed only across rows that share the pool's kernel lane "
                f"{pool.kernel_lane.as_pair()}; {disagreeing} ran another lane")
        pools.append(pool)
    return ShapeRuntimeTable(_string(body["table_id"], "table_id"), context,
                             tuple(sorted(rows.values(), key=lambda row: row.key)), tuple(pools),
                             source_path)


def load_shape_table(path: str | Path) -> ShapeRuntimeTable:
    """Parse the file and verify every row's receipt bytes against its digest."""
    table = parse_shape_table(_json(path), source_path=str(path))
    for receipt, expected in sorted({(row.measurement.receipt_path, row.measurement.receipt_sha256)
                                     for row in table.rows}):
        receipt_path = Path(receipt)
        if not receipt_path.is_absolute():
            receipt_path = Path(path).parent / receipt_path
        try:
            actual = file_sha256hex(receipt_path)
        except OSError as exc:
            raise ShapeRuntimeError(f"cannot read shape-time receipt {receipt_path}: {exc}") from exc
        if actual != expected:
            raise ShapeRuntimeError(f"shape-time receipt SHA-256 mismatch: {receipt_path}")
    return table


# --------------------------------------------------------------------------- #
# Admission
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ShapeTableScope:
    """The independently supplied identity a table must equal to be read."""

    contract_sha256: str
    tessera_commit: str
    runtime_image_digest: str
    tensor_parallel: int
    platform: str
    residency: str
    execution_mode: str


def _launch_refusal(table: EligibilityTable, scope: ShapeTableScope, *, structure: str,
                    family: str, rate: int, m: int, lane: KernelLane) -> tuple[str | None, str | None]:
    """``(cell_id, None)`` when a pinned cell names and admits this launch, else ``(None, why)``."""
    regime = regime_for_m(m)
    if regime not in table.regimes:
        return None, (f"the pinned table declares regimes {list(table.regimes)}, not {regime!r} "
                      f"(M={m})")
    context = ServingContext(scope.platform, structure, scope.residency,
                             scope.runtime_image_digest, scope.execution_mode)
    cells = [cell for cell in table.cells
             if cell.family == family and cell.regime == regime and cell.is_trellis
             and rate in cell.rungs_q256
             and cell_matches_serving_context(cell, context, serving_source_sha256=None)]
    if not cells:
        return None, (f"no pinned lane cell covers {family} R{rate} {structure} in regime "
                      f"{regime!r} on {scope.platform}/{scope.residency}/{scope.execution_mode} "
                      f"at {scope.runtime_image_digest}")
    naming = [cell for cell in cells if lane.as_pair() in cell.executes]
    if not naming:
        return None, (f"cell(s) {sorted(cell.id for cell in cells)} execute "
                      f"{sorted({pair for cell in cells for pair in cell.executes})}, not the row's "
                      f"launch {lane.as_pair()}")
    reasons = []
    for cell in naming:
        if cell.predicates:
            reasons.append(f"cell {cell.id!r} predicates on unit facts a shape row does not carry")
            continue
        # Ask the one lane decision about THIS launch only: a cell that names
        # a compact and a fused launch together is otherwise decided by
        # whichever lane it lists first.
        admits, why = cell_lane_admits(replace(cell, executes=(lane.as_pair(),)), rate, table.lanes)
        if admits:
            return cell.id, None
        reasons.append(why)
    return None, "; ".join(reasons)


def admit_shape_table(table: ShapeRuntimeTable, *, scope: ShapeTableScope,
                      eligibility: EligibilityTable) -> ShapeRuntimeTable:
    """Admit every row against the pinned contract, or refuse the whole table.

    Nothing is admitted row by row: a table that claims a launch the pinned
    contract does not back is a table whose other claims are not read either.
    """
    context = table.context
    drift = [name for name in ("contract_sha256", "tessera_commit", "runtime_image_digest",
                               "tensor_parallel", "platform", "residency", "execution_mode")
             if getattr(context, name) != getattr(scope, name)]
    if drift:
        raise ShapeRuntimeError(
            "shape table context differs from the pinned scope on "
            + ", ".join(f"{name} ({getattr(context, name)!r} != {getattr(scope, name)!r})"
                        for name in drift))
    if not eligibility.present:
        raise ShapeRuntimeError(f"no pinned lane eligibility table: {eligibility.absent_reason}")
    if eligibility.contract_sha256 != scope.contract_sha256:
        raise ShapeRuntimeError(
            f"the eligibility table read ({eligibility.contract_sha256}) is not the pinned "
            f"contract ({scope.contract_sha256})")
    refusals, cells = [], {}
    for row in table.rows:
        key = row.key
        cell_id, why = _launch_refusal(eligibility, scope, structure=key.structure,
                                       family=key.family, rate=key.rate_q256, m=key.m,
                                       lane=row.kernel_lane)
        if why:
            refusals.append(f"{key.label()} on {row.kernel_lane.as_pair()}: {why}")
        else:
            cells[key.label()] = cell_id
    pooled = 0
    for pool in table.rate_pools:
        for m in table.context.regimes:
            if table.lookup(ShapeKey(pool.structure, pool.rank_local_shape, pool.family,
                                     pool.rates_q256[0], m)) is None:
                continue
            for rate in pool.rates_q256:
                cell_id, why = _launch_refusal(eligibility, scope, structure=pool.structure,
                                               family=pool.family, rate=rate, m=m,
                                               lane=pool.kernel_lane)
                label = ShapeKey(pool.structure, pool.rank_local_shape, pool.family, rate, m).label()
                if why:
                    refusals.append(f"pooled {label} on {pool.kernel_lane.as_pair()}: {why}")
                else:
                    cells.setdefault(label, cell_id)
                    pooled += 1
    if refusals:
        raise ShapeRuntimeError(f"shape table refused ({len(refusals)} launch(es)): "
                                + " | ".join(refusals))
    admission = {"schema": ADMISSION_SCHEMA, "contract_sha256": eligibility.contract_sha256,
                 "table_sha256": identity_sha256(table.as_dict()),
                 "rows_admitted": len(table.rows), "pooled_rates_admitted": pooled,
                 "cell_by_key": dict(sorted(cells.items())),
                 "per_unit_weight_identity": "not_claimed"}
    return replace(table, admission=MappingProxyType(admission))


# --------------------------------------------------------------------------- #
# Unit derivation
# --------------------------------------------------------------------------- #

def _role(name: str) -> str | None:
    leaf = str(name).rsplit(".", 1)[-1]
    return leaf if leaf in MOE_MEMBER_ROLE_AXES else None


def served_operator(member_shapes: Mapping[str, Sequence[int]], *, structure: str,
                    tensor_parallel: int, where: str = "unit") -> str:
    """The rank-local shape of the ONE operator a serving unit executes.

    Raises :class:`ShapeRuntimeError` naming why no shape can be derived; the
    caller records that as a measurement gap, never as a zero.
    """
    if not member_shapes:
        raise ShapeRuntimeError(f"{where}: no members")
    if tensor_parallel > 1 and any(_role(name) is None for name in member_shapes):
        raise ShapeRuntimeError(
            f"{where}: members {sorted(n for n in member_shapes if _role(n) is None)[:4]} carry no "
            f"role spelling, so no tensor-parallel cut rule gives their rank-local shape at "
            f"TP{tensor_parallel}")
    try:
        local = rank_local_member_shapes(member_shapes, tensor_parallel=tensor_parallel, where=where)
    except RuntimePriceError as exc:
        raise ShapeRuntimeError(str(exc)) from exc
    if any(len(shape) != 2 for shape in local.values()):
        raise ShapeRuntimeError(f"{where}: every member of a served GEMM must be 2-D")
    if structure == STRUCTURE_DENSE:
        if len(local) == 1:
            (n, k), = local.values()
            return f"{n}x{k}"
        axes = {MOE_MEMBER_ROLE_AXES.get(_role(name)) for name in local}
        if axes != {0}:
            raise ShapeRuntimeError(
                f"{where}: {len(local)} members compose into one operator only as fused "
                "column-parallel siblings (gate/up), and these are not")
        widths = {shape[1] for shape in local.values()}
        if len(widths) != 1:
            raise ShapeRuntimeError(f"{where}: fused siblings disagree on input extent {sorted(widths)}")
        return f"{sum(shape[0] for shape in local.values())}x{widths.pop()}"
    if structure != STRUCTURE_ROUTED_MOE:
        raise ShapeRuntimeError(f"{where}: structure must be one of {sorted(STRUCTURES)}")
    experts: dict[str, dict[str, tuple[int, ...]]] = {}
    for name, shape in local.items():
        prefix, leaf = str(name).rsplit(".", 1)
        experts.setdefault(prefix, {})[leaf] = shape
    stacks = set()
    for prefix, roles in experts.items():
        gate = roles.get("gate_proj", roles.get("w1"))
        up = roles.get("up_proj", roles.get("w3"))
        down = roles.get("down_proj", roles.get("w2"))
        if len(roles) != 3 or None in (gate, up, down):
            raise ShapeRuntimeError(
                f"{where}: expert {prefix} carries roles {sorted(roles)}, not one gate, up and down")
        if gate != up or gate[1] != down[0] or gate[0] != down[1]:
            raise ShapeRuntimeError(f"{where}: expert {prefix} is not one [gate|up] -> down block")
        stacks.add((gate[0], gate[1]))
    if len(stacks) != 1:
        raise ShapeRuntimeError(f"{where}: experts disagree on their geometry {sorted(stacks)}")
    (i_local, hidden), = stacks
    return f"E{len(experts)}:w13={2 * i_local}x{hidden}:w2={hidden}x{i_local}"


@dataclass(frozen=True)
class ShapePricing:
    """What :func:`build_shape_runtime_resources` returns.

    ``resources`` has exactly the type ``measured_runtime_prices.build_runtime_resources``
    returns, for the PRICED options only; ``gaps`` names every option left
    out and why.
    """

    resources: Mapping[tuple[str, str], RuntimeResources]
    prefill: Mapping[tuple[str, str], PricedTime]
    decode: Mapping[tuple[str, str], PricedTime]
    gaps: tuple[Mapping, ...]
    operators: Mapping[str, str]
    regime_m: int
    table_identity: Mapping = field(default_factory=dict)

    def time_candidates(self, candidates: Mapping[str, list]) -> dict[str, list]:
        """The candidate set restricted to priced options; refuses a unit with none."""
        kept = {unit: [c for c in options if (unit, c.fmt) in self.resources]
                for unit, options in candidates.items()}
        empty = sorted(unit for unit, options in kept.items() if not options)
        if empty:
            raise ShapeRuntimeError(
                f"{len(empty)} unit(s) have no option the shape table prices at M={self.regime_m}, "
                f"so no time-aware assignment exists: {empty[:8]}; gaps: "
                + "; ".join(f"{g['unit']}@{g['format']}: {g['reason']}"
                            for g in self.gaps if g["unit"] in set(empty[:8]))[:2000])
        return kept

    def gap_report(self) -> dict:
        by_reason: dict[str, int] = {}
        for gap in self.gaps:
            by_reason[gap["kind"]] = by_reason.get(gap["kind"], 0) + 1
        return {"regime_m": self.regime_m, "priced_options": len(self.resources),
                "unpriced_options": len(self.gaps), "by_kind": dict(sorted(by_reason.items())),
                "gaps": [dict(gap) for gap in self.gaps],
                "reading": ("an unpriced option is absent from the time-aware candidate set and is "
                            "a measurement gap, never a zero-time option")}

    def operator_sum_bootstrap(self, assignment: Mapping[str, str], *, draws: int, seed: int,
                               axis: str = "prefill", offset_ms: float = 0.0) -> dict:
        """The sum's dispersion, drawing each distinct measurement ONCE per draw.

        Units that read the same row (all 42 routed groups of one shape and
        rate) share one draw weighted by how many read it, so the interval
        carries the full correlation a shared measurement has.
        """
        priced = self.prefill if axis == "prefill" else self.decode
        counts: dict[str, int] = {}
        samples: dict[str, tuple[float, ...]] = {}
        for unit, fmt in sorted(assignment.items()):
            time = priced.get((unit, fmt))
            if time is None:
                raise ShapeRuntimeError(f"{unit}@{fmt} has no {axis} time; it cannot be in a priced sum")
            counts[time.source_id] = counts.get(time.source_id, 0) + 1
            samples[time.source_id] = time.samples_ms
        ids = sorted(counts)
        result = bootstrap_sum([samples[i] for i in ids], draws=draws, seed=seed, offset_ms=offset_ms,
                               multiplicities=[counts[i] for i in ids])
        result["distinct_measurements"] = len(ids)
        return result


def build_shape_runtime_resources(table: ShapeRuntimeTable, candidates: Mapping[str, list], *,
                                  option_members: Mapping[tuple[str, str], Mapping[str, str]],
                                  member_shapes: Mapping[str, Sequence[int]],
                                  member_structure: Mapping[str, str], regime_m: int,
                                  published_formats: Mapping[str, Mapping[str, Any]]) -> ShapePricing:
    """Price every candidate option from shape rows; leave the unpriced out.

    ``option_members`` maps each DP option to the member ``{name: format}`` it
    expands to (the allocator's ``measured_option_assignments``);
    ``member_shapes`` is each member's full source geometry from probe stats
    and ``member_structure`` its ``unit_structure``. ``published_formats`` is
    the pinned contract's ``formats[]`` table: a format name becomes
    ``(family, rate)`` only through ``lane_eligibility.resolve_payload_rung``.

    Byte fields: ``serialized_bytes`` is the candidate's own exact payload. The
    table prices no device residency, scratch or activation, so those are 0 and
    no device budget may be evaluated against them (the allocator refuses one).
    """
    if not table.admitted:
        raise ShapeRuntimeError("shape table rows are priced only after admit_shape_table")
    if regime_m not in table.context.regimes:
        raise ShapeRuntimeError(
            f"--pact-regime {regime_m} is not a regime the table measured {list(table.context.regimes)}")
    tp = table.context.tensor_parallel
    resources, prefill, decode, gaps, operators = {}, {}, {}, [], {}
    operator_refusals: dict[str, str] = {}

    def gap(unit, fmt, kind, reason):
        gaps.append(MappingProxyType({"unit": unit, "format": fmt, "kind": kind, "reason": reason}))

    for unit, options in sorted(candidates.items()):
        for candidate in sorted(options, key=lambda c: c.fmt):
            key = unit, candidate.fmt
            members = option_members.get(key)
            if not members:
                raise ShapeRuntimeError(f"{key}: no member expansion supplied")
            names = tuple(sorted(members))
            structures = {member_structure.get(name) for name in names}
            if len(structures) != 1 or None in structures:
                gap(unit, candidate.fmt, "structure",
                    f"members name structures {sorted(map(str, structures))}, not one")
                continue
            structure = structures.pop()
            if unit not in operators and unit not in operator_refusals:
                try:
                    operators[unit] = served_operator(
                        {name: tuple(member_shapes[name]) for name in names},
                        structure=structure, tensor_parallel=tp, where=unit)
                except (ShapeRuntimeError, KeyError) as exc:
                    operator_refusals[unit] = str(exc)
            if unit in operator_refusals:
                gap(unit, candidate.fmt, "operator_shape", operator_refusals[unit])
                continue
            rungs = {resolve_payload_rung(fmt, published_formats) for fmt in members.values()}
            families = {family for family, _k, _rate in rungs}
            rates = {rate for _family, _k, rate in rungs}
            if len(families) != 1 or len(rates) != 1 or None in rates:
                kind = "mixed_rate_operator" if len(families) == 1 and None not in rates else "not_rate_addressed"
                gap(unit, candidate.fmt, kind,
                    f"the served operator carries families {sorted(families)} at rates "
                    f"{sorted(map(str, rates))}; a shape row keys one rate-addressed family and rate")
                continue
            family, rate = families.pop(), rates.pop()
            if type(candidate.memory_bytes) is not int:
                raise ShapeRuntimeError(f"{key}: candidate bytes must be an exact integer")
            found = table.lookup(ShapeKey(structure, operators[unit], family, rate, regime_m))
            if found is None:
                gap(unit, candidate.fmt, "no_time_row",
                    f"no row or rate pool times {structure} {operators[unit]} {family} R{rate} "
                    f"at M={regime_m}")
                continue
            one = (table.lookup(ShapeKey(structure, operators[unit], family, rate, DECODE_M))
                   if DECODE_M in table.context.regimes else None)
            prefill[key] = found
            if one is not None:
                decode[key] = one
            resources[key] = RuntimeResources(
                prefill_ms=found.median_ms, decode_ms=None if one is None else one.median_ms,
                serialized_bytes=candidate.memory_bytes, resident_bytes=0, peak_scratch_bytes=0,
                activation_bytes=0)
    return ShapePricing(MappingProxyType(resources), MappingProxyType(prefill),
                        MappingProxyType(decode), tuple(gaps), MappingProxyType(operators),
                        regime_m, MappingProxyType(table.identity()))


# --------------------------------------------------------------------------- #
# Tessera receipt consumer (stub until RobTand/tessera#688 publishes a schema)
# --------------------------------------------------------------------------- #

SHAPE_TIME_PANEL_SCHEMA = "tessera.shape_time_panel.v1"


def consume_shape_time_panel(receipts: Sequence[Path], *, table_id: str) -> dict:
    """STUB: convert ``tessera.shape_time_panel.v1`` receipts into a v1 table.

    Tessera has not published this receipt's schema (RobTand/tessera#688 is
    open, PQ #1583 follow-up). Guessing its fields would make this module the
    second author of Tessera's receipt, so it refuses instead. When #688 lands
    this becomes the converter, on ``native_receipt_table``'s pattern.
    """
    raise ShapeRuntimeError(
        f"{SHAPE_TIME_PANEL_SCHEMA} has no published schema yet (RobTand/tessera#688); "
        "the receipt->table converter is a stub until it does (PQ #1583)")


def main(argv: Sequence[str] | None = None) -> int:
    """``convert`` exits 2 until #688; ``check`` parses a table and exits 0 or 2."""
    parser = argparse.ArgumentParser(prog="python -m prismaquant.shape_runtime_prices")
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert", help="receipts -> table (stub until tessera#688)")
    convert.add_argument("--out", type=Path, required=True)
    convert.add_argument("--table-id", required=True)
    convert.add_argument("--receipts", type=Path, nargs="+", required=True)
    check = sub.add_parser("check", help="parse a table and verify its receipt digests")
    check.add_argument("table", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "convert":
            consume_shape_time_panel(args.receipts, table_id=args.table_id)
            return 0
        table = load_shape_table(args.table)
        print(json.dumps(table.identity()), flush=True)
        return 0
    except RuntimePriceError as exc:
        print(f"[shape-runtime-prices] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
