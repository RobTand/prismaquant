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
rate moves the time on one lane and can decide the lane. Under contracts v42
to v44 (Tessera #685) a routed E4M3 stack at R1024 took the fused lane
(17.5 ms at M=2048, TP2 rank-local, the after-#640 bench) and the same stack
at R896 took the compact adapter (109 ms). Since v45 (tessera#694) R896 takes
the fused lane too, at 1.65x to 1.80x of R1024's fused time on Tessera's
PACT bench (TP1, GLM-5.3-Flash layer 3, tessera#701), and a routed plan
outside the lane's ``column_rates_routed_moe`` still takes the compact
adapter.

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
``native_routed_fused_window`` at R1024 admits. One at R896 admits since
contract v45 and was refused before it, because the v44 lane's ``requires``
did not read that wire.

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

Tessera's checker emits ``tessera.shape_time_observation.v1``. Conversion
requires its explicitly selected PB completion and the independently reviewed
checker configuration; a panel or observation alone is insufficient.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass, field, replace
from statistics import median
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .digests import DIRECT_ASCII_SPACED_LAX, DIRECT_ASCII_STRICT, bytes_sha256hex, file_sha256hex
from .lane_eligibility import (
    STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE, STRUCTURES, EligibilityTable,
    LaneEligibilityError, ServingContext, cell_lane_admits,
    cell_matches_serving_context, resolve_payload_rung,
)
from .measured_runtime_prices import (
    MOE_MEMBER_ROLE_AXES, OperatorMeasurement, RuntimePriceError, RuntimeResources,
    _integer, _json, _object, _string, bootstrap_sum, rank_local_member_shapes,
)
from .runtime_provenance import ArtifactReader, _strict_json as _parse_bound_json
from .schemas import strict_json_loads

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
    """Parse the file and re-bind every row to the authenticated raw samples.

    A checker receipt is rejoined through the public PB reader and the
    independent reviewed source config, then its projection is compared to
    the table. Bare panels and observations refuse. Other legacy artifacts
    retain the digest-only check.
    """
    table = parse_shape_table(_json(path), source_path=str(path))
    for receipt, expected in sorted({(row.measurement.receipt_path, row.measurement.receipt_sha256)
                                     for row in table.rows}):
        receipt_path = Path(receipt)
        if not receipt_path.is_absolute():
            receipt_path = Path(path).parent / receipt_path
        receipt_path = receipt_path.resolve()
        try:
            _, raw = ArtifactReader(Path()).bytes(
                {"path": str(receipt_path), "sha256": expected}, "shape-time receipt")
        except (OSError, RuntimePriceError) as exc:
            raise ShapeRuntimeError(f"cannot read shape-time receipt {receipt_path}: {exc}") from exc
        _rebind_shape_row_to_receipt(table, receipt_path, raw)
    return table


def _rebind_shape_row_to_receipt(table: ShapeRuntimeTable, receipt_path: Path, raw: bytes) -> None:
    """Compare each row reading this receipt against its authenticated samples."""
    try:
        panel = _parse_observation_json(raw)
    except (ValueError, UnicodeError) as exc:
        raise ShapeRuntimeError(f"shape-time receipt {receipt_path}: invalid JSON: {exc}") from exc
    if isinstance(panel, Mapping) and panel.get("schema") == CHECKER_RECEIPT_SCHEMA:
        projection = _verify_checker_receipt(panel)
        if DIRECT_ASCII_STRICT.text(table.context.as_dict()) != DIRECT_ASCII_STRICT.text(
                projection["context"].as_dict()):
            raise ShapeRuntimeError("shape table context differs from its checker observation")
        expected_key = ShapeKey(projection["structure"], projection["rank_local_shape"],
                                projection["family"], projection["rate_q256"], projection["m"])
        for row in table.rows:
            row_receipt = Path(row.measurement.receipt_path)
            if not row_receipt.is_absolute():
                row_receipt = Path(table.source_path).parent / row_receipt
            if row_receipt.resolve() != receipt_path:
                continue
            if row.key != expected_key or row.kernel_lane != projection["lane"]:
                raise ShapeRuntimeError("shape row key/lane differs from its checker observation")
            if row.measurement.samples_ms != projection["samples_ms"]:
                raise ShapeRuntimeError("shape row samples differ from the receipt's raw samples")
            if (row.measurement.method != "cuda_events"
                    or row.measurement.warmup_iterations != projection["warmup_iterations"]):
                raise ShapeRuntimeError("shape row measurement differs from its checker observation")
        return
    if isinstance(panel, Mapping) and panel.get("schema") in (
            SHAPE_TIME_PANEL_SCHEMA, SHAPE_TIME_OBSERVATION_SCHEMA):
        raise ShapeRuntimeError("shape-time receipt requires an authenticated PB checker completion")


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

    def kernel_lane_histogram(self, assignment: Mapping[str, str], *, axis: str = "prefill") -> dict:
        """How many units of ``assignment`` ride each priced kernel lane.

        Reads the lane each option's own measurement carries (``symbol/decoder``),
        so the histogram is the priced route, not a derived guess (PQ #1585).
        """
        priced = self.prefill if axis == "prefill" else self.decode
        counts: dict[str, int] = {}
        for unit, fmt in sorted(assignment.items()):
            time = priced.get((unit, fmt))
            if time is None:
                raise ShapeRuntimeError(f"{unit}@{fmt} has no {axis} time; it has no priced kernel lane")
            label = f"{time.kernel_lane.symbol}/{time.kernel_lane.decoder}"
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))


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
# Tessera observation consumer (tessera.shape_time_observation.v1)
# --------------------------------------------------------------------------- #
#
# The producer publishes ONE versioned, input-bound handoff per validated
# panel (``tessera.shape_time_observation.v1``, RobTand/tessera#856, from the
# existing ``tools/tessera_shape_time_panel.py check`` path). PQ reads that
# observation and its bound panel bytes: it does not import or vendor
# ``tessera.serving``, does not deserialize the producer's private validation
# token, and does not restate a second semantic validator. Everything the
# producer proved is joined to a selected PB completion and a reviewed
# checker source/command/environment. Hash and length detect subsequent
# artifact changes. Everything
# that is a PQ admission decision (pinned contract, lane, context) is made by
# PQ's own ``admit_shape_table`` against the pinned ``EligibilityTable``.

SHAPE_TIME_PANEL_SCHEMA = "tessera.shape_time_panel.v1"
SHAPE_TIME_OBSERVATION_SCHEMA = "tessera.shape_time_observation.v1"
CHECKER_RECEIPT_SCHEMA = "prismaquant.shape_time_checker_receipt.v1"
CHECKER_CONFIG_SCHEMA = "prismaquant.shape_time_checker_config.v1"
CHECKER_CONFIG_PATH = Path(__file__).parent / "tessera_runtime" / "shape_time_checker_config.json"
CHECKER_RESULT_MAX_BYTES = 2 * 1024 * 1024
CHECKER_EVIDENCE_MAX_BYTES = 4 * 1024 * 1024
_OBSERVATION_CLAIMS = {"time_claim": TIME_CLAIM, "certifies_placement": False,
                       "served_p95": "not_claimed"}
_FLAT_SHA = re.compile(r"[0-9a-f]{64}")
_FLAT_COMMIT = re.compile(r"[0-9a-f]{40}")
_OBSERVATION_TOP = ("schema", "status", "claims", "gpu_executed", "panel",
                    "expected_panel_sha256", "request", "expected_runtime", "contract",
                    "evidence", "preflight", "producer", "replay", "invocation", "scope",
                    "scope_id", "cell_id", "kernel_lane", "structure", "rank_local_shape",
                    "family", "payload", "timing", "sampling", "operator_projection",
                    "energy_status")
_OBSERVATION_RUNTIME = ("image", "tessera_commit", "serving_source_sha256", "contract_sha256",
                        "platform", "torch", "vllm", "serve_flags", "execution_mode",
                        "residency", "tp_rank", "tp_degree", "package_root")
_OBSERVATION_EVIDENCE = ("runtime", "producer", "contract", "wire", "preparation", "samples",
                         "routes", "trace", "telemetry", "native_binary", "runtime_origins")
_OBSERVATION_SCOPE = ("route", "grid", "q256", "structure", "mode", "execution_mode",
                      "regime", "tp_degree", "requested_platform", "shape")
_OBSERVATION_SAMPLING = ("method", "sample_unit", "warmup_iterations", "n", "samples_ms",
                         "interval_unix")
_OBSERVATION_PROJECTION = ("batch_size", "rows", "reading")
_OBSERVATION_REPLAY = ("source_tree_sha256", "source_tree_members", "tool_source_sha256", "tool")
_OBSERVATION_BINDING = ("path", "bytes", "sha256")


def _observation_sha(value, where) -> str:
    if not isinstance(value, str) or not _FLAT_SHA.fullmatch(value):
        raise ShapeRuntimeError(f"{where}: requires a lowercase SHA-256")
    return value


def _observation_number(value, where, *, positive=True) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ShapeRuntimeError(
            f"{where}: requires a finite {'positive ' if positive else ''}number")
    return float(value)


def _observation_sha_in(value, candidates, where) -> None:
    if value not in candidates:
        raise ShapeRuntimeError(f"{where}: {value!r} names no bound artifact")


def _observation_lane(value, where) -> KernelLane:
    """The observation records the launch as the ``[symbol, decoder]`` pair."""
    if not isinstance(value, list) or len(value) != 2:
        raise ShapeRuntimeError(f"{where}: requires a [symbol, decoder] pair")
    symbol = _string(value[0], where + ".symbol")
    decoder = _string(value[1], where + ".decoder")
    return KernelLane(symbol, decoder)


def _verify_checker_receipt(receipt: Mapping) -> dict:
    """Join an explicit PB completion to independently reviewed checker bytes."""
    top = _object(receipt, ("schema", "observation", "selector"), "checker receipt")
    if top["schema"] != CHECKER_RECEIPT_SCHEMA:
        raise ShapeRuntimeError("observation requires a versioned PB checker receipt")
    selector = _object(top["selector"], ("action_key", "published_unix", "attempt"),
                       "checker selector")
    _observation_sha(selector["action_key"], "checker action key")
    published = _observation_number(selector["published_unix"], "checker publication time")
    attempt = _integer(selector["attempt"], "checker attempt", 1)
    reader = ArtifactReader(Path())
    try:
        observation_path, raw = reader.bytes(top["observation"], "checker observation",
                                             max_bytes=CHECKER_RESULT_MAX_BYTES)
        observation = _parse_observation_json(raw)
    except (RuntimePriceError, ValueError, UnicodeError) as exc:
        raise ShapeRuntimeError(f"checker observation refused: {exc}") from exc
    try:
        from .staged_lease import client_sdk
        sdk = client_sdk()
        result = sdk.read_verified_action_result(
            sdk.PoolQueue(), selector["action_key"], published_unix=published,
            attempt=attempt, max_result_bytes=CHECKER_RESULT_MAX_BYTES,
            max_evidence_bytes=CHECKER_EVIDENCE_MAX_BYTES)
        sdk.bind_standard_capture_command(result["request"])
    except Exception as exc:
        raise ShapeRuntimeError(f"PB checker completion refused: {exc}") from exc
    if (result.get("action_key") != selector["action_key"]
            or result.get("published_unix") != published or result.get("attempt") != attempt):
        raise ShapeRuntimeError("PB checker result differs from the explicit selector")
    request = result["request"]
    params = request.get("params", {})
    snapshot = params.get("checkout_snapshot", {})
    source = snapshot.get("input")
    if not isinstance(source, Mapping) or source not in result.get("inputs", []):
        raise ShapeRuntimeError("PB checker carries no sealed source snapshot input")
    try:
        config = _object(_json(CHECKER_CONFIG_PATH), ("schema", "checkers"), "checker config")
    except (OSError, ValueError) as exc:
        raise ShapeRuntimeError(f"reviewed checker configuration unavailable: {exc}") from exc
    if config["schema"] != CHECKER_CONFIG_SCHEMA or not isinstance(config["checkers"], list):
        raise ShapeRuntimeError("reviewed checker configuration has an unsupported schema")
    matches = []
    for entry in config["checkers"]:
        pin = _object(entry, ("snapshot", "cwd", "working_directory", "command", "environment",
                              "observation_output"),
                      "reviewed checker")
        if (snapshot == pin["snapshot"] and params.get("command") == pin["command"]
                and params.get("cwd") == pin["cwd"]
                and request["task"].get("working_directory") == pin["working_directory"]
                and request.get("environment") == pin["environment"]
                and str(observation_path) == pin["observation_output"]):
            matches.append(pin)
    if len(matches) != 1:
        raise ShapeRuntimeError("PB checker source/command/environment is not independently reviewed")
    payload = result.get("payload")
    if not isinstance(payload, bytes) or len(payload) > CHECKER_RESULT_MAX_BYTES:
        raise ShapeRuntimeError("PB checker carries no bounded owned result bytes")
    emitted = []
    for line in payload.splitlines():
        try:
            value = _parse_observation_json(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(value, Mapping) and value.get("schema") == SHAPE_TIME_OBSERVATION_SCHEMA:
            emitted.append(value)
    if len(emitted) != 1 or DIRECT_ASCII_STRICT.encoded(emitted[0]) + b"\n" != raw:
        raise ShapeRuntimeError("observation bytes differ from the PB checker's owned output")
    projection = _observation_projection(observation)
    projection["observation_bytes"] = raw
    return projection


def _observation_context(payload: Mapping, where: str, *, scope: Mapping,
                         scope_where: str) -> ShapeRuntimeContext:
    runtime = _object(payload, _OBSERVATION_RUNTIME, where)
    serve_flags = runtime["serve_flags"]
    if (not isinstance(serve_flags, dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in serve_flags.items())):
        raise ShapeRuntimeError(f"{where}.serve_flags: requires observed string values")
    if serve_flags.get("TESSERA_SERVE_MODE") != runtime["residency"]:
        raise ShapeRuntimeError(f"{where}: residency disagrees with the observed serve mode")
    if runtime["execution_mode"] != "eager" or runtime["residency"] != "resident":
        raise ShapeRuntimeError(f"{where}: the observation slice is eager/resident only")
    if type(runtime["tp_degree"]) is not int or type(runtime["tp_rank"]) is not int \
            or runtime["tp_rank"] != 0:
        raise ShapeRuntimeError(f"{where}: the observation names an explicit TP rank 0 world")
    tp_degree = _integer(runtime["tp_degree"], "observation.tp_degree", 1)
    return ShapeRuntimeContext(
        runtime_image_digest=runtime["image"],
        tessera_commit=runtime["tessera_commit"],
        contract_sha256=runtime["contract_sha256"],
        tensor_parallel=tp_degree,
        platform=runtime["platform"],
        execution_mode=runtime["execution_mode"],
        residency=runtime["residency"],
        batch_size=1,
        regimes=_observation_regimes(scope, scope_where),
    )


def _observation_regimes(scope_payload: Mapping, where: str) -> tuple[int, ...]:
    """The M the observation's own scope names, as the one measured regime.

    The producer times one operator apply at a single M; PQ declares exactly
    that M and does not add a decode row for M=1. A caller that later asks
    :meth:`ShapeRuntimeTable.lookup` for an unmeasured regime gets ``None`` --
    a gap, never a fabricated number.
    """
    scope = _object(scope_payload, _OBSERVATION_SCOPE, where)
    shape = _object(scope["shape"], ("M", "N", "K"), where + ".shape")
    return (_integer(shape["M"], where + " M", 1),)

def read_shape_time_observation(path: str | Path) -> tuple[Mapping, tuple[Path, bytes]]:
    """Return bounded strict JSON and its owned ``(path, raw)`` bytes.

    Authentication comes from the selected checker receipt, independently of
    this caller-supplied file's own digest.
    """
    root = Path(path).resolve()
    reader = ArtifactReader(root.parent)
    try:
        _, raw = reader.bytes({"path": str(root), "sha256": file_sha256hex(root)}, "observation",
                              max_bytes=CHECKER_RESULT_MAX_BYTES)
    except (OSError, RuntimePriceError) as exc:
        raise ShapeRuntimeError(f"shape-time observation: cannot read {root}: {exc}") from exc
    try:
        value = _parse_observation_json(raw)
    except (ValueError, UnicodeError) as exc:
        raise ShapeRuntimeError(f"shape-time observation: invalid JSON {root}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ShapeRuntimeError("shape-time observation: expected a JSON object")
    return value, (root, raw)


def _parse_observation_json(raw: bytes) -> Any:
    return strict_json_loads(
        raw, duplicate=lambda key: ShapeRuntimeError(
            f"shape-time observation: duplicate JSON key {key!r}"),
        constant=lambda token: ShapeRuntimeError(
            f"shape-time observation: nonfinite JSON constant {token}"))


def _observation_binding(reader: ArtifactReader, reference, where: str) -> tuple[Path, bytes]:
    return reader.bytes(reference, where)


def _observation_projection(observation: Mapping) -> dict:
    """Check the observation against its own bound bytes; return its projection.

    This is the PQ consumer side of the handoff. It re-binds every artifact the
    authenticated checker output names and checks its internal consistency,
    then reports
    the context, key, lane, measurement and receipt the table row is built
    from. Producer semantics are supplied by the reviewed checker's bound
    execution; this function is called only after that completion is joined.
    Admission remains :func:`admit_shape_table`'s job.
    """
    top = _object(observation, _OBSERVATION_TOP, "shape-time observation")
    if top["schema"] != SHAPE_TIME_OBSERVATION_SCHEMA:
        raise ShapeRuntimeError(
            f"shape-time observation schema must be {SHAPE_TIME_OBSERVATION_SCHEMA}, "
            f"not {top['schema']!r}")
    if top["status"] != "validated":
        raise ShapeRuntimeError("shape-time observation status must be 'validated'")
    if top["gpu_executed"] is not False:
        raise ShapeRuntimeError("shape-time observation must record a CUDA-disabled replay")
    if dict(top["claims"]) != _OBSERVATION_CLAIMS:
        raise ShapeRuntimeError(f"shape-time observation claims must be exactly {_OBSERVATION_CLAIMS}")
    reader = ArtifactReader(Path())
    panel_path, panel_raw = reader.bytes(top["panel"], "observation.panel")
    _observation_sha(top["expected_panel_sha256"], "observation.expected_panel_sha256")
    panel = _parse_bound_json(panel_raw, panel_path, "observation.panel")
    _equal_strict(panel.get("schema"), SHAPE_TIME_PANEL_SCHEMA, "observation.panel schema")
    if panel.get("status") != "measured":
        raise ShapeRuntimeError("observation.panel must be a measured shape-time panel")
    request = reader.json(top["request"], "observation.request")[1]
    expected_runtime = reader.json(top["expected_runtime"], "observation.expected_runtime")[1]
    reader.bytes(top["contract"], "observation.contract")
    evidence = _object(top["evidence"], _OBSERVATION_EVIDENCE, "observation.evidence")
    bound: dict[str, tuple[Path, bytes]] = {}
    for name in evidence:
        bound[name] = reader.bytes(evidence[name], f"observation.evidence.{name}")
    preflight = _object(top["preflight"], ("result", "phase"), "observation.preflight")
    reader.bytes(preflight["result"], "observation.preflight.result")
    reader.bytes(preflight["phase"], "observation.preflight.phase")
    if evidence["contract"] != panel.get("evidence", {}).get("contract"):
        raise ShapeRuntimeError("observation contract binding differs from its panel")
    if evidence != panel.get("evidence"):
        raise ShapeRuntimeError("observation evidence bindings differ from its panel")
    if preflight != panel.get("preflight"):
        raise ShapeRuntimeError("observation preflight bindings differ from its panel")
    if panel.get("runtime") != expected_runtime:
        raise ShapeRuntimeError("observation expected runtime differs from the panel runtime")
    scope_doc = _object(top["scope"], _OBSERVATION_SCOPE, "observation.scope")
    context = _observation_context(expected_runtime, "observation.expected_runtime",
                                   scope=scope_doc, scope_where="observation.scope")
    observed = _observation_context(panel["runtime"], "observation.panel.runtime",
                                    scope=scope_doc, scope_where="observation.scope")
    if context != observed:
        raise ShapeRuntimeError("observation runtime context differs from the panel runtime")
    claims = panel.get("claims", top["claims"])
    if dict(claims) != _OBSERVATION_CLAIMS:
        raise ShapeRuntimeError("observation panel claims differ from the observation claims")
    if top["energy_status"] != "hold":
        raise ShapeRuntimeError("observation energy must remain HOLD in this slice")
    if panel.get("energy", {}).get("status") != "hold":
        raise ShapeRuntimeError("observation panel energy must remain HOLD in this slice")
    plan = panel.get("plan")
    if not isinstance(plan, Mapping) or plan.get("gpu_executed") is not False:
        raise ShapeRuntimeError("observation panel requires an unmeasured CPU census plan")
    plan_rows = plan.get("rows")
    if not isinstance(plan_rows, list) or len(plan_rows) != 1:
        raise ShapeRuntimeError("the observation slice carries exactly one plan row")
    plan_scope = _object(plan_rows[0].get("scope"), _OBSERVATION_SCOPE, "observation.panel.plan.scope")
    if plan_scope != top["scope"]:
        raise ShapeRuntimeError("observation scope differs from its panel plan")
    if top["scope_id"] != plan_rows[0].get("id"):
        raise ShapeRuntimeError("observation scope_id differs from its panel plan")
    shape = _object(top["scope"]["shape"], ("M", "N", "K"), "observation.scope.shape")
    m = _integer(shape["M"], "observation scope M", 1)
    n = _integer(shape["N"], "observation scope N", 1)
    k = _integer(shape["K"], "observation scope K", 1)
    structure = _string(top["structure"], "observation.structure")
    rank_local = validate_shape(structure, _string(top["rank_local_shape"], "observation.rank_local_shape"),
                                "observation.rank_local_shape")
    if structure == STRUCTURE_DENSE and rank_local != f"{n}x{k}":
        raise ShapeRuntimeError(
            f"observation rank_local_shape {rank_local!r} differs from its NxK {n}x{k}")
    if top["scope"]["structure"] != structure or top["scope"]["regime"] != regime_for_m(m):
        raise ShapeRuntimeError("observation structure/regime disagrees with its scope")
    if top["scope"]["tp_degree"] != context.tensor_parallel or top["scope"]["tp_degree"] != 1:
        raise ShapeRuntimeError("observation scope is not the TP1 world it declares")
    if top["scope"]["requested_platform"] != context.platform:
        raise ShapeRuntimeError("observation platform disagrees with its scope")
    family = _string(top["family"], "observation.family")
    payload = _object(top["payload"], ("route", "grid", "q256", "rows", "columns"), "observation.payload")
    if payload["route"] != top["scope"]["route"] or payload["grid"] != top["scope"]["grid"]:
        raise ShapeRuntimeError("observation payload route/grid differs from its scope")
    if payload["rows"] != n or payload["columns"] != k or payload["q256"] != top["scope"]["q256"]:
        raise ShapeRuntimeError("observation payload geometry differs from its scope")
    rate = _integer(top["scope"]["q256"], "observation scope q256", 1)
    lane = _observation_lane(top["kernel_lane"], "observation.kernel_lane")
    samples = _object(top["sampling"], _OBSERVATION_SAMPLING, "observation.sampling")
    if samples["method"] != "cuda_events" or samples["sample_unit"] != "single_apply":
        raise ShapeRuntimeError("observation must time one CUDA-event single apply")
    values = samples["samples_ms"]
    if not isinstance(values, list) or len(values) < 3:
        raise ShapeRuntimeError("observation requires at least three repeated CUDA-event samples")
    finite = tuple(_observation_number(v, "observation.samples_ms") for v in values)
    if samples["n"] != len(finite):
        raise ShapeRuntimeError("observation sampling count differs from its raw samples")
    warmup = _integer(samples["warmup_iterations"], "observation.warmup_iterations")
    interval = samples["interval_unix"]
    if (not isinstance(interval, list) or len(interval) != 2
            or _observation_number(interval[1], "observation.interval end")
            <= _observation_number(interval[0], "observation.interval start", positive=False)):
        raise ShapeRuntimeError("observation sampling interval is invalid")
    summary = _object(top["timing"], ("method", "n", "median_ms", "p25_ms", "p75_ms", "iqr_ms",
                                      "quartiles"), "observation.timing")
    if summary["method"] != "cuda_events" or _integer(summary["n"], "observation.timing.n") != len(finite):
        raise ShapeRuntimeError("observation timing summary count/method differs from its samples")
    derived = _timing_summary(finite)
    for name in ("median_ms", "p25_ms", "p75_ms", "iqr_ms", "quartiles"):
        if DIRECT_ASCII_STRICT.text(summary[name]) != DIRECT_ASCII_STRICT.text(derived[name]):
            raise ShapeRuntimeError(f"observation timing {name} differs from its raw samples")
    samples_path, samples_raw = bound["samples"]
    raw_samples = _parse_bound_json(samples_raw, samples_path, "observation.evidence.samples")
    if raw_samples.get("samples_ms") != list(samples["samples_ms"]):
        raise ShapeRuntimeError("observation samples differ from its bound raw sample bytes")
    if raw_samples.get("warmup_iterations") != warmup:
        raise ShapeRuntimeError("observation warmup count differs from its bound raw sample bytes")
    if raw_samples.get("interval_unix") != list(interval):
        raise ShapeRuntimeError("observation sample interval differs from its bound raw sample bytes")
    routes_path, routes_raw = bound["routes"]
    routes = _parse_bound_json(routes_raw, routes_path, "observation.evidence.routes")
    records = routes.get("records")
    if not isinstance(records, list) or len(records) != len(finite):
        raise ShapeRuntimeError("observation requires one fresh route record per timed sample")
    pairs = {(record.get("symbol"), record.get("decoder")) for record in records
             if isinstance(record, Mapping)}
    if pairs != {(lane.symbol, lane.decoder)}:
        raise ShapeRuntimeError("observation lane differs from its raw route records")
    projection = _object(top["operator_projection"], _OBSERVATION_PROJECTION,
                         "observation.operator_projection")
    if projection["batch_size"] != 1 or projection["rows"] != m:
        raise ShapeRuntimeError(
            "observation operator projection must read one M-row operator at batch_size=1")
    producer_binding = _object(top["producer"], _OBSERVATION_BINDING, "observation.producer")
    _observation_sha(producer_binding["sha256"], "observation.producer.sha256")
    producer = reader.json(producer_binding, "observation.producer")[1]
    if request.get("producer_identity") != producer_binding:
        raise ShapeRuntimeError("observation producer differs from its bound request")
    replay = _object(top["replay"], _OBSERVATION_REPLAY, "observation.replay")
    _observation_sha(replay["source_tree_sha256"], "observation.replay.source_tree_sha256")
    _observation_sha(replay["tool_source_sha256"], "observation.replay.tool_source_sha256")
    _integer(replay["source_tree_members"], "observation.replay.source_tree_members", 1)
    _object(replay["tool"], _OBSERVATION_BINDING, "observation.replay.tool")
    invocation = _object(top["invocation"], ("command", "phase", "returncode"), "observation.invocation")
    if invocation["phase"] != "runtime-preflight" or invocation["returncode"] != 0:
        raise ShapeRuntimeError("observation must be issued after a successful CPU preflight")
    command = invocation["command"]
    if not isinstance(command, list) or any(not isinstance(v, str) for v in command):
        raise ShapeRuntimeError("observation invocation command must be an explicit argv")
    for required in ("CUDA_VISIBLE_DEVICES=", "--preflight", "--job-sha256"):
        if required not in command:
            raise ShapeRuntimeError(f"observation invocation lost the owned CPU marker {required!r}")
    return {"panel_path": panel_path, "panel": panel, "request": request,
            "expected_runtime": expected_runtime, "context": context, "m": m, "structure": structure,
            "rank_local_shape": rank_local, "family": family, "rate_q256": rate, "lane": lane,
            "samples_ms": finite, "warmup_iterations": warmup, "summary": summary,
            "cell_id": _string(top["cell_id"], "observation.cell_id"),
            "producer": producer, "replay": replay, "invocation": invocation,
            "panel_binding": top["panel"], "request_binding": top["request"],
            "contract_binding": top["contract"], "evidence": dict(evidence), "preflight": dict(preflight)}


def _equal_strict(actual, expected, where: str) -> None:
    if actual != expected:
        raise ShapeRuntimeError(f"{where}: {actual!r} != {expected!r}")


def _timing_summary(samples: Sequence[float]) -> dict:
    values = [float(v) for v in samples]
    q1, _mid, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {"method": "cuda_events", "n": len(values), "median_ms": float(median(values)),
            "p25_ms": float(q1), "p75_ms": float(q3), "iqr_ms": float(q3 - q1),
            "quartiles": "statistics.quantiles.inclusive"}


def consume_shape_time_observation(observations: Sequence[Path], *, table_id: str,
                                   checker_receipts: Sequence[Path] | None = None,
                                   expected_scope: "ShapeTableScope | None" = None,
                                   eligibility: EligibilityTable | None = None) -> ShapeRuntimeTable:
    """Convert validated observations into an admitted proposal table.

    Every observation is bound to its exact bytes through
    :func:`read_shape_time_observation`; then its projection is turned into the
    existing :class:`ShapeRuntimeContext`/:class:`ShapeKey`/
    :class:`KernelLane`/:class:`OperatorMeasurement`; then the table is parsed
    and, when an expected scope and eligibility table are supplied, admitted by
    the unchanged :func:`admit_shape_table`.

    A malformed, incomplete or unsupported observation refuses the WHOLE
    conversion: nothing is skipped and no output is written. One observation
    with one row produces one table row; no menu completeness is claimed.
    """
    if not observations:
        raise ShapeRuntimeError("shape-time conversion requires at least one observation")
    if checker_receipts is None or len(checker_receipts) != len(observations):
        raise ShapeRuntimeError("conversion requires one explicit PB checker receipt per observation")
    projections: list[dict] = []
    for index, path in enumerate(observations):
        _observation, (_path, raw) = read_shape_time_observation(path)
        proof, (_receipt_path, proof_raw) = read_shape_time_observation(checker_receipts[index])
        projection = _verify_checker_receipt(proof)
        if raw != projection["observation_bytes"]:
            raise ShapeRuntimeError("checker receipt names different observation bytes")
        projection["receipt_path"] = str(Path(checker_receipts[index]).resolve())
        projection["receipt_sha256"] = bytes_sha256hex(proof_raw)
        if any(DIRECT_ASCII_STRICT.text(projection["context"].as_dict())
               == DIRECT_ASCII_STRICT.text(previous["context"].as_dict())
               and projection["rank_local_shape"] == previous["rank_local_shape"]
               and projection["family"] == previous["family"]
               and projection["rate_q256"] == previous["rate_q256"]
               and projection["m"] == previous["m"]
               for previous in projections):
            raise ShapeRuntimeError(
                f"observation {index}: duplicate shape key {projection['rank_local_shape']} "
                f"{projection['family']} R{projection['rate_q256']} M{projection['m']}")
        projections.append(projection)
    contexts = {DIRECT_ASCII_STRICT.text(projection["context"].as_dict()) for projection in projections}
    if len(contexts) != 1:
        raise ShapeRuntimeError("every observation in one conversion must share one runtime context")
    context = projections[0]["context"]
    rows = []
    for projection in projections:
        measurement = OperatorMeasurement(
            method="cuda_events", samples_ms=projection["samples_ms"],
            warmup_iterations=projection["warmup_iterations"],
            receipt_path=projection["receipt_path"],
            receipt_sha256=projection["receipt_sha256"])
        key = ShapeKey(projection["structure"], projection["rank_local_shape"],
                       projection["family"], projection["rate_q256"], projection["m"])
        rows.append(ShapeRow(key, projection["lane"], measurement))
    document = {"schema": SCHEMA, "table_id": table_id, "status": "proposal_data",
                "composition": "sequential_operator_sum", "claims": dict(CLAIMS),
                "context": context.as_dict(),
                "rows": [row.as_dict() for row in sorted(rows, key=lambda r: r.key)],
                "rate_pools": []}
    table = parse_shape_table(document)
    if expected_scope is not None:
        if eligibility is None:
            raise ShapeRuntimeError(
                "admission requires PQ's pinned EligibilityTable; none was supplied")
        table = admit_shape_table(table, scope=expected_scope, eligibility=eligibility)
    return table


def write_shape_table(table: ShapeRuntimeTable, out: str | Path) -> tuple[Path, str]:
    """Publish the table to ``--out`` durably; a crash leaves old or new bytes."""
    path = Path(out)
    payload = DIRECT_ASCII_STRICT.encoded(table.as_dict()) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path, file_sha256hex(path)


def _load_shape_table_scope(path: Path) -> ShapeTableScope:
    body = _object(_json(path), tuple(ShapeTableScope.__dataclass_fields__), "expected scope")
    return ShapeTableScope(**{name: body[name] for name in ShapeTableScope.__dataclass_fields__})


def _convert_eligibility(scope: "ShapeTableScope | None") -> "EligibilityTable | None":
    """Load PQ's OWN pinned eligibility table for the scope, or abstain.

    The pinned-runtime gate is the allocator's own live owner
    (``tessera_lane.allocation_shape_price_scope``); a missing or uninstalled
    pinned runtime refuses rather than admitting on the observation's word.
    With no expected scope there is nothing to admit against, and the
    converted table is returned unadmitted.
    """
    if scope is None:
        return None
    if scope.contract_sha256 != _pinned_contract_sha256():
        raise ShapeRuntimeError(
            f"expected scope contract {scope.contract_sha256} is not PQ's pinned contract")
    from .tessera_lane import allocation_shape_price_scope
    from .tessera_serving_scope import ServingTarget
    target = ServingTarget(platform=scope.platform, runtime_image=scope.runtime_image_digest,
                           execution_mode=scope.execution_mode, residency=scope.residency)
    live_scope, eligibility, _formats = allocation_shape_price_scope(
        target, tensor_parallel=scope.tensor_parallel)
    if live_scope != scope:
        raise ShapeRuntimeError(
            f"expected scope {scope} differs from the pinned runtime's own scope {live_scope}")
    return eligibility


def _pinned_contract_sha256() -> str:
    """PQ's tracked serving pin's contract digest, without importing a runtime."""
    from .tessera_serving_runtime_pin import load_tessera_serving_runtime_pin
    return load_tessera_serving_runtime_pin().contract_sha256


def main(argv: Sequence[str] | None = None) -> int:
    """``convert`` turns validated observations into a table; ``check`` re-reads one."""
    parser = argparse.ArgumentParser(prog="python -m prismaquant.shape_runtime_prices")
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert", help="tessera.shape_time_observation.v1 -> table")
    convert.add_argument("--out", type=Path, required=True)
    convert.add_argument("--table-id", required=True)
    convert.add_argument("--observations", type=Path, nargs="+", required=True,
                         help="one or more tessera.shape_time_observation.v1 documents")
    convert.add_argument("--checker-receipts", type=Path, nargs="+")
    convert.add_argument("--expected-scope", type=Path,
                         help="JSON file the table context must equal (PQ's pinned b40 scope)")
    check = sub.add_parser("check", help="parse a table and verify its receipt digests/samples")
    check.add_argument("table", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "convert":
            scope = _load_shape_table_scope(args.expected_scope) if args.expected_scope else None
            table = consume_shape_time_observation(args.observations, table_id=args.table_id,
                                                   checker_receipts=args.checker_receipts,
                                                   expected_scope=scope,
                                                   eligibility=_convert_eligibility(scope))
            path, digest = write_shape_table(table, args.out)
            print(DIRECT_ASCII_SPACED_LAX.text({**table.identity(), "source_path": str(path),
                                              "out_sha256": digest}), flush=True)
            return 0
        table = load_shape_table(args.table)
        print(json.dumps(table.identity()), flush=True)
        return 0
    except RuntimePriceError as exc:
        print(f"[shape-runtime-prices] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
