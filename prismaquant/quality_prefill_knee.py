"""Deterministic development-point selection on a quality-prefill frontier.

Specification section 9 of ``docs/design/tessera_quality_prefill_experiment.md``.

This module answers one narrow question: given a set of *proposed* points that
an allocator already produced, which one does the frozen geometric rule
nominate for measurement, and which neighbours and endpoint controls must be
measured beside it?

It computes nothing about quality or speed. Every number it reads was measured
or proposed somewhere else, and every number it writes is a copy or a
normalised coordinate. In particular it never upgrades a point's
``measurement_status``: a knee proposed from screened points is a proposal
about screened points.

Three properties are deliberate:

* **The rule proposes a region, not a decision.** Section 9's closing sentence
  is that Rob's actual quality/budget choice remains an explicit configuration
  input. So :func:`select_development_point` always returns the adjacent
  non-dominated neighbours and both feasible endpoint controls alongside the
  nominee, and the caller measures all of them.
* **Degeneracy refuses.** A zero-width range, fewer than three non-dominated
  points, or two candidates separated by less than the caller's declared
  resolution yields no unique knee. The refusal names which of those it was.
* **No RNG.** Ordering is by exact numeric comparison and then by
  ``assignment_id`` string order. There is no ``random`` import and
  ``tests/test_quality_prefill_knee.py`` asserts its absence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .quality_prefill_contract import (
    COVERAGE_TYPES,
    CURRENCIES,
    QualityPrefillContractError,
    canonical_sha256,
)

__all__ = [
    "KneeSelectionError",
    "FrontierPoint",
    "Normalisation",
    "KneeSelection",
    "read_points",
    "feasible_points",
    "nondominated",
    "select_development_point",
]

#: The rule's own identity. A change to the geometry is a new version, never a
#: silent change to this one: published selections cite it.
KNEE_RULE = MappingProxyType({"name": "chord_perpendicular_v1", "version": "v1"})

#: Statuses that may appear in one frontier. ``measured`` and every screening
#: status are kept apart, because a frontier that mixes them is a confound
#: rather than a frontier (principle 3: a screen is not a result).
_MEASURED = "measured"


class KneeSelectionError(QualityPrefillContractError):
    """A frontier cannot nominate a development point.

    This subclasses the contract error because every case it reports is a
    fixable document: a wider sweep, a declared resolution, or a single
    currency. It is emphatically *not* the fleet-capability refusal that
    ``quality_prefill_pb_adapter.DecompositionUnavailable`` is.
    """


@dataclass(frozen=True)
class FrontierPoint:
    """One proposed operating point, exactly as the frontier report carries it."""

    point_id: str
    assignment_id: str
    bytes: int
    prefill_budget: int
    quality_value: float
    currency: str
    units: str
    measurement_status: str

    def as_point(self) -> dict[str, object]:
        """Return the eight-key mapping ``validate_frontier_report`` accepts."""
        return {
            "point_id": self.point_id,
            "assignment_id": self.assignment_id,
            "bytes": self.bytes,
            "prefill_budget": self.prefill_budget,
            "quality_value": self.quality_value,
            "currency": self.currency,
            "units": self.units,
            "measurement_status": self.measurement_status,
        }


@dataclass(frozen=True)
class Normalisation:
    """The published [0,1] transform and the chord it induces.

    Section 9 requires the normalisation and chord to be published, because the
    same points under a different range nominate a different knee. Storing them
    beside the nominee is what makes a selection re-derivable.
    """

    latency_min: int
    latency_max: int
    quality_min: float
    quality_max: float
    chord_from_point_id: str
    chord_to_point_id: str

    def latency(self, value: int) -> float:
        return (value - self.latency_min) / (self.latency_max - self.latency_min)

    def quality(self, value: float) -> float:
        return (value - self.quality_min) / (self.quality_max - self.quality_min)

    def as_dict(self) -> dict[str, object]:
        return {
            "latency_min": self.latency_min,
            "latency_max": self.latency_max,
            "quality_min": self.quality_min,
            "quality_max": self.quality_max,
            "chord_from_point_id": self.chord_from_point_id,
            "chord_to_point_id": self.chord_to_point_id,
        }


@dataclass(frozen=True)
class KneeSelection:
    """What the rule nominates, and everything measured beside it."""

    knee_point_id: str
    neighbour_point_ids: tuple[str, ...]
    endpoint_point_ids: tuple[str, ...]
    nondominated_point_ids: tuple[str, ...]
    improvements: Mapping[str, float]
    normalisation: Normalisation
    byte_budget: int
    currency: str
    units: str
    measurement_status: str
    rule: Mapping[str, object]

    def measurement_roster(self) -> tuple[str, ...]:
        """The exact points section 10 requires be measured for this selection.

        The knee, its two available neighbours and both feasible endpoint
        controls, deduplicated, in non-dominated latency order.
        """
        wanted = {self.knee_point_id, *self.neighbour_point_ids, *self.endpoint_point_ids}
        return tuple(pid for pid in self.nondominated_point_ids if pid in wanted)

    def as_dict(self) -> dict[str, object]:
        return {
            "rule": dict(self.rule),
            "byte_budget": self.byte_budget,
            "currency": self.currency,
            "units": self.units,
            "measurement_status": self.measurement_status,
            "knee_point_id": self.knee_point_id,
            "neighbour_point_ids": list(self.neighbour_point_ids),
            "endpoint_point_ids": list(self.endpoint_point_ids),
            "nondominated_point_ids": list(self.nondominated_point_ids),
            "improvements": {k: self.improvements[k] for k in sorted(self.improvements)},
            "normalisation": self.normalisation.as_dict(),
            "measurement_roster": list(self.measurement_roster()),
        }

    def identity_sha256(self) -> str:
        """SHA-256 over canonical JSON of :meth:`as_dict`, no RNG anywhere."""
        return canonical_sha256(self.as_dict())


def _fail(message: str) -> None:
    raise KneeSelectionError(message)


def read_points(values: Iterable[Mapping[str, object]]) -> tuple[FrontierPoint, ...]:
    """Read frontier-report point mappings into typed points.

    Refuses a mixed currency, a mixed unit, a mixed measurement status and a
    duplicate point id. Each of those would make the comparison that follows
    meaningless rather than merely imprecise, so none of them is normalised
    away.
    """
    points: list[FrontierPoint] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        where = f"points[{index}]"
        if not isinstance(raw, Mapping):
            _fail(f"{where} must be a mapping")
        missing = {
            "point_id",
            "assignment_id",
            "bytes",
            "prefill_budget",
            "quality_value",
            "currency",
            "units",
            "measurement_status",
        } - set(raw)
        if missing:
            _fail(f"{where} is missing required key(s): {', '.join(sorted(missing))}")
        point_id = raw["point_id"]
        if not isinstance(point_id, str) or not point_id:
            _fail(f"{where}.point_id must be a non-empty string")
        if point_id in seen:
            _fail(f"{where}.point_id {point_id!r} is already present")
        seen.add(point_id)
        assignment_id = raw["assignment_id"]
        if not isinstance(assignment_id, str) or not assignment_id:
            _fail(f"{where}.assignment_id must be a non-empty string")
        for name in ("bytes", "prefill_budget"):
            if type(raw[name]) is not int or raw[name] < 1:
                _fail(f"{where}.{name} must be a positive int")
        quality = raw["quality_value"]
        if type(quality) not in (int, float) or quality != quality or quality in (
            float("inf"),
            float("-inf"),
        ):
            _fail(f"{where}.quality_value must be a finite number")
        if quality < 0.0:
            _fail(f"{where}.quality_value must not be negative")
        if raw["currency"] not in CURRENCIES:
            _fail(f"{where}.currency {raw['currency']!r} is not a known currency")
        if not isinstance(raw["units"], str) or not raw["units"]:
            _fail(f"{where}.units must be a non-empty string")
        if raw["measurement_status"] not in COVERAGE_TYPES:
            _fail(
                f"{where}.measurement_status {raw['measurement_status']!r} is not a "
                "declared coverage type"
            )
        points.append(
            FrontierPoint(
                point_id=point_id,
                assignment_id=assignment_id,
                bytes=int(raw["bytes"]),
                prefill_budget=int(raw["prefill_budget"]),
                quality_value=float(quality),
                currency=str(raw["currency"]),
                units=str(raw["units"]),
                measurement_status=str(raw["measurement_status"]),
            )
        )
    if not points:
        _fail("a frontier needs at least one point")
    currencies = {point.currency for point in points}
    if len(currencies) != 1:
        _fail(
            "one frontier carries one currency; got "
            + ", ".join(sorted(currencies))
        )
    units = {point.units for point in points}
    if len(units) != 1:
        _fail("one frontier carries one unit; got " + ", ".join(sorted(units)))
    statuses = {point.measurement_status for point in points}
    if len(statuses) != 1:
        _fail(
            "one frontier carries one measurement status, because a measured point "
            "and a screened point are not comparable; got "
            + ", ".join(sorted(statuses))
        )
    return tuple(points)


def feasible_points(
    points: Sequence[FrontierPoint], *, byte_budget: int
) -> tuple[FrontierPoint, ...]:
    """Points whose exact serialised bytes fit the budget.

    Bytes are exact integers, so the comparison is exact; there is no tolerance
    and no rounding to a gigabyte.
    """
    if type(byte_budget) is not int or byte_budget < 1:
        _fail("byte_budget must be a positive int")
    return tuple(point for point in points if point.bytes <= byte_budget)


def nondominated(points: Sequence[FrontierPoint]) -> tuple[FrontierPoint, ...]:
    """Return the non-dominated set, ordered by increasing prefill budget.

    ``a`` dominates ``b`` when ``a`` is no worse on both axes and strictly
    better on at least one. Ties on both axes keep both points: they are
    different assignments that happen to price alike, and deleting one would
    hide a distinct execution route (section 5.4).
    """
    keep: list[FrontierPoint] = []
    for point in points:
        dominated = any(
            other is not point
            and other.prefill_budget <= point.prefill_budget
            and other.quality_value <= point.quality_value
            and (
                other.prefill_budget < point.prefill_budget
                or other.quality_value < point.quality_value
            )
            for other in points
        )
        if not dominated:
            keep.append(point)
    return tuple(
        sorted(keep, key=lambda p: (p.prefill_budget, p.quality_value, p.assignment_id))
    )


def _endpoints(
    frontier: Sequence[FrontierPoint],
) -> tuple[FrontierPoint, FrontierPoint]:
    """The fastest feasible point and the best-quality feasible point.

    Both are controls in their own right; section 10 requires measuring them
    whether or not the knee is near either.
    """
    fastest = min(
        frontier, key=lambda p: (p.prefill_budget, p.quality_value, p.assignment_id)
    )
    best = min(
        frontier, key=lambda p: (p.quality_value, p.prefill_budget, p.assignment_id)
    )
    return fastest, best


def select_development_point(
    points: Iterable[Mapping[str, object]],
    *,
    byte_budget: int,
    min_separation: float,
) -> KneeSelection:
    """Nominate a development point by the frozen chord-perpendicular rule.

    ``min_separation`` is the caller's declared resolution in normalised
    perpendicular units: two candidates closer than it are not distinguishable
    by this frontier's evidence and the function refuses rather than picking
    the numerically larger one. It has no default, because a default would be
    a threshold invented here rather than derived from the measurement
    (principle 2).

    Raises :class:`KneeSelectionError` when the frontier is degenerate, too
    small, or cannot separate its top two candidates.
    """
    if type(min_separation) is not float and type(min_separation) is not int:
        _fail("min_separation must be a number")
    if min_separation < 0.0 or min_separation != min_separation:
        _fail("min_separation must be a finite non-negative number")

    typed = read_points(points)
    feasible = feasible_points(typed, byte_budget=byte_budget)
    if not feasible:
        _fail(
            f"no point fits the byte budget {byte_budget}; the smallest is "
            f"{min(point.bytes for point in typed)} bytes"
        )
    frontier = nondominated(feasible)
    if len(frontier) < 3:
        _fail(
            "a chord and an interior both need points: the non-dominated set has "
            f"{len(frontier)}, and the rule needs at least 3"
        )

    fastest, best = _endpoints(frontier)
    if fastest.point_id == best.point_id:
        _fail(
            "the fastest and the best-quality feasible points are the same point; "
            "there is no trade-off to find a knee on"
        )
    latency_min, latency_max = fastest.prefill_budget, best.prefill_budget
    quality_min, quality_max = best.quality_value, fastest.quality_value
    if latency_max == latency_min:
        _fail(
            "degenerate latency range: every feasible non-dominated point proposes "
            f"{latency_min}, so no knee exists"
        )
    if quality_max == quality_min:
        _fail(
            "degenerate quality range: every feasible non-dominated point prices "
            f"{quality_min}, so no knee exists"
        )

    normalisation = Normalisation(
        latency_min=latency_min,
        latency_max=latency_max,
        quality_min=quality_min,
        quality_max=quality_max,
        chord_from_point_id=fastest.point_id,
        chord_to_point_id=best.point_id,
    )

    # The chord runs from the fastest endpoint (x=0, y=1) to the best-quality
    # endpoint (x=1, y=0) in normalised coordinates. Perpendicular improvement
    # is the signed distance towards the origin: a point below the chord buys
    # more quality per unit of prefill than linear interpolation would.
    ax, ay = normalisation.latency(fastest.prefill_budget), normalisation.quality(
        fastest.quality_value
    )
    bx, by = normalisation.latency(best.prefill_budget), normalisation.quality(
        best.quality_value
    )
    dx, dy = bx - ax, by - ay
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0.0:
        _fail("the endpoint chord has zero length; no knee exists")

    improvements: dict[str, float] = {}
    interior: list[FrontierPoint] = []
    for point in frontier:
        px = normalisation.latency(point.prefill_budget)
        py = normalisation.quality(point.quality_value)
        # Positive when the point lies on the improving side of the chord.
        cross = dx * (py - ay) - dy * (px - ax)
        improvements[point.point_id] = -cross / length
        if point.point_id not in (fastest.point_id, best.point_id):
            interior.append(point)

    if not interior:
        _fail("the non-dominated set is its own two endpoints; no interior knee exists")

    ranked = sorted(
        interior,
        key=lambda p: (
            -improvements[p.point_id],
            p.prefill_budget,
            p.assignment_id,
        ),
    )
    winner = ranked[0]
    if improvements[winner.point_id] <= 0.0:
        _fail(
            "no interior point improves on the endpoint chord; the frontier is "
            "linear or convex here and proposes no knee"
        )
    if len(ranked) > 1:
        gap = improvements[winner.point_id] - improvements[ranked[1].point_id]
        if gap < min_separation:
            _fail(
                "two candidates are within the declared resolution "
                f"{min_separation}: {winner.point_id} and {ranked[1].point_id} differ "
                f"by {gap}. This frontier proposes a region, not a unique knee"
            )

    order = [point.point_id for point in frontier]
    index = order.index(winner.point_id)
    neighbours = tuple(
        order[position]
        for position in (index - 1, index + 1)
        if 0 <= position < len(order) and order[position] != winner.point_id
    )

    return KneeSelection(
        knee_point_id=winner.point_id,
        neighbour_point_ids=neighbours,
        endpoint_point_ids=(fastest.point_id, best.point_id),
        nondominated_point_ids=tuple(order),
        improvements=MappingProxyType(dict(improvements)),
        normalisation=normalisation,
        byte_budget=int(byte_budget),
        currency=typed[0].currency,
        units=typed[0].units,
        measurement_status=typed[0].measurement_status,
        rule=KNEE_RULE,
    )
