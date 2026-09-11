"""Deterministic finite-domain adaptive interpolation for research studies.

This module is deliberately format-neutral and CPU-only.  It selects the next
coordinate from a caller-declared finite integer domain using only values that
have already been measured.  A caller obtains a :class:`AdaptiveProbe`, runs
its real measurement, and supplies that value to :meth:`record_measurement`.
There is no oracle callback and no path that silently substitutes a different
coordinate when a requested one is unavailable.

An accepted interval means that its predeclared sentinel measurement agreed
with the *frozen parent* piecewise-linear prediction within the declared
relative tolerance.  It is empirical evidence, not a mathematical assertion
about every unseen coordinate in that interval.  In particular, an accepted
midpoint must never be reported as validation of the whole domain.
"""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Literal

from .anchored_shape import AnchoredShapeError


InterpolationMode = Literal["log2", "value"]


def _finite_positive(value: object, where: str) -> float:
    if isinstance(value, bool):
        raise AnchoredShapeError(f"{where} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnchoredShapeError(f"{where} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise AnchoredShapeError(f"{where} must be a positive finite number")
    return result


def _domain(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise AnchoredShapeError("candidate coordinates must be a finite integer sequence")
    try:
        coordinates = tuple(values)
    except TypeError as exc:
        raise AnchoredShapeError("candidate coordinates must be a finite integer sequence") from exc
    if len(coordinates) < 2:
        raise AnchoredShapeError("candidate coordinates need two endpoints")
    if any(type(value) is not int for value in coordinates):
        raise AnchoredShapeError("candidate coordinates must be integers")
    if any(left >= right for left, right in zip(coordinates, coordinates[1:])):
        raise AnchoredShapeError("candidate coordinates must be strictly increasing")
    return coordinates


def _mode(value: str) -> InterpolationMode:
    if not isinstance(value, str) or value not in {"log2", "value"}:
        raise AnchoredShapeError("interpolation mode must be 'log2' or 'value'")
    return value  # type: ignore[return-value]


def _relative_tolerance(value: object) -> float:
    if isinstance(value, bool):
        raise AnchoredShapeError("relative tolerance must be a finite nonnegative number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnchoredShapeError("relative tolerance must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise AnchoredShapeError("relative tolerance must be a finite nonnegative number")
    return result


def _measurement_cap(value: object) -> int:
    if type(value) is not int or value < 2:
        raise AnchoredShapeError("max_measurements must be an integer of at least two")
    return value


def _checks_per_interval(value: object) -> int:
    if type(value) is not int or value not in {1, 2}:
        raise AnchoredShapeError("checks_per_interval must be one or two")
    return value


def _validate_measurements(
    domain: tuple[int, ...], values: Mapping[int, float],
) -> dict[int, float]:
    measured: dict[int, float] = {}
    for coordinate, value in values.items():
        if type(coordinate) is not int or coordinate not in domain:
            raise AnchoredShapeError("measurement coordinate is outside the candidate domain")
        if coordinate in measured:
            raise AnchoredShapeError("measurement coordinate is repeated")
        measured[coordinate] = _finite_positive(value, "measurement value")
    if len(measured) < 2:
        raise AnchoredShapeError("adaptive interpolation needs two measured endpoints")
    ordered = sorted(measured.items())
    for (_, left), (_, right) in zip(ordered, ordered[1:]):
        if right >= left:
            raise AnchoredShapeError(
                "measurements must strictly decrease as coordinate rises; "
                "a nonmonotone curve is refused"
            )
    return measured


def _predict_between(
    mode: InterpolationMode,
    left_coordinate: int,
    left_value: float,
    right_coordinate: int,
    right_value: float,
    coordinate: int,
) -> float:
    if coordinate == left_coordinate:
        return left_value
    if coordinate == right_coordinate:
        return right_value
    if not left_coordinate < coordinate < right_coordinate:
        raise AnchoredShapeError("prediction extrapolates outside a measured interval")
    try:
        ratio = (coordinate - left_coordinate) / (right_coordinate - left_coordinate)
        if not math.isfinite(ratio):
            raise ValueError("coordinate ratio is not finite")
        if mode == "value":
            prediction = left_value + ratio * (right_value - left_value)
        else:
            prediction = 2.0 ** (
                math.log2(left_value)
                + ratio * (math.log2(right_value) - math.log2(left_value))
            )
    except (OverflowError, ValueError) as exc:
        raise AnchoredShapeError("interpolation is not representable") from exc
    if not math.isfinite(prediction) or prediction <= 0.0:
        raise AnchoredShapeError("interpolation is not representable")
    return prediction


def _sentinel_targets(
    domain: tuple[int, ...],
    left_coordinate: int,
    right_coordinate: int,
    checks_per_interval: int,
) -> tuple[int, ...]:
    """Return the one midpoint or two approximately third-point sentinels."""
    try:
        left_index = domain.index(left_coordinate)
        right_index = domain.index(right_coordinate)
    except ValueError as exc:
        raise AnchoredShapeError("interval endpoint is outside the candidate domain") from exc
    span = right_index - left_index
    if span < 2:
        return ()
    if checks_per_interval == 1:
        offsets = (span // 2,)
    else:
        offsets = ((span + 1) // 3, (2 * span + 2) // 3)
    return tuple(sorted({
        domain[left_index + min(span - 1, max(1, offset))]
        for offset in offsets
    }))


@dataclass(frozen=True, slots=True)
class AdaptiveProbe:
    """One requested measurement and its immutable pre-reveal prediction."""

    coordinate: int
    left_coordinate: int
    right_coordinate: int
    prediction: float


@dataclass(frozen=True, slots=True)
class AdaptiveCheck:
    """An observed sentinel and the residual of its frozen parent prediction."""

    coordinate: int
    prediction: float
    measured: float
    relative_residual: float


@dataclass(frozen=True, slots=True)
class AdaptiveInterval:
    """A bracket that still needs its predetermined sentinel checks."""

    left_coordinate: int
    right_coordinate: int
    checks: tuple[AdaptiveCheck, ...] = ()


@dataclass(frozen=True, slots=True)
class AdaptiveIntervalEvidence:
    """Empirical acceptance of one parent interval, never a shape proof."""

    left_coordinate: int
    right_coordinate: int
    checks: tuple[AdaptiveCheck, ...]


@dataclass(frozen=True, slots=True)
class AdaptiveAnchoredCurve:
    """Immutable adaptive PWL state over a fixed finite integer domain.

    ``accepted_intervals`` only records sentinel agreement.  It intentionally
    has no ``validated`` property: passing one or two points cannot establish
    an unseen function's shape.  ``unverified_intervals`` records brackets
    still awaiting a request; if the cap is exhausted they remain there rather
    than being relabelled accepted.
    """

    candidate_coordinates: tuple[int, ...]
    mode: InterpolationMode
    relative_tolerance: float
    max_measurements: int
    checks_per_interval: int
    measurements: Mapping[int, float]
    _active_intervals: tuple[AdaptiveInterval, ...]
    pending_checks: tuple[AdaptiveProbe, ...]
    accepted_intervals: tuple[AdaptiveIntervalEvidence, ...]

    def __post_init__(self) -> None:
        domain = _domain(self.candidate_coordinates)
        mode = _mode(self.mode)
        tolerance = _relative_tolerance(self.relative_tolerance)
        cap = _measurement_cap(self.max_measurements)
        checks_per_interval = _checks_per_interval(self.checks_per_interval)
        measurements = _validate_measurements(domain, self.measurements)
        if len(measurements) > cap:
            raise AnchoredShapeError("measurement cap is lower than recorded unique measurements")
        if domain[0] not in measurements or domain[-1] not in measurements:
            raise AnchoredShapeError("the candidate-domain endpoints must be measured")
        active_pairs: set[tuple[int, int]] = set()
        for interval in self._active_intervals:
            if not isinstance(interval, AdaptiveInterval):
                raise AnchoredShapeError("active interval type is invalid")
            pair = interval.left_coordinate, interval.right_coordinate
            if pair in active_pairs or not interval.left_coordinate < interval.right_coordinate:
                raise AnchoredShapeError("active interval is invalid or repeated")
            active_pairs.add(pair)
            if (pair[0] not in measurements or pair[1] not in measurements
                    or pair[0] not in domain or pair[1] not in domain):
                raise AnchoredShapeError("active interval endpoint is not measured")
            targets = _sentinel_targets(domain, *pair, checks_per_interval)
            if not targets:
                raise AnchoredShapeError("active interval has no interior candidate")
            if len(interval.checks) >= len(targets):
                raise AnchoredShapeError("active interval has completed all of its sentinels")
            for index, check in enumerate(interval.checks):
                if not isinstance(check, AdaptiveCheck) or check.coordinate != targets[index]:
                    raise AnchoredShapeError("active interval check is not the next deterministic sentinel")
                if measurements.get(check.coordinate) != check.measured:
                    raise AnchoredShapeError("active interval check is not an exact measurement")
                expected = _predict_between(
                    mode, pair[0], measurements[pair[0]], pair[1],
                    measurements[pair[1]], check.coordinate,
                )
                expected_residual = abs(check.measured - expected) / check.measured
                if check.prediction != expected or check.relative_residual != expected_residual:
                    raise AnchoredShapeError("active interval check does not match its frozen parent")
                if check.relative_residual > tolerance:
                    raise AnchoredShapeError("a failed sentinel must split its parent immediately")
        accepted_pairs: set[tuple[int, int]] = set()
        for evidence in self.accepted_intervals:
            if not isinstance(evidence, AdaptiveIntervalEvidence):
                raise AnchoredShapeError("accepted interval evidence type is invalid")
            pair = evidence.left_coordinate, evidence.right_coordinate
            if (pair in accepted_pairs or not pair[0] < pair[1]
                    or pair[0] not in domain or pair[1] not in domain
                    or pair[0] not in measurements or pair[1] not in measurements):
                raise AnchoredShapeError("accepted interval is invalid or repeated")
            accepted_pairs.add(pair)
            targets = _sentinel_targets(domain, *pair, checks_per_interval)
            if not targets or len(evidence.checks) != len(targets):
                raise AnchoredShapeError("accepted interval lacks all deterministic sentinels")
            for check, target in zip(evidence.checks, targets):
                if not isinstance(check, AdaptiveCheck) or check.coordinate != target:
                    raise AnchoredShapeError("accepted interval check is not a deterministic sentinel")
                if measurements.get(check.coordinate) != check.measured:
                    raise AnchoredShapeError("accepted interval check is not an exact measurement")
                expected = _predict_between(
                    mode, pair[0], measurements[pair[0]], pair[1],
                    measurements[pair[1]], check.coordinate,
                )
                expected_residual = abs(check.measured - expected) / check.measured
                if (check.prediction != expected
                        or check.relative_residual != expected_residual
                        or check.relative_residual > tolerance):
                    raise AnchoredShapeError("accepted interval check does not support empirical acceptance")
        sorted_pairs = sorted((*active_pairs, *accepted_pairs))
        if any(left_right[0] < previous[1]
               for previous, left_right in zip(sorted_pairs, sorted_pairs[1:])):
            raise AnchoredShapeError("active and accepted intervals overlap")
        interval_pairs = (*active_pairs, *accepted_pairs)
        for left, right in zip(domain, domain[1:]):
            covered_by_interval = any(
                interval_left <= left and right <= interval_right
                for interval_left, interval_right in interval_pairs
            )
            covered_by_exact_step = left in measurements and right in measurements
            if not covered_by_interval and not covered_by_exact_step:
                raise AnchoredShapeError(
                    "active and accepted intervals do not cover the candidate domain"
                )
        if len(self.pending_checks) > 1:
            raise AnchoredShapeError("at most one deterministic measurement request may be pending")
        if len(measurements) + len(self.pending_checks) > cap:
            raise AnchoredShapeError("pending request would exceed the measurement cap")
        for probe in self.pending_checks:
            if not isinstance(probe, AdaptiveProbe):
                raise AnchoredShapeError("pending probe type is invalid")
            pair = probe.left_coordinate, probe.right_coordinate
            if pair not in active_pairs or probe.coordinate in measurements:
                raise AnchoredShapeError("pending probe does not belong to an unmeasured active interval")
            if (type(probe.coordinate) is not int or probe.coordinate not in domain
                    or not pair[0] < probe.coordinate < pair[1]
                    or not math.isfinite(probe.prediction) or probe.prediction <= 0.0):
                raise AnchoredShapeError("pending probe is invalid")
            parent = next(
                interval for interval in self._active_intervals
                if (interval.left_coordinate, interval.right_coordinate) == pair
            )
            completed = {check.coordinate for check in parent.checks}
            next_target = next(
                (target for target in _sentinel_targets(domain, *pair, checks_per_interval)
                 if target not in completed),
                None,
            )
            expected = _predict_between(
                mode, pair[0], measurements[pair[0]], pair[1],
                measurements[pair[1]], probe.coordinate,
            )
            if probe.coordinate != next_target or probe.prediction != expected:
                raise AnchoredShapeError("pending probe does not match its frozen parent")
        object.__setattr__(self, "candidate_coordinates", domain)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "relative_tolerance", tolerance)
        object.__setattr__(self, "max_measurements", cap)
        object.__setattr__(self, "checks_per_interval", checks_per_interval)
        object.__setattr__(self, "measurements", MappingProxyType(measurements))

    @classmethod
    def start(
        cls,
        candidate_coordinates: Sequence[int],
        endpoint_values: Mapping[int, float],
        *,
        mode: InterpolationMode = "log2",
        relative_tolerance: float,
        max_measurements: int,
        checks_per_interval: int = 1,
    ) -> "AdaptiveAnchoredCurve":
        """Start with exactly the two measured endpoints of a fixed domain."""
        domain = _domain(candidate_coordinates)
        if set(endpoint_values) != {domain[0], domain[-1]}:
            raise AnchoredShapeError("start requires exactly the two endpoint measurements")
        active: tuple[AdaptiveInterval, ...] = ()
        if len(domain) > 2:
            active = (AdaptiveInterval(domain[0], domain[-1]),)
        return cls(
            domain, mode, relative_tolerance, max_measurements,
            checks_per_interval, endpoint_values, active, (), (),
        )

    @property
    def measurement_count(self) -> int:
        """Actual unique revealed values; pending requests are not measurements."""
        return len(self.measurements)

    @property
    def unverified_intervals(self) -> tuple[AdaptiveInterval, ...]:
        """Unaccepted brackets that have no request currently in flight."""
        pending_pairs = {
            (probe.left_coordinate, probe.right_coordinate)
            for probe in self.pending_checks
        }
        return tuple(
            interval for interval in self._active_intervals
            if (interval.left_coordinate, interval.right_coordinate) not in pending_pairs
        )

    @property
    def budget_exhausted(self) -> bool:
        """Whether further unique reveals are forbidden while work remains."""
        return bool(self._active_intervals) and (
            self.measurement_count + len(self.pending_checks)
            >= self.max_measurements
        )

    @property
    def all_candidates_measured(self) -> bool:
        """True only when every finite-domain coordinate is an exact anchor."""
        return len(self.measurements) == len(self.candidate_coordinates)

    def predict(self, coordinate: int) -> float:
        """Return an exact anchor or current PWL prediction; never extrapolate."""
        if type(coordinate) is not int or coordinate not in self.candidate_coordinates:
            raise AnchoredShapeError("prediction coordinate is outside the finite candidate domain")
        if coordinate in self.measurements:
            return self.measurements[coordinate]
        measured_coordinates = tuple(sorted(self.measurements))
        upper_index = bisect_left(measured_coordinates, coordinate)
        if upper_index == 0 or upper_index == len(measured_coordinates):
            raise AnchoredShapeError("prediction extrapolates outside measured endpoints")
        left, right = measured_coordinates[upper_index - 1], measured_coordinates[upper_index]
        return _predict_between(
            self.mode, left, self.measurements[left], right,
            self.measurements[right], coordinate,
        )

    def _targets_for(self, interval: AdaptiveInterval) -> tuple[int, ...]:
        return _sentinel_targets(
            self.candidate_coordinates, interval.left_coordinate,
            interval.right_coordinate, self.checks_per_interval,
        )

    def request_next(self) -> tuple["AdaptiveAnchoredCurve", AdaptiveProbe | None]:
        """Request the deterministic next sentinel without receiving any truth.

        The selection is solely a stable ordering of currently unverified
        brackets (widest coordinate span, then lower endpoint) and their
        predeclared sentinel positions.
        """
        if self.pending_checks or self.measurement_count >= self.max_measurements:
            return self, None
        candidates = [
            interval for interval in self._active_intervals
            if len(interval.checks) < len(self._targets_for(interval))
        ]
        if not candidates:
            return self, None
        interval = min(
            candidates,
            key=lambda item: (
                -(item.right_coordinate - item.left_coordinate),
                item.left_coordinate,
                item.right_coordinate,
            ),
        )
        completed = {check.coordinate for check in interval.checks}
        coordinate = next(
            target for target in self._targets_for(interval)
            if target not in completed
        )
        probe = AdaptiveProbe(
            coordinate, interval.left_coordinate, interval.right_coordinate,
            _predict_between(
                self.mode, interval.left_coordinate,
                self.measurements[interval.left_coordinate],
                interval.right_coordinate,
                self.measurements[interval.right_coordinate], coordinate,
            ),
        )
        return replace(self, pending_checks=(probe,)), probe

    def record_measurement(self, coordinate: int, value: float) -> "AdaptiveAnchoredCurve":
        """Consume one requested real measurement and update the state machine.

        The residual is intentionally calculated from ``AdaptiveProbe``'s
        stored prediction before the new anchor changes any local PWL segment.
        A failure splits the parent immediately; with two sentinels, a passing
        first check waits for the other sentinel before accepting the parent.
        """
        if len(self.pending_checks) != 1:
            raise AnchoredShapeError("measurement does not match a pending deterministic request")
        probe = self.pending_checks[0]
        if type(coordinate) is not int or coordinate != probe.coordinate:
            raise AnchoredShapeError("measurement does not match the pending coordinate")
        if coordinate in self.measurements:
            raise AnchoredShapeError("a measured coordinate cannot be revealed again")
        measured_value = _finite_positive(value, "measurement value")
        expanded_measurements = dict(self.measurements)
        expanded_measurements[coordinate] = measured_value
        _validate_measurements(self.candidate_coordinates, expanded_measurements)
        residual = abs(measured_value - probe.prediction) / measured_value
        if not math.isfinite(residual):
            raise AnchoredShapeError("measurement residual is not representable")
        matching = next(
            (interval for interval in self._active_intervals
             if (interval.left_coordinate, interval.right_coordinate)
             == (probe.left_coordinate, probe.right_coordinate)),
            None,
        )
        if matching is None:
            raise AnchoredShapeError("pending measurement parent is absent")
        check = AdaptiveCheck(coordinate, probe.prediction, measured_value, residual)
        checked_parent = replace(matching, checks=(*matching.checks, check))
        remaining = [
            interval for interval in self._active_intervals if interval != matching
        ]
        accepted = list(self.accepted_intervals)
        if residual <= self.relative_tolerance and (
            len(checked_parent.checks) == len(self._targets_for(checked_parent))
        ):
            accepted.append(AdaptiveIntervalEvidence(
                checked_parent.left_coordinate, checked_parent.right_coordinate,
                checked_parent.checks,
            ))
        elif residual <= self.relative_tolerance:
            remaining.append(checked_parent)
        else:
            # Every already revealed point inside the failed parent becomes an
            # exact bracket boundary.  This prevents an earlier passing
            # sentinel from being silently dropped when a later sentinel fails.
            parent_points = sorted(
                point for point in expanded_measurements
                if matching.left_coordinate <= point <= matching.right_coordinate
            )
            domain_positions = {
                point: self.candidate_coordinates.index(point)
                for point in parent_points
            }
            for left, right in zip(parent_points, parent_points[1:]):
                if domain_positions[right] - domain_positions[left] > 1:
                    remaining.append(AdaptiveInterval(left, right))
        return replace(
            self, measurements=expanded_measurements,
            _active_intervals=tuple(sorted(
                remaining,
                key=lambda item: (item.left_coordinate, item.right_coordinate),
            )), pending_checks=(), accepted_intervals=tuple(accepted),
        )


__all__ = [
    "AdaptiveAnchoredCurve",
    "AdaptiveCheck",
    "AdaptiveInterval",
    "AdaptiveIntervalEvidence",
    "AdaptiveProbe",
    "InterpolationMode",
]
