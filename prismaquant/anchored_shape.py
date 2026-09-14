"""Format-neutral shared log shapes and sparse per-unit anchor corrections.

Existing shared-shape logarithms are base ten. ``EndpointCurvatureModel`` is
explicitly log2 when its ``mode`` is ``"log2"``. Callers own currency, equivalence segments,
measurement identity, validation policy and fallback. A shared panel removes
per-unit log offsets before fitting its feature basis. One anchor restores a
unit's level; two or more also fit a residual slope in a centered, scaled
coordinate. Real anchors remain exact, including in an overdetermined fit.

Audit measurements never refit the model. An explicit later fit may consume
accepted audit measurements without changing the frozen earlier predictions.
This module performs scalar offline arithmetic and imports no model runtime.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType


class AnchoredShapeError(ValueError):
    """Invalid or unrepresentable numerical shape input."""


def _label(value: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnchoredShapeError(f"{where} must be a nonempty string")
    return value


def _finite(value: float, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise AnchoredShapeError(f"{where} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnchoredShapeError(f"{where} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise AnchoredShapeError(f"{where} must be finite" + (" and positive" if positive else ""))
    return result


def _tuple_of_finite(values: Sequence[float], where: str, *, positive: bool = False) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise AnchoredShapeError(f"{where} must be a finite numeric sequence")
    try:
        return tuple(_finite(value, where, positive=positive) for value in values)
    except TypeError as exc:
        raise AnchoredShapeError(f"{where} must be a finite numeric sequence") from exc


def _curvature_basis_width(feature_width: int, degree: int) -> int:
    return 1 + feature_width + (feature_width * (feature_width + 1) // 2 if degree == 2 else 0)


def _rate_span(rate_lo: float, rate_hi: float) -> float:
    try:
        span = rate_hi - rate_lo
    except OverflowError as exc:
        raise AnchoredShapeError("endpoint curvature rate span is not finite and positive") from exc
    if not math.isfinite(span) or span <= 0.0:
        raise AnchoredShapeError("endpoint curvature rate span is not finite and positive")
    return span


@dataclass(frozen=True)
class EndpointCurvatureCurve:
    """A bound endpoint curve with its t-polynomial already expanded."""

    mode: str
    rate_lo: float
    rate_hi: float
    left: float
    right: float
    c0: float
    c1: float
    c2: float

    def __post_init__(self) -> None:
        if self.mode not in {"value", "log2"}:
            raise AnchoredShapeError("endpoint curvature mode is invalid")
        for name in ("rate_lo", "rate_hi"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        _rate_span(self.rate_lo, self.rate_hi)
        for name in ("left", "right"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=True))
        for name in ("c0", "c1", "c2"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))

    def predict(self, rate: float) -> float:
        """Predict one declared in-range rate, retaining exact endpoints."""
        coordinate = _finite(rate, "prediction rate")
        if coordinate == self.rate_lo:
            return self.left
        if coordinate == self.rate_hi:
            return self.right
        if coordinate < self.rate_lo or coordinate > self.rate_hi:
            raise AnchoredShapeError("endpoint curvature prediction extrapolates outside its rate span")
        t = (coordinate - self.rate_lo) / _rate_span(self.rate_lo, self.rate_hi)
        if not 0.0 < t < 1.0 or not math.isfinite(t):
            raise AnchoredShapeError("endpoint curvature prediction coordinate is not representable")
        curvature = self.c0 + t * (self.c1 + t * self.c2)
        try:
            if self.mode == "value":
                value = ((1.0 - t) * self.left + t * self.right
                         + self.left * t * (1.0 - t) * curvature)
            else:
                log_left = math.log2(self.left)
                log_value = (log_left + t * (math.log2(self.right) - log_left)
                             + t * (1.0 - t) * curvature)
                value = 2.0 ** log_value
        except (OverflowError, ValueError) as exc:
            raise AnchoredShapeError("endpoint curvature prediction is not a positive finite float") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise AnchoredShapeError("endpoint curvature prediction is not a positive finite float")
        return value


@dataclass(frozen=True)
class EndpointCurvatureModel:
    """A caller-fitted standardized curvature polynomial.

    Feature order is ``[log2(left), log2(right)-log2(left), t, *context]``.
    The caller owns fit rows, segment identities, calibration identity, and
    admission policy; this object only binds a frozen fitted polynomial.
    """

    mode: str
    rate_lo: float
    rate_hi: float
    degree: int
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.mode not in {"value", "log2"}:
            raise AnchoredShapeError("endpoint curvature mode is invalid")
        for name in ("rate_lo", "rate_hi"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        _rate_span(self.rate_lo, self.rate_hi)
        if type(self.degree) is not int or self.degree not in (1, 2):
            raise AnchoredShapeError("endpoint curvature degree must be one or two")
        means = _tuple_of_finite(self.means, "endpoint curvature means")
        scales = _tuple_of_finite(self.scales, "endpoint curvature scales", positive=True)
        coefficients = _tuple_of_finite(self.coefficients, "endpoint curvature coefficients")
        if len(means) < 3 or len(scales) != len(means):
            raise AnchoredShapeError("endpoint curvature feature width is invalid")
        if len(coefficients) != _curvature_basis_width(len(means), self.degree):
            raise AnchoredShapeError("endpoint curvature coefficient width is invalid")
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "coefficients", coefficients)

    def bind(self, left: float, right: float, context: Sequence[float] = ()) -> EndpointCurvatureCurve:
        """Freeze a unit's endpoints/context and analytically expand curvature in t."""
        left = _finite(left, "endpoint curvature left", positive=True)
        right = _finite(right, "endpoint curvature right", positive=True)
        context = _tuple_of_finite(context, "endpoint curvature context")
        if len(context) != len(self.means) - 3:
            raise AnchoredShapeError("endpoint curvature context width is invalid")
        try:
            raw = (math.log2(left), math.log2(right) - math.log2(left), 0.0, *context)
            intercept = tuple((value - mean) / scale
                              for value, mean, scale in zip(raw, self.means, self.scales))
        except (OverflowError, ValueError) as exc:
            raise AnchoredShapeError("endpoint curvature binding is not representable") from exc
        if not all(math.isfinite(value) for value in intercept):
            raise AnchoredShapeError("endpoint curvature binding is not representable")
        slope = [0.0] * len(intercept)
        slope[2] = 1.0 / self.scales[2]
        c0, c1, c2 = self.coefficients[0], 0.0, 0.0
        coefficient_index = 1
        for index in range(len(intercept)):
            coefficient = self.coefficients[coefficient_index]
            coefficient_index += 1
            c0 += coefficient * intercept[index]
            c1 += coefficient * slope[index]
        if self.degree == 2:
            for left_index in range(len(intercept)):
                for right_index in range(left_index, len(intercept)):
                    coefficient = self.coefficients[coefficient_index]
                    coefficient_index += 1
                    a, b = intercept[left_index], intercept[right_index]
                    da, db = slope[left_index], slope[right_index]
                    c0 += coefficient * a * b
                    c1 += coefficient * (a * db + da * b)
                    c2 += coefficient * da * db
        if not all(math.isfinite(value) for value in (c0, c1, c2)):
            raise AnchoredShapeError("endpoint curvature expansion is not representable")
        return EndpointCurvatureCurve(self.mode, self.rate_lo, self.rate_hi,
                                      left, right, c0, c1, c2)


def fit_endpoint_curvature(
    observations: Sequence[LogShapeObservation],
    coordinates: Mapping[str, float],
    *,
    mode: str = "value",
    degree: int = 1,
) -> EndpointCurvatureModel:
    """Fit the study's standardized ridge/Huber endpoint-curvature polynomial.

    Every supplied pilot unit must have both endpoint keys and at least one
    measured interior key. This reader neither supplies missing anchors nor
    decides whether the fitted curve is admissible for a caller's segment.
    """
    if mode not in {"value", "log2"}:
        raise AnchoredShapeError("endpoint curvature mode is invalid")
    if type(degree) is not int or degree not in (1, 2):
        raise AnchoredShapeError("endpoint curvature degree must be one or two")
    if not coordinates:
        raise AnchoredShapeError("endpoint curvature coordinates are empty")
    coordinate_map = {
        _label(key, "endpoint curvature key"): _finite(value, "endpoint curvature coordinate")
        for key, value in coordinates.items()
    }
    if len(coordinate_map) < 3 or len(set(coordinate_map.values())) != len(coordinate_map):
        raise AnchoredShapeError("endpoint curvature coordinates do not identify an interior span")
    rate_lo, rate_hi = min(coordinate_map.values()), max(coordinate_map.values())
    span = _rate_span(rate_lo, rate_hi)
    by_unit: dict[str, dict[str, float]] = defaultdict(dict)
    for observation in observations:
        if not isinstance(observation, LogShapeObservation):
            raise AnchoredShapeError("endpoint curvature observation type is invalid")
        if observation.key not in coordinate_map:
            raise AnchoredShapeError("unknown endpoint curvature observation key")
        unit_rows = by_unit[observation.unit]
        if observation.key in unit_rows:
            raise AnchoredShapeError("duplicate endpoint curvature unit/key observation")
        unit_rows[observation.key] = observation.value
    if not by_unit:
        raise AnchoredShapeError("endpoint curvature observations are empty")
    low_keys = {key for key, rate in coordinate_map.items() if rate == rate_lo}
    high_keys = {key for key, rate in coordinate_map.items() if rate == rate_hi}
    if len(low_keys) != 1 or len(high_keys) != 1:
        raise AnchoredShapeError("endpoint curvature endpoint keys are ambiguous")
    low_key, high_key = next(iter(low_keys)), next(iter(high_keys))
    feature_rows: list[tuple[float, float, float]] = []
    response_rows: list[float] = []
    try:
        for unit, values in sorted(by_unit.items()):
            if low_key not in values or high_key not in values:
                raise AnchoredShapeError(f"endpoint curvature unit {unit!r} lacks an endpoint")
            interiors = [(rate, key) for key, rate in coordinate_map.items()
                         if rate_lo < rate < rate_hi and key in values]
            if not interiors:
                raise AnchoredShapeError(f"endpoint curvature unit {unit!r} lacks an interior observation")
        # The study's numerical order is rate-major then row-major.
        for rate, key in sorted((rate, key) for key, rate in coordinate_map.items()
                                if rate_lo < rate < rate_hi):
            t = (rate - rate_lo) / span
            denominator = t * (1.0 - t)
            if not 0.0 < t < 1.0 or not math.isfinite(denominator) or denominator <= 0.0:
                raise AnchoredShapeError("endpoint curvature interior coordinate is not representable")
            for unit in sorted(by_unit):
                values = by_unit[unit]
                if key not in values:
                    continue
                left, right, value = values[low_key], values[high_key], values[key]
                log_left = math.log2(left)
                log_slope = math.log2(right) - log_left
                if mode == "log2":
                    response = (math.log2(value) - (log_left + t * log_slope)) / denominator
                else:
                    chord = (1.0 - t) * left + t * right
                    response = (value - chord) / left / denominator
                if not math.isfinite(response):
                    raise AnchoredShapeError("endpoint curvature response is not representable")
                feature_rows.append((log_left, log_slope, t))
                response_rows.append(response)
    except AnchoredShapeError:
        raise
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise AnchoredShapeError("endpoint curvature fitting arithmetic is not representable") from exc
    if len(feature_rows) < 12:
        raise AnchoredShapeError("endpoint curvature has insufficient interior observations")
    try:
        # Kept lazy so scalar anchored-shape users do not import NumPy.
        import numpy as np
    except ImportError as exc:
        raise AnchoredShapeError("endpoint curvature fitting requires NumPy") from exc
    x, y = np.asarray(feature_rows, dtype=np.float64), np.asarray(response_rows, dtype=np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    z = (x - mean) / scale
    columns = [np.ones(len(z))]
    columns.extend(z[:, index] for index in range(z.shape[1]))
    if degree == 2:
        columns.extend(z[:, left_index] * z[:, right_index]
                       for left_index in range(z.shape[1])
                       for right_index in range(left_index, z.shape[1]))
    design = np.column_stack(columns)
    ridge = np.eye(design.shape[1]) * (len(y) * 1e-5)
    ridge[0, 0] = 0
    weights = np.ones(len(y))
    coefficients = np.zeros(design.shape[1])
    try:
        for _ in range(5):
            weighted = design * weights[:, None]
            coefficients = np.linalg.solve(design.T @ weighted + ridge, weighted.T @ y)
            residual = y - design @ coefficients
            mad = np.median(np.abs(residual - np.median(residual)))
            if mad == 0:
                break
            ratio = np.abs(residual) / (1.5 * mad)
            weights = np.ones(len(y))
            np.divide(1, ratio, out=weights, where=ratio > 1)
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError, ValueError) as exc:
        raise AnchoredShapeError("endpoint curvature fit is not representable") from exc
    if not (np.isfinite(mean).all() and np.isfinite(scale).all() and np.isfinite(coefficients).all()):
        raise AnchoredShapeError("endpoint curvature fit is not representable")
    return EndpointCurvatureModel(mode, rate_lo, rate_hi, degree,
                                  tuple(mean.tolist()), tuple(scale.tolist()),
                                  tuple(coefficients.tolist()))


@dataclass(frozen=True)
class LogShapeObservation:
    unit: str
    key: str
    value: float

    def __post_init__(self) -> None:
        _label(self.unit, "observation unit")
        _label(self.key, "observation key")
        object.__setattr__(self, "value", _finite(self.value, "observation value", positive=True))


@dataclass(frozen=True)
class SharedLogShape:
    coefficients: tuple[float, ...]
    reference_key: str
    log_shape_by_key: Mapping[str, float]
    design_rank: int
    n_units: int
    n_observations: int

    def __post_init__(self) -> None:
        values = {_label(key, "shape key"): _finite(value, "log shape")
                  for key, value in self.log_shape_by_key.items()}
        coefficients = tuple(_finite(value, "shape coefficient") for value in self.coefficients)
        if not values or self.reference_key not in values:
            raise AnchoredShapeError("shape reference is absent")
        if not coefficients or self.design_rank != len(coefficients):
            raise AnchoredShapeError("shape design rank is insufficient")
        if self.n_units < 1 or self.n_observations < 2*self.n_units:
            raise AnchoredShapeError("shape observation count is insufficient")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "log_shape_by_key", MappingProxyType(values))


@dataclass(frozen=True)
class AnchorCorrection:
    shape: SharedLogShape
    anchors: Mapping[str, float]
    coordinates: Mapping[str, float]
    center: float
    scale: float
    intercept: float
    slope: float

    def __post_init__(self) -> None:
        # Freeze caller-owned mappings so later measurement/refit cannot
        # change predictions used to evaluate an earlier audit.
        if not isinstance(self.shape, SharedLogShape):
            raise AnchoredShapeError("shared shape type is invalid")
        anchors = {_label(key, "anchor key"): _finite(value, "anchor value", positive=True)
                   for key, value in self.anchors.items()}
        coordinates = {_label(key, "coordinate key"): _finite(value, "coordinate")
                       for key, value in self.coordinates.items()}
        if not anchors or not set(anchors) <= set(coordinates):
            raise AnchoredShapeError("anchor keys are empty or unknown")
        if set(coordinates) != set(self.shape.log_shape_by_key):
            raise AnchoredShapeError("coordinate domain differs from shared shape")
        if len(set(coordinates.values())) != len(coordinates):
            raise AnchoredShapeError("duplicate coordinates do not identify distinct rungs")
        for name in ("center", "scale", "intercept", "slope"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=name == "scale"))
        object.__setattr__(self, "anchors", MappingProxyType(anchors))
        object.__setattr__(self, "coordinates", MappingProxyType(coordinates))


@dataclass(frozen=True)
class AnchoredAuditRow:
    key: str
    predicted: float
    measured: float
    absolute_log10_error: float


@dataclass(frozen=True)
class AnchoredAudit:
    rows: tuple[AnchoredAuditRow, ...]

    @property
    def max_absolute_log10_error(self) -> float:
        return max(row.absolute_log10_error for row in self.rows)


def rank_and_solve(
    x_rows: Sequence[Sequence[float]],
    y_rows: Sequence[float],
) -> tuple[int, tuple[float, ...]]:
    """Solve centered least squares through normal equations with pivoting."""
    if not x_rows:
        raise AnchoredShapeError("shape design is empty")
    width = len(x_rows[0])
    if width < 1 or any(len(row) != width for row in x_rows):
        raise AnchoredShapeError("shape feature width is invalid")
    # Plugin features have no prescribed unit.  Normalize every column before
    # rank detection/solving so a plugin expressing the same coordinate in
    # 1e-7 rather than 1 cannot turn an identifiable design into rank zero.
    column_scales = [
        math.sqrt(math.fsum(row[index] * row[index] for row in x_rows))
        for index in range(width)
    ]
    active = [scale > 0.0 and math.isfinite(scale) for scale in column_scales]
    if not all(active):
        return sum(active), tuple()
    normalized = [
        [row[index] / column_scales[index] for index in range(width)]
        for row in x_rows
    ]
    gram = [
        [math.fsum(row[i] * row[j] for row in normalized) for j in range(width)]
        for i in range(width)
    ]
    rhs = [
        math.fsum(row[i] * value for row, value in zip(normalized, y_rows))
        for i in range(width)
    ]
    scale = max((abs(value) for row in gram for value in row), default=1.0)
    tolerance = scale * 1e-12
    augmented = [gram[index] + [rhs[index]] for index in range(width)]
    rank = 0
    for column in range(width):
        pivot = max(
            range(rank, width), key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) <= tolerance:
            continue
        augmented[rank], augmented[pivot] = augmented[pivot], augmented[rank]
        divisor = augmented[rank][column]
        augmented[rank] = [value / divisor for value in augmented[rank]]
        for row in range(width):
            if row == rank:
                continue
            factor = augmented[row][column]
            if abs(factor) <= tolerance:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[rank],
                )
            ]
        rank += 1
    if rank != width:
        return rank, tuple()
    solution = [0.0] * width
    for row in range(width):
        pivot_columns = [
            column for column in range(width)
            if abs(augmented[row][column] - 1.0) <= 1e-9
            and all(
                abs(augmented[other][column]) <= 1e-9
                for other in range(width) if other != row
            )
        ]
        if len(pivot_columns) != 1:
            raise AnchoredShapeError("shape solve pivot reconstruction failed")
        solution[pivot_columns[0]] = augmented[row][-1]
    return rank, tuple(
        solution[index] / column_scales[index] for index in range(width)
    )


def fit_centered_log_shape(
    observations: Sequence[LogShapeObservation],
    features_by_key: Mapping[str, Sequence[float]],
    *,
    reference_key: str | None = None,
) -> SharedLogShape:
    """Fit log shape after removing each panel unit's own log-cost level.

    A caller may declare unmeasured target keys, provided the observed panel
    identifies every feature column. Feature and observation iteration order
    is retained to preserve the established AURA fitter's arithmetic.
    """
    if not observations or not features_by_key:
        raise AnchoredShapeError("shape input is empty")
    features = {
        _label(key, "feature key"): tuple(_finite(value, "shape feature") for value in row)
        for key, row in features_by_key.items()
    }
    widths = {len(row) for row in features.values()}
    if len(widths) != 1 or not next(iter(widths)):
        raise AnchoredShapeError("shape feature width is invalid")
    width = next(iter(widths))
    reference = min(features) if reference_key is None else reference_key
    if reference not in features:
        raise AnchoredShapeError("unknown shape reference key")
    by_unit: dict[str, list[tuple[tuple[float, ...], float]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    duplicate = False
    for observation in observations:
        if not isinstance(observation, LogShapeObservation):
            raise AnchoredShapeError("shape observation type is invalid")
        pair = observation.unit, observation.key
        if pair in seen:
            duplicate = True
        seen.add(pair)
        if observation.key not in features:
            raise AnchoredShapeError("unknown shape observation key")
        by_unit[observation.unit].append((features[observation.key], math.log10(observation.value)))
    x_rows: list[tuple[float, ...]] = []
    y_rows: list[float] = []
    try:
        for unit, rows in sorted(by_unit.items()):
            if len(rows) < 2:
                raise AnchoredShapeError(f"panel unit {unit!r} has fewer than two rungs")
            feature_means = tuple(math.fsum(row[index] for row, _ in rows) / len(rows)
                                  for index in range(width))
            value_mean = math.fsum(value for _, value in rows) / len(rows)
            for row, value in rows:
                x_rows.append(tuple(row[index] - feature_means[index] for index in range(width)))
                y_rows.append(value - value_mean)
        rank, coefficients = rank_and_solve(x_rows, y_rows)
        if rank != width:
            raise AnchoredShapeError(f"panel design rank is {rank} of {width}")
        if duplicate:
            raise AnchoredShapeError("duplicate unit/key shape observation")
        log_shape = {
            key: math.fsum(coefficient * (feature - features[reference][index])
                           for index, (coefficient, feature) in enumerate(zip(coefficients, row)))
            for key, row in features.items()
        }
    except (OverflowError, ZeroDivisionError) as exc:
        raise AnchoredShapeError("shape design arithmetic is not representable") from exc
    return SharedLogShape(coefficients, reference, log_shape, rank, len(by_unit), len(observations))


def fit_anchor_correction(
    shape: SharedLogShape,
    anchors: Mapping[str, float],
    coordinates: Mapping[str, float],
) -> AnchorCorrection:
    """Restore level from one anchor, or fit level and tilt from two or more.

    The coordinate map declares the entire supported target domain. Center
    and scale it before solving, so coordinate origin and units cannot create
    an artificial rank failure. Beyond two anchors the affine residual is a
    least-squares proposal; each measured anchor still predicts exactly.
    """
    if not isinstance(shape, SharedLogShape):
        raise AnchoredShapeError("shared shape type is invalid")
    if not anchors:
        raise AnchoredShapeError("anchor input is empty")
    if set(coordinates) != set(shape.log_shape_by_key):
        raise AnchoredShapeError("coordinate domain differs from shared shape")
    coords = {key: _finite(value, "coordinate") for key, value in coordinates.items()}
    if len(set(coords.values())) != len(coords):
        raise AnchoredShapeError("duplicate coordinates do not identify distinct rungs")
    values: dict[str, float] = {}
    for key, value in anchors.items():
        if key not in coords:
            raise AnchoredShapeError("unknown anchor key")
        values[key] = _finite(value, "anchor value", positive=True)
    lower, upper = min(coords.values()), max(coords.values())
    center = lower/2.0 + upper/2.0
    scale = max(abs(lower - center), abs(upper - center)) or 1.0
    normalized = {key: (value - center)/scale for key, value in coords.items()}
    if any(not math.isfinite(value) for value in normalized.values()):
        raise AnchoredShapeError("coordinate normalization is not representable")
    if len(set(normalized.values())) != len(coords):
        raise AnchoredShapeError("coordinate normalization loses distinct rungs")
    xs = [normalized[key] for key in values]
    residuals = [_finite(math.log10(value) - shape.log_shape_by_key[key], "anchor residual")
                 for key, value in values.items()]
    if len(values) == 1:
        intercept, slope = residuals[0], 0.0
    else:
        try:
            x_mean = math.fsum(xs)/len(xs)
            y_mean = math.fsum(residuals)/len(residuals)
            rank, coefficients = rank_and_solve([(x-x_mean,) for x in xs],
                                                [y-y_mean for y in residuals])
        except (OverflowError, ZeroDivisionError) as exc:
            raise AnchoredShapeError("anchor correction arithmetic is not representable") from exc
        if rank != 1:
            raise AnchoredShapeError("anchor correction design rank is insufficient")
        slope = coefficients[0]
        intercept = y_mean - slope*x_mean
    return AnchorCorrection(shape, values, coords, center, scale,
                            _finite(intercept, "residual intercept"),
                            _finite(slope, "residual slope"))


def predict_anchored(correction: AnchorCorrection, key: str) -> float:
    """Predict one declared key, returning actual anchor values verbatim."""
    if not isinstance(correction, AnchorCorrection):
        raise AnchoredShapeError("anchor correction type is invalid")
    _label(key, "prediction key")
    if key not in correction.coordinates:
        raise AnchoredShapeError("unknown prediction key")
    if key in correction.anchors:
        return correction.anchors[key]
    coordinate = (correction.coordinates[key] - correction.center)/correction.scale
    try:
        log_value = math.fsum((correction.shape.log_shape_by_key[key],
                               correction.intercept, correction.slope*coordinate))
        value = 10.0**log_value
    except (OverflowError, ValueError) as exc:
        raise AnchoredShapeError("prediction is not representable as a positive finite float") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise AnchoredShapeError("prediction is not representable as a positive finite float")
    return value


def audit_anchored(
    correction: AnchorCorrection,
    observations: Mapping[str, float],
) -> AnchoredAudit:
    """Freeze untouched audit predictions and errors without fitting them."""
    if not observations:
        raise AnchoredShapeError("audit input is empty")
    if set(observations) & set(correction.anchors):
        raise AnchoredShapeError("audit key is also a fitted anchor")
    rows = []
    for key, value in sorted(observations.items()):
        measured = _finite(value, "audit value", positive=True)
        predicted = predict_anchored(correction, key)
        rows.append(AnchoredAuditRow(key, predicted, measured,
                                    abs(math.log10(predicted) - math.log10(measured))))
    return AnchoredAudit(tuple(rows))


__all__ = [
    "AnchoredShapeError", "EndpointCurvatureModel", "EndpointCurvatureCurve",
    "LogShapeObservation", "SharedLogShape", "AnchorCorrection",
    "AnchoredAuditRow", "AnchoredAudit", "fit_centered_log_shape", "fit_anchor_correction",
    "predict_anchored", "audit_anchored", "fit_endpoint_curvature",
]
