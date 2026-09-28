"""PACT candidate generator: the exact lower convex hull of (time, Δloss).

PACT (PQ #1584) prices every DP option with two numbers: its predicted Δloss
(``Candidate.predicted_dloss``) and the operator-sum time the shape-time table
gives it at one regime M (``shape_runtime_prices``). The exact nondominated
set of whole assignments in that plane is too large for the exact solver
(``solve_runtime_frontier`` refuses it at ``max_states`` on GLM-5.3's 132 x 3
problem), so this module generates only the vertices of its lower-left convex
hull.

Why the hull loses no pick
--------------------------
Every PACT selection rule maximises an affine function of (time, Δloss):
argmin Δloss, argmin time, and ``select_development_point``'s
chord-perpendicular distance in coordinates normalised by the two endpoints
(themselves hull vertices). The maximum of an affine function over a finite
point set is attained at a vertex of its convex hull, so the exact lower hull
contains every pick those rules can make. The one thing a hull cannot answer
is a rule that is NOT affine over the whole set, such as a separation test
between the top two points: that is then evaluated among hull vertices only
(PQ #1585 owns that consequence).

How it is generated (dichotomic parametric search)
--------------------------------------------------
1. Probe the two endpoints: minimum Δloss, and minimum time.
2. For an adjacent pair (A, B) with t_A > t_B and d_A < d_B, probe the
   weighted sum ``w_d·Δloss + w_t·time`` with ``w_d = t_A − t_B`` and
   ``w_t = d_B − d_A`` (the segment's normal, so A and B score the same).
3. A result strictly below the segment is a new vertex; recurse on both
   halves. Otherwise AB is a hull edge.

λ = w_t / w_d, the Δloss a millisecond buys along an edge, is recorded beside
every edge as a DIAGNOSTIC. It never enters a selection objective; it only
generates the candidates (CLAUDE.md §9: λ is a candidate generator only).

Each probe is exact, in exact arithmetic. An option's probe cost
``w_d·Δloss + w_t·time`` is computed as an exact rational (every float64 is
one), and sums never round. When the byte budget binds, the probe is one
single-budget multiple-choice knapsack, solved by
``exact_mckp.solve_exact_mckp``. It returns the minimum cost, then the fewest
bytes, then the lexically first assignment: the order
``allocator_solver.solve_runtime_frontier`` ranks its final frontier by, and
the same answer that solver gives whenever its float sums are exact. It is
not that solver. That solver keeps lexical order as a dominance coordinate in
every intermediate fold, because a float sum or a later peak can erase a
strict difference. Here both coordinates are exact sums, so the fold keeps
only the (bytes, cost) Pareto set. It also drops prefixes that an exact LP
bound proves cannot reach the incumbent. On GLM-5.3's 132 units x 7 E4M3 rungs
the order coordinate had held 458,239 states at unit 47 before a refusal
(``exact_mckp`` has the numbers and the proof). ``max_states`` and
``max_transitions`` bound that search the same way and refuse rather than
truncate. When the budget provably cannot bind -- every unit's largest option
fits together -- the constraint is vacuous, no term couples two units, and the
probe's minimum is the sum of per-unit minima: each unit's exact minimiser,
ties to the smallest format name. Bytes do not enter that key, since no
assignment can exceed the budget; this is the tie rule the slack probe
always had (it used to flatten every unit's bytes to its largest option).

"Strictly below" is decided against the float resolution of the sums being
compared, never a chosen epsilon: a sequential float64 sum of n terms is
within ``(n − 1)·u·Σ|terms|`` of exact (Higham), with ``u = 2⁻⁵³``; each term
here is a product (one more rounding), so the resolution is
``(n + 1)·u·(S_probe + S_segment)`` where ``S = Σ|w_d·d_u| + Σ|w_t·t_u|``.
Point coordinates are summed with ``math.fsum``.

Limits, stated
--------------
* A time ceiling is a REPORT bound here. The constrained set's own boundary
  vertex (minimum Δloss subject to time ≤ C) needs time as a live solver
  coordinate and is not generated.
* A separation test between the top two candidates sees hull vertices, not
  every point of the exact frontier (see above).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping, Sequence

from .allocator_solver import Candidate, RuntimeFrontierLimitError
from .exact_mckp import exact_integer_costs, solve_exact_mckp

SCHEMA = "prismaquant.pact_hull.v1"
CANDIDATE_GENERATOR = "lower_convex_hull_dichotomic"
#: Unit roundoff of IEEE-754 binary64, the dtype every cost and time here is.
FLOAT64_UNIT_ROUNDOFF = 2.0 ** -53
#: The exact probe's bounds, the same defaults ``solve_runtime_frontier``
#: has; a caller may raise them (``--pact-max-states`` /
#: ``--pact-max-transitions``).
DEFAULT_MAX_STATES = 100_000
DEFAULT_MAX_TRANSITIONS = 8_000_000

BYTE_AXIS_SLACK = "slack_flattened"
BYTE_AXIS_LIVE = "live"


class PactHullError(ValueError):
    """A hull request this module refuses."""


@dataclass(frozen=True)
class HullPoint:
    """One whole assignment, with its exactly summed coordinates."""

    assignment: Mapping[str, str]
    predicted_dloss: float
    time_ms: float
    memory_bytes: int

    def key(self) -> tuple:
        return tuple(sorted(self.assignment.items()))


@dataclass(frozen=True)
class HullProbe:
    """What one solver call was asked and what it answered."""

    index: int
    weights: tuple[float, float]
    lambda_dloss_per_ms: float | None
    segment: tuple[int, int] | None
    result: int
    value: float | None
    segment_value: float | None
    resolution: float | None
    strictly_below: bool | None
    byte_axis: str
    solver: Mapping


@dataclass(frozen=True)
class LowerHull:
    """The hull vertices, accuracy end first, and every probe that built them."""

    vertices: tuple[HullPoint, ...]
    #: ``edges[i]`` joins ``vertices[i]`` and ``vertices[i + 1]``: its λ (Δloss
    #: per ms) is a diagnostic.
    edge_lambdas: tuple[float, ...]
    probes: tuple[HullProbe, ...]
    points: tuple[HullPoint, ...]
    byte_axis: str
    max_memory_bytes: int
    slack_bound_bytes: int
    n_units: int
    dominated_probe_points: tuple[int, ...] = field(default=())

    def finding_probe(self, vertex: HullPoint) -> HullProbe:
        """The first probe whose answer was this vertex; its weights re-derive it."""
        pid = next(i for i, point in enumerate(self.points) if point.key() == vertex.key())
        return next(p for p in self.probes if p.result == pid)

    def as_dict(self) -> dict:
        index = {point.key(): i for i, point in enumerate(self.points)}
        return {
            "schema": SCHEMA,
            "candidate_generator": CANDIDATE_GENERATOR,
            "n_units": self.n_units,
            "byte_axis": self.byte_axis,
            "max_memory_bytes": self.max_memory_bytes,
            "slack_bound_bytes": self.slack_bound_bytes,
            "probe_count": len(self.probes),
            "vertex_point_ids": [index[v.key()] for v in self.vertices],
            "vertex_probe_ids": [self.finding_probe(v).index for v in self.vertices],
            "edge_lambda_dloss_per_ms": list(self.edge_lambdas),
            "edge_lambda_role": "diagnostic only; never a selection objective",
            "resolution_rule": "(n_units + 1) * 2**-53 * (S_probe + S_segment), "
                               "S = sum |w_d*d_u| + sum |w_t*t_u| (float64 sum bound)",
            "probes": [{
                "index": p.index, "weights": list(p.weights),
                "lambda_dloss_per_ms": p.lambda_dloss_per_ms,
                "segment_point_ids": None if p.segment is None else list(p.segment),
                "result_point_id": p.result, "value": p.value,
                "segment_value": p.segment_value, "resolution": p.resolution,
                "strictly_below": p.strictly_below, "byte_axis": p.byte_axis,
                "solver": dict(p.solver)} for p in self.probes],
            "dominated_probe_point_ids": list(self.dominated_probe_points),
        }


def _check_inputs(candidates: Mapping[str, Sequence[Candidate]],
                  time_ms: Mapping[tuple[str, str], float]) -> None:
    if not candidates:
        raise PactHullError("no units to allocate")
    for unit, options in candidates.items():
        if not options:
            raise PactHullError(f"{unit}: no priced option")
        for option in options:
            value = time_ms.get((unit, option.fmt))
            if value is None:
                raise PactHullError(f"{unit}@{option.fmt}: no time; an unpriced option must be "
                                    "absent from the candidate set, never priced here")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise PactHullError(f"{unit}@{option.fmt}: time must be finite and nonnegative")
            if not math.isfinite(float(option.predicted_dloss)):
                raise PactHullError(f"{unit}@{option.fmt}: predicted_dloss must be finite")


def _byte_axis(candidates, max_memory_bytes: int) -> tuple[str, dict[str, int], int]:
    unit_max = {u: max(int(c.memory_bytes) for c in candidates[u]) for u in candidates}
    slack_bound = sum(unit_max.values())
    axis = BYTE_AXIS_SLACK if slack_bound <= int(max_memory_bytes) else BYTE_AXIS_LIVE
    return axis, unit_max, slack_bound


def _exact_option_costs(candidates, time_ms, w_d: float, w_t: float) -> dict:
    """Every option's ``w_d·Δloss + w_t·time`` as an exact integer, one scale.

    Weights, Δloss and time are float64 values, so each product and each sum
    is an exact dyadic rational; ``exact_integer_costs`` scales all of them by
    one power of two. Nothing is rounded, so no tie and no strict difference
    depends on float rounding.
    """
    wd, wt = Fraction(w_d), Fraction(w_t)
    scaled, _denominator = exact_integer_costs({
        (unit, c.fmt): wd * Fraction(float(c.predicted_dloss))
        + wt * Fraction(float(time_ms[(unit, c.fmt)]))
        for unit in candidates for c in candidates[unit]})
    return scaled


def _solve_weighted(candidates, time_ms, w_d: float, w_t: float, *, byte_axis: str,
                    max_memory_bytes: int, max_states: int, max_transitions: int,
                    hints: Sequence[Mapping[str, str]] = ()) -> tuple[dict[str, str], dict]:
    """One exact probe: min Σ(w_d·Δloss + w_t·time) subject to bytes ≤ budget.

    Slack byte axis: the constraint is vacuous (every unit's largest option
    fits together), so nothing couples two units and the minimum of the sum is
    the sum of per-unit minima. Each unit's answer is its exact minimiser, ties
    to the smallest format name; bytes do not enter the key. Live byte axis:
    ``solve_exact_mckp``, whose ``max_states``/``max_transitions`` refusal
    stands; its key is minimum cost, then fewest bytes, then lexical order.
    One problem has one byte axis, so every probe of a hull uses one key.
    ``hints`` are feasible assignments already found (a probed segment's two
    ends); they only tighten the live probe's bound and never change its answer,
    so a replay without them re-derives the same assignment.
    """
    cost = _exact_option_costs(candidates, time_ms, w_d, w_t)
    if byte_axis == BYTE_AXIS_SLACK:
        assignment = {}
        for unit in sorted(candidates):
            best = min(candidates[unit], key=lambda c: (cost[(unit, c.fmt)], c.fmt))
            assignment[unit] = best.fmt
        return assignment, {"method": "separable_per_unit_minimum"}
    units = {unit: [(c.fmt, cost[(unit, c.fmt)], int(c.memory_bytes))
                    for c in candidates[unit]] for unit in sorted(candidates)}
    diag: dict = {}
    try:
        assignment = solve_exact_mckp(units, max_bytes=int(max_memory_bytes),
                                      max_states=max_states,
                                      max_transitions=max_transitions, diagnostics=diag,
                                      incumbent_hints=hints)
    except RuntimeFrontierLimitError as exc:
        # The refusal stands; its measured growth (per-unit frontier sizes up
        # to the refused unit) travels with it so a caller can report it.
        exc.diagnostics = diag
        raise
    if assignment is None:
        raise PactHullError(
            f"no assignment fits max_memory_bytes={int(max_memory_bytes)} "
            f"(byte axis {byte_axis})")
    return assignment, diag


def probe_assignment(candidates: Mapping[str, Sequence[Candidate]],
                     time_ms: Mapping[tuple[str, str], float], weights: Sequence[float], *,
                     max_memory_bytes: int, max_states: int = DEFAULT_MAX_STATES,
                     max_transitions: int = DEFAULT_MAX_TRANSITIONS) -> dict[str, str]:
    """Re-run ONE recorded probe: the assignment those weights select.

    A replay re-derives a hull vertex from the probe that found it rather than
    rebuilding the whole hull; the byte axis is decided from the same inputs by
    the same rule, so the same inputs return the same assignment.
    """
    _check_inputs(candidates, time_ms)
    w_d, w_t = (float(w) for w in weights)
    axis, _unit_max, _ = _byte_axis(candidates, max_memory_bytes)
    assignment, _diag = _solve_weighted(
        candidates, time_ms, w_d, w_t, byte_axis=axis,
        max_memory_bytes=max_memory_bytes, max_states=max_states,
        max_transitions=max_transitions)
    return assignment


def dichotomic_lower_hull(candidates: Mapping[str, Sequence[Candidate]],
                      time_ms: Mapping[tuple[str, str], float], *,
                      max_memory_bytes: int, max_states: int = DEFAULT_MAX_STATES,
                      max_transitions: int = DEFAULT_MAX_TRANSITIONS) -> LowerHull:
    """Every vertex of the lower-left convex hull of feasible (time, Δloss).

    ``candidates`` are the priced options of each serving unit (unpriced ones
    already removed, ``ShapePricing.time_candidates``); ``time_ms`` is each
    option's operator time. Raises :class:`PactHullError` when no assignment
    fits ``max_memory_bytes``, and lets the solver's own
    ``RuntimeFrontierLimitError`` through unchanged.
    """
    _check_inputs(candidates, time_ms)
    units = sorted(candidates)
    n = len(units)
    byte_axis, _unit_max, slack_bound = _byte_axis(candidates, max_memory_bytes)
    options = {u: {c.fmt: c for c in candidates[u]} for u in units}

    points: list[HullPoint] = []
    point_index: dict[tuple, int] = {}
    probes: list[HullProbe] = []

    def terms(point: HullPoint, w_d: float, w_t: float) -> tuple[list[float], float]:
        values = []
        for unit, fmt in point.assignment.items():
            values.append(w_d * float(options[unit][fmt].predicted_dloss))
            values.append(w_t * float(time_ms[(unit, fmt)]))
        return values, math.fsum(abs(v) for v in values)

    def probe(w_d: float, w_t: float, segment=None) -> int:
        hints = () if segment is None else tuple(points[i].assignment for i in segment)
        assignment, diag = _solve_weighted(
            candidates, time_ms, w_d, w_t, byte_axis=byte_axis,
            max_memory_bytes=max_memory_bytes, max_states=max_states,
            max_transitions=max_transitions, hints=hints)
        point = HullPoint(
            assignment=assignment,
            predicted_dloss=math.fsum(float(options[u][f].predicted_dloss)
                                      for u, f in assignment.items()),
            time_ms=math.fsum(float(time_ms[(u, f)]) for u, f in assignment.items()),
            memory_bytes=sum(int(options[u][f].memory_bytes) for u, f in assignment.items()))
        key = point.key()
        if key not in point_index:
            point_index[key] = len(points)
            points.append(point)
        pid = point_index[key]
        solver = {"method": diag["method"], "frontier_size": diag.get("frontier_size"),
                  "max_fold_size": max(diag.get("frontier_sizes") or [0]),
                  "fold_states": sum(diag.get("frontier_sizes") or []),
                  "pruned_by_bound": diag.get("pruned_by_bound"),
                  "pruned_infeasible": diag.get("pruned_infeasible"),
                  "transitions": diag.get("transitions")}
        value = segment_value = resolution = below = None
        lam = None
        if segment is not None:
            a = points[segment[0]]
            p_terms, p_abs = terms(points[pid], w_d, w_t)
            a_terms, a_abs = terms(a, w_d, w_t)
            value, segment_value = math.fsum(p_terms), math.fsum(a_terms)
            resolution = (n + 1) * FLOAT64_UNIT_ROUNDOFF * (p_abs + a_abs)
            below = value < segment_value - resolution
            lam = w_t / w_d
        probes.append(HullProbe(len(probes), (w_d, w_t), lam, segment, pid, value,
                                segment_value, resolution, below, byte_axis, solver))
        return pid

    def dominates(p: HullPoint, q: HullPoint) -> bool:
        return (p.time_ms <= q.time_ms and p.predicted_dloss <= q.predicted_dloss
                and (p.time_ms < q.time_ms or p.predicted_dloss < q.predicted_dloss))

    accurate = probe(1.0, 0.0)
    fast = probe(0.0, 1.0)
    stack = [(accurate, fast)]
    while stack:
        ia, ib = stack.pop()
        a, b = points[ia], points[ib]
        # A segment is probed only between two points neither of which
        # weakly dominates the other; otherwise it has no interior normal.
        if not (a.time_ms > b.time_ms and a.predicted_dloss < b.predicted_dloss):
            continue
        w_d = a.time_ms - b.time_ms
        w_t = b.predicted_dloss - a.predicted_dloss
        before = len(probes)
        ip = probe(w_d, w_t, segment=(ia, ib))
        if not probes[before].strictly_below or ip in (ia, ib):
            continue
        stack.append((ip, ib))
        stack.append((ia, ip))

    candidates_set = list(range(len(points)))
    dominated = tuple(i for i in candidates_set
                      if any(dominates(points[j], points[i]) for j in candidates_set if j != i))
    kept = [i for i in candidates_set if i not in dominated]
    # The surviving probe points; a collinear or above-segment probe answer is
    # a point the search saw, not a vertex, so keep only strict-hull members:
    # a point is a vertex when it is strictly below the chord of its two
    # neighbours in time order, at the same float resolution.
    kept.sort(key=lambda i: (-points[i].time_ms, points[i].predicted_dloss))
    vertices = list(kept)
    changed = True
    while changed and len(vertices) > 2:
        changed = False
        for k in range(1, len(vertices) - 1):
            a, p, b = (points[vertices[k - 1]], points[vertices[k]], points[vertices[k + 1]])
            w_d, w_t = a.time_ms - b.time_ms, b.predicted_dloss - a.predicted_dloss
            p_terms, p_abs = terms(p, w_d, w_t)
            a_terms, a_abs = terms(a, w_d, w_t)
            if not (math.fsum(p_terms) < math.fsum(a_terms)
                    - (n + 1) * FLOAT64_UNIT_ROUNDOFF * (p_abs + a_abs)):
                del vertices[k]
                changed = True
                break
    hull = tuple(points[i] for i in vertices)
    lambdas = tuple((hull[k + 1].predicted_dloss - hull[k].predicted_dloss)
                    / (hull[k].time_ms - hull[k + 1].time_ms) for k in range(len(hull) - 1))
    return LowerHull(vertices=hull, edge_lambdas=lambdas, probes=tuple(probes),
                     points=tuple(points), byte_axis=byte_axis,
                     max_memory_bytes=int(max_memory_bytes), slack_bound_bytes=slack_bound,
                     n_units=n, dominated_probe_points=dominated)
