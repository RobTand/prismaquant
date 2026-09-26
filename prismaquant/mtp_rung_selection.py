"""Throughput-optimal draft-rung selector for MTP / spec-decode modules — canon.

Normative spec: ``docs/design/mtp_rung_selection.md`` (Robert, 2026-07-20). A draft can
NEVER change outputs (rejection sampling reproduces the target distribution
exactly), so this selector optimises **throughput only** — there is no quality
gate on the draft. The reference integration is
``scripts/build_hy3_mtp_cb_inputs.py --rung-select auto``.

Objective (per spec-decode cycle, ``k`` speculative tokens; doc §1):

    T(b) = (1 + Σ_{i=1..k} Π_{j<=i} a_j(b)) / (t + k·d(b))

with per-position acceptance approximated as ``a(b)^i`` for position ``i`` (see
``_throughput``). Cost side is exact: ``d(b) = d0 + c·b``. Acceptance side is the
Fisher/Pinsker shape ``a(b) = a_inf − β·sqrt(E(b))`` with ``E(b) = Σ_i h_i·MSE_i``
per rung, and ``(a_inf, β)`` **fit from served acceptance measurements**.

This module is pure-Python (stdlib + optional scipy for the Lambert-W
cross-check); it never imports torch, so it stays importable in the CPU driver.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Mapping, Optional

_LN2 = math.log(2.0)
# Doc §3.6: degenerate iff the cost side varies < 1% across the menu. This is
# the spec constant, not a tunable heuristic (exposed only for testability).
_DEGENERATE_FRACTION = 0.01


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RungPoint:
    """One draft-menu candidate: an aggregate draft encoding.

    ``bits`` is the params-weighted draft bpw ``b``; ``resident_bytes`` is the
    draft's serve-time footprint at this rung (feeds the memory gate);
    ``E`` = ``Σ_i h_i·MSE_i`` is the calibrated error proxy that drives ``a(b)``.
    """

    name: str
    bits: float
    resident_bytes: int
    E: float

    def __post_init__(self) -> None:
        if self.bits <= 0:
            raise ValueError(f"RungPoint {self.name}: bits must be > 0")
        if self.resident_bytes < 0:
            raise ValueError(f"RungPoint {self.name}: resident_bytes < 0")
        if self.E < 0 or not math.isfinite(self.E):
            raise ValueError(f"RungPoint {self.name}: E must be finite and >= 0")


@dataclass(frozen=True)
class ServeConstants:
    """Cost-side constants (doc §2). All times in ms.

    ``t_ms`` = target verify-step time; ``d0_ms`` = rung-independent drafter
    overhead (shared lm_head read + attention + KV + host/launch — host-dominated
    on an eager drafter); ``c_ms_per_bit`` = drafter time per bit/weight so that
    ``d(b) = d0_ms + c_ms_per_bit·b``.
    """

    t_ms: float
    d0_ms: float
    c_ms_per_bit: float

    def __post_init__(self) -> None:
        if self.t_ms <= 0:
            raise ValueError("ServeConstants: t_ms must be > 0")
        if self.d0_ms < 0 or self.c_ms_per_bit < 0:
            raise ValueError("ServeConstants: d0_ms and c_ms_per_bit must be >= 0")

    def d(self, bits: float) -> float:
        """One drafter forward at ``bits`` bits/weight (ms)."""
        return self.d0_ms + self.c_ms_per_bit * bits


@dataclass(frozen=True)
class AcceptancePoint:
    """A served acceptance measurement at one rung.

    Identify the rung by ``rung_name`` (matched against ``RungPoint.name`` and
    ``E_by_bits`` keys) OR by ``bits``; at least one is required. The fit maps a
    point to its ``E`` via ``key = bits if bits is not None else rung_name``.
    """

    measured_acceptance: float
    rung_name: Optional[str] = None
    bits: Optional[float] = None

    def __post_init__(self) -> None:
        if self.rung_name is None and self.bits is None:
            raise ValueError("AcceptancePoint: give rung_name or bits")
        if not (0.0 <= self.measured_acceptance <= 1.0):
            raise ValueError(
                f"AcceptancePoint: acceptance {self.measured_acceptance} "
                "outside [0, 1]")

    @property
    def key(self):
        """Resolution key into ``E_by_bits``: ``bits`` if set, else ``rung_name``."""
        return self.bits if self.bits is not None else self.rung_name


def group_product_menu(groups: Mapping, rows: Mapping, *, params: Mapping):
    """Draft rungs from per-unit rows: one uniform rung per declared group.

    ``groups`` maps a group name to its unit names; every unit of a group takes
    the same rung (a routed stack is uniform per layer; a fused pair shares one
    scheme). ``rows[unit][rung]`` is ``(E, resident_bytes)``, and ``params``
    gives each unit's parameter count. A menu point is one rung per group; its
    ``E`` and ``resident_bytes`` are sums over every unit, and ``bits`` is the
    params-weighted ``8 * bytes / params``.

    A rung some unit of a group lacks cannot be that group's uniform rung, so
    it is left out of the menu and reported in ``incomplete``
    (``{group: {rung: [units lacking it]}}``). Returns ``(menu, incomplete)``.
    """
    import itertools

    names = list(groups)
    if not names or any(not groups[g] for g in names):
        raise ValueError("group_product_menu: every group needs at least one unit")
    seen = [unit for g in names for unit in groups[g]]
    if len(seen) != len(set(seen)):
        raise ValueError("group_product_menu: a unit belongs to two groups")
    offered, incomplete = {}, {}
    for g in names:
        units = list(groups[g])
        rungs = sorted({rung for unit in units for rung in rows.get(unit, {})})
        offered[g] = [r for r in rungs if all(r in rows.get(u, {}) for u in units)]
        missing = {r: sorted(u for u in units if r not in rows.get(u, {}))
                   for r in rungs if r not in offered[g]}
        if missing:
            incomplete[g] = missing
        if not offered[g]:
            raise ValueError(f"group_product_menu: group {g!r} has no rung priced for every unit")
    total_params = sum(int(params[u]) for u in seen)
    if total_params <= 0:
        raise ValueError("group_product_menu: parameter count must be positive")
    menu = []
    for combo in itertools.product(*(offered[g] for g in names)):
        E = resident = 0
        for g, rung in zip(names, combo):
            for unit in groups[g]:
                unit_E, unit_bytes = rows[unit][rung]
                E += float(unit_E)
                resident += int(unit_bytes)
        menu.append(RungPoint(name="|".join(f"{g}={r}" for g, r in zip(names, combo)),
                              bits=8.0 * resident / total_params, resident_bytes=resident, E=E))
    return menu, incomplete


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
@dataclass
class SelectionResult:
    rung: RungPoint
    regime: str  # "degenerate" | "interior"
    per_rung_T: dict
    provenance: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Acceptance fit:  a(b) = a_inf − β·sqrt(E(b))   (linear in sqrt(E); doc §2)
# --------------------------------------------------------------------------- #
def fit_acceptance(points, E_by_bits: Mapping):
    """Fit ``(a_inf, β)`` from served acceptance measurements.

    Returns ``(a_inf, beta, fit_mode)`` where ``fit_mode`` is:

      * ``"least_squares"`` — ≥2 points with distinct ``sqrt(E)``: ordinary
        least-squares line ``a = a_inf − β·x`` over ``x = sqrt(E)``.
      * ``"single_point"`` — exactly 1 point, OR ≥2 points that all share one
        ``E`` (no fidelity spread → no slope). ``a_inf`` is taken from the
        highest-fidelity (lowest-E) measured rung; ``beta`` is ``None`` (there is
        no slope to estimate — the doc forbids assuming one). The selector must
        then fall back to the degenerate branch.
      * ``"no_data"`` — 0 points: ``a_inf = beta = None``.

    ``E_by_bits`` maps each point's ``key`` (its ``bits`` if set, else
    ``rung_name``) to ``E(b)``; a missing key is a hard error (never fabricate E).
    """
    pts = list(points)
    if not pts:
        return None, None, "no_data"

    xs, ys = [], []
    for p in pts:
        if p.key not in E_by_bits:
            raise KeyError(
                f"fit_acceptance: no E for acceptance point key {p.key!r}")
        E = float(E_by_bits[p.key])
        if E < 0 or not math.isfinite(E):
            raise ValueError(f"fit_acceptance: bad E={E} for key {p.key!r}")
        xs.append(math.sqrt(E))
        ys.append(float(p.measured_acceptance))

    n = len(pts)
    x_span = max(xs) - min(xs)
    # Single point, or a degenerate cluster with no fidelity spread → no slope.
    if n == 1 or x_span <= 1e-12:
        # a_inf := acceptance at the highest-fidelity (lowest-E ⇒ lowest-x) rung;
        # average ties so a repeated-rung calibration is not order-dependent.
        x_min = min(xs)
        best = [y for x, y in zip(xs, ys) if abs(x - x_min) <= 1e-12]
        return sum(best) / len(best), None, "single_point"

    # Ordinary least squares for the line a = intercept + slope·x (β = −slope).
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    return intercept, -slope, "least_squares"


# --------------------------------------------------------------------------- #
# Throughput and the continuous cross-check
# --------------------------------------------------------------------------- #
def _acceptance(a_inf, beta, E: float) -> float:
    """a(b) = a_inf − β·sqrt(E), clamped to the probability domain [0, 1].

    Clamping enforces that acceptance is a probability (a mathematical bound, not
    a heuristic band-aid on the optimiser); ``provenance['a_clamped']`` records
    whether any menu rung's fitted value fell outside [0, 1].
    """
    a = a_inf - (beta or 0.0) * math.sqrt(max(E, 0.0))
    return min(1.0, max(0.0, a))


def _throughput(a: float, d: float, t: float, k: int) -> float:
    """T = (1 + Σ_{i=1..k} a^i) / (t + k·d).

    Per-position cumulative acceptance ``Π_{j<=i} a_j`` is approximated by
    ``a^i`` (all positions share the fitted ``a(b)``; doc §1). k=1 reduces to
    ``(1 + a) / (t + d)``.
    """
    num, term = 1.0, 1.0
    for _ in range(k):
        term *= a
        num += term
    return num / (t + k * d)


def _continuous_bstar_fixed_point(a_inf, beta, const: ServeConstants,
                                  b0: float, iters: int = 3):
    """Lambert-W continuous optimum via fixed-point iteration (scipy-free).

    Solves the k=1 stationarity condition (doc §3.4)
        2^{-b}·β·[ln2·(t+d0+c·b) + c] = c·(1 + a_inf)
    rearranged to b = (1/ln2)·ln( β·[ln2·(t+d0+c·b)+c] / (c·(1+a_inf)) ). The
    RHS depends on b only through a log, so 2–3 iterations converge. Theory /
    sanity cross-check on the discrete argmax — never the shipped selection.
    Returns None when the curve has no interior optimum (β, c, or a_inf unusable).
    """
    c = const.c_ms_per_bit
    if beta is None or beta <= 0 or c <= 0 or a_inf is None or (1.0 + a_inf) <= 0:
        return None
    b = b0
    for _ in range(max(1, iters)):
        bracket = _LN2 * (const.t_ms + const.d0_ms + c * b) + c
        arg = beta * bracket / (c * (1.0 + a_inf))
        if arg <= 0 or not math.isfinite(arg):
            return None
        b = math.log(arg) / _LN2
        if not math.isfinite(b):
            return None
    return b


def _continuous_bstar_lambertw(a_inf, beta, const: ServeConstants):
    """Closed form via scipy's Lambert-W (the W_{-1} branch), when computable.

    Independent cross-check on the fixed point. Returns ``(value, status)``
    so provenance can record WHICH solver answered instead of collapsing
    every failure into a bare ``None``:

      * ``("scipy_lambertw", b*)`` — scipy evaluated W_{-1} directly;
      * ``("log_space_continuation", b*)`` — the argument is smaller than
        float64 can represent (``ln M < -700``, i.e. the d0-dominated regime
        ``(t+d0)/c >~ 1023``); the branch is continued exactly by solving
        ``s - ln(s) = -ln(M)`` (Newton; unique root s > 1, convex, converged
        to machine precision), which is the same equation W_{-1} solves in
        the only regime where its argument underflows;
      * ``(None, "no_real_solution")`` — ``M >= 1/e`` or non-finite: no real
        point on the -1 branch;
      * ``(None, "invalid_fit_constants")`` / ``(None, "scipy_absent")``.

    Audit 2026-08-21: the old form computed ``M = (1+a_inf)/(beta*exp(g/c))``
    directly. For eager-drafter constants (Hy3: t=76 ms, d0=50 ms,
    c=0.1 ms/bit -> (t+d0)/c = 1260) ``math.exp(g/c)`` raised OverflowError
    and the helper returned None — silently self-disabling on roughly the
    top half of the plausible (t+d0)/c range even though a real W_{-1}
    exists there. Rescaling into ``log_M = ln((1+a_inf)/beta) - g/c``
    removes the overflow entirely: the argument handed to scipy is
    ``-exp(log_M)``, representable whenever ``log_M > -700``, and continued
    analytically below that.
    """
    try:
        from scipy.special import lambertw  # noqa: PLC0415
    except Exception:
        return None, "scipy_absent"
    c = const.c_ms_per_bit
    if beta is None or beta <= 0 or c <= 0 or a_inf is None or (1.0 + a_inf) <= 0:
        return None, "invalid_fit_constants"
    g_over_c = (_LN2 * (const.t_ms + const.d0_ms) + c) / c
    try:
        log_M = math.log((1.0 + a_inf) / beta) - g_over_c
    except (OverflowError, ValueError):
        return None, "no_real_solution"
    if not math.isfinite(log_M) or log_M >= -1.0:
        return None, "no_real_solution"  # -M outside [-1/e, 0): no real W_{-1}
    if log_M >= -700.0:
        w = lambertw(-math.exp(log_M), k=-1)
        if abs(w.imag) > 1e-9 or not math.isfinite(w.real):
            return None, "no_real_solution"
        s = -w.real
        method = "scipy_lambertw"
    else:
        # s := -W_{-1}(-M) >= 1 satisfies s - ln(s) = L with L = -ln(M).
        L = -log_M
        s = L + math.log(L)
        for _ in range(60):
            step = (s - math.log(s) - L) * s / (s - 1.0)   # Newton
            s -= step
            if not (s > 1.0) or not math.isfinite(s):
                return None, "continuation_did_not_converge"
            if abs(step) <= 1e-14 * s:
                break
        else:
            return None, "continuation_did_not_converge"
        method = "log_space_continuation"
    return s / _LN2 - (const.t_ms + const.d0_ms) / c - 1.0 / _LN2, method


# --------------------------------------------------------------------------- #
# The selector (doc §3)
# --------------------------------------------------------------------------- #
def select_rung(menu, constants: ServeConstants, accept_points,
                mem_budget_bytes: int, k: int = 1,
                h_source: str = "unknown",
                degenerate_fraction: float = _DEGENERATE_FRACTION,
                ) -> SelectionResult:
    """Pick the throughput-optimal draft rung (doc §3).

    Order of operations:
      1. **Memory gate first** (doc §3.5): keep rungs with
         ``resident_bytes <= mem_budget_bytes``. NOTE: the doc gate is
         ``weights + draft + profiling-peak + 3 GiB margin <= usable pool`` —
         everything except the draft's own resident bytes (weights, profiling
         peak, **and the 3 GiB margin**) is the CALLER's responsibility to net
         out of the usable pool before passing ``mem_budget_bytes``. This
         function compares only the draft footprint against that net budget.
      2. Fit ``(a_inf, β)`` from the served acceptance points.
      3. **Degenerate-regime branch** (doc §3.6): if the cost side varies less
         than ``degenerate_fraction`` of the cycle (``k·c·Δb`` vs ``t + k·d``),
         OR the fit has no usable slope (0/1 acceptance point), the argmax
         provably lands on the acceptance-max rung — pick the **highest-fidelity
         (lowest-E) rung passing the gate** and record ``regime='degenerate'``.
      4. Else **discrete argmax** of ``T(b)`` over the passing rungs (the menu is
         discrete; the continuous Lambert-W optimum lives in provenance as a
         cross-check only).

    Raises ValueError if no rung passes the memory gate (nothing to ship).
    """
    menu = list(menu)
    accept_points = list(accept_points)  # may be iterated several times below
    if not menu:
        raise ValueError("select_rung: empty menu")
    if k < 1:
        raise ValueError(f"select_rung: k must be >= 1, got {k}")

    # 1. Memory gate ---------------------------------------------------------
    passing = [r for r in menu if r.resident_bytes <= mem_budget_bytes]
    excluded = [r for r in menu if r.resident_bytes > mem_budget_bytes]
    if not passing:
        smallest = min(menu, key=lambda r: r.resident_bytes)
        raise ValueError(
            f"select_rung: no rung fits mem_budget_bytes={mem_budget_bytes} "
            f"(smallest is {smallest.name} @ {smallest.resident_bytes} B)")

    # 2. Fit -----------------------------------------------------------------
    E_by_bits = {r.name: r.E for r in menu}
    a_inf, beta, fit_mode = fit_acceptance(accept_points, E_by_bits)

    # per-rung acceptance + a-clamp bookkeeping (only when a_inf is known)
    a_clamped = False
    if a_inf is not None:
        for r in passing:
            raw = a_inf - (beta or 0.0) * math.sqrt(max(r.E, 0.0))
            if raw < 0.0 or raw > 1.0:
                a_clamped = True
                break

    # per-rung throughput (needs a_inf; single_point uses a=a_inf constant)
    per_rung_T = {}
    for r in passing:
        if a_inf is None:
            per_rung_T[r.name] = None
        else:
            a = _acceptance(a_inf, beta, r.E)
            per_rung_T[r.name] = _throughput(a, constants.d(r.bits),
                                             constants.t_ms, k)

    # 3. Degenerate test -----------------------------------------------------
    bits = [r.bits for r in passing]
    b_min, b_max = min(bits), max(bits)
    b_mid = 0.5 * (b_min + b_max)
    cost_span_ms = k * constants.c_ms_per_bit * (b_max - b_min)
    cycle_ms = constants.t_ms + k * constants.d(b_mid)
    ratio = cost_span_ms / cycle_ms if cycle_ms > 0 else 0.0
    cost_flat = ratio < degenerate_fraction
    insufficient_slope = beta is None  # single_point / no_data
    degenerate = cost_flat or insufficient_slope

    if degenerate:
        # Highest fidelity == lowest E among passing rungs. Tie-break toward
        # more bits, then name, for determinism.
        chosen = min(passing, key=lambda r: (r.E, -r.bits, r.name))
        regime = "degenerate"
        reason = "cost_flat" if cost_flat else "insufficient_acceptance_data"
    else:
        # 4. Discrete argmax of T. Tie-break toward higher fidelity (lower E).
        chosen = max(passing, key=lambda r: (per_rung_T[r.name], -r.E, r.name))
        regime = "interior"
        reason = None

    b_star = _continuous_bstar_fixed_point(a_inf, beta, constants, b_mid)
    b_star_lw, lw_status = _continuous_bstar_lambertw(a_inf, beta, constants)

    provenance = {
        "schema": "mtp_rung_selection/1",
        "selected_rung": chosen.name,
        "selected_bits": chosen.bits,
        "regime": regime,
        "degenerate_reason": reason,
        "k": k,
        "h_source": h_source,
        "constants": {
            "t_ms": constants.t_ms,
            "d0_ms": constants.d0_ms,
            "c_ms_per_bit": constants.c_ms_per_bit,
        },
        "fit": {
            "a_inf": a_inf,
            "beta": beta,
            "fit_mode": fit_mode,
            "n_points": len(list(accept_points)),
            "beta_negative": (beta is not None and beta < 0),
            "points": [
                {"rung": p.rung_name, "bits": p.bits,
                 "E": E_by_bits.get(p.key),
                 "sqrt_E": (math.sqrt(E_by_bits[p.key])
                            if p.key in E_by_bits else None),
                 "acceptance": p.measured_acceptance}
                for p in accept_points
            ],
        },
        "memory": {
            "mem_budget_bytes": int(mem_budget_bytes),
            "passing": [r.name for r in passing],
            "excluded": [{"name": r.name, "resident_bytes": r.resident_bytes}
                         for r in excluded],
        },
        "menu": [
            {"name": r.name, "bits": r.bits, "resident_bytes": r.resident_bytes,
             "E": r.E, "passes_gate": r.resident_bytes <= mem_budget_bytes}
            for r in menu
        ],
        "degenerate_test": {
            "cost_span_ms": cost_span_ms,
            "cycle_ms": cycle_ms,
            "ratio": ratio,
            "threshold": degenerate_fraction,
            "cost_flat": cost_flat,
            "insufficient_slope": insufficient_slope,
            "b_min": b_min,
            "b_max": b_max,
        },
        "per_rung_T": per_rung_T,
        "continuous_bstar": b_star,
        "continuous_method": "fixed_point" if b_star is not None else None,
        "continuous_bstar_lambertw": b_star_lw,
        "continuous_bstar_lambertw_status": lw_status,
        "a_clamped": a_clamped,
    }
    # Provenance must be JSON-serialisable (doc §3.7); fail fast if not.
    json.dumps(provenance)
    return SelectionResult(rung=chosen, regime=regime, per_rung_T=per_rung_T,
                           provenance=provenance)
