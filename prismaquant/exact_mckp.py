"""Exact single-budget multiple-choice knapsack, with the runtime solver's tie rule.

The problem
-----------
Choose one option per unit to minimise ``Σ cost`` subject to
``Σ bytes ≤ max_bytes``. Every cost is an exact integer, and so is every sum:
a caller holding float64 values scales them to integers first
(:func:`exact_integer_costs`). A float64 is a dyadic rational, so that scaling
loses nothing. Among assignments of equal minimum cost the answer is the one
with the fewest bytes, then the lexically first assignment (units in sorted
order, formats compared as strings). That is the order
``allocator_solver.solve_runtime_frontier`` sorts its final frontier by when
its only other active axis is constant, so this solver returns the assignment
that solver ranks first whenever the solver's own float sums are exact and
it is called the way PACT used to call it: prefill 0 on every row, no decode
axis, no device axis. PACT's hull probe (``pact_hull``) is the caller: one
probe is one call.

Why a separate solver
---------------------
``solve_runtime_frontier`` keeps canonical assignment order as a dominance
coordinate in every intermediate fold. It needs that coordinate for two
reasons: a later ``max`` over peak axes can erase a strict difference, and a
float sum can erase a strict loss difference. Neither applies here. Both
coordinates are sums, and the sums are exact, so a strict difference stays
strict through every later fold. With the order coordinate the fold keeps
every prefix that is dominated in (bytes, cost) but lexically earlier than
all of its dominators. On GLM-5.3 (132 units, up to 7 rungs) that is 458,239
states at unit 47 and a refusal at unit 48. Here the fold keeps the 2-D
(bytes, cost) Pareto set, and exactly equal vectors keep the lexically
earlier prefix. That changes no answer (proof below).

Bound pruning
-------------
Each fold also drops a prefix that no completion can make optimal:

* **Infeasible.** Its bytes plus the smallest bytes each remaining unit can
  take exceed the budget.
* **Bounded out.** Its cost plus a lower bound on the remaining units' cost
  EXCEEDS the incumbent: the cost of some feasible assignment already in hand.
  The lower bound is the LP relaxation of the remaining units at the
  remaining bytes. It is the greedy over every unit's lower convex hull of
  (bytes, cost), a standard MCKP result, evaluated exactly: the one
  fractional step is compared by cross-multiplication, never divided. The
  incumbent is the cheapest of: the LP solution with its fractional step
  dropped, completed from every feasible extension in the fold's first pass
  (before any bound or dominance pruning), which is feasible by construction;
  and a caller's known-feasible assignments (``incumbent_hints``).

The comparison is strict. A prefix whose bound EQUALS the incumbent stays,
because it could complete to an assignment of equal cost that wins the byte or
lexical tie.

Why no answer changes
---------------------
Let ``a*`` be the answer: the minimum over feasible assignments of the key
(cost, bytes, formats). Let ``p_k`` be its prefix after k units and ``s_k``
its suffix. Two facts first:

* **Frontier order is lexical order.** Each fold extends its parents in
  frontier order, and each parent's options in format-string order, and
  numbers the extensions in that order. The passes only delete, and the
  survivors are put back in that numbering. By induction the frontier is
  sorted by the prefix's formats tuple, and the numbering is a strict order
  on distinct prefixes.
* **The incumbent never drops below ``cost(a*)``.** Every value it takes is
  the cost of a feasible assignment: the root's LP-rounded completion, a
  validated hint, or an LP-rounded completion read in a fold's first pass.
  So lowering it mid-fold, before the bound pass, cannot prune ``a*``.

By induction, after fold k the frontier holds a prefix ``q`` for which
``q + s_k`` is feasible and its key is at most ``key(a*)``. Because ``a*`` is
the unique minimum, ``q = p_k``, and so the final fold holds ``a*``, which
the final pick (minimum cost, then bytes, then frontier order) returns.

* **Extension.** ``q`` extended by ``a*``'s option at unit k has bytes at
  most ``p_k``'s and cost at most ``p_k``'s.
* **Infeasibility test.** The extension keeps ``s_k``'s bytes within the
  budget, so the test cannot drop it.
* **Bound test.** Its remaining room is at least ``s_k``'s bytes, so the LP
  bound there is at most ``s_k``'s cost. Then cost plus bound is at most
  ``cost(a*)``, which is at most the incumbent. The strict test keeps it.
* **Dominance.** If the extension is dominated, some survivor has bytes and
  cost no larger, and it is lexically earlier when both are equal. (The
  entry that drops another is itself kept: the running minimum the sweep
  compares against is always the cost of the last entry it kept.) That
  survivor, completed by ``s_k``, is feasible with a key no larger.

Nothing here is a tolerance. The only numbers compared are exact integers.

Bounds and refusals
-------------------
``max_states`` bounds each fold's surviving frontier and ``max_transitions``
bounds the total (prefix, option) pairs attempted, with the same meaning as
``solve_runtime_frontier``'s bounds. Crossing either raises
:class:`~prismaquant.allocator_solver.RuntimeFrontierLimitError`. It never
returns a truncated answer. ``None`` means the search completed and found no
feasible assignment.
"""
from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
from numbers import Integral
from typing import Hashable, Mapping, Sequence

from .allocator_solver import RuntimeFrontierLimitError, _runtime_int

METHOD = "exact_mckp_pareto2_lp_bound"
TIE_RULE = "min_cost_then_min_bytes_then_lexical_formats"


def exact_integer_costs(values: Mapping[Hashable, object]) -> tuple[dict, int]:
    """Scale exact rationals to integers over one power-of-two denominator.

    ``values`` maps any key to a float, an int or a ``Fraction``. A float is a
    dyadic rational (``Fraction(x)`` is exact), so the common denominator of
    floats is the largest one among them. Returns ``(scaled, denominator)``
    with ``scaled[key] / denominator == values[key]`` exactly. Refuses a
    non-finite value or a non-dyadic ``Fraction``, which no power-of-two
    denominator holds.
    """
    exact = {}
    for key, value in values.items():
        if isinstance(value, bool):
            raise ValueError(f"{key!r}: a cost must be a number, not a bool")
        try:
            exact[key] = value if isinstance(value, Fraction) else Fraction(value)
        except (ValueError, OverflowError, TypeError) as exc:
            raise ValueError(f"{key!r}: a cost must be finite ({exc})") from None
    denominator = 1
    for key, value in exact.items():
        den = value.denominator
        if den & (den - 1):
            raise ValueError(f"{key!r}: {value} is not dyadic")
        denominator = max(denominator, den)
    return ({key: value.numerator * (denominator // value.denominator)
             for key, value in exact.items()}, denominator)


def _exact_int(value, label: str) -> int:
    """An exact integer of either sign (a cost); bytes use ``_runtime_int``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _lower_hull_steps(options: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], list]:
    """The (bytes, cost) lower convex hull of one unit, as a base and its steps.

    The base is the fewest-bytes option (the cheaper one on a byte tie). The
    steps walk the hull toward the cheapest option: every step adds bytes
    (``db > 0``), saves cost (``dc < 0``), and saves strictly less per byte
    than the step before it. Dropping points above the hull or on it, and
    options that cost no less than one with fewer bytes, leaves the LP value
    unchanged at every room.
    """
    by_bytes: dict[int, int] = {}
    for b, c in options:
        if b not in by_bytes or c < by_bytes[b]:
            by_bytes[b] = c
    chain: list[tuple[int, int]] = []
    for b in sorted(by_bytes):
        c = by_bytes[b]
        if chain and c >= chain[-1][1]:
            continue  # more bytes, no cheaper: never on the LP path
        while len(chain) >= 2:
            (b1, c1), (b2, c2) = chain[-2], chain[-1]
            # chain[-1] stays only if it lies strictly below the chord from
            # chain[-2] to (b, c): (c2 - c1)/(b2 - b1) < (c - c1)/(b - b1).
            if (c2 - c1) * (b - b1) < (c - c1) * (b2 - b1):
                break
            chain.pop()
        chain.append((b, c))
    base = chain[0]
    steps = [(b2 - b1, c2 - c1) for (b1, c1), (b2, c2) in zip(chain, chain[1:])]
    return base, steps


class _SuffixBound:
    """The LP relaxation of units ``k..n-1`` at any remaining byte room.

    ``room`` below ``min_bytes`` is infeasible. Otherwise the relaxation takes
    every unit's base, then the merged hull steps in decreasing saving per
    byte while they fit, then a fraction of the next one.
    """

    __slots__ = ("min_bytes", "base_cost", "cum_bytes", "cum_cost", "steps")

    def __init__(self, min_bytes: int, base_cost: int, steps: list[tuple[int, int]], *,
                 presorted: bool = False):
        self.min_bytes = min_bytes
        self.base_cost = base_cost
        # Decreasing saving per byte: -dc/db, compared exactly. A caller that
        # already holds the steps in that order passes presorted=True.
        self.steps = (list(steps) if presorted else
                      sorted(steps, key=lambda s: Fraction(-s[1], s[0]), reverse=True))
        self.cum_bytes = [0]
        self.cum_cost = [0]
        for db, dc in self.steps:
            self.cum_bytes.append(self.cum_bytes[-1] + db)
            self.cum_cost.append(self.cum_cost[-1] + dc)

    def evaluate(self, room: int) -> tuple[int, int, int, int] | None:
        """``(whole, db, extra, dc)`` for the LP value at ``room``, or None if infeasible.

        The LP value is ``whole + extra * dc / db`` (``db > 0``, ``extra < db``;
        ``db == 0`` means no fractional step). ``whole`` is also the cost of a
        feasible integral completion: the full steps taken, the fraction dropped.
        """
        slack = room - self.min_bytes
        if slack < 0:
            return None
        p = bisect_right(self.cum_bytes, slack) - 1
        whole = self.base_cost + self.cum_cost[p]
        if p == len(self.steps):
            return whole, 0, 0, 0
        db, dc = self.steps[p]
        return whole, db, slack - self.cum_bytes[p], dc


def solve_exact_mckp(
    units: Mapping[str, Sequence[tuple[str, int, int]]],
    *,
    max_bytes: int,
    max_states: int = 100_000,
    max_transitions: int = 8_000_000,
    diagnostics: dict | None = None,
    incumbent_hints: Sequence[Mapping[str, str]] = (),
) -> dict[str, str] | None:
    """The minimum (cost, bytes, formats) assignment under ``max_bytes``.

    ``units`` maps each unit name to its options ``(fmt, cost, bytes)``, with
    ``cost`` an exact integer (any sign) and ``bytes`` a nonnegative integer.
    Returns ``{unit: fmt}``, or ``None`` when no assignment fits. See the
    module docstring for the exactness argument.

    ``incumbent_hints`` are complete assignments the caller knows fit, such as
    the two ends of the hull segment a probe bisects. Each one's exact cost
    is a valid incumbent, so it can only tighten the bound; the answer does
    not depend on it. A hint that names another unit set or format, or does
    not fit, is refused.
    """
    diag = diagnostics if diagnostics is not None else {}
    diag.update(complete=False, feasible=False, frontier_sizes=[], candidate_counts=[],
                transitions=0, refusal=None, approximation="none", method=METHOD,
                tie_rule=TIE_RULE, dimensions=["memory_bytes", "cost"],
                intermediate_tie_coordinate=False, pruned_infeasible=0,
                pruned_by_bound=0)
    max_bytes = _runtime_int(max_bytes, "max_bytes")
    max_states = _runtime_int(max_states, "max_states", positive=True)
    max_transitions = _runtime_int(max_transitions, "max_transitions", positive=True)
    diag.update(max_states=max_states, max_transitions=max_transitions)
    if not units:
        raise ValueError("no units to allocate")
    names = sorted(units)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("unit names must be nonempty strings")
    menus: list[list[tuple[str, int, int]]] = []
    for name in names:
        seen, menu = set(), []
        for option in units[name]:
            fmt, cost, size = option
            if not isinstance(fmt, str) or not fmt:
                raise ValueError(f"{name!r}: a format must be a nonempty string")
            if fmt in seen:
                raise ValueError(f"duplicate option ({name!r}, {fmt!r})")
            seen.add(fmt)
            menu.append((fmt, _exact_int(cost, f"{name}@{fmt} cost"),
                         _runtime_int(size, f"{name}@{fmt} bytes")))
        if not menu:
            raise ValueError(f"{name!r}: empty option menu")
        # Formats in string order: option index order is lexical order.
        menus.append(sorted(menu, key=lambda o: o[0]))

    n = len(names)
    hulls = [_lower_hull_steps([(b, c) for _f, c, b in menu]) for menu in menus]
    # Every hull step, sorted ONCE by decreasing saving per byte. A unit's own
    # steps already save strictly less each, so the stable sort keeps them in
    # hull order; each suffix's sorted list is this one filtered by unit.
    ranked = sorted(((Fraction(-dc, db), k, db, dc)
                     for k, (_base, unit_steps) in enumerate(hulls) for db, dc in unit_steps),
                    key=lambda step: step[0], reverse=True)
    bounds: list[_SuffixBound] = [None] * (n + 1)  # type: ignore[list-item]
    bounds[n] = _SuffixBound(0, 0, [], presorted=True)
    min_bytes, base_cost = 0, 0
    for k in range(n - 1, -1, -1):
        (b0, c0), _unit_steps = hulls[k]
        min_bytes += b0
        base_cost += c0
        bounds[k] = _SuffixBound(min_bytes, base_cost,
                                 [(db, dc) for _key, unit, db, dc in ranked if unit >= k],
                                 presorted=True)
    diag["min_total_bytes"] = bounds[0].min_bytes
    root = bounds[0].evaluate(max_bytes)
    if root is None:
        diag.update(complete=True, feasible=False)
        return None
    # The incumbent: a feasible assignment's exact cost, improved every fold.
    incumbent = root[0]
    priced = [{fmt: (cost, size) for fmt, cost, size in menu} for menu in menus]
    for hint in incumbent_hints:
        if set(hint) != set(names):
            raise ValueError("an incumbent hint must assign exactly the solver's units")
        try:
            chosen = [priced[k][hint[name]] for k, name in enumerate(names)]
        except KeyError as exc:
            raise ValueError(f"an incumbent hint names an unpriced format {exc}") from None
        if sum(size for _cost, size in chosen) > max_bytes:
            raise ValueError("an incumbent hint does not fit max_bytes")
        incumbent = min(incumbent, sum(cost for cost, _size in chosen))
    diag["incumbent_hints"] = len(incumbent_hints)

    # Frontier states in lexical prefix order: (bytes, cost). back[k][i] is
    # (parent index in fold k-1, option index) for state i of fold k.
    frontier: list[tuple[int, int]] = [(0, 0)]
    back: list[list[tuple[int, int]]] = []
    for k, name in enumerate(names):
        menu = menus[k]
        pairs = len(frontier) * len(menu)
        if diag["transitions"] + pairs > max_transitions:
            diag.update(refusal="max_transitions", refused_unit=name,
                        attempted_transitions=diag["transitions"] + pairs)
            raise RuntimeFrontierLimitError(
                f"exact MCKP at {name!r} requires {diag['transitions'] + pairs} attempted "
                f"transitions, over max_transitions={max_transitions}; exact search refused "
                "without returning a partial allocation")
        diag["transitions"] += pairs
        rest = bounds[k + 1]
        # Pass 1: extend, drop the infeasible, and read the incumbent each
        # extension's LP-rounded completion offers.
        extended = []
        gen = 0
        for parent, (b_prefix, c_prefix) in enumerate(frontier):
            for j, (_fmt, cost, size) in enumerate(menu):
                b = b_prefix + size
                lp = rest.evaluate(max_bytes - b)
                if lp is None:
                    diag["pruned_infeasible"] += 1
                    continue
                c = c_prefix + cost
                if c + lp[0] < incumbent:
                    incumbent = c + lp[0]
                extended.append((b, c, gen, parent, j, lp))
                gen += 1
        # Pass 2: drop what the LP bound proves cannot reach the incumbent.
        # c + whole + extra*dc/db > incumbent, cross-multiplied by db > 0.
        kept = []
        for entry in extended:
            b, c, _gen, _parent, _j, (whole, db, extra, dc) = entry
            gap = c + whole - incumbent
            if (gap * db + extra * dc > 0) if db else gap > 0:
                diag["pruned_by_bound"] += 1
                continue
            kept.append(entry)
        diag["candidate_counts"].append(len(kept))
        # Pass 3: the 2-D Pareto set. Sorted by (bytes, cost, lexical order),
        # a state survives only below every cost seen at no more bytes; an
        # exact duplicate loses to the lexically earlier one sorted first.
        kept.sort(key=lambda e: (e[0], e[1], e[2]))
        survivors = []
        best = None
        for entry in kept:
            if best is None or entry[1] < best:
                best = entry[1]
                survivors.append(entry)
                if len(survivors) > max_states:
                    diag.update(refusal="max_states", refused_unit=name,
                                frontier_size_lower_bound=len(survivors))
                    raise RuntimeFrontierLimitError(
                        f"exact MCKP frontier at {name!r} exceeds max_states={max_states}; "
                        "exact search refused without truncating nondominated states")
        # Back to lexical order: generation order is (parent rank, format).
        survivors.sort(key=lambda e: e[2])
        frontier = [(e[0], e[1]) for e in survivors]
        back.append([(e[3], e[4]) for e in survivors])
        diag["frontier_sizes"].append(len(frontier))
        if not frontier:
            break

    diag["incumbent"] = incumbent
    if not frontier:
        # Unreachable: the root is feasible, so an optimum exists, and the
        # module docstring's induction keeps it in every fold. A refusal
        # rather than a silent None, should that argument ever break.
        raise RuntimeError("exact MCKP lost every state although an assignment fits")
    # The answer: the minimum (cost, bytes, lexical order) state.
    best_index = min(range(len(frontier)), key=lambda i: (frontier[i][1], frontier[i][0], i))
    total_bytes, total_cost = frontier[best_index]
    chosen: list[str] = [""] * n
    index = best_index
    for k in range(n - 1, -1, -1):
        parent, j = back[k][index]
        chosen[k] = menus[k][j][0]
        index = parent
    diag.update(complete=True, feasible=True, frontier_size=len(frontier),
                total_cost=total_cost, total_bytes=total_bytes)
    return dict(zip(names, chosen))
