"""Exact single-budget MCKP (PQ #1584): the PACT hull's binding-budget probe.

Three references:

* brute force over every assignment, in exact rational arithmetic, with the
  (cost, bytes, lexical formats) key;
* the unbounded ``solve_runtime_frontier``, whose first-ranked allocation is
  the same key whenever its float sums are exact (integer-valued costs here);
* a GLM-5.3 E4M3 prefix (``tests/fixtures/pact_glm53_e4m3_prefix_1584.json``),
  real float costs, where the two solvers must agree on the assignment and on
  its exact sums.
"""
import itertools
import json
import random
from fractions import Fraction
from pathlib import Path

import pytest

from prismaquant.allocator_solver import (
    Candidate, RuntimeFrontierLimitError, solve_runtime_frontier)
from prismaquant.exact_mckp import (
    _SuffixBound, _lower_hull_steps, exact_integer_costs, solve_exact_mckp)
from prismaquant.measured_runtime_prices import RuntimeResources

FIXTURE = Path(__file__).parent / "fixtures" / "pact_glm53_e4m3_prefix_1584.json"


def _brute(units, max_bytes):
    names = sorted(units)
    best = None
    for choice in itertools.product(*(sorted(units[u], key=lambda o: o[0]) for u in names)):
        size = sum(o[2] for o in choice)
        if size > max_bytes:
            continue
        key = (sum((Fraction(o[1]) for o in choice), Fraction(0)), size,
               tuple(o[0] for o in choice))
        if best is None or key < best:
            best = key
    return None if best is None else (dict(zip(names, best[2])), best[0], best[1])


def _runtime_reference(units, max_bytes):
    """``solve_runtime_frontier``'s first allocation, called as the hull used to."""
    cands, resources = {}, {}
    for u, options in units.items():
        cands[u] = [Candidate(fmt=f, bits_per_param=0.0, memory_bytes=b, predicted_dloss=c)
                    for f, c, b in options]
        for f, _c, b in options:
            resources[(u, f)] = RuntimeResources(
                prefill_ms=0.0, decode_ms=None, serialized_bytes=b, resident_bytes=0,
                peak_scratch_bytes=0, activation_bytes=0)
    frontier = solve_runtime_frontier(cands, resources, max_memory_bytes=max_bytes,
                                      max_prefill_ms=0.0, max_states=10**9,
                                      max_transitions=10**12)
    return dict(frontier[0].assignment) if frontier else None


def _exact_sums(units, assignment):
    options = {(u, f): (c, b) for u, row in units.items() for f, c, b in row}
    return (sum((Fraction(options[(u, f)][0]) for u, f in assignment.items()), Fraction(0)),
            sum(options[(u, f)][1] for u, f in assignment.items()))


def _integer_units(units):
    """Float or Fraction costs scaled to exact integers, the solver's input."""
    scaled, _den = exact_integer_costs({(u, f): c for u, row in units.items()
                                        for f, c, _b in row})
    return {u: [(f, scaled[(u, f)], b) for f, _c, b in row] for u, row in units.items()}


def _tied_problem(rng, n_units, n_formats):
    """Integer-valued costs over a tiny range, so ties of every kind occur.

    Duplicate options (same cost and bytes under two names), equal cost at
    different bytes, and equal bytes at different cost are all common, and
    every float sum is exact (small integers).
    """
    fmts = ["A", "B", "C", "D", "E", "F", "G"][:n_formats]
    units = {}
    for u in range(n_units):
        row, used = [], rng.sample(fmts, rng.randint(1, n_formats))
        for f in used:
            if row and rng.random() < 0.3:
                _f, c, b = rng.choice(row)          # an exact duplicate option
            else:
                c, b = float(rng.randint(-3, 6)), rng.randint(0, 5)
            row.append((f, c, b))
        units[f"u{u:02d}"] = row
    return units


@pytest.mark.parametrize("seed", range(40))
def test_equals_brute_force_and_the_unbounded_runtime_solver_on_engineered_ties(seed):
    rng = random.Random(seed)
    units = _tied_problem(rng, n_units=rng.randint(1, 5), n_formats=rng.randint(1, 4))
    lo = sum(min(o[2] for o in row) for row in units.values())
    hi = sum(max(o[2] for o in row) for row in units.values())
    for budget in sorted({max(lo - 1, 0), lo, (lo + hi) // 2, hi - 1, hi, hi + 3}):
        expected = _brute(units, budget)
        diag = {}
        got = solve_exact_mckp(_integer_units(units), max_bytes=budget, diagnostics=diag)
        reference = _runtime_reference(units, budget)
        if expected is None:
            assert got is None and reference is None and diag["feasible"] is False
            continue
        assert got == expected[0], (seed, budget)
        assert reference == expected[0], (seed, budget)
        assert _exact_sums(units, got) == (expected[1], expected[2])
        assert (diag["total_cost"], diag["total_bytes"]) == (expected[1], expected[2])
        assert diag["complete"] and diag["intermediate_tie_coordinate"] is False


@pytest.mark.parametrize("seed", range(20))
def test_incumbent_hints_tighten_the_bound_and_never_change_the_answer(seed):
    rng = random.Random(3000 + seed)
    units = _integer_units(_tied_problem(rng, n_units=rng.randint(2, 5), n_formats=4))
    names = sorted(units)
    hi = sum(max(o[2] for o in row) for row in units.values())
    lo = sum(min(o[2] for o in row) for row in units.values())
    budget = (lo + hi) // 2
    fitting = [dict(zip(names, (o[0] for o in choice)))
               for choice in itertools.product(*(units[u] for u in names))
               if sum(o[2] for o in choice) <= budget]
    plain = {}
    answer = solve_exact_mckp(units, max_bytes=budget, diagnostics=plain)
    # The optimum itself, and arbitrary feasible assignments, as hints.
    for hints in ([answer], rng.sample(fitting, min(3, len(fitting))), fitting):
        diag = {}
        assert solve_exact_mckp(units, max_bytes=budget, diagnostics=diag,
                                incumbent_hints=hints) == answer
        assert diag["incumbent"] <= plain["incumbent"]
        assert diag["total_cost"] == plain["total_cost"]
    with pytest.raises(ValueError, match="exactly the solver's units"):
        solve_exact_mckp(units, max_bytes=budget, incumbent_hints=[{names[0]: "A"}])
    with pytest.raises(ValueError, match="unpriced format"):
        solve_exact_mckp(units, max_bytes=budget, incumbent_hints=[{u: "Z" for u in names}])
    too_big = {u: max(units[u], key=lambda o: o[2])[0] for u in names}
    if sum(max(o[2] for o in units[u]) for u in names) > budget:
        with pytest.raises(ValueError, match="does not fit"):
            solve_exact_mckp(units, max_bytes=budget, incumbent_hints=[too_big])


@pytest.mark.parametrize("seed", range(12))
def test_equals_brute_force_with_dyadic_costs_of_any_sign(seed):
    rng = random.Random(1000 + seed)
    units = {}
    for u in range(rng.randint(2, 6)):
        units[f"u{u}"] = [(f, rng.choice([-1, 1]) * rng.randint(0, 64) / 2 ** rng.randint(0, 40),
                           rng.randint(0, 40)) for f in rng.sample("PQRST", rng.randint(1, 3))]
    lo = sum(min(o[2] for o in row) for row in units.values())
    hi = sum(max(o[2] for o in row) for row in units.values())
    for budget in (lo, lo + (hi - lo) // 3, lo + 2 * (hi - lo) // 3, hi):
        expected = _brute(units, budget)
        assert solve_exact_mckp(_integer_units(units), max_bytes=budget) == expected[0]


def test_a_float_sum_cannot_erase_a_strict_difference():
    # 2**53 + 1 rounds to 2**53 in float64, so the float solver sees a cost
    # tie and takes the smaller-bytes option; the exact sum says it costs 1
    # more.
    units = {"u0": [("P", 2.0 ** 53, 0)],
             "u1": [("Q", 1.0, 0), ("R", 0.0, 1)]}
    assert _runtime_reference(units, 1) == {"u0": "P", "u1": "Q"}
    assert solve_exact_mckp(_integer_units(units), max_bytes=1) == {"u0": "P", "u1": "R"}
    assert _brute(units, 1)[0] == {"u0": "P", "u1": "R"}


def test_equal_cost_prefers_fewer_bytes_then_the_lexically_first_assignment():
    units = {"a": [("X", 5, 3), ("Y", 5, 2), ("Z", 5, 2)],
             "b": [("X", 1, 1), ("Y", 0, 2)],
             "c": [("M", 2, 0), ("N", 2, 0)]}
    # a: Y and Z tie on (cost, bytes) -> Y (lexically first); b: at room for
    # b=Y the total cost is lower; c: M and N tie exactly -> M.
    assert solve_exact_mckp(units, max_bytes=4) == {"a": "Y", "b": "Y", "c": "M"}
    assert solve_exact_mckp(units, max_bytes=3) == {"a": "Y", "b": "X", "c": "M"}
    assert solve_exact_mckp(units, max_bytes=1) is None


@pytest.mark.parametrize("seed", range(20))
def test_the_lp_bound_is_a_lower_bound_and_its_rounded_completion_fits(seed):
    rng = random.Random(2000 + seed)
    rows = [[(rng.randint(0, 30), rng.randint(-20, 40)) for _ in range(rng.randint(1, 5))]
            for _ in range(rng.randint(1, 5))]
    min_bytes, base, steps = 0, 0, []
    for row in rows:
        (b0, c0), unit_steps = _lower_hull_steps(row)
        assert (b0, c0) == min(row)
        assert all(db > 0 and dc < 0 for db, dc in unit_steps)
        min_bytes, base, steps = min_bytes + b0, base + c0, steps + unit_steps
    bound = _SuffixBound(min_bytes, base, steps)
    totals = [(sum(b for b, _c in choice), sum(c for _b, c in choice))
              for choice in itertools.product(*rows)]
    for room in range(0, sum(max(b for b, _c in row) for row in rows) + 2):
        feasible = [c for b, c in totals if b <= room]
        value = bound.evaluate(room)
        if not feasible:
            assert value is None
            continue
        whole, db, extra, dc = value
        lp = Fraction(whole) + (Fraction(extra * dc, db) if db else 0)
        assert lp <= min(feasible)
        # The fraction dropped leaves a feasible integral completion.
        assert whole >= min(feasible)


def test_refuses_rather_than_truncates():
    # The first seeded instance whose exact fold holds more than one state.
    for seed in range(50):
        rng = random.Random(seed)
        units = {f"u{u}": [(f, rng.randint(0, 50), rng.randint(0, 50)) for f in "ABCD"]
                 for u in range(6)}
        budget = sum(max(o[2] for o in row) for row in units.values()) // 2
        diag = {}
        answer = solve_exact_mckp(units, max_bytes=budget, diagnostics=diag)
        peak = max(diag["frontier_sizes"])
        if peak > 1:
            break
    assert peak > 1
    assert answer == _brute(units, budget)[0]
    refused = {}
    with pytest.raises(RuntimeFrontierLimitError, match="max_states"):
        solve_exact_mckp(units, max_bytes=budget, max_states=peak - 1, diagnostics=refused)
    assert refused["refusal"] == "max_states" and refused["complete"] is False
    with pytest.raises(RuntimeFrontierLimitError, match="max_transitions"):
        solve_exact_mckp(units, max_bytes=budget, max_transitions=diag["transitions"] - 1)
    assert solve_exact_mckp(units, max_bytes=budget, max_states=peak,
                            max_transitions=diag["transitions"]) == answer


def test_input_refusals():
    with pytest.raises(ValueError, match="no units"):
        solve_exact_mckp({}, max_bytes=1)
    with pytest.raises(ValueError, match="empty option menu"):
        solve_exact_mckp({"u": []}, max_bytes=1)
    with pytest.raises(ValueError, match="duplicate option"):
        solve_exact_mckp({"u": [("A", 1, 1), ("A", 2, 1)]}, max_bytes=1)
    with pytest.raises(ValueError, match="must be an integer"):
        solve_exact_mckp({"u": [("A", 0.5, 1)]}, max_bytes=1)
    with pytest.raises(ValueError, match="nonnegative"):
        solve_exact_mckp({"u": [("A", 1, -1)]}, max_bytes=1)


def test_exact_integer_costs_scale_floats_without_loss():
    values = {"a": 0.1, "b": -3.0, "c": 2.0 ** -60, "d": 0, "e": Fraction(3, 8)}
    scaled, den = exact_integer_costs(values)
    assert den & (den - 1) == 0
    assert all(Fraction(scaled[k], den) == Fraction(v) for k, v in values.items())
    with pytest.raises(ValueError, match="not dyadic"):
        exact_integer_costs({"x": Fraction(1, 3)})
    with pytest.raises(ValueError, match="finite"):
        exact_integer_costs({"x": float("inf")})
    with pytest.raises(ValueError, match="not a bool"):
        exact_integer_costs({"x": True})


def _glm_prefix(n_units):
    doc = json.loads(FIXTURE.read_text())
    return doc, {u: doc["units"][u] for u in sorted(doc["units"])[:n_units]}


@pytest.mark.parametrize("n_units, weights, room", [
    *((12, w, r) for w in ((1.0, 0.0), (0.0, 1.0), "edge") for r in (0.25, 0.5, 0.75)),
    # All 20 fixture units: the prefix through the first 7-rung routed unit.
    (20, "edge", 0.5)])
def test_identity_with_the_unbounded_runtime_solver_on_a_glm53_prefix(n_units, weights, room):
    doc, raw = _glm_prefix(n_units)
    if weights == "edge":
        weights = tuple(doc["edge_weights"])
    w_d, w_t = weights
    # The hull's float cost, one per option, fed to BOTH solvers: the
    # reference sums it in float, the new probe sums its exact value.
    units = {u: [(f, w_d * d + w_t * t, b) for f, d, b, t in row] for u, row in raw.items()}
    lo = sum(min(o[2] for o in row) for row in units.values())
    hi = sum(max(o[2] for o in row) for row in units.values())
    budget = lo + int((hi - lo) * room)
    diag = {}
    got = solve_exact_mckp(_integer_units(units), max_bytes=budget, diagnostics=diag)
    reference = _runtime_reference(units, budget)
    assert got == reference
    exact_cost, exact_bytes = _exact_sums(units, got)
    assert (exact_cost, exact_bytes) == _exact_sums(units, reference)
    _scaled, den = exact_integer_costs({(u, f): c for u, row in units.items()
                                        for f, c, _b in row})
    assert Fraction(diag["total_cost"], den) == exact_cost
    assert diag["total_bytes"] == exact_bytes <= budget
