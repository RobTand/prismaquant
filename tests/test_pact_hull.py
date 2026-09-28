"""PACT lower convex hull (PQ #1584): exact against brute force.

Every assignment of a small problem (8 units x 3 options = 6561) is
enumerated, the feasible ones under each byte budget are kept, and their
lower-left convex hull is computed in exact rational arithmetic. The
dichotomic generator must return exactly those vertices, for a slack budget
(the byte axis is flattened) and for binding ones (real bytes stay live).
"""
import itertools
import math
import random
from fractions import Fraction

import pytest

from prismaquant import pact_hull
from prismaquant.allocator_solver import Candidate

FORMATS = ("A", "B", "C")


def _problem(seed, n_units=8):
    rng = random.Random(seed)
    candidates, time_ms = {}, {}
    for u in range(n_units):
        unit = f"u{u}"
        row = []
        for fmt in FORMATS:
            row.append(Candidate(fmt=fmt, bits_per_param=0.0,
                                 memory_bytes=rng.randrange(1_000, 5_000),
                                 predicted_dloss=rng.uniform(0.001, 1.0)))
            time_ms[(unit, fmt)] = rng.uniform(0.5, 20.0)
        candidates[unit] = row
    return candidates, time_ms


def _brute_hull(candidates, time_ms, budget):
    units = sorted(candidates)
    points = {}
    for choice in itertools.product(*(candidates[u] for u in units)):
        if sum(c.memory_bytes for c in choice) > budget:
            continue
        d = sum((Fraction(c.predicted_dloss) for c in choice), Fraction(0))
        t = sum((Fraction(time_ms[(u, c.fmt)]) for u, c in zip(units, choice)), Fraction(0))
        points[tuple((u, c.fmt) for u, c in zip(units, choice))] = (t, d)
    if not points:
        return []
    # Pareto set in (t, d), fastest first, then the lower convex chain.
    ordered = sorted(points.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    pareto, best_d = [], None
    for key, (t, d) in ordered:
        if best_d is None or d < best_d:
            pareto.append((key, t, d))
            best_d = d
    chain = []
    for key, t, d in pareto:
        while len(chain) >= 2:
            (_, t1, d1), (_, t2, d2) = chain[-2], chain[-1]
            # Keep chain[-1] only when it turns strictly convex (lower hull).
            if (t2 - t1) * (d - d1) - (d2 - d1) * (t - t1) > 0:
                break
            chain.pop()
        chain.append((key, t, d))
    return [dict(key) for key, _t, _d in chain]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_hull_equals_brute_force_for_slack_and_binding_budgets(seed):
    candidates, time_ms = _problem(seed)
    lo = sum(min(c.memory_bytes for c in cs) for cs in candidates.values())
    hi = sum(max(c.memory_bytes for c in cs) for cs in candidates.values())
    budgets = [hi, hi + 10_000] + [lo + (hi - lo) * k // 5 for k in (1, 2, 3, 4)]
    for budget in budgets:
        expected = _brute_hull(candidates, time_ms, budget)
        hull = pact_hull.dichotomic_lower_hull(candidates, time_ms, max_memory_bytes=budget)
        got = [dict(v.assignment) for v in hull.vertices]
        # Accuracy end first in the hull; the brute chain runs fastest first.
        assert got == list(reversed(expected)), (seed, budget)
        assert hull.byte_axis == (pact_hull.BYTE_AXIS_SLACK if budget >= hi
                                  else pact_hull.BYTE_AXIS_LIVE)
        assert all(p.byte_axis == hull.byte_axis for p in hull.probes)
        assert all(v.memory_bytes <= budget for v in hull.vertices)
        # Two endpoint probes, one discovering probe per interior vertex and
        # one confirming probe per edge.
        if len(got) >= 2:
            assert len(hull.probes) == 2 * len(got) - 1
        # Convexity: the Δloss a millisecond buys rises toward the fast end.
        lambdas = hull.edge_lambdas
        assert all(x > 0 for x in lambdas)
        assert all(lambdas[k] < lambdas[k + 1] for k in range(len(lambdas) - 1))


def test_endpoints_are_the_two_single_objective_optima():
    candidates, time_ms = _problem(7)
    hull = pact_hull.dichotomic_lower_hull(candidates, time_ms, max_memory_bytes=10**9)
    best = {u: min(cs, key=lambda c: c.predicted_dloss).fmt for u, cs in candidates.items()}
    fast = {u: min(cs, key=lambda c: time_ms[(u, c.fmt)]).fmt for u, cs in candidates.items()}
    assert dict(hull.vertices[0].assignment) == best
    assert dict(hull.vertices[-1].assignment) == fast
    record = hull.as_dict()
    assert record["candidate_generator"] == "lower_convex_hull_dichotomic"
    assert record["probe_count"] == len(hull.probes)
    assert record["edge_lambda_role"].startswith("diagnostic only")


def test_an_endpoint_tie_resolves_to_the_dominating_point_without_an_epsilon():
    # u0's two options tie on Δloss; the slower one must not survive as the
    # accuracy endpoint.
    candidates = {
        "u0": [Candidate("A", 0.0, 10, 0.5), Candidate("B", 0.0, 10, 0.5),
               Candidate("C", 0.0, 10, 0.9)],
        "u1": [Candidate("A", 0.0, 10, 0.1), Candidate("B", 0.0, 10, 0.3),
               Candidate("C", 0.0, 10, 0.2)],
    }
    time_ms = {("u0", "A"): 9.0, ("u0", "B"): 4.0, ("u0", "C"): 1.0,
               ("u1", "A"): 8.0, ("u1", "B"): 2.0, ("u1", "C"): 5.0}
    hull = pact_hull.dichotomic_lower_hull(candidates, time_ms, max_memory_bytes=100)
    assert all(v.assignment["u0"] != "A" for v in hull.vertices)
    assert dict(hull.vertices[0].assignment) == {"u0": "B", "u1": "A"}


def test_refusals():
    candidates, time_ms = _problem(0, n_units=2)
    missing = dict(time_ms)
    missing.pop(("u0", "A"))
    with pytest.raises(pact_hull.PactHullError, match="unpriced option must be absent"):
        pact_hull.dichotomic_lower_hull(candidates, missing, max_memory_bytes=10**9)
    with pytest.raises(pact_hull.PactHullError, match="no assignment fits"):
        pact_hull.dichotomic_lower_hull(candidates, time_ms, max_memory_bytes=1)
    with pytest.raises(pact_hull.PactHullError, match="no priced option"):
        pact_hull.dichotomic_lower_hull({"u0": []}, {}, max_memory_bytes=1)


def test_the_resolution_is_the_float64_sum_bound():
    candidates, time_ms = _problem(5)
    hull = pact_hull.dichotomic_lower_hull(candidates, time_ms, max_memory_bytes=10**9)
    n = hull.n_units
    for p in hull.probes:
        if p.segment is None:
            assert p.resolution is None and p.lambda_dloss_per_ms is None
            continue
        a = hull.points[p.segment[0]]
        w_d, w_t = p.weights
        scale = math.fsum(abs(w_d * [c for c in candidates[u] if c.fmt == f][0].predicted_dloss)
                          + abs(w_t * time_ms[(u, f)]) for u, f in a.assignment.items())
        assert p.resolution >= (n + 1) * 2.0 ** -53 * scale
        assert p.lambda_dloss_per_ms == pytest.approx(w_t / w_d)


def test_a_live_probe_over_its_state_bound_is_refused_with_its_measured_growth():
    # The first seeded problem whose exact probe folds hold more than one state.
    for seed in range(3, 60):
        candidates, time_ms = _problem(seed, n_units=6)
        budget = sum(min(c.memory_bytes for c in row) for row in candidates.values()) + 3_000
        exact = pact_hull.dichotomic_lower_hull(candidates, time_ms, max_memory_bytes=budget)
        peak = max(p.solver["max_fold_size"] for p in exact.probes)
        if peak > 1:
            break
    assert exact.byte_axis == pact_hull.BYTE_AXIS_LIVE
    assert peak > 1
    with pytest.raises(pact_hull.RuntimeFrontierLimitError) as refused:
        pact_hull.dichotomic_lower_hull(candidates, time_ms, max_memory_bytes=budget,
                                        max_states=peak - 1)
    sizes = refused.value.diagnostics["frontier_sizes"]
    assert refused.value.diagnostics["refusal"] == "max_states"
    assert refused.value.diagnostics["frontier_size_lower_bound"] > peak - 1
    assert all(size <= peak - 1 for size in sizes)
    # A raised bound returns the same exact hull.
    raised = pact_hull.dichotomic_lower_hull(
        candidates, time_ms, max_memory_bytes=budget,
        max_states=10 * pact_hull.DEFAULT_MAX_STATES,
        max_transitions=10 * pact_hull.DEFAULT_MAX_TRANSITIONS)
    assert [dict(v.assignment) for v in raised.vertices] == \
        [dict(v.assignment) for v in exact.vertices]


def _order_coordinate_problem(n_units=20):
    """Every unit: A costs 2 at 2 bytes, B costs 1 at 1 byte, same time.

    A is dominated in (bytes, Δloss) and lexically first, so a fold that keeps
    lexical order as a dominance coordinate keeps one prefix per count of A's
    (nothing lexically earlier dominates it), while the (bytes, Δloss) Pareto
    set is the all-B prefix alone.
    """
    candidates = {f"u{u:02d}": [Candidate("A", 0.0, 2, 2.0), Candidate("B", 0.0, 1, 1.0)]
                  for u in range(n_units)}
    time_ms = {(u, f): 1.0 for u in candidates for f in ("A", "B")}
    return candidates, time_ms


def test_a_binding_probe_holds_only_the_two_dimensional_pareto_set():
    candidates, time_ms = _order_coordinate_problem()
    # 20 units: every smallest option fits in 20 bytes, every largest in 40.
    budget = 30
    assert pact_hull._byte_axis(candidates, budget)[0] == pact_hull.BYTE_AXIS_LIVE
    assignment = pact_hull.probe_assignment(candidates, time_ms, (1.0, 0.0),
                                            max_memory_bytes=budget, max_states=2)
    assert assignment == {u: "B" for u in candidates}


def test_a_probe_prices_every_option_exactly():
    # w_d*d + w_t*t = 2**53 + 1 rounds to 2**53 in float64 and ties P with Q,
    # whose exact cost is 1 lower. Both byte axes must pick Q.
    candidates = {"u0": [Candidate("P", 0.0, 0, 2.0 ** 53), Candidate("Q", 0.0, 1, 2.0 ** 53)],
                  "u1": [Candidate("X", 0.0, 0, 0.0), Candidate("Y", 0.0, 5, 1.0)]}
    time_ms = {("u0", "P"): 1.0, ("u0", "Q"): 0.0, ("u1", "X"): 0.0, ("u1", "Y"): 1.0}
    for budget, axis in ((1, pact_hull.BYTE_AXIS_LIVE), (6, pact_hull.BYTE_AXIS_SLACK)):
        assert pact_hull._byte_axis(candidates, budget)[0] == axis
        assert pact_hull.probe_assignment(candidates, time_ms, (1.0, 1.0),
                                          max_memory_bytes=budget) == {"u0": "Q", "u1": "X"}
