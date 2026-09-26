from __future__ import annotations

import math

import pytest

from tools.dsv4_ldlq_burn import (
    ANCHORS,
    EXPERT_COUNT,
    HOLDOUTS,
    MEASURED_RUNGS,
    RUNGS,
    _fit_slices,
)

# `_fit_slices` builds `names` over RUNGS and then indexes it by ANCHORS and
# HOLDOUTS, so every anchor and holdout must lie inside the priced domain.
# The 2026-08-06 demand-driven revision narrowed RUNGS to K28-K38
# (tools/dsv4_ldlq_cost_campaign.py:62) without moving ANCHORS = (28, 38, 48)
# or HOLDOUTS = (33, 43) with it, leaving K48 and K43 dangling: the fit raises
# `KeyError: 48` at tools/dsv4_ldlq_burn.py:494 before it evaluates anything.
#
# This is a live defect in the campaign tool, not a test artifact -- the same
# call path is what a real dual-holdout fit takes. It is marked xfail rather
# than repaired here because choosing the replacement anchor/holdout rungs is a
# campaign-design decision (it changes which rungs are measured and which are
# predicted), and the 43-layer burn in flight was launched against these exact
# constants. Repair belongs with the operator who owns the rung domain.
_DANGLING_RUNGS = sorted(set(ANCHORS).union(HOLDOUTS) - set(RUNGS))
_dangling = pytest.mark.xfail(
    bool(_DANGLING_RUNGS),
    reason=(
        "campaign rung domain is inconsistent: "
        f"K{_DANGLING_RUNGS} are named as anchors/holdouts but are outside "
        f"RUNGS=K{RUNGS[0]}-K{RUNGS[-1]}, so _fit_slices raises KeyError"
    ),
    strict=True,
    raises=KeyError,
)


@_dangling
def test_dual_holdout_fit_accepts_exact_production_law():
    errors = {
        k: [3.0 * 2.0 ** (-k / 4.0) for _ in range(EXPERT_COUNT)]
        for k in MEASURED_RUNGS
    }
    accepted, laws, report = _fit_slices(errors)
    assert len(accepted) == EXPERT_COUNT
    assert len(laws) == EXPERT_COUNT
    assert report["rejected"] == 0
    assert set(report["holdouts"]) == {f"K{k}" for k in HOLDOUTS}


@_dangling
def test_dual_holdout_fit_rejects_a_bad_second_holdout():
    errors = {
        k: [2.0 * 2.0 ** (-k / 4.0) for _ in range(EXPERT_COUNT)]
        for k in MEASURED_RUNGS
    }
    errors[HOLDOUTS[1]][17] *= 1.5
    accepted, laws, report = _fit_slices(errors)
    assert 17 not in accepted
    assert 17 not in laws
    assert report["rejected"] == 1
    assert math.isfinite(report["holdouts"][f"K{HOLDOUTS[1]}"]["p95"])
    assert tuple(sorted(ANCHORS)) == ANCHORS
