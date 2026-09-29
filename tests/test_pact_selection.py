"""PQ #1585: the PACT selection record.

A materiality test derived from the table's own bootstrap interval, three
picks on the exact frontier, and a roster record that a shipcard can carry.
"""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path

import pytest

from prismaquant import pact_selection
from prismaquant.pact_selection import (
    HIGH_PREFILL_RULE,
    SCHEMA,
    PactSelectionError,
    select_pact,
    validate_pact_selection,
)
from prismaquant.quality_prefill_contract import canonical_sha256

FIXTURES = Path(__file__).parent / "fixtures"
TABLE = {"table_sha256": "0" * 64}
BUDGET = 10**12


def _pt(pid, time_ms, dloss, *, half=None, lanes=None, nbytes=1000):
    return {
        "point_id": pid,
        "assignment_sha256": canonical_sha256({"assignment": pid}),
        "bytes": nbytes,
        "time_ms": time_ms,
        "time_interval_ms": None if half is None else [time_ms - half, time_ms + half],
        "predicted_dloss": dloss,
        "kernel_lanes": lanes,
    }


def _select(points, **kwargs):
    kwargs.setdefault("regime_m", 2048)
    kwargs.setdefault("table_identity", TABLE)
    kwargs.setdefault("frontier_scope", "exact_frontier")
    return select_pact(points, **kwargs)


def _two_candidate_frontier(half):
    """Fastest, A, B, best. A and B are the top two interior candidates.

    In the normalised chord frame A's improvement exceeds B's by 0.005/sqrt(2).
    """
    d = lambda y: 0.01 + y * 0.99  # noqa: E731
    return [
        _pt("fast", 100.0, 1.0, half=0.5),
        _pt("A", 300.0, d(0.200), half=half),
        _pt("B", 310.0, d(0.195), half=half),
        _pt("best", 1100.0, 0.01, half=0.5),
    ]


def _cubic_frontier(half, n=21):
    pts = []
    for i in range(n):
        x = i / (n - 1)
        pts.append(_pt(f"c{i:02d}", 100.0 + 1000.0 * x, 0.01 + 0.99 * (1 - x) ** 3, half=half))
    return pts


# ---------------------------------------------------------------- materiality


def test_flat_when_endpoint_intervals_overlap():
    pts = _two_candidate_frontier(half=1.0)
    pts[0]["time_interval_ms"] = [90.0, 700.0]
    pts[-1]["time_interval_ms"] = [600.0, 1200.0]
    record = _select(pts)
    assert record["materiality"]["verdict"] == "flat"
    assert record["picks"]["high_accuracy"]["point_id"] == "best"
    assert record["picks"]["balanced"]["point_id"] == "best"
    assert record["picks"]["high_prefill"]["point_id"] == "best"
    assert record["picks"]["balanced"]["status"] == "flat_time_axis"
    roster_ids = {entry["point_id"] for entry in record["roster"]}
    assert {"fast", "best"} <= roster_ids


def test_touching_intervals_are_flat_and_a_gap_is_material():
    touching = _two_candidate_frontier(half=1.0)
    touching[0]["time_interval_ms"] = [90.0, 600.0]
    touching[-1]["time_interval_ms"] = [600.0, 1200.0]
    assert _select(touching)["materiality"]["verdict"] == "flat"
    gap = copy.deepcopy(touching)
    gap[0]["time_interval_ms"] = [90.0, 599.999]
    assert _select(gap)["materiality"]["verdict"] == "material"


def test_materiality_is_the_span_against_the_pooled_one_sided_widths():
    pts = _two_candidate_frontier(half=1.0)
    pts[0]["time_interval_ms"] = [90.0, 130.0]
    pts[-1]["time_interval_ms"] = [1000.0, 1200.0]
    m = _select(pts)["materiality"]
    assert m["span_ms"] == pytest.approx(1000.0)
    assert m["fastest_upper_width_ms"] == pytest.approx(30.0)
    assert m["best_lower_width_ms"] == pytest.approx(100.0)
    assert m["resolution_ms"] == pytest.approx(130.0)
    assert m["pooled_interval_ms"] == [90.0, 1200.0]
    assert m["verdict"] == "material"


def test_missing_endpoint_interval_is_unresolved_and_refused_by_default():
    pts = _two_candidate_frontier(half=1.0)
    pts[0]["time_interval_ms"] = None
    with pytest.raises(PactSelectionError, match="interval"):
        _select(pts)
    record = _select(pts, allow_unresolved=True)
    assert record["materiality"]["verdict"] == "unresolved"
    assert record["min_separation"]["value"] == 0.0
    assert record["min_separation"]["source"] == "no_intervals"


def test_single_point_frontier_is_its_own_pick():
    record = _select([_pt("only", 100.0, 0.05, half=1.0)])
    assert record["materiality"]["verdict"] == "single_point"
    assert {pick["point_id"] for pick in record["picks"].values()} == {"only"}


# ------------------------------------------------- derived min_separation


def test_min_separation_is_derived_from_the_top_two_interval_half_widths():
    record = _select(_two_candidate_frontier(half=0.5))
    span = 1000.0
    expected = (0.5 + 0.5) / (span * math.sqrt(2.0))
    assert record["min_separation"]["value"] == pytest.approx(expected)
    assert record["min_separation"]["source"] == "top_two_interval_half_widths"
    assert record["picks"]["balanced"]["point_id"] == "A"


def test_top_two_refusal_sees_a_non_vertex_exact_frontier_point():
    # B lies above the hull (it is not a vertex) yet is within the noise of A.
    narrow = _select(_two_candidate_frontier(half=1.0))
    assert narrow["picks"]["balanced"]["status"] == "selected"
    wide = _select(_two_candidate_frontier(half=20.0))
    balanced = wide["picks"]["balanced"]
    assert balanced["status"] == "refused"
    assert balanced["point_id"] is None
    assert "separat" in balanced["reason"]
    assert wide["picks"]["high_prefill"]["status"] == "refused"
    # The accuracy endpoint never depends on the knee.
    assert wide["picks"]["high_accuracy"]["point_id"] == "best"


def test_the_derivation_has_no_numeric_literal_threshold():
    source = Path(pact_selection.__file__).read_text()
    for banned in ("0.05", "0.1 ", "1e-3", "1e-6", "0.01"):
        assert banned not in source, banned


# ------------------------------------------------------------- three picks


def test_three_picks_on_a_convex_frontier():
    record = _select(_cubic_frontier(half=0.01))
    picks = record["picks"]
    assert picks["high_accuracy"]["point_id"] == "c20"  # argmin predicted_dloss
    assert picks["balanced"]["status"] == "selected"
    assert picks["high_prefill"]["status"] == "selected"
    times = {e["point_id"]: e["time_ms"] for e in record["roster"]}
    assert times[picks["high_prefill"]["point_id"]] < times[picks["balanced"]["point_id"]]
    assert times[picks["balanced"]["point_id"]] < times[picks["high_accuracy"]["point_id"]]
    assert record["rule"]["high_prefill"] == dict(HIGH_PREFILL_RULE)


def test_balanced_is_select_development_point_on_the_same_points():
    from prismaquant.quality_prefill_knee import select_development_point

    frontier = _cubic_frontier(half=0.01)
    record = _select(frontier)
    ns = [
        {
            "point_id": p["point_id"],
            "assignment_id": p["assignment_sha256"],
            "bytes": p["bytes"],
            "prefill_budget": round(p["time_ms"] * 1_000_000),
            "quality_value": p["predicted_dloss"],
            "currency": "joint_aura_predicted_dloss",
            "units": "x",
            "measurement_status": "measured",
        }
        for p in frontier
    ]
    direct = select_development_point(
        ns, byte_budget=BUDGET, min_separation=record["min_separation"]["value"]
    )
    assert record["picks"]["balanced"]["point_id"] == direct.knee_point_id


def test_high_prefill_refuses_with_a_reason_when_the_sub_frontier_is_too_small():
    record = _select(_two_candidate_frontier(half=1.0))
    high = record["picks"]["high_prefill"]
    assert high["status"] == "refused"
    assert high["point_id"] is None
    assert "fewer than 3" in high["reason"]


def test_byte_budget_defaults_to_the_largest_point_and_is_recorded():
    frontier = _cubic_frontier(half=0.01)
    assert _select(frontier)["byte_budget"] == 1000
    small = _select(frontier, byte_budget=999)["picks"]["high_accuracy"]
    assert small["status"] == "refused"


# ------------------------------------------------------------------ record


def test_record_shape_roster_and_lane_histogram():
    frontier = _cubic_frontier(half=0.01)
    for point in frontier:
        point["kernel_lanes"] = {"cutlass_sm120": 3, "fallback": 1}
    record = _select(frontier)
    assert record["schema"] == SCHEMA == "prismaquant.pact_selection.v1"
    assert record["regime_m"] == 2048
    assert record["table_identity"] == TABLE
    assert record["frontier_scope"] == "exact_frontier"
    assert record["quality_axis"]["interval"] is None
    assert record["measurement_status"] == "predicted_proposal"
    picks = {p["point_id"] for p in record["picks"].values() if p["point_id"]}
    roster_ids = [e["point_id"] for e in record["roster"]]
    assert picks <= set(roster_ids)
    assert len(roster_ids) == len(set(roster_ids))
    for entry in record["roster"]:
        assert entry["kernel_lanes"] == {"cutlass_sm120": 3, "fallback": 1}
        assert entry["roles"] and entry["assignment_sha256"]
    body = {k: v for k, v in record.items() if k != "identity_sha256"}
    assert record["identity_sha256"] == canonical_sha256(body)
    assert validate_pact_selection(record) == record


def test_record_is_deterministic():
    a = _select(_cubic_frontier(half=0.01))
    b = _select(list(reversed(_cubic_frontier(half=0.01))))
    assert a == b


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["picks"]["balanced"].__setitem__("point_id", "c00"),
        lambda r: r.__setitem__("regime_m", 512),
        lambda r: r["materiality"].__setitem__("verdict", "flat"),
        lambda r: r["roster"].pop(),
        lambda r: r.__setitem__("schema", "prismaquant.pact_selection.v0"),
    ],
)
def test_validate_refuses_a_tampered_record(mutate):
    record = _select(_cubic_frontier(half=0.01))
    bad = copy.deepcopy(record)
    mutate(bad)
    with pytest.raises(PactSelectionError):
        validate_pact_selection(bad)


def test_validate_refuses_an_identity_that_was_recomputed_over_an_inconsistent_body():
    record = _select(_cubic_frontier(half=0.01))
    bad = copy.deepcopy(record)
    bad["picks"]["balanced"]["point_id"] = "not-in-roster"
    bad["identity_sha256"] = canonical_sha256(
        {k: v for k, v in bad.items() if k != "identity_sha256"}
    )
    with pytest.raises(PactSelectionError, match="roster"):
        validate_pact_selection(bad)


# -------------------------------------------------- the scratch frontiers


def _scratch(name, *, half_fraction=None):
    with (FIXTURES / name).open() as handle:
        rows = list(csv.DictReader(handle))
    points = []
    for row in rows:
        t = float(row["t_linear_ms"])
        half = None if half_fraction is None else t * half_fraction
        points.append(
            _pt(row["point"], t, float(row["predicted_dloss"]), half=half,
                nbytes=int(row["bytes_total"]))
        )
    return points


def test_balanced_rule_reproduces_point_96_on_the_pre_685_frontier():
    # The scratch table has no bootstrap columns: unresolved, resolution 0.
    record = _select(_scratch("pact_frontier_pre685_m2048.csv"), allow_unresolved=True)
    assert record["materiality"]["verdict"] == "unresolved"
    assert record["picks"]["balanced"]["point_id"] == "96"
    assert record["picks"]["high_accuracy"]["point_id"] == "0"
    assert validate_pact_selection(record) == record


def test_after_685_frontier_reports_and_records_its_verdict():
    scratch = _scratch("pact_frontier_after640_m2048.csv")
    record = _select(scratch, allow_unresolved=True)
    assert record["materiality"]["verdict"] == "unresolved"
    assert "no interval" in record["materiality"]["reason"]
    # With labelled fixture intervals the verdict follows the table: a 0.5%
    # bootstrap is far inside the span, a 60% one swallows it.
    material = _select(_scratch("pact_frontier_after640_m2048.csv", half_fraction=0.005))
    assert material["materiality"]["verdict"] == "material"
    flat = _select(_scratch("pact_frontier_after640_m2048.csv", half_fraction=0.6))
    assert flat["materiality"]["verdict"] == "flat"
    assert flat["picks"]["balanced"]["point_id"] == flat["picks"]["high_accuracy"]["point_id"]
