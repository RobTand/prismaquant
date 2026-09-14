"""Section 9's development-point rule: what it nominates and what it refuses."""

from __future__ import annotations

import pathlib

import pytest

from prismaquant.quality_prefill_knee import (
    KNEE_RULE,
    KneeSelectionError,
    feasible_points,
    nondominated,
    read_points,
    select_development_point,
)

JOINT = "joint_aura_predicted_dloss"
SCALAR = "output_mse_under_route_activation_contract"


def point(pid, *, latency, quality, size=1000, status="predicted_screen", currency=JOINT):
    return {
        "point_id": pid,
        "assignment_id": f"a.{pid}",
        "bytes": size,
        "prefill_budget": latency,
        "quality_value": quality,
        "currency": currency,
        "units": "nats",
        "measurement_status": status,
    }


def convex_frontier():
    """A frontier with a real elbow at ``mid``.

    Latency 10 -> 40 buys most of the quality; 40 -> 100 buys very little, so
    the perpendicular improvement peaks at 40.
    """
    return [
        point("fast", latency=10, quality=1.00),
        point("early", latency=20, quality=0.70),
        point("mid", latency=40, quality=0.25),
        point("late", latency=70, quality=0.15),
        point("best", latency=100, quality=0.10),
    ]


def test_the_module_imports_no_rng():
    source = pathlib.Path("prismaquant/quality_prefill_knee.py").read_text()
    assert "import random" not in source
    assert "from random" not in source
    assert "secrets" not in source


def test_the_elbow_is_nominated_with_both_neighbours_and_both_endpoints():
    selection = select_development_point(
        convex_frontier(), byte_budget=2000, min_separation=0.01
    )
    assert selection.knee_point_id == "mid"
    assert selection.neighbour_point_ids == ("early", "late")
    assert selection.endpoint_point_ids == ("fast", "best")
    # Section 10 measures the knee, both neighbours and both endpoint controls.
    assert selection.measurement_roster() == ("fast", "early", "mid", "late", "best")
    assert selection.rule == KNEE_RULE


def test_the_selection_is_byte_identical_across_repeats():
    first = select_development_point(
        convex_frontier(), byte_budget=2000, min_separation=0.01
    )
    second = select_development_point(
        list(reversed(convex_frontier())), byte_budget=2000, min_separation=0.01
    )
    assert first.identity_sha256() == second.identity_sha256()


def test_input_order_does_not_change_the_nominee():
    ordered = select_development_point(
        convex_frontier(), byte_budget=2000, min_separation=0.01
    )
    shuffled_like = [convex_frontier()[i] for i in (3, 0, 4, 1, 2)]
    assert (
        select_development_point(
            shuffled_like, byte_budget=2000, min_separation=0.01
        ).knee_point_id
        == ordered.knee_point_id
    )


def test_the_normalisation_and_chord_are_published():
    selection = select_development_point(
        convex_frontier(), byte_budget=2000, min_separation=0.01
    )
    published = selection.normalisation.as_dict()
    assert published["latency_min"] == 10 and published["latency_max"] == 100
    assert published["quality_min"] == 0.10 and published["quality_max"] == 1.00
    assert published["chord_from_point_id"] == "fast"
    assert published["chord_to_point_id"] == "best"


def test_a_dominated_point_never_becomes_the_knee():
    points = convex_frontier()
    points.append(point("dominated", latency=45, quality=0.90))
    selection = select_development_point(points, byte_budget=2000, min_separation=0.01)
    assert "dominated" not in selection.nondominated_point_ids
    assert selection.knee_point_id == "mid"


def test_a_point_over_the_byte_budget_is_infeasible():
    points = convex_frontier()
    points.append(point("huge", latency=15, quality=0.05, size=999_999))
    selection = select_development_point(points, byte_budget=2000, min_separation=0.01)
    assert "huge" not in selection.nondominated_point_ids


def test_a_budget_no_point_fits_refuses_with_the_smallest_size():
    with pytest.raises(KneeSelectionError, match="no point fits the byte budget 10"):
        select_development_point(convex_frontier(), byte_budget=10, min_separation=0.01)


def test_a_linear_frontier_proposes_no_knee():
    straight = [
        point("a", latency=10, quality=1.0),
        point("b", latency=20, quality=0.9),
        point("c", latency=30, quality=0.8),
    ]
    with pytest.raises(KneeSelectionError, match="no interior point improves"):
        select_development_point(straight, byte_budget=2000, min_separation=0.0)


def test_two_candidates_inside_the_declared_resolution_refuse():
    points = convex_frontier()
    points.append(point("twin", latency=41, quality=0.251))
    with pytest.raises(KneeSelectionError, match="within the declared resolution"):
        select_development_point(points, byte_budget=2000, min_separation=0.5)


def test_fewer_than_three_nondominated_points_refuses():
    two = [point("a", latency=10, quality=1.0), point("b", latency=20, quality=0.5)]
    with pytest.raises(KneeSelectionError, match="needs at least 3"):
        select_development_point(two, byte_budget=2000, min_separation=0.0)


def test_a_degenerate_quality_range_refuses():
    flat = [
        point("a", latency=10, quality=0.5),
        point("b", latency=20, quality=0.5),
        point("c", latency=30, quality=0.5),
    ]
    with pytest.raises(KneeSelectionError, match="needs at least 3"):
        # Equal quality at higher latency is dominated, so the frontier
        # collapses to one point before the range check can fire.
        select_development_point(flat, byte_budget=2000, min_separation=0.0)


def test_mixed_currencies_refuse_rather_than_being_compared():
    points = convex_frontier()
    points.append(point("scalar", latency=25, quality=0.4, currency=SCALAR))
    with pytest.raises(KneeSelectionError, match="one frontier carries one currency"):
        select_development_point(points, byte_budget=2000, min_separation=0.01)


def test_a_measured_point_is_not_compared_against_a_screened_one():
    points = convex_frontier()
    points.append(point("served", latency=25, quality=0.4, status="measured"))
    with pytest.raises(
        KneeSelectionError, match="one frontier carries one measurement status"
    ):
        select_development_point(points, byte_budget=2000, min_separation=0.01)


def test_the_status_is_copied_and_never_upgraded():
    selection = select_development_point(
        convex_frontier(), byte_budget=2000, min_separation=0.01
    )
    assert selection.measurement_status == "predicted_screen"
    assert selection.as_dict()["measurement_status"] == "predicted_screen"


def test_a_measured_frontier_keeps_its_measured_label():
    measured = [
        point(pid, latency=lat, quality=q, status="measured")
        for pid, lat, q in (("f", 10, 1.0), ("m", 40, 0.25), ("b", 100, 0.10))
    ]
    selection = select_development_point(
        measured, byte_budget=2000, min_separation=0.0
    )
    assert selection.measurement_status == "measured"
    assert selection.knee_point_id == "m"


def test_duplicate_point_ids_refuse():
    points = convex_frontier() + [point("mid", latency=50, quality=0.2)]
    with pytest.raises(KneeSelectionError, match="already present"):
        select_development_point(points, byte_budget=2000, min_separation=0.01)


def test_an_unknown_coverage_type_refuses():
    bad = convex_frontier()
    bad[0]["measurement_status"] = "probably_fine"
    with pytest.raises(KneeSelectionError, match="not a declared coverage type"):
        read_points(bad)


def test_a_negative_quality_value_refuses():
    bad = convex_frontier()
    bad[0]["quality_value"] = -1.0
    with pytest.raises(KneeSelectionError, match="must not be negative"):
        read_points(bad)


def test_min_separation_has_no_default():
    with pytest.raises(TypeError):
        select_development_point(convex_frontier(), byte_budget=2000)


def test_equal_cost_points_are_both_kept():
    """Two assignments that price alike are distinct routes, not one point."""
    twins = read_points(
        [
            point("a", latency=10, quality=1.0),
            point("b", latency=40, quality=0.3),
            point("c", latency=40, quality=0.3),
            point("d", latency=100, quality=0.1),
        ]
    )
    kept = {p.point_id for p in nondominated(twins)}
    assert {"b", "c"} <= kept


def test_feasible_points_uses_exact_integer_bytes():
    typed = read_points([point("a", latency=10, quality=1.0, size=1024)])
    assert feasible_points(typed, byte_budget=1024) == typed
    assert feasible_points(typed, byte_budget=1023) == ()
