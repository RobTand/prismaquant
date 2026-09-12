"""Pure synthetic tests for the offline complete-curve oracle."""
from itertools import combinations
import json
import math
import sys

import pytest

import experiments.sparse_rate_curve_oracle as oracle
from experiments.sparse_rate_curve_oracle import minimum_anchors
from prismaquant.adaptive_anchored_shape import _predict_between


def brute_force_count(rates, values, mode, tolerance):
    """Independent exhaustive reference for small finite rosters."""
    for additional in range(len(rates) - 1):
        for middle in combinations(range(1, len(rates) - 1), additional):
            anchors = (0, *middle, len(rates) - 1)
            valid = True
            for left, right in zip(anchors, anchors[1:]):
                if values[left] <= values[right]:
                    valid = False
                    break
                for point in range(left + 1, right):
                    prediction = _predict_between(mode, rates[left], values[left], rates[right], values[right], rates[point])
                    if abs(prediction / values[point] - 1.0) > tolerance:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return len(anchors)
    return None


@pytest.mark.parametrize("mode", ["value", "log2"])
@pytest.mark.parametrize("values", [
    [100.0, 83.0, 70.0, 55.0, 41.0],
    [128.0, 100.0, 75.0, 48.0, 33.0],
    [100.0, 94.0, 52.0, 47.0, 18.0],
])
def test_exact_cardinality_matches_all_anchor_subsets(mode, values):
    rates = [10, 20, 30, 40, 50]
    tolerance = .08
    result = minimum_anchors(rates, values, mode=mode, max_relative_error=tolerance)
    assert result["optimal_anchor_count"] == brute_force_count(rates, values, mode, tolerance)


@pytest.mark.parametrize("mode", ["value", "log2"])
def test_straight_line_uses_only_two_endpoints(mode):
    rates = [10, 20, 30, 40, 50]
    values = ([100.0 - 2.0 * index for index in range(5)] if mode == "value"
              else [100.0 * (0.8 ** index) for index in range(5)])
    result = minimum_anchors(rates, values, mode=mode, max_relative_error=1e-12)
    assert result["status"] == "optimal"
    assert result["optimal_anchor_count"] == 2
    assert result["anchor_rates"] == [10, 50]
    assert result["actual_max_relative_error"] <= 1e-12


def test_hard_bend_requires_more_than_endpoints():
    result = minimum_anchors([10, 20, 30, 40, 50], [100.0, 99.0, 30.0, 29.0, 1.0],
                             mode="value", max_relative_error=.01)
    assert result["status"] == "optimal"
    assert result["optimal_anchor_count"] > 2
    assert result["actual_max_relative_error"] <= .01


def test_equal_cardinality_paths_use_the_lowest_predecessor_index():
    result = minimum_anchors([10, 20, 30, 40], [100.0, 80.0, 50.0, 10.0],
                             mode="value", max_relative_error=.15)
    assert result["optimal_anchor_count"] == 3
    assert result["anchor_indices"] == [0, 1, 3]
    assert result["tie_break"] == "lowest_predecessor_index"


def test_nonmonotone_shape_is_explicitly_refused():
    result = minimum_anchors([10, 20, 30], [10.0, 12.0, 11.0], mode="value", max_relative_error=.01)
    assert result["status"] == "refused_shape"
    assert result["optimal_anchor_count"] is None
    assert result["actual_max_relative_error"] is None
    assert result["require_strict_decrease"] is True


@pytest.mark.parametrize("values,expected_count", [
    ([10.0, 10.0, 10.0], 2),
    ([10.0, 8.0, 9.0, 1.0], 4),
])
def test_nonmonotone_opt_in_finds_minimum_path_that_strict_default_refuses(values, expected_count):
    rates = list(range(10, 10 * (len(values) + 1), 10))
    strict = minimum_anchors(rates, values, mode="value", max_relative_error=.1)
    relaxed = minimum_anchors(
        rates, values, mode="value", max_relative_error=.1,
        require_strict_decrease=False,
    )
    assert strict["status"] == "refused_shape"
    assert relaxed["status"] == "optimal"
    assert relaxed["optimal_anchor_count"] == expected_count
    assert relaxed["require_strict_decrease"] is False


def test_evaluate_curve_forwards_the_strict_decrease_policy(monkeypatch):
    monkeypatch.setattr(oracle, "validate_curve", lambda curve: ((10, 20, 30), (5.0, 5.0, 5.0)))
    relaxed = oracle.evaluate_curve(
        object(), mode="value", tolerance=0.0, require_strict_decrease=False,
    )
    strict = oracle.evaluate_curve(object(), mode="value", tolerance=0.0)
    assert relaxed["status"] == "optimal"
    assert relaxed["require_strict_decrease"] is False
    assert strict["status"] == "refused_shape"
    assert strict["require_strict_decrease"] is True


def test_allow_nonmonotone_cli_records_the_selected_policy(tmp_path, monkeypatch):
    curve_path = tmp_path / "curve.json"
    curve_path.write_text(json.dumps({
        "curve_id": "synthetic", "qname": "test.linear", "family": "test",
        "activation_contract": "test", "source_identity": {"test": True},
        "calibration_identity": {"test": True}, "recipe_identity": {"test": True},
    }))
    out = tmp_path / "out"
    monkeypatch.setattr(oracle, "validate_curve", lambda curve: ((10, 20, 30), (5.0, 5.0, 5.0)))
    monkeypatch.setattr(sys, "argv", [
        "sparse_rate_curve_oracle.py", "--curve", str(curve_path), "--out", str(out),
        "--allow-nonmonotone",
    ])
    oracle.main()
    report = json.loads((out / "report.json").read_text())
    assert report["require_strict_decrease"] is False
    assert {result["require_strict_decrease"] for result in report["results"].values()} == {False}


@pytest.mark.parametrize("values,tolerance", [
    ([10.0, math.inf, 1.0], .01), ([10.0, math.nan, 1.0], .01), ([10.0, 5.0, 1.0], math.inf),
])
def test_invalid_finite_bounds_are_refused(values, tolerance):
    with pytest.raises(ValueError):
        minimum_anchors([10, 20, 30], values, mode="value", max_relative_error=tolerance)


@pytest.mark.parametrize("require_strict_decrease", [0, 1, None, "false"])
def test_strict_decrease_policy_must_be_boolean(require_strict_decrease):
    with pytest.raises(ValueError, match="require_strict_decrease"):
        minimum_anchors(
            [10, 20, 30], [10.0, 5.0, 1.0], mode="value", max_relative_error=.01,
            require_strict_decrease=require_strict_decrease,
        )
