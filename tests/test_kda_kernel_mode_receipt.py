"""CPU checks of the kernel-mode receipt's comparisons (PQ #1214 E4).

The receipt itself runs the Triton kernel on a GLM decoder layer inside the
campaign image. What it records about a difference is plain tensor
arithmetic, checked here: the plane comparison's scale (reference magnitude,
elements that differ bit for bit) and the per-group top-k comparison.
"""
from __future__ import annotations

import pytest
import torch

from experiments.kda_kernel_mode_receipt import _compare, _compare_routes, _group_routes


def _planes(values):
    return {(0, index): torch.tensor(row, dtype=torch.bfloat16) for index, row in enumerate(values)}


def test_a_plane_comparison_records_its_scale_and_its_differing_elements():
    reference = _planes([[1.0, -2.0, 4.0, 0.5], [-1.0, 0.25, 8.0, -0.5]])
    candidate = _planes([[1.0, -2.0, 4.0, 0.5], [-1.0, 0.25, 8.0, -0.5]])
    candidate[(0, 1)][2] = 8.0625  # one element, one bf16 step above 8
    record = _compare(candidate, reference)
    assert record["equal"] is False and record["unequal_count"] == 1
    assert record["elements"] == 8 and record["differing_elements"] == 1
    assert record["differing_fraction"] == 1 / 8
    assert record["max_abs_diff"] == 0.0625
    mean_abs = (1 + 2 + 4 + 0.5 + 1 + 0.25 + 8 + 0.5) / 8
    assert record["reference_mean_abs"] == pytest.approx(mean_abs, rel=0, abs=0)
    assert record["reference_max_abs"] == 8.0
    assert record["max_abs_diff_over_reference_mean_abs"] == pytest.approx(0.0625 / mean_abs)


def test_identical_planes_differ_nowhere():
    reference = _planes([[1.0, -2.0], [3.0, 4.0]])
    record = _compare(_planes([[1.0, -2.0], [3.0, 4.0]]), reference)
    assert record["equal"] is True and record["differing_elements"] == 0
    assert record["max_abs_diff"] == 0.0


def test_differing_elements_are_counted_bit_for_bit():
    # torch.equal calls a signed zero equal; its bits, and so the plane's
    # digest, differ.
    record = _compare(_planes([[-0.0, 1.0]]), _planes([[0.0, 1.0]]))
    assert record["equal"] is True
    assert record["differing_elements"] == 1


def test_planes_of_another_shape_are_not_compared_element_by_element():
    record = _compare({(0, 0): torch.zeros(2, dtype=torch.bfloat16)},
                      {(0, 0): torch.zeros(3, dtype=torch.bfloat16)})
    assert record == {"equal": False, "keys": "shapes or dtypes differ"}


def _routes(*rows):
    return torch.tensor(rows, dtype=torch.int64)


def test_the_same_experts_in_another_order_are_not_a_changed_route():
    first = [_routes([0, 1, 2], [3, 4, 5])]
    second = [_routes([2, 0, 1], [5, 4, 3])]
    assert _compare_routes(first, second) == {
        "comparable": True, "tokens": 2, "tokens_with_a_changed_expert": 0,
        "changed_expert_slots": 0}


def test_a_changed_route_counts_its_token_and_each_expert_it_gained():
    first = [_routes([0, 1, 2], [3, 4, 5]), _routes([6, 7, 8])]
    second = [_routes([0, 1, 9], [3, 4, 5]), _routes([6, 10, 11])]
    assert _compare_routes(first, second) == {
        "comparable": True, "tokens": 3, "tokens_with_a_changed_expert": 2,
        "changed_expert_slots": 3}


def test_routes_of_another_group_count_are_not_compared():
    assert _compare_routes([_routes([0, 1])], []) == {"comparable": False}
    assert _compare_routes(None, [_routes([0, 1])]) == {"comparable": False}


def test_group_routes_take_the_first_pass_and_check_the_others():
    group_a, group_b = _routes([0, 1]), _routes([2, 3])
    # Two probes over two groups, probe-major, as the capture runs them.
    routes, stable = _group_routes([group_a, group_b, group_a.clone(), group_b.clone()], 2, 2)
    assert stable is True and [r.tolist() for r in routes] == [[[0, 1]], [[2, 3]]]
    routes, stable = _group_routes([group_a, group_b, group_a, _routes([2, 4])], 2, 2)
    assert stable is False and routes[1].tolist() == [[2, 3]]


def test_an_arm_with_another_forward_count_has_no_group_routes():
    assert _group_routes([_routes([0, 1])] * 3, 2, 2) == (None, False)
