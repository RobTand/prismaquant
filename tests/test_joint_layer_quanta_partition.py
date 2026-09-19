"""The window partition's two sealed sources (derivation v2).

The plan's ``retained_window_budget_derivation`` block is the partition's
original home (§4.1).  A run plan that deliberately carries none — the
single-run plan the prepared completion binds, whose re-plan sibling exists
only to carry this partition — may name the same sealed record explicitly.
Two sealed sources never disagree silently, and there is no default.

These tests are synthetic: they run everywhere, mount or no mount.  The
plan-carried path against the real sealed campaign is exercised by the
mount-bound tests in ``test_joint_layer_quanta.py``.
"""

import pytest

from prismaquant.joint_layer_quanta import _windows_by_layer

PARTITION = {
    "schema": "prismaquant.joint_retained_window_budget_derivation.v1",
    "windows_by_layer": {"0": 1, "1": 2, "2": 8},
}
PLAN_WITH = {"retained_window_budget_derivation": PARTITION}
PLAN_WITHOUT = {"model": "/models/x"}


def test_a_plan_without_the_block_and_no_input_refuses():
    with pytest.raises(ValueError, match="no sealed retained-window partition"):
        _windows_by_layer(PLAN_WITHOUT)


def test_two_sealed_sources_that_disagree_refuse():
    other = dict(PARTITION, windows_by_layer={"0": 2})
    with pytest.raises(ValueError, match="disagree"):
        _windows_by_layer(PLAN_WITH, other)


def test_two_sealed_sources_that_agree_are_the_plan_path():
    assert _windows_by_layer(PLAN_WITH, PARTITION) == {0: 1, 1: 2, 2: 8}


def test_an_explicit_input_alone_is_the_explicit_path():
    assert _windows_by_layer(PLAN_WITHOUT, PARTITION) == {0: 1, 1: 2, 2: 8}


def test_a_record_without_windows_by_layer_refuses():
    with pytest.raises(ValueError, match="windows_by_layer"):
        _windows_by_layer(PLAN_WITHOUT, {"schema": "not-a-partition"})
