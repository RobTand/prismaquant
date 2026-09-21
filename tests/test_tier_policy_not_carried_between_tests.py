"""The strict staged-tier policy does not outlive the test that set it (PQ #906).

The two tests below are one scenario in two steps, in file order on one
worker: the first leaves the process-global policy active, as an entry point
driven in-process does, and the second is any later offline test.
"""
from __future__ import annotations

from prismaquant.staged_tier_policy import (
    activate_staged_tier_policy, policy_is_active)


def test_step_1_a_test_leaves_the_policy_active():
    activate_staged_tier_policy("ram,ssd")
    assert policy_is_active()


def test_step_2_the_next_test_starts_without_it():
    assert not policy_is_active()
