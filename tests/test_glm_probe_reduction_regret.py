"""The regret study draws and estimates through the campaign table (C-MIG-01).

``experiments/glm_probe_reduction_regret.py`` reuses the campaign's own
stack-sample draw and Horvitz-Thompson estimator. Both moved to
``prismaquant.tessera_campaign_table``; these tiny inputs exercise the study's
public ``draw`` and ``ht_total`` against that owner. No study data is read.
"""
import math

import pytest

from experiments import glm_probe_reduction_regret as regret
from prismaquant import tessera_campaign_table as campaign_table


def test_the_study_samples_through_the_campaign_table():
    assert regret.SAMPLING_SOURCE == "prismaquant.tessera_campaign_table"
    assert regret.TC is campaign_table


def test_draw_and_total_are_the_owner_draw_and_estimator(monkeypatch):
    weights = {str(expert): float(1 + expert) for expert in range(8)}
    draw, used = regret.draw(weights, 4, seed=7, stack="layer3")
    assert used == 4
    assert draw == campaign_table.draw_stack_sample(weights, 4, seed=7, stack="layer3")
    sampled = [int(unit) for unit in draw["units"]]
    pi = {int(k): float(v) for k, v in draw["inclusion_probability"].items() if float(v) > 0}
    y = {expert: 0.5 * (expert + 1) ** 2 for expert in sampled}

    estimator = campaign_table._horvitz_thompson_stack
    seen = []

    def spy(sample, member_dloss):
        seen.append((sample.sampled_experts, dict(sample.inclusion_prob), dict(member_dloss)))
        return estimator(sample, member_dloss)

    monkeypatch.setattr(campaign_table, "_horvitz_thompson_stack", spy)
    total = regret.ht_total(sampled, pi, y)
    assert seen == [(tuple(sampled), pi, y)]
    assert total == pytest.approx(math.fsum(y[e] / pi[e] for e in sampled), rel=1e-12)


def test_a_single_random_draw_is_widened_by_one():
    """One certainty unit and n=2 leaves one random draw, which the owner refuses."""
    weights = {"0": 100.0, "1": 1.0, "2": 1.0, "3": 1.0}
    with pytest.raises(RuntimeError, match="exactly one"):
        campaign_table.draw_stack_sample(weights, 2, seed=0, stack="layer0")
    draw, used = regret.draw(weights, 2, seed=0, stack="layer0")
    assert used == 3
    assert draw == campaign_table.draw_stack_sample(weights, 3, seed=0, stack="layer0")
