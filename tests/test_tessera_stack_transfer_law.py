"""The stack transfer law: does it recover a known slope, and what is its error?

RobTand/prismaquant#495 part 2.  The law predicts a packed routed stack's cost
at the rungs it did NOT census, from the rung it did plus a small expert
sample.  ``docs/results/glm_tessera_probe_reduction_regret_2026-09-10.md``
section 3.2 defines it (schedule ``b_law``) and section 3.4 point 3 is why the
slope may be pooled across stacks at all.

The study's own per-expert rows live inside the frozen campaign workspace and
are not reachable from a test, so the fixture here is synthetic with slopes
that are known by construction -- which is the stronger check anyway: a fit
that recovers a planted slope and a prediction that lands inside its own
declared error are both falsifiable statements, and neither is available from
a summary table.
"""
import math
import random

import pytest

from prismaquant.tessera_rate_surface import (
    STACK_TRANSFER_REFERENCE_Q256,
    StackRateSample,
    fit_stack_transfer_law,
    predict_stack_rates,
)
from prismaquant.tessera_formats import TesseraFormatError

#: The campaign's three rungs.  832 and 1088 are what the schedule stops
#: censusing; 960 is the reference every prediction regresses on.
REFERENCE = STACK_TRANSFER_REFERENCE_Q256
TARGETS = (832, 1088)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

#: The planted law, per (projection, rate).  Slopes near 1 with a level shift
#: is the shape the study measured: a lower rung costs more, and the
#: cross-expert ordering is nearly preserved.
PLANTED = {
    832: {"gate_proj": (1.06, 1.40), "up_proj": (1.03, 1.25),
          "down_proj": (0.97, 1.55)},
    1088: {"gate_proj": (0.94, -1.30), "up_proj": (0.98, -1.10),
           "down_proj": (1.02, -1.45)},
}


def _stack(name, *, base, sampled, noise, rng, level=0.0):
    """One stack whose experts obey ``PLANTED`` up to log2 noise of ``noise``.

    ``base`` is a per-projection vector of log2 reference values shared by every
    stack and PERMUTED per stack, so each stack has its own per-expert ordering
    while all stacks share one mean reference level.  That is deliberate: the
    pooled fit centres on a single mean over all the pooled stacks, so a
    per-stack intercept that correlates with a per-stack mean reference level
    biases the slope (see ``fit_stack_transfer_law``).  Holding the level common
    and letting ``level`` move only the intercept isolates the property under
    test -- the slope -- from that known sensitivity, which has its own test.
    """
    experts = len(base[PROJECTIONS[0]])
    order = {projection: rng.sample(range(experts), experts)
             for projection in PROJECTIONS}
    reference = {
        expert: {projection: 2.0 ** base[projection][order[projection][expert]]
                 for projection in PROJECTIONS}
        for expert in range(experts)}
    sampled_mse = {}
    for rate in TARGETS:
        sampled_mse[rate] = {}
        for expert in range(experts):
            row = {}
            for projection in PROJECTIONS:
                slope, intercept = PLANTED[rate][projection]
                row[projection] = 2.0 ** (
                    intercept + level
                    + slope * math.log2(reference[expert][projection])
                    + rng.gauss(0.0, noise))
            sampled_mse[rate][expert] = row
    return StackRateSample(
        stack=name,
        projections=PROJECTIONS,
        experts=tuple(range(experts)),
        weights=None,
        reference_q256=REFERENCE,
        reference_mse=reference,
        sampled_experts=tuple(sorted(sampled)),
        sampled_mse=sampled_mse,
        currency="output_mse_under_route_activation_contract",
    )


def _truth_total(stack, rate):
    """The weighted total the census would have measured at ``rate``."""
    return math.fsum(
        stack.weight(expert) * math.fsum(
            float(stack.sampled_mse[rate][expert][projection])
            for projection in stack.projections)
        for expert in stack.experts)


def _base(rng, experts):
    """One shared per-projection vector of log2 reference values."""
    return {projection: [-9.0 + 0.35 * index + rng.gauss(0.0, 1.1)
                         for _ in range(experts)]
            for index, projection in enumerate(PROJECTIONS)}


@pytest.fixture
def population():
    """Twelve stacks of 48 experts, 9 of them sampled -- the study's 3%-ish.

    The held-out stack keeps its full frame at both target rungs so the test
    can compare against a truth the law never saw; a production stack would
    only ever hold the 9.
    """
    rng = random.Random(20260911)
    base = _base(rng, 48)
    stacks = {}
    for index in range(12):
        sampled = rng.sample(range(48), 9)
        stacks[f"s:layer{index}"] = _stack(
            f"s:layer{index}", base=base, sampled=sampled, noise=0.05,
            rng=rng, level=0.03 * (index - 5.5))
    return stacks


def test_the_pooled_fit_recovers_the_planted_slopes(population):
    """A slope pooled over eleven stacks lands on the one that generated them."""
    law = fit_stack_transfer_law(population, hold_out="s:layer0")
    assert law.reference_q256 == REFERENCE
    assert law.target_q256 == TARGETS
    assert "s:layer0" not in law.pooled_stacks
    assert len(law.pooled_stacks) == 11
    for rate in TARGETS:
        # 11 stacks x 9 sampled experts.
        assert law.pooled_rows[rate] == 99
        for projection in PROJECTIONS:
            planted = PLANTED[rate][projection][0]
            assert law.slope[rate][projection] == pytest.approx(planted, abs=0.02)
            # The residual carries the planted per-expert noise (0.05 log2) AND
            # the between-stack intercept spread the pooled centring cannot
            # absorb (sd ~0.1), so it is conservative by construction -- which
            # is the direction an error field may err in.
            assert 0.05 < law.residual_sd_log2[rate][projection] < 0.25


def test_the_prediction_lands_inside_its_own_declared_model_error(population):
    """Leave one stack out, predict its two rungs, check against the truth.

    The claim being tested is the one the row will carry: the stack TOTAL's
    error is the common-mode intercept term, so the realised |log2| error must
    sit inside a few of those, not inside the per-expert residual.
    """
    for held_out, stack in sorted(population.items()):
        law = fit_stack_transfer_law(population, hold_out=held_out)
        result = predict_stack_rates(stack, law)
        assert sorted(result["predicted"]) == list(TARGETS)
        assert result["intercept_sample_size"] == 9
        for rate in TARGETS:
            truth = _truth_total(stack, rate)
            error = abs(math.log2(result["predicted"][rate] / truth))
            declared = result["model_error"][rate]["stack_total_sd_log2"]
            assert declared > 0.0
            assert error <= 3.0 * declared, (
                f"{held_out} rung {rate}: |log2 pred/true| {error} exceeds "
                f"three declared model sd ({declared})")
            # And the prediction is actually good, not merely inside a
            # generous bar: 9 experts fit this stack's own intercept, whose
            # standard error is the planted 0.05 log2 over sqrt(9).
            assert error <= 0.06, (
                f"{held_out} rung {rate}: |log2 pred/true| {error}")


def test_the_model_error_names_itself_and_denies_being_a_sampling_error(population):
    """A reader, and a gate, must not be able to mistake it for an HT stderr."""
    law = fit_stack_transfer_law(population, hold_out="s:layer3")
    result = predict_stack_rates(population["s:layer3"], law)
    for rate in TARGETS:
        error = result["model_error"][rate]
        assert error["kind"] == "transfer_law_model_error_log2"
        assert error["is_sampling_error"] is False
        assert set(error["per_expert_residual_sd_log2"]) == set(PROJECTIONS)
        # The common-mode term is the per-expert one shrunk by sqrt(n): it is
        # a different number, and smaller, for every stack in this fixture.
        assert error["stack_total_sd_log2"] < min(
            error["per_expert_residual_sd_log2"].values())
        # Reported, never applied.
        assert error["smearing_factor_not_applied"] > 1.0
    assert "dloss_stderr" not in result
    assert "estimator" not in result


def test_a_law_refuses_to_predict_the_stack_it_pooled(population):
    """Fitting without a hold-out and then predicting a member is refused."""
    law = fit_stack_transfer_law(population)
    with pytest.raises(TesseraFormatError, match="pooled this stack's own experts"):
        predict_stack_rates(population["s:layer5"], law)
    other = fit_stack_transfer_law(population, hold_out="s:layer5")
    with pytest.raises(TesseraFormatError, match="held out"):
        predict_stack_rates(population["s:layer6"], other)


def test_a_stack_whose_reference_is_not_a_census_is_refused():
    """Every expert is predicted from its own reference value, so all are needed."""
    rng = random.Random(7)
    stack = _stack("s:layer0", base=_base(rng, 6), sampled=[0, 1, 2],
                   noise=0.01, rng=rng)
    partial = dict(stack.reference_mse)
    partial.pop(5)
    with pytest.raises(TesseraFormatError, match="must be a census"):
        StackRateSample(
            stack=stack.stack, projections=stack.projections,
            experts=stack.experts, reference_q256=REFERENCE,
            reference_mse=partial, sampled_experts=stack.sampled_experts,
            sampled_mse=stack.sampled_mse, currency=stack.currency)


def test_weights_reproduce_the_campaign_total_convention(population):
    """``sum_e w_e * sum_p mse`` -- the same product the HT path estimates."""
    stack = population["s:layer1"]
    weighted = StackRateSample(
        stack=stack.stack, projections=stack.projections,
        experts=stack.experts, reference_q256=REFERENCE,
        reference_mse=stack.reference_mse,
        sampled_experts=stack.sampled_experts, sampled_mse=stack.sampled_mse,
        weights={e: 2.0 + e for e in stack.experts}, currency=stack.currency)
    assert weighted.reference_total() == pytest.approx(math.fsum(
        (2.0 + e) * math.fsum(stack.reference_mse[e][p] for p in PROJECTIONS)
        for e in stack.experts))
    law = fit_stack_transfer_law(population, hold_out=stack.stack)
    plain = predict_stack_rates(stack, law)
    scaled = predict_stack_rates(weighted, law)
    for rate in TARGETS:
        # The per-expert predictions do not depend on the weights; only the
        # total does, and it does so exactly linearly.
        assert scaled["predicted_per_expert"][rate] == pytest.approx(
            plain["predicted_per_expert"][rate])
        assert scaled["predicted"][rate] == pytest.approx(math.fsum(
            (2.0 + e) * plain["predicted_per_expert"][rate][e]
            for e in stack.experts))
