"""Public bootstrap uncertainty must survive power-of-two sample counts.

The reference range follows Var(mean of n draws with replacement) =
population_variance / n. It does not reimplement or constrain the random
generator.
"""
from __future__ import annotations

import math

import pytest

from prismaquant.saturation_select import bootstrap_mean_ci


@pytest.mark.parametrize("count", [2, 4, 8], ids=["two-samples", "four-samples", "eight-samples"])
def test_nonconstant_bootstrap_has_the_known_sampling_scale(count):
    samples = list(range(count))
    mean, stderr = bootstrap_mean_ci(samples, n_boot=4096, seed=17)
    expected_mean = math.fsum(samples) / count
    population_variance = math.fsum((value - expected_mean) ** 2 for value in samples) / count
    expected_stderr = math.sqrt(population_variance / count)
    assert mean == expected_mean
    assert 0.75 * expected_stderr < stderr < 1.25 * expected_stderr, (
        f"nonconstant n={count} bootstrap SE {stderr} is outside the broad "
        f"statistical range around {expected_stderr}")


def test_constant_bootstrap_has_zero_uncertainty():
    assert bootstrap_mean_ci([3.25] * 8, n_boot=1024, seed=17) == (3.25, 0.0)


def test_bootstrap_is_repeatable_without_prescribing_rng():
    samples = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    first = bootstrap_mean_ci(samples, n_boot=1024, seed=29)
    second = bootstrap_mean_ci(samples, n_boot=1024, seed=29)
    assert first == second


def test_bootstrap_keeps_process_random_state():
    import random

    before = random.getstate()
    bootstrap_mean_ci([0.0, 1.0], n_boot=128, seed=17)
    assert random.getstate() == before
