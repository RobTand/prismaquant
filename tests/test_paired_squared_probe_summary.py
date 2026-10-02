"""CPU scalar controls for covariance; no source or model-price admission."""
import math

import pytest

from prismaquant.joint_aura import paired_squared_probe_summary


def test_pair_retains_covariance_that_independent_standard_errors_lose():
    result = paired_squared_probe_summary([25., 16., 25., 16.], [9., 0., 9., 0.])
    assert result == dict(mean_difference=8., paired_standard_error=0.,
                          difference_per_probe=[8.] * 4)
    assert not any('identity' in key or 'currency' in key for key in result)


@pytest.mark.parametrize('a,b', [([], []), ([1.], [1.]), ([1., 2.], [1.]),
    ([1., 2.], [1., 2., 3.]), ([float('nan'), 2.], [1., 2.]),
    ([float('inf'), 2.], [1., 2.]), ([-1., 2.], [1., 2.]),
    ([True, 2.], [1., 2.]), ([1e308, 0.], [0., 0.])])
def test_raw_pair_refuses_missing_nonfinite_or_invalid_samples(a, b):
    with pytest.raises(ValueError):
        paired_squared_probe_summary(a, b)


@pytest.mark.parametrize('a,b', [([1., 4., 9., 16.], [0., 1., 4., 9.]),
    ([1e-200, 1e-100, 1., 1e100], [0., 1e-100, .5, 1e100]),
    ([.1, .2, .3, .4], [.3, .2, .1, .05])])
def test_extracted_owner_keeps_original_operation_order_exact(a, b):
    values = [.5 * (left - right) for left, right in zip(a, b)]
    mean = sum(values) / len(values)
    variance = sum((value - mean)**2 for value in values) / (len(values) - 1)
    old = dict(mean_difference=mean, paired_standard_error=math.sqrt(variance / len(values)),
               difference_per_probe=values)
    assert paired_squared_probe_summary(a, b) == old
