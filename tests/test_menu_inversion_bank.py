"""Mutation-sensitive controls for the explicitly historical raw-bank screen."""
import copy
import math

import pytest

from experiments.alloc_lead_asd.analyze_menu_inversion_bank import FAMILIES, RATES, UNITS, inspect_bank


def _bank():
    bank = {'units': {}}
    for unit in UNITS:
        formats = {}
        for family in FAMILIES:
            for rate in RATES:
                signed = [1., 2., 3., 4.]
                squared = [x*x for x in signed]
                mean = sum(squared) / 4
                formats[f'{family}_R{rate}'] = dict(
                    signed_per_probe=signed, x2_per_probe=squared,
                    predicted_dloss=.5*mean,
                    predicted_dloss_stderr=.5*math.sqrt(sum((x-mean)**2 for x in squared)/3/4),
                    probe_ids=[7000, 7001, 7002, 7003], probe_identity={'historical_stub': True},
                    probe_identity_sha256='a'*64, hessian_identity={'historical_stub': True},
                    joint_operator_identity=dict(qname=unit, format=f'{family}_R{rate}',
                        probe_identity_sha256='a'*64, source_weight={'historical_stub': unit},
                        activation={'historical_stub': family}, arithmetic='<dict len=19>'))
        bank['units'][unit] = formats
    return bank


def test_screen_keeps_published_field_equality_separate_from_source_admission():
    result = inspect_bank(_bank())
    assert set(result) == set(UNITS)
    for families in result.values():
        for report in families.values():
            assert report['published_identity_fields_equal'] is True
            assert len(report['comparisons']) == 6
            for comparison in report['comparisons']:
                assert comparison['paired_standard_error'] == 0
                assert comparison['descriptive_paired_signal_to_se'] is None
                assert comparison['positive_on_every_supplied_probe'] is False
                assert 'cost_currency' not in comparison


@pytest.mark.parametrize('mutation', ['probe_ids', 'source_weight', 'hessian', 'coordinate',
                                      'signed', 'mean', 'missing_unit', 'duplicate_probe'])
def test_screen_refuses_changed_alignment_or_published_samples(mutation):
    bank = _bank()
    row = bank['units'][UNITS[1]][f'{FAMILIES[0]}_R960']
    if mutation == 'probe_ids':
        row['probe_ids'] = [7001, 7000, 7002, 7003]
    elif mutation == 'source_weight':
        row['joint_operator_identity']['source_weight'] = {'foreign': True}
    elif mutation == 'hessian':
        row['hessian_identity'] = {'foreign': True}
    elif mutation == 'coordinate':
        row['joint_operator_identity']['qname'] = UNITS[0]
    elif mutation == 'signed':
        row['signed_per_probe'][0] = float('nan')
    elif mutation == 'mean':
        row['predicted_dloss'] += 1
    elif mutation == 'missing_unit':
        del bank['units'][UNITS[0]]
    elif mutation == 'duplicate_probe':
        row['x2_per_probe'] = row['x2_per_probe'] + [row['x2_per_probe'][0]]
    with pytest.raises(ValueError):
        inspect_bank(bank)
