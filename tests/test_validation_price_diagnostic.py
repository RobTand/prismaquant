import pytest
from prismaquant.validate_assignments_kl import _assignment_cost_summary


def test_a_row_without_predicted_dloss_is_unknown_instead_of_a_zero_price():
    summary = _assignment_cost_summary(
        {'u': {'NVFP4': {'output_mse': 3.0, 'output_mse_measured': True}}},
        {'u': 'NVFP4'}, stats={'u': {}})
    assert summary['predicted_dloss_sum'] is None
    assert summary['price_missing_count'] == 1


def test_diagnostic_is_independent_of_unrecorded_operator_environment(monkeypatch):
    monkeypatch.setenv('PRISMAQUANT_COST_UCB_Z', '99')
    summary = _assignment_cost_summary(
        {'u': {'FP8_E4M3': {'predicted_dloss': 3.0, 'predicted_dloss_stderr': 0.5,
                            'output_mse': 3.0, 'output_mse_measured': True}}},
        {'u': 'FP8_E4M3'}, stats={'u': {}})
    assert summary['predicted_dloss_sum'] == 3.0
    assert summary['predicted_dloss_basis'] == 'unadjusted_cost_row_estimate'


def test_unknown_activation_at_the_optimal_zero_price_withholds_the_estimate():
    summary = _assignment_cost_summary(
        {'u': {'NVFP4': {'predicted_dloss': 0.0, 'weight_mse': 0.0,
                        'output_mse': 0.0, 'output_mse_measured': False}}},
        {'u': 'NVFP4'}, stats={'u': {}})
    assert summary['predicted_dloss_sum'] is None
    assert summary['price_missing_count'] == 1
