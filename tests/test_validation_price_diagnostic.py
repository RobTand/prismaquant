import pytest
from prismaquant.validate_assignments_kl import _assignment_cost_summary


def test_render_score_placeholder_uses_measured_output_and_sensitivity():
    summary = _assignment_cost_summary(
        {'u': {'NVFP4': {'predicted_dloss': 0.0, 'output_mse': 3.0,
                        'weight_mse': 0.25, 'output_mse_measured': True}}},
        {'u': 'NVFP4'}, stats={'u': {'h_trace': 4.0}})
    assert summary['predicted_dloss_sum'] == 6.0
    assert summary['predicted_dloss_basis'] == 'unadjusted_cost_row_estimate'


def test_missing_sensitivity_is_unknown_instead_of_a_zero_price():
    summary = _assignment_cost_summary(
        {'u': {'NVFP4': {'output_mse': 3.0, 'output_mse_measured': True}}},
        {'u': 'NVFP4'}, stats={'u': {}})
    assert summary['predicted_dloss_sum'] is None
    assert summary['price_missing_count'] == 1


def test_diagnostic_is_independent_of_unrecorded_operator_environment(monkeypatch):
    monkeypatch.setenv('PRISMAQUANT_COST_UCB_Z', '99')
    monkeypatch.setenv('PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR', '1')
    summary = _assignment_cost_summary(
        {'u': {'FP8_E4M3': {'output_mse': 3.0, 'fisher_output_mse': 100.0,
                            'output_mse_measured': True}}},
        {'u': 'FP8_E4M3'}, stats={'u': {'h_trace': 4.0}})
    assert summary['predicted_dloss_sum'] == 6.0


def test_unknown_activation_at_the_optimal_zero_price_withholds_the_estimate():
    summary = _assignment_cost_summary(
        {'u': {'NVFP4': {'weight_mse': 0.0, 'output_mse': 0.0,
                        'output_mse_measured': False}}},
        {'u': 'NVFP4'}, stats={'u': {'h_trace': 4.0}})
    assert summary['predicted_dloss_sum'] is None
    assert summary['price_missing_count'] == 1
