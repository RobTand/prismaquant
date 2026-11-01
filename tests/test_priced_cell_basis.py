from prismaquant.priced_cell import CostPricingPolicy, price_cell


def test_aggregated_anchored_zero_keeps_its_measured_provenance():
    cell = price_cell({'h_trace': 4.0}, {
        'cost_source': 'production_arm_render',
        'cost_currency': 'aura_predicted_dloss',
        'fisher_application_count': 1,
        'activation_pricing_applied': True,
        'predicted_dloss': 0.0,
    }, format_name='NVFP4', policy=CostPricingPolicy(0.0, False))
    assert cell.predicted_dloss == 0.0
    assert cell.activation_branch == 'weight_only_activation_calibrated'
    assert not cell.unmeasured_activation_at_zero
