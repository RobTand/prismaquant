"""Loaded phases bind policy separately from raw wire and source custody."""
from pathlib import Path

import pytest

from prismaquant import tessera_cell_measurement as cell_measurement
from prismaquant import tessera_campaign as campaign
from prismaquant.tessera_campaign_cli import parse_campaign_arguments
from prismaquant.tessera_campaign_requests import (
    LoadedPricingRequest, ResidentCampaignWeights,
    SelectedCampaignWeights, loaded_campaign_request,
)


def _bound(extra=(), *, selected=False, stream=False):
    args = parse_campaign_arguments(['--model', 'fixture', '--out', 'cost.pkl',
        '--cache-dir', 'cache', '--menu-mode', 'research', '--campaign-identity-threads', '17', *extra])
    args.calibration_input_receipt = None
    return args, loaded_campaign_request(args, mode='research', selected_source=selected,
        streaming_head=stream, wire_dir=Path('cache/wire'), checkpoint=Path('anchors.json'),
        identity_threads=17)


def test_loaded_pricing_keeps_explicit_zero_and_false_controls_and_raw_wire_separate():
    args, request = _bound(['--seed=0', '--anchor-budget=0', '--deadline-seconds=0',
        '--publication-overlap-bytes=0', '--campaign-identity-bytes=0'])
    assert isinstance(request, LoadedPricingRequest)
    assert isinstance(request.source, ResidentCampaignWeights)
    assert request.recipe.anchor_budget == request.publication.deadline_seconds == 0
    assert request.publication.overlap_bytes == request.encoding.identity_bytes == 0
    assert request.recipe.placement.exhaustive_rate_grid is False
    assert request.publication.requested_identity_threads == 17
    assert request.seed is None and request.selection_requested is False
    assert args.seed == 0 and vars(args)['campaign_identity_threads'] == 17


@pytest.mark.parametrize('stream', [False, True])
def test_selected_request_carries_resolved_head_and_actual_capture_contract(stream):
    import json
    from test_tessera_row_stream import LOAD_POLICY, REFERENCE_POLICY

    extra = ['--streaming', '--units', 'selection.json', '--max-rounds', '1',
        '--calibration-cache', 'capture.json', '--calibration-cache-sha256', 'a'*64,
        '--calibration-census', 'census.json', '--row-head', 'stream' if stream else 'load-all',
        '--capture-load-policy', json.dumps(LOAD_POLICY),
        '--export-hessian-reference-policy', json.dumps(REFERENCE_POLICY)]
    if not stream:
        extra.extend(['--seed-checkpoint', 'seed.json', '--seed-wire-dir', 'seed-wires'])
    args, request = _bound(extra, selected=True, stream=stream)
    assert isinstance(request.source, SelectedCampaignWeights)
    assert request.source.capture == 'capture.json' and request.source.capture_sha256 == 'a'*64
    assert request.source.census == 'census.json'
    assert request.source.snapshot_policy == 'whole-layer-v1'
    assert request.source.head == args.row_head == ('stream' if stream else 'load-all')
    assert request.source.load_policy == LOAD_POLICY
    assert request.outputs.hessian_reference_policy == REFERENCE_POLICY
    if stream:
        assert request.seed is None
    else:
        assert request.seed.checkpoint == 'seed.json' and request.seed.wire_dir == 'seed-wires'


@pytest.mark.parametrize('hessian', ['require', 'off'])
def test_actual_loaded_census_keeps_shared_constructors_without_creating_h_source(
        monkeypatch, tmp_path, hessian):
    from test_tessera_canonical_census import fixture
    from prismaquant.production_weight_cache import ProductionWeightCache

    actual, argv, _model, _contract, _calls, output = fixture(monkeypatch, tmp_path)
    argv[argv.index('--hessian')+1] = hessian
    events = []
    real_memo = cell_measurement.encoder_kwargs_memo
    def memo(source, *args, **kwargs):
        assert source is None
        events.append('memo')
        return real_memo(source, *args, **kwargs)
    real_init = ProductionWeightCache.__init__
    def initialize(self, *args, **kwargs):
        events.append('cache')
        return real_init(self, *args, **kwargs)
    real_census = actual.run_census
    def publish(request, **kwargs):
        events.append('census')
        assert not kwargs['expert_targets']
        return real_census(request, **kwargs)
    def unexpected_h_source(*args, **kwargs):
        pytest.fail('census created an H-aware producer source')
    monkeypatch.setattr(cell_measurement, "encoder_kwargs_memo", memo)
    monkeypatch.setattr(ProductionWeightCache, '__init__', initialize)
    monkeypatch.setattr(actual, 'run_census', publish)
    monkeypatch.setattr(actual.th, 'activation_source', unexpected_h_source)
    assert actual.main([*argv, '--attention-implementation', 'eager']) == 0
    assert events == ['memo', 'cache', 'census']
    assert output.exists() and not (tmp_path/'cost.pkl').exists()
    assert not (tmp_path/'campaign.anchors.json').exists()


def test_loaded_operation_has_all_actual_global_bindings():
    from tools.verify_campaign_operation_globals import unbound_global_reads

    assert unbound_global_reads(campaign.run_loaded_campaign) == ()
