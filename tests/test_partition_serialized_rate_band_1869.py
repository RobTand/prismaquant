"""Actual producer-shaped rate-band provenance at the host merge boundary."""
import json
import pytest
from prismaquant.tessera_campaign import parse_rate_band
from test_dispatch_tessera_campaign_expert_partitions import dispatch, partition_payloads  # pyright: ignore[reportMissingImports]

@pytest.mark.parametrize('band', [[768, 768], (768, 768)])
def test_typed_band_preserves_exact_rate(band):
    assert parse_rate_band(band) == (768, 768)

@pytest.mark.parametrize('representation', ['json-list', 'tuple'])
def test_real_merge_accepts_serialized_rate_band(partition_payloads, representation):
    payloads, census, coverage = partition_payloads
    for payload in payloads.values():
        band = (768, 768)
        payload['provenance']['rate_band'] = json.loads(json.dumps(band)) if representation == 'json-list' else band
    merged = dispatch.merge_payloads(payloads, census=census, capture_sha256='merged', plan_coverage=coverage)
    assert set(merged['costs']) == {name for p in payloads.values() for name in p['costs']}
    assert all(set(prices) == {'TESSERA_E4M3_K1_R768'} for prices in merged['costs'].values())
    assert merged['provenance']['coverage']['priced_groups'] == 1

@pytest.mark.parametrize('band', [[], [768], [768,768,768], [True,True], [768.0,768.0],
    ['768','768'], [0,768], [-1,768], [896,768], [768,None]])
def test_invalid_typed_band_refuses(band):
    with pytest.raises(RuntimeError):
        parse_rate_band(band)

@pytest.mark.parametrize('text,expected', [('768,768',(768,768)), (' 768, 896 ',(768,896)), ('',None), (None,None)])
def test_cli_contract_unchanged(text, expected):
    assert parse_rate_band(text) == expected
