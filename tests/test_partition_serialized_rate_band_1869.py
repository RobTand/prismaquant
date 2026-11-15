"""Check partition merges and rate-band inputs at the merge boundary."""
from prismaquant import tessera_campaign_table as campaign_table
import json
import copy
from test_tessera_campaign_fanout import _shard, SCOPE
from test_tessera_campaign_expert_partitions import GROUP, roster, selection
import pytest
from prismaquant.tessera_campaign_table import parse_rate_band

@pytest.mark.parametrize('band', [[768, 768], (768, 768)])
def test_typed_band_preserves_exact_rate(band):
    assert parse_rate_band(band) == (768, 768)

@pytest.mark.parametrize('representation', ['json-list', 'tuple'])
def test_real_merge_accepts_serialized_rate_band(partition_payloads, representation):
    payloads, census, coverage = partition_payloads
    for payload in payloads.values():
        band = (768, 768)
        payload['provenance']['rate_band'] = json.loads(json.dumps(band)) if representation == 'json-list' else band
    merged = campaign_table.merge_payloads(payloads, census=census, capture_sha256='merged', plan_coverage=coverage)
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


@pytest.fixture
def partition_payloads(monkeypatch):
    from prismaquant import tessera_expert_projection as projection
    records, stacks = roster()
    monkeypatch.setattr(projection, "carried_units", lambda value: ({}, records, stacks))
    names = sorted(records)
    scope = {**copy.deepcopy(SCOPE), "dense_targets": [], "dense_all": [],
             "expert_targets": names, "anchor_groups": {GROUP: names},
             "calibration_census": {"counts": dict.fromkeys(names, 16384),
                                    "token_count": 16384, "token_count_min": 16384}}
    census = {"anchor_groups": {GROUP: names}, "expert_projection": {},
              "counts": dict.fromkeys(names, 16384)}
    selections = {f"row{i}": selection(i) for i in range(2)}
    payloads = {}
    for row_id, units in selections.items():
        priced = units["groups"][0]["partition"]["members"]
        shards = [_shard(GROUP, name) for name in priced]
        payload = copy.deepcopy(shards[0])
        for field in ("costs", "leave_one_anchor_out", "menu_sizes", "anchor_counts"):
            payload[field] = {name: shard[field][name] for name, shard in zip(priced, shards)}
        for name, formats in payload["costs"].items():
            payload["costs"][name] = {"TESSERA_E4M3_K1_R768": next(iter(formats.values()))}
        payload["formats"] = ["TESSERA_E4M3_K1_R768"]
        payload["provenance"].update({"unit_selection": {**units, "selected": True},
            "campaign_scope": scope, "rate_band": "768,768", "max_rounds": 1,
            "anchor_groups": {GROUP: priced},
            "surfaces": {name: shard["provenance"]["surfaces"][name]
                         for name, shard in zip(priced, shards)}})
        payloads[row_id] = payload
    coverage = {"expected_groups": [GROUP], "excluded_rows": [],
                "reason": "fixed-rate expert partitions", "expected_partitions": selections}
    return payloads, census, coverage


def test_partition_merge_matches_whole_group_cost_union(partition_payloads):
    payloads, census, coverage = partition_payloads
    merged = campaign_table.merge_payloads(payloads, census=census,
                                     capture_sha256="merged", plan_coverage=coverage)
    whole = copy.deepcopy(payloads["row0"])
    for field in ("costs", "leave_one_anchor_out", "menu_sizes", "anchor_counts"):
        whole[field] = {name: value for payload in payloads.values()
                        for name, value in payload[field].items()}
    whole["provenance"]["surfaces"] = {name: value for payload in payloads.values()
                                        for name, value in payload["provenance"]["surfaces"].items()}
    whole["provenance"]["anchor_groups"] = {GROUP: sorted(roster()[0])}
    whole["provenance"]["unit_selection"] = {
        "schema": "prismaquant.tessera_campaign_units.v1", "selected": True,
        "groups": [{"key": GROUP, "members": sorted(roster()[0])}]}
    reference = campaign_table.merge_payloads({"whole": whole}, census=census,
                                        capture_sha256="merged")
    for field in ("costs", "leave_one_anchor_out", "menu_sizes", "anchor_counts", "formats"):
        assert merged[field] == reference[field]
    assert merged["provenance"]["anchor_groups"] == {GROUP: sorted(roster()[0])}
    assert merged["provenance"]["unit_selection"] == reference["provenance"]["unit_selection"]


@pytest.mark.parametrize("damage", ["missing", "overlap", "changed", "no_plan", "cost_gap", "anchor_gap", "seed"])
def test_partition_merge_refuses_coverage_or_identity_gap(partition_payloads, damage):
    payloads, census, coverage = partition_payloads
    payloads, coverage = copy.deepcopy(payloads), copy.deepcopy(coverage)
    if damage == "missing":
        payloads.pop("row1")
    elif damage == "overlap":
        coverage["expected_partitions"]["row1"] = selection(0)
    elif damage == "changed":
        payloads["row1"]["provenance"]["unit_selection"]["groups"][0]["partition"]["index"] = 0
    elif damage == "no_plan":
        coverage = None
    elif damage == "cost_gap":
        payloads["row1"]["costs"].pop(next(iter(payloads["row1"]["costs"])))
    elif damage == "anchor_gap":
        groups = payloads["row1"]["provenance"]["anchor_groups"]
        groups[GROUP] = groups[GROUP][:-1]
    elif damage == "seed":
        payloads["row1"]["provenance"]["seed_checkpoint"] = {"path": "foreign"}
    with pytest.raises(campaign_table.MergeRefused, match="partition"):
        campaign_table.merge_payloads(payloads, census=census, capture_sha256="merged", plan_coverage=coverage)


@pytest.mark.parametrize("damage", ["anchor_key", "wrong_rate", "empty_prices"])
def test_partition_merge_checks_actual_group_and_priced_rung(partition_payloads, damage):
    payloads, census, coverage = partition_payloads
    row = payloads["row0"]
    if damage == "anchor_key":
        row["provenance"]["anchor_groups"] = {"s:wrong": row["provenance"]["anchor_groups"][GROUP]}
    elif damage == "wrong_rate":
        row["costs"] = {name: {"TESSERA_E4M3_K1_R1024": next(iter(formats.values()))}
                        for name, formats in row["costs"].items()}
        row["formats"] = ["TESSERA_E4M3_K1_R1024"]
    else:
        row["costs"] = dict.fromkeys(row["costs"], {})
    with pytest.raises(campaign_table.MergeRefused, match="partition"):
        campaign_table.merge_payloads(payloads, census=census, capture_sha256="merged",
                                plan_coverage=coverage)
