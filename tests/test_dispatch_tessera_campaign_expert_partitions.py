"""Real planner and merge coverage for opt-in unsampled expert partitions."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_tessera_campaign as dispatch  # pyright: ignore[reportMissingImports]
from test_tessera_campaign_fanout import _partition_workspace, _plan_args, _write_model, _shard, SCOPE  # pyright: ignore[reportMissingImports]
from test_tessera_campaign_expert_partitions import GROUP, roster, selection  # pyright: ignore[reportMissingImports]


@pytest.fixture
def planned(tmp_path, monkeypatch):
    from experiments import glm_data_manifests as manifests
    from prismaquant import tessera_expert_projection as projection
    monkeypatch.setattr(manifests, "SHARED_MOUNT", str(tmp_path))
    spec_path, workspace = _partition_workspace(tmp_path, wide=False)
    spec = json.loads(spec_path.read_text())
    spec["campaign_argv"] = ["--rate-band", "768,768", "--max-rounds", "1"]
    spec_path.write_text(json.dumps(spec))
    records, stacks = roster()
    monkeypatch.setattr(projection, "carried_units", lambda value: ({}, records, stacks))
    census = json.loads((workspace / "census.json").read_text())
    census["anchor_groups"] = {GROUP: sorted(records), "u:dense": ["dense"]}
    census["unit_shapes"] = {name: [8, 8] for name in [*records, "dense"]}
    census["expert_projection"] = {"synthetic_test_projection": True}
    _write_model(Path(spec["model"]), census["unit_shapes"])
    (workspace / "census.json").write_text(json.dumps(census))
    seen = []
    monkeypatch.setattr(dispatch, "_row_memory_gb",
                        lambda spec, members, census, **kw: seen.append(sorted(members)) or 5)
    return spec_path, workspace, seen


def test_real_plan_emits_distinct_rows_with_own_expert_demands(planned):
    spec, workspace, seen = planned
    assert dispatch.cmd_plan(_plan_args(spec, workspace, experts_per_row=2)) == 0
    plan = json.loads((workspace / "plan.json").read_text())
    routed = [row for row in plan["rows"] if GROUP in row["groups"]]
    assert len(routed) == 2, "whole routed group still cannot split across rows"
    assert len({row["row_id"] for row in routed}) == 2
    priced = [set(row["members"]) for row in routed]
    assert not priced[0] & priced[1]
    assert priced[0] | priced[1] == set(roster()[0])
    assert all(len(members) == 6 for members in priced)
    assert sorted(seen, key=str) == sorted([sorted(p) for p in priced] + [["dense"]], key=str)
    for row in routed:
        record = json.loads(Path(row["units"]).read_text())
        assert record["schema"] == "prismaquant.tessera_campaign_units.v3"
        assert record["groups"][0]["members"] == sorted(roster()[0])
        assert record["groups"][0]["partition"]["members"] == row["members"]
    actions = json.loads((workspace / "manifest.json").read_text())
    assert len(actions) == 3
    assert all(Path(action["data_manifest"]).is_file() for action in actions)


def test_partition_readsets_contain_only_chunk_captures_and_weight_bytes(planned, monkeypatch):
    spec, workspace, _seen = planned
    assert dispatch.cmd_plan(_plan_args(spec, workspace, experts_per_row=2)) == 0
    plan = json.loads((workspace / "plan.json").read_text())
    producer = dispatch._manifest_producer()
    # One-byte records make this tiny fixture's distinct tensor ranges visible;
    # production still rounds to the existing ZFS record size.
    monkeypatch.setattr(sys.modules[producer.Campaign.__module__], "RECORD_SIZE", 1)
    root = workspace / "captures"
    root.mkdir()
    names = [*roster()[0], "dense"]
    entries = {}
    for index, name in enumerate(names):
        path = root / f"capture-{index}.pt"
        path.write_bytes(b"capture")
        entries[name] = {"path": path.name}
    capture_manifest = root / "manifest.json"
    capture_manifest.write_text(json.dumps({"entries": entries}))
    with_captures = {**plan, "calibration_cache": {"path": str(capture_manifest)}}
    campaign = producer.Campaign(str(workspace), plan=with_captures)
    actions = json.loads((workspace / "manifest.json").read_text())
    argv_by_row = {producer.row_id_of(action): action["argv"] for action in actions}
    for row in plan["rows"]:
        if GROUP not in row["groups"]:
            continue
        expected = set(row["members"])
        assert set(campaign.members(row["row_id"])) == expected
        manifest = producer.build_manifest(campaign, row["row_id"], {}, argv_by_row[row["row_id"]])
        capture_paths = {entry["path"] for entry in manifest["entries"] if entry["path"].endswith(".pt")}
        assert capture_paths == {str(root / entries[name]["path"]) for name in expected}
        assert manifest["annotations"]["bytes"]["weight_extents"] == len(expected) * 8 * 8 * 2
        assert manifest["annotations"]["counts"]["captures"] == len(expected)


@pytest.mark.parametrize("argv,overrides", [
    (["--rate-band", "768,896", "--max-rounds", "1"], {}),
    (["--rate-band", "768,768", "--max-rounds", "2"], {}),
    (["--max-rounds", "1"], {}),
    (["--rate-band", "768,768", "--max-rounds", "1"], {"seed_checkpoint": "other"}),
])
def test_partition_plan_refuses_unsupported_contract_before_publication(planned, argv, overrides):
    spec_path, workspace, seen = planned
    spec = json.loads(spec_path.read_text())
    spec["campaign_argv"] = argv
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(RuntimeError, match="partition"):
        dispatch.cmd_plan(_plan_args(spec_path, workspace, experts_per_row=2, **overrides))
    assert not (workspace / "plan.json").exists()
    assert not seen


def test_merge_coverage_is_derived_from_all_planned_partitions(tmp_path):
    rows = []
    for index in range(2):
        path = tmp_path / f"row{index}.json"
        path.write_text(json.dumps(selection(index)))
        rows.append({"row_id": f"row{index}", "groups": [GROUP], "units": str(path),
                     "members": selection(index)["groups"][0]["partition"]["members"]})
    coverage = dispatch.declared_coverage({"rows": rows})
    assert coverage is not None, "partition coverage still inferred only from surviving payloads"
    assert coverage["expected_groups"] == [GROUP]
    assert coverage["expected_partitions"] == {row["row_id"]: json.loads(Path(row["units"]).read_text())
                                               for row in rows}


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
    merged = dispatch.merge_payloads(payloads, census=census,
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
    reference = dispatch.merge_payloads({"whole": whole}, census=census,
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
    with pytest.raises(dispatch.MergeRefused, match="partition"):
        dispatch.merge_payloads(payloads, census=census, capture_sha256="merged", plan_coverage=coverage)


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
    with pytest.raises(dispatch.MergeRefused, match="partition"):
        dispatch.merge_payloads(payloads, census=census, capture_sha256="merged",
                                plan_coverage=coverage)
