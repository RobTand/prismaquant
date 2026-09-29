"""Opt-in CPU planner checks; synthetic timings are not performance evidence."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_tessera_campaign as dispatch  # pyright: ignore[reportMissingImports]
from experiments import glm_data_manifests
from test_tessera_campaign_fanout import _plan_args, _write_model


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(glm_data_manifests, "SHARED_MOUNT", str(tmp_path))

    def make(dense=6, routed=2, startup=20.0, share=0.4, slots=2):
        model = tmp_path / "model"
        model.mkdir()
        groups = {f"u:dense-{i:03d}": [f"dense-{i}"] for i in range(dense)}
        # Include a fused group: its members must never be split.
        groups["g:fused"] = groups.pop("u:dense-000") + ["fused-sibling"]
        groups.update({f"s:stack-{i:03d}": [f"expert-{i}-0", f"expert-{i}-1"]
                       for i in range(routed)})
        shapes = {name: [8, 8] for members in groups.values() for name in members}
        _write_model(model, shapes)
        workspace = tmp_path / "campaign"
        workspace.mkdir()
        census = workspace / "census.json"
        census.write_text(json.dumps({"model": str(model), "anchor_groups": groups,
                                      "layer_stride": 1, "unit_shapes": shapes}))
        spec = tmp_path / "spec.json"
        spec.write_text(json.dumps({"model": str(model), "campaign_argv": [],
                                    "cwd": str(tmp_path), "python": "python3",
                                    "env": {}, "headroom_gb": 0}))
        profile = tmp_path / "work.json"
        profile.write_text(json.dumps({
            "schema": "prismaquant.campaign_row_work.v1",
            "census_sha256": hashlib.sha256(census.read_bytes()).hexdigest(),
            "campaign_argv": [], "startup_seconds": startup,
            "max_startup_fraction": share, "gpu_slots": slots,
            "group_pricing_seconds": {key: 10.0 for key in groups
                                      if not key.startswith("s:")},
            "evidence": {"startup": "synthetic-test", "pricing": "synthetic-test"},
        }))
        return spec, workspace, profile

    return make


def _snapshot(workspace):
    return {str(path.relative_to(workspace)): path.read_bytes()
            for path in workspace.rglob("*") if path.is_file()}


def _manifest_by_id(workspace):
    rows = json.loads((workspace / "manifest.json").read_text())
    return {Path(row["argv"][row["argv"].index("--units") + 1]).stem: row
            for row in rows}


@pytest.mark.parametrize("dense,routed,startup,share", [
    (6, 2, 20.0, 0.4), (90, 42, 88.3, 0.2),
])
def test_measured_objective_packs_dense_and_keeps_routed_bytes(
        campaign, dense, routed, startup, share):
    spec, workspace, profile = campaign(dense, routed, startup, share)
    assert dispatch.cmd_plan(_plan_args(spec, workspace)) == 0
    legacy = _snapshot(workspace)
    legacy_plan = json.loads(legacy["plan.json"])
    legacy_manifest = _manifest_by_id(workspace)
    routed_ids = {entry["row_id"] for entry in legacy_plan["rows"]
                  if entry["groups"][0].startswith("s:")}

    args = _plan_args(spec, workspace, row_work_profile=profile)
    assert dispatch.cmd_plan(args) == 0
    plan = json.loads((workspace / "plan.json").read_text())
    packed = [row for row in plan["rows"] if not row["groups"][0].startswith("s:")]
    assert len(packed) == 2, "the opt-in profile was ignored; dense rows still pay startup"
    assert all(row["predicted_work"]["startup_fraction"] <= share for row in packed)
    covered = [key for row in plan["rows"] for key in row["groups"]]
    assert sorted(covered) == sorted(json.loads((workspace / "census.json").read_text())["anchor_groups"])
    assert len(covered) == len(set(covered))
    census_groups = json.loads((workspace / "census.json").read_text())["anchor_groups"]
    members = []
    for row in plan["rows"]:
        selection = json.loads(Path(row["units"]).read_text())
        for group in selection["groups"]:
            assert group["members"] == sorted(census_groups[group["key"]])
            members.extend(group["members"])
    assert sorted(members) == sorted(name for group in census_groups.values() for name in group)
    assert len(members) == len(set(members))
    assert plan["row_work_profile"]["sha256"] == hashlib.sha256(profile.read_bytes()).hexdigest()
    manifest = _manifest_by_id(workspace)
    for row_id in routed_ids:
        assert manifest[row_id] == legacy_manifest[row_id]
        for relative in (f"units/{row_id}.json",
                         f"data-manifests/{row_id}.data-manifest.json"):
            assert (workspace / relative).read_bytes() == legacy[relative]
    before = _snapshot(workspace)
    assert dispatch.cmd_plan(args) == 0
    assert _snapshot(workspace) == before
    # The existing verifier still re-derives the packed row's demand/read set.
    assert dispatch.main(["check", "--workspace", str(workspace)]) == 0


@pytest.mark.parametrize("field,value,message", [
    ("schema", "other", "schema"),
    ("census_sha256", "0" * 64, "census"),
    ("campaign_argv", ["--rates", "other"], "campaign_argv"),
    ("startup_seconds", 0, "startup_seconds"),
    ("startup_seconds", True, "startup_seconds"),
    ("startup_seconds", float("nan"), "startup_seconds"),
    ("max_startup_fraction", 1.0, "max_startup_fraction"),
    ("gpu_slots", 0, "gpu_slots"),
    ("gpu_slots", 1.5, "gpu_slots"),
    ("group_pricing_seconds", {}, "group_pricing_seconds"),
    ("evidence", {}, "evidence"),
])
def test_invalid_or_unbound_profile_preserves_existing_plan(campaign, field, value, message):
    spec, workspace, profile = campaign()
    assert dispatch.cmd_plan(_plan_args(spec, workspace)) == 0
    before = _snapshot(workspace)
    record = json.loads(profile.read_text())
    record[field] = value
    profile.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match=message):
        dispatch.cmd_plan(_plan_args(spec, workspace, row_work_profile=profile))
    assert _snapshot(workspace) == before


@pytest.mark.parametrize("slots,expected", [(1, 3), (2, 2)])
def test_slot_envelope_changes_full_wave_count(campaign, slots, expected):
    spec, workspace, profile = campaign(dense=9, routed=0, slots=slots)
    assert dispatch.cmd_plan(_plan_args(spec, workspace, row_work_profile=profile)) == 0
    plan = json.loads((workspace / "plan.json").read_text())
    assert len(plan["rows"]) == expected
    assert all(row["predicted_work"]["startup_fraction"] <= 0.4 for row in plan["rows"])


def test_heterogeneous_work_reduces_row_count_to_meet_the_objective(campaign):
    spec, workspace, profile = campaign(dense=3, routed=0, share=0.5)
    record = json.loads(profile.read_text())
    record["group_pricing_seconds"] = dict(zip(sorted(record["group_pricing_seconds"]),
                                               [80.0, 5.0, 5.0]))
    profile.write_text(json.dumps(record))
    assert dispatch.cmd_plan(_plan_args(spec, workspace, row_work_profile=profile)) == 0
    plan = json.loads((workspace / "plan.json").read_text())
    assert len(plan["rows"]) == 1
    assert plan["rows"][0]["predicted_work"]["pricing_seconds"] == 90.0
    assert plan["rows"][0]["predicted_work"]["startup_fraction"] <= 0.5


@pytest.mark.parametrize("value", [0, -1, True, "10", float("nan"), float("inf")])
def test_invalid_group_prediction_preserves_existing_plan(campaign, value):
    spec, workspace, profile = campaign()
    assert dispatch.cmd_plan(_plan_args(spec, workspace)) == 0
    before = _snapshot(workspace)
    record = json.loads(profile.read_text())
    record["group_pricing_seconds"]["g:fused"] = value
    profile.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="group_pricing_seconds"):
        dispatch.cmd_plan(_plan_args(spec, workspace, row_work_profile=profile))
    assert _snapshot(workspace) == before


@pytest.mark.parametrize("extra", ["unknown", "s:stack-000"])
def test_extra_group_prediction_preserves_existing_plan(campaign, extra):
    spec, workspace, profile = campaign()
    assert dispatch.cmd_plan(_plan_args(spec, workspace)) == 0
    before = _snapshot(workspace)
    record = json.loads(profile.read_text())
    record["group_pricing_seconds"][extra] = 10.0
    profile.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="group_pricing_seconds"):
        dispatch.cmd_plan(_plan_args(spec, workspace, row_work_profile=profile))
    assert _snapshot(workspace) == before


def test_nonfinite_aggregate_prediction_preserves_existing_plan(campaign):
    spec, workspace, profile = campaign()
    assert dispatch.cmd_plan(_plan_args(spec, workspace)) == 0
    before = _snapshot(workspace)
    record = json.loads(profile.read_text())
    record["group_pricing_seconds"] = {key: 1e308 for key in record["group_pricing_seconds"]}
    profile.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="group_pricing_seconds total is nonfinite"):
        dispatch.cmd_plan(_plan_args(spec, workspace, row_work_profile=profile))
    assert _snapshot(workspace) == before


def test_count_two_still_uses_sorted_contiguous_bundles(campaign):
    spec, workspace, _ = campaign()
    assert dispatch.cmd_plan(_plan_args(spec, workspace, groups_per_row=2)) == 0
    plan = json.loads((workspace / "plan.json").read_text())
    groups = sorted(json.loads((workspace / "census.json").read_text())["anchor_groups"])
    assert [row["groups"] for row in plan["rows"]] == [groups[i:i + 2] for i in range(0, len(groups), 2)]
    assert "row_work_profile" not in plan
    assert all("predicted_work" not in row for row in plan["rows"])


def test_not_enough_work_refuses_instead_of_claiming_the_objective(campaign):
    spec, workspace, profile = campaign(startup=1000.0)
    with pytest.raises(RuntimeError, match="insufficient dense work"):
        dispatch.cmd_plan(_plan_args(spec, workspace, row_work_profile=profile))
    assert not (workspace / "manifest.json").exists()


def test_profile_does_not_combine_with_count_packing(campaign):
    spec, workspace, profile = campaign()
    with pytest.raises(RuntimeError, match="groups-per-row"):
        dispatch.cmd_plan(_plan_args(spec, workspace, row_work_profile=profile,
                                     groups_per_row=2))
    assert not (workspace / "manifest.json").exists()


def test_memory_fit_still_declines_packed_rows_without_shrinking_demand(campaign, monkeypatch):
    spec, workspace, profile = campaign()
    record = json.loads(spec.read_text())
    record["box_memory_gb"] = 104
    spec.write_text(json.dumps(record))
    seen = []

    def demand(spec, members, census, **kwargs):
        seen.append(list(members))
        return 70 if len(members) > 2 else 5

    monkeypatch.setattr(dispatch, "_row_memory_gb", demand)
    assert dispatch.cmd_plan(_plan_args(spec, workspace, row_work_profile=profile)) == 0
    plan = json.loads((workspace / "plan.json").read_text())
    assert len(plan["inadmissible_rows"]) == 2
    assert {row["mem_gb"] for row in plan["inadmissible_rows"]} == {70}
    assert len(seen) == 4
    assert len(_manifest_by_id(workspace)) == 2
