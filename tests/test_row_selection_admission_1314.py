"""Check the existing whole-group contract before admitting a campaign row."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_tessera_campaign as dispatch  # pyright: ignore[reportMissingImports]

V1 = "prismaquant.tessera_campaign_units.v1"
V2 = "prismaquant.tessera_campaign_units.v2"
STACK = "s:stack"
MEMBERS = [f"expert.{expert}.{role}" for expert in range(2)
           for role in ("gate", "up", "down")]


@pytest.fixture
def row_case(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    selection_path = workspace / "units.json"
    census = {"anchor_groups": {STACK: MEMBERS, "g:dense": ["dense"]},
              "unit_shapes": {name: [8, 4 + index % 3]
                              for index, name in enumerate([*MEMBERS, "dense"])}}
    spec = {"model": str(tmp_path / "model"), "campaign_argv": [],
            "cwd": str(tmp_path), "python": "python3", "env": {},
            "process_baseline_bytes": 0, "headroom_gb": 0,
            "max_act_rows": 32, "box_memory_gb": 64}
    row = {"argv": ["python3", "-m", "prismaquant.tessera_campaign",
                    "--model", spec["model"], "--units", str(selection_path),
                    "--out", str(workspace / "cost.pkl"),
                    "--max-rounds", "1", "--rate-band", "832,1088"],
           "demand": {"mem_gb": 8}}
    model_reads = []
    monkeypatch.setattr(dispatch, "_model_bytes",
                        lambda model: model_reads.append(model) or 256)

    def publish(selection):
        selection_path.write_text(json.dumps(selection))
        (workspace / "census.json").write_text(json.dumps(census))
        (workspace / "spec.json").write_text(json.dumps(spec))
        (workspace / "manifest.json").write_text(json.dumps([row]))
        return SimpleNamespace(workspace=workspace, spec=workspace / "spec.json",
                               census=workspace / "census.json", manifest=None,
                               box_memory_gb=64, wait_s=1)

    return spec, census, row, model_reads, publish


def selection(*entries, schema=V1):
    return {"schema": schema, "groups": list(entries)}


def whole_group():
    return {"key": STACK, "members": list(MEMBERS)}


@pytest.mark.parametrize("band", ["832,1088", "896,896"])
def test_undeclared_partial_group_is_refused_before_memory_math(row_case, band):
    spec, census, row, reads, publish = row_case
    row["argv"][-1] = band
    publish(selection({"key": STACK, "members": MEMBERS[:3]}))
    with pytest.raises(dispatch.DemandRefused, match="invalid unit selection"):
        dispatch.verify_row_demand(spec, census, row, label="slice-a")
    assert reads == []


@pytest.mark.parametrize("bad", [
    selection({"key": "s:unknown", "members": list(MEMBERS)}),
    selection({"key": STACK, "members": [*MEMBERS[:-1], "dense"]}),
    selection(whole_group(), whole_group()),
    selection(whole_group(), schema="prismaquant.tessera_campaign_units.v999"),
    selection({**whole_group(), "sampled": MEMBERS[:3]}),
])
def test_invalid_group_contract_is_refused_before_memory_math(row_case, bad):
    spec, census, row, reads, publish = row_case
    publish(bad)
    with pytest.raises(dispatch.DemandRefused, match="invalid unit selection"):
        dispatch.verify_row_demand(spec, census, row, label="slice-a")
    assert reads == []


def test_valid_whole_groups_keep_their_derived_demand(row_case):
    spec, census, row, reads, publish = row_case
    publish(selection(whole_group(), {"key": "g:dense", "members": ["dense"]}))
    result = dispatch.verify_row_demand(spec, census, row, label="whole")
    expected = dispatch._row_memory_demand(spec, [*MEMBERS, "dense"], census)
    assert {key: result[key] for key in expected} == expected
    assert reads == [spec["model"], spec["model"]]


def test_schema_valid_sample_keeps_own_member_demand_and_whole_group_identity(row_case):
    """Check schema/group demand, not adoption of a persisted packed draw.

    This control omits ``stack_samples``. The downstream runtime's
    ``_validate_stack_sample`` check is outside this admission slice.
    """
    spec, census, row, reads, publish = row_case
    sampled = MEMBERS[:3]
    publish(selection({**whole_group(), "sampled": sampled,
                       "inclusion_probability": {name: 0.5 for name in sampled}},
                      schema=V2))
    result = dispatch.verify_row_demand(spec, census, row)
    expected = dispatch._row_memory_demand(spec, sampled, census)
    assert {key: result[key] for key in expected} == expected
    whole = dispatch._row_memory_demand(spec, MEMBERS, census)
    assert result["plan_bytes"] < whole["plan_bytes"]


@pytest.mark.parametrize("command", [dispatch.cmd_check, dispatch.cmd_submit])
def test_check_and_submit_refuse_before_manifest_or_submission_work(row_case,
                                                                  monkeypatch, command):
    _, _, _, reads, publish = row_case
    args = publish(selection({"key": STACK, "members": MEMBERS[:3]}))
    called = []
    monkeypatch.setattr(dispatch, "require_data_manifests", lambda *a, **k: called.append("check"))
    monkeypatch.setattr(dispatch, "attach_data_manifests",
                        lambda workspace, rows: called.append("attach") or rows)
    monkeypatch.setattr(dispatch, "_pbcampaign", lambda *a, **k: called.append("submit") or 0)
    before = {path.name: path.read_bytes() for path in args.workspace.iterdir()}
    with pytest.raises(dispatch.DemandRefused, match="invalid unit selection"):
        command(args)
    assert reads == []
    assert called == []
    assert {path.name: path.read_bytes() for path in args.workspace.iterdir()} == before
