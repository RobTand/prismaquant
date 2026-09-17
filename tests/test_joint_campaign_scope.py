"""The joint pass's scope is an identity, not a tally.

``load_measured_anchor_input`` already refuses a plan whose measured journals
do not cover the plan's own cells, but the plan's cells are whatever its bound
census says they are: a plan whose census is a coherently narrowed roster
verifies cleanly and is still not the campaign.  These are the regressions for
the acceptance that closes that gap -- the roster and the window set are
compared by identity, and the campaign identity is fixed outside the plan.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from tools.dispatch_tessera_campaign import (
    CAMPAIGN_IDENTITY_SCHEMA, CAMPAIGN_SCOPE_SCHEMA, COMPLETE_CAMPAIGN_SCOPE,
    DIAGNOSTIC_SCOPE, PANEL_SCHEMA, PANEL_STATUS, ScopeRefused,
    campaign_identity, joint_campaign_scope, verify_joint_campaign_scope)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bind(root: Path, units, groups, *, nsamples=512, seqlen=512, draw=None,
          calibration=b"calibration tokens", capture=b'{"schema": "capture"}'):
    """A census, campaign plan and calibration bound the way a joint plan binds them."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    census_path = root / "census.json"
    census = {"schema": "prismaquant.tessera_campaign_census.v1",
              "model": "/fixture/model",
              # The draw: which windows, over which corpus revision and which
              # tokenizer ids. Two censuses that differ here are two
              # calibrations, however equal their rosters and window counts.
              "text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64,
              "seed": 0, "layer_stride": 1,
              "nsamples": nsamples, "seqlen": seqlen,
              "unit_shapes": {name: [16, 16] for name in units},
              "anchor_groups": {group: sorted(members)
                                for group, members in groups.items()}}
    census.update(draw or {})
    census_path.write_text(json.dumps(census, sort_keys=True))
    campaign_path = root / "campaign-plan.json"
    campaign = {"schema": "prismaquant.tessera_campaign_plan.v1",
                "census": str(census_path),
                "rows": [{"row_id": "row-0", "members": sorted(units),
                          "groups": sorted(groups)}]}
    campaign_path.write_text(json.dumps(campaign, sort_keys=True))
    calibration_path = root / "calibration_tokens.safetensors"
    calibration_path.write_bytes(calibration)
    capture_path = root / "capture_manifest.json"
    capture_path.write_bytes(capture)
    checkpoint_path = root / "cost.anchors.json"
    checkpoint_path.write_text(json.dumps(
        {"identity": {"units": {name: {"menu": ["BF16", "TESSERA_E4M3"]}
                                for name in sorted(units)}}}, sort_keys=True))
    return {"census": {"path": str(census_path), "sha256": _sha(census_path)},
            "campaign_plan": {"path": str(campaign_path),
                              "sha256": _sha(campaign_path)},
            "calibration_input": {"path": str(calibration_path),
                                  "sha256": _sha(calibration_path)},
            "canonical_capture": {"path": str(capture_path),
                                  "sha256": _sha(capture_path)},
            "merged_checkpoint": {"path": str(checkpoint_path),
                                  "sha256": _sha(checkpoint_path)}}


def _plan(inputs, *, windows=512, seqlen=512, panel=None, units=None, groups=None):
    # The calibration and the capture are plan-level bindings, not ``inputs``
    # entries; the roster and the checkpoint are ``inputs``.
    inputs = dict(inputs)
    calibration_input = inputs.pop("calibration_input")
    canonical_capture = inputs.pop("canonical_capture")
    if units is None or groups is None:
        census = json.loads(Path(inputs["census"]["path"]).read_text())
        units = len(census["unit_shapes"]) if units is None else units
        groups = len(census["anchor_groups"]) if groups is None else groups
    plan = {"schema": "prismaquant.tessera_joint_aura.plan.v1",
            "calibration_input": calibration_input,
            "canonical_capture": canonical_capture,
            "inputs": {**inputs, "required_source_units": units,
                       "required_campaign_groups": groups},
            "execution": {"calib_seqlen": seqlen, "n_calib_samples": windows}}
    if panel is not None:
        plan["joint_eval"] = panel
    return plan


def _panel(size=16, seqlen=512):
    return {"schema": PANEL_SCHEMA, "status": PANEL_STATUS,
            "selection": {"algorithm": "python_random_permutation_prefix_v1",
                          "seed": 237, "size": size,
                          "indices": list(range(size))},
            "shape": [size, seqlen], "eval_ids_sha256": "0" * 64}


def test_the_scope_is_derived_from_the_bound_roster_not_a_declared_count(tmp_path):
    inputs = _bind(tmp_path, ["a", "b", "c"], {"g0": ["a", "b"], "g1": ["c"]})
    scope = joint_campaign_scope(_plan(inputs))
    assert scope["schema"] == CAMPAIGN_SCOPE_SCHEMA
    assert scope["kind"] == COMPLETE_CAMPAIGN_SCOPE
    assert scope["source_unit_count"] == 3
    assert scope["campaign_group_count"] == 2
    assert scope["window_count"] == 512
    assert scope["campaign_window_count"] == 512
    assert scope["calib_seqlen"] == 512
    assert scope["selection_sha256"] is None
    assert scope["source_roster_sha256"] == canonical_json_sha256(
        sorted(["a", "b", "c"]), where="roster")
    # The draw, the capture and the candidate roster are part of the scope,
    # not of the plan's self-description.
    assert scope["calibration_sha256"] == canonical_json_sha256({
        "census_draw": {"model": "/fixture/model", "text_sha256": "a" * 64,
                        "fit_ids_sha256": "b" * 64, "seed": 0, "layer_stride": 1},
        "window_count": 512, "calib_seqlen": 512,
        "calibration_input_sha256": inputs["calibration_input"]["sha256"],
        "canonical_capture_sha256": inputs["canonical_capture"]["sha256"],
    }, where="calibration")
    assert scope["campaign_checkpoint_sha256"] == \
        inputs["merged_checkpoint"]["sha256"]


def test_an_equal_size_calibration_swap_is_a_different_campaign(tmp_path):
    """Same roster, same window count, different bytes: a different draw."""
    campaign = campaign_identity(joint_campaign_scope(_plan(_bind(
        tmp_path / "campaign", ["a", "b", "c"], {"g0": ["a", "b"], "g1": ["c"]}))))
    swapped = _plan(_bind(tmp_path / "swapped", ["a", "b", "c"],
                          {"g0": ["a", "b"], "g1": ["c"]},
                          calibration=b"other calibration tokens"))
    scope = joint_campaign_scope(swapped)
    assert scope["source_roster_sha256"] == campaign["source_roster_sha256"]
    assert scope["source_unit_count"] == campaign["source_unit_count"]
    with pytest.raises(ScopeRefused, match="calibration_sha256"):
        verify_joint_campaign_scope(
            swapped, require_scope=COMPLETE_CAMPAIGN_SCOPE, campaign=campaign)


def test_a_reduced_candidate_roster_is_a_different_campaign(tmp_path):
    """The candidate (unit, format) roster is bound, not just its unit count."""
    inputs = _bind(tmp_path / "campaign", ["a", "b", "c"],
                   {"g0": ["a", "b"], "g1": ["c"]})
    campaign = campaign_identity(joint_campaign_scope(_plan(inputs)))
    checkpoint = Path(inputs["merged_checkpoint"]["path"])
    document = json.loads(checkpoint.read_text())
    for entry in document["identity"]["units"].values():
        entry["menu"] = entry["menu"][:1]        # same units, fewer rungs
    checkpoint.write_text(json.dumps(document, sort_keys=True))
    reduced = {**inputs,
               "merged_checkpoint": {"path": str(checkpoint),
                                     "sha256": _sha(checkpoint)}}
    scope = joint_campaign_scope(_plan(reduced))
    assert scope["source_unit_count"] == campaign["source_unit_count"]
    assert scope["source_roster_sha256"] == campaign["source_roster_sha256"]
    with pytest.raises(ScopeRefused, match="campaign_checkpoint_sha256"):
        verify_joint_campaign_scope(
            _plan(reduced), require_scope=COMPLETE_CAMPAIGN_SCOPE, campaign=campaign)


def test_an_unidentified_draw_refuses_rather_than_pricing_another_one(tmp_path):
    """A census with no corpus/tokenizer identity is not a calibration."""
    inputs = _bind(tmp_path, ["a", "b", "c"], {"g0": ["a", "b"], "g1": ["c"]})
    census_path = Path(inputs["census"]["path"])
    census = json.loads(census_path.read_text())
    del census["text_sha256"]
    census_path.write_text(json.dumps(census, sort_keys=True))
    rebound = {**inputs, "census": {"path": str(census_path),
                                    "sha256": _sha(census_path)}}
    with pytest.raises(ScopeRefused, match="text_sha256"):
        joint_campaign_scope(_plan(rebound))


def test_an_identity_sealed_before_the_draw_binding_refuses_to_stand(tmp_path):
    """A pre-existing six-field identity says so, rather than comparing None."""
    inputs = _bind(tmp_path, ["a", "b", "c"], {"g0": ["a", "b"], "g1": ["c"]})
    scope = joint_campaign_scope(_plan(inputs))
    stale = {key: value for key, value in campaign_identity(scope).items()
             if key not in ("calibration_sha256", "campaign_checkpoint_sha256")}
    with pytest.raises(ScopeRefused, match="must be re-sealed"):
        verify_joint_campaign_scope(
            _plan(inputs), require_scope=COMPLETE_CAMPAIGN_SCOPE, campaign=stale)


def test_a_roster_of_equal_length_with_different_members_is_a_different_campaign(tmp_path):
    """The count is not the acceptance: the members are."""
    campaign = campaign_identity(joint_campaign_scope(_plan(_bind(
        tmp_path / "campaign", ["a", "b", "c"], {"g0": ["a", "b"], "g1": ["c"]}))))
    other = _plan(_bind(tmp_path / "other", ["a", "b", "d"],
                        {"g0": ["a", "b"], "g1": ["d"]}))
    assert joint_campaign_scope(other)["source_unit_count"] == 3          # same tally
    with pytest.raises(ScopeRefused, match="source_roster_sha256"):
        verify_joint_campaign_scope(
            other, require_scope=COMPLETE_CAMPAIGN_SCOPE, campaign=campaign)


def test_a_narrowed_window_set_without_a_frozen_panel_refuses(tmp_path):
    """Narrowing ``n_calib_samples`` is a narrower scope, not the campaign's."""
    inputs = _bind(tmp_path, ["a", "b", "c"], {"g0": ["a", "b"], "g1": ["c"]})
    campaign = campaign_identity(joint_campaign_scope(_plan(inputs)))
    with pytest.raises(ScopeRefused, match="evaluates all 512 windows"):
        joint_campaign_scope(_plan(inputs, windows=16))
    with pytest.raises(ScopeRefused, match="evaluates all 512 windows"):
        verify_joint_campaign_scope(_plan(inputs, windows=16),
                                    require_scope=COMPLETE_CAMPAIGN_SCOPE,
                                    campaign=campaign)


def test_a_diagnostic_subset_is_explicit_and_is_not_the_campaign(tmp_path):
    inputs = _bind(tmp_path, ["a", "b", "c"], {"g0": ["a", "b"], "g1": ["c"]})
    full = joint_campaign_scope(_plan(inputs))
    campaign = campaign_identity(full)
    panel = _panel()
    pilot = _plan(inputs, windows=512, panel=panel)
    scope = joint_campaign_scope(pilot)
    assert scope["kind"] == DIAGNOSTIC_SCOPE
    assert scope["window_count"] == 16 and scope["campaign_window_count"] == 512
    assert scope["selection_sha256"] == canonical_json_sha256(panel, where="panel")
    # It can be submitted as the diagnostic it is ...
    assert verify_joint_campaign_scope(
        pilot, require_scope=DIAGNOSTIC_SCOPE, campaign=campaign) == scope
    # ... and never as the campaign's score.
    with pytest.raises(ScopeRefused, match="requires a complete_campaign scope"):
        verify_joint_campaign_scope(
            pilot, require_scope=COMPLETE_CAMPAIGN_SCOPE, campaign=campaign)


def test_a_narrowed_census_cannot_read_as_the_campaign(tmp_path):
    """A plan is self-consistent with whatever census it binds."""
    campaign = campaign_identity(joint_campaign_scope(_plan(_bind(
        tmp_path / "campaign", ["a", "b", "c", "d"],
        {"g0": ["a", "b"], "g1": ["c", "d"]}))))
    narrowed = _plan(_bind(tmp_path / "narrow", ["a", "b"],
                           {"g0": ["a", "b"]}, nsamples=128),
                     windows=128, units=2, groups=1)
    assert joint_campaign_scope(narrowed)["kind"] == COMPLETE_CAMPAIGN_SCOPE
    with pytest.raises(ScopeRefused, match="source_unit_count"):
        verify_joint_campaign_scope(
            narrowed, require_scope=COMPLETE_CAMPAIGN_SCOPE, campaign=campaign)


def test_a_missing_campaign_identity_refuses(tmp_path):
    inputs = _bind(tmp_path, ["a", "b", "c"], {"g0": ["a", "b"], "g1": ["c"]})
    with pytest.raises(ScopeRefused, match=CAMPAIGN_IDENTITY_SCHEMA):
        verify_joint_campaign_scope(_plan(inputs),
                                    require_scope=COMPLETE_CAMPAIGN_SCOPE,
                                    campaign=None)


def test_the_unknown_scope_refuses(tmp_path):
    inputs = _bind(tmp_path, ["a", "b", "c"], {"g0": ["a", "b"], "g1": ["c"]})
    with pytest.raises(ScopeRefused, match="unknown required scope"):
        verify_joint_campaign_scope(
            _plan(inputs), require_scope="whatever",
            campaign=campaign_identity(joint_campaign_scope(_plan(inputs))))


def test_the_frozen_identity_names_the_campaign_not_one_subset(tmp_path):
    inputs = _bind(tmp_path, ["a", "b", "c"], {"g0": ["a", "b"], "g1": ["c"]})
    record = campaign_identity(joint_campaign_scope(_plan(inputs)))
    assert record["schema"] == CAMPAIGN_IDENTITY_SCHEMA
    assert set(record) == {"schema", "source_unit_count", "source_roster_sha256",
                           "campaign_group_count", "campaign_group_roster_sha256",
                           "campaign_window_count", "calib_seqlen",
                           "calibration_sha256", "campaign_checkpoint_sha256"}


def test_a_rebound_census_is_refused_before_the_roster_is_read(tmp_path):
    inputs = _bind(tmp_path, ["a", "b", "c"], {"g0": ["a", "b"], "g1": ["c"]})
    plan = _plan(inputs)
    Path(inputs["census"]["path"]).write_text("{}")
    with pytest.raises(ScopeRefused, match="which hashes to"):
        joint_campaign_scope(plan)
