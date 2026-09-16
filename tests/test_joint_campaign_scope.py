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


def _bind(root: Path, units, groups, *, nsamples=512, seqlen=512):
    """A census and campaign plan bound the way a joint plan binds them."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    census_path = root / "census.json"
    census = {"schema": "prismaquant.tessera_campaign_census.v1",
              "nsamples": nsamples, "seqlen": seqlen,
              "unit_shapes": {name: [16, 16] for name in units},
              "anchor_groups": {group: sorted(members)
                                for group, members in groups.items()}}
    census_path.write_text(json.dumps(census, sort_keys=True))
    campaign_path = root / "campaign-plan.json"
    campaign = {"schema": "prismaquant.tessera_campaign_plan.v1",
                "census": str(census_path),
                "rows": [{"row_id": "row-0", "members": sorted(units),
                          "groups": sorted(groups)}]}
    campaign_path.write_text(json.dumps(campaign, sort_keys=True))
    return {"census": {"path": str(census_path), "sha256": _sha(census_path)},
            "campaign_plan": {"path": str(campaign_path),
                              "sha256": _sha(campaign_path)}}


def _plan(inputs, *, windows=512, seqlen=512, panel=None, units=None, groups=None):
    if units is None or groups is None:
        census = json.loads(Path(inputs["census"]["path"]).read_text())
        units = len(census["unit_shapes"]) if units is None else units
        groups = len(census["anchor_groups"]) if groups is None else groups
    plan = {"schema": "prismaquant.tessera_joint_aura.plan.v1",
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
                           "campaign_window_count", "calib_seqlen"}


def test_a_rebound_census_is_refused_before_the_roster_is_read(tmp_path):
    inputs = _bind(tmp_path, ["a", "b", "c"], {"g0": ["a", "b"], "g1": ["c"]})
    plan = _plan(inputs)
    Path(inputs["census"]["path"]).write_text("{}")
    with pytest.raises(ScopeRefused, match="which hashes to"):
        joint_campaign_scope(plan)
