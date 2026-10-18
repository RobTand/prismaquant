"""Each overlapping public submit must hand off its own checked snapshot."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import campaign_data_manifests
from test_tessera_campaign_fanout import _partition_workspace, _plan_args

import dispatch_tessera_campaign as dispatch


@pytest.mark.parametrize("change_demand", [True, False],
                         ids=["distinct-checked-snapshot", "identical-retry"])
def test_overlapping_submit_keeps_its_checked_manifest(tmp_path, monkeypatch,
                                                      change_demand):
    monkeypatch.setattr(campaign_data_manifests, "SHARED_MOUNT", str(tmp_path))
    spec, workspace = _partition_workspace(tmp_path, wide=False,
                                           box_memory_gb=None)
    assert dispatch.cmd_plan(_plan_args(spec, workspace)) == 0
    manifest = workspace / "manifest.json"
    original = json.loads(manifest.read_text())
    assert len(original) == 1
    second = copy.deepcopy(original)
    if change_demand:
        second[0]["demand"]["mem_gb"] += 1
    args = SimpleNamespace(workspace=workspace, wait_s=7)
    checked = []
    real_verify = dispatch.verify_manifest_demands

    def observe_real_check(spec, census, rows, **kwargs):
        records = real_verify(spec, census, rows, **kwargs)
        checked.append(copy.deepcopy(rows))
        return records

    monkeypatch.setattr(dispatch, "verify_manifest_demands", observe_real_check)
    entered = []
    consumed = {}
    refused = []

    def delayed_external_reader(path, *, wait_s, receipts=None):
        ordinal = len(entered)
        entered.append(Path(path))
        assert wait_s == 7
        assert receipts == workspace / "receipts.json"
        if ordinal == 0:
            # The first client has not opened its submitted pathname yet.
            # A second real submit validates another ordinary editable request
            # and publishes before that first external read proceeds.
            replacement = manifest.with_suffix(".replacement.json")
            replacement.write_text(json.dumps(second) + "\n")
            replacement.replace(manifest)
            try:
                assert dispatch.cmd_submit(args) == 0
            except RuntimeError as error:
                # A named differing-publication refusal is also a safe result.
                assert change_demand
                assert "manifest" in str(error).lower()
                assert "differ" in str(error).lower()
                refused.append(str(error))
        consumed[ordinal] = json.loads(Path(path).read_text())
        return 0

    # Only the final external PB boundary is replaced. Planning, source proof,
    # demand admission, readset derivation and manifest publication stay real.
    monkeypatch.setattr(dispatch, "_pbcampaign", delayed_external_reader)
    assert dispatch.cmd_submit(args) == 0
    assert checked == [original, second]
    assert consumed[0] == original, (
        "the first client reopened a submitted pathname overwritten by the "
        "second submit and consumed a different checked request")
    if refused:
        assert len(entered) == 1
        assert "fresh workspace" in refused[0].lower()
    else:
        assert len(entered) == 2
        assert consumed[1] == second
    assert json.loads(manifest.read_text()) == second
