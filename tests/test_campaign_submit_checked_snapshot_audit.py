"""Submit the rows whose demand was checked, or refuse a changed manifest."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import campaign_data_manifests
from test_tessera_campaign_fanout import _partition_workspace, _plan_args

import dispatch_tessera_campaign as dispatch


@pytest.fixture
def planned(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign_data_manifests, "SHARED_MOUNT", str(tmp_path))
    spec, workspace = _partition_workspace(
        tmp_path, wide=False, box_memory_gb=None)
    assert dispatch.cmd_plan(_plan_args(spec, workspace)) == 0
    manifest = workspace / "manifest.json"
    rows = json.loads(manifest.read_text())
    assert len(rows) == 1
    assert rows[0]["demand"]["mem_gb"] > 1
    calls = []

    def record_submission(path, *, wait_s, receipts=None):
        calls.append({"path": Path(path), "rows": json.loads(Path(path).read_text()),
                      "wait_s": wait_s, "receipts": receipts})
        return 0

    monkeypatch.setattr(dispatch, "_pbcampaign", record_submission)
    args = SimpleNamespace(workspace=workspace, wait_s=7)
    return args, manifest, rows, calls


def _replace_demand(manifest, rows):
    changed = copy.deepcopy(rows)
    changed[0]["demand"]["mem_gb"] = 1
    replacement = manifest.with_suffix(".replacement.json")
    replacement.write_text(json.dumps(changed) + "\n")
    replacement.replace(manifest)
    return changed


def test_submit_uses_the_checked_rows_when_manifest_is_replaced(planned, monkeypatch):
    args, manifest, original, calls = planned
    real_verify = dispatch.verify_manifest_demands
    checked = []
    changed = []

    def replace_after_real_check(spec, census, rows, **kwargs):
        records = real_verify(spec, census, rows, **kwargs)
        checked.append(copy.deepcopy(rows))
        changed.append(_replace_demand(manifest, original))
        return records

    monkeypatch.setattr(dispatch, "verify_manifest_demands", replace_after_real_check)
    try:
        result = dispatch.cmd_submit(args)
    except dispatch.DemandRefused:
        assert checked == [original]
        assert calls == []
        return

    assert result == 0
    assert checked == [original]
    assert changed == [json.loads(manifest.read_text())]
    assert changed[0][0]["demand"]["mem_gb"] == 1
    assert len(calls) == 1
    assert calls[0]["rows"] == original, (
        "cmd_submit sent a different manifest snapshot from the rows whose "
        "demand passed the real validator")


def test_unchanged_checked_rows_reach_the_submission_boundary(planned):
    args, manifest, original, calls = planned
    assert dispatch.cmd_submit(args) == 0
    assert len(calls) == 1
    assert calls[0] == {"path": args.workspace / dispatch.SUBMITTED_MANIFEST,
                        "rows": original, "wait_s": 7,
                        "receipts": args.workspace / "receipts.json"}
    assert json.loads(manifest.read_text()) == original


def test_underdeclared_rows_are_refused_before_submission(planned):
    args, manifest, original, calls = planned
    _replace_demand(manifest, original)
    with pytest.raises(dispatch.DemandRefused, match="declared demand.mem_gb 1"):
        dispatch.cmd_submit(args)
    assert calls == []
    assert not (args.workspace / dispatch.SUBMITTED_MANIFEST).exists()


def test_check_uses_the_checked_rows_when_manifest_is_replaced(planned, monkeypatch):
    args, manifest, original, calls = planned
    real_verify = dispatch.verify_manifest_demands
    checked = []

    def replace_after_real_check(spec, census, rows, **kwargs):
        records = real_verify(spec, census, rows, **kwargs)
        checked.append(copy.deepcopy(rows))
        changed = copy.deepcopy(original)
        changed[0].pop("data_manifest")
        replacement = manifest.with_suffix(".replacement.json")
        replacement.write_text(json.dumps(changed) + "\n")
        replacement.replace(manifest)
        return records

    monkeypatch.setattr(dispatch, "verify_manifest_demands", replace_after_real_check)

    assert dispatch.cmd_check(args) == 0
    assert checked == [original]
    assert "data_manifest" not in json.loads(manifest.read_text())[0]
    assert calls == []


def test_submitted_manifest_publisher_accepts_a_relative_workspace(planned, monkeypatch):
    args, manifest, original, calls = planned
    workspace = args.workspace
    monkeypatch.chdir(workspace.parent)

    submitted = dispatch._publish_submitted_manifest(Path(workspace.name), original)

    assert submitted == workspace / dispatch.SUBMITTED_MANIFEST
    assert json.loads(submitted.read_text()) == original
    assert json.loads(manifest.read_text()) == original
    assert calls == []
