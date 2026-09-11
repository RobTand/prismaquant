"""A pricing row is bounded by whether it is committing anchors, not by a clock.

Two GLM-5.3 rows were killed at a 14,400 s limit this dispatcher sealed into
every row, while both were still journalling anchors -- 5,952 and 5,920 of
them durably on disk at the moment the worker signalled them (PrismaBuild
#480).  The seal is gone; what replaces it is a declaration of the phases a
row walks and the quiet it is allowed in each, and a report the campaign makes
after every journal flush.
"""
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_tessera_campaign as dispatch

from prismaquant import prismabuild_progress


def spec():
    return {
        "model": "/mnt/shared/model", "cwd": "/original/checkout",
        "python": "python3", "campaign_argv": [],
        "env": {"PYTHONPATH": ".:/producer/src"},
        "container": {"image": "qualified:fixed", "mounts": [
            {"source": "/mnt/shared", "target": "/mnt/shared"},
            {"source": "/producer", "target": "/producer", "readonly": True},
        ]},
    }


# -- the seal --------------------------------------------------------------

def test_a_row_seals_no_deadline_of_its_own(tmp_path):
    row = dispatch._row(spec(), ["--units", "/mnt/shared/units.json"],
                        mem_gb=48, timeout_s=None)
    assert "timeout_s" not in row
    assert row["progress_phases"] == ["startup=3600", "pricing=900",
                                      "finalize=1800"]


def test_an_explicitly_asked_for_deadline_is_still_sealed():
    row = dispatch._row(spec(), [], mem_gb=48, timeout_s=86400)
    assert row["timeout_s"] == 86400
    # Both, and they compose: the hard deadline caps the cost, the phases end
    # a row that stops early.
    assert row["progress_phases"][1] == "pricing=900"


def test_nonpricing_rows_do_not_promise_anchor_progress():
    """Census/capture exit before the pricing journal reporter is available."""

    row = dispatch._row(spec(), [], mem_gb=48, timeout_s=7200,
                        progress_phases=())
    assert row["timeout_s"] == 7200
    assert "progress_phases" not in row


def test_census_and_capture_keep_their_deadlines_without_anchor_progress(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({
        **spec(), "model": str(model), "cwd": str(tmp_path),
    }))
    args = SimpleNamespace(spec=str(spec_path), workspace=str(tmp_path),
                           timeout_s=7200, submit=False)

    assert dispatch.cmd_census(args) == 0
    census_row = json.loads((tmp_path / "census-manifest.json").read_text())[0]
    assert census_row["timeout_s"] == 7200
    assert "progress_phases" not in census_row

    (tmp_path / "census.json").write_text(json.dumps({
        "model": str(model), "counts": {}, "unit_shapes": {},
    }))
    assert dispatch.cmd_capture(args) == 0
    capture_row = json.loads((tmp_path / "capture-manifest.json").read_text())[0]
    assert capture_row["timeout_s"] == 7200
    assert "progress_phases" not in capture_row


def test_the_declared_quiet_is_shorter_than_the_limit_that_killed_the_rows():
    """The bound got *tighter*; what changed is what it is a bound on."""

    total = sum(grace for _, grace in dispatch.CAMPAIGN_PROGRESS_PHASES)
    assert total == 6300
    assert total < 14400


def test_the_pricing_allowance_covers_the_measured_commit_cadence():
    """18.4 s per committed batch, fitted over 23 completed 864-unit rows."""

    pricing = dict(dispatch.CAMPAIGN_PROGRESS_PHASES)["pricing"]
    assert pricing > 40 * 18.4


# -- the report ------------------------------------------------------------

def test_the_reporter_writes_the_record_prismabuild_accepts(tmp_path, monkeypatch):
    path = tmp_path / "key.progress"
    monkeypatch.setenv(prismabuild_progress.PATH_ENV, str(path))
    monkeypatch.setenv(prismabuild_progress.TOKEN_ENV, "minted-token")
    assert prismabuild_progress.report("pricing", 5920) is True
    record = json.loads(path.read_text())
    assert record["schema"] == "prismabuild.action_progress.v1"
    assert record["token"] == "minted-token"
    assert record["phase"] == "pricing"
    assert record["units_completed"] == 5920
    assert record["unit"] == "anchors"
    assert isinstance(record["reported_unix"], float)


def test_a_row_that_was_not_admitted_under_the_contract_reports_nothing(monkeypatch):
    monkeypatch.delenv(prismabuild_progress.PATH_ENV, raising=False)
    monkeypatch.delenv(prismabuild_progress.TOKEN_ENV, raising=False)
    assert prismabuild_progress.report("pricing", 1) is False


def test_an_unwritable_destination_does_not_fail_the_row(tmp_path, monkeypatch):
    monkeypatch.setenv(prismabuild_progress.PATH_ENV,
                       str(tmp_path / "absent" / "key.progress"))
    monkeypatch.setenv(prismabuild_progress.TOKEN_ENV, "t")
    assert prismabuild_progress.report("pricing", 1) is False


# -- the channel into the container ---------------------------------------

def test_the_container_carries_the_progress_channel(monkeypatch):
    runner = importlib.import_module("tools.tessera_campaign_container")
    monkeypatch.setenv(prismabuild_progress.PATH_ENV,
                       "/mnt/shared/prismabuild-fleet/pb-queue/claimed/k.progress")
    monkeypatch.setenv(prismabuild_progress.TOKEN_ENV, "minted-token")
    argv = runner.docker_command(spec(), ["python3"], cwd="/snapshot", uid=1,
                                 gid=1, image_id="sha256:x", environ=os.environ)
    assert (f"{prismabuild_progress.PATH_ENV}="
            "/mnt/shared/prismabuild-fleet/pb-queue/claimed/k.progress") in argv
    assert f"{prismabuild_progress.TOKEN_ENV}=minted-token" in argv


def test_a_progress_file_outside_a_writable_mount_is_refused(monkeypatch):
    """Silence the worker cannot tell from a stall is worth failing at launch."""

    runner = importlib.import_module("tools.tessera_campaign_container")
    monkeypatch.setenv(prismabuild_progress.PATH_ENV, "/var/queue/k.progress")
    monkeypatch.setenv(prismabuild_progress.TOKEN_ENV, "t")
    with pytest.raises(RuntimeError, match="not inside any writable"):
        runner.docker_command(spec(), ["python3"], cwd="/snapshot", uid=1,
                              gid=1, image_id="sha256:x", environ=os.environ)


def test_a_readonly_mount_is_not_somewhere_to_report(monkeypatch):
    runner = importlib.import_module("tools.tessera_campaign_container")
    monkeypatch.setenv(prismabuild_progress.PATH_ENV, "/producer/k.progress")
    monkeypatch.setenv(prismabuild_progress.TOKEN_ENV, "t")
    with pytest.raises(RuntimeError, match="not inside any writable"):
        runner.docker_command(spec(), ["python3"], cwd="/snapshot", uid=1,
                              gid=1, image_id="sha256:x", environ=os.environ)


def test_a_row_without_the_contract_adds_no_container_environment(monkeypatch):
    runner = importlib.import_module("tools.tessera_campaign_container")
    monkeypatch.delenv(prismabuild_progress.PATH_ENV, raising=False)
    monkeypatch.delenv(prismabuild_progress.TOKEN_ENV, raising=False)
    argv = runner.docker_command(spec(), ["python3"], cwd="/snapshot", uid=1,
                                 gid=1, image_id="sha256:x", environ=os.environ)
    assert not [entry for entry in argv if entry.startswith("PRISMABUILD_")]
