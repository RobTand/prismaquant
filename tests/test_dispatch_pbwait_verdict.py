"""Native completion is exact-ID success, never a display table or early artifact."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools import dispatch_joint_quanta as dispatch
from prismaquant import flux_jobs, tessera_campaign_rows

SAVED = {"job_id": "f123", "record_dir": "/records/rows/00000", "launch_sha256": "a" * 64}

@pytest.mark.parametrize("status,exit_code,error,admitted", [
    ("succeeded", 0, None, True), ("failed", 7, None, False),
    ("running", None, None, False), ("unknown", 0, "query rejected", False),
    ("canceled", None, None, False), ("dependency-failed", None, None, False),
    ("succeeded", 1, None, False)])
def test_native_gateway_requires_complete_success(monkeypatch, status, exit_code, error, admitted):
    monkeypatch.setattr(flux_jobs, "load_submission", lambda path: flux_jobs.FluxSubmission("f123", Path(path), "a" * 64))
    observed = []
    def collect(path, *, wait):
        observed.append((path, wait))
        return [{"status": status, "payload_exit": exit_code, "query_error": error}]
    monkeypatch.setattr(tessera_campaign_rows, "collect_rows", collect)
    assert dispatch.Gateway().is_terminal_executed(SAVED) is admitted
    assert observed == [(Path("/records"), False)]

def test_another_id_cannot_authorize_reuse(monkeypatch):
    monkeypatch.setattr(flux_jobs, "load_submission", lambda path: flux_jobs.FluxSubmission("other", Path(path), "a" * 64))
    monkeypatch.setattr(tessera_campaign_rows, "collect_rows", lambda *a, **k: pytest.fail("must refuse before collection"))
    result = dispatch.Gateway().status(SAVED)
    assert result["status"] == "unknown" and "identity differs" in result["query_error"]

def test_native_pilot_authority_reads_only_a_saved_native_execution(tmp_path):
    """A selection naming no saved native record refuses; the capture-command hold is gone."""
    from prismaquant import native_results
    from prismaquant.joint_dispatch_pilot import PilotRefused
    gateway = dispatch.Gateway()
    assert not hasattr(gateway, "bind_capture_command")
    publication = {"path": str(tmp_path / "publication.json"), "sha256": "b" * 64, "bytes": 1}
    selection = {"schema": native_results.SELECTION_SCHEMA, "job_id": "f123", "record_dir": str(tmp_path / "missing"),
                 "launch_sha256": "a" * 64, "publication": publication, "expected_launch": {},
                 "producer_id": "x", "inputs": {}, "runtime": {}, "max_publication_bytes": 1,
                 "max_evidence_bytes": 1, "max_artifact_bytes": 1}
    with pytest.raises(PilotRefused, match="native pilot result refused"):
        gateway.read_verified_result(dispatch.PilotSelector(selection))
