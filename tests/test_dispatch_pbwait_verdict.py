"""Native completion is exact-ID success, never a display table or early artifact."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import dispatch_joint_quanta as dispatch
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

def test_native_pilot_authority_fails_closed_without_loading_pb():
    from prismaquant.joint_dispatch_pilot import PilotRefused
    gateway = dispatch.Gateway()
    for method in (gateway.read_verified_result, gateway.bind_capture_command):
        with pytest.raises(PilotRefused, match="native .* authority is unavailable"):
            method(object())
