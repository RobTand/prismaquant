"""Development identity drift does not waive authenticated acquisition controls."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_campaign_acquisition_pb_handoff import (
    handoff, real_joint_run, plan, records, dispatch, load_joint_campaign_acquisition,
)


def rewrite_request(handoff, change):
    document = json.loads(handoff.raw_request)
    change(document)
    raw = json.dumps(document, allow_nan=False).encode()
    handoff.request.write_bytes(raw)
    return {"path": str(handoff.request), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.mark.parametrize("field", ["domain_pins", "export_sha256", "grammar_sha256"])
def test_loader_stamps_only_recorded_producer_identity(handoff, monkeypatch, capsys, field):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    def change(document):
        if field == "domain_pins":
            document[field] = {**document[field], "tessera_git": "recorded-before-refresh"}
        else:
            document["producer_source_state"][field] = "0" * 64
    binding = rewrite_request(handoff, change)
    acquired = load_joint_campaign_acquisition(binding)
    assert acquired["requests"] == handoff.acquisition["requests"]
    assert acquired["source_weights"] == handoff.acquisition["source_weights"]
    assert "[DEV-MODE]" in capsys.readouterr().out
    handoff.acquisition = acquired
    merged = dispatch._merge_acquisition_rows(records(handoff),
        acquisition=acquired, scope_groups=handoff.census["anchor_groups"])
    assert merged["origin"]["request_sha256"] == binding["sha256"]
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(ValueError, match="domain_pins|producer_source_state"):
        load_joint_campaign_acquisition(binding)


def test_actual_cli_plans_acquisition_after_producer_refresh(handoff, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    binding = rewrite_request(handoff, lambda document:
        document["producer_source_state"].update(export_sha256="0" * 64))
    spec = json.loads(handoff.spec.read_text())
    spec["campaign_argv"][-1] = binding["sha256"]
    handoff.spec.write_text(json.dumps(spec))
    root = Path(dispatch.__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, str(root / "tools/dispatch_tessera_campaign.py"),
        "plan", "--spec", str(handoff.spec), "--workspace", str(handoff.workspace)],
        cwd=handoff.root, env=dict(os.environ), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "[DEV-MODE]" in completed.stdout
    planned = json.loads((handoff.workspace / "plan.json").read_text())
    assert planned["acquisition"]["binding"] == binding
    assert len(planned["rows"]) == 2
    assert len(planned["acquisition"]["deferred_groups"]) == 1


@pytest.mark.parametrize("relocate_cost", [False, True])
def test_manifest_accepts_authenticated_locator_aliases(handoff, monkeypatch, capsys, relocate_cost):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    planned, actions = plan(handoff)
    document = json.loads(handoff.raw_request)
    if relocate_cost:
        cost = handoff.root / "cost-alias.pkl"
        cost.write_bytes(handoff.raw_cost)
        document["cost_path"] = str(cost)
        raw = json.dumps(document, allow_nan=False).encode()
    else:
        raw = handoff.raw_request
    alias = handoff.root / "request-alias.json"
    alias.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    for action in actions:
        argv = action["argv"]
        argv[argv.index("--acquisition-request") + 1] = str(alias)
        argv[argv.index("--acquisition-request-sha256") + 1] = digest
    dispatch._check_acquisition_manifest(actions, planned, handoff.census, handoff.acquisition)
    assert "[DEV-MODE]" in capsys.readouterr().out
    assert dispatch._plan_acquisition(planned, handoff.census) == handoff.acquisition
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(dispatch.DemandRefused, match="locator|binding"):
        dispatch._check_acquisition_manifest(actions, planned, handoff.census, handoff.acquisition)


@pytest.mark.parametrize("change", ["missing", "digest", "request_content", "cost_content"])
def test_manifest_still_refuses_actual_control_disagreement(handoff, monkeypatch, change):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    planned, actions = plan(handoff)
    action = copy.deepcopy(actions[0])
    argv = action["argv"]
    if change == "missing":
        for flag in ("--acquisition-request", "--acquisition-request-sha256"):
            index = argv.index(flag)
            del argv[index:index + 2]
    elif change == "digest":
        argv[argv.index("--acquisition-request-sha256") + 1] = "0" * 64
    else:
        document = json.loads(handoff.raw_request)
        if change == "request_content":
            document["total_requested_quality_measurements"] += 1
        else:
            cost = handoff.root / "different-cost.pkl"
            cost.write_bytes(handoff.raw_cost + b"different owned bytes")
            document.update(cost_path=str(cost), cost_sha256=hashlib.sha256(cost.read_bytes()).hexdigest())
        raw = json.dumps(document).encode()
        request = handoff.root / "different-request.json"
        request.write_bytes(raw)
        argv[argv.index("--acquisition-request") + 1] = str(request)
        argv[argv.index("--acquisition-request-sha256") + 1] = hashlib.sha256(raw).hexdigest()
    with pytest.raises((dispatch.DemandRefused, ValueError), match="binding|control|identity mismatch"):
        dispatch._check_acquisition_manifest([action, *actions[1:]], planned,
            handoff.census, handoff.acquisition)


def test_seal_lint_scans_acquisition_input_owner():
    import test_no_new_seals as lint
    path = "prismaquant/tessera_acquisition_inputs.py"
    sources = lint._campaign_sources()
    assert path in sources, "the authenticated acquisition input owner is outside the seal ratchet"
    sources[path] += '\n\ndef injected_identity_wall(recorded_identity, running_identity):\n    if recorded_identity != running_identity:\n        raise ValueError("recorded producer drift")\n'
    assert any("injected_identity_wall" in problem for problem in lint.violations(sources))
