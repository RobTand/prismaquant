"""Remaining #2774 interim-record admission, continuing #2535.

All authorizations below are SYNTHETIC intake fixtures. They exercise an
independently configured decision channel, NOT an actual CEO decision, flat
coefficient selection, scientific validation, or production rollout. #2972
must supply the real decision; #3013 owns affected-population enforcement.

Proposed extension of the retained c46b833e policy: authorized_interim_record
is a separately supplied trusted configuration input, never authority taken
from the price profile itself. A digest checks equality, not authenticity.
The CLI exposes that same input as --depth-price-authorization JSON_PATH.
Only an explicit decision selecting band_loss_multiplier.v1 is executable by
this contract. Other numerical meanings must stop rather than be guessed.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import sys

import pytest

from prismaquant import allocator, priced_cell
from prismaquant.joint_aura_rows import joint_identity_sha256 as digest
from prismaquant.layer_config import load_assignment
from test_routed_depth_pricing import (
    CONTRACT, EARLY, LATE, LOW, _bindings, _cli_fixture, _payload, _price, _record,
)

AUTHORIZATION_SCHEMA = "prismaquant.routed_depth_authorization.v1"


def _interim_intake_fixture(payload, *, multipliers=(0.25, 1.0)):
    profile = _record(payload, multipliers=multipliers)
    profile.pop("measurement_receipt")
    profile["state"] = "interim_flat"
    # Test-only out-of-band decision. No fixture is an authorization artifact
    # for real model/lane application and no trusted issuer is hard-coded.
    decision = {
        "schema": AUTHORIZATION_SCHEMA,
        "decision_ref": "synthetic-test-only:decision-1",
        "decided_by": "SYNTHETIC fixture decision-maker",
        "decided_at": "2026-01-01T00:00:00Z",
        **{key: copy.deepcopy(profile[key]) for key in (
            "scope", "bands", "calibration_contract", "parameters",
            "source_table_sha256", "band_roster_sha256", "parameters_sha256")},
    }
    profile["authorization_sha256"] = digest(decision)
    return profile, decision


def _interim_policy(payload, profile, decision, *, production=True):
    factory = getattr(priced_cell.CostPricingPolicy, "from_depth_profile", None)
    assert callable(factory), "#2774: adopt the existing #2535 policy intake, not a new allocator"
    assert "authorized_interim_record" in inspect.signature(factory).parameters, (
        "#2774: interim numeric decision must be an independent configuration input")
    return factory(payload, profile_record=profile, production=production,
                   authorized_interim_record=decision, ucb_z=0.0)


def test_interim_numeric_record_applies_exact_values_and_stays_unvalidated():
    payload, _, _ = _payload()
    profile, decision = _interim_intake_fixture(payload)
    before = copy.deepcopy((payload, profile, decision))
    policy = _interim_policy(payload, profile, decision)
    # These numbers exercise application only; they do not claim to repair
    # the actual affected model's routed-depth sensitivity.
    assert _price(payload, policy, EARLY[0]) == 0.625
    assert _price(payload, policy, LATE[0]) == 2.5
    metadata = policy.depth_profile
    assert metadata["state"] == "interim_flat"
    assert metadata["validated"] is False
    assert metadata["authorization_sha256"] == digest(decision)
    for key, value in _bindings(payload, profile).items():
        assert metadata[key] == value
    assert "qualification_receipts_sha256" not in metadata
    assert (payload, profile, decision) == before
    # The admitted decision and metadata must be owned snapshots.
    decision["parameters"]["early"]["loss_multiplier"] = 99
    profile["parameters"]["early"]["loss_multiplier"] = 99
    metadata["parameters"]["early"]["loss_multiplier"] = 99
    assert _price(payload, policy, EARLY[0]) == 0.625
    assert policy.depth_profile["parameters"]["early"]["loss_multiplier"] == 0.25


@pytest.mark.parametrize("production", [False, True])
@pytest.mark.parametrize("mutation", [
    "absent", "embedded_only", "digest", "changed_parameters", "scope",
    "roster", "source", "contract", "version", "missing_values",
    "missing_provenance", "nonfinite",
])
def test_interim_refuses_self_authorization_or_changed_decision(mutation, production):
    payload, _, _ = _payload()
    profile, decision = _interim_intake_fixture(payload)
    if mutation == "absent":
        decision = None
    elif mutation == "embedded_only":
        profile["authorization"] = copy.deepcopy(decision)
        decision = None
    elif mutation == "digest":
        profile["authorization_sha256"] = "f" * 64
    elif mutation == "changed_parameters":
        profile["parameters"]["early"]["loss_multiplier"] = 0.5
        # Self-consistent profile digests are not the independent decision.
        profile.update(_bindings(payload, profile))
    else:
        if mutation == "scope":
            decision["scope"]["model_id"] = "foreign-model"
        elif mutation == "roster":
            decision["bands"]["early"].pop()
            decision["band_roster_sha256"] = digest(decision["bands"])
        elif mutation == "source":
            decision["source_table_sha256"] = "e" * 64
        elif mutation == "contract":
            decision["calibration_contract"] = "unknown_flat_meaning.v99"
        elif mutation == "version":
            decision["schema"] = "unknown.v99"
        elif mutation == "missing_values":
            decision.pop("parameters")
        elif mutation == "missing_provenance":
            decision.pop("decision_ref")
        else:
            decision["parameters"]["early"]["loss_multiplier"] = math.inf
        # Equality of a hash alone cannot license an incomplete decision or
        # incompatible semantics. Do not allow the error to be merely stale SHA.
        if mutation != "nonfinite":
            profile["authorization_sha256"] = digest(decision)
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError):
        _interim_policy(payload, profile, decision, production=production)
    assert payload == before


@pytest.mark.parametrize("purpose", ["research", "production"])
def test_interim_cli_uses_configured_decision_and_retains_source_bytes(purpose, tmp_path, monkeypatch):
    _interim_policy(*_interim_smoke_arguments())  # assert the intake exists before CLI parsing
    import pickle

    selections = []
    for index, factors in enumerate(((0.25, 1.0), (1.0, 0.25))):
        root = tmp_path / str(index)
        root.mkdir()
        argv, source, payload = _cli_fixture(root)
        payload["provenance"]["depth_price_scope"] = _payload()[0]["provenance"]["depth_price_scope"]
        source.write_bytes(pickle.dumps(payload))
        raw = source.read_bytes()
        profile, decision = _interim_intake_fixture(payload, multipliers=factors)
        profile_path, decision_path = root / "profile.json", root / "decision.json"
        profile_path.write_text(json.dumps(profile))
        decision_path.write_text(json.dumps(decision))
        monkeypatch.setattr(sys, "argv", argv + [
            "--depth-price-profile", str(profile_path), "--depth-price-purpose", purpose,
            "--depth-price-authorization", str(decision_path)])
        allocator.main()
        assignment = load_assignment(root / "layer.json")
        selections.append({name for name in EARLY + LATE if assignment[name] == "BF16"})
        metadata = json.loads((root / "layer.json").read_text())["__prismaquant__"]["routed_depth_pricing"]
        assert metadata["state"] == "interim_flat" and metadata["validated"] is False
        assert metadata["authorization_sha256"] == digest(decision)
        assert metadata["source_table_sha256"] == digest(payload)
        assert metadata["selected_prices"] == {
            name: (0.0 if assignment[name] == "BF16" else
                   4.5 * (factors[0] if name in EARLY else factors[1]))
            for name in EARLY + LATE}
        assert source.read_bytes() == raw
    assert selections == [set(LATE), set(EARLY)]


def _interim_smoke_arguments():
    payload, _, _ = _payload()
    profile, decision = _interim_intake_fixture(payload)
    return payload, profile, decision


def test_interim_cli_missing_decision_publishes_no_assignment_or_seeds(tmp_path, monkeypatch):
    _interim_policy(*_interim_smoke_arguments())
    import pickle

    argv, source, payload = _cli_fixture(tmp_path)
    payload["provenance"]["depth_price_scope"] = _payload()[0]["provenance"]["depth_price_scope"]
    source.write_bytes(pickle.dumps(payload))
    profile, _ = _interim_intake_fixture(payload)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    before = source.read_bytes()
    monkeypatch.setattr(sys, "argv", argv + ["--depth-price-profile", str(path)])
    with pytest.raises((ValueError, SystemExit), match="(?i)(authoriz|decision|numeric|flat)"):
        allocator.main()
    assert source.read_bytes() == before
    assert not (tmp_path / "layer.json").exists()
    assert not list((tmp_path / "seeds").glob("*.json"))


def test_interim_authorization_does_not_authorize_replacement_measurements():
    from test_joint_aura_assignment_diagnostics import _row

    payload, profile, decision = _interim_smoke_arguments()
    policy = _interim_policy(payload, profile, decision)
    payload["costs"][EARLY[0]][LOW] = _row(EARLY[0], [9, 9], fmt=LOW)
    with pytest.raises(ValueError):
        _price(payload, policy, EARLY[0])


def test_interim_authorization_contract_is_documented_without_qualification_claim():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    design = (root / "docs/design/joint_aura_runtime_allocation.md").read_text()
    architecture = (root / "docs/ARCHITECTURE.md").read_text()
    changelog = (root / "CHANGELOG.md").read_text()
    for term in (AUTHORIZATION_SCHEMA, "--depth-price-authorization", "authorization_sha256",
                 "interim_flat", "unvalidated"):
        assert term in design, f"#2774: interim decision intake documentation omits {term}"
    assert "authorized_interim_record" in architecture
    assert "2774" in changelog
