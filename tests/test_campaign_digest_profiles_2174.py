"""Campaign's direct JSON, strict UTF-8 and sealed receipt bytes stay exact."""
import hashlib
import json
import sys

import pytest

from prismaquant import cluster_campaign as campaign


def _old_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


@pytest.mark.parametrize("value", [None, {}, {"z": "δ/数据", "a": [1, True, None]},
    {10: "ten", 9: "nine"}, [0.0, -0.0, 1e-300, 1.7976931348623157e308],
    ("tuple", {"newline": "a\nb", "nul": "a\x00b"})])
def test_campaign_identity_keeps_direct_json_profile_not_roundtrip(value):
    raw = _old_bytes(value)
    assert campaign._canonical_bytes(value) == raw
    assert campaign.canonical_sha256(value) == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "one", "2": "two"},
                                  {"unsupported": object()}, "\ud800"])
def test_campaign_json_refuses_with_unchanged_domain_error_and_native_cause(value):
    with pytest.raises((TypeError, ValueError)) as old:
        _old_bytes(value)
    with pytest.raises(campaign.CampaignContractError) as new:
        campaign.canonical_sha256(value)
    assert str(new.value) == "value is not finite canonical JSON data"
    assert type(new.value.__cause__) is type(old.value)
    assert str(new.value.__cause__) == str(old.value)


@pytest.mark.parametrize("identity,stage,attempt", [
    ("a" * 64, "plain", 1), ("δ", "数据", "002"),
    ("a\x00b", "c\x00d", True), ("zero", "negative", -1),
    ("rounded", "attempt", 1.9),
])
def test_attempt_owner_keeps_legacy_nul_framing_and_int_coercion(identity, stage, attempt):
    raw = f"{identity}\0{stage}\0{int(attempt)}".encode("utf-8")
    assert campaign._owner_token(identity, stage, attempt) == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("attempt", ["bad", None])
def test_attempt_owner_preserves_native_coercion_refusal(attempt):
    with pytest.raises((TypeError, ValueError)) as old:
        int(attempt)
    with pytest.raises(type(old.value)) as new:
        campaign._owner_token("identity", "stage", attempt)
    assert str(new.value) == str(old.value)


def test_sealed_stage_receipt_keeps_child_binding_and_final_lf():
    child = [sys.executable, "-c", "print('数据/δ')"]
    token = "qwen38-125b:materialize-plan"
    child_sha = hashlib.sha256(_old_bytes(child)).hexdigest()
    expected = _old_bytes({"schema": campaign.SEALED_STAGE_RECEIPT_SCHEMA_V1,
                           "token": token, "child_argv_sha256": child_sha}) + b"\n"
    assert campaign.sealed_stage_receipt_bytes(token, child) == expected
    assert campaign.sealed_stage_receipt_sha256(token, child) == hashlib.sha256(expected).hexdigest()
    changed = [*child[:-1], "print('different')"]
    assert campaign.sealed_stage_receipt_sha256(token, changed) != hashlib.sha256(expected).hexdigest()
