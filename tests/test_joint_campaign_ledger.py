import copy

import pytest

from prismaquant import joint_campaign_ledger as ledger


@pytest.fixture
def record():
    return ledger.new_quantum_record(
        layer=0,
        campaign={"plan_path": "/plan", "prepared_path": "/prepared",
                  "plan_sha256": "a" * 64, "prepared_sha256": "b" * 64,
                  "read_manifest_sha256": "c" * 64, "unit_roster_sha256": "d" * 64,
                  "campaign_scope": {"kind": "complete_campaign"}},
        read_set={"total_bytes": 16}, chunks=[{"name": "chunk-000"}],
        windows=[{"window_index": 0}],
        adjoint={"checkpoint_boundary": 1, "chain_layers": [], "receipt_sha256": None},
        output_space={"root": "/run/layer-quanta/layer-000"})


def test_canonical_record_identity_keeps_the_v1_wire_recipe():
    record = ledger.seal_quantum_record({"schema": ledger.LAYER_QUANTUM_SCHEMA, "note": "ü"})
    assert record["identity_sha256"] == "4aa7ac5911c1ded8959b400a0d982bec77991ebd34df1e7a9edcc9d8343632e9"
    ledger.require_quantum_identity(record)
    with pytest.raises(ValueError):
        ledger.seal_quantum_record({"schema": ledger.LAYER_QUANTUM_SCHEMA, "note": float("nan")})


@pytest.mark.parametrize("mode", ["0", "1"])
@pytest.mark.parametrize("key", ["schema", "chunks", "windows", "adjoint", "output_space"])
def test_body_tampering_refuses_in_both_modes(record, monkeypatch, mode, key):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    record[key] = "edited"
    with pytest.raises(ValueError):
        ledger.require_quantum_campaign(record, record["campaign"])


@pytest.mark.parametrize("mode", ["0", "1"])
@pytest.mark.parametrize("key", ["plan_sha256", "prepared_sha256", "read_manifest_sha256", "campaign_scope", "unit_roster_sha256"])
def test_campaign_rules_distinguish_run_seals_from_the_measured_roster(record, monkeypatch, mode, key):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    campaign = {**record["campaign"], key: {"kind": "other"} if key == "campaign_scope" else "e" * 64}
    if mode == "0" or key == "unit_roster_sha256":
        with pytest.raises(ValueError):
            ledger.require_quantum_campaign(record, campaign)
    else:
        ledger.require_quantum_campaign(record, campaign)


def test_unbound_and_slice_bound_records_have_distinct_admission(record):
    ledger.require_quantum_campaign(record, record["campaign"])
    expected = {**record["campaign"], "adjoint_slice_sha256": "f" * 64}
    with pytest.raises(ValueError, match="unbound"):
        ledger.require_quantum_campaign(record, expected)
    bound = ledger.seal_quantum_record({**record, "adjoint": {"slice_sha256": "f" * 64, "slice_path": "/slice"}})
    ledger.require_quantum_campaign(bound, expected)
    with pytest.raises(ValueError, match="another stage-A slice"):
        ledger.require_quantum_campaign(bound, {**expected, "adjoint_slice_sha256": "e" * 64})
    with pytest.raises(ValueError, match="whole stage-A receipt"):
        ledger.require_quantum_campaign(record, {**record["campaign"], "adjoint_receipt_sha256": "f" * 64})
    receipt_bound = ledger.seal_quantum_record({**record, "adjoint": {"receipt_sha256": "f" * 64}})
    with pytest.raises(ValueError, match="whole stage-A receipt"):
        ledger.require_quantum_campaign(receipt_bound, receipt_bound["campaign"])


@pytest.mark.parametrize("kind", ["boundary", "executable"])
def test_readset_replacement_preserves_the_original_and_unknown_fields(record, kind):
    record = ledger.seal_quantum_record({**record, "future_field": {"values": [1]}})
    original = copy.deepcopy(record)
    block = {"manifest_path": "/manifest", "phases": ["head"]}
    fresh = ledger.replace_quantum_readset(record, kind=kind, block=block)
    assert record == original
    assert fresh["future_field"] == original["future_field"]
    assert fresh["identity_sha256"] != original["identity_sha256"]
    block["phases"].append("late edit")
    ledger.require_quantum_identity(fresh)
    assert fresh[f"{kind}_readset"]["phases"] == ["head"]
    record["future_field"]["values"].append(2)
    with pytest.raises(ValueError, match="identity"):
        ledger.replace_quantum_readset(record, kind=kind, block=block)


@pytest.mark.parametrize("defect", ["duplicate", "campaign", "mixed_binding", "receipt"])
def test_record_sets_refuse_mixed_generations(record, defect):
    other = ledger.new_quantum_record(layer=1, campaign=record["campaign"], read_set={},
                                     chunks=[], windows=[], adjoint=record["adjoint"], output_space={})
    if defect == "duplicate":
        other["quantum_id"] = record["quantum_id"]
    elif defect == "campaign":
        other["campaign"]["plan_path"] = "/other-plan"
    elif defect == "mixed_binding":
        other["adjoint"]["slice_sha256"] = "f" * 64
    else:
        other["adjoint"]["receipt_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        ledger.require_quantum_record_set([record, other])
