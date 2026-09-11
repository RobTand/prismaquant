"""A campaign merge must preserve the journal's existing trust boundary."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def _journal(tmp_path):
    from prismaquant.cost_stage_checkpoint import prepare_journal, write_unit, unit_path

    row = tmp_path / "row"
    manifest = row / "cost.anchors.json"
    parts, digest, _ = prepare_journal(
        row / "cost.anchors.json.parts", stage="Tessera campaign", resume=True,
        identity={"units": {"a": {"weight": "wa"}}}, qnames=["a"],
        manifest_path=manifest)
    write_unit(parts, stage="Tessera campaign", qname="a", identity_sha256=digest,
               state={"measured": {"rung": 1.0}})
    return row, manifest, unit_path(parts, "a")


POLICY = {"schema": "prismaquant.tessera_campaign_family_restriction.v1",
          "dense": ["TESSERA_BF16_K1", "TESSERA_E4M3_K1"],
          "routed_moe": ["TESSERA_E4M3_K1"]}


def _restricted_row(tmp_path, row_id, qname, structure, policy=POLICY):
    """One row's journal, restricted to its own unit the way a campaign row is."""
    from prismaquant.cost_stage_checkpoint import prepare_journal, write_unit

    row = tmp_path / row_id
    manifest = row / "cost.anchors.json"
    identity = {"units": {qname: {"weight": "w" + qname}}}
    if policy is not None:
        identity["family_restriction"] = {
            "policy": policy, "structure_by_unit": {qname: structure}}
    parts, digest, _ = prepare_journal(
        row / "cost.anchors.json.parts", stage="Tessera campaign", resume=True,
        identity=identity, qnames=[qname], manifest_path=manifest)
    write_unit(parts, stage="Tessera campaign", qname=qname,
               identity_sha256=digest, state={"measured": {"rung": 1.0}})
    return row


def test_merge_checkpoint_unions_a_dense_row_and_a_routed_row(tmp_path):
    """The journal side of the restriction is per-selection, so it reconciles.

    A dense row and a routed row of one campaign carry different
    ``structure_by_unit`` maps by construction. Requiring the whole
    ``family_restriction`` to be equal refuses every census that fans dense
    and routed units onto separate rows (RobTand/prismaquant#487).
    """
    import dispatch_tessera_campaign as dispatch

    rows = {"row-0000": str(_restricted_row(tmp_path, "row-0000", "a", "dense")),
            "row-0001": str(_restricted_row(tmp_path, "row-0001", "b", "routed_moe"))}
    merged = dispatch.merge_checkpoint(rows, tmp_path / "merged" / "cost.anchors.json")
    assert merged["identity"]["family_restriction"] == {
        "policy": POLICY, "structure_by_unit": {"a": "dense", "b": "routed_moe"}}
    assert sorted(merged["identity"]["units"]) == ["a", "b"]
    assert [entry["qname"] for entry in merged["units"]] == ["a", "b"]


def test_merge_checkpoint_refuses_a_conflicting_restricted_structure(tmp_path):
    """The union is not an overwrite: one unit, two structures, is a refusal."""
    import dispatch_tessera_campaign as dispatch

    first = _restricted_row(tmp_path, "row-0000", "a", "dense")
    second = _restricted_row(tmp_path, "row-0001", "b", "routed_moe")
    manifest = second / "cost.anchors.json"
    value = json.loads(manifest.read_text())
    value["identity"]["family_restriction"]["structure_by_unit"]["a"] = "routed_moe"
    manifest.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="a"):
        dispatch.merge_checkpoint({"row-0000": str(first), "row-0001": str(second)},
                                  tmp_path / "merged.json")


def test_merge_checkpoint_refuses_a_restricted_row_beside_an_unrestricted_one(tmp_path):
    """A restriction the other row never carried is a different campaign."""
    import dispatch_tessera_campaign as dispatch

    rows = {"row-0000": str(_restricted_row(tmp_path, "row-0000", "a", "dense")),
            "row-0001": str(_restricted_row(tmp_path, "row-0001", "b", None,
                                            policy=None))}
    with pytest.raises(RuntimeError, match="restrict"):
        dispatch.merge_checkpoint(rows, tmp_path / "merged.json")


def test_merge_checkpoint_refuses_two_policies(tmp_path):
    """Policy is the campaign-wide half and still has to be equal."""
    import dispatch_tessera_campaign as dispatch

    other = {**POLICY, "dense": ["TESSERA_E4M3_K1"]}
    rows = {"row-0000": str(_restricted_row(tmp_path, "row-0000", "a", "dense")),
            "row-0001": str(_restricted_row(tmp_path, "row-0001", "b", "routed_moe",
                                            policy=other))}
    with pytest.raises(RuntimeError, match="policy"):
        dispatch.merge_checkpoint(rows, tmp_path / "merged.json")


@pytest.mark.parametrize("field,value", [
    ("payload_sha256", "0" * 64),
    ("identity_sha256", "0" * 64),
    ("qname", "another-unit"),
    ("stage", "another-stage"),
    ("schema", "another-schema"),
])
def test_merge_refuses_a_shard_the_journal_itself_refuses(tmp_path, field, value):
    import dispatch_tessera_campaign as dispatch

    row, _, shard = _journal(tmp_path)
    envelope = pickle.loads(shard.read_bytes())
    envelope[field] = value
    shard.write_bytes(pickle.dumps(envelope))
    out = tmp_path / "merged" / "cost.anchors.json"
    with pytest.raises(RuntimeError, match=field):
        dispatch.merge_checkpoint({"row-0000": str(row)}, out)
    assert not out.exists()


def test_merge_refuses_a_manifest_with_a_wrong_identity_digest(tmp_path):
    import dispatch_tessera_campaign as dispatch

    row, manifest, _ = _journal(tmp_path)
    value = json.loads(manifest.read_text())
    value["identity_sha256"] = "0" * 64
    manifest.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="identity_sha256"):
        dispatch.merge_checkpoint({"row-0000": str(row)}, tmp_path / "merged.json")


def test_merge_refuses_a_manifest_that_drops_an_identity_unit(tmp_path):
    import dispatch_tessera_campaign as dispatch

    row, manifest, _ = _journal(tmp_path)
    value = json.loads(manifest.read_text())
    value["units"] = []
    manifest.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="units"):
        dispatch.merge_checkpoint({"row-0000": str(row)}, tmp_path / "merged.json")


def test_valid_merged_journal_resumes_through_the_owned_reader(tmp_path):
    import dispatch_tessera_campaign as dispatch
    from prismaquant.cost_stage_checkpoint import prepare_journal

    row, _, _ = _journal(tmp_path)
    out = tmp_path / "merged" / "cost.anchors.json"
    merged = dispatch.merge_checkpoint({"row-0000": str(row)}, out)
    _, _, states = prepare_journal(
        out.with_name(out.name + ".parts"), stage="Tessera campaign", resume=True,
        identity=merged["identity"], qnames=["a"], manifest_path=out)
    assert states == {"a": {"measured": {"rung": 1.0}}}


def test_refusal_order_does_not_depend_on_anchor_insertion_order():
    from test_tessera_stack_sample_cost import _anchor
    from prismaquant import tessera_campaign as campaign

    def payload(names):
        return campaign.campaign_cost_payload(
            {name: {"TESSERA_BF16_K1": [
                _anchor(campaign, name, "TESSERA_BF16_K1", "TESSERA_BF16_K1_R1792",
                        1792, 1.0)]} for name in names}, {}, loo={}, provenance={})

    assert payload(["b", "a"])["non_interpolable"] == payload(["a", "b"])["non_interpolable"]


def test_merge_accepts_stack_incomplete_rung_refusal_without_family():
    import dispatch_tessera_campaign as dispatch
    from test_tessera_campaign_fanout import _payloads

    payloads = _payloads()
    refusal = {"qname": "a", "format_name": "TESSERA_BF16_K1_R1792",
               "reason": "stack_rung_incomplete_over_sample", "missing_experts": [2]}
    payloads["row-0000"]["non_interpolable"] = [refusal]
    merged = dispatch.merge_payloads(payloads, census={"counts": {}}, capture_sha256="merged")
    assert merged["non_interpolable"] == [refusal]


def test_merge_carries_the_union_of_identity_migration_records(tmp_path):
    import dispatch_tessera_campaign as dispatch
    from test_tessera_campaign_fanout import _payloads

    record = {"schema": "prismaquant.identity_migration.v1", "proof_bundle_sha256": "p" * 64,
              "old_pins": {"prismaquant_source_sha256": "a" * 64}, "new_pins": {"prismaquant_source_sha256": "b" * 64},
              "old_identity_sha256": "row-specific", "shards": 3}
    row, manifest, _ = _journal(tmp_path)
    data = json.loads(manifest.read_text())
    data["identity_migration"] = [record]
    manifest.write_text(json.dumps(data))
    out = tmp_path / "merged" / "cost.anchors.json"
    merged = dispatch.merge_checkpoint({"row-0000": str(row)}, out)
    carried = json.loads(out.read_text())["identity_migration"]
    assert merged["identity_migration"] == carried
    assert carried[0]["proof_bundle_sha256"] == "p" * 64 and "old_identity_sha256" not in carried[0]

    payloads = _payloads()
    for row_id, payload in payloads.items():
        payload["provenance"]["identity_migration"] = [dict(record, old_identity_sha256=row_id)]
    merged_payload = dispatch.merge_payloads(payloads, census={"counts": {}}, capture_sha256="merged")
    assert merged_payload["provenance"]["identity_migration"] == carried
    del payloads[sorted(payloads)[0]]["provenance"]["identity_migration"]
    merged_payload = dispatch.merge_payloads(payloads, census={"counts": {}}, capture_sha256="merged")
    assert merged_payload["provenance"]["identity_migration"] == carried
