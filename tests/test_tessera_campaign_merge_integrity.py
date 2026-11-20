"""A campaign merge must preserve the journal's existing trust boundary."""
from __future__ import annotations
from prismaquant import tessera_campaign_table as campaign_table

import json
import pickle
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def _journal(tmp_path, row_id="row", *, states=None, sampling=None):
    from prismaquant.cost_stage_checkpoint import prepare_journal, write_unit, unit_path

    row = tmp_path / row_id
    manifest = row / "cost.anchors.json"
    if states is None:
        states = {"a": {"measured": {"rung": 1.0}}}
    identity = {"units": {name: {"weight": "w" + name} for name in states}}
    if sampling is not None:
        identity["stack_sampling_identity"] = sampling
    parts, digest, _ = prepare_journal(
        row / "cost.anchors.json.parts", stage="Tessera campaign", resume=True,
        identity=identity, qnames=sorted(states), manifest_path=manifest)
    for name, state in states.items():
        write_unit(parts, stage="Tessera campaign", qname=name, identity_sha256=digest,
                   state=state)
    return row, manifest, unit_path(parts, next(iter(states)))


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

    rows = {"row-0000": str(_restricted_row(tmp_path, "row-0000", "a", "dense")),
            "row-0001": str(_restricted_row(tmp_path, "row-0001", "b", "routed_moe"))}
    merged = campaign_table.merge_checkpoint(rows, tmp_path / "merged" / "cost.anchors.json")
    assert merged["identity"]["family_restriction"] == {
        "policy": POLICY, "structure_by_unit": {"a": "dense", "b": "routed_moe"}}
    assert sorted(merged["identity"]["units"]) == ["a", "b"]
    assert [entry["qname"] for entry in merged["units"]] == ["a", "b"]


def test_merge_checkpoint_refuses_a_conflicting_restricted_structure(tmp_path):
    """The union is not an overwrite: one unit, two structures, is a refusal."""

    first = _restricted_row(tmp_path, "row-0000", "a", "dense")
    second = _restricted_row(tmp_path, "row-0001", "b", "routed_moe")
    manifest = second / "cost.anchors.json"
    value = json.loads(manifest.read_text())
    value["identity"]["family_restriction"]["structure_by_unit"]["a"] = "routed_moe"
    manifest.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="a"):
        campaign_table.merge_checkpoint({"row-0000": str(first), "row-0001": str(second)},
                                  tmp_path / "merged.json")


def test_merge_checkpoint_refuses_a_restricted_row_beside_an_unrestricted_one(tmp_path):
    """A restriction the other row never carried is a different campaign."""

    rows = {"row-0000": str(_restricted_row(tmp_path, "row-0000", "a", "dense")),
            "row-0001": str(_restricted_row(tmp_path, "row-0001", "b", None,
                                            policy=None))}
    with pytest.raises(RuntimeError, match="restrict"):
        campaign_table.merge_checkpoint(rows, tmp_path / "merged.json")


def test_merge_checkpoint_refuses_two_policies(tmp_path):
    """Policy is the campaign-wide half and still has to be equal."""

    other = {**POLICY, "dense": ["TESSERA_E4M3_K1"]}
    rows = {"row-0000": str(_restricted_row(tmp_path, "row-0000", "a", "dense")),
            "row-0001": str(_restricted_row(tmp_path, "row-0001", "b", "routed_moe",
                                            policy=other))}
    with pytest.raises(RuntimeError, match="policy"):
        campaign_table.merge_checkpoint(rows, tmp_path / "merged.json")


@pytest.mark.parametrize("field,value", [
    ("payload_sha256", "0" * 64),
    ("identity_sha256", "0" * 64),
    ("qname", "another-unit"),
    ("stage", "another-stage"),
    ("schema", "another-schema"),
])
def test_merge_refuses_a_shard_the_journal_itself_refuses(tmp_path, field, value):

    row, _, shard = _journal(tmp_path)
    envelope = pickle.loads(shard.read_bytes())
    envelope[field] = value
    shard.write_bytes(pickle.dumps(envelope))
    out = tmp_path / "merged" / "cost.anchors.json"
    with pytest.raises(RuntimeError, match=field):
        campaign_table.merge_checkpoint({"row-0000": str(row)}, out)
    assert not out.exists()


def test_merge_refuses_a_manifest_with_a_wrong_identity_digest(tmp_path):

    row, manifest, _ = _journal(tmp_path)
    value = json.loads(manifest.read_text())
    value["identity_sha256"] = "0" * 64
    manifest.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="identity_sha256"):
        campaign_table.merge_checkpoint({"row-0000": str(row)}, tmp_path / "merged.json")


def test_merge_refuses_a_manifest_that_drops_an_identity_unit(tmp_path):

    row, manifest, _ = _journal(tmp_path)
    value = json.loads(manifest.read_text())
    value["units"] = []
    manifest.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="units"):
        campaign_table.merge_checkpoint({"row-0000": str(row)}, tmp_path / "merged.json")


def _resume_merged(out, merged):
    from prismaquant.cost_stage_checkpoint import prepare_journal

    _, _, states = prepare_journal(
        out.with_name(out.name + ".parts"), stage="Tessera campaign", resume=True,
        identity=merged["identity"], qnames=sorted(merged["identity"]["units"]),
        manifest_path=out)
    return states


def test_valid_merged_journal_resumes_through_the_owned_reader(tmp_path):
    row, _, _ = _journal(tmp_path)
    out = tmp_path / "merged" / "cost.anchors.json"
    merged = campaign_table.merge_checkpoint({"row-0000": str(row)}, out)
    assert _resume_merged(out, merged) == {"a": {"measured": {"rung": 1.0}}}


def _anchor_rows(name, family, points):
    from test_tessera_stack_sample_cost import _anchor

    return [_anchor(name, family, f"{family}_R{rate}", rate, loss)
            for rate, loss in points]


def test_authenticated_row_journal_union_prices_like_one_campaign(tmp_path):
    family = "TESSERA_E2M1_K1"
    rows, states = {}, {}
    samples = {}
    for layer in (7, 8):
        module = f"model.layers.{layer}.feed_forward.experts"
        packed = f"{module}.down_proj"
        experts = (0, 1, 2, 3)
        sample = campaign_table.StackExpertSample(
            packed_qname=packed, packed_experts_module=module, packed_param="down_proj",
            num_experts=4, stack_h_trace=4.0, h_trace_per_expert=(1.0,) * 4,
            sampled_experts=experts, inclusion_prob=dict.fromkeys(experts, 1.0),
            members={e: (f"{module}.{e}.w2",) for e in experts}, seed=3,
            design="census", transfer_law_experts=(0, 1, 2))
        samples[packed] = sample
        row_states = {}
        for expert, (name,) in sample.members.items():
            points = [(960, expert + 1.0)]
            if expert < 3:
                points.append((832, 2.0 * (expert + 1.0)))
            anchors = _anchor_rows(name, family, points)
            row_states[name] = {
                "anchors": [vars(anchor) for anchor in anchors],
                "wire_records": {anchor.format_name: {"file": f"{name}.{anchor.format_name}",
                                                       "blob_bytes": anchor.wire_bytes}
                                 for anchor in anchors},
            }
        row_id = f"row-{layer}"
        rows[row_id], _, _ = _journal(tmp_path, row_id, states=row_states,
                                      sampling={packed: {"transfer_law_experts": [0, 1, 2]}})
        states.update(row_states)

    def price(evidence):
        anchors = {name: {family: [campaign_table.CampaignAnchor(**a)
                                   for a in state["anchors"]]}
                   for name, state in evidence.items()}
        return campaign_table.campaign_cost_payload(
            anchors, {}, loo={}, provenance={}, stack_samples=samples)

    whole = price(states)
    assert set(whole["costs"]) == set(samples)
    for packed, formats in whole["costs"].items():
        target = formats[f"{family}_R832"]
        assert target["output_mse"] == pytest.approx(5.0)
        assert target["transfer_law"]["held_out"] == packed
        assert target["transfer_law"]["pooled_stacks"] == sorted(samples.keys() - {packed})

    for index, row_order in enumerate((rows, dict(reversed(list(rows.items()))))):
        out = tmp_path / f"merged-{index}" / "cost.anchors.json"
        merged = campaign_table.merge_checkpoint(row_order, out)
        assert set(merged["identity"]["stack_sampling_identity"]) == set(samples)
        resumed = _resume_merged(out, merged)
        assert resumed == states
        assert price(resumed) == whole


def test_refusal_order_does_not_depend_on_anchor_insertion_order():
    from prismaquant import tessera_campaign as campaign

    def payload(names):
        family = "TESSERA_BF16_K1"
        return campaign_table.campaign_cost_payload(
            {name: {family: _anchor_rows(name, family, [(1792, 1.0)])}
             for name in names}, {}, loo={}, provenance={})

    assert payload(["b", "a"])["non_interpolable"] == payload(["a", "b"])["non_interpolable"]


def test_merge_accepts_stack_incomplete_rung_refusal_without_family():
    from test_tessera_campaign_fanout import _payloads

    payloads = _payloads()
    refusal = {"qname": "a", "format_name": "TESSERA_BF16_K1_R1792",
               "reason": "stack_rung_incomplete_over_sample", "missing_experts": [2]}
    payloads["row-0000"]["non_interpolable"] = [refusal]
    merged = campaign_table.merge_payloads(payloads, census={"counts": {}}, capture_sha256="merged")
    assert merged["non_interpolable"] == [refusal]


def test_merge_carries_the_union_of_identity_migration_records(tmp_path):
    from test_tessera_campaign_fanout import _payloads

    record = {"schema": "prismaquant.identity_migration.v1", "proof_bundle_sha256": "p" * 64,
              "old_pins": {"prismaquant_source_sha256": "a" * 64}, "new_pins": {"prismaquant_source_sha256": "b" * 64},
              "old_identity_sha256": "row-specific", "shards": 3}
    row, manifest, _ = _journal(tmp_path)
    data = json.loads(manifest.read_text())
    data["identity_migration"] = [record]
    manifest.write_text(json.dumps(data))
    out = tmp_path / "merged" / "cost.anchors.json"
    merged = campaign_table.merge_checkpoint({"row-0000": str(row)}, out)
    carried = json.loads(out.read_text())["identity_migration"]
    assert merged["identity_migration"] == carried
    assert carried[0]["proof_bundle_sha256"] == "p" * 64 and "old_identity_sha256" not in carried[0]

    payloads = _payloads()
    for row_id, payload in payloads.items():
        payload["provenance"]["identity_migration"] = [dict(record, old_identity_sha256=row_id)]
    merged_payload = campaign_table.merge_payloads(payloads, census={"counts": {}}, capture_sha256="merged")
    assert merged_payload["provenance"]["identity_migration"] == carried
    del payloads[sorted(payloads)[0]]["provenance"]["identity_migration"]
    merged_payload = campaign_table.merge_payloads(payloads, census={"counts": {}}, capture_sha256="merged")
    assert merged_payload["provenance"]["identity_migration"] == carried


def test_the_uncalled_table_names_stay_deleted():
    assert not {"_bound_pickle", "REFERENCES_NAME", "SCALES_NAME"} & set(vars(campaign_table))
