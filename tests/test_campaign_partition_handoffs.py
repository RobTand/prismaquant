"""Partition handoffs use real reference/journal owners, without GPU claims."""
from types import SimpleNamespace

import pytest

from prismaquant import cost_stage_checkpoint as journal
from prismaquant import tessera_calibration_cache as cache
from prismaquant import tessera_campaign as campaign
from prismaquant import tessera_export_lane as export
from tools import dispatch_tessera_campaign as dispatch
from test_tessera_campaign_expert_partitions import roster, selection
from test_tessera_hessian_reference_handoff import handoff, rows as reference_rows


def test_partition_reference_union_uses_priced_subset_not_full_group(handoff, monkeypatch):
    import torch

    fixture = handoff
    dirs, payloads = reference_rows(fixture)
    names = sorted(fixture["H"])
    # Isolate the reference-roster boundary: producer/chunk validation is
    # separately exercised against real carried producer records.
    for index, name in enumerate(names):
        payloads[name]["provenance"]["unit_selection"] = {
            "schema": campaign.UNITS_SCHEMA_V3,
            "groups": [{"key": "s:fixture", "members": names,
                "partition": {"schema": campaign.EXPERT_PARTITION_SCHEMA,
                    "experts_per_row": 1, "index": index, "count": len(names),
                    "rate_q256": 768, "members": [name]}}]}
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("eager reference H read"))
    path, _, digest = dispatch.merge_export_inputs(dirs, payloads,
        out_cache=fixture["tmp"] / "partition-merged", identity=fixture["calibration"],
        policy="fixture", static_scales={}, census=fixture["census"])
    assert digest == export.hessian_capture_sha256(
        fixture["H"], {**fixture["calibration"], "hessian_role": "fit"})
    with cache.open_hessian_reference(path) as owner:
        assert set(owner) == set(names)
        assert owner.receipt()["loaded_entries"] == 0


def test_partition_journal_union_matches_actual_whole_unit_identity(tmp_path, monkeypatch):
    class Api:
        @staticmethod
        def encoder_source_sha256():
            return "encoder"

        @staticmethod
        def tensor_identity(value):
            return {"id": value}

    monkeypatch.setattr(campaign, "_checkpoint_identity_api", lambda: Api)
    monkeypatch.setattr(campaign.th, "encoder_recipe", lambda: {"recipe": 1})
    monkeypatch.setattr("prismaquant.production_weight_cache._production_cache_source_sha256",
                        lambda: "package")
    names = sorted(roster()[0])
    formats = ["TESSERA_E4M3_K1_R768", "TESSERA_E4M3_K1_R1024"]
    args = SimpleNamespace(model="fixture", units="whole.json", rate_band="768,768",
                           max_rounds=1, family_restriction=None, source_scope=None)
    common = dict(args=args, calibration_identity={"text_sha256": "same-draw"},
                  serving_scope=None, static_scales={}, static_scale_policy="same-policy")

    def identity(priced):
        return campaign._campaign_checkpoint_identity(
            weights={name: "weight:" + name for name in priced},
            acts={name: None for name in priced}, hessians={name: None for name in priced},
            menus={name: [SimpleNamespace(format_name=fmt) for fmt in formats] for name in priced},
            **common)

    whole = identity(names)
    row_dirs = {}
    states = {name: {"qname": name, "measured_anchor": 768} for name in names}
    for index in range(2):
        priced = selection(index)["groups"][0]["partition"]["members"]
        row = tmp_path / f"row{index}"
        manifest = row / "cost.anchors.json"
        root, digest, _ = journal.prepare_journal(
            str(manifest) + ".parts", stage="Tessera campaign", resume=False,
            identity=identity(priced), qnames=priced, manifest_path=manifest)
        for name in priced:
            journal.write_unit(root, stage="Tessera campaign", qname=name,
                               identity_sha256=digest, state=states[name])
        row_dirs[f"row{index}"] = str(row)
    destination = tmp_path / "merged" / "cost.anchors.json"
    merged = dispatch.merge_checkpoint(row_dirs, destination)
    assert merged["identity"] == journal.canonical_json(whole, where="whole identity")
    assert merged["identity_sha256"] == journal.canonical_json_sha256(whole, where="whole identity")
    _, _, resumed = journal.prepare_journal(
        str(destination) + ".parts", stage="Tessera campaign", resume=True,
        identity=whole, qnames=names, manifest_path=destination)
    assert resumed == states
    assert {name: unit["menu"] for name, unit in merged["identity"]["units"].items()} == {
        name: sorted(formats) for name in names}
