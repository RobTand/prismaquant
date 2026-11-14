"""Partition handoffs use real reference/journal owners, without GPU claims."""

from prismaquant import tessera_campaign_wire as campaign_wire
from prismaquant import tessera_campaign_table as campaign_table
from types import SimpleNamespace


from prismaquant import cost_stage_checkpoint as journal
from prismaquant import tessera_campaign as campaign
from test_tessera_campaign_expert_partitions import roster, selection


def test_partition_journal_union_matches_actual_whole_unit_identity(tmp_path, monkeypatch):
    class Api:
        @staticmethod
        def encoder_source_sha256():
            return "encoder"

        @staticmethod
        def tensor_identity(value):
            return {"id": value}

    monkeypatch.setattr(campaign_wire, 'require_checkpoint_identity_api', lambda: Api)
    monkeypatch.setattr(campaign.th, "encoder_recipe", lambda: {"recipe": 1})
    monkeypatch.setattr("prismaquant.production_weight_cache._production_cache_source_sha256",
                        lambda: "package")
    names = sorted(roster()[0])
    formats = ["TESSERA_E4M3_K1_R768", "TESSERA_E4M3_K1_R1024"]
    args = SimpleNamespace(model="fixture", units="whole.json", rate_band="768,768",
                           max_rounds=1, family_restriction=None, source_scope=None)
    common = dict(settings=vars(args), calibration_identity={"text_sha256": "same-draw"},
                  serving_scope=None, static_scales={}, static_scale_policy="same-policy")

    def identity(priced):
        return campaign.campaign_checkpoint_identity(
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
    merged = campaign_table.merge_checkpoint(row_dirs, destination)
    assert merged["identity"] == journal.canonical_json(whole, where="whole identity")
    assert merged["identity_sha256"] == journal.canonical_json_sha256(whole, where="whole identity")
    _, _, resumed = journal.prepare_journal(
        str(destination) + ".parts", stage="Tessera campaign", resume=True,
        identity=whole, qnames=names, manifest_path=destination)
    assert resumed == states
    assert {name: unit["menu"] for name, unit in merged["identity"]["units"].items()} == {
        name: sorted(formats) for name in names}
