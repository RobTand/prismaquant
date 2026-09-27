"""A composed H reference keeps the original per-unit priced commitments."""

import json
import copy

import pytest
import torch

from prismaquant import tessera_calibration_cache as cache
from prismaquant import tessera_export_lane as export
from prismaquant import glm_mtp_selection
from tools.build_tessera_hessian_collection import collection_document, publish_collection
from test_tessera_hessian_reference_handoff import handoff, write_row  # noqa: F401


def _children(handoff):
    return [write_row(handoff, name)[0] for name in ("a", "b")]


def test_collection_handoff_opens_disjoint_original_references(handoff):
    children = _children(handoff)
    output = handoff["tmp"] / "union.collection.references.json"
    published = publish_collection(children, output)
    assert output.is_file()
    assert published["units"] == 2
    with cache.open_hessian_reference(output) as owner:
        assert set(owner) == {"a", "b"}
        assert owner.receipt()["verified_units"] == []
        for name in ("a", "b"):
            with cache.open_hessian_reference(children[("a", "b").index(name)]) as child:
                assert owner.commitment(name) == child.commitment(name)
            torch.testing.assert_close(owner[name], handoff["H"][name], rtol=0, atol=0)
        receipt = owner.receipt()
        assert receipt["verified_units"] == ["a", "b"]
        assert all(len(row["verified_units"]) == 1 for row in receipt["references"])
    assert owner.receipt()["closed"] is True


def test_collection_refuses_child_mutation_and_roster_omission(handoff):
    children = _children(handoff)
    output = handoff["tmp"] / "union.collection.references.json"
    publish_collection(children, output)
    document = json.loads(output.read_text())
    document["units"].remove("b")
    output.write_text(json.dumps(document))
    with pytest.raises(Exception, match="roster|omits|committed"):
        cache.open_hessian_reference(output)

    output.write_text(json.dumps(collection_document(children)))
    children[1].write_text(children[1].read_text() + " ")
    with pytest.raises(Exception, match="checksum|changed|differs"):
        cache.open_hessian_reference(output)


def test_collection_author_refuses_overlapping_rosters(handoff):
    path = write_row(handoff, "a")[0]
    with pytest.raises(ValueError, match="distinct"):
        collection_document([path, path])


def test_export_collection_rebinds_each_selected_unit_to_original_child(handoff, monkeypatch):
    children = _children(handoff)
    output = handoff["tmp"] / "union.collection.references.json"
    publish_collection(children, output)
    fmt = "TESSERA_E4M3_K1_R1024"
    with cache.open_hessian_reference(children[0]) as body, \
            cache.open_hessian_reference(children[1]) as mtp:
        from tessera.hessian_capture import capture_sha256_from_units
        body_seal = capture_sha256_from_units(body.provenance, body.committed_units())
        mtp_seal = capture_sha256_from_units(mtp.provenance, mtp.committed_units())
        block = {"reference_binding": body.binding(), "capture_sha256": body_seal}
        priced = {"provenance": {"hessian": {
            "capture_path": str(children[1]), "capture_sha256": mtp_seal,
            "reference_binding": mtp.binding()}},
            "costs": {"b": {fmt: {"hessian_identity": {
                "applied": True, "capture_sha256": mtp_seal,
                "reference_binding": mtp.binding()}}}}}
    with cache.open_hessian_reference(output) as owner:
        selection = {"rung_by_group": {"mtp": fmt},
            "cost_path": str(handoff["tmp"] / "bound.pkl"),
            "mtp_expert_wire_binding_schema": glm_mtp_selection.WIRE_BINDING_SCHEMA,
            "mtp_expert_wires": {"b": {"identity": {"calibration": {
                "hessian": owner.commitment("b")}}}},
            "mtp_expert_source_bindings": {"b": {"m3": {
                "path": str(handoff["tmp"] / "m3.pkl"), "sha256": "a" * 64}}},
        }
        for field in ("mtp_joint_cost_sha256", "mtp_expert_projection",
                      "mtp_expert_wire_roots"):
            selection[field] = field
        monkeypatch.setattr(glm_mtp_selection, "backfill_mtp_selection_wires",
                            lambda *_args: {"__prismaquant__": {"mtp_selection": selection}})
        monkeypatch.setattr(glm_mtp_selection, "_bound_payload", lambda *_args, **_kwargs: priced)
        proof = export._mtp_hessian_collection_proof(
            {}, {"mtp_selection": selection}, {"a": fmt, "b": fmt}, block, owner)
        assert proof["unit_capture_sha256"] == {"a": body_seal, "b": mtp_seal}
        assert proof["binding"] == owner.binding()

        tampered = copy.deepcopy(priced)
        tampered["costs"]["b"][fmt]["hessian_identity"]["capture_sha256"] = "0" * 64
        monkeypatch.setattr(glm_mtp_selection, "_bound_payload", lambda *_args, **_kwargs: tampered)
        with pytest.raises(export.TesseraExportLaneError, match="receipt H differs"):
            export._mtp_hessian_collection_proof(
                {}, {"mtp_selection": selection}, {"a": fmt, "b": fmt}, block, owner)

        with pytest.raises(export.TesseraExportLaneError, match="body selected unit"):
            export._mtp_hessian_collection_proof(
                {}, {"mtp_selection": selection}, {"a": fmt, "b": fmt,
                                                             "uncovered": fmt}, block, owner)
