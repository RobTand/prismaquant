"""T4 overlay catalog: journal anchors, result rebinding, and script hygiene.

The first catalog stored each cell's scalar cost row as its anchor, and
``attach_candidate_overlay`` refused all 36,288 cells (PB 26073882c2ca,
c45669ebd5ce). The builder now takes the E2M1 journal's ``CampaignAnchor`` row
through the journal's own sealed reader. Rebuilding changes every cell digest,
so the qualification results are rebound by a per-cell proof that only the
anchor changed.

CPU-only; no campaign file is read.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import assemble_t4_overlay as assemble  # noqa: E402
import build_t4_overlay_catalog as builder  # noqa: E402
import rebind_t4_qualified_results as rebind  # noqa: E402
from prismaquant.cost_stage_checkpoint import unit_path, write_unit  # noqa: E402
from prismaquant.tessera_campaign import CampaignAnchor  # noqa: E402
from prismaquant.tessera_joint_aura import STAGE  # noqa: E402

FMT = builder.FMT
QNAME = "model.language_model.layers.3.mlp.experts.7.down_proj"
SEAL = "5" * 64


def _anchor(**changes):
    row = {"qname": QNAME, "family": "TESSERA_E2M1_K2", "format_name": FMT,
           "body_rate_q256": 896, "dloss": 3.36e-05, "dloss_stderr": 0.0,
           "memory_bytes": 4194816, "bits_per_param": 4.0005, "activation_contract": "fp4_e2m1",
           "activation_quantized": True, "wire_bytes": 4195550, "seconds": 6.07,
           "hessian_applied": True, "input_global_scale": 0.558, "encoding_batch_size": 8}
    row.update(changes)
    return row


def _scalar(anchor):
    return {"output_mse": anchor["dloss"], "tessera_family": anchor["family"],
            "tessera_body_rate_q256": anchor["body_rate_q256"],
            "activation_contract": anchor["activation_contract"],
            "activation_quantized": anchor["activation_quantized"],
            "input_global_scale": anchor["input_global_scale"], "wire_bytes": anchor["wire_bytes"],
            "output_mse_measured": True, "cost_source": "tessera_campaign_measured",
            "tessera_provenance": "measured", "currency": builder.MEASURED_CURRENCY}


def _journal(tmp_path, anchors, record):
    parts = tmp_path / "cost.anchors.json.parts"
    write_unit(parts, stage=STAGE, qname=QNAME, identity_sha256=SEAL,
               state={"anchors": anchors, "wire_records": {FMT: record}})
    manifest = tmp_path / "cost.anchors.json"
    manifest.write_text(json.dumps({
        "identity_sha256": SEAL, "stage": STAGE,
        "units": [{"qname": QNAME, "file": str(unit_path(parts, QNAME).relative_to(parts))}]}))
    return manifest


# ------------------------------------------------------------ journal anchors

def test_the_anchor_is_the_journal_row_and_loads_as_a_campaign_anchor(tmp_path):
    record = {"file": "wire.tsr", "blob_bytes": 4195550}
    manifest = _journal(tmp_path, [_anchor(), _anchor(format_name="OTHER")], record)
    journal, _digest, units = builder.open_anchor_journal(manifest)
    anchor, wire = builder.journal_anchor(units[QNAME], stage=STAGE, qname=QNAME,
                                          identity_sha256=journal["identity_sha256"], fmt=FMT)
    assert anchor == _anchor() and wire == record
    builder.require_anchor_matches_scalar(anchor, _scalar(anchor), qname=QNAME)
    assert CampaignAnchor(**anchor).dloss == _anchor()["dloss"]


def test_the_scalar_cost_row_is_not_an_anchor():
    """The first catalog's anchor spelling does not construct."""
    with pytest.raises(TypeError):
        CampaignAnchor(**_scalar(_anchor()))


@pytest.mark.parametrize("field,value", [("dloss", 1.0), ("family", "TESSERA_E4M3_K1"),
                                          ("body_rate_q256", 832), ("input_global_scale", 0.5),
                                          ("wire_bytes", 1)])
def test_a_journal_row_the_loader_would_refuse_refuses_the_build(field, value):
    anchor = _anchor()
    scalar = _scalar(anchor)
    with pytest.raises(AssertionError):
        builder.require_anchor_matches_scalar(_anchor(**{field: value}), scalar, qname=QNAME)


def test_a_foreign_journal_seal_refuses(tmp_path):
    manifest = _journal(tmp_path, [_anchor()], {})
    _journal_doc, _digest, units = builder.open_anchor_journal(manifest)
    with pytest.raises(RuntimeError, match="identity_sha256"):
        builder.journal_anchor(units[QNAME], stage=STAGE, qname=QNAME,
                               identity_sha256="6" * 64, fmt=FMT)


# ----------------------------------------------------------------- rebinding

def _cells():
    previous = {"qname": QNAME, "format": FMT, "anchor": _scalar(_anchor()),
                "record": {"blob_sha256": "b" * 64}, "render": "/r.pt"}
    return previous, {**previous, "anchor": _anchor()}


def _result(cell):
    value = {"qname": QNAME, "format": FMT, "cell_sha256": rebind.cell_sha256(cell),
             "verified_cell": {"render_origin": "encoded"}}
    value["verified_cell_sha256"] = rebind.cell_sha256(value["verified_cell"])
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def test_an_anchor_only_change_rebinds_and_the_assembler_admits_it():
    previous, new = _cells()
    raw = _result(previous)
    row = rebind.rebind_cell(new, previous, raw)
    assert row["previous_cell_sha256"] != row["cell_sha256"]
    value = assemble.qualified_result(new, raw, {(QNAME, FMT): row})
    assert value["verified_cell"] == {"render_origin": "encoded"}
    # Without the rebinding the result does not bind the new cell.
    with pytest.raises(AssertionError):
        assemble.qualified_result(new, raw, None)


def test_a_change_beyond_the_anchor_refuses_rebinding():
    previous, new = _cells()
    new = {**new, "render": "/other.pt"}
    with pytest.raises(AssertionError, match="other than the anchor"):
        rebind.rebind_cell(new, previous, _result(previous))


@pytest.mark.parametrize("tamper", ["result", "cell", "previous_anchor"])
def test_the_assembler_rechecks_every_rebinding_row(tamper):
    previous, new = _cells()
    raw = _result(previous)
    row = rebind.rebind_cell(new, previous, raw)
    if tamper == "result":
        raw = raw.replace(b"encoded", b"synthesized")
    elif tamper == "cell":
        new = {**new, "render": "/other.pt"}
    else:
        row = {**row, "previous_anchor": {"output_mse": 0.0}}
    with pytest.raises(AssertionError):
        assemble.qualified_result(new, raw, {(QNAME, FMT): row})


# --------------------------------------------- a prepared rebuilt for R13

def _bind(path, value):
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": rebind.sha(raw)}


def _rebind_fixture(tmp_path, *, new_prepared_changes=None, same_prepared=False):
    """Two catalogs whose cells differ in their anchor and whose headers
    bind the prepared of R12 and of R13, which differ in ``plan_sha256``."""
    prepared = {"plan_sha256": "a" * 64, "production_cache": {"sha256": "c" * 64}}
    old = _bind(tmp_path / "r12-prepared.json", prepared)
    new = old if same_prepared else _bind(
        tmp_path / "r13-prepared.json", {**prepared, "plan_sha256": "b" * 64, **(new_prepared_changes or {})})
    previous_cell, new_cell = _cells()
    header = {"schema": "prismaquant.t4_adopted_catalog.v1", "format": FMT}
    previous = _bind(tmp_path / "previous.json", {**header, "old_prepared": old, "cells": [previous_cell]})
    catalog = _bind(tmp_path / "catalog.json", {**header, "old_prepared": new, "cells": [new_cell]})
    qualified = tmp_path / "qualified"
    qualified.mkdir()
    rebind.result_path(qualified, QNAME).write_bytes(_result(previous_cell))
    argv = ["rebind_t4_qualified_results.py", "--catalog", catalog["path"],
            "--catalog-sha256", catalog["sha256"], "--previous-catalog", previous["path"],
            "--previous-catalog-sha256", previous["sha256"], "--qualified-dir", str(qualified),
            "--out", str(tmp_path / "rebinding.json")]
    return argv, old, new, tmp_path / "rebinding.json"


def test_a_prepared_that_changed_only_its_plan_digest_rebinds(tmp_path, monkeypatch):
    """R13 re-prepared R12's cache under its own plan (RobTand/prismaquant#1117)."""
    argv, old, new, out = _rebind_fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    rebind.main()
    document = json.loads(out.read_bytes())
    assert document["old_prepared_rebound"] == {"previous": old, "new": new,
                                                "differs_only_in": ["plan_sha256"]}
    assert len(document["rows"]) == 1


def test_an_unchanged_prepared_writes_no_rebound_field(tmp_path, monkeypatch):
    argv, _old, _new, out = _rebind_fixture(tmp_path, same_prepared=True)
    monkeypatch.setattr(sys, "argv", argv)
    rebind.main()
    assert "old_prepared_rebound" not in json.loads(out.read_bytes())


@pytest.mark.parametrize("changes", [{"production_cache": {"sha256": "d" * 64}}, {"extra": 1}])
def test_a_prepared_that_changed_beyond_its_plan_digest_refuses(tmp_path, monkeypatch, changes):
    argv, _old, _new, out = _rebind_fixture(tmp_path, new_prepared_changes=changes)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(AssertionError, match="beyond its plan digest"):
        rebind.main()
    assert not out.exists()


def test_a_rebuilt_prepared_that_does_not_match_its_binding_refuses(tmp_path, monkeypatch):
    argv, _old, new, out = _rebind_fixture(tmp_path)
    Path(new["path"]).write_bytes(b"{}")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(AssertionError, match="SHA256 mismatch"):
        rebind.main()
    assert not out.exists()


# ------------------------------------------------------------ script hygiene

IMPORT_SAFE = ["audit_existing_t4.py", "audit_t4_render_paths.py", "pilot_existing_t4.py",
               "inspect_prepared_cache.py", "inspect_t4_binding.py", "build_t4_overlay_catalog.py",
               "assemble_t4_overlay.py", "build_t4_logical_request.py", "build_t4_recovery_request.py",
               "rebind_t4_qualified_results.py", "check_t4_overlay_prepare.py"]


def _is_main_guard(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and getattr(node.test.left, "id", None) == "__name__")


@pytest.mark.parametrize("name", IMPORT_SAFE)
def test_the_campaign_scripts_do_nothing_at_import(name):
    tree = ast.parse((ROOT / "tools" / name).read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) or _is_main_guard(node):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # docstring
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            calls = [n for n in ast.walk(node.value) if isinstance(n, ast.Call)]
            assert all(getattr(c.func, "id", None) == "Path" for c in calls), \
                f"{name}: call at import (line {node.lineno})"
            continue
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "attr", None) == "insert"
                and getattr(getattr(node.value.func, "value", None), "attr", None) == "path"):
            continue  # sys.path.insert
        pytest.fail(f"{name}: statement runs at import (line {node.lineno})")


PREVIOUS_CATALOG_SHA256 = "71238bdab85bdccabed921ce94b03459d5aba3607021b85007bf41f3e376fbc8"


@pytest.mark.parametrize("name", ["build_t4_logical_request.py", "build_t4_recovery_request.py",
                                  "assemble_t4_overlay.py"])
def test_request_and_assembly_inputs_are_arguments(name):
    """No checkout, venv or catalog digest is baked in (WS-M finding b)."""
    source = (ROOT / "tools" / name).read_text()
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert not [s for s in strings if s.startswith("/home/")], name
    assert PREVIOUS_CATALOG_SHA256 not in source, name
    assert "1902e677b456b0a6e40da2fe45fe0bd38fccee5226e238359d4f9b060a963bf5" not in source, name
