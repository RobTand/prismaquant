"""The T4 overlay catalog builder, end to end on a fixture campaign (PQ #1437).

The builder takes its inputs as flags since PQ #1432. With no ``--workspace``
and no ``--carry`` it rebuilds the R13 A4 catalog, and it must still write the
v1 bytes it always wrote. ``--carry`` copies an existing catalog's cells byte
for byte, so the rebinding of that catalog still admits them, and
``--workspace`` adds cells from a merged campaign workspace, with or without a
reseal proof.

CPU-only; the fixture is one routed unit in ``tmp_path``.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import assemble_t4_overlay as assemble  # noqa: E402
import build_t4_overlay_catalog as builder  # noqa: E402
from prismaquant import format_registry as fr  # noqa: E402
from prismaquant.cost_stage_checkpoint import canonical_json_sha256, unit_path, write_unit  # noqa: E402
from prismaquant.digests import bytes_sha256hex as sha  # noqa: E402
from prismaquant.joint_aura import activation_identity  # noqa: E402
from prismaquant.joint_catalog_extension import (  # noqa: E402
    CATALOG_SCHEMA_V1, CATALOG_SCHEMA_V2, added_format_recipes, catalog_view)
from prismaquant.tessera_joint_aura import STAGE  # noqa: E402

QNAME = "model.language_model.layers.3.mlp.experts.7.down_proj"
OLDFMT = "TESSERA_E4M3_K1_R1024"
R896 = builder.FMT
R768 = "TESSERA_E2M1_K2_R768"
E4M3 = "TESSERA_E4M3_K1_R1152"
SEAL = "5" * 64
IDENTITY = {"unit": QNAME, "source": {"sha256": "1" * 64}, "projection": "down",
            "calibration": {"sha256": "2" * 64}, "encoder_fixture_id": "f" * 64,
            "encoder_source_sha256": "4" * 64}


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": str(path), "sha256": sha(raw)}


def _render_path(row_dir, fmt):
    return row_dir / "cache" / (QNAME.replace("/", "__").replace(".", "_") + "__" + fmt + ".pt")


def _original(root):
    """The original prepared, its PWC and the plan owning the reference cell."""
    maxima = {QNAME: 0.75}
    old_identity = {**IDENTITY, "recipe": {"old": True}}
    verified = {"wire_sha256": "9" * 64, "source_weight": {"shape": [16, 16]},
                "encoding_identity_sha256": canonical_json_sha256(old_identity, where="fixture old identity")}
    cache = SimpleNamespace(metadata={"verified_cells": {(QNAME, OLDFMT): verified}},
                            activation_max_abs=maxima)
    pwc = _write(root / "original" / "production.pkl", pickle.dumps(cache))
    prepared = {"production_cache": pwc, "formats_by_qname": {QNAME: [OLDFMT, "BF16"]},
                "source_model_identity": {"model": "fixture"}, "calibration_input": {"shape": [8, 8]},
                "source_execution": {"fixture": True}, "reader_identity": {"reader": "fixture"},
                "projection_backend": {"backend": "fixture"}}
    prepared_path = root / "original" / "prepared.json"
    _write(prepared_path, json.dumps(prepared).encode())
    ref_dir = root / "reference-row"
    write_unit(ref_dir / "cost.anchors.json.parts", stage=STAGE, qname=QNAME, identity_sha256=SEAL,
               state={"wire_records": {OLDFMT: {"blob_sha256": verified["wire_sha256"],
                                                "identity": old_identity}}})
    return prepared_path, maxima, ref_dir


def _workspace(workspace, row_dir, formats, maxima, *, expert_wires=True):
    """A merged workspace pricing ``formats`` for the unit: cost, journal, wires, renders."""
    recipe = added_format_recipes()
    anchors, records, scalars = [], {}, {}
    wire_dir = workspace / "wires"
    for index, fmt in enumerate(formats):
        wire = wire_dir / f"{fmt}.tsr"
        _write(wire, b"w" * (100 + index))
        _write(_render_path(row_dir, fmt), b"r" * (200 + index))
        records[fmt] = {"file": wire.name, "blob_bytes": 100 + index,
                        "identity": {**IDENTITY, "encoder_source_sha256": "8" * 64, "recipe": recipe(fmt)}}
        scale = activation_identity(fr.get_format(fmt), maxima, QNAME)["input_global_scale"]
        anchor = {"qname": QNAME, "family": fmt.rsplit("_R", 1)[0], "format_name": fmt,
                  "body_rate_q256": int(fmt.rsplit("_R", 1)[1]), "dloss": 1e-5 * (index + 1),
                  "activation_contract": "fixture", "activation_quantized": True,
                  "input_global_scale": scale, "wire_bytes": 100 + index}
        anchors.append(anchor)
        scalars[fmt] = {"output_mse": anchor["dloss"], "tessera_family": anchor["family"],
                        "tessera_body_rate_q256": anchor["body_rate_q256"],
                        "activation_contract": anchor["activation_contract"],
                        "activation_quantized": True, "input_global_scale": scale,
                        "wire_bytes": anchor["wire_bytes"], "output_mse_measured": True,
                        "cost_source": "tessera_campaign_measured", "tessera_provenance": "measured",
                        "currency": builder.MEASURED_CURRENCY}
    parts = workspace / "cost.anchors.json.parts"
    write_unit(parts, stage=STAGE, qname=QNAME, identity_sha256=SEAL,
               state={"anchors": anchors, "wire_records": records})
    _write(workspace / "cost.anchors.json", json.dumps({
        "identity_sha256": SEAL, "stage": STAGE,
        "units": [{"qname": QNAME, "file": str(unit_path(parts, QNAME).relative_to(parts))}]}).encode())
    cost = {"costs": {QNAME: scalars}, "provenance": {"wire_dir": str(wire_dir)}}
    if expert_wires:
        cost["tessera_expert_wires"] = {QNAME: records}
    _write(workspace / "cost.pkl", pickle.dumps(cost))
    plan = workspace.parent / "plan.json"
    _write(plan, json.dumps({"rows": [{"dir": str(row_dir), "members": [QNAME]}]}).encode())
    return workspace / "cost.pkl", plan


def _legacy_fixture(root):
    """The R13 layout under a ``--census-base`` of ``root / 'base'``."""
    prepared, maxima, ref_dir = _original(root)
    base = root / "base"
    rel = builder.LEGACY_WORKSPACE_COST.relative_to(builder.BASE)
    _workspace(base / rel.parent, root / "e2m1-row", [R896], maxima)
    _write(base / builder.REFERENCE_PLAN.relative_to(builder.BASE),
           json.dumps({"rows": [{"dir": str(ref_dir), "members": [QNAME]}]}).encode())
    proof = _write(root / "proof.json", b'{"fixture": "proof"}\n')
    return SimpleNamespace(prepared=prepared, base=base, proof=proof, maxima=maxima, ref_dir=ref_dir)


def _run(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["build_t4_overlay_catalog.py", *argv])
    module.main()


def _legacy(module, fx, monkeypatch, out):
    monkeypatch.setattr(module, "PROOF", Path(fx.proof["path"]))
    monkeypatch.setattr(module, "PROOF_SHA256", fx.proof["sha256"])
    _run(module, monkeypatch, "--out", str(out), "--prepared", str(fx.prepared),
         "--census-base", str(fx.base))
    return out.read_bytes()


def _pre1432_builder():
    """The builder as it was before PQ #1432 (4fd2bb39), for the byte comparison."""
    spec = importlib.util.spec_from_file_location(
        "pre1432_build_t4_overlay_catalog", ROOT / "tests/fixtures/_pre1432_build_t4_overlay_catalog.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_no_flag_build_writes_the_bytes_the_pre_1432_builder_wrote(tmp_path, monkeypatch):
    fx = _legacy_fixture(tmp_path)
    old = _legacy(_pre1432_builder(), fx, monkeypatch, tmp_path / "old-catalog.json")
    new = _legacy(builder, fx, monkeypatch, tmp_path / "new-catalog.json")
    assert new == old
    catalog = json.loads(new)
    assert catalog["schema"] == CATALOG_SCHEMA_V1 and catalog["format"] == R896
    assert catalog_view(catalog)["formats"] == (R896,)


def _carry_and_workspace(tmp_path, monkeypatch, *extra):
    """The legacy catalog carried, plus a proofless workspace of two new formats."""
    fx = _legacy_fixture(tmp_path)
    legacy_path = tmp_path / "r13-catalog.json"
    legacy_raw = _legacy(builder, fx, monkeypatch, legacy_path)
    carried = {"path": str(legacy_path), "sha256": sha(legacy_raw)}
    cost, plan = _workspace(tmp_path / "gamut" / "merged", tmp_path / "gamut-row", [R768, E4M3],
                            fx.maxima, expert_wires=False)
    reference = tmp_path / "base" / builder.REFERENCE_PLAN.relative_to(builder.BASE)
    out = tmp_path / "gamut-catalog.json"
    _run(builder, monkeypatch, "--out", str(out), "--prepared", str(fx.prepared),
         "--carry", carried["path"], carried["sha256"], "--workspace", str(cost), str(plan),
         "--reference-plan", str(reference), "--no-proof", *extra)
    return json.loads(legacy_raw), carried, json.loads(out.read_bytes()), out


def test_a_carry_and_workspace_build_writes_one_v2_catalog(tmp_path, monkeypatch):
    legacy, carried, catalog, out = _carry_and_workspace(tmp_path, monkeypatch)
    view = catalog_view(catalog)
    assert catalog["schema"] == CATALOG_SCHEMA_V2
    assert view["formats"] == (R768, R896, E4M3)
    assert catalog["carried_from"] == [carried]
    # Carried cells are copied byte for byte; their source is the carried catalog's.
    assert catalog["cells"][0] == legacy["cells"][0]
    assert catalog["sources"][0] == {key: legacy[key] for key in ("cost", "anchor_journal", "reseal_proof")}
    assert catalog["cell_sources"] == [0, 1, 1]
    new = {cell["format"]: cell for cell in catalog["cells"][1:]}
    assert set(new) == {R768, E4M3}
    assert catalog["sources"][1]["reseal_proof"] is None
    for fmt, cell in new.items():
        assert cell["catalog_source_adoption"]["encoder_source_proof"] is None
        assert cell["adopted_source_hessian"]["reseal_proof_sha256"] is None
        assert cell["record"]["identity"]["recipe"] == added_format_recipes()(fmt)
        assert cell["catalog_source_adoption"]["reference_pair"] == [QNAME, OLDFMT]


def test_a_rebinding_of_the_carried_catalog_still_admits_its_cells(tmp_path, monkeypatch):
    """R13's rebinding binds the carried catalog; the v2 catalog declares it
    in ``carried_from``, so the assembler reads the carried cells' results
    through it and the new cells' results directly."""
    legacy, carried, catalog, out = _carry_and_workspace(tmp_path, monkeypatch)
    cell = legacy["cells"][0]
    row = {"qname": QNAME, "format": R896, "result_sha256": "0" * 64, "previous_cell_sha256": "0" * 64,
           "cell_sha256": "0" * 64, "previous_anchor": cell["anchor"]}
    rebinding = {"schema": assemble.REBINDING_SCHEMA, "catalog": carried,
                 "qualified_dir": str(tmp_path / "qualified"), "rows": [row]}
    rows = assemble.rebinding_rows_for(
        rebinding, catalog_binding={"path": str(out), "sha256": sha(out.read_bytes())},
        catalog=catalog, qualified_dir=lambda fmt: tmp_path / "qualified")
    assert set(rows) == {(QNAME, R896)}


def test_a_workspace_build_names_its_proof_or_declares_none(tmp_path, monkeypatch, capsys):
    fx = _legacy_fixture(tmp_path)
    cost, plan = _workspace(tmp_path / "gamut" / "merged", tmp_path / "gamut-row", [R768], fx.maxima)
    with pytest.raises(SystemExit):
        _run(builder, monkeypatch, "--out", str(tmp_path / "x.json"), "--prepared", str(fx.prepared),
             "--workspace", str(cost), str(plan))
    assert "--no-proof" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


def test_a_carried_catalog_of_another_original_refuses(tmp_path, monkeypatch):
    fx = _legacy_fixture(tmp_path)
    legacy_path = tmp_path / "r13-catalog.json"
    legacy = json.loads(_legacy(builder, fx, monkeypatch, legacy_path))
    legacy["old_prepared"] = {"path": "/elsewhere/prepared.json", "sha256": "0" * 64}
    foreign = _write(tmp_path / "foreign-catalog.json", json.dumps(legacy).encode())
    with pytest.raises(AssertionError, match="extends another prepared"):
        _run(builder, monkeypatch, "--out", str(tmp_path / "y.json"), "--prepared", str(fx.prepared),
             "--carry", foreign["path"], foreign["sha256"])
    assert not (tmp_path / "y.json").exists()
