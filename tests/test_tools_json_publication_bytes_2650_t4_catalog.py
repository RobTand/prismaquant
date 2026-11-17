"""The T4 catalog tools publish the same JSON bytes through bound profiles.

The literals below come from the unedited tools. The catalog builder runs end
to end on a one-unit fixture. A patch of ``catalog_view`` swaps the catalog for
``EDGE``, so that the bytes do not depend on paths, inodes or times.

CPU-only; every file is in ``tmp_path``.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from prismaquant import format_registry as fr  # noqa: E402
from prismaquant.cost_stage_checkpoint import canonical_json_sha256, unit_path, write_unit  # noqa: E402
from prismaquant.digests import JsonProfile, bytes_sha256hex as sha  # noqa: E402
from prismaquant.joint_aura import activation_identity  # noqa: E402
from prismaquant.joint_catalog_extension import added_format_recipes  # noqa: E402
from prismaquant.tessera_joint_aura import STAGE  # noqa: E402
from tools import assemble_t4_overlay as assemble  # noqa: E402
from tools import build_t4_overlay_catalog as builder  # noqa: E402

QNAME = "model.language_model.layers.3.mlp.experts.7.down_proj"
OLDFMT = "TESSERA_E4M3_K1_R1024"
SEAL = "5" * 64
IDENTITY = {"unit": QNAME, "source": {"sha256": "1" * 64}, "projection": "down",
            "calibration": {"sha256": "2" * 64}, "encoder_fixture_id": "f" * 64,
            "encoder_source_sha256": "4" * 64}
EDGE = {"schema": "edge", "status": "edge-status", "text": "caf\u00e9 \u2603 \U0001f600 \"q\" \\ \n\t",
        "ints": {10: "ten", 2: "two", 1: "one"}, "mixed": {"b": 1, "a": [1, 2.5, [], {}, None, True], "": {}},
        "nan": float("nan"), "inf": float("inf"), "ninf": float("-inf"), "empty": {},
        "nested": [[1, [2]], {"z": {"y": []}}]}
DOC_BYTES = (
    b'{\n  "empty": {},\n  "inf": Infinity,\n  "ints": {\n    "1": "one",\n    "2": "two",\n    "10": "ten"\n  },\n'
    b'  "mixed": {\n    "": {},\n    "a": [\n      1,\n      2.5,\n      [],\n      {},\n      null,\n      true\n'
    b'    ],\n    "b": 1\n  },\n  "nan": NaN,\n  "nested": [\n    [\n      1,\n      [\n        2\n      ]\n    ],\n'
    b'    {\n      "z": {\n        "y": []\n      }\n    }\n  ],\n  "ninf": -Infinity,\n  "schema": "edge",\n'
    b'  "status": "edge-status",\n'
    b'  "text": "caf\\u00e9 \\u2603 \\ud83d\\ude00 \\"q\\" \\\\ \\n\\t"\n}\n')
DOC_SHA = "42d2bf584178492784bd834589d1cff94aabd2ed1510d588d763d92074cf9aae"
CATALOG_BYTES = (
    b'{"empty":{},"inf":Infinity,"ints":{"1":"one","2":"two","10":"ten"},"mixed":{"":{},"a":[1,2.5,[],{},null,true],'
    b'"b":1},"nan":NaN,"nested":[[1,[2]],{"z":{"y":[]}}],"ninf":-Infinity,"schema":"edge","status":"edge-status",'
    b'"text":"caf\\u00e9 \\u2603 \\ud83d\\ude00 \\"q\\" \\\\ \\n\\t"}\n')
CATALOG_SHA = "e446682772cba89f13a8f84d27f87e922de5c9d34e7ac826e6f87f0d60efdbc8"


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": str(path), "sha256": sha(raw)}


def _render_path(row_dir, fmt):
    return row_dir / "cache" / (QNAME.replace("/", "__").replace(".", "_") + "__" + fmt + ".pt")


def _fixture(root):
    """The R13 layout under a ``--census-base`` of ``root / 'base'``: one unit, one format."""
    maxima = {QNAME: 0.75}
    old_identity = {**IDENTITY, "recipe": {"old": True}}
    verified = {"wire_sha256": "9" * 64, "source_weight": {"shape": [16, 16]},
                "encoding_identity_sha256": canonical_json_sha256(old_identity, where="fixture old identity")}
    cache = SimpleNamespace(metadata={"verified_cells": {(QNAME, OLDFMT): verified}}, activation_max_abs=maxima)
    pwc = _write(root / "original" / "production.pkl", pickle.dumps(cache))
    prepared = {"production_cache": pwc, "formats_by_qname": {QNAME: [OLDFMT, "BF16"]},
                "source_model_identity": {"model": "fixture"}, "calibration_input": {"shape": [8, 8]},
                "source_execution": {"fixture": True}, "reader_identity": {"reader": "fixture"},
                "projection_backend": {"backend": "fixture"}}
    prepared_path = root / "original" / "prepared.json"
    _write(prepared_path, json.dumps(prepared).encode())
    ref_dir = root / "reference-row"
    write_unit(ref_dir / "cost.anchors.json.parts", stage=STAGE, qname=QNAME, identity_sha256=SEAL,
               state={"wire_records": {OLDFMT: {"blob_sha256": verified["wire_sha256"], "identity": old_identity}}})
    base = root / "base"
    workspace = base / builder.LEGACY_WORKSPACE_COST.relative_to(builder.BASE).parent
    fmt = builder.FMT
    wire = workspace / "wires" / f"{fmt}.tsr"
    _write(wire, b"w" * 100)
    _write(_render_path(root / "e2m1-row", fmt), b"r" * 200)
    record = {"file": wire.name, "blob_bytes": 100,
              "identity": {**IDENTITY, "encoder_source_sha256": "8" * 64, "recipe": added_format_recipes()(fmt)}}
    scale = activation_identity(fr.get_format(fmt), maxima, QNAME)["input_global_scale"]
    anchor = {"qname": QNAME, "family": fmt.rsplit("_R", 1)[0], "format_name": fmt,
              "body_rate_q256": int(fmt.rsplit("_R", 1)[1]), "dloss": 1e-5, "activation_contract": "fixture",
              "activation_quantized": True, "input_global_scale": scale, "wire_bytes": 100}
    scalar = {"output_mse": anchor["dloss"], "tessera_family": anchor["family"],
              "tessera_body_rate_q256": anchor["body_rate_q256"], "activation_contract": "fixture",
              "activation_quantized": True, "input_global_scale": scale, "wire_bytes": 100,
              "output_mse_measured": True, "cost_source": "tessera_campaign_measured",
              "tessera_provenance": "measured", "currency": builder.MEASURED_CURRENCY}
    parts = workspace / "cost.anchors.json.parts"
    write_unit(parts, stage=STAGE, qname=QNAME, identity_sha256=SEAL,
               state={"anchors": [anchor], "wire_records": {fmt: record}})
    _write(workspace / "cost.anchors.json", json.dumps({
        "identity_sha256": SEAL, "stage": STAGE,
        "units": [{"qname": QNAME, "file": str(unit_path(parts, QNAME).relative_to(parts))}]}).encode())
    cost = {"costs": {QNAME: {fmt: scalar}}, "provenance": {"wire_dir": str(workspace / "wires")},
            "tessera_expert_wires": {QNAME: {fmt: record}}}
    _write(workspace / "cost.pkl", pickle.dumps(cost))
    _write(workspace.parent / "plan.json",
           json.dumps({"rows": [{"dir": str(root / "e2m1-row"), "members": [QNAME]}]}).encode())
    _write(base / builder.REFERENCE_PLAN.relative_to(builder.BASE),
           json.dumps({"rows": [{"dir": str(ref_dir), "members": [QNAME]}]}).encode())
    proof = _write(root / "proof.json", b'{"fixture": "proof"}\n')
    return prepared_path, base, proof


def _build(tmp_path, monkeypatch, before_publish=None):
    """Run the builder ``main`` with ``EDGE`` as the catalog; return the out path and the catalog object."""
    prepared, base, proof = _fixture(tmp_path)
    out = tmp_path / "out" / "catalog.json"
    seen = []

    def view(catalog):
        seen.append(catalog)
        catalog.clear()
        catalog.update(EDGE)
        if before_publish:
            before_publish(out)

    monkeypatch.setattr(builder, "PROOF", Path(proof["path"]))
    monkeypatch.setattr(builder, "PROOF_SHA256", proof["sha256"])
    monkeypatch.setattr(builder, "catalog_view", view)
    monkeypatch.setattr(builder, "catalog_sources", lambda catalog: [0])
    monkeypatch.setattr(sys, "argv", ["build_t4_overlay_catalog.py", "--out", str(out), "--prepared",
                                      str(prepared), "--census-base", str(base)])
    return out, seen


def test_t4_catalog_assembler_doc_bytes():
    raw = assemble.doc(EDGE)
    assert raw == DOC_BYTES, repr(raw)
    assert assemble.sha(raw) == DOC_SHA
    assert assemble.doc({}) == b"{}\n"


def test_t4_catalog_assembler_publish_bytes_and_refusals(tmp_path):
    path = tmp_path / "pub" / "doc.json"
    assert assemble.publish(path, DOC_BYTES) == {"path": str(path), "sha256": DOC_SHA}
    assert path.read_bytes() == DOC_BYTES
    assert not path.with_suffix(".json.tmp").exists()
    with pytest.raises(AssertionError) as exists:
        assemble.publish(path, b"other\n")
    assert str(exists.value) == str(path) and exists.value.__cause__ is None
    assert path.read_bytes() == DOC_BYTES
    fresh = tmp_path / "pub" / "fresh.json"
    stale = fresh.with_suffix(".json.tmp")
    stale.write_bytes(b"stale")
    with pytest.raises(FileExistsError) as busy:
        assemble.publish(fresh, b"new\n")
    assert str(busy.value) == f"[Errno 17] File exists: '{stale}'" and busy.value.__cause__ is None
    assert not fresh.exists() and stale.read_bytes() == b"stale"


def test_t4_catalog_builder_bytes_and_stdout(tmp_path, monkeypatch, capsysbinary):
    out, _ = _build(tmp_path, monkeypatch)
    builder.main()
    raw = out.read_bytes()
    assert raw == CATALOG_BYTES, repr(raw)
    assert sha(raw) == CATALOG_SHA
    printed = re.sub(rb'"seconds": [0-9.e+-]+', b'"seconds": 0', capsysbinary.readouterr().out)
    assert printed == (
        f'{{"path": "{out}", "sha256": "{CATALOG_SHA}", "schema": "edge", "cells": 1, '
        f'"cells_by_format": {{"{builder.FMT}": 1}}, "carried_from": [], "seconds": 0, '
        f'"status": "edge-status"}}\n').encode()


def test_t4_catalog_builder_refusal_leaves_the_file_alone(tmp_path, monkeypatch, capsysbinary):
    def occupy(out):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"kept\n")

    out, _ = _build(tmp_path, monkeypatch, occupy)
    with pytest.raises(FileExistsError) as busy:
        builder.main()
    assert str(busy.value) == f"[Errno 17] File exists: '{out}'" and busy.value.__cause__ is None
    assert out.read_bytes() == b"kept\n"
    assert capsysbinary.readouterr().out == b""


def test_t4_catalog_profile_calls(tmp_path, monkeypatch):
    calls = []
    real = JsonProfile.text

    def spy(self, value):
        calls.append((self.name, value))
        return real(self, value)

    monkeypatch.setattr(JsonProfile, "text", spy)
    assemble.doc(EDGE)
    assert [(name, value is EDGE) for name, value in calls] == [("direct-ascii-indent2-lax", True)]
    del calls[:]
    out, seen = _build(tmp_path, monkeypatch)
    builder.main()
    assert [name for name, value in calls if value is seen[0]] == ["direct-ascii-lax"]
