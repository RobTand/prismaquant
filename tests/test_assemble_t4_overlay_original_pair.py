"""The T4 overlay assembler takes its original plan and prepared as arguments.

The assembler bound R12's pair as constants, so it could not build a Stage B
A4 pair against R13, and ``joint_catalog_extension`` refuses a band whose run
header does not name the pair's originals (RobTand/prismaquant#1117). The
pair is now named by ``--original-plan`` and ``--original-prepared``, each
with its SHA-256. With no flags, the assembler binds the old constants and
publishes the same bytes.

CPU-only; the fixture is a two-cell overlay in ``tmp_path``.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import assemble_t4_overlay as assemble  # noqa: E402
import rebind_t4_qualified_results as rebind  # noqa: E402

FMT = "TESSERA_E2M1_K2_R3"
QNAMES = ["model.layers.3.mlp.experts.7.down_proj", "model.layers.4.mlp.experts.1.up_proj"]
PUBLISHED = ["plan.json", "prepare/production.pkl", "prepare/prepared.json", "catalog-pair-inputs.json"]


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": str(path), "sha256": _sha(raw)}


def _stat(path):
    s = path.stat()
    return dict(inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def _pair(root, name, cache_binding):
    """One original plan and prepared, told apart by ``name``."""
    plan = {"name": name, "inputs": {"source": name}, "output_root": str(root / name),
            "execution": {"retained_operator_windows": {"budget": 1}}, "max_gpu_bytes": 1}
    prepared = {"name": name, "production_cache": cache_binding,
                "formats_by_qname": {q: ["NVFP4", "BF16"] for q in QNAMES}}
    plan_binding = _write(root / name / "plan.json", json.dumps(plan).encode())
    prepared_binding = _write(root / name / "prepare/prepared.json", json.dumps(prepared).encode())
    return plan_binding, prepared_binding


def _fixture(root):
    cache = SimpleNamespace(weights={("model.embed", "BF16"): "/embed.pt"},
                            _lru_paths={}, metadata={"verified_cells": {}})
    cache_binding = _write(root / "production.pkl", pickle.dumps(cache))
    r12 = _pair(root, "r12", cache_binding)
    r13 = _pair(root, "r13", cache_binding)
    served = _write(root / "served.json", b'{"served": true}\n')
    resources = _write(root / "resources.json", json.dumps(
        {"budget": 7, "limits": {"gpu_bytes": 9}}).encode())
    cells = []
    for index, qname in enumerate(QNAMES):
        wire = root / "wire" / f"{index}.tsr"
        render = root / "render" / f"{index}.pt"
        _write(wire, b"wire" * (index + 1))
        _write(render, b"render" * (index + 1))
        cell = {"qname": qname, "format": FMT, "source_weight": {"shape": [4, 4]},
                "activation": {"input_global_scale": 0.5}, "encoding_identity_sha256": "e" * 64,
                "render_origin": "encoded", "render_comparison": "independent_render_vs_wire",
                "catalog_source_adoption": {"schema": "adoption"}, "anchor": {"dloss": 1e-5},
                "wire": str(wire), "wire_stat": _stat(wire),
                "render": str(render), "render_stat": _stat(render)}
        cells.append(cell)
        receipt = {key: cell[key] for key in ("source_weight", "activation", "encoding_identity_sha256",
                                              "render_origin", "render_comparison",
                                              "catalog_source_adoption")}
        receipt.update(render_file_sha256="f" * 64, rendered_weight={"content_sha256": "c" * 64})
        result = {"qname": qname, "format": FMT, "cell_sha256": rebind.cell_sha256(cell),
                  "verified_cell": receipt, "verified_cell_sha256": rebind.cell_sha256(receipt)}
        _write(root / "qualified" / (_sha(qname.encode()) + ".json"), json.dumps(result).encode())
    catalog = _write(root / "catalog.json", json.dumps({"cells": cells}).encode())
    return SimpleNamespace(root=root, r12=r12, r13=r13, served=served, resources=resources,
                           catalog=catalog, qualified=root / "qualified", out=root / "overlay",
                           cells=len(cache.weights) + len(cells))


@pytest.fixture
def fx(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(assemble, "OLDPLAN", Path(fixture.r12[0]["path"]))
    monkeypatch.setattr(assemble, "OLDPREP", Path(fixture.r12[1]["path"]))
    monkeypatch.setattr(assemble, "EXTENDED_CELLS", fixture.cells)
    return fixture


def _run(fx, monkeypatch, *extra):
    argv = ["assemble_t4_overlay.py",
            "--served-activation-policy", fx.served["path"],
            "--served-activation-policy-sha256", fx.served["sha256"],
            "--stage-b-resource-policy", fx.resources["path"],
            "--stage-b-resource-policy-sha256", fx.resources["sha256"],
            "--catalog", fx.catalog["path"], "--catalog-sha256", fx.catalog["sha256"],
            "--qualified-dir", str(fx.qualified), "--out", str(fx.out), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    assemble.main()
    return {name: (fx.out / name).read_bytes() for name in PUBLISHED}


def _flags(pair):
    plan, prepared = pair
    return ["--original-plan", plan["path"], "--original-plan-sha256", plan["sha256"],
            "--original-prepared", prepared["path"], "--original-prepared-sha256", prepared["sha256"]]


def test_the_original_pair_is_named_by_flags(fx, monkeypatch):
    published = _run(fx, monkeypatch, *_flags(fx.r13))
    inputs = json.loads(published["catalog-pair-inputs.json"])
    assert inputs["original_plan"] == fx.r13[0]
    assert inputs["original_prepared"] == fx.r13[1]
    assert json.loads(published["plan.json"])["name"] == "r13"
    prepared = json.loads(published["prepare/prepared.json"])
    assert prepared["name"] == "r13"
    assert prepared["formats_by_qname"][QNAMES[0]] == ["NVFP4", FMT, "BF16"]


def test_no_flags_bind_the_default_pair_and_publish_the_same_bytes(fx, monkeypatch):
    unflagged = _run(fx, monkeypatch)
    inputs = json.loads(unflagged["catalog-pair-inputs.json"])
    assert inputs["original_plan"] == fx.r12[0]
    assert inputs["original_prepared"] == fx.r12[1]
    shutil.rmtree(fx.out)
    assert _run(fx, monkeypatch, *_flags(fx.r12)) == unflagged


@pytest.mark.parametrize("which", ["plan", "prepared"])
def test_a_path_without_its_sha_refuses_and_publishes_nothing(fx, monkeypatch, capsys, which):
    flags = _flags(fx.r13)
    index = flags.index(f"--original-{which}-sha256")
    del flags[index:index + 2]
    with pytest.raises(SystemExit):
        _run(fx, monkeypatch, *flags)
    assert "together" in capsys.readouterr().err
    assert not fx.out.exists()


@pytest.mark.parametrize("which", ["plan", "prepared"])
def test_a_sha_without_its_path_refuses_and_publishes_nothing(fx, monkeypatch, capsys, which):
    flags = _flags(fx.r13)
    index = flags.index(f"--original-{which}")
    del flags[index:index + 2]
    with pytest.raises(SystemExit):
        _run(fx, monkeypatch, *flags)
    assert "together" in capsys.readouterr().err
    assert not fx.out.exists()


@pytest.mark.parametrize("which", ["plan", "prepared"])
def test_a_sha_mismatch_refuses_and_publishes_nothing(fx, monkeypatch, capsys, which):
    flags = _flags(fx.r13)
    flags[flags.index(f"--original-{which}-sha256") + 1] = "0" * 64
    with pytest.raises(SystemExit):
        _run(fx, monkeypatch, *flags)
    assert "SHA-256" in capsys.readouterr().err
    assert not fx.out.exists()
