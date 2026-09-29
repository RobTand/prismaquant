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

from prismaquant.digests import DIRECT_ASCII_LAX, bytes_sha256hex  # noqa: E402
from tools import assemble_t4_overlay as assemble  # noqa: E402
from tools import build_t4_logical_request as builder  # noqa: E402
from tools import rebind_t4_qualified_results as rebind  # noqa: E402

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
    return {'inode': s.st_ino, 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


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
                "render": str(render), "render_stat": _stat(render),
                "record": {"blob_sha256": hashlib.sha256(b"wire" * (index + 1)).hexdigest()}}
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


# -- several added formats (PQ #1432) -------------------------------------------

FMT2 = "TESSERA_E4M3_K1_R5"


def _result_bytes(cell, bound_cell):
    """A qualification result for ``cell`` that binds ``bound_cell``'s digest."""
    receipt = {key: cell[key] for key in ("source_weight", "activation", "encoding_identity_sha256",
                                          "render_origin", "render_comparison", "catalog_source_adoption")}
    receipt.update(render_file_sha256="f" * 64, rendered_weight={"content_sha256": "c" * 64})
    return json.dumps({"qname": cell["qname"], "format": cell["format"],
                       "cell_sha256": rebind.cell_sha256(bound_cell), "verified_cell": receipt,
                       "verified_cell_sha256": rebind.cell_sha256(receipt)}).encode()


def _multi_format(fx, *, declare_carried=True):
    """A catalog carrying the fixture's cells byte-identical from an earlier
    catalog, whose results were rebound from older anchors as R13's were, plus
    a second format qualified directly in its own directory."""
    root = fx.root
    cells = json.loads(Path(fx.catalog["path"]).read_bytes())["cells"]
    carried = _write(root / "carried-catalog.json", json.dumps({"cells": cells}).encode())
    rows = []
    for cell in cells:
        prior = {**cell, "anchor": {"dloss": 2e-5}}
        raw = _result_bytes(cell, prior)
        _write(fx.qualified / (_sha(cell["qname"].encode()) + ".json"), raw)
        rows.append(rebind.rebind_cell(cell, prior, raw))
    rebinding = _write(root / "rebinding.json", json.dumps(
        {"schema": rebind.SCHEMA, "catalog": carried, "qualified_dir": str(fx.qualified), "rows": rows}).encode())
    second = root / "qualified-e4m3"
    added = []
    for index, qname in enumerate(QNAMES):
        wire = root / "wire2" / f"{index}.tsr"
        render = root / "render2" / f"{index}.pt"
        _write(wire, b"w2" * (index + 1))
        _write(render, b"r2" * (index + 1))
        cell = {**cells[index], "format": FMT2, "wire": str(wire), "wire_stat": _stat(wire),
                "render": str(render), "render_stat": _stat(render),
                "record": {"blob_sha256": hashlib.sha256(b"w2" * (index + 1)).hexdigest()}}
        added.append(cell)
        _write(second / (_sha(qname.encode()) + ".json"), _result_bytes(cell, cell))
    catalog = {"cells": cells + added, **({"carried_from": [carried]} if declare_carried else {})}
    fx.catalog = _write(root / "catalog-multi.json", json.dumps(catalog).encode())
    fx.cells += len(added)
    return ["--rebinding", rebinding["path"], "--rebinding-sha256", rebinding["sha256"],
            "--format-qualified-dir", f"{FMT2}={second}"]


def test_a_multi_format_catalog_assembles_with_carried_and_new_cells(fx, monkeypatch):
    """Carried cells pass through the previous catalog's rebinding; the new
    format's cells are qualified directly from their own directory; every
    unit's added formats go, sorted, before its terminal BF16."""
    extra = _multi_format(fx)
    published = _run(fx, monkeypatch, *_flags(fx.r13), *extra)
    prepared = json.loads(published["prepare/prepared.json"])
    assert all(prepared["formats_by_qname"][q] == ["NVFP4", FMT, FMT2, "BF16"] for q in QNAMES)
    assert prepared["measured_cells"] == fx.cells == 5


def test_a_rebinding_of_an_undeclared_catalog_refuses(fx, monkeypatch):
    extra = _multi_format(fx, declare_carried=False)
    with pytest.raises(AssertionError, match="neither this catalog"):
        _run(fx, monkeypatch, *_flags(fx.r13), *extra)
    assert not fx.out.exists()


def test_builder_pair_outputs_feed_rebinding_and_assembly_without_renames(fx, monkeypatch):
    """Simulate qualification metadata only; never read source bytes or run a decoder."""
    original = json.loads(Path(fx.catalog['path']).read_bytes())['cells']
    cells = [{**c, 'format': fmt,
              'wire': f'/mnt/shared/synthetic/{index}/{fmt}/wire',
              'render': f'/mnt/shared/synthetic/{index}/{fmt}/render'}
             for index, c in enumerate(original) for fmt in (FMT, FMT2)]
    catalog = {'schema': 'prismaquant.t4_adopted_catalog.v2', 'cells': cells,
               'formats': [FMT, FMT2], 'cell_sources': [0] * len(cells),
               'sources': [{'cost': {'path': 'synthetic', 'sha256': 'b' * 64}}]}
    previous = _write(fx.root / 'builder-catalog.json', DIRECT_ASCII_LAX.encoded(catalog))
    request_path = fx.root / 'request.json'
    monkeypatch.setattr(sys, 'argv', ['builder', '--root', str(fx.root / 'fresh-campaign'),
                        '--catalog', previous['path'], '--catalog-sha256', previous['sha256'],
                        '--python', '/pinned/python', '--qualifier-checkout', '/independent-worker',
                        '--out', str(request_path)])
    builder.main()
    tasks = json.loads(request_path.read_bytes())['roster']['tasks']
    assert all(len(t['payload']['reads']) == 2 for t in tasks)
    before = {}
    for task in tasks:
        cell = task['payload']['cell']
        path = Path(task['payload']['output'])
        raw = _result_bytes(cell, cell)
        _write(path, raw)
        before[path] = raw
    fx.qualified = fx.root / 'fresh-campaign' / 'qualified'
    fx.catalog = previous
    fx.out = fx.root / 'direct-pair-overlay'
    monkeypatch.setattr(assemble, 'fence_cell_artifacts', lambda c: None)
    direct = _run(fx, monkeypatch, *_flags(fx.r13))
    assert json.loads(direct['prepare/prepared.json'])['measured_cells'] == len(cells) + 1
    fx.out = fx.root / 'rebound-pair-overlay'
    catalog['cells'] = [{**c, 'anchor': {'dloss': 2e-5}} for c in cells]
    fx.catalog = _write(fx.root / 'reanchored-catalog.json', DIRECT_ASCII_LAX.encoded(catalog))
    rebound_path = fx.root / 'pair-rebinding.json'
    monkeypatch.setattr(sys, 'argv', ['rebind', '--catalog', fx.catalog['path'],
                        '--catalog-sha256', fx.catalog['sha256'],
                        '--previous-catalog', previous['path'],
                        '--previous-catalog-sha256', previous['sha256'],
                        '--qualified-dir', str(fx.qualified), '--out', str(rebound_path)])
    rebind.main()
    assert len(json.loads(rebound_path.read_bytes())['rows']) == len(cells)
    monkeypatch.setattr(assemble, 'fence_cell_artifacts', lambda c: None)
    published_weights = {}
    original_dumps = pickle.dumps

    def capture_cache(value, *args, **kwargs):
        published_weights.update(value.weights)
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(assemble.pickle, 'dumps', capture_cache)
    published = _run(fx, monkeypatch, *_flags(fx.r13), '--rebinding', str(rebound_path),
                     '--rebinding-sha256', bytes_sha256hex(rebound_path.read_bytes()))
    assert json.loads(published['prepare/prepared.json'])['measured_cells'] == len(cells) + 1
    assert set(published_weights) == {('model.embed', 'BF16')} | {(c['qname'], c['format']) for c in cells}
    assert all(published_weights[c['qname'], c['format']] == c['render'] for c in cells)
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.parametrize('consumer', ['assemble', 'rebind'])
def test_pair_and_qname_files_refuse_ambiguous_auto_consumer_intake(fx, monkeypatch, consumer):
    cells = json.loads(Path(fx.catalog['path']).read_bytes())['cells']
    for cell in cells:
        pair_id = DIRECT_ASCII_LAX.sha256([cell['qname'], cell['format']])
        pair_path = fx.qualified / (pair_id + '.json')
        _write(pair_path, _result_bytes(cell, cell))
        assert pair_path.read_bytes() == rebind.result_path(fx.qualified, cell['qname']).read_bytes()
    if consumer == 'assemble':
        with pytest.raises(ValueError, match='ambiguous'):
            _run(fx, monkeypatch)
        assert not fx.out.exists()
    else:
        out = fx.root / 'ambiguous-rebinding.json'
        monkeypatch.setattr(sys, 'argv', ['rebind', '--catalog', fx.catalog['path'],
                            '--catalog-sha256', fx.catalog['sha256'],
                            '--previous-catalog', fx.catalog['path'],
                            '--previous-catalog-sha256', fx.catalog['sha256'],
                            '--qualified-dir', str(fx.qualified), '--out', str(out)])
        with pytest.raises(ValueError, match='ambiguous'):
            rebind.main()
        assert not out.exists()


@pytest.mark.parametrize('key', ['pair', 'qname'])
def test_explicit_consumer_key_resolves_ambiguous_directory(fx, monkeypatch, key):
    cells = json.loads(Path(fx.catalog['path']).read_bytes())['cells']
    for cell in cells:
        pair_id = DIRECT_ASCII_LAX.sha256([cell['qname'], cell['format']])
        _write(fx.qualified / (pair_id + '.json'), _result_bytes(cell, cell))
    published = _run(fx, monkeypatch, '--qualified-key', key)
    prepared = json.loads(published['prepare/prepared.json'])
    assert prepared['measured_cells'] == fx.cells
    out = fx.root / 'explicit-rebinding.json'
    monkeypatch.setattr(sys, 'argv', ['rebind', '--catalog', fx.catalog['path'],
                        '--catalog-sha256', fx.catalog['sha256'],
                        '--previous-catalog', fx.catalog['path'],
                        '--previous-catalog-sha256', fx.catalog['sha256'],
                        '--qualified-dir', str(fx.qualified), '--qualified-key', key, '--out', str(out)])
    rebind.main()
    assert len(json.loads(out.read_bytes())['rows']) == len(cells)


def test_a_second_format_without_its_directory_refuses(fx, monkeypatch):
    extra = _multi_format(fx)
    with pytest.raises(AssertionError):
        _run(fx, monkeypatch, *_flags(fx.r13), *extra[:4])
    assert not fx.out.exists()
