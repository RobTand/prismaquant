"""Stage B source reads are not the parent's byte tiling (PQ #900).

Use real safetensors headers and the existing producer/receipt fixtures. The
same quantum needs the holes in its chain layer and in its own layer filled;
neither a neighbouring phase nor a union of staged files covers one tensor.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant import joint_layer_quanta as jl
from test_quantum_executable_readset import (
    CALIB, N_PROBES, RENDER_PREREQ, STRIDED, _bound_inputs,
    _exec_argv, _exec_campaign, _exec_receipt,
)
from test_stagea_readset_source_coverage import _write_shard

PREFIX = "model.language_model.layers."


def _source_model(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    weight_map = {}
    for layer in range(4):
        # A small F32-shaped span and a large straddling span in every shard.
        names = [(f"{PREFIX}{layer}.norm.weight", 32),
                 (f"{PREFIX}{layer}.mlp.weight", 8192)]
        if layer == 3:
            # Own-layer source at the start of its neighbour's shard.
            names.insert(0, (f"{PREFIX}2.extra.weight", 16))
        shard = f"layer-{layer}.safetensors"
        for name in _write_shard(model / shard, names):
            weight_map[name] = shard
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    spans = jl.read_layer_source_spans(
        str(model), 4, checkpoint_layers_prefix=PREFIX)
    return model, spans


def _with_source_paths(parent, model):
    parent = copy.deepcopy(parent)
    for layer in range(4):
        # Only the first 200 bytes of each shard were declared by the parent.
        parent["entries"][layer + 1]["path"] = str(
            model / f"layer-{layer}.safetensors")
    return parent


def _phase_entries(manifest, name):
    phase = next(row for row in manifest["read_plan"]["phases"]
                 if row["name"] == name)
    return [manifest["entries"][i] for i in phase["entry_indices"]]


def _build(record, receipt, parent, **kwargs):
    return jl.build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=CALIB,
        render_prerequisite=RENDER_PREREQ, **kwargs)


def test_chain_and_own_source_phases_cover_every_reader_tensor(tmp_path):
    record, receipt, parent, _ = _bound_inputs(tmp_path)
    model, spans = _source_model(tmp_path)
    parent = _with_source_paths(parent, model)
    before = copy.deepcopy((record, receipt, parent))
    legacy = _build(record, receipt, parent)
    manifest = _build(record, receipt, parent, layer_source_spans=spans)
    for layer, name in ((3, "chain-003-source"), (2, "own-002-source")):
        missing = jl.uncovered_source_spans(
            _phase_entries(manifest, name), spans[layer])
        assert not missing, f"{name} leaves actual source tensors undeclared: {missing}"
    assert (record, receipt, parent) == before
    # Append only: preexisting phase entry indices still name the same bytes.
    assert manifest["entries"][:len(legacy["entries"])] == legacy["entries"]
    assert "source_completion" not in legacy["annotations"]
    note = manifest["annotations"]["source_completion"]
    assert note["layers_checked"] == 2
    assert note["source_spans"] == len(spans[2]) + len(spans[3])
    assert {row["layer"] for row in note["added"]} == {2, 3}
    assert len({(row["path"], row["offset"]) for row in manifest["entries"]}) \
        == manifest["entry_count"]
    assert _build(record, receipt, parent, layer_source_spans=spans) == manifest


@pytest.mark.parametrize("bad_spans", [{}, {2: []}, {2: [], 3: []}])
def test_incomplete_reader_spans_refuse(tmp_path, bad_spans):
    record, receipt, parent, _ = _bound_inputs(tmp_path)
    with pytest.raises(ValueError, match="no source spans"):
        _build(record, receipt, parent, layer_source_spans=bad_spans)


def test_binder_keeps_tiling_and_rejects_a_rehashed_missing_tensor(tmp_path):
    record, receipt, parent, kwargs = _bound_inputs(tmp_path)
    model, spans = _source_model(tmp_path)
    parent = _with_source_paths(parent, model)
    manifest = _build(record, receipt, parent, layer_source_spans=spans)
    before = copy.deepcopy(record)
    kwargs.update(manifest=manifest, layer_source_spans=spans,
                  manifest_sha256=hashlib.sha256(
                      jl.seal_manifest_bytes(manifest)).hexdigest())
    fresh = jl.bind_quantum_executable(record, receipt, parent, **kwargs)
    assert record == before
    for key in ("campaign", "read_set", "chunks", "windows", "output_space", "adjoint"):
        assert fresh[key] == record[key]
    assert fresh["identity_sha256"] != record["identity_sha256"]
    body = {key: value for key, value in fresh.items() if key != "identity_sha256"}
    assert fresh["identity_sha256"] == jl.canonical_sha256(body)
    # Remove a completion by replacing it with the old, valid-but-incomplete
    # manifest and rehash it. Rebuild against the reader still refuses.
    kwargs["manifest"] = _build(record, receipt, parent)
    kwargs["manifest_sha256"] = hashlib.sha256(
        jl.seal_manifest_bytes(kwargs["manifest"])).hexdigest()
    with pytest.raises(ValueError, match="do not originate"):
        jl.bind_quantum_executable(record, receipt, parent, **kwargs)


def test_tensor_straddling_two_entries_gets_one_covering_entry(tmp_path):
    record, receipt, parent, _ = _bound_inputs(tmp_path)
    path = parent["entries"][4]["path"]
    # The phase covers [0,200) and [200,300). The reader cannot stitch the
    # staged files together, so their union must not satisfy this tensor.
    parent["entries"].append(
        {"path": path, "offset": 200, "bytes": 100, "sha256": None})
    parent["entry_count"] += 1
    parent["total_bytes"] += 100
    parent["annotations"]["phases"][-1]["bytes"] += 100
    parent["annotations"]["phases"][-1]["cumulative_bytes"] += 100
    spans = {2: [(parent["entries"][3]["path"], 10, 190)],
             3: [(path, 150, 250)]}
    manifest = _build(record, receipt, parent, layer_source_spans=spans)
    assert {"path": path, "offset": 150, "bytes": 100, "sha256": None} in \
        _phase_entries(manifest, "chain-003-source")


def test_neighbour_source_phase_does_not_cover_own_small_tensor(tmp_path):
    record, receipt, parent, _ = _bound_inputs(tmp_path)
    path = parent["entries"][4]["path"]
    spans = {2: [(parent["entries"][3]["path"], 10, 190), (path, 30, 46)],
             3: [(path, 50, 190)]}
    legacy = _build(record, receipt, parent)
    assert not jl.uncovered_source_spans(legacy["entries"], spans[2])
    assert jl.uncovered_source_spans(
        _phase_entries(legacy, "own-002-source"), spans[2])
    manifest = _build(record, receipt, parent, layer_source_spans=spans)
    assert not jl.uncovered_source_spans(
        _phase_entries(manifest, "own-002-source"), spans[2])


def test_completion_refuses_a_conflicting_existing_start(tmp_path):
    record, receipt, parent, _ = _bound_inputs(tmp_path)
    spans = {2: [(parent["entries"][3]["path"], 0, 250)],
             3: [(parent["entries"][4]["path"], 10, 190)]}
    with pytest.raises(ValueError, match="refusing to repeat"):
        _build(record, receipt, parent, layer_source_spans=spans)


def _cli_fixture(tmp_path):
    campaign = _exec_campaign(tmp_path)
    model, spans = _source_model(tmp_path)
    plan = json.loads(campaign["plan_path"].read_text())
    plan["model"] = str(model)
    campaign["plan_path"].write_text(json.dumps(plan))
    campaign["plan_sha"] = hashlib.sha256(campaign["plan_path"].read_bytes()).hexdigest()
    parent = _with_source_paths(json.loads(campaign["parent_path"].read_text()), model)
    campaign["parent_path"].write_text(json.dumps(parent))
    campaign["parent_sha"] = hashlib.sha256(campaign["parent_path"].read_bytes()).hexdigest()
    receipt, space = _exec_receipt(tmp_path, campaign)
    receipt_path = space / "adjoint-capture.json"
    receipt_path.write_text(json.dumps(receipt))
    return campaign, receipt_path, spans


def test_cli_completes_in_new_generation_preserving_prior_tiling(tmp_path):
    import regenerate_joint_quanta as regen
    campaign, receipt_path, spans = _cli_fixture(tmp_path)
    original = tmp_path / "original-records"
    base = _exec_argv(tmp_path, campaign)
    assert regen.main(base + ["--records-out", str(original)]) == 0
    old = {p.name: p.read_bytes() for p in original.glob("*.json")}
    new = tmp_path / "metadata-generation"
    args = base + ["--metadata-root", str(new),
                   "--compare-existing", str(original),
                   "--adjoint-receipt", str(receipt_path),
                   "--executable-readsets", "--source-layers-prefix", PREFIX]
    assert regen.main(args + ["--check-only"]) == 0
    assert not new.exists()
    assert regen.main(args) == 0
    assert old == {p.name: p.read_bytes() for p in original.glob("*.json")}
    for path in sorted((new / "records").glob("layer-*.json")):
        record = json.loads(path.read_text())
        prior = json.loads(old[path.name])
        for key in ("campaign", "chunks", "windows", "output_space"):
            assert record[key] == prior[key]
        for key in ("source_phase", "entry_count", "total_bytes"):
            assert record["read_set"][key] == prior["read_set"][key]
        manifest_path = Path(record["executable_readset"]["manifest_path"])
        wire = manifest_path.read_bytes()
        assert hashlib.sha256(wire).hexdigest() == record["executable_readset"]["manifest_sha256"]
        manifest = json.loads(gzip.decompress(wire))
        for layer in record["adjoint"]["chain_layers"] + [record["layer"]]:
            name = (jl.executable_own_source_phase_name(layer)
                    if layer == record["layer"] else jl.executable_source_phase_name(layer))
            assert not jl.uncovered_source_spans(_phase_entries(manifest, name), spans[layer])
        # Validate the actual completed v2 document through PB, with portable
        # fixture paths re-rooted under its declared mount.
        import prismabuild.core as core
        import prismabuild.storage_tiers as tiers
        checked = copy.deepcopy(manifest)
        for entry in checked["entries"]:
            entry["path"] = "/mnt/shared/fixture" + entry["path"]
        normalized = core.validate_data_manifest(checked)
        assert len(tiers.manifest_phase_ranges(normalized)) == len(manifest["read_plan"]["phases"])
    # First-writer replay is idempotent; it never edits either generation.
    assert regen.main(args) == 0


def test_cli_wrong_reader_prefix_refuses_without_publication(tmp_path, capsys):
    import regenerate_joint_quanta as regen
    campaign, receipt_path, _ = _cli_fixture(tmp_path)
    new = tmp_path / "bad-generation"
    assert regen.main(_exec_argv(tmp_path, campaign) + [
        "--metadata-root", str(new), "--adjoint-receipt", str(receipt_path),
        "--executable-readsets", "--source-layers-prefix", "wrong.layers."]) == 3
    assert "wrong prefix or wrong layer count" in capsys.readouterr().err
    assert not new.exists()
