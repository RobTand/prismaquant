"""Executable source phases must stage source weights only (PQ #909).

A mixed parent layer phase -- source weights plus rendered-cache files,
exactly the production shape from PQ #909 (real layer 9: 4,350 entries /
84.72 GB, only 3 entries / 11.779 GB under the source model) -- must not
reach the chain/own source phases. Rendered weights stay under the
accepted produced-output lifecycle; the executable readset stages what
the checkpoint reader actually reads.

RED basis: the first two tests, run pre-fix through PrismaBuild as action
``96b358550ab5`` (rc 1, 2 failed), staged the wrong corpus through the
existing builder API -- rendered bytes plus a string-prefix sibling --
not a missing symbol. The fix threads the sealed plan's source model
root through build/bind/emit/regenerator as ``source_model_root``.
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
)

SOURCE_ROOT = "/fixture/model"
RENDER_DIR = "/fixture/workspace/rows/row-0086/cache"
EVIL_SIBLING = "/fixture/model-evil/shard.pt"


def _insert_parent_entry(parent, phase_name, entry):
    """Insert one entry at the end of a parent phase, keeping it sealed.

    The row's byte range grows by the entry and every later cumulative
    mark shifts, so ``phase_ranges`` and the tiling-agreement check still
    hold: the parent tiling never changes shape, it only carries the
    mixed corpus.
    """
    rows = {row["name"]: row for row in jl.phase_ranges(parent)}
    position = rows[phase_name]["entry_end"]
    names = [row["name"] for row in parent["annotations"]["phases"]]
    counts = {n: rows[n]["entry_end"] - rows[n]["entry_begin"]
              for n in names}
    parent["entries"].insert(position, dict(entry))
    cursor = 0
    running = 0
    for name in names:
        group = parent["entries"][
            cursor:cursor + counts[name] + (1 if name == phase_name else 0)]
        size = sum(item["bytes"] for item in group)
        row = next(r for r in parent["annotations"]["phases"]
                   if r["name"] == name)
        row["bytes"] = size
        running += size
        row["cumulative_bytes"] = running
        cursor += len(group)
    assert cursor == len(parent["entries"])
    parent["entry_count"] = len(parent["entries"])
    parent["total_bytes"] = running
    return parent


def _mixed_parent(parent):
    _insert_parent_entry(parent, "layer-3", {
        "path": f"{RENDER_DIR}/model_language_model_layers_9_mlp_experts_0"
                "_down_proj__TESSERA_BF16_K1_R1024.pt",
        "offset": 0, "bytes": 64, "sha256": None})
    _insert_parent_entry(parent, "layer-2", {
        "path": f"{RENDER_DIR}/model_language_model_layers_8_mlp_gate_proj"
                "__TESSERA_BF16_K1_R1024.pt",
        "offset": 0, "bytes": 48, "sha256": None})
    _insert_parent_entry(parent, "layer-2", {
        "path": EVIL_SIBLING, "offset": 0, "bytes": 32, "sha256": None})
    # The recomputed table must still describe the same tiling.
    jl.phase_ranges(parent)
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


def _non_source(entries):
    return [item for item in entries
            if not (item["path"] == SOURCE_ROOT
                    or item["path"].startswith(SOURCE_ROOT + "/"))]


def test_chain_and_own_source_phases_stage_no_rendered_cache(tmp_path):
    record, receipt, parent, _ = _bound_inputs(tmp_path)
    manifest = _build(record, receipt, _mixed_parent(parent),
                      source_model_root=SOURCE_ROOT)
    for name in ("chain-003-source", "own-002-source"):
        staged = _phase_entries(manifest, name)
        wrong = _non_source(staged)
        wrong_bytes = sum(item["bytes"] for item in wrong)
        assert not wrong, (
            f"{name} stages {wrong_bytes} non-source bytes "
            f"({len(wrong)} entries, first "
            f"{wrong[0]['path']}:{wrong[0]['offset']}); rendered cache "
            "must stay under the produced-output lifecycle")
    # Both source entries survive; the corpus is restricted, not emptied.
    staged = (_phase_entries(manifest, "chain-003-source")
              + _phase_entries(manifest, "own-002-source"))
    assert {(item["path"], item["offset"]) for item in staged} == {
        ("/fixture/model/shard-l3.pt", 0),
        ("/fixture/model/shard-l2.pt", 0)}


def test_string_prefix_sibling_is_not_a_source_path(tmp_path):
    record, receipt, parent, _ = _bound_inputs(tmp_path)
    manifest = _build(record, receipt, _mixed_parent(parent),
                      source_model_root=SOURCE_ROOT)
    staged = (_phase_entries(manifest, "chain-003-source")
              + _phase_entries(manifest, "own-002-source"))
    evil = [item for item in staged if item["path"] == EVIL_SIBLING]
    assert not evil, (
        f"a bare string prefix admits the sibling {EVIL_SIBLING}; "
        "source selection must match a path-component boundary")


def test_source_selection_matches_component_boundary():
    assert jl._is_source_model_path(SOURCE_ROOT, SOURCE_ROOT)
    assert jl._is_source_model_path(f"{SOURCE_ROOT}/shard.pt", SOURCE_ROOT)
    assert jl._is_source_model_path(
        f"{SOURCE_ROOT}/nested/dir/shard.pt", SOURCE_ROOT)
    assert jl._is_source_model_path(
        f"{SOURCE_ROOT}/shard.pt", SOURCE_ROOT + "/")
    assert not jl._is_source_model_path(EVIL_SIBLING, SOURCE_ROOT)
    assert not jl._is_source_model_path(f"{RENDER_DIR}/a.pt", SOURCE_ROOT)
    assert not jl._is_source_model_path("", SOURCE_ROOT)
    assert not jl._is_source_model_path(None, SOURCE_ROOT)
    with pytest.raises(ValueError, match="absolute"):
        jl._source_extent_entries(
            {"entries": [{"path": "a", "offset": 0, "bytes": 1}],
             "annotations": {"phases": []}},
            layers=[], source_model_root="relative/root")


def test_render_only_layer_phase_refuses(tmp_path):
    record, receipt, parent, _ = _bound_inputs(tmp_path)
    parent = copy.deepcopy(parent)
    parent["entries"][3] = {
        "path": f"{RENDER_DIR}/only-render.pt",
        "offset": 0, "bytes": 200, "sha256": None}
    with pytest.raises(ValueError, match="no source extent"):
        _build(record, receipt, parent, source_model_root=SOURCE_ROOT)


def test_unfiltered_build_reproduces_historical_bytes(tmp_path):
    record, receipt, parent, _ = _bound_inputs(tmp_path)
    legacy = _build(record, receipt, parent)
    filtered = _build(record, receipt, parent,
                      source_model_root=SOURCE_ROOT)
    assert filtered == legacy


def _bind_kwargs(record, manifest, root, source_model_root):
    return dict(
        manifest=manifest,
        manifest_path=f"{root}/layer-quanta/adjoint/bound-readsets/"
                      "layer-002.executable.json.gz",
        manifest_sha256=hashlib.sha256(
            jl.seal_manifest_bytes(manifest)).hexdigest(),
        output_root=root,
        strided_boundaries=STRIDED, n_probes=N_PROBES,
        calib=dict(CALIB), render_prerequisite=dict(RENDER_PREREQ),
        source_model_root=source_model_root)


def test_binder_rederives_filtered_manifest_with_root(tmp_path):
    record, receipt, parent, _ = _bound_inputs(tmp_path)
    parent = _mixed_parent(parent)
    manifest = _build(record, receipt, parent,
                      source_model_root=SOURCE_ROOT)
    root = "/mnt/shared/run"
    before = copy.deepcopy(record)
    fresh = jl.bind_quantum_executable(
        record, receipt, parent,
        **_bind_kwargs(record, manifest, root, SOURCE_ROOT))
    assert record == before
    assert fresh["identity_sha256"] != record["identity_sha256"]
    # The binder rederives: a root mismatch on either side refuses.
    with pytest.raises(ValueError, match="do not originate"):
        jl.bind_quantum_executable(
            record, receipt, parent,
            **_bind_kwargs(record, manifest, root, None))
    legacy = _build(record, receipt, parent)
    with pytest.raises(ValueError, match="do not originate"):
        jl.bind_quantum_executable(
            record, receipt, parent,
            **_bind_kwargs(record, legacy, root, SOURCE_ROOT))


def test_emit_forwards_source_model_root(tmp_path):
    record, receipt, parent, _ = _bound_inputs(tmp_path)
    parent = _mixed_parent(parent)
    root = str(tmp_path / "run")
    rows = jl.emit_quantum_executable_readsets(
        receipt, [record], parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ),
        output_root=root, source_model_root=SOURCE_ROOT)
    assert len(rows) == 1
    staged = (_phase_entries(rows[0]["manifest"], "chain-003-source")
              + _phase_entries(rows[0]["manifest"], "own-002-source"))
    assert not _non_source(staged)
    assert rows[0]["record"]["identity_sha256"] != record["identity_sha256"]
    legacy = jl.emit_quantum_executable_readsets(
        receipt, [record], parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ), output_root=root)
    staged = (_phase_entries(legacy[0]["manifest"], "chain-003-source")
              + _phase_entries(legacy[0]["manifest"], "own-002-source"))
    assert _non_source(staged), \
        "without a root the historical (unfiltered) bytes must reproduce"


def test_cli_filters_rendered_cache_using_sealed_plan_root(tmp_path):
    from test_stageb_readset_source_coverage import _cli_fixture
    from test_quantum_executable_readset import _exec_argv
    import regenerate_joint_quanta as regen
    campaign, receipt_path, _ = _cli_fixture(tmp_path)
    parent = json.loads(campaign["parent_path"].read_text())
    model_root = json.loads(campaign["plan_path"].read_text())["model"]
    _mixed_parent(parent)
    # Re-point the mixed fixture at the sealed source model: rendered
    # entries live outside it by construction.
    for entry in parent["entries"]:
        if entry["path"].startswith("/fixture/model/"):
            entry["path"] = entry["path"].replace(
                "/fixture/model/", model_root + "/", 1)
    campaign["parent_path"].write_text(json.dumps(parent))
    campaign["parent_sha"] = hashlib.sha256(
        campaign["parent_path"].read_bytes()).hexdigest()
    original = tmp_path / "original-records"
    base = _exec_argv(tmp_path, campaign)
    assert regen.main(base + ["--records-out", str(original)]) == 0
    new = tmp_path / "metadata-generation"
    args = base + ["--metadata-root", str(new),
                   "--compare-existing", str(original),
                   "--adjoint-receipt", str(receipt_path),
                   "--executable-readsets"]
    assert regen.main(args + ["--check-only"]) == 0
    assert not new.exists()
    assert regen.main(args) == 0
    assert regen.main(args) == 0
    for path in sorted((new / "records").glob("layer-*.json")):
        record = json.loads(path.read_text())
        manifest_path = Path(record["executable_readset"]["manifest_path"])
        wire = manifest_path.read_bytes()
        assert hashlib.sha256(wire).hexdigest() == \
            record["executable_readset"]["manifest_sha256"]
        manifest = json.loads(gzip.decompress(wire))
        for phase in manifest["read_plan"]["phases"]:
            if not phase["name"].endswith("-source"):
                continue
            for index in phase["entry_indices"]:
                entry = manifest["entries"][index]
                assert entry["path"] == model_root or entry["path"].startswith(
                    model_root + "/"), entry["path"]
