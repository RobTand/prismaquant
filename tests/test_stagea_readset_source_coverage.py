"""The stage-A read manifest declares what the reader reads (PQ #898).

The 512-sample GLM-5.3-Flash campaign's parent manifest dropped the tails of
layers 9, 19, 29 and 39. Each tail sits at the head of a shard that an earlier
phase already held a header entry for at ``(path, 0)``; PrismaBuild refuses a
repeated ``(path, offset)``, so the tail's own extent was dropped rather than
clipped. ``build_adjoint_manifest`` copied the hole forward, nothing compared
the readset to the reader, and the run refused at the layer-9 prefetch.

The fixture is that shape in miniature: layer 1's tail opens shard B, layer 0
lives behind it, and the parent gives layer 0 a header entry at ``(B, 0)``.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

from prismaquant.joint_layer_quanta import (
    SOURCE_COMPLETION_SCHEMA,
    build_adjoint_manifest,
    complete_source_extent,
    uncovered_source_spans,
)
from prismaquant.layer_streaming import streaming_source_plan

PUBLISHED_PB_SRC = Path("/mnt/shared/prismabuild-fleet/repo/src")
#: Checkpoint names. The fixture has no config, so the loader's profile is the
#: text-only default, which names these layers ``model.layers.N`` live.
PREFIX = "model.language_model.layers."
LIVE_PREFIX = "model.layers."
HEADER_ENTRY_BYTES = 1024


def _write_shard(path: Path, tensors: list[tuple[str, int]]) -> dict[str, tuple[int, int]]:
    """A safetensors file by hand. Returns each tensor's absolute file span."""
    header, cursor = {}, 0
    for name, size in tensors:
        header[name] = {"dtype": "U8", "shape": [size],
                        "data_offsets": [cursor, cursor + size]}
        cursor += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(cursor))
    base = 8 + len(raw)
    return {name: (base + row["data_offsets"][0], base + row["data_offsets"][1])
            for name, row in header.items()}


@pytest.fixture
def campaign(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    shard_a = _write_shard(model / "a.safetensors", [
        (f"{PREFIX}1.attn.weight", 30000), (f"{PREFIX}1.mlp.experts.0.weight", 20000)])
    shard_b = _write_shard(model / "b.safetensors", [
        (f"{PREFIX}1.mlp.experts.1.weight", 15000),
        (f"{PREFIX}1.mlp.experts.2.weight", 15000),
        (f"{PREFIX}0.attn.weight", 40000),
        ("model.visual.blocks.0.weight", 5000),
        (f"{PREFIX}2.mtp.weight", 7000)])
    weight_map = {name: "a.safetensors" for name in shard_a}
    weight_map.update({name: "b.safetensors" for name in shard_b})
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    a, b = str(model / "a.safetensors"), str(model / "b.safetensors")
    layer0 = shard_b[f"{PREFIX}0.attn.weight"]
    assert shard_b[f"{PREFIX}1.mlp.experts.1.weight"][0] < HEADER_ENTRY_BYTES, (
        "the fixture's tail has to begin inside the header entry")
    entries = [
        {"path": str(tmp_path / "calib.pt"), "offset": 0, "bytes": 10, "sha256": None},
        # layer 0: the shard's header, then its own tensor.
        {"path": b, "offset": 0, "bytes": HEADER_ENTRY_BYTES, "sha256": None},
        {"path": b, "offset": layer0[0], "bytes": layer0[1] - layer0[0], "sha256": None},
        # layer 1: shard A whole. Its tail in shard B starts at (b, 0) too,
        # and is the entry the parent's producer dropped.
        {"path": a, "offset": 0, "bytes": (model / "a.safetensors").stat().st_size,
         "sha256": None},
    ]
    phases, cumulative, cursor = [], 0, 0
    for name, count in (("head", 1), ("layer-0", 2), ("layer-1", 1)):
        size = sum(entry["bytes"] for entry in entries[cursor:cursor + count])
        cursor += count
        cumulative += size
        phases.append({"name": name, "bytes": size, "cumulative_bytes": cumulative})
    parent = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "produced_by": {"tool": "fixture"}, "mount_prefix": str(tmp_path),
        "entries": entries, "entry_count": len(entries), "total_bytes": cumulative,
        "annotations": {"campaign_scope": {"campaign": "readset-fixture"},
                        "phases": phases},
    }
    return {"model": str(model), "parent": parent, "a": a, "b": b,
            "tail": (shard_b[f"{PREFIX}1.mlp.experts.1.weight"][0],
                     shard_b[f"{PREFIX}1.mlp.experts.2.weight"][1])}


def _build(campaign, spans=None):
    return build_adjoint_manifest(
        {"model": campaign["model"]}, campaign["parent"],
        plan_path="/fixture/plan.json", plan_sha256="a" * 64,
        prepared_path="/fixture/prepared.json", prepared_sha256="b" * 64,
        parent_manifest_sha256="c" * 64, output_root="/fixture/out",
        **({} if spans is None else {"layer_source_spans": spans}))


def _spans(campaign):
    return streaming_source_plan(campaign["model"], layers_prefix=LIVE_PREFIX,
                                 layers=range(2))["layer_spans"]


def _phase(manifest, name):
    return next(row for row in manifest["read_plan"]["phases"] if row["name"] == name)


def test_the_reader_side_is_every_tensor_under_the_layers_prefix(campaign):
    spans = _spans(campaign)
    assert sorted(spans) == [0, 1]
    assert [Path(path).name for path, _start, _end in spans[1]] == [
        "a.safetensors", "a.safetensors", "b.safetensors", "b.safetensors"]
    assert [(Path(path).name, end - start) for path, start, end in spans[0]] == [
        ("b.safetensors", 40000)]


def test_a_layer_past_the_count_and_a_tower_outside_the_prefix_are_not_read(campaign):
    spans = _spans(campaign)
    read = {(path, start) for rows in spans.values() for path, start, _end in rows}
    assert len(read) == 5


def test_a_prefix_that_names_nothing_refuses(campaign):
    # The checkpoint's prefix is not the loader's for a text-only profile.
    with pytest.raises(ValueError, match="wrong prefix or wrong layer count"):
        streaming_source_plan(campaign["model"], layers_prefix=PREFIX,
                              layers=range(2))


def test_a_tensor_the_index_names_and_the_shard_lacks_refuses(campaign):
    index = Path(campaign["model"]) / "model.safetensors.index.json"
    body = json.loads(index.read_text())
    body["weight_map"][f"{PREFIX}0.ghost.weight"] = "b.safetensors"
    index.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="does not hold tensor"):
        _spans(campaign)


def test_the_parent_alone_carries_the_hole_forward(campaign):
    manifest = _build(campaign)
    layer1 = [manifest["entries"][index]
              for index in _phase(manifest, "forward-001")["entry_indices"]]
    assert len(uncovered_source_spans(layer1, _spans(campaign)[1])) == 2
    assert "source_completion" not in manifest["annotations"]


def test_the_dropped_tail_is_declared_once_the_reader_is_consulted(campaign):
    manifest = _build(campaign, _spans(campaign))
    start, end = campaign["tail"]
    added = manifest["annotations"]["source_completion"]
    assert added == {"schema": SOURCE_COMPLETION_SCHEMA, "layers_checked": 2,
                     "source_spans": 5,
                     "added": [{"layer": 1, "path": campaign["b"],
                                "offset": start, "bytes": end - start}]}
    forward, chain = _phase(manifest, "forward-001"), _phase(manifest, "chain-001")
    assert forward["entry_indices"] == chain["entry_indices"]
    layer1 = [manifest["entries"][index] for index in forward["entry_indices"]]
    assert uncovered_source_spans(layer1, _spans(campaign)[1]) == []
    assert forward["bytes"] == sum(entry["bytes"] for entry in layer1)
    assert manifest["entry_count"] == len(manifest["entries"]) == 5
    assert manifest["total_bytes"] == sum(e["bytes"] for e in manifest["entries"])
    assert manifest["read_plan"]["read_bytes"] == \
        manifest["read_plan"]["phases"][-1]["cumulative_bytes"]


def test_the_added_entry_does_not_repeat_the_header_entrys_offset(campaign):
    manifest = _build(campaign, _spans(campaign))
    keys = [(entry["path"], entry["offset"]) for entry in manifest["entries"]]
    assert len(set(keys)) == len(keys)
    assert (campaign["b"], 0) in keys and (campaign["b"], campaign["tail"][0]) in keys


def test_a_complete_parent_gains_nothing(campaign):
    start, end = campaign["tail"]
    parent = campaign["parent"]
    parent["entries"].append({"path": campaign["b"], "offset": start,
                              "bytes": end - start, "sha256": None})
    parent["entry_count"] += 1
    parent["total_bytes"] += end - start
    parent["annotations"]["phases"][-1]["bytes"] += end - start
    parent["annotations"]["phases"][-1]["cumulative_bytes"] += end - start
    manifest = _build(campaign, _spans(campaign))
    assert manifest["annotations"]["source_completion"]["added"] == []
    assert manifest["entries"] == _build(campaign)["entries"]


def test_without_the_reader_the_manifest_is_what_it_always_was(campaign):
    assert _build(campaign) == build_adjoint_manifest(
        {"model": campaign["model"]}, campaign["parent"],
        plan_path="/fixture/plan.json", plan_sha256="a" * 64,
        prepared_path="/fixture/prepared.json", prepared_sha256="b" * 64,
        parent_manifest_sha256="c" * 64, output_root="/fixture/out",
        layer_source_spans=None)


def test_two_entries_that_meet_inside_a_tensor_do_not_cover_it():
    entries = [{"path": "/m/s", "offset": 0, "bytes": 100},
               {"path": "/m/s", "offset": 100, "bytes": 100}]
    assert uncovered_source_spans(entries, [("/m/s", 10, 90), ("/m/s", 90, 110)]) \
        == [("/m/s", 90, 110)]


def test_an_entry_that_starts_with_the_tensors_and_stops_short_refuses():
    entries = [{"path": "/m/s", "offset": 40, "bytes": 10}]
    with pytest.raises(ValueError, match=r"refusing to repeat a \(path, offset\)"):
        complete_source_extent(entries, [("/m/s", 40, 90)],
                               taken={("/m/s", 40)}, where="phase layer-7")


def test_a_layer_the_reader_does_not_name_refuses(campaign):
    spans = _spans(campaign)
    del spans[1]
    with pytest.raises(ValueError, match="phase layer-1 has no source spans"):
        _build(campaign, spans)


def test_prismabuild_accepts_the_completed_manifest(campaign):
    if not (PUBLISHED_PB_SRC / "prismabuild" / "core.py").is_file():
        pytest.skip(f"published PrismaBuild not visible at {PUBLISHED_PB_SRC}")
    if str(PUBLISHED_PB_SRC) not in sys.path:
        sys.path.insert(0, str(PUBLISHED_PB_SRC))
    import prismabuild.core as core
    if not Path(core.__file__).resolve().is_relative_to(PUBLISHED_PB_SRC.resolve()):
        pytest.skip("a different prismabuild is already imported")
    import prismabuild.storage_tiers as tiers
    manifest = _build(campaign, _spans(campaign))
    core.validate_data_manifest(manifest)
    # The completion's index runs are not contiguous; PrismaBuild's read-order
    # expansion has to accept that, or the seal refuses the whole manifest.
    assert len(tiers.manifest_phase_ranges(manifest)) == \
        len(manifest["read_plan"]["phases"])
