"""Safetensors header readers delegate to one owner (PQ #1303 slice).

The owning reader is ``prismaquant.source_read_plan.read_safetensors_header``
(the only one with length bounds, staged-read support and a header-is-object
check). Every other site that hand-parsed the 8-byte length prefix and the
JSON header must route through it. These tests pin two things per site:
byte-and-behavior identity against the old inline parse over a real fixture
checkpoint, and actual delegation (a raising stub in the consumer's namespace
must surface through the site's public callable).
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def _write_shard(path: Path, *, metadata=True) -> dict:
    """Write one small, real safetensors shard; return its header dict."""
    header = {
        "__metadata__": {"format": "pt"} if metadata else {"x": "1"},
        "tensor.a.weight": {"dtype": "F32", "shape": [2, 2],
                            "data_offsets": [0, 16]},
        "tensor.b.weight": {"dtype": "BF16", "shape": [4],
                            "data_offsets": [16, 24]},
    }
    blob = json.dumps(header, separators=(",", ":")).encode()
    payload = bytes(24)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + payload)
    return header


def _old_parse(path: Path) -> dict:
    """The pre-consolidation inline parse, kept here as the equality oracle."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


@pytest.fixture()
def checkpoint(tmp_path):
    shard = tmp_path / "model-00001-of-00002.safetensors"
    header = _write_shard(shard)
    other = tmp_path / "model-00002-of-00002.safetensors"
    _write_shard(other, metadata=False)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"tensor.a.weight": shard.name,
                        "tensor.b.weight": shard.name}}))
    return tmp_path, shard, header


def test_footprint_header_is_the_inline_parse(checkpoint):
    from prismaquant import footprint
    _dir, shard, _header = checkpoint
    assert footprint._read_safetensors_header(str(shard)) == _old_parse(shard)


def test_artifact_completeness_header_filters_metadata_like_before(checkpoint):
    from prismaquant import artifact_completeness
    _dir, shard, _header = checkpoint
    old = {k: v for k, v in _old_parse(shard).items() if k != "__metadata__"}
    assert artifact_completeness._read_safetensors_header(shard) == old


def test_pipeline_parameter_count_matches_the_inline_parse(checkpoint):
    from prismaquant import pipeline
    directory, _shard, _header = checkpoint
    # the model carries an index, so the counter reads exactly the indexed
    # shards -- the same set the old inline parse read
    index = json.loads((Path(directory) / "model.safetensors.index.json").read_text())
    old_total = 0
    for name in sorted(set(index["weight_map"].values())):
        shard = Path(directory) / name
        for tensor, meta in _old_parse(shard).items():
            if tensor == "__metadata__":
                continue
            old_total += int(meta["shape"][0]) * (
                int(meta["shape"][1]) if len(meta["shape"]) > 1 else 1)
    assert pipeline._safetensors_parameter_count(directory) == old_total


def test_autoscale_resident_bytes_matches_the_inline_parse(checkpoint):
    from prismaquant import autoscale
    _dir, shard, header = checkpoint
    old_bytes = sum(int(m["shape"][0]) * (int(m["shape"][1])
                                          if len(m["shape"]) > 1 else 1) * 2
                    for k, m in header.items() if k != "__metadata__")
    assert autoscale._shard_resident_bytes(shard, dtype_bytes=2) == old_bytes


def test_tp2_header_is_the_inline_parse(checkpoint):
    import tp2_budget_plan
    _dir, shard, _header = checkpoint
    assert tp2_budget_plan.read_safetensors_header(shard) == _old_parse(shard)


def test_tp2_refusals_stay_header_errors(tmp_path):
    import tp2_budget_plan
    truncated = tmp_path / "trunc.safetensors"
    truncated.write_bytes(b"\x01\x02\x03")
    with pytest.raises(tp2_budget_plan.HeaderError):
        tp2_budget_plan.read_safetensors_header(truncated)


def test_chain_roll_spans_match_the_inline_parse(checkpoint):
    import chain_roll_bench
    directory, _shard, header = checkpoint
    spans = chain_roll_bench._safetensors_spans(
        Path(directory), ["tensor.a.weight", "tensor.b.weight"])
    assert spans, "the bench must still produce merged read spans"
    covered = {(s["path"], s["offset"], s["bytes"]) for s in spans}
    first = Path(directory) / "model-00001-of-00002.safetensors"
    blob_len = len(json.dumps(header, separators=(",", ":")).encode())
    # header + both contiguous tensors merge into one whole-file read
    assert (str(first), 0, 8 + blob_len + 24) in covered


ROUTING_SITES = [
    ("prismaquant.footprint", "_read_safetensors_header"),
    ("prismaquant.artifact_completeness", "_read_safetensors_header"),
    ("prismaquant.pipeline", "_safetensors_parameter_count"),
    ("prismaquant.autoscale", "_shard_resident_bytes"),
]


def _raise(path, *args, **kwargs):
    raise AssertionError("routed through the owning reader")


@pytest.mark.parametrize("module_name, callable_name", ROUTING_SITES)
def test_prismaquant_sites_route_through_the_owner(
        monkeypatch, checkpoint, module_name, callable_name):
    import importlib
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "read_safetensors_header", _raise, raising=False)
    directory, shard, _header = checkpoint
    arguments = {
        "_read_safetensors_header": (str(shard),),
        "_safetensors_parameter_count": (str(directory),),
        "_shard_resident_bytes": (shard, 2),
    }[callable_name]
    with pytest.raises(AssertionError, match="routed through"):
        getattr(module, callable_name)(*arguments)


def test_tool_sites_route_through_the_owner(monkeypatch, checkpoint):
    import chain_roll_bench
    import tp2_budget_plan
    directory, shard, _header = checkpoint
    monkeypatch.setattr(chain_roll_bench, "read_safetensors_header", _raise,
                        raising=False)
    with pytest.raises(AssertionError, match="routed through"):
        chain_roll_bench._safetensors_spans(Path(directory), ["tensor.a.weight"])
    monkeypatch.setattr(tp2_budget_plan.source_read_plan,
                        "read_safetensors_header", _raise, raising=False)
    with pytest.raises(tp2_budget_plan.HeaderError, match="routed through"):
        tp2_budget_plan.read_safetensors_header(shard)
