"""Streaming-audit regression: `estimate_per_layer_bytes` sized the layer
cache from *on-disk* bytes, but fp8-native checkpoints dequant to bf16 in
the layer cache (`_read_layer_to_device` casts every floating tensor to
the execution dtype) — a 2x resident undercount that overshoots the
memory budget. The estimate must be dtype-aware the way
`streaming_model._estimate_layer_cache_bytes` already is.
"""
import json

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import autoscale as A


def _write_model(tmp_path, tensors):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "toy"}))
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return str(tmp_path)


def test_fp8_source_estimate_doubles_vs_disk_bytes(tmp_path):
    numel = 256 * 256
    path = _write_model(
        tmp_path / "fp8",
        {"model.layers.0.proj.weight":
         torch.randn(256, 256).to(torch.float8_e4m3fn)},
    )
    disk_tensor_bytes = numel  # fp8 = 1 byte/elem on disk
    per_layer_weight, _ = A.estimate_per_layer_bytes(
        path, num_layers=1, hidden_size=256, nsamples=1, seqlen=1,
        dtype_bytes=2,
    )
    # resident = numel * 2 (bf16 after block dequant), body fudge 0.90
    assert per_layer_weight == int(disk_tensor_bytes * 2 * 0.90)


def test_bf16_source_estimate_matches_disk_bytes(tmp_path):
    numel = 256 * 256
    path = _write_model(
        tmp_path / "bf16",
        {"model.layers.0.proj.weight":
         torch.randn(256, 256, dtype=torch.bfloat16)},
    )
    per_layer_weight, _ = A.estimate_per_layer_bytes(
        path, num_layers=1, hidden_size=256, nsamples=1, seqlen=1,
        dtype_bytes=2,
    )
    assert per_layer_weight == int(numel * 2 * 0.90)


def test_fp8_and_bf16_checkpoints_estimate_the_same_resident_size(tmp_path):
    """Same model at fp8 vs bf16 on disk must produce the SAME resident
    estimate — the layer cache holds the dequanted bf16 tensors either way."""
    fp8 = _write_model(
        tmp_path / "fp8",
        {"w": torch.randn(128, 128).to(torch.float8_e4m3fn)})
    bf16 = _write_model(
        tmp_path / "bf16",
        {"w": torch.randn(128, 128, dtype=torch.bfloat16)})
    kw = dict(num_layers=1, hidden_size=128, nsamples=1, seqlen=1,
              dtype_bytes=2)
    w_fp8, _ = A.estimate_per_layer_bytes(fp8, **kw)
    w_bf16, _ = A.estimate_per_layer_bytes(bf16, **kw)
    assert w_fp8 == w_bf16


def test_declared_fp4_expert_estimate_quadruples_disk_bytes(tmp_path):
    """MXFP4 nibble-packed I8 experts (declared via config expert_dtype)
    dequant to 2 logical elements x bf16 per on-disk byte — 4x the disk
    size, not the verbatim 1x that under-evicted before every load."""
    packed_bytes = 256 * 128
    d = tmp_path / "fp4"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(
        json.dumps({"model_type": "toy", "expert_dtype": "fp4"}))
    save_file(
        {"model.layers.0.mlp.experts.0.gate_proj.weight":
         torch.zeros(256, 128, dtype=torch.int8)},
        str(d / "model.safetensors"),
    )
    per_layer_weight, _ = A.estimate_per_layer_bytes(
        str(d), num_layers=1, hidden_size=64, nsamples=1, seqlen=1,
        dtype_bytes=2,
    )
    assert per_layer_weight == int(packed_bytes * 2 * 2 * 0.90)


def test_declared_fp4_expert_estimate_covers_uint8_packs(tmp_path):
    """`layer_streaming._check_mxfp4_packed_grid` accepts int8 AND uint8
    nibble-packs, so both must price at 2 logical elements x execution
    dtype per packed byte. Sizing only I8 left a U8-packed checkpoint with
    the 4x undercount that under-evicts before every load."""
    packed_bytes = 256 * 128
    d = tmp_path / "fp4u8"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(
        json.dumps({"model_type": "toy", "expert_dtype": "fp4"}))
    save_file(
        {"model.layers.0.mlp.experts.0.gate_proj.weight":
         torch.zeros(256, 128, dtype=torch.uint8)},
        str(d / "model.safetensors"),
    )
    per_layer_weight, _ = A.estimate_per_layer_bytes(
        str(d), num_layers=1, hidden_size=64, nsamples=1, seqlen=1,
        dtype_bytes=2,
    )
    assert per_layer_weight == int(packed_bytes * 2 * 2 * 0.90)


@pytest.mark.parametrize("packed_dtype", [torch.int8, torch.uint8])
def test_undeclared_byte_expert_stays_verbatim(tmp_path, packed_dtype):
    """No expert_dtype declaration -> a 1-byte integer expert tensor is
    genuine int8/uint8, kept 1 byte/elem (the explicit config declaration,
    never a shape/name heuristic alone, is what flips MXFP4 pricing)."""
    packed_bytes = 256 * 128
    path = _write_model(
        tmp_path / f"raw-{packed_dtype}".replace(".", "-"),
        {"model.layers.0.mlp.experts.0.gate_proj.weight":
         torch.zeros(256, 128, dtype=packed_dtype)},
    )
    per_layer_weight, _ = A.estimate_per_layer_bytes(
        path, num_layers=1, hidden_size=64, nsamples=1, seqlen=1,
        dtype_bytes=2,
    )
    assert per_layer_weight == int(packed_bytes * 0.90)


def test_non_float_tensors_kept_verbatim(tmp_path):
    numel = 1024
    path = _write_model(
        tmp_path / "ints",
        {"tok": torch.zeros(numel, dtype=torch.int64)})
    per_layer_weight, _ = A.estimate_per_layer_bytes(
        path, num_layers=1, hidden_size=64, nsamples=1, seqlen=1,
        dtype_bytes=2,
    )
    assert per_layer_weight == int(numel * 8 * 0.90)


def test_unparseable_blob_falls_back_to_file_size(tmp_path):
    d = tmp_path / "junk"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "toy"}))
    (d / "model.safetensors").write_bytes(b"\0" * 4096)
    per_layer_weight, _ = A.estimate_per_layer_bytes(
        str(d), num_layers=1, hidden_size=64, nsamples=1, seqlen=1,
        dtype_bytes=2,
    )
    assert per_layer_weight == int(4096 * 0.90)


def test_float64_source_estimate_casts_floats_and_keeps_integer_bytes_verbatim(tmp_path):
    path = _write_model(tmp_path, {'model.layers.0.proj.weight': torch.zeros(10, 10, dtype=torch.float64),
                                  'model.layers.0.index': torch.zeros(20, dtype=torch.uint8)})
    before = {item.name: item.read_bytes() for item in tmp_path.iterdir()}
    assert A.estimate_per_layer_bytes(path, num_layers=2, hidden_size=10,
        nsamples=1, seqlen=1, dtype_bytes=2, act_mult=1) == (99, 119)
    assert {item.name: item.read_bytes() for item in tmp_path.iterdir()} == before
