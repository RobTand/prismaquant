"""FP8 block-dequant robustness (numerical audit 2026-07-02, §3.7).

a) A transposed `(in_blocks, out_blocks)` scale grid is numel-compatible
   with the dequant reshape and previously mis-scaled every block
   silently — it must now raise.
b) The dequant block size must come from the checkpoint-declared
   `quantization_config.weight_block_size` (previously hardcoded 128x128
   while the config value was checked for presence but never read), and
   missing/null declarations on fp8 checkpoints must hard-fail.
c) A float8 source tensor with no entry in the scale map must fail fast
   instead of silently casting raw codes to bf16 (the historical ±448
   range bug), with PRISMAQUANT_ALLOW_UNSCALED_FP8=1 as the explicit
   escape.
"""
import json

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.layer_streaming import (
    Fp8ScaleInvMap,
    _apply_fp8_dequant_inplace,
    _apply_source_scale_values,
    _build_fp8_scale_inv_map,
    _dequant_fp8_block_weight,
    _read_layer_to_device,
)

CPU = torch.device("cpu")


def _write_fp8_checkpoint(tmp_path, *, weight, scale_inv, block_size,
                          name="model.layers.0.mlp.down_proj"):
    """Minimal single-shard fp8 checkpoint with a block scale sibling."""
    cfg = {"model_type": "toy", "architectures": ["ToyForCausalLM"]}
    if block_size != "omit-config-key":
        cfg["quantization_config"] = {
            "quant_method": "fp8",
            "weight_block_size": block_size,
        }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    tensors = {f"{name}.weight": weight}
    if scale_inv is not None:
        tensors[f"{name}.weight_scale_inv"] = scale_inv
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return name


# ---------------------------------------------------------------------------
# a) transposed scale grid raises
# ---------------------------------------------------------------------------

def test_dequant_fp8_block_weight_rejects_transposed_scale():
    w = torch.randn(128, 256).to(torch.float8_e4m3fn).to(torch.bfloat16)
    good = torch.ones(1, 2)   # (out_blocks, in_blocks)
    bad = torch.ones(2, 1)    # transposed — numel-compatible

    _dequant_fp8_block_weight(w, good)  # sanity: correct grid passes
    with pytest.raises(ValueError, match="transposed"):
        _dequant_fp8_block_weight(w, bad)


def test_dequant_fp8_block_weight_rejects_scalar_scale():
    w = torch.randn(128, 128).to(torch.float8_e4m3fn).to(torch.bfloat16)
    with pytest.raises(ValueError, match="expected"):
        _dequant_fp8_block_weight(w, torch.tensor(2.0))


def test_apply_fp8_dequant_inplace_rejects_transposed_scale(tmp_path):
    w = torch.randn(128, 256).to(torch.float8_e4m3fn)
    name = _write_fp8_checkpoint(
        tmp_path, weight=w, scale_inv=torch.ones(2, 1),
        block_size=[128, 128])
    fp8_map = _build_fp8_scale_inv_map(str(tmp_path))
    out = {name + ".weight": w.to(torch.bfloat16)}
    with pytest.raises(ValueError, match="transposed"):
        _apply_fp8_dequant_inplace(out, fp8_map, CPU)


# ---------------------------------------------------------------------------
# b) declared weight_block_size is read and used
# ---------------------------------------------------------------------------

def test_declared_block_size_64_dequants_correctly(tmp_path):
    torch.manual_seed(0)
    out_dim, in_dim, blk = 128, 192, 64
    codes = torch.randn(out_dim, in_dim).to(torch.float8_e4m3fn)
    # Power-of-two scales make the bf16 multiply exact vs the fp32
    # reference expansion.
    scale = torch.tensor(
        [[0.5, 2.0, 4.0], [1.0, 8.0, 0.25]], dtype=torch.float32)
    assert scale.shape == (out_dim // blk, in_dim // blk)
    name = _write_fp8_checkpoint(
        tmp_path, weight=codes, scale_inv=scale, block_size=[64, 64])

    fp8_map = _build_fp8_scale_inv_map(str(tmp_path))
    assert fp8_map.block == (64, 64)
    assert name + ".weight" in fp8_map

    shard = str(tmp_path / "model.safetensors")
    key = name + ".weight"
    tensors = _read_layer_to_device(
        "model.layers.0.",
        {key: shard},
        {key: key},
        torch.bfloat16,
        CPU,
        fp8_scale_inv_map=fp8_map,
    )
    got = tensors[key]
    assert got.dtype == torch.bfloat16

    expanded = scale.repeat_interleave(blk, dim=0).repeat_interleave(
        blk, dim=1)
    ref = codes.to(torch.float32) * expanded
    torch.testing.assert_close(got.to(torch.float32), ref, rtol=0.0, atol=0.0)

    # A 128x128 assumption on this checkpoint would have produced a
    # (1, 2)-grid expectation; the declared-block map must instead
    # validate against the real (2, 3) grid — proven by the exact match
    # above and by the transposed test rejecting wrong grids.


def test_fp8_checkpoint_without_declared_block_size_fails(tmp_path):
    w = torch.randn(128, 128).to(torch.float8_e4m3fn)
    _write_fp8_checkpoint(
        tmp_path, weight=w, scale_inv=torch.ones(1, 1),
        block_size="omit-config-key")
    with pytest.raises(RuntimeError, match="weight_block_size"):
        _build_fp8_scale_inv_map(str(tmp_path))


def test_fp8_checkpoint_with_null_block_size_fails(tmp_path):
    # Real case: Mistral-Medium-3.5 ships quant_method=fp8 with
    # weight_block_size: null and scalar per-tensor scales.
    w = torch.randn(128, 128).to(torch.float8_e4m3fn)
    _write_fp8_checkpoint(
        tmp_path, weight=w, scale_inv=torch.ones(1, 1), block_size=None)
    with pytest.raises(RuntimeError, match="weight_block_size"):
        _build_fp8_scale_inv_map(str(tmp_path))


def test_fp8_checkpoint_with_malformed_block_size_fails(tmp_path):
    w = torch.randn(128, 128).to(torch.float8_e4m3fn)
    _write_fp8_checkpoint(
        tmp_path, weight=w, scale_inv=torch.ones(1, 1),
        block_size=[128, 128, 128])
    with pytest.raises(RuntimeError, match="unsupported"):
        _build_fp8_scale_inv_map(str(tmp_path))


def test_bf16_checkpoint_yields_empty_map_without_config_demands(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "toy"}))
    save_file(
        {"model.layers.0.proj.weight": torch.randn(8, 8, dtype=torch.bfloat16)},
        str(tmp_path / "model.safetensors"),
    )
    fp8_map = _build_fp8_scale_inv_map(str(tmp_path))
    assert fp8_map == {}
    assert fp8_map.block is None


# ---------------------------------------------------------------------------
# c) unmapped float8 tensors fail fast
# ---------------------------------------------------------------------------

def _unmapped_fp8_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "toy"}))
    key = "model.layers.0.proj.weight"
    save_file(
        {key: torch.randn(8, 8).to(torch.float8_e4m3fn)},
        str(tmp_path / "model.safetensors"),
    )
    return key, str(tmp_path / "model.safetensors")


def test_unmapped_fp8_tensor_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_ALLOW_UNSCALED_FP8", raising=False)
    key, shard = _unmapped_fp8_dir(tmp_path)
    with pytest.raises(RuntimeError, match="fp8_scale_inv_map"):
        _read_layer_to_device(
            "model.layers.0.", {key: shard}, {key: key},
            torch.bfloat16, CPU, fp8_scale_inv_map=None,
        )
    # Same guard with an EMPTY map (fp8-native checkpoint whose scale
    # scan found nothing).
    with pytest.raises(RuntimeError, match="scale"):
        _read_layer_to_device(
            "model.layers.0.", {key: shard}, {key: key},
            torch.bfloat16, CPU, fp8_scale_inv_map=Fp8ScaleInvMap(),
        )


def test_unmapped_fp8_tensor_escape_env_allows_raw_cast(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_ALLOW_UNSCALED_FP8", "1")
    key, shard = _unmapped_fp8_dir(tmp_path)
    tensors = _read_layer_to_device(
        "model.layers.0.", {key: shard}, {key: key},
        torch.bfloat16, CPU, fp8_scale_inv_map=None,
    )
    assert tensors[key].dtype == torch.bfloat16


def test_mapped_fp8_tensor_passes_the_guard(tmp_path, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_ALLOW_UNSCALED_FP8", raising=False)
    w = torch.randn(128, 128).to(torch.float8_e4m3fn)
    name = _write_fp8_checkpoint(
        tmp_path, weight=w, scale_inv=torch.ones(1, 1), block_size=[128, 128])
    fp8_map = _build_fp8_scale_inv_map(str(tmp_path))
    key = name + ".weight"
    shard = str(tmp_path / "model.safetensors")
    tensors = _read_layer_to_device(
        "model.layers.0.", {key: shard}, {key: key},
        torch.bfloat16, CPU, fp8_scale_inv_map=fp8_map,
    )
    assert tensors[key].dtype == torch.bfloat16


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('path_kind', ['batched-fp8', 'fallback-fp8', 'mxfp4'])
def test_dequant_copy_policy_preserves_disabled_and_original_arithmetic(
        tmp_path, enabled, path_kind):
    """CPU policy/arithmetic control; no CUDA or original-provider acceptance."""
    if path_kind == 'mxfp4':
        weight = torch.full((4, 128), 0x22, dtype=torch.int8)
        scales = torch.full((4, 8), 127, dtype=torch.uint8)
        block = (1, 1)
        declared = frozenset({'weight'})
        expected = torch.ones((4, 256), dtype=torch.bfloat16)
    else:
        block = (3, 3) if path_kind == 'fallback-fp8' else (1, 1)
        shape = (10, 22) if path_kind == 'fallback-fp8' else (4, 8)
        weight = torch.ones(shape, dtype=torch.bfloat16)
        scales = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        declared = frozenset()
        expected = scales.repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)
        expected = expected[:shape[0], :shape[1]].to(torch.bfloat16)
    shard = tmp_path / 'scale.safetensors'
    save_file({'scale': scales}, str(shard))
    mapping = Fp8ScaleInvMap({'weight': (str(shard), 'scale')},
                            block=block, mxfp4_names=declared)
    calls = []
    retained = []

    class Copies:
        def retain(self, tensor):
            retained.append(tensor)

        def copy(self, tensor, **kwargs):
            calls.append(dict(kwargs))
            assert kwargs['non_blocking'] is enabled
            return tensor.to(device=CPU, **kwargs)

    copies = Copies()
    copies.enabled = enabled
    out = {'weight': weight}
    assert _apply_source_scale_values(out, mapping, CPU,
                                      source_authentication=None, copies=copies) == 1
    assert len(calls) == (1 if path_kind == 'fallback-fp8' else 2)
    assert retained and out['weight'].dtype == torch.bfloat16
    assert torch.equal(out['weight'], expected)
