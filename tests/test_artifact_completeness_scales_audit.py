"""Static compressed-tensors schemes require their persisted weight scales."""
from __future__ import annotations

import json
import math
import struct

import pytest

from prismaquant import artifact_completeness as completeness


class _Profile:
    def source_tensor_name(self, name):
        return name

    def to_vllm_internal_name(self, name):
        return "language_model.lm_head" if name == "lm_head" else name

    def fused_sibling_leaf_mapping(self):
        return {"gate_up_proj": ("gate_proj", "up_proj")}


@pytest.fixture(autouse=True)
def _profile(monkeypatch):
    monkeypatch.setattr(completeness, "_detect_profile_quietly", lambda _root: _Profile())


def _scheme(encoding):
    if encoding == "fp8_channel":
        return {
            "format": "float-quantized",
            "weights": {"num_bits": 8, "type": "float", "strategy": "channel",
                        "symmetric": True, "dynamic": False},
            "input_activations": {"num_bits": 8, "type": "float", "strategy": "token",
                                  "symmetric": True, "dynamic": True},
        }
    if encoding == "nvfp4_packed":
        return {
            "format": "nvfp4-pack-quantized",
            "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group",
                        "group_size": 16, "symmetric": True, "dynamic": False,
                        "scale_dtype": "torch.float8_e4m3fn"},
            "input_activations": {"num_bits": 4, "type": "float", "strategy": "tensor_group",
                                  "group_size": 16, "symmetric": True, "dynamic": "local",
                                  "scale_dtype": "torch.float8_e4m3fn"},
        }
    assert encoding == "mxfp4_packed"
    return {
        "format": "mxfp4-pack-quantized",
        "weights": {"num_bits": 4, "type": "float", "strategy": "group",
                    "group_size": 32, "symmetric": True, "dynamic": False,
                    "scale_dtype": "torch.uint8"},
    }


def _write_artifact(root, *, scheme, targets, tensors):
    root.mkdir()
    header = {}
    offset = 0
    for name, dtype, shape in tensors:
        size = math.prod(shape) * {"F8_E4M3": 1, "U8": 1, "F32": 4, "BF16": 2}[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    config = {}
    if scheme is not None:
        config["quantization_config"] = {
            "quant_method": "compressed-tensors", "format": scheme["format"],
            "config_groups": {"group_0": {**scheme, "targets": targets}}, "ignore": [],
        }
    (root / "config.json").write_text(json.dumps(config))


def _quantized_artifact(root, *, encoding, fused, with_scale):
    unit = "model.layers.0.mlp.gate_up_proj" if fused else "lm_head"
    targets = (["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"]
               if fused else ["re:^language_model[.]lm_head$"])
    if encoding == "fp8_channel":
        tensors = [(unit + ".weight", "F8_E4M3", (4, 32))]
        scale = (unit + ".weight_scale", "F32", (4, 1))
    else:
        tensors = [(unit + ".weight_packed", "U8", (4, 16))]
        scale = (unit + ".weight_scale", "U8", (4, 1))
        if encoding == "nvfp4_packed":
            scale = (unit + ".weight_scale", "F8_E4M3", (4, 2))
            tensors += [(unit + ".weight_global_scale", "F32", (1,)),
                        (unit + ".input_global_scale", "F32", (1,))]
    if with_scale:
        tensors.append(scale)
    _write_artifact(root, scheme=_scheme(encoding), targets=targets, tensors=tensors)
    return unit


@pytest.mark.parametrize("encoding", ["fp8_channel", "nvfp4_packed", "mxfp4_packed"])
@pytest.mark.parametrize("fused", [False, True], ids=["wrapper_target", "fused_half_targets"])
@pytest.mark.parametrize("with_scale", [False, True], ids=["missing_scale", "complete"])
def test_static_config_group_requires_its_weight_scale(tmp_path, encoding, fused, with_scale):
    root = tmp_path / "artifact"
    unit = _quantized_artifact(root, encoding=encoding, fused=fused, with_scale=with_scale)
    report = completeness.check_artifact_completeness(root)
    if with_scale:
        assert report.ok, report.failure_text()
        assert report.cb_units == [unit]
        completeness.assert_artifact_complete(root)
    else:
        assert not report.ok
        assert unit in report.missing_scale
        with pytest.raises(completeness.ArtifactIncomplete, match=unit):
            completeness.assert_artifact_complete(root)


def test_completeness_cli_refuses_missing_static_channel_scale(tmp_path, capsys):
    root = tmp_path / "artifact"
    _quantized_artifact(root, encoding="fp8_channel", fused=False, with_scale=False)
    status = completeness.main([str(root)])
    output = capsys.readouterr().out
    assert status == 1
    assert "INCOMPLETE" in output
    assert "lm_head" in output


def test_plain_float_weight_requires_no_scale_or_quantization_config(tmp_path):
    root = tmp_path / "artifact"
    _write_artifact(root, scheme=None, targets=[], tensors=[("lm_head.weight", "BF16", (4, 32))])
    report = completeness.check_artifact_completeness(root)
    assert report.ok, report.failure_text()
    completeness.assert_artifact_complete(root)
