"""A failed fused-mapping lookup must stop the export and the completeness check (#2443).

THE DEFECT THESE TESTS PIN. ``profile.fused_sibling_leaf_mapping()`` can raise: the
GLM profile reads its mapping from the Tessera runtime lane, and an interpreter
without ``tessera.serving.dense_ownership`` fails the lookup. Two callers used to
swallow every exception from it:

* ``export_native_compressed._fused_modules_mapping_for_profile`` fell through to
  the vLLM registry and the structure spec, which hold no GLM fused groups after
  #2371. It returned ``{}``, so the export skipped the fill-in of fused siblings
  into ``ignore`` and wrote a quantization config without them. No error appeared.
* ``artifact_completeness.check_artifact_completeness`` set its fusion map to ``{}``.
  Fused units were then reported as claimed by no mechanism: a wrong diagnosis for
  an environment failure.

Measured on PrismaBuild x86 (action in ``pq2443-measure2-20261008.log``): with the
lookup healthy the export mapping holds four GLM groups; with the lookup raising it
was ``{}`` and the profile call itself raised ``ModuleNotFoundError``.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

import prismaquant.artifact_completeness as completeness
import prismaquant.tessera_lane as lane
from prismaquant.export_native_compressed import _fused_modules_mapping_for_profile
from prismaquant.model_profiles.glm5_next import Glm5NextProfile


class _LookupFailed(RuntimeError):
    pass


class _RaisingProfile:
    """A profile whose fused-mapping getter raises, and that offers no fallback."""

    def fused_sibling_leaf_mapping(self):
        raise _LookupFailed("the fused-mapping lookup failed")

    def to_vllm_internal_name(self, name: str) -> str:
        return name

    def source_tensor_name(self, name: str) -> str:
        return name


def _glm_profile() -> Glm5NextProfile:
    profile = Glm5NextProfile()
    kinds = ["linear_attention"] * 46
    kinds[3] = kinds[45] = "deepseek_sparse_attention"
    profile._declare_config_document({"text_config": {"layer_types": kinds}})
    return profile


def _write_minimal_artifact(root: Path) -> None:
    """One FP8 weight with its scale and a config that declares nothing."""

    root.mkdir(parents=True, exist_ok=True)
    tensors = (
        ("model.layers.0.mlp.gate_proj.weight", "F8_E4M3", (32, 8), 32 * 8),
        ("model.layers.0.mlp.gate_proj.weight_scale", "F32", (32, 1), 32 * 4),
    )
    header: dict[str, object] = {}
    offset = 0
    for name, dtype, shape, span in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + span]}
        offset += span
    blob = json.dumps(header).encode("utf-8")
    with (root / "model.safetensors").open("wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(b"\0" * offset)
    (root / "quant_config.json").write_text(json.dumps({
        "quant_method": "compressed-tensors", "format": "float-quantized",
        "config_groups": {}, "ignore": [],
    }), encoding="utf-8")


def test_export_mapping_propagates_a_failed_profile_lookup():
    with pytest.raises(_LookupFailed):
        _fused_modules_mapping_for_profile(_RaisingProfile())


def test_completeness_check_propagates_a_failed_profile_lookup(tmp_path, monkeypatch):
    root = tmp_path / "artifact"
    _write_minimal_artifact(root)
    monkeypatch.setattr(completeness, "_detect_profile_quietly",
                        lambda _root: _RaisingProfile())
    with pytest.raises(_LookupFailed):
        completeness.check_artifact_completeness(root)


def test_glm_export_mapping_stops_when_the_lane_lookup_fails(monkeypatch):
    """The real case: the GLM lane lookup fails and the export must not return {}."""

    def lookup_failed():
        raise ModuleNotFoundError("No module named 'tessera.serving.dense_ownership'")

    monkeypatch.setattr(lane, "glm_fused_sibling_leaf_mapping", lookup_failed)
    with pytest.raises(ModuleNotFoundError):
        _fused_modules_mapping_for_profile(_glm_profile())


def test_glm_export_mapping_is_complete_when_the_lookup_works():
    """Control: with a working lookup the four GLM fused groups reach the export."""

    mapping = _fused_modules_mapping_for_profile(_glm_profile())
    assert set(mapping) == {
        "fused_qkv_a_proj", "gate_up_proj", "in_proj_qkvbfg_a", "wk_weights_proj"}
    assert mapping["in_proj_qkvbfg_a"] == (
        "q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")
