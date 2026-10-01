"""Stock AutoConfig consumes declared metadata without changing its model root."""
from __future__ import annotations

import hashlib
import json

import pytest

from prismaquant import streaming_model, staged_whole_file
from prismaquant.staged_tier_policy import TierPolicyRefused
from test_streamed_metadata_staged_reads import (  # noqa: F401
    checkpoint, _activate, _deny_pool_opens, _forget_state,
)

pytestmark = pytest.mark.own_process


def _config(checkpoint, extra=None):
    root, _old, _index, manifest = checkpoint
    config = {"model_type": "llama", "hidden_size": 32,
              "intermediate_size": 64, "num_hidden_layers": 1,
              "num_attention_heads": 2, "num_key_value_heads": 2,
              "architectures": ["LlamaForCausalLM"]}
    config.update(extra or {})
    raw = json.dumps(config).encode()
    (root / "config.json").write_bytes(raw)
    for entry in manifest["entries"]:
        if entry["path"] == str(root / "config.json"):
            entry["bytes"] = len(raw)
            entry["sha256"] = hashlib.sha256(raw).hexdigest()
    return root, config, manifest


@pytest.mark.parametrize("extra", [{}, {
    "model_type": "mistral", "layer_types": ["full_attention"],
    "architectures": ["MistralForCausalLM"],
}])
def test_stock_class_defaults_and_root_survive_staged_input(checkpoint, tmp_path, monkeypatch, extra):
    from transformers import AutoConfig

    root, _cfg, manifest = _config(checkpoint, extra)
    baseline = AutoConfig.from_pretrained(str(root), trust_remote_code=True,
                                           local_files_only=True)
    _activate(tmp_path, monkeypatch, manifest)
    _deny_pool_opens(monkeypatch, root)
    result = streaming_model.load_streaming_auto_config(
        str(root), str(root), local_files_only=True)
    assert type(result) is type(baseline)
    assert result.to_dict() == baseline.to_dict()
    assert result.name_or_path == str(root)


def test_missing_config_material_refuses_no_pool_fallback(checkpoint, tmp_path, monkeypatch):
    root, _cfg, manifest = _config(checkpoint)
    _activate(tmp_path, monkeypatch, manifest, skip={str(root / "config.json")})
    _deny_pool_opens(monkeypatch, root)
    with pytest.raises(TierPolicyRefused):
        streaming_model.load_streaming_auto_config(str(root), str(root), local_files_only=True)


def test_bad_config_bytes_refuse_before_hf(checkpoint, tmp_path, monkeypatch):
    root, _cfg, manifest = _config(checkpoint)
    _activate(tmp_path, monkeypatch, manifest)
    original = staged_whole_file.read_staged_entry

    def damaged(*args, **kwargs):
        return original(*args, **kwargs) + b" "

    monkeypatch.setattr(staged_whole_file, "read_staged_entry", damaged)
    _deny_pool_opens(monkeypatch, root)
    with pytest.raises(TierPolicyRefused, match="digest"):
        streaming_model.load_streaming_auto_config(str(root), str(root), local_files_only=True)


@pytest.mark.parametrize("extra", [
    {"configuration_files": ["config.other.json"]},
    {"auto_map": {"AutoConfig": "rogue.CustomConfig"}},
])
def test_indirect_or_dynamic_config_is_not_an_undeclared_fallback(checkpoint, tmp_path, monkeypatch, extra):
    root, _cfg, manifest = _config(checkpoint, extra)
    _activate(tmp_path, monkeypatch, manifest)
    _deny_pool_opens(monkeypatch, root)
    with pytest.raises(RuntimeError, match="configuration|AutoConfig"):
        streaming_model.load_streaming_auto_config(str(root), str(root), local_files_only=True)


@pytest.mark.parametrize("fail", [False, True])
def test_config_buffer_closes_on_stock_return_or_failure(checkpoint, tmp_path, monkeypatch, fail):
    import os
    from transformers import AutoConfig
    from prismaquant import io_engine

    root, _cfg, manifest = _config(checkpoint)
    _activate(tmp_path, monkeypatch, manifest)
    _deny_pool_opens(monkeypatch, root)
    paths = []
    original = io_engine.SealedBuffer

    class TrackedBuffer(original):
        def __init__(self, size):
            super().__init__(size)
            paths.append(self.path)

    monkeypatch.setattr(io_engine, "SealedBuffer", TrackedBuffer)
    if fail:
        def rejected(*args, **kwargs):
            assert kwargs["_configuration_file"] == paths[-1]
            assert os.path.isfile(paths[-1])
            raise ValueError("stock config rejected")
        monkeypatch.setattr(AutoConfig, "from_pretrained", rejected)
        with pytest.raises(ValueError, match="stock config rejected"):
            streaming_model.load_streaming_auto_config(str(root), str(root), local_files_only=True)
    else:
        streaming_model.load_streaming_auto_config(str(root), str(root), local_files_only=True)
    assert len(paths) == 1
    with pytest.raises(FileNotFoundError):
        os.stat(paths[0])


def test_existing_buffer_exact_bytes_and_immutable_lifetime():
    from prismaquant.io_engine import SealedBuffer

    buffer = SealedBuffer(3)
    try:
        with pytest.raises(ValueError, match="declared size"):
            buffer.fill_bytes(b"ab")
        buffer.fill_bytes(b"abc")
        buffer.seal()
        assert bytes(buffer) == b"abc"
        with pytest.raises(RuntimeError, match="not writable"):
            buffer.fill_bytes(b"xyz")
    finally:
        buffer.close()
    with pytest.raises(RuntimeError, match="not writable"):
        buffer.fill_bytes(b"abc")
