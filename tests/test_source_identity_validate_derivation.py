"""Validator derivation matches the streaming runner (PQ #854).

`validate_cached_streamed_model_identity` must derive its live config the
same way `_build_streaming_context` does -- profile-gated staging plus the
real meta skeleton, which applies constructor defaults -- instead of a
hardcoded text-only `AutoConfig` load.  A hardcoded derivation can never
agree with an umbrella cache on a multimodal-skeleton family, and stopping
at `AutoConfig` misses constructor normalization even on the right branch.

All tests are monkeypatch-scoped (auto-reverting): no `os.environ` writes,
no module-global mutation.  Fixture shards are kilobytes; no model payload
scale is involved anywhere here.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

transformers = pytest.importorskip("transformers")

from prismaquant import cost_streaming as cs


def _llama_config_dict():
    try:
        from transformers import LlamaConfig
    except ImportError:
        pytest.skip("no LlamaConfig in installed transformers")
    return LlamaConfig().to_dict()


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_llama_config_dict()))
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 196608},
        "weight_map": {"a.weight": "a.safetensors",
                       "b.weight": "b.safetensors"},
    }))
    shards = {}
    for name, payload in (("a.safetensors", b"a" * 65536),
                          ("b.safetensors", b"b" * 131072)):
        path = root / name
        path.write_bytes(payload)
        shards[name] = path
    return root, shards


def _runner(shard_paths, config_dict):
    return SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(
            to_dict=lambda: dict(config_dict))),
        context=SimpleNamespace(
            weight_ckpt={},
            weight_shard={name: str(path)
                          for name, path in shard_paths.items()}))


def _build_cache(root, shards, config_dict):
    cache = root / "identity-cache.json"
    identity = cs.build_streamed_model_identity(
        _runner(shards, config_dict), str(root), identity_cache_path=cache)
    assert cache.is_file()
    return cache, identity


def test_text_only_end_to_end_agrees_with_runner_derivation(checkpoint):
    """Text-only family: build-then-validate agrees through the new derivation.

    The cache is built from a plain `LlamaConfig` runner while the validator
    derives live config through the real meta skeleton.  Agreement proves the
    shared derivation preserves existing text-only behavior.
    """
    root, shards = checkpoint
    config_dict = _llama_config_dict()
    config_dict["_name_or_path"] = str(root)
    cache, identity = _build_cache(root, shards, config_dict)
    validated = cs.validate_cached_streamed_model_identity(str(root), cache)
    assert validated["content_sha256"] == identity["content_sha256"]


def test_multimodal_branch_uses_multimodal_staging_and_skeleton(
        monkeypatch, checkpoint):
    """A multimodal-skeleton profile takes the multimodal branch end to end.

    Spies record the branch; every other call delegates to the real
    function, and the Llama fixture resolves through the real meta
    skeleton, so a passing validation proves the wiring, not a mock.
    """
    import prismaquant.sensitivity_probe as probe
    import prismaquant.streaming_model as streaming

    root, shards = checkpoint
    calls: list = []

    real_stage_mm = probe.stage_multimodal
    real_skeleton = streaming.build_streaming_skeleton

    def spy_stage_mm(source):
        calls.append("stage_multimodal")
        return real_stage_mm(source)

    def refuse_stage_text_only(source):
        raise AssertionError("text-only staging must not run here")

    def spy_skeleton(config, *, multimodal, **kwargs):
        calls.append(("skeleton", multimodal))
        return real_skeleton(config, multimodal=multimodal, **kwargs)

    class _MultimodalProfile:
        requires_multimodal_skeleton = lambda self: True  # noqa: E731

    monkeypatch.setattr(
        "prismaquant.model_profiles.detect_profile",
        lambda source: _MultimodalProfile())
    # The helper under test imports these names from their home modules.
    monkeypatch.setattr(probe, "stage_multimodal", spy_stage_mm)
    monkeypatch.setattr(probe, "stage_text_only", refuse_stage_text_only)
    monkeypatch.setattr(streaming, "build_streaming_skeleton", spy_skeleton)

    config_dict = _llama_config_dict()
    config_dict["_name_or_path"] = str(root)
    cache, identity = _build_cache(root, shards, config_dict)
    validated = cs.validate_cached_streamed_model_identity(str(root), cache)
    assert validated["content_sha256"] == identity["content_sha256"]
    assert "stage_multimodal" in calls
    assert ("skeleton", True) in calls


def test_live_config_comes_from_skeleton_not_raw_autoconfig(
        monkeypatch, checkpoint):
    """The constructor step is load-bearing in the derivation.

    The stub skeleton returns a config with one normalized field that the
    raw `AutoConfig` dict lacks.  A cache built from the normalized form
    validates only because derivation reads the skeleton return.
    """
    import prismaquant.streaming_model as streaming

    root, shards = checkpoint
    raw = _llama_config_dict()
    normalized = dict(raw)
    normalized["rms_norm_eps"] = float(raw.get("rms_norm_eps", 1e-6)) / 2
    normalized["_name_or_path"] = str(root)

    def stub_skeleton(config, *, multimodal, **kwargs):
        assert multimodal is False
        return SimpleNamespace(config=SimpleNamespace(
            to_dict=lambda: dict(normalized)))

    monkeypatch.setattr(streaming, "build_streaming_skeleton", stub_skeleton)
    runner_config = dict(normalized)
    cache, identity = _build_cache(root, shards, runner_config)
    validated = cs.validate_cached_streamed_model_identity(str(root), cache)
    assert validated["content_sha256"] == identity["content_sha256"]

    # The same cache refuses when the runner config was the raw form: the
    # skeleton normalization is part of the bound identity, not advisory.
    cache2 = root / "identity-cache-raw.json"
    cs.build_streamed_model_identity(
        _runner(shards, dict(raw, _name_or_path=str(root))),
        str(root), identity_cache_path=cache2)
    with pytest.raises(RuntimeError, match="live config differs"):
        cs.validate_cached_streamed_model_identity(str(root), cache2)


def test_mutated_semantic_field_still_refuses(monkeypatch, checkpoint):
    """Comparison strictness is unchanged: one drifted field refuses by name."""
    import prismaquant.streaming_model as streaming

    root, shards = checkpoint
    base = _llama_config_dict()
    base["_name_or_path"] = str(root)
    cache, _ = _build_cache(root, shards, base)

    drifted = dict(base)
    drifted["hidden_size"] = int(base["hidden_size"]) + 1

    def stub_skeleton(config, *, multimodal, **kwargs):
        return SimpleNamespace(config=SimpleNamespace(
            to_dict=lambda: dict(drifted)))

    monkeypatch.setattr(streaming, "build_streaming_skeleton", stub_skeleton)
    with pytest.raises(RuntimeError, match=r"changed=\['hidden_size'\]"):
        cs.validate_cached_streamed_model_identity(str(root), cache)


def test_skeleton_failure_is_fail_closed(monkeypatch, checkpoint):
    """A derivation failure refuses validation rather than skipping it."""
    import prismaquant.streaming_model as streaming

    root, shards = checkpoint
    config_dict = _llama_config_dict()
    config_dict["_name_or_path"] = str(root)
    cache, _ = _build_cache(root, shards, config_dict)

    def refuse_skeleton(config, *, multimodal, **kwargs):
        raise RuntimeError("no skeleton here")

    monkeypatch.setattr(streaming, "build_streaming_skeleton", refuse_skeleton)
    with pytest.raises(RuntimeError, match="cannot validate the live source config"):
        cs.validate_cached_streamed_model_identity(str(root), cache)
