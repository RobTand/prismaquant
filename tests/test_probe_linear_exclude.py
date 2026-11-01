"""Probe Linear-exclusion composition (profile hook).

The faithful DSv4 vendored forward instantiates the compressor +
indexer, whose nn.Linear leaves are OUTSIDE the D0.1 serving
contract's quantizable set. The probe must keep them out of its
inventory (on an FP8-source checkpoint they would carry zero legal
candidates and trip the allocator coverage refusal), which is done
via ModelProfile.probe_linear_exclude_extra composed into
incremental_probe.resolve_linear_exclude.
"""
import re

from prismaquant.incremental_probe import (
    resolve_linear_exclude,
)
from prismaquant.probe_inventory import BASE_LINEAR_EXCLUDE
from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile
from prismaquant.model_profiles.default import DefaultProfile


def _composed(profile) -> str:
    extra = profile.probe_linear_exclude_extra()
    if extra:
        return f"(?:{BASE_LINEAR_EXCLUDE}|{extra})"
    return BASE_LINEAR_EXCLUDE


def test_default_profile_extra_is_empty():
    assert DefaultProfile().probe_linear_exclude_extra() == ""


def test_dsv4_extra_excludes_compressor_and_indexer():
    pat = re.compile(_composed(DeepseekV4Profile()))
    excluded = [
        "model.layers.3.self_attn.compressor.wkv",
        "model.layers.3.self_attn.compressor.wgate",
        "model.layers.3.self_attn.indexer.wq_b",
        "model.layers.3.self_attn.indexer.weights_proj",
        "model.layers.3.self_attn.indexer.compressor.wkv",
        "model.layers.3.mlp.gate",  # baseline router exclusion intact
    ]
    kept = [
        "model.layers.3.self_attn.wq_a",
        "model.layers.3.self_attn.wq_b",
        "model.layers.3.self_attn.wkv",
        "model.layers.3.self_attn.wo_b",
        "model.layers.3.mlp.experts.7.down_proj",
        "model.layers.3.mlp.shared_experts.gate_proj",
    ]
    for name in excluded:
        assert pat.search(name), f"should be excluded: {name}"
    for name in kept:
        assert not pat.search(name), f"must stay probeable: {name}"


def test_resolver_composes_profile_extra(monkeypatch):
    import prismaquant.incremental_probe as ip

    monkeypatch.setattr(
        ip, "_detect_profile_for_shards", lambda _p: DeepseekV4Profile()
    )
    pat = resolve_linear_exclude("/nonexistent")
    assert re.search(pat, "model.layers.0.self_attn.indexer.wkv")
    assert not re.search(pat, "model.layers.0.self_attn.wkv")


def test_resolver_tolerates_profiles_without_hook(monkeypatch):
    import prismaquant.incremental_probe as ip

    class _Legacy:  # no probe_linear_exclude_extra attribute
        pass

    monkeypatch.setattr(ip, "_detect_profile_for_shards", lambda _p: _Legacy())
    assert resolve_linear_exclude("/nonexistent") == BASE_LINEAR_EXCLUDE
