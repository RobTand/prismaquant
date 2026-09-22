"""Stage B must register the served A4 arithmetic before pricing or caching."""
from types import SimpleNamespace

import pytest

from prismaquant import nvfp4_activation_contract as contract
from prismaquant.joint_cost_quantum import bind_joint_served_quantizer


def test_a4_requires_registered_operator_and_returns_exact_build(monkeypatch):
    calls = []
    identity = {"backend": "registered_op", "image_content_sha256": "a" * 64}
    def bind(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(as_record=lambda: identity)
    monkeypatch.setattr(contract, "bind_served_quantizer_identity", bind)
    result = bind_joint_served_quantizer({"q": ["BF16", "TESSERA_E2M1_K2_R896"]})
    assert result == identity
    assert len(calls) == 1 and calls[0]["require"] is True


def test_a4_registration_failure_is_not_a_model_fallback(monkeypatch):
    def refuse(**kwargs):
        raise contract.ServedQuantizerUnboundError("operator unavailable")
    monkeypatch.setattr(contract, "bind_served_quantizer_identity", refuse)
    with pytest.raises(contract.ServedQuantizerUnboundError, match="operator unavailable"):
        bind_joint_served_quantizer({"q": ["TESSERA_E2M1_K2_R896"]})


def test_a16_does_not_require_vllm_extension(monkeypatch):
    def unwanted(**kwargs):
        pytest.fail("A16 has no served static activation quantizer")
    monkeypatch.setattr(contract, "bind_served_quantizer_identity", unwanted)
    assert bind_joint_served_quantizer({"q": ["BF16", "TESSERA_BF16_K1_R1024"]}) is None


def test_source_build_retains_stage_a_derivative_and_admitted_prefetch(monkeypatch):
    from prismaquant import cost_streaming, model_profiles
    from prismaquant.joint_cost_quantum import build_quantum_source_runner
    prefetch = dict(max_cache_slots=2, prefetch_workers=1, prefetch_lookahead=1,
                    cache_headroom_gb=2, prefetch_min_available_gb=2,
                    require_prefetched_residency=True)
    derivative = {"schema": "prismaquant.glm_source_derivative.v1", "version": "fixture"}
    def build(path, **kwargs):
        assert path == "/source"
        assert kwargs["source_derivative"] == derivative
        assert {k: kwargs[k] for k in prefetch} == prefetch
        assert kwargs["attn_implementation"] == "eager"
        return "built"
    monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm", build)
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _: None)
    config = {"model": "/source", "execution": {"source_derivative": derivative},
              "source_prefetch": prefetch}
    assert build_quantum_source_runner(config, offload_folder="/offload") == "built"
    config.pop("source_prefetch")
    with pytest.raises(ValueError, match="complete source_prefetch"):
        build_quantum_source_runner(config, offload_folder="/offload")
