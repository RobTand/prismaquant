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
