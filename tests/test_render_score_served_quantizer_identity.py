"""A retained activation-aware cost is only reusable under its own arithmetic.

The render-score key is ``qname|FMT`` and the retained record carries the static
G it was priced at; neither of those changes when the process binds a different
activation quantiser.  A model-priced row and a registered-operator row are
different objects -- the retained 84-group differential priced 24 of 172,032
probed elements one E2M1 code apart -- so the reuse paths must compare the
recorded arithmetic and its build, and refuse the mismatch.

Rows that never touched the quantiser are the exemption, and it is a
mathematical one: a weight-only or dynamically scored row has no static G, so
its cost does not depend on which A-side arithmetic the run bound.

These tests call the reuse validators directly with records in the shape
``_render_score_record`` writes.  Building a served record through the writer
needs a Tessera format, which needs the pinned ``tessera`` package; the
validator is the surface under test and it is reachable without one.
"""
from __future__ import annotations

import pytest

import torch

from prismaquant import nvfp4_activation_contract as owner
from prismaquant import production_weight_cache as pwc
from prismaquant.production_weight_cache import (
    _render_score_record,
    _check_resumed_render_score_policies,
    production_cache_priced_input_global_scales,
)
from prismaquant import format_registry as fr

POLICY = owner.resolve_input_global_scale_policy()
QNAME = "model.language_model.layers.10.mlp.experts::w13"


def _registered_identity(**overrides) -> owner.ServedQuantizerIdentity:
    fields = {"backend": owner.SERVED_QUANTIZER_BACKEND_REGISTERED_OP,
              "op": owner.SERVED_QUANTIZER_OP, "platform": "sm_121",
              "torch": "2.13.0+cu130", "torch_git": "cf30153c4c131c8164ee7798e5022d810682e2cb",
              "vllm": "0.1.dev20073+g8e685d198", "image_content_sha256": "d" * 64,
              "dequant_kernel": owner.SERVED_QUANTIZER_DEQUANT_KERNEL}
    fields.update(overrides)
    return owner.ServedQuantizerIdentity(**fields)


def _model_identity() -> owner.ServedQuantizerIdentity:
    return owner.ServedQuantizerIdentity(
        backend=owner.SERVED_QUANTIZER_BACKEND_MODEL)


def _record(*, identity, max_abs: float = 1.6796875, static: bool = True):
    record = {
        "qname": QNAME, "format": "TESSERA_E2M1_K2_R896",
        "activation_max_abs": max_abs,
        "input_global_scale": (
            owner.input_global_scale_from_max_abs(max_abs, policy=POLICY)
            if static else None),
        "input_global_scale_policy": POLICY if static else None,
    }
    if identity is not None:
        record["served_quantizer"] = identity.as_record()
    return {f"{QNAME}|TESSERA_E2M1_K2_R896": record}


def _bind(identity) -> None:
    owner._reset_served_quantizer_identity_for_tests()
    owner.bind_served_quantizer_identity(identity=identity, require=False)


def test_identical_identity_reuses_the_retained_cost():
    identity = _registered_identity()
    _bind(identity)

    assert _check_resumed_render_score_policies(
        _record(identity=identity), policy=POLICY, where="test") == 1


def test_a_record_predating_the_stamp_cannot_be_reused_as_operator_pricing():
    _bind(_registered_identity())

    with pytest.raises(owner.ServedQuantizerUnboundError, match="no served-quantizer identity"):
        _check_resumed_render_score_policies(
            _record(identity=None), policy=POLICY, where="test")


def test_a_model_priced_row_cannot_be_reused_as_operator_pricing():
    _bind(_registered_identity())

    with pytest.raises(owner.ServedQuantizerUnboundError, match="differing axes are backend"):
        _check_resumed_render_score_policies(
            _record(identity=_model_identity()), policy=POLICY, where="test")


def test_a_different_build_of_the_same_operator_is_a_different_quantiser():
    _bind(_registered_identity())
    other_build = _registered_identity(
        vllm="0.28.1rc1.dev397+gfd4a15126", image_content_sha256="e" * 64)

    with pytest.raises(owner.ServedQuantizerUnboundError,
                       match="differing axes are vllm, image_content_sha256"):
        _check_resumed_render_score_policies(
            _record(identity=other_build), policy=POLICY, where="test")


def test_a_model_bound_run_still_reuses_its_own_rows():
    _bind(_model_identity())

    assert _check_resumed_render_score_policies(
        _record(identity=_model_identity()), policy=POLICY, where="test") == 1


def test_weight_only_rows_are_untouched_by_the_arithmetic_binding():
    """No static G, no activation quantiser in the cost, no refusal."""
    _bind(_registered_identity())

    assert _check_resumed_render_score_policies(
        _record(identity=None, static=False), policy=POLICY, where="test") == 0


def test_the_measurement_side_refuses_the_same_mismatch():
    """The KL hook reads costs through this path; a mismatch refuses there too."""
    class _Cache:
        metadata = {"render_scores": {"records": {}}}

    _bind(_registered_identity())

    with pytest.raises(owner.ServedQuantizerUnboundError, match="differing axes are backend"):
        _Cache.metadata["render_scores"]["records"] = _record(
            identity=_model_identity())
        production_cache_priced_input_global_scales(_Cache(), where="test")

    _Cache.metadata["render_scores"]["records"] = _record(
        identity=_registered_identity())
    assert production_cache_priced_input_global_scales(_Cache(), where="test") == {
        QNAME: owner.input_global_scale_from_max_abs(1.6796875, policy=POLICY)}


def test_the_writer_stamps_the_arithmetic_the_contract_actually_priced_with(monkeypatch):
    """A row's stamp and a row's arithmetic are ONE answer.

    The contract here carries an explicit binding (build B) while the process is
    bound to another (build A).  ``quantize_dequantize`` prices with the
    contract's, so the record must say build B -- a writer that read the process
    binding would publish a row that lies about how it was produced, and a later
    reuse would then accept or refuse it on the wrong grounds.
    """
    from prismaquant import format_registry as fr

    # The contract prices with the Torch model (which a CPU box can run) while
    # the PROCESS is bound to the registered operator: the two disagree, and the
    # record must follow the contract that actually priced the row.
    explicit = _model_identity()
    _bind(_registered_identity())
    contract = owner.StaticActivationContract(
        measured_as_served=True, served_quantizer=explicit)
    monkeypatch.setattr(fr, "canonical_format_name", lambda name: name)
    monkeypatch.setattr(fr, "get_format", lambda name: object())
    monkeypatch.setattr(pwc, "_static_activation_contract_of", lambda spec: contract)

    record = pwc._render_score_record(
        qname=QNAME, fmt="NVFP4", render_format="NVFP4",
        reference_weight=torch.zeros(2, 16, dtype=torch.bfloat16),
        rendered_weight=torch.zeros(2, 16, dtype=torch.bfloat16),
        activations=torch.zeros(1, 16, dtype=torch.bfloat16),
        activation_max_abs=1.0)

    assert record["served_quantizer"]["backend"] == owner.SERVED_QUANTIZER_BACKEND_MODEL
    assert record["served_quantizer"] == explicit.as_record()
    # ... and a run bound to that same arithmetic reuses it.
    _bind(explicit)
    assert _check_resumed_render_score_policies(
        {f"{QNAME}|NVFP4": record}, policy=POLICY, where="test") == 1


def _registry_with(monkeypatch, contract):
    from prismaquant import format_registry as fr

    monkeypatch.setattr(fr, "canonical_format_name", lambda name: name)
    monkeypatch.setattr(fr, "get_format", lambda name: object())
    monkeypatch.setattr(pwc, "_static_activation_contract_of", lambda spec: contract)


@pytest.fixture(autouse=True)
def _the_registry_answers_in_this_module(monkeypatch):
    """Every record here names a format, and the reader now RESOLVES it.

    Scoped to this module on purpose: the fail-closed behaviour below is what
    production sees, and a repository-wide fixture would hide it.
    """
    _registry_with(monkeypatch, None)


def test_a_contract_bound_build_reuses_through_the_real_read(monkeypatch):
    """A correctly stamped contract-bound row must not be refused by the run's
    OWN binding: the reader resolves the contract the row was priced through."""
    explicit = _model_identity()
    _bind(_registered_identity())          # the process prices with something else
    _registry_with(monkeypatch, owner.StaticActivationContract(
        measured_as_served=True, served_quantizer=explicit))
    records = _record(identity=explicit)
    records[f"{QNAME}|TESSERA_E2M1_K2_R896"]["format"] = "NVFP4"

    assert _check_resumed_render_score_policies(
        records, policy=POLICY, where="test") == 1

    class _Cache:
        metadata = {"render_scores": {"records": records}}

    assert production_cache_priced_input_global_scales(_Cache(), where="test") == {
        QNAME: owner.input_global_scale_from_max_abs(1.6796875, policy=POLICY)}


def test_the_real_read_still_rejects_another_build(monkeypatch):
    """The same path, with a contract-bound row stamped for a different build."""
    explicit = _model_identity()
    _bind(_model_identity())
    _registry_with(monkeypatch, owner.StaticActivationContract(
        measured_as_served=True, served_quantizer=explicit))
    records = _record(identity=_registered_identity())
    records[f"{QNAME}|TESSERA_E2M1_K2_R896"]["format"] = "NVFP4"

    with pytest.raises(owner.ServedQuantizerUnboundError, match="differing axes are backend"):
        _check_resumed_render_score_policies(records, policy=POLICY, where="test")


def test_an_unresolvable_format_is_refused_not_reinterpreted(monkeypatch):
    """A missing package or corrupt registry row must fail closed."""
    from prismaquant import format_registry as fr

    _bind(_registered_identity())
    monkeypatch.setattr(
        fr, "get_format",
        lambda name: (_ for _ in ()).throw(KeyError("no such format")))

    with pytest.raises(RuntimeError, match="cannot be checked against the arithmetic"):
        _check_resumed_render_score_policies(
            _record(identity=_registered_identity()), policy=POLICY, where="test")
