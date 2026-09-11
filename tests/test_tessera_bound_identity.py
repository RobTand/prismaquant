"""Actual producer identities remain exact across a bounded immutable unit."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


def fixture(*, projected=False):
    from prismaquant import tessera_campaign as tc, tessera_hessian as th
    from prismaquant.tessera_formats import parse_tessera_format_name
    name = "model.layers.0.proj"
    weight = torch.arange(32 * 256, dtype=torch.float32).reshape(32, 256).to(torch.bfloat16) / 1024
    hessian = torch.eye(256)
    source = th.activation_source({name: hessian}, th.calibration_identity(
        "fixture", [torch.arange(16).reshape(1, 16)], fit_tokens=16))
    formats = [f"TESSERA_{family}_K1_R{rung}" for family in ("BF16", "E4M3")
               for rung in (896, 1024, 1152)] + ["TESSERA_E2M1_K2_R896"]
    anchors = []
    for fmt in formats:
        family, rung = parse_tessera_format_name(fmt)
        anchors.append(tc.CampaignAnchor(qname=name, format_name=fmt, family=family.name,
            body_rate_q256=rung, dloss=0.1, dloss_stderr=0.0, memory_bytes=8192,
            bits_per_param=4.0, activation_contract="fixture", activation_quantized=True,
            wire_bytes=8192, seconds=0.1, hessian_applied=True,
            input_global_scale=0.125 if "E2M1" in fmt else None))
    projection = None if not projected else dict(tensor=name + ".weight", source_tensor="packed.weight",
        source_layout="packed", source_slice={"expert": 0}, expert=0,
        projection="gate_proj", group="w13", rows=32, cols=256)
    kwargs = dict(weights={name: weight}, menus={name: [SimpleNamespace(format_name=f) for f in formats]},
        calibration_source=source, static_scales={name: 0.125},
        projected_units={} if projection is None else {name: projection})
    return tc, name, weight, hessian, source, anchors, projection, kwargs


@pytest.mark.parametrize("projected", [False, True])
def test_bound_unit_reuses_actual_source_h_hashes_with_identical_producer_identity(monkeypatch, projected):
    from tessera import cached_unit
    tc, name, weight, hessian, source, anchors, projection, kwargs = fixture(projected=projected)
    expected = [tc._checkpoint_anchor_identity(anchor, **kwargs) for anchor in anchors]
    actual_hash = cached_unit.tensor_identity
    calls = []
    def observed(value):
        calls.append(id(value))
        return actual_hash(value)
    monkeypatch.setattr(cached_unit, "tensor_identity", observed)
    with tc.bind_checkpoint_unit_identity(anchors, source_weight=weight,
            calibration_source=source, projected_unit=projection, static_scales=kwargs["static_scales"]) as bound:
        actual = [tc._checkpoint_anchor_identity(anchor, **kwargs, bound_unit=bound) for anchor in anchors]
        assert actual == expected
        assert calls.count(id(weight)) == 1
        assert calls.count(id(hessian)) == 1
        actual[0]["source"]["sha256"] = "0" * 64
        assert tc._checkpoint_anchor_identity(anchors[0], **kwargs, bound_unit=bound) == expected[0]
    with pytest.raises(ValueError, match="closed"):
        tc._checkpoint_anchor_identity(anchors[0], **kwargs, bound_unit=bound)


@pytest.mark.parametrize("change", ["source_values", "source_storage", "source_view", "h_values", "h_replaced", "settings", "provenance", "projection"])
def test_bound_unit_refuses_lifetime_mutation(change):
    tc, name, weight, hessian, source, anchors, projection, kwargs = fixture(projected=True)
    bound = tc.bind_checkpoint_unit_identity(anchors, source_weight=weight,
        calibration_source=source, projected_unit=projection, static_scales=kwargs["static_scales"])
    if change == "source_values":
        weight[0, 0] += 1
    elif change == "source_storage":
        weight.data = weight.clone()
    elif change == "source_view":
        kwargs["weights"][name] = weight.t().contiguous().t()
    elif change == "h_values":
        hessian[0, 0] += 1
    elif change == "h_replaced":
        source.hessians[name] = hessian.clone()
    elif change == "settings":
        object.__setattr__(source, "ldlq_sigma", source.ldlq_sigma + 1)
    elif change == "provenance":
        source.provenance["fit_tokens"] += 1
    else:
        projection["source_slice"]["expert"] += 1
    from tessera.errors import GrammarError
    with pytest.raises((ValueError, RuntimeError, GrammarError), match="changed|moved|differs|bound"):
        tc._checkpoint_anchor_identity(anchors[0], **kwargs, bound_unit=bound)


def test_bound_unit_keeps_per_anchor_static_scale_and_hessian_gates():
    tc, name, weight, hessian, source, anchors, projection, kwargs = fixture()
    with tc.bind_checkpoint_unit_identity(anchors, source_weight=weight,
            calibration_source=source, projected_unit=projection, static_scales=kwargs["static_scales"]) as bound:
        with pytest.raises(RuntimeError, match="Hessian applicability"):
            tc._checkpoint_anchor_identity(replace(anchors[0], hessian_applied=False), **kwargs, bound_unit=bound)
        with pytest.raises(tc.ActivationScaleContractError):
            tc._checkpoint_anchor_identity(replace(anchors[-1], input_global_scale=0.25), **kwargs, bound_unit=bound)


def test_bound_unit_refuses_nonfinite_source():
    tc, name, weight, hessian, source, anchors, projection, kwargs = fixture()
    weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        tc.bind_checkpoint_unit_identity(anchors, source_weight=weight,
            calibration_source=source, projected_unit=projection, static_scales=kwargs["static_scales"])


def test_campaign_hold_reuses_exact_run_receipts_and_survives_equivalent_owner_handoff(monkeypatch):
    """Campaign startup owns one producer source/H receipt per unit."""
    from tessera import cached_unit
    tc, name, weight, hessian, source, anchors, projection, kwargs = fixture(projected=True)
    args = SimpleNamespace(family_restriction=None)
    common = dict(weights=kwargs["weights"], acts={name: torch.ones(2, 2)},
        hessians={name: hessian}, menus=kwargs["menus"], args=args,
        calibration_identity=source.provenance, serving_scope=None,
        static_scales=kwargs["static_scales"], static_scale_policy="fixture",
        expert_projection=None)
    expected = tc._campaign_checkpoint_identity(**common)
    actual_hash = cached_unit.tensor_identity
    calls = []
    def observed(value):
        calls.append(id(value))
        return actual_hash(value)
    monkeypatch.setattr(cached_unit, "tensor_identity", observed)
    metadata_bounds, scratch = tc._campaign_identity_metadata_plan(
        weights=kwargs["weights"], menus=kwargs["menus"], calibration_source=source,
        projected_units={name: projection}, static_scales=kwargs["static_scales"])
    assert scratch > 0
    bound = tc._campaign_bound_identities(weights=kwargs["weights"], menus=kwargs["menus"],
        calibration_source=source, projected_units={name: projection},
        static_scales=kwargs["static_scales"], metadata_bounds=metadata_bounds)
    actual = tc._campaign_checkpoint_identity(**common, bound_units=bound)
    assert actual == expected
    # Exact identity parity is the same-campaign resume/merge compatibility
    # predicate; source identity is intentionally still a normal identity field.
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    assert canonical_json_sha256(actual, where="bound") == canonical_json_sha256(
        expected, where="legacy")
    assert calls.count(id(weight)) == 1
    # Tessera's one producer receipt seals H for capture provenance and then
    # stamps its per-unit H identity. Both are producer-owned calls, once at
    # startup rather than once for every published anchor.
    assert calls.count(id(hessian)) == 2
    assert bound[name].observed_metadata_bytes() > 0
    replacement = tc.th.activation_source({name: hessian}, source.provenance)
    bound[name].replace_calibration_source(replacement)
    replacement_kwargs = {**kwargs, "calibration_source": replacement}
    assert tc._checkpoint_anchor_identity(anchors[0], **replacement_kwargs,
        bound_unit=bound[name]) == tc._checkpoint_anchor_identity(
            anchors[0], **replacement_kwargs)
    bound[name].close()


def test_h_free_hold_accepts_later_shared_reference_owner(monkeypatch):
    tc, name, weight, hessian, source, anchors, _, kwargs = fixture()
    h_free = [replace(anchor, hessian_applied=False) for anchor in anchors]
    # This isolates the holder's lifetime branch: real H-free producer wires
    # have the same `calibration: None` receipt, while grammar applicability is
    # separately exercised by the producer integration suite.
    monkeypatch.setattr(tc, "_checkpoint_anchor_identity", lambda *a, **k: {
        "source": {"sha256": "a" * 64}, "calibration": None, "recipe": {}})
    bound = tc.bind_checkpoint_unit_identity(h_free, source_weight=weight,
        calibration_source=source, projected_unit=None,
        static_scales=kwargs["static_scales"], retain_source_receipt=False)
    replacement = tc.th.activation_source({name: hessian}, source.provenance)
    bound.replace_calibration_source(replacement)
    bound.close()


def test_campaign_identity_hold_closes_during_error_unwind():
    from contextlib import ExitStack
    tc, name, weight, _, source, anchors, projection, kwargs = fixture(projected=True)
    bound = tc.bind_checkpoint_unit_identity(anchors, source_weight=weight,
        calibration_source=source, projected_unit=projection,
        static_scales=kwargs["static_scales"], retain_source_receipt=False)
    with pytest.raises(RuntimeError, match="publisher failed"):
        with ExitStack() as scope:
            scope.callback(bound.close)
            raise RuntimeError("publisher failed")
    with pytest.raises(ValueError, match="closed"):
        bound.campaign_inputs()


def test_a_bound_identity_is_derived_on_the_writer_without_a_tensor_read(monkeypatch):
    """The deferred derivation is device-free, so it may run on the writer.

    The producer's graph-capture contract forbids surprise device work from
    another thread while a capture may be running, so the writer is CPU/IO
    only.  Binding hashes the weight and Hessian once, on the calling thread;
    every derivation after that is a template copy plus Tessera's recipe and
    reads no tensor on any thread.  The receipts it seals are the ones the
    inline path computes.
    """
    import threading
    from tessera import cached_unit
    from prismaquant.tessera_publication import BoundedPublisher
    tc, name, weight, hessian, source, anchors, projection, kwargs = fixture()
    expected = [tc._checkpoint_anchor_identity(anchor, **kwargs) for anchor in anchors]
    actual_hash = cached_unit.tensor_identity
    hashed_on = []

    def observed(value):
        hashed_on.append(threading.current_thread().name)
        return actual_hash(value)

    monkeypatch.setattr(cached_unit, "tensor_identity", observed)
    with tc.bind_checkpoint_unit_identity(anchors, source_weight=weight,
            calibration_source=source, projected_unit=projection,
            static_scales=kwargs["static_scales"]) as bound:
        assert hashed_on and set(hashed_on) == {threading.current_thread().name}
        hashed_on.clear()
        journalled, derived_on = [], []
        pub = BoundedPublisher(budget_bytes=1 << 20)
        ledger = tc._AnchorPublicationLedger(publisher=pub,
            make_record=lambda anchor, identity: identity,
            journal_anchor=lambda anchor, record: journalled.append(record))
        try:
            for anchor in anchors:
                def derive(anchor=anchor):
                    derived_on.append(threading.current_thread().name)
                    return tc._checkpoint_anchor_identity(anchor, **kwargs, bound_unit=bound)
                ledger.record(anchor, derive=derive)
            assert ledger.drain() == len(anchors)
        finally:
            pub.close()
        assert journalled == expected
        assert derived_on == ["tessera-publication"] * len(anchors)
        assert hashed_on == [], "a deferred derivation read a tensor"

