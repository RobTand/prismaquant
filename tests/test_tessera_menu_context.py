"""Explicit serving scope survives menu lookup, provenance and campaign reuse."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from prismaquant import serving_profiles as profiles
from prismaquant import tessera_campaign as campaign
from prismaquant import tessera_menu as menu


def _context(**changed):
    from prismaquant.lane_eligibility import ServingContext

    values = dict(platform="sm_121", structure="dense", residency="resident",
                  runtime_image="fixture/runtime@sha256:" + "1" * 64,
                  execution_mode="eager")
    values.update(changed)
    return ServingContext(**values)


def _contract(monkeypatch, payload, *, context=None):
    import copy
    from prismaquant import tessera_runtime_contract as runtime
    from tests.conftest import set_cell_census

    payload = copy.deepcopy(payload)
    selected = {}
    for cell in payload["lane_eligibility"]["cells"]:
        if cell["family"] == "TESSERA_E4M3_K1" and cell["regime"] not in selected:
            selected[cell["regime"]] = cell
    payload["lane_eligibility"]["cells"] = list(selected.values())
    for cell in selected.values():
        cell["structure"] = "dense"
        cell["route_status"] = "backed_with_serve_flag"
        cell["requires_serve_flags"] = ["TESSERA_SERVE_MODE=resident"]
        set_cell_census(payload, cell, [1024])
        if context is not None:
            cell["runtime"]["image"] = context.runtime_image
            cell["runtime"]["execution_modes"] = [context.execution_mode]
    parsed = runtime._parse(payload, commit="owned-menu-fixture",
                            sha="owned-menu-fixture", path="owned-menu-fixture")
    monkeypatch.setattr(menu, "tessera_runtime_contract", lambda: parsed)


def test_menu_admission_records_its_matching_context(monkeypatch, legacy_v5_contract):
    context = _context()
    _contract(monkeypatch, legacy_v5_contract, context=context)
    admitted = menu.route_admission("TESSERA_E4M3_K1_R1024", serving_context=context)
    assert admitted.attested
    assert admitted.serving_context == context


@pytest.mark.parametrize("context", [None, _context(structure="routed_moe")])
def test_v5_menu_admission_cannot_borrow_another_context(monkeypatch, context, legacy_v5_contract):
    _contract(monkeypatch, legacy_v5_contract, context=_context())
    admitted = menu.route_admission("TESSERA_E4M3_K1_R1024", serving_context=context)
    assert not admitted.attested
    assert admitted.requires_serving_context
    assert "context" in admitted.detail


def test_menu_expansion_preserves_explicit_scope_and_prices_real_geometry(
        monkeypatch, legacy_v5_contract):
    from prismaquant.tessera_formats import get_tessera_family

    context = _context()
    _contract(monkeypatch, legacy_v5_contract, context=context)
    family = get_tessera_family("TESSERA_E4M3_K1")
    rows = menu.expand_tessera_menu(
        (2048, 4096), families=(family,), step_q256=256,
        serving_context=context)
    assert [row.format_name for row in rows] == ["TESSERA_E4M3_K1_R1024"]
    assert rows[0].admission.serving_context == context
    assert rows[0].memory_bytes > 0
    assert menu.expand_tessera_menu(
        (2048, 4096), families=(family,), step_q256=256,
        serving_context=replace(context, structure="routed_moe")) == []


def test_resolved_lane_identity_and_serialization_retain_scope(monkeypatch):
    calls = []

    def admit(name, *, serving_context=None):
        calls.append(serving_context)
        return SimpleNamespace(
            activation_contract="fp8_per_token_dynamic", source="fixture",
            detail="fixture", route_status="backed_with_serve_flag",
            requires_serve_flags=("TESSERA_SERVE_MODE=resident",),
            serving_context=serving_context,
        )

    monkeypatch.setattr(menu, "route_admission", admit)
    first = _context()
    second = replace(first, execution_mode="compiled")
    a = menu.tessera_resolved_serving_lane(
        "TESSERA_E4M3_K1_R1024", serving_context=first)
    b = menu.tessera_resolved_serving_lane(
        "TESSERA_E4M3_K1_R1024", serving_context=second)
    assert calls == [first, second]
    assert a.as_dict()["serving_context"] == first.as_dict()
    assert b.as_dict()["serving_context"] == second.as_dict()
    assert a.route_key() != b.route_key()


def test_profile_tessera_fallback_forwards_explicit_scope(monkeypatch):
    context = _context()
    calls = []
    sentinel = object()
    monkeypatch.setattr(profiles, "load_serving_profile", lambda profile: SimpleNamespace(
        serving_lane_for=lambda *args, **kwargs: None))

    def resolve(name, *, runtime_version="", serving_context=None):
        calls.append(serving_context)
        return sentinel

    monkeypatch.setattr(menu, "tessera_resolved_serving_lane", resolve)
    assert profiles.serving_lane_route(
        "fixture", "TESSERA_E4M3_K1_R1024", serving_context=context) is sentinel
    assert calls == [context]


@pytest.mark.parametrize("different", [
    {"structure": "routed_moe"}, {"residency": "streamed"},
    {"runtime_image": "fixture/runtime@sha256:" + "2" * 64},
    {"execution_mode": "compiled"}, {"platform": "sm_90"},
])
def test_campaign_menu_cache_separates_scope_and_reuses_equal_scope(monkeypatch, different):
    first = _context()
    second = replace(first, **different)
    calls = []

    def expand(shape, **kwargs):
        context = kwargs.get("serving_context")
        calls.append(context)
        return [context]

    monkeypatch.setattr(menu, "expand_tessera_menu", expand)
    weights = {name: SimpleNamespace(shape=(32, 32)) for name in ("a", "b", "c")}
    result = campaign.expand_menus_for_targets(
        weights, list(weights), mode=menu.MENU_ATTESTED, tp_degree=1,
        parallel_kind=menu.PARALLEL_NONE,
        context_by_unit={"a": first, "b": second, "c": replace(first)})
    assert calls == [first, second]
    assert result["a"] is result["c"]
    assert result["a"] is not result["b"]


def test_campaign_partial_context_map_keeps_missing_scope_unbound(monkeypatch):
    first = _context()
    calls = []

    def expand(shape, **kwargs):
        context = kwargs.get("serving_context")
        calls.append(context)
        return [context]

    monkeypatch.setattr(menu, "expand_tessera_menu", expand)
    weights = {name: SimpleNamespace(shape=(32, 32)) for name in ("a", "b")}
    result = campaign.expand_menus_for_targets(
        weights, list(weights), mode=menu.MENU_ATTESTED, tp_degree=1,
        parallel_kind=menu.PARALLEL_NONE, context_by_unit={"a": first})
    assert calls == [first, None]
    assert result["b"] == [None]


def _legacy_contract(monkeypatch, payload):
    _contract(monkeypatch, payload)


@pytest.mark.parametrize("context", [
    None, _context(), _context(execution_mode="compiled"),
    _context(structure="routed_moe"),
])
def test_legacy_menu_cannot_attest_a_new_runtime_context(monkeypatch, context, legacy_v4_contract):
    _legacy_contract(monkeypatch, legacy_v4_contract)

    admission = menu.route_admission(
        "TESSERA_E4M3_K1_R1024", serving_context=context)

    assert admission.attested is (context is None)
    assert admission.admits(menu.MENU_ATTESTED) is (context is None)
    assert admission.admits(menu.MENU_RESEARCH)
    assert admission.serving_context == context
    assert not admission.requires_serving_context
    if context is not None:
        assert "context" in admission.detail
        assert "v4" in admission.detail
        assert admission.requires_serve_flags == ()


def test_legacy_context_refusal_survives_resolved_route_provenance(
        monkeypatch, legacy_v4_contract):
    _legacy_contract(monkeypatch, legacy_v4_contract)
    context = _context(structure="routed_moe", execution_mode="compiled")

    resolved = menu.tessera_resolved_serving_lane(
        "TESSERA_E4M3_K1_R1024", serving_context=context)

    assert resolved.route_status == "unattested"
    assert resolved.as_dict()["route_status"] == "unattested"
    assert resolved.as_dict()["serving_context"] == context.as_dict()
    assert resolved.requires_serve_flags == ()
