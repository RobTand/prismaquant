"""Typed decisions retain the pin, population and caller-owned facts."""
import json

import pytest

from prismaquant import tessera_menu as menu
from prismaquant import tessera_render as render
from prismaquant import tessera_runtime_contract as runtime
from prismaquant.lane_eligibility import ServingContext, UnitStructuralFacts, resolve_unit_route
from prismaquant.tessera_formats import get_tessera_family
from tests.tessera_metadata_fixture import install_metadata


@pytest.fixture
def reference(monkeypatch):
    payload = json.loads(runtime.installed_runtime_snapshot().raw)
    row = next(cell for cell in payload["lane_eligibility"]["cells"]
               if cell["family"] == "TESSERA_E2M1_K2" and cell["structure"] == "dense")
    context = ServingContext(
        platform=row["platform"], structure=row["structure"], residency="resident",
        runtime_image=row["runtime"]["image"], execution_mode="eager")
    snapshot = install_metadata(monkeypatch, payload)
    return payload, snapshot, context


def test_native_verdict_carries_the_admitted_conditions(reference):
    _, _, context = reference
    verdict = render.tessera_lane_admission("TESSERA_E2M1_K2_R896", serving_context=context)
    assert verdict.admitted
    assert verdict.kind == "native"
    assert verdict.reason == ""
    assert verdict.rate_q256 == 896
    assert verdict.route_status == "backed_with_serve_flag"
    assert verdict.requires_serve_flags == ("TESSERA_SERVE_MODE=resident",)
    assert verdict.requires_plugins == ("tessera",)
    assert {cell.regime for cell in verdict.cells} == {"decode", "batch"}


def test_unbound_population_reports_a_context_refusal(reference):
    verdict = render.tessera_lane_admission("TESSERA_E2M1_K2_R896")
    assert verdict.kind == "context_missing"
    assert not verdict.admitted
    assert verdict.route_status == "unattested"
    assert verdict.cells == ()


def test_pin_refusal_keeps_matched_cells_without_admitted_conditions(reference, monkeypatch):
    payload, _, context = reference
    # Retain the declared pin while capturing answer-equivalent changed bytes.
    payload["contract_version"] += 1
    install_metadata(monkeypatch, payload, pinned=False)
    verdict = render.tessera_lane_admission("TESSERA_E2M1_K2_R896", serving_context=context)
    assert verdict.kind == "pin"
    assert not verdict.admitted
    assert verdict.route_status == "unattested"
    assert verdict.requires_serve_flags == verdict.requires_plugins == ()
    assert {cell.regime for cell in verdict.cells} == {"decode", "batch"}


def test_one_menu_expansion_captures_one_runtime_for_routes_and_tp(reference, monkeypatch):
    _, snapshot, context = reference
    calls = []
    def capture():
        calls.append(snapshot.sha256)
        return snapshot
    monkeypatch.setattr(runtime, "installed_runtime_snapshot", capture)
    rows = menu.expand_tessera_menu(
        (2048, 4096), families=(get_tessera_family("TESSERA_E2M1_K2"),),
        step_q256=256, serving_context=context,
        tp_degree=2, parallel_kind=menu.PARALLEL_COLUMN)
    assert [row.format_name for row in rows] == ["TESSERA_E2M1_K2_R896"]
    assert calls == [snapshot.sha256]


def test_returned_predicate_values_cannot_mutate_later_queries(reference, monkeypatch):
    payload, _, context = reference
    for cell in payload["lane_eligibility"]["cells"]:
        if cell["family"] == "TESSERA_E2M1_K2" and cell["structure"] == "dense":
            cell["predicates"] = [{"fact": "in_features", "op": "in", "value": [4096]}]
    snapshot = install_metadata(monkeypatch, payload)
    first = render.tessera_lane_admission("TESSERA_E2M1_K2_R896", serving_context=context)
    first.cells[0].predicates[0][2][0] = 9999
    fresh = render.tessera_lane_admission("TESSERA_E2M1_K2_R896", serving_context=context)
    assert fresh.admitted
    assert fresh.cells[0].predicates[0][2] == [4096]
    facts = UnitStructuralFacts(
        qname="owned.weight", format_name="TESSERA_E2M1_K2_R896",
        payload_family="TESSERA_E2M1_K2", k=None, n_sub=None, rate_q256=896,
        structure="dense", role_split=False, in_features=4096, out_features=2048)
    route = resolve_unit_route(
        facts, snapshot.eligibility, platform=context.platform, residency=context.residency,
        runtime_image=context.runtime_image, execution_mode=context.execution_mode)
    assert route.route_status == "backed_with_serve_flag"


@pytest.mark.parametrize("rate", [None, True, 0, -1])
def test_native_record_refuses_missing_or_invalid_rate(reference, rate):
    from prismaquant.lane_eligibility import NativeRungVerdict
    _, _, context = reference
    cells = render.tessera_lane_admission(
        "TESSERA_E2M1_K2_R896", serving_context=context).cells
    with pytest.raises(ValueError, match="matched cells and a positive integer rate"):
        NativeRungVerdict("native", cells=cells, rate_q256=rate)


def test_native_record_refuses_absent_matched_cells():
    from prismaquant.lane_eligibility import NativeRungVerdict
    with pytest.raises(ValueError, match="matched cells and a positive integer rate"):
        NativeRungVerdict("native", rate_q256=896)
