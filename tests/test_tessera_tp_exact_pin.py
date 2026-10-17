"""The serving pin supplies TP facts independently of the optional dev pin."""
import json

import pytest

from prismaquant import tessera_menu as menu
from prismaquant import tessera_runtime_contract as runtime
from prismaquant.lane_eligibility import ServingContext
from prismaquant.tessera_serving_runtime_pin import require_pinned_tessera_runtime


@pytest.fixture
def pinned_payload(monkeypatch):
    monkeypatch.delenv(runtime.TESSERA_DEV_PIN_ENV, raising=False)
    require_pinned_tessera_runtime()
    return json.loads(runtime.contract_path().read_bytes())


@pytest.mark.parametrize("family", [
    "TESSERA_E2M1_K2", "TESSERA_E4M3_K1", "TESSERA_BF16_K1",
])
def test_exact_serving_pin_attests_only_the_published_tp_ceiling(pinned_payload, family):
    unit = next(row for row in pinned_payload["tensor_parallel"]["units"]
                if row["unit"] == family)
    ceiling = unit["max_world_size"]
    assert ceiling > 1

    assert menu.tessera_tp_world_attested(family, ceiling) == (True, "")
    attested, reason = menu.tessera_tp_world_attested(family, ceiling + 1)
    assert not attested
    assert f"world size {ceiling}" in reason


def test_tp2_menu_keeps_only_the_same_scoped_native_rungs(pinned_payload):
    cell = next(row for row in pinned_payload["lane_eligibility"]["cells"]
                if row["family"] == "TESSERA_E2M1_K2" and row["structure"] == "dense")
    context = ServingContext(
        platform=cell["platform"], structure=cell["structure"],
        residency="resident", runtime_image=cell["runtime"]["image"],
        execution_mode="eager")
    shape = (2048, 1024)

    whole = menu.expand_tessera_menu(
        shape, mode=menu.MENU_ATTESTED, serving_context=context)
    sharded = menu.expand_tessera_menu(
        shape, mode=menu.MENU_ATTESTED, tp_degree=2,
        parallel_kind=menu.PARALLEL_COLUMN, serving_context=context)

    assert whole
    assert [row.format_name for row in sharded] == [row.format_name for row in whole]
    assert all(row.admission.attested for row in sharded)
    assert menu.expand_tessera_menu(
        shape, mode=menu.MENU_ATTESTED, tp_degree=2,
        parallel_kind=menu.PARALLEL_COLUMN) == []


def test_tp_fallback_refuses_answer_equivalent_unpinned_bytes(
        pinned_payload, monkeypatch, tmp_path):
    changed = {**pinned_payload,
               "contract_version": pinned_payload["contract_version"] + 1}
    path = tmp_path / "runtime_contract.json"
    path.write_text(json.dumps(changed))
    monkeypatch.setattr(runtime, "contract_path", lambda: path)

    attested, reason = menu.tessera_tp_world_attested("TESSERA_E4M3_K1", 2)

    assert not attested
    assert "pinned" in reason


def test_tp_fallback_remains_closed_world_for_an_unpublished_family(pinned_payload):
    attested, reason = menu.tessera_tp_world_attested("TESSERA_LM8_K1", 2)
    assert not attested
    assert "does not list" in reason


# ---------------------------------------------------------------------------
# The loader-axis leg reads the same exact-pinned bytes as the world leg
# ---------------------------------------------------------------------------

E2M1 = "TESSERA_E2M1_K2"
SHAPE = (2048, 1024)


def _refusing_row_axis(payload):
    """The pinned document with one published ``refused`` axis, nothing else moved."""
    from copy import deepcopy

    changed = deepcopy(payload)
    unit = next(row for row in changed["tensor_parallel"]["units"]
                if row["unit"] == E2M1)
    unit["loader_axes"]["row"]["status"] = runtime.TP_LOADER_AXIS_REFUSED
    return changed


def _dense_e2m1_context(payload):
    cell = next(row for row in payload["lane_eligibility"]["cells"]
                if row["family"] == E2M1 and row["structure"] == "dense")
    return ServingContext(
        platform=cell["platform"], structure=cell["structure"],
        residency="resident", runtime_image=cell["runtime"]["image"],
        execution_mode="eager")


def _families(rows):
    return {row.family for row in rows}


@pytest.fixture
def refused_row_axis(pinned_payload, monkeypatch):
    from tests.tessera_metadata_fixture import install_metadata

    install_metadata(monkeypatch, _refusing_row_axis(pinned_payload))
    return _dense_e2m1_context(pinned_payload)


def test_exact_pin_publishes_the_refused_axis_to_direct_callers(refused_row_axis):
    axes = menu.tessera_loader_axes()
    assert axes is not None and axes[E2M1]["row"] == runtime.TP_LOADER_AXIS_REFUSED

    loads, reason = menu.tessera_tp_axis_legal(
        E2M1, tp_degree=2, parallel_kind=menu.PARALLEL_COLUMN)
    assert not loads
    assert reason == f"tp_axis_refused:{E2M1}:{E2M1}:row"


@pytest.mark.parametrize("require_attested_world", [False, True])
def test_exact_pin_refused_axis_refuses_a_tp2_column_cut(
        refused_row_axis, require_attested_world):
    rung = next(row for row in menu.expand_tessera_menu(
        SHAPE, mode=menu.MENU_ATTESTED, serving_context=refused_row_axis)
        if row.family == E2M1).body_rate_q256

    legal, reason = menu.tessera_tp_legal(
        E2M1, rung, SHAPE, tp_degree=2, parallel_kind=menu.PARALLEL_COLUMN,
        require_attested_world=require_attested_world)
    assert not legal
    assert reason == f"tp_axis_refused:{E2M1}:{E2M1}:row"


def test_exact_pin_tp2_menu_drops_only_the_refused_cut(refused_row_axis):
    whole = menu.expand_tessera_menu(
        SHAPE, mode=menu.MENU_ATTESTED, serving_context=refused_row_axis)
    column = menu.expand_tessera_menu(
        SHAPE, mode=menu.MENU_ATTESTED, tp_degree=2,
        parallel_kind=menu.PARALLEL_COLUMN, serving_context=refused_row_axis)
    row = menu.expand_tessera_menu(
        SHAPE, mode=menu.MENU_ATTESTED, tp_degree=2,
        parallel_kind=menu.PARALLEL_ROW, serving_context=refused_row_axis)

    assert E2M1 in _families(whole)
    assert [r.format_name for r in column] == [
        r.format_name for r in whole if r.family != E2M1]
    assert [r.format_name for r in row] == [r.format_name for r in whole]


@pytest.mark.parametrize("tp_degree,parallel_kind", [
    (1, menu.PARALLEL_COLUMN), (2, menu.PARALLEL_NONE),
])
def test_exact_pin_refused_axis_is_not_asked_without_a_cut(
        refused_row_axis, tp_degree, parallel_kind):
    whole = menu.expand_tessera_menu(
        SHAPE, mode=menu.MENU_ATTESTED, serving_context=refused_row_axis)
    controlled = menu.expand_tessera_menu(
        SHAPE, mode=menu.MENU_ATTESTED, tp_degree=tp_degree,
        parallel_kind=parallel_kind, serving_context=refused_row_axis)
    assert [r.format_name for r in controlled] == [r.format_name for r in whole]


def test_menu_expansion_reads_tp_facts_from_its_one_capture(
        refused_row_axis, monkeypatch):
    captured = runtime.installed_runtime_snapshot
    reads = []

    def counted():
        reads.append(1)
        return captured()

    monkeypatch.setattr(runtime, "installed_runtime_snapshot", counted)
    menu.expand_tessera_menu(
        SHAPE, mode=menu.MENU_ATTESTED, tp_degree=2,
        parallel_kind=menu.PARALLEL_COLUMN, serving_context=refused_row_axis)
    assert len(reads) == 1


def test_unpinned_bytes_publish_no_loader_fact(pinned_payload, monkeypatch):
    from tests.tessera_metadata_fixture import install_metadata

    install_metadata(monkeypatch, _refusing_row_axis(pinned_payload), pinned=False)
    assert menu.tessera_loader_axes() is None
    assert menu.tessera_tp_axis_legal(
        E2M1, tp_degree=2, parallel_kind=menu.PARALLEL_COLUMN) == (True, "")
    attested, reason = menu.tessera_tp_world_attested(E2M1, 2)
    assert not attested and reason.startswith("tp_unattested:")
