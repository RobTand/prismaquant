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
