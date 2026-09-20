"""Driver plan tests: deterministic, supported interfaces only, hosts explicit."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / "tools" / "fleet_qualification_driver.py"


def _driver():
    spec = importlib.util.spec_from_file_location(
        "fleet_qualification_driver", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pins():
    return {"pb_candidate": {"rev": "a" * 40}, "pq": {"head": "b" * 40},
            "published_generation": {"generation": "c"}}


def test_plan_is_deterministic_for_declared_pins():
    """Same pins in, same plan out: byte-identical JSON."""
    driver = _driver()
    first = driver.build_plan(pins=_pins())
    second = driver.build_plan(pins=_pins())
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["pins"] == _pins()


def test_every_row_uses_a_supported_entrypoint():
    """Rows submit through pbrun/pbtest/pbcampaign only -- no scheduler."""
    driver = _driver()
    plan = driver.build_plan(pins=_pins())
    assert plan["schema"] == "prismaquant.fleet_qualification.plan.v1"
    for stage in plan["stages"]:
        for row in stage["rows"]:
            assert row["entrypoint"] in driver.ENTRYPOINTS, row
            assert row["host"] in driver.HOSTS, row


def test_hosts_are_explicit_per_row_with_no_layer_sharding():
    """Host coverage is the purpose: each row names its box, never a shard map."""
    driver = _driver()
    plan = driver.build_plan(pins=_pins())
    seen = {row["host"] for stage in plan["stages"] for row in stage["rows"]}
    assert seen == set(driver.HOSTS)
    blob = json.dumps(plan)
    assert "shard" not in blob and "layers" not in blob


def test_membership_rows_carry_their_requirement_and_emit_nothing():
    """JOIN/RESIGN/handoff plan rows require the unpublished API; no mutation."""
    driver = _driver()
    plan = driver.build_plan(pins=_pins())
    membership = next(stage for stage in plan["stages"]
                      if "membership" in stage["name"])
    assert membership["rows"]
    for row in membership["rows"]:
        assert "PB728" in row["requires"], row
        assert "JOIN" in row["requires"] and "RESIGN" in row["requires"]
    lines = driver.format_invocations(plan)
    assert any("membership" in line for line in lines)
    assert not any("fleet_qualification_driver" in line for line in lines)


def test_print_plan_lists_supported_invocations_only(tmp_path, capsys):
    """--print-plan emits exact invocations; --submit is not exercised here."""
    driver = _driver()
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps(_pins()))
    assert driver.main(["--pins", str(pins_path), "--print-plan"]) == 0
    out = capsys.readouterr().out
    assert "tools/pbtest.py" in out and "tools/pbrun.py" in out
    assert "membership-join-resign-handoff-plan" in out
