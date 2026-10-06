"""The allocator CLI reaches a lane through the allocation protocol (#1558).

Decoupling step 6, part 2b. ``allocator.py`` imports no lane module. What a
lane adds to an allocation -- its flags, serving scope, menu token, cost-table
identity checks, metadata blocks and selection side outputs -- arrives through
the one plugin that provides ``allocation_menu``. ``_StockAllocationLane`` is
the protocol's stock answers and its one written statement.
"""
from __future__ import annotations

import argparse
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from prismaquant import allocator
from prismaquant import lane_spec

ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> list[str]:
    return sorted(name for name, _ in inspect.getmembers(allocator._StockAllocationLane)
                  if not name.startswith("_"))


def test_the_tessera_plugin_is_the_allocation_lane():
    assert allocator._allocation_lane() is lane_spec.lane_plugin("tessera")


def test_the_tessera_plugin_implements_the_whole_protocol():
    plugin = lane_spec.lane_plugin("tessera")
    missing = [name for name in _protocol() if not callable(getattr(plugin, name, None))]
    assert not missing, missing


def test_the_stock_lane_answers_without_a_lane(monkeypatch):
    monkeypatch.setattr(lane_spec, "single_lane_plugin", lambda name: None)
    lane = allocator._allocation_lane()
    assert lane is allocator._StockAllocationLane
    parser = argparse.ArgumentParser()
    lane.allocation_arguments(parser)
    args = parser.parse_args([])
    assert lane.allocation_serving_target(args, target_platform="sm_121") is None
    assert lane.allocation_contexts(None, {}, None) is None
    menu = lane.allocation_menu(["NVFP4", "BF16"], ("NVFP4",), context_by_unit=None)
    assert (menu.formats, menu.widths, menu.refusal_cause) == (["NVFP4", "BF16"], {}, "")
    assert lane.allocation_hessian_identity({}, {}) == {}
    assert lane.allocation_runtime_identity() == {}
    assert lane.allocation_layer_config_meta(menu_report={}) == {}
    assert lane.allocation_selection_meta(menu_report={}) == {}
    assert lane.allocation_selection_request_path(args) is None
    assert lane.allocation_expert_projection({}, {}) == {}
    with pytest.raises(LookupError):
        lane.allocation_unit_context(None, "u", None)
    with pytest.raises(SystemExit, match="no declared allocation lane"):
        lane.allocation_routed_unit_rates(
            {}, {}, cost_data={}, per_linear_legal_formats=None,
            budget_bytes=1, reserve_bytes=0,
            artifact_size_for=lambda assignment: None, canonical_format=str)


def test_the_lane_owns_its_flags():
    parser = argparse.ArgumentParser()
    lane_spec.lane_plugin("tessera").allocation_arguments(parser)
    flags = {action.dest for action in parser._actions}
    assert {"tessera_platform", "tessera_runtime_image", "tessera_execution_mode",
            "tessera_residency", "tessera_materialization_plan"} <= flags


def test_the_selection_meta_keeps_its_stamp_order():
    """selection.json writes these three keys on every run, in this order."""
    meta = lane_spec.lane_plugin("tessera").allocation_selection_meta(
        menu_report={}, menu_report_agg={}, menu_widths={},
        group_menu_report={}, dev_pin={"commit": "x"})
    assert list(meta) == ["tessera_group_knapsack", "tessera_dev_pin", "tessera_menu"]
    assert meta["tessera_menu"] == {"per_linear": {}, "aggregated": {}}


def test_a_stock_layer_config_carries_no_lane_block():
    meta = lane_spec.lane_plugin("tessera").allocation_layer_config_meta(
        menu_report={}, menu_report_agg={}, menu_widths={}, group_menu_report={},
        hessian_identity={}, dev_pin={}, assignment={"u": "NVFP4"},
        cost_data={"costs": {}})
    assert meta == {}


def test_importing_the_allocator_imports_no_lane_module():
    code = textwrap.dedent("""
        import sys
        import prismaquant.allocator
        loaded = sorted(m for m in sys.modules
                        if m == "tessera" or m.startswith(("tessera.", "prismaquant.tessera_")))
        assert not loaded, loaded
    """)
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
