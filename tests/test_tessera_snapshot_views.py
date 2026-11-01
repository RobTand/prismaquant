"""A query cannot change later admission by editing returned metadata."""
from pathlib import Path

import pytest

from prismaquant import tessera_menu as menu
from prismaquant import tessera_runtime_contract as runtime


@pytest.fixture
def captured(tmp_path):
    source = runtime.installed_runtime_snapshot()
    path = tmp_path / "runtime_contract.json"
    path.write_bytes(source.raw)
    return runtime.read_runtime_snapshot(path)


def test_lane_and_platform_views_do_not_change_later_facts(captured):
    table = captured.eligibility
    lane = next(row for row in table.lanes if row.requires)
    expected_arity = lane.requires["grid_arities"]
    lane.requires["grid_arities"] = (99,)
    platform = next(iter(table.platform_entries.values()))
    family = next(name for name, route in platform.executes.items() if route)
    expected_route = platform.executes[family]
    platform.executes[family] = None

    fresh = captured.eligibility
    fresh_lane = next(row for row in fresh.lanes if row.extension == lane.extension)
    assert fresh_lane.requires["grid_arities"] == expected_arity
    assert fresh.platform_entries[platform.key].executes[family] == expected_route


def test_tp_views_do_not_change_later_limits_or_axes(captured):
    limits, axes = captured.tensor_parallel
    family = next(name for name, values in axes.items() if values)
    axis = next(iter(axes[family]))
    expected_limit = limits[family]
    expected_axis = axes[family][axis]
    limits[family] = 0
    axes[family][axis] = "unpublished"

    fresh_limits, fresh_axes = captured.tensor_parallel
    assert fresh_limits[family] == expected_limit
    assert fresh_axes[family][axis] == expected_axis


def test_declared_contract_views_do_not_change_the_reader(captured):
    contract = runtime.read_tessera_contract(captured.path, captured.sha256, "fixture")
    family = next(iter(contract.reader_rate_range))
    expected_range = contract.reader_rate_range[family]
    expected_limit = contract.max_world_size[family]
    contract.reader_rate_range.pop(family)
    contract.max_world_size[family] = 0

    fresh = runtime.read_tessera_contract(captured.path, captured.sha256, "fixture")
    assert fresh.reader_rate_range[family] == expected_range
    assert fresh.max_world_size[family] == expected_limit


def test_file_tp_views_do_not_change_later_queries(captured):
    limits = runtime.published_tensor_parallel_limits(captured.path, captured.sha256)
    axes = runtime.published_tensor_parallel_axes(captured.path, captured.sha256)
    family = next(name for name, values in axes.items() if values)
    axis = next(iter(axes[family]))
    expected_limit = limits[family]
    expected_axis = axes[family][axis]
    limits[family] = 0
    axes[family][axis] = "unpublished"

    assert runtime.published_tensor_parallel_limits(captured.path, captured.sha256)[family] == expected_limit
    assert runtime.published_tensor_parallel_axes(captured.path, captured.sha256)[family][axis] == expected_axis


def test_editing_returned_limits_cannot_admit_an_unpublished_tp_degree(captured, monkeypatch):
    monkeypatch.delenv(runtime.TESSERA_DEV_PIN_ENV, raising=False)
    monkeypatch.setattr(runtime, "contract_path", lambda: Path(captured.path))
    limits = runtime.pinned_tensor_parallel_limits()
    family = next(iter(limits))
    unpublished = limits[family] + 1
    limits[family] = unpublished

    assert menu.tessera_tp_world_attested(family, unpublished)[0] is False
