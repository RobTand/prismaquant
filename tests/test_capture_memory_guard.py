"""The bounded capture guard holds the CPU cap and the device envelope apart.

The pilot row is 21 GiB of enforced CPU cap (docker ``--memory``), 80 GiB of
device envelope (``max_gpu_bytes``) and a 101 GiB aggregate PrismaBuild
reservation. A guard that adds the cgroup charge to the whole CUDA reservation
and compares the sum with the SMALLEST limit refuses that row at construction:
21 GiB less a 2 GiB margin cannot hold 80 GiB of residency, so the arithmetic
refused every row of this shape before any of it ran.

These tests pin the split, and pin that the un-split behaviour every existing
caller relies on is unchanged.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from prismaquant import memory_management as mm

GiB = 1024**3


def _cgroup(tmp_path: Path, *, cap_bytes: int, current_bytes: int) -> dict:
    """A minimal cgroup v2 tree: one scope with a finite cap and a charge."""
    root = tmp_path / "cgroup"
    scope = root / "scope"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text(str(cap_bytes))
    (scope / "memory.current").write_text(str(current_bytes))
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/scope\n")
    return {"cgroup_root": root, "membership": membership}


@pytest.fixture()
def host(monkeypatch):
    """A 121 GiB box with 100 GiB available, and a chosen CUDA reservation."""
    state = {"reserved": 0, "available": 100 * GiB}

    def reserved(device=None, *args, **kwargs):
        return state["reserved"]

    def host_info():
        return state["available"], 121 * GiB

    monkeypatch.setattr(torch.cuda, "memory_reserved", reserved)
    monkeypatch.setattr(mm, "_host_memory_info", host_info)
    return state


def test_the_pilot_row_is_bounded_rather_than_refused(tmp_path, host):
    """21 GiB CPU + 80 GiB device is bounded; the aggregate is their sum."""
    tree = _cgroup(tmp_path, cap_bytes=21 * GiB, current_bytes=5 * GiB)
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=80 * GiB, **tree)
    host["reserved"] = 40 * GiB
    record = guard.check("before_joint_source_authentication")

    assert guard.cpu_cap_bytes == 21 * GiB
    assert guard.device_bytes == 80 * GiB
    assert record["enforced"] == "split-cpu-device-host"
    assert record["aggregate_envelope_bytes"] == 101 * GiB
    assert record["cpu_refusal_threshold_bytes"] == 19 * GiB
    assert record["device_refusal_threshold_bytes"] == 80 * GiB
    assert record["host_floor_bytes"] >= mm.MIN_HOST_FLOOR_BYTES


def test_the_same_reading_refuses_without_the_split(tmp_path, host):
    """This is the conflation, unchanged for every existing caller."""
    tree = _cgroup(tmp_path, cap_bytes=21 * GiB, current_bytes=5 * GiB)
    guard = mm.CaptureMemoryGuard("cuda", **tree)
    host["reserved"] = 40 * GiB
    with pytest.raises(RuntimeError) as refused:
        guard.check("before_joint_source_authentication")
    assert "capture CPU memory refusal" in str(refused.value)
    assert guard.device_bytes is None


def test_the_cpu_cap_still_refuses_on_its_own_budget(tmp_path, host):
    tree = _cgroup(tmp_path, cap_bytes=21 * GiB, current_bytes=20 * GiB)
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=80 * GiB, **tree)
    with pytest.raises(RuntimeError) as refused:
        guard.check("cpu side")
    assert "capture CPU memory refusal" in str(refused.value)
    assert "2147483648-byte margin" in str(refused.value)


def test_the_device_envelope_refuses_on_its_own_budget(tmp_path, host):
    tree = _cgroup(tmp_path, cap_bytes=21 * GiB, current_bytes=1 * GiB)
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=80 * GiB, **tree)
    host["reserved"] = 81 * GiB
    with pytest.raises(RuntimeError) as refused:
        guard.check("device side")
    assert "capture device memory refusal" in str(refused.value)


def test_a_future_device_allocation_is_charged_to_the_envelope(tmp_path, host):
    tree = _cgroup(tmp_path, cap_bytes=21 * GiB, current_bytes=1 * GiB)
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=80 * GiB, **tree)
    host["reserved"] = 70 * GiB
    guard.check("headroom", reserve_device_bytes=9 * GiB)
    with pytest.raises(RuntimeError) as refused:
        guard.check("overshoot", reserve_device_bytes=11 * GiB)
    assert "capture device memory refusal" in str(refused.value)


def test_the_host_floor_is_held_beside_both_budgets(tmp_path, host):
    tree = _cgroup(tmp_path, cap_bytes=21 * GiB, current_bytes=1 * GiB)
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=80 * GiB, **tree)
    host["available"] = mm.MIN_HOST_FLOOR_BYTES - 1
    with pytest.raises(RuntimeError) as refused:
        guard.check("host floor")
    assert "capture physical memory refusal" in str(refused.value)


def test_a_floor_below_the_minimum_is_refused_at_construction(tmp_path):
    tree = _cgroup(tmp_path, cap_bytes=21 * GiB, current_bytes=1 * GiB)
    with pytest.raises(RuntimeError) as refused:
        mm.CaptureMemoryGuard("cuda", device_bytes=80 * GiB,
                              host_floor_bytes=1 * GiB, **tree)
    assert "host floor must be at least" in str(refused.value)


def test_a_device_reservation_without_an_envelope_refuses(tmp_path, host):
    """The budgets are not interchangeable; an unstated one is not inferred."""
    tree = _cgroup(tmp_path, cap_bytes=21 * GiB, current_bytes=1 * GiB)
    guard = mm.CaptureMemoryGuard("cuda", **tree)
    with pytest.raises(ValueError) as refused:
        guard.check("device without envelope", reserve_device_bytes=1 * GiB)
    assert "needs a declared device envelope" in str(refused.value)


def test_enforce_device_envelope_sets_the_allocator_fraction(monkeypatch):
    """``max_gpu_bytes`` becomes a bound, not a comparison after the fact."""
    seen = {}

    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda device: type("P", (), {"total_memory": 121 * GiB})())
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction",
        lambda fraction, device=None: seen.update(fraction=fraction, device=device))

    record = mm.enforce_device_envelope("cuda", 80 * GiB)
    assert record["enforced"] is True
    assert record["device_total_bytes"] == 121 * GiB
    assert record["device_envelope_bytes"] == 80 * GiB
    assert seen["fraction"] == pytest.approx(80 / 121)


def test_enforce_device_envelope_refuses_a_budget_that_bounds_nothing(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda device: type("P", (), {"total_memory": 121 * GiB})())
    for bad in (0, -1, None, True, 121 * GiB, 200 * GiB):
        with pytest.raises(RuntimeError):
            mm.enforce_device_envelope("cuda", bad)
    assert mm.enforce_device_envelope("cpu", 80 * GiB)["enforced"] is False
