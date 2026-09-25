"""The capture guard's host floor is the box watchdog's, above the hang (PQ #1158).

On 2026-09-14 a vLLM A8 routed-expert load (tessera#501) hung both Sparks at
8.5 GiB ``MemAvailable``, memory PSI full 89. The box watchdog set after that
hang acts below 16 GiB ``MemAvailable`` on each GB10. The guard's default host
floor was 8 GiB, under the hang itself: it admitted an allocation that left the
box in the range the hang happened in, and the only thing between a row and the
hang was the watchdog, acting from outside the row with no refusal receipt.

The incident numbers are pinned here, not read from the module, so a change to
the module's constant cannot move the evidence it is checked against.

The fit tests put the rows the floor governs on a GB10 as it was measured with
nothing running on it: sparklina, 2026-09-07 04:59:15Z, 0 broker jobs,
``MemAvailable`` 115.638 GiB of a 121.627 GiB ``MemTotal``
(PrismaBuild ``docs/gb10_memory_104_capacity_2026-09-07.md``). So the box keeps
about 5.99 GiB for itself, and a row can hold at most 115.638 - 16 = 99.638 GiB
before its guard refuses. On GB10 the device allocates out of host memory, so a
row's cgroup charge and its CUDA reservation both come out of ``MemAvailable``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from prismaquant import memory_management as mm

GiB = 1024**3

#: ``MemAvailable`` when the 2026-09-14 A8 load hung both Sparks (PQ #1158).
HANG_MEM_AVAILABLE_BYTES = int(8.5 * GiB)
#: The GB10 box watchdog's ``MemAvailable`` floor, set after that hang (Tessera
#: ``docs/measurements/tessera-glm53-a4-stub-tp2-served-2026-09-14.md``).
WATCHDOG_FLOOR_BYTES = 16 * GiB
#: An idle GB10 (see the module docstring).
IDLE_MEM_AVAILABLE_BYTES = int(115.638 * GiB)
MEM_TOTAL_BYTES = int(121.627 * GiB)


def _cgroup(tmp_path: Path, *, cap_bytes: int, current_bytes: int) -> dict:
    root = tmp_path / "cgroup"
    scope = root / "scope"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text(str(cap_bytes))
    (scope / "memory.current").write_text(str(current_bytes))
    # No page cache: the committed bytes are memory.current (PQ #1157).
    (scope / "memory.stat").write_text("anon 0\nfile 0\nshmem 0\nfile_dirty 0\nfile_writeback 0\n")
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/scope\n")
    return {"cgroup_root": root, "membership": membership}


@pytest.fixture()
def host(monkeypatch):
    state = {"reserved": 0, "available": 100 * GiB}
    monkeypatch.setattr(torch.cuda, "memory_reserved",
                        lambda device=None, *a, **k: state["reserved"])
    monkeypatch.setattr(mm, "_host_memory_info",
                        lambda: (state["available"], MEM_TOTAL_BYTES))
    return state


#: The three guard shapes the default reaches, each at a campaign's own caps:
#: the un-split legacy guard, the split qualification guard of the Stage A
#: pilot row (21 GiB cgroup cap + 80 GiB device envelope), and the aggregate
#: Stage B / COST guard (the R13 policy: 28 GiB + 66 GiB).
SHAPES = {
    "unsplit": dict(cap_bytes=40 * GiB, kwargs={}),
    "split": dict(cap_bytes=21 * GiB, kwargs={"device_bytes": 80 * GiB}),
    "aggregate": dict(cap_bytes=28 * GiB, kwargs={"device_bytes": 70866960384,
                                                   "aggregate_envelope": True}),
}


def _guard(tmp_path, shape, *, current_bytes=GiB):
    spec = SHAPES[shape]
    return mm.CaptureMemoryGuard("cuda", **spec["kwargs"], **_cgroup(
        tmp_path, cap_bytes=spec["cap_bytes"], current_bytes=current_bytes))


def test_the_default_floor_is_above_the_hang():
    assert mm.DEFAULT_HOST_FLOOR_BYTES > HANG_MEM_AVAILABLE_BYTES


def test_the_default_floor_is_the_box_watchdogs():
    """One number: the guard refuses where the watchdog would act, not below it."""
    assert mm.DEFAULT_HOST_FLOOR_BYTES == WATCHDOG_FLOOR_BYTES
    assert WATCHDOG_FLOOR_BYTES > HANG_MEM_AVAILABLE_BYTES
    # The module names the evidence it restates, and names it as measured.
    assert mm.BOX_WATCHDOG_FLOOR_BYTES == WATCHDOG_FLOOR_BYTES
    assert mm.HANG_MEM_AVAILABLE_BYTES == HANG_MEM_AVAILABLE_BYTES


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_every_guard_shape_holds_the_default_floor(tmp_path, host, shape):
    guard = _guard(tmp_path, shape)
    assert guard.host_floor_bytes == mm.DEFAULT_HOST_FLOOR_BYTES
    assert guard.snapshot()["host_floor_bytes"] == WATCHDOG_FLOOR_BYTES


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_guard_refuses_at_the_hangs_mem_available(tmp_path, host, shape):
    """The box as it was when it hung: the guard refuses, it does not admit."""
    guard = _guard(tmp_path, shape)
    host["available"] = HANG_MEM_AVAILABLE_BYTES
    with pytest.raises(RuntimeError) as refused:
        guard.check("at the 2026-09-14 hang")
    assert "capture physical memory refusal" in str(refused.value)


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_an_allocation_that_would_leave_the_hang_zone_is_refused(tmp_path, host, shape):
    """Charged before it is made, an allocation must leave the watchdog's floor."""
    host["available"] = 20 * GiB
    side = "reserve_device_bytes" if shape != "unsplit" else "reserve_bytes"
    _guard(tmp_path / "fits", shape).check(
        "leaves the floor", **{side: 20 * GiB - WATCHDOG_FLOOR_BYTES})
    guard = _guard(tmp_path / "hangs", shape)
    with pytest.raises(RuntimeError) as refused:
        guard.check("leaves the hang's MemAvailable",
                    **{side: 20 * GiB - HANG_MEM_AVAILABLE_BYTES})
    assert "capture physical memory refusal" in str(refused.value)


def test_the_stage_a_pilot_row_fits_at_its_own_ceiling(tmp_path, host):
    """21 GiB cgroup cap + 80 GiB envelope, full to what its own guard admits.

    The split guard admits at most ``cap - MARGIN_BYTES`` = 19 GiB committed
    and 80 GiB reserved, 99 GiB together. On an idle GB10 that leaves
    115.638 - 99 = 16.638 GiB available, 0.638 GiB above the floor.
    """
    guard = _guard(tmp_path, "split", current_bytes=21 * GiB - mm.CaptureMemoryGuard.MARGIN_BYTES)
    host["reserved"] = 80 * GiB
    host["available"] = IDLE_MEM_AVAILABLE_BYTES - 99 * GiB
    record = guard.check("stage A ceiling")
    assert 0 < record["host_mem_available_bytes"] - WATCHDOG_FLOOR_BYTES < GiB
    # The same row one reservation earlier: the last 10 GiB of the envelope is
    # charged before it is allocated, and it still fits.
    guard = _guard(tmp_path / "before", "split",
                   current_bytes=21 * GiB - mm.CaptureMemoryGuard.MARGIN_BYTES)
    host["reserved"] = 70 * GiB
    host["available"] = IDLE_MEM_AVAILABLE_BYTES - 89 * GiB
    guard.check("stage A last reservation", reserve_device_bytes=10 * GiB)


def test_the_r13_stage_b_row_fits_at_its_own_ceiling(tmp_path, host):
    """28 GiB + 66 GiB less the 2 GiB margin is 92 GiB: 23.638 GiB stay available."""
    guard = _guard(tmp_path, "aggregate", current_bytes=26 * GiB)
    host["reserved"] = 66 * GiB
    host["available"] = IDLE_MEM_AVAILABLE_BYTES - 92 * GiB
    guard.check("R13 ceiling")


def test_a_ceiling_past_the_watchdog_floor_is_refused_by_the_guard(tmp_path, host):
    """The 104 GiB COST plan (24 GiB + 80 GiB, 2026-09-18) cannot reach its ceiling.

    Its aggregate guard admits up to 102 GiB, which on an idle GB10 leaves
    13.638 GiB available: under the watchdog's floor, where the old 8 GiB floor
    admitted it. The guard now refuses there, with a receipt, instead of
    leaving the row to the watchdog. The most such a row can hold is
    115.638 - 16 = 99.638 GiB.
    """
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=80 * GiB, aggregate_envelope=True,
                                  **_cgroup(tmp_path, cap_bytes=24 * GiB, current_bytes=22 * GiB))
    host["reserved"] = 80 * GiB
    host["available"] = IDLE_MEM_AVAILABLE_BYTES - 102 * GiB
    with pytest.raises(RuntimeError) as refused:
        guard.check("104 GiB plan ceiling")
    assert "capture physical memory refusal" in str(refused.value)
