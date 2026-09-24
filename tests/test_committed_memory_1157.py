"""Clean file pages are never committed memory (PQ #1157).

The layer-044 capture-workspace profile (PQ #1151, attempt 1, on sparklina,
2026-09-24) was refused before any GPU work:

    before_quantum_reverse:44: actual cgroup-plus-CUDA baseline 44561817600
    exceeds declared metadata/runtime/source/auxiliary owners 43132886892

Its cgroup held about 4.7 GB of anon beside about 21.8 GB of clean page cache
left by the own-source read. The guard read ``memory.current``, which counts
both, and the kernel drops clean cache before it refuses an allocation. So the
plan refused memory that was never committed.

The fixture is the same row's cgroup on sparky (attempt 2), read verbatim at
its largest clean file charge during the own-source read: 4.71 GB of anon and
18.43 GB of clean cache. The tests put the refusal's own observed and declared
bytes around it, drive the real guard and the real plan check, and then make
the cache dirty to show that the check still bites.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from prismaquant import joint_retained_window_plan as plan
from prismaquant import memory_management as mm
from prismaquant.joint_retained_window_plan import RetainedWindowBudget

FIXTURE = json.loads((Path(__file__).parent / "fixtures"
                      / "cgroup_memory_stat_layer044_sparky.json").read_text())
#: ``budget`` of ``a4/stageb-resource-policy.r13.json``.
R13_BUDGET = {
    "schema": "prismaquant.joint_retained_window_budget.v1",
    "auxiliary_reserve_bytes": 2147483648, "boundary_reserve_bytes": 2281701376,
    "candidate_delta_bytes": 201326592, "load_buffer_bytes": 402662764,
    "max_windows_per_layer": 25, "metadata_reserve_bytes": 21474836480,
    "physical_limit_bytes": 100931731456, "read_page_reserve_bytes": 4096,
    "retained_render_cap_bytes": 6023929799, "runtime_reserve_bytes": 4294967296,
    "safety_margin_bytes": 2147483648, "statistics_cap_bytes": 5939134464,
    "workspace_reserve_bytes": 17179869184,
}
#: ``limits.gpu_bytes`` of the same policy: the device envelope of the guard.
R13_GPU_BYTES = 70866960384
#: The refusal's observed cgroup-plus-CUDA bytes and its declared limit.
REFUSED_OBSERVED = 44561817600
REFUSED_LIMIT = 43132886892
#: The source bytes that make the R13 owners state the refusal's limit.
REFUSED_SOURCE = (REFUSED_LIMIT - R13_BUDGET["metadata_reserve_bytes"]
                  - R13_BUDGET["runtime_reserve_bytes"])
#: The key the plan's callers compare with the declared owners. Before #1157
#: both callers read the raw charge; reading it through ``getattr`` keeps this
#: test runnable on that tree, so the regression shows there as the refusal.
OBSERVED_KEY = getattr(plan, "OBSERVED_BASELINE_KEY",
                       "conservative_cgroup_plus_cuda_reserved_bytes")


def _stat(text):
    return dict((key, int(value)) for key, value in
                (line.split() for line in text.splitlines()))


def _format(stat):
    return "".join(f"{key} {value}\n" for key, value in stat.items())


def _cgroup(tmp_path, stat_text):
    scope = tmp_path / "cgroup" / "scope"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text(str(FIXTURE["memory.max"]))
    (scope / "memory.current").write_text(str(FIXTURE["memory.current"]))
    (scope / "memory.stat").write_text(stat_text)
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/scope\n")
    return {"cgroup_root": tmp_path / "cgroup", "membership": membership}


@pytest.fixture()
def host(monkeypatch):
    """The refusal's CUDA reservation, on a box with room on the host."""
    reserved = REFUSED_OBSERVED - FIXTURE["memory.current"]
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: reserved)
    # check_operator_allocation empties the allocator cache first; no device
    # work is under test.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(mm, "_host_memory_info", lambda: (100 * 1024**3, 121 * 1024**3))
    return reserved


def _stage_b_guard(tmp_path, stat_text):
    """The guard ``operator_window_guard`` builds for a Stage B row."""
    return mm.CaptureMemoryGuard("cuda", device_bytes=R13_GPU_BYTES, aggregate_envelope=True,
                                 **_cgroup(tmp_path, stat_text))


def _require_baseline(observed):
    RetainedWindowBudget.from_dict(R13_BUDGET).require_observed_baseline(
        observed_bytes=observed[OBSERVED_KEY], source_bytes=REFUSED_SOURCE,
        actual_auxiliary_bytes=0, label="before_quantum_reverse:44")


def test_the_fixture_is_the_refused_shape():
    stat = _stat(FIXTURE["memory.stat"])
    clean = stat["file"] - stat["shmem"] - stat["file_dirty"] - stat["file_writeback"]
    assert FIXTURE["memory.max"] == 30064771072 == 28 * 1024**3
    assert stat["anon"] < 5 * 10**9 < 18 * 10**9 < clean
    assert FIXTURE["memory.current"] > clean + stat["anon"]


def test_the_refused_baseline_is_admitted_once_clean_cache_is_left_out(tmp_path, host):
    """The attempt-1 refusal, with this row's real memory.stat under it."""
    guard = _stage_b_guard(tmp_path, FIXTURE["memory.stat"])
    observed = guard.check("before_quantum_reverse:44")
    assert observed["conservative_cgroup_plus_cuda_reserved_bytes"] == REFUSED_OBSERVED
    _require_baseline(observed)

    stat = _stat(FIXTURE["memory.stat"])
    clean = stat["file"] - stat["shmem"] - stat["file_dirty"] - stat["file_writeback"]
    assert observed["cgroup_clean_file_bytes"] == clean
    assert observed["cgroup_committed_bytes"] == FIXTURE["memory.current"] - clean
    assert observed[OBSERVED_KEY] == REFUSED_OBSERVED - clean <= REFUSED_LIMIT


def test_the_check_still_bites_when_the_cache_is_dirty(tmp_path, host):
    """Mutate the driver: the same pages, waiting to be written, are committed."""
    stat = _stat(FIXTURE["memory.stat"])
    stat["file_dirty"] = stat["file"] - stat["shmem"] - stat["file_writeback"]
    guard = _stage_b_guard(tmp_path, _format(stat))
    observed = guard.check("before_quantum_reverse:44")
    assert observed[OBSERVED_KEY] == REFUSED_OBSERVED
    with pytest.raises(RuntimeError, match=f"cgroup-plus-CUDA baseline {REFUSED_OBSERVED} "
                                           f"exceeds declared .* owners {REFUSED_LIMIT}"):
        _require_baseline(observed)


def test_the_cgroup_side_of_the_guard_reads_committed_bytes(tmp_path, host):
    """The kernel's own cap is held against committed bytes, not the page cache.

    Clean cache fills a cgroup up to its cap, because the kernel reclaims it
    only there. The same row, with the cache grown to the cap, is admitted;
    the same bytes held as anon are refused.
    """
    stat = _stat(FIXTURE["memory.stat"])
    clean = stat["file"] - stat["shmem"] - stat["file_dirty"] - stat["file_writeback"]
    grown = FIXTURE["memory.max"] - FIXTURE["memory.current"] - 4096
    cached = dict(stat, file=stat["file"] + grown)
    tree = _cgroup(tmp_path / "cached", _format(cached))
    (tree["cgroup_root"] / "scope" / "memory.current").write_text(
        str(FIXTURE["memory.current"] + grown))
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=R13_GPU_BYTES,
                                  aggregate_envelope=True, **tree)
    record = guard.check("full of clean cache")
    assert record["cgroup_current_bytes"] > record["cpu_refusal_threshold_bytes"]
    assert record["cgroup_committed_bytes"] < record["cpu_refusal_threshold_bytes"]

    held = dict(stat, anon=stat["anon"] + clean + grown, file=stat["file"] - clean)
    tree = _cgroup(tmp_path / "held", _format(held))
    (tree["cgroup_root"] / "scope" / "memory.current").write_text(
        str(FIXTURE["memory.current"] + grown))
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=R13_GPU_BYTES,
                                  aggregate_envelope=True, **tree)
    with pytest.raises(RuntimeError, match="capture aggregate memory refusal"):
        guard.check("full of anon")


@pytest.mark.parametrize("missing", mm.COMMITTED_MEMORY_STAT_KEYS
                         if hasattr(mm, "COMMITTED_MEMORY_STAT_KEYS") else ["file"])
def test_a_stat_without_a_committed_key_refuses(tmp_path, host, missing):
    stat = _stat(FIXTURE["memory.stat"])
    del stat[missing]
    guard = _stage_b_guard(tmp_path, _format(stat))
    with pytest.raises(RuntimeError, match="memory.stat lacks"):
        guard.check("before_quantum_reverse:44")
    # The refusal latches, as every other failed observation does.
    with pytest.raises(RuntimeError):
        guard.check("after")


def test_a_race_never_counts_less_than_the_stat_states(tmp_path):
    """Cache dropped between the two reads cannot lower committed below anon+."""
    stat = _stat(FIXTURE["memory.stat"])
    stated = stat["anon"] + stat["shmem"] + stat["file_dirty"] + stat["file_writeback"]
    assert mm.committed_cgroup_bytes(stated // 2, stat) == stated


def test_the_device_side_of_an_allocation_is_held_against_the_envelope(tmp_path, host):
    """The capture's CUDA allocations reach the device envelope (PQ #1157)."""
    from prismaquant import joint_statistics_replay as replay

    guard = _stage_b_guard(tmp_path, FIXTURE["memory.stat"])
    room = R13_GPU_BYTES - host
    record = replay.check_operator_allocation(
        guard, "before_joint_window_backward", reserve_bytes=0, reserve_device_bytes=room)
    assert record["future_device_allocation_bytes"] == room
    with pytest.raises(RuntimeError, match="capture device memory refusal"):
        replay.check_operator_allocation(
            guard, "before_joint_window_backward:over", reserve_bytes=0,
            reserve_device_bytes=room + 1)


def test_a_guard_without_a_device_envelope_takes_the_sum(tmp_path, host):
    from prismaquant import joint_statistics_replay as replay

    guard = mm.CaptureMemoryGuard("cuda", **_cgroup(tmp_path, FIXTURE["memory.stat"]))
    guard.check = lambda label, *, reserve_bytes=0, reserve_device_bytes=0: dict(
        label=label, reserve_bytes=reserve_bytes, reserve_device_bytes=reserve_device_bytes)
    record = replay.check_operator_allocation(guard, "sum", reserve_bytes=3,
                                              reserve_device_bytes=4)
    assert record == dict(label="sum", reserve_bytes=7, reserve_device_bytes=0)
