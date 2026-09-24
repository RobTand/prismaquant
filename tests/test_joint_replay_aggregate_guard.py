"""The joint replay guard bounds the plan's conservative sum, not the cgroup alone.

The GLM-5.3-Flash run stage (PB ``99e7880171961782…``, 2026-09-18 04:37Z) died
after a 9.9 h prepare on ``retained COST plan exceeds the actual PB physical
guard``. The plan's retained budget states ``physical_limit_bytes`` = 104 GiB,
the 24 GiB container cap plus the 80 GiB device envelope, because every phase
reservation on that path is the conservative ``cgroup + cuda_reserved`` sum. The
guard it was compared with was built without the envelope, so its cap was the
24 GiB cgroup cap: a check that had never passed on this shape since it landed
(``c6baaeb70a``, 2026-09-13).

These tests drive the real guard at the campaign's own numbers (the budget dict
below is the plan's, byte for byte) and the real comparison, then mutate the
DRIVER so the check is shown to bite, not just to pass.
"""
from __future__ import annotations

import pytest
import torch

from prismaquant import memory_management as mm
from prismaquant.joint_retained_window_plan import RetainedWindowBudget
from prismaquant import joint_statistics_replay as replay
from prismaquant.autoscale import BOUNDED_CAPTURE_ENV

GiB = 1024**3

#: ``execution.retained_operator_windows.budget`` of plan ``0b2cc0066bb612e3…``.
CAMPAIGN_BUDGET = {
    "schema": "prismaquant.joint_retained_window_budget.v1",
    "auxiliary_reserve_bytes": 2147483648, "boundary_reserve_bytes": 2281701376,
    "candidate_delta_bytes": 4194304, "load_buffer_bytes": 536870912,
    "max_windows_per_layer": 2, "metadata_reserve_bytes": 21474836480,
    "physical_limit_bytes": 111669149696, "read_page_reserve_bytes": 4096,
    "retained_render_cap_bytes": 2147483648, "runtime_reserve_bytes": 4294967296,
    "safety_margin_bytes": 2147483648, "statistics_cap_bytes": 34359738368,
    "workspace_reserve_bytes": 17179869184,
}
CONTAINER_CAP = 24 * GiB      # run spec ``cpu_memory_gb`` 24
DEVICE_ENVELOPE = 80 * GiB    # plan ``max_gpu_bytes``


def _cgroup(tmp_path, *, cap_bytes, current_bytes):
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
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None, *a, **k: state["reserved"])
    monkeypatch.setattr(mm, "_host_memory_info", lambda: (state["available"], 121 * GiB))
    return state


def test_the_campaign_plan_is_refused_by_a_cgroup_only_guard(tmp_path, host):
    """The 04:37Z failure, reproduced: 104 GiB plan against a 24 GiB cap."""
    budget = RetainedWindowBudget.from_dict(CAMPAIGN_BUDGET)
    guard = mm.CaptureMemoryGuard("cuda", **_cgroup(tmp_path, cap_bytes=CONTAINER_CAP, current_bytes=GiB))
    assert guard.physical_cap_bytes == CONTAINER_CAP
    with pytest.raises(RuntimeError) as refused:
        budget.require_physical_guard(guard)
    assert "retained COST plan exceeds the actual PB physical guard" in str(refused.value)
    assert str(111669149696) in str(refused.value) and str(CONTAINER_CAP) in str(refused.value)


def test_the_campaign_plan_fits_the_aggregate_guard(tmp_path, host):
    """24 GiB cap + 80 GiB envelope = the plan's own 104 GiB, so it runs."""
    budget = RetainedWindowBudget.from_dict(CAMPAIGN_BUDGET)
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=DEVICE_ENVELOPE, aggregate_envelope=True,
                                  **_cgroup(tmp_path, cap_bytes=CONTAINER_CAP, current_bytes=GiB))
    assert guard.physical_cap_bytes == CONTAINER_CAP + DEVICE_ENVELOPE == budget.physical_limit_bytes
    assert guard.separate_reservations is False
    assert guard.check.separates_cpu_and_device_reservations is False
    budget.require_physical_guard(guard)


@pytest.mark.parametrize("change", [
    {"physical_limit_bytes": 111669149696 + 1},          # one byte wider than the guard
    {"safety_margin_bytes": mm.CaptureMemoryGuard.MARGIN_BYTES - 1},  # less margin than it refuses at
])
def test_the_comparison_still_bites_under_the_aggregate_guard(tmp_path, host, change):
    budget = RetainedWindowBudget.from_dict({**CAMPAIGN_BUDGET, **change})
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=DEVICE_ENVELOPE, aggregate_envelope=True,
                                  **_cgroup(tmp_path, cap_bytes=CONTAINER_CAP, current_bytes=GiB))
    with pytest.raises(RuntimeError, match="exceeds the actual PB physical guard"):
        budget.require_physical_guard(guard)


def test_the_aggregate_guard_charges_the_conservative_sum_to_the_sum_of_envelopes(tmp_path, host):
    """The run's shape mid-way: 10 GiB cgroup, 45 GiB device, a 20 GiB workspace reserve."""
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=DEVICE_ENVELOPE, aggregate_envelope=True,
                                  **_cgroup(tmp_path, cap_bytes=CONTAINER_CAP, current_bytes=10 * GiB))
    host["reserved"] = 45 * GiB
    record = guard.check("admit_retained_cost:layer-3", reserve_bytes=20 * GiB)
    assert record["enforced"] == "cgroup-plus-cuda-reserved-against-aggregate"
    assert record["aggregate_envelope_bytes"] == 104 * GiB
    assert record["aggregate_refusal_threshold_bytes"] == 102 * GiB
    assert record["conservative_cgroup_plus_cuda_reserved_bytes"] == 55 * GiB
    # The same reading under the cgroup-only guard is the refusal the run hit next.
    plain = mm.CaptureMemoryGuard("cuda", **_cgroup(tmp_path / "plain", cap_bytes=CONTAINER_CAP, current_bytes=10 * GiB))
    with pytest.raises(RuntimeError, match="capture CPU memory refusal"):
        plain.check("admit_retained_cost:layer-3", reserve_bytes=20 * GiB)


def test_the_aggregate_guard_refuses_past_the_sum(tmp_path, host):
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=DEVICE_ENVELOPE, aggregate_envelope=True,
                                  **_cgroup(tmp_path, cap_bytes=CONTAINER_CAP, current_bytes=10 * GiB))
    host["reserved"] = 45 * GiB
    with pytest.raises(RuntimeError, match="capture aggregate memory refusal"):
        guard.check("admit_retained_cost:layer-3", reserve_bytes=48 * GiB)  # 103 > 102


def test_the_aggregate_guard_still_holds_the_cgroup_cap_the_kernel_enforces(tmp_path, host):
    """A 23 GiB cgroup charge fits the 102 GiB sum and is refused anyway."""
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=DEVICE_ENVELOPE, aggregate_envelope=True,
                                  **_cgroup(tmp_path, cap_bytes=CONTAINER_CAP, current_bytes=23 * GiB))
    host["reserved"] = 10 * GiB
    with pytest.raises(RuntimeError, match="capture aggregate memory refusal"):
        guard.check("before_retained_cost:layer-0")


def test_the_aggregate_guard_still_holds_the_device_envelope(tmp_path, host):
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=DEVICE_ENVELOPE, aggregate_envelope=True,
                                  **_cgroup(tmp_path, cap_bytes=CONTAINER_CAP, current_bytes=GiB))
    host["reserved"] = 79 * GiB
    with pytest.raises(RuntimeError, match="capture device memory refusal"):
        guard.check("before_joint_window_backward", reserve_device_bytes=2 * GiB)


def test_an_aggregate_envelope_needs_a_device_envelope(tmp_path):
    with pytest.raises(RuntimeError, match="needs a declared device envelope"):
        mm.CaptureMemoryGuard("cuda", aggregate_envelope=True,
                              **_cgroup(tmp_path, cap_bytes=CONTAINER_CAP, current_bytes=GiB))


def test_the_replay_guard_is_the_aggregate_one_when_the_envelope_is_declared(tmp_path, host, monkeypatch):
    """``operator_window_guard`` is the only constructor on the replay path."""
    for name, value in BOUNDED_CAPTURE_ENV.items():
        monkeypatch.setenv(name, value)
    built = {}
    real = mm.CaptureMemoryGuard

    def constructor(device, **kwargs):
        built.update(kwargs)
        tree = tmp_path / str(len(built) + sum(1 for _ in tmp_path.iterdir()))
        return real(device, **kwargs, **_cgroup(tree, cap_bytes=CONTAINER_CAP, current_bytes=GiB))
    monkeypatch.setattr(mm, "CaptureMemoryGuard", constructor)

    guard = replay.operator_window_guard("cuda:0", device_bytes=DEVICE_ENVELOPE)
    assert built == {"device_bytes": DEVICE_ENVELOPE, "aggregate_envelope": True}
    assert guard.physical_cap_bytes == CONTAINER_CAP + DEVICE_ENVELOPE
    built.clear()
    plain = replay.operator_window_guard("cuda:0")
    assert built == {} and plain.physical_cap_bytes == CONTAINER_CAP
    assert replay.operator_window_guard("cpu", device_bytes=DEVICE_ENVELOPE) is None
