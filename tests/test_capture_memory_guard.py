"""The bounded capture guard holds the CPU cap and the device envelope apart.

The pilot row is 21 GiB of enforced CPU cap (docker ``--memory``), 80 GiB of
device envelope (``max_gpu_bytes``) and a 101 GiB aggregate PrismaBuild
reservation. A guard that adds the cgroup charge to the whole CUDA reservation
and compares the sum with the SMALLEST limit refuses that row at construction:
21 GiB less a 2 GiB margin cannot hold 80 GiB of residency, so the arithmetic
refused every row of this shape before any of it ran.

These tests pin the split, and pin that the un-split behaviour every existing
caller relies on is unchanged.

The second half pins the CALLERS, because the split only bounds the row if the
bytes arrive on the right side of it. `prefetch_capture` and the qualification
unit's own reservation charged one number -- the payload plus the tensors about
to move to the device -- to ``reserve_bytes``, which is the cgroup cap the
kernel enforces. Under 21 GiB / 80 GiB that refused the row's first unit while
the row held no device residency at all, and the same conflation is what a
deeper device residency would have hidden.
"""
from __future__ import annotations

import importlib.metadata
import json
import threading
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
    # No page cache: the committed bytes are memory.current (PQ #1157).
    (scope / "memory.stat").write_text("anon 0\nfile 0\nshmem 0\nfile_dirty 0\nfile_writeback 0\n")
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


def _real_allocator_api(monkeypatch, *, total_memory=121 * GiB, current_device=0):
    """A stub that REFUSES what the real allocator api refuses.

    ``get_device_properties("cuda")`` accepts the unspecified device;
    ``set_per_process_memory_fraction(f, "cuda")`` does not, and raises
    ``Expected a torch.device with a specified index or an integer``. A stub
    that accepted everything is exactly why the first version reached the real
    fleet: it proved the call, not the contract.
    """
    seen = {"fractions": {}}

    def set_fraction(fraction, device=None):
        if isinstance(device, bool) or not isinstance(device, int):
            raise ValueError(
                "Expected a torch.device with a specified index or an integer, "
                f"but got: {device}")
        seen["fractions"][device] = fraction
        seen["fraction"] = fraction
        seen["device"] = device

    def get_fraction(device=None):
        if isinstance(device, bool) or not isinstance(device, int):
            raise ValueError(
                "Expected a torch.device with a specified index or an integer, "
                f"but got: {device}")
        return seen["fractions"][device]

    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda device: type("P", (), {"total_memory": total_memory})())
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", set_fraction)
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", get_fraction,
                        raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)
    return seen


def test_enforce_device_envelope_sets_the_allocator_fraction(monkeypatch):
    """``max_gpu_bytes`` becomes a bound, not a comparison after the fact.

    The device the caller names is the unspecified ``cuda``, which the
    properties call resolves and the allocator call does not: the envelope is
    therefore set on the resolved index and reads back through the same api.
    """
    seen = _real_allocator_api(monkeypatch)

    record = mm.enforce_device_envelope("cuda", 80 * GiB)
    assert record["enforced"] is True
    assert record["device_total_bytes"] == 121 * GiB
    assert record["device_envelope_bytes"] == 80 * GiB
    assert record["allocator_device_index"] == 0
    assert seen["device"] == 0
    assert seen["fraction"] == pytest.approx(80 / 121)
    assert torch.cuda.get_per_process_memory_fraction(0) == pytest.approx(80 / 121)


def test_the_allocator_index_is_the_one_the_fraction_is_set_on(monkeypatch):
    """A named index is passed through; the unspecified form resolves."""
    seen = _real_allocator_api(monkeypatch, current_device=3)
    record = mm.enforce_device_envelope("cuda:3", 80 * GiB)
    assert record["allocator_device_index"] == 3
    assert seen["device"] == 3
    # `cuda` with no index is the shape the plan uses, and it must resolve to
    # the current device rather than reaching the api as `torch.device('cuda')`.
    record = mm.enforce_device_envelope("cuda", 80 * GiB)
    assert record["allocator_device_index"] == 3
    assert mm.allocator_device("cuda") == 3
    assert mm.allocator_device("cuda:1") == 1
    assert mm.allocator_device("cpu").type == "cpu"


def test_enforce_device_envelope_refuses_a_budget_that_bounds_nothing(monkeypatch):
    _real_allocator_api(monkeypatch)
    for bad in (0, -1, None, True, 121 * GiB, 200 * GiB):
        with pytest.raises(RuntimeError):
            mm.enforce_device_envelope("cuda", bad)
    assert mm.enforce_device_envelope("cpu", 80 * GiB)["enforced"] is False


# ---------------------------------------------------------------------------
# The callers, not just the guard
# ---------------------------------------------------------------------------

def test_reserve_allocation_keeps_an_unsplit_callback_conservative(tmp_path, host):
    """A caller taught the split hands over two numbers; nothing else changes.

    The legacy shape -- one ``resource_check(label, reserve_bytes=N)`` -- is
    what every other caller in the tree still uses, and it keeps the
    conservative sum, so teaching this seam the split cannot move an unrelated
    caller's arithmetic.
    """
    seen = []
    mm.reserve_allocation(lambda label, *, reserve_bytes=0: seen.append(reserve_bytes),
                          "legacy", cpu_bytes=3 * GiB, device_bytes=4 * GiB)
    assert seen == [7 * GiB]

    tree = _cgroup(tmp_path, cap_bytes=21 * GiB, current_bytes=1 * GiB)
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=80 * GiB, **tree)
    # The capability travels with the CALLABLE the readers are handed, not with
    # the class: a bound method carries no attributes of its own, so a property
    # on the class is invisible to ``getattr(guard.check, ...)``.
    assert getattr(guard.check, mm.SEPARATE_RESERVATIONS) is True
    record = mm.reserve_allocation(guard.check, "split",
                                   cpu_bytes=3 * GiB, device_bytes=4 * GiB)
    assert record["future_allocation_bytes"] == 3 * GiB
    assert record["future_device_allocation_bytes"] == 4 * GiB

    unsplit = mm.CaptureMemoryGuard("cuda", **_cgroup(
        tmp_path / "unsplit", cap_bytes=21 * GiB, current_bytes=1 * GiB))
    assert getattr(unsplit.check, mm.SEPARATE_RESERVATIONS) is False
    record = mm.reserve_allocation(unsplit.check, "unsplit",
                                   cpu_bytes=3 * GiB, device_bytes=4 * GiB)
    assert record["future_allocation_bytes"] == 7 * GiB
    assert record["enforced"] == "cgroup-plus-cuda-reserved"


def _published_capture(tmp_path):
    """A real one-unit capture, exactly as ``publish_capture`` writes it."""
    from prismaquant import tessera_calibration_cache as cc

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"bounded source fixture")
    transformers = importlib.metadata.version("transformers")
    census = dict(model=str(source), counts={"a": 5}, max_abs={"a": 4.0},
                  unit_shapes={"a": [3, 2]}, layer_stride=1,
                  anchor_groups={"u:a": ["a"]},
                  model_load_contract=dict(schema="prismaquant.pretrained_initialization.v1",
                      scope="checkpoint_missing_state", status="completed",
                      transformers_version=transformers),
                  attention_implementation="eager",
                  capture_runtime=dict(torch=torch.__version__,
                      cuda=torch.version.cuda, transformers=transformers))
    census_path = tmp_path / "census.json"
    census_path.write_text(json.dumps(census))
    capture_id = cc.capture_identity(
        census_path, calibration={"fit_ids_sha256": "draw"}, max_act_rows=2,
        model_load_contract=census["model_load_contract"],
        attention_implementation="eager")
    record = cc.publish_capture(
        tmp_path / "capture", census_path=census_path, identity=capture_id,
        acts={"a": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        hessians={"a": torch.eye(2) * 13},
        counts=census["counts"], maxima=census["max_abs"])
    return record, census, capture_id


def _no_cuda_move(monkeypatch):
    """``.to('cuda')`` and the device sync become no-ops on this CPU box.

    The fixture's tensors are 16 bytes and this box has no device; what the
    caller DECIDES those bytes are is the subject, so the transfer itself is
    stubbed rather than the decision.
    """
    original = torch.Tensor.to

    def to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if isinstance(target, str) and target.startswith("cuda"):
            return self
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)


def _recording_guard(guard):
    """A split-aware recorder in front of a real guard's own ``check``.

    The label's own record is what the test reads; the guard's ``last`` is
    whatever call came last, and a reader makes several (the hash brackets, the
    decode). The capability attribute is stated here because that is the
    contract ``reserve_allocation`` reads: a callable that does not declare it
    is handed the conservative sum instead.
    """
    records = []

    def resource_check(label, *, reserve_bytes=0, reserve_device_bytes=0):
        record = guard.check(label, reserve_bytes=reserve_bytes,
                             reserve_device_bytes=reserve_device_bytes)
        records.append((label, record))
        return record

    resource_check.separates_cpu_and_device_reservations = True
    return resource_check, records


def test_the_capture_seam_charges_the_device_half_to_the_envelope(
        tmp_path, host, monkeypatch):
    """The real ``prefetch_capture`` under 21 GiB CPU and 80 GiB device.

    One unit's declared payload is the campaign's order (10 GiB) while its
    tensors are 16 bytes: the guard's classification is the claim. The CPU half
    fits a 21 GiB cap holding 6 GiB (threshold 19 GiB, so 16 GiB); the same
    number DOUBLED -- the conflation, one budget -- is 26 GiB and refuses. So
    this fails on the caller that handed the guard its sum, which is exactly
    what the row did at its first unit.
    """
    from prismaquant import tessera_calibration_cache as cc

    record, census, capture_id = _published_capture(tmp_path)
    tree = _cgroup(tmp_path / "cgroup21", cap_bytes=21 * GiB, current_bytes=6 * GiB)
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=80 * GiB, **tree)
    resource_check, records = _recording_guard(guard)
    monkeypatch.setattr(cc, "_capture_storage_bytes", lambda *a, **k: 10 * GiB)
    _no_cuda_move(monkeypatch)

    values, _ = cc.prefetch_capture(
        record["path"], expected_identity=capture_id, census=census, names=["a"],
        device="cuda", expected_sha256=record["sha256"],
        resource_check=resource_check)

    assert set(values[0]) == {"a"}
    charge = next(record for label, record in records
                  if label == "before_capture_prefetch:a")
    assert charge["future_allocation_bytes"] == 10 * GiB
    assert charge["future_device_allocation_bytes"] == 10 * GiB
    assert charge["cpu_refusal_threshold_bytes"] == 19 * GiB
    assert charge["device_refusal_threshold_bytes"] == 80 * GiB
    # And the same bytes as ONE number refuse, which is what the row did: 6 GiB
    # held plus 20 GiB requested against a 19 GiB threshold.
    conflated = mm.CaptureMemoryGuard("cuda", **_cgroup(
        tmp_path / "conflated", cap_bytes=21 * GiB, current_bytes=6 * GiB))
    with pytest.raises(RuntimeError, match="capture CPU memory refusal"):
        conflated.check("conflated", reserve_bytes=20 * GiB)


def test_the_capture_seam_still_refuses_on_the_cpu_cap_alone(
        tmp_path, host, monkeypatch):
    """The split does not soften the CPU side: a payload over the cap refuses."""
    from prismaquant import tessera_calibration_cache as cc

    record, census, capture_id = _published_capture(tmp_path)
    tree = _cgroup(tmp_path / "cgroup8", cap_bytes=8 * GiB, current_bytes=6 * GiB)
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=80 * GiB, **tree)
    monkeypatch.setattr(cc, "_capture_storage_bytes", lambda *a, **k: 10 * GiB)
    _no_cuda_move(monkeypatch)

    with pytest.raises(RuntimeError, match="capture CPU memory refusal"):
        cc.prefetch_capture(
            record["path"], expected_identity=capture_id, census=census,
            names=["a"], device="cuda", expected_sha256=record["sha256"],
            resource_check=guard.check)
    assert guard.last["future_allocation_bytes"] == 10 * GiB
    assert guard.last["future_device_allocation_bytes"] == 10 * GiB


def test_the_parallel_capture_window_sums_both_budgets(tmp_path, host):
    """N readers in flight present the SUM of each budget to the guard."""
    from prismaquant import tessera_calibration_cache as cc

    tree = _cgroup(tmp_path / "cgroup", cap_bytes=21 * GiB, current_bytes=6 * GiB)
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=80 * GiB, **tree)
    reservation = cc._ConcurrentReservation(guard.check)
    registered = threading.Event()
    release = threading.Event()

    def reader():
        reservation.check("reader", reserve_bytes=4 * GiB,
                          reserve_device_bytes=30 * GiB)
        registered.set()
        release.wait(5)
        reservation.release()

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        assert registered.wait(5)
        reservation.check("consumer", reserve_bytes=6 * GiB,
                          reserve_device_bytes=20 * GiB)
        assert guard.last["future_allocation_bytes"] == 10 * GiB
        assert guard.last["future_device_allocation_bytes"] == 50 * GiB
    finally:
        release.set()
        thread.join(5)


def test_both_gpu_commands_are_capped_before_any_device_work(tmp_path, monkeypatch):
    """``run`` was uncapped: the envelope reached ``prepare`` and that was it.

    ``max_gpu_bytes`` is applied on the execution path -- after the refusals
    that need no device, before the process builds a CUDA context, a streamed
    runner or a tensor -- for BOTH GPU commands. ``synthesize`` is the CPU
    command and does not go through here at all. The budget that bounds nothing
    refuses on both; a bounding one is SET on both, and the sentinel that stops
    each command is the projection prewarm, which is the first thing after the
    cap that needs the device (the import happens inside the function, so the
    module attribute is what the call sees).
    """
    from prismaquant import gpu_guard, joint_projection_backend, tessera_joint_aura

    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path",
                        lambda *a, **k: torch.device("cuda"))
    # The same real-api stub the envelope tests use: this path resolves the
    # allocator's index, so a stub that accepts the unspecified device would
    # hide the very failure the fleet hit.
    seen = _real_allocator_api(monkeypatch)
    seen["calls"] = 0
    real_set = torch.cuda.set_per_process_memory_fraction

    def counting_set(fraction, device=None):
        seen["calls"] += 1
        return real_set(fraction, device)

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", counting_set)

    class PrewarmReached(Exception):
        pass

    def prewarm(*_args, **_kwargs):
        raise PrewarmReached

    monkeypatch.setattr(joint_projection_backend, "prewarm_projection_backend", prewarm)
    base = {"output_root": str(tmp_path), "execution": {"production_act_scales": "0"}}

    for command in ("prepare", "run"):
        with pytest.raises(RuntimeError, match="device envelope"):
            tessera_joint_aura.execute(command, {**base, "max_gpu_bytes": 200 * GiB},
                                       plan_sha256="0" * 64)
    assert seen["calls"] == 0, "a budget that bounds nothing must not be set"

    for command in ("prepare", "run"):
        with pytest.raises(PrewarmReached):
            tessera_joint_aura.execute(command, {**base, "max_gpu_bytes": 80 * GiB},
                                       plan_sha256="0" * 64)
    assert seen["calls"] == 2
    assert seen["fraction"] == pytest.approx(80 / 121)
