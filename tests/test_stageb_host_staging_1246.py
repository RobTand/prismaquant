"""Stage B capture groups stage their host bytes in one held buffer (PQ #1246).

A Stage B capture pass used to allocate host memory once per capture group,
at three sites: each read of the cotangent scratch (``ExactCotangentScratch``
allocated a fresh tensor and ``_direct_io`` read into it), the stacks of the
incoming cotangents and of the layer inputs (``_stack_to_device``'s
``torch.cat``), and the input cotangent's copy to the host (``.to("cpu")``)
plus one ``.clone()`` per stored batch. The allocator's purge delay is zero
under the bounded capture contract (#366), so every one of those allocations
faulted its pages in again, and on a fragmented box a huge-page fault
compacts memory first.

These tests pin the fix on the one-pass spill fixture of
``tests/test_stageb_one_pass_spill.py``, grown to 1,024 tokens per sample so
a cotangent slot (32 KiB of bf16) stands clear of the capture's small host
allocations, with the cotangent plane in a real ``ExactCotangentScratch`` on
the direct-I/O path the campaign runs:

* **Per group.** A torch profiler around each capture pass counts the host
  allocations, of at least one slot, that the capture's thread makes between
  one group's start and the next. The boundary reads that
  ``aura.exact_activation.prefetch`` stages are not the capture's and are
  left out. Before the fix a group of two stored batches made seven such
  allocations and a batch at ``capture_batch=1`` made two; after it, none.
* **The guard.** The held buffer is admitted to the capture guard
  (``before_stage_b_plane_staging``) before it is allocated, and it is
  committed before the pass's backward admission reads the guard.
* **The bytes.** ``read_into`` and the staging helpers return exactly the
  bytes the allocating paths returned.
"""
from __future__ import annotations

import bisect
import contextlib
import dataclasses
import json
import mmap
from types import SimpleNamespace

import pytest
import torch

import prismaquant.aura_cost as aura
import prismaquant.joint_adjoint_checkpoints as checkpoints
import prismaquant.joint_replay_spill as spill_mod
import prismaquant.perturbed_x_cache as scratch_mod
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.joint_adjoint_checkpoints import _stack_to_device, chain_layers_for
from prismaquant.joint_adjoint_slices import adjoint_slice_sha256, stage_a_slice
from prismaquant.joint_retained_window_plan import RetainedWindowBudget

import test_joint_cost_quantum_runtime as rt
import test_stageb_one_pass_spill as spill_tests
from test_stageb_cotangent_scratch import _direct_io_supported
from test_stageb_one_pass_spill import (
    FORMATS,
    N_PROBES,
    RENDER_FORMATS,
    VOCAB,
    WIDTH,
    _MoELM,
    _chain,
    _clear_output,
    _policy_budget,
    _prepared,
    _runner,
    _spill_root,
    _targets,
)
from test_streamed_cost_checkpoints import _model_identity

#: The spill suite's execution, bound before ``_run`` patches its name.
_SPILL_EXECUTION = spill_tests._execution

SAMPLES, WINDOW, TOKENS = 8, 4, 1024
#: One stored batch's boundary or cotangent entry: 1,024 tokens of width 16
#: in bf16, a whole number of direct-I/O blocks.
SLOT_BYTES = TOKENS * WIDTH * 2
#: Layer 1 walks no chain, so its capture passes are the whole quantum's
#: GPU work on the plane.
LAYER = 1
#: The campaign's accumulation (``stage-b-spec.v7-gpu68.json``); the capture
#: batch is the test's.
CAMPAIGN_ACCUMULATION = "accumulation=operator_gemm,chunk_rows=65536"
GROUP_MARKER = "pq1246/group"
PREFETCH_RANGE = "aura.exact_activation.prefetch"
MEMORY_EVENT = "[memory]"
STAGING_ADMISSION = "before_stage_b_plane_staging"
#: The boundary owner's auxiliary ceiling at 1,024 tokens; see ``_execution``.
AUXILIARY_BYTES = 1 << 30


def _calibration():
    generator = torch.Generator().manual_seed(1246)
    return torch.randint(0, VOCAB, (SAMPLES, TOKENS), generator=generator)


def _execution(root):
    """The spill suite's execution with read windows of four 32 KiB entries.

    At 1,024 tokens the per-sample metadata that the boundary owner retains
    and checks (ids, positions and masks) is far above the spill suite's
    1 MiB auxiliary ceiling, so the ceiling here is 1 GiB. It is a bound the
    owner checks, not an allocation. The retained plan charges the owner's
    ceiling, so its auxiliary reserve is the same 1 GiB, and its physical
    limit grows by the same bytes: every window keeps the allowance the
    fixture's records were planned with.
    """
    execution = _SPILL_EXECUTION(root)
    policy = rt._boundary_policy(root / "boundaries", window=WINDOW)
    policy["max_resident_bytes"] = WINDOW * (1 + N_PROBES) * SLOT_BYTES + SLOT_BYTES
    policy["max_auxiliary_bytes"] = AUXILIARY_BYTES
    policy["max_artifact_bytes"] = 256 << 20
    execution["boundary_storage"] = policy
    retained = execution["retained_operator_windows"]
    budget = RetainedWindowBudget.from_dict(retained["budget"])
    grown = AUXILIARY_BYTES - budget.auxiliary_reserve_bytes
    budget = dataclasses.replace(
        budget, auxiliary_reserve_bytes=AUXILIARY_BYTES,
        physical_limit_bytes=budget.physical_limit_bytes + grown)
    execution["retained_operator_windows"] = {**retained, "budget": budget.as_dict()}
    return execution


def _regime(capture_batch):
    return f"capture_batch={capture_batch},{CAMPAIGN_ACCUMULATION}"


@pytest.fixture(scope="module")
def staging_campaign(tmp_path_factory):
    """One Stage A capture at 1,024 tokens per sample plus sealed records."""
    from prismaquant.joint_cost_stage_a import run_adjoint_capture_core
    from prismaquant.joint_statistics_replay import preflight_joint_operator_admission

    root = tmp_path_factory.mktemp("host-staging-1246")
    device = spill_tests._device()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
        patch.setenv("PRISMAQUANT_DEV_MODE", "1")
        torch.manual_seed(715)
        state = {key: tensor.detach().clone()
                 for key, tensor in _MoELM().state_dict().items()}
        model, context, runner = _runner(state, device)
        linears = _targets(model, runner.profile)
        weights = {(qname, fmt): module.weight.detach().to("cpu").clone() + 0.03125
                   for qname, module in linears.items() for fmt in RENDER_FORMATS}
        cache, linears = _prepared(model, context, runner, weights, root / "shared")
        formats_by_qname = {qname: list(FORMATS) for qname in linears}
        policy, budget, _retained = _policy_budget()
        names_by_layer = {layer: sorted(n for n in linears
                                        if runner.layer_index_for_qname(n) == layer)
                          for layer in (0, 1)}
        preflight = preflight_joint_operator_admission(
            names_by_layer, linears, {qname: list(RENDER_FORMATS) for qname in linears},
            cache, policy=policy, retained_budget=budget, source_bytes=1 << 20)
        output_root = root / "campaign"
        _model_a, _context_a, runner_a = _runner(state, device)
        receipt = run_adjoint_capture_core(
            runner_a, _calibration(), execution=_execution(root / "exec"),
            output_root=output_root, stride=2,
            source_model_identity=_model_identity("joint-source"),
            unit_roster_sha256=rt._hex("a"), plan_sha256=rt._hex("d"),
            prepared_sha256=rt._hex("e"), read_manifest_sha256=rt._hex("f"),
            implementation_sha256=aura._aura_source_sha256())
    assert [c["boundary"] for c in receipt["checkpoints"]] == [2]
    records, slices = {}, {}
    for layer in (0, 1):
        slices[layer] = stage_a_slice(json.loads(json.dumps(receipt)), layer)
        windows = rt._windows_records(preflight[layer])
        record = rt._quantum_record(
            output_root=output_root, layer=layer,
            checkpoint_boundary=slices[layer]["checkpoint"]["boundary"], chain=[],
            windows=[{"window_index": index} for index in range(len(windows))],
            total_bytes=sum(w["render_file_upper_bound_bytes"] for w in windows),
            plan_sha=rt._hex("d"), prepared_sha=rt._hex("e"),
            adjoint_sha=adjoint_slice_sha256(slices[layer]))
        record["adjoint"]["chain_layers"] = list(chain_layers_for(2, layer))
        record["identity_sha256"] = canonical_json_sha256(
            {k: v for k, v in record.items() if k != "identity_sha256"}, where="record")
        records[layer] = record
    assert records[LAYER]["adjoint"]["chain_layers"] == []
    return SimpleNamespace(root=root, state=state, weights=weights, receipt=receipt,
                           records=records, slices=slices, output_root=output_root,
                           device=device, formats_by_qname=formats_by_qname,
                           preflight=preflight)


def _scratch_root(tmp_path):
    root = tmp_path / "cotangent-scratch"
    root.mkdir()
    assert _direct_io_supported(root), (
        f"{root} has no direct I/O on an 8 KiB grid: the campaign's cotangent scratch "
        "reads with O_DIRECT, and this test measures that path")
    return root


def _run(campaign, monkeypatch, *, capture_batch, spill_root, scratch_root, guard=None):
    """Layer 1's quantum under the spill, its plane in the cotangent scratch."""
    _clear_output(campaign, LAYER)
    monkeypatch.delenv("PRISMAQUANT_STAGE_B_KERNEL_PROFILE", raising=False)
    monkeypatch.setenv("PRISMAQUANT_STAGE_B_COTANGENT_ROOT", str(scratch_root))
    monkeypatch.setenv("PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES", str(1 << 26))
    with monkeypatch.context() as patch:
        patch.setattr(spill_tests, "_calibration", _calibration)
        patch.setattr(spill_tests, "_execution", _execution)
        payload, state = spill_tests._quantum(
            campaign, monkeypatch, layer=LAYER, spill_root=spill_root,
            ceiling=1 << 30, regime=_regime(capture_batch), guard=guard)
    assert payload is not None, _chain(state.error)
    return payload, state


def _instrument(monkeypatch):
    """Profile each capture pass and mark each capture group's start.

    The quantum imports ``_free_gib`` from ``aura_cost`` when it starts, and
    both the batched capture group and the one-batch capture loop call it once,
    first, per group; the marker is an empty ``record_function`` range there.
    The profiler opens after ``spill.capture`` has set its arenas up and
    closes before it tears them down, so a pass's profile holds its groups.
    """
    seen = SimpleNamespace(passes=[], scratches=[], direct={"read": 0, "write": 0})
    real_free_gib = aura._free_gib

    def marked_free_gib(*args, **kwargs):
        with torch.profiler.record_function(GROUP_MARKER):
            pass
        return real_free_gib(*args, **kwargs)

    monkeypatch.setattr(aura, "_free_gib", marked_free_gib)
    original_capture = spill_mod.StageBReplaySpill.capture

    @contextlib.contextmanager
    def profiled_capture(self, *args, **kwargs):
        with original_capture(self, *args, **kwargs) as observer:
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    profile_memory=True) as profile:
                yield observer
            seen.passes.append(profile.profiler.kineto_results.events())

    monkeypatch.setattr(spill_mod.StageBReplaySpill, "capture", profiled_capture)
    original_init = scratch_mod.ExactCotangentScratch.__init__

    def recorded_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        seen.scratches.append(self)

    monkeypatch.setattr(scratch_mod.ExactCotangentScratch, "__init__", recorded_init)
    original_direct_io = scratch_mod.ExactCotangentScratch._direct_io

    def counted_direct_io(self, call, view, offset, size, what, *rest):
        seen.direct[what] += 1
        return original_direct_io(self, call, view, offset, size, what, *rest)

    monkeypatch.setattr(scratch_mod.ExactCotangentScratch, "_direct_io", counted_direct_io)
    return seen


def _enclosing(ops, starts, time_ns):
    """The innermost few CPU ops open at ``time_ns`` on the capture's thread."""
    index = bisect.bisect_right(starts, time_ns)
    chain = [name for start, end, name in ops[max(0, index - 512):index] if end >= time_ns]
    return " > ".join(chain[-4:]) or "<no enclosing op>"


def _pass_groups(events, *, threshold):
    """Per capture group, the capture thread's host allocations of one pass.

    Returns ``(groups, before)``: for each group, the count of all its host
    allocations and the list ``(bytes, enclosing ops)`` of those of at least
    ``threshold`` bytes; ``before`` lists the large ones made after the
    profile opened and before the first group.
    """
    from torch.autograd import DeviceType

    markers = sorted((event.start_ns(), event.start_thread_id())
                     for event in events if event.name() == GROUP_MARKER)
    assert markers, "the capture pass marked no group start"
    threads = {thread for _start, thread in markers}
    assert len(threads) == 1, f"group starts on several threads: {sorted(threads)}"
    (thread,) = threads
    starts = [start for start, _thread in markers]
    prefetch = [(event.start_ns(), event.end_ns()) for event in events
                if event.name() == PREFETCH_RANGE and event.start_thread_id() == thread]
    ops = sorted((event.start_ns(), event.end_ns(), event.name()) for event in events
                 if event.start_thread_id() == thread
                 and event.device_type() == DeviceType.CPU
                 and event.name() not in (MEMORY_EVENT, GROUP_MARKER))
    op_starts = [op[0] for op in ops]
    groups = [{"allocations": 0, "large": []} for _ in starts]
    before = []
    for event in events:
        if (event.name() != MEMORY_EVENT or event.device_type() != DeviceType.CPU
                or event.start_thread_id() != thread or event.nbytes() <= 0):
            continue
        at = event.start_ns()
        if any(start <= at <= end for start, end in prefetch):
            continue
        large = event.nbytes() >= threshold
        index = bisect.bisect_right(starts, at) - 1
        if index < 0:
            if large:
                before.append((event.nbytes(), _enclosing(ops, op_starts, at)))
            continue
        groups[index]["allocations"] += 1
        if large:
            groups[index]["large"].append((event.nbytes(), _enclosing(ops, op_starts, at)))
    return groups, before


@pytest.mark.parametrize("capture_batch", [1, 2], ids=["capture-b1", "capture-b2"])
def test_capture_groups_allocate_no_host_memory_per_group(staging_campaign, monkeypatch,
                                                          tmp_path, capture_batch):
    """A capture group reuses the pass's held buffer: no host allocation per group.

    Counts, per capture group of each probe's capture pass, the host
    allocations of at least one cotangent slot that the capture's thread
    makes. Before PQ #1246 a group of two stored batches made seven (two
    scratch reads, two stacks, the copy to the host and two clones) and a
    batch at ``capture_batch=1`` made two (the scratch read and the copy to
    the host).
    """
    if staging_campaign.device.type != "cuda":
        pytest.skip("the capture's host copies are allocations only off a CUDA device")
    scratch_root = _scratch_root(tmp_path)
    seen = _instrument(monkeypatch)
    _run(staging_campaign, monkeypatch, capture_batch=capture_batch,
         spill_root=_spill_root(tmp_path), scratch_root=scratch_root)
    assert len(seen.scratches) == 1, len(seen.scratches)
    (scratch,) = seen.scratches
    assert scratch._direct is not None, (
        "the cotangent scratch fell back to buffered I/O; the campaign reads with O_DIRECT")
    assert seen.direct["read"] > 0, seen.direct
    slot_bytes = min(size for _offset, size, _shape, _dtype in scratch._slots.values())
    threshold = min(slot_bytes, TOKENS * WIDTH * torch.empty((), dtype=spill_tests.DTYPE)
                    .element_size())
    assert threshold == SLOT_BYTES, (threshold, slot_bytes)
    assert len(seen.passes) == N_PROBES, len(seen.passes)
    groups_per_pass = -(-SAMPLES // capture_batch)
    breakdown, per_group, staged = [], [], []
    for probe, events in enumerate(seen.passes):
        groups, before = _pass_groups(events, threshold=threshold)
        assert len(groups) == groups_per_pass, (probe, len(groups))
        assert all(group["allocations"] > 0 for group in groups), (
            "the profiler recorded no host allocation in some group; a count of zero "
            f"large ones would prove nothing: {[g['allocations'] for g in groups]}")
        staged.append(before)
        for index, group in enumerate(groups):
            per_group.append(len(group["large"]))
            for nbytes, where in group["large"]:
                breakdown.append(f"probe {probe} group {index}: {nbytes} B in {where}")
    print(f"PQ1246 capture_batch={capture_batch} threshold={threshold} "
          f"large host allocations per group={per_group} "
          f"before the first group={staged} direct I/O calls={seen.direct}")
    assert not breakdown, (
        f"capture groups allocated host memory of at least one slot ({threshold} B) "
        f"per group: counts {per_group}\n" + "\n".join(breakdown))


def test_plane_staging_is_admitted_before_it_is_allocated(staging_campaign, monkeypatch,
                                                          tmp_path):
    """The held buffer is charged to the capture guard, then committed (#366).

    For every probe's capture pass the guard sees
    ``before_stage_b_plane_staging`` reserve at least the buffer's allocation,
    the buffer is allocated and touched right after it, it reserves nothing
    more while held, and only then does the pass's backward admission read
    the guard.
    """
    guard = spill_tests._RecordingGuard(staging_campaign.device)
    staging_class = getattr(checkpoints, "PlaneHostStaging", None)
    if staging_class is not None:
        original_held = staging_class.held

        @contextlib.contextmanager
        def held(self):
            with original_held(self) as stage:
                buffer = self._buffer
                guard.admissions.append((
                    "allocated", 0 if buffer is None else buffer.untyped_storage().nbytes()))
                guard.admissions.append(("reserve_while_held", self.reserve_host_bytes))
                yield stage

        monkeypatch.setattr(staging_class, "held", held)
    _run(staging_campaign, monkeypatch, capture_batch=2, spill_root=_spill_root(tmp_path),
         scratch_root=_scratch_root(tmp_path), guard=guard)
    labels = [label for label, _reserve in guard.admissions]
    backward = [index for index, label in enumerate(labels)
                if label == "before_joint_window_backward"]
    staging = [index for index, label in enumerate(labels) if label == STAGING_ADMISSION]
    assert len(backward) == N_PROBES, labels
    assert len(staging) == N_PROBES, (
        f"expected one {STAGING_ADMISSION} admission per capture pass (PQ #1246): {labels}")
    for admitted, backward_index in zip(staging, backward):
        assert labels[admitted:backward_index + 1] == [
            STAGING_ADMISSION, "allocated", "reserve_while_held",
            "before_joint_window_backward"], labels
        reserve = guard.admissions[admitted][1]
        allocated = guard.admissions[admitted + 1][1]
        assert 2 * SLOT_BYTES <= allocated <= reserve, (allocated, reserve)
        assert guard.admissions[admitted + 2][1] == 0


# -- the bytes: read_into and the staging helpers ----------------------------

def _grid_records(shape, dtype, count):
    size = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
    return [{"name": f"cotangent-0-{index}", "shape": list(shape), "dtype": str(dtype),
             "tensor_bytes": size} for index in range(count)]


def _bits(shape, dtype, salt):
    """Values with signed zeros, infinities, a NaN and subnormal patterns."""
    width = torch.empty((), dtype=dtype).element_size()
    integer = {2: torch.int16, 4: torch.int32}[width]
    count = torch.Size(shape).numel()
    info = torch.iinfo(integer)
    generator = torch.Generator().manual_seed(1246 + salt)
    raw = torch.randint(info.min, info.max, (count,), generator=generator, dtype=torch.int64)
    raw[:4] = torch.tensor([0, info.min, 1, -1])
    return raw.to(integer).view(dtype).reshape(shape)


def _page_aligned(nbytes):
    raw = torch.empty(nbytes + mmap.PAGESIZE, dtype=torch.uint8)
    return raw.narrow(0, (-raw.data_ptr()) % mmap.PAGESIZE, nbytes)


@pytest.mark.parametrize("grid", [True, False], ids=["direct", "buffered"])
def test_scratch_read_into_returns_the_bytes_a_fresh_read_returns(tmp_path, grid):
    """``read_into`` fills a caller's tensor with ``__getitem__``'s bytes."""
    from prismaquant.perturbed_x_cache import ExactCotangentScratch

    if grid and not _direct_io_supported(tmp_path):
        pytest.skip("tmp_path has no direct I/O on an 8 KiB grid")
    shape, dtype = ((2, 2048), torch.bfloat16) if grid else ((3, 5), torch.float32)
    arena = ExactCotangentScratch(_grid_records(shape, dtype, 3), directory=tmp_path,
                                  max_bytes=1 << 20)
    try:
        assert (arena._direct is not None) == grid
        nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        values = [_bits(shape, dtype, index) for index in range(3)]
        for index, value in enumerate(values):
            arena[0, index] = value
        assert arena.slot_layout((0, 1)) == (shape, dtype, nbytes)
        buffer = _page_aligned(4 * nbytes + 64)
        for index, value in enumerate(values):
            # On the memory grid, and off it by one element: the bounce path.
            for offset in (0, torch.empty((), dtype=dtype).element_size()):
                out = buffer[offset:offset + nbytes].view(dtype).view(shape)
                out.fill_(0)
                assert arena.read_into((0, index), out) is out
                assert torch.equal(out.view(torch.uint8), arena[0, index].view(torch.uint8))
                assert torch.equal(out.view(torch.uint8), value.view(torch.uint8))
        good = buffer[:nbytes].view(dtype).view(shape)
        for bad in (good.view(-1), good.to(torch.float16),
                    torch.empty(tuple(reversed(shape)), dtype=dtype).t(),
                    torch.empty(shape, dtype=dtype, device="meta")):
            with pytest.raises(ValueError, match="read target"):
                arena.read_into((0, 0), bad)
        arena._written.discard((0, 2))
        with pytest.raises(RuntimeError, match="not ready"):
            arena.read_into((0, 2), good)
    finally:
        arena.close()
    assert list(tmp_path.iterdir()) == []


def _staging(plane, keys, *, group_batches, dtype):
    from prismaquant.joint_adjoint_checkpoints import PlaneHostStaging

    return PlaneHostStaging(plane, keys, group_batches=group_batches, dtype=dtype)


def test_plane_staging_stacks_the_bytes_the_allocating_stack_did():
    """``incoming`` and ``boundaries`` return ``_stack_to_device``'s tensors.

    On a plane that keeps its tensors, with ragged rows and a conversion
    from float32 to bf16, the staged stacks equal the allocating stacks bit
    for bit, and neither result aliases the held buffer.
    """
    rows = (1, 2, 1)
    plane = {(0, index): _bits((count, 4, 8), torch.bfloat16, index)
             for index, count in enumerate(rows)}
    keys = list(plane)
    boundaries = [torch.randn(count, 4, 8, generator=torch.Generator().manual_seed(index))
                  for index, count in enumerate(rows)]
    stage = _staging(plane, keys, group_batches=3, dtype=torch.bfloat16)
    # Three stored batches of the largest entry, two rows of 4 x 8 in bf16.
    assert stage.capacity_bytes == 3 * (2 * 4 * 8) * 2
    assert stage.reserve_host_bytes == stage.capacity_bytes + mmap.PAGESIZE
    with stage.held():
        assert stage.reserve_host_bytes == 0
        incoming = stage.incoming(keys, device="cpu")
        expected = _stack_to_device([plane[key] for key in keys], device="cpu")
        assert incoming.dtype == expected.dtype and torch.equal(
            incoming.view(torch.int16), expected.view(torch.int16))
        stacked = stage.boundaries(boundaries, device="cpu", dtype=torch.bfloat16)
        reference = _stack_to_device(boundaries, device="cpu", dtype=torch.bfloat16)
        assert torch.equal(stacked.view(torch.int16), reference.view(torch.int16))
        held = stage._buffer.untyped_storage().data_ptr()
        for result in (incoming, stacked):
            assert result.untyped_storage().data_ptr() != held
        # One entry of a kept plane moves as it always moved.
        assert stage.incoming(keys[:1], device="cpu") is plane[keys[0]]
        with pytest.raises(RuntimeError, match="admitted"):
            stage.boundaries([torch.zeros(4, 4, 8)] * 3, device="cpu", dtype=torch.float32)
    assert stage.reserve_host_bytes == stage.capacity_bytes + mmap.PAGESIZE
    with pytest.raises(RuntimeError, match="not held"):
        stage.incoming(keys, device="cpu")


def test_plane_staging_stores_owned_rows_on_a_kept_plane_and_the_scratch(tmp_path):
    """``store`` writes each stored batch's rows back, as the capture did.

    A kept plane gets owned host tensors that later reuse of the buffer
    cannot change; the scratch gets its slots written from the buffer and
    reads back the same bytes.
    """
    from prismaquant.perturbed_x_cache import ExactCotangentScratch

    gradient = _bits((3, 1024), torch.float32, 7)
    kept = {(0, index): torch.zeros(count, 1024) for index, count in enumerate((1, 2))}
    stage = _staging(kept, list(kept), group_batches=2, dtype=torch.float32)
    with stage.held():
        stage.store(list(kept), [1, 2], gradient)
        stage.incoming(list(kept), device="cpu").fill_(0)
        stage._buffer.fill_(255)
    assert torch.equal(kept[0, 0].view(torch.int32), gradient[:1].view(torch.int32))
    assert torch.equal(kept[0, 1].view(torch.int32), gradient[1:].view(torch.int32))
    with stage.held(), pytest.raises(RuntimeError, match="wrong rows"):
        stage.store(list(kept), [1, 1], gradient)

    records = [{"name": f"cotangent-0-{index}", "shape": [1, 1024], "dtype": "torch.float32",
                "tensor_bytes": 4096} for index in range(3)]
    arena = ExactCotangentScratch(records, directory=tmp_path, max_bytes=1 << 20)
    try:
        keys = [(0, 0), (0, 1), (0, 2)]
        for key in keys:
            arena[key] = torch.zeros(1, 1024)
        stage = _staging(arena, keys, group_batches=3, dtype=torch.float32)
        with stage.held():
            stage.store(keys, [1, 1, 1], gradient)
            stacked = stage.incoming(keys, device="cpu")
        assert torch.equal(stacked.view(torch.int32), gradient.view(torch.int32))
        for row, key in enumerate(keys):
            assert torch.equal(arena[key].view(torch.int32),
                               gradient[row:row + 1].view(torch.int32))
    finally:
        arena.close()
