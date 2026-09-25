#!/usr/bin/env python3
"""PQ #1246 measurement: the host traffic of Stage B capture groups.

Run with the working directory at the root of a PrismaQuant checkout; the
same file measures the checkout before the fix and after it. It builds the
one-pass spill fixture of ``tests/test_stageb_one_pass_spill.py`` at a chosen
model width and token count (16 MiB cotangent slots by default, the size of
a campaign plane's slots), runs Stage A once, then runs layer 1's quantum
under the spill, its cotangent plane in an ``O_DIRECT`` scratch, at a chosen
capture batch, ``--repeats`` times. For every capture group it records:

* the job cgroup's ``memory.stat`` deltas (``thp_fault_alloc``,
  ``pgfault``, direct reclaim) and the host's ``/proc/vmstat`` THP and
  compaction deltas, from the group's start to the next group's start, or
  to its pass's end for the last group;
* the capture thread's host allocations of at least one slot (torch
  profiler with ``profile_memory``) and their enclosing ops;
* the time at the three sites: the scratch reads, the stacks of the
  incoming cotangents and layer inputs, and the store of the input
  cotangent, from ``record_function`` ranges around the functions that
  implement each site in this checkout.

It digests Stage A's receipt, the quantum's payload (its costs, statistics
and the rest, key by key) and each capture pass's plane, so the runs of two
checkouts compare bit for bit. Writes one JSON file.

The records bind the digest of the whole ``prismaquant`` package
(``_production_cache_source_sha256``): every cost row carries it in its
probe identity, and Stage A's receipt carries it as the implementation
digest. A fix changes that digest by construction, so the harness records
the checkout's real digest and then pins the function to one constant for
the whole run, in both checkouts. Everything else in the payload then
compares byte for byte, except three metadata entries that bind the run's
own Stage A session (``adjoint_slice_sha256``, ``checkpoint_identity_sha256``
and ``distributed_quantum``), which differ between any two runs of one
checkout.
"""
from __future__ import annotations

import argparse
import bisect
import contextlib
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

CHECKOUT = Path.cwd()
sys.path[:0] = [str(CHECKOUT), str(CHECKOUT / "tests")]

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import prismaquant.aura_cost as aura  # noqa: E402
import prismaquant.joint_adjoint_checkpoints as checkpoints  # noqa: E402
import prismaquant.joint_cost_quantum as jcq  # noqa: E402
import prismaquant.joint_replay_spill as spill_mod  # noqa: E402
import prismaquant.perturbed_x_cache as scratch_mod  # noqa: E402
from prismaquant.cost_stage_checkpoint import canonical_json_sha256  # noqa: E402
from prismaquant.joint_adjoint_checkpoints import chain_layers_for  # noqa: E402
from prismaquant.joint_adjoint_slices import adjoint_slice_sha256, stage_a_slice  # noqa: E402
from prismaquant.joint_retained_window_plan import (  # noqa: E402
    EXECUTION_SCHEMA, RetainedWindowBudget)

import test_joint_cost_quantum_runtime as rt  # noqa: E402
import test_stageb_one_pass_spill as spill_tests  # noqa: E402
from test_joint_operator_windows import policy as operator_policy  # noqa: E402
from test_stageb_cotangent_scratch import _direct_io_supported  # noqa: E402
from test_streamed_cost_checkpoints import _model_identity  # noqa: E402

GROUP_MARKER = "pq1246/group"
PREFETCH_RANGE = "aura.exact_activation.prefetch"
MEMORY_EVENT = "[memory]"
SITE_READ, SITE_STACK, SITE_STORE = "pq1246/site1_read", "pq1246/site2_stack", "pq1246/site3_store"
SCRATCH_WRITE = "pq1246/scratch_write"
LAYER = 1
CAMPAIGN_ACCUMULATION = "accumulation=operator_gemm,chunk_rows=65536"
CGROUP_KEYS = ("thp_fault_alloc", "thp_collapse_alloc", "pgfault", "pgmajfault",
               "pgscan_direct", "pgsteal_direct", "anon", "anon_thp", "file")
VMSTAT_KEYS = ("thp_fault_alloc", "thp_fault_fallback", "compact_stall", "compact_fail",
               "compact_success", "pgfault")


# -- the scaled fixture ------------------------------------------------------

class Scale:
    """The fixture's sizes. ``k`` scales every byte bound the spill suite pins
    at its width of 16, so the plan's relations hold at this width."""

    def __init__(self, args):
        self.width = args.width
        self.tokens = args.tokens
        self.samples = args.samples
        self.window = args.window
        self.k = max(1, args.width // 16)
        self.dtype = spill_tests.DTYPE
        self.slot_bytes = args.tokens * args.width * torch.empty((), dtype=self.dtype).element_size()
        self.auxiliary_bytes = args.auxiliary_gib << 30
        self.artifact_bytes = args.artifact_gib << 30
        self.scratch_max_bytes = args.scratch_gib << 30
        self.spill_ceiling = args.spill_gib << 30
        self.source_bytes = self.k << 20


def policy_budget(scale):
    """``_policy_budget`` of the spill suite, every byte bound times ``k``."""
    k, inter, width = scale.k, spill_tests.INTER, scale.width
    single = inter * width * 4 * 2
    policy = operator_policy(
        max_statistics_bytes=k << 20, max_candidate_bytes=k << 20,
        max_render_resident_bytes=k << 20, max_load_buffer_bytes=k << 20,
        workspace_reserve_bytes=k << 20, max_replay_cotangent_bytes=k << 20)
    budget = RetainedWindowBudget(
        k * (50 << 20), k << 20, k << 20, k << 20, k << 20,
        k << 20, k << 20, k << 20, k << 20, inter * width * 4,
        3 * single, k * (4 << 20), 16)
    grown = scale.auxiliary_bytes - budget.auxiliary_reserve_bytes
    if grown > 0:
        budget = RetainedWindowBudget(**{
            **{name: getattr(budget, name) for name in RetainedWindowBudget.__dataclass_fields__},
            "auxiliary_reserve_bytes": scale.auxiliary_bytes,
            "physical_limit_bytes": budget.physical_limit_bytes + grown})
    retained = {"schema": EXECUTION_SCHEMA, "budget": budget.as_dict(),
                "source_reserve_bytes": scale.source_bytes,
                "source_loading_reserve_bytes": 2 * scale.source_bytes}
    return policy, budget, retained


def make_execution(scale):
    def execution(root):
        policy, _budget, retained = policy_budget(scale)
        boundary = rt._boundary_policy(root / "boundaries", window=scale.window)
        boundary["max_resident_bytes"] = (
            scale.window * (1 + spill_tests.N_PROBES) * scale.slot_bytes + scale.slot_bytes)
        boundary["max_auxiliary_bytes"] = scale.auxiliary_bytes
        boundary["max_artifact_bytes"] = scale.artifact_bytes
        return {"n_probes": spill_tests.N_PROBES, "seed_base": 7000, "probe_microbatch": 1,
                "token_scope": "all", "temperature": 1.0, "production_act_scales": "0",
                "boundary_storage": boundary, "operator_windows": dict(policy),
                "retained_operator_windows": retained, "min_free_gib": 0,
                "device_envelope_bytes": None}
    return execution


def make_calibration(scale):
    def calibration():
        generator = torch.Generator().manual_seed(1246)
        return torch.randint(0, spill_tests.VOCAB, (scale.samples, scale.tokens),
                             generator=generator)
    return calibration


def build_campaign(root, device, scale, execution, calibration):
    """Stage A once, and the sealed records the quantum consumes."""
    from prismaquant.joint_cost_stage_a import run_adjoint_capture_core
    from prismaquant.joint_statistics_replay import preflight_joint_operator_admission

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
        patch.setenv("PRISMAQUANT_DEV_MODE", "1")
        torch.manual_seed(715)
        state = {key: tensor.detach().clone()
                 for key, tensor in spill_tests._MoELM().state_dict().items()}
        model, context, runner = spill_tests._runner(state, device)
        linears = spill_tests._targets(model, runner.profile)
        weights = {(qname, fmt): module.weight.detach().to("cpu").clone() + 0.03125
                   for qname, module in linears.items() for fmt in spill_tests.RENDER_FORMATS}
        cache, linears = spill_tests._prepared(model, context, runner, weights, root / "shared")
        formats_by_qname = {qname: list(spill_tests.FORMATS) for qname in linears}
        policy, budget, _retained = policy_budget(scale)
        names_by_layer = {layer: sorted(n for n in linears
                                        if runner.layer_index_for_qname(n) == layer)
                          for layer in (0, 1)}
        preflight = preflight_joint_operator_admission(
            names_by_layer, linears,
            {qname: list(spill_tests.RENDER_FORMATS) for qname in linears},
            cache, policy=policy, retained_budget=budget, source_bytes=scale.source_bytes)
        output_root = root / "campaign"
        _model_a, _context_a, runner_a = spill_tests._runner(state, device)
        started = time.time()
        receipt = run_adjoint_capture_core(
            runner_a, calibration(), execution=execution(root / "exec"),
            output_root=output_root, stride=2,
            source_model_identity=_model_identity("joint-source"),
            unit_roster_sha256=rt._hex("a"), plan_sha256=rt._hex("d"),
            prepared_sha256=rt._hex("e"), read_manifest_sha256=rt._hex("f"),
            implementation_sha256=aura._aura_source_sha256())
        stage_a_s = time.time() - started
    if [c["boundary"] for c in receipt["checkpoints"]] != [2]:
        raise RuntimeError(f"unexpected Stage A checkpoints {receipt['checkpoints']}")
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
    if records[LAYER]["adjoint"]["chain_layers"] != []:
        raise RuntimeError("layer 1 walks a chain; the measurement needs a capture-only quantum")
    return SimpleNamespace(root=root, state=state, weights=weights, receipt=receipt,
                           records=records, slices=slices, output_root=output_root,
                           device=device, formats_by_qname=formats_by_qname,
                           preflight=preflight, stage_a_s=stage_a_s,
                           windows=[len(w.original_full_target_names)
                                    for w in preflight[LAYER]])


# -- digests -----------------------------------------------------------------

def _feed(value, h):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to("cpu").contiguous()
        h.update(f"T{tensor.dtype}{tuple(tensor.shape)}|".encode())
        if tensor.numel():
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        h.update(f"A{array.dtype}{array.shape}|".encode())
        h.update(array.tobytes())
    elif isinstance(value, dict):
        h.update(b"{")
        for key in sorted(value, key=repr):
            _feed(key, h)
            h.update(b":")
            _feed(value[key], h)
            h.update(b",")
        h.update(b"}")
    elif isinstance(value, (list, tuple)):
        h.update(b"[" if isinstance(value, list) else b"(")
        for item in value:
            _feed(item, h)
            h.update(b",")
        h.update(b"]")
    elif isinstance(value, float):
        h.update(b"f" + struct.pack("<d", value))
    elif isinstance(value, (np.floating, np.integer)):
        h.update(f"n{value.dtype}|".encode() + value.tobytes())
    else:
        h.update(f"{type(value).__name__}|{value!r}".encode())


def digest(value):
    h = hashlib.sha256()
    _feed(value, h)
    return h.hexdigest()


PINNED_SOURCE_SHA256 = hashlib.sha256(b"pq1246 pinned producer source").hexdigest()


def pin_producer_source():
    """Pin the package digest the records bind; return the checkout's real one.

    Rebinds ``_production_cache_source_sha256`` and ``_aura_source_sha256``
    on their home modules and on every loaded ``prismaquant`` module that
    imported them by name. A module imported later reads the pinned
    attribute from its home module.
    """
    import prismaquant.production_weight_cache as pwc

    real = pwc._production_cache_source_sha256()
    originals = {"_production_cache_source_sha256": pwc._production_cache_source_sha256,
                 "_aura_source_sha256": aura._aura_source_sha256}
    pinned = {"_production_cache_source_sha256": lambda package_root=None: PINNED_SOURCE_SHA256,
              "_aura_source_sha256": lambda: PINNED_SOURCE_SHA256}
    rebound = []
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("prismaquant") or module is None:
            continue
        for attribute, original in originals.items():
            if getattr(module, attribute, None) is original:
                setattr(module, attribute, pinned[attribute])
                rebound.append(f"{name}.{attribute}")
    if aura._aura_source_sha256() != PINNED_SOURCE_SHA256:
        raise RuntimeError("the producer source digest is not pinned")
    return {"real": real, "pinned": PINNED_SOURCE_SHA256, "rebound": rebound}


def payload_digests(payload):
    """Key by key; the metadata entries one level deeper, since they hold receipts."""
    out = {}
    for key in sorted(payload, key=str):
        if key == "provenance" and isinstance(payload[key], dict):
            out[key] = {sub: digest(v) for sub, v in sorted(payload[key].items(), key=str)}
        else:
            out[str(key)] = digest(payload[key])
    return out


# -- the job's counters --------------------------------------------------------

def _cgroup_dir():
    text = Path("/proc/self/cgroup").read_text()
    path = next(line.split("::", 1)[1] for line in text.splitlines() if line.startswith("0::"))
    directory = Path("/sys/fs/cgroup") / path.lstrip("/")
    while not (directory / "memory.stat").exists() and directory != Path("/sys/fs/cgroup"):
        directory = directory.parent
    return directory


class Counters:
    def __init__(self):
        self.cgroup = _cgroup_dir()
        self.stat_path = self.cgroup / "memory.stat"

    def describe(self):
        def read(name):
            try:
                return (self.cgroup / name).read_text().strip()
            except OSError as exc:
                return f"<{exc}>"
        return {"cgroup": str(self.cgroup), "memory.max": read("memory.max"),
                "procs": read("cgroup.procs").split(), "memory.current": read("memory.current")}

    def snapshot(self):
        wall = time.time()
        stat = dict(line.split() for line in self.stat_path.read_text().splitlines())
        vmstat = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
        current = int((self.cgroup / "memory.current").read_text())
        return {"t": wall, "current": current,
                "cgroup": {k: int(stat[k]) for k in CGROUP_KEYS if k in stat},
                "vmstat": {k: int(vmstat[k]) for k in VMSTAT_KEYS if k in vmstat}}


def _delta(later, earlier):
    return {"s": later["t"] - earlier["t"],
            "cgroup": {k: later["cgroup"][k] - earlier["cgroup"].get(k, 0)
                       for k in later["cgroup"]},
            "vmstat": {k: later["vmstat"][k] - earlier["vmstat"].get(k, 0)
                       for k in later["vmstat"]}}


# -- instrumentation -------------------------------------------------------------

def _ranged(label, function):
    def wrapper(*args, **kwargs):
        with torch.profiler.record_function(label):
            return function(*args, **kwargs)
    wrapper.__wrapped__ = function
    return wrapper


def instrument(patch, counters, fixed):
    """Group markers, per-site ranges, and a profiled, counted capture pass."""
    seen = SimpleNamespace(passes=[], scratches=[], in_capture=False, pass_marks=None,
                           direct={"read": 0, "write": 0},
                           bounced={"direct": 0, "bounced": 0})
    real_free_gib = aura._free_gib

    def marked_free_gib(*args, **kwargs):
        if seen.in_capture:
            seen.pass_marks.append(counters.snapshot())
            with torch.profiler.record_function(GROUP_MARKER):
                pass
        return real_free_gib(*args, **kwargs)

    patch.setattr(aura, "_free_gib", marked_free_gib)
    scratch_cls = scratch_mod.ExactCotangentScratch
    original_init = scratch_cls.__init__

    def recorded_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        seen.scratches.append(self)

    patch.setattr(scratch_cls, "__init__", recorded_init)
    original_direct_io = scratch_cls._direct_io

    def counted_direct_io(self, call, view, offset, size, what):
        seen.direct[what] += int(seen.in_capture)
        with torch.profiler.record_function(f"pq1246/io_{what}"):
            return original_direct_io(self, call, view, offset, size, what)

    patch.setattr(scratch_cls, "_direct_io", counted_direct_io)
    original_device_buffer = scratch_cls._device_buffer

    def counted_device_buffer(self, tensor, size):
        view, bounced = original_device_buffer(self, tensor, size)
        if seen.in_capture:
            seen.bounced["bounced" if bounced else "direct"] += 1
        return view, bounced

    patch.setattr(scratch_cls, "_device_buffer", counted_device_buffer)
    patch.setattr(scratch_cls, "__setitem__", _ranged(SCRATCH_WRITE, scratch_cls.__setitem__))
    if fixed:
        staging = checkpoints.PlaneHostStaging
        patch.setattr(scratch_cls, "read_into", _ranged(SITE_READ, scratch_cls.read_into))
        patch.setattr(staging, "incoming", _ranged(SITE_STACK, staging.incoming))
        patch.setattr(staging, "boundaries", _ranged(SITE_STACK, staging.boundaries))
        patch.setattr(staging, "store", _ranged(SITE_STORE, staging.store))
    else:
        patch.setattr(scratch_cls, "__getitem__", _ranged(SITE_READ, scratch_cls.__getitem__))
        patch.setattr(jcq, "_stack_to_device", _ranged(SITE_STACK, jcq._stack_to_device))
    original_capture = spill_mod.StageBReplaySpill.capture

    @contextlib.contextmanager
    def profiled_capture(self, probe_index, *args, **kwargs):
        with original_capture(self, probe_index, *args, **kwargs) as observer:
            seen.pass_marks = []
            start = counters.snapshot()
            seen.in_capture = True
            try:
                with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU],
                        profile_memory=True) as profile:
                    yield observer
            finally:
                seen.in_capture = False
            end = counters.snapshot()
            events = profile.profiler.kineto_results.events()
            plane = {}
            for scratch in seen.scratches:
                for key in sorted(scratch._written):
                    if key[0] == int(probe_index):
                        plane[f"{key[0]}-{key[1]}"] = digest(scratch[key])
            seen.passes.append({"probe": int(probe_index), "start": start, "end": end,
                                "marks": seen.pass_marks, "events": events, "plane": plane})

    patch.setattr(spill_mod.StageBReplaySpill, "capture", profiled_capture)
    return seen


# -- the profile, per group -------------------------------------------------------

def _union(intervals):
    """Sorted, disjoint cover of possibly nested or overlapping intervals."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _within(union, at):
    """Whether ``at`` falls in ``union``, a list ``_union`` returned."""
    index = bisect.bisect_right(union, (at, float("inf"))) - 1
    return index >= 0 and union[index][0] <= at <= union[index][1]


def _clip(start, end, lo, hi):
    return max(0, min(end, hi) - max(start, lo))


def analyse_pass(record, *, threshold):
    """Per group: allocations of a slot or more by where, and the time per site."""
    from torch.autograd import DeviceType

    events = record["events"]
    markers = sorted((e.start_ns(), e.start_thread_id()) for e in events
                     if e.name() == GROUP_MARKER)
    if not markers:
        raise RuntimeError("a capture pass marked no group")
    threads = {thread for _start, thread in markers}
    if len(threads) != 1:
        raise RuntimeError(f"group starts on several threads: {sorted(threads)}")
    (thread,) = threads
    starts = [start for start, _thread in markers]
    last = max(e.end_ns() for e in events if e.start_thread_id() == thread)
    bounds = list(zip(starts, starts[1:] + [last]))
    own = [e for e in events if e.start_thread_id() == thread]
    ranges = {}
    for e in own:
        if e.name().startswith("pq1246/") or e.name() == PREFETCH_RANGE:
            ranges.setdefault(e.name(), []).append((e.start_ns(), e.end_ns()))
    for name in ranges:
        ranges[name].sort()
    prefetch = _union(ranges.get(PREFETCH_RANGE, []))
    sites = _union(ranges.get(SITE_READ, []) + ranges.get(SITE_STACK, [])
                   + ranges.get(SITE_STORE, []))
    stacks = _union(ranges.get(SITE_STACK, []))
    stores = _union(ranges.get(SITE_STORE, []))
    ops = sorted((e.start_ns(), e.end_ns(), e.name()) for e in own
                 if e.device_type() == DeviceType.CPU and e.name().startswith("aten::"))
    # Top-level aten ops: not inside another aten op on this thread.
    top, open_end = [], -1
    for start, end, name in ops:
        if start >= open_end:
            top.append((start, end, name))
            open_end = end
    allocations = [(e.start_ns(), e.nbytes()) for e in own
                   if e.name() == MEMORY_EVENT and e.device_type() == DeviceType.CPU
                   and e.nbytes() > 0]
    groups = []
    for index, (lo, hi) in enumerate(bounds):
        group = {"large_at_sites": 0, "large_in_prefetch": 0, "large_other": 0,
                 "large_bytes_at_sites": 0, "allocations": 0, "ns": hi - lo,
                 "site_ns": {"read": 0, "stack": 0, "store": 0}, "other_large": []}
        for at, nbytes in allocations:
            if not lo <= at < hi:
                continue
            group["allocations"] += 1
            if nbytes < threshold:
                continue
            if _within(prefetch, at):
                group["large_in_prefetch"] += 1
            elif _within(sites, at) or any(s <= at <= e and name in ("aten::to", "aten::clone")
                                           for s, e, name in top):
                group["large_at_sites"] += 1
                group["large_bytes_at_sites"] += nbytes
            else:
                group["large_other"] += 1
                where = [name for s, e, name in top if s <= at <= e]
                group["other_large"].append((nbytes, where[-1] if where else "<none>"))
        read = sum(_clip(s, e, lo, hi) for s, e in ranges.get(SITE_READ, []))
        stack = sum(_clip(s, e, lo, hi) for s, e in ranges.get(SITE_STACK, []))
        store = sum(_clip(s, e, lo, hi) for s, e in ranges.get(SITE_STORE, []))
        # A stack range holds the reads it makes (after the fix) and a store
        # range the scratch writes it makes: time each site once.
        nested_read = sum(_clip(s, e, lo, hi) for s, e in ranges.get(SITE_READ, [])
                          if _within(stacks, s))
        nested_write = sum(_clip(s, e, lo, hi) for s, e in ranges.get(SCRATCH_WRITE, [])
                           if _within(stores, s))
        # Before the fix the store is inline: the copy to the host (an aten::to
        # that allocates on the host) and the clones, outside the site ranges.
        inline_store = 0
        for s, e, name in top:
            if not lo <= s < hi or _within(sites, s) or _within(prefetch, s):
                continue
            if name == "aten::clone" or (name == "aten::to" and any(
                    s <= at <= e and nbytes >= threshold for at, nbytes in allocations)):
                inline_store += e - s
        group["site_ns"] = {"read": read, "stack": stack - nested_read,
                            "store": store - nested_write + inline_store}
        groups.append(group)
    marks = record["marks"]
    stats = []
    for index in range(len(marks)):
        later = marks[index + 1] if index + 1 < len(marks) else record["end"]
        stats.append(_delta(later, marks[index]))
    if len(stats) != len(groups):
        raise RuntimeError(f"{len(stats)} counter marks for {len(groups)} profiled groups")
    for group, stat in zip(groups, stats):
        group["counters"] = stat
    return groups


# -- the run ----------------------------------------------------------------------

def run(args):
    fixed = hasattr(checkpoints, "PlaneHostStaging")
    producer_source = pin_producer_source()
    scale = Scale(args)
    spill_tests.WIDTH = scale.width  # the fixture's model and budgets read it
    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    device = spill_tests._device()
    execution = make_execution(scale)
    calibration = make_calibration(scale)
    counters = Counters()
    result = {"label": args.label, "fixed": fixed, "checkout": str(CHECKOUT),
              "device": str(device), "torch": torch.__version__,
              "env": {k: os.environ.get(k) for k in ("MIMALLOC_PURGE_DELAY",
                                                    "PRISMAQUANT_RELEASE_SOURCE_PAGES",
                                                    "CUDA_VISIBLE_DEVICES")},
              "thp": {name: Path(f"/sys/kernel/mm/transparent_hugepage/{name}").read_text().strip()
                      for name in ("enabled", "defrag")},
              "host": os.uname().nodename, "cgroup": counters.describe(),
              "producer_source_sha256": producer_source,
              "scale": {"width": scale.width, "tokens": scale.tokens, "samples": scale.samples,
                        "window": scale.window, "capture_batch": args.capture_batch,
                        "slot_bytes": scale.slot_bytes, "k": scale.k},
              "files": {name: hashlib.sha256((CHECKOUT / name).read_bytes()).hexdigest()
                        for name in ("prismaquant/joint_cost_quantum.py",
                                     "prismaquant/joint_adjoint_checkpoints.py",
                                     "prismaquant/perturbed_x_cache.py")}}
    started = time.time()
    campaign = build_campaign(work / "campaign", device, scale, execution, calibration)
    receipt = json.loads(json.dumps(campaign.receipt))
    result["stage_a"] = {"s": campaign.stage_a_s, "receipt_sha256": digest(receipt),
                         "receipt": receipt, "windows": campaign.windows,
                         "wall": [started, time.time()]}
    print(f"PQ1246 stage A done in {campaign.stage_a_s:.1f}s windows={campaign.windows}",
          flush=True)
    scratch_root = work / "cotangent-scratch"
    scratch_root.mkdir()
    if not _direct_io_supported(scratch_root):
        raise RuntimeError(f"{scratch_root} has no direct I/O on an 8 KiB grid")
    regime = f"capture_batch={args.capture_batch},{CAMPAIGN_ACCUMULATION}"
    repeats = []
    for repeat in range(args.repeats):
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("PRISMAQUANT_STAGED_RANGE_WAIT_S", "0")
            patch.delenv("PRISMAQUANT_STAGE_B_KERNEL_PROFILE", raising=False)
            patch.setenv("PRISMAQUANT_STAGE_B_COTANGENT_ROOT", str(scratch_root))
            patch.setenv("PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES", str(scale.scratch_max_bytes))
            patch.setattr(spill_tests, "_calibration", calibration)
            patch.setattr(spill_tests, "_execution", execution)
            seen = instrument(patch, counters, fixed)
            spill_root = work / f"spill-{repeat}"
            spill_root.mkdir()
            spill_tests._clear_output(campaign, LAYER)
            began = time.time()
            payload, state = spill_tests._quantum(
                campaign, patch, layer=LAYER, spill_root=spill_root,
                ceiling=scale.spill_ceiling, regime=regime)
            ended = time.time()
            if payload is None:
                raise RuntimeError(spill_tests._chain(state.error))
            if len(seen.scratches) != 1 or seen.scratches[0]._direct is None:
                raise RuntimeError("the plane is not in one direct-I/O cotangent scratch")
            slot = min(size for _o, size, _s, _d in seen.scratches[0]._slots.values())
            if slot != scale.slot_bytes:
                raise RuntimeError(f"slot {slot} != {scale.slot_bytes}")
            passes = []
            for record in seen.passes:
                groups = analyse_pass(record, threshold=scale.slot_bytes)
                # The pass's head: from the profile's start to its first group,
                # where a held staging buffer is allocated and written.
                passes.append({"probe": record["probe"], "plane": record["plane"],
                               "wall": [record["start"]["t"], record["end"]["t"]],
                               "pass": _delta(record["end"], record["start"]),
                               "head": _delta(record["marks"][0], record["start"]),
                               "current": {"start": record["start"]["current"],
                                           "end": record["end"]["current"],
                                           "file_end": record["end"]["cgroup"].get("file")},
                               "groups": groups})
            repeats.append({"wall": [began, ended], "passes": passes,
                            "direct_io_calls": seen.direct,
                            "direct_io_buffers": seen.bounced,
                            "direct_io_grid": list(seen.scratches[0]._direct),
                            "payload": payload_digests(payload),
                            "costs_rows": sum(len(rows) for rows in payload["costs"].values())})
            shutil.rmtree(spill_root, ignore_errors=True)
            print(f"PQ1246 repeat {repeat} done in {ended - began:.1f}s", flush=True)
    result["repeats"] = repeats
    result["wall"] = [started, time.time()]
    summary = summarize(repeats)
    result["summary"] = summary
    print("PQ1246 SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return result


def summarize(repeats):
    groups = [g for r in repeats for p in r["passes"] for g in p["groups"]]

    def stat(values):
        values = sorted(values)
        n = len(values)
        return {"n": n, "sum": sum(values), "mean": sum(values) / n if n else None,
                "min": values[0] if n else None, "median": values[n // 2] if n else None,
                "max": values[-1] if n else None}

    out = {"groups": len(groups)}
    for key in ("large_at_sites", "large_in_prefetch", "large_other", "allocations"):
        out[key] = stat([g[key] for g in groups])
    for key in ("read", "stack", "store"):
        out[f"site_{key}_ms"] = stat([g["site_ns"][key] / 1e6 for g in groups])
    out["sites_ms"] = stat([sum(g["site_ns"].values()) / 1e6 for g in groups])
    out["group_ms"] = stat([g["ns"] / 1e6 for g in groups])
    for key in CGROUP_KEYS:
        if key in ("anon", "anon_thp", "file"):
            continue
        out[f"cgroup_{key}"] = stat([g["counters"]["cgroup"].get(key, 0) for g in groups])
    for key in VMSTAT_KEYS:
        out[f"vmstat_{key}"] = stat([g["counters"]["vmstat"].get(key, 0) for g in groups])
    heads = [p["head"] for r in repeats for p in r["passes"]]
    for key in ("thp_fault_alloc", "pgfault", "pgscan_direct"):
        out[f"head_cgroup_{key}"] = stat([h["cgroup"].get(key, 0) for h in heads])
    out["direct_io_buffers"] = [r["direct_io_buffers"] for r in repeats]
    out["direct_io_grid"] = repeats[0]["direct_io_grid"] if repeats else None
    out["peak_current_bytes"] = max(max(p["current"]["start"], p["current"]["end"])
                                    for r in repeats for p in r["passes"])
    out["payload_identical_across_repeats"] = all(
        r["payload"] == repeats[0]["payload"] for r in repeats)
    out["plane_identical_across_repeats"] = all(
        [p["plane"] for p in r["passes"]] == [p["plane"] for p in repeats[0]["passes"]]
        for r in repeats)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--capture-batch", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--auxiliary-gib", type=int, default=4)
    parser.add_argument("--artifact-gib", type=int, default=8)
    parser.add_argument("--scratch-gib", type=int, default=2)
    parser.add_argument("--spill-gib", type=int, default=8)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("PRISMAQUANT_STAGED_RANGE_WAIT_S", "0")
    try:
        result = run(args)
    finally:
        if not args.keep_work:
            shutil.rmtree(args.work, ignore_errors=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True, indent=1, default=str))
    print(f"PQ1246 wrote {out}", flush=True)


if __name__ == "__main__":
    main()
