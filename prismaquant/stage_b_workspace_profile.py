"""Measure the device workspace of a Stage B capture pass (PQ #1151).

The Stage B guard charges ``workspace_reserve_bytes`` for every stored batch a
capture pass carries (``joint_cost_quantum.replay_backward``). Until this
measurement, that reserve was a declared 16 GiB with no measurement behind it.
This module measures what one pass actually holds on the device, on a real
layer with real boundaries, and writes a JSON receipt.

The measurement is opt-in. :data:`PROFILE_ENV` names the receipt path;
without it nothing here runs. The quantum calls
:func:`profile_capture_workspace` at the start of probe 0's spill capture,
where the retained window, the source layer and the cotangent plane are
resident exactly as a production pass sees them, and stops the quantum with
:class:`CaptureWorkspaceProfiled` once the receipt is written.

The ladder runs in this order, and each step is admitted by the live guard
before it runs:

1. the window backward with no observer, one stored batch per pass, under the
   declared reserve;
2. the same with ``capture_batch`` stored batches per pass, under the first
   step's measured peak times the batch ratio;
3. the capture pass with its spill observer at one stored batch, under the
   declared reserve;
4. the capture pass at 2 and then 4 stored batches, each under the previous
   step's measured peak times the batch ratio.

A step the guard refuses is recorded with the refusal and ends the ladder,
because every later step is larger. A pass that stops on a free-memory floor
or an allocator refusal is recorded the same way, with the peak it reached as
a lower bound.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import threading
import time
from pathlib import Path

PROFILE_ENV = "PRISMAQUANT_STAGE_B_WORKSPACE_PROFILE"
GROUPS_ENV = "PRISMAQUANT_STAGE_B_WORKSPACE_PROFILE_GROUPS"
SCHEMA = "prismaquant.stage_b_capture_workspace_profile.v1"
DEFAULT_GROUPS = 16
MIN_GROUPS = 8
HOST_SAMPLE_INTERVAL_S = 0.01
#: The cgroup ``memory.stat`` fields every group's host peak reports first.
HOST_FIELDS = ("anon", "file", "shmem", "file_dirty", "file_writeback",
               "kernel", "sock", "pagetables", "slab_unreclaimable",
               "active_file", "inactive_file")


class _LadderDone(Exception):
    """Leaves the spill observer's context without ending the probe."""


class CaptureWorkspaceProfiled(Exception):
    """The measurement is written; the quantum stops without publishing."""

    @classmethod
    def found_in(cls, error):
        """This exception if ``error`` is one or was raised from one."""
        seen = set()
        while error is not None and id(error) not in seen:
            if isinstance(error, cls):
                return error
            seen.add(id(error))
            error = error.__cause__ or error.__context__
        return None

    def __init__(self, path, sha256):
        super().__init__(f"Stage B capture workspace profile written to {path} "
                         f"(sha256 {sha256})")
        self.path = str(path)
        self.sha256 = str(sha256)


def profile_request(environ=None):
    """The requested receipt path and group count, or ``None`` when unset."""
    environ = os.environ if environ is None else environ
    raw = environ.get(PROFILE_ENV)
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{PROFILE_ENV} must be an absolute path, got {raw!r}")
    groups = environ.get(GROUPS_ENV)
    count = DEFAULT_GROUPS if groups in (None, "") else int(groups)
    if count < MIN_GROUPS:
        raise ValueError(f"{GROUPS_ENV} must be at least {MIN_GROUPS}: routing "
                         "varies the transient, so a smaller sample says little")
    return {"path": path, "groups": count}


def ladder(capture_batch):
    """The measured settings, in admission order.

    Each entry is ``(name, observer, batch, basis)``. ``basis`` is either
    ``("declared", batch)`` -- the declared per-batch reserve times the batch
    -- or ``(earlier_name, ratio)`` -- that step's measured peak times the
    ratio.
    """
    capture_batch = int(capture_batch)
    if capture_batch < 1:
        raise ValueError("capture batch must be a positive integer")
    steps = [("window-backward-b1", False, 1, ("declared", 1))]
    if capture_batch > 1:
        steps.append((f"window-backward-b{capture_batch}", False, capture_batch,
                      ("window-backward-b1", capture_batch)))
    steps.append(("capture-b1", True, 1, ("declared", 1)))
    previous, batch = "capture-b1", 1
    while batch < capture_batch:
        following = min(2 * batch, capture_batch)
        name = f"capture-b{following}"
        steps.append((name, True, following, (previous, following // batch)))
        previous, batch = name, following
    return steps


def parse_memory_stat(text):
    values = {}
    for line in text.splitlines():
        key, _, raw = line.partition(" ")
        if raw.strip().lstrip("-").isdigit():
            values[key] = int(raw)
    return values


class HostPeakSampler:
    """Samples the cgroup at a fixed interval and keeps each window's peak.

    The peak of a window is the sample with the largest ``memory.current``;
    its whole ``memory.stat`` is kept, so the receipt shows what the charged
    bytes were made of at that instant.
    """

    def __init__(self, scope, interval_s=HOST_SAMPLE_INTERVAL_S):
        self.scope = Path(scope)
        self.interval_s = float(interval_s)
        self._lock = threading.Lock()
        self._window = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="stage-b-workspace-host",
                                        daemon=True)
        self.samples = 0

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()

    def _read(self):
        current = int((self.scope / "memory.current").read_text())
        stat = parse_memory_stat((self.scope / "memory.stat").read_text())
        return current, stat

    def _loop(self):
        while not self._stop.is_set():
            try:
                current, stat = self._read()
            except OSError:
                current, stat = None, None
            now = time.time()
            with self._lock:
                self.samples += 1
                window = self._window
                if window is not None and current is not None and (
                        window["peak"] is None or current > window["peak"]["current"]):
                    window["peak"] = {"unix": now, "current": current, "stat": stat}
            self._stop.wait(self.interval_s)

    def open(self):
        current, stat = self._read()
        with self._lock:
            self._window = {"peak": {"unix": time.time(), "current": current, "stat": stat},
                            "start": {"current": current, "stat": stat}}

    def close(self):
        current, stat = self._read()
        with self._lock:
            window, self._window = self._window, None
        if window["peak"] is None or current > window["peak"]["current"]:
            window["peak"] = {"unix": time.time(), "current": current, "stat": stat}
        peak = window["peak"]
        return {
            "start_current_bytes": window["start"]["current"],
            "peak_current_bytes": peak["current"],
            "peak_unix": peak["unix"],
            "peak_stat": {field: peak["stat"].get(field) for field in HOST_FIELDS},
            "peak_stat_all": peak["stat"],
        }


def _summary(values):
    return {"max": max(values), "min": min(values),
            "mean": statistics.fmean(values),
            "spread": max(values) - min(values),
            "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0}


def measure_group(run, group, *, device, host):
    """One pass over ``group`` with the device peak counters reset first."""
    import torch

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    host.open()
    started = time.time()
    failure = None
    try:
        run(group)
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        # A free-memory floor or an allocator refusal inside the pass is a
        # reading too: keep the peak it reached, and let the ladder end.
        failure = f"{type(exc).__name__}: {exc}"
        torch.cuda.synchronize(device)
    finished = time.time()
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    host_peak = host.close()
    return {
        "indices": [int(item[0]) for item in group],
        "tokens": int(sum(item[2].numel() // item[2].shape[-1] for item in group)),
        "start_unix": started, "end_unix": finished, "wall_s": finished - started,
        "allocated_before_bytes": allocated, "reserved_before_bytes": reserved,
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "allocated_delta_bytes": peak_allocated - allocated,
        "reserved_delta_bytes": peak_reserved - reserved,
        "host": host_peak,
        "failure": failure,
    }


def _admit(guard, label, reserve_device_bytes, *, device):
    """The live guard's verdict on one step, charged to the device side."""
    from .aura_cost import _release_streamed_anchor_allocator_cache

    _release_streamed_anchor_allocator_cache(device)
    # The workspace is device memory, so it is charged to the device side
    # when the guard holds one; a guard without a device envelope takes it
    # as a plain reservation, which is the stricter charge.
    split = ({"reserve_device_bytes": int(reserve_device_bytes)}
             if guard.device_bytes is not None
             else {"reserve_bytes": int(reserve_device_bytes)})
    try:
        observed = guard.check(label, **split)
    except RuntimeError as exc:
        return {"admitted": False, "refusal": str(exc), "reserve_device_bytes":
                int(reserve_device_bytes), "guard": dict(guard.last or {})}
    return {"admitted": True, "reserve_device_bytes": int(reserve_device_bytes),
            "guard": dict(observed)}


def run_ladder(items, *, run_group, observed, guard, declared_workspace_bytes,
               capture_reserve_bytes, capture_batch, groups, device, host):
    """Run :func:`ladder` over ``items`` and return the settings measured.

    ``run_group(group, observer)`` is the quantum's own capture-group pass;
    ``observed()`` opens the spill observer's context. Every step uses the
    first ``groups * batch`` items, cut into consecutive groups of ``batch``.
    """
    settings, measured = [], {}
    steps = ladder(capture_batch)
    plain = [step for step in steps if not step[1]]
    captured = [step for step in steps if step[1]]

    def run_step(step, observer):
        name, with_observer, batch, basis = step
        reference, factor = basis
        if reference == "declared":
            reserve = int(declared_workspace_bytes) * int(factor)
            basis_text = (f"declared workspace_reserve_bytes {declared_workspace_bytes} "
                          f"x {factor}")
        else:
            reserve = measured[reference] * int(factor)
            basis_text = f"measured peak of {reference} ({measured[reference]}) x {factor}"
        if with_observer:
            reserve += int(capture_reserve_bytes)
            basis_text += f" + spill capture reserve {capture_reserve_bytes}"
        admission = _admit(guard, f"stage_b_workspace_profile:{name}", reserve,
                           device=device)
        admission["basis"] = basis_text
        setting = {"name": name, "observer": with_observer, "capture_batch": batch,
                   "admission": admission}
        settings.append(setting)
        if not admission["admitted"]:
            return False
        chunk = items[:groups * batch]
        cut = [chunk[start:start + batch] for start in range(0, len(chunk), batch)]
        records = []
        for group in cut:
            records.append(measure_group(lambda group: run_group(group, observer), group,
                                         device=device, host=host))
            if records[-1].get("failure"):
                break
        allocated = [record["allocated_delta_bytes"] for record in records]
        reserved = [record["reserved_delta_bytes"] for record in records]
        peak = max(max(allocated), max(reserved))
        setting.update(groups=records,
                       allocated_delta=_summary(allocated),
                       reserved_delta=_summary(reserved),
                       host_peak_current=_summary(
                           [record["host"]["peak_current_bytes"] for record in records]))
        if records[-1].get("failure"):
            # The pass stopped before its end, so its peak only bounds the
            # true one from below; it prices nothing.
            setting.update(failure=records[-1]["failure"], lower_bound_peak_bytes=peak)
            return False
        measured[name] = peak
        setting.update(peak_bytes=peak, peak_per_batch_bytes=-(-peak // batch))
        return True

    for step in plain:
        if not run_step(step, None):
            return settings, False
    # Leave the observer's context by an exception, never normally: a normal
    # exit ends the probe's capture, which validates a spill this ladder only
    # partly wrote. The spill is discarded with the quantum either way.
    complete = True
    try:
        with observed() as observer:
            for step in captured:
                if not run_step(step, observer):
                    complete = False
                    break
            raise _LadderDone()
    except _LadderDone:
        pass
    return settings, complete


def write_profile(path, profile):
    from .cost_stage_checkpoint import atomic_write_bytes

    raw = (json.dumps(profile, sort_keys=True, indent=2, allow_nan=False,
                      default=str) + "\n").encode()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(Path(path), raw)
    return hashlib.sha256(raw).hexdigest()


def profile_capture_workspace(request, *, storage, batches, layer, run_group, observed,
                              guard, declared_workspace_bytes, capture_reserve_bytes,
                              capture_batch, device, identity):
    """Measure the ladder at the start of probe 0's spill capture, then stop.

    Reads the first ``groups * capture_batch`` boundaries of the layer under
    the spill phase the caller has entered, keeps its own copies (a prefetch
    window releases its leases when it closes), runs :func:`run_ladder`, and
    raises :class:`CaptureWorkspaceProfiled` with the receipt's path and
    digest.
    """
    import torch

    from .cost_streaming import prefetched_boundary_batches

    if guard is None:
        raise RuntimeError("the capture workspace profile needs the live capture guard")
    groups = int(request["groups"])
    needed = groups * int(capture_batch)
    if needed > len(batches):
        raise RuntimeError(f"the profile needs {needed} stored batches; the layer has "
                           f"{len(batches)}")
    started = time.time()
    items = []
    with prefetched_boundary_batches(storage, batches, layer) as reverse_batches:
        for item in reverse_batches:
            items.append((item[0], item[1], item[2].clone(), item[3]))
            if len(items) == needed:
                break
    read_s = time.time() - started
    with HostPeakSampler(guard.scope) as host:
        settings, complete = run_ladder(
            items, run_group=run_group, observed=observed, guard=guard,
            declared_workspace_bytes=declared_workspace_bytes,
            capture_reserve_bytes=capture_reserve_bytes, capture_batch=capture_batch,
            groups=groups, device=device, host=host)
        host_samples = host.samples
    admitted = [setting for setting in settings if "peak_bytes" in setting]
    per_batch = max((setting["peak_per_batch_bytes"] for setting in admitted), default=None)
    profile = {
        "schema": SCHEMA,
        "identity": dict(identity),
        "started_unix": started, "finished_unix": time.time(),
        "boundary_read_s": read_s,
        "groups_per_setting": groups,
        "capture_batch": int(capture_batch),
        "declared_workspace_reserve_bytes": int(declared_workspace_bytes),
        "spill_capture_reserve_bytes": int(capture_reserve_bytes),
        "device": {"name": torch.cuda.get_device_name(device),
                   "torch": str(torch.__version__), "cuda": torch.version.cuda,
                   "envelope_bytes": guard.device_bytes},
        "cgroup": {"scope": str(guard.scope), "cap_bytes": int(guard.cap_bytes),
                   "margin_bytes": int(guard.margin_bytes),
                   "host_sample_interval_s": HOST_SAMPLE_INTERVAL_S,
                   "host_samples": host_samples},
        "ladder_complete": complete,
        "settings": settings,
        "measured": {
            "workspace_per_batch_bytes": per_batch,
            "basis": ("the largest per-stored-batch device peak over every admitted "
                      "setting: max(allocated delta, reserved delta) of one pass, "
                      "divided by its stored batches and rounded up"),
        },
    }
    sha256 = write_profile(request["path"], profile)
    raise CaptureWorkspaceProfiled(request["path"], sha256)


# ---------------------------------------------------------------- chain phase
#: The receipt path of a chain-phase measurement (PQ #1163). Set, the quantum
#: measures every chain roll in-process and stops after the chain.
CHAIN_PROFILE_ENV = "PRISMAQUANT_STAGE_B_CHAIN_PROFILE"
#: A JSON object describing the chain owner the measurement was admitted
#: under, recorded verbatim in the receipt.
CHAIN_OWNER_ENV = "PRISMAQUANT_STAGE_B_CHAIN_PROFILE_OWNER"
CHAIN_SCHEMA = "prismaquant.stage_b_chain_workspace_profile.v1"
MEMINFO_SAMPLE_INTERVAL_S = 0.1


class ChainWorkspaceProfiled(CaptureWorkspaceProfiled):
    """The chain measurement is written; the quantum stops after its chain."""

    def __init__(self, path, sha256):
        Exception.__init__(self, f"Stage B chain workspace profile written to {path} "
                                 f"(sha256 {sha256})")
        self.path = str(path)
        self.sha256 = str(sha256)


def _mem_available_bytes():
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable")


class _MemAvailableFloor:
    """The box's lowest MemAvailable over a window, sampled on a thread.

    On GB10 unified memory, device allocations are host memory, so
    MemAvailable is the box-level reading of what a roll takes.
    """

    def __init__(self, interval_s=MEMINFO_SAMPLE_INTERVAL_S):
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.minimum = None
        self.samples = 0
        self._thread = threading.Thread(target=self._loop, name="stage-b-chain-meminfo",
                                        daemon=True)

    def _sample(self):
        value, now = _mem_available_bytes(), time.time()
        with self._lock:
            self.samples += 1
            if self.minimum is None or value < self.minimum["bytes"]:
                self.minimum = {"bytes": value, "unix": now}

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._sample()
            except (OSError, RuntimeError):
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()
        self._sample()


def chain_roll_owner_bytes(roll):
    """The owner bytes one measured chain roll states (PQ #1163).

    ``bytes`` is the roll's device workspace: the larger of the allocated
    and reserved deltas over the CUDA allocator's reading right after the
    roll's admission released the cache, which is what the guard charges
    before the roll. ``device_resident_bytes`` is the CUDA reservation that
    admission read, and ``host_committed_bytes`` the larger of the cgroup's
    committed bytes at that admission and at the roll's cgroup peak
    (:func:`prismaquant.memory_management.committed_cgroup_bytes`). Together
    they are every owner resident in the chain phase as the guard reads it.
    The workspace includes the successor source bytes that land on the
    device during the roll: the chain leaves that read in flight (PQ #1166).
    """
    from .memory_management import committed_cgroup_bytes

    layer = roll["layer"]
    admission, host = roll.get("admission"), roll.get("host")
    if admission is None:
        raise ValueError(f"chain roll of layer {layer} has no admission reading, so its "
                         "resident owners were not measured")
    if host is None:
        raise ValueError(f"chain roll of layer {layer} has no cgroup reading, so its "
                         "host committed bytes were not measured")
    at_peak = committed_cgroup_bytes(int(host["peak_current_bytes"]), host["peak_stat_all"])
    return {"bytes": max(int(roll["allocated_delta_bytes"]), int(roll["reserved_delta_bytes"])),
            "device_resident_bytes": int(admission["cuda_reserved_bytes"]),
            "host_committed_bytes": max(int(admission["cgroup_committed_bytes"]), at_peak)}


class ChainRollProfile:
    """Measures each chain roll of one quantum in-process (PQ #1163).

    For every roll it records the CUDA allocator's allocated and reserved
    peaks over the reading taken right after the roll's admission released
    the cache, the roll's wall time, the cgroup's peak and the box's lowest
    MemAvailable. :meth:`finish` writes the receipt, releases the cache the
    chain left and records how much it held, then stops the quantum with
    :class:`ChainWorkspaceProfiled`. A roll that fails writes the receipt
    with its failure and the peaks it reached as lower bounds, then
    re-raises, so the row still fails.
    """

    @classmethod
    def requested(cls, *, guard, device, identity, environ=None):
        environ = os.environ if environ is None else environ
        raw = environ.get(CHAIN_PROFILE_ENV)
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            raise ValueError(f"{CHAIN_PROFILE_ENV} must be an absolute path, got {raw!r}")
        owner = environ.get(CHAIN_OWNER_ENV)
        return cls(path, guard=guard, device=device, identity=identity,
                   owner=None if not owner else json.loads(owner))

    def __init__(self, path, *, guard, device, identity, owner=None):
        self.path = Path(path)
        # The quantum's identity, chain regime included, goes into every
        # receipt this profile writes, a failed roll's too.
        self.identity = dict(identity)
        self.guard = guard
        self.device = device
        self.owner = owner
        self.rolls = []
        self.started = time.time()

    def _profile(self, identity, *, released=None):
        import torch

        measured = {str(roll["layer"]): {
            "workspace_bytes": max(roll["allocated_delta_bytes"], roll["reserved_delta_bytes"]),
            "complete": roll["failure"] is None} for roll in self.rolls}
        owners = {str(roll["layer"]): dict(chain_roll_owner_bytes(roll),
                                           complete=roll["failure"] is None)
                  for roll in self.rolls
                  if roll.get("admission") is not None and roll.get("host") is not None}
        return {
            "schema": CHAIN_SCHEMA,
            "identity": dict(identity),
            "admitted_owner": self.owner,
            "started_unix": self.started, "finished_unix": time.time(),
            "device": {"name": torch.cuda.get_device_name(self.device),
                       "torch": str(torch.__version__), "cuda": torch.version.cuda,
                       "envelope_bytes": getattr(self.guard, "device_bytes", None)},
            "cgroup": (None if self.guard is None else
                       {"scope": str(self.guard.scope), "cap_bytes": int(self.guard.cap_bytes),
                        "margin_bytes": int(self.guard.margin_bytes),
                        "host_sample_interval_s": HOST_SAMPLE_INTERVAL_S}),
            "meminfo_sample_interval_s": MEMINFO_SAMPLE_INTERVAL_S,
            "rolls": self.rolls,
            "released_after_chain": released,
            "measured": {
                "workspace_bytes_by_layer": measured,
                "owners_by_layer": owners,
                "basis": ("max(allocated delta, reserved delta) of one chain roll over the "
                          "CUDA allocator's reading right after the roll's admission "
                          "released the cache; this is the quantity the guard charges "
                          "before the roll (reserve_device_bytes)"),
            },
        }

    def _write(self, profile):
        return write_profile(self.path, profile)

    def measure(self, layer, run, *, admission, reserve_device_bytes):
        import torch

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        host = None if self.guard is None else HostPeakSampler(self.guard.scope)
        failure, result = None, None
        with (host if host is not None else _nullcontext()), _MemAvailableFloor() as floor:
            if host is not None:
                host.open()
            started = time.time()
            try:
                result = run()
                torch.cuda.synchronize(self.device)
            except BaseException as exc:
                failure = exc
            finished = time.time()
            host_peak = None if host is None else host.close()
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        self.rolls.append({
            "layer": int(layer), "start_unix": started, "end_unix": finished,
            "wall_s": finished - started, "backwards": result,
            "admission": None if admission is None else dict(admission),
            "reserve_device_bytes": reserve_device_bytes,
            "allocated_before_bytes": allocated, "reserved_before_bytes": reserved,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "allocated_delta_bytes": peak_allocated - allocated,
            "reserved_delta_bytes": peak_reserved - reserved,
            "host": host_peak, "mem_available_min": floor.minimum,
            "mem_available_samples": floor.samples,
            "failure": None if failure is None else f"{type(failure).__name__}: {failure}",
        })
        if failure is not None:
            # The peaks bound the roll from below; the receipt is written
            # and the row still fails with the roll's own error.
            self._write(self._profile(dict(self.identity, layer_failed=int(layer))))
            raise failure
        return result

    def finish(self):
        import torch

        torch.cuda.synchronize(self.device)
        before = torch.cuda.memory_reserved(self.device)
        mem_before = _mem_available_bytes()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_reserved(self.device)
        released = {"reserved_before_bytes": before, "reserved_after_bytes": after,
                    "released_bytes": before - after,
                    "mem_available_before_bytes": mem_before,
                    "mem_available_after_bytes": _mem_available_bytes()}
        sha256 = self._write(self._profile(self.identity, released=released))
        raise ChainWorkspaceProfiled(self.path, sha256)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_exc):
        return False
