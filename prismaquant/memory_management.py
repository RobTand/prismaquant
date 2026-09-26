"""Shared memory controls for cache-heavy quantization passes."""
from __future__ import annotations

import gc
import os
import sys
import weakref
from pathlib import Path
from typing import Iterable

import torch
from . import io_spans


_BUDGET_EVICTORS: "weakref.WeakSet[object]" = weakref.WeakSet()


#: The floor a bounded row must keep available on the host, whatever else it
#: reserves. A row is bounded on the box it runs on, so a guard that let the
#: host be driven into reclaim to satisfy one allocation would be bounding the
#: cgroup rather than the box.
MIN_HOST_FLOOR_BYTES = 3*1024**3

#: ``MemAvailable`` when the Sparks hung on 2026-09-14: a vLLM A8 routed-expert
#: load (tessera#501) took both GB10s down at 8.5 GiB available, memory PSI
#: full 89 (PQ #1158). It is the measured edge of the failure the host floor
#: exists to refuse, so a floor at or below it admits that failure.
HANG_MEM_AVAILABLE_BYTES = int(8.5*1024**3)

#: The GB10 box watchdog's ``MemAvailable`` floor. The watchdog was set after
#: that hang, one per box, and acts below 16 GiB available or at memory PSI
#: full avg10 20 and above. It is fleet configuration: no repository holds it,
#: and nothing a row can read at run time publishes its floor (no PrismaBuild
#: offer field, host announcement or launch environment carries it). So the
#: value is restated here from the served receipts that record it, Tessera
#: ``docs/measurements/tessera-glm53-a4-stub-tp2-served-2026-09-14.md`` and
#: ``tessera-glm53-a4-stub-tp2-world-size-2026-09-15.md`` ("Memory watchdog").
#: Its scope is the two GB10s; no other box's watchdog is recorded. A watchdog
#: that moves its floor must move this line with it.
BOX_WATCHDOG_FLOOR_BYTES = 16*1024**3

#: The floor a capture guard holds unless its caller states one: the box
#: watchdog's, and this is PrismaQuant's one home of it. A lower floor admits
#: allocations the watchdog then ends from outside the row, with no refusal
#: receipt; the 8 GiB this replaced sat below the hang itself. A higher one
#: refuses memory the box lets a row hold. On an idle GB10 (115.638 GiB
#: available of 121.627 GiB, PrismaBuild
#: ``docs/gb10_memory_104_capacity_2026-09-07.md``) a row can hold
#: 115.638 - 16 = 99.638 GiB before this floor refuses it.
DEFAULT_HOST_FLOOR_BYTES = BOX_WATCHDOG_FLOOR_BYTES

#: The attribute a memory-guard callback carries when it can hold the CPU and
#: device sides of a future allocation apart. A callback without it keeps the
#: conservative single-budget arithmetic it was written against.
SEPARATE_RESERVATIONS = "separates_cpu_and_device_reservations"

#: The ``memory.stat`` keys the committed-memory definition reads. Every
#: cgroup v2 kernel publishes them; a stat without one of them refuses rather
#: than reading the missing key as zero.
COMMITTED_MEMORY_STAT_KEYS = ('anon', 'file', 'shmem', 'file_dirty', 'file_writeback')


def read_memory_stat(path) -> dict:
    """Parse one cgroup ``memory.stat`` file into ``{key: bytes}``."""
    stat = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise RuntimeError(f'cgroup memory.stat line is malformed: {line!r}')
        stat[fields[0]] = int(fields[1])
    return stat


def clean_file_bytes(stat) -> int:
    """File pages the kernel can drop without writing anything first.

    ``file`` counts every page-cache page charged to the cgroup, including
    tmpfs and shared memory (``shmem``) and pages still waiting to be written
    (``file_dirty``, ``file_writeback``). What remains is clean cache: a read
    of a checkpoint or an NFS source leaves it behind, and the kernel reclaims
    it before it would refuse an allocation at ``memory.max``.
    """
    missing = [key for key in COMMITTED_MEMORY_STAT_KEYS if key not in stat]
    if missing:
        raise RuntimeError(
            f'cgroup memory.stat lacks {missing}; the committed-memory '
            'definition cannot be computed without them')
    values = {key: stat[key] for key in COMMITTED_MEMORY_STAT_KEYS}
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise RuntimeError(f'cgroup memory.stat values are invalid: {values}')
    return max(0, values['file'] - values['shmem'] - values['file_dirty']
               - values['file_writeback'])


def committed_cgroup_bytes(current_bytes: int, stat) -> int:
    """The cgroup memory that reclaim cannot free without writing it out.

    This is the one committed-memory definition. The capture guard admits
    against it, and the retained plan's observed-baseline check compares it
    with its declared owners (PQ #1141): anon, shmem, kernel memory, dirty and
    writeback file pages, and every other charge, which is ``memory.current``
    less the clean file pages of :func:`clean_file_bytes`.

    Clean file pages are never committed. ``memory.current`` counts them, so a
    guard that reads ``memory.current`` refuses a row for page cache the
    kernel would have dropped: on sparklina, the layer-44 own-source read left
    about 21.8 GB of clean NFS cache charged beside 4.7 GB of anon, and the
    retained plan refused a baseline that its declared owners held with about
    20 GB to spare.

    Written as a subtraction, not a sum of the committed keys, so that a charge
    nobody listed here (reclaimable slab, socket buffers, zswap, swap cache)
    stays committed. Pass a ``stat`` read BEFORE ``current_bytes``: cache that
    grows between the two reads then counts as committed, which is the safe
    side. Cache the kernel drops between the reads is the unsafe side, so the
    result never falls below what the same stat states is committed outright
    (anon, shmem, dirty and writeback pages).
    """
    if type(current_bytes) is not int or current_bytes < 0:
        raise RuntimeError(f'cgroup memory.current is invalid: {current_bytes!r}')
    clean = clean_file_bytes(stat)
    stated = stat['anon'] + stat['shmem'] + stat['file_dirty'] + stat['file_writeback']
    return max(current_bytes - clean, stated)


def reserve_allocation(resource_check, label, *, cpu_bytes=0, device_bytes=0):
    """Charge one future allocation to the budget it actually belongs to.

    A reader that is about to hold a payload in this process AND move tensors to
    the device has one ``resource_check`` call and two budgets. Under a split
    guard the two are enforced by different things -- the cgroup cap by the
    kernel, the device envelope by the CUDA allocator -- so a caller that adds
    them into one number charges the device residency to the CPU cap, which is
    how a row holding 80 GiB of device residency beside a 21 GiB CPU cap refuses
    at its first unit.

    A callback that does not declare ``SEPARATE_RESERVATIONS`` -- every caller
    that has not been taught the split -- gets the SUM through ``reserve_bytes``
    exactly as before, so legacy arithmetic does not change.
    """
    if resource_check is None:
        return
    if getattr(resource_check, SEPARATE_RESERVATIONS, False):
        return resource_check(label, reserve_bytes=cpu_bytes,
                              reserve_device_bytes=device_bytes)
    return resource_check(label, reserve_bytes=cpu_bytes + device_bytes)


class GPUMemoryBudgetExceeded(RuntimeError):
    """Raised when cache eviction cannot bring CUDA memory under budget."""


def allocator_device(device):
    """The device the CUDA ALLOCATOR api accepts for ``device``.

    ``get_device_properties`` resolves the unspecified ``cuda`` to the current
    device, and ``set_per_process_memory_fraction`` does not: it raises
    ``ValueError: Expected a torch.device with a specified index or an integer,
    but got: cuda``. The envelope is therefore taken against whatever device
    the caller named and the fraction is SET on that device's resolved index,
    which is what makes the two calls describe the same device.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return device
    return device.index if device.index is not None else torch.cuda.current_device()


def enforce_device_envelope(device, device_bytes, *, where="joint capture"):
    """Cap this process's CUDA allocator at ``device_bytes``.

    ``max_gpu_bytes`` was a POST-HOC check: the pass allocated, then compared
    ``torch.cuda.max_memory_allocated`` with the declared budget and refused
    after the fact. That is a report, not a bound -- on a unified-memory GB10 the
    device and the host are one pool, so an overshoot is charged to the box's
    DRAM and can take the host down with it before the comparison is ever
    reached.

    ``torch.cuda.set_per_process_memory_fraction`` is the enforcement that
    already exists for exactly this: the caching allocator refuses to hand out
    more than the fraction of total device memory it names, and it fails with an
    OOM inside the row instead of on the box. The fraction is derived from the
    declared budget and the device's own reported total, so the plan's number is
    what is enforced and no second constant is introduced.

    WHAT THIS DOES NOT COVER, because the fraction belongs to the caching
    allocator: the CUDA context, NCCL and the rest of the runtime's own
    allocations, and any native or driver allocation made outside torch's
    allocator, are not counted against the fraction and the driver can still
    hand them out. On a unified-memory device they come from the same physical
    pool as everything else, so this bounds THIS PROCESS'S TORCH allocations
    rather than claiming nothing else can oversubscribe the device.

    Returns the record to stamp: the fraction, the device total it was taken
    against, and the budget. Refuses a budget that is not a positive number of
    bytes, or one at or above the device's total, because a fraction of 1.0 is
    the unbounded case wearing the same interface.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return {"enforced": False, "reason": f"{device.type} device has no CUDA allocator"}
    if (isinstance(device_bytes, bool) or not isinstance(device_bytes, int)
            or device_bytes <= 0):
        raise RuntimeError(
            f"{where}: the device envelope must be a positive number of bytes, "
            f"got {device_bytes!r}")
    total = int(torch.cuda.get_device_properties(device).total_memory)
    if device_bytes >= total:
        raise RuntimeError(
            f"{where}: the device envelope {device_bytes} is not below the "
            f"device's total memory {total}, so it bounds nothing; lower the "
            "plan's max_gpu_bytes or run where the device is larger")
    fraction = device_bytes / total
    # The allocator api refuses the unspecified form, so the index is resolved
    # once and reported: a receipt that says which device the fraction was set
    # on is what lets an operator check it against the device it was sized for.
    allocator_index = allocator_device(device)
    torch.cuda.set_per_process_memory_fraction(fraction, allocator_index)
    return {"enforced": True, "fraction": fraction, "device_total_bytes": total,
            "device_envelope_bytes": int(device_bytes), "device": str(device),
            "allocator_device_index": (allocator_index
                                       if isinstance(allocator_index, int)
                                       else None)}


class CaptureMemoryGuard:
    """Fail closed on the conservative cgroup-plus-CUDA capture footprint.

    CUDA can be absent from GB10's cgroup charge. Adding the entire allocator
    reservation is deliberately conservative even where charges overlap. The
    guard never changes a cache policy or drops system-wide page caches.

    Every reading is ABSOLUTE: the cgroup's COMMITTED bytes
    (:func:`committed_cgroup_bytes`, which is ``memory.current`` less the clean
    file pages the kernel reclaims before it refuses an allocation) plus the
    whole CUDA reservation for the process, not the growth a phase caused.
    Every admission below reads committed bytes wherever it once read
    ``memory.current``; the reading still carries the raw charge as
    ``cgroup_current_bytes`` and ``conservative_cgroup_plus_cuda_reserved_bytes``
    beside ``committed_cgroup_plus_cuda_reserved_bytes`` (PQ #1157). ``check``
    compares that absolute reading, plus the caller's future allocation, with
    the cgroup cap less the margin, so its own arithmetic is already in one
    unit. What is not in that unit is a phase PLAN, which states deltas; a
    caller that admits a plan against the raw cap is out by whatever this
    process already held. ``baseline`` is that floor, measured at the first
    ``check``, and ``baseline_bytes`` is what such a caller subtracts. It stays
    the raw charge (``memory.current`` plus CUDA): the Tessera lane subtracts
    it from its cap, and ``baseline['committed_bytes']`` is recorded beside it.
    ``peak_checkpoint`` and ``peak_by_checkpoint_prefix`` say where the peak
    was observed, so a plan that undercharges is attributable from one
    receipt instead of a rerun.

    ``MARGIN_BYTES`` is that physical safety margin as a class attribute so a
    producer that sizes the cgroup cap a row will run under reads the number
    this guard refuses on instead of restating it. A second copy of it would
    drift, and the drift would show up as a row that PrismaBuild admits and
    the guard then refuses.
    """

    #: Physical safety margin held back from the cgroup cap on every check.
    MARGIN_BYTES = 2*1024**3

    def __init__(self, device, *, cgroup_root=Path('/sys/fs/cgroup'),
                 membership=Path('/proc/self/cgroup'),
                 device_bytes: "int | None" = None,
                 aggregate_envelope: bool = False,
                 host_floor_bytes: int = DEFAULT_HOST_FLOOR_BYTES):
        """``device_bytes`` splits the guard; without it, nothing changes.

        THE AGGREGATE. ``aggregate_envelope=True`` (only with a declared
        ``device_bytes``) is the third shape, for a caller whose plan was
        written against the CONSERVATIVE SUM ``cgroup + cuda_reserved`` rather
        than against the two sides apart: the retained COST plan
        (``joint_retained_window_plan.RetainedWindowBudget``) states one
        ``physical_limit_bytes`` that is ``cpu cap + device envelope`` and
        charges every reservation it makes to that one number, because it does
        not know -- on a unified-memory box it cannot know -- which side a
        resident render or a statistics window lands on. Comparing that sum with
        the cgroup cap alone refused the GLM-5.3-Flash run stage on 2026-09-18
        (``104 GiB > 24 GiB``) after a 9.9 h prepare, on a check that had never
        passed since it landed (``c6baaeb70a``). In this mode ``check`` holds:

          * ``committed + cuda_reserved + every reservation`` against
            ``cpu cap + device_bytes - MARGIN_BYTES``, the plan's own arithmetic;
          * ``committed`` alone against ``cpu cap - MARGIN_BYTES``, because
            the kernel still enforces that cap whatever the sum says;
          * ``cuda_reserved + device reservation`` against ``device_bytes``;
          * the host's available memory against ``host_floor_bytes``.

        It does NOT hold the two sides apart, so ``separate_reservations`` is
        False and :func:`reserve_allocation` charges the conservative sum.

        THE SPLIT. A bounded row holds two budgets that are enforced by two
        different things, and a guard that adds them into one number against the
        smaller of the two refuses every row that holds a large device
        residency beside a small CPU cap -- which is the shape of this campaign
        (21 GiB cgroup cap, 80 GiB device envelope, 101 GiB aggregate
        reservation). So when a caller states the device envelope, ``check``
        holds:

          * the cgroup's own committed bytes against ``cap - MARGIN_BYTES``,
            which is the CPU side the kernel enforces with ``--memory``;
          * ``torch.cuda.memory_reserved`` against ``device_bytes``, which is
            the device side (and ``enforce_device_envelope`` is what makes that
            a bound rather than a report);
          * the host's available memory against ``host_floor_bytes``.

        The aggregate the caller reserved from PrismaBuild is the SUM of the two
        budgets -- 21 GiB + 80 GiB = 101 GiB for this campaign -- and it is not a
        third quantity for this guard to derive, because on GB10 the cgroup does
        not charge device memory. The host floor is NOT part of that sum: it is
        memory the box has to keep free, and on a unified-memory box it is what
        the job can still draw on rather than something reserved from it.

        WITHOUT ``device_bytes`` the guard keeps its original conservative
        predicate: committed bytes plus the whole CUDA reservation against
        ``cap - MARGIN_BYTES``. Every existing caller is in that mode, and its
        arithmetic is unchanged apart from the committed-memory definition.
        """
        self.device = torch.device(device)
        if type(host_floor_bytes) is not int or host_floor_bytes < MIN_HOST_FLOOR_BYTES:
            raise RuntimeError(
                f'capture guard host floor must be at least {MIN_HOST_FLOOR_BYTES} '
                f'bytes, got {host_floor_bytes!r}')
        if device_bytes is not None and (
                type(device_bytes) is not int or device_bytes <= 0):
            raise RuntimeError(
                f'capture guard device envelope must be a positive number of '
                f'bytes when declared, got {device_bytes!r}')
        if aggregate_envelope and device_bytes is None:
            raise RuntimeError(
                'an aggregate capture envelope needs a declared device envelope; '
                'without one there is nothing to add to the cgroup cap')
        self.device_bytes = device_bytes
        self.aggregate_envelope = bool(aggregate_envelope)
        root = Path(cgroup_root)
        entries = [line.split(':', 2)[2] for line in Path(membership).read_text().splitlines()
                   if line.startswith('0::')]
        if len(entries) != 1 or not entries[0].startswith('/') or '..' in Path(entries[0]).parts:
            raise RuntimeError('capture memory guard requires a cgroup v2 membership')
        current = root/entries[0].lstrip('/')
        limits = []
        for scope in [current, *current.parents]:
            if scope != root and root not in scope.parents:
                break
            limit_path = scope/'memory.max'
            if not limit_path.is_file():
                if scope == root:
                    break  # The host's root cgroup has no configurable limit.
                raise RuntimeError('capture memory guard cannot inspect its cgroup ancestors')
            raw = limit_path.read_text().strip()
            if raw != 'max':
                limits.append((int(raw), scope))
            if scope == root:
                break
        if not limits:
            raise RuntimeError('bounded capture requires a finite cgroup memory budget')
        self.cap_bytes, self.scope = min(limits, key=lambda pair: pair[0])
        self.cpu_cap_bytes = self.cap_bytes
        self.margin_bytes = self.MARGIN_BYTES
        self.host_floor_bytes = host_floor_bytes
        if self.cap_bytes <= self.margin_bytes:
            raise RuntimeError('capture budget cannot hold its physical safety margin')
        self.failure = None
        self.peak_bytes = 0
        self.peak_cpu_bytes = 0
        self.peak_committed_bytes = 0
        self.peak_device_bytes = 0
        self.peak_checkpoint = None
        self.peak_by_checkpoint_prefix = {}
        self.baseline = None
        self.min_available_bytes = None
        self.last = None
        # The reservations of the most recent check, which is the phase the
        # process is in: ``headroom_bytes`` leaves room for them (PQ #1291).
        self._reserve = (0, 0)
        self._reclaimers = []
        # ``check`` is an INSTANCE ATTRIBUTE holding a closure, not the method:
        # the callers hand ``guard.check`` to a reader as a ``resource_check``
        # callable, and a capability has to travel with THAT object. A bound
        # method carries no attributes of its own, so a property on the class
        # is invisible to ``getattr(guard.check, ...)`` and every split caller
        # would silently fall back to the conservative sum.
        def check(label, *, reserve_bytes=0, reserve_device_bytes=0):
            return self._check(label, reserve_bytes=reserve_bytes,
                               reserve_device_bytes=reserve_device_bytes)
        check.separates_cpu_and_device_reservations = self.separate_reservations
        check.__name__ = "check"
        check.__qualname__ = f"{type(self).__name__}.check"
        check.__doc__ = ("Refuse the moment either budget or the host floor is "
                         "exceeded; see CaptureMemoryGuard._check.")
        self.check = check

    @property
    def separate_reservations(self) -> bool:
        """Whether this guard can hold the two budgets apart.

        True only with a declared device envelope: without one, a device
        reservation has no budget of its own and :meth:`_check` refuses to take
        one, so a caller routing through :func:`reserve_allocation` gets the
        conservative sum exactly as every caller did before the split.
        """
        return self.device_bytes is not None and not self.aggregate_envelope

    @property
    def physical_cap_bytes(self) -> int:
        """The one number a plan's ``physical_limit_bytes`` is compared with.

        The cgroup cap alone unless this guard holds an aggregate envelope, in
        which case it is ``cpu cap + device_bytes``: the same sum the plan
        states and the same threshold ``check`` refuses against.
        """
        if self.aggregate_envelope:
            return self.cpu_cap_bytes + self.device_bytes
        return self.cap_bytes

    def _check(self, label, *, reserve_bytes=0, reserve_device_bytes=0):
        """Refuse the moment either budget or the host floor is exceeded.

        ``reserve_bytes`` is a future allocation charged to the cgroup -- the
        CPU side. ``reserve_device_bytes`` is one charged to the device
        envelope, and is only meaningful where the caller declared one; without
        a declared envelope the whole conservative sum is charged to the cgroup
        cap exactly as before.
        """
        if self.failure is not None:
            raise RuntimeError(self.failure)
        try:
            if type(reserve_bytes) is not int or reserve_bytes < 0:
                raise ValueError('capture future allocation reservation must be nonnegative bytes')
            if type(reserve_device_bytes) is not int or reserve_device_bytes < 0:
                raise ValueError(
                    'capture future device allocation reservation must be nonnegative bytes')
            if reserve_device_bytes and self.device_bytes is None:
                raise ValueError(
                    'a device reservation needs a declared device envelope; without '
                    'one the guard charges CUDA to the cgroup cap and cannot tell '
                    'the two budgets apart')
            self._reserve = (reserve_bytes, reserve_device_bytes)
            observed = self._observe()
            deficits = self._deficits(observed, reserve_bytes, reserve_device_bytes)
            shortfall = max(deficits['budget'], deficits['host'])
            device_shortfall = deficits['device']
            if (shortfall > 0 or device_shortfall > 0) and self._reclaimers:
                # Bytes a reader holds ahead of its consumer are reclaimable
                # (``io_engine.ReadStream.reclaim``): they are dropped before
                # this check would refuse, and it reads the process again. A
                # budget or host-floor shortfall asks the host reclaimers; a
                # device-envelope shortfall asks the device ones (CUDA tensors
                # kept for reuse, PQ #1348). On unified memory both sides
                # draw on one pool, so a host reclaimer that holds CUDA bytes
                # frees them in its own order (``ordered_reclaimer``).
                freed = 0
                for reclaim, device_side in list(self._reclaimers):
                    need = device_shortfall if device_side else shortfall
                    if need > 0:
                        freed += int(reclaim(need))
                if freed:
                    observed = self._observe()
                    deficits = self._deficits(observed, reserve_bytes, reserve_device_bytes)
            cap, stat, current = observed['cap'], observed['stat'], observed['current']
            committed, reserved = observed['committed'], observed['reserved']
            available = observed['available']
            self.last = dict(label=str(label), cgroup_current_bytes=current,
                cgroup_committed_bytes=committed,
                cgroup_clean_file_bytes=clean_file_bytes(stat),
                cuda_reserved_bytes=reserved,
                conservative_cgroup_plus_cuda_reserved_bytes=current+reserved,
                committed_cgroup_plus_cuda_reserved_bytes=committed+reserved,
                host_mem_available_bytes=available, cap_bytes=cap,
                cpu_cap_bytes=self.cpu_cap_bytes,
                future_allocation_bytes=reserve_bytes,
                future_device_allocation_bytes=reserve_device_bytes,
                refusal_threshold_bytes=cap-self.margin_bytes)
            if self.device_bytes is None:
                # The un-split guard: CPU and device charged to one budget, which
                # is the conservative answer when the caller has not said which
                # budget the device residency belongs to.
                self.last['enforced'] = 'cgroup-plus-cuda-reserved'
            elif self.aggregate_envelope:
                # The aggregate guard: the plan's conservative sum against the
                # sum of the two envelopes, plus the cgroup cap the kernel holds
                # on its own and the device envelope the allocator holds.
                self.last.update(enforced='cgroup-plus-cuda-reserved-against-aggregate',
                    device_envelope_bytes=self.device_bytes,
                    aggregate_envelope_bytes=cap+self.device_bytes,
                    aggregate_refusal_threshold_bytes=cap+self.device_bytes-self.margin_bytes,
                    device_refusal_threshold_bytes=self.device_bytes,
                    host_floor_bytes=self.host_floor_bytes,
                    cpu_refusal_threshold_bytes=cap-self.margin_bytes)
            else:
                # The split guard: the kernel's own CPU budget, the device's own
                # envelope, and the host floor are three separate refusals. The
                # aggregate is their sum and travels in the receipt, but adding
                # them here is what refused every row holding a large device
                # residency beside a small CPU cap.
                self.last.update(enforced='split-cpu-device-host',
                    device_envelope_bytes=self.device_bytes,
                    aggregate_envelope_bytes=self.cpu_cap_bytes+self.device_bytes,
                    device_refusal_threshold_bytes=self.device_bytes,
                    host_floor_bytes=self.host_floor_bytes,
                    cpu_refusal_threshold_bytes=cap-self.margin_bytes)
            over_budget = deficits['budget'] > 0
            over_device = deficits['device'] > 0
            if self.baseline is None:
                # The FIRST reading is what this process already held before
                # any planned phase became resident: the interpreter, torch,
                # the CUDA runtime, and every page this process had touched.
                # A phase plan states deltas over that floor, so a caller
                # that compares a plan with the raw cap compares two
                # different quantities (RobTand/prismaquant#390). It is
                # measured here, in the row's own process, because no
                # producer-side constant can know a consumer's floor.
                self.baseline = dict(label=str(label), bytes=current+reserved,
                    measured_in_process=True, cgroup_current_bytes=current,
                    cuda_reserved_bytes=reserved,
                    committed_bytes=committed+reserved,
                    cgroup_committed_bytes=committed)
            if current+reserved > self.peak_bytes:
                self.peak_bytes = current+reserved
                self.peak_checkpoint = str(label)
            self.peak_cpu_bytes = max(self.peak_cpu_bytes, current)
            self.peak_committed_bytes = max(self.peak_committed_bytes,
                                            committed+reserved)
            self.peak_device_bytes = max(self.peak_device_bytes, reserved)
            # Labels carry a per-unit suffix after ':'; the prefixes are the
            # bounded set of phase names, so this attributes a peak to the
            # phase that held it without growing with the roster.
            prefix = str(label).split(':', 1)[0]
            self.peak_by_checkpoint_prefix[prefix] = max(
                self.peak_by_checkpoint_prefix.get(prefix, 0), current+reserved)
            self.min_available_bytes = (available if self.min_available_bytes is None
                                       else min(self.min_available_bytes, available))
            if over_budget and self.aggregate_envelope:
                raise RuntimeError(
                    f'capture aggregate memory refusal: the cgroup has '
                    f'{committed} bytes committed ({current} charged), '
                    f'{reserved} bytes are reserved on '
                    f'{self.device} and {reserve_bytes+reserve_device_bytes} more '
                    f'is requested against a {cap+self.device_bytes}-byte aggregate '
                    f'envelope ({cap}-byte cgroup cap) less a '
                    f'{self.margin_bytes}-byte margin')
            if over_budget:
                raise RuntimeError(
                    f'capture CPU memory refusal: the cgroup has '
                    f'{committed} bytes committed ({current} charged) and '
                    f'{reserve_bytes} more is '
                    f'requested against a {cap}-byte cap less a '
                    f'{self.margin_bytes}-byte margin')
            if over_device:
                raise RuntimeError(
                    f'capture device memory refusal: {reserved} bytes are '
                    f'reserved on {self.device} and {reserve_device_bytes} more '
                    f'is requested against a {self.device_bytes}-byte envelope')
            if deficits['host'] > 0:
                raise RuntimeError(f'capture physical memory refusal: {self.last}')
        except Exception as error:
            self.failure = str(error)
            raise
        return dict(self.last)

    def _observe(self):
        """One reading of the cgroup, the CUDA reservation and the host."""
        raw = (self.scope/'memory.max').read_text().strip()
        cap = self.cap_bytes if raw == 'max' else min(self.cap_bytes, int(raw))
        # The stat first, then the charge: see committed_cgroup_bytes.
        stat = read_memory_stat(self.scope/'memory.stat')
        current = int((self.scope/'memory.current').read_text())
        committed = committed_cgroup_bytes(current, stat)
        reserved = int(torch.cuda.memory_reserved(self.device))
        host = _host_memory_info()
        if host is None or current < 0 or reserved < 0:
            raise RuntimeError('capture memory observations are unavailable')
        available, total = host
        if not 0 <= available <= total:
            raise RuntimeError('capture host memory observations are invalid')
        return dict(cap=cap, stat=stat, current=current, committed=committed,
                    reserved=reserved, available=available)

    def _deficits(self, observed, reserve_bytes, reserve_device_bytes, *,
                  host_reserve=False):
        """By how many bytes each refusal of :meth:`_check` is exceeded.

        ``budget`` is the cgroup (or aggregate) refusal, ``device`` the device
        envelope's and ``host`` the host floor's; a positive value refuses.
        ``host_reserve`` also holds the CPU reservation against the cgroup cap
        on its own in the aggregate guard, which ``_check`` does not (the plan
        does not say which side its reservation lands on); ``headroom_bytes``
        asks for it, because what it admits lands on the host.
        """
        cap, committed = observed['cap'], observed['committed']
        reserved, available = observed['reserved'], observed['available']
        limit = cap - self.margin_bytes
        if self.device_bytes is None:
            return {'budget': committed + reserved + reserve_bytes - limit,
                    'device': 0,
                    'host': self.host_floor_bytes + reserve_bytes - available}
        host = self.host_floor_bytes + reserve_bytes + reserve_device_bytes - available
        device = reserved + reserve_device_bytes - self.device_bytes
        if self.aggregate_envelope:
            return {'budget': max(
                        committed + reserved + reserve_bytes + reserve_device_bytes
                        - (limit + self.device_bytes),
                        committed + (reserve_bytes if host_reserve else 0) - limit),
                    'device': device, 'host': host}
        return {'budget': committed + reserve_bytes - limit, 'device': device, 'host': host}

    def headroom_bytes(self) -> int:
        """Host bytes that may still be allocated now without a refusal.

        A reading, not a check: it records nothing and never refuses. It is
        the largest allocation on the host side that keeps every term of the
        most recent check's reservations (the phase the process is in) within
        the budget, the aggregate envelope and the host floor, with that
        reservation also held against the cgroup cap on its own. Negative when
        the process is already past one of them. A reader that runs ahead of
        its consumer sizes its depth by it (``io_engine``, PQ #1291).
        """
        if self.failure is not None:
            return 0
        reserve_bytes, reserve_device_bytes = self._reserve
        deficits = self._deficits(self._observe(), reserve_bytes, reserve_device_bytes,
                                  host_reserve=True)
        return -max(deficits['budget'], deficits['host'])

    def device_headroom_bytes(self) -> int:
        """Device bytes that may still be allocated now without a refusal (PQ #1348).

        The device side's :meth:`headroom_bytes`: the largest CUDA allocation
        that keeps every term of the most recent check's reservations within
        each limit that allocation lands in. Without a device envelope CUDA is
        charged beside the cgroup, so that is the budget; with one it is the
        device envelope, and the aggregate envelope where this guard holds
        one. The host floor counts either way: on unified memory a CUDA
        allocation takes host pages. A reading, not a check; negative when
        the process is already past one of them.
        """
        if self.failure is not None:
            return 0
        reserve_bytes, reserve_device_bytes = self._reserve
        observed = self._observe()
        deficits = self._deficits(observed, reserve_bytes, reserve_device_bytes)
        if self.device_bytes is None:
            return -max(deficits['budget'], deficits['host'])
        terms = [deficits['device'], deficits['host']]
        if self.aggregate_envelope:
            terms.append(observed['committed'] + observed['reserved'] + reserve_bytes
                         + reserve_device_bytes
                         - (observed['cap'] - self.margin_bytes + self.device_bytes))
        return -max(terms)

    def add_reclaimer(self, reclaim, *, device=False):
        """Let ``reclaim(shortfall_bytes) -> freed`` drop bytes before a refusal.

        Returns a callable that removes it again. A check that would refuse on
        the budget or the host floor calls every host reclaimer with its
        shortfall first and then reads the process again. ``device=True``
        registers a device reclaimer instead (PQ #1348): a check that would
        refuse on the device envelope calls it with that shortfall, and a
        host shortfall does not.
        """
        entry = (reclaim, bool(device))
        self._reclaimers.append(entry)

        def remove():
            if entry in self._reclaimers:
                self._reclaimers.remove(entry)
        return remove

    def snapshot(self):
        return dict(scope=str(self.scope), budget_bytes=self.cap_bytes,
            physical_cap_bytes=self.physical_cap_bytes,
            device_envelope_bytes=self.device_bytes,
            aggregate_envelope=self.aggregate_envelope,
            margin_bytes=self.margin_bytes, host_floor_bytes=self.host_floor_bytes,
            peak_conservative_bytes=self.peak_bytes,
            peak_committed_bytes=self.peak_committed_bytes,
            peak_checkpoint=self.peak_checkpoint,
            peak_by_checkpoint_prefix=dict(self.peak_by_checkpoint_prefix),
            baseline=None if self.baseline is None else dict(self.baseline),
            min_host_available_bytes=self.min_available_bytes,
            last_checkpoint=None if self.last is None else dict(self.last))

    def baseline_bytes(self):
        """The measured process floor every delta plan is compared against.

        Refuses before the first ``check`` rather than defaulting to zero: a
        zero floor is the arithmetic this guard exists to stop, and it would
        read as "the process holds nothing" on a box where it holds gigabytes.
        """
        if self.baseline is None:
            raise RuntimeError('capture memory baseline is unmeasured; check() first')
        return int(self.baseline['bytes'])


def env_flag_enabled(name: str, *, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        parsed = int(value)
    except ValueError:
        return int(default)
    return max(parsed, 0)


def register_budget_evictor(evictor: object) -> None:
    try:
        _BUDGET_EVICTORS.add(evictor)
    except TypeError:
        pass


def unregister_budget_evictor(evictor: object) -> None:
    try:
        _BUDGET_EVICTORS.discard(evictor)
    except TypeError:
        pass


def model_device(model) -> torch.device:
    for p in model.parameters():
        if not p.is_meta:
            return p.device
    return torch.device("cpu")


def cuda_memory_info(device: torch.device | None = None) -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except TypeError:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    if _use_host_available_for_uma(device):
        host_info = _host_memory_info()
        if host_info is not None:
            host_available, _host_total = host_info
            # Integrated CUDA devices share the host memory pool.  On GB10,
            # mem_get_info() reports reclaimable page cache as unavailable,
            # which makes cache and lane guardrails far too conservative
            # after streamed weight snapshots.  MemAvailable already includes
            # reclaimable cache, so use it as the better free-memory signal.
            free_bytes = max(int(free_bytes), int(host_available))
            free_bytes = min(int(free_bytes), int(total_bytes))
    return int(free_bytes), int(total_bytes)


def _use_host_available_for_uma(device: torch.device | None = None) -> bool:
    mode = os.environ.get("PRISMAQUANT_UMA_MEMORY_INFO", "auto")
    mode = str(mode).strip().lower()
    if mode in {"0", "false", "no", "off", "cuda"}:
        return False
    if mode in {"1", "true", "yes", "on", "host", "uma"}:
        return True
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:
        return False
    return bool(getattr(props, "is_integrated", False))


def ordered_reclaimer(*reclaims):
    """One reclaimer that asks each of ``reclaims`` in turn for what is still short.

    A guard asks every reclaimer it holds for the whole shortfall. Where
    several draw on one pool (GB10's unified memory: renders and spill
    chunks read ahead, CUDA tensors kept for reuse), registering them as one
    ordered reclaimer drops the cheapest to restore first and stops once the
    shortfall is covered (PQ #1348). ``None`` entries are skipped.
    """
    reclaims = tuple(reclaim for reclaim in reclaims if reclaim is not None)

    def reclaim(shortfall_bytes):
        freed = 0
        for step in reclaims:
            if freed >= shortfall_bytes:
                break
            freed += int(step(shortfall_bytes - freed))
        return freed
    return reclaim


def _host_memory_info() -> tuple[int, int] | None:
    """``(MemAvailable, MemTotal)`` in bytes, or ``None`` when unreadable."""
    try:
        values = io_spans.read_meminfo()
        return values["MemAvailable"], values["MemTotal"]
    except Exception:
        return None


def _dynamic_gpu_memory_budget_bytes(
    device: torch.device | None = None,
) -> int | None:
    info = cuda_memory_info(device)
    if info is None:
        return None
    free_bytes, total_bytes = info
    used_bytes = total_bytes - free_bytes

    device_reserve = max(
        int(total_bytes * max(
            env_float("PRISMAQUANT_GPU_MEM_RESERVE_FRACTION", 0.05),
            0.0,
        )),
        int(max(env_float("PRISMAQUANT_GPU_MEM_RESERVE_GB", 2.0), 0.0) * 1024 ** 3),
    )
    budget = total_bytes - device_reserve

    host_info = _host_memory_info()
    if host_info is not None:
        host_available, host_total = host_info
        host_reserve = max(
            int(host_total * max(
                env_float("PRISMAQUANT_HOST_MEM_RESERVE_FRACTION", 0.05),
                0.0,
            )),
            int(max(env_float("PRISMAQUANT_HOST_MEM_RESERVE_GB", 4.0), 0.0) * 1024 ** 3),
        )
        host_deficit = max(0, host_reserve - host_available)
        if host_deficit:
            # On UMA systems, CUDA allocations and host memory share the same
            # physical pool. Lower the CUDA cache budget by the observed host
            # deficit so registered caches are evicted before swap pressure
            # turns into a system OOM.
            budget = min(budget, used_bytes - host_deficit)

    return max(int(budget), 0)


def max_gpu_memory_bytes(device: torch.device | None = None) -> int | None:
    """Return the cache budget for CUDA-visible allocations.

    `PRISMAQUANT_MAX_GPU_MEM_GB` remains an explicit override. Without it,
    derive the budget from the live device size and host memory pressure so
    cache-heavy passes scale across 24 GB, 48 GB, 96 GB, UMA, and larger hosts
    without baking in one workstation's usable-memory ceiling.
    """
    raw = os.environ.get("PRISMAQUANT_MAX_GPU_MEM_GB")
    if raw is not None and raw.strip() != "":
        try:
            gb = float(raw)
        except ValueError:
            gb = -1.0 if raw.strip().lower() in {"0", "off", "false", "none"} else 0.0
        if gb <= 0.0:
            return None
        return int(gb * 1024 ** 3)
    return _dynamic_gpu_memory_budget_bytes(device)


def _gb(num_bytes: int | float) -> float:
    return float(num_bytes) / float(1024 ** 3)


def _tensor_tree_nbytes(value, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    if isinstance(value, torch.Tensor):
        key = id(value)
        if key in seen:
            return 0
        seen.add(key)
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_tree_nbytes(child, seen) for child in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_tree_nbytes(child, seen) for child in value)
    return 0


def _graph_entry_static_nbytes(entry) -> int:
    seen: set[int] = set()
    total = 0
    for attr in ("static_hidden", "static_args", "static_kwargs", "static_output"):
        if hasattr(entry, attr):
            total += _tensor_tree_nbytes(getattr(entry, attr), seen)
    return total


def _evictor_pool_id(evictor: object) -> str:
    getter = getattr(evictor, "graph_pool_id", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception as exc:
            return f"unavailable:{type(exc).__name__}"
    pool = getattr(evictor, "graph_pool", None)
    if callable(pool):
        try:
            handle = pool()
        except Exception as exc:
            return f"unavailable:{type(exc).__name__}"
        if handle is None:
            return "private"
        return f"shared:{id(handle):x}"
    return "unknown"


def report_graph_memory(label: str = "") -> None:
    """Print per-registry occupancy + pool footprint. Used at phase boundaries."""
    if not env_flag_enabled("PRISMAQUANT_GRAPH_AUDIT", default=False):
        return
    label_text = str(label or "-")
    evictors = list(_BUDGET_EVICTORS)
    registry_count = 0
    total_entries = 0
    total_static_bytes = 0
    for evictor in evictors:
        entries = getattr(evictor, "entries", None)
        if entries is None:
            continue
        try:
            entry_items = list(entries.values())
        except AttributeError:
            continue
        registry_count += 1
        entry_count = len(entry_items)
        static_bytes = sum(_graph_entry_static_nbytes(entry) for entry in entry_items)
        total_entries += entry_count
        total_static_bytes += static_bytes
        registry_label = getattr(evictor, "label", type(evictor).__name__)
        print(
            "[graph-audit] "
            f"label={label_text} registry={registry_label} "
            f"class={type(evictor).__name__} entries={entry_count} "
            f"static_bytes={static_bytes} static_gb={_gb(static_bytes):.6f} "
            f"pool={_evictor_pool_id(evictor)}",
            file=sys.stderr,
            flush=True,
        )

    allocated_current = None
    mem_info = None
    if torch.cuda.is_available():
        try:
            allocated_current = int(
                torch.cuda.memory_stats().get("allocated_bytes.all.current", 0)
            )
        except Exception:
            allocated_current = None
        mem_info = cuda_memory_info()
    free_text = total_text = "n/a"
    if mem_info is not None:
        free_text = str(mem_info[0])
        total_text = str(mem_info[1])
    allocated_text = "n/a" if allocated_current is None else str(allocated_current)
    print(
        "[graph-audit] "
        f"label={label_text} summary registries={registry_count} "
        f"entries={total_entries} static_bytes={total_static_bytes} "
        f"static_gb={_gb(total_static_bytes):.6f} "
        f"allocated_bytes_current={allocated_text} "
        f"mem_free_bytes={free_text} mem_total_bytes={total_text}",
        file=sys.stderr,
        flush=True,
    )


def _unique_evictors(evictors: Iterable[object]) -> list[object]:
    out: list[object] = []
    seen: set[int] = set()
    for evictor in evictors:
        if evictor is None:
            continue
        key = id(evictor)
        if key in seen:
            continue
        seen.add(key)
        out.append(evictor)
    return out


def _drop_released_cuda_memory(*, synchronize: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if synchronize:
            torch.cuda.synchronize()


def enforce_gpu_memory_budget(
    evictors: Iterable[object] = (),
    *,
    device: torch.device | None = None,
    reason: str = "",
) -> int:
    """Evict oldest registered cache entries until used CUDA memory is in budget.

    ``torch.cuda.mem_get_info`` reports driver-visible free memory. We compare
    ``total - free`` against ``PRISMAQUANT_MAX_GPU_MEM_GB`` so the budget acts
    as a hard ceiling even when PyTorch's caching allocator is holding blocks.
    """
    budget_bytes = max_gpu_memory_bytes(device)
    if budget_bytes is None:
        return 0
    info = cuda_memory_info(device)
    if info is None:
        return 0
    free_bytes, total_bytes = info
    used_bytes = total_bytes - free_bytes
    if used_bytes <= budget_bytes:
        return 0

    candidates = _unique_evictors([*evictors, *_BUDGET_EVICTORS])
    evicted = 0
    while used_bytes > budget_bytes:
        progress = False
        for evictor in candidates:
            evict_one = getattr(evictor, "evict_oldest_for_memory_budget", None)
            if not callable(evict_one):
                continue
            if evict_one():
                evicted += 1
                progress = True
                _drop_released_cuda_memory()
                info = cuda_memory_info(device)
                if info is None:
                    return evicted
                free_bytes, total_bytes = info
                used_bytes = total_bytes - free_bytes
                if used_bytes <= budget_bytes:
                    break
        if not progress:
            detail = f" during {reason}" if reason else ""
            raise GPUMemoryBudgetExceeded(
                "CUDA memory budget exceeded"
                f"{detail}: used={_gb(used_bytes):.2f}GB "
                f"budget={_gb(budget_bytes):.2f}GB "
                f"total={_gb(total_bytes):.2f}GB. "
                "No registered cache entries remain to evict; the model or "
                "other allocations exceed PRISMAQUANT_MAX_GPU_MEM_GB."
            )
    return evicted


def phase_boundary_memory_cleanup(label: str | None = None) -> None:
    """Release allocator-held memory and collect Python garbage at phase edges."""
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        if label:
            print(
                f"[memory] cleanup {label}: empty_cache failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
