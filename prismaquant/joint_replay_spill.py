"""Stage B one-pass replay: spill each probe's X/G once, replay windows from it.

The windowed replay (``joint_statistics_replay.observe_and_project_retained_
windows``) runs the whole layer forward and backward once per (retained
window, probe), because each window's statistics lease hooks only that
window's Linears. On GLM-5.3-Flash that is about 15 layer passes per probe.

This module runs ONE forward and backward per probe with a spill observer on
every pending target Linear. At each backward invocation it records the
Linear's bf16 input ``x`` and its selected bf16 output gradient to a
local-NVMe scratch file. Each retained window then feeds its own fresh
``JointOperatorStatisticsLease`` from the spill through the lease's one
statistics arithmetic (``_observe_invocation``). The records are
bit-identical to the windowed path's because:

* every statistics key is an independent FP32 accumulator of one Linear, so
  only each Linear's own ``add_`` order matters, and the replay feeds each
  Linear's invocations in firing order;
* the spilled operands are the exact bf16 tensors the live hook reads at fire
  time, rebuilt with the same shape and strides and, on CUDA, the same
  address residue modulo 512 bytes, so ``.float()``, the activation QDQ and
  the GEMMs see identical operands (see ``StageBReplaySpill._same_layout``
  for why a host residue is not held);
* a non-dense selected gradient is stored contiguous, which is what the live
  path's ``reshape(...).float()`` makes of it anyway. A non-dense ``x`` is
  refused, because it reaches the activation QDQ as is.

The spill is laid out per Linear (``_Window``): one input stream per Linear
that first read a tensor, and one gradient stream per Linear and probe, so a
per-operator GEMM over a Linear's rows reads two ordered streams.

The arithmetic identity and the resource policy are unchanged; the replay
mode is recorded only in the quantum counters. Without the spill
environment the windowed path runs unchanged and is the bitwise reference.

The scratch is declared like the #956 cotangent sink: an environment root and
a byte ceiling, forwarded by the campaign container launcher through an
identity bind. The layer's spill bytes are bounded from geometry before any
GPU work and the whole bound is allocated up front; see
``perturbed_x_cache.StageBSpillScratch``.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import bisect
import hashlib
import os
import queue
import threading
import time

import torch

from .joint_aura import (
    JointOperatorStatisticsLease,
    SignedJointProjectionLease,
    select_invocation_gradient,
)
from .routed_experts import PackedExpertProjection

SPILL_ENV = ("PRISMAQUANT_STAGE_B_SPILL_ROOT", "PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES")
REPLAY_WINDOWED = "windowed"
REPLAY_SPILL = "one_pass_spill"
SPILL_DTYPES = (torch.bfloat16, torch.float16)
#: Replayed operands keep the live tensor's address residue modulo this many
#: bytes: the CUDA caching allocator's block alignment, so no kernel variant
#: chosen on pointer alignment can differ between the two paths.
ADDRESS_ALIGNMENT = 512
#: Write arenas and read buffers are at least this large (or one largest
#: tensor pair), and never larger than the layer needs.
ARENA_BYTES = 256 << 20
READ_BYTES = 64 << 20
ARENA_COUNT = 3
READ_BUFFER_COUNT = 2
#: Writer and read-ahead threads by default; tests run the same I/O inline.
DEFAULT_THREADS = True
_TOP_K_KEYS = ("num_experts_per_tok", "num_experts_per_token", "moe_top_k",
               "num_active_experts")


def stage_b_spill_config(environ=None):
    """``(root, max_bytes)`` when the spill is declared, else ``None``."""
    environ = os.environ if environ is None else environ
    root, ceiling = (environ.get(name) for name in SPILL_ENV)
    if root is None and ceiling is None:
        return None
    if not root or not ceiling or not ceiling.isdecimal() or int(ceiling) <= 0:
        raise ValueError(
            "Stage B spill requires both PRISMAQUANT_STAGE_B_SPILL_ROOT and a "
            "positive PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES")
    return root, int(ceiling)


def experts_per_token(model):
    """The declared routed top-k, or ``None`` when the config states none."""
    config = getattr(model, "config", None)
    for candidate in (config, getattr(config, "text_config", None)):
        if candidate is None:
            continue
        for key in _TOP_K_KEYS:
            value = (candidate.get(key) if isinstance(candidate, Mapping)
                     else getattr(candidate, key, None))
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


@dataclass(frozen=True)
class SpillGeometry:
    """Upper bound on one layer's spill, from shapes and token counts only."""

    element_size: int
    experts_per_token: int | None
    n_probes: int
    tokens: int
    max_batch_tokens: int
    window_x_bytes: tuple
    window_g_bytes: tuple
    x_bytes: int
    g_bytes_per_probe: int
    total_bytes: int
    batch_x_bytes: int
    batch_bytes: int
    largest_tensor_bytes: int

    def as_dict(self):
        return {
            "element_size": self.element_size,
            "experts_per_token": self.experts_per_token,
            "n_probes": self.n_probes, "tokens": self.tokens,
            "max_batch_tokens": self.max_batch_tokens,
            "x_bytes": self.x_bytes, "g_bytes_per_probe": self.g_bytes_per_probe,
            "total_bytes": self.total_bytes, "batch_x_bytes": self.batch_x_bytes,
            "batch_bytes": self.batch_bytes,
            "largest_tensor_bytes": self.largest_tensor_bytes,
            "windows": len(self.window_x_bytes),
        }


def spill_geometry(linears, window_names, *, pending, batch_tokens, n_probes,
                   element_size, experts_per_token):
    """Bound the spill of ``pending`` targets over ``window_names``.

    One invocation per target per sample is assumed; the scratch refuses at
    the first byte past this bound, so a model that breaks the assumption
    fails closed instead of overrunning the disk.

    * A dense target spills ``x`` and its gradient for every token.
    * A token visits each routed expert at most once and at most ``top_k``
      experts. A packed projection sits in one window, so its gradient rows
      over its ``m`` pending experts, across all windows, are at most
      ``tokens * min(m, top_k)``. Without a declared top-k the bound is
      ``tokens * m``, looser but still exact.
    * An expert's ``x`` is written once per window that holds any projection
      of its packed parameter (gate and up read the same rows), ``c`` times
      in all. A token then writes at most the ``top_k`` largest ``c`` of the
      experts it visits, so the parameter's input rows are at most ``tokens``
      times that sum. Bounding each window by ``min(m, top_k)`` instead would
      charge ``top_k`` rows per token to every window a parameter spans.

    ``window_x_bytes`` and ``window_g_bytes`` bound each window alone; the
    totals bound the layer and are what the scratch reserves.
    """
    if type(n_probes) is not int or n_probes <= 0:
        raise ValueError("Stage B spill geometry requires a positive probe count")
    batch_tokens = [int(tokens) for tokens in batch_tokens]
    if not batch_tokens or any(tokens <= 0 for tokens in batch_tokens):
        raise ValueError("Stage B spill geometry requires positive batch token counts")

    def rows(experts):
        return experts if experts_per_token is None else min(experts, experts_per_token)

    def largest(counts):
        ordered = sorted(counts, reverse=True)
        return sum(ordered if experts_per_token is None else ordered[:experts_per_token])

    per_token_x, per_token_g, widest = [], [], 0
    dense_x = dense_g = 0
    layer_in, layer_out = {}, {}
    for names in window_names:
        x_width = g_width = 0
        packed_in, packed_out = {}, {}
        for name in names:
            if name not in pending:
                continue
            module = linears[name]
            out_features, in_features = (int(size) for size in module.weight.shape)
            widest = max(widest, out_features, in_features)
            if isinstance(module, PackedExpertProjection):
                key = (module.module_qname, module.param_name)
                width, experts = packed_in.setdefault(key, [in_features, set()])
                if width != in_features:
                    raise RuntimeError(f"packed parameter input width differs for {name}")
                experts.add(int(module.expert_id))
                role = (key, module.projection_name)
                width, experts = packed_out.setdefault(role, [out_features, set()])
                if width != out_features:
                    raise RuntimeError(f"packed projection output width differs for {name}")
                experts.add(int(module.expert_id))
            else:
                x_width += in_features
                g_width += out_features
        dense_x += x_width
        dense_g += g_width
        for key, (width, experts) in packed_in.items():
            layer_width, windows = layer_in.setdefault(key, [width, {}])
            if layer_width != width:
                raise RuntimeError(f"packed parameter input width differs for {key}")
            for expert in experts:
                windows[expert] = windows.get(expert, 0) + 1
        for role, (width, experts) in packed_out.items():
            layer_width, seen = layer_out.setdefault(role, [width, set()])
            if layer_width != width or seen & experts:
                raise RuntimeError(f"packed projection {role} is split inconsistently "
                                   "across windows")
            seen |= experts
        x_width += sum(width * rows(len(experts)) for width, experts in packed_in.values())
        g_width += sum(width * rows(len(experts)) for width, experts in packed_out.values())
        per_token_x.append(x_width)
        per_token_g.append(g_width)
    layer_x = dense_x + sum(width * largest(windows.values())
                            for width, windows in layer_in.values())
    layer_g = dense_g + sum(width * rows(len(experts))
                            for width, experts in layer_out.values())
    tokens, widest_batch = sum(batch_tokens), max(batch_tokens)
    window_x = tuple(width * tokens * element_size for width in per_token_x)
    window_g = tuple(width * tokens * element_size for width in per_token_g)
    x_bytes = layer_x * tokens * element_size
    g_bytes = layer_g * tokens * element_size
    return SpillGeometry(
        element_size=int(element_size), experts_per_token=experts_per_token,
        n_probes=n_probes, tokens=tokens, max_batch_tokens=widest_batch,
        window_x_bytes=window_x, window_g_bytes=window_g,
        x_bytes=x_bytes, g_bytes_per_probe=g_bytes,
        total_bytes=x_bytes + n_probes * g_bytes,
        batch_x_bytes=layer_x * widest_batch * element_size,
        batch_bytes=(layer_x + layer_g) * widest_batch * element_size,
        largest_tensor_bytes=widest * widest_batch * element_size)


def _is_dense(tensor):
    """Non-overlapping and dense: its elements fill exactly ``numel`` slots."""
    if tensor.is_contiguous():
        return True
    expected = 1
    for stride, size in sorted((stride, size) for size, stride in
                               zip(tensor.shape, tensor.stride()) if size != 1):
        if stride != expected:
            return False
        expected *= size
    return True


def _contiguous_strides(shape):
    strides, step = [], 1
    for size in reversed(shape):
        strides.append(step)
        step *= max(int(size), 1)
    return tuple(reversed(strides))


def _layout(tensor, *, dense):
    if dense:
        return (tuple(tensor.shape), tuple(tensor.stride()),
                tensor.data_ptr() % ADDRESS_ALIGNMENT)
    return (tuple(tensor.shape), _contiguous_strides(tensor.shape), 0)


def _storage_order(tensor):
    """The tensor's elements in storage order, as one flat view (or copy)."""
    if _is_dense(tensor):
        return tensor.as_strided((tensor.numel(),), (1,))
    return tensor.contiguous().view(-1)


def _placed(cursor, residue):
    return cursor + (residue - cursor) % ADDRESS_ALIGNMENT


class _Entry:
    __slots__ = ("logical", "nbytes", "layout", "digest")

    def __init__(self, logical, nbytes, layout):
        self.logical, self.nbytes, self.layout, self.digest = logical, nbytes, layout, None


class _Window:
    """One retained window's spilled streams and its probe-0 record order.

    Streams are per Linear. A Linear's output gradients are one stream per
    probe. An input stream belongs to the Linear that first read the tensor;
    a Linear that reads the same tensor afterwards (the up projection of an
    expert whose gate projection read it first) reads its owner's stream,
    and must do so for every one of its invocations. So each Linear's rows
    are one ordered input stream and one ordered gradient stream, which is
    what a per-operator GEMM over the Linear's rows reads.
    """

    def __init__(self, names):
        self.names = tuple(names)
        # owner -> input entries in first-use order
        self.entries: dict[str, list[_Entry]] = {}
        self.x_logical: dict[str, int] = {}
        self.x_runs: dict[str, list[tuple[int, int, int]]] = {}
        self.x_starts: dict[str, list[int]] = {}
        # Linear -> the input stream all of its records read
        self.x_source: dict[str, str] = {}
        # (name, owner, entry, g_logical, g_bytes, g_layout) in firing order
        self.records: list[tuple] = []
        self.g_logical: dict[str, int] = {}
        self.g_runs: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
        self.g_starts: dict[tuple[str, int], list[int]] = {}
        # (owner, entry) -> the last plan position that reads it
        self.last_ref: dict[tuple[str, int], int] = {}
        # [(owner, chunk)] in replay order
        self.plan: list[tuple] = []
        self.entry_cursor: dict[str, int] = {}
        self.record_cursor = 0
        self.dedupe: dict = {}


class _Arena:
    __slots__ = ("buffer", "view", "used", "parts", "event", "probe")

    def __init__(self, nbytes, pinned):
        self.buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pinned)
        self.view = memoryview(self.buffer.numpy())
        self.used, self.parts, self.event, self.probe = 0, [], None, None


class _SpillObserver(SignedJointProjectionLease):
    """The statistics lease's hooks and packed wrapper, recording instead.

    It reuses ``SignedJointProjectionLease``'s forward hooks and its packed
    ``F.linear``/``F.grouped_mm`` wrapper unchanged, and the gradient
    selection the statistics lease uses (``select_invocation_gradient``).
    """

    def __init__(self, session, modules, specs_by_qname, *, activation_max_abs,
                 projection_backend):
        self._session = session
        super().__init__(modules, specs_by_qname, {},
                         activation_max_abs=activation_max_abs,
                         projection_backend=projection_backend)
        self._sources = {name: JointOperatorStatisticsLease._source_fingerprint(module.weight)
                         for name, module in self.modules.items()}
        self._pending = 0
        self.observed = 0

    def _validate_delta_coverage(self, name, module):
        if name not in self.specs or not self.specs[name]:
            raise ValueError(f"Stage B spill missing spec coverage for {name}")

    def _observe(self, name, source_weight, x, output, output_slice=None, row_slice=None):
        if not self.active:
            raise RuntimeError("Stage B spill forward outside active capture")
        if JointOperatorStatisticsLease._source_fingerprint(source_weight) != self._sources[name]:
            raise RuntimeError(f"Stage B spill source weight changed for {name}")
        if not isinstance(x, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise TypeError(f"Stage B spill Linear {name} needs Tensor input/output")
        held = [x.detach(), source_weight]
        fired = False

        def spill(gradient):
            nonlocal fired
            if not self.active or fired:
                raise RuntimeError("Stage B spill backward outside active capture")
            fired = True
            try:
                x_held, weight = held
                with torch.no_grad():
                    selected = select_invocation_gradient(
                        name, weight, x_held, gradient,
                        output_slice=output_slice, row_slice=row_slice)
                    self._session._record(name, x_held, selected)
                self._pending -= 1
                self.observed += 1
            except BaseException:
                self.active = False
                raise
            finally:
                held.clear()
            return gradient

        if output.requires_grad:
            self._pending += 1
            output.register_hook(spill)

    def end_batch(self):
        if self._pending:
            raise RuntimeError("Stage B spill has pending backward observations")
        self._session._end_batch()

    def finish_probe(self):
        raise RuntimeError("Stage B spill observer projects nothing")


class StageBReplaySpill:
    """One layer quantum's spill: capture per probe, then replay per window.

    ``window_names`` are the sealed retained windows in index order, already
    reduced to their pending targets. ``threads`` runs the NVMe writer and
    the window read-ahead on their own threads; ``False`` does the same I/O
    inline. ``None`` takes ``DEFAULT_THREADS``.
    """

    def __init__(self, *, root, max_bytes, geometry, window_names, n_probes,
                 dtype, device, threads=None):
        from .perturbed_x_cache import StageBSpillScratch

        if dtype not in SPILL_DTYPES:
            raise RuntimeError(
                f"Stage B spill replays 16-bit measurement only, not {dtype}: "
                "an FP32 operand is used in place and its layout is not rebuilt")
        if not isinstance(geometry, SpillGeometry) or geometry.n_probes != n_probes:
            raise ValueError("Stage B spill geometry does not match the probe count")
        self.geometry = geometry
        self.n_probes = int(n_probes)
        self.dtype = dtype
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            # Tensors always name their index; compare against the same spelling.
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.element_size = torch.empty((), dtype=dtype).element_size()
        if self.element_size != geometry.element_size:
            raise ValueError("Stage B spill geometry element size differs")
        self._cuda = self.device.type == "cuda"
        self._threads = DEFAULT_THREADS if threads is None else bool(threads)
        self._windows = [_Window(names) for names in window_names]
        self._window_of = {}
        for index, window in enumerate(self._windows):
            for name in window.names:
                if name in self._window_of:
                    raise ValueError(f"Stage B spill window membership repeats {name}")
                self._window_of[name] = index
        pair = 2 * (geometry.largest_tensor_bytes + ADDRESS_ALIGNMENT)
        per_probe = geometry.x_bytes + geometry.g_bytes_per_probe
        self.arena_bytes = max(pair, min(ARENA_BYTES, per_probe + pair))
        self.read_bytes = max(pair, min(READ_BYTES, per_probe + pair))
        self._arenas: list[_Arena] = []
        self._free: queue.Queue | None = None
        self._pending: queue.Queue | None = None
        self._writer = None
        self._write_error = None
        self._arena = None
        self._probe = None
        self._captured = 0
        self._failed = False
        self._read_buffers: list = []
        self.telemetry = {
            "bound_bytes": geometry.total_bytes, "ceiling_bytes": int(max_bytes),
            "x_bytes_written": 0, "g_bytes_written": 0, "bytes_read": 0,
            "runs_written": 0, "reads": 0, "x_streams": 0, "shared_input_linears": 0,
            "records_per_probe": [], "x_entries": 0, "x_digest_checks": 0,
            "capture_wall_s": [], "replay_wall_s": 0.0,
            "hook_wait_s": 0.0, "writer_busy_s": 0.0, "reader_wait_s": 0.0,
            "arena_bytes": self.arena_bytes,
            "arenas": ARENA_COUNT if self._threads else 1,
            "read_buffer_bytes": self.read_bytes,
            "read_buffers": READ_BUFFER_COUNT if self._threads else 1,
            "threads": self._threads,
        }
        self._scratch = StageBSpillScratch(directory=root, max_bytes=max_bytes,
                                           nbytes=geometry.total_bytes)

    # -- reservations the caller charges to its capture guard ---------------
    @property
    def capture_reserve_bytes(self):
        """Pinned arenas plus every target input held until its backward."""
        arenas = 0 if self._arenas else self.arena_bytes * self.telemetry["arenas"]
        return arenas + self.geometry.batch_x_bytes

    @property
    def replay_reserve_bytes(self):
        """Pinned read buffers plus the device staging a window keeps live."""
        buffers = 0 if self._read_buffers else self.read_bytes * self.telemetry["read_buffers"]
        return buffers + 2 * (self.read_bytes + ADDRESS_ALIGNMENT)

    # -- lifetime -------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        writer, self._writer = self._writer, None
        if writer is not None:
            self._pending.put(None)
            writer.join()
        self._arenas.clear()
        self._arena = None
        self._read_buffers.clear()
        for window in self._windows:
            window.dedupe.clear()
        self._scratch.close()

    def _require_healthy(self):
        if self._failed:
            raise RuntimeError("Stage B spill failed earlier in this quantum")
        if self._write_error is not None:
            self._failed = True
            raise RuntimeError("Stage B spill writer failed") from self._write_error

    # -- capture --------------------------------------------------------------
    @contextmanager
    def capture(self, probe_index, modules, specs_by_qname, *, activation_max_abs,
                projection_backend):
        """Observe one probe's single forward/backward over every pending target."""
        self._require_healthy()
        if probe_index != self._captured or probe_index >= self.n_probes:
            raise RuntimeError("Stage B spill captures probes once, in index order")
        if set(modules) != set(self._window_of):
            raise RuntimeError("Stage B spill capture roster differs from its windows")
        self._probe = int(probe_index)
        for window in self._windows:
            window.entry_cursor = {}
            window.record_cursor = 0
            window.g_logical = {}
            window.dedupe.clear()
        self._start_arenas()
        started = time.time()
        records_before = self._records_seen = 0
        try:
            observer = _SpillObserver(self, modules, specs_by_qname,
                                      activation_max_abs=activation_max_abs,
                                      projection_backend=projection_backend)
            with observer:
                observer.begin_probe()
                yield observer
                if observer._pending:
                    raise RuntimeError("Stage B spill has pending backward observations")
                observer.active = False
            self._end_capture()
        except BaseException:
            self._failed = True
            raise
        self.telemetry["records_per_probe"].append(self._records_seen - records_before)
        self.telemetry["capture_wall_s"].append(time.time() - started)
        self._captured += 1

    def _start_arenas(self):
        if self._arenas:
            return
        count = ARENA_COUNT if self._threads else 1
        self._arenas = [_Arena(self.arena_bytes, self._cuda) for _ in range(count)]
        self._free = queue.Queue()
        for arena in self._arenas[1:]:
            self._free.put(arena)
        self._arena = self._arenas[0]
        if self._threads:
            self._pending = queue.Queue()
            self._writer = threading.Thread(target=self._writer_loop,
                                            name="stage-b-spill-writer", daemon=True)
            self._writer.start()

    def _record(self, name, x, selected):
        """Runs inside a backward hook: stage one invocation's operands."""
        self._require_healthy()
        index = self._window_of.get(name)
        if index is None:
            raise RuntimeError(f"Stage B spill observed an undeclared target {name}")
        if x.dtype != self.dtype or selected.dtype != self.dtype:
            raise RuntimeError(f"Stage B spill operand dtype differs for {name}")
        if x.device != self.device or selected.device != self.device:
            raise RuntimeError(f"Stage B spill operand residency differs for {name}")
        if not _is_dense(x):
            raise RuntimeError(
                f"Stage B spill refuses a non-dense input for {name}: the "
                "activation QDQ reads it in place, so a copy is not the same operand")
        window = self._windows[index]
        key = (x.untyped_storage().data_ptr(), x.storage_offset(), tuple(x.shape),
               tuple(x.stride()), x._version)
        held = window.dedupe.get(key)
        if held is None:
            owner = name
            layout = _layout(x, dense=True)
            nbytes = x.numel() * self.element_size
            entries = window.entries.setdefault(owner, [])
            entry = window.entry_cursor.get(owner, 0)
            if self._probe == 0:
                logical = window.x_logical.get(owner, 0)
                entries.append(_Entry(logical, nbytes, layout))
                window.x_logical[owner] = logical + nbytes
            elif (entry >= len(entries)
                    or not self._same_layout(entries[entry].layout, layout)
                    or entries[entry].nbytes != nbytes):
                raise RuntimeError(
                    f"Stage B spill probe {self._probe} input layout differs from "
                    f"probe 0 for {name}")
            window.entry_cursor[owner] = entry + 1
            # The held reference keeps this storage from being reused inside
            # the sample, so an address key cannot name two tensors.
            window.dedupe[key] = (owner, entry, x)
            self._stage(index, ("x", owner), entry, entries[entry].logical, x)
        else:
            owner, entry = held[0], held[1]
        if window.x_source.setdefault(name, owner) != owner:
            raise RuntimeError(
                f"Stage B spill Linear {name} reads inputs first read by more than "
                "one Linear; its rows would not be one input stream")
        g_layout = _layout(selected, dense=_is_dense(selected))
        g_bytes = selected.numel() * self.element_size
        logical = window.g_logical.get(name, 0)
        if self._probe == 0:
            window.records.append((name, owner, entry, logical, g_bytes, g_layout))
        else:
            cursor = window.record_cursor
            if (cursor >= len(window.records)
                    or window.records[cursor][:4] != (name, owner, entry, logical)
                    or window.records[cursor][4] != g_bytes
                    or not self._same_layout(window.records[cursor][5], g_layout)):
                raise RuntimeError(
                    f"Stage B spill probe {self._probe} invocation order or gradient "
                    f"layout differs from probe 0 at {name}")
        window.record_cursor += 1
        window.g_logical[name] = logical + g_bytes
        self._records_seen += 1
        self._stage(index, ("g", name), window.record_cursor - 1, logical, selected)

    def _same_layout(self, recorded, observed):
        """Probe 0's layout is the one every probe is replayed at.

        Shape and strides must match on every device. The address residue
        must match on CUDA, where the caching allocator's 512-byte blocks
        make it a deterministic function of the view offset, so a mismatch
        means a different operand. A host allocation's residue is not: the
        windowed replay itself runs each window's pass at a different host
        address, and host kernels on these fresh or elementwise operands
        do not depend on it.
        """
        return recorded[:2] == observed[:2] and (not self._cuda or recorded[2] == observed[2])

    def _stage(self, window_index, stream, index, logical, tensor):
        """Copy ``tensor`` into the arena as part of ``stream`` at ``logical``."""
        nbytes = tensor.numel() * self.element_size
        arena = self._arena
        offset = (arena.used + 255) & ~255
        if offset + nbytes > len(arena.view):
            self._flush()
            arena, offset = self._arena, 0
            if nbytes > len(arena.view):
                raise RuntimeError("Stage B spill tensor exceeds its arena bound")
        if nbytes:
            destination = arena.buffer.narrow(0, offset, nbytes).view(self.dtype)
            destination.copy_(_storage_order(tensor), non_blocking=self._cuda)
        arena.parts.append((window_index, stream, index, logical, nbytes, offset))
        arena.used = offset + nbytes
        arena.probe = self._probe

    def _end_batch(self):
        for window in self._windows:
            window.dedupe.clear()

    def _flush(self):
        arena = self._arena
        if not arena.parts:
            return
        if self._cuda:
            arena.event = torch.cuda.Event()
            arena.event.record(torch.cuda.current_stream(self.device))
        if self._threads:
            self._pending.put(arena)
            started = time.time()
            self._arena = self._free.get()
            self.telemetry["hook_wait_s"] += time.time() - started
            self._require_healthy()
        else:
            self._write_arena(arena)
            self._reset(arena)

    def _drain(self):
        """Flush the current arena and wait until every arena is written."""
        self._flush()
        if self._threads:
            parked = [self._arena]
            started = time.time()
            while len(parked) < len(self._arenas):
                parked.append(self._free.get())
            self.telemetry["hook_wait_s"] += time.time() - started
            self._arena = parked[0]
            for arena in parked[1:]:
                self._free.put(arena)
        self._require_healthy()

    @staticmethod
    def _reset(arena):
        arena.used, arena.parts, arena.event, arena.probe = 0, [], None, None

    def _writer_loop(self):
        while True:
            arena = self._pending.get()
            if arena is None:
                return
            try:
                if self._write_error is None:
                    self._write_arena(arena)
            except BaseException as exc:  # surfaced on the capture thread
                self._write_error = exc
            finally:
                self._reset(arena)
                self._free.put(arena)

    def _write_arena(self, arena):
        """Write one arena: one file run per (window, stream) it holds."""
        started = time.time()
        if arena.event is not None:
            arena.event.synchronize()
        probe, view = arena.probe, arena.view
        groups: dict[tuple, list] = {}
        for part in arena.parts:
            groups.setdefault((part[0], part[1]), []).append(part)
        spans = []
        for (window_index, (kind, stream)), parts in groups.items():
            window = self._windows[window_index]
            logical = parts[0][3]
            for part in parts:
                if part[3] != logical:
                    raise RuntimeError("Stage B spill stream is not contiguous")
                logical += part[4]
            total = logical - parts[0][3]
            if kind == "x":
                for _, _, entry, _, nbytes, offset in parts:
                    digest = hashlib.sha256(view[offset:offset + nbytes]).digest()
                    record = window.entries[stream][entry]
                    if probe == 0:
                        record.digest = digest
                    elif digest != record.digest:
                        raise RuntimeError(
                            f"Stage B spill probe {probe} input differs from probe 0 "
                            f"in window {window_index} ({stream}, entry {entry}); "
                            "the replay would not be the windowed arithmetic")
                    else:
                        self.telemetry["x_digest_checks"] += 1
                if probe != 0:
                    continue
                runs = window.x_runs.setdefault(stream, [])
            else:
                runs = window.g_runs.setdefault((stream, probe), [])
            if runs and runs[-1][0] + runs[-1][2] != parts[0][3]:
                raise RuntimeError("Stage B spill runs are out of stream order")
            if not runs and parts[0][3] != 0:
                raise RuntimeError("Stage B spill stream does not start at zero")
            file_offset = self._scratch.allocate(total)
            self._scratch.write(file_offset, [view[offset:offset + nbytes]
                                              for *_, nbytes, offset in parts])
            runs.append((parts[0][3], file_offset, total))
            spans.append((file_offset, total))
            key = "x_bytes_written" if kind == "x" else "g_bytes_written"
            self.telemetry[key] += total
            self.telemetry["runs_written"] += 1
        if spans:
            start = min(offset for offset, _ in spans)
            end = max(offset + size for offset, size in spans)
            self._scratch.sync_and_release(start, end - start)
        self.telemetry["writer_busy_s"] += time.time() - started

    def _end_capture(self):
        self._drain()
        probe = self._probe
        for index, window in enumerate(self._windows):
            if probe == 0:
                window.plan = self._plan(window)
                window.x_starts = {owner: [run[0] for run in runs]
                                   for owner, runs in window.x_runs.items()}
                self.telemetry["x_entries"] += sum(len(e) for e in window.entries.values())
                self.telemetry["x_streams"] += len(window.entries)
                self.telemetry["shared_input_linears"] += sum(
                    1 for name, owner in window.x_source.items() if name != owner)
            elif (window.record_cursor != len(window.records)
                    or any(window.entry_cursor.get(owner, 0) != len(entries)
                           for owner, entries in window.entries.items())):
                raise RuntimeError(
                    f"Stage B spill probe {probe} observed a different invocation "
                    f"count than probe 0 in window {index}")
            for name in {record[0] for record in window.records}:
                runs = window.g_runs.get((name, probe), [])
                window.g_starts[(name, probe)] = [run[0] for run in runs]
            if any(entry.digest is None for entries in window.entries.values()
                   for entry in entries):
                raise RuntimeError("Stage B spill input digest missing")
            window.dedupe.clear()
        if probe == self.n_probes - 1:
            # The last capture is done: release the pinned write arenas now.
            writer, self._writer = self._writer, None
            if writer is not None:
                self._pending.put(None)
                writer.join()
            self._arenas.clear()
            self._arena = None

    # -- replay ---------------------------------------------------------------
    def _plan(self, window):
        """Replay order: one input stream's records at a time, in read chunks.

        Each input stream's records (its owner's and those of any Linear that
        shares it) replay in firing order; every Linear belongs to exactly one
        stream, so every statistics key keeps its own order. A chunk fits one
        read buffer.
        """
        streams: dict[str, list[int]] = {}
        for position, record in enumerate(window.records):
            streams.setdefault(record[1], []).append(position)
        order = [name for name in window.names if name in streams]
        plan = []
        for owner in order:
            records, new, gradients, cursor, next_entry = [], [], [], 0, 0
            entries = window.entries[owner]
            for position in streams[owner]:
                _, _, entry, _, g_bytes, g_layout = window.records[position]
                need = []
                if entry == next_entry:
                    need.append(("x", entry, entries[entry].nbytes, entries[entry].layout[2]))
                elif entry > next_entry:
                    raise RuntimeError("Stage B spill entries are not in first-use order")
                need.append(("g", position, g_bytes, g_layout[2]))
                end = cursor
                for _, _, nbytes, residue in need:
                    end = _placed(end, residue) + nbytes
                if records and end > self.read_bytes:
                    plan.append((owner, (tuple(records), tuple(new), tuple(gradients), cursor)))
                    records, new, gradients, cursor = [], [], [], 0
                for kind, item, nbytes, residue in need:
                    offset = _placed(cursor, residue)
                    (new if kind == "x" else gradients).append((item, offset))
                    cursor = offset + nbytes
                if entry == next_entry:
                    next_entry += 1
                window.last_ref[(owner, entry)] = position
                records.append(position)
            if next_entry != len(entries):
                raise RuntimeError("Stage B spill input stream has an unread entry")
            if records:
                plan.append((owner, (tuple(records), tuple(new), tuple(gradients), cursor)))
        if any(chunk[3] > self.read_bytes for _, chunk in plan):
            raise RuntimeError("Stage B spill record exceeds its read buffer")
        return plan

    @staticmethod
    def _physical(runs, starts, logical, nbytes):
        run = bisect.bisect_right(starts, logical) - 1
        if run < 0:
            raise RuntimeError("Stage B spill stream index is incomplete")
        start, file_offset, size = runs[run]
        if logical + nbytes > start + size:
            raise RuntimeError("Stage B spill stream index is incomplete")
        return file_offset + (logical - start)

    def _fill(self, window, probe, item, buffer):
        """Read one chunk's tensors into ``buffer`` at their planned offsets."""
        owner, (_records, new, gradients, _used) = item
        view = buffer[1]
        pieces = []
        entries, x_runs, x_starts = (window.entries[owner], window.x_runs[owner],
                                     window.x_starts[owner])
        for entry, offset in new:
            record = entries[entry]
            pieces.append((self._physical(x_runs, x_starts, record.logical, record.nbytes),
                           view[offset:offset + record.nbytes]))
        for position, offset in gradients:
            name, _, _, logical, nbytes, _ = window.records[position]
            pieces.append((self._physical(window.g_runs[(name, probe)],
                                          window.g_starts[(name, probe)], logical, nbytes),
                           view[offset:offset + nbytes]))
        pieces.sort(key=lambda piece: piece[0])
        group, start, end = [], None, None
        for file_offset, target in pieces:
            if group and file_offset != end:
                self.telemetry["bytes_read"] += self._scratch.read_into(start, group)
                self.telemetry["reads"] += 1
                group = []
            if not group:
                start = end = file_offset
            group.append(target)
            end += len(target)
        if group:
            self.telemetry["bytes_read"] += self._scratch.read_into(start, group)
            self.telemetry["reads"] += 1

    def _start_read_buffers(self):
        if self._read_buffers:
            return
        count = READ_BUFFER_COUNT if self._threads else 1
        for _ in range(count):
            tensor = torch.empty(self.read_bytes, dtype=torch.uint8, pin_memory=self._cuda)
            # [tensor, memoryview, event of the H2D copy that last read it]
            self._read_buffers.append([tensor, memoryview(tensor.numpy()), None])

    def _chunks(self, window, probe, plan):
        """Yield ``(chunk, buffer)`` with bounded read-ahead on another thread."""
        self._start_read_buffers()
        if not self._threads:
            buffer = self._read_buffers[0]
            for item in plan:
                if buffer[2] is not None:
                    buffer[2].synchronize()
                self._fill(window, probe, item, buffer)
                yield item, buffer
            return
        free, ready, stop = queue.Queue(), queue.Queue(), threading.Event()
        for buffer in self._read_buffers:
            free.put(buffer)

        def read():
            try:
                for item in plan:
                    buffer = free.get()
                    if stop.is_set() or buffer is None:
                        return
                    if buffer[2] is not None:
                        buffer[2].synchronize()
                    self._fill(window, probe, item, buffer)
                    ready.put((item, buffer))
            except BaseException as exc:
                ready.put(exc)

        reader = threading.Thread(target=read, name="stage-b-spill-reader", daemon=True)
        reader.start()
        try:
            for _ in plan:
                started = time.time()
                item = ready.get()
                self.telemetry["reader_wait_s"] += time.time() - started
                if isinstance(item, BaseException):
                    raise RuntimeError("Stage B spill read failed") from item
                yield item
                free.put(item[1])
        finally:
            stop.set()
            free.put(None)
            reader.join()

    def replay(self, window_index, probe_index, lease):
        """Feed ``lease`` this window's spilled invocations for one probe."""
        self._require_healthy()
        if probe_index >= self._captured:
            raise RuntimeError("Stage B spill replay precedes its probe's capture")
        window = self._windows[window_index]
        if set(lease.modules) != set(window.names):
            raise RuntimeError("Stage B spill replay lease roster differs from its window")
        started = time.time()
        live: dict[tuple[str, int], torch.Tensor] = {}
        es = self.element_size
        try:
            for item, buffer in self._chunks(window, probe_index, window.plan):
                owner, (records, new, gradients, used) = item
                staging = torch.empty(used + 2 * ADDRESS_ALIGNMENT, dtype=torch.uint8,
                                      device=self.device)
                shift = (-staging.data_ptr()) % ADDRESS_ALIGNMENT
                staging.narrow(0, shift, used).copy_(buffer[0].narrow(0, 0, used),
                                                     non_blocking=self._cuda)
                if self._cuda:
                    buffer[2] = torch.cuda.Event()
                    buffer[2].record(torch.cuda.current_stream(self.device))
                typed = staging.narrow(0, 0, (staging.numel() // es) * es).view(self.dtype)
                entries = window.entries[owner]
                for entry, offset in new:
                    shape, stride, _ = entries[entry].layout
                    live[(owner, entry)] = torch.as_strided(
                        typed, shape, stride, (shift + offset) // es)
                gradient_at = dict(gradients)
                for position in records:
                    name, _, entry, _, _, (shape, stride, _) = window.records[position]
                    gradient = torch.as_strided(typed, shape, stride,
                                                (shift + gradient_at[position]) // es)
                    lease._observe_invocation(name, lease.modules[name].weight,
                                              live[(owner, entry)], gradient)
                    if window.last_ref[(owner, entry)] == position:
                        del live[(owner, entry)]
                del staging, typed
            if live:
                raise RuntimeError("Stage B spill replay left an input unconsumed")
        except BaseException:
            self._failed = True
            raise
        finally:
            live.clear()
        self.telemetry["replay_wall_s"] += time.time() - started
