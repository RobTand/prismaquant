"""Production-faithful rendered-weight cache for measured candidates.

Per-Linear candidate probes and real-KL gates need to measure the same
rendered weights that export will ship. Without this cache the perturbation
installed into the model is bare RTN. The export pipeline renders weights with
several activation-aware passes; the shipped δw is much smaller than the RTN
δw at the same format.

This module pre-renders `W_tilde[name, fmt]` once, using the production
quantization path:

  IMPLEMENTED (v1):
    * scalar GPTQ (with damp sweep when env-enabled)
    * optional scalar scale-sweep
    * joint NVFP4 fused-sibling globals (q/k/v share a per-tensor scale,
      gate/up share theirs)
    * calibrated `input_global_scale` per fused-sibling group
      (max_abs(activations) / 6.0; the same value the export persists
      to the artifact)
    * progressive local render gates using the shared mechanism order
      baseline -> format scale rule -> GPTQ -> optional scale_sweep;
      individual formats explicitly opt out of unsupported mechanisms,
      and regressive candidates fall back to the previous accepted render
      while recording metadata
    * FP8_DYNAMIC/FP8_E4M3 per-row scale search when scale_sweep is enabled;
      explicit MXFP8 E8M0 scale search remains opt-in. These refine the current
      accepted render rather than starting a separate format-specific path
    * activation-weighted GPTQ for FP8_DYNAMIC/FP8_E4M3. Explicit MXFP8 keeps
      GPTQ support for research/legacy artifacts. NVFP4 is the only production
      format that uses joint_scale_opt; MXFP8 uses the canonical E8M0 scale rule.
    * retired Fisher-weighted local objectives are archived under
      ``archive/fisher_2026-05-15/`` and are not part of the production
      pipeline
    * retired input-axis rotation experiments are not part of the production
      cache path.

  KNOWN GAPS (v2 work, NOT implemented):
    * batched NVFP4 GPTQ + scale-sweep across same-shape Linears
      (defaults-on in the export when activations are cached;
      mathematically equivalent to scalar but ~3-8× faster on MoE)
    * block-output match (post-GPTQ refinement against BF16 block output)
    * any export-only refinements added after this docstring is written

  FP8_DYNAMIC / BF16:
    * FP8_DYNAMIC is represented by the canonical FP8_E4M3 format name:
      per-output-row FP32 weight scales and per-token dynamic activation
      scales. It uses GPTQ damp-sweep by default in production render.
    * Explicit MXFP8/MXFP4 formats remain available only when requested.
    * BF16 is passthrough.

PerturbedActivationCache installs `W_tilde` (and applies the calibrated
`input_global_scale` on activations) instead of RTN-quantizing on the
fly, so per-Linear probes, frontier validation, and polish gates use the
same δw the export will deliver, modulo the v2 gaps above.

Usage:

    cache = fill_production_weight_cache(
        model, calib_ids, qnames=qnames, formats=["NVFP4"],
    )
    cache.validate_coverage(qnames, ["NVFP4"])  # raise on misses
    perturbed_cache = PerturbedActivationCache(
        ..., production_weight_cache=cache,
    )
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import stat
import zipfile
from typing import NamedTuple

import torch
import torch.nn as nn

from prismaquant.activation_sampling import update_priority_reservoir
from prismaquant.build_rtn_cache import iter_quantizable_tensors
from prismaquant.cost_stage_checkpoint import atomic_write_bytes, unique_temp_suffix
from prismaquant.digests import canonical_json
from prismaquant.render_score import (
    gate_render_candidate,
    normalize_row_weights,
    resolve_render_mechanism_order,
    score_render_error,
)
from prismaquant.source_prefetch import prefetch_files_to_page_cache


ACTIVATION_HOOK_SCOPE_SCHEMA = (
    "prismaquant.production_weight_cache.activation_hook_scope.v1"
)
ACTIVATION_HOOK_SCOPE_KEY = "activation_hook_scope"
RENDER_IDENTITY_SCHEMA = (
    "prismaquant.production_weight_cache.render_identity.v1"
)
RENDER_IDENTITY_SIDECAR_FILENAME = "render_identity.json"
#: The resolved NVFP4 input-global-scale policy, as a render lever (#227).
#: ``_resolve_production_render_levers`` is the ONE place this process's
#: policy is turned into a stamped value, so the directory render identity,
#: the CB render contract, the union per-shard identity, the score records and
#: every consumer that reads ``ProductionWeightCache.levers`` all name the same
#: one.  The policy stays a live setting in the contract owner; what is frozen
#: here is what THIS fill priced with.
RENDER_LEVER_INPUT_GLOBAL_SCALE_POLICY = "nvfp4_input_global_scale_policy"
RENDER_IDENTITY_MTP_APPEND_SCHEMA = (
    "prismaquant.production_weight_cache.mtp_append_identity.v1"
)
RENDER_IDENTITY_PACKED_APPEND_SCHEMA = (
    "prismaquant.production_weight_cache.packed_append_identity.v1"
)
RENDER_IDENTITY_PACKED_STREAMING_APPEND_SCHEMA = (
    "prismaquant.production_weight_cache.packed_streaming_append_identity.v1"
)
MTP_APPEND_SIDECAR_KEY = "mtp_append"
PACKED_APPEND_SIDECAR_KEY = "packed_expert_append"
PACKED_STREAMING_APPEND_SIDECAR_KEY = "packed_expert_streaming_append"
PRE_GUARD_ADMISSION_SIDECAR_KEY = "pre_guard_admission"


# Formats whose WEIGHT PLANE is the same artifact and must therefore get the
# same production render.
#
# NVFP4 and NVFP4A16 are one serialization: identical packed 4-bit weights,
# identical group scales, identical per-tensor global scale. They differ only in
# whether the config group declares `input_activations`, which vLLM reads to
# pick W4A4 (CUTLASS) or W4A16 (Marlin) at RUNTIME. The bytes on disk are the
# same bytes.
#
# The render gate used to be `fmt == "NVFP4"` literally, so NVFP4A16 fell
# through to the registry RTN path and shipped weights that had never seen
# GPTQ, static_act_order or JSO -- measured max|diff| 0.0217 against NVFP4 on a
# 512x1024 Linear. That made A16 strictly WORSE on the weight plane, so any
# A16-vs-A4 comparison measured the rendering difference rather than the
# activation contract: a rendering confound, principle 8's exact prohibition,
# and it silently biased the allocator against ever choosing A16.
#
# GPTQ's objective is ||WX - W_hat X|| under the CALIBRATION activations, which
# are the true unquantized ones in both contracts. Nothing in the weight solve
# depends on whether activations are quantized at serve time, so the same
# render is not merely acceptable for both -- it is the correct one, and
# sharing it makes the activation axis genuinely byte-free AND render-free.
# That is what lets one production cache entry serve both contracts.
NVFP4_RENDER_EQUIVALENT = frozenset({"NVFP4", "NVFP4A16"})


def _render_base_format(fmt: str) -> str:
    """The render-base spelling of a requested format name.

    Upper-casing is this module's one normalizer for a format identity: it
    also spells the cache filenames, the manifest keys and the render-score
    records, so the render base has to agree with them and it stays as it is.
    Enumerated over the whole registry (80 rows + 3 aliases), it changes the
    spelling of exactly one registered name -- ``INT4_W4A16_g128``, the only
    mixed-case row -- and no other name, alias or Tessera rung.  That one name
    is not made unresolvable HERE, though: it is unresolvable because case was
    decided at each caller instead of in the registry, so the fix is in
    ``format_registry.canonical_format_name``, which now resolves a name
    case-insensitively as its last probe (#218).  Do not paper over it here.
    """
    return str(fmt).strip().upper()


def _cache_weight_filename(qname: str, fmt: str) -> str:
    safe = qname.replace("/", "__").replace(".", "_")
    return f"{safe}__{fmt}.pt"


def _cache_pair_identity_filename(qname: str, fmt: str) -> str:
    return _cache_weight_filename(qname, fmt) + ".identity.json"


_UNCACHED_PACKED_EXPERT_RE = re.compile(
    r"\.experts(?:\.\d+)?\."
    r"(?:gate_up_proj|down_proj|gate_proj|up_proj|w1|w2|w3)$"
)


def is_uncached_packed_expert_qname(qname: str) -> bool:
    """Return True for packed-MoE expert tensor qnames.

    Legacy residency helpers may skip these names when no concrete packed
    render is expected. Production build/export gates must be stricter:
    non-BF16 packed experts are rendered by
    ``fill_packed_expert_cache_entries`` and missing entries are an early
    coverage failure unless the explicit RTN research escape hatch is set.
    """
    return bool(_UNCACHED_PACKED_EXPERT_RE.search(str(qname)))


_PACKED_EXPERT_PARAM_RE = re.compile(
    r"\.experts\."
    r"(?:gate_up_proj|down_proj|gate_proj|up_proj|w1|w2|w3)$"
)


def is_packed_expert_param_qname(qname: str) -> bool:
    """Return True only for PACKED (3-D, un-indexed) expert param qnames.

    Stricter than ``is_uncached_packed_expert_qname``, which also matches
    per-expert-indexed names (``...experts.5.down_proj``) — those are plain
    nn.Linear modules (DSv4/MiniMax layouts) whose activation replay capture
    IS per-projection-correct. Use this predicate when a behavior must apply
    only to packed tensors, where one module plan carries several projection
    params and the module-input capture is the wrong tensor for ``down``.
    """
    return bool(_PACKED_EXPERT_PARAM_RE.search(str(qname)))


#: Reader-lease windows a retained PWC window pinned its staged renders
#: under (PQ #1210), in the exact activation cache's counter shape (PQ #997):
#: ``windows_batched`` lease windows serving ``entries_batched`` entries, and
#: ``batch_fallbacks`` windows whose batched lease was refused and whose
#: loads then leased one entry at a time. Only the window's own thread
#: updates them.
PWC_WINDOW_LEASE_COUNTERS = {"windows_batched": 0, "entries_batched": 0,
                             "batch_fallbacks": 0}


class _WindowLeaseRef(NamedTuple):
    """The declared file one retained-window group-lease member stands for."""
    path: str


class _WindowTrackedWeights(dict):
    """Index resident owners while retaining the public cache's dict API.

    The large disk-backed roster needs one initial walk. Later window checks
    inspect only live tensor keys, including values directly assigned by a
    caller. Storage identities themselves are recomputed at every boundary so
    views, aliases and changed tensor backing cannot undercount memory.
    """

    def __init__(self, values=()):
        super().__init__(values)
        self.tensor_keys = {key for key, value in self.items()
                            if isinstance(value, torch.Tensor)}

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if isinstance(value, torch.Tensor):
            self.tensor_keys.add(key)
        else:
            self.tensor_keys.discard(key)

    def __delitem__(self, key):
        super().__delitem__(key)
        self.tensor_keys.discard(key)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def __ior__(self, other):
        self.update(other)
        return self

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, *default):
        value = super().pop(key, *default)
        self.tensor_keys.discard(key)
        return value

    def popitem(self):
        key, value = super().popitem()
        self.tensor_keys.discard(key)
        return key, value

    def clear(self):
        super().clear()
        self.tensor_keys.clear()

    @classmethod
    def fromkeys(cls, keys, value=None):
        return cls(dict.fromkeys(keys, value))

    def __reduce__(self):
        # The index is process-local bookkeeping, regenerated on demand.
        return dict, (dict(self),)


@dataclass
class ProductionWeightCache:
    """Dict-like cache of production-faithful dequantized weights.

    Keys: ``(qname, fmt_canonical)``.  Values are EITHER:
      * ``torch.Tensor`` ([out, in] float32 or bf16) — in-memory cache
      * ``str`` (a path) — points to a per-Linear .pt file on disk;
        ``get()`` lazy-loads on first access and memoizes the tensor

    Disk-streaming mode (when ``cache_dir`` is set during fill) keeps
    fill-time peak memory bounded — only one weight is in RAM at a time
    instead of the full ~25 GB stack of all rendered Linears.  At
    validation or recache time the lazy-load caches each weight in memory
    after first access, so steady-state behavior matches the in-memory mode.

    ``activation_max_abs[qname]`` is the calibrated max(|activations|)
    used by the act-clip step in the export pipeline.  PerturbedActivation
    Cache reads this and clamps activations to ``[-max_abs, +max_abs]``
    before per-group RTN, matching the export's act-clip behavior.

    Note: the *exported metadata* convention for this field is
    ``input_global_scale = 448 * 6 / max_abs`` (reciprocal — vLLM
    multiplies activations by it; legacy ``6 / max_abs`` under
    PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE=0).  We store ``max_abs``
    directly here because that's the value the act-clip path needs;
    consumers convert via
    ``export_native_compressed._nvfp4_input_global_scale_from_max_abs``.
    """
    weights: dict[tuple[str, str], object]  # tensor OR str(path)
    levers: dict[str, bool]
    activation_max_abs: dict[str, float] | None = None
    failed: dict[tuple[str, str], str] | None = None
    cache_dir: str | None = None  # set when disk-streaming was used at fill time
    metadata: dict[str, object] | None = None
    # Backward-compat alias for code that still reads ``activation_scales``.
    activation_scales: dict[str, float] | None = None
    # LRU eviction state for memoized tensor loads.  When non-None, the
    # in-memory cache holds at most ``mem_lru_max_bytes`` of tensor data;
    # least-recently-used entries are evicted back to their on-disk
    # filename when the budget is exceeded.  Default OFF for backward
    # compat; opt-in via ``enable_lru(...)``.
    _lru_order: list[tuple[str, str]] | None = None
    _lru_paths: dict[tuple[str, str], str] | None = None
    _lru_bytes: int = 0
    _lru_max_bytes: int = 0
    _cb_verified_keys: set[tuple[str, str]] | None = None
    _file_load_max_bytes: int = 0
    _file_load_receipts: dict | None = None
    _expected_file_sha256: dict[tuple[str, str], str] | None = None

    def __post_init__(self) -> None:
        # Normalize to ``activation_max_abs`` if a caller used the legacy
        # name.  After this, both attributes hold the same dict (max_abs).
        if self.activation_max_abs is None and self.activation_scales is not None:
            self.activation_max_abs = self.activation_scales
        elif self.activation_scales is None and self.activation_max_abs is not None:
            self.activation_scales = self.activation_max_abs

    def validate_cb_render_identity(
        self,
        *,
        expected_context=None,
        expected_qnames: Sequence[str] | None = None,
        col_weights: Mapping[str, torch.Tensor] | None = None,
        require_for_formats: Sequence[str] = (),
        require_source_complete: bool = True,
        where: str = "ProductionWeightCache",
    ):
        """Validate and return this cache's persisted CB producer context.

        The context is read from the cache metadata, never reconstructed from
        the current process environment.  Non-CB caches return ``None``.
        """
        return validate_production_cache_cb_render_identity(
            self,
            expected_context=expected_context,
            expected_qnames=expected_qnames,
            col_weights=col_weights,
            require_for_formats=require_for_formats,
            require_source_complete=require_source_complete,
            where=where,
        )

    def enable_lru(self, max_bytes: int) -> None:
        """Bound the in-memory tensor footprint to ``max_bytes`` via LRU
        eviction.  Required for very large disk-streamed caches (e.g.
        Qwen3.6-27B's ~46 GB of bf16 weights wouldn't fit in a 121 GB
        UMA box alongside the model + working set)."""
        self._lru_max_bytes = int(max_bytes)
        self._lru_order = []
        self._lru_paths = {}
        self._lru_bytes = 0
        self._file_load_receipts = None

    def _evict_to_budget(self) -> None:
        if self._lru_order is None or self._lru_max_bytes <= 0:
            return
        while self._lru_bytes > self._lru_max_bytes and self._lru_order:
            evict_key = self._lru_order.pop(0)
            t = self.weights.get(evict_key)
            if isinstance(t, torch.Tensor):
                self._lru_bytes -= t.element_size() * t.numel()
                # Restore the filename so subsequent lookups still resolve.
                if self._lru_paths is not None and evict_key in self._lru_paths:
                    self.weights[evict_key] = self._lru_paths[evict_key]
                if self._cb_verified_keys is not None:
                    self._cb_verified_keys.discard(evict_key)
                if self._file_load_receipts is not None:
                    self._file_load_receipts.pop(evict_key, None)

    def compact_for_pickle(self) -> int:
        """Restore disk-backed resident tensors to path references.

        A recache/polish pass may lazily load many disk-streamed entries into
        ``weights``.  Pickling that state would serialize the tensors and turn a
        small manifest into a multi-GB file.  This method keeps the cache
        portable by replacing resident LRU-loaded tensors with their original
        paths before serialization.  Returns the number of entries compacted.
        """
        compacted = 0
        for key, path in (self._lru_paths or {}).items():
            if isinstance(self.weights.get(key), torch.Tensor):
                self.weights[key] = path
                compacted += 1
        if self.cache_dir:
            cache_dir = Path(self.cache_dir)
            for key, value in list(self.weights.items()):
                if not isinstance(value, torch.Tensor):
                    continue
                fname = _cache_weight_filename(key[0], key[1])
                if (cache_dir / fname).is_file():
                    self.weights[key] = fname
                    compacted += 1
        self._lru_order = [] if self._lru_order is not None else None
        self._lru_bytes = 0
        self._cb_verified_keys = None
        self._file_load_receipts = None
        self._forget_window_archive_bytes()
        return compacted

    def release_resident_tensors(self, keys: Sequence[tuple[str, str]] | None = None) -> int:
        """Drop disk-backed resident tensors, keeping every key resolvable.

        Same operation as :meth:`compact_for_pickle` — that one is named for
        why it was first needed — but callers reach for this one when the goal
        is *memory*, not serialization: after a one-way install has copied the
        render set into a live model, nothing reads the cache's resident copies
        again, and on a 27B assignment they are up to the full LRU cap of dead
        bytes inside a shared 121.6 GB pool.

        Entries that were never disk-backed are left resident, because dropping
        those would lose data rather than free a re-readable copy. A later
        ``get()`` on a released key simply reloads it from disk.

        ``keys`` limits release to entries with a recorded load path. It never
        adopts an unrelated same-named file as backing for an in-memory tensor.
        Omitting ``keys`` preserves the legacy whole-cache compaction behavior.
        """
        if keys is None:
            return self.compact_for_pickle()
        released = 0
        for key in dict.fromkeys(keys):
            value = self.weights.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            path = (self._lru_paths or {}).get(key)
            if path is None:
                continue  # A non-disk-backed value remains its only owner.
            self.weights[key] = path
            if self._lru_order is not None and key in self._lru_order:
                self._lru_order.remove(key)
                self._lru_bytes -= value.numel() * value.element_size()
            if self._cb_verified_keys is not None:
                self._cb_verified_keys.discard(key)
            if self._file_load_receipts is not None:
                self._file_load_receipts.pop(key, None)
            released += 1
        return released

    @staticmethod
    def _window_storage(tensor):
        if (type(tensor) not in (torch.Tensor, nn.Parameter)
                or tensor.layout != torch.strided or tensor.device.type == 'meta'):
            raise RuntimeError('PWC resident window has unaccountable tensor storage')
        storage = tensor.untyped_storage()
        return (tensor.device, storage.data_ptr()), storage.nbytes()

    def _window_resident_storages(self):
        if not isinstance(self.weights, _WindowTrackedWeights):
            self.weights = _WindowTrackedWeights(self.weights)
        storages = {}
        for key in self.weights.tensor_keys:
            identity, nbytes = self._window_storage(self.weights[key])
            storages[identity] = max(storages.get(identity, 0), nbytes)
        return storages

    @staticmethod
    def _window_archive_storage_bytes(source):
        """Only ordinary uncompressed Torch archives have accountable loads."""
        from .perturbed_x_cache import torch_archive_storage_bytes
        return torch_archive_storage_bytes(source)

    def _window_keys(self, keys):
        if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)):
            raise TypeError('PWC window keys must be a finite sequence')
        resolved, seen = [], set()
        for pair in keys:
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or any(not isinstance(part, str) or not part for part in pair)):
                raise ValueError('PWC window needs (name, format) keys')
            key = self.resolve_key(*pair)
            if key is None:
                raise RuntimeError(f'PWC window missing cache entry {pair}')
            if key not in seen:
                resolved.append(key)
                seen.add(key)
        return tuple(resolved)

    @staticmethod
    def _window_limits(max_resident_bytes, max_workers):
        if type(max_resident_bytes) is not int or max_resident_bytes <= 0:
            raise ValueError('PWC window requires a positive resident byte budget')
        if (type(max_workers) is not int or max_workers <= 0
                or max_workers > len(os.sched_getaffinity(0))):
            raise ValueError('PWC window workers exceed assigned CPU affinity')

    def _window_file(self, key):
        value = self.weights[key]
        if not isinstance(value, (str, Path)):
            raise RuntimeError('PWC window has an unaccountable cache input')
        path = Path(self._path_for_value(value)).absolute()
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError('PWC window requires a regular file, not a symlink')
        # ONE ARCHIVE SCAN PER FILE PER LIFETIME. The storage total is a pure
        # function of the file's bytes, and the file identity this cache
        # already trusts for that -- the stat signature every window read
        # re-checks -- is the memo key, so a file that changed is a miss and is
        # rescanned. A research plan runs preflight more than once over the
        # same key and each scan was a fresh open plus a central-directory
        # read over cold NFS: 11.0% of the prepare's main-thread wall time
        # (#693). The bytes-backed scan at ``_load_file_tensor`` is a different
        # call on the loader thread and is never memoized. A retained window
        # does not scan here at all (``_window_file_bound``, PQ #1210).
        memo = self._window_archive_memo()
        signature = self._file_signature(before)
        remembered = memo.get(str(path))
        if remembered is not None and remembered[0] == signature:
            storage_bytes = remembered[1]
        else:
            storage_bytes = self._window_archive_storage_bytes(path)
            if self._file_signature(path.lstat()) != signature:
                raise RuntimeError('PWC window file changed during preflight')
            memo[str(path)] = (signature, storage_bytes)
        estimate = self.estimate_nbytes([key])
        if estimate != before.st_size or storage_bytes > estimate:
            raise RuntimeError('PWC window file storage estimate changed')
        return path, before, estimate, storage_bytes

    def _window_file_bound(self, key):
        """Stat one retained-window file and charge its length (PQ #1210).

        The retained window charges each file its length, the bound the
        sealed plan already charges (``retained_admission_targets``): an
        ordinary uncompressed Torch archive stores every tensor byte inside
        the file, so its archive storage cannot exceed the file's length.
        The archive itself is parsed once, on the bytes the loader has just
        read and hashed (``_decode_file_tensor``), before anything is
        deserialized, and a load whose storage exceeds this charge, or whose
        archive is not an ordinary uncompressed Torch archive, fails its
        window there. Parsing it again here meant a second open of every
        declared pool file only to read its central directory: 5.7% of row
        041's main thread in the render windows, on top of the loads.
        """
        value = self.weights[key]
        if not isinstance(value, (str, Path)):
            raise RuntimeError('PWC window has an unaccountable cache input')
        path = Path(self._path_for_value(value)).absolute()
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError('PWC window requires a regular file, not a symlink')
        estimate = self.estimate_nbytes([key])
        if estimate != before.st_size:
            raise RuntimeError('PWC window file storage estimate changed')
        return path, before, estimate

    def _window_archive_memo(self):
        """Per-file archive storage totals for the current window lifetime.

        One small entry per distinct backing path, so the memo cannot outgrow
        the roster the cache already holds a path for, and it is dropped when a
        window closes or the cache is compacted.
        """
        memo = getattr(self, '_window_archive_bytes', None)
        if memo is None:
            memo = self._window_archive_bytes = {}
        return memo

    def _forget_window_archive_bytes(self) -> None:
        self._window_archive_bytes = None

    def plan_resident_windows(self, keys, *, max_resident_bytes: int, max_workers: int,
                              max_load_buffer_bytes: int | None = None):
        """Plan finite research quanta in input order without loading tensors.

        All existing PWC tensor backing storages count, including unrelated
        entries and storage hidden behind views; aliases count once. Incoming
        standard uncompressed Torch files use the existing conservative file
        estimate.

        A quantum is closed by the two byte budgets its caller already admits
        and nothing else: the resident cap covers every backing storage, and
        the serialized cap covers the load buffers the quantum reads at once
        (default: the resident cap, matching ``resident_window``). ``max_workers``
        is the loader concurrency, which ``prefetch`` bounds on its own pool; it
        is no longer a second, unpriced width cap. That cap made the width of a
        quantum a function of the CPU count instead of the admitted bytes, and
        on the joint-AURA walk it split every five-render unit into a four-key
        quantum at 12.5% of its byte budget plus a one-key quantum that read a
        single file on a single thread (#693).

        Plans are key-only hints; ``resident_window`` revalidates each entry.
        """
        self._window_limits(max_resident_bytes, max_workers)
        buffer_cap = (max_resident_bytes if max_load_buffer_bytes is None
                      else max_load_buffer_bytes)
        if type(buffer_cap) is not int or buffer_cap <= 0:
            raise ValueError('PWC window needs a positive serialized buffer budget')
        keys = self._window_keys(keys)
        baseline = sum(self._window_resident_storages().values())
        if baseline > max_resident_bytes:
            raise RuntimeError('PWC existing resident storage exceeds window budget')
        windows, window, nbytes, buffer_bytes = [], [], baseline, 0
        for key in keys:
            value = self.weights[key]
            incoming = 0 if isinstance(value, torch.Tensor) else self._window_file(key)[2]
            if baseline + incoming > max_resident_bytes:
                raise RuntimeError(f'PWC single entry exceeds resident window budget: {key}')
            if incoming > buffer_cap:
                raise RuntimeError(f'PWC single serialized load buffer exceeds window budget: {key}')
            if window and (nbytes + incoming > max_resident_bytes
                           or buffer_bytes + incoming > buffer_cap):
                windows.append(tuple(window))
                window, nbytes, buffer_bytes = [], baseline, 0
            window.append(key)
            nbytes += incoming
            buffer_bytes += incoming
        if window:
            windows.append(tuple(window))
        return tuple(windows)

    def _retained_window_preflight(self, keys, *, max_resident_bytes,
                                   max_workers, max_load_buffer_bytes):
        """Price one retained tensor lifetime and its bounded load quanta."""
        self._window_limits(max_resident_bytes, max_workers)
        buffer_cap = (max_resident_bytes if max_load_buffer_bytes is None
                      else max_load_buffer_bytes)
        if type(buffer_cap) is not int or buffer_cap <= 0:
            raise ValueError('PWC retained window needs a positive serialized buffer budget')
        keys = self._window_keys(keys)
        if not keys:
            raise RuntimeError('PWC retained window needs nonempty keys')
        baseline = sum(self._window_resident_storages().values())
        if baseline > max_resident_bytes:
            raise RuntimeError('PWC existing resident storage exceeds retained window budget')

        # Preflight every selected file before the first load. The persistent
        # charge is the file's length, which bounds its archive storage (see
        # ``_window_file_bound``), while the serialized file buffer is charged
        # only within the quantum that reads it.
        files, file_costs = {}, {}
        persistent_bytes = baseline
        incoming_lru_bytes = 0
        quanta, quantum = [], []
        quantum_buffer_bytes = peak_buffer_bytes = 0
        for key in keys:
            value = self.weights[key]
            file_bytes = storage_bytes = 0
            if not isinstance(value, torch.Tensor):
                path, observed, file_bytes = self._window_file_bound(key)
                storage_bytes = file_bytes
                # A bound, not an exact total: the read's own parse must not
                # exceed it (``_decode_file_tensor``).
                files[str(path)] = (observed, file_bytes, storage_bytes, False)
                file_costs[key] = (str(path), file_bytes, storage_bytes)
                persistent_bytes += storage_bytes
                incoming_lru_bytes += storage_bytes
                if file_bytes > buffer_cap:
                    raise RuntimeError(f'PWC single serialized load buffer exceeds budget: {key}')
                if persistent_bytes > max_resident_bytes:
                    raise RuntimeError(f'PWC retained resident storage exceeds budget: {key}')
            if quantum and (len(quantum) == max_workers
                            or quantum_buffer_bytes + file_bytes > buffer_cap):
                quanta.append(tuple(quantum))
                peak_buffer_bytes = max(peak_buffer_bytes, quantum_buffer_bytes)
                quantum, quantum_buffer_bytes = [], 0
            quantum.append(key)
            quantum_buffer_bytes += file_bytes
        if quantum:
            quanta.append(tuple(quantum))
            peak_buffer_bytes = max(peak_buffer_bytes, quantum_buffer_bytes)
        if (self._lru_order is not None and self._lru_max_bytes > 0
                and self._lru_bytes + incoming_lru_bytes > self._lru_max_bytes):
            raise RuntimeError('PWC retained window exceeds available LRU budget')
        return (keys, tuple(quanta), files, file_costs, peak_buffer_bytes,
                buffer_cap)

    def plan_retained_window(self, keys, *, max_resident_bytes: int,
                             max_workers: int, max_load_buffer_bytes: int | None = None):
        """Return key-only bounded prefetch quanta for one retained lifetime.

        All selected files and existing tensor storages are checked without
        deserializing. This is a hint: ``retained_window`` repeats preflight
        immediately before opening the context and guards each actual read.
        """
        return self._retained_window_preflight(
            keys, max_resident_bytes=max_resident_bytes, max_workers=max_workers,
            max_load_buffer_bytes=max_load_buffer_bytes)[1]

    def retained_key_costs(self, keys):
        """Price selected incoming PWC entries without loading or roster walks.

        Returned keys are concrete aliases. Existing resident tensors have no
        *incoming* charge; ``retained_window`` counts their complete backing
        storages in its baseline. A disk entry is charged its file length for
        both its incoming storage and its serialized buffer, exactly as the
        retained window charges it (``_window_file_bound``, PQ #1210) and as
        the sealed plan does (``retained_admission_targets``). The context
        repeats this preflight before loading because paths can change.
        """
        costs = {}
        for key in self._window_keys(keys):
            if isinstance(self.weights[key], torch.Tensor):
                self._window_storage(self.weights[key])
                costs[key] = {'incoming_storage_bytes': 0, 'serialized_bytes': 0}
            else:
                _, _, file_bytes = self._window_file_bound(key)
                costs[key] = {'incoming_storage_bytes': file_bytes,
                              'serialized_bytes': file_bytes}
        return costs

    def retained_read_entries(self, keys, *, group, render_identities: bool = False):
        """The IO engine entries that load ``keys`` for one retained window.

        One ``io_engine.ReadEntry`` per selected disk-backed key, in window
        order, all in ``group``. Each is priced exactly as the retained window
        prices it (``_window_file_bound``: the file's length bounds both its
        serialized buffer and its archive storage), bound to its expected
        digest, and decoded by :meth:`_decode_file_tensor` against that same
        bound. With ``render_identities`` the reading thread also hashes the
        tensor it decoded (PQ #1192), before the consumer sees it. A stream of
        these entries can run ahead of the window that will admit them
        (:meth:`retained_window` with ``stream=``, PQ #1291).

        The engine reads each file into a sealed memfd and the decoder maps it
        (:meth:`_decode_file_tensor`), so a decoded render holds exactly its
        file's pages (to within one 4 KiB page) until it is dropped: the
        entry's charge is its file length, with no separate measure. The
        mapping is private, so a write to a render would copy pages into
        anonymous memory rather than touch the sealed bytes. Nothing on the
        Stage B path writes one: a probe reads each render through
        :meth:`resident_render_identity` (a hash) and copies it to the device
        (``.to(..., copy=True)``) before it subtracts the source. The
        identity's guard includes the tensor's version counter, so a
        ``torch`` write in place fails the next probe's identity read
        (``tests/test_render_identity_once_1192.py`` writes a streamed render
        in place and requires that refusal); a write that bypasses the
        counter (through NumPy or a raw pointer) is not seen, and none exists
        on this path.
        """
        from .io_engine import ReadEntry
        expected = getattr(self, "_expected_file_sha256", None)
        bound = getattr(self, "_file_load_max_bytes", 0)
        derive = (partial(self._loaded_render_identity, requested=True)
                  if render_identities else None)
        entries = []
        for key in self._window_keys(keys):
            if isinstance(self.weights[key], torch.Tensor):
                continue
            path, observed, file_bytes = self._window_file_bound(key)
            window_entry = (observed, file_bytes, file_bytes, False)
            entries.append(ReadEntry(
                key=key, path=str(path), size=file_bytes,
                limit=min(bound, file_bytes) if bound else file_bytes,
                held_bytes=file_bytes,
                expected_sha256=None if expected is None else expected.get(key),
                decoder=partial(self._decode_file_tensor, window_entry),
                group=group, declared_stat=observed, derive=derive))
        return entries

    @contextmanager
    def retained_window(self, keys, *, max_resident_bytes: int, max_workers: int,
                        max_load_buffer_bytes: int | None = None,
                        release_file_pages: bool = False,
                        before_load_quantum=None,
                        render_identities: bool = False,
                        stream=None, stream_group=None):
        """Keep many selected renders resident across repeated consumer passes.

        Only PWC owns the tensors. Every selected file is preflighted and
        charged before the first load, then every key must be resident before
        the consumer runs. The cap covers all PWC backing storages; serialized
        buffers have a separate cap. The consumer must release borrowed
        references before context exit. Selected disk-backed entries are
        restored to paths on success or failure. When requested, checked file
        page advice runs once the window's loads are done. It is not a
        guarantee of physical reclaim; ``before_load_quantum`` may enforce a
        live host guard before the loads, on this thread, using current
        resident bytes, the incoming storage not yet read and its serialized
        buffers.

        The loads go through the IO engine (``io_engine``, PQ #1294). With
        ``stream``, an ``io_engine.ReadStream`` built from
        :meth:`retained_read_entries`, this window takes ``stream_group`` from
        a stream that may already have read it while an earlier window
        computed (PQ #1291), and the guard is charged only what is still
        unread. Without one, the window reads its own entries on demand. Either
        way each file is read once, hashed once against its bound digest,
        its archive parsed once on those bytes, under the staged renders'
        group lease where the strict tier policy pins one (PQ #1210), and the
        serialized buffers in flight stay within ``max_load_buffer_bytes``.
        ``load_quanta`` in the receipt is the preflight's admission plan: the
        width ``max_workers`` and the buffer cap admit each quantum; the
        engine's own concurrency follows its measured rates within the same
        buffer cap.

        With ``render_identities``, each reading thread also hashes the tensor
        it loaded (``_cb_cache_tensor_identity``) before the tensor is handed
        out, and ``resident_render_identity`` serves that hash for the rest
        of the load's lifetime (PQ #1192).
        """
        if getattr(self, '_resident_window_files', None) is not None:
            raise RuntimeError('PWC resident windows cannot be nested')
        if type(release_file_pages) is not bool:
            raise ValueError('PWC window page release must be boolean')
        if before_load_quantum is not None and not callable(before_load_quantum):
            raise TypeError('PWC before_load_quantum must be callable')
        if type(render_identities) is not bool:
            raise ValueError('PWC window render identities must be boolean')
        (keys, quanta, files, file_costs, peak_buffer_bytes, buffer_cap) = (
            self._retained_window_preflight(
                keys, max_resident_bytes=max_resident_bytes,
                max_workers=max_workers,
                max_load_buffer_bytes=max_load_buffer_bytes))
        own = None
        if stream is None and file_costs:
            from .io_engine import FixedBudget, read_stream
            own = stream = read_stream(
                self.retained_read_entries(keys, group=0,
                                           render_identities=render_identities),
                budget=FixedBudget(buffer_bytes=buffer_cap),
                lease_counters=PWC_WINDOW_LEASE_COUNTERS)
            stream_group = 0
        elif stream is not None:
            if stream.group_keys(stream_group) != tuple(key for key in keys if key in file_costs):
                raise RuntimeError('PWC retained window keys differ from its stream group')
        self._resident_window_files = files
        self._resident_window_receipt_keys = frozenset(file_costs)
        self._resident_window_render_identities = render_identities
        try:
            loaded = 0
            if file_costs:
                unread_bytes, unread_serialized = stream.demand(stream_group)
                if before_load_quantum is not None:
                    before_load_quantum({
                        'resident_bytes': sum(self._window_resident_storages().values()),
                        'remaining_incoming_storage_bytes': unread_bytes,
                        'next_serialized_bytes': min(unread_serialized, buffer_cap),
                    })
                loaded = self._admit_prefetched(
                    (item.entry.key, self.weights.get(item.entry.key), item.value,
                     item.observed, item.derived)
                    for item in stream.take(stream_group))
            for key in keys:
                self.get_resident(*key)
            if sum(self._window_resident_storages().values()) > max_resident_bytes:
                raise RuntimeError('PWC actual backing storage exceeds retained window budget')
            advised_paths = set()
            if release_file_pages:
                from .perturbed_x_cache import release_activation_cache_file_pages
                for path, (observed, *_rest) in files.items():
                    release_activation_cache_file_pages(path, expected_stat=observed)
                    advised_paths.add(path)
            # Catch selected source drift before exposing the window.
            for key in keys:
                self.get_resident(*key)
            actual = sum(self._window_resident_storages().values())
            if actual > max_resident_bytes:
                raise RuntimeError('PWC actual backing storage exceeds retained window budget')
            yield {'keys': keys, 'loaded': loaded, 'resident_bytes': actual,
                   'budget_bytes': max_resident_bytes,
                   'load_buffer_capacity_bytes': peak_buffer_bytes,
                   'load_buffer_budget_bytes': buffer_cap,
                   'file_pages_advised': len(advised_paths),
                   'load_quanta': quanta}
        finally:
            # The window's tensors go first: the stream charges them to its
            # budget until it is released, so releasing it while they were
            # still alive would let it read into memory they hold.
            try:
                self.release_resident_tensors(keys)
            finally:
                try:
                    if own is not None:
                        own.close()
                    elif stream is not None and file_costs:
                        stream.release()
                finally:
                    self._resident_window_files = None
                    self._resident_window_receipt_keys = frozenset()
                    self._resident_window_render_identities = False
                    self._forget_window_archive_bytes()

    @contextmanager
    def resident_window(self, keys, *, max_resident_bytes: int, max_workers: int,
                        max_load_buffer_bytes: int | None = None, release_file_pages: bool = False):
        """Prefetch one research quantum, then expose strict resident lookups.

        This owns no tensor dictionary. PWC remains the only cache; selected
        disk-backed entries are released on success/failure. Callers must drop
        their borrowed tensor references at the boundary. Nested windows refuse.
        The resident cap covers all cache backing storages. Serialized loader
        buffers have a separate aggregate cap (default: resident cap); both
        caps, allocator overhead and consumer workspaces need phase admission.
        Page-release advice is optional and is not a physical-memory guarantee.
        """
        if getattr(self, '_resident_window_files', None) is not None:
            raise RuntimeError('PWC resident windows cannot be nested')
        if type(release_file_pages) is not bool:
            raise ValueError('PWC window page release must be boolean')
        buffer_cap = max_resident_bytes if max_load_buffer_bytes is None else max_load_buffer_bytes
        if type(buffer_cap) is not int or buffer_cap <= 0:
            raise ValueError('PWC window needs a positive serialized buffer budget')
        windows = self.plan_resident_windows(keys, max_resident_bytes=max_resident_bytes,
                                             max_workers=max_workers,
                                             max_load_buffer_bytes=buffer_cap)
        if len(windows) != 1:
            raise RuntimeError('PWC resident_window requires one nonempty planned quantum')
        keys = windows[0]
        files = {}
        for key in keys:
            value = self.weights[key]
            if not isinstance(value, torch.Tensor):
                path, observed, estimate, storage_bytes = self._window_file(key)
                # Exact: the read must parse to this storage total.
                files[str(path)] = (observed, estimate, storage_bytes, True)
        # Different keys reading one file still allocate separate load buffers.
        buffer_bytes = sum(files[str(Path(self._path_for_value(self.weights[key])).absolute())][1]
                           for key in keys if not isinstance(self.weights[key], torch.Tensor))
        if buffer_bytes > buffer_cap:
            raise RuntimeError('PWC serialized load buffers exceed window budget')
        incoming_storage = sum(files[str(Path(self._path_for_value(self.weights[key])).absolute())][2]
                               for key in keys if not isinstance(self.weights[key], torch.Tensor))
        if (self._lru_order is not None and self._lru_max_bytes > 0
                and self._lru_bytes + incoming_storage > self._lru_max_bytes):
            raise RuntimeError('PWC resident window exceeds available LRU budget')
        self._resident_window_files = files
        self._resident_window_receipt_keys = frozenset(
            key for key in keys if not isinstance(self.weights[key], torch.Tensor))
        try:
            loaded = self.prefetch(keys, max_workers=max_workers)
            for key in keys:
                self.get_resident(*key)
            actual = sum(self._window_resident_storages().values())
            if actual > max_resident_bytes:
                raise RuntimeError('PWC actual backing storage exceeds resident window budget')
            if release_file_pages:
                from .perturbed_x_cache import release_activation_cache_file_pages
                for path, (observed, *_rest) in files.items():
                    release_activation_cache_file_pages(path, expected_stat=observed)
            yield {'keys': keys, 'loaded': loaded, 'resident_bytes': actual,
                   'budget_bytes': max_resident_bytes, 'load_buffer_capacity_bytes': buffer_bytes,
                   'load_buffer_budget_bytes': buffer_cap, 'file_pages_advised': len(files) if release_file_pages else 0}
        finally:
            self.release_resident_tensors(keys)
            self._resident_window_files = None
            self._resident_window_receipt_keys = frozenset()
            self._forget_window_archive_bytes()

    def _path_for_value(self, value: object) -> str:
        path = str(value)
        if self.cache_dir and not Path(path).is_absolute():
            path = str(Path(self.cache_dir) / path)
        return path

    def _name_candidates(self, name: str) -> list[str]:
        candidates = [name]
        if name.endswith(".weight"):
            candidates.append(name[:-len(".weight")])
        if name.startswith("model.language_model."):
            candidates.append("model." + name[len("model.language_model."):])
        return list(dict.fromkeys(candidates))

    def _format_candidates(self, fmt: str) -> list[str]:
        raw = str(fmt)
        candidates = [raw, raw.upper()]
        try:
            from prismaquant import format_registry as fr
            candidates.append(fr.canonical_format_name(raw))
        except Exception:
            pass
        if "MXFP8_E4M3" in candidates:
            candidates.append("MXFP8")
        if "MXFP8" in candidates:
            candidates.append("MXFP8_E4M3")
        if "FP8_E4M3" in candidates:
            candidates.append("FP8")
        if "FP8" in candidates:
            candidates.append("FP8_E4M3")
        return list(dict.fromkeys(candidates))

    def resolve_key(self, name: str, fmt: str) -> tuple[str, str] | None:
        """Resolve recipe aliases to the concrete stored cache key."""
        for cand in self._name_candidates(name):
            for fmt_cand in self._format_candidates(fmt):
                key = (cand, fmt_cand)
                if key in self.weights:
                    return key
        return None

    def estimate_nbytes(
        self,
        keys: Sequence[tuple[str, str]] | None = None,
    ) -> int:
        """Estimate resident bytes for cache entries without loading them."""
        total = 0
        for key in (list(self.weights) if keys is None else list(keys)):
            value = self.weights.get(key)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                total += value.element_size() * value.numel()
            else:
                total += Path(self._path_for_value(value)).stat().st_size
        return total

    def assignment_keys(
        self,
        assignment: Mapping[str, str],
        *,
        include_packed_experts: bool = False,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Return concrete non-BF16 cache keys needed by an assignment.

        This centralizes recipe alias handling for recache, polish, KL
        probes, and export: callers should ask the cache which stored key a
        recipe entry maps to, then feed those keys into ``prefetch``. Legacy
        residency callers leave ``include_packed_experts`` false because
        packed experts may be handled out-of-band; production build/export
        gates set it true once ``fill_packed_expert_cache_entries`` is
        responsible for those 3-D entries.
        """
        from prismaquant import format_registry as fr

        cb_formats = [
            (str(qname), fr.canonical_format_name(str(fmt)))
            for qname, fmt in assignment.items()
            if _is_cb_format_name(fr.canonical_format_name(str(fmt)))
        ]
        if cb_formats:
            cb_expected = []
            for qname, fmt in cb_formats:
                resolved = self.resolve_key(qname, fmt)
                if resolved is not None:
                    cb_expected.append(resolved[0])
            self.validate_cb_render_identity(
                expected_qnames=cb_expected,
                require_for_formats=[fmt for _qname, fmt in cb_formats],
                where="ProductionWeightCache assignment",
            )

        keys: list[tuple[str, str]] = []
        missing: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for qname, fmt in assignment.items():
            fmt_canon = fr.canonical_format_name(str(fmt))
            if fmt_canon == "BF16":
                continue
            key = self.resolve_key(str(qname), fmt_canon)
            if key is None:
                if (
                    is_uncached_packed_expert_qname(str(qname))
                    and not include_packed_experts
                ):
                    continue
                missing.append((str(qname), fmt_canon))
                continue
            if key not in seen:
                keys.append(key)
                seen.add(key)
        return keys, missing

    def assignment_file_paths(
        self,
        assignment: Mapping[str, str],
    ) -> tuple[list[Path], list[tuple[str, str]], list[tuple[str, str]]]:
        """Return disk files backing an assignment without loading tensors.

        This is a page-cache residency helper for validation paths that
        destructively materialize one assignment into the model. It reuses the
        same cache key resolution as ``prefetch_assignment`` but intentionally
        does not call ``torch.load`` or create another rendered-weight cache.
        """
        keys, missing = self.assignment_keys(assignment)
        paths: list[Path] = []
        in_memory: list[tuple[str, str]] = []
        seen_paths: set[Path] = set()
        for key in keys:
            value = self.weights.get(key)
            if value is None:
                missing.append(key)
                continue
            if isinstance(value, torch.Tensor):
                path_value = (
                    self._lru_paths.get(key)
                    if self._lru_paths is not None else None
                )
                if path_value is None:
                    in_memory.append(key)
                    continue
                value = path_value
            path = Path(self._path_for_value(value)).resolve()
            if path not in seen_paths:
                paths.append(path)
                seen_paths.add(path)
        return paths, missing, in_memory

    def prefetch_assignment_file_pages(
        self,
        assignment: Mapping[str, str],
        *,
        mode: str = "require",
        max_resident_bytes: int | None = None,
        headroom_gb: float = 24.0,
        max_workers: int = 4,
        progress: bool = False,
        log_prefix: str = "[prod-cache-files]",
    ) -> dict[str, object]:
        """Prefetch assignment cache files into the OS page cache.

        Unlike ``prefetch_assignment``, this keeps rendered weights out of the
        Python heap. The following ``get`` calls still go through
        ``ProductionWeightCache`` and its LRU, but deserialization reads from
        resident file pages instead of faulting against NVMe.
        """
        paths, missing, in_memory = self.assignment_file_paths(assignment)
        mode = str(mode or "off").lower()
        if paths:
            stats = prefetch_files_to_page_cache(
                paths,
                mode=mode,
                max_resident_bytes=max_resident_bytes,
                headroom_gb=headroom_gb,
                workers=max_workers,
                progress=progress,
                log_prefix=log_prefix,
                label="production cache files",
            )
        else:
            stats = {
                "mode": mode,
                "label": "production cache files",
                "files": 0,
                "bytes": 0,
                "max_resident_bytes": int(max_resident_bytes or 0),
                "available_bytes": None,
                "prefetched_bytes": 0,
                "elapsed_seconds": 0.0,
                "skipped": True,
                "reason": "no disk-backed production cache files",
            }
        stats["keys"] = len(paths) + len(in_memory)
        stats["in_memory"] = len(in_memory)
        stats["missing"] = len(missing)
        if missing:
            stats["missing_sample"] = missing[:8]
            msg = (
                f"production cache missing {len(missing)} assignment entries; "
                f"sample={missing[:8]}"
            )
            if mode == "require":
                raise RuntimeError(msg)
            if progress:
                print(f"{log_prefix} WARNING: {msg}", flush=True)
        return stats

    def prefetch_assignment(
        self,
        assignment: Mapping[str, str],
        *,
        max_resident_bytes: int | None = None,
        max_workers: int = 4,
        require: bool = False,
        progress: bool = False,
        log_prefix: str = "[prod-cache]",
    ) -> dict[str, object]:
        """Prefetch rendered weights required by a concrete assignment.

        ``require`` converts missing entries or resident-budget overflow into
        a hard failure.  That is the production-safe mode for GPU-bound
        recache/export runs because it prevents accidental NVMe streaming.
        """
        keys, missing = self.assignment_keys(assignment)
        nbytes = self.estimate_nbytes(keys)
        budget = (
            int(max_resident_bytes)
            if max_resident_bytes is not None and int(max_resident_bytes) > 0
            else None
        )
        stats: dict[str, object] = {
            "keys": len(keys),
            "missing": len(missing),
            "bytes": int(nbytes),
            "budget_bytes": int(budget or 0),
            "loaded": 0,
            "skipped": False,
        }
        if missing:
            stats["missing_sample"] = missing[:8]
            msg = (
                f"production cache missing {len(missing)} assignment entries; "
                f"sample={missing[:8]}"
            )
            if require:
                raise RuntimeError(msg)
            if progress:
                print(f"{log_prefix} WARNING: {msg}", flush=True)
        if budget is not None and nbytes > budget:
            stats["skipped"] = True
            msg = (
                "production cache preload would exceed resident budget: "
                f"{nbytes / 1024**3:.2f} GiB needed, "
                f"{budget / 1024**3:.2f} GiB budget"
            )
            if require:
                raise RuntimeError(msg)
            if progress:
                print(f"{log_prefix} WARNING: {msg}; skipping preload", flush=True)
            return stats

        if progress:
            print(
                f"{log_prefix} preloading production cache: "
                f"{len(keys)} entries, {nbytes / 1024**3:.2f} GiB",
                flush=True,
            )
        loaded = self.prefetch(keys, max_workers=max_workers)
        stats["loaded"] = int(loaded)
        if progress:
            print(
                f"{log_prefix} preloaded {loaded}/{len(keys)} production "
                "cache entries",
                flush=True,
            )
        return stats

    def _record_lru_load(
        self,
        key: tuple[str, str],
        original_value: object,
        tensor: torch.Tensor,
    ) -> None:
        if self._lru_paths is None:
            self._lru_paths = {}
        if key not in self._lru_paths:
            self._lru_paths[key] = str(original_value)
        if self._lru_order is None:
            return
        if key in self._lru_order:
            self._lru_order.remove(key)
        self._lru_bytes += tensor.element_size() * tensor.numel()
        self._lru_order.append(key)
        self._evict_to_budget()

    def _validate_loaded_cb_pair_tensor(
        self,
        key: tuple[str, str],
        tensor: torch.Tensor,
    ) -> None:
        """Verify an admitted CB shard on its necessary consumer load.

        Resume admission validates the small identity sidecar without an eager
        full-cache scan. The first real consumer load verifies the logical
        tensor digest from the cache manifest, combining integrity checking
        with I/O AURA/export already has to perform.
        """
        if not _is_cb_format_name(key[1]):
            return
        if self._cb_verified_keys is not None and key in self._cb_verified_keys:
            return
        metadata = self.metadata if isinstance(self.metadata, Mapping) else {}
        artifact_set = metadata.get("cb_cache_pair_artifacts")
        if artifact_set is None:
            return
        if not isinstance(artifact_set, Mapping):
            raise RuntimeError(
                "ProductionWeightCache CB pair artifact metadata is malformed"
            )
        records = artifact_set.get("records")
        record_key = f"{key[0]}|{key[1]}"
        record = records.get(record_key) if isinstance(records, Mapping) else None
        expected = record.get("tensor") if isinstance(record, Mapping) else None
        if not isinstance(expected, Mapping):
            raise RuntimeError(
                f"ProductionWeightCache CB pair artifact identity is missing "
                f"for {key[0]}@{key[1]}"
            )
        observed = _cb_cache_tensor_identity(tensor)
        difference = first_identity_difference(
            expected,
            observed,
            path="tensor",
        )
        if difference is not None:
            field, stored, current = difference
            raise RuntimeError(
                f"ProductionWeightCache CB shard integrity refused for "
                f"{key[0]}@{key[1]}: identity field '{field}' differs: "
                f"stored={identity_value_for_error(stored)} "
                f"current={identity_value_for_error(current)}"
            )
        if self._cb_verified_keys is None:
            self._cb_verified_keys = set()
        self._cb_verified_keys.add(key)

    def enable_file_load_receipts(self, *, max_file_bytes: int) -> None:
        """Capture content provenance on the necessary, bounded PWC file read.

        Receipts describe bytes actually deserialized, not a previous file scan.
        They exist only while the matching tensor remains resident and unchanged.
        This is opt-in for preparation; ordinary cache loads retain their path.
        """
        if type(max_file_bytes) is not int or max_file_bytes <= 0:
            raise ValueError("PWC file receipt needs a positive read buffer bound")
        if self._lru_order is None or self._lru_max_bytes <= 0:
            raise RuntimeError("PWC file receipts require bounded LRU residency")
        if any(isinstance(value, torch.Tensor) for value in self.weights.values()):
            raise RuntimeError("PWC file receipt capture requires unloaded entries")
        self._file_load_max_bytes = max_file_bytes
        self._file_load_receipts = {}
        self._expected_file_sha256 = None

    def require_file_load_sha256(
        self, expected: Mapping[tuple[str, str], str], *, max_file_bytes: int,
    ) -> None:
        """Bind all prepared renders to the bytes this cache actually loads.

        The existing bounded loader hashes its serialized read before the
        tensor enters the LRU. A resident hit also checks the file/tensor
        receipt so an already loaded tensor cannot escape its lifetime fence.
        """
        if not isinstance(expected, Mapping) or set(expected) != set(self.weights):
            raise ValueError("expected SHA256 must cover the complete PWC render roster")
        if any(not isinstance(value, str) for value in self.weights.values()):
            raise RuntimeError("prepared PWC render proof requires disk-backed entries")
        if any(not isinstance(digest, str) or len(digest) != 64 or
               any(c not in "0123456789abcdef" for c in digest)
               for digest in expected.values()):
            raise ValueError("prepared PWC render proof requires SHA256 digests")
        self.enable_file_load_receipts(max_file_bytes=max_file_bytes)
        self._expected_file_sha256 = dict(expected)

    def _check_expected_file_sha256(self, key, observed) -> None:
        expected = getattr(self, "_expected_file_sha256", None)
        if expected is None:
            return
        receipt = observed[0] if observed is not None else None
        if receipt is None or receipt["sha256"] != expected[key]:
            raise RuntimeError(f"{key[0]}@{key[1]}: prepared PWC render checksum changed")

    def disable_file_load_receipts(self) -> None:
        self._file_load_max_bytes = 0
        self._file_load_receipts = None
        self._expected_file_sha256 = None

    @staticmethod
    def _file_signature(value):
        from .perturbed_x_cache import cache_file_stat_signature
        return cache_file_stat_signature(value)

    @staticmethod
    def _file_tensor_guard(tensor):
        return (tensor._version, tensor.untyped_storage().data_ptr(),
                tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()),
                str(tensor.dtype), str(tensor.device))

    def _load_file_tensor(self, value, key=None):
        """Load one shard, from PrismaBuild's stage tier when it holds it.

        The bounded read is ``io_engine.load_file``: the tier comes from the
        residency map, the bytes are hashed once and held to the map's digest,
        and this cache's decoder (:meth:`_decode_file_tensor`) turns them into
        the tensor. A load inside an active resident window is bounded by, and
        fenced against, that window's preflight of the file.

        The unbounded ``torch.load`` branch is deliberately not redirected: it
        computes no digest, so a staged copy there would be admitted on the
        map's word alone. Both joint stages take the bounded branch (prepare
        through ``enable_file_load_receipts``, run through
        ``require_file_load_sha256``). Under the active allowed-tier policy the
        unbounded branch refuses outright (no digest, no bulk HDD open), and
        the bounded branch requires the caller's digest binding.
        """
        from .staged_tier_policy import policy_is_active, refuse_pool_bulk_read
        path = Path(self._path_for_value(value)).absolute()
        limit = getattr(self, "_file_load_max_bytes", 0)
        window_files = getattr(self, '_resident_window_files', None)
        window_entry = None
        if window_files is not None:
            window_entry = window_files.get(str(path))
            if window_entry is None:
                raise RuntimeError('PWC load is outside the active resident window')
            limit = min(limit, window_entry[1]) if limit else window_entry[1]
        if not limit:
            if policy_is_active():
                # No digest is computed on this branch, so a staged copy
                # would be admitted on the map's word alone — and the pool
                # copy is a bulk HDD open either way. Refuse outright.
                raise refuse_pool_bulk_read(
                    str(path), "unbounded-read-has-no-digest-binding")
            return torch.load(path, map_location="cpu", weights_only=True), None
        from .io_engine import load_file
        expected = getattr(self, "_expected_file_sha256", None)
        binding = None if expected is None or key is None else expected.get(key)
        return load_file(
            path, limit, binding=binding,
            decode=partial(self._decode_file_tensor, window_entry),
            declared_signature=(None if window_entry is None
                                else self._file_signature(window_entry[0])))

    def _decode_file_tensor(self, window_entry, raw, receipt, staged):
        """Turn one file's verified bytes into its tensor: ``(tensor, guard)``.

        The decoder ``io_engine.load_file`` runs on the bytes it just read and
        hashed. With a window entry (the file's preflight), the archive is
        parsed once here, before anything is deserialized: a resident window's
        preflight parsed the file and this total must equal it; a retained
        window charged the file's length, which this total must not exceed
        (``_window_file_bound``, PQ #1210). A staged copy that fails a fence
        raises ``StagedReadRefused``, which falls back like a read fence does.

        A stream's read arrives as an ``io_engine.SealedBuffer`` (PQ #1291):
        the archive is parsed through the memfd and the tensor is loaded with
        ``mmap=True`` from it, so the tensor maps the verified pages instead of
        copying them into a ``torch`` allocation, which on this platform would
        not return its pages to the cgroup when freed. The mapping is
        private: nothing written to the tensor reaches the sealed bytes.
        """
        from .io_engine import SealedBuffer
        from .residency_map import StagedReadRefused
        del receipt
        sealed = isinstance(raw, SealedBuffer)
        if window_entry is not None:
            _observed, _buffer, storage_bound, exact = window_entry
            if sealed:
                with open(raw.path, "rb") as archive:
                    storage_bytes = self._window_archive_storage_bytes(archive)
            else:
                storage_bytes = self._window_archive_storage_bytes(io.BytesIO(raw))
            if exact and storage_bytes != storage_bound:
                if staged:
                    raise StagedReadRefused(
                        'staged archive storage differs from the window preflight')
                raise RuntimeError('PWC window archive storage changed during its read')
            if not exact and storage_bytes > storage_bound:
                if staged:
                    raise StagedReadRefused(
                        'staged archive storage exceeds the window charge')
                raise RuntimeError('PWC window archive storage exceeds its charged bound')
        if sealed:
            tensor = torch.load(raw.path, mmap=True, map_location="cpu", weights_only=True)
        else:
            tensor = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            if staged:
                raise StagedReadRefused('staged copy is not a tensor shard')
            raise RuntimeError("PWC file receipt requires a tensor shard")
        if window_entry is not None and self._window_storage(tensor)[1] > storage_bytes:
            raise RuntimeError('PWC loaded backing storage exceeds its archive bound')
        return tensor, self._file_tensor_guard(tensor)

    def _record_file_load(self, key, tensor, observed) -> None:
        if observed is not None and self.weights.get(key) is tensor:
            if self._file_load_receipts is None:
                self._file_load_receipts = {}
            # This is the same resident object, released with its LRU entry.
            # The third slot holds what is derived from this one load, such
            # as its render identity, and is dropped with the receipt.
            self._file_load_receipts[key] = (tensor, observed, {})

    def _loaded_render_identity(self, tensor, observed, requested=None):
        """Hash one just-loaded render on its loader thread, if the window asks.

        Only a retained window opened with ``render_identities`` asks (the
        active window's flag, or ``requested`` for an entry read ahead of its
        window), and only a load with a receipt (``observed``) can keep the
        result. The tensor is still private to the reading thread here, so its
        bytes and its guard are the ones the receipt records. ``hashlib``
        releases the GIL for each 8 MiB chunk, so the readers hash in parallel.
        """
        if requested is None:
            requested = getattr(self, '_resident_window_render_identities', False)
        if observed is None or not requested:
            return None
        return {"rendered_identity": _cb_cache_tensor_identity(tensor)}

    def resident_render_identity(self, name: str, fmt: str, tensor) -> dict:
        """Return the content identity of one resident render, hashed once per load.

        The identity is ``_cb_cache_tensor_identity`` of the tensor: its
        shape, dtype, logical bytes and the SHA-256 of its bytes. A Stage B
        window reads each resident render once per probe, and hashing it on
        every read repeated the same work (PQ #1192).

        The identity is kept in the file-load receipt of the exact tensor
        object this cache loaded, so it lives exactly as long as that
        receipt. Every path that ends a resident lifetime drops the receipt
        and the identity with it: ``release_resident_tensors`` (a retained
        window's exit), an LRU eviction, ``enable_lru``,
        ``compact_for_pickle`` and a failed ``file_load_receipt``. A later
        load of the same key is a new object with a new receipt, and it is
        hashed again. The identity is served only while ``tensor`` is still
        this key's resident object and its guard (version counter, storage
        pointer, offset, shape, stride, dtype and device) equals the guard
        taken when its bytes were loaded; otherwise the call refuses. A
        tensor this cache did not load through a receipt refuses too, so a
        caller holding one hashes it itself.
        """
        key = self.resolve_key(name, fmt)
        entry = None if key is None else (self._file_load_receipts or {}).get(key)
        if (entry is None or entry[0] is not tensor
                or self.weights.get(key) is not tensor):
            raise RuntimeError(
                f"PWC render identity has no matching resident load: {name}@{fmt}")
        if self._file_tensor_guard(tensor) != entry[1][2]:
            raise RuntimeError(f"PWC resident render changed after its load: {name}@{fmt}")
        derived = entry[2]
        if "rendered_identity" not in derived:
            derived["rendered_identity"] = _cb_cache_tensor_identity(tensor)
        identity = derived["rendered_identity"]
        # A copy, so a caller that edits its record cannot edit the memo.
        return {**identity, "shape": list(identity["shape"])}

    def file_load_receipt(self, key, tensor) -> dict:
        """Return a detached receipt for this exact resident tensor lifetime."""
        entry = (self._file_load_receipts or {}).get(key)
        if entry is None or entry[0] is not tensor or self.weights.get(key) is not tensor:
            if self._file_load_receipts is not None:
                self._file_load_receipts.pop(key, None)
            raise RuntimeError("PWC file receipt has no matching resident tensor")
        receipt, signature, guard = entry[1]
        if (self._file_tensor_guard(tensor) != guard or
                self._file_signature(Path(receipt["path"]).lstat()) != signature):
            self._file_load_receipts.pop(key, None)
            raise RuntimeError("PWC file receipt tensor or source file changed")
        return dict(receipt)

    def prefetch(self, keys: Sequence[tuple[str, str]] | None = None,
                 max_workers: int = 4, *, executor=None) -> int:
        """Eagerly load (a subset of) cache entries via a thread pool.

        ``keys=None`` prefetches every entry that's still on disk (the
        common case at polish startup).  Returns the number of newly-
        materialized tensors.

        Disk-streamed caches typically have torch.load latency ~50 ms
        per file (deserialization-bound, not I/O-bound).  Loading
        serially through 496 entries = ~25 sec; with 4 threads this
        drops to ~6 sec.  Subsequent ``.get()`` calls hit the in-memory
        copy (no torch.load), so per-trial materialization in polish
        becomes essentially free.

        ``executor`` is a caller-owned pool of at most ``max_workers``
        threads to load on instead of a fresh one (a retained window keeps
        one for all its quanta, PQ #1210). This call is still a barrier:
        it returns, or raises, only after every load it submitted has
        finished, so nothing it started is in flight past it.
        """
        from concurrent.futures import ThreadPoolExecutor, wait

        if getattr(self, "_file_load_max_bytes", 0):
            if type(max_workers) is not int or not 0 < max_workers <= len(os.sched_getaffinity(0)):
                raise ValueError("PWC file receipt workers exceed assigned CPU affinity")
        if keys is None:
            keys = [k for k, v in self.weights.items()
                    if not isinstance(v, torch.Tensor)]
        else:
            keys = [k for k in keys
                    if not isinstance(self.weights.get(k), torch.Tensor)]
        if not keys:
            return 0

        def _load_one(key):
            value = self.weights.get(key)
            if value is None or isinstance(value, torch.Tensor):
                return None
            tensor, receipt = self._load_file_tensor(value, key)
            # Hashed here, on the loader thread, before the tensor is handed
            # out, when the window asked for it (PQ #1192).
            return key, value, tensor, receipt, self._loaded_render_identity(tensor, receipt)

        if executor is None:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return self._admit_prefetched(pool.map(_load_one, keys))
        futures = [executor.submit(_load_one, key) for key in keys]
        try:
            return self._admit_prefetched(future.result() for future in futures)
        finally:
            for future in futures:
                future.cancel()
            wait(futures)

    def _admit_prefetched(self, items) -> int:
        """Admit ``prefetch``'s loads, in key order, as each one lands."""
        loaded_count = 0
        for item in items:
            if item is None:
                continue
            key, original_value, tensor, receipt, derived = item
            if isinstance(self.weights.get(key), torch.Tensor):
                continue
            self._check_expected_file_sha256(key, receipt)
            self._validate_loaded_cb_pair_tensor(key, tensor)
            self.weights[key] = tensor
            self._record_lru_load(key, original_value, tensor)
            self._record_file_load(key, tensor, receipt)
            entry = (self._file_load_receipts or {}).get(key)
            if derived and entry is not None and entry[0] is tensor:
                entry[2].update(derived)
            if getattr(self, "_expected_file_sha256", None) is not None:
                self.file_load_receipt(key, tensor)
            loaded_count += 1
        return loaded_count

    def _resolve_to_tensor(self, key: tuple[str, str], *, resident_only=False) -> torch.Tensor | None:
        """Return the tensor at ``key`` (lazy-load from disk if needed).
        With LRU enabled, the freshly-loaded tensor is bookkept and the
        oldest entries get evicted back to filenames when the byte budget
        is exceeded.  Returns None if the key isn't present."""
        v = self.weights.get(key)
        if resident_only and not isinstance(v, torch.Tensor):
            raise RuntimeError(f'PWC cache entry is not resident: {key}')
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            self._validate_loaded_cb_pair_tensor(key, v)
            if getattr(self, "_expected_file_sha256", None) is not None:
                self._check_expected_file_sha256(key, (self.file_load_receipt(key, v),))
            if resident_only and (getattr(self, '_file_load_max_bytes', 0)
                                  or key in (self._file_load_receipts or {})
                                  or key in getattr(self, '_resident_window_receipt_keys', ())):
                self.file_load_receipt(key, v)
            # Refresh LRU position.
            if self._lru_order is not None:
                if key in self._lru_order:
                    self._lru_order.remove(key)
                self._lru_order.append(key)
            return v
        # Treat anything non-tensor as a filename / path.
        loaded, receipt = self._load_file_tensor(v, key)
        self._check_expected_file_sha256(key, receipt)
        self._validate_loaded_cb_pair_tensor(key, loaded)
        self.weights[key] = loaded
        self._record_lru_load(key, v, loaded)
        self._record_file_load(key, loaded, receipt)
        if getattr(self, "_expected_file_sha256", None) is not None:
            self.file_load_receipt(key, loaded)
        return loaded

    def get(self, name: str, fmt: str, *, resident_only=False) -> torch.Tensor | None:
        key = self.resolve_key(name, fmt)
        if key is not None:
            if _is_cb_format_name(key[1]):
                self.validate_cb_render_identity(
                    expected_qnames=[key[0]],
                    require_for_formats=[key[1]],
                    where=f"ProductionWeightCache get({key[0]}@{key[1]})",
                )
            return self._resolve_to_tensor(key, resident_only=resident_only)
        if resident_only:
            raise RuntimeError(f'PWC missing cache entry: {name}@{fmt}')
        return None

    def get_resident(self, name: str, fmt: str) -> torch.Tensor:
        """Return a resident tensor with CB/receipt guards; never load a file."""
        return self.get(name, fmt, resident_only=True)

    def relocate(self, new_cache_dir: str | Path) -> None:
        """Point the cache at a new on-disk directory of .pt shards.

        Used when a pickled cache is moved to a new host or when a
        cache_dir set inside one container is re-mounted at a different
        path on a second container.  No tensor reload happens here; the
        next ``get()`` will resolve against the new path.
        """
        self.cache_dir = str(new_cache_dir) if new_cache_dir is not None else None

    def verify_files(
        self,
        expected: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, list[tuple[str, str]]]:
        """Verify every disk-resident cache entry's .pt file exists.

        Returns ``{"present": [...], "missing": [...], "in_memory": [...]}``
        keyed by (qname, fmt).  In-memory entries (already-loaded tensors)
        are reported separately and never count as missing.

        On a disk-streaming cache that has been moved or whose backing
        directory was deleted, this is the canonical way to detect the
        problem at startup rather than at first ``get()`` (which raises
        FileNotFoundError mid-polish).  Callers should treat any
        ``missing`` entry as fatal: the cache must be rebuilt or its
        directory restored before use.

        ``expected``, when given, restricts the check to that subset of
        keys.  Default checks every entry in ``self.weights``.
        """
        present: list[tuple[str, str]] = []
        missing: list[tuple[str, str]] = []
        in_memory: list[tuple[str, str]] = []
        keys = list(self.weights) if expected is None else list(expected)
        for key in keys:
            v = self.weights.get(key)
            if v is None:
                missing.append(key)
                continue
            if isinstance(v, torch.Tensor):
                in_memory.append(key)
                continue
            path = str(v)
            if self.cache_dir and not Path(path).is_absolute():
                path = str(Path(self.cache_dir) / path)
            if Path(path).is_file():
                present.append(key)
            else:
                missing.append(key)
        return {"present": present, "missing": missing, "in_memory": in_memory}

    def __contains__(self, key: tuple[str, str]) -> bool:
        # Mirror the alias-resolution that ``get`` performs.
        name, fmt = key
        return self.resolve_key(name, fmt) is not None

    def __len__(self) -> int:
        return len(self.weights)

    def coverage_report(
        self,
        expected_qnames: Sequence[str],
        formats: Sequence[str],
    ) -> dict:
        """Return a dict with ``hits``, ``misses``, ``failed`` lists keyed
        by (qname, fmt).  Use ``validate_coverage`` to raise on any miss.

        Crucially, this checks key membership only — does NOT lazy-load
        tensors — so it stays cheap on disk-streaming caches with
        thousands of entries totalling tens of GB.
        """
        hits: list[tuple[str, str]] = []
        misses: list[tuple[str, str]] = []
        for q in expected_qnames:
            for f in formats:
                if f.upper() == "BF16":
                    continue
                if self.resolve_key(q, f.upper()) is not None:
                    hits.append((q, f.upper()))
                else:
                    misses.append((q, f.upper()))
        return {
            "hits": hits,
            "misses": misses,
            "failed": list((self.failed or {}).keys()),
        }

    def validate_coverage(
        self,
        expected_qnames: Sequence[str],
        formats: Sequence[str],
    ) -> None:
        """Raise ``RuntimeError`` if any (qname, fmt) is missing from the
        cache.  Call this immediately after fill to catch silent gaps
        from naming aliases or render failures."""
        report = self.coverage_report(expected_qnames, formats)
        if report["misses"] or report["failed"]:
            samples = (report["misses"][:5] + report["failed"][:5])
            raise RuntimeError(
                f"ProductionWeightCache coverage failure: "
                f"{len(report['misses'])} misses, "
                f"{len(report['failed'])} failed renders; "
                f"sample={samples}"
            )

class _LinearActivationCollector:
    """Hook every quantizable nn.Linear's input on a forward pass.

    Stores up to ``max_rows`` rows of activations per Linear (concatenated
    across calibration samples) on the configured resident device.  Only handles
    ``nn.Linear`` for now — packed MoE experts route through different
    APIs in the export pipeline and would need a separate collector.

    ``store_qnames`` controls which Linears get full activation tensors
    stored (memory-bounded by ``max_rows``).  All Linears in
    ``qnames`` get a per-Linear scalar ``max_abs`` recorded — that's
    cheap (one float per Linear) and needed by the cache's act-clip
    metadata even for Linears whose render is skipped via resume.
    """

    def __init__(
        self,
        model: nn.Module,
        qnames: set[str],
        max_rows: int,
        store_qnames: set[str] | None = None,
        *,
        store_device: torch.device | str | None = None,
        store_dtype: torch.dtype = torch.float32,
        profile=None,
    ):
        self.model = model
        self.profile = profile
        self.qnames = qnames
        self.store_qnames = set(store_qnames) if store_qnames is not None else set(qnames)
        self.max_rows = int(max_rows)
        self.store_device = torch.device(store_device or "cpu")
        self.store_dtype = store_dtype
        self.activations: dict[str, list[torch.Tensor]] = {}
        self._activation_priorities: dict[str, torch.Tensor] = {}
        self._activation_generator = torch.Generator(device="cpu")
        self._activation_generator.manual_seed(42)
        self.max_abs: dict[str, float] = {}
        self._max_abs_tensors: dict[str, torch.Tensor] = {}
        self._handles: list = []
        self._name_by_id: dict[int, str] = {}
        for full_name, mod, attr in iter_quantizable_tensors(model, self.profile):
            if attr != "weight" or not isinstance(mod, nn.Linear):
                continue
            qname = full_name[:-7] if full_name.endswith(".weight") else full_name
            if qname not in qnames and full_name not in qnames:
                continue
            key = qname
            self._name_by_id[id(mod)] = key
            if key in self.store_qnames:
                self.activations[key] = []

    def install(self) -> None:
        for mod_id, key in self._name_by_id.items():
            for full_name, mod, attr in iter_quantizable_tensors(
                self.model,
                self.profile,
            ):
                if id(mod) != mod_id or attr != "weight":
                    continue
                self._handles.append(
                    mod.register_forward_pre_hook(self._make_hook(key))
                )
                break

    def _make_hook(self, key: str):
        def hook(module, args):
            if not args:
                return
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return
            # Always update the cheap per-Linear max_abs scalar — needed
            # even for Linears we won't store activations for (so cache
            # has act-clip values for every assigned Linear).
            x_abs_max = x.detach().abs().amax()
            prev = self._max_abs_tensors.get(key)
            self._max_abs_tensors[key] = (
                x_abs_max.detach()
                if prev is None
                else torch.maximum(prev, x_abs_max.detach())
            )
            # Draw this call's row priorities for EVERY hooked Linear,
            # stored or not. One generator feeds every Linear's reservoir, so
            # the slice of the stream a Linear receives is a function of how
            # many rows every earlier hook consumed. Drawing only for stored
            # Linears makes a run that stores a subset -- a `--resume` build
            # that skips already-rendered units, or any per-unit split of a
            # render -- keep DIFFERENT rows than the full run, and the
            # rendered bytes follow the rows. Draws are CPU floats; the cost
            # is noise next to the D2H copy below.
            last_dim = int(x.shape[-1]) if x.dim() >= 1 else 0
            n_rows = (x.numel() // last_dim) if last_dim > 0 else 0
            new_priorities = (
                torch.rand(
                    int(n_rows),
                    generator=self._activation_generator,
                    dtype=torch.float32,
                    device="cpu",
                )
                if n_rows > 0 and self.max_rows > 0
                else None
            )
            # Only store the full activation tensor if this Linear is in
            # the store set.  Memory bound: store_qnames × max_rows × in.
            if key not in self.store_qnames:
                return
            # NOT non_blocking: async D2H into pageable memory can read
            # the tensor before the producing kernel finishes under GPU
            # contention, deterministically corrupting the snapshot to NaN
            # (2026-07-11, inline-export agent repro).
            flat = x.detach().reshape(-1, x.shape[-1]).to(
                device=self.store_device,
                dtype=self.store_dtype,
            )
            current = (
                torch.cat(self.activations[key], dim=0)
                if self.activations[key]
                else None
            )
            sampled, priorities = update_priority_reservoir(
                current,
                self._activation_priorities.get(key),
                flat,
                max_rows=self.max_rows,
                new_priorities=new_priorities,
            )
            self.activations[key] = [] if sampled is None else [sampled]
            if priorities is None:
                self._activation_priorities.pop(key, None)
            else:
                self._activation_priorities[key] = priorities
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def collected(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, parts in self.activations.items():
            if not parts:
                continue
            out[key] = torch.cat(parts, dim=0)
        self.max_abs = {
            key: float(value.detach().to("cpu").item())
            for key, value in self._max_abs_tensors.items()
        }
        return out


@contextmanager
def _temporarily_install_act_aware(
    activations: Mapping[str, torch.Tensor],
    levers: Mapping[str, object],
):
    """Install module-level state expected by ``_quantize_2d``.

    The export module reads ``_CACHED_ACTIVATIONS`` and ``_ACT_AWARE_FLAGS``
    from its own globals to decide what passes to run.  We mutate these
    inside a try/finally so concurrent export work isn't disturbed.
    """
    from prismaquant import export_native_compressed as enc

    prev_cache = enc._CACHED_ACTIVATIONS
    prev_flags = dict(enc._ACT_AWARE_FLAGS)
    prev_scale_rule = enc._NVFP4_SCALE_RULE
    enc._CACHED_ACTIVATIONS = _DictActivations(activations)
    enc._ACT_AWARE_FLAGS = {
        "gptq": bool(levers.get("gptq", True)),
        "scale_sweep": bool(levers.get("scale_sweep", False)),
        "static_act_order": bool(levers.get("static_act_order", False)),
        "joint_scale_opt": bool(levers.get("joint_scale_opt", False)),
    }
    enc._NVFP4_SCALE_RULE = enc.resolve_nvfp4_scale_rule(
        str(levers.get("nvfp4_scale_rule", "static_6"))
    )
    try:
        yield
    finally:
        enc._CACHED_ACTIVATIONS = prev_cache
        enc._ACT_AWARE_FLAGS.clear()
        enc._ACT_AWARE_FLAGS.update(prev_flags)
        enc._NVFP4_SCALE_RULE = prev_scale_rule


class _DictActivations:
    """`.get(name)` shim matching `_LazyActivationCache`'s interface."""

    def __init__(self, mapping: Mapping[str, torch.Tensor]):
        self._mapping = mapping

    def get(self, name: str) -> torch.Tensor | None:
        a = self._mapping.get(name)
        if a is None and name.endswith(".weight"):
            a = self._mapping.get(name[:-7])
        return a


class _FisherRowWeightCache:
    """Lazy loader for h-detail `g2_per_token` vectors."""

    _FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")

    def __init__(
        self,
        h_detail_dir: str | Path | None,
        fused_sibling_mapping: Mapping[str, Sequence[str]] | None = None,
    ):
        self.detail_dir = Path(h_detail_dir) if h_detail_dir else None
        self.fused_sibling_mapping = {
            str(fused): tuple(str(member) for member in members)
            for fused, members in (fused_sibling_mapping or {}).items()
        }
        self._cache: dict[str, torch.Tensor | None] = {}
        self.loads = 0
        self.misses = 0

    def _path_for_name(self, name: str) -> Path | None:
        if self.detail_dir is None:
            return None
        return self.detail_dir / (self._FNAME_SUB.sub("__", name) + ".pt")

    def _load_exact(self, name: str) -> torch.Tensor | None:
        if self.detail_dir is None:
            return None
        path = self._path_for_name(name)
        if path is None:
            return None
        if not path.is_file():
            return None
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            weights = blob.get("g2_per_token") if isinstance(blob, dict) else None
            if not isinstance(weights, torch.Tensor) or weights.numel() == 0:
                weights = None
            else:
                weights = weights.detach().to(torch.float32).cpu()
        except Exception:
            weights = None
        return weights

    def _split_fused_names(self, qname: str) -> tuple[str, ...]:
        if "." not in qname:
            return ()
        prefix, leaf = qname.rsplit(".", 1)
        members = self.fused_sibling_mapping.get(leaf)
        if not members:
            return ()
        return tuple(f"{prefix}.{member}" for member in members)

    @staticmethod
    def _combine_split_weights(parts: Sequence[torch.Tensor]) -> torch.Tensor | None:
        tensors = [
            p.detach().reshape(-1).to(torch.float32).cpu()
            for p in parts
            if isinstance(p, torch.Tensor) and p.numel() > 0
        ]
        if not tensors:
            return None
        n = min(int(t.numel()) for t in tensors)
        if n <= 0:
            return None
        stacked = torch.stack([t[:n] for t in tensors], dim=0)
        return stacked.mean(dim=0)

    def get(self, qname: str) -> torch.Tensor | None:
        if self.detail_dir is None:
            return None
        if qname in self._cache:
            return self._cache[qname]

        weights = self._load_exact(qname)
        if weights is None:
            split = self._split_fused_names(qname)
            if split:
                parts = [
                    part for name in split
                    if (part := self._load_exact(name)) is not None
                ]
                weights = self._combine_split_weights(parts)

        if weights is None:
            self.misses += 1
        else:
            self.loads += 1
        self._cache[qname] = weights
        return weights


def _fused_sibling_leaf_mapping_from_profile(profile) -> dict[str, tuple[str, ...]]:
    if profile is None:
        return {}
    getter = getattr(profile, "fused_sibling_leaf_mapping", None)
    if not callable(getter):
        return {}
    try:
        mapping = getter()
    except Exception:
        return {}
    return {
        str(fused): tuple(str(member) for member in members)
        for fused, members in (mapping or {}).items()
    }


def _damp_sweep_enabled() -> bool:
    # lazy import: production_weight_cache <-> export_native_compressed
    # are mutually lazy to avoid import cycles
    from prismaquant.export_native_compressed import gptq_damp_sweep_enabled
    return gptq_damp_sweep_enabled()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = int(default)
    return max(lo, min(hi, value))


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except Exception:
        value = float(default)
    return max(lo, min(hi, value))


@contextmanager
def _temporary_nvfp4_scale_rule(rule: str):
    from prismaquant import export_native_compressed as enc

    previous = enc._NVFP4_SCALE_RULE
    enc._NVFP4_SCALE_RULE = enc.resolve_nvfp4_scale_rule(rule)
    try:
        yield
    finally:
        enc._NVFP4_SCALE_RULE = previous

def _store_rendered_weight_entry(
    *,
    weights: dict[tuple[str, str], object],
    cache_dir_path: Path | None,
    qname: str,
    fmt: str,
    tensor: torch.Tensor,
    weight_dtype: torch.dtype,
    durable: bool = False,
) -> None:
    from prismaquant import format_registry as fr

    fmt = fr.canonical_format_name(str(fmt).strip().upper())
    stored = _canonical_rendered_weight_tensor(
        tensor,
        weight_dtype=weight_dtype,
    )
    if cache_dir_path is not None:
        fname = _cache_weight_filename(qname, fmt)
        final_path = cache_dir_path / fname
        # Per-process staging: a fan-out that re-renders a cell another writer
        # is publishing must not share its inode. The suffix adds exactly one
        # dot so torch's archive name -- and the published bytes -- are the
        # same as a direct save (``unique_temp_suffix``).
        tmp_path = cache_dir_path / (fname + unique_temp_suffix())
        torch.save(stored, tmp_path)
        if durable:
            fd = os.open(tmp_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        os.replace(tmp_path, final_path)
        if durable:
            directory_fd = os.open(cache_dir_path, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        weights[(qname, fmt)] = fname
        del stored
    else:
        weights[(qname, fmt)] = stored


def _canonical_rendered_weight_tensor(
    tensor: torch.Tensor,
    *,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    """Return the exact tensor representation a cache shard would contain.

    Renderers may compute in FP32 even when the source model is BF16.  Any
    transient consumer must score this canonical representation, rather than
    the wider temporary, so its scalar describes the bytes later installed by
    validation/export.
    """
    target_dtype = (
        weight_dtype if weight_dtype != torch.float32 else torch.bfloat16
    )
    return tensor.detach().to(
        dtype=target_dtype,
        device="cpu",
    ).contiguous()

@dataclass
class _RenderedCandidate:
    label: str
    weight: torch.Tensor
    score: float
    metric: str
    scale_rule: str
    package: tuple[str, ...]
    has_gptq: bool


def _render_score_for_gate(
    reference_weight: torch.Tensor,
    rendered_weight: torch.Tensor,
    activations: torch.Tensor | None,
) -> tuple[float, str]:
    """Score a local render candidate with the shared scorer.

    Activations should normally be present in production cache renders.  The
    weight-MSE fallback keeps pure RTN/FourOverSix unit tests and non-act-aware
    formats measurable without adding a second scoring abstraction.

    Progressive-gate callers must pass the SAME clipped activation matrix the
    GPTQ loop optimized under (``_gate_activation_matrix``), not the raw
    capture: mixing clipped optimization with unclipped gates biases
    accept-vs-RTN decisions near ties by the outlier rows — the exact
    mismatch export's shared-matrix contract
    (``_activation_matrix_for_gptq``) exists to prevent.
    """
    if (
        activations is not None
        and activations.numel() > 0
        and int(activations.shape[-1]) == int(reference_weight.shape[1])
    ):
        return (
            score_render_error(
                reference_weight,
                rendered_weight,
                activations,
                row_weights=None,
            ),
            "output_mse",
        )
    diff = (
        reference_weight.detach().to(torch.float32)
        - rendered_weight.detach().to(
            device=reference_weight.device,
            dtype=torch.float32,
        )
    )
    return float(diff.pow(2).mean().item()), "weight_mse"


def _gate_activation_matrix(
    acts_for_render: torch.Tensor | None,
    cols: int,
    *,
    device: torch.device,
    act_clip_threshold: float | None,
    act_clip_rescale: str | None,
    fisher_row_weights: torch.Tensor | None,
) -> torch.Tensor | None:
    """Return the clip-consistent activation matrix for progressive gates.

    Gates must score candidates under the objective GPTQ optimized: export's
    ``_activation_matrix_for_gptq`` applies the quantile/threshold clip and
    the Fisher row weighting before the Hessian build ("intentionally shared
    by the Hessian build, damping sweep evaluator, and do-no-harm gate"), so
    the cache-fill gate reuses it with identical settings instead of scoring
    on the raw capture.
    """
    if acts_for_render is None:
        return None
    from prismaquant import export_native_compressed as enc

    return enc._activation_matrix_for_gptq(
        acts_for_render,
        int(cols),
        device=device,
        clip_threshold=act_clip_threshold,
        clip_rescale=act_clip_rescale,
        row_weights=fisher_row_weights,
    )


def _render_score_record_key(qname: str, fmt: str) -> str:
    return f"{qname}|{fmt.upper()}"


def _render_score_normalizer(
    reference_weight: torch.Tensor,
    activations: torch.Tensor | None,
    metric: str,
) -> tuple[float, int]:
    rows, cols = reference_weight.shape
    if (
        metric in {"output_mse", "fisher_output_mse"}
        and activations is not None
        and activations.numel() > 0
        and int(activations.shape[-1]) == int(cols)
    ):
        n_act_rows = int(activations.reshape(-1, cols).shape[0])
        return float(max(1, n_act_rows) * int(rows)), n_act_rows
    return float(max(1, int(rows) * int(cols))), 0


def _render_score_record(
    *,
    qname: str,
    fmt: str,
    render_format: str,
    reference_weight: torch.Tensor,
    rendered_weight: torch.Tensor,
    activations: torch.Tensor | None,
    activation_max_abs: float | None,
    input_global_scale_policy: str | None = None,
) -> dict[str, object]:
    raw_score, raw_metric = _render_score_for_gate(
        reference_weight.detach().to(torch.float32),
        rendered_weight,
        activations,
    )
    score = raw_score
    metric = raw_metric
    activation_quantized = False
    activation_clipped = False
    from prismaquant import format_registry as fr

    spec = fr.get_format(fr.canonical_format_name(fmt))
    contract = _static_activation_contract_of(spec)
    # Which activation quantizer the row is scored under is the SPEC's answer
    # (``static_activation_contract``), not its name's (#205):
    #
    #   * served-measurement contract (a Tessera W4A4 rung): the served oracle
    #     at the unit's calibrated G, no pre-clip (the clamp lives in G), and
    #     the record keeps the maximum and the G it was priced at.  A unit
    #     with no maximum refuses by name -- there is no dynamic serving path
    #     whose score would mean anything.
    #   * screen-default contract (stock NVFP4): the historical clip to the
    #     calibrated maximum + the row's dynamic RTN.
    #   * no contract (FP8/MX dynamic W8A8): the row's dynamic quantizer,
    #     unclipped -- applying NVFP4's maximum there prices the wrong kernel.
    #
    # Either contract retains the calibration maximum in the record: it is the
    # unit's scale identity, and a reader must be able to see which contract
    # the row was priced under.
    served_measurement = contract is not None and contract.measured_as_served
    recorded_max_abs = activation_max_abs if contract is not None else None
    activation_clip_max = None if served_measurement else recorded_max_abs
    # Which policy this row is priced under: the one the caller resolved for
    # the whole fill (its render levers), else the live one.  Resolved ONCE
    # here so the G that scores and the policy the record stamps are the same
    # answer (#227).
    policy = _resolve_render_input_global_scale_policy(input_global_scale_policy)
    input_global_scale: float | None = None
    if (
        activations is not None
        and activations.numel() > 0
        and int(activations.shape[-1]) == int(reference_weight.shape[1])
    ):
        if served_measurement:
            input_global_scale = contract.require_input_global_scale(
                activation_max_abs, qname=qname,
                consumer=f"production cache render score @ {fmt}",
                policy=policy)
            g = float(input_global_scale)
            activation_quantize = lambda t: contract.quantize_dequantize(t, g)  # noqa: E731
        else:
            activation_quantize = spec.activation_quantize_dequantize
        try:
            score, metric, activation_quantized, activation_clipped = (
                _local_forward_render_score(
                    reference_weight=reference_weight,
                    rendered_weight=rendered_weight,
                    activations=activations,
                    activation_quantize=activation_quantize,
                    activation_max_abs=activation_clip_max,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"activation-aware render scoring failed for {qname} @ {fmt}: "
                f"{exc}"
            ) from exc
    normalizer, activation_rows = _render_score_normalizer(
        reference_weight,
        activations,
        metric,
    )
    # weight_mse is the original prismaquant cost surrogate: pure
    # (W_orig - W_rendered)^2 averaged over weights. Activation-independent,
    # low variance; the allocator multiplies by h_trace for predicted_dloss.
    ref_f = reference_weight.detach().to(
        device=rendered_weight.device, dtype=torch.float32,
    )
    rendered_f = rendered_weight.detach().to(torch.float32)
    diff = ref_f - rendered_f
    n_weights = int(diff.numel())
    weight_mse = float(diff.pow(2).mean().item()) if n_weights > 0 else 0.0
    rows, cols = reference_weight.shape
    return {
        "qname": str(qname),
        "format": str(fmt).upper(),
        "render_format": str(render_format).upper(),
        "metric": str(metric),
        "score": float(score),
        "score_sum": float(score) * float(normalizer),
        "raw_render_metric": str(raw_metric),
        "raw_render_score": float(raw_score),
        "raw_render_score_sum": float(raw_score) * float(normalizer),
        "weight_mse": float(weight_mse),
        "weight_mse_sum": float(weight_mse) * float(n_weights),
        "n_weights": int(n_weights),
        "normalizer": float(normalizer),
        "activation_rows": int(activation_rows),
        "activation_quantized": bool(activation_quantized),
        "activation_clipped": bool(activation_clipped),
        "activation_max_abs": (
            float(recorded_max_abs)
            if recorded_max_abs is not None and recorded_max_abs > 0
            else None
        ),
        # The static G the row was priced at, when its contract is the served
        # one; None for rows scored under a dynamic quantizer.
        "input_global_scale": (
            float(input_global_scale) if input_global_scale is not None else None
        ),
        # ... and the policy that G came out of, so a later resume or KL hook
        # can say WHY a retained cost no longer matches (#227).  Stamped only
        # where it priced something: a dynamically scored row does not depend
        # on it, and claiming otherwise would invalidate FP8/MX rows for a
        # setting that never touched them.
        "input_global_scale_policy": (
            str(policy) if input_global_scale is not None else None
        ),
        # ... and WHICH activation quantiser arithmetic priced it, with the build
        # and image that arithmetic ran in.  A static-G row is scored through
        # ``contract.quantize_dequantize``, which runs the registered served
        # operator or this tree's Torch model depending on the process binding;
        # a retained cost is only a cost of the arithmetic that produced it, so
        # the identity travels with the record (RobTand/prismaquant#567).  None
        # where no static G was priced: a dynamically scored row never touched
        # this quantiser, and invalidating it would be the opposite error.
        "served_quantizer": (
            None if input_global_scale is None
            else _scored_served_quantizer_record(contract)
        ),
        "out_features": int(rows),
        "in_features": int(cols),
    }


def _scored_served_quantizer_record(contract) -> "dict | None":
    """The identity of the arithmetic THIS row was priced with.

    Read through the one effective-identity accessor, so the arithmetic the row
    ran (``contract.quantize_dequantize``) and the arithmetic it is stamped with
    cannot be two answers: a contract carrying its own explicit binding stamps
    that, not the process's.  Never resolved, probed or imported here -- the
    cache records what priced the row, and an unstamped row is refused at reuse
    rather than filled in later.
    """
    from prismaquant.nvfp4_activation_contract import (
        effective_served_quantizer_identity,
    )

    identity = effective_served_quantizer_identity(contract)
    return None if identity is None else identity.as_record()


def _resolve_format_spec(fmt):
    """The ``FormatSpec`` a requested format name resolves to, or ``None``.

    ONE home for "does this requested name resolve to a spec, and if so which"
    (principle 8).  Every predicate in this module that classifies a name off
    the format menu goes through here, so they cannot answer that question
    differently again.  Two properties, both load-bearing:

    * It does not upper-case.  ``fr.canonical_format_name`` settles case
      itself; an extra ``.upper()`` here is redundant at best, and at worst it
      destroys a mixed-case registered name before any resolver sees it
      (#218).  These predicates are handed ``_render_base_format`` output,
      which has already normalized -- so they resolve the name they are given.

    * It answers ``None`` rather than raising.  These are predicates over a
      requested format MENU, and a menu can carry a name this registry does
      not own; "not a format we know" is an answer, not an error.  A caller
      that wants to refuse an unknown format raises its own named refusal
      where the menu is validated -- not a bare ``KeyError`` escaping from a
      question about activation scales, mid-fill, before a tensor is
      rendered.
    """
    from prismaquant import format_registry as fr

    try:
        return fr.get_format(fr.canonical_format_name(str(fmt).strip()))
    except Exception:
        return None


def _static_activation_contract_of(spec):
    """``spec.static_activation_contract``, tolerant of the bare stand-ins
    tests hand ``fr.get_format`` back (a namespace with only the callable)."""
    return getattr(spec, "static_activation_contract", None)


def _format_uses_static_activation_clip(fmt: str) -> bool:
    """Return whether local scoring should apply a calibrated activation max.

    True for a format whose serving contract is a calibrated STATIC per-unit
    activation scale (``FormatSpec.static_activation_contract``, read from the
    spec -- a Tessera W4A4 rung answers yes with no "NVFP4" in its name).
    MXFP8/FP8 dynamic serving computes activation scales at runtime, so
    applying a calibrated maximum to those formats prices the wrong kernel
    contract.  Whether the maximum is applied as a pre-clip or as the G of the
    served oracle is the contract's ``measured_as_served``; this predicate only
    says the maximum belongs to the row.

    Total over the menu: a name this registry does not resolve is not served
    under a static activation contract, so it answers False (#218).
    """
    return _static_activation_contract_of(_resolve_format_spec(fmt)) is not None


def _formats_need_static_activation_max(formats) -> bool:
    """Whether a cache fill over ``formats`` must measure per-unit max|x|.

    The fill collects the calibrated activation maximum only when some
    requested format is served under a static per-unit activation scale; it
    used to ask ``"NVFP4" in formats``, which a Tessera-only fill answered
    "no" -- so its records could never retain a maximum and its assignment-KL
    hooks never had a scale identity to price with (#205).
    """
    return any(_format_uses_static_activation_clip(fmt) for fmt in formats)


def _local_forward_render_score(
    *,
    reference_weight: torch.Tensor,
    rendered_weight: torch.Tensor,
    activations: torch.Tensor,
    activation_quantize,
    activation_max_abs: float | None,
    row_chunk: int = 128,
) -> tuple[float, str, bool, bool]:
    rows, cols = reference_weight.shape
    if rendered_weight.shape != reference_weight.shape:
        return float("inf"), "output_mse", False, False
    if activations.shape[-1] != cols:
        return float("inf"), "output_mse", False, False
    device = reference_weight.device
    ref_t = reference_weight.detach().to(device=device, dtype=torch.float32).t()
    rendered_t = rendered_weight.detach().to(device=device, dtype=torch.float32).t()
    x = activations.detach().to(device=device, dtype=torch.float32).reshape(-1, cols)
    clipped = False
    if (
        activation_max_abs is not None
        and float(activation_max_abs) > 0
        and _env_flag("PRISMAQUANT_PROD_ACT_SCALES", True)
    ):
        x_quant_input = x.clamp(-float(activation_max_abs), float(activation_max_abs))
        clipped = True
    else:
        x_quant_input = x

    total = torch.zeros((), dtype=torch.float32, device=device)
    quantized_any = False
    with torch.no_grad():
        for start in range(0, x.shape[0], int(row_chunk)):
            x_ref = x[start:start + int(row_chunk)]
            x_q = activation_quantize(x_quant_input[start:start + int(row_chunk)])
            if x_q is not x_ref:
                quantized_any = quantized_any or not torch.equal(
                    x_q.detach().to(device=device, dtype=torch.float32),
                    x_ref,
                )
            x_q = x_q.to(device=device, dtype=torch.float32)
            y_ref = x_ref @ ref_t
            y_q = x_q @ rendered_t
            err = (y_ref - y_q).pow(2)
            total = total + err.sum()
    score = float(total.item()) / max(1, int(x.shape[0]) * int(rows))
    return score, "output_mse", bool(quantized_any), bool(clipped)


def _load_render_score_sidecar(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.is_file():
        return {}

    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"failed to load render-score sidecar {path}: {exc}; "
            "refusing resume"
        ) from exc
    records = raw.get("records") if isinstance(raw, Mapping) else None
    if not isinstance(records, Mapping):
        raise RuntimeError(
            f"render-score sidecar {path} does not contain a records object; "
            "refusing resume"
        )
    out: dict[str, dict[str, object]] = {}
    for key, value in records.items():
        if not isinstance(value, Mapping):
            raise RuntimeError(
                f"render-score sidecar {path} has a non-object record for "
                f"{key!r}; refusing resume"
            )
        out[str(key)] = dict(value)
    return out


def _write_render_score_sidecar(
    path: Path | None,
    records: Mapping[str, Mapping[str, object]],
) -> None:
    if path is None:
        return
    import json as _json

    payload = {
        "schema": "prismaquant.production_render_scores.v1",
        "records": dict(sorted((str(k), dict(v)) for k, v in records.items())),
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(_json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _render_score_record_priced_scale(
    record: Mapping[str, object],
    *,
    key: str,
    where: str,
) -> tuple[str, float, float, str | None, object] | None:
    """``(qname, priced_G, max_abs, priced_policy)`` for a served-contract row.

    ``None`` for a row that carries no static G -- one scored under a dynamic
    quantizer (FP8/MX), or scored without activations.  Nothing about those
    rows depends on the input-global-scale policy, so nothing about them is
    invalidated by a change to it.

    A row that carries a G but no maximum to have derived it from is neither:
    it is an inconsistent record, and it fails closed here rather than being
    admitted as "nothing to check".
    """
    priced = record.get("input_global_scale")
    if priced is None:
        return None
    qname = str(record.get("qname") or key)
    max_abs = record.get("activation_max_abs")
    try:
        priced_value = float(priced)  # type: ignore[arg-type]
        max_abs_value = float(max_abs)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise RuntimeError(
            f"{where}: render score {key!r} records "
            f"input_global_scale={priced!r} with activation_max_abs="
            f"{max_abs!r}; refusing to reuse a cost whose static scale cannot "
            "be checked against the current policy"
        ) from None
    if (
        not math.isfinite(priced_value) or priced_value <= 0.0
        or not math.isfinite(max_abs_value) or max_abs_value <= 0.0
    ):
        raise RuntimeError(
            f"{where}: render score {key!r} records a non-positive static "
            f"activation scale (input_global_scale={priced!r}, "
            f"activation_max_abs={max_abs!r}); refusing to reuse it"
        )
    policy = record.get("input_global_scale_policy")
    return (
        qname,
        priced_value,
        max_abs_value,
        None if policy is None else str(policy),
        record.get("served_quantizer"),
    )


def _render_score_read_contract(record: Mapping[str, object]):
    """The contract a retained render score was priced through, or ``None``.

    A row's arithmetic is the CONTRACT's -- an explicit binding on the contract
    beats the process's -- so a reader that compared every retained cost against
    the process binding alone would refuse a row that was correctly stamped for
    a contract-bound build.  The record names its format, and the registry is
    the one owner of which contract that format carries, so the read resolves
    the same object the write priced through rather than a second expectation.

    ``None`` (unknown format, no contract, or a registry that cannot answer) is
    not an error here: the caller falls back to the process binding, which is
    what a row with no contract-bound arithmetic was priced under.
    """
    fmt = record.get("format")
    if not isinstance(fmt, str) or not fmt:
        # The one narrowed legacy case, and it is a property of the record's
        # shape rather than of a failed lookup: ``_render_score_record`` has
        # always stamped ``format``, so a static-G record without one was not
        # written by this writer.  It carries no format to resolve, so the
        # caller's process binding is the only expectation available.
        return None
    try:
        from prismaquant import format_registry as fr

        spec = fr.get_format(fr.canonical_format_name(fmt))
    except Exception as exc:
        # FAIL CLOSED.  A format this tree cannot resolve -- a missing package,
        # a corrupt registry row, a name whose render lane is gone -- is not
        # evidence that the row was priced under the process's arithmetic.
        # Reinterpreting a static-G cost under the global binding is exactly how
        # a model-priced cost would be admitted as registered-operator pricing.
        raise RuntimeError(
            f"render score record names format {fmt!r}, which this tree cannot "
            f"resolve ({type(exc).__name__}: {exc}); a static-G cost cannot be "
            "checked against the arithmetic that priced it, so it is refused "
            "rather than reinterpreted under another binding"
        ) from exc
    try:
        return _static_activation_contract_of(spec)
    except Exception as exc:
        raise RuntimeError(
            f"render score record names format {fmt!r} whose activation "
            f"contract cannot be read ({type(exc).__name__}: {exc}); refusing "
            "to reinterpret a static-G cost under another binding"
        ) from exc


def _check_resumed_render_score_policies(
    records: Mapping[str, Mapping[str, object]],
    *,
    policy: str,
    where: str,
) -> int:
    """Refuse retained activation-aware costs priced under another policy.

    The resume loop admits a shard as rendered by file presence and keeps its
    existing score by ``qname|FMT`` key, so without this a cache filled under
    ``PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE=0`` (G = 6/amax) silently
    contributes its old costs to a run that quantizes activations at
    ``448*6/amax`` -- the same weights, the same maximum, a different served
    A-side, and therefore a different cost (#227).

    The check is the recorded G against the G this policy derives from the
    SAME recorded maximum, so it holds for records written before the policy
    stamp existed as well as after; the stamp only lets the refusal name the
    two policies.  Returns how many rows carried a static G.
    """
    from prismaquant.nvfp4_activation_contract import (
        input_global_scale_from_max_abs,
        require_matching_input_global_scale,
        require_matching_served_quantizer,
    )

    checked = 0
    for key in sorted(records):
        record = records[key]
        if not isinstance(record, Mapping):
            continue
        priced = _render_score_record_priced_scale(
            record, key=key, where=where)
        if priced is None:
            continue
        qname, priced_value, max_abs_value, priced_policy, priced_quantizer = priced
        checked += 1
        # The second axis of the same question: the cost must have been priced
        # by the arithmetic this run is bound to.  Weight-only and dynamically
        # scored rows never reach here (``priced is None`` above), which is what
        # keeps their caches reusable.
        require_matching_served_quantizer(
            priced_quantizer, qname=qname, consumer=where,
            contract=_render_score_read_contract(record))
        require_matching_input_global_scale(
            priced_value,
            input_global_scale_from_max_abs(max_abs_value, policy=policy),
            qname=qname,
            consumer=where,
            priced_policy=priced_policy,
            applied_policy=policy,
        )
    return checked


def production_cache_priced_input_global_scales(
    cache, *, where: str = "ProductionWeightCache",
) -> dict[str, float]:
    """The static G each unit's retained render score was priced at.

    The measurement side of #227: an assignment-KL hook derives its own G from
    the cache's calibrated maximum and the CURRENT policy, and must compare it
    against the G the cost it is measuring against was priced at.  Read from
    the cache's own ``render_scores`` provenance, never from the environment.

    Rows without a static G (dynamic quantizers) contribute nothing.  Two rows
    for one unit disagreeing is the very defect this guards -- G is a function
    of the unit's maximum and one policy -- so it refuses by name rather than
    picking one.
    """
    metadata = cache.metadata if isinstance(
        getattr(cache, "metadata", None), Mapping) else {}
    section = metadata.get("render_scores")
    records = section.get("records") if isinstance(section, Mapping) else None
    if not isinstance(records, Mapping):
        return {}
    priced_scales: dict[str, float] = {}
    from prismaquant.nvfp4_activation_contract import (
        require_matching_served_quantizer,
    )
    for key in sorted(records):
        record = records[key]
        if not isinstance(record, Mapping):
            continue
        priced = _render_score_record_priced_scale(
            record, key=str(key), where=where)
        if priced is None:
            continue
        qname, priced_value, _max_abs, _policy, priced_quantizer = priced
        # The measurement side of the same refusal: a KL hook about to measure
        # against these costs reads them through here, so the arithmetic check
        # belongs on this path too, not only on resume.
        require_matching_served_quantizer(
            priced_quantizer, qname=qname, consumer=where,
            contract=_render_score_read_contract(record))
        previous = priced_scales.get(qname)
        if previous is not None and previous != priced_value:
            raise RuntimeError(
                f"{where}: {qname!r} has render scores priced at two static "
                f"activation scales ({previous!r} and {priced_value!r}); the "
                "cache mixes input-global-scale policies and cannot be "
                "measured against"
            )
        priced_scales[qname] = priced_value
    return priced_scales


def _load_activation_max_abs_sidecar(path: Path) -> dict[str, float]:
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"failed to load activation max-abs sidecar {path}: {exc}; "
            "refusing resume"
        ) from exc
    if not isinstance(raw, Mapping):
        raise RuntimeError(
            f"activation max-abs sidecar {path} is not a JSON object; "
            "refusing resume"
        )

    values: dict[str, float] = {}
    for qname, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(
                f"activation max-abs sidecar {path} has a non-numeric value "
                f"for {qname!r}; refusing resume"
            )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise RuntimeError(
                f"activation max-abs sidecar {path} has an invalid value "
                f"for {qname!r}: {value!r}; refusing resume"
            )
        values[str(qname)] = numeric
    return values


def _format_supports_render_mechanism(fmt: str, mechanism: str) -> bool:
    """Return whether a shared render mechanism is meaningful for ``fmt``.

    The production render pipeline is format-agnostic in order. Individual
    formats opt out of mechanisms whose math or exported schema does not
    apply.
    """

    fmt_u = str(fmt).strip().upper()
    mech = str(mechanism).strip()
    if fmt_u == "NVFP4":
        return mech in {
            "four_over_six",
            "gptq",
            "static_act_order",
            "joint_scale_opt",
            "fisher_gptq",
            "scale_sweep",
        }
    if fmt_u in {"FP8_E4M3", "FP8_E5M2"}:
        return mech == "gptq" or (mech == "scale_sweep" and fmt_u == "FP8_E4M3")
    if fmt_u == "MXFP4":
        return mech in {"gptq", "static_act_order"}
    if fmt_u in {"MXFP8_E4M3", "MXFP8_E5M2"}:
        return mech in {"gptq", "static_act_order"} or (
            mech == "scale_sweep" and fmt_u == "MXFP8_E4M3"
        )
    if _weighted_render_family(fmt_u) is not None:
        # k-quant superblocks pick scales per block, so the scalar-column
        # mechanisms (gptq/jso/scale_sweep/act_order) do not apply. The
        # imatrix-weighted search IS the deliberate render (col_weights) —
        # re-vet R3 made it reachable from this path.
        return mech == "weighted_vq"
    return False


# Format families whose EXPORTER renders imatrix-weighted, so a production
# render of the same (qname, fmt) is only exporter-faithful when it applies the
# same per-input-column vector. Keyed off the family exactly as
# `measure_quant_cost._cost_render_uses_imatrix` keys the inline cost render —
# one invariant, two call sites, no third opinion.
WEIGHTED_RENDER_FAMILIES = ("gguf",)


def _expert_col_weights(
    stack_cw: torch.Tensor | None, index: int, n_experts: int,
) -> torch.Tensor | None:
    """Slice a packed-expert imatrix entry for expert ``index``.

    **Delegates** to ``measure_quant_cost._item_col_weights`` rather than
    re-deriving the rule: a ``(E, in)`` stack is indexed per expert, anything
    else is a single pooled vector shared by every expert. One definition, two
    readers — if the cost stage and the cache ever sliced the same pickle
    differently they would weight the same tensor differently, which is the
    rendering confound wearing a new hat.
    """
    if stack_cw is None:
        return None
    from prismaquant.measure_quant_cost import _item_col_weights

    return _item_col_weights(stack_cw, index, n_experts)


def _weighted_render_family(fmt: str) -> str | None:
    """The weighted-render family of ``fmt``, or ``None`` for every format
    whose render ignores ``col_weights`` (NVFP4/FP8/MX/BF16/INT)."""
    family = getattr(_resolve_format_spec(fmt), "family", None)
    return family if family in WEIGHTED_RENDER_FAMILIES else None


def _is_cb_format_name(fmt: str) -> bool:
    """Refuse a retired codebook rung name; every live format answers False.

    The codebook families were archived on 2026-09-25 (#1304), so no
    resolvable format is one. ``_resolve_format_spec`` swallows the
    registry's refusal, which would let a stale ``NVFP4_CB_K*``/``FP8_CB_K*``
    cache key read as "not a codebook format" and be reused; this raises
    ``fr.RetiredFormatError`` for it instead.

    Kept only because ``ProductionWeightCache`` still asks it (its
    ``assignment_keys``, ``_validate_loaded_cb_pair_tensor`` and ``get``);
    delete it with those call sites once #1318 lands (#1328).
    """
    from prismaquant.schemas import refuse_retired_codebook_format

    refuse_retired_codebook_format(str(fmt).strip())
    return False


#: One canonical-JSON value normalizer (PQ #1302): ``digests`` owns it.
_canonical_json_value = canonical_json


def _source_weight_value_identity(
    weight: torch.Tensor,
) -> tuple[list[int], str]:
    tensor = torch.as_tensor(weight).detach()
    shape = [int(dim) for dim in tensor.shape]
    digest = hashlib.sha256()
    max_chunk_elements = 4 * 1024 * 1024  # <=16 MiB after fp32 conversion

    def _chunks_c_order(value: torch.Tensor):
        if value.numel() <= max_chunk_elements or value.ndim == 0:
            yield value
            return
        trailing = math.prod(int(dim) for dim in value.shape[1:])
        if trailing <= max_chunk_elements:
            step = max(1, max_chunk_elements // max(trailing, 1))
            for start in range(0, int(value.shape[0]), step):
                yield value[start:start + step]
            return
        # A single leading slice is still too large. Recurse dimension by
        # dimension; concatenating these chunks is exactly C-order traversal.
        for index in range(int(value.shape[0])):
            yield from _chunks_c_order(value[index])

    for chunk in _chunks_c_order(tensor):
        cpu = chunk.to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(
            cpu.numpy().astype("<f4", copy=False).tobytes(order="C")
        )
        del cpu
    return shape, digest.hexdigest()


_IDENTITY_MISSING = object()


def first_identity_difference(
    stored: object,
    expected: object,
    *,
    path: str = "",
) -> tuple[str, object, object] | None:
    """Return the first canonical-identity difference, including its field.

    Resume gates use this rather than a single aggregate digest so an operator
    is told *which* value-bearing input changed.  Mapping keys are traversed in
    sorted order and sequences in index order, making the reported field stable.
    """
    if isinstance(stored, Mapping) and isinstance(expected, Mapping):
        keys = sorted(set(str(key) for key in stored) | set(
            str(key) for key in expected
        ))
        stored_by_name = {str(key): value for key, value in stored.items()}
        expected_by_name = {str(key): value for key, value in expected.items()}
        for key in keys:
            child = f"{path}.{key}" if path else key
            left = stored_by_name.get(key, _IDENTITY_MISSING)
            right = expected_by_name.get(key, _IDENTITY_MISSING)
            if left is _IDENTITY_MISSING or right is _IDENTITY_MISSING:
                return child, left, right
            different = first_identity_difference(left, right, path=child)
            if different is not None:
                return different
        return None
    if (
        isinstance(stored, Sequence)
        and not isinstance(stored, (str, bytes, bytearray))
        and isinstance(expected, Sequence)
        and not isinstance(expected, (str, bytes, bytearray))
    ):
        if len(stored) != len(expected):
            child = f"{path}.length" if path else "length"
            return child, len(stored), len(expected)
        for index, (left, right) in enumerate(zip(stored, expected)):
            child = f"{path}[{index}]" if path else f"[{index}]"
            different = first_identity_difference(left, right, path=child)
            if different is not None:
                return different
        return None
    if type(stored) is not type(expected) or stored != expected:
        return path or "value", stored, expected
    return None


def identity_value_for_error(value: object) -> str:
    if value is _IDENTITY_MISSING:
        return "<missing>"
    text = repr(value)
    return text if len(text) <= 240 else text[:237] + "..."


def _canonical_json_sha256(value: object, *, where: str) -> str:
    canonical = _canonical_json_value(value, where=where)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _production_cache_git_commit() -> str:
    """Resolve the exact source commit once for pair-shard identity.

    A missing commit is not a useful producer identity, so unlike reporting
    provenance this is deliberately fail-closed.  The optional override is for
    immutable/container source mounts whose checkout metadata is unavailable;
    it must still be a full hexadecimal object id.
    """
    repo_root = Path(__file__).resolve().parents[1]
    override = str(os.environ.get(
        "PRISMAQUANT_IDENTITY_GIT_COMMIT", ""
    )).strip().lower()
    if override:
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", override) is None:
            raise RuntimeError(
                "PRISMAQUANT_IDENTITY_GIT_COMMIT must be a full 40- or "
                "64-character hexadecimal commit id"
            )
        return override
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        raise RuntimeError(
            "production cache identity cannot resolve git commit"
        ) from exc
    commit = result.stdout.strip().lower()
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit) is None:
        raise RuntimeError(
            "production cache identity resolved an invalid git commit "
            f"{commit!r}"
        )
    return commit


def _production_cache_source_sha256(
    package_root: Path | None = None,
) -> str:
    """Hash every durable PrismaQuant package input used by the producer.

    A hand-maintained import list is not fail-closed: a newly introduced
    transitive renderer dependency could otherwise change bytes without
    changing a resumable pair's identity.  Hash the complete installed
    package tree instead, excluding only interpreter bytecode caches.  This
    also binds packaged lattice/codebook data and model-profile JSON.
    """
    root = (
        Path(package_root)
        if package_root is not None
        else Path(__file__).resolve().parent
    )
    if not root.is_dir():
        raise RuntimeError(
            f"production source identity cannot read package root {root}"
        )
    identity_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.relative_to(root).parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    if not identity_paths:
        raise RuntimeError(
            f"production source identity found no files under {root}"
        )
    digest = hashlib.sha256()
    for path in identity_paths:
        relative = path.relative_to(root).as_posix()
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"production source identity cannot read {relative}"
            ) from exc
        encoded_name = relative.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


#: Feed width for host-tensor digests. ``hashlib`` releases the GIL for
#: buffers this size, so one memoryview fed in wide slices hashes at the
#: same single-core rate as one contiguous feed, without the ``tobytes()``
#: copy that used to double resident bytes per tensored identity (PQ #725).
_TENSOR_DIGEST_CHUNK_BYTES = 8 << 20


def _cb_cache_tensor_identity(tensor: torch.Tensor) -> dict[str, object]:
    stored = tensor.detach().to(device="cpu").contiguous()
    # ``cast("B")`` flattens the dimensions without copying, so slices below
    # are byte windows, never copies. (A multi-dimensional memoryview's bare
    # ``len`` would be the first axis, not the byte count.) An empty tensor
    # has zeros in its shape, which ``cast`` refuses, so it takes the empty
    # feed directly -- the same sha256 of zero bytes ``tobytes()`` produced.
    nbytes = stored.nbytes
    digest = hashlib.sha256()
    if nbytes:
        view = memoryview(stored.view(torch.uint8).numpy()).cast("B")
        for offset in range(0, len(view), _TENSOR_DIGEST_CHUNK_BYTES):
            digest.update(view[offset:offset + _TENSOR_DIGEST_CHUNK_BYTES])
    return {
        "shape": [int(dim) for dim in stored.shape],
        "dtype": str(stored.dtype),
        "logical_bytes": nbytes,
        "content_sha256": digest.hexdigest(),
    }


def validate_production_cache_cb_render_identity(
    cache: ProductionWeightCache,
    *,
    expected_context=None,
    expected_qnames: Sequence[str] | None = None,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    require_for_formats: Sequence[str] = (),
    require_source_complete: bool = True,
    where: str = "ProductionWeightCache",
):
    """Refuse a cache that stores, or is asked for, a retired codebook rung.

    The codebook lane and its render identity were archived on 2026-09-25
    (#1304). A stale manifest that still keys a weight by an
    ``NVFP4_CB_K*``/``FP8_CB_K*`` name raises ``fr.RetiredFormatError`` here
    instead of being reused. A lone ``cb_render_identity`` metadata key with
    no codebook weight is inert provenance and is ignored. Returns ``None``.

    Kept only because ``ProductionWeightCache.validate_cb_render_identity``
    still calls it; delete both once #1318 lands (#1328). The keyword
    parameters are that method's call shape and are not read.
    """
    for fmt in require_for_formats:
        _is_cb_format_name(fmt)
    for _qname, fmt in (getattr(cache, "weights", {}) or {}):
        _is_cb_format_name(fmt)
    return None


def _render_nvfp4_progressive_candidate(
    *,
    qname: str,
    weight_scaled: torch.Tensor,
    activations_scaled: torch.Tensor | None,
    levers: Mapping[str, object],
    scale_rule: str,
    joint_global_real: torch.Tensor | None,
    act_clip_threshold: float | None,
    act_clip_rescale: str | None,
    fisher_row_weights: torch.Tensor | None,
    include_gptq: bool,
    include_scale_sweep: bool,
) -> torch.Tensor:
    from prismaquant import export_native_compressed as enc

    with _temporary_nvfp4_scale_rule(scale_rule):
        current = enc._rtn_dequant_nvfp4(
            weight_scaled,
            group_size=16,
            global_real_override=joint_global_real,
        )
        if activations_scaled is None or activations_scaled.numel() == 0:
            return current
        if include_gptq:
            if _damp_sweep_enabled():
                current = enc._gptq_obs_rounding_nvfp4_swept(
                    weight_scaled,
                    activations_scaled,
                    group_size=16,
                    global_real_override=joint_global_real,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    static_act_order=bool(
                        levers.get("static_act_order", False)
                    ),
                    joint_scale_opt=bool(
                        levers.get("joint_scale_opt", False)
                    ),
                )
            else:
                current = enc._gptq_obs_rounding_nvfp4(
                    weight_scaled,
                    activations_scaled,
                    group_size=16,
                    damp=enc._resolve_gptq_damp_for_role(qname),
                    global_real_override=joint_global_real,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    static_act_order=bool(
                        levers.get("static_act_order", False)
                    ),
                    joint_scale_opt=bool(
                        levers.get("joint_scale_opt", False)
                    ),
                )
        if include_scale_sweep:
            current = enc._scale_sweep_nvfp4(
                current,
                activations_scaled,
                group_size=16,
                global_real_override=joint_global_real,
                reference_weight=weight_scaled,
                clip_threshold=act_clip_threshold,
                clip_rescale=act_clip_rescale,
                fisher_row_weights=fisher_row_weights,
            )
        return current


def _render_nvfp4_progressively(
    weight: torch.Tensor,
    *,
    qname: str,
    activations: Mapping[str, torch.Tensor],
    levers: Mapping[str, object],
    joint_global_real: torch.Tensor | None,
    act_clip_threshold: float | None,
    act_clip_rescale: str | None,
    fisher_row_weights: torch.Tensor | None,
    gate_trace: list[dict[str, object]] | None,
) -> torch.Tensor:
    from prismaquant import export_native_compressed as enc

    requested_rule = enc.resolve_nvfp4_scale_rule(
        str(levers.get("nvfp4_scale_rule", "static_6"))
    )
    f6_enabled = requested_rule == enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE
    gptq_enabled = bool(levers.get("gptq", True))
    scale_sweep_enabled = bool(levers.get("scale_sweep", False))
    static_act_order_enabled = bool(
        gptq_enabled and levers.get("static_act_order", False)
    )
    joint_scale_opt_enabled = bool(
        gptq_enabled and levers.get("joint_scale_opt", False)
    )
    gptq_modifiers = tuple(
        name for name, enabled in (
            ("static_act_order", static_act_order_enabled),
            ("joint_scale_opt", joint_scale_opt_enabled),
        )
        if enabled
    )
    gptq_scale_rule = (
        enc.NVFP4_SCALE_RULE_JOINT_MSE
        if joint_scale_opt_enabled
        else None
    )
    min_gain = _env_float(
        "PRISMAQUANT_RENDER_GATE_MIN_GAIN",
        0.0,
        lo=-1.0,
        hi=1.0,
    )

    reference = weight.detach().to(device=weight.device, dtype=torch.float32)
    acts = activations.get(qname)
    acts_for_render = (
        acts.detach().to(device=weight.device, dtype=torch.float32)
        if acts is not None and int(acts.shape[-1]) == int(weight.shape[1])
        else None
    )
    # Score gate candidates on the same clipped/weighted matrix the GPTQ
    # loop optimizes under (audit 2026-07-02 §3.9): the loop passes
    # act_clip_threshold/act_clip_rescale/fisher_row_weights into
    # _activation_matrix_for_gptq, so the gate must too.
    acts_for_gate = _gate_activation_matrix(
        acts_for_render,
        int(weight.shape[1]),
        device=weight.device,
        act_clip_threshold=act_clip_threshold,
        act_clip_rescale=act_clip_rescale,
        fisher_row_weights=fisher_row_weights,
    )
    reference_for_render = reference

    def candidate(
        *,
        label: str,
        scale_rule: str,
        package: tuple[str, ...],
        include_gptq: bool,
        include_scale_sweep: bool,
    ) -> _RenderedCandidate:
        rendered_scaled = _render_nvfp4_progressive_candidate(
            qname=qname,
            weight_scaled=reference_for_render,
            activations_scaled=acts_for_render,
            levers=levers,
            scale_rule=scale_rule,
            joint_global_real=joint_global_real,
            act_clip_threshold=act_clip_threshold,
            act_clip_rescale=act_clip_rescale,
            fisher_row_weights=fisher_row_weights,
            include_gptq=include_gptq,
            include_scale_sweep=include_scale_sweep,
        )
        rendered = rendered_scaled
        score, metric = _render_score_for_gate(
            reference,
            rendered,
            acts_for_gate,
        )
        return _RenderedCandidate(
            label=label,
            weight=rendered,
            score=float(score),
            metric=metric,
            scale_rule=scale_rule,
            package=package,
            has_gptq=bool(include_gptq),
        )

    static_rule = enc.NVFP4_SCALE_RULE_STATIC_6
    current = candidate(
        label="rtn_static_6",
        scale_rule=static_rule,
        package=(),
        include_gptq=False,
        include_scale_sweep=False,
    )
    if gate_trace is not None:
        gate_trace.append({
            "mechanism": "baseline",
            "selected": current.label,
            "score": float(current.score),
            "metric": current.metric,
            "scale_rule": current.scale_rule,
            "package": list(current.package),
        })

    def apply_gate(
        *,
        mechanism: str,
        candidates: Sequence[_RenderedCandidate],
    ) -> None:
        nonlocal current
        if not candidates:
            return
        best = min(candidates, key=lambda item: item.score)
        decision = gate_render_candidate(
            baseline_score=current.score,
            candidate_score=best.score,
            metric=best.metric,
            min_relative_gain=min_gain,
        )
        accepted = bool(decision.accepted)
        if gate_trace is not None:
            gate_trace.append({
                "mechanism": mechanism,
                "accepted": accepted,
                "selected": best.label if accepted else current.label,
                "candidate": best.label,
                "baseline_score": float(current.score),
                "candidate_score": float(best.score),
                "relative_gain": float(decision.relative_gain),
                "metric": best.metric,
                "reason": str(decision.reason),
                "scale_rule": best.scale_rule,
                "package": list(best.package),
                "candidates": [
                    {
                        "label": cand.label,
                        "score": float(cand.score),
                        "metric": cand.metric,
                        "scale_rule": cand.scale_rule,
                        "package": list(cand.package),
                    }
                    for cand in candidates
                ],
            })
        if accepted:
            old = current.weight
            current = best
            if old is not best.weight:
                del old
        for cand in candidates:
            if cand is not current:
                del cand.weight

    if f6_enabled:
        apply_gate(
            mechanism="four_over_six",
            candidates=[
                candidate(
                    label="four_over_six",
                    scale_rule=enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
                    package=("four_over_six",),
                    include_gptq=False,
                    include_scale_sweep=False,
                )
            ],
        )

    if gptq_enabled and acts_for_render is not None:
        gptq_name = "fisher_gptq" if fisher_row_weights is not None else "gptq"
        primary_scale_rule = gptq_scale_rule or current.scale_rule
        primary_package = (
            (gptq_name, "gptq") if gptq_name != "gptq" else ("gptq",)
        )
        primary_package = tuple(dict.fromkeys((*gptq_modifiers, *primary_package)))
        packages: list[_RenderedCandidate] = [
            candidate(
                label="+".join((primary_scale_rule, *gptq_modifiers, gptq_name)),
                scale_rule=primary_scale_rule,
                package=primary_package,
                include_gptq=True,
                include_scale_sweep=False,
            )
        ]
        if (
            f6_enabled
            and not joint_scale_opt_enabled
            and current.scale_rule != enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE
        ):
            packages.append(candidate(
                label=f"four_over_six+{gptq_name}",
                scale_rule=enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
                package=(
                    (*gptq_modifiers, "four_over_six", gptq_name, "gptq")
                    if gptq_name != "gptq" else
                    (*gptq_modifiers, "four_over_six", "gptq")
                ),
                include_gptq=True,
                include_scale_sweep=False,
            ))
        apply_gate(mechanism=gptq_name, candidates=packages)

    if scale_sweep_enabled and acts_for_render is not None:
        scale_candidates: list[_RenderedCandidate] = [
            candidate(
                label=f"{current.label}+scale_sweep",
                scale_rule=current.scale_rule,
                package=tuple(dict.fromkeys((*current.package, "scale_sweep"))),
                include_gptq=current.has_gptq,
                include_scale_sweep=True,
            )
        ]
        if gptq_enabled and not current.has_gptq:
            gptq_name = "fisher_gptq" if fisher_row_weights is not None else "gptq"
            scale_rule = gptq_scale_rule or current.scale_rule
            pkg = (
                (gptq_name, "gptq", "scale_sweep")
                if gptq_name != "gptq" else
                ("gptq", "scale_sweep")
            )
            pkg = tuple(dict.fromkeys((*gptq_modifiers, *pkg)))
            scale_candidates.append(candidate(
                label="+".join((scale_rule, *gptq_modifiers, gptq_name, "scale_sweep")),
                scale_rule=scale_rule,
                package=pkg,
                include_gptq=True,
                include_scale_sweep=True,
            ))
        if (
            f6_enabled
            and not joint_scale_opt_enabled
            and current.scale_rule != enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE
        ):
            gptq_name = "fisher_gptq" if fisher_row_weights is not None else "gptq"
            include_gptq = bool(gptq_enabled)
            pkg = [*gptq_modifiers, "four_over_six"]
            if include_gptq:
                if gptq_name != "gptq":
                    pkg.extend([gptq_name, "gptq"])
                else:
                    pkg.append("gptq")
            pkg.append("scale_sweep")
            scale_candidates.append(candidate(
                label="+".join(pkg),
                scale_rule=enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
                package=tuple(pkg),
                include_gptq=include_gptq,
                include_scale_sweep=True,
            ))
        apply_gate(mechanism="scale_sweep", candidates=scale_candidates)

    return current.weight.to(device=weight.device, dtype=weight.dtype).contiguous()


def _summarize_render_gate_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "enabled": True,
        "entries": int(len(records)),
        "mechanisms": {},
    }
    mechanisms: dict[str, dict[str, object]] = {}
    for record in records:
        for step in record.get("trace", []):  # type: ignore[union-attr]
            if not isinstance(step, Mapping):
                continue
            mech = str(step.get("mechanism", "unknown"))
            if mech == "baseline":
                continue
            bucket = mechanisms.setdefault(mech, {
                "accepted": 0,
                "rejected": 0,
                "reasons": {},
                "package_accepted": 0,
            })
            accepted = bool(step.get("accepted", False))
            if accepted:
                bucket["accepted"] = int(bucket["accepted"]) + 1
                package = step.get("package")
                if (
                    isinstance(package, Sequence)
                    and not isinstance(package, str)
                    and mech in package
                ):
                    bucket["package_accepted"] = int(bucket["package_accepted"]) + 1
            else:
                bucket["rejected"] = int(bucket["rejected"]) + 1
            reason = str(step.get("reason", "unknown"))
            reasons = bucket["reasons"]
            if isinstance(reasons, dict):
                reasons[reason] = int(reasons.get(reason, 0)) + 1

            package = step.get("package")
            if isinstance(package, Sequence) and not isinstance(package, str):
                for member in package:
                    member_name = str(member)
                    if member_name == mech:
                        continue
                    member_bucket = mechanisms.setdefault(member_name, {
                        "accepted": 0,
                        "rejected": 0,
                        "reasons": {},
                        "package_accepted": 0,
                    })
                    if accepted:
                        member_bucket["package_accepted"] = (
                            int(member_bucket["package_accepted"]) + 1
                        )
    summary["mechanisms"] = mechanisms
    return summary


def render_production_weight(
    weight: torch.Tensor,
    fmt: str,
    *,
    qname: str,
    activations: Mapping[str, torch.Tensor],
    levers: Mapping[str, object],
    joint_global_real: torch.Tensor | None = None,
    input_global_scale: float | None = None,
    act_clip_threshold: float | None = None,
    act_clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    col_weights: torch.Tensor | None = None,
    gate_trace: list[dict[str, object]] | None = None,
) -> torch.Tensor:
    """Compute the production-faithful dequantized weight for ``(qname, fmt)``.

    Returns a tensor matching ``weight.shape`` and dtype.  For NVFP4 this
    runs GPTQ + scale_sweep (the activation-aware passes) with the joint
    fused-sibling NVFP4 global if supplied; for BF16 and RTN-only formats
    it falls back to the registry quantize_dequantize because those formats
    don't benefit from activation-aware refinement in the production pipeline.

    ``joint_global_real`` is the max-across-fused-siblings NVFP4 global
    used to keep q/k/v (or gate/up) per-tensor scales unified — same as
    the export's ``_compute_nvfp4_joint_global``.  When ``None`` the
    per-Linear computed value is used (legacy behavior, only correct for
    isolated Linears with no fused siblings).

    ``act_clip_threshold`` is an optional scalar clamp for activation-aware
    render passes. ``fisher_row_weights`` optionally weights local objectives
    by per-token gradient² from h-detail.

    ``col_weights`` is the per-input-column imatrix vector (``(in_features,)``)
    for this ``qname`` — re-vet **R3**. It is applied for the weighted-render
    families ONLY (``WEIGHTED_RENDER_FAMILIES``: GGUF; the retired codebook
    lane, archived 2026-09-25, #1304, was the other), whose exporters ship
    imatrix-weighted bytes; for every other format the argument is inert and
    the render is **bit-identical** whether it is passed or not (pinned by
    ``tests/test_col_weights_render_identity.py``). The vector is supplied by
    the caller rather than derived here from ``activations`` on purpose: the
    exporter's vector is the harvested ``artifacts/cb_col_weights.pkl``, which
    includes the synthesized packed-expert entries
    (``moe_imatrix.synthesize_packed_expert_col_weights``) that no raw
    activation matrix on this path can reproduce. With it, a weighted lane's
    cost, KL and shipped bytes can come from ONE render through
    ``ProductionWeightCache``.
    """
    from prismaquant import format_registry as fr

    fmt = fr.canonical_format_name(str(fmt).strip().upper())
    # Tessera owns its own byte path. It is intercepted here, ahead of the
    # format cascade, because the cascade's terminal branch is the registry's
    # ``quantize_dequantize`` -- for Tessera a weights-only *reconstruction*,
    # not the decoded wire, and not the H-aware encode that ships. One render
    # for the surrogate, the KL and the bytes (principle 8) means this one.
    if fr.is_tessera_format_name(fmt):
        # Imported inside the branch: ``tessera_render`` pulls in the
        # ``tessera`` package, and an NVFP4-only pipeline must not.
        from prismaquant.tessera_render import render_tessera_production

        return render_tessera_production(
            weight,
            fmt,
            qname=qname,
            activations=activations,
            levers=levers,
        )
    # Resolved before any branch so a retired or unknown name refuses here
    # (``fr.RetiredFormatError`` for a stale codebook rung), never inside a
    # predicate that would read it as "not weighted".
    fr.get_format(fmt)
    if col_weights is not None and not (
        _format_supports_render_mechanism(fmt, "weighted_vq")
        and bool(levers.get("weighted_vq", True))
    ):
        # Inert for every non-weighted family: fall through to the exact
        # pre-R3 code path, no branch, no dtype churn.
        col_weights = None
    clip_rescale = "none"
    if str(act_clip_rescale or "none").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
        "none",
    }:
        raise ValueError("activation clip rescaling is not supported")
    progressive_gates = _env_flag("PRISMAQUANT_RENDER_PROGRESSIVE_GATES", True)
    if fmt in NVFP4_RENDER_EQUIVALENT and progressive_gates:
        return _render_nvfp4_progressively(
            weight,
            qname=qname,
            activations=activations,
            levers=levers,
            joint_global_real=joint_global_real,
            act_clip_threshold=act_clip_threshold,
            act_clip_rescale=clip_rescale,
            fisher_row_weights=fisher_row_weights,
            gate_trace=gate_trace,
        )

    if fmt not in NVFP4_RENDER_EQUIVALENT:
        spec = fr.get_format(fmt)
        acts = activations.get(qname)
        acts_for_render = (
            acts.detach().to(device=weight.device, dtype=torch.float32)
            if acts is not None and int(acts.shape[-1]) == int(weight.shape[1])
            else None
        )
        # R3: the weighted families' one render definition, shared with the
        # inline cost render and the emulation path; the registry QDQ when
        # ``col_weights`` is inert.
        from prismaquant.emu_forward_kl import weighted_quantize_dequantize

        baseline = weighted_quantize_dequantize(
            spec,
            weight.detach(),
            None if col_weights is None
            else col_weights.reshape(-1).to(weight.device),
        ).to(device=weight.device, dtype=weight.dtype)
        reference = weight.detach().to(torch.float32)
        # Clip-consistent gate matrix — same contract as the NVFP4
        # progressive path (audit 2026-07-02 §3.9): score under the matrix
        # the GPTQ/scale-sweep passes optimize on.
        acts_for_gate = _gate_activation_matrix(
            acts_for_render,
            int(weight.shape[1]),
            device=weight.device,
            act_clip_threshold=act_clip_threshold,
            act_clip_rescale=clip_rescale,
            fisher_row_weights=fisher_row_weights,
        )
        baseline_score, baseline_metric = _render_score_for_gate(
            reference,
            baseline,
            acts_for_gate,
        )
        current = _RenderedCandidate(
            label=f"{fmt.lower()}+" + (
                "weighted_vq" if col_weights is not None else "rtn"),
            weight=baseline.contiguous(),
            score=float(baseline_score),
            metric=baseline_metric,
            scale_rule="",
            package=(),
            has_gptq=False,
        )
        if gate_trace is not None:
            gate_trace.append({
                "mechanism": "baseline",
                "selected": current.label,
                "score": float(current.score),
                "metric": current.metric,
                "package": [],
            })

        def _apply_non_nv_gate(
            *,
            mechanism: str,
            candidates: Sequence[_RenderedCandidate],
        ) -> None:
            nonlocal current
            if not candidates:
                return
            best = min(candidates, key=lambda item: item.score)
            decision = gate_render_candidate(
                baseline_score=current.score,
                candidate_score=best.score,
                metric=best.metric,
                min_relative_gain=_env_float(
                    "PRISMAQUANT_RENDER_GATE_MIN_GAIN",
                    0.0,
                    lo=-1.0,
                    hi=1.0,
                ),
            )
            if gate_trace is not None:
                gate_trace.append({
                    "mechanism": mechanism,
                    "accepted": bool(decision.accepted),
                    "selected": (
                        best.label if decision.accepted else current.label
                    ),
                    "candidate": best.label,
                    "baseline_score": float(current.score),
                    "candidate_score": float(best.score),
                    "relative_gain": float(decision.relative_gain),
                    "metric": best.metric,
                    "reason": str(decision.reason),
                    "package": list(best.package),
                    "candidates": [
                        {
                            "label": cand.label,
                            "score": float(cand.score),
                            "metric": cand.metric,
                            "package": list(cand.package),
                        }
                        for cand in candidates
                    ],
                })
            if decision.accepted:
                old = current.weight
                current = best
                if old is not best.weight:
                    del old
            for cand in candidates:
                if cand is not current and cand.weight is not current.weight:
                    del cand.weight

        def _non_nv_candidate(
            *,
            label: str,
            weight_dq: torch.Tensor,
            package: tuple[str, ...],
            has_gptq: bool,
        ) -> _RenderedCandidate:
            rendered = weight_dq.to(device=weight.device, dtype=weight.dtype).contiguous()
            score, metric = _render_score_for_gate(
                reference, rendered, acts_for_gate,
            )
            return _RenderedCandidate(
                label=label,
                weight=rendered,
                score=float(score),
                metric=metric,
                scale_rule="",
                package=package,
                has_gptq=bool(has_gptq),
            )

        if (
            _format_supports_render_mechanism(fmt, "gptq")
            and bool(levers.get("gptq", True))
            and acts_for_render is not None
        ):
            from prismaquant import export_native_compressed as enc

            joint_scale_opt = bool(
                levers.get("joint_scale_opt", False)
                and _format_supports_render_mechanism(fmt, "joint_scale_opt")
            )
            static_act_order = bool(
                levers.get("static_act_order", False)
                and _format_supports_render_mechanism(fmt, "static_act_order")
            )
            base_package = (
                ("joint_scale_opt", "gptq")
                if joint_scale_opt else
                ("gptq",)
            )
            use_damp_sweep = (
                _damp_sweep_enabled()
            )

            def _gptq_candidate(use_static_act_order: bool) -> _RenderedCandidate:
                package = tuple(dict.fromkeys((
                    *(
                        ("static_act_order",)
                        if use_static_act_order else
                        ()
                    ),
                    *base_package,
                )))
                if fmt == "MXFP4":
                    if use_damp_sweep:
                        _q, _s, candidate = enc._gptq_obs_rounding_mxfp4_swept(
                            reference,
                            acts_for_render,
                            group_size=32,
                            clip_threshold=act_clip_threshold,
                            clip_rescale=clip_rescale,
                            fisher_row_weights=fisher_row_weights,
                            static_act_order=use_static_act_order,
                        )
                    else:
                        _q, _s, candidate = enc._gptq_obs_rounding_mxfp4(
                            reference,
                            acts_for_render,
                            group_size=32,
                            clip_threshold=act_clip_threshold,
                            clip_rescale=clip_rescale,
                            fisher_row_weights=fisher_row_weights,
                            static_act_order=use_static_act_order,
                        )
                elif use_damp_sweep:
                    _q, _s, candidate = enc._gptq_obs_rounding_fp8_like_swept(
                        reference,
                        acts_for_render,
                        fmt=fmt,
                        group_size=32,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=clip_rescale,
                        fisher_row_weights=fisher_row_weights,
                        joint_scale_opt=joint_scale_opt,
                        static_act_order=use_static_act_order,
                    )
                else:
                    _q, _s, candidate = enc._gptq_obs_rounding_fp8_like(
                        reference,
                        acts_for_render,
                        fmt=fmt,
                        group_size=32,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=clip_rescale,
                        fisher_row_weights=fisher_row_weights,
                        joint_scale_opt=joint_scale_opt,
                        static_act_order=use_static_act_order,
                    )
                return _non_nv_candidate(
                    label=f"{fmt.lower()}+{'+'.join(package)}",
                    weight_dq=candidate,
                    package=package,
                    has_gptq=True,
                )

            gptq_candidates = [_gptq_candidate(False)]
            if static_act_order:
                gptq_candidates.append(_gptq_candidate(True))
            _apply_non_nv_gate(
                mechanism="gptq",
                candidates=gptq_candidates,
            )

        if (
            _format_supports_render_mechanism(fmt, "scale_sweep")
            and bool(levers.get("scale_sweep", False))
            and acts_for_render is not None
        ):
            if fmt == "MXFP8_E4M3":
                from prismaquant.export_native_compressed import (
                    _mxfp8_scale_sweep_quantize,
                )

                _, _, w_dq = _mxfp8_scale_sweep_quantize(
                    current.weight.detach().to(torch.float32),
                    acts_for_render,
                    group_size=32,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                )
            else:
                from prismaquant.export_native_compressed import (
                    _fp8_dynamic_scale_sweep_quantize,
                )

                _, _, w_dq = _fp8_dynamic_scale_sweep_quantize(
                    current.weight.detach().to(torch.float32),
                    acts_for_render,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                )
            candidate = _non_nv_candidate(
                label=f"{current.label}+scale_sweep",
                weight_dq=w_dq,
                package=tuple(dict.fromkeys((*current.package, "scale_sweep"))),
                has_gptq=current.has_gptq,
            )
            if progressive_gates:
                _apply_non_nv_gate(
                    mechanism="scale_sweep",
                    candidates=[candidate],
                )
                return current.weight.contiguous()
            return candidate.weight.contiguous()
        return current.weight.contiguous()

    from prismaquant.export_native_compressed import _quantize_2d

    with _temporarily_install_act_aware(activations, levers):
        result = _quantize_2d(
            weight.detach().clone(),
            fmt="NVFP4",
            linear_name=qname,
            nvfp4_global_real_override=joint_global_real,
            input_global_scale_override=input_global_scale,
            act_clip_threshold=act_clip_threshold,
            act_clip_rescale=clip_rescale,
            fisher_row_weights=fisher_row_weights,
            compute_only=True,
        )
    w_dq = result["_w_dq"]
    return w_dq.to(device=weight.device, dtype=weight.dtype).contiguous()


def _resolve_production_render_levers(
    levers: Mapping[str, object] | None,
) -> dict:
    """Normalize a caller lever dict to the production render contract.

    Shared by ``fill_production_weight_cache`` (resident) and the streaming
    driver so both apply IDENTICAL damp/JSO/act-order/scale-rule defaulting —
    the rendered weights must be byte-identical between the two paths.
    """
    levers = dict(levers) if levers is not None else {}
    default_optional_levers = not bool(levers.pop("none", False))
    if not default_optional_levers:
        for name in (
            "gptq",
            "scale_sweep",
            "fisher_gptq",
            "static_act_order",
            "joint_scale_opt",
        ):
            levers.setdefault(name, False)
    levers.setdefault("gptq", True)
    levers.setdefault(
        "gptq_damp_sweep",
        bool(levers.get("gptq", True))
        and _damp_sweep_enabled(),
    )
    if not levers["gptq_damp_sweep"]:
        # Provenance: which fixed damp the no-sweep renders used
        # (PRISMAQUANT_GPTQ_DAMP, default 0.01) — without this, two
        # fixed-damp caches are metadata-indistinguishable.
        from prismaquant.export_native_compressed import (
            _resolve_gptq_fixed_damp,
        )
        levers.setdefault("gptq_fixed_damp", _resolve_gptq_fixed_damp())
    levers.setdefault("scale_sweep", False)
    levers.setdefault(
        "static_act_order",
        _env_flag("PRISMAQUANT_GPTQ_STATIC_ACT_ORDER", False),
    )
    levers.setdefault(
        "joint_scale_opt",
        _env_flag("PRISMAQUANT_NVFP4_JOINT_SCALE_OPT", False),
    )
    if not bool(levers.get("gptq", True)):
        levers["static_act_order"] = False
        levers["joint_scale_opt"] = False
    levers.setdefault(
        "fisher_gptq",
        _env_flag("PRISMAQUANT_FISHER_WEIGHTED_GPTQ", False),
    )
    from prismaquant.export_native_compressed import (
        NVFP4_SCALE_RULE_ENV,
        NVFP4_SCALE_RULE_JOINT_MSE,
        resolve_nvfp4_scale_rule,
    )
    if (
        bool(levers.get("joint_scale_opt", False))
        and "nvfp4_scale_rule" not in levers
        and NVFP4_SCALE_RULE_ENV not in os.environ
    ):
        levers["nvfp4_scale_rule"] = NVFP4_SCALE_RULE_JOINT_MSE
    levers.setdefault("nvfp4_scale_rule", resolve_nvfp4_scale_rule())
    # The A-side scale policy is a render input like the W-side scale rule
    # beside it: it already reaches ``render_production_weight`` as this
    # fill's ``input_global_scale``, and it decides what every
    # activation-aware render SCORE means.  Resolve it here, once, and let it
    # travel with the levers -- into the directory render identity, the
    # union per-shard identity and each score record --
    # so a cache priced under one policy cannot be silently resumed, rescored
    # or KL-validated under another (#227).  An explicit caller value wins and
    # is canonicalized; a bad one refuses here rather than mid-fill.
    levers[RENDER_LEVER_INPUT_GLOBAL_SCALE_POLICY] = (
        _resolve_render_input_global_scale_policy(
            levers.get(RENDER_LEVER_INPUT_GLOBAL_SCALE_POLICY)
        )
    )
    return levers


def _resolve_render_input_global_scale_policy(
    value: object | None = None,
) -> str:
    """Canonical input-global-scale policy id, from the contract owner."""
    from prismaquant.nvfp4_activation_contract import (
        resolve_input_global_scale_policy,
    )

    return resolve_input_global_scale_policy(
        None if value is None else str(value)
    )


def _render_levers_input_global_scale_policy(
    levers: Mapping[str, object] | None,
) -> str:
    """The policy a resolved lever mapping was priced under.

    Falls back to the live resolution for a lever mapping built before #227
    (or by a caller that never went through
    :func:`_resolve_production_render_levers`), which is what those callers
    already got.
    """
    value = (
        levers.get(RENDER_LEVER_INPUT_GLOBAL_SCALE_POLICY)
        if isinstance(levers, Mapping) else None
    )
    return _resolve_render_input_global_scale_policy(value)


def _resolve_render_mechanism_plan(levers: Mapping[str, object]):
    """Build the render-mechanism ordering plan from normalized levers."""
    enabled_mechanisms: list[str] = []
    if str(levers.get("nvfp4_scale_rule", "")).strip() == "four_over_six_mse":
        enabled_mechanisms.append("four_over_six")
    if bool(levers.get("gptq", True)):
        enabled_mechanisms.append("gptq")
    if bool(levers.get("static_act_order", False)):
        enabled_mechanisms.append("static_act_order")
    if bool(levers.get("joint_scale_opt", False)):
        enabled_mechanisms.append("joint_scale_opt")
    if bool(levers.get("fisher_gptq", False)):
        enabled_mechanisms.append("fisher_gptq")
    if bool(levers.get("scale_sweep", False)):
        enabled_mechanisms.append("scale_sweep")
    mechanism_plan = resolve_render_mechanism_order(enabled_mechanisms)
    if mechanism_plan.errors:
        raise ValueError(
            "invalid render mechanism plan: " + "; ".join(mechanism_plan.errors)
        )
    return mechanism_plan


def _qname_set_sha256(qnames: Iterable[str]) -> str:
    """Digest an unordered qname set.

    Sorted, so the digest is a property of the SET and not of the order the
    caller happened to enumerate it in. Two ``fill_production_weight_cache``
    calls with equal digests hooked the same Linears, which (with an equal
    ``calib_hash``) is what makes their rendered rows equal.
    """
    payload = "\n".join(sorted(str(q) for q in qnames))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_activation_hook_scope(
    value: object, *, where: str = "activation_hook_scope"
) -> dict[str, object]:
    """Validate an ``activation_hook_scope`` stamp (#147).

    Every rule below is derived from the writer in
    ``fill_production_weight_cache``: the exact five fields, the hooked set's
    sha256, and ``render_narrowed`` as the comparison of the two counts.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    raw = dict(value)
    if raw.get("schema") != ACTIVATION_HOOK_SCOPE_SCHEMA:
        raise ValueError(f"{where} has unsupported schema")
    digest = str(raw.get("hooked_qnames_sha256", "")).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{where}.hooked_qnames_sha256 must be a SHA-256 digest")
    hooked = raw.get("hooked_qnames")
    rendered = raw.get("rendered_qnames")
    narrowed = raw.get("render_narrowed")
    if (
        not isinstance(hooked, int)
        or isinstance(hooked, bool)
        or hooked < 0
        or not isinstance(rendered, int)
        or isinstance(rendered, bool)
        or rendered < 0
        or rendered > hooked
    ):
        raise ValueError(
            f"{where} has malformed hooked/rendered qname counts"
        )
    if not isinstance(narrowed, bool) or narrowed != (rendered != hooked):
        raise ValueError(
            f"{where}.render_narrowed must equal (rendered_qnames != "
            "hooked_qnames)"
        )
    if set(raw) != {
        "schema",
        "hooked_qnames_sha256",
        "hooked_qnames",
        "rendered_qnames",
        "render_narrowed",
    }:
        raise ValueError(f"{where} has unsupported fields")
    return {
        "schema": ACTIVATION_HOOK_SCOPE_SCHEMA,
        "hooked_qnames_sha256": digest,
        "hooked_qnames": int(hooked),
        "rendered_qnames": int(rendered),
        "render_narrowed": bool(narrowed),
    }


def activation_hook_scope_of(source: object) -> dict[str, object] | None:
    """Return the validated hook scope carried by a cache or its metadata.

    ``None`` when the stamp is absent (caches rendered before #130). Present
    but malformed is an error, never a silent skip.
    """
    metadata = source
    if hasattr(source, "metadata"):
        metadata = getattr(source, "metadata")
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get(ACTIVATION_HOOK_SCOPE_KEY)
    if raw is None:
        return None
    return validate_activation_hook_scope(raw)


def assert_same_activation_hook_scope(
    reference: Mapping[str, object] | None,
    candidate: Mapping[str, object] | None,
    *,
    where: str,
) -> dict[str, object] | None:
    """Refuse two activation renderings that cannot be the same one (#147).

    The correct rule for striped campaigns: shards of one campaign hook the
    whole enumeration and render a slice each, so EQUAL hook digests (plus an
    equal hooked count) is the pass condition. Binding per-shard digests into
    the render identity would refuse every striped union; comparing nothing
    lets two renderings ship under one name. Returns the validated reference
    scope (``None`` when both sides predate the stamp).
    """
    if reference is None and candidate is None:
        return None
    if reference is None or candidate is None:
        raise ValueError(
            f"{where}: activation hook scope is present on one side and "
            "missing on the other; cannot prove the same rendering"
        )
    validated_reference = validate_activation_hook_scope(
        reference, where=f"{where} reference"
    )
    validated_candidate = validate_activation_hook_scope(
        candidate, where=f"{where} candidate"
    )
    if (
        validated_candidate["hooked_qnames_sha256"]
        != validated_reference["hooked_qnames_sha256"]
        or validated_candidate["hooked_qnames"]
        != validated_reference["hooked_qnames"]
    ):
        raise ValueError(
            f"{where}: activation hook digests differ: reference="
            f"{validated_reference['hooked_qnames_sha256']} "
            f"candidate={validated_candidate['hooked_qnames_sha256']}; "
            "the two caches were rendered against different enumerations"
        )
    return validated_reference


def _pre_guard_trusted_shards(cache_dir_path: str | Path) -> list[str]:
    """Sorted names of rendered ``.pt`` shards already on disk.

    This is the same file-presence predicate the resume loop admits units
    by, so the admission record names exactly the shards the guard lets
    through on trust.
    """
    try:
        entries = list(Path(cache_dir_path).iterdir())
    except OSError:
        return []
    return sorted(
        entry.name
        for entry in entries
        if entry.is_file() and entry.suffix == ".pt"
    )


def build_pre_guard_admission(trusted_shards: Sequence[str]) -> dict[str, object]:
    """Build the trust-admission record for a pre-guard directory (#146).

    A directory that holds rendered shards but no sidecar is admitted on
    trust — refusing would strand every existing cache — but the admission
    is recorded in the sidecar itself: that shards were admitted, how many,
    and their sorted shard names. A fresh directory (no shards) carries no
    record at all.
    """
    names = [str(name) for name in trusted_shards]
    if not names or any(not name for name in names):
        raise ValueError("pre-guard admission requires a nonempty shard list")
    if list(names) != sorted(names):
        raise ValueError("pre-guard admission shard names must be sorted")
    return {
        "admitted_on_trust": True,
        "trusted_shard_count": len(names),
        "trusted_shards": list(names),
    }


def validate_pre_guard_admission(
    value: object, *, where: str = "pre-guard admission"
) -> dict[str, object]:
    """Validate a ``pre_guard_admission`` sidecar section.

    Every rule below is derived from the writer in
    :func:`build_pre_guard_admission`: the exact field set, the true
    admission flag, and the shard count matching the named shards. Present
    but malformed is an error, never a silent skip.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    raw = dict(value)
    if set(raw) != {
        "admitted_on_trust",
        "trusted_shard_count",
        "trusted_shards",
    }:
        raise ValueError(f"{where} has unsupported fields")
    if raw.get("admitted_on_trust") is not True:
        raise ValueError(f"{where}.admitted_on_trust must be true")
    count = raw.get("trusted_shard_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(
            f"{where}.trusted_shard_count must be a positive int"
        )
    shards = raw.get("trusted_shards")
    if (
        not isinstance(shards, list)
        or not shards
        or any(not isinstance(name, str) or not name for name in shards)
        or list(shards) != sorted(shards)
    ):
        raise ValueError(
            f"{where}.trusted_shards must be a sorted nonempty string list"
        )
    if count != len(shards):
        raise ValueError(
            f"{where}.trusted_shard_count must match len(trusted_shards)"
        )
    return {
        "admitted_on_trust": True,
        "trusted_shard_count": int(count),
        "trusted_shards": list(shards),
    }


def build_production_cache_render_identity(
    *,
    render_scope: str,
    requested_formats: Sequence[str],
    levers: Mapping[str, object],
    mechanism_plan,
    calib_hash: str,
    eligible_qnames: Iterable[str],
    render_formats_by_qname: Mapping[str, Sequence[str]],
    max_act_rows: int,
) -> dict[str, object]:
    """Build the render identity a cache directory's shards were rendered under.

    This is the directory-level equivalent of the union campaign's
    ``_render_identity`` (which binds the same render semantics per shard
    bundle): every value-bearing input that changes the rendered bytes is a
    field here — the render scope, the requested format menu, the resolved
    levers plus their mechanism order, the calibration hash, the hooked
    enumeration digest (#130: what the shared priority stream — and therefore
    every Linear's rows — is a function of), the exact rendered
    ``qname|fmt`` pairs (what an assignment scope or an
    ``--include-qnames-file`` narrowing renders), and ``max_act_rows`` (the
    reservoir size, hence the rows kept). Two fills that admit the same
    sidecar rendered the same bytes; anything else refuses with the differing
    field named.
    """
    if not isinstance(levers, Mapping):
        raise ValueError("production cache render identity requires levers")
    ordered = getattr(mechanism_plan, "ordered", mechanism_plan)
    if ordered is None:
        raise ValueError(
            "production cache render identity requires a render mechanism plan"
        )
    mechanism_records = []
    for raw in ordered:
        if isinstance(raw, Mapping):
            record = {
                str(key): raw[key]
                for key in (
                    "name", "operation", "scope", "gate_metric"
                )
                if key in raw
            }
        else:
            record = {
                "name": str(getattr(raw, "name", raw)),
                **({
                    key: getattr(raw, key)
                    for key in ("operation", "scope", "gate_metric")
                    if hasattr(raw, key)
                }),
            }
        mechanism_records.append(record)
    hooked = sorted({str(q) for q in eligible_qnames})
    rendered_pairs = sorted({
        f"{str(qname)}|{str(fmt).strip().upper()}"
        for qname, fmts in render_formats_by_qname.items()
        for fmt in fmts
    })
    digest = str(calib_hash or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{32,64}", digest) is None:
        raise ValueError(
            "production cache render identity requires a value-bearing "
            "calib_hash"
        )
    rows = int(max_act_rows)
    if rows < 1:
        raise ValueError("production cache render identity requires max_act_rows >= 1")
    return {
        "schema": RENDER_IDENTITY_SCHEMA,
        "render_scope": str(render_scope),
        "requested_formats": sorted({
            str(fmt).strip().upper()
            for fmt in requested_formats
            if str(fmt).strip()
        }),
        "levers": _canonical_json_value(
            dict(levers), where="production cache render levers"
        ),
        "render_mechanism_order": _canonical_json_value(
            mechanism_records, where="production cache render mechanism order"
        ),
        "calib_hash": digest,
        "hooked_qnames_sha256": _qname_set_sha256(hooked),
        "hooked_qnames": len(hooked),
        "rendered_pairs": rendered_pairs,
        "max_act_rows": rows,
    }


def validate_production_cache_render_identity(
    value: object, *, where: str = "production cache render identity"
) -> dict[str, object]:
    """Validate a render-identity sidecar payload.

    Every rule below is derived from the writer in
    :func:`build_production_cache_render_identity`: the exact field set, the
    sorted format/pair lists, and the digest shapes. Present but malformed is
    an error, never a silent skip.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    raw = dict(value)
    if raw.get("schema") != RENDER_IDENTITY_SCHEMA:
        raise ValueError(f"{where} has unsupported schema")
    base_fields = {
        "schema",
        "render_scope",
        "requested_formats",
        "levers",
        "render_mechanism_order",
        "calib_hash",
        "hooked_qnames_sha256",
        "hooked_qnames",
        "rendered_pairs",
        "max_act_rows",
    }
    append_fields = {
        MTP_APPEND_SIDECAR_KEY,
        PACKED_APPEND_SIDECAR_KEY,
        PACKED_STREAMING_APPEND_SIDECAR_KEY,
    }
    admission_fields = {PRE_GUARD_ADMISSION_SIDECAR_KEY}
    # Not `set(raw) < base_fields`: that is a PROPER-subset test, so a
    # sidecar missing a base field but carrying an append section is
    # neither a subset nor an unknown-field case and would fall through
    # here.  Every base field does raise on its own below, so nothing
    # got past this in practice -- but a field check should refuse for
    # the reason it names.
    if (
        set(raw) - base_fields - append_fields - admission_fields
        or not base_fields <= set(raw)
    ):
        raise ValueError(f"{where} has unsupported fields")
    if not isinstance(raw.get("render_scope"), str) or not raw["render_scope"]:
        raise ValueError(f"{where}.render_scope must be a nonempty string")
    formats = raw.get("requested_formats")
    if (
        not isinstance(formats, list)
        or not formats
        or any(not isinstance(fmt, str) or not fmt for fmt in formats)
        or list(formats) != sorted(formats)
    ):
        raise ValueError(
            f"{where}.requested_formats must be a sorted nonempty string list"
        )
    levers = raw.get("levers")
    if not isinstance(levers, Mapping) or not levers:
        raise ValueError(f"{where}.levers must be a nonempty JSON object")
    _canonical_json_value(dict(levers), where=f"{where}.levers")
    mechanism_order = raw.get("render_mechanism_order")
    if not isinstance(mechanism_order, list) or not all(
        isinstance(record, Mapping) for record in mechanism_order
    ):
        raise ValueError(
            f"{where}.render_mechanism_order must be a list of JSON objects"
        )
    digest = str(raw.get("calib_hash", "")).strip().lower()
    if re.fullmatch(r"[0-9a-f]{32,64}", digest) is None:
        raise ValueError(f"{where}.calib_hash must be a hex digest")
    hook_digest = str(raw.get("hooked_qnames_sha256", "")).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", hook_digest) is None:
        raise ValueError(
            f"{where}.hooked_qnames_sha256 must be a SHA-256 digest"
        )
    hooked = raw.get("hooked_qnames")
    if not isinstance(hooked, int) or isinstance(hooked, bool) or hooked < 0:
        raise ValueError(f"{where}.hooked_qnames must be a non-negative int")
    pairs = raw.get("rendered_pairs")
    if (
        not isinstance(pairs, list)
        or not pairs
        or any(not isinstance(pair, str) or "|" not in pair for pair in pairs)
        or list(pairs) != sorted(pairs)
    ):
        raise ValueError(
            f"{where}.rendered_pairs must be a sorted nonempty "
            '"qname|FMT" string list'
        )
    rows = raw.get("max_act_rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
        raise ValueError(f"{where}.max_act_rows must be a positive int")
    identity: dict[str, object] = {
        "schema": RENDER_IDENTITY_SCHEMA,
        "render_scope": str(raw["render_scope"]),
        "requested_formats": list(formats),
        "levers": _canonical_json_value(
            dict(levers), where=f"{where}.levers"
        ),
        "render_mechanism_order": _canonical_json_value(
            list(mechanism_order),
            where=f"{where}.render_mechanism_order",
        ),
        "calib_hash": digest,
        "hooked_qnames_sha256": hook_digest,
        "hooked_qnames": int(hooked),
        "rendered_pairs": list(pairs),
        "max_act_rows": int(rows),
    }
    # Append sections (#170) ride in the same sidecar but are owned by their
    # append writers, not by the base fill. The base fill neither writes nor
    # compares them; each append compare-or-writes its own section through
    # ``_check_and_record_append_identity``.
    if MTP_APPEND_SIDECAR_KEY in raw:
        identity[MTP_APPEND_SIDECAR_KEY] = validate_mtp_append_identity(
            raw[MTP_APPEND_SIDECAR_KEY],
            where=f"{where}.{MTP_APPEND_SIDECAR_KEY}",
        )
    if PACKED_APPEND_SIDECAR_KEY in raw:
        identity[PACKED_APPEND_SIDECAR_KEY] = validate_packed_append_identity(
            raw[PACKED_APPEND_SIDECAR_KEY],
            where=f"{where}.{PACKED_APPEND_SIDECAR_KEY}",
        )
    if PACKED_STREAMING_APPEND_SIDECAR_KEY in raw:
        identity[PACKED_STREAMING_APPEND_SIDECAR_KEY] = (
            validate_packed_streaming_append_identity(
                raw[PACKED_STREAMING_APPEND_SIDECAR_KEY],
                where=f"{where}.{PACKED_STREAMING_APPEND_SIDECAR_KEY}",
            )
        )
    # The trust admission (#146) is history, not render input: it is owned
    # by the guard writers, carried forward by every later sidecar write,
    # and never compared, so a guarded directory carrying it still resumes
    # cleanly against a caller identity computed without it.
    if PRE_GUARD_ADMISSION_SIDECAR_KEY in raw:
        identity[PRE_GUARD_ADMISSION_SIDECAR_KEY] = validate_pre_guard_admission(
            raw[PRE_GUARD_ADMISSION_SIDECAR_KEY],
            where=f"{where}.{PRE_GUARD_ADMISSION_SIDECAR_KEY}",
        )
    return identity


def build_mtp_append_identity(
    *,
    max_act_rows: int,
    activation_source_hash: str,
    activation_rows: Mapping[str, int],
    source_prefix: str,
    source_tensor_count: int,
    profile_name: str,
) -> dict[str, object]:
    """Build the render identity for one MTP append (#170).

    Every value-bearing input that changes the appended bytes is a field
    here: the activation-source content hash (what the probe rows contain,
    not the directory path that held them), the per-module row counts, the
    reservoir size, and the profile/source binding the synthesized module
    came from. Deliberately NOT bound: the render narrowing (which
    ``mtp.*`` subset this call appends) — the append replaces its scope, so
    consecutive stripes stay legal exactly as
    ``test_mtp_append_replaces_prior_scope_without_double_counting`` pins.
    """
    rows = int(max_act_rows)
    if rows < 1:
        raise ValueError("MTP append identity requires max_act_rows >= 1")
    source_hash = str(activation_source_hash or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{32,64}", source_hash) is None:
        raise ValueError(
            "MTP append identity requires a value-bearing "
            "activation_source_hash"
        )
    if not isinstance(activation_rows, Mapping) or not activation_rows:
        raise ValueError(
            "MTP append identity requires a nonempty activation_rows mapping"
        )
    counts = {
        str(qname): int(count)
        for qname, count in activation_rows.items()
    }
    if (
        any(not qname for qname in counts)
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
            for count in counts.values()
        )
    ):
        raise ValueError(
            "MTP append identity requires positive per-module row counts"
        )
    prefix = str(source_prefix or "").strip()
    if not prefix:
        raise ValueError("MTP append identity requires source_prefix")
    total = int(source_tensor_count)
    if total < 1:
        raise ValueError(
            "MTP append identity requires source_tensor_count >= 1"
        )
    profile = str(profile_name or "").strip()
    if not profile:
        raise ValueError("MTP append identity requires profile_name")
    return {
        "schema": RENDER_IDENTITY_MTP_APPEND_SCHEMA,
        "profile_name": profile,
        "source_prefix": prefix,
        "source_tensor_count": total,
        "max_act_rows": rows,
        "activation_source_hash": source_hash,
        "activation_rows": _canonical_json_value(
            counts, where="MTP append activation rows"
        ),
    }


def validate_mtp_append_identity(
    value: object, *, where: str = "MTP append identity"
) -> dict[str, object]:
    """Validate an MTP append identity section.

    Every rule below is derived from the writer in
    :func:`build_mtp_append_identity`: the exact field set, the digest
    shape, and the positive row counts. Present but malformed is an error,
    never a silent skip.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    raw = dict(value)
    if raw.get("schema") != RENDER_IDENTITY_MTP_APPEND_SCHEMA:
        raise ValueError(f"{where} has unsupported schema")
    if set(raw) != {
        "schema",
        "profile_name",
        "source_prefix",
        "source_tensor_count",
        "max_act_rows",
        "activation_source_hash",
        "activation_rows",
    }:
        raise ValueError(f"{where} has unsupported fields")
    if not isinstance(raw.get("profile_name"), str) or not raw["profile_name"]:
        raise ValueError(f"{where}.profile_name must be a nonempty string")
    if not isinstance(raw.get("source_prefix"), str) or not raw["source_prefix"]:
        raise ValueError(f"{where}.source_prefix must be a nonempty string")
    total = raw.get("source_tensor_count")
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        raise ValueError(
            f"{where}.source_tensor_count must be a positive int"
        )
    rows = raw.get("max_act_rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
        raise ValueError(f"{where}.max_act_rows must be a positive int")
    source_hash = str(raw.get("activation_source_hash", "")).strip().lower()
    if re.fullmatch(r"[0-9a-f]{32,64}", source_hash) is None:
        raise ValueError(
            f"{where}.activation_source_hash must be a hex digest"
        )
    counts = raw.get("activation_rows")
    if (
        not isinstance(counts, Mapping)
        or not counts
        or any(not isinstance(qname, str) or not qname for qname in counts)
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
            for count in counts.values()
        )
    ):
        raise ValueError(
            f"{where}.activation_rows must map qnames to positive ints"
        )
    return {
        "schema": RENDER_IDENTITY_MTP_APPEND_SCHEMA,
        "profile_name": str(raw["profile_name"]),
        "source_prefix": str(raw["source_prefix"]),
        "source_tensor_count": int(total),
        "max_act_rows": int(rows),
        "activation_source_hash": source_hash,
        "activation_rows": _canonical_json_value(
            {str(qname): int(count) for qname, count in counts.items()},
            where=f"{where}.activation_rows",
        ),
    }


def build_packed_expert_append_identity(
    *,
    module_token_budget: int,
    max_rows_per_expert: int,
    eval_rows_per_expert: int,
    render_mode: str,
    gate_calibration_hash: str | None,
    gate_token_budget: int | None,
    hooked_qnames: Iterable[str],
    max_layers: int | None,
    pair_fit_calibration_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the render identity for one packed-expert append (#170).

    Every value-bearing input that changes the appended bytes is a field
    here: the fit reservoir budget, the per-expert row caps, the render
    mode, the gate corpus hash plus its own budget (``None`` without a
    cross-domain gate corpus), and the hooked-module enumeration digest
    (#145: what the shared priority stream — and therefore every module's
    kept rows — is a function of). Deliberately NOT bound:

    * the render narrowing (which packed tensors this call renders) — the
      eager format-menu build and the lazy per-Pareto-point gap-fill render
      different subsets with identical bytes, so binding the subset would
      refuse the exact sequence M4 relies on.

    The fit corpus IS bound, but per pair (#173), not per section: each
    ``"qname|FMT"`` entry records the ``calibration_data_hash`` of the fit
    corpus its bytes were fitted on. A section-level bind was tried for
    #170 and reverted — it refused the sanctioned M4 lazy gap-fill, which
    legitimately renders a disjoint split of the cache under the
    render-split calibration. Per-pair binding lets that split record its
    own entries: disjoint pairs merge, while the same pair under a
    different hash refuses (existing shards are never re-rendered, so the
    recorded hash is always the bytes on disk). Pipeline builds pass one
    calib to the fill and all its appends, so the base identity's
    ``calib_hash`` still covers that path.
    """
    budget = int(module_token_budget)
    if budget < 1:
        raise ValueError(
            "packed-expert append identity requires module_token_budget >= 1"
        )
    fit_rows = int(max_rows_per_expert)
    if fit_rows < 1:
        raise ValueError(
            "packed-expert append identity requires max_rows_per_expert >= 1"
        )
    eval_rows = int(eval_rows_per_expert)
    if eval_rows < 1:
        raise ValueError(
            "packed-expert append identity requires eval_rows_per_expert >= 1"
        )
    mode = str(render_mode or "").strip()
    if mode not in {"batched", "per_expert"}:
        raise ValueError(
            "packed-expert append identity requires "
            "render_mode in {'batched', 'per_expert'}"
        )
    gate_hash: str | None = None
    if gate_calibration_hash is not None:
        gate_hash = str(gate_calibration_hash).strip().lower()
        if re.fullmatch(r"[0-9a-f]{32,64}", gate_hash) is None:
            raise ValueError(
                "packed-expert append identity requires a hex "
                "gate_calibration_hash"
            )
    gate_budget: int | None = None
    if gate_token_budget is not None:
        gate_budget = int(gate_token_budget)
        if gate_budget < 1:
            raise ValueError(
                "packed-expert append identity requires "
                "gate_token_budget >= 1"
            )
    if (gate_hash is None) != (gate_budget is None):
        raise ValueError(
            "packed-expert append identity requires gate_token_budget "
            "exactly when a gate corpus is present"
        )
    hooked = sorted({str(q) for q in hooked_qnames})
    layers: int | None = None
    if max_layers is not None:
        layers = int(max_layers)
        if layers < 1:
            raise ValueError(
                "packed-expert append identity requires max_layers >= 1"
            )
    pair_map = _validate_pair_fit_map(
        pair_fit_calibration_hashes,
        where="packed-expert append identity",
    )
    return {
        "schema": RENDER_IDENTITY_PACKED_APPEND_SCHEMA,
        "module_token_budget": budget,
        "max_rows_per_expert": fit_rows,
        "eval_rows_per_expert": eval_rows,
        "render_mode": mode,
        "gate_calibration_hash": gate_hash,
        "gate_token_budget": gate_budget,
        "hooked_qnames_sha256": _qname_set_sha256(hooked),
        "hooked_qnames": len(hooked),
        "max_layers": layers,
        "pair_fit_calibration_hashes": pair_map,
    }


def _validate_pair_fit_map(
    value: Mapping[str, str] | None, *, where: str
) -> dict[str, str]:
    """Validate a ``pair -> fit calibration hash`` map (#173).

    Keys are ``"qname|FMT"`` pair names (nonempty, pipe-separated, exactly
    as the packed render loop names them); values are content hashes of
    the fit corpus the pair's bytes were fitted on. The shape rule is
    derived from ``calibration_data_hash`` (blake2b-128 hex, 32 chars;
    the 32..64 window admits longer digests without admitting garbage).
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{where} requires pair_fit_calibration_hashes to be a JSON "
            "object"
        )
    pair_map: dict[str, str] = {}
    for pair, fit_hash in value.items():
        if (
            not isinstance(pair, str)
            or not pair
            or "|" not in pair
            or not pair.split("|")[0]
            or not pair.split("|")[1]
        ):
            raise ValueError(
                f"{where}.pair_fit_calibration_hashes keys must be "
                '"qname|FMT" pair names'
            )
        digest = str(fit_hash or "").strip().lower() if isinstance(
            fit_hash, str
        ) else ""
        if re.fullmatch(r"[0-9a-f]{32,64}", digest) is None:
            raise ValueError(
                f"{where}.pair_fit_calibration_hashes[{pair!r}] must be a "
                "hex digest"
            )
        pair_map[str(pair)] = digest
    return _canonical_json_value(
        pair_map, where=f"{where}.pair_fit_calibration_hashes"
    )


def validate_packed_append_identity(
    value: object, *, where: str = "packed-expert append identity"
) -> dict[str, object]:
    """Validate a packed-expert append identity section.

    Every rule below is derived from the writer in
    :func:`build_packed_expert_append_identity`: the exact field set, the
    digest shapes, and the gate-hash/gate-budget pairing. Present but
    malformed is an error, never a silent skip.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    raw = dict(value)
    if raw.get("schema") != RENDER_IDENTITY_PACKED_APPEND_SCHEMA:
        raise ValueError(f"{where} has unsupported schema")
    base_field_set = {
        "schema",
        "module_token_budget",
        "max_rows_per_expert",
        "eval_rows_per_expert",
        "render_mode",
        "gate_calibration_hash",
        "gate_token_budget",
        "hooked_qnames_sha256",
        "hooked_qnames",
        "max_layers",
    }
    # #173 added ``pair_fit_calibration_hashes``; a sidecar written by the
    # #170 guard has the base ten fields and no pair map. Admit it (its
    # pairs are adopted on the next append) rather than stranding the
    # directory — fail-closed still holds, since every base field below
    # raises on its own and the merged map refuses same-pair conflicts.
    # Two distinct refusals, not one: an unknown field and a missing base
    # field are different faults, and a single set-equality test reports
    # both as "unsupported fields" -- a message that names the wrong
    # reason for half the sidecars it rejects.
    unknown = set(raw) - base_field_set - {"pair_fit_calibration_hashes"}
    if unknown:
        raise ValueError(
            f"{where} has unsupported fields: {sorted(unknown)}"
        )
    absent = base_field_set - set(raw)
    if absent:
        raise ValueError(
            f"{where} is missing required fields: {sorted(absent)}"
        )
    budget = raw.get("module_token_budget")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError(f"{where}.module_token_budget must be a positive int")
    fit_rows = raw.get("max_rows_per_expert")
    if (
        not isinstance(fit_rows, int)
        or isinstance(fit_rows, bool)
        or fit_rows < 1
    ):
        raise ValueError(
            f"{where}.max_rows_per_expert must be a positive int"
        )
    eval_rows = raw.get("eval_rows_per_expert")
    if (
        not isinstance(eval_rows, int)
        or isinstance(eval_rows, bool)
        or eval_rows < 1
    ):
        raise ValueError(
            f"{where}.eval_rows_per_expert must be a positive int"
        )
    if raw.get("render_mode") not in {"batched", "per_expert"}:
        raise ValueError(
            f"{where}.render_mode must be 'batched' or 'per_expert'"
        )
    gate_hash_raw = raw.get("gate_calibration_hash")
    gate_hash: str | None = None
    if gate_hash_raw is not None:
        gate_hash = str(gate_hash_raw).strip().lower()
        if re.fullmatch(r"[0-9a-f]{32,64}", gate_hash) is None:
            raise ValueError(
                f"{where}.gate_calibration_hash must be a hex digest"
            )
    gate_budget_raw = raw.get("gate_token_budget")
    gate_budget: int | None = None
    if gate_budget_raw is not None:
        if (
            not isinstance(gate_budget_raw, int)
            or isinstance(gate_budget_raw, bool)
            or gate_budget_raw < 1
        ):
            raise ValueError(
                f"{where}.gate_token_budget must be a positive int"
            )
        gate_budget = int(gate_budget_raw)
    if (gate_hash is None) != (gate_budget is None):
        raise ValueError(
            f"{where} requires gate_token_budget exactly when a gate "
            "corpus is present"
        )
    hook_digest = str(raw.get("hooked_qnames_sha256", "")).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", hook_digest) is None:
        raise ValueError(
            f"{where}.hooked_qnames_sha256 must be a SHA-256 digest"
        )
    hooked = raw.get("hooked_qnames")
    if not isinstance(hooked, int) or isinstance(hooked, bool) or hooked < 0:
        raise ValueError(f"{where}.hooked_qnames must be a non-negative int")
    layers_raw = raw.get("max_layers")
    layers: int | None = None
    if layers_raw is not None:
        if (
            not isinstance(layers_raw, int)
            or isinstance(layers_raw, bool)
            or layers_raw < 1
        ):
            raise ValueError(f"{where}.max_layers must be a positive int")
        layers = int(layers_raw)
    pair_raw = raw.get("pair_fit_calibration_hashes")
    pair_map: dict[str, str] = {}
    if pair_raw is not None:
        if not isinstance(pair_raw, Mapping):
            raise ValueError(
                f"{where}.pair_fit_calibration_hashes must be a JSON object"
            )
        pair_map = _validate_pair_fit_map(
            {str(pair): fit for pair, fit in pair_raw.items()},
            where=where,
        )
    return {
        "schema": RENDER_IDENTITY_PACKED_APPEND_SCHEMA,
        "module_token_budget": int(budget),
        "max_rows_per_expert": int(fit_rows),
        "eval_rows_per_expert": int(eval_rows),
        "render_mode": str(raw["render_mode"]),
        "gate_calibration_hash": gate_hash,
        "gate_token_budget": gate_budget,
        "hooked_qnames_sha256": hook_digest,
        "hooked_qnames": int(hooked),
        "max_layers": layers,
        "pair_fit_calibration_hashes": pair_map,
    }


def build_packed_streaming_append_identity(
    *,
    module_token_budget: int,
    max_rows_per_expert: int,
    eval_rows_per_expert: int,
    render_mode: str,
    max_layers: int | None,
    layers: Mapping[str, str],
) -> dict[str, object]:
    """Build the render identity for streaming packed-expert appends (#172).

    One streaming call sees a single materialized module (``_render_packed_layer``
    feeds each layer's input snapshot via ``module_acts_override``), so no
    single section can describe the directory — the #170 guard recorded
    nothing at all for that path and the union merged silently. This section
    is the union: shared budgets/mode plus a per-layer map keyed by the
    experts-module qname each call saw, each entry holding the content hash
    of that module's activation snapshot. Consecutive layers merge; the same
    module under a different budget or a different snapshot refuses.
    Deliberately NOT bound: the render narrowing (which packed tensors each
    layer renders) and any cross-domain gate corpus (unsupported on the
    streaming path, so there is none to bind).
    """
    budget = int(module_token_budget)
    if budget < 1:
        raise ValueError(
            "packed streaming append identity requires "
            "module_token_budget >= 1"
        )
    fit_rows = int(max_rows_per_expert)
    if fit_rows < 1:
        raise ValueError(
            "packed streaming append identity requires "
            "max_rows_per_expert >= 1"
        )
    eval_rows = int(eval_rows_per_expert)
    if eval_rows < 1:
        raise ValueError(
            "packed streaming append identity requires "
            "eval_rows_per_expert >= 1"
        )
    mode = str(render_mode or "").strip()
    if mode not in {"batched", "per_expert"}:
        raise ValueError(
            "packed streaming append identity requires "
            "render_mode in {'batched', 'per_expert'}"
        )
    max_l: int | None = None
    if max_layers is not None:
        max_l = int(max_layers)
        if max_l < 1:
            raise ValueError(
                "packed streaming append identity requires max_layers >= 1"
            )
    layer_map = _validate_streaming_layer_map(
        layers, where="packed streaming append identity"
    )
    if not layer_map:
        raise ValueError(
            "packed streaming append identity requires a nonempty layers map"
        )
    return {
        "schema": RENDER_IDENTITY_PACKED_STREAMING_APPEND_SCHEMA,
        "module_token_budget": budget,
        "max_rows_per_expert": fit_rows,
        "eval_rows_per_expert": eval_rows,
        "render_mode": mode,
        "max_layers": max_l,
        "layers": layer_map,
    }


def _validate_streaming_layer_map(
    value: object, *, where: str
) -> dict[str, str]:
    """Validate a ``layers`` map of experts-module qname to snapshot hash."""
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{where}.layers must be a nonempty JSON object")
    layer_map: dict[str, str] = {}
    for qname, snapshot_hash in value.items():
        if not isinstance(qname, str) or not qname:
            raise ValueError(
                f"{where}.layers keys must be nonempty qnames"
            )
        digest = (
            str(snapshot_hash or "").strip().lower()
            if isinstance(snapshot_hash, str)
            else ""
        )
        if re.fullmatch(r"[0-9a-f]{32,64}", digest) is None:
            raise ValueError(
                f"{where}.layers[{qname!r}] must be a hex digest"
            )
        layer_map[str(qname)] = digest
    return _canonical_json_value(layer_map, where=f"{where}.layers")


def validate_packed_streaming_append_identity(
    value: object, *, where: str = "packed streaming append identity"
) -> dict[str, object]:
    """Validate a packed streaming append identity section.

    Every rule below is derived from the writer in
    :func:`build_packed_streaming_append_identity`: the exact field set,
    the shared budgets, and the per-layer snapshot hashes. Present but
    malformed is an error, never a silent skip.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    raw = dict(value)
    if raw.get("schema") != RENDER_IDENTITY_PACKED_STREAMING_APPEND_SCHEMA:
        raise ValueError(f"{where} has unsupported schema")
    if set(raw) != {
        "schema",
        "module_token_budget",
        "max_rows_per_expert",
        "eval_rows_per_expert",
        "render_mode",
        "max_layers",
        "layers",
    }:
        raise ValueError(f"{where} has unsupported fields")
    budget = raw.get("module_token_budget")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError(f"{where}.module_token_budget must be a positive int")
    fit_rows = raw.get("max_rows_per_expert")
    if (
        not isinstance(fit_rows, int)
        or isinstance(fit_rows, bool)
        or fit_rows < 1
    ):
        raise ValueError(
            f"{where}.max_rows_per_expert must be a positive int"
        )
    eval_rows = raw.get("eval_rows_per_expert")
    if (
        not isinstance(eval_rows, int)
        or isinstance(eval_rows, bool)
        or eval_rows < 1
    ):
        raise ValueError(
            f"{where}.eval_rows_per_expert must be a positive int"
        )
    if raw.get("render_mode") not in {"batched", "per_expert"}:
        raise ValueError(
            f"{where}.render_mode must be 'batched' or 'per_expert'"
        )
    layers_raw = raw.get("max_layers")
    max_l: int | None = None
    if layers_raw is not None:
        if (
            not isinstance(layers_raw, int)
            or isinstance(layers_raw, bool)
            or layers_raw < 1
        ):
            raise ValueError(f"{where}.max_layers must be a positive int")
        max_l = int(layers_raw)
    layer_map = _validate_streaming_layer_map(
        raw.get("layers"), where=where
    )
    return {
        "schema": RENDER_IDENTITY_PACKED_STREAMING_APPEND_SCHEMA,
        "module_token_budget": int(budget),
        "max_rows_per_expert": int(fit_rows),
        "eval_rows_per_expert": int(eval_rows),
        "render_mode": str(raw["render_mode"]),
        "max_layers": max_l,
        "layers": layer_map,
    }


def _refuse_append_mismatch(
    cache_dir: Path, section_key: str, field: str, cached: object,
    now: object,
) -> ValueError:
    return ValueError(
        f"production cache directory {cache_dir} {section_key} render "
        f"identity differs at {field!r}: "
        f"cached={identity_value_for_error(cached)} "
        f"current={identity_value_for_error(now)}; the directory holds "
        "appends from a different configuration — rebuild this directory "
        "(fresh --cache-dir) instead of appending to it"
    )


def _merge_packed_pair_map_or_refuse(
    cache_dir: Path,
    section_key: str,
    stored_section: Mapping[str, object],
    current: Mapping[str, object],
    stored_raw: dict[str, object],
    sidecar_path: Path,
    *,
    progress: bool = True,
) -> None:
    """Merge one resident packed append's per-pair fit map (#173).

    Shared fields (budgets, mode, gate corpus, hooked enumeration) compare
    exactly and refuse on the first difference. The per-pair fit map unions:
    pairs the directory has never seen are recorded, pairs already recorded
    under the same hash are idempotent, and the same pair under a different
    hash refuses — the bytes on disk are the first render's (existing shards
    are never re-rendered), so a second hash for the same pair names bytes
    that do not exist. The sanctioned M4 lazy gap-fill renders disjoint
    pairs under the render-split calib, so it merges without refusing.
    """
    stored_shared = {
        key: value
        for key, value in stored_section.items()
        if key != "pair_fit_calibration_hashes"
    }
    current_shared = {
        key: value
        for key, value in current.items()
        if key != "pair_fit_calibration_hashes"
    }
    difference = first_identity_difference(stored_shared, current_shared)
    if difference is not None:
        field, cached, now = difference
        raise _refuse_append_mismatch(
            cache_dir, section_key, field, cached, now
        )
    stored_pairs = dict(stored_section.get("pair_fit_calibration_hashes") or {})
    current_pairs = dict(current.get("pair_fit_calibration_hashes") or {})
    if not isinstance(stored_pairs, Mapping) or not isinstance(
        current_pairs, Mapping
    ):
        raise _refuse_append_mismatch(
            cache_dir, section_key, "pair_fit_calibration_hashes",
            stored_pairs, current_pairs,
        )
    for pair, fit_hash in current_pairs.items():
        cached_hash = stored_pairs.get(pair)
        if cached_hash is not None and cached_hash != fit_hash:
            raise _refuse_append_mismatch(
                cache_dir, section_key,
                f"pair_fit_calibration_hashes[{pair!r}]",
                cached_hash, fit_hash,
            )
    if all(
        stored_pairs.get(pair) == fit_hash
        for pair, fit_hash in current_pairs.items()
    ):
        if progress:
            print(
                f"[prod-cache] append resume: {section_key} render identity "
                "matches; reusing on-disk shards",
                flush=True,
            )
        return
    merged_section = dict(stored_section)
    merged_section["pair_fit_calibration_hashes"] = {
        **stored_pairs, **current_pairs,
    }
    merged = dict(stored_raw)
    merged[section_key] = merged_section
    atomic_write_bytes(
        sidecar_path,
        json.dumps(merged, indent=2, sort_keys=True).encode("utf-8"),
    )


def _merge_streaming_layer_map_or_refuse(
    cache_dir: Path,
    section_key: str,
    stored_section: Mapping[str, object],
    current: Mapping[str, object],
    stored_raw: dict[str, object],
    sidecar_path: Path,
    *,
    progress: bool = True,
) -> None:
    """Merge one streaming packed append's per-layer map (#172).

    Shared budgets/mode compare exactly and refuse on the first difference,
    so layers appended under different budgets cannot land in one directory.
    The per-layer map unions keyed by experts-module qname: unseen modules
    are recorded, a re-appended module under the same snapshot hash is
    idempotent, and the same module under a different snapshot hash refuses.
    """
    stored_shared = {
        key: value
        for key, value in stored_section.items()
        if key != "layers"
    }
    current_shared = {
        key: value for key, value in current.items() if key != "layers"
    }
    difference = first_identity_difference(stored_shared, current_shared)
    if difference is not None:
        field, cached, now = difference
        raise _refuse_append_mismatch(
            cache_dir, section_key, field, cached, now
        )
    stored_layers = dict(stored_section.get("layers") or {})
    current_layers = dict(current.get("layers") or {})
    if not isinstance(stored_layers, Mapping) or not isinstance(
        current_layers, Mapping
    ):
        raise _refuse_append_mismatch(
            cache_dir, section_key, "layers", stored_layers, current_layers,
        )
    for qname, snapshot_hash in current_layers.items():
        cached_hash = stored_layers.get(qname)
        if cached_hash is not None and cached_hash != snapshot_hash:
            raise _refuse_append_mismatch(
                cache_dir, section_key, f"layers[{qname!r}]",
                cached_hash, snapshot_hash,
            )
    if all(
        stored_layers.get(qname) == snapshot_hash
        for qname, snapshot_hash in current_layers.items()
    ):
        if progress:
            print(
                f"[prod-cache] append resume: {section_key} render identity "
                "matches; reusing on-disk shards",
                flush=True,
            )
        return
    merged_section = dict(stored_section)
    merged_section["layers"] = {**stored_layers, **current_layers}
    merged = dict(stored_raw)
    merged[section_key] = merged_section
    atomic_write_bytes(
        sidecar_path,
        json.dumps(merged, indent=2, sort_keys=True).encode("utf-8"),
    )


def _check_and_record_append_identity(
    cache_dir_path: str | Path,
    section_key: str,
    current_section: Mapping[str, object],
    *,
    progress: bool = True,
) -> None:
    """Enforce one append configuration per cache directory (#170).

    The base ``render_identity.json`` guard (#146) covers only
    ``fill_production_weight_cache``; the MTP and packed-expert appends
    stream further shards into the same directory after that fill returns,
    under their own value-bearing inputs (budgets, gate corpus, activation
    source). Each append compare-or-writes its own section of the sidecar:
    the first append records it, every later append compares and refuses on
    the first differing field. A directory whose sidecar predates the section
    gains it without disturbing the base identity; a directory with no
    sidecar at all records an append-only file the base fill later adopts.

    Two sections merge rather than compare whole (#172, #173): the resident
    packed section unions its per-pair fit map (disjoint pairs merge, the
    same pair under a different hash refuses), and the streaming section
    unions its per-layer map (new layers merge, the same module under a
    different budget or snapshot refuses).

    A ``pre_guard_admission`` record (#146), when present, is carried
    forward untouched by the section merge and by that later adoption; an
    append running first in a pre-guard directory records it the same way
    the base fill does.
    """
    if section_key == MTP_APPEND_SIDECAR_KEY:
        current = validate_mtp_append_identity(
            _canonical_json_value(
                dict(current_section),
                where="MTP append identity",
            ),
            where="MTP append identity",
        )
    elif section_key == PACKED_APPEND_SIDECAR_KEY:
        current = validate_packed_append_identity(
            _canonical_json_value(
                dict(current_section),
                where="packed-expert append identity",
            ),
            where="packed-expert append identity",
        )
    elif section_key == PACKED_STREAMING_APPEND_SIDECAR_KEY:
        current = validate_packed_streaming_append_identity(
            _canonical_json_value(
                dict(current_section),
                where="packed streaming append identity",
            ),
            where="packed streaming append identity",
        )
    else:
        raise ValueError(
            f"unknown production cache append identity section {section_key!r}"
        )
    cache_dir = Path(cache_dir_path)
    sidecar_path = cache_dir / RENDER_IDENTITY_SIDECAR_FILENAME
    if not sidecar_path.is_file():
        payload: dict[str, object] = {section_key: current}
        trusted = _pre_guard_trusted_shards(cache_dir)
        if trusted:
            # An append running first in a pre-guard directory meets the
            # same shards-on-trust admission as the base fill; the record
            # goes in now so the base fill's later adoption carries it
            # forward instead of laundering it.
            if progress:
                print(
                    "[prod-cache] WARNING: cache directory "
                    f"{cache_dir} holds rendered shards but no "
                    f"{RENDER_IDENTITY_SIDECAR_FILENAME} (pre-guard "
                    "directory); admitting existing shards on trust and "
                    f"recording the {PRE_GUARD_ADMISSION_SIDECAR_KEY} "
                    f"alongside the {section_key} section.",
                    flush=True,
                )
            payload[PRE_GUARD_ADMISSION_SIDECAR_KEY] = (
                build_pre_guard_admission(trusted)
            )
        atomic_write_bytes(
            sidecar_path,
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        )
        return
    try:
        stored_raw = json.loads(sidecar_path.read_bytes().decode("utf-8"))
    except Exception as exc:
        raise ValueError(
            f"production cache directory {cache_dir} has an unreadable "
            f"{RENDER_IDENTITY_SIDECAR_FILENAME}: {exc}; refusing append — "
            "rebuild this directory (fresh --cache-dir) instead of mixing "
            "renders"
        ) from exc
    if not isinstance(stored_raw, Mapping):
        raise ValueError(
            f"production cache directory {cache_dir} has an invalid "
            f"{RENDER_IDENTITY_SIDECAR_FILENAME}: expected a JSON object; "
            "refusing append — rebuild this directory (fresh --cache-dir) "
            "instead of mixing renders"
        )
    stored_raw = dict(stored_raw)
    if (
        stored_raw.get("schema") != RENDER_IDENTITY_SCHEMA
        and MTP_APPEND_SIDECAR_KEY not in stored_raw
        and PACKED_APPEND_SIDECAR_KEY not in stored_raw
        and PACKED_STREAMING_APPEND_SIDECAR_KEY not in stored_raw
    ):
        raise ValueError(
            f"production cache directory {cache_dir} has an unrecognized "
            f"{RENDER_IDENTITY_SIDECAR_FILENAME}; refusing append — rebuild "
            "this directory (fresh --cache-dir) instead of mixing renders"
        )
    stored_section_raw = stored_raw.get(section_key)
    if stored_section_raw is None:
        merged = dict(stored_raw)
        merged[section_key] = current
        atomic_write_bytes(
            sidecar_path,
            json.dumps(merged, indent=2, sort_keys=True).encode("utf-8"),
        )
        return
    try:
        if section_key == MTP_APPEND_SIDECAR_KEY:
            stored_section = validate_mtp_append_identity(
                stored_section_raw,
                where=(
                    f"cache directory {cache_dir} MTP append identity"
                ),
            )
        elif section_key == PACKED_STREAMING_APPEND_SIDECAR_KEY:
            stored_section = validate_packed_streaming_append_identity(
                stored_section_raw,
                where=(
                    f"cache directory {cache_dir} packed streaming append "
                    "identity"
                ),
            )
        else:
            stored_section = validate_packed_append_identity(
                stored_section_raw,
                where=(
                    f"cache directory {cache_dir} packed-expert append "
                    "identity"
                ),
            )
    except ValueError as exc:
        raise ValueError(
            f"production cache directory {cache_dir} has an invalid "
            f"{RENDER_IDENTITY_SIDECAR_FILENAME} section {section_key!r}: "
            f"{exc}; refusing append — rebuild this directory (fresh "
            "--cache-dir) instead of appending to it"
        ) from exc
    if section_key == PACKED_APPEND_SIDECAR_KEY:
        _merge_packed_pair_map_or_refuse(
            cache_dir, section_key, stored_section, current, stored_raw,
            sidecar_path, progress=progress,
        )
        return
    if section_key == PACKED_STREAMING_APPEND_SIDECAR_KEY:
        _merge_streaming_layer_map_or_refuse(
            cache_dir, section_key, stored_section, current, stored_raw,
            sidecar_path, progress=progress,
        )
        return
    difference = first_identity_difference(stored_section, current)
    if difference is not None:
        field, cached, now = difference
        raise ValueError(
            f"production cache directory {cache_dir} {section_key} render "
            f"identity differs at {field!r}: "
            f"cached={identity_value_for_error(cached)} "
            f"current={identity_value_for_error(now)}; the directory holds "
            "appends from a different configuration — rebuild this directory "
            "(fresh --cache-dir) instead of appending to it"
        )
    if progress:
        print(
            f"[prod-cache] append resume: {section_key} render identity "
            "matches; reusing on-disk shards",
            flush=True,
        )


def _check_production_cache_render_identity(
    cache_dir_path: Path,
    current: Mapping[str, object],
    *,
    progress: bool = True,
) -> None:
    """Enforce one rendering per cache directory (#146, extended by #170).

    The resume loop admits a unit as done by file presence alone, so without
    this gate a directory resumed under a different scope, include-file,
    lever string or calibration silently mixes units rendered under different
    conditions. The first fill writes its render identity into
    ``render_identity.json``; every later fill compares and refuses on the
    first differing field. A missing sidecar means a pre-guard directory:
    warn and admit its shards on trust (refusing would strand existing work),
    then write the current identity so later mismatches refuse. The admission
    itself is recorded in the sidecar as ``pre_guard_admission`` — the
    trusted shard count plus their sorted shard names — so a downstream
    artifact can read that part of the directory rests on unverified shards.
    A fresh directory (no shards) carries no such record.

    Append sections (#170, #172) are compared by their own writers, never
    here: the stored base projection is compared against the current base
    identity while recorded ``mtp_append`` / ``packed_expert_append`` /
    ``packed_expert_streaming_append`` sections are preserved untouched. A
    sidecar that holds only append sections (an append ran before any base
    fill) is adopted the same way: the current base identity is merged in
    alongside the validated sections. The trust admission is history rather
    than render input, so it is carried forward by the adoption instead of
    compared: a guarded directory carrying it still resumes cleanly against
    a caller identity computed without it.
    """
    current_identity = validate_production_cache_render_identity(
        _canonical_json_value(dict(current), where="production cache render identity"),
        where="production cache render identity",
    )
    # The base fill owns the base fields only; a caller-supplied append
    # section is never its to write.
    current_identity = {
        key: value
        for key, value in current_identity.items()
        if key not in (
            MTP_APPEND_SIDECAR_KEY,
            PACKED_APPEND_SIDECAR_KEY,
            PACKED_STREAMING_APPEND_SIDECAR_KEY,
        )
    }
    sidecar_path = cache_dir_path / RENDER_IDENTITY_SIDECAR_FILENAME
    if not sidecar_path.is_file():
        trusted_shards = _pre_guard_trusted_shards(cache_dir_path)
        if trusted_shards:
            print(
                "[prod-cache] WARNING: cache directory "
                f"{cache_dir_path} holds rendered shards but no "
                f"{RENDER_IDENTITY_SIDECAR_FILENAME} (pre-guard directory); "
                "admitting existing shards on trust — confirm the directory "
                "was rendered under one configuration, then resume only with "
                "identical settings. Writing the current render identity "
                f"with a {PRE_GUARD_ADMISSION_SIDECAR_KEY} record naming "
                "the trusted shards; future mismatches refuse.",
                flush=True,
            )
            current_identity = {
                **current_identity,
                PRE_GUARD_ADMISSION_SIDECAR_KEY: build_pre_guard_admission(
                    trusted_shards
                ),
            }
        atomic_write_bytes(
            sidecar_path,
            json.dumps(
                current_identity, indent=2, sort_keys=True
            ).encode("utf-8"),
        )
        return
    try:
        stored_raw = json.loads(sidecar_path.read_bytes().decode("utf-8"))
    except Exception as exc:
        raise ValueError(
            f"production cache directory {cache_dir_path} has an unreadable "
            f"{RENDER_IDENTITY_SIDECAR_FILENAME}: {exc}; refusing resume — "
            "rebuild this directory (fresh --cache-dir) instead of mixing "
            "renders"
        ) from exc
    if not isinstance(stored_raw, Mapping):
        raise ValueError(
            f"production cache directory {cache_dir_path} has an invalid "
            f"{RENDER_IDENTITY_SIDECAR_FILENAME}: expected a JSON object; "
            "refusing resume — rebuild this directory (fresh --cache-dir) "
            "instead of mixing renders"
        )
    stored_raw = dict(stored_raw)
    base_fields = set(current_identity)
    stored_base_raw = {
        key: stored_raw[key] for key in base_fields if key in stored_raw
    }
    stored_sections: dict[str, object] = {}
    for key in (
        MTP_APPEND_SIDECAR_KEY,
        PACKED_APPEND_SIDECAR_KEY,
        PACKED_STREAMING_APPEND_SIDECAR_KEY,
    ):
        if key in stored_raw:
            stored_sections[key] = stored_raw[key]
    if set(stored_base_raw) != base_fields:
        # No base identity on file. An append-only sidecar (an append ran
        # before any base fill, so the file holds sections and no base
        # fields at all) is adopted: keep its validated sections and record
        # the current base identity alongside them. A file with some — but
        # not all — base fields is neither a valid sidecar nor an
        # append-only one, so it refuses fail-closed like any other invalid
        # sidecar.
        unknown = (
            set(stored_raw) - base_fields
            - {
                MTP_APPEND_SIDECAR_KEY,
                PACKED_APPEND_SIDECAR_KEY,
                PACKED_STREAMING_APPEND_SIDECAR_KEY,
            }
            - {PRE_GUARD_ADMISSION_SIDECAR_KEY}
        )
        if not stored_base_raw and stored_sections and not unknown:
            try:
                adopted = validate_production_cache_render_identity(
                    {**stored_raw, **current_identity},
                    where=(
                        f"cache directory {cache_dir_path} render identity"
                    ),
                )
            except ValueError as exc:
                raise ValueError(
                    f"production cache directory {cache_dir_path} has an "
                    f"invalid {RENDER_IDENTITY_SIDECAR_FILENAME}: {exc}; "
                    "refusing resume — rebuild this directory (fresh "
                    "--cache-dir) instead of mixing renders"
                ) from exc
            print(
                "[prod-cache] WARNING: cache directory "
                f"{cache_dir_path} holds append sections but no base render "
                "identity (append ran before any base fill); adopting the "
                "recorded append configuration and writing the current base "
                "identity — resume only with identical settings.",
                flush=True,
            )
            atomic_write_bytes(
                sidecar_path,
                json.dumps(adopted, indent=2, sort_keys=True).encode("utf-8"),
            )
            return
        raise ValueError(
            f"production cache directory {cache_dir_path} has an invalid "
            f"{RENDER_IDENTITY_SIDECAR_FILENAME}: missing base render "
            "identity fields; refusing resume — rebuild this directory "
            "(fresh --cache-dir) instead of mixing renders"
        )
    try:
        stored = validate_production_cache_render_identity(
            stored_raw,
            where=f"cache directory {cache_dir_path} render identity",
        )
    except ValueError as exc:
        raise ValueError(
            f"production cache directory {cache_dir_path} has an invalid "
            f"{RENDER_IDENTITY_SIDECAR_FILENAME}: {exc}; refusing resume — "
            "rebuild this directory (fresh --cache-dir) instead of mixing "
            "renders"
        ) from exc
    stored_base = {
        key: stored[key] for key in base_fields
    }
    difference = first_identity_difference(stored_base, current_identity)
    if difference is not None:
        field, cached, now = difference
        raise ValueError(
            f"production cache directory {cache_dir_path} render identity "
            f"differs at {field!r}: cached={identity_value_for_error(cached)} "
            f"current={identity_value_for_error(now)}; the directory holds "
            "renders from a different configuration — rebuild this directory "
            "(fresh --cache-dir) instead of resuming it"
        )
    if progress:
        print(
            "[prod-cache] resume: render identity matches "
            f"({len(current_identity['rendered_pairs'])} pairs); "
            "reusing on-disk shards",
            flush=True,
        )


def fill_production_weight_cache(
    model: nn.Module,
    calib_ids: torch.Tensor,
    qnames: Sequence[str],
    *,
    formats: Sequence[str] = ("NVFP4",),
    render_assignment: Mapping[str, str] | None = None,
    render_qnames: Iterable[str] | None = None,
    levers: Mapping[str, bool] | None = None,
    max_act_rows: int = 256,
    progress: bool = True,
    cache_dir: str | Path | None = None,
    recache_pass: bool = False,
    recache_assignment: Mapping[str, str] | None = None,
    recache_profile=None,
    recache_include_activation_quant: bool = True,
    recache_microbatch_size: int = 1,
    h_detail_dir: str | Path | None = None,
    col_weights: Mapping[str, torch.Tensor] | None = None,
) -> ProductionWeightCache:
    """End-to-end fill: collect activations, render production δw per
    (qname, fmt), return a `ProductionWeightCache`.

    Args:
      model: live HF model on the export device.
      calib_ids: ``[N, T]`` token id tensor for activation collection.
      qnames: which Linears are eligible to render (skips MoE packed
        experts; handle those separately via `_quantize_3d_packed`
        extensions). This is also the set the activation collector HOOKS,
        and the rendered bytes are a function of it: pass the whole
        enumeration and narrow with `render_assignment`/`render_qnames`, never
        by shortening this list (#130).
      formats: which formats to pre-render when `render_assignment` is not
        supplied.
      render_assignment: optional concrete export assignment. When supplied,
        render exactly the non-BF16 `(qname, fmt)` entries used by that
        assignment instead of the full `qnames x formats` menu.
      render_qnames: optional render-only narrowing, orthogonal to
        `render_assignment`. Render just these qnames (a stripe, a shard) while
        still hooking all of `qnames`, so the shard's bytes are the whole
        run's bytes. `None` renders everything in scope.
      levers: which production levers to enable (default: GPTQ with optional
        joint NVFP4 scale optimization when requested by the caller).
      recache_pass: when True, run a second calibration forward with the
        concrete production assignment installed from this cache and refit
        ``activation_max_abs`` under quantized upstream weights.
      recache_assignment: required when ``recache_pass`` is True.  Candidate
        caches with multiple possible formats per Linear are ambiguous; recache
        needs the actual export assignment.
      h_detail_dir: optional probe h-detail directory retained for archived
        Fisher ablations. V1 production defaults do not require it.
      col_weights: optional ``{qname: (in_features,)}`` imatrix map (re-vet
        R3). Applied only to the weighted-render families
        (``WEIGHTED_RENDER_FAMILIES``); every other format's rendered bytes are
        bit-identical whether it is supplied or not. This lets a weighted
        lane's allocator cost, frontier KL and shipped bytes all come from ONE
        ``ProductionWeightCache`` render instead of a separate
        skeleton-requantize path.
    """
    if recache_pass and not recache_assignment:
        raise ValueError(
            "recache_pass=True requires recache_assignment with the concrete "
            "production assignment"
        )
    levers = _resolve_production_render_levers(levers)
    mechanism_plan = _resolve_render_mechanism_plan(levers)
    if progress and mechanism_plan.ordered:
        print(
            "[prod-cache] render mechanism order: "
            + " -> ".join(spec.name for spec in mechanism_plan.ordered),
            flush=True,
        )

    from prismaquant import format_registry as fr
    from prismaquant.schemas import refuse_retired_codebook_format

    def _canon(fmt: str) -> str:
        canonical = fr.canonical_format_name(str(fmt).strip().upper())
        # The render loop records a render failure and moves on, and resumes
        # an existing shard without rendering. Neither may absorb a retired
        # codebook rung (archived 2026-09-25, #1304), so it refuses here.
        refuse_retired_codebook_format(canonical)
        return canonical

    requested_formats = tuple(
        dict.fromkeys(_canon(f) for f in formats if str(f).strip())
    )
    # #130/#135: ``eligible_qnames`` is the enumeration the activation
    # collector hooks, and it is deliberately NOT narrowed by anything below.
    # One shared generator feeds every hooked Linear's priority reservoir, so
    # the rows a Linear keeps are a function of how many rows every EARLIER
    # hook consumed. Narrowing the hook set therefore changes the rows, the
    # GPTQ Hessian and the rendered bytes. The invariant this buys: the bytes
    # of a (qname, fmt) pair depend on ``qnames`` and the calibration, never on
    # which subset of them this call happens to render. That is what makes a
    # stripe, an assignment scope and a resume all reproduce the whole run.
    eligible_qnames = set(qnames)
    render_scope_qnames: set[str] | None = (
        {str(q) for q in render_qnames} if render_qnames is not None else None
    )
    if render_assignment is not None:
        render_formats_by_qname: dict[str, tuple[str, ...]] = {}
        for qname, fmt in render_assignment.items():
            q = str(qname)
            if q not in eligible_qnames:
                continue
            if render_scope_qnames is not None and q not in render_scope_qnames:
                continue
            fmt_canon = _canon(fmt)
            if fmt_canon == "BF16":
                continue
            render_formats_by_qname[q] = (fmt_canon,)
        qname_set = set(render_formats_by_qname)
        render_scope = "assignment"
    else:
        non_bf16_formats = tuple(
            f for f in requested_formats if f != "BF16"
        )
        render_formats_by_qname = {
            q: non_bf16_formats
            for q in eligible_qnames
            if render_scope_qnames is None or q in render_scope_qnames
        }
        qname_set = {
            q for q, fmts in render_formats_by_qname.items() if fmts
        }
        render_scope = "format-menu"

    if not qname_set:
        return ProductionWeightCache(
            weights={},
            levers=dict(levers),
            metadata={
                "render_scope": render_scope,
                "requested_formats": list(requested_formats),
                "requested_entries": 0,
            },
        )
    model_profile = recache_profile
    if model_profile is None:
        from .model_profiles import (
            DeadVendoredOverrideError,
            profile_from_model,
        )
        try:
            model_profile = profile_from_model(model)
        except DeadVendoredOverrideError:
            # This is the render that produces the production weight cache --
            # the bytes an export later ships. A dead override means those
            # bytes would be rendered from UPSTREAM modelling code, and
            # `model_profile = None` also drops the profile's pinned names, so
            # components the profile forbids quantizing get quantized (#202).
            raise
        except Exception:
            model_profile = None

    if progress:
        requested_entries = sum(
            len(fmts) for fmts in render_formats_by_qname.values()
        )
        print(f"[prod-cache] levers={dict(sorted(levers.items()))}", flush=True)
        print(
            f"[prod-cache] render_scope={render_scope} "
            f"qnames={len(qname_set)} entries={requested_entries}",
            flush=True,
        )

    # RESUME: when disk-streaming is on and prior shards exist, only
    # collect activations for Linears whose shards we still need to
    # render.  On a job that's 99%+ complete this drops activation
    # collection memory + compute by 99% — and lets a borderline-OOM
    # job finish on the same hardware.
    cache_dir_path: Path | None = None
    if cache_dir is not None:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        from prismaquant.perturbed_x_cache import calibration_data_hash

        # #146: resume admits a unit as done by file presence alone, so a
        # directory resumed under a different scope, include-file, lever
        # string or calibration would silently mix units rendered under
        # different conditions. Compare this call's render identity against
        # the sidecar before any shard is read or written; refuse on mismatch.
        _check_production_cache_render_identity(
            cache_dir_path,
            build_production_cache_render_identity(
                render_scope=render_scope,
                requested_formats=requested_formats,
                levers=levers,
                mechanism_plan=mechanism_plan,
                calib_hash=calibration_data_hash(calib_ids),
                eligible_qnames=eligible_qnames,
                render_formats_by_qname=render_formats_by_qname,
                max_act_rows=max_act_rows,
            ),
            progress=progress,
        )
    render_score_sidecar_path: Path | None = (
        cache_dir_path / "render_scores.json"
        if cache_dir_path is not None else None
    )
    render_score_records: dict[str, dict[str, object]] = (
        _load_render_score_sidecar(render_score_sidecar_path)
    )
    # Every retained cost enters the fill here -- the disk-resume branch below
    # keeps them by key, and a fully-rendered directory never reaches that
    # branch at all -- so this is the one place to ask whether they were
    # priced under the activation-scale policy this run applies (#227).  A
    # directory whose render identity was compared already refused on
    # ``levers.nvfp4_input_global_scale_policy``; this still holds for the
    # pre-guard directory admitted on trust, for a sidecar restored beside a
    # fresh identity, and for in-memory callers.
    render_score_policy = _render_levers_input_global_scale_policy(levers)
    _check_resumed_render_score_policies(
        render_score_records,
        policy=render_score_policy,
        where=(
            "production cache resume"
            if cache_dir_path is None
            else f"production cache resume @ {cache_dir_path}"
        ),
    )
    if progress and render_score_records:
        print(
            f"[prod-cache] resume: loaded {len(render_score_records)} "
            "render-score entries from sidecar",
            flush=True,
        )

    fmt_set = {
        fmt
        for fmts in render_formats_by_qname.values()
        for fmt in fmts
    }
    render_base_fmt_set = {_render_base_format(fmt) for fmt in fmt_set}
    # Whether any requested format is served under a calibrated STATIC
    # activation scale, so the fill must measure per-unit max|x| (the scale
    # identity its records retain and the assignment-KL hooks price with).
    # Read from the specs, not from ``"NVFP4" in`` the set: a Tessera W4A4
    # rung is served under the same contract with a different name (#205).
    needs_static_activation_max = _formats_need_static_activation_max(
        render_base_fmt_set)
    # Store activations for every missing rendered format.  NVFP4 needs them
    # for GPTQ/JSO; FP8_DYNAMIC/FP8_E4M3 and explicit MX formats need them
    # for their activation-aware renders, and the production-render allocator
    # cost always needs them to score the final local forward error after the
    # format's activation quantizer.
    activation_aware_formats = set(fmt_set)
    qnames_to_render: set[str] = set(qname_set)
    missing_formats_by_qname: dict[str, set[str]] = {
        q: set(render_formats_by_qname.get(q, ())) for q in qname_set
    }
    if cache_dir_path is not None:
        # A qname is FULLY done if every requested format has a shard.
        prerendered = 0
        for q in list(qname_set):
            missing = {
                f for f in render_formats_by_qname.get(q, ())
                if not (cache_dir_path / _cache_weight_filename(q, f)).is_file()
            }
            missing_formats_by_qname[q] = missing
            if not missing:
                qnames_to_render.discard(q)
                prerendered += 1
        if progress and prerendered:
            print(
                f"[prod-cache] resume: {prerendered} qnames already on disk "
                f"({len(qnames_to_render)} still need rendering)",
                flush=True,
            )
    qnames_needing_activation = set()
    for q, missing in missing_formats_by_qname.items():
        requested = tuple(render_formats_by_qname.get(q, ()))
        missing_activation_render = any(
            f in activation_aware_formats for f in missing
        )
        missing_activation_score = any(
            f in activation_aware_formats
            and _render_score_record_key(q, f) not in render_score_records
            for f in requested
        )
        if missing_activation_render or missing_activation_score:
            qnames_needing_activation.add(q)
    device = next(model.parameters()).device
    activation_store_device = (
        device if device.type == "cuda" else torch.device("cpu")
    )
    activation_store_dtype = torch.float32
    if progress and qnames_needing_activation:
        print(
            f"[prod-cache] activation_capture "
            f"store_device={activation_store_device} "
            f"store_dtype={activation_store_dtype} "
            f"qnames={len(qnames_needing_activation)}",
            flush=True,
        )
    # RESUME: if all qnames are already rendered AND we have either a
    # sidecar OR no need for max_abs (no NVFP4 in formats), skip the
    # forward pass entirely.  Avoids OOM from the model's forward pass
    # itself on big models (e.g. linear-attention torch fallback can
    # spike memory mid-pass on Qwen3.5/3.6 27B+).
    sidecar_path: Path | None = (
        cache_dir_path / "activation_max_abs.json"
        if cache_dir_path is not None else None
    )
    skip_forward = (
        cache_dir_path is not None
        and not qnames_needing_activation
        and (
            (sidecar_path is not None and sidecar_path.is_file())
            or not needs_static_activation_max
        )
    )
    collector = None  # may stay None on the skip_forward path
    if skip_forward:
        if progress:
            print(
                "[prod-cache] resume: all qnames pre-rendered + max_abs "
                "available, skipping activation forward pass",
                flush=True,
            )
        activations: dict[str, torch.Tensor] = {}
    else:
        # Hook the WHOLE eligible enumeration -- not the render-narrowed
        # ``qname_set``. Hooking is cheap (an amax plus one CPU
        # ``torch.rand`` per call), and hooking the full set is what keeps the
        # shared priority stream identical across a stripe, an assignment
        # scope and a resume (#130/#135). Only STORE full activations for
        # Linears we still need to render.
        collector = _LinearActivationCollector(
            model,
            qnames=eligible_qnames,
            max_rows=max_act_rows,
            store_qnames=qnames_needing_activation,
            store_device=activation_store_device,
            store_dtype=activation_store_dtype,
            profile=model_profile,
        )
        collector.install()
        try:
            with torch.no_grad():
                for i in range(calib_ids.size(0)):
                    batch = calib_ids[i:i + 1].to(device)
                    try:
                        model(batch, use_cache=False)
                    except TypeError:
                        # Some non-HF or older model wrappers do not expose
                        # use_cache. The cache is only an inference speed
                        # feature; activation collection is still correct
                        # without the explicit flag on those models.
                        model(batch)
        finally:
            collector.remove()
        activations = collector.collected()

    if device.type == "cuda" and qnames_needing_activation:
        cpu_activations = [
            name for name, acts in activations.items()
            if acts.device.type != "cuda"
        ]
        if cpu_activations:
            raise RuntimeError(
                "production cache captured non-CUDA activations for "
                f"{len(cpu_activations)} Linears; sample={cpu_activations[:3]}"
            )

    if progress:
        activation_bytes = sum(
            int(t.numel()) * int(t.element_size())
            for t in activations.values()
        )
        activation_devices = sorted({str(t.device) for t in activations.values()})
        print(
            f"[prod-cache] collected activations for "
            f"{len(activations)} of {len(qname_set)} rendered "
            f"({len(eligible_qnames)} hooked) Linears "
            f"resident_bytes={activation_bytes:,} "
            f"devices={activation_devices}",
            flush=True,
        )

    weights: dict[tuple[str, str], object] = {}
    failed: dict[tuple[str, str], str] = {}
    qname_to_module: dict[str, nn.Module] = {}

    if cache_dir_path is not None and progress:
        print(f"[prod-cache] streaming cache to {cache_dir_path}/", flush=True)

    for full_name, mod, attr in iter_quantizable_tensors(model, model_profile):
        if attr != "weight" or not isinstance(mod, nn.Linear):
            continue
        qname = full_name[:-7] if full_name.endswith(".weight") else full_name
        if qname in qname_set:
            qname_to_module[qname] = mod

    fused_sibling_mapping = (
        _fused_sibling_leaf_mapping_from_profile(model_profile)
        if model_profile is not None
        else {}
    )
    fisher_rows = (
        _FisherRowWeightCache(
            h_detail_dir,
            fused_sibling_mapping or None,
        )
        if (bool(levers.get("fisher_gptq", False)) and h_detail_dir)
        else None
    )
    if progress and bool(levers.get("fisher_gptq", False)):
        if fisher_rows is None:
            print(
                "[prod-cache] Fisher weighting requested but no h_detail_dir "
                "was provided; falling back to unweighted objectives",
                flush=True,
            )
        else:
            print(
                f"[prod-cache] Fisher weighting using h-detail dir "
                f"{fisher_rows.detail_dir}",
                flush=True,
            )

    # HIGH-1: compute joint NVFP4 fused-sibling globals so q/k/v share a
    # per-tensor scale (and gate/up likewise), matching the export's
    # `_compute_nvfp4_joint_global` behavior.  Without this each sibling
    # gets its own scale and vLLM's loader either rejects the artifact or
    # silently runs with degraded accuracy.
    joint_globals: dict[str, torch.Tensor] = {}
    needs_nvfp4_render = any(
        any(_render_base_format(fmt) == "NVFP4" for fmt in missing)
        for missing in missing_formats_by_qname.values()
    )
    if needs_nvfp4_render:
        from prismaquant.export_native_compressed import (
            _compute_nvfp4_joint_global,
        )
        synthetic_assignment = {q: "NVFP4" for q in qname_to_module}
        joint_globals = _compute_nvfp4_joint_global(
            model,
            synthetic_assignment,
            profile=model_profile,
        )
        if progress:
            print(
                f"[prod-cache] computed joint NVFP4 globals for "
                f"{len(joint_globals)} fused-sibling members",
                flush=True,
            )

    # MED-3: per-Linear calibrated max_abs used by the export's act-clip
    # step.  For fused-sibling groups the value is unified (max across
    # siblings), matching the export's joint input_global_scale derivation.
    # We store max_abs directly (not 6/max_abs) — see ProductionWeightCache
    # docstring on the convention difference.
    activation_max_abs: dict[str, float] = {}

    # RESUME: load previously-computed max_abs values from the sidecar
    # JSON if disk-streaming + sidecar exists.  Lets a resume run skip
    # both activation collection and max_abs recomputation for already-
    # rendered qnames.  ``sidecar_path`` was defined earlier (before the
    # forward-skip decision); re-using it here.
    if sidecar_path is not None and sidecar_path.is_file():
        activation_max_abs.update(
            _load_activation_max_abs_sidecar(sidecar_path)
        )
        if progress:
            print(
                f"[prod-cache] resume: loaded {len(activation_max_abs)} "
                f"max_abs entries from sidecar",
                flush=True,
            )

    if needs_static_activation_max:
        # Group by fused sibling key for max-across-siblings unification.
        from prismaquant.decision_units import fused_group_key

        per_qname_max_abs: dict[str, float] = {}
        for qname, _ in qname_to_module.items():
            # 1. Sidecar (resume) wins — these are the pre-computed values
            #    from a prior run.
            if qname in activation_max_abs:
                per_qname_max_abs[qname] = activation_max_abs[qname]
                continue
            # 2. Collector's per-Linear scalar (always populated for
            #    Linears that were hooked, even if no full activation
            #    tensor was stored).  ``collector`` is None on the
            #    skip_forward path, in which case we can only fall
            #    through to the activations-tensor path (which is
            #    empty on skip_forward, so we just continue).
            mx = (
                collector.max_abs.get(qname, 0.0)
                if collector is not None else 0.0
            )
            if mx <= 0:
                a = activations.get(qname)
                if a is None:
                    continue
                mx = float(a.abs().max().item())
            if mx <= 0:
                continue
            per_qname_max_abs[qname] = mx

        # Unify across fused sibling groups by taking the max.
        groups: dict[str, list[str]] = {}
        for qname in per_qname_max_abs:
            gk = (
                fused_group_key(model_profile, qname)
                if model_profile is not None else qname
            )
            groups.setdefault(gk, []).append(qname)
        for gk, members in groups.items():
            shared = max(per_qname_max_abs[m] for m in members)
            for m in members:
                activation_max_abs[m] = shared
        if progress and activation_max_abs:
            print(
                f"[prod-cache] computed activation max_abs for "
                f"{len(activation_max_abs)} Linears "
                f"({len(groups)} fused groups)",
                flush=True,
            )
        # Persist max_abs to sidecar so future resume runs can skip
        # activation collection entirely for completed qnames.
        if sidecar_path is not None and activation_max_abs:
            atomic_write_bytes(
                sidecar_path,
                json.dumps(activation_max_abs, indent=2).encode("utf-8"),
            )

    # MEM: free per-Linear activation tensors after each render so peak
    # memory stays bounded.  On 27B, 497 Linears × ~10K in_features × 512
    # rows × fp32 = ~10 GB just for activations.  Freeing in-loop drops
    # this to ~20 MB resident.
    import gc as _gc
    activations_local = dict(activations)  # shallow copy; we'll pop entries
    n = sum(len(render_formats_by_qname.get(q, ())) for q in qname_to_module)
    done = 0
    skipped_resumed = 0
    skipped_prewritten = 0
    render_gate_records: list[dict[str, object]] = []
    for qname, mod in qname_to_module.items():
        weight = mod.weight.data
        joint = joint_globals.get(qname)
        max_abs = activation_max_abs.get(qname)
        row_weights = (
            fisher_rows.get(qname)
            if bool(levers.get("fisher_gptq", False))
            and fisher_rows is not None
            else None
        )
        # _quantize_2d's input_global_scale_override expects the export
        # convention.  It only affects emitted metadata in compute_only
        # mode (not the dequantized weight values), but we pass the
        # correct convention so the metadata is honest in case future
        # code consumes it.  Routed through the export helper so it
        # tracks PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE (audit C1).
        from prismaquant.export_native_compressed import (
            _nvfp4_input_global_scale_from_max_abs,
        )
        export_scale = (
            _nvfp4_input_global_scale_from_max_abs(
                float(max_abs), policy=render_score_policy)
            if (max_abs is not None and max_abs > 0)
            else None
        )
        for fmt in render_formats_by_qname.get(qname, ()):
            fmt_key = str(fmt).upper()
            render_fmt = _render_base_format(fmt_key)
            key = (qname, fmt_key)
            if key in weights:
                skipped_prewritten += 1
                done += 1
                if progress and done % 25 == 0:
                    print(f"[prod-cache] {done}/{n}", flush=True)
                continue
            # RESUME: in disk-streaming mode, if a shard already exists
            # for (qname, fmt) on disk, treat it as previously rendered
            # and skip re-rendering.  This lets a job that OOM'd at 95%
            # resume without re-doing the work — just rebuild the manifest
            # from the surviving .pt files.
            if cache_dir_path is not None:
                fname = _cache_weight_filename(qname, fmt_key)
                disk_path = cache_dir_path / fname
                if disk_path.is_file():
                    weights[(qname, fmt_key)] = fname
                    skipped_resumed += 1
                    score_key = _render_score_record_key(qname, fmt_key)
                    if score_key not in render_score_records:
                        try:
                            cached = torch.load(
                                disk_path,
                                map_location=weight.device,
                                weights_only=True,
                            ).to(device=weight.device, dtype=weight.dtype)
                            render_score_records[score_key] = _render_score_record(
                                qname=qname,
                                fmt=fmt_key,
                                render_format=render_fmt,
                                reference_weight=weight,
                                rendered_weight=cached,
                                activations=activations_local.get(qname),
                                activation_max_abs=max_abs,
                                input_global_scale_policy=render_score_policy,
                            )
                            del cached
                        except Exception:
                            pass
                    # Do NOT pop activations_local[qname] here: this
                    # loop iterates through every format for this
                    # Linear, and a later format in the same outer
                    # iteration may still need the activation tensor
                    # to render.  The outer pop after the format loop
                    # drops it once all formats are done.
                    continue
            try:
                gate_trace: list[dict[str, object]] = []
                w_dq = render_production_weight(
                    weight, render_fmt,
                    qname=qname,
                    activations=activations_local,
                    levers=levers,
                    joint_global_real=joint,
                    input_global_scale=export_scale,
                    fisher_row_weights=row_weights,
                    col_weights=(
                        None if col_weights is None
                        else col_weights.get(qname)
                    ),
                    gate_trace=gate_trace,
                )
                render_score_records[_render_score_record_key(qname, fmt_key)] = (
                    _render_score_record(
                        qname=qname,
                        fmt=fmt_key,
                        render_format=render_fmt,
                        reference_weight=weight,
                        rendered_weight=w_dq,
                        activations=activations_local.get(qname),
                        activation_max_abs=max_abs,
                        input_global_scale_policy=render_score_policy,
                    )
                )
                if gate_trace:
                    render_gate_records.append({
                        "qname": qname,
                        "format": fmt_key,
                        "render_format": render_fmt,
                        "trace": gate_trace,
                    })
            except Exception as e:
                failed[(qname, fmt_key)] = str(e)
                if progress:
                    print(
                        f"[prod-cache] FAILED {qname} @ {fmt}: {e}",
                        flush=True,
                    )
                continue
            # MEM: store as the model's native dtype (bf16 by default)
            # rather than fp32 — _quantize_2d's compute_only path returns
            # fp32 but we always re-cast at install time, so storing fp32
            # is wasteful (2× memory).  On 27B this drops the cache from
            # ~25 GB to ~12 GB.
            _store_rendered_weight_entry(
                weights=weights,
                cache_dir_path=cache_dir_path,
                qname=qname,
                fmt=fmt_key,
                tensor=w_dq,
                weight_dtype=weight.dtype,
            )
            done += 1
            del w_dq
            if progress and done % 25 == 0:
                print(f"[prod-cache] {done}/{n}", flush=True)
                _write_render_score_sidecar(
                    render_score_sidecar_path,
                    render_score_records,
                )
        # Free this Linear's activation tensor — won't render this qname
        # again, and the activation can be tens of MB on big models.
        activations_local.pop(qname, None)
        if done % 50 == 0:
            _gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
    if progress:
        print(
            f"[prod-cache] rendered {len(weights)} (qname, fmt) entries "
            f"({skipped_resumed} resumed from disk, "
            f"{skipped_prewritten} pre-existing entries); "
            f"{len(failed)} failures",
            flush=True,
        )
    _write_render_score_sidecar(render_score_sidecar_path, render_score_records)
    render_gate_summary = _summarize_render_gate_records(render_gate_records)
    cache = ProductionWeightCache(
        weights=weights,
        levers=dict(levers),
        activation_max_abs=activation_max_abs or None,
        failed=failed,
        cache_dir=str(cache_dir_path) if cache_dir_path is not None else None,
        metadata={
            "render_mechanism_order": [
                {
                    "name": spec.name,
                    "operation": spec.operation,
                    "scope": spec.scope,
                    "gate_metric": spec.gate_metric,
                }
                for spec in mechanism_plan.ordered
            ],
            "render_failures": {
                f"{qname}|{fmt}": str(error)
                for (qname, fmt), error in sorted(failed.items())
            },
            "render_gates": {
                **render_gate_summary,
                "records": render_gate_records,
            },
            "render_scores": {
                "schema": "prismaquant.production_render_scores.v1",
                "entries": int(len(render_score_records)),
                "records": dict(sorted(render_score_records.items())),
                "cost_semantics": (
                    "score is the rendered candidate's local forward-error "
                    "mean after the format activation quantizer; score_sum "
                    "is score multiplied by activation_rows * out_features "
                    "for output metrics, or by parameter count for weight_mse "
                    "fallback. raw_render_score records the post-render "
                    "weight-only reconstruction objective used by local "
                    "render gates."
                ),
            },
            "four_over_six": (
                render_gate_summary.get("mechanisms", {}).get("four_over_six", {
                    "accepted": 0,
                    "rejected": 0,
                    "package_accepted": 0,
                    "reasons": {},
                })
                if isinstance(render_gate_summary.get("mechanisms"), dict)
                else {
                    "accepted": 0,
                    "rejected": 0,
                    "package_accepted": 0,
                    "reasons": {},
                }
            ),
            "fisher_weighted_gptq": {
                "enabled": bool(levers.get("fisher_gptq", False)),
                "h_detail_dir": str(h_detail_dir) if h_detail_dir else None,
                "loaded": (
                    int(fisher_rows.loads)
                    if fisher_rows is not None
                    and bool(levers.get("fisher_gptq", False))
                    else 0
                ),
                "misses": (
                    int(fisher_rows.misses)
                    if fisher_rows is not None
                    and bool(levers.get("fisher_gptq", False))
                    else 0
                ),
            },
            "render_scope": render_scope,
            # #130: render_scope alone cannot tell a reader which rows this
            # cache was rendered against -- a stripe and a whole build both
            # stamp "format-menu". The hooked enumeration is what the shared
            # priority stream is a function of, so record it: two caches with
            # equal hook digests and equal calibration rendered the same rows,
            # whatever subset each one rendered.
            ACTIVATION_HOOK_SCOPE_KEY: {
                "schema": ACTIVATION_HOOK_SCOPE_SCHEMA,
                "hooked_qnames_sha256": _qname_set_sha256(eligible_qnames),
                "hooked_qnames": len(eligible_qnames),
                "rendered_qnames": len(qname_set),
                "render_narrowed": len(qname_set) != len(eligible_qnames),
            },
            "requested_formats": list(requested_formats),
            "requested_entries": int(n),
        },
    )
    if recache_pass:
        from prismaquant.production_recache import recache_production_weight_cache

        if progress:
            print("[prod-cache] running production activation re-cache", flush=True)
        cache.prefetch_assignment(
            recache_assignment or {},
            max_resident_bytes=(
                cache._lru_max_bytes if cache._lru_max_bytes > 0 else None
            ),
            max_workers=4,
            require=False,
            progress=progress,
        )
        recache_production_weight_cache(
            model,
            calib_ids,
            recache_assignment or {},
            cache,
            profile=model_profile,
            include_activation_quant=recache_include_activation_quant,
            microbatch_size=recache_microbatch_size,
            progress=progress,
        )
        compacted = cache.compact_for_pickle()
        if progress and compacted:
            print(
                f"[prod-cache] compacted {compacted} resident cache tensors "
                "back to path references after re-cache",
                flush=True,
            )
    return cache


class _PackedExpertActivationCollector:
    """Capture module-level input ``X`` for each packed-experts module.

    Packed 3-D MoE experts are not ``nn.Linear`` and so are invisible to
    ``_LinearActivationCollector``.  This collector hooks the experts module
    itself and reservoir-samples its module-level input (the pre-routing token
    hidden states ``[*, hidden]``).  At render time the captured ``X`` is split
    into per-expert routed rows by ``derive_per_expert_activations`` — the same
    routing the live forward used — so each expert's GPTQ Hessian sees exactly
    the rows that route to it.

    The per-module token budget (``module_token_budget``) is intentionally much
    larger than the per-Linear ``max_rows`` budget: with top-k routing over E
    experts each expert only sees ~``top_k/E`` of the tokens, so a stable
    per-expert Hessian needs ``~max_rows_per_expert * E / top_k`` module tokens.

    An optional ``row_consumer(qname, X)`` observes each full, live input before
    storage selection. It must consume synchronously rather than retain X;
    campaigns use it to accumulate full-calibration Hessians without creating
    a second module-input cache. An optional boundary_consumer receives the
    original module, positional and keyword arguments before row selection or
    dtype conversion, preserving actual routing tensors for native validation.
    The default leaves reservoir sampling intact.
    """

    def __init__(
        self,
        model: nn.Module,
        experts_qnames: set[str],
        *,
        module_token_budget: int,
        store_device: torch.device | str,
        store_dtype: torch.dtype = torch.float32,
        profile=None,
        store_qnames: set[str] | None = None,
        row_consumer=None,
        boundary_consumer=None,
    ):
        self.model = model
        self.profile = profile
        self.row_consumer = row_consumer
        self.boundary_consumer = boundary_consumer
        # #145: ``experts_qnames`` is the enumeration the collector HOOKS --
        # the full visible module set -- and ``store_qnames`` narrows only
        # which modules keep full activation tensors. One shared generator
        # feeds every hooked module's priority reservoir, so the rows a
        # module keeps are a function of how many rows every EARLIER hook
        # consumed; narrowing the hook set therefore changes the rows, the
        # per-expert Hessians and the rendered bytes. This mirrors the dense
        # ``_LinearActivationCollector`` (``qnames``/``store_qnames``) fix
        # from #130/#135: hook the whole enumeration, narrow only the render.
        self.experts_qnames = set(experts_qnames)
        self.store_qnames = (
            set(store_qnames) if store_qnames is not None
            else set(experts_qnames)
        )
        self.module_token_budget = int(module_token_budget)
        self.store_device = torch.device(store_device)
        self.store_dtype = store_dtype
        self.activations: dict[str, list[torch.Tensor]] = {}
        self._priorities: dict[str, torch.Tensor] = {}
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(1234)
        self._handles: list = []
        self._modules_by_qname: dict[str, nn.Module] = {}
        from prismaquant.sensitivity_probe import _is_packed_experts_module
        for qname, mod in model.named_modules():
            if not _is_packed_experts_module(mod, self.profile):
                continue
            if qname not in self.experts_qnames:
                continue
            self._modules_by_qname[qname] = mod
            if qname in self.store_qnames:
                self.activations[qname] = []

    def install(self) -> None:
        for qname, mod in self._modules_by_qname.items():
            if self.boundary_consumer is None:
                handle = mod.register_forward_pre_hook(self._make_hook(qname))
            else:
                handle = mod.register_forward_pre_hook(self._make_hook(qname), with_kwargs=True)
            self._handles.append(handle)

    def _make_hook(self, qname: str):
        def hook(module, args, kwargs=None):
            if self.boundary_consumer is not None:
                self.boundary_consumer(qname, module, args, kwargs or {})
            if not args or not isinstance(args[0], torch.Tensor):
                return
            x = args[0]
            if self.row_consumer is not None:
                self.row_consumer(qname, x)
            # Draw this call's row priorities for EVERY hooked module,
            # stored or not. One generator feeds every module's reservoir, so
            # the slice of the stream a module receives is a function of how
            # many rows every earlier hook consumed. Drawing only for stored
            # modules makes a run that renders a subset -- an assignment scope
            # that BF16s one layer's experts, or the force_format frontier
            # build vs the export build -- keep DIFFERENT rows than the full
            # run, and the rendered bytes follow the rows. Draws are CPU
            # floats; the cost is noise next to the D2H copy below. Only
            # modules in the store set keep the full activation tensor.
            # Memory bound: store_qnames x module_token_budget x hidden.
            last_dim = int(x.shape[-1]) if x.dim() >= 1 else 0
            n_rows = (x.numel() // last_dim) if last_dim > 0 else 0
            new_priorities = (
                torch.rand(
                    int(n_rows),
                    generator=self._gen,
                    dtype=torch.float32,
                    device="cpu",
                )
                if n_rows > 0 and self.module_token_budget > 0
                else None
            )
            if qname not in self.store_qnames:
                return
            # NOT non_blocking: async D2H into pageable memory can read
            # the tensor before the producing kernel finishes under GPU
            # contention, deterministically corrupting the snapshot to NaN
            # (2026-07-11, inline-export agent repro).
            flat = x.detach().reshape(-1, x.shape[-1]).to(
                device=self.store_device,
                dtype=self.store_dtype,
            )
            current = (
                torch.cat(self.activations[qname], dim=0)
                if self.activations[qname]
                else None
            )
            sampled, priorities = update_priority_reservoir(
                current,
                self._priorities.get(qname),
                flat,
                max_rows=self.module_token_budget,
                new_priorities=new_priorities,
            )
            self.activations[qname] = [] if sampled is None else [sampled]
            if priorities is None:
                self._priorities.pop(qname, None)
            else:
                self._priorities[qname] = priorities
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def collected(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for qname, parts in self.activations.items():
            if parts:
                out[qname] = torch.cat(parts, dim=0)
        return out


PACKED_EXPERT_RENDER_SCORE_SCHEMA = (
    "prismaquant.production_render_scores.packed_expert.v1"
)
PACKED_EXPERT_RENDER_GATE_SCHEMA = (
    "prismaquant.production_render_gates.packed_expert.v1"
)
PACKED_ACTIVATION_HOOK_SCOPE_SCHEMA = (
    "prismaquant.production_weight_cache.packed_activation_hook_scope.v1"
)
PACKED_ACTIVATION_HOOK_SCOPE_KEY = "activation_hook_scope_packed"


def _stamp_packed_hook_scope(
    cache: ProductionWeightCache,
    *,
    hook_qnames: set[str],
) -> None:
    """Stamp which packed-experts enumeration a render hooked (#145).

    The stamp carries ONLY the hooked enumeration (digest + count), never a
    rendered count: the M4 lazy gap-fill calls this path repeatedly with
    overlapping assignments, so any rendered subset would describe the last
    call rather than the cache. The hook enumeration is what the shared
    priority stream is a function of, so equal digests (+ equal calibration)
    is the readable claim "these packed bytes were rendered against the same
    rows". A repeat call that hooked a different enumeration is a different
    rendering and is refused rather than silently re-stamped.
    """
    scope = {
        "schema": PACKED_ACTIVATION_HOOK_SCOPE_SCHEMA,
        "hooked_qnames_sha256": _qname_set_sha256(hook_qnames),
        "hooked_qnames": len(hook_qnames),
    }
    metadata = cache.metadata
    if metadata is None:
        cache.metadata = metadata = {}
    existing = metadata.get(PACKED_ACTIVATION_HOOK_SCOPE_KEY)
    if existing is None:
        metadata[PACKED_ACTIVATION_HOOK_SCOPE_KEY] = scope
        return
    if not isinstance(existing, Mapping):
        raise ValueError(
            "ProductionWeightCache activation_hook_scope_packed is malformed"
        )
    if (
        existing.get("schema") != scope["schema"]
        or existing.get("hooked_qnames_sha256")
        != scope["hooked_qnames_sha256"]
        or int(existing.get("hooked_qnames", -1)) != scope["hooked_qnames"]
    ):
        raise ValueError(
            "packed-expert activation hook enumeration differs from the "
            "cache's stamped rendering; refusing to mix two renderings in "
            "one cache"
        )


def packed_activation_hook_scope_of(
    source: object,
) -> dict[str, object] | None:
    """Canonical reader for the packed activation-hook stamp (#145).

    Returns the stamped enumeration (schema, digest, count), or ``None``
    when the cache rendered no packed experts. A present-but-malformed
    stamp is refused: "no stamp" (dense-only cache) and "a stamp that
    cannot be checked" must not be conflated. ``union_production_cache``
    and the cost/export pair gate share this reader so the digest
    comparison is one rule, not three.
    """
    metadata = (
        source.metadata if isinstance(source, ProductionWeightCache) else source
    )
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "packed activation hook scope needs a cache or metadata mapping"
        )
    scope = metadata.get(PACKED_ACTIVATION_HOOK_SCOPE_KEY)
    if scope is None:
        return None
    if not isinstance(scope, Mapping):
        raise ValueError(
            "ProductionWeightCache activation_hook_scope_packed is malformed"
        )
    digest = str(scope.get("hooked_qnames_sha256", "")).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(
            "ProductionWeightCache activation_hook_scope_packed digest "
            "is malformed"
        )
    if scope.get("schema") != PACKED_ACTIVATION_HOOK_SCOPE_SCHEMA:
        raise ValueError(
            "ProductionWeightCache activation_hook_scope_packed schema "
            "is unsupported"
        )
    count = scope.get("hooked_qnames")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or int(count) < 0
    ):
        raise ValueError(
            "ProductionWeightCache activation_hook_scope_packed count "
            "is malformed"
        )
    return {
        "schema": PACKED_ACTIVATION_HOOK_SCOPE_SCHEMA,
        "hooked_qnames_sha256": digest,
        "hooked_qnames": int(count),
    }


def _packed_expert_score_rows(
    per_expert_acts: Sequence[torch.Tensor | None] | None,
    expert: int,
    *,
    eval_rows: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return the scoring rows for one packed expert, or ``None``.

    The tail of the expert's routed fit rows is used, capped at
    ``eval_rows`` — the same slice the batched do-no-harm gate holds out
    when a same-corpus holdout is available.  Experts the router never
    sent a token to have no activation evidence and score weight-only.
    """
    if per_expert_acts is None or expert >= len(per_expert_acts):
        return None
    rows_tensor = per_expert_acts[expert]
    if rows_tensor is None or rows_tensor.numel() == 0:
        return None
    n_rows = int(rows_tensor.shape[0])
    cap = int(eval_rows) if int(eval_rows) > 0 else n_rows
    tail = rows_tensor[max(0, n_rows - cap):]
    return tail.detach().to(device=device, dtype=torch.float32)


def _packed_expert_render_score_record(
    *,
    qname: str,
    fmt: str,
    render_format: str,
    reference: torch.Tensor,
    rendered: torch.Tensor,
    per_expert_acts: Sequence[torch.Tensor | None] | None,
    activation_max_abs: float | None,
    eval_rows: int,
    device: torch.device,
    score_rows_source: str,
    input_global_scale_policy: str | None = None,
) -> dict[str, object]:
    """Score one packed 3-D expert cache entry honestly.

    Every expert in the stack is scored with the SAME scorer the dense path
    uses (``_render_score_record`` on ``[out, in]`` slices), against its own
    routed rows; the per-expert records are then summed into the one record
    the cache key owns.  Sums, not means-of-means: ``score`` is
    ``sum(score_sum) / sum(normalizer)`` so an expert with more routed rows
    weighs proportionally, exactly as a single dense Linear's rows do.

    Experts with no routed rows have no activation evidence.  They are
    excluded from the output metric (averaging an undefined error in would
    be a fabrication) and counted in ``experts_without_activations``;
    ``weight_mse`` is activation-independent and always covers every expert.
    """
    if reference.dim() != 3:
        raise RuntimeError(
            f"packed expert render score for {qname}@{fmt} needs a 3-D "
            f"[E, out, in] reference, got shape {tuple(reference.shape)}"
        )
    if tuple(rendered.shape) != tuple(reference.shape):
        raise RuntimeError(
            f"packed expert render score for {qname}@{fmt}: rendered shape "
            f"{tuple(rendered.shape)} differs from reference "
            f"{tuple(reference.shape)}"
        )
    n_experts = int(reference.shape[0])
    per_expert: list[dict[str, object]] = []
    for expert in range(n_experts):
        acts_e = _packed_expert_score_rows(
            per_expert_acts, expert, eval_rows=eval_rows, device=device,
        )
        per_expert.append(
            _render_score_record(
                qname=f"{qname}.e{expert}",
                fmt=fmt,
                render_format=render_format,
                reference_weight=reference[expert].detach().to(
                    device=device, dtype=torch.float32,
                ),
                rendered_weight=rendered[expert].detach().to(device=device),
                activations=acts_e,
                activation_max_abs=activation_max_abs,
                input_global_scale_policy=input_global_scale_policy,
            )
        )
        del acts_e
    if not per_expert:
        raise RuntimeError(
            f"packed expert render score for {qname}@{fmt} has zero experts"
        )
    with_acts = [
        record for record in per_expert
        if int(record["activation_rows"]) > 0  # type: ignore[arg-type]
    ]
    scored = with_acts or per_expert
    metrics = {str(record["metric"]) for record in scored}
    raw_metrics = {str(record["raw_render_metric"]) for record in scored}
    if len(metrics) != 1 or len(raw_metrics) != 1:
        raise RuntimeError(
            f"packed expert render score for {qname}@{fmt} mixes metrics: "
            f"metric={sorted(metrics)} raw={sorted(raw_metrics)}"
        )
    normalizer = sum(float(record["normalizer"]) for record in scored)
    score_sum = sum(float(record["score_sum"]) for record in scored)
    raw_score_sum = sum(
        float(record["raw_render_score_sum"]) for record in scored
    )
    weight_mse_sum = sum(
        float(record["weight_mse_sum"]) for record in per_expert
    )
    n_weights = sum(int(record["n_weights"]) for record in per_expert)
    activation_rows = sum(int(record["activation_rows"]) for record in scored)
    clip_values = {
        float(record["activation_max_abs"])
        for record in scored
        if record["activation_max_abs"] is not None
    }
    # Every expert in the stack was priced from the one ``activation_max_abs``
    # this entry owns, so a served-contract stack has exactly one static G and
    # one policy.  Carry them the way the dense record does, so a packed cost
    # is checkable against a later policy too (#227); a stack scored under a
    # dynamic quantizer records neither.
    scale_values = {
        float(record["input_global_scale"])
        for record in scored
        if record.get("input_global_scale") is not None
    }
    if len(scale_values) > 1:
        raise RuntimeError(
            f"packed expert render score for {qname}@{fmt} mixes static "
            f"activation scales {sorted(scale_values)}; one entry is priced "
            "under one input_global_scale"
        )
    policy_values = {
        str(record["input_global_scale_policy"])
        for record in scored
        if record.get("input_global_scale_policy") is not None
    }
    return {
        "qname": str(qname),
        "format": str(fmt).upper(),
        "render_format": str(render_format).upper(),
        "schema": PACKED_EXPERT_RENDER_SCORE_SCHEMA,
        "metric": next(iter(metrics)),
        "score": float(score_sum / normalizer) if normalizer > 0 else 0.0,
        "score_sum": float(score_sum),
        "raw_render_metric": next(iter(raw_metrics)),
        "raw_render_score": (
            float(raw_score_sum / normalizer) if normalizer > 0 else 0.0
        ),
        "raw_render_score_sum": float(raw_score_sum),
        "weight_mse": (
            float(weight_mse_sum / n_weights) if n_weights > 0 else 0.0
        ),
        "weight_mse_sum": float(weight_mse_sum),
        "n_weights": int(n_weights),
        "normalizer": float(normalizer),
        "activation_rows": int(activation_rows),
        "activation_quantized": any(
            bool(record["activation_quantized"]) for record in scored
        ),
        "activation_clipped": any(
            bool(record["activation_clipped"]) for record in scored
        ),
        "activation_max_abs": (
            float(max(clip_values)) if clip_values else None
        ),
        "input_global_scale": (
            float(next(iter(scale_values))) if scale_values else None
        ),
        "input_global_scale_policy": (
            str(next(iter(policy_values)))
            if scale_values and policy_values else None
        ),
        "out_features": int(reference.shape[1]),
        "in_features": int(reference.shape[2]),
        "packed_experts": int(n_experts),
        "experts_scored_with_activations": int(len(with_acts)),
        "experts_without_activations": int(n_experts - len(with_acts)),
        "expert_score_aggregation": "sum_over_experts",
        "score_rows_source": str(score_rows_source),
        "score_rows_per_expert_cap": int(eval_rows),
    }


def _packed_expert_render_gate_record(
    *,
    qname: str,
    fmt: str,
    render_format: str,
    render_mode: str,
    batched: bool,
    resumed: bool,
    coverage_record: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return the truthful render-gate record for a packed expert entry.

    The packed path runs NO progressive-gate mechanism, so this record
    carries an EMPTY trace: synthesizing a step would claim a gate evaluated
    candidates it never saw.  What it records instead is what actually
    executed, plus the per-expert GPTQ-vs-RTN do-no-harm decision counts the
    render did measure.
    """
    if resumed:
        mechanism = "resumed_shard"
        progressive_gate = "not-run"
        detail = (
            "packed-MoE expert entry resumed from an existing shard; no "
            "render mechanism ran in this process. Render telemetry is the "
            "producing run's, carried in packed_expert_coverage."
        )
    elif batched:
        mechanism = "batched_gptq_fixed_damp"
        progressive_gate = "not-run"
        detail = (
            "packed-MoE expert render: no progressive-gate mechanism ran. "
            "The batched path applies one fixed-damp GPTQ column update "
            "across the layer's experts; joint_scale_opt, static_act_order "
            "and the damp sweep are deliberately not run (each forces a "
            "per-expert loop). The GPTQ-vs-RTN do-no-harm decision recorded "
            "below WAS measured, per expert."
        )
    else:
        mechanism = "render_production_weight_per_expert"
        progressive_gate = "ran-untraced"
        detail = (
            "packed-MoE expert render in per_expert mode: progressive gates "
            "ran inside render_production_weight for each expert, but the "
            "packed path collects no per-expert gate trace, so no mechanism "
            "step is claimed here."
        )
    record: dict[str, object] = {
        "qname": str(qname),
        "format": str(fmt).upper(),
        "render_format": str(render_format).upper(),
        "schema": PACKED_EXPERT_RENDER_GATE_SCHEMA,
        "trace": [],
        "progressive_gate": progressive_gate,
        "mechanism": mechanism,
        "render_mode": str(render_mode),
        "detail": detail,
    }
    if isinstance(coverage_record, Mapping):
        record["expert_do_no_harm"] = {
            field: coverage_record[field]
            for field in (
                "n_experts",
                "empty_experts",
                "rtn_fallbacks",
                "heldout_reverts",
                "gptq_experts",
                "cross_gated_experts",
                "gate_mode",
            )
            if field in coverage_record
        }
    return record


def _packed_expert_metadata_scope(
    metadata: Mapping[str, object],
    key: str,
) -> dict[str, object]:
    value = metadata.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeError(
            f"ProductionWeightCache metadata.{key} is not a mapping"
        )
    return dict(value)


def _finalize_packed_expert_cache_metadata(
    cache: ProductionWeightCache,
    *,
    packed_scope_names: set[str],
    score_records: Mapping[str, Mapping[str, object]],
    gate_records: Mapping[tuple[str, str], Mapping[str, object]],
    coverage: Mapping[str, Mapping[str, object]],
    cache_dir_path: Path | None,
) -> None:
    """Restore the cache's render-identity invariants after a packed append.

    ``union_production_cache`` treats three counters as invariants of a
    materialized cache: ``requested_entries == len(cache)``, one
    ``render_scores`` record per cache key, and one ``render_gates`` record
    per non-MTP cache key.  A packed-expert append that only pushes keys
    into ``cache.weights`` leaves all three stale and the exact union
    refuses at its first subcommand.

    The packed append is replaced as ONE exact scope (the MTP doctrine at
    ``mtp_production_cache``): records for packed-expert tensor names that
    are no longer cache keys are dropped, records for the keys present are
    upserted, and every counter is RECOMPUTED from the cache rather than
    incremented — the M4 lazy gap-fill calls this path repeatedly with
    overlapping assignments, and an increment drifts on every rerun.
    """
    if cache.metadata is None:
        cache.metadata = {}
    metadata = cache.metadata
    live_keys = set(cache.weights)

    def _is_orphan_packed_key(qname: str, fmt: str) -> bool:
        return (
            qname in packed_scope_names
            and (qname, fmt) not in live_keys
        )

    score_meta = _packed_expert_metadata_scope(metadata, "render_scores")
    existing_scores = score_meta.get("records", {})
    if not isinstance(existing_scores, Mapping):
        raise RuntimeError(
            "ProductionWeightCache render_scores.records is not a mapping"
        )
    merged_scores: dict[str, dict[str, object]] = {}
    for raw_key, value in existing_scores.items():
        if not isinstance(value, Mapping):
            raise RuntimeError(
                f"ProductionWeightCache render score is not a mapping: "
                f"{raw_key!r}"
            )
        parts = str(raw_key).rsplit("|", 1)
        if len(parts) == 2 and _is_orphan_packed_key(parts[0], parts[1]):
            continue
        merged_scores[str(raw_key)] = dict(value)
    for raw_key, value in score_records.items():
        merged_scores[str(raw_key)] = dict(value)
    score_meta.setdefault(
        "schema", "prismaquant.production_render_scores.v1"
    )
    score_meta["entries"] = int(len(merged_scores))
    score_meta["records"] = dict(sorted(merged_scores.items()))
    metadata["render_scores"] = score_meta

    # render_gates is written by the non-streaming dense fill only. When it is
    # absent the union skips its coverage check entirely; creating a
    # packed-only structure here would turn that skip into a guaranteed
    # refusal, so absence is preserved.
    if "render_gates" in metadata:
        gate_meta = _packed_expert_metadata_scope(metadata, "render_gates")
        existing_gate_records = gate_meta.get("records", [])
        if not isinstance(existing_gate_records, list):
            raise RuntimeError(
                "ProductionWeightCache render_gates.records is not a list"
            )
        merged_gates: dict[tuple[str, str], dict[str, object]] = {}
        for value in existing_gate_records:
            if not isinstance(value, Mapping):
                raise RuntimeError(
                    "ProductionWeightCache render gate record is not a mapping"
                )
            pair = (str(value.get("qname", "")), str(value.get("format", "")))
            if _is_orphan_packed_key(*pair):
                continue
            merged_gates[pair] = dict(value)
        for pair, value in gate_records.items():
            merged_gates[pair] = dict(value)
        ordered_gates = [
            merged_gates[pair] for pair in sorted(merged_gates)
        ]
        enabled = gate_meta.get("enabled", True)
        summary = _summarize_render_gate_records(ordered_gates)
        summary["enabled"] = enabled
        metadata["render_gates"] = {**summary, "records": ordered_gates}

    if coverage or "packed_expert_coverage" in metadata:
        merged_coverage = _packed_expert_metadata_scope(
            metadata, "packed_expert_coverage"
        )
        merged_coverage.update({
            str(qname): dict(record) for qname, record in coverage.items()
        })
        metadata["packed_expert_coverage"] = dict(
            sorted(merged_coverage.items())
        )

    metadata["requested_entries"] = int(len(cache.weights))
    if cache_dir_path is not None:
        _write_render_score_sidecar(
            cache_dir_path / "render_scores.json", merged_scores,
        )


def fill_packed_expert_cache_entries(
    cache: ProductionWeightCache,
    model: nn.Module,
    calib_ids: torch.Tensor | None,
    *,
    render_assignment: Mapping[str, str] | None = None,
    force_format: str | None = None,
    levers: Mapping[str, object],
    profile,
    module_token_budget: int = 32768,
    max_rows_per_expert: int = 2048,
    eval_rows_per_expert: int = 128,
    cache_dir: str | Path | None = None,
    progress: bool = True,
    max_layers: int | None = None,
    render_mode: str = "batched",
    gate_calib_ids: torch.Tensor | None = None,
    gate_token_budget: int | None = None,
    module_acts_override: Mapping[str, torch.Tensor] | None = None,
    col_weights: Mapping[str, torch.Tensor] | None = None,
) -> dict:
    """Render packed-MoE experts into ``ProductionWeightCache`` entries.

    ``render_mode``:
      * ``"batched"`` (default, production): vectorize the GPTQ column update
        across all experts of a layer via ``gptq_obs_rounding_nvfp4_batched``
        — ~E× fewer Python iterations than per-expert. Fixed damp, no JSO/
        act-order (those force a per-expert loop that is ~hours on a 35B).
      * ``"per_expert"``: render each expert through the full
        ``render_production_weight`` stack (progressive gates + damp-sweep +
        JSO + do-no-harm). Faithful to the dense path but ~hours on a 35B;
        used for the fidelity A/B that justifies the production default.

    ``gate_calib_ids``: optional token ids from a corpus DISJOINT from
    ``calib_ids``. When given, the per-expert GPTQ-vs-RTN do-no-harm gate is
    judged on this corpus's routed rows instead of a same-corpus held-out
    slice, and GPTQ fits on ALL fit-corpus rows. Rationale (2026-06-09 served
    A/B): per-expert Hessians are thin (sparse routing), so GPTQ overfits its
    calibration *domain* — a same-domain holdout catches in-sample overfit but
    passed renders that LOST on served cross-domain KL (RTN 0.0302 vs GPTQ
    0.0334 confident). Experts the gate corpus never routes to fall back to
    the same-domain holdout. Default ``None`` preserves prior behavior.

    This closes the packed-expert RTN-by-omission gap: every non-BF16 packed
    expert tensor receives a deliberate cache render keyed by routed
    per-expert activations. In production's default ``"batched"`` mode that
    render is fixed-damp batched GPTQ without dense JSO/act-order/damp-sweep,
    followed by the measured GPTQ-vs-RTN gate below. The non-default
    ``"per_expert"`` mode is the one that calls ``render_production_weight``
    and matches the dense GPTQ+damp-sweep+JSO stack. Export then reads the
    cached 3-D dequant and re-packs it — exactly the
    ``_pack_production_cached_2d`` contract, lifted to packed experts.

    Hook invariant (#145, mirrors the dense #130/#135 one): the activation
    collector hooks EVERY packed-experts module this call can see
    (``eligible_experts_qnames``), and ``render_assignment``/``force_format`` narrow
    only which tensors get rendered. The bytes of a packed (tensor, fmt)
    pair are therefore a function of the visible modules and the
    calibration, never of which subset a call renders — a ``force_format``
    frontier build and an assignment-scoped export build that BF16s one
    layer's experts render identical bytes for the rest.

    Args:
      cache: the ``ProductionWeightCache`` produced by
        ``fill_production_weight_cache``.  New 3-D ``(experts_qname.pn, fmt)``
        entries are added in place (and streamed to ``cache_dir`` shards when
        disk-streaming is on).
      render_assignment: the concrete export assignment (recipe_key -> fmt).
        Only non-BF16 packed-expert tensors are rendered (BF16 = passthrough).
      levers: the resolved production lever dict. It is consumed by
        ``render_mode="per_expert"``; the production ``"batched"`` path uses
        its fixed fast recipe regardless of dense-linears JSO/damp-sweep
        levers.
      module_token_budget: reservoir size for each experts module's input X.
      max_rows_per_expert: cap on routed rows fed to each expert's GPTQ.
      max_layers: debug/timing cap — render only the first N experts modules.
      col_weights: optional ``{qname: vector}`` imatrix map (re-vet R3),
        keyed by the packed tensor name (``…experts.w1`` etc.)
        with either one shared ``(in_features,)`` vector or a per-expert
        ``(E, in_features)`` stack — the layout
        ``moe_imatrix.synthesize_packed_expert_col_weights`` emits. Applied
        only on the per-expert render path and only for the weighted-render
        families; NVFP4/FP8 expert bytes are unchanged.
      module_acts_override: streaming build. ``{experts_qname: X}`` module-level
        input snapshots sourced from the probe's activation cache instead of a
        fresh forward pass. When supplied, no calibration forward runs, in-scope
        experts are restricted to the supplied modules (the currently-installed
        decoder layer), ``calib_ids`` may be ``None``, and the cross-domain gate
        corpus is unsupported.

    Returns a coverage dict consumed by the export enforcement gate:
      ``{experts_qname.pn: {"fmt", "n_experts", "min_rows", "median_rows",
      "rendered", "had_activations"}}``.
    """
    import time as _time
    from prismaquant.sensitivity_probe import (
        _is_packed_experts_module,
        _packed_experts_param_names,
    )
    from prismaquant.measure_quant_cost import (
        derive_per_expert_activations,
        _packed_experts_parent_module,
    )
    from prismaquant.export_native_compressed import (
        _split_packed_expert_tensor,
        compute_nvfp4_global_real,
    )
    from prismaquant import format_registry as fr

    # Device probe: the streamed export materializes ONLY the current
    # decoder layer (+ head) — on a multimodal-forced skeleton
    # (glm5_next) the module-order-first parameter is a META visual-tower
    # weight, so `next(model.parameters())` is not a valid device probe
    # (it silently .to(meta)'d the activation snapshot and crashed the
    # router forward). Use the first NON-meta parameter instead.
    device = None
    for _probe_param in model.parameters():
        if not _probe_param.is_meta:
            device = _probe_param.device
            break
    if device is None:
        device = next(model.parameters()).device
    cache_dir_path = Path(cache_dir) if cache_dir is not None else None
    if cache_dir_path is not None:
        cache_dir_path.mkdir(parents=True, exist_ok=True)
    from prismaquant.schemas import refuse_retired_codebook_format

    def _canon(fmt: str) -> str:
        canonical = fr.canonical_format_name(str(fmt).strip().upper())
        # A retired codebook rung (archived 2026-09-25, #1304) refuses here,
        # before the render loop can record it as a failure or resume it.
        refuse_retired_codebook_format(canonical)
        return canonical

    if force_format is None and render_assignment is None:
        raise ValueError(
            "fill_packed_expert_cache_entries requires render_assignment or "
            "force_format")
    if force_format is not None and render_assignment is not None:
        raise ValueError(
            "fill_packed_expert_cache_entries: pass render_assignment OR "
            "force_format, not both")
    if module_acts_override is None and calib_ids is None:
        raise ValueError(
            "fill_packed_expert_cache_entries requires calib_ids unless "
            "module_acts_override supplies the experts-module input snapshot")
    if module_acts_override is not None and gate_calib_ids is not None:
        raise ValueError(
            "fill_packed_expert_cache_entries: module_acts_override (streaming) "
            "does not support a cross-domain gate corpus")

    # 1. Resolve in-scope packed-expert tensors (non-BF16 in the assignment).
    #    Each entry: (experts_qname, mod, parent, pn, full, fmt).
    #
    #    #145: ``eligible_experts_qnames`` is the enumeration the activation
    #    collector hooks, and it is deliberately NOT narrowed by the
    #    render assignment below. One shared generator feeds every hooked
    #    module's priority reservoir, so the rows a module keeps are a
    #    function of how many rows every EARLIER hook consumed. Narrowing
    #    the hook set therefore changes the rows, the per-expert Hessians
    #    and the rendered bytes. The invariant this buys -- the same one
    #    #130/#135 bought the dense path: the bytes of a (tensor, fmt) pair
    #    depend on the visible modules and the calibration, never on which
    #    subset of them this call happens to render. That is what makes a
    #    force_format frontier build and an assignment-scoped export build
    #    reproduce each other's bytes.
    in_scope: list[tuple[str, nn.Module, nn.Module, str, str, str]] = []
    experts_qnames: set[str] = set()
    eligible_experts_qnames: set[str] = set()
    # Every packed-expert tensor name this call can SEE, BF16 and
    # out-of-assignment included. This is the exact scope inside which stale
    # render-score / render-gate records may be pruned at the end of the call;
    # dense and MTP records, and other layers' packed records on the streaming
    # path, are never in it.
    all_packed_fullnames: set[str] = set()
    modules_seen = 0
    for experts_qname, mod in model.named_modules():
        if not _is_packed_experts_module(mod, profile):
            continue
        # Streaming build: only the currently-installed decoder layer's experts
        # module is materialized (others are on meta) and only it has an X in
        # the override. Skip everything else so we never touch a meta param.
        if (
            module_acts_override is not None
            and experts_qname not in module_acts_override
        ):
            continue
        if max_layers is not None and modules_seen >= max_layers:
            break
        modules_seen += 1
        # Hooked whether or not this module renders below: the hook set is
        # the visible enumeration, and the render assignment narrows only
        # the render (#145).
        eligible_experts_qnames.add(experts_qname)
        parent = _packed_experts_parent_module(model, experts_qname)
        for pn in _packed_experts_param_names(mod, profile):
            full = f"{experts_qname}.{pn}" if experts_qname else pn
            all_packed_fullnames.add(full)
            if force_format is not None:
                # Force-format mode: render every packed expert at one format,
                # ignoring render_assignment. Used by the format-menu frontier
                # build (eager NVFP4) and by per-Pareto-point lazy FP8 gap-fill.
                fmt = _canon(force_format)
            else:
                try:
                    recipe_key = profile.live_to_recipe_name(full)
                except Exception:
                    recipe_key = full
                fmt = render_assignment.get(recipe_key)
                if fmt is None and recipe_key != full:
                    fmt = render_assignment.get(full)
                if fmt is None:
                    continue
                fmt = _canon(fmt)
            if fmt == "BF16":
                continue
            in_scope.append((experts_qname, mod, parent, pn, full, fmt))
            experts_qnames.add(experts_qname)

    coverage: dict[str, dict[str, object]] = {}
    if not in_scope:
        if progress:
            print("[prod-cache/experts] no non-BF16 packed experts in scope",
                  flush=True)
        return coverage
    if cache_dir_path is not None:
        from prismaquant.perturbed_x_cache import calibration_data_hash

        if module_acts_override is None:
            _check_and_record_append_identity(
                cache_dir_path,
                PACKED_APPEND_SIDECAR_KEY,
                build_packed_expert_append_identity(
                    module_token_budget=int(module_token_budget),
                    max_rows_per_expert=int(max_rows_per_expert),
                    eval_rows_per_expert=int(eval_rows_per_expert),
                    render_mode=str(render_mode),
                    gate_calibration_hash=(
                        calibration_data_hash(gate_calib_ids)
                        if gate_calib_ids is not None
                        else None
                    ),
                    gate_token_budget=(
                        int(gate_token_budget or module_token_budget)
                        if gate_calib_ids is not None
                        else None
                    ),
                    hooked_qnames=eligible_experts_qnames,
                    max_layers=max_layers,
                    pair_fit_calibration_hashes={
                        f"{full}|{fmt}": calibration_data_hash(calib_ids)
                        for (_q, _m, _p, _pn, full, fmt) in in_scope
                    },
                ),
                progress=progress,
            )
        else:
            # #172: streaming builds render one materialized layer per call
            # from supplied snapshots — each call sees a different single
            # module, so no single section can describe the directory. Record
            # a per-layer section list keyed by the module this call saw
            # (the union of what each streaming call rendered) BEFORE any
            # render, so a hand-mixed streaming configuration refuses instead
            # of merging silently. The render narrowing (which packed tensors
            # this layer renders) is deliberately NOT bound, exactly as on
            # the resident path.
            snapshot_layers = {
                str(experts_qname): calibration_data_hash(
                    module_acts_override[experts_qname]
                )
                for experts_qname in sorted(experts_qnames)
                if experts_qname in module_acts_override
            }
            if snapshot_layers:
                _check_and_record_append_identity(
                    cache_dir_path,
                    PACKED_STREAMING_APPEND_SIDECAR_KEY,
                    build_packed_streaming_append_identity(
                        module_token_budget=int(module_token_budget),
                        max_rows_per_expert=int(max_rows_per_expert),
                        eval_rows_per_expert=int(eval_rows_per_expert),
                        render_mode=str(render_mode),
                        max_layers=max_layers,
                        layers=snapshot_layers,
                    ),
                    progress=progress,
                )
    if module_acts_override is None:
        # Resident builds hook the visible enumeration through the shared
        # collector, so the stamp names a real rendering. Streaming builds
        # render one materialized layer per call from supplied snapshots --
        # no collector runs, nothing is hooked, and each call sees a
        # different single module -- so there is no enumeration to stamp.
        # The per-layer sidecar union (#172) is what guards that path, not
        # this stamp.
        _stamp_packed_hook_scope(cache, hook_qnames=eligible_experts_qnames)

    # Render-identity bookkeeping for this append. Every packed key that ends
    # up in ``cache.weights`` owes a render-score record and a render-gate
    # record, and the counters are recomputed from the cache at the end of the
    # call (``_finalize_packed_expert_cache_metadata``). Records already on the
    # cache are honored as-is so a repeat call with an overlapping assignment
    # neither re-renders nor re-scores.
    existing_score_records = _packed_expert_metadata_scope(
        cache.metadata or {}, "render_scores",
    ).get("records", {})
    if not isinstance(existing_score_records, Mapping):
        raise RuntimeError(
            "ProductionWeightCache render_scores.records is not a mapping"
        )
    packed_score_records: dict[str, dict[str, object]] = {}
    packed_gate_records: dict[tuple[str, str], dict[str, object]] = {}
    existing_coverage_records = _packed_expert_metadata_scope(
        cache.metadata or {}, "packed_expert_coverage",
    )

    def _has_render_score(qname: str, fmt: str) -> bool:
        score_key = _render_score_record_key(qname, fmt)
        return (
            score_key in packed_score_records
            or score_key in existing_score_records
        )

    resolved_packed_levers = _resolve_production_render_levers(levers)
    packed_score_policy = _render_levers_input_global_scale_policy(
        resolved_packed_levers
    )
    # Retained packed costs are reused by key exactly as the dense ones are,
    # so they answer the same question before this append prices anything
    # else (#227).
    _check_resumed_render_score_policies(
        existing_score_records,
        policy=packed_score_policy,
        where="packed expert cache resume",
    )

    if progress:
        print(
            f"[prod-cache/experts] {len(in_scope)} packed-expert tensors across "
            f"{len(experts_qnames)} modules "
            f"({len(eligible_experts_qnames)} hooked); capturing module "
            f"activations (budget={module_token_budget} tokens/module)",
            flush=True,
        )

    # RESUME: persist per-param activation max_abs (the export's W4A4 input
    # scale) to a sidecar so a resumed build — which skips re-rendering existing
    # shards — still has the calibrated scale. Without this, a resumed build
    # ships the 1.0 placeholder (Codex blocker). Load it up front; the render
    # loop writes it as each tensor is (re)computed.
    import json as _json
    if cache.activation_max_abs is None:
        cache.activation_max_abs = {}
    expert_sidecar_path: Path | None = (
        cache_dir_path / "packed_expert_max_abs.json"
        if cache_dir_path is not None else None
    )
    if expert_sidecar_path is not None and expert_sidecar_path.is_file():
        try:
            loaded_scales = _json.loads(expert_sidecar_path.read_text())
            if isinstance(loaded_scales, Mapping):
                cache.activation_max_abs.update({
                    str(qname): float(value)
                    for qname, value in loaded_scales.items()
                })
        except Exception:
            pass

    in_scope_keys = [(full, fmt) for (_q, _m, _p, _pn, full, fmt) in in_scope]

    def _persist_expert_sidecar() -> None:
        if expert_sidecar_path is None:
            return
        # MERGE with the existing sidecar — a subset render (the M4 lazy
        # per-Pareto-point FP8 gap-fill) must not destroy the eager build's
        # scale entries for every OTHER expert in the shared cache_dir.
        merged: dict[str, float] = {}
        if expert_sidecar_path.is_file():
            try:
                merged.update(_json.loads(expert_sidecar_path.read_text()))
            except Exception:
                pass
        for full_k, _fmt in in_scope_keys:
            v = cache.activation_max_abs.get(full_k)
            if v is not None:
                merged[full_k] = float(v)
        tmp = expert_sidecar_path.with_suffix(".json.tmp")
        tmp.write_text(_json.dumps(merged, indent=2))
        os.replace(tmp, expert_sidecar_path)
    # Only capture activations if some tensor still needs rendering OR is
    # missing its calibrated scale — a fully-resumed build with a complete
    # sidecar skips the forward entirely.
    def _needs_work(full: str, fmt: str) -> bool:
        # A missing render score is work too: scoring needs this module's
        # routed activations, so the capture below must run for it exactly as
        # the dense path's ``missing_activation_score`` makes it run there.
        if (full, fmt) in cache.weights:
            return (
                cache.activation_max_abs.get(full) is None
                or not _has_render_score(full, fmt)
            )
        if cache_dir_path is not None:
            shard = cache_dir_path / _cache_weight_filename(full, fmt)
            if shard.is_file():
                return (
                    cache.activation_max_abs.get(full) is None
                    or not _has_render_score(full, fmt)
                )
        return True

    work_remaining = any(_needs_work(full, fmt) for full, fmt in in_scope_keys)

    module_acts: dict[str, torch.Tensor] = {}
    if module_acts_override is not None:
        # Streaming build: the experts-module input snapshot X comes from the
        # probe's activation cache (keyed by the experts-module qname), not a
        # fresh forward pass — the model is only ever one layer resident here.
        # Store on CPU/fp32 exactly as the collector would so the downstream
        # derive/render path is byte-identical to the resident build.
        module_acts = {
            q: t.detach().reshape(-1, t.size(-1)).to(
                device="cpu", dtype=torch.float32,
            )
            for q, t in module_acts_override.items()
            if q in experts_qnames
        }
        if progress:
            print(
                f"[prod-cache/experts] streaming: using {len(module_acts)} "
                "supplied module activation snapshot(s) (no forward)",
                flush=True,
            )
    elif work_remaining:
        # 2. Capture module-level X for each experts module (one calib forward).
        # Store the (large) per-module reservoirs on CPU/host, NOT in the CUDA
        # allocator pool: on the GB10 unified-memory box the model already
        # occupies ~67 GB of GPU pool, and 40 modules × budget × hidden × fp32
        # in the CUDA pool fragments/OOMs the capture (observed). CPU storage
        # keeps them out of the CUDA segment pool; derive moves one module's X
        # back to GPU transiently. Pinned for faster H2D at render.
        #
        # Hook the WHOLE eligible enumeration -- not the render-narrowed
        # ``experts_qnames``. Hooking is cheap (one CPU ``torch.rand`` per
        # call), and hooking the full set is what keeps the shared priority
        # stream identical across a force_format frontier build and an
        # assignment-scoped export build (#145). Only STORE full activations
        # for modules we still need to render.
        collector = _PackedExpertActivationCollector(
            model,
            eligible_experts_qnames,
            store_qnames=experts_qnames,
            module_token_budget=module_token_budget,
            store_device=torch.device("cpu"),
            store_dtype=torch.float32,
            profile=profile,
        )
        collector.install()
        t_cap = _time.monotonic()
        try:
            with torch.no_grad():
                for i in range(calib_ids.size(0)):
                    batch = calib_ids[i:i + 1].to(device)
                    try:
                        model(batch, use_cache=False)
                    except TypeError:
                        model(batch)
        finally:
            collector.remove()
        module_acts = collector.collected()
        if progress:
            print(
                f"[prod-cache/experts] captured {len(module_acts)} module "
                f"activations in {_time.monotonic() - t_cap:.1f}s",
                flush=True,
            )
    elif progress:
        print(
            "[prod-cache/experts] resume: all packed experts rendered + scales "
            "in sidecar, skipping activation capture",
            flush=True,
        )

    # 2b. Cross-domain gate corpus: capture a SECOND reservoir per module from
    # the disjoint gate corpus. Same CPU-storage rationale as the fit reservoir.
    gate_module_acts: dict[str, torch.Tensor] = {}
    if work_remaining and gate_calib_ids is not None:
        gate_collector = _PackedExpertActivationCollector(
            model,
            eligible_experts_qnames,
            store_qnames=experts_qnames,
            module_token_budget=int(gate_token_budget or module_token_budget),
            store_device=torch.device("cpu"),
            store_dtype=torch.float32,
            profile=profile,
        )
        gate_collector.install()
        t_cap = _time.monotonic()
        try:
            with torch.no_grad():
                for i in range(gate_calib_ids.size(0)):
                    batch = gate_calib_ids[i:i + 1].to(device)
                    try:
                        model(batch, use_cache=False)
                    except TypeError:
                        model(batch)
        finally:
            gate_collector.remove()
        gate_module_acts = gate_collector.collected()
        if progress:
            print(
                f"[prod-cache/experts] captured {len(gate_module_acts)} "
                f"cross-domain gate-module activations in "
                f"{_time.monotonic() - t_cap:.1f}s",
                flush=True,
            )

    # 3. Render each in-scope tensor per-expert through render_production_weight.
    derived_by_module: dict[str, dict] = {}
    gate_derived_by_module: dict[str, dict | None] = {}
    # Index of each module's LAST in-scope param, so we can free its (large,
    # GPU-resident) derived per-expert activations + captured X as soon as both
    # its params (gate_up + down) are rendered — otherwise they accumulate
    # across all 40 modules (~2.5 GB/module on GPU) and OOM the box.
    last_idx_by_module: dict[str, int] = {}
    for _i, (_q, _m, _p, _pn, _full, _fmt) in enumerate(in_scope):
        last_idx_by_module[_q] = _i
    weights = cache.weights
    t0 = _time.monotonic()
    for idx, (experts_qname, mod, parent, pn, full, fmt) in enumerate(in_scope):
        key = (full, fmt)
        fname = (
            _cache_weight_filename(full, fmt)
            if cache_dir_path is not None else None
        )
        shard_exists = (key in weights) or (
            fname is not None and (cache_dir_path / fname).is_file()
        )
        need_scale = cache.activation_max_abs.get(full) is None
        score_key = _render_score_record_key(full, fmt)
        gate_key = (full, str(fmt).upper())
        need_score = not _has_render_score(full, fmt)
        # Fully done (shard + calibrated scale + render score): register the
        # path and re-stamp this entry's gate record, which the union counts
        # per cache key on every rerun.
        if shard_exists and not need_scale and not need_score:
            if key not in weights and fname is not None:
                weights[key] = fname
            packed_gate_records[gate_key] = _packed_expert_render_gate_record(
                qname=full,
                fmt=fmt,
                render_format=fmt,
                render_mode=render_mode,
                batched=False,
                resumed=True,
                coverage_record=(
                    coverage.get(full) or existing_coverage_records.get(full)
                ),
            )
            continue

        X = module_acts.get(experts_qname)
        if X is None or X.numel() == 0:
            raise RuntimeError(
                f"[prod-cache/experts] no captured activations for "
                f"{experts_qname}; cannot render {full} through the deliberate "
                f"path (would silently fall back to RTN). Increase the calib "
                f"set or module_token_budget."
            )
        mod_dtype = getattr(mod, pn).dtype
        if experts_qname not in derived_by_module:
            # The router forward inside derive_per_expert_activations runs in
            # the model's compute dtype (bf16); X was stored as fp32 for the
            # reservoir, so cast it back to the experts module dtype before
            # routing. The per-expert activations are re-promoted to fp32 by
            # render_production_weight's GPTQ path.
            derived_by_module[experts_qname] = derive_per_expert_activations(
                mod, X.to(device=device, dtype=mod_dtype), parent,
                capture_down=True,
                max_rows_per_expert=max_rows_per_expert,
            )
        derived = derived_by_module[experts_qname]
        if gate_module_acts and experts_qname not in gate_derived_by_module:
            # Cross-domain gate rows: same routing derivation, capped at the
            # eval budget — the gate only judges, it never fits.
            Xg = gate_module_acts.get(experts_qname)
            gate_derived_by_module[experts_qname] = (
                derive_per_expert_activations(
                    mod, Xg.to(device=device, dtype=mod_dtype), parent,
                    capture_down=True,
                    max_rows_per_expert=eval_rows_per_expert,
                )
                if Xg is not None and Xg.numel() > 0 else None
            )
        gate_derived = gate_derived_by_module.get(experts_qname)
        packed_param = getattr(mod, pn).detach().float()  # [E, out, in]
        E = packed_param.shape[0]
        proj_split = _split_packed_expert_tensor(packed_param, pn, profile)
        is_down = len(proj_split) == 1
        acts_key = "down" if is_down else "gate_up"
        per_expert_acts = derived[acts_key]
        row_counts = derived["row_counts"]

        # Per-param activation max_abs for the export's input_global_scale
        # (W4A4 needs the calibrated activation clip; without it experts ship
        # the 1.0 placeholder — Codex blocker). gate/up share the module input
        # X (measured uncapped over the full captured reservoir); down sees the
        # post-SwiGLU intermediate. One scale per packed param, applied to all
        # experts (the fused MoE kernel quantizes the input once).
        if is_down:
            param_max_abs = 0.0
            for e in range(E):
                a = per_expert_acts[e] if e < len(per_expert_acts) else None
                if a is not None and a.numel() > 0:
                    param_max_abs = max(
                        param_max_abs, float(a.detach().abs().max().item()))
        else:
            param_max_abs = float(X.detach().abs().max().item())
        # First calibrated scale wins per qname: activation_max_abs is keyed
        # per-param (shared across formats), so a later render of ANOTHER
        # format (M4 lazy FP8 gap-fill) must not clobber the scale the eager
        # NVFP4 rung calibrated, measured under, and will ship. A fresh build
        # (no prior scale, no sidecar) always writes.
        if param_max_abs > 0 and cache.activation_max_abs.get(full) is None:
            cache.activation_max_abs[full] = param_max_abs
            _persist_expert_sidecar()

        # Resume with a missing scale and/or a missing render score: the shard
        # is already on disk, so score the bytes it holds. Never re-render.
        if shard_exists:
            if key not in weights and fname is not None:
                weights[key] = fname
            if need_score:
                stored_value = weights.get(key)
                resumed_render = (
                    stored_value
                    if isinstance(stored_value, torch.Tensor)
                    else torch.load(
                        cache._path_for_value(stored_value),
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                packed_score_records[score_key] = (
                    _packed_expert_render_score_record(
                        qname=full,
                        fmt=fmt,
                        render_format=fmt,
                        reference=packed_param,
                        rendered=resumed_render,
                        per_expert_acts=per_expert_acts,
                        activation_max_abs=cache.activation_max_abs.get(full),
                        eval_rows=eval_rows_per_expert,
                        device=device,
                        score_rows_source="fit_corpus_tail",
                        input_global_scale_policy=packed_score_policy,
                    )
                )
                del resumed_render
            packed_gate_records[gate_key] = _packed_expert_render_gate_record(
                qname=full,
                fmt=fmt,
                render_format=fmt,
                render_mode=render_mode,
                batched=False,
                resumed=True,
                coverage_record=(
                    coverage.get(full) or existing_coverage_records.get(full)
                ),
            )
            continue

        # Per-expert global: for split gate/up the joint is max over slices so
        # gate and up share one scale (matches the export packer); for down the
        # per-expert global stands alone.
        per_expert_global: list[torch.Tensor | None] = [None] * E
        for e in range(E):
            cands = [
                compute_nvfp4_global_real(sp[e].float(), group_size=16)
                for _, sp in proj_split
            ]
            per_expert_global[e] = (
                torch.stack(cands).max() if len(cands) > 1 else cands[0]
            )

        empties = sum(
            1 for e in range(E)
            if e >= len(per_expert_acts)
            or per_expert_acts[e] is None
            or per_expert_acts[e].numel() == 0
        )

        rtn_fallbacks = 0
        heldout_reverts = 0
        cross_gated = 0
        use_batched = (render_mode == "batched" and fmt == "NVFP4")
        if use_batched:
            from prismaquant.export_batched_gptq import (
                gptq_obs_rounding_nvfp4_batched,
            )
            from prismaquant.export_native_compressed import _rtn_dequant_nvfp4
            src = packed_param.to(device=device, dtype=torch.float32)
            hidden_in = packed_param.shape[2]
            # Split each expert's routed rows into a RENDER set (fit GPTQ) and a
            # HELD-OUT eval set (decide GPTQ-vs-RTN). The held-out gate makes the
            # render provably non-regressive vs RTN on data the expert was not
            # fit to — catching the rank-deficient-Hessian overfit where
            # in-sample GPTQ always "wins" but generalizes worse than RTN
            # (observed on down_proj at low row counts). When a cross-domain
            # gate corpus is available for this expert, the eval set comes from
            # THAT corpus (and GPTQ fits on all fit-corpus rows) — a same-domain
            # holdout cannot catch calibration-DOMAIN overfit, which is the
            # failure the 2026-06-09 served A/B actually exhibited. Empty
            # experts (no rows) get source RTN, never the batched core's
            # all-zero dead-column output.
            per_expert_gw = derived.get("gate_weights")
            gate_acts_lists = (
                gate_derived[acts_key] if gate_derived is not None else None
            )
            gate_gw_lists = (
                gate_derived.get("gate_weights")
                if gate_derived is not None else None
            )
            render_acts: list[torch.Tensor] = []
            eval_acts: list[torch.Tensor | None] = []
            eval_gw: list[torch.Tensor | None] = []
            for e in range(E):
                a = per_expert_acts[e] if e < len(per_expert_acts) else None
                gw = (
                    per_expert_gw[e]
                    if per_expert_gw is not None and e < len(per_expert_gw)
                    else None
                )
                xa = None
                xw = None
                if gate_acts_lists is not None and e < len(gate_acts_lists):
                    cand = gate_acts_lists[e]
                    if cand is not None and cand.numel() > 0:
                        xa = cand
                        if gate_gw_lists is not None and e < len(gate_gw_lists):
                            xw = gate_gw_lists[e]
                if a is None or a.numel() == 0:
                    render_acts.append(
                        torch.zeros((0, hidden_in), device=device,
                                    dtype=torch.float32))
                    eval_acts.append(None)
                    eval_gw.append(None)
                    continue
                a = a.to(device=device, dtype=torch.float32)
                if xa is not None:
                    # Cross-domain do-no-harm: fit on ALL fit-corpus rows,
                    # judge on the disjoint gate corpus's routed rows.
                    render_acts.append(a)
                    eval_acts.append(
                        xa.to(device=device, dtype=torch.float32))
                    eval_gw.append(
                        xw.to(device=device, dtype=torch.float32).pow(2)
                        if xw is not None else None
                    )
                    cross_gated += 1
                    continue
                n = int(a.shape[0])
                n_eval = min(eval_rows_per_expert, n // 5) if n >= 20 else 0
                if n_eval > 0:
                    render_acts.append(a[:n - n_eval])
                    eval_acts.append(a[n - n_eval:])
                    # Router-weight the held-out objective: served MoE scales
                    # each token's expert output by its top_k gate weight, so
                    # the row's output error² scales by gate_weight².
                    eval_gw.append(
                        gw[n - n_eval:].to(
                            device=device, dtype=torch.float32).pow(2)
                        if gw is not None else None
                    )
                else:
                    render_acts.append(a)
                    eval_acts.append(None)  # too few rows to hold out
                    eval_gw.append(None)
            overrides = torch.stack([
                per_expert_global[e].to(device=device, dtype=torch.float32)
                for e in range(E)
            ])
            rendered = gptq_obs_rounding_nvfp4_batched(
                src, render_acts, group_size=16,
                global_real_overrides=overrides,
                static_act_order=False,
                joint_scale_opt=False,
            )  # fp32 [E,out,in]; held-out do-no-harm below, store casts to bf16
            for e in range(E):
                w_rtn = _rtn_dequant_nvfp4(
                    src[e], group_size=16, global_real_override=overrides[e],
                )
                src_a = (
                    per_expert_acts[e] if e < len(per_expert_acts) else None
                )
                if src_a is None or src_a.numel() == 0:
                    # empty expert: no activation evidence -> source RTN
                    rendered[e] = w_rtn
                    rtn_fallbacks += 1
                    continue
                ev = eval_acts[e]
                # held-out gate when available, else in-sample render set
                gate_acts = ev if ev is not None else render_acts[e]
                row_w = eval_gw[e] if ev is not None else None
                if gate_acts is None or gate_acts.numel() == 0:
                    rendered[e] = w_rtn
                    rtn_fallbacks += 1
                    continue
                s_gptq = score_render_error(
                    src[e], rendered[e], gate_acts, row_weights=row_w)
                s_rtn = score_render_error(
                    src[e], w_rtn, gate_acts, row_weights=row_w)
                if s_rtn <= s_gptq:
                    rendered[e] = w_rtn
                    rtn_fallbacks += 1
                    if ev is not None:
                        heldout_reverts += 1
            # Free the render-phase GPU transients before the CPU store copy
            # below: `src` alone is 18G fp32 for a GLM-class gate_up stack and
            # would otherwise survive into the NEXT param's render (peak
            # doubling on the 118G unified pool).
            if E > 0:
                del w_rtn
            del src, overrides, render_acts, eval_acts, eval_gw
        else:
            # Per-expert full-stack render (fmt != NVFP4, or the A/B mode).
            rendered = torch.empty_like(packed_param)
            stack_cw = None if col_weights is None else col_weights.get(full)
            for e in range(E):
                w_e = packed_param[e]  # [out, in]
                acts_e = (
                    per_expert_acts[e] if e < len(per_expert_acts) else None
                )
                has_acts = acts_e is not None and acts_e.numel() > 0
                w_dq = render_production_weight(
                    w_e, fmt,
                    qname=f"{full}.e{e}",
                    activations=(
                        {f"{full}.e{e}": acts_e.to(device)} if has_acts else {}
                    ),
                    levers=levers,
                    joint_global_real=per_expert_global[e],
                    col_weights=_expert_col_weights(stack_cw, e, int(E)),
                )
                rendered[e] = w_dq.to(rendered.dtype)
                del w_dq
            rendered = rendered.to(getattr(mod, pn).dtype)

        pos_rows = sorted(r for r in row_counts if r > 0)
        coverage_record = {
            "fmt": fmt,
            "n_experts": int(E),
            "min_rows": int(pos_rows[0]) if pos_rows else 0,
            "median_rows": int(pos_rows[len(pos_rows) // 2]) if pos_rows else 0,
            "empty_experts": int(empties),
            "rtn_fallbacks": int(rtn_fallbacks),
            "heldout_reverts": int(heldout_reverts),
            "gptq_experts": int(E - rtn_fallbacks),
            "render_mode": render_mode,
            "gate_mode": (
                "cross-domain" if (use_batched and gate_derived is not None)
                else "in-domain-holdout"
            ),
            "cross_gated_experts": int(cross_gated),
            "rendered": True,
            "had_activations": True,
        }
        # Score every packed expert honestly, on the tensors already
        # materialized here — the source param and the render that is about to
        # be stored. No re-render, and the record covers the whole stack.
        packed_score_records[score_key] = _packed_expert_render_score_record(
            qname=full,
            fmt=fmt,
            render_format=fmt,
            reference=packed_param,
            rendered=rendered,
            per_expert_acts=per_expert_acts,
            activation_max_abs=cache.activation_max_abs.get(full),
            eval_rows=eval_rows_per_expert,
            device=device,
            score_rows_source="fit_corpus_tail",
            input_global_scale_policy=packed_score_policy,
        )
        packed_gate_records[gate_key] = _packed_expert_render_gate_record(
            qname=full,
            fmt=fmt,
            render_format=fmt,
            render_mode=render_mode,
            batched=bool(use_batched),
            resumed=False,
            coverage_record=coverage_record,
        )
        _store_rendered_weight_entry(
            weights=weights,
            cache_dir_path=cache_dir_path,
            qname=full,
            fmt=fmt,
            tensor=rendered,
            weight_dtype=getattr(mod, pn).dtype,
        )
        coverage[full] = coverage_record
        del rendered, packed_param, proj_split
        # Free this module's GPU-resident derived activations + captured X once
        # its last param is rendered (bounds peak to ~one module's working set).
        if idx == last_idx_by_module.get(experts_qname):
            derived = None
            gate_derived = None
            derived_by_module.pop(experts_qname, None)
            module_acts.pop(experts_qname, None)
            gate_derived_by_module.pop(experts_qname, None)
            gate_module_acts.pop(experts_qname, None)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if progress:
            print(
                f"[prod-cache/experts] {idx + 1}/{len(in_scope)} {full} @ {fmt} "
                f"E={E} empty={empties} rtn_fallback={rtn_fallbacks} "
                f"(heldout_reverts={heldout_reverts}, "
                f"cross_gated={cross_gated}) gptq={E - rtn_fallbacks} "
                f"min_rows={coverage[full]['min_rows']} "
                f"({_time.monotonic() - t0:.1f}s elapsed)",
                flush=True,
            )

    _finalize_packed_expert_cache_metadata(
        cache,
        packed_scope_names=all_packed_fullnames,
        score_records=packed_score_records,
        gate_records=packed_gate_records,
        coverage=coverage,
        cache_dir_path=cache_dir_path,
    )

    if progress:
        print(
            f"[prod-cache/experts] rendered {len(coverage)} packed-expert "
            f"tensors in {_time.monotonic() - t0:.1f}s",
            flush=True,
        )
    return coverage
