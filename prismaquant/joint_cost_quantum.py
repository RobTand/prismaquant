"""Stage B of the distributed joint-AURA cost campaign: the layer quantum.

``python3 -m prismaquant.joint_cost_quantum`` -- one PB action per layer
(contract ``docs/design/distributed_campaign_2026-09-19.md`` §6): verify the
quantum record's four digest-checked inputs (exit 3, nothing written, on any
mismatch), load the same head inputs the single consumer loads, rebuild the
incoming cotangent at boundary L+1 by chaining at most S-1 render-free source
backwards from the nearest strided adjoint checkpoint, then replay this
layer's sealed retained windows through ``ProductionWeightCache`` in sealed
window order -- the same kernels, the same call order, the same
within-layer FP accumulation the single run performs -- committing each
finished unit to this quantum's own identity-bound journal and reporting
accepted progress at chunk granularity.

The numerics are reused, not reimplemented: the per-window replay is
``joint_statistics_replay.observe_and_project_retained_windows`` (the single
run's retained reverse step), the chain leg is
``joint_adjoint_checkpoints.render_free_layer_roll`` (the completed-layer leg
of the single run's reverse walk), boundaries flow through the existing
``StreamedBoundaryArtifacts`` reader, renders through the PWC's verified
loads, and journals through ``aura_cost``'s checkpoint grammar. This module
orchestrates those seams for one layer and owns nothing else -- no second
cache, no cross-layer state, no writes outside
``<output_root>/layer-quanta/layer-NNN/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import pickle
import socket
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import torch

from .cost_stage_checkpoint import atomic_write_bytes, canonical_json_sha256
from .cost_currency import probe_identity_seals, probe_identity_walls_differ
from .dev_mode import seal_check
from .io_spans import IoSpanLog, failure_outcome, read_proc_io, stage_span_log
from .joint_adjoint_checkpoints import (
    QUANTUM_COUNTERS_SCHEMA,
    QUANTUM_RECORD_SCHEMA,
    QUANTUM_STATUS_SCHEMA,
    GpuPowerSampler,
    KernelTimeProfiler,
    adjoint_space,
    boundary_entry_directory,
    chain_layers_for,
    _chain_group_batch,
    _require_per_sample_state,
    _stack_to_device,
    dev_mode_stamp,
    load_adjoint_checkpoint,
    require_dev_mode,
)
from .joint_adjoint_slices import (
    AdjointSliceRefused,
    ChainRegimeRefused,
    adjoint_slice_sha256,
    chain_regime_of,
    load_adjoint_slice,
    slice_run_header,
    verify_adjoint_slice,
)
from .joint_layer_quanta import (
    CHECKPOINT_LOAD_PHASE,
    PREPARED_INPUT_SCHEMA,
    check_adjoint_run_identity,
    check_prepared_windows_against_resolved,
    executable_bound_phase_name,
    executable_own_source_phase_name,
    executable_render_phase_name,
    executable_replay_phase_name,
    executable_spill_phase_name,
    executable_source_phase_name,
    qname_layer,
    quantum_source_layer_order,
)
from .source_read_plan import chain_prefetch_window
from .joint_quantum_handoff import (
    HANDOFF_LOAD_PHASE,
    HandoffEmitter,
    QuantumHandoffRefused,
    bind_handoff_publication,
    load_handoff_inputs,
    load_quantum_handoff,
    require_band_serial_readset,
)

#: Exit codes (§6.2/§6.4): 3 is the identity refusal -- nothing written; 4 is
#: the clean gap (status.json says gapped; PB retries the sealed action key).
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_IDENTITY_REFUSED = 3
EXIT_GAPPED = 4

IDENTITY_REFUSED_MARKER = "quantum_identity_refused"

#: Opt in to Stage B's kernel-time sessions, one around the chain and one
#: around each retained window (PQ #1029). Off by default, as Stage A's are
#: (#899): each session's close sums every CUDA kernel it recorded through
#: ``key_averages``, and on a GLM-shaped proxy that close was 15.6 s of one
#: quantum's main thread. The GPU power sampler stays on. ``1`` restores the
#: kernel-time sum.
KERNEL_PROFILE_ENV = "PRISMAQUANT_STAGE_B_KERNEL_PROFILE"
KERNEL_PROFILE_NOT_MEASURED = (
    "not measured: each torch.profiler session's close sums every kernel it "
    f"recorded (PQ #1029); {KERNEL_PROFILE_ENV}=1 opts in")


def _stage_b_kernel_profiler() -> KernelTimeProfiler:
    """A Stage B kernel-time profiler: not measured unless asked for."""
    if os.environ.get(KERNEL_PROFILE_ENV) == "1":
        return KernelTimeProfiler()
    return KernelTimeProfiler(not_measured=KERNEL_PROFILE_NOT_MEASURED)


class QuantumIdentityRefused(RuntimeError):
    """A digest, schema or binding mismatch: refuse before writing anything."""


def require_slice_bf16_reduction(adjoint_slice, allow: bool, *, where: str) -> None:
    """Refuse a quantum whose bf16 reduction flag differs from Stage A's (#1065).

    The quantum rebuilds Stage A's chain from its checkpoint, so it must run
    that chain under the arithmetic Stage A ran it under. The slice's run
    identity carries ``allow_bf16_reduced_precision_reduction: false`` when
    Stage A ran with the flag off and nothing when it ran at PyTorch's
    default (PQ #1028). ``allow`` is this quantum's own setting. Certified
    mode refuses a mismatch and names both settings; dev mode records it,
    as the prepared implementation-digest check does.
    """
    from .dev_mode import dev_mode_enabled, dev_warning
    from .matmul_arithmetic import (
        BF16_REDUCTION_ENV, BF16_REDUCTION_FIELD, MatmulArithmeticRefused,
        bf16_reduction_of)

    identity = adjoint_slice.get("run_identity") if isinstance(adjoint_slice, dict) else None
    if not isinstance(identity, dict):
        raise QuantumIdentityRefused(f"{where}: the stage-A slice carries no run identity")
    try:
        stamped = bf16_reduction_of(identity)
    except MatmulArithmeticRefused as exc:
        raise QuantumIdentityRefused(
            f"{where}: the stage-A slice's bf16 reduction stamp: {exc}") from exc
    if stamped == bool(allow):
        return

    def spelled(value):
        return f"{BF16_REDUCTION_ENV} unset" if value else f"{BF16_REDUCTION_ENV}=off"

    message = (f"{where}: Stage A ran its chain with {BF16_REDUCTION_FIELD}={stamped} "
               f"({spelled(stamped)}), this quantum runs with "
               f"{BF16_REDUCTION_FIELD}={bool(allow)} ({spelled(allow)})")
    if dev_mode_enabled():
        dev_warning(f"{message}; recorded, not gated (dev mode)")
        return
    raise QuantumIdentityRefused(message)


# --------------------------------------------------------------------------
# Record verification (§6.2 step 1)
# --------------------------------------------------------------------------


def _digest_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_hex(value: object, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise QuantumIdentityRefused(f"{label} is not a 64-hex sha256: {text!r}")
    return text


def verify_quantum_identity(
    *,
    quantum_path: Path, quantum_sha256: str,
    plan_path: Path, plan_sha256: str,
    prepared_path: Path, prepared_sha256: str,
    adjoint_path: Path, adjoint_sha256: str,
    output_root: Path,
) -> tuple[dict, dict]:
    """Verify the four digests and the record's campaign binding.

    ``adjoint_path`` is the quantum's stage-A slice (PQ #993), never a whole
    receipt: the file is the slice's canonical JSON, so its digest is the
    ``slice_sha256`` the record binds.

    Returns ``(record, adjoint_slice)``; raises :class:`QuantumIdentityRefused`
    (the caller exits 3) with nothing written. The producer's
    ``check_quantum_for_campaign`` runs too when the producer module has
    landed; until then the same checks run here from the record's own fields,
    so a record built for another campaign revision refuses the same way.
    """
    try:
        _require_hex(quantum_sha256, "--quantum-sha256")
        if _digest_of(quantum_path) != quantum_sha256:
            raise QuantumIdentityRefused(
                f"quantum record digest mismatch at {quantum_path}")
        record = json.loads(Path(quantum_path).read_text())
        if record.get("schema") != QUANTUM_RECORD_SCHEMA:
            raise QuantumIdentityRefused(
                f"quantum record schema mismatch: {record.get('schema')!r}")
        body = {key: value for key, value in record.items()
                if key != "identity_sha256"}
        recomputed = canonical_json_sha256(body, where="quantum record identity")
        if recomputed != record.get("identity_sha256"):
            raise QuantumIdentityRefused(
                "quantum record identity_sha256 does not bind its own bytes: "
                f"stored={record.get('identity_sha256')!r} recomputed={recomputed}")
        layer = int(record["layer"])
        if record.get("quantum_id") != f"layer-{layer:03d}":
            raise QuantumIdentityRefused(
                f"quantum_id {record.get('quantum_id')!r} does not name layer {layer}")
        campaign = record["campaign"]
        for label, supplied, path in (
                ("plan", plan_sha256, plan_path),
                ("prepared", prepared_sha256, prepared_path)):
            _require_hex(supplied, f"--{label}-sha256")
            if _digest_of(path) != supplied:
                raise QuantumIdentityRefused(f"{label} digest mismatch at {path}")
            # A run seal (PQ #1147): dev mode prints the record's binding and
            # runs under the supplied, digest-checked file.
            seal_check(f"quantum record {label}", campaign.get(f"{label}_sha256"), supplied,
                       where=str(quantum_path),
                       refusal=QuantumIdentityRefused(
                           f"quantum record binds another {label}: "
                           f"record={campaign.get(f'{label}_sha256')!r} argv={supplied}"))
        _require_hex(campaign.get("read_manifest_sha256"),
                     "record read_manifest_sha256")
        adjoint = record["adjoint"]
        if adjoint.get("receipt_sha256") is not None:
            raise QuantumIdentityRefused(
                "quantum record binds a whole stage-A receipt; a quantum reads "
                "only its slice (PQ #993): re-seal the record")
        if adjoint.get("slice_sha256") is None:
            raise QuantumIdentityRefused(
                "quantum record is unbound (pre-stage-A): re-seal it against "
                "its stage-A slice with bind_adjoint_slice -- a new "
                "identity, never an edit (producer D3) -- before publishing")
        _require_hex(adjoint.get("slice_sha256"), "record adjoint slice_sha256")
        _require_hex(adjoint_sha256, "--adjoint-slice-sha256")
        if adjoint_sha256 != adjoint["slice_sha256"]:
            raise QuantumIdentityRefused(
                "quantum record binds another stage-A slice: "
                f"record={adjoint['slice_sha256']!r} argv={adjoint_sha256}")
        try:
            adjoint_slice = load_adjoint_slice(
                adjoint_path, adjoint_sha256, layer=layer,
                checkpoint_boundary=int(adjoint["checkpoint_boundary"]))
        except AdjointSliceRefused as exc:
            raise QuantumIdentityRefused(f"stage-A slice refused: {exc}") from exc
        # The slice's run header must answer for this campaign: its plan,
        # prepared and scope, or -- for a catalog extension -- the original
        # run the extension binds (checked by the extension owner).
        header = slice_run_header(adjoint_slice)
        try:
            # load_adjoint_slice above checked that the header's stride
            # places this layer at the record's checkpoint.
            check_adjoint_run_identity(
                header, plan_sha256=plan_sha256, prepared_sha256=prepared_sha256,
                scope=campaign["campaign_scope"],
                catalog_extension=record.get("catalog_extension"))
        except ValueError as exc:
            raise QuantumIdentityRefused(
                f"stage-A slice does not answer for this campaign: {exc}") from exc
        chunks = record["chunks"]
        total = int(record["read_set"]["total_bytes"])
        cursor = 0
        for chunk in chunks:
            start, end = int(chunk["start_bytes"]), int(chunk["end_bytes"])
            if start != cursor or end <= start:
                raise QuantumIdentityRefused(
                    f"quantum chunk table does not tile [0, {total}): {chunk!r}")
            cursor = end
        if cursor != total:
            raise QuantumIdentityRefused(
                f"quantum chunk table ends at {cursor}, not {total}")
        expected_chain = chain_layers_for(int(adjoint["checkpoint_boundary"]), layer)
        if tuple(int(c) for c in adjoint["chain_layers"]) != expected_chain:
            raise QuantumIdentityRefused(
                "quantum chain_layers do not match its checkpoint boundary: "
                f"{adjoint['chain_layers']!r} vs {expected_chain!r}")
        seen: set[str] = set()
        for position, window in enumerate(record["windows"]):
            # D2 (producer PR #785): the record seals ordered window indices
            # only -- per-window names and byte sizes are recomputed at
            # runtime from the sealed budget (resolve_quantum_windows) because
            # statistics bytes need module geometry the producer cannot see.
            # What is checked here is the index order itself: 0..n-1, sealed
            # order, no gaps, no repeats. Extra keys (e.g. advisory names in
            # dispatcher/joiner fixtures) are ignored here and cross-checked
            # at resolution, never trusted.
            if (not isinstance(window, Mapping)
                    or window.get("window_index") != position):
                raise QuantumIdentityRefused(
                    f"quantum windows do not seal the ordered index slice "
                    f"0..{len(record['windows']) - 1}: {window!r} at position "
                    f"{position}")
            for name in window.get("names", ()):
                if name in seen:
                    raise QuantumIdentityRefused(
                        f"quantum windows repeat unit {name!r}")
                seen.add(name)
        expected_root = Path(output_root) / "layer-quanta" / f"layer-{layer:03d}"
        if Path(record["output_space"]["root"]).resolve() != expected_root.resolve():
            raise QuantumIdentityRefused(
                f"quantum output_space.root {record['output_space']['root']} "
                f"is not {expected_root}")
        try:
            from .joint_layer_quanta import check_quantum_for_campaign
        except ImportError:
            pass
        else:  # pragma: no cover - producer-owned check once it lands
            try:
                check_quantum_for_campaign(record, {
                    "plan_sha256": plan_sha256,
                    "prepared_sha256": prepared_sha256,
                    "read_manifest_sha256": campaign["read_manifest_sha256"],
                    "unit_roster_sha256": campaign["unit_roster_sha256"],
                    "campaign_scope": campaign["campaign_scope"],
                    "adjoint_slice_sha256": adjoint_slice_sha256(adjoint_slice),
                })
            except ValueError as exc:
                raise QuantumIdentityRefused(
                    f"producer campaign check refuses: {exc}") from exc
    except QuantumIdentityRefused:
        raise
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise QuantumIdentityRefused(
            f"quantum record is not a valid {QUANTUM_RECORD_SCHEMA} document: {exc}") from exc
    return record, adjoint_slice


_HEX64 = re.compile(r"[0-9a-f]{64}")


def chain_readset_sha256(record: Mapping) -> str:
    """The data-manifest digest a chain-mode row stages, as its record seals it.

    The dispatcher's rule, spelled once more here because the dispatcher
    cannot be imported on the GPU side (``tools/dispatch_joint_quanta.py``:
    ``_row_manifest_sha256`` and the ``executable is not None`` branch of
    ``quantum_argv``). An executable row stages its executable readset; any
    other row stages its slice manifest. A present but malformed executable
    block refuses rather than fall back to the slice manifest, exactly as
    the dispatcher refuses to publish such a row.
    """
    executable = record.get("executable_readset")
    if executable is not None:
        digest = (executable.get("manifest_sha256")
                  if isinstance(executable, Mapping) else None)
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise QuantumIdentityRefused(
                "the record's executable readset seals no manifest digest")
        return digest
    read_set = record.get("read_set")
    digest = (read_set.get("manifest_sha256")
              if isinstance(read_set, Mapping) else None)
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise QuantumIdentityRefused("the record's read set seals no manifest digest")
    return digest


def require_chain_readset(record: Mapping, *, data_manifest_sha256) -> None:
    """Refuse unless a chain-mode row stages the manifest its record seals.

    The chain-mode twin of ``require_band_serial_readset`` (PQ #1008): the
    staged manifest is the one the dispatcher derives from the sealed
    record, so any other digest, or none, names a row that stages other
    bytes than the record declares. It would also fail at its first
    unstaged read; this names the cause before any read.
    """
    if not isinstance(data_manifest_sha256, str) or \
            not _HEX64.fullmatch(data_manifest_sha256):
        raise QuantumIdentityRefused(
            "a chain-mode quantum needs --data-manifest-sha256: the staged "
            "data manifest is the one its record seals")
    if data_manifest_sha256 != chain_readset_sha256(record):
        raise QuantumIdentityRefused(
            "the staged data manifest is not the readset the record seals")


# --------------------------------------------------------------------------
# The chunk frontier: windows completed -> the chunk phase being read
# --------------------------------------------------------------------------


class ChunkFrontier:
    """Map completed sealed windows onto the record's chunk byte ranges.

    The slice's non-render share (the head phase's source extents and head
    inputs) is the frontier's starting offset; each completed window advances
    it by its declared render-byte upper bound. Chunk phases are entered when
    the frontier crosses their start byte -- the same currency the promotion
    window advances on, reported at chunk granularity (§5.2/§6.2 step 5).
    """

    def __init__(self, *, chunks, windows):
        self._starts = [int(chunk["start_bytes"]) for chunk in chunks]
        self._names = [str(chunk["name"]) for chunk in chunks]
        total = int(chunks[-1]["end_bytes"]) if chunks else 0
        render = sum(int(window["render_file_upper_bound_bytes"]) for window in windows)
        self._frontier = max(0, total - render)
        self.index = 0 if chunks else -1

    def window_done(self, window: Mapping) -> None:
        self._frontier += int(window["render_file_upper_bound_bytes"])
        while (self.index + 1 < len(self._starts)
               and self._frontier >= self._starts[self.index + 1]):
            self.index += 1

    @property
    def phase_name(self) -> str | None:
        return self._names[self.index] if 0 <= self.index < len(self._names) else None


# --------------------------------------------------------------------------
# Progress: accepted commits at chunk granularity (§6.2 step 5)
# --------------------------------------------------------------------------


class QuantumProgress:
    """Cumulative durable units under the declared head/chunk phase names.

    The currency is the one the single run reports: units are cost rows
    journalled plus whatever the head walk committed before them, committed
    only once the journal shard is on disk.
    """

    def __init__(self, *, frontier: ChunkFrontier, base_units=0, log=None):
        from .joint_run_progress import declared_phases

        self._phases = declared_phases()
        self._log = log or (lambda message: print(message, flush=True))
        self._base_units = int(base_units)
        self._units = 0
        self._committed = (None, None)
        self._phase = None
        self._phase_index = -1
        self._frontier = frontier
        self._undeclared: set[str] = set()
        self.commits = 0

    def _enter(self, name: str) -> None:
        if self._phases is None:
            return
        try:
            index = self._phases.index(str(name))
        except ValueError:
            # Chunk names are not staging phases on an executable row
            # (joint_layer_quanta.quantum_executable_phase_names), so this
            # fires once per chunk there. commit() keeps reporting under the
            # phase already entered; say that, once per name (PQ #1187).
            if str(name) not in self._undeclared:
                self._undeclared.add(str(name))
                where = (f"units commit under {self._phase!r}" if self._phase
                         else "nothing commits until a declared phase is entered")
                self._log(f"quantum progress: phase {name!r} is not one this "
                          f"action declared; {where}")
            return
        if index > self._phase_index:
            self._phase_index = index
            self._phase = str(name)

    def enter_head(self, units: int) -> None:
        self._enter("head")
        self.priced(units)
        self.commit()

    def enter_read_phase(self, name: str) -> None:
        """Report a staging read phase without pricing new units.

        Read transitions (checkpoint load, layer install, probe passes)
        advance the phase the tier loop stages ahead of, while the
        committed durable-unit count only moves when journalled work
        lands. Undeclared names commit nothing, as everywhere here.
        """
        self._enter(name)
        self.commit()

    def units(self) -> int:
        return self._base_units + self._units

    def priced(self, count: int) -> None:
        if int(count) > self._units:
            self._units = int(count)

    def window_done(self, window: Mapping) -> None:
        self._frontier.window_done(window)
        if self._frontier.phase_name is not None:
            self._enter(self._frontier.phase_name)

    def commit(self) -> bool:
        """Report the cumulative count under the current phase (inert no-op)."""
        from .prismabuild_progress import report

        if self._phase is None:
            return False
        advanced = (self._committed == (None, None)
                    or self.units() > self._committed[0]
                    or self._phase != self._committed[1])
        if not advanced:
            return False
        if report(self._phase, self.units(), unit="quantum_units"):
            self.commits += 1
        self._committed = (self.units(), self._phase)
        return True


# --------------------------------------------------------------------------
# Counters (§8.1)
# --------------------------------------------------------------------------


def quantum_io_spans(quantum_id, sampler) -> IoSpanLog:
    """The quantum's span log: ``/proc/self/io``, residency tiers and GPU watts."""
    return stage_span_log(str(quantum_id), power_sampler=sampler)


class QuantumCounters:
    """Rob's two metrics, per chunk phase and per window (§8.1).

    ``bytes_from_ram``/``bytes_from_stage``/``bytes_from_pool`` are deltas of
    the existing residency-map accounting, snapshotted at phase and window
    boundaries; GPU energy is the 1 Hz power sampler plus profiler kernel
    time. Utilization percentages are never recorded: on GB10 they are
    non-diagnostic (AGENTS.md principle 13).
    """

    def __init__(self, *, quantum_id, identity_sha256, chunks, frontier: ChunkFrontier,
                 io_spans: IoSpanLog | None = None, sampler=None,
                 started: float | None = None):
        from .residency_map import residency_report

        self._report = residency_report
        self._frontier = frontier
        self.quantum_id = str(quantum_id)
        self.identity_sha256 = str(identity_sha256)
        # A caller that started the power sampler and the span log before
        # the head (``run_layer_quantum``) passes both, with the time it
        # started them, so wall time and joules cover the same interval.
        self.started = time.time() if started is None else float(started)
        self.sampler = GpuPowerSampler().start() if sampler is None else sampler
        self.io = io_spans if io_spans is not None else quantum_io_spans(
            self.quantum_id, self.sampler)
        self.total_kernel_active_s = 0.0
        self._kernel_error = None
        self.phases = [{"name": str(chunk["name"]),
                        "start_bytes": int(chunk["start_bytes"]),
                        "end_bytes": int(chunk["end_bytes"]),
                        "entered_unix": None, "units_at_entry": None,
                        "bytes_from_ram": 0, "bytes_from_stage": 0,
                        "bytes_from_pool": 0}
                       for chunk in chunks]
        self.windows: list[dict] = []
        self.chain = {"layers": 0, "backwards": 0, "wall_s": 0.0, "kernel_active_s": 0.0}
        # How the target layer's statistics were replayed (PQ #994). The
        # mode is telemetry only: both modes produce the same records.
        # ``layer_passes`` counts full target-layer forward/backward passes.
        self.replay = {"mode": "windowed", "layer_passes": 0,
                       "noncontiguous_cotangent_seeds": 0}
        self._phase_cursor = None
        self._window_cursor = None

    def _snapshot(self):
        report = self._report()
        if report is None:
            return None
        return (int(report.get("bytes_from_ram", 0)),
                int(report.get("bytes_from_stage", 0)),
                int(report.get("bytes_from_pool", 0)),
                int(report.get("fallback_count", 0)) + int(report.get("ram_fallback_count", 0)))

    @staticmethod
    def _delta(later, earlier):
        if later is None or earlier is None:
            return (0, 0, 0, 0)
        return tuple(a - b for a, b in zip(later, earlier))

    def open(self) -> None:
        self._phase_cursor = self._snapshot()

    def enter_phase(self) -> None:
        """Close the current chunk phase's tier delta and open the next."""
        now = self._snapshot()
        index = self._frontier.index
        if self._phase_cursor is not None and 0 <= index < len(self.phases):
            block = self.phases[index]
            if block["entered_unix"] is None:
                block["entered_unix"] = time.time()
            ram, stage, pool, _refused = self._delta(now, self._phase_cursor)
            block["bytes_from_ram"] += ram
            block["bytes_from_stage"] += stage
            block["bytes_from_pool"] += pool
        self._phase_cursor = now

    def mark_phase_units(self, units: int) -> None:
        index = self._frontier.index
        if 0 <= index < len(self.phases):
            block = self.phases[index]
            if block["units_at_entry"] is None:
                block["units_at_entry"] = int(units)

    def open_window(self, window_index: int, window: Mapping) -> None:
        self._window_cursor = self._snapshot()
        self.windows.append({
            "window_index": int(window_index),
            "names": list(window["names"]),
            "render_bytes_upper_bound": int(window["render_file_upper_bound_bytes"]),
            "statistics_bytes": int(window.get("statistics_bytes", 0)),
            "candidate_count": int(window.get("candidate_count", 0)),
            "bytes_from_ram": 0, "bytes_from_stage": 0, "bytes_from_pool": 0,
            "kernel_active_s": None, "wall_s": None,
        })

    def close_window(self, window_index: int, *, kernel_active_s, wall_s) -> None:
        now = self._snapshot()
        block = self.windows[window_index]
        ram, stage, pool, _refused = self._delta(now, self._window_cursor)
        block["bytes_from_ram"] = ram
        block["bytes_from_stage"] = stage
        block["bytes_from_pool"] = pool
        block["kernel_active_s"] = kernel_active_s
        block["wall_s"] = wall_s
        self._window_cursor = now

    def chain_step(self, *, layers: int, backwards: int, wall_s: float,
                   kernel_active_s) -> None:
        self.chain["layers"] += int(layers)
        self.chain["backwards"] += int(backwards)
        self.chain["wall_s"] += float(wall_s)
        # None (not measured) is sticky: a partial sum is not the chain's.
        if kernel_active_s is None or self.chain["kernel_active_s"] is None:
            self.chain["kernel_active_s"] = None
        else:
            self.chain["kernel_active_s"] += float(kernel_active_s)

    def kernel_block(self, profiler: KernelTimeProfiler) -> None:
        self.total_kernel_active_s += profiler.kernel_active_s
        if profiler.error and self._kernel_error is None:
            self._kernel_error = profiler.error

    def finish(self, *, units_done: int | None, units_total: int) -> dict:
        """The counters document. ``units_done`` is ``None`` for a failed run."""
        gpu = self.sampler.stop()
        wall_s = time.time() - self.started
        kernel_active_s = (self.total_kernel_active_s
                           if self._kernel_error is None else None)
        joules = gpu.get("gpu_joules")
        final = self._snapshot()
        counters = {
            "schema": QUANTUM_COUNTERS_SCHEMA,
            "quantum_id": self.quantum_id,
            "identity_sha256": self.identity_sha256,
            "units": [None if units_done is None else int(units_done),
                      int(units_total)],
            "wall_s": wall_s,
            "kernel_active_s": kernel_active_s,
            "kernel_active_ratio": (
                kernel_active_s / wall_s
                if kernel_active_s is not None and wall_s > 0 else None),
            "gpu_joules": joules,
            "gpu_power_w_p50": gpu.get("gpu_power_w_p50"),
            "gpu_power_w_p95": gpu.get("gpu_power_w_p95"),
            "gpu_power_w_max": gpu.get("gpu_power_w_max"),
            "gpu_sampler_samples": gpu.get("sample_count"),
            "gpu_power_envelope_w": 140.0,
            "work_per_joule": {
                "units_per_kwh": (
                    units_done / (joules / 3.6e6)
                    if joules and units_done is not None else None),
            },
            "chain": dict(self.chain),
            "replay": dict(self.replay),
            "phases": self.phases,
            "windows": self.windows,
            # Every closed span, in close order (prismaquant.io_spans).
            "io_spans": list(self.io.records),
        }
        if self._kernel_error:
            counters["kernel_profiler_error"] = self._kernel_error
        if "sampler_error" in gpu:
            counters["gpu_sampler_error"] = gpu["sampler_error"]
        if final is not None:
            counters["bytes_from_ram"] = final[0]
            counters["bytes_from_stage"] = final[1]
            counters["bytes_from_pool"] = final[2]
            counters["residency_refusals"] = final[3]
        else:
            counters["bytes_from_ram"] = None
            counters["bytes_from_stage"] = None
            counters["bytes_from_pool"] = None
            counters["residency_refusals"] = None
            counters["residency_accounting"] = "absent (no residency map bound)"
        return counters


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _render_proof_sides(value, source, *, side: str) -> str:
    """Both sides of a failed prepared render proof, for its refusal (PQ #1102)."""
    live_bytes = ("n/a (meta)" if source.is_meta
                  else source.numel() * source.element_size())
    return (f"prepared shape {value.get('shape')} dtype {value.get('dtype')} "
            f"bytes {value.get('logical_bytes')}; {side} shape {list(source.shape)} "
            f"dtype {source.dtype} bytes {live_bytes}")


def _install_with_settlement(runner, layer: int, *, operator_windows,
                             order, settle_successors: bool = True) -> None:
    """Install one layer of the quantum's own walk, then prefetch its next.

    ``order`` is the quantum's install order
    (``joint_layer_quanta.quantum_source_layer_order``), and the executable
    readset declares a source phase for each of its layers (PQ #1095).

    The layer is asked for first. ``install(require_prefetched=True)``
    refuses a layer that is neither resident nor in flight, and nothing
    before the walk's first layer prefetches it; for any later layer the
    call hands back the read already in flight, or owns the resident entry
    until the install claims it (PQ #1124). After the install, the next
    ``prefetch_lookahead`` layers of the order are prefetched, and never a
    layer the quantum does not install. The single run's reverse walk
    prefetches ``layer - 1`` after every layer; here that read would be the
    next quantum's source, which this readset does not declare and the
    strict reader refuses. The context's own top-up is off for the same
    reason.

    Under operator windows, ``settle_successors`` waits for the prefetched
    layers before returning. A chain step passes ``False``: its successors'
    reads overlap the chain roll, and the consumer waits for each one only
    under the phase that stages it, when it installs a chain layer or in
    :func:`_await_own_source` (PQ #1166). The own layer's call settles an
    empty window, which refuses any prefetch still in flight.
    """
    order = tuple(order)
    position = order.index(layer)
    runner.context.schedule_prefetch(layer)
    runner.context.install(
        layer,
        require_prefetched=runner.require_prefetched_residency,
        prefetch_following=False,
    )
    successors = chain_prefetch_window(order, position, runner.prefetch_lookahead)
    for successor in successors:
        runner.context.schedule_prefetch(successor)
    if operator_windows is None or not settle_successors:
        return
    _settle_sources(runner, successors)


def _settle_sources(runner, layers) -> None:
    """Wait for the prefetches of ``layers``; refuse any other in flight."""
    settle = getattr(runner.context, "settle_prefetched_layers", None)
    if callable(settle):
        settle(tuple(layers))
    elif torch.device(runner.device).type == "cuda":
        raise RuntimeError(
            "joint operator replay requires source prefetch settlement")


def _await_own_source(runner, layer: int, *, operator_windows) -> None:
    """Wait for the quantum's own layer source under its source phase.

    The chain step prefetches the own layer and leaves the read in flight
    while the chain rolls (PQ #1166). The caller has reported
    ``own-LLL-source``, the phase that stages the layer, and calls this
    before the layer step's first memory observation, which must see no
    pending owner and no loader temporaries. With no chain, nothing has
    prefetched the layer yet, and this starts its read.
    """
    runner.context.schedule_prefetch(layer)
    if operator_windows is None:
        return
    _settle_sources(runner, (layer,))


def _rebuild_batches(runner, *, partitions, shared_pass):
    """Rebuild the reverse walk's per-batch owners from stored artifacts.

    Metadata (ids, positions, embeddings, mask) comes from the same
    ``_prepare`` call the capture ran, on the same calibration rows; the
    shared forward pass state comes from the strided checkpoint.
    ``partitions`` is the capture's own row-sliced batch list, in order.
    """
    from .cost_streaming import StreamedForwardBoundaries

    batches = []
    for batch_index, rows in enumerate(partitions):
        ids, position_ids, hidden, embeddings, mask = runner._prepare(rows)
        del hidden
        batches.append(StreamedForwardBoundaries(
            ids, position_ids, embeddings, mask, [], shared_pass.get(batch_index)))
    return batches


def _boundary_entry_record(adjoint_slice: dict, batch_index: int, boundary: int) -> dict:
    for entry in adjoint_slice["boundary_entries"][str(boundary)]:
        if entry["name"] == f"boundary-{batch_index}-{boundary}-at-{boundary}":
            return entry
    raise RuntimeError(
        f"stage-A slice has no boundary entry for batch {batch_index} "
        f"boundary {boundary}")


def record_window_indices(record: dict) -> list[int]:
    """The record's sealed window-index slice, validated (D2 handshake).

    Returns ``[0..n-1]``; raises :class:`QuantumIdentityRefused` when the
    record does not seal the ordered index slice. Per-window names and byte
    sizes are never read from the record -- see :func:`resolve_quantum_windows`.
    """
    windows = record["windows"]
    indices = [window.get("window_index") if isinstance(window, Mapping) else None
               for window in windows]
    if indices != list(range(len(windows))):
        raise QuantumIdentityRefused(
            f"quantum {record.get('quantum_id')!r} windows do not seal the "
            f"ordered index slice 0..{len(windows) - 1}: {indices!r}")
    return indices


def resolve_quantum_windows(
    record: dict, *, layer: int, names, linears, render_formats,
    production_cache, operator_windows, retained_budget, source_bytes,
) -> list[dict]:
    """Recompute this layer's sealed retained windows and handshake the record.

    Producer D2 (PR #785): the record seals ordered window *indices* only.
    Membership and footprints are recomputed here with the same preflight the
    single run falls back to (``preflight_joint_operator_admission`` --
    deterministic from the sealed budget, the installed source geometry and
    the PWC candidate files), then handshook: the record's index slice must
    be exactly ``0..n-1`` for the ``n`` recomputed windows, and advisory
    ``names`` carried beside an index must equal the recomputed membership.
    Returns the resolved windows (index, names, statistics/render bytes,
    candidate count) -- the only window facts the replay, the chunk frontier
    and the counters may use. Refuses (exit 3) on any mismatch: a stale or
    retargeted record is a new identity, never a silent repack.
    """
    from .joint_statistics_replay import preflight_joint_operator_admission

    indices = record_window_indices(record)
    windows_by_layer = preflight_joint_operator_admission(
        {int(layer): list(names)}, linears,
        {name: list(render_formats[name]) for name in names},
        production_cache, policy=operator_windows,
        retained_budget=retained_budget, source_bytes=int(source_bytes))
    admitted = list((windows_by_layer or {}).get(int(layer), ()))
    if len(admitted) != len(indices):
        raise QuantumIdentityRefused(
            f"quantum {record.get('quantum_id')!r} seals {len(indices)} "
            f"windows but the sealed budget admits {len(admitted)} for layer "
            f"{layer}: re-seal the record, refusing")
    resolved = []
    for position, window in zip(indices, admitted):
        entry = record["windows"][position]
        if "names" in entry and list(entry["names"]) != list(
                window.original_full_target_names):
            raise QuantumIdentityRefused(
                f"quantum {record.get('quantum_id')!r} window {position} "
                f"names disagree with the sealed budget: "
                f"{list(entry['names'])!r} vs "
                f"{list(window.original_full_target_names)!r}")
        resolved.append({
            "window_index": int(position),
            "names": list(window.original_full_target_names),
            "statistics_bytes": int(window.statistics_bytes),
            "render_file_upper_bound_bytes": int(window.render_file_upper_bound_bytes),
            "candidate_count": int(window.candidate_count),
        })
    return resolved


# --------------------------------------------------------------------------
# Shared seams: the roster and the retained budget, in one place
# --------------------------------------------------------------------------


def quantum_runtime_execution(config, *, replay_regime):
    """The execution block a launched quantum's core runs under.

    The plan's ``execution`` block, plus the settings the plan keeps at its
    top level: the device envelope (``max_gpu_bytes``) and the free-UMA floor
    (``min_free_gib``), which the chain roll and the retained replay check.
    Before PQ #1163 the floor was not carried, so the core read 0. The
    replay regime is the launch setting ``run_layer_quantum`` resolved.
    """
    execution = dict(config["execution"])
    execution.setdefault("device_envelope_bytes", config.get("max_gpu_bytes"))
    if "min_free_gib" in config:
        execution.setdefault("min_free_gib", config["min_free_gib"])
    execution["replay_regime"] = replay_regime
    return execution


def quantum_retained_state(execution):
    """The sealed retained budget, operator policy and source cap (§6).

    One normalization shared by window resolution (before any GPU work) and
    the replay core, so both derive the same windows from the same bytes.
    """
    from .cost_streaming import normalize_boundary_storage
    from .joint_retained_window_plan import (
        RetainedWindowBudget,
        normalize_retained_execution,
    )
    from .joint_statistics_replay import normalize_operator_windows

    operator_windows = normalize_operator_windows(execution.get("operator_windows"))
    retained_operator_windows = execution.get("retained_operator_windows")
    if retained_operator_windows is None:
        raise RuntimeError(
            "layer quantum requires the plan's retained operator windows "
            "(the campaign of record replays sealed retained windows)")
    retained_operator_windows = normalize_retained_execution(
        retained_operator_windows,
        operator_windows=operator_windows,
        boundary_storage=(None if execution.get("boundary_storage") is None else
                          normalize_boundary_storage(
                              execution["boundary_storage"])),
    )
    return SimpleNamespace(
        operator_windows=operator_windows,
        retained_operator_windows=retained_operator_windows,
        retained_budget=RetainedWindowBudget.from_dict(
            retained_operator_windows["budget"]),
        source_bytes=int(retained_operator_windows["source_reserve_bytes"]),
    )


def quantum_layer_roster(runner, formats_by_qname, layer):
    """This layer's units, formats and live linears (the unit_filter seam, §6).

    One roster shared by window resolution and the replay core: resolution
    hands these modules to the preflight, the core replays them.
    """
    from .aura_cost import _ZERO_COST_FORMATS, _target_linears
    from .routed_experts import profile_declared_packed_expert_projections

    layer = int(layer)
    profile = runner.profile
    linears = _target_linears(runner.model, include_lm_head=False,
                              include_routed_experts=True, profile=profile)
    packed_members = profile_declared_packed_expert_projections(runner.model, profile)
    linears.update({member.qname: member for member in packed_members})
    layer_names = [name for name in formats_by_qname
                   if runner.layer_index_for_qname(name) == layer]
    if not layer_names:
        raise RuntimeError(f"the prepared roster has no units in layer {layer}")
    names = sorted(layer_names)
    unit_formats = {name: tuple(formats_by_qname[name]) for name in names}
    for name in names:
        if name not in linears:
            raise RuntimeError(f"quantum unit {name} is not an eligible live Linear")
    fmts = list(dict.fromkeys(fmt for name in names for fmt in unit_formats[name]))
    render_formats = {
        name: tuple(fmt for fmt in unit_formats[name]
                    if fmt not in _ZERO_COST_FORMATS)
        for name in names
    }
    if any(not render_formats[name] for name in names):
        raise RuntimeError("layer quantum requires a measured candidate per unit")
    return SimpleNamespace(
        layer=layer, profile=profile, linears=linears, names=names,
        unit_formats=unit_formats, fmts=fmts, render_formats=render_formats,
        packed_members=packed_members)


# --------------------------------------------------------------------------
# Static prepared-input bridge (PQ #917)
# --------------------------------------------------------------------------


def _verified_cells(production_cache) -> dict:
    """The production cache's verified cells, keyed ``(qname, format)``."""
    metadata = getattr(production_cache, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("the production cache carries no metadata: refusing")
    raw_verified = metadata.get("verified_cells")
    if not isinstance(raw_verified, Mapping) or not raw_verified:
        raise ValueError(
            "the production cache carries no verified cells: refusing")
    verified: dict[tuple[str, str], dict] = {}
    for key, value in raw_verified.items():
        if not isinstance(key, (list, tuple)) or len(key) != 2:
            raise ValueError(
                "a verified cell names no (qname, format) pair: refusing")
        verified[(str(key[0]), str(key[1]))] = value
    return verified


def _verified_render_shapes(verified: Mapping, names, render_formats: Mapping
                            ) -> dict[str, tuple[int, int]]:
    """Each unit's render shape, which every render format must agree on.

    Every cell must also carry its render digest: this is the check the
    prepared-input bridge seals the digests after.
    """
    result: dict[str, tuple[int, int]] = {}
    for name in names:
        shapes = set()
        for fmt in render_formats[name]:
            cell = verified.get((name, fmt))
            if not isinstance(cell, dict):
                raise ValueError(
                    f"prepared renders hold no verified cell for "
                    f"{name}@{fmt}: refusing")
            cell_digest = cell.get("render_file_sha256")
            if type(cell_digest) is not str or not re.fullmatch(
                    r"[0-9a-f]{64}", cell_digest):
                raise ValueError(
                    f"verified cell {name}@{fmt} carries no render digest: "
                    "refusing")
            shape = (cell.get("rendered_weight") or {}).get("shape")
            if (not isinstance(shape, (list, tuple)) or len(shape) != 2
                    or any(type(dim) is not int or dim <= 0
                           for dim in shape)):
                raise ValueError(
                    f"verified cell {name}@{fmt} carries no render shape: "
                    "refusing")
            shapes.add(tuple(shape))
        if len(shapes) != 1:
            raise ValueError(
                f"verified cell shapes disagree across {name!r} formats: "
                "refusing")
        result[name] = shapes.pop()
    return result


def derive_layer_prepared_inputs(record: dict, *, execution: Mapping,
                                 formats_by_qname: Mapping,
                                 production_cache,
                                 prepared_sha256: str,
                                 production_pkl_sha256: str,
                                 unit_roster_sha256: str) -> dict:
    """Derive the static prepared-input contract for one quantum record.

    The offline metadata bridge the post-capture generator uses: the
    once-loaded production pickle's verified cells supply each render's
    digest, the PWC's own key resolution supplies each whole-file path,
    and a current stat supplies each size -- no payload is rehashed and
    nothing is re-prepared or re-rendered. Per-window rosters come from
    the existing retained planners run on shape-only twins (the planners
    only read shapes, candidate file sizes and the sealed budget), and
    are compared with :func:`resolve_quantum_windows` before anything
    seals: a drift between the two planner paths refuses rather than
    sealing a regrouped 360-window partition. Every roster unit stages
    its complete render formats (the runtime's zero-cost exclusion);
    whole-file entries seal offset 0 with the verified digest.

    Raises ``ValueError`` (never a partial contract) when the roster,
    the verified cells, the PWC files, the planners or the digests
    disagree.
    """
    import stat as _stat

    from .aura_cost import _ZERO_COST_FORMATS
    from .joint_statistics_replay import preflight_joint_operator_admission

    layer = record.get("layer")
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(
            f"prepared inputs need a record layer, not {layer!r}: refusing")
    if not isinstance(formats_by_qname, Mapping) or not formats_by_qname:
        raise ValueError("prepared inputs need the prepared unit roster: "
                         "refusing")
    for digest, label in ((prepared_sha256, "prepared_sha256"),
                          (production_pkl_sha256, "production_pkl_sha256"),
                          (unit_roster_sha256, "unit_roster_sha256")):
        if type(digest) is not str or not re.fullmatch(
                r"[0-9a-f]{64}", digest):
            raise ValueError(f"prepared inputs seal no {label}: refusing")
    unassigned = sorted(
        name for name in formats_by_qname if qname_layer(name) is None)
    if unassigned:
        raise ValueError(
            f"prepared roster units outside the layer grammar: "
            f"{unassigned[:4]!r} -- refusing")
    names = sorted(name for name in formats_by_qname
                   if qname_layer(name) == layer)
    if not names:
        raise ValueError(
            f"the prepared roster holds no unit for layer {layer}: refusing")
    render_formats: dict[str, list[str]] = {}
    for name in names:
        fmts = [fmt for fmt in formats_by_qname[name]
                if fmt not in _ZERO_COST_FORMATS]
        if not fmts:
            raise ValueError(
                f"unit {name!r} has no measured render format: refusing")
        render_formats[name] = fmts
    verified = _verified_cells(production_cache)
    weights = getattr(production_cache, "weights", None)
    if not isinstance(weights, Mapping):
        raise ValueError("the production cache carries no weights: refusing")
    stubs = {name: SimpleNamespace(weight=SimpleNamespace(shape=shape))
             for name, shape in _verified_render_shapes(
                 verified, names, render_formats).items()}
    try:
        retained = quantum_retained_state(execution)
        admitted = preflight_joint_operator_admission(
            {layer: names}, stubs,
            {name: list(render_formats[name]) for name in names},
            production_cache, policy=retained.operator_windows,
            retained_budget=retained.retained_budget,
            source_bytes=retained.source_bytes)
        preflight_windows = list((admitted or {}).get(layer, ()))
        if not preflight_windows:
            raise ValueError(
                f"the sealed budget admits no retained window for layer "
                f"{layer}: refusing")
        resolved = resolve_quantum_windows(
            record, layer=layer, names=names, linears=stubs,
            render_formats={name: list(render_formats[name])
                            for name in names},
            production_cache=production_cache,
            operator_windows=retained.operator_windows,
            retained_budget=retained.retained_budget,
            source_bytes=retained.source_bytes)
    except (RuntimeError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError(f"prepared roster derivation refuses: "
                         f"{exc}") from exc
    if [list(window.original_full_target_names)
            for window in preflight_windows] != [
                window["names"] for window in resolved]:
        raise ValueError(
            f"quantum {record.get('quantum_id')!r} preflight and resolved "
            "window rosters drift: refusing to seal a regrouping")
    windows = []
    for window in resolved:
        members = []
        entries = []
        for qname in window["names"]:
            for fmt in render_formats[qname]:
                key = production_cache.resolve_key(qname, fmt)
                if key is None:
                    raise ValueError(
                        f"PWC candidate entry missing for {qname}@{fmt}: "
                        "refusing")
                value = weights.get(key)
                if not isinstance(value, (str, Path)):
                    raise ValueError(
                        f"prepared render {qname}@{fmt} is not file-backed: "
                        "refusing")
                path = str(Path(
                    production_cache._path_for_value(value)).absolute())
                try:
                    observed = os.lstat(path)
                except OSError as exc:
                    raise ValueError(
                        f"prepared render file unreadable at {path}: "
                        f"{exc}") from exc
                if not _stat.S_ISREG(observed.st_mode):
                    raise ValueError(
                        f"prepared render at {path} is not a regular file: "
                        "refusing")
                if observed.st_size <= 0:
                    raise ValueError(
                        f"prepared render at {path} is empty: refusing")
                members.append([qname, fmt])
                entries.append({
                    "qname": qname, "fmt": fmt, "path": path, "offset": 0,
                    "bytes": observed.st_size,
                    "sha256": verified[(qname, fmt)][
                        "render_file_sha256"]})
        windows.append({"window_index": window["window_index"],
                        "members": members, "entries": entries})
    return {"schema": PREPARED_INPUT_SCHEMA,
            "production_pkl_sha256": production_pkl_sha256,
            "unit_roster_sha256": unit_roster_sha256,
            "prepared_sha256": prepared_sha256,
            "windows": windows}


def derive_layer_spill_bound(prepared_inputs: Mapping, *, execution: Mapping,
                             production_cache, profile, model_config: Mapping,
                             replay_regime=None, block: int) -> dict:
    """Seal one layer quantum's spill bound for the one-pass replay.

    The bound is ``spill_geometry`` over the layer's whole roster, with
    every input the quantum's own geometry reads, taken from what the
    record already seals:

    * the retained windows and their members from ``prepared_inputs``
      (:func:`derive_layer_prepared_inputs`), which the quantum resolves
      and compares before it replays;
    * each unit's shape from the production cache's verified render cells,
      and its packed-expert role from ``profile``
      (``joint_replay_spill.sealed_spill_targets``);
    * the probe count, calibration shape and probe microbatch from the
      plan's ``execution`` block, and the capture batch from the launch's
      ``replay_regime``;
    * the routed top-k from the source ``model_config`` (``config.json``);
    * the 16-bit dtype the source runner measures in.

    ``block`` is the direct-I/O grid the reservation is sized on, the one
    input the plan does not carry; the scratch refuses a coarser live grid.
    A resume spills a subset of the roster, which reserves no more.
    """
    from .joint_replay_regime import normalize_replay_regime
    from .joint_replay_spill import (
        SPILL_SEAL_DTYPE,
        experts_per_token,
        seal_spill_bound,
        sealed_spill_targets,
        spill_capture_batch_tokens,
        spill_geometry,
    )

    windows = prepared_inputs.get("windows") if isinstance(
        prepared_inputs, Mapping) else None
    if not isinstance(windows, list) or not windows:
        raise ValueError("a spill bound needs the layer's sealed prepared "
                         "windows: refusing")
    window_names: list[tuple[str, ...]] = []
    render_formats: dict[str, list[str]] = {}
    for window in windows:
        names = []
        for qname, fmt in window["members"]:
            render_formats.setdefault(qname, []).append(fmt)
            if qname not in names:
                names.append(qname)
        window_names.append(tuple(names))
    shapes = _verified_render_shapes(_verified_cells(production_cache),
                                     sorted(render_formats), render_formats)
    execution = dict(execution)
    capture_batch = normalize_replay_regime(replay_regime)["capture_batch"]
    try:
        geometry = spill_geometry(
            sealed_spill_targets(shapes, profile), window_names,
            pending=set(render_formats),
            batch_tokens=spill_capture_batch_tokens(
                execution["n_calib_samples"], execution["calib_seqlen"],
                probe_microbatch=int(execution.get("probe_microbatch", 0)),
                capture_batch=capture_batch),
            n_probes=execution["n_probes"],
            element_size=torch.empty((), dtype=getattr(
                torch, SPILL_SEAL_DTYPE)).element_size(),
            experts_per_token=experts_per_token(
                SimpleNamespace(config=model_config)))
        return seal_spill_bound(geometry, block=block,
                                capture_batch=capture_batch,
                                element_dtype=SPILL_SEAL_DTYPE)
    except (RuntimeError, KeyError, TypeError) as exc:
        raise ValueError(f"spill bound derivation refuses: {exc}") from exc


def prepare_retained_window_read(window_index: int, *, record: Mapping,
                                 progress) -> str:
    """Enter the window's render phase and await its staged renders (PQ #917).

    The production ``before_window`` body, factored so the same code runs
    in the quantum and in tests: the render phase is entered first (read
    transitions never price units), then the window's exact sealed
    entries are awaited through the existing bounded strict readiness
    API -- one deadline for the whole window, movers still owned
    entirely by PrismaBuild, no HDD fallback, no global barrier. The
    sealed executable manifest is the declaration: these few entries are
    what the dispatched row stages, so an uncovered span waits the bound
    and a foreign span refuses immediately.

    Returns ``"ready"`` when every entry is covered and published,
    ``"legacy"`` for rows without a sealed prepared contract (phase
    announced, nothing to await), ``"unavailable"`` when no resolver is
    bound (the load's own lease checks decide), and
    ``"unready-<verdict>"`` otherwise (the load below still owns every
    check and refuses on unlanded bytes).
    """
    import time as _time

    from .residency_map import RANGE_HIT, residency_resolver
    from .residency_shard_reader import (
        await_staged_spans,
        staged_range_wait_s,
    )
    from .staged_lease import stage_cover_is_published, stage_covers_are_published

    window_index = int(window_index)
    block = record.get("executable_readset")
    prepared = block.get("prepared_input") if isinstance(block, dict) else None
    progress.enter_read_phase(executable_render_phase_name(window_index))
    windows = prepared.get("windows") if isinstance(prepared, dict) else None
    entry = next((window for window in (windows or [])
                  if isinstance(window, dict)
                  and window.get("window_index") == window_index), None)
    if entry is None:
        return "legacy"
    resolver = residency_resolver()
    if resolver is None:
        return "unavailable"
    wanted = [(item["path"], int(item.get("offset", 0)),
               int(item.get("offset", 0)) + int(item["bytes"]),
               int(item["bytes"]))
              for item in entry.get("entries", [])]
    if not wanted:
        return "legacy"
    # One cover lookup proves the whole window (PQ #1210), as it does for the
    # exact-cache and adjoint reads (PQ #997); a batch that cannot say which
    # entry is missing falls back to one lookup per entry.
    verdict = await_staged_spans(
        resolver, wanted,
        deadline=_time.monotonic() + staged_range_wait_s(),
        published=stage_cover_is_published,
        published_batch=stage_covers_are_published)
    print(f"[residency] retained render window {window_index:02d}: "
          f"{verdict} for {len(wanted)} staged entr"
          f"{'y' if len(wanted) == 1 else 'ies'}", flush=True)
    return "ready" if verdict == RANGE_HIT else f"unready-{verdict}"


# --------------------------------------------------------------------------
# The layer quantum core (§6.2 steps 2-6)
# --------------------------------------------------------------------------


def bind_joint_served_quantizer(formats_by_qname):
    """Require the actual served static-A4 operator before Stage B pricing.

    A registered binding includes the inspected image and extension build.
    A missing operator refuses; the Torch arithmetic model is never a price.
    A16/dynamic-only rosters do not load the serving extension.
    """
    from . import format_registry as fr
    from .nvfp4_activation_contract import bind_served_quantizer_identity
    from .perturbed_x_cache import _served_nvfp4_act_qdq_enabled

    served_override = _served_nvfp4_act_qdq_enabled()
    for fmt in sorted({fmt for formats in formats_by_qname.values() for fmt in formats}):
        contract = fr.get_format(fmt).static_activation_contract
        if contract is not None and (contract.measured_as_served or served_override):
            identity = bind_served_quantizer_identity(
                require=True, context="joint Stage B activation pricing")
            if contract.served_quantizer is not None and contract.served_quantizer != identity:
                raise RuntimeError("joint Stage B format overrides the served quantizer binding")
            return identity.as_record()
    return None


def build_quantum_source_runner(config, *, offload_folder,
                                sealed_head_tensors=None):
    """Rebuild the same sealed BF16 source used by Stage A.

    ``sealed_head_tensors`` is the resident head the quantum's executable
    readset declares (``executable_readset.head_source.tensors``, PQ #1095):
    the streaming context refuses before its first head read when the head
    it selects differs.
    """
    from .cost_streaming import build_streamed_causal_lm
    from .model_profiles import detect_profile
    from .tessera_joint_aura import _source_prefetch

    return build_streamed_causal_lm(
        config["model"], device=torch.device("cuda"), dtype=torch.bfloat16,
        offload_folder=str(offload_folder), profile=detect_profile(config["model"]),
        attn_implementation="eager", source_authentication=None,
        source_derivative=config["execution"].get("source_derivative"),
        **({"sealed_head_tensors": sealed_head_tensors}
           if sealed_head_tensors is not None else {}),
        **_source_prefetch(config))


def quantum_adjoint_space(record, adjoint_slice, output_root):
    """Read the original capture namespace when only the candidate catalog moved."""
    if record.get("catalog_extension") is None:
        return adjusted_space(output_root)
    directory = Path(adjoint_slice["boundary_storage"]["directory"])
    if (not directory.is_absolute() or directory.name != "exact-boundaries"
            or ".." in directory.parts):
        raise RuntimeError("catalog extension capture has no canonical original adjoint namespace")
    space = directory.parent
    checkpoint = adjoint_slice["checkpoint"]
    expected = space / "checkpoints" / f"boundary-{int(checkpoint['boundary']):03d}" / "entries"
    from .joint_adjoint_slices import (
        ADJOINT_CHECKPOINT_REFERENCED_SCHEMAS, checkpoint_cotangent_plane)
    # The slice was verified upstream (``verify_adjoint_slice``); a copied
    # checkpoint keeps the one-directory rule unchanged. Referenced: v2, and
    # v3 whose one shared-state row is its pack (PQ #1037).
    referenced = checkpoint.get("schema") in ADJOINT_CHECKPOINT_REFERENCED_SCHEMAS
    plane = {}
    if referenced:
        try:
            plane = checkpoint_cotangent_plane(checkpoint)
        except ValueError as exc:
            raise RuntimeError(f"catalog extension checkpoint refused: {exc}") from exc
    own = (checkpoint.get("shared_state_entries", []) if referenced
           else checkpoint.get("activation_entries", [])
           + checkpoint.get("shared_state_entries", []))
    for entry in own:
        if Path(entry["path"]).parent != expected:
            raise RuntimeError("catalog extension checkpoint escaped the original capture namespace")
    # PQ #1036: a referenced plane lives in the same capture's generation.
    for row in plane.values():
        if Path(row["path"]).parent != (
                directory / str(checkpoint["session"]["generation"]) / "entries"):
            raise RuntimeError("catalog extension checkpoint escaped the original capture namespace")
    return space


def run_layer_quantum_core(
    runner, production_cache, calib_ids, formats_by_qname, *,
    record, adjoint_slice, execution, output_root,
    projection_backend=None, resume=False,
    resolved_windows,
    counters: QuantumCounters, progress: QuantumProgress,
    adjoint_handoff=None, handoff_emitter=None,
) -> dict:
    """Execute one layer quantum and return its payload (§6.4 ``cost.pkl``).

    ``adjoint_handoff`` (band-serial, PQ #996) is quantum ``layer + 1``'s
    handoff, already bound by :func:`~prismaquant.joint_quantum_handoff.
    load_quantum_handoff`: its boundary-``layer + 1`` plane and owner states
    replace the checkpoint load and the render-free chain. ``handoff_emitter``
    publishes this quantum's own final plane and owner states for
    ``layer - 1`` once the retained windows finish. Neither enters the
    record, the journal identity or the payload.
    """
    from . import format_registry as fr
    from .aura_cost import (
        _assemble_streamed_aura_payload,
        _aura_source_sha256,
        _aura_unit_state,
        _build_aura_checkpoint_identity,
        _checkpoint_git_commit,
        _free_gib,
        _prepare_aura_checkpoints,
        _restore_aura_unit_state,
        _write_aura_unit_checkpoint,
        _ZERO_COST_FORMATS,
    )
    from .cost_streaming import (
        StreamedBoundaryArtifacts,
        _state_storage_bytes,
        normalize_boundary_storage,
        prefetched_boundary_batches,
        validate_streamed_model_identity,
    )
    from .joint_aura import (
        JointOperatorStatisticsLease,
        activation_identity,
        identity_sha256,
        make_joint_aura_entry,
        source_execution_identity,
        squared_signed,
        validate_joint_aura_entry,
        validated_probe_identity,
    )
    from .joint_adjoint_checkpoints import (
        load_adjoint_checkpoint,
        reference_from_record,
        render_free_layer_roll,
    )
    from .joint_retained_window_plan import OBSERVED_BASELINE_KEY
    from .joint_statistics_replay import (
        check_operator_allocation,
        observe_and_project_retained_windows,
        operator_window_guard,
        statistics_arithmetic_identity,
    )
    from .kl_fisher import ROW_PROBE_LAYOUT
    from .production_weight_cache import _cb_cache_tensor_identity
    from .production_weight_cache import production_cache_cb_render_provenance
    from .routed_experts import refresh_packed_expert_projections
    from .sensitivity_probe import SharedStateCotangents, kv_cotangent_path_enabled

    layer = int(record["layer"])
    quantum_id = str(record["quantum_id"])
    # The quantum reads its stage-A slice and nothing else (PQ #993): a whole
    # receipt or band, or a slice the record does not bind, refuses here.
    verify_adjoint_slice(adjoint_slice, layer=layer,
                         checkpoint_boundary=int(record["adjoint"]["checkpoint_boundary"]))
    if adjoint_slice_sha256(adjoint_slice) != record["adjoint"].get("slice_sha256"):
        raise RuntimeError(f"quantum {quantum_id} is handed a stage-A slice its record "
                           "does not bind")
    # The chain a quantum rebuilds from its checkpoint runs Stage A's own
    # regime (RobTand/prismaquant#997): the batch size sets the rounding,
    # so a chain at another batch size would not be Stage A's chain.
    try:
        chain_regime = chain_regime_of(adjoint_slice["run_identity"])
    except ChainRegimeRefused as exc:
        raise QuantumIdentityRefused(
            f"quantum {quantum_id}: the stage-A slice's chain regime: {exc}") from exc
    # The bf16 reduction flag sets the chain's rounding too (PQ #1065): the
    # live flag, which the runtime pinned from its environment, must be the
    # one the slice's run identity records.
    require_slice_bf16_reduction(
        adjoint_slice, bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        where=f"quantum {quantum_id}")
    checkpoint_dir = Path(record["output_space"]["checkpoint_dir"])
    n_probes = int(execution["n_probes"])
    seed_base = int(execution["seed_base"])
    token_scope = "all"
    temperature = 1.0
    probe_microbatch = int(execution.get("probe_microbatch", 0))
    # The free-UMA floor the chain roll and the retained replay check. A
    # missing floor is not a zero floor (PQ #1163): the plan keeps
    # ``min_free_gib`` at its top level, and ``quantum_runtime_execution``
    # carries it here.
    if (type(execution.get("min_free_gib")) not in (int, float)
            or not execution["min_free_gib"] >= 0):
        raise QuantumIdentityRefused(
            f"quantum {quantum_id}: its execution declares no nonnegative min_free_gib; "
            "the chain roll and the retained replay would run with no free-UMA floor")
    min_free_gib = float(execution["min_free_gib"])
    # The Stage B replay regime (#994): the capture batch and the statistics
    # accumulation. The default stamps nothing and replays bitwise; any other
    # regime changes the arithmetic, so it is stamped into the statistics
    # identity below and replays only from the spill.
    from .joint_replay_regime import (
        DEFAULT_REPLAY_REGIME,
        ReplayRegimeRefused,
        handoff_regime_refusal,
        normalize_replay_regime,
        replay_regime_identity,
    )
    from .joint_replay_spill import stage_b_spill_config
    try:
        replay_regime = normalize_replay_regime(execution.get("replay_regime"))
    except ReplayRegimeRefused as exc:
        raise QuantumIdentityRefused(f"quantum {quantum_id}: {exc}") from exc
    if replay_regime != DEFAULT_REPLAY_REGIME:
        if stage_b_spill_config() is None:
            raise QuantumIdentityRefused(
                f"quantum {quantum_id}: replay regime {replay_regime} replays from "
                "the spill; declare PRISMAQUANT_STAGE_B_SPILL_ROOT and "
                "PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES")
        counters.replay["regime"] = replay_regime_identity(replay_regime)
    # A band-serial producer (PQ #996) captures at the slice's chain batch
    # size, so the plane it hands off is the one the consumer's chain
    # rebuild ends on; any other capture batch refuses before GPU work.
    if handoff_emitter is not None:
        refusal = handoff_regime_refusal(
            replay_regime, chain_batch_size=chain_regime["batch_size"])
        if refusal:
            raise QuantumIdentityRefused(f"quantum {quantum_id}: {refusal}")
        if int(handoff_emitter.capture_batch) != replay_regime["capture_batch"]:
            raise QuantumIdentityRefused(
                f"quantum {quantum_id}: the handoff emitter was built for capture "
                f"batch {handoff_emitter.capture_batch}, and this launch captures "
                f"at {replay_regime['capture_batch']}")
    capture_batch = replay_regime["capture_batch"]
    # PQ #1151: an opt-in measurement of the capture workspace, read once.
    from .stage_b_workspace_profile import profile_capture_workspace, profile_request
    workspace_profile = profile_request()
    # PQ #1011: an executable read plan is sealed for one replay mode, and a
    # launch in the other mode would stage reads this quantum never makes.
    sealed_spill = False
    sealed_block = record.get("executable_readset")
    if isinstance(sealed_block, dict):
        from .joint_layer_quanta import normalize_replay_mode
        try:
            sealed_mode = normalize_replay_mode(sealed_block.get("replay_mode"))
        except ValueError as exc:
            raise QuantumIdentityRefused(f"quantum {quantum_id}: {exc}") from exc
        launched_mode = "windowed" if stage_b_spill_config() is None else "spill"
        if sealed_mode != launched_mode:
            raise QuantumIdentityRefused(
                f"quantum {quantum_id}: its read plan is sealed for the "
                f"{sealed_mode} replay, but this launch runs the {launched_mode} "
                "replay (PRISMAQUANT_STAGE_B_SPILL_ROOT); regenerate the "
                f"executable readsets with --replay-mode {launched_mode}")
        sealed_spill = sealed_mode == "spill"

    retained = quantum_retained_state(execution)
    operator_windows = retained.operator_windows
    retained_operator_windows = retained.retained_operator_windows
    retained_budget = retained.retained_budget

    # ---- roster: this layer's units only (the unit_filter seam, §6) -------
    roster = quantum_layer_roster(runner, formats_by_qname, layer)
    profile, linears, names = roster.profile, roster.linears, roster.names
    unit_formats, fmts, render_formats = (
        roster.unit_formats, roster.fmts, roster.render_formats)
    packed_members = roster.packed_members
    served_quantizer = bind_joint_served_quantizer(unit_formats)
    from .joint_served_activation import joint_activation_maxima, operator_policy_record
    pricing_maxima = joint_activation_maxima(production_cache)
    activation_policy = getattr(production_cache, "_joint_served_activation", None)
    resource_policy = getattr(production_cache, "_joint_stage_b_resource_policy", None)
    # The record seals window indices only (D2); membership comes from the
    # resolved handshake the caller ran, which refuses stale records. What is
    # checked here is coverage: the sealed budget must admit exactly this
    # layer's roster -- no foreign units, none missing.
    resolved_names = sorted(
        name for window in resolved_windows for name in window["names"])
    if resolved_names != names:
        raise RuntimeError(
            f"resolved windows cover {resolved_names!r}, not layer {layer}'s "
            f"roster {names!r}")

    # ---- identity blocks (mirrors compute_aura_cost_streamed's) -----------
    batch_rows = min(probe_microbatch or len(calib_ids), len(calib_ids))
    row_offsets = list(range(0, len(calib_ids), batch_rows))
    probe_layout = None
    execution_partition = None
    if probe_microbatch:
        sequence_length = int(calib_ids.shape[1])
        probe_layout = {
            "schema": ROW_PROBE_LAYOUT,
            "global_rows": len(calib_ids),
            "sequence_length": sequence_length,
            "selected_tokens_per_row": sequence_length,
            "vocab_size": int(runner._head().weight.shape[0]),
            "token_scope": token_scope,
            "global_token_count": len(calib_ids) * sequence_length,
        }
        execution_partition = {
            "schema": "prismaquant.aura.streamed_microbatch.v1",
            "requested_rows": probe_microbatch,
            "effective_rows": batch_rows,
            "partition_count": len(row_offsets),
            "row_order": "contiguous_complete_sequences",
            "gradient_diagnostics": "sum_output_operators_fp32_before_norm",
        }

    model_identity = validate_streamed_model_identity(
        production_cache.metadata["source_model_identity"], where="layer quantum")
    joint_probe_identity = {
        "schema": "prismaquant.joint_aura.probes.v2",
        "source_model": model_identity,
        "calibration_sha256": hashlib.sha256(
            calib_ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "calibration_shape": list(calib_ids.shape),
        "calibration_dtype": str(calib_ids.dtype),
        "n_probes": n_probes, "seed_base": seed_base,
        "token_scope": token_scope, "temperature": temperature,
        "distribution": "rademacher", "normalization": "global_kl_fisher",
        "producer_source_sha256": _aura_source_sha256(),
        "source_execution": source_execution_identity(runner.model),
        "arithmetic": statistics_arithmetic_identity(runner.dtype, projection_backend,
                                                     replay_regime=replay_regime),
    }
    joint_probe_identity["arithmetic"]["operator_windows"] = operator_windows
    joint_probe_identity["arithmetic"]["gradient_diagnostics"] = (
        "sum_output_operators_fp32_before_norm")
    if served_quantizer is not None:
        joint_probe_identity["arithmetic"]["served_quantizer"] = served_quantizer
    if activation_policy is not None:
        joint_probe_identity["arithmetic"]["served_activation_policy"] = activation_policy[0]
    if resource_policy is not None:
        joint_probe_identity["arithmetic"]["stage_b_resource_policy"] = resource_policy
    if probe_layout is not None:
        joint_probe_identity["noise_layout"] = probe_layout
        joint_probe_identity["arithmetic"]["execution_partition"] = execution_partition
    # The probe identity is final here. It carries the source model's
    # identity (8.8 MB of JSON for GLM-5.3), which every operator and every
    # (unit, format) row serialized again, several times each, while the GPU
    # idled (PQ #1183). Validate and hash it once, and check fresh rows
    # through that. The rows themselves carry this ordinary dict, as before;
    # the provenance digest below proves it did not change after this point.
    joint_probe = validated_probe_identity(joint_probe_identity)

    def validate_row(row):
        if row["probe_identity"] is joint_probe_identity:
            row = {**row, "probe_identity": joint_probe}
        return validate_joint_aura_entry(row)

    prepared_render_identities = production_cache.metadata["verified_cells"]
    expected_pairs = {(name, fmt) for name in names for fmt in render_formats[name]}
    joint_cache_renders: dict[str, dict[str, dict]] = {}
    # Before install, a streamed Linear is the meta skeleton's parameter: its
    # shape is the checkpoint's, its dtype is torch's default
    # (``build_streaming_skeleton`` passes none, so GLM's wrapper config gets
    # float32 against a bf16 checkpoint).  Only the shape is compared here.
    # The dtype and byte half of the prepared proof is checked on the
    # installed tensor, per layer, before its first render is consumed (PQ
    # #1102; the run's ``_require_installed_render_sources`` since 20dede4a).
    for name, fmt in sorted(expected_pairs):
        value = prepared_render_identities[(name, fmt)]["rendered_weight"]
        source = linears[name].weight
        if (value.get("shape") != list(source.shape)
                or not isinstance(value.get("dtype"), str)
                or type(value.get("logical_bytes")) is not int):
            raise RuntimeError(
                f"prepared render tensor proof differs from the source for {name}@{fmt}: "
                f"{_render_proof_sides(value, source, side='skeleton')}")
        joint_cache_renders.setdefault(name, {})[fmt] = dict(value)
    joint_run_identity = {
        "schema": "prismaquant.joint_aura.run.v2",
        "probe_identity": joint_probe_identity,
        "cached_rendered_weights": joint_cache_renders,
        "activation_contracts": {
            name: {fmt: activation_identity(fr.get_format(fmt),
                                            pricing_maxima or {}, name)
                   for fmt in unit_formats[name]}
            for name in names
        },
    }
    if served_quantizer is not None:
        joint_run_identity["served_quantizer"] = served_quantizer

    # ---- journal ---------------------------------------------------------
    checkpoint_git_commit = _checkpoint_git_commit()
    from prismaquant.aura_cost import is_cb_format

    cb_provenance = (production_cache_cb_render_provenance(
        production_cache, require_for_formats=fmts, where="layer quantum cache")
        if any(is_cb_format(fmt) for fmt in fmts) else {})
    extra = {
        "streaming": True,
        "retained_operator_windows": retained_operator_windows,
        "streamed_microbatch": execution_partition,
        "streamed_boundary_storage": None,  # bound below, before the journal
        "joint_aura": joint_run_identity,
        "streamed_model_identity": model_identity,
        "streamed_formats_by_qname": {name: list(unit_formats[name]) for name in names},
        "unmeasured_streamed_formats_by_qname": {name: [] for name in names},
        "include_routed_experts": True,
        "diagnostic_weight_mse_pairs": [],
        "streamed_gradient_harvest": "post_accumulate_per_parameter",
        "streamed_cotangent_rollover": "in_place_per_probe",
        "streamed_boundary_release": "progressive_reverse",
        "distributed_quantum": {
            "quantum_id": quantum_id,
            "identity_sha256": record["identity_sha256"],
            "adjoint_slice_sha256": record["adjoint"]["slice_sha256"],
            "checkpoint_boundary": int(record["adjoint"]["checkpoint_boundary"]),
            "chain_layers": [int(c) for c in record["adjoint"]["chain_layers"]],
            "windows": len(record["windows"]),
            "chunks": [chunk["name"] for chunk in record["chunks"]],
        },
    }

    # ---- boundary storage: read-attached to the adjoint capture ----------
    source_adjoint_space = quantum_adjoint_space(record, adjoint_slice, output_root)
    storage_policy = normalize_boundary_storage(execution["boundary_storage"])
    storage_policy["directory"] = str(boundary_entry_directory(source_adjoint_space))
    storage = StreamedBoundaryArtifacts(storage_policy)
    storage.attach(adjoint_slice["boundary_storage"]["session"], n_probes=n_probes,
                   forward_recovery=adjoint_slice["boundary_storage"].get("forward_recovery"))
    extra["streamed_boundary_storage"] = storage.identity

    identity = _build_aura_checkpoint_identity(
        model=runner.model, calib_ids=calib_ids, names=names, linears=linears,
        formats=fmts, chunks=[names], n_probes=n_probes, token_scope=token_scope,
        temperature=temperature, seed_base=seed_base, dw_dtype="float32",
        include_lm_head=False, hook_harvest=True,
        allow_packed_expert_omission=False, probe_microbatch=probe_microbatch,
        collect_col_energy=False, require_production_cache=True,
        production_cache=production_cache, cb_provenance=cb_provenance,
        git_commit=checkpoint_git_commit, extra_identity=extra,
    )
    checkpoint_root, checkpoint_identity_sha256, completed_states = (
        _prepare_aura_checkpoints(checkpoint_dir, resume=resume, identity=identity,
                                  names=names))

    s2: dict[tuple[str, str], float] = {}
    s4: dict[tuple[str, str], float] = {}
    x2_probe: dict[tuple[str, str], list[float]] = {}
    dw_src: dict[tuple[str, str], str] = {}
    g_trace: dict[str, float] = {}
    joint_rows: dict[str, dict[str, dict]] = {}
    joint_components: dict[tuple[str, str], list[dict]] = {}
    joint_operators: dict[tuple[str, str], dict] = {}
    joint_source_tensors: dict[str, dict] = {}
    operator_window_receipts: list[dict] = []
    completed_units: set[str] = set()
    for name in names:
        state = completed_states.get(name)
        if state is None:
            continue
        _restore_aura_unit_state(
            name, state, nonzero_formats=render_formats[name], n_probes=n_probes,
            collect_col_energy=False, s2=s2, s4=s4, x2_probe=x2_probe, dw_src=dw_src,
            g_trace=g_trace, col_energy={}, diagnostic_weight_mse_pairs=set(),
            weight_mse_diagnostic={}, require_source_weight_identity=False,
            source_weight_identity={}, observation_counts=None)
        rows = state.get("joint_aura_rows")
        if not isinstance(rows, Mapping) or set(rows) != set(unit_formats[name]):
            raise RuntimeError(f"joint AURA checkpoint row scope mismatch for {name}")
        for fmt, row in rows.items():
            try:
                if not validate_joint_aura_entry(row):
                    raise ValueError("not a joint row")
                operator = row["joint_operator_identity"]
                if operator["qname"] != name or operator["format"] != fmt:
                    raise ValueError("probe/operator alignment mismatch")
                # What the row measured (the calibration draw, the probes)
                # refuses in both modes; its producer source and arithmetic
                # (the Stage B resource policy among them) are run seals
                # (PQ #1147): dev mode prints them and reuses the row.
                if probe_identity_walls_differ(joint_probe_identity, row["probe_identity"]):
                    raise ValueError("probe/operator alignment mismatch")
                seal_check("joint probe identity", probe_identity_seals(joint_probe_identity),
                           probe_identity_seals(row["probe_identity"]),
                           where=f"joint AURA checkpoint {name}@{fmt}",
                           same=row["probe_identity"] == joint_probe_identity,
                           refusal=lambda: ValueError("probe/operator alignment mismatch"))
                if fmt in render_formats[name]:
                    if operator["rendered_weight"] != joint_cache_renders[name][fmt]:
                        raise ValueError("actual rendered-weight identity mismatch")
                    if row["x2_per_probe"] != x2_probe[(name, fmt)]:
                        raise ValueError("legacy/joint sample alignment mismatch")
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"joint AURA checkpoint identity mismatch for {name}@{fmt}: "
                    f"{exc}") from exc
        joint_rows[name] = dict(rows)
        completed_units.add(name)

    progress.enter_head(len(completed_units))

    # ---- checkpoint + chain (§6.2 step 3) --------------------------------
    # Read-phase reporting follows the executable readset contract only:
    # legacy slice rows keep byte-identical progress (head/chunks), while
    # bound rows report each staged phase as its bytes are consumed.
    executable = isinstance(record.get("executable_readset"), dict)
    checkpoint_record = adjoint_slice["checkpoint"]
    if int(checkpoint_record["boundary"]) != int(record["adjoint"]["checkpoint_boundary"]):
        raise RuntimeError(
            "stage-A slice does not carry the record's checkpoint boundary "
            f"{record['adjoint']['checkpoint_boundary']}")

    # ---- one-pass replay spill (PQ #994) ---------------------------------
    # Declared by environment, like the #956 cotangent sink, and outside the
    # quantum identity: both replay modes produce the same records. The
    # bound is geometry only and the scratch is allocated here, before the
    # chain or any GPU work, so an undersized ceiling or disk refuses first.
    # A resume recaptures the spill for its pending targets.
    from contextlib import nullcontext

    from .joint_replay_spill import (
        REPLAY_SPILL,
        SpillBoundRefused,
        StageBReplaySpill,
        experts_per_token,
        require_sealed_spill_bound,
        spill_capture_batch_tokens,
        spill_geometry,
        stage_b_spill_config,
    )
    from .perturbed_x_cache import SpillGridRefused

    spill = None
    spill_config = stage_b_spill_config()
    spill_pending = {name for name in names
                     if render_formats[name] and name not in completed_units}
    # The capture carries ``capture_batch`` stored batches through each
    # layer pass: consecutive batches, never across a sealed read window
    # (the window's tensors are released when it closes), the last group
    # ragged when the batch count is not a multiple.
    capture_groups = [list(range(start, min(start + capture_batch, len(row_offsets))))
                      for start in range(0, len(row_offsets), capture_batch)]
    if capture_batch > 1 and int(storage.config["prefetch_batches"]) % capture_batch:
        raise QuantumIdentityRefused(
            f"quantum {quantum_id}: capture batch {capture_batch} does not divide "
            f"the sealed read window of {storage.config['prefetch_batches']} batches")
    if spill_config is not None and spill_pending:
        if replay_regime != DEFAULT_REPLAY_REGIME:
            from .joint_replay_spill import (
                ReplayRegimeInadmissible,
                require_row_local_activation_qdq,
            )
            from .joint_served_activation import joint_activation_maxima
            try:
                checked = require_row_local_activation_qdq(
                    {name: linears[name] for name in spill_pending},
                    {name: {fmt: fr.get_format(fmt) for fmt in render_formats[name]}
                     for name in spill_pending},
                    joint_activation_maxima(production_cache),
                    device=runner.device, dtype=runner.dtype)
            except ReplayRegimeInadmissible as exc:
                raise QuantumIdentityRefused(f"quantum {quantum_id}: {exc}") from exc
            counters.replay["row_local_qdq"] = [list(pair) for pair in checked]
        spill_windows = [tuple(name for name in window["names"] if name in spill_pending)
                         for window in resolved_windows]
        spill_batch_tokens = spill_capture_batch_tokens(
            len(calib_ids), int(calib_ids.shape[1]),
            probe_microbatch=probe_microbatch, capture_batch=capture_batch)
        if len(spill_batch_tokens) != len(capture_groups):
            raise RuntimeError("Stage B spill capture groups differ from the "
                               "stored batches")
        spill_element_size = torch.empty((), dtype=runner.dtype).element_size()
        spill_top_k = experts_per_token(runner.model)
        spill_bound = spill_geometry(
            linears, spill_windows, pending=spill_pending,
            batch_tokens=spill_batch_tokens, n_probes=n_probes,
            element_size=spill_element_size, experts_per_token=spill_top_k)
        # The ceiling is the record's sealed reservation for the whole roster
        # (the dispatcher sets it). The quantum recomputes that geometry from
        # its live modules before any GPU work; a resume's pending subset
        # then reserves no more than the seal.
        sealed_grid = None
        if sealed_spill:
            sealed_bound = sealed_block.get("spill_bound")
            if sealed_bound is None:
                raise QuantumIdentityRefused(
                    f"quantum {quantum_id}: its executable readset is sealed for "
                    "the spill replay but seals no spill bound; regenerate it")
            roster_bound = spill_geometry(
                linears, [tuple(window["names"]) for window in resolved_windows],
                pending={name for name in names if render_formats[name]},
                batch_tokens=spill_batch_tokens, n_probes=n_probes,
                element_size=spill_element_size, experts_per_token=spill_top_k)
            try:
                sealed_grid = require_sealed_spill_bound(
                    sealed_bound, roster_bound, capture_batch=capture_batch,
                    element_dtype=str(runner.dtype).removeprefix("torch."),
                    ceiling=spill_config[1])
            except SpillBoundRefused as exc:
                raise QuantumIdentityRefused(f"quantum {quantum_id}: {exc}") from exc
        try:
            spill = StageBReplaySpill(
                root=spill_config[0], max_bytes=spill_config[1], geometry=spill_bound,
                window_names=spill_windows, n_probes=n_probes, dtype=runner.dtype,
                device=runner.device, accumulation=replay_regime["accumulation"],
                chunk_rows=replay_regime["chunk_rows"], max_block=sealed_grid)
        except SpillGridRefused as exc:
            raise QuantumIdentityRefused(f"quantum {quantum_id}: {exc}") from exc
        counters.replay.update(mode=REPLAY_SPILL, spill_geometry=spill_bound.as_dict())
        if capture_batch > 1:
            counters.replay["capture_groups"] = len(capture_groups)

    # Band-serial (PQ #996): the handoff is the plane this quantum's chain
    # would end on, so the chain below walks no layers.
    chain_layers = ([] if adjoint_handoff is not None else
                    [int(c) for c in record["adjoint"]["chain_layers"]])
    source_order = quantum_source_layer_order(chain_layers, layer)
    if executable:
        progress.enter_read_phase(CHECKPOINT_LOAD_PHASE if adjoint_handoff is None
                                  else HANDOFF_LOAD_PHASE)
    with storage, (spill if spill is not None else nullcontext()):
        with counters.io.span(CHECKPOINT_LOAD_PHASE if adjoint_handoff is None
                              else HANDOFF_LOAD_PHASE):
            if adjoint_handoff is None:
                grad_plane, shared_adjoint, shared_pass = load_adjoint_checkpoint(
                    source_adjoint_space, checkpoint_record,
                    cotangent_factory=storage.checkpoint_cotangent_sink,
                    shared_state_max_bytes=storage.config["max_auxiliary_bytes"],
                    max_resident_bytes=storage.config["max_resident_bytes"],
                    residency_check=storage.reserve_resident)
            else:
                grad_plane, shared_adjoint, shared_pass = load_handoff_inputs(
                    adjoint_handoff, checkpoint_record, n_probes=n_probes,
                    n_batches=len(row_offsets),
                    cotangent_factory=storage.checkpoint_cotangent_sink,
                    shared_state_max_bytes=storage.config["max_auxiliary_bytes"],
                    max_resident_bytes=storage.config["max_resident_bytes"],
                    residency_check=storage.reserve_resident)
        cotangent_owners = [[SharedStateCotangents(enabled=kv_cotangent_path_enabled())
                             for _ in row_offsets] for _ in range(n_probes)]
        for (probe, batch), state in shared_adjoint.items():
            cotangent_owners[probe][batch].load_state_dict(state)
        state = None
        del shared_adjoint
        partitions = [calib_ids[offset:offset + batch_rows]
                      for offset in row_offsets]
        batches = _rebuild_batches(runner, partitions=partitions,
                                   shared_pass=shared_pass)
        del shared_pass
        needed = sorted(set(chain_layers) | {layer})
        for batch_index, batch in enumerate(batches):
            batch.activations_cpu = [
                (_boundary_entry_record_to_reference(
                    _boundary_entry_record(adjoint_slice, batch_index, boundary))
                 if boundary in needed else None)
                for boundary in range(runner.num_layers + 1)]
        storage.watch_auxiliary(batches, cotangent_owners)
        storage.check_auxiliary(batches, cotangents=cotangent_owners)
        if spill is not None and capture_batch > 1:
            # Before the chain: a batched capture merges samples, so every
            # sample's pass state must be empty (no profile shared state, no
            # shared-state cotangent). Rechecked per group at capture time.
            try:
                _require_per_sample_state(
                    runner, batches, layer,
                    [owner for owners in cotangent_owners for owner in owners],
                    range(len(batches)), where="batched Stage B spill capture")
            except ChainRegimeRefused as exc:
                raise QuantumIdentityRefused(f"quantum {quantum_id}: {exc}") from exc

        # ---- the render-free chain: stage A's arithmetic, reused ---------
        # The guard exists before the chain (PQ #1163): each roll is a
        # batched backward at Stage A's chain regime, admitted before it
        # runs with the workspace the plan priced for that regime.
        guard = operator_window_guard(
            runner.device,
            device_bytes=execution.get("device_envelope_bytes"))
        if guard is not None:
            retained_budget.require_physical_guard(guard)
        # Every roll's planned workspace, before the first one runs: a chain
        # layer whose shape the plan did not price refuses here.
        chain_workspace_bytes = {}
        if chain_layers and guard is not None:
            try:
                chain_workspace_bytes = {
                    int(chain_layer): retained_budget.chain_workspace_bytes(
                        chain_regime["batch_size"], fused=chain_regime["probe_fusion"],
                        layer=int(chain_layer))
                    for chain_layer in chain_layers}
            except RuntimeError as exc:
                raise QuantumIdentityRefused(
                    f"quantum {quantum_id}: chain {chain_layers} cannot be admitted: "
                    f"{exc}") from exc
        from .stage_b_workspace_profile import ChainRollProfile
        chain_profile = ChainRollProfile.requested(
            guard=guard, device=runner.device,
            identity={"quantum_id": quantum_id,
                      "record_identity_sha256": record.get("identity_sha256"),
                      "layer": int(layer), "chain_layers": list(chain_layers),
                      "chain_regime": dict(chain_regime), "n_probes": n_probes,
                      "n_batches": len(batches), "min_free_gib": min_free_gib,
                      "chain_workspace_bytes_by_layer": {
                          str(key): value for key, value in chain_workspace_bytes.items()}})
        chain_started = time.time()
        chain_backwards = 0
        chain_kernel = _stage_b_kernel_profiler()
        chain_kernel.__enter__()
        try:
            for chain_layer in chain_layers:
                with counters.io.span("chain-layer", layer=int(chain_layer)):
                    try:
                        if executable:
                            progress.enter_read_phase(
                                executable_source_phase_name(chain_layer))
                        _install_with_settlement(runner, chain_layer,
                                                 operator_windows=operator_windows,
                                                 order=source_order,
                                                 settle_successors=False)
                        if executable:
                            progress.enter_read_phase(
                                executable_bound_phase_name(chain_layer))
                        if guard is not None:
                            # The free-UMA floor inside the roll stays as a
                            # second check (``_chain_free_floor``).
                            chain_admission = check_operator_allocation(
                                guard, f"before_chain_layer_roll:{int(chain_layer)}",
                                reserve_bytes=0,
                                reserve_device_bytes=chain_workspace_bytes[int(chain_layer)])
                        else:
                            chain_admission = None

                        def roll_chain_layer(chain_layer=chain_layer):
                            return render_free_layer_roll(
                                runner, storage=storage, batches=batches, layer=chain_layer,
                                cotangents=cotangent_owners, n_probes=n_probes,
                                incoming_entries=None,
                                incoming_tensor=lambda probe, batch: grad_plane[(probe, batch)],
                                roll=lambda tensor, batch, probe: grad_plane.__setitem__(
                                    (probe, batch), tensor),
                                min_free_gib=min_free_gib,
                                batch_size=chain_regime["batch_size"],
                                probe_fusion=chain_regime["probe_fusion"])

                        if chain_profile is None:
                            backwards = roll_chain_layer()
                        else:
                            backwards = chain_profile.measure(
                                int(chain_layer), roll_chain_layer,
                                admission=chain_admission,
                                reserve_device_bytes=chain_workspace_bytes.get(
                                    int(chain_layer)))
                        chain_backwards += backwards
                    finally:
                        runner.context.unload(chain_layer)
        finally:
            chain_kernel.__exit__(None, None, None)
        counters.kernel_block(chain_kernel)
        counters.chain_step(layers=len(chain_layers),
                            backwards=chain_backwards,
                            wall_s=time.time() - chain_started,
                            kernel_active_s=(None if chain_kernel.error
                                             else chain_kernel.kernel_active_s))
        if chain_profile is not None:
            # A measurement of the chain phase (PQ #1163) stops here, before
            # the retained reverse step, with its receipt written.
            chain_profile.finish()

        # ---- layer L: the single run's retained reverse step --------------
        def retained_source_phase(stage_label):
            snapshot = runner.context.source_residency_snapshot(
                range(runner.num_layers), include_head=True)
            if any(owner.get("state") in ("pending", "failed")
                   for owner in snapshot["owners"]):
                raise RuntimeError(
                    "layer quantum source phase has an unsettled or failed "
                    "source owner")
            source_bytes = snapshot["unique_storage_bytes"]
            source_cap = retained_operator_windows["source_reserve_bytes"]
            if source_bytes > source_cap:
                raise RuntimeError(
                    "layer quantum source owners exceed the sealed retained "
                    "source cap")
            if guard is not None:
                observed = check_operator_allocation(
                    guard, f"before_quantum_{stage_label}:{layer}", reserve_bytes=0)
                retained_budget.require_observed_baseline(
                    observed_bytes=observed[OBSERVED_BASELINE_KEY],
                    source_bytes=source_bytes, actual_auxiliary_bytes=0,
                    label=f"before_quantum_{stage_label}:{layer}")
                check_operator_allocation(
                    guard, f"admit_quantum_{stage_label}:{layer}",
                    reserve_bytes=(retained_budget.boundary_reserve_bytes
                                   + retained_budget.workspace_reserve_bytes
                                   + retained_operator_windows[
                                       "source_loading_reserve_bytes"]))

        if executable:
            progress.enter_read_phase(
                executable_own_source_phase_name(layer))
        with counters.io.span("own-source", layer=int(layer)):
            # The chain step left this layer's read in flight during the
            # roll. The consumer waits for it here, under the phase that
            # stages it, before the replay's first observation (PQ #1166).
            _await_own_source(runner, layer, operator_windows=operator_windows)
            if guard is not None:
                # An observation before any replay phase is admitted: it
                # releases the chain's retired blocks and charges no future
                # allocation.
                check_operator_allocation(guard, "before_layer_quantum_replay",
                                          reserve_bytes=0)
            production_cache.enable_lru(retained_budget.retained_render_cap_bytes)
            retained_source_phase("source_loading")
            _install_with_settlement(runner, layer, operator_windows=operator_windows,
                                     order=source_order)
        if packed_members:
            from .routed_experts import PackedExpertProjection

            members = [linears[name] for name in names
                       if isinstance(linears[name], PackedExpertProjection)]
            if members:
                linears.update({member.qname: member for member in
                                refresh_packed_expert_projections(members, profile)})
        retained_source_phase("reverse")

        # Prepared render proofs vs the installed tensors -- the run's
        # ``_require_installed_render_sources``, scoped to this layer.
        for name in names:
            renders = joint_cache_renders.get(name)
            if not renders:
                continue
            source = linears[name].weight
            if source.is_meta:
                raise RuntimeError(f"installed source is still meta for {name}")
            for fmt, value in renders.items():
                if (value["shape"] != list(source.shape)
                        or value["dtype"] != str(source.dtype)
                        or value["logical_bytes"] != source.numel() * source.element_size()):
                    raise RuntimeError(
                        f"prepared render tensor proof differs from the "
                        f"installed source for {name}@{fmt}: "
                        f"{_render_proof_sides(value, source, side='installed')}")

        pending = [name for name in names if name not in completed_units]
        for name in pending:
            g_trace.setdefault(name, 0.0)
            for fmt in render_formats[name]:
                key = (name, fmt)
                dw_src[key] = "rendered"
                s2.setdefault(key, 0.0)
                s4.setdefault(key, 0.0)
                x2_probe.setdefault(key, [])

        def _record_joint_operator(name, fmt, source, rendered):
            scales = pricing_maxima or {}
            activation = activation_identity(fr.get_format(fmt), scales, name)
            if fmt in render_formats[name]:
                # A measured render is the window's resident PWC tensor,
                # read by every probe: its loader thread hashed it once, when
                # it loaded it (PQ #1192). The comparison below still runs on
                # every probe.
                rendered_identity = production_cache.resident_render_identity(
                    name, fmt, rendered)
            else:
                # A zero-cost row carries its source as its render, once.
                rendered_identity = _cb_cache_tensor_identity(rendered)
            if fmt in render_formats[name]:
                if (rendered_identity != joint_cache_renders[name][fmt]
                        or activation != joint_run_identity[
                            "activation_contracts"][name][fmt]):
                    raise RuntimeError(
                        f"joint AURA actual render/activation identity changed "
                        f"for {name}@{fmt}")
            if name not in joint_source_tensors:
                joint_source_tensors[name] = _cb_cache_tensor_identity(source)
            joint_operators[(name, fmt)] = {
                "schema": "prismaquant.joint_aura.operator.v2",
                "qname": name, "format": fmt,
                "source_weight": joint_source_tensors[name],
                "rendered_weight": rendered_identity,
                "activation": activation,
                "arithmetic": joint_probe_identity["arithmetic"],
                "probe_identity_sha256": identity_sha256(joint_probe),
            }
            if activation_policy is not None and fmt != "BF16":
                policy_record = operator_policy_record(
                    *activation_policy, name, fmt, prepared_render_identities[name, fmt]["activation"])
                if policy_record is not None:
                    joint_operators[(name, fmt)]["served_activation_policy"] = policy_record
            joint_components.setdefault((name, fmt), [])

        # Zero-cost passthrough rows carry the source as their own render,
        # exactly as the single run records them before its windows replay.
        with torch.no_grad():
            for name in pending:
                for fmt in unit_formats[name]:
                    if fmt in _ZERO_COST_FORMATS:
                        weight = linears[name].weight.data
                        _record_joint_operator(name, fmt, weight, weight)

        def commit_streamed_units(targets):
            targets = [name for name in targets if name not in completed_units]
            if not targets:
                return 0
            if source_execution_identity(runner.model) != joint_probe_identity[
                    "source_execution"]:
                raise RuntimeError(
                    "joint AURA source execution backend changed during measurement")
            for name in targets:
                rows = {}
                for fmt in unit_formats[name]:
                    if fmt in _ZERO_COST_FORMATS:
                        components = [{"weight": 0.0, "activation": 0.0,
                                       "mixed": 0.0, "total": 0.0}
                                      for _ in range(n_probes)]
                    else:
                        components = joint_components[(name, fmt)]
                    row = make_joint_aura_entry(
                        operator_identity=joint_operators[(name, fmt)],
                        probe_identity=joint_probe,
                        signed_components=components,
                    )
                    row["probe_identity"] = joint_probe_identity
                    rows[fmt] = row
                joint_rows[name] = rows
                _write_aura_unit_checkpoint(
                    checkpoint_root, qname=name,
                    identity_sha256=checkpoint_identity_sha256,
                    state={**_aura_unit_state(
                        name, render_formats[name], s2=s2, s4=s4,
                        x2_probe=x2_probe, dw_src=dw_src, g_trace=g_trace,
                        col_energy={}, weight_mse_diagnostic={},
                        source_weight_identity={}, observation_counts=None),
                        "joint_aura_rows": joint_rows[name]})
                completed_units.add(name)
            progress.priced(len(completed_units))
            progress.commit()
            counters.mark_phase_units(len(completed_units))
            return len(targets)

        noncontiguous_seeds: set[tuple[int, int]] = set()
        capture_group_cache: dict = {}

        def capture_group(group, active_probe, observer):
            # One backward over a batch group (capture_batch > 1), as the
            # batched render-free chain runs it: the samples' pass state is
            # empty, so there is nothing to graft or harvest, and the input
            # cotangent is split back per stored batch.
            indices = [item[0] for item in group]
            _require_per_sample_state(
                runner, batches, layer,
                [cotangent_owners[active_probe][index] for index in indices],
                indices, where="batched Stage B spill capture")
            x_in = out = incoming_grad = gradient = None
            try:
                storage.check_auxiliary(batches, cotangents=cotangent_owners)
                if _free_gib() < min_free_gib:
                    raise RuntimeError("joint window replay crossed free UMA floor")
                cpu_rng = torch.get_rng_state()
                cuda_rng = (torch.cuda.get_rng_state(runner.device)
                            if torch.device(runner.device).type == "cuda" else None)
                incoming_grad = _stack_to_device(
                    [grad_plane[(active_probe, index)] for index in indices],
                    device=runner.device)
                x_in = _stack_to_device(
                    [item[2] for item in group], device=runner.device,
                    dtype=runner.dtype).detach().requires_grad_(True)
                batch = (batches[indices[0]] if len(indices) == 1 else
                         _chain_group_batch(runner, batches, indices, capture_group_cache))
                out = runner.isolated_layer(batch, layer, x_in, pass_state={})
                torch.autograd.backward([out], [incoming_grad])
                if observer is not None:
                    # None only under the workspace profile's window backward
                    # (PQ #1151): the same pass with no spill hooks installed.
                    observer.end_batch()
                if not torch.equal(cpu_rng, torch.get_rng_state()) or (
                        cuda_rng is not None and not torch.equal(
                            cuda_rng, torch.cuda.get_rng_state(runner.device))):
                    raise RuntimeError("joint operator replay source consumed Torch RNG")
                if x_in.grad is None:
                    raise RuntimeError("joint operator replay produced no input cotangent")
                gradient = x_in.grad.detach().to("cpu")
                start = 0
                for index, item in zip(indices, group):
                    rows = int(item[2].shape[0])
                    grad_plane[(active_probe, index)] = (
                        gradient if len(indices) == 1
                        else gradient[start:start + rows].clone())
                    start += rows
                if start != int(gradient.shape[0]):
                    raise RuntimeError("batched Stage B spill capture split its cotangent "
                                       "into the wrong rows")
                storage.check_auxiliary(batches, cotangents=cotangent_owners)
            finally:
                group = x_in = out = incoming_grad = gradient = None

        def batched_capture(active_probe, observer):
            last, expected = len(batches) - 1, iter(capture_groups)
            with prefetched_boundary_batches(storage, batches, layer) as reverse_batches:
                pending_items = []
                for item in reverse_batches:
                    pending_items.append(item)
                    if len(pending_items) < capture_batch and item[0] != last:
                        continue
                    group, pending_items = pending_items, []
                    if [entry[0] for entry in group] != next(expected, None):
                        raise RuntimeError(
                            "Stage B spill capture group differs from its geometry")
                    capture_group(group, active_probe, observer)
                    group = None
                if pending_items or next(expected, None) is not None:
                    raise RuntimeError("Stage B spill capture left batches ungrouped")

        def replay_backward(*, final, lease, probe, observer=None):
            active_probe = int(probe)
            if observer is not None and (not final or lease is not None):
                raise RuntimeError(
                    "the spill capture is the probe's one final pass, with no "
                    "statistics lease")
            # One phase admission per pass, as check_operator_allocation's
            # contract states: it synchronizes, empties the allocator cache
            # and charges the guard, which is too much work per sample. The
            # lease is fresh, so its whole statistics capacity is still to
            # come; the per-sample floors below stay. The workspace is the
            # retained budget's, the same quantity its derivation planned the
            # capture pass with (PQ #1151): one reserve per stored batch.
            # The backward's workspace, the lease's statistics and the inputs
            # the spill capture holds are CUDA allocations, so they are charged
            # to the device side; only the spill's pinned host arenas are not
            # (PQ #1157), and ``spill.capture`` has allocated those by now.
            if guard is not None:
                check_operator_allocation(
                    guard, "before_joint_window_backward",
                    reserve_bytes=(0 if observer is None
                                   else spill.capture_reserve_host_bytes),
                    reserve_device_bytes=(
                        retained_budget.capture_workspace_bytes(
                            1 if observer is None else capture_batch)
                        + (0 if lease is None
                           else lease.statistics_capacity_bytes
                           - lease.resident_statistics_bytes)
                        + (0 if observer is None
                           else spill.capture_reserve_device_bytes)))
            if observer is not None and capture_batch > 1:
                batched_capture(active_probe, observer)
                counters.replay["layer_passes"] += 1
                return
            with prefetched_boundary_batches(storage, batches, layer) as reverse_batches:
                for batch_index, batch, boundary_cpu, _unused in reverse_batches:
                    owner = cotangent_owners[active_probe][batch_index]
                    # A fork seeds its roots from contiguous clones of these
                    # accumulators, the final pass from the originals. Only
                    # when every original is contiguous are the two seed
                    # layouts, and so the windows' bits, the same (#994).
                    if not all(tensor.is_contiguous()
                               for tensor in owner.resident_tensors()):
                        noncontiguous_seeds.add((active_probe, batch_index))
                        counters.replay["noncontiguous_cotangent_seeds"] = len(
                            noncontiguous_seeds)
                        if observer is not None:
                            raise RuntimeError(
                                "Stage B spill replay needs contiguous shared-state "
                                "cotangent accumulators: the windowed replay seeds "
                                "its earlier windows from contiguous forks, so a "
                                "non-contiguous original would not reproduce them")
                    replay_owner = None
                    try:
                        if not final:
                            replay_owner = owner.fork_for_replay(
                                max_resident_bytes=operator_windows[
                                    "max_replay_cotangent_bytes"])
                            owner = replay_owner
                        storage.check_auxiliary(
                            batches, cotangents=cotangent_owners,
                            extra=() if final else owner.resident_tensors())
                        if _free_gib() < min_free_gib:
                            raise RuntimeError(
                                "joint window replay crossed free UMA floor")
                        cpu_rng = torch.get_rng_state()
                        cuda_rng = (torch.cuda.get_rng_state(runner.device)
                                    if torch.device(runner.device).type == "cuda"
                                    else None)
                        incoming_grad = grad_plane[(active_probe, batch_index)].to(
                            runner.device)
                        x_in = boundary_cpu.to(
                            device=runner.device, dtype=runner.dtype
                        ).detach().requires_grad_(True)
                        isolated = profile.isolated_layer_pass_state(
                            batch.shared_pass_state, runner.layers[layer])
                        isolated = owner.graft(isolated)
                        out = runner.isolated_layer(batch, layer, x_in,
                                                    pass_state=isolated)
                        roots, root_grads = owner.produced_roots()
                        torch.autograd.backward(
                            [out, *roots], [incoming_grad, *root_grads])
                        if observer is not None:
                            observer.end_batch()
                        owner.harvest()
                        if not final:
                            if _state_storage_bytes(owner.resident_tensors()) > \
                                    operator_windows["max_replay_cotangent_bytes"]:
                                raise RuntimeError(
                                    "joint replay cotangents exceed their "
                                    "resident budget")
                            storage.check_auxiliary(
                                batches, cotangents=cotangent_owners,
                                extra=owner.resident_tensors())
                        if not torch.equal(cpu_rng, torch.get_rng_state()) or (
                                cuda_rng is not None and not torch.equal(
                                    cuda_rng, torch.cuda.get_rng_state(runner.device))):
                            raise RuntimeError(
                                "joint operator replay source consumed Torch RNG")
                        if x_in.grad is None:
                            raise RuntimeError(
                                "joint operator replay produced no input cotangent")
                        if final:
                            grad_plane[(active_probe, batch_index)] = (
                                x_in.grad.detach().to("cpu"))
                            storage.check_auxiliary(batches,
                                                    cotangents=cotangent_owners)
                    finally:
                        boundary_cpu = None
                        out = x_in = incoming_grad = isolated = None
                        roots = root_grads = None
                        replay_owner = owner = None
            counters.replay["layer_passes"] += 1

        def consume_window_probe(probe_index, terms, diagnostics, window_receipt):
            operator_window_receipts.append(dict(layer=layer,
                                                 probe_index=probe_index,
                                                 **window_receipt))
            for name, diagnostic in diagnostics.items():
                g_trace[name] += diagnostic["g_trace"]
            for key, components in terms.items():
                joint_components[key].append(components)
                value = squared_signed(components["total"])
                s2[key] += value
                s4[key] += value * value
                x2_probe[key].append(value)

        measured = {name: linears[name] for name in names if render_formats[name]}
        source_seal = {name: JointOperatorStatisticsLease._source_fingerprint(
            module.weight) for name, module in measured.items()}
        sealed_windows = [
            SimpleNamespace(
                original_full_target_names=tuple(window["names"]),
                statistics_bytes=int(window["statistics_bytes"]),
                render_file_upper_bound_bytes=int(
                    window["render_file_upper_bound_bytes"]),
                candidate_count=int(window["candidate_count"]),
            )
            for window in resolved_windows
        ]

        window_kernel: KernelTimeProfiler | None = None
        window_started = time.time()
        replay_window: int | None = None
        # One span per window, opened before its staged-render wait and
        # closed after its units commit. A window whose units were all
        # journalled earlier gets no after_window call; its span closes as
        # ``skipped`` when the next window opens, or when the replay returns.
        window_span = None

        def close_skipped_window():
            nonlocal window_span, window_kernel
            if window_kernel is not None:
                # A skipped window's profiler ends here, not at an
                # after_window. On a resume it holds the spill captures
                # (PQ #1172), so its kernel time is counted.
                window_kernel.__exit__(None, None, None)
                counters.kernel_block(window_kernel)
                window_kernel = None
            if window_span is not None and not window_span.closed:
                counters.io.close(window_span, outcome="skipped")
            window_span = None

        def before_window(window_index, window_names):
            nonlocal window_kernel, window_started, replay_window, window_span
            del window_names
            close_skipped_window()
            window_span = counters.io.open("window", window=int(window_index))
            # The wait has its own child span, as the commit does, so a slow
            # window names which of the two it spent (PQ #1207). Without an
            # executable readset there is no wait, and the span says so.
            with counters.io.span("window-wait", window=int(window_index)):
                if executable:
                    # PQ #917: the production window-readiness body -- the
                    # render phase first, then the bounded staged-render
                    # wait over the window's exact sealed entries, before
                    # observe_and_project_retained_windows opens the PWC
                    # retained window. The retained load below still owns
                    # every lease check. Skipped/resumed windows still
                    # enter the phase: the sealed list never changes.
                    prepare_retained_window_read(
                        window_index, record=record, progress=progress)
            replay_window = int(window_index)
            window_kernel = _stage_b_kernel_profiler()
            window_kernel.__enter__()
            window_started = time.time()
            counters.enter_phase()
            counters.open_window(window_index,
                                 resolved_windows[window_index])

        def backward_reporting(*, probe_index, final, lease):
            # Report each replay probe pass under the executable contract.
            # A complete resume uses the last visited window's phases;
            # before any window is visited, None denotes window zero.
            # Ordering: observe_and_project_retained_windows opens the
            # candidate retained_window before invoking this callback, so
            # the retained PWC lifetime already holds when the replay phase
            # is entered here and the boundary prefetch inside
            # replay_backward runs under the already-reported phase.
            if executable:
                # A spill-sealed plan has no window replay phases: the
                # zero-pending resume reads under the probe's spill phase.
                progress.enter_read_phase(
                    executable_spill_phase_name(probe_index) if sealed_spill
                    else executable_replay_phase_name(replay_window, probe_index))
            with counters.io.span("replay", window=replay_window,
                                  probe=int(probe_index), mode="window"):
                return replay_backward(
                    final=final, lease=lease, probe=probe_index)

        def after_window(window_index, window_names):
            nonlocal window_kernel
            kernel_active_s = None
            try:
                if window_kernel is not None:
                    window_kernel.__exit__(None, None, None)
                    counters.kernel_block(window_kernel)
                    kernel_active_s = (window_kernel.kernel_active_s
                                       if not window_kernel.error else None)
            finally:
                window_kernel = None
            counters.close_window(window_index,
                                  kernel_active_s=kernel_active_s,
                                  wall_s=time.time() - window_started)
            with counters.io.span("commit", window=int(window_index)) as commit_span:
                commit_span.attrs["units"] = commit_streamed_units(window_names)
            progress.window_done(resolved_windows[window_index])
            counters.enter_phase()
            counters.mark_phase_units(len(completed_units))
            progress.commit()
            if window_span is not None:
                counters.io.close(window_span)

        spill_driver = None
        if spill is not None:
            spill_modules = {name: module for name, module in measured.items()
                             if name in spill_pending}
            spill_specs = {name: {fmt: fr.get_format(fmt) for fmt in render_formats[name]}
                           for name in spill_modules}

            def spill_observer(probe_index):
                return spill.capture(
                    probe_index, spill_modules, spill_specs,
                    activation_max_abs=joint_activation_maxima(production_cache),
                    projection_backend=projection_backend)

            def profile_group(group, observer):
                # capture_group replaces each batch's incoming cotangent with
                # the layer input's, and the ladder passes the same batches
                # again, so every pass gets the incoming cotangent back
                # (PQ #1151). A read of the scratch is already a fresh copy.
                keys = [(0, int(item[0])) for item in group]
                saved = {key: grad_plane[key] for key in keys}
                try:
                    capture_group(group, 0, observer)
                finally:
                    for key, tensor in saved.items():
                        grad_plane[key] = tensor
                    saved = None

            def spill_capture(probe_index):
                # The probe's one pass reads its boundaries under its spill
                # phase, where the spill-sealed read plan stages them (PQ
                # #1011). No statistics lease exists yet, so no statistics
                # hook fires during it.
                if executable:
                    progress.enter_read_phase(
                        executable_spill_phase_name(probe_index))
                if workspace_profile is not None and int(probe_index) == 0:
                    # PQ #1151: measure the capture workspace where the
                    # production pass would run, then stop the quantum.
                    with counters.io.span("workspace-profile", probe=int(probe_index)):
                        profile_capture_workspace(
                            workspace_profile, storage=storage, batches=batches,
                            layer=layer,
                            run_group=profile_group,
                            observed=lambda: spill_observer(probe_index),
                            guard=guard,
                            declared_workspace_bytes=(
                                retained_budget.workspace_reserve_bytes),
                            capture_reserve_bytes=spill.capture_reserve_bytes,
                            capture_batch=capture_batch, device=runner.device,
                            identity={
                                "quantum_id": quantum_id, "layer": layer,
                                "record_identity_sha256": record.get("identity_sha256"),
                                "replay_regime": dict(replay_regime),
                                "probe_microbatch": probe_microbatch,
                                "stored_batches": len(batches),
                                "device_envelope_bytes": execution.get(
                                    "device_envelope_bytes"),
                                "git_commit": _checkpoint_git_commit()})
                with counters.io.span("spill-capture", probe=int(probe_index)), \
                        spill_observer(probe_index) as observer:
                    replay_backward(final=True, lease=None, probe=probe_index,
                                    observer=observer)

            def spill_replay(*, window_index, probe_index, lease):
                # Reads nothing: the window's render phase stays current.
                if guard is not None:
                    check_operator_allocation(
                        guard, "before_joint_spill_window_replay", reserve_bytes=(
                            operator_windows["workspace_reserve_bytes"]
                            + lease.statistics_capacity_bytes
                            - lease.resident_statistics_bytes
                            + spill.replay_reserve_bytes))
                with counters.io.span("replay", window=int(window_index),
                                      probe=int(probe_index), mode="spill"):
                    spill.replay(window_index, probe_index, lease)

            spill_driver = SimpleNamespace(capture=spill_capture, replay=spill_replay)

        counters.open()
        try:
            observe_and_project_retained_windows(
                measured,
                {name: {fmt: fr.get_format(fmt) for fmt in render_formats[name]}
                 for name in measured},
                production_cache, operator_windows,
                retained_budget=retained_budget, n_probes=n_probes,
                source_bytes=retained_operator_windows["source_reserve_bytes"],
                backward=backward_reporting,
                record_operator=_record_joint_operator,
                consume_probe=consume_window_probe,
                collect_col_energy=False, backend=projection_backend,
                guard=guard, source_fingerprints=source_seal,
                completed_names=set(measured) & completed_units,
                sealed_windows=sealed_windows,
                before_window=before_window,
                after_window=after_window,
                spill=spill_driver,
                # The loader threads hash each render as they load it, so
                # _record_joint_operator reads a hash (PQ #1192).
                render_identities=True,
            )
            close_skipped_window()
        finally:
            if spill is not None:
                counters.replay["spill"] = dict(spill.telemetry)
            if window_kernel is not None:
                window_kernel.__exit__(None, None, None)
                window_kernel = None
            runner.context.unload(layer)

        # ---- the tail after the last window (PQ #1187) ---------------------
        # One top-level span from here to the return: the handoff write, the
        # payload assembly and the final check of every row. It commits no
        # units (nothing here is journalled; the rows are written under
        # ``records-out``), so it runs under the last window's render phase
        # and its stall allowance. A failure leaves it open, and the failure
        # path closes it as interrupted, as it does a window's.
        payload_span = counters.io.open(
            "payload", units=len(joint_rows),
            rows=sum(len(rows) for rows in joint_rows.values()))
        # ---- band-serial handoff for layer - 1 (PQ #996) ------------------
        # The final pass above wrote the boundary-``layer`` plane and
        # harvested every owner; both stay readable until ``storage`` closes
        # (the plane may live in its cotangent scratch).
        if handoff_emitter is not None:
            with counters.io.span("handoff-out"):
                handoff_emitter.emit(grad_plane=grad_plane,
                                     cotangent_owners=cotangent_owners,
                                     n_probes=n_probes, n_batches=len(row_offsets))

    # ---- payload (§6.4 cost.pkl) -----------------------------------------
    payload = _assemble_streamed_aura_payload(
        linears=linears, names=names, formats=fmts, formats_by_qname=unit_formats,
        unmeasured_formats_by_qname={}, n_probes=n_probes, token_scope=token_scope,
        seed_base=seed_base, temperature=temperature, dw_dtype="float32",
        measurement_dtype=runner.dtype, n_linear_chunks=1, calib_ids=calib_ids,
        omitted_packed_experts=[], cb_provenance=cb_provenance,
        checkpoint_git_commit=checkpoint_git_commit, collect_col_energy=False,
        s2=s2, s4=s4, x2_probe=x2_probe, dw_src=dw_src, g_trace=g_trace,
        col_energy={}, weight_mse_diagnostic={})
    payload["costs"] = joint_rows
    probe_identity_sha256 = identity_sha256(joint_probe_identity)
    if probe_identity_sha256 != identity_sha256(joint_probe):
        raise RuntimeError("joint probe identity changed after it was validated")
    payload["provenance"].update({
        "cost_mode": "aura", "joint_activation": True,
        "cost_currency": "joint_aura_predicted_dloss",
        "joint_aura_identity": joint_run_identity,
        "joint_aura_identity_sha256": identity_sha256(joint_run_identity),
        "probe_identity": joint_probe_identity,
        "probe_identity_sha256": probe_identity_sha256,
        "joint_operator_windows": operator_window_receipts,
        "measurement_status": "research",
        "uncertainty_scope": "probe_sampling_conditional_on_fixed_calibration",
        "distributed_quantum": extra["distributed_quantum"],
        "campaign_binding": {
            "plan_sha256": record["campaign"]["plan_sha256"],
            "prepared_sha256": record["campaign"]["prepared_sha256"],
            "read_manifest_sha256": record["campaign"]["read_manifest_sha256"],
            "campaign_scope": record["campaign"].get("campaign_scope"),
            "unit_roster_sha256": record["campaign"].get("unit_roster_sha256"),
        },
        "adjoint_slice_sha256": record["adjoint"]["slice_sha256"],
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
        **({"catalog_extension": record["catalog_extension"]}
           if record.get("catalog_extension") is not None else {}),
        **({"served_activation_policy": activation_policy[0]} if activation_policy is not None else {}),
        **({"stage_b_resource_policy": resource_policy} if resource_policy is not None else {}),
    })
    if set(joint_rows) != set(names):
        raise RuntimeError("layer quantum incomplete unit coverage")
    for name, rows in joint_rows.items():
        if set(rows) != set(unit_formats[name]):
            raise RuntimeError(
                f"layer quantum candidate scope mismatch for {name}")
        for fmt, row in rows.items():
            if not validate_row(row):
                raise RuntimeError(f"invalid measured joint cost for {name}@{fmt}")
    counters.io.close(payload_span)
    return payload


# --------------------------------------------------------------------------
# Orchestration: head inputs, writes, exit codes (§6.2 steps 2 and 6)
# --------------------------------------------------------------------------


def _io_counters() -> dict:
    return read_proc_io()


def _resolved_units(resolved_windows) -> int:
    return (sum(len(window["names"]) for window in resolved_windows)
            if resolved_windows is not None else 0)


def write_failure_counters(record, *, counters, io_spans, sampler, error,
                           units_total=0, stamps=None) -> Path | None:
    """Write a failed run's counters to its counters path, and return it.

    The document is the success document with ``units`` done as ``None``
    and an ``outcome`` block naming the error and the spans it interrupted.
    Spans still open close as ``interrupted``. Before the quantum built its
    counters (a failure in the head), the document carries the spans, the
    GPU power and the outcome only. ``stamps`` are merged in as they are
    (the dispatcher's load-phase grace). Nothing else is written:
    ``status.json`` and ``cost.pkl`` stay the success path's.

    Never raises: a failure here is printed, and the run's own error is the
    one that propagates.
    """
    try:
        open_names = io_spans.open_names
        io_spans.close_open(error=error)
        outcome = failure_outcome(error, open_spans=open_names)
        if counters is not None:
            document = counters.finish(units_done=None, units_total=units_total)
        else:
            gpu = sampler.stop()
            document = {
                "schema": QUANTUM_COUNTERS_SCHEMA,
                "quantum_id": record.get("quantum_id"),
                "identity_sha256": record.get("identity_sha256"),
                "units": [None, int(units_total)],
                "gpu_joules": gpu.get("gpu_joules"),
                "gpu_power_w_p50": gpu.get("gpu_power_w_p50"),
                "gpu_power_w_p95": gpu.get("gpu_power_w_p95"),
                "gpu_power_w_max": gpu.get("gpu_power_w_max"),
                "gpu_sampler_samples": gpu.get("sample_count"),
                "gpu_power_envelope_w": 140.0,
                "io_spans": list(io_spans.records),
            }
            if "sampler_error" in gpu:
                document["gpu_sampler_error"] = gpu["sampler_error"]
        document.update(stamps or {})
        document["outcome"] = outcome
        space = record.get("output_space") or {}
        target = space.get("counters") or (
            str(Path(space["root"]) / "counters.json") if space.get("root") else None)
        if target is None:
            print("quantum counters: no output space to write the failure "
                  "counters into", flush=True)
            return None
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            path, (json.dumps(document, sort_keys=True, indent=2,
                              allow_nan=False, default=str) + "\n").encode())
        print(f"quantum counters: wrote the failed run's counters to {path} "
              f"(open spans {open_names})", flush=True)
        return path
    except Exception as exc:  # noqa: BLE001 -- never mask the run's error
        print(f"quantum counters: could not write the failure counters: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None


def publish_quantum_outputs(record, *, payload, result, counters,
                            units_total=None) -> dict:
    """Write the quantum's outputs (§6.4), only under its output space.

    ``cost.pkl`` only when every row passed ``validate_joint_aura_entry``
    (checked by the core before returning); ``status.json`` is the last act.
    Every path comes from the record's ``output_space``; nothing is written
    anywhere else by this stage. ``units_total`` is the layer's unit count
    (the caller counts the resolved windows' names -- the record seals
    indices only, D2); when omitted it falls back to the payload's own unit
    count, or 0 for a gapped quantum that never resolved.
    """
    if units_total is None:
        units_total = len(payload["costs"]) if payload is not None else 0
    units_total = int(units_total)
    units_done = len(payload["costs"]) if payload is not None else 0
    status = "complete" if payload is not None and units_done == units_total else "gapped"
    if payload is not None:
        cost_path = Path(record["output_space"]["cost_payload"])
        atomic_write_bytes(
            cost_path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        result["cost"] = {
            "path": str(cost_path),
            "sha256": hashlib.sha256(cost_path.read_bytes()).hexdigest(),
        }
    result["status"] = status
    result["units_done"] = units_done
    result["units_total"] = units_total
    result["passed"] = status == "complete"
    counters = {**counters, "outcome": {"status": status}}
    atomic_write_bytes(
        Path(record["output_space"]["counters"]),
        (json.dumps(counters, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    atomic_write_bytes(
        Path(record["output_space"]["results"]),
        (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    status_record = {
        "schema": QUANTUM_STATUS_SCHEMA,
        "quantum_id": record["quantum_id"],
        "identity_sha256": record["identity_sha256"],
        "status": status,
        "units": [units_done, units_total],
        "unix": time.time(),
    }
    root = Path(record["output_space"]["root"])
    status_path = Path(record["output_space"].get("status", str(root / "status.json")))
    atomic_write_bytes(
        status_path,
        (json.dumps(status_record, sort_keys=True, allow_nan=False) + "\n").encode())
    return status_record


def run_layer_quantum(
    config, *, record, adjoint_slice, plan_sha256, prepared, output_root,
    data_manifest_sha256=None, resume=False, adjoint_handoff=None,
    emit_handoff=False, progress_grace=None,
) -> dict:
    """Load the head phase and execute one quantum (§6.2 steps 2-6).

    ``adjoint_handoff`` is a bound handoff from quantum ``layer + 1``
    (band-serial, PQ #996); ``emit_handoff`` publishes this quantum's own for
    ``layer - 1``. Both are chosen by the dispatcher; neither falls back.
    ``progress_grace`` is the dispatcher's grace stamps (load and compute
    phases): stamped into results.json and counters.json as
    ``progress_grace``, never read.
    """
    import torch

    from .aura_cost import _aura_source_sha256
    from .calibration_data import load_calibration_input
    from .cost_streaming import build_streamed_causal_lm
    from .gpu_guard import require_cuda_hot_path
    from .joint_prewarm_phases import HEAD_PHASE
    from .joint_projection_backend import executing_image, prewarm_projection_backend
    from .model_profiles import detect_profile
    from .production_weight_cache import ProductionWeightCache
    from .residency_map import bind_residency_manifest, residency_report
    from .tessera_joint_aura import (
        ACTIVATION_SCALE_ENV,
        _bound,
        _prepare_file_read_bound,
        _preflight_run_prepared,
        _same,
        _seed_source_identity_cache,
        load_measured_anchor_input,
        require_prepared_digests,
    )
    from .tessera_reader import load_declared_reader

    require_cuda_hot_path("joint_cost_quantum", "cuda")
    from .autoscale import require_bounded_capture_environment

    execution = config["execution"]
    # The replay regime is a launch setting sealed in the campaign container
    # spec, never a plan field: the plan is bound to the prepared inputs.
    from .joint_replay_regime import (
        ReplayRegimeRefused, handoff_regime_refusal, replay_regime_from_environment)
    if "replay_regime" in execution:
        raise QuantumIdentityRefused(
            "the Stage B replay regime is a launch setting "
            "(PRISMAQUANT_STAGE_B_REPLAY_REGIME), not a plan execution field")
    try:
        replay_regime = replay_regime_from_environment(os.environ)
    except ReplayRegimeRefused as exc:
        raise QuantumIdentityRefused(str(exc)) from exc
    # The bf16 reduction flag is a launch setting of the same kind (PQ #1028).
    from .matmul_arithmetic import (
        MatmulArithmeticRefused, bf16_reduction_from_environment, pin_matmul_arithmetic)
    try:
        bf16_reduction = bf16_reduction_from_environment(os.environ)
    except MatmulArithmeticRefused as exc:
        raise QuantumIdentityRefused(str(exc)) from exc
    # PQ #1065: compare the setting with the slice's stamp before any head
    # work. The core repeats the check on the verified slice and the live
    # flag, for every caller.
    if isinstance(adjoint_slice, dict) and "run_identity" in adjoint_slice:
        require_slice_bf16_reduction(
            adjoint_slice, bf16_reduction,
            where=f"quantum {record.get('quantum_id', '?')}")
    handoff_capture_batch = None
    if emit_handoff:
        # The producer's capture batch must be the batch size the slice's
        # chain regime rolls at (PQ #994, #997): the plane it hands off is
        # then the consumer's chain plane. The core repeats the check on the
        # verified slice; this one refuses before any head work.
        from .joint_adjoint_slices import ChainRegimeRefused, chain_regime_of
        from .joint_replay_regime import normalize_replay_regime
        try:
            chain_batch_size = chain_regime_of(
                adjoint_slice["run_identity"])["batch_size"]
        except (ChainRegimeRefused, KeyError, TypeError) as exc:
            raise QuantumIdentityRefused(
                f"the stage-A slice's chain regime: {exc}") from exc
        refusal = handoff_regime_refusal(replay_regime,
                                         chain_batch_size=chain_batch_size)
        if refusal:
            raise QuantumIdentityRefused(refusal)
        handoff_capture_batch = normalize_replay_regime(replay_regime)["capture_batch"]
    from .joint_stageb_resources import enforce_device_policy
    # Bind the readset before the first head read, so the head slice and
    # every declared entry after it resolve through residency (PQ #1024).
    bind_residency_manifest(
        data_manifest_sha256 or record["campaign"].get("read_manifest_sha256"))
    head_slice = None
    readset_block = record.get("executable_readset")
    if isinstance(readset_block, dict) and readset_block.get("head_slice") is not None:
        # PQ #1010: the head intake ran once, at prepare. The slice is the
        # first head read; the device limits it carries were verified there.
        from .joint_stage_b_head import (
            HeadSliceRefused, head_slice_limits, read_quantum_head_slice)
        try:
            head_slice, head_slice_binding, head_files = read_quantum_head_slice(
                config, record=record, prepared=prepared, plan_sha256=plan_sha256)
        except HeadSliceRefused as exc:
            raise QuantumIdentityRefused(str(exc)) from exc
        device_envelope = enforce_device_policy(
            config, verified_limits=head_slice_limits(head_slice))
    else:
        device_envelope = enforce_device_policy(config)
    if (config.get("qualification_window") is not None
            or execution.get("retained_operator_windows") is not None):
        require_bounded_capture_environment(os.environ)

    os.environ[ACTIVATION_SCALE_ENV] = execution["production_act_scales"]
    torch.set_num_threads(1)
    pin_matmul_arithmetic()

    layer = int(record["layer"])
    space = Path(record["output_space"]["root"])
    space.mkdir(parents=True, exist_ok=True)

    result = {
        "schema": "prismaquant.joint_cost_quantum.execution.v1",
        "command": "layer-quantum",
        "quantum_id": record["quantum_id"],
        "identity_sha256": record["identity_sha256"],
        "plan_sha256": plan_sha256,
        "adjoint_slice_sha256": record["adjoint"]["slice_sha256"],
        "env": {"host": socket.gethostname(), "started_epoch": time.time(),
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "affinity": sorted(os.sched_getaffinity(0))},
        "dev_mode": dev_mode_stamp(),
        "phases": [], "passed": False,
    }
    result["env"]["container_content_sha256"] = executing_image()
    stamps = ({} if progress_grace is None
              else {"progress_grace": list(progress_grace)})
    result.update(stamps)
    result["device_envelope"] = device_envelope
    # Band-serial mode is a fact about this execution, never about the
    # record or the payload: both are the chain-mode bytes (PQ #996).
    result["adjoint_cotangent_source"] = (
        {"mode": "chain"} if adjoint_handoff is None else
        {"mode": "handoff", "handoff_sha256": adjoint_handoff["handoff_sha256"],
         "producer": dict(adjoint_handoff["producer"])})
    handoff_emitter = None
    if emit_handoff:
        # Before any GPU work: an admitted action without the produced-output
        # binding its handoff needs refuses here, like a digest mismatch.
        try:
            handoff_emitter = HandoffEmitter(
                record=record, adjoint_slice=adjoint_slice,
                boundary_storage=execution["boundary_storage"],
                capture_batch=handoff_capture_batch,
                publication=bind_handoff_publication(
                    boundary_storage=execution["boundary_storage"]))
        except QuantumHandoffRefused as exc:
            raise QuantumIdentityRefused(str(exc)) from exc

    started, before_io = time.time(), _io_counters()
    # The power sampler and the span log start before the head, so the head
    # has a span and its watts, and the counters' joules and wall time cover
    # the same interval. A failure writes what they hold (PQ #1144 follow-up).
    power = GpuPowerSampler().start()
    io_spans = quantum_io_spans(record["quantum_id"], power)
    runner = None
    payload = None
    counters = None
    resolved_windows: list[dict] | None = None
    try:
        # The ``head`` span covers everything before the core enters
        # checkpoint-load: the head read, the source runner and the window
        # handshake. That is the stretch PrismaBuild reports as ``head``.
        head_span = io_spans.open("head")
        # Bind before cache/intake work. The core repeats this idempotently
        # for direct callers and stamps the actual arithmetic in row identity.
        if head_slice is not None:
            from .joint_stage_b_head import load_quantum_head, read_prepared_head
            try:
                prepared_header = read_prepared_head(head_files)
            except HeadSliceRefused as exc:
                raise QuantumIdentityRefused(str(exc)) from exc
        else:
            prepared_header = json.loads(_bound(prepared, "prepared anchors").read_text())
        result["served_quantizer"] = bind_joint_served_quantizer(
            prepared_header["formats_by_qname"])
        reader = load_declared_reader(config.get("reader"))
        reader_identity = None if reader is None else reader.identity
        implementation = _aura_source_sha256()
        projection_backend = prewarm_projection_backend(
            execution.get("projection_backend"), device="cuda")
        result["projection_backend"] = projection_backend.identity
        if head_slice is not None:
            try:
                head = load_quantum_head(
                    config, record=record, head_slice=head_slice, files=head_files,
                    completion=prepared_header, plan_sha256=plan_sha256,
                    implementation_sha256=implementation,
                    reader_identity=reader_identity,
                    projection_backend=projection_backend.identity,
                    progress_phase=HEAD_PHASE)
            except HeadSliceRefused as exc:
                raise QuantumIdentityRefused(str(exc)) from exc
            completion, cache = head.completion, head.cache
            formats_by_qname = head.formats_by_qname
            ids, calibration = head.calibration_ids, head.calibration
            result["calibration_input"] = calibration
            result["renders_synthesized_now"] = 0
            result["wire_validation"] = "historical-qualified-wire"
            result["head_slice"] = {
                **head_slice_binding,
                "producer_implementation_sha256": head.producer_implementation_sha256}
            head_units, head_cells = head.units, head.measured_cells
            progress_base = head.progress_units
            identity_cache_bytes = head.identity_cache_bytes
        else:
            _preflight_run_prepared(prepared, plan_sha256=plan_sha256,
                                    implementation_sha256=implementation,
                                    reader_identity=reader_identity,
                                    projection_backend=projection_backend.identity)
            data = load_measured_anchor_input(
                config["inputs"], reader=reader, synthesis_device="cuda",
                progress_phase=HEAD_PHASE,
                head_checkpoint=space / "checkpoints" / "head-walk",
                head_resume=resume,
                require_existing_renders=True, verify_payloads=False,
                # The plan's allowance, as Stage A, the prepare and the
                # head-slice producer pass it (PQ #1023).
                historical_encoder_reuse=config.get("historical_encoder_reuse"))
            _same(config["model"], data.census["model"], "requested source model")
            _same(data.census["attention_implementation"], "eager",
                  "qualified source attention")
            ids, calibration = load_calibration_input(
                config["calibration_input"]["path"],
                expected_sha256=config["calibration_input"]["sha256"],
                n_samples=execution["n_calib_samples"],
                seqlen=execution["calib_seqlen"])
            original_draw = data.payload["provenance"]["hessian"]["calibration_identity"]
            for name in ("fit_ids_sha256", "text_sha256", "nsamples", "seqlen", "seed"):
                _same(calibration["provenance"].get(name), original_draw.get(name),
                      f"original full draw {name}")
            result["calibration_input"] = calibration
            result["renders_synthesized_now"] = data.synthesized_now

            completion = json.loads(_bound(prepared, "prepared anchors").read_text())
            _same(completion.get("schema"), "prismaquant.tessera_joint_aura.prepared.v3",
                  "prepared schema")
            require_prepared_digests(completion, plan_sha256=plan_sha256,
                                     implementation_sha256=implementation)
            _same(completion.get("calibration_input"), calibration,
                  "prepared calibration")
            _same(completion["formats_by_qname"],
                  {n: list(v) for n, v in data.formats_by_qname.items()},
                  "prepared exact candidate roster")
            cache = pickle.loads(
                _bound(completion["production_cache"], "qualified PWC").read_bytes())
            if not isinstance(cache, ProductionWeightCache):
                raise RuntimeError("prepared cache is not ProductionWeightCache")
            _same(cache.metadata["inputs"], data.inputs, "prepared source bindings")
            # The two policies are run seals (PQ #1147): dev mode prints a
            # difference and runs under the plan's own policy.
            seal_check("prepared Stage B resource policy", config.get("stage_b_resource_policy"),
                       completion.get("stage_b_resource_policy"), where="Stage B quantum",
                       refusal=lambda: ValueError(
                           "prepared Stage B resource policy: identity mismatch"))
            if config.get("stage_b_resource_policy") is not None:
                cache._joint_stage_b_resource_policy = dict(config["stage_b_resource_policy"])
            seal_check("prepared served activation policy", config.get("served_activation_policy"),
                       completion.get("served_activation_policy"), where="Stage B quantum",
                       refusal=lambda: ValueError(
                           "prepared served activation policy: identity mismatch"))
            if config.get("served_activation_policy") is not None:
                if record.get("catalog_extension") is None:
                    raise RuntimeError("served activation policy requires an explicit catalog extension")
                from .joint_served_activation import activate_policy
                activate_policy(cache, config["served_activation_policy"])
            expected_renders = {pair: cache.metadata["verified_cells"][pair]["render_file_sha256"]
                                for pair in data.cells}
            cache.require_file_load_sha256(
                expected_renders,
                max_file_bytes=_prepare_file_read_bound(
                    data, max_render_bytes=config["max_render_bytes"]))
            result["wire_validation"] = "historical-qualified-wire"
            formats_by_qname = data.formats_by_qname
            head_units, head_cells = len(data.formats_by_qname), len(data.cells)
            progress_base = data.progress_committed
            identity_cache_bytes = None

        from .cost_streaming import build_streamed_model_identity
        if identity_cache_bytes is not None:
            # The slice declared the bound cache and the head read it: no
            # copy into the output space, and nothing is written back.
            identity_cache = {"identity_cache_bytes": identity_cache_bytes}
        else:
            identity_cache = {"identity_cache_path": _seed_source_identity_cache(
                config, space / "run")}
        runner = build_quantum_source_runner(
            config, offload_folder=space / "run" / "offload",
            sealed_head_tensors=((record.get("executable_readset") or {})
                                 .get("head_source") or {}).get("tensors"))

        source = build_streamed_model_identity(runner, config["model"], **identity_cache)
        # A run seal (PQ #1147): dev mode stamps a source other than the
        # prepared one and continues.
        seal_check("prepared source identity", completion.get("source_model_identity"),
                   source, where="prepared completion versus the running source",
                   refusal=lambda: ValueError("prepared source identity: identity mismatch"))
        result.update(source_model_identity=source,
                      units=head_units, measured_cells=head_cells)

        execution_runtime = quantum_runtime_execution(config, replay_regime=replay_regime)
        # D2 handshake, before any GPU work or progress: the record seals
        # window indices only, so membership and footprints are recomputed
        # from the sealed budget and handshook here. The chunk frontier,
        # counters and progress all size from the resolved windows -- never
        # from the record's index entries.
        retained = quantum_retained_state(execution_runtime)
        roster = quantum_layer_roster(runner, formats_by_qname, layer)
        if head_slice is not None:
            _same(roster.names, sorted(head_slice["intake"]["layer_formats"]),
                  "head slice layer roster")
        resolved_windows = resolve_quantum_windows(
            record, layer=layer, names=roster.names, linears=roster.linears,
            render_formats=roster.render_formats, production_cache=cache,
            operator_windows=retained.operator_windows,
            retained_budget=retained.retained_budget,
            source_bytes=retained.source_bytes)
        executable_block = record.get("executable_readset")
        if isinstance(executable_block, dict) and isinstance(
                executable_block.get("prepared_input"), dict):
            # PQ #917: the sealed prepared membership must equal the live
            # geometry recomputed above, before any GPU work or progress.
            try:
                check_prepared_windows_against_resolved(
                    executable_block["prepared_input"].get("windows", []),
                    resolved_windows,
                    quantum_id=record.get("quantum_id"))
            except ValueError as exc:
                raise QuantumIdentityRefused(str(exc)) from exc
        result["resolved_windows"] = len(resolved_windows)
        counters = QuantumCounters(
            quantum_id=record["quantum_id"], identity_sha256=record["identity_sha256"],
            chunks=record["chunks"],
            frontier=ChunkFrontier(chunks=record["chunks"],
                                   windows=resolved_windows),
            io_spans=io_spans, sampler=power, started=started)
        # Head-phase currency continues from the head-committed base (§6.2
        # step 5): the same cumulative units the single run reports.
        progress = QuantumProgress(frontier=counters._frontier,
                                   base_units=progress_base)
        io_spans.close(head_span)
        payload = run_layer_quantum_core(
            runner, cache, ids.to(runner.device), formats_by_qname,
            record=record, adjoint_slice=adjoint_slice, execution=execution_runtime,
            output_root=output_root, projection_backend=projection_backend,
            resume=resume, resolved_windows=resolved_windows,
            counters=counters, progress=progress,
            adjoint_handoff=adjoint_handoff, handoff_emitter=handoff_emitter)
        if handoff_emitter is not None:
            result["handoff"] = dict(handoff_emitter.published)
        torch.cuda.synchronize()
        result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if result["peak_gpu_bytes"] > config["max_gpu_bytes"]:
            raise RuntimeError("observed GPU allocation exceeds declared budget")
    except BaseException as error:
        # Before the runner's teardown, which can take the box down with it
        # (Stage A, PQ #899): the counters of a failed run are the evidence
        # of where it failed. This covers exceptions, not SIGKILL.
        write_failure_counters(record, counters=counters, io_spans=io_spans,
                               sampler=power, error=error,
                               units_total=_resolved_units(resolved_windows),
                               stamps=stamps)
        raise
    finally:
        # Between the core and ``records-out`` (PQ #1187). On a failure the
        # counters were written above, so this span reaches the log only.
        with io_spans.span("teardown"):
            if runner is not None:
                completed, runner = runner, None
                completed.shutdown()
            result["env"]["finished_epoch"] = time.time()
            result["phases"].append({"phase": "layer-quantum", "start_epoch": started,
                                     "end_epoch": result["env"]["finished_epoch"]})
            result["io_before"], result["io_after"] = before_io, _io_counters()
            residency = residency_report()
            if residency is not None:
                result["residency"] = residency

    # ---- writes: only under layer-quanta/layer-NNN/ (§6.4) ---------------
    units_total = _resolved_units(resolved_windows)
    records_span = io_spans.open("records-out")
    counters_done = {**counters.finish(
        units_done=len(payload["costs"]) if payload else 0,
        units_total=units_total), **stamps}
    try:
        status_record = publish_quantum_outputs(
            record, payload=payload, result=result, counters=counters_done,
            units_total=units_total)
    except BaseException as error:
        io_spans.close(records_span, error=error)
        raise
    # Its line is the record of the writes; counters.json is already out.
    io_spans.close(records_span)
    result["passed"] = status_record["status"] == "complete"
    return result


def progress_grace_stamps(raw: str | None) -> list | None:
    """The dispatcher's grace stamps (load and compute phases), as recorded.

    The stamps explain the grace PrismaBuild enforces; the quantum reads
    nothing from them. A value that is not a JSON list of objects is kept
    as it came, under ``unparsed``, and the run continues.
    """
    if raw is None:
        return None
    try:
        stamps = json.loads(raw)
    except ValueError:
        stamps = None
    if isinstance(stamps, list) and all(isinstance(s, dict) for s in stamps):
        return stamps
    print("[joint-quantum] --progress-grace-derivation is not a JSON list of "
          "objects; recording it unparsed", flush=True)
    return [{"unparsed": raw}]


def build_parser() -> argparse.ArgumentParser:
    """The layer-quantum CLI surface, shared by ``main`` and by tests that
    prove a dispatcher payload parses before any gate runs."""
    parser = argparse.ArgumentParser(
        description="Run one layer quantum of the distributed joint-AURA "
                    "cost campaign (contract §6).")
    parser.add_argument("--quantum", type=Path, required=True,
                        help="the producer's sealed layer-quantum record")
    parser.add_argument("--quantum-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--adjoint-slice", type=Path, required=True,
                        help="the quantum's stage-A slice (PQ #993); never a "
                             "whole receipt")
    parser.add_argument("--adjoint-slice-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-tool", default=None,
                        help="optional cprofile wrapping (default off; the "
                             "single run's profile applies to the campaign "
                             "of record, not every quantum)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-manifest-sha256")
    parser.add_argument("--adjoint-handoff", type=Path, default=None,
                        help="band-serial (PQ #996): quantum layer+1's "
                             "handoff.json; replaces the checkpoint load and "
                             "the render-free chain")
    parser.add_argument("--adjoint-handoff-sha256", default=None)
    parser.add_argument("--progress-grace-derivation", default=None,
                        help="the dispatcher's load-phase grace stamps (JSON "
                             "list): copied into results.json and "
                             "counters.json, never read as a setting")
    parser.add_argument("--emit-adjoint-handoff", action="store_true",
                        help="band-serial (PQ #996): publish this quantum's "
                             "final input cotangent and owner states for "
                             "layer-1")
    from .staged_tier_policy import DEFAULT_ALLOWED_TIERS
    parser.add_argument("--allowed-tiers", default=DEFAULT_ALLOWED_TIERS,
                        help="sealed staged-tier declaration for GPU-consumed "
                             "bulk inputs: comma subset of {ram,ssd}, RAM "
                             "first (default %(default)s). Pool/HDD bulk "
                             "opens refuse under this declaration.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.device != "cuda":
        parser.error("layer quanta are a GPU hot path; --device must be cuda")
    progress_grace = progress_grace_stamps(args.progress_grace_derivation)
    try:
        require_dev_mode("joint_cost_quantum")
        record, adjoint_slice = verify_quantum_identity(
            quantum_path=args.quantum, quantum_sha256=args.quantum_sha256,
            plan_path=args.plan, plan_sha256=args.plan_sha256,
            prepared_path=args.prepared, prepared_sha256=args.prepared_sha256,
            adjoint_path=args.adjoint_slice, adjoint_sha256=args.adjoint_slice_sha256,
            output_root=args.output_root)
        adjoint_handoff = None
        if (args.adjoint_handoff is None) != (args.adjoint_handoff_sha256 is None):
            raise QuantumIdentityRefused(
                "--adjoint-handoff and --adjoint-handoff-sha256 go together")
        if args.adjoint_handoff is not None:
            try:
                adjoint_handoff = load_quantum_handoff(
                    args.adjoint_handoff, args.adjoint_handoff_sha256,
                    record=record, adjoint_slice=adjoint_slice)
                require_band_serial_readset(
                    record, adjoint_handoff, adjoint_slice["checkpoint"],
                    output_root=args.output_root,
                    data_manifest_sha256=args.data_manifest_sha256)
            except QuantumHandoffRefused as exc:
                raise QuantumIdentityRefused(f"adjoint handoff: {exc}") from exc
        else:
            require_chain_readset(
                record, data_manifest_sha256=args.data_manifest_sha256)
    except QuantumIdentityRefused as exc:
        print(f"{IDENTITY_REFUSED_MARKER}: {exc}", flush=True)
        return EXIT_IDENTITY_REFUSED
    from .tessera_joint_aura import _load_plan

    # The readset is bound inside run_layer_quantum; until then the plan
    # itself is the only input read (PQ #1024).
    config = _load_plan(args.plan, args.plan_sha256, defer_pool_reads=True)
    if Path(config["output_root"]).resolve() != Path(args.output_root).resolve():
        print(f"{IDENTITY_REFUSED_MARKER}: plan output_root "
              f"{config['output_root']} is not --output-root {args.output_root}",
              flush=True)
        return EXIT_IDENTITY_REFUSED
    from .staged_tier_policy import activate_staged_tier_policy
    try:
        allowed = activate_staged_tier_policy(args.allowed_tiers)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"[STAGED-TIER] bulk inputs serve from {','.join(sorted(allowed))}; "
          f"pool/HDD bulk opens refuse", flush=True)
    profiler = None
    if args.profile_tool == "cprofile":
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()
    try:
        result = run_layer_quantum(
            config, record=record, adjoint_slice=adjoint_slice, plan_sha256=args.plan_sha256,
            prepared={"path": str(args.prepared), "sha256": args.prepared_sha256},
            output_root=args.output_root,
            data_manifest_sha256=args.data_manifest_sha256, resume=args.resume,
            adjoint_handoff=adjoint_handoff,
            emit_handoff=args.emit_adjoint_handoff,
            progress_grace=progress_grace)
    except QuantumIdentityRefused as exc:
        # The D2 window handshake or the producer campaign check refused a
        # stale record after the digest gate passed: same exit 3, nothing
        # committed (resolution runs before any journal or payload write).
        print(f"{IDENTITY_REFUSED_MARKER}: {exc}", flush=True)
        return EXIT_IDENTITY_REFUSED
    finally:
        if profiler is not None:
            profiler.disable()
            import io
            import pstats

            profiler.dump_stats(str(Path(record["output_space"]["root"]) / "profile.pstats"))
            text = io.StringIO()
            pstats.Stats(profiler, stream=text).sort_stats(
                "cumulative").print_stats(100)
            (Path(record["output_space"]["root"]) / "profile.txt").write_text(
                text.getvalue())
    print(json.dumps({key: result[key] for key in (
        "quantum_id", "passed", "status", "units_done", "units_total")}))
    if result["status"] != "complete":
        return EXIT_GAPPED
    return EXIT_OK


def _boundary_entry_record_to_reference(entry: dict):
    from .joint_adjoint_checkpoints import reference_from_record

    return reference_from_record(entry)


def adjusted_space(output_root) -> Path:
    """The adjoint namespace inside this campaign's output root."""
    return adjoint_space(output_root)


__all__ = [
    "EXIT_GAPPED", "EXIT_IDENTITY_REFUSED", "EXIT_FAILURE", "EXIT_OK",
    "EXIT_USAGE", "IDENTITY_REFUSED_MARKER", "ChunkFrontier",
    "QuantumCounters", "QuantumIdentityRefused", "QuantumProgress",
    "adjusted_space", "quantum_layer_roster", "quantum_retained_state",
    "record_window_indices", "resolve_quantum_windows",
    "run_layer_quantum", "run_layer_quantum_core", "verify_quantum_identity",
]


if __name__ == "__main__":
    raise SystemExit(main())
