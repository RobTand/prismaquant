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
import pickle
import socket
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import torch

from .cost_stage_checkpoint import atomic_write_bytes, canonical_json_sha256
from .joint_adjoint_checkpoints import (
    QUANTUM_COUNTERS_SCHEMA,
    QUANTUM_RECORD_SCHEMA,
    QUANTUM_STATUS_SCHEMA,
    GpuPowerSampler,
    KernelTimeProfiler,
    adjoint_space,
    boundary_entry_directory,
    chain_layers_for,
    dev_mode_stamp,
    load_adjoint_checkpoint,
    load_adjoint_receipt,
    require_dev_mode,
)

#: Exit codes (§6.2/§6.4): 3 is the identity refusal -- nothing written; 4 is
#: the clean gap (status.json says gapped; PB retries the sealed action key).
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_IDENTITY_REFUSED = 3
EXIT_GAPPED = 4

IDENTITY_REFUSED_MARKER = "quantum_identity_refused"


class QuantumIdentityRefused(RuntimeError):
    """A digest, schema or binding mismatch: refuse before writing anything."""


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

    Returns ``(record, adjoint_receipt)``; raises :class:`QuantumIdentityRefused`
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
            if campaign.get(f"{label}_sha256") != supplied:
                raise QuantumIdentityRefused(
                    f"quantum record binds another {label}: "
                    f"record={campaign.get(f'{label}_sha256')!r} argv={supplied}")
        _require_hex(campaign.get("read_manifest_sha256"),
                     "record read_manifest_sha256")
        adjoint = record["adjoint"]
        if adjoint.get("receipt_sha256") is None:
            raise QuantumIdentityRefused(
                "quantum record is unbound (pre-stage-A): re-seal it against "
                "the stage-A receipt with bind_adjoint_receipt -- a new "
                "identity, never an edit (producer D3) -- before publishing")
        _require_hex(adjoint.get("receipt_sha256"), "record adjoint receipt_sha256")
        _require_hex(adjoint_sha256, "--adjoint-sha256")
        if _digest_of(adjoint_path) != adjoint_sha256:
            raise QuantumIdentityRefused(
                f"adjoint receipt digest mismatch at {adjoint_path}")
        if adjoint["receipt_sha256"] != adjoint_sha256:
            raise QuantumIdentityRefused(
                "quantum record binds another adjoint receipt: "
                f"record={adjoint['receipt_sha256']!r} argv={adjoint_sha256}")
        receipt = load_adjoint_receipt(adjoint_path, adjoint_sha256)
        if receipt.get("status") != "complete":
            raise QuantumIdentityRefused(
                f"adjoint receipt status is {receipt.get('status')!r}, not complete")
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
                    "adjoint_receipt_sha256": adjoint_sha256,
                })
            except ValueError as exc:
                raise QuantumIdentityRefused(
                    f"producer campaign check refuses: {exc}") from exc
    except QuantumIdentityRefused:
        raise
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise QuantumIdentityRefused(
            f"quantum record is not a valid {QUANTUM_RECORD_SCHEMA} document: {exc}") from exc
    return record, receipt


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
        self.commits = 0

    def _enter(self, name: str) -> None:
        if self._phases is None:
            return
        try:
            index = self._phases.index(str(name))
        except ValueError:
            self._log(f"quantum progress: phase {name!r} was not declared; "
                      f"committing nothing under it")
            return
        if index > self._phase_index:
            self._phase_index = index
            self._phase = str(name)

    def enter_head(self, units: int) -> None:
        self._enter("head")
        self.priced(units)
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


class QuantumCounters:
    """Rob's two metrics, per chunk phase and per window (§8.1).

    ``bytes_from_ram``/``bytes_from_stage``/``bytes_from_pool`` are deltas of
    the existing residency-map accounting, snapshotted at phase and window
    boundaries; GPU energy is the 1 Hz power sampler plus profiler kernel
    time. Utilization percentages are never recorded: on GB10 they are
    non-diagnostic (AGENTS.md principle 13).
    """

    def __init__(self, *, quantum_id, identity_sha256, chunks, frontier: ChunkFrontier):
        from .residency_map import residency_report

        self._report = residency_report
        self._frontier = frontier
        self.quantum_id = str(quantum_id)
        self.identity_sha256 = str(identity_sha256)
        self.started = time.time()
        self.sampler = GpuPowerSampler().start()
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
        self.chain["kernel_active_s"] += float(kernel_active_s or 0.0)

    def kernel_block(self, profiler: KernelTimeProfiler) -> None:
        self.total_kernel_active_s += profiler.kernel_active_s
        if profiler.error and self._kernel_error is None:
            self._kernel_error = profiler.error

    def finish(self, *, units_done: int, units_total: int) -> dict:
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
            "units": [int(units_done), int(units_total)],
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
                    units_done / (joules / 3.6e6) if joules else None),
            },
            "chain": dict(self.chain),
            "phases": self.phases,
            "windows": self.windows,
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


def _install_with_settlement(runner, layer: int, *, operator_windows) -> None:
    """Install one source layer the way the single run's reverse walk does."""
    runner.context.install(
        layer,
        require_prefetched=runner.require_prefetched_residency,
        **({"prefetch_following": False} if operator_windows is not None else {}),
    )
    if operator_windows is None:
        runner.schedule_reverse_prefetch(layer)
    else:
        successors = range(max(0, layer - runner.prefetch_lookahead), layer)
        for successor in reversed(successors):
            runner.context.schedule_prefetch(successor)
        settle = getattr(runner.context, "settle_prefetched_layers", None)
        if callable(settle):
            settle(successors)
        elif torch.device(runner.device).type == "cuda":
            raise RuntimeError(
                "joint operator replay requires source prefetch settlement")


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


def _boundary_entry_record(receipt: dict, batch_index: int, boundary: int) -> dict:
    for entry in receipt["boundary_entries"][str(boundary)]:
        if entry["name"] == f"boundary-{batch_index}-{boundary}-at-{boundary}":
            return entry
    raise RuntimeError(
        f"adjoint receipt has no boundary entry for batch {batch_index} "
        f"boundary {boundary}")


def record_window_indices(record: dict) -> list[int]:
    """The record's sealed window-index slice, validated (D2 handshake).

    Returns ``[0..n-1]``; raises :class:`QuantumIdentityRefused` when the
    record does not seal the ordered index slice. Per-window names and byte
    sizes are never read from the record -- see :func:`resolve_quantum_windows`.
    """
    windows = record["windows"]
    indices = [window["window_index"] if isinstance(window, Mapping) else None
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
                              execution["boundary_storage"]).config),
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
# The layer quantum core (§6.2 steps 2-6)
# --------------------------------------------------------------------------


def run_layer_quantum_core(
    runner, production_cache, calib_ids, formats_by_qname, *,
    record, receipt, execution, output_root,
    projection_backend=None, resume=False,
    resolved_windows,
    counters: QuantumCounters, progress: QuantumProgress,
) -> dict:
    """Execute one layer quantum and return its payload (§6.4 ``cost.pkl``)."""
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
    )
    from .joint_adjoint_checkpoints import (
        load_adjoint_checkpoint,
        reference_from_record,
        render_free_layer_roll,
    )
    from .joint_statistics_replay import (
        check_operator_allocation,
        observe_and_project_retained_windows,
        operator_window_guard,
        statistics_arithmetic_identity,
    )
    from .kl_fisher import ROW_PROBE_LAYOUT
    from .perturbed_x_cache import _cb_cache_tensor_identity
    from .production_weight_cache import production_cache_cb_render_provenance
    from .routed_experts import refresh_packed_expert_projections
    from .sensitivity_probe import SharedStateCotangents, kv_cotangent_path_enabled

    layer = int(record["layer"])
    quantum_id = str(record["quantum_id"])
    checkpoint_dir = Path(record["output_space"]["checkpoint_dir"])
    n_probes = int(execution["n_probes"])
    seed_base = int(execution["seed_base"])
    token_scope = "all"
    temperature = 1.0
    probe_microbatch = int(execution.get("probe_microbatch", 0))
    min_free_gib = float(execution.get("min_free_gib", 0.0))

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
        "arithmetic": statistics_arithmetic_identity(runner.dtype, projection_backend),
    }
    joint_probe_identity["arithmetic"]["operator_windows"] = operator_windows
    joint_probe_identity["arithmetic"]["gradient_diagnostics"] = (
        "sum_output_operators_fp32_before_norm")
    if probe_layout is not None:
        joint_probe_identity["noise_layout"] = probe_layout
        joint_probe_identity["arithmetic"]["execution_partition"] = execution_partition

    prepared_render_identities = production_cache.metadata["verified_cells"]
    expected_pairs = {(name, fmt) for name in names for fmt in render_formats[name]}
    joint_cache_renders: dict[str, dict[str, dict]] = {}
    for name, fmt in sorted(expected_pairs):
        value = prepared_render_identities[(name, fmt)]["rendered_weight"]
        source = linears[name].weight
        if (value["shape"] != list(source.shape)
                or value["logical_bytes"] != source.numel() * source.element_size()):
            raise RuntimeError(
                f"prepared render tensor proof differs from the source for {name}@{fmt}")
        joint_cache_renders.setdefault(name, {})[fmt] = dict(value)
    joint_run_identity = {
        "schema": "prismaquant.joint_aura.run.v2",
        "probe_identity": joint_probe_identity,
        "cached_rendered_weights": joint_cache_renders,
        "activation_contracts": {
            name: {fmt: activation_identity(fr.get_format(fmt),
                                            production_cache.activation_max_abs or {}, name)
                   for fmt in unit_formats[name]}
            for name in names
        },
    }

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
            "adjoint_receipt_sha256": record["adjoint"]["receipt_sha256"],
            "checkpoint_boundary": int(record["adjoint"]["checkpoint_boundary"]),
            "chain_layers": [int(c) for c in record["adjoint"]["chain_layers"]],
            "windows": len(record["windows"]),
            "chunks": [chunk["name"] for chunk in record["chunks"]],
        },
    }

    # ---- boundary storage: read-attached to the adjoint capture ----------
    storage_policy = normalize_boundary_storage(execution["boundary_storage"])
    storage_policy["directory"] = str(boundary_entry_directory(adjusted_space(output_root)))
    storage = StreamedBoundaryArtifacts(storage_policy)
    storage.attach(receipt["boundary_storage"]["session"], n_probes=n_probes)
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
                if (row["probe_identity"] != joint_probe_identity
                        or operator["qname"] != name or operator["format"] != fmt):
                    raise ValueError("probe/operator alignment mismatch")
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
    checkpoint_record = next(
        (entry for entry in receipt["checkpoints"]
         if int(entry["boundary"]) == int(record["adjoint"]["checkpoint_boundary"])),
        None)
    if checkpoint_record is None:
        raise RuntimeError(
            "adjoint receipt does not carry the record's checkpoint boundary "
            f"{record['adjoint']['checkpoint_boundary']}")
    cotangent_plane, shared_adjoint, shared_pass = load_adjoint_checkpoint(
        adjusted_space(output_root), checkpoint_record)
    grad_plane: dict[tuple[int, int], torch.Tensor] = dict(cotangent_plane)
    cotangent_owners = [[SharedStateCotangents(enabled=kv_cotangent_path_enabled())
                         for _ in row_offsets] for _ in range(n_probes)]
    for (probe, batch), state in shared_adjoint.items():
        cotangent_owners[probe][batch].load_state_dict(state)

    with storage:
        partitions = [calib_ids[offset:offset + batch_rows]
                      for offset in row_offsets]
        batches = _rebuild_batches(runner, partitions=partitions,
                                   shared_pass=shared_pass)
        needed = sorted({int(c) for c in record["adjoint"]["chain_layers"]} | {layer})
        for batch_index, batch in enumerate(batches):
            batch.activations_cpu = [
                (_boundary_entry_record_to_reference(
                    _boundary_entry_record(receipt, batch_index, boundary))
                 if boundary in needed else None)
                for boundary in range(runner.num_layers + 1)]
        storage.watch_auxiliary(batches, cotangent_owners)
        storage.check_auxiliary(batches, cotangents=cotangent_owners)

        # ---- the render-free chain: stage A's arithmetic, reused ---------
        chain_started = time.time()
        chain_backwards = 0
        chain_kernel = KernelTimeProfiler()
        chain_kernel.__enter__()
        try:
            for chain_layer in (int(c) for c in record["adjoint"]["chain_layers"]):
                try:
                    _install_with_settlement(runner, chain_layer,
                                             operator_windows=operator_windows)
                    backwards = render_free_layer_roll(
                        runner, storage=storage, batches=batches, layer=chain_layer,
                        cotangents=cotangent_owners, n_probes=n_probes,
                        incoming_entries=None,
                        incoming_tensor=lambda probe, batch: grad_plane[(probe, batch)],
                        roll=lambda tensor, batch, probe: grad_plane.__setitem__(
                            (probe, batch), tensor),
                        min_free_gib=min_free_gib)
                    chain_backwards += backwards
                finally:
                    runner.context.unload(chain_layer)
        finally:
            chain_kernel.__exit__(None, None, None)
        counters.kernel_block(chain_kernel)
        counters.chain_step(layers=len(record["adjoint"]["chain_layers"]),
                            backwards=chain_backwards,
                            wall_s=time.time() - chain_started,
                            kernel_active_s=chain_kernel.kernel_active_s)

        # ---- layer L: the single run's retained reverse step --------------
        guard = operator_window_guard(
            runner.device,
            device_bytes=execution.get("device_envelope_bytes"))
        if guard is not None:
            check_operator_allocation(guard, "before_layer_quantum_replay")

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
                    observed_bytes=observed[
                        "conservative_cgroup_plus_cuda_reserved_bytes"],
                    source_bytes=source_bytes, actual_auxiliary_bytes=0,
                    label=f"before_quantum_{stage_label}:{layer}")
                check_operator_allocation(
                    guard, f"admit_quantum_{stage_label}:{layer}",
                    reserve_bytes=(retained_budget.boundary_reserve_bytes
                                   + retained_budget.workspace_reserve_bytes
                                   + retained_operator_windows[
                                       "source_loading_reserve_bytes"]))

        production_cache.enable_lru(retained_budget.retained_render_cap_bytes)
        if guard is not None:
            retained_budget.require_physical_guard(guard)
        retained_source_phase("source_loading")

        _install_with_settlement(runner, layer, operator_windows=operator_windows)
        if packed_members:
            from .production_weight_cache import PackedExpertProjection

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
                        f"installed source for {name}@{fmt}")

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
            scales = production_cache.activation_max_abs or {}
            activation = activation_identity(fr.get_format(fmt), scales, name)
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
                "probe_identity_sha256": identity_sha256(joint_probe_identity),
            }
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
                return
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
                    rows[fmt] = make_joint_aura_entry(
                        operator_identity=joint_operators[(name, fmt)],
                        probe_identity=joint_probe_identity,
                        signed_components=components,
                    )
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

        def replay_backward(*, final, lease, probe):
            active_probe = int(probe)
            with prefetched_boundary_batches(storage, batches, layer) as reverse_batches:
                for batch_index, batch, boundary_cpu, _unused in reverse_batches:
                    owner = cotangent_owners[active_probe][batch_index]
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
                        if guard is not None:
                            check_operator_allocation(
                                guard, "before_joint_window_backward", reserve_bytes=(
                                    operator_windows["workspace_reserve_bytes"]
                                    + (0 if lease is None
                                       else lease.statistics_capacity_bytes
                                       - lease.resident_statistics_bytes)))
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

        def before_window(window_index, window_names):
            nonlocal window_kernel, window_started
            del window_names
            window_kernel = KernelTimeProfiler()
            window_kernel.__enter__()
            window_started = time.time()
            counters.enter_phase()
            counters.open_window(window_index,
                                 resolved_windows[window_index])

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
            commit_streamed_units(window_names)
            progress.window_done(resolved_windows[window_index])
            counters.enter_phase()
            counters.mark_phase_units(len(completed_units))
            progress.commit()

        counters.open()
        try:
            observe_and_project_retained_windows(
                measured,
                {name: {fmt: fr.get_format(fmt) for fmt in render_formats[name]}
                 for name in measured},
                production_cache, operator_windows,
                retained_budget=retained_budget, n_probes=n_probes,
                source_bytes=retained_operator_windows["source_reserve_bytes"],
                backward=lambda *, probe_index, final, lease: replay_backward(
                    final=final, lease=lease, probe=probe_index),
                record_operator=_record_joint_operator,
                consume_probe=consume_window_probe,
                collect_col_energy=False, backend=projection_backend,
                guard=guard, source_fingerprints=source_seal,
                completed_names=set(measured) & completed_units,
                sealed_windows=sealed_windows,
                before_window=before_window,
                after_window=after_window,
            )
        finally:
            if window_kernel is not None:
                window_kernel.__exit__(None, None, None)
                window_kernel = None
            runner.context.unload(layer)

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
    payload["provenance"].update({
        "cost_mode": "aura", "joint_activation": True,
        "cost_currency": "joint_aura_predicted_dloss",
        "joint_aura_identity": joint_run_identity,
        "joint_aura_identity_sha256": identity_sha256(joint_run_identity),
        "probe_identity": joint_probe_identity,
        "probe_identity_sha256": identity_sha256(joint_probe_identity),
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
        "adjoint_receipt_sha256": record["adjoint"]["receipt_sha256"],
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
    })
    if set(joint_rows) != set(names):
        raise RuntimeError("layer quantum incomplete unit coverage")
    for name, rows in joint_rows.items():
        if set(rows) != set(unit_formats[name]):
            raise RuntimeError(
                f"layer quantum candidate scope mismatch for {name}")
        for fmt, row in rows.items():
            if not validate_joint_aura_entry(row):
                raise RuntimeError(f"invalid measured joint cost for {name}@{fmt}")
    return payload


# --------------------------------------------------------------------------
# Orchestration: head inputs, writes, exit codes (§6.2 steps 2 and 6)
# --------------------------------------------------------------------------


def _io_counters() -> dict:
    values = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value)
    return values


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
    config, *, record, receipt, plan_sha256, prepared, output_root,
    data_manifest_sha256=None, resume=False,
) -> dict:
    """Load the head phase and execute one quantum (§6.2 steps 2-6)."""
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
    )
    from .tessera_reader import load_declared_reader

    require_cuda_hot_path("joint_cost_quantum", "cuda")
    from .autoscale import require_bounded_capture_environment

    execution = config["execution"]
    if (config.get("qualification_window") is not None
            or execution.get("retained_operator_windows") is not None):
        require_bounded_capture_environment(os.environ)

    os.environ[ACTIVATION_SCALE_ENV] = execution["production_act_scales"]
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    layer = int(record["layer"])
    space = Path(record["output_space"]["root"])
    space.mkdir(parents=True, exist_ok=True)
    bind_residency_manifest(
        data_manifest_sha256 or record["campaign"].get("read_manifest_sha256"))

    result = {
        "schema": "prismaquant.joint_cost_quantum.execution.v1",
        "command": "layer-quantum",
        "quantum_id": record["quantum_id"],
        "identity_sha256": record["identity_sha256"],
        "plan_sha256": plan_sha256,
        "adjoint_receipt_sha256": record["adjoint"]["receipt_sha256"],
        "env": {"host": socket.gethostname(), "started_epoch": time.time(),
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "affinity": sorted(os.sched_getaffinity(0))},
        "dev_mode": dev_mode_stamp(),
        "phases": [], "passed": False,
    }
    result["env"]["container_content_sha256"] = executing_image()

    started, before_io = time.time(), _io_counters()
    runner = None
    payload = None
    counters = None
    resolved_windows: list[dict] | None = None
    try:
        reader = load_declared_reader(config.get("reader"))
        reader_identity = None if reader is None else reader.identity
        implementation = _aura_source_sha256()
        projection_backend = prewarm_projection_backend(
            execution.get("projection_backend"), device="cuda")
        result["projection_backend"] = projection_backend.identity
        _preflight_run_prepared(prepared, plan_sha256=plan_sha256,
                                implementation_sha256=implementation,
                                reader_identity=reader_identity,
                                projection_backend=projection_backend.identity)
        data = load_measured_anchor_input(
            config["inputs"], reader=reader, synthesis_device="cuda",
            progress_phase=HEAD_PHASE,
            head_checkpoint=space / "checkpoints" / "head-walk",
            head_resume=resume,
            require_existing_renders=True, verify_payloads=False)
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
        _same(completion.get("plan_sha256"), plan_sha256, "prepared plan")
        _same(completion.get("implementation_sha256"), implementation,
              "prepared implementation")
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
        expected_renders = {pair: cache.metadata["verified_cells"][pair]["render_file_sha256"]
                            for pair in data.cells}
        cache.require_file_load_sha256(
            expected_renders,
            max_file_bytes=_prepare_file_read_bound(
                data, max_render_bytes=config["max_render_bytes"]))
        result["wire_validation"] = "historical-qualified-wire"

        identity_cache_path = _seed_source_identity_cache(config, space / "run")
        runner = build_streamed_causal_lm(
            config["model"], device=torch.device("cuda"), dtype=torch.bfloat16,
            offload_folder=str(space / "run" / "offload"),
            profile=detect_profile(config["model"]), attn_implementation="eager",
            source_authentication=None)
        from .cost_streaming import build_streamed_model_identity

        source = build_streamed_model_identity(runner, config["model"],
                                               identity_cache_path=identity_cache_path)
        _same(completion.get("source_model_identity"), source,
              "prepared source identity")
        result.update(source_model_identity=source,
                      units=len(data.formats_by_qname),
                      measured_cells=len(data.cells))

        execution_runtime = dict(execution)
        execution_runtime.setdefault("device_envelope_bytes", config.get("max_gpu_bytes"))
        # D2 handshake, before any GPU work or progress: the record seals
        # window indices only, so membership and footprints are recomputed
        # from the sealed budget and handshook here. The chunk frontier,
        # counters and progress all size from the resolved windows -- never
        # from the record's index entries.
        retained = quantum_retained_state(execution_runtime)
        roster = quantum_layer_roster(runner, data.formats_by_qname, layer)
        resolved_windows = resolve_quantum_windows(
            record, layer=layer, names=roster.names, linears=roster.linears,
            render_formats=roster.render_formats, production_cache=cache,
            operator_windows=retained.operator_windows,
            retained_budget=retained.retained_budget,
            source_bytes=retained.source_bytes)
        result["resolved_windows"] = len(resolved_windows)
        counters = QuantumCounters(
            quantum_id=record["quantum_id"], identity_sha256=record["identity_sha256"],
            chunks=record["chunks"],
            frontier=ChunkFrontier(chunks=record["chunks"],
                                   windows=resolved_windows))
        # Head-phase currency continues from the head-committed base (§6.2
        # step 5): the same cumulative units the single run reports.
        progress = QuantumProgress(frontier=counters._frontier,
                                   base_units=data.progress_committed)
        payload = run_layer_quantum_core(
            runner, cache, ids.to(runner.device), data.formats_by_qname,
            record=record, receipt=receipt, execution=execution_runtime,
            output_root=output_root, projection_backend=projection_backend,
            resume=resume, resolved_windows=resolved_windows,
            counters=counters, progress=progress)
        torch.cuda.synchronize()
        result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if result["peak_gpu_bytes"] > config["max_gpu_bytes"]:
            raise RuntimeError("observed GPU allocation exceeds declared budget")
    finally:
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
    units_total = (sum(len(window["names"]) for window in resolved_windows)
                   if resolved_windows is not None else 0)
    counters_done = counters.finish(
        units_done=len(payload["costs"]) if payload else 0,
        units_total=units_total)
    status_record = publish_quantum_outputs(
        record, payload=payload, result=result, counters=counters_done,
        units_total=units_total)
    result["passed"] = status_record["status"] == "complete"
    return result


def main(argv=None) -> int:
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
    parser.add_argument("--adjoint", type=Path, required=True,
                        help="the stage-A adjoint-capture receipt")
    parser.add_argument("--adjoint-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-tool", default=None,
                        help="optional cprofile wrapping (default off; the "
                             "single run's profile applies to the campaign "
                             "of record, not every quantum)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-manifest-sha256")
    args = parser.parse_args(argv)
    if args.device != "cuda":
        parser.error("layer quanta are a GPU hot path; --device must be cuda")
    try:
        require_dev_mode("joint_cost_quantum")
        record, receipt = verify_quantum_identity(
            quantum_path=args.quantum, quantum_sha256=args.quantum_sha256,
            plan_path=args.plan, plan_sha256=args.plan_sha256,
            prepared_path=args.prepared, prepared_sha256=args.prepared_sha256,
            adjoint_path=args.adjoint, adjoint_sha256=args.adjoint_sha256,
            output_root=args.output_root)
    except QuantumIdentityRefused as exc:
        print(f"{IDENTITY_REFUSED_MARKER}: {exc}", flush=True)
        return EXIT_IDENTITY_REFUSED
    from .tessera_joint_aura import _load_plan

    config = _load_plan(args.plan, args.plan_sha256)
    if Path(config["output_root"]).resolve() != Path(args.output_root).resolve():
        print(f"{IDENTITY_REFUSED_MARKER}: plan output_root "
              f"{config['output_root']} is not --output-root {args.output_root}",
              flush=True)
        return EXIT_IDENTITY_REFUSED
    profiler = None
    if args.profile_tool == "cprofile":
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()
    try:
        result = run_layer_quantum(
            config, record=record, receipt=receipt, plan_sha256=args.plan_sha256,
            prepared={"path": str(args.prepared), "sha256": args.prepared_sha256},
            output_root=args.output_root,
            data_manifest_sha256=args.data_manifest_sha256, resume=args.resume)
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
