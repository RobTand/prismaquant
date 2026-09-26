"""Bind retained COST execution to a sealed PrismaBuild V2 read schedule.

The caller supplies the same descriptor it sealed as PB's ``data_manifest``
input and in the action specification. This module authenticates those bytes
and their application-level annotations; it cannot attest how PB populated a
worker environment. It neither reads declared data files nor warms them.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import stat
from typing import Callable, Mapping
import zlib

from .joint_retained_window_plan import RetainedWindowBudget
from .schemas import Contract, strict_json_loads


SCHEMA = "prismaquant.prismabuild.data_manifest.v2"
COMPLETED_SCHEMA = "prismaquant.joint_cost.validated_completed_units.v1"
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_DECODED_BYTES = 512 * 1024 * 1024
MAX_ENTRIES = 1_000_000
MAX_READS = 4_000_000
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_LAYER = re.compile(r"^.*\.layers\.(\d+)(?:\.|$)")
_ENTRY_KEYS = {"path", "offset", "bytes", "sha256"}
_ROOT_KEYS = {"schema", "produced_by", "mount_prefix", "entries", "entry_count",
              "total_bytes", "annotations", "read_plan"}
_ANNOTATION_KEYS = {"entry_point", "mode", "plan", "plan_sha256", "prepared",
                    "prepared_sha256", "source_owner_cap_bytes", "retained_budget",
                    "probes_per_window", "source_prefetch_lookahead_layers",
                    "tail_retained_source_layers", "window_partition_sha256", "windows",
                    "validated_completed_units_sha256", "validated_completed_units",
                    "sha256_present", "sha256_absent_reason", "argv"}
_WINDOW_KEYS = {"phase", "layer", "window_index", "original_full_target_names",
                "active_pending_names", "statistics_bytes",
                "render_file_upper_bound_bytes", "candidate_count"}


_CHECK = Contract(ValueError, "joint COST read schedule: ")
_require = _CHECK.require


def _int(value: object, label: str, *, positive: bool = False) -> int:
    _require(type(value) is int and value >= (1 if positive else 0),
             f"{label} must be an exact {'positive' if positive else 'nonnegative'} integer")
    return value


def _sha(value: object, label: str) -> str:
    _require(type(value) is str and _HEX.fullmatch(value) is not None,
             f"{label} must be a lowercase SHA-256 digest")
    return value


def _named_text(value: object, label: str) -> str:
    _require(type(value) is str and bool(value)
             and not any(ord(char) < 32 for char in value),
             f"{label} must be nonempty text without control characters")
    return value


def _object(value: object, keys: set[str], label: str) -> dict:
    _require(type(value) is dict and set(value) == keys, f"{label} fields differ")
    return value


def _decode_unique(raw: bytes) -> dict:
    try:
        return strict_json_loads(
            raw.decode("utf-8"),
            duplicate=lambda key: _CHECK.exception(f"duplicate JSON key {key!r}"),
            constant=lambda value: _CHECK.exception(f"non-finite JSON {value}"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("joint COST read schedule: invalid UTF-8 JSON") from exc


def _read_sealed(path: str | Path, sha256: str, size: int) -> dict:
    _sha(sha256, "manifest_sha256")
    _int(size, "manifest_bytes", positive=True)
    _require(size <= MAX_MANIFEST_BYTES, "manifest exceeds PB's 64 MiB input limit")
    path = os.fspath(path)
    _require(os.path.isabs(path), "manifest_path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        before = os.fstat(handle.fileno())
        _require(stat.S_ISREG(before.st_mode), "manifest descriptor is not a regular file")
        _require(before.st_size == size, "manifest descriptor byte count differs")
        raw = handle.read(size + 1)
        after = os.fstat(handle.fileno())
        _require(len(raw) == size and before.st_dev == after.st_dev
                 and before.st_ino == after.st_ino and before.st_size == after.st_size
                 and before.st_mtime_ns == after.st_mtime_ns,
                 "manifest bytes changed during read")
    _require(hashlib.sha256(raw).hexdigest() == sha256, "manifest SHA-256 differs")
    if raw.startswith(b"\x1f\x8b"):
        try:
            decoder = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
            decoded = decoder.decompress(raw, MAX_DECODED_BYTES + 1)
        except zlib.error as exc:
            raise ValueError("joint COST read schedule: invalid gzip data manifest") from exc
        _require(len(decoded) <= MAX_DECODED_BYTES and decoder.eof
                 and not decoder.unused_data and not decoder.unconsumed_tail,
                 "gzip data manifest needs exactly one bounded complete member")
        raw = decoded
    return _decode_unique(raw)


def _validate_pb_v2(manifest: object) -> tuple[dict, tuple[str, ...]]:
    m = _object(manifest, _ROOT_KEYS, "manifest")
    _require(m["schema"] == SCHEMA, "V2 schema required")
    _require(type(m["produced_by"]) is dict, "produced_by must be an object")
    _require(type(m["annotations"]) is dict and "phases" not in m["annotations"],
             "V2 annotations must be an object without phases")
    prefix = _named_text(m["mount_prefix"], "mount_prefix")
    _require(prefix.startswith("/") and prefix != "/"
             and posixpath.normpath(prefix) == prefix, "invalid mount_prefix")
    entries = m["entries"]
    _require(type(entries) is list and 0 < len(entries) <= MAX_ENTRIES,
             "invalid entry roster")
    seen, total, sizes = set(), 0, []
    for i, raw in enumerate(entries):
        row = _object(raw, _ENTRY_KEYS, f"entry[{i}]")
        path = _named_text(row["path"], f"entry[{i}] path")
        _require(path.startswith(prefix + "/")
                 and posixpath.normpath(path) == path,
                 f"entry[{i}] path outside normalized mount")
        offset = _int(row["offset"], f"entry[{i}] offset")
        size = _int(row["bytes"], f"entry[{i}] bytes", positive=True)
        _require(row["sha256"] is None or (type(row["sha256"]) is str
                 and _HEX.fullmatch(row["sha256"]) is not None),
                 f"entry[{i}] SHA-256 invalid")
        _require((path, offset) not in seen, f"entry[{i}] repeats path and offset")
        seen.add((path, offset))
        sizes.append(size)
        total += size
    _require(_int(m["entry_count"], "entry_count") == len(entries)
             and _int(m["total_bytes"], "total_bytes") == total,
             "unique entry accounting differs")
    plan = _object(m["read_plan"], {"phases", "read_bytes"}, "read_plan")
    phases = plan["phases"]
    _require(type(phases) is list and phases, "read_plan needs phases")
    used, names, cumulative, reads = set(), [], 0, 0
    for i, raw in enumerate(phases):
        row = _object(raw, {"name", "entry_indices", "bytes", "cumulative_bytes"},
                      f"phase[{i}]")
        name, indices = row["name"], row["entry_indices"]
        _named_text(name, f"phase[{i}] name")
        _require(name.strip() == name and name not in names,
                 f"phase[{i}] has an invalid or duplicate name")
        _require(type(indices) is list, f"phase[{i}] indices must be an array")
        reads += len(indices)
        _require(reads <= MAX_READS, "read references exceed PB limit")
        local, size = set(), 0
        for index in indices:
            _int(index, f"phase[{i}] entry index")
            _require(index < len(entries) and index not in local,
                     f"phase[{i}] has invalid or repeated entry index")
            local.add(index)
            used.add(index)
            size += sizes[index]
        cumulative += size
        _require(_int(row["bytes"], f"phase[{i}] bytes") == size
                 and _int(row["cumulative_bytes"], f"phase[{i}] cumulative bytes") == cumulative,
                 f"phase[{i}] byte accounting differs")
        names.append(name)
    _require(used == set(range(len(entries)))
             and _int(plan["read_bytes"], "read_bytes") == cumulative,
             "read-plan coverage or aggregate bytes differ")
    return m, tuple(names)


@dataclass(frozen=True)
class CostReadWindow:
    phase: str
    layer: int
    window_index: int
    original_full_target_names: tuple[str, ...]
    active_pending_names: tuple[str, ...]
    statistics_bytes: int
    render_file_upper_bound_bytes: int
    candidate_count: int


class JointCostReadSchedule:
    def __init__(self, phases: tuple[str, ...],
                 progress_callback: Callable[[str, int], object] | None,
                 runtime_binder: Callable[[Mapping[int, object], object],
                                          tuple[CostReadWindow, ...]],
                 identity: dict[str, object]):
        self.phases = phases
        self.windows: tuple[CostReadWindow, ...] = ()
        self._by_layer: dict[int, tuple[CostReadWindow, ...]] = {}
        self._progress_callback = progress_callback
        self._runtime_binder = runtime_binder
        self._identity = identity
        self._bound = False
        self._phase_index = -1
        self._committed_units = 0

    @property
    def current_phase(self) -> str | None:
        return None if self._phase_index < 0 else self.phases[self._phase_index]

    @property
    def identity(self) -> dict[str, object]:
        return dict(self._identity)

    def bind_runtime(self, target_names_by_layer: Mapping[int, object],
                     validated_completed_units: object) -> None:
        """Bind actual measured targets and independently validated checkpoints once."""
        _require(not self._bound, "runtime roster was already bound")
        _require(self._phase_index in (-1, 1),
                 "runtime binding is permitted before traversal or after cost_head")
        windows = self._runtime_binder(target_names_by_layer, validated_completed_units)
        self.windows = windows
        for layer in sorted({window.layer for window in windows}):
            self._by_layer[layer] = tuple(w for w in windows if w.layer == layer)
        self._bound = True

    def windows_for_layer(self, layer: int) -> tuple[CostReadWindow, ...]:
        _require(self._bound, "runtime target roster is not bound")
        _int(layer, "layer")
        return self._by_layer.get(layer, ())

    def enter_phase(self, name: str, committed_units: int) -> None:
        """Advance only to the next declared phase; report durable cumulative units."""
        count = _int(committed_units, "committed_units")
        next_index = self._phase_index + 1
        if self._phase_index >= 0 and name == self.phases[self._phase_index]:
            _require(count > self._committed_units,
                     "same-phase report requires a new durable committed unit")
        else:
            _require(next_index < len(self.phases) and name == self.phases[next_index],
                     "phase traversal skipped, repeated, or moved backward")
            _require(self._bound or next_index < 2,
                     "runtime roster must be bound before cost_capture")
            _require(count >= self._committed_units, "committed units moved backward")
            self._phase_index = next_index
        self._committed_units = count
        if self._progress_callback is not None:
            self._progress_callback(name, count)


def load_joint_cost_read_schedule(*, manifest_path: str | Path, manifest_sha256: str,
                                  manifest_bytes: int, plan_path: str | Path,
                                  plan_sha256: str, prepared_path: str | Path,
                                  prepared_sha256: str,
                                  retained_budget: RetainedWindowBudget,
                                  source_owner_cap_bytes: int, n_probes: int,
                                  target_names_by_layer: Mapping[int, object] | None = None,
                                  validated_completed_units: object | None = None,
                                  progress_callback: Callable[[str, int], object] | None = None
                                  ) -> JointCostReadSchedule:
    """Load the exact sealed PB V2 roster and bind it to runtime COST inputs."""
    _sha(plan_sha256, "plan_sha256")
    _sha(prepared_sha256, "prepared_sha256")
    _require(type(retained_budget) is RetainedWindowBudget, "retained budget object required")
    source_cap = _int(source_owner_cap_bytes, "source_owner_cap_bytes", positive=True)
    _require(_int(n_probes, "n_probes", positive=True) == 4, "COST V2 requires four probes")
    _require(progress_callback is None or callable(progress_callback), "progress callback invalid")
    m, phases = _validate_pb_v2(_read_sealed(manifest_path, manifest_sha256, manifest_bytes))
    a = _object(m["annotations"], _ANNOTATION_KEYS, "COST annotations")
    _require(a["entry_point"] == "prismaquant.tessera_joint_aura:run"
             and a["mode"] == "retained_cost_v2", "COST V2 annotation mode differs")
    _require(a["plan"] == os.path.abspath(plan_path) and a["plan_sha256"] == plan_sha256,
             "joint plan binding differs")
    _require(a["prepared"] == os.path.abspath(prepared_path)
             and a["prepared_sha256"] == prepared_sha256,
             "prepared completion binding differs")
    _require(a["retained_budget"] == retained_budget.as_dict()
             and a["source_owner_cap_bytes"] == source_cap
             and a["probes_per_window"] == n_probes,
             "retained budget, source cap, or probe count differs")
    _require(type(a["source_prefetch_lookahead_layers"]) is int
             and a["source_prefetch_lookahead_layers"] == 1,
             "source prefetch lookahead differs")
    _require(a["sha256_present"] is False and type(a["sha256_absent_reason"]) is str
             and bool(a["sha256_absent_reason"]), "data-byte hash annotation differs")
    _require(a["argv"] is None or (type(a["argv"]) is list
             and all(type(x) is str for x in a["argv"])), "argv annotation invalid")
    _sha(a["window_partition_sha256"], "window_partition_sha256")
    _require(phases[:2] == ("cost_setup", "cost_head"),
             "COST setup/head phase prefix differs")

    schedule = JointCostReadSchedule(phases, progress_callback,
        lambda names, completed: _bind_runtime(a, phases, retained_budget, source_cap,
                                                plan_sha256, prepared_sha256,
                                                names, completed),
        {"manifest_sha256": manifest_sha256, "manifest_bytes": manifest_bytes,
         "window_partition_sha256": a["window_partition_sha256"]})
    _require((target_names_by_layer is None) == (validated_completed_units is None),
             "runtime roster and completed names must be supplied together")
    if target_names_by_layer is not None:
        schedule.bind_runtime(target_names_by_layer, validated_completed_units)
    return schedule


def _bind_runtime(a: dict, phases: tuple[str, ...], retained_budget: RetainedWindowBudget,
                  source_cap: int, plan_sha256: str, prepared_sha256: str,
                  target_names_by_layer: Mapping[int, object],
                  validated_completed_units: object) -> tuple[CostReadWindow, ...]:
    _require(isinstance(target_names_by_layer, Mapping), "target roster mapping required")
    target_layers: dict[int, tuple[str, ...]] = {}
    for layer, names in target_names_by_layer.items():
        _int(layer, "target roster layer")
        _require(type(names) in (tuple, list) and names,
                 f"target roster layer {layer} must have names")
        row = tuple(names)
        _require(all(type(n) is str and n for n in row) and row == tuple(sorted(set(row))),
                 f"target roster layer {layer} must be sorted unique names")
        _require(all((match := _LAYER.match(n)) is not None
                     and int(match.group(1)) == layer for n in row),
                 f"target roster layer {layer} qnames differ")
        target_layers[layer] = row
    full_roster = set().union(*(set(v) for v in target_layers.values()))
    _require(bool(full_roster), "measured target roster is empty")
    _require(type(validated_completed_units) in (list, tuple, set, frozenset)
             and all(type(name) is str and name for name in validated_completed_units),
             "validated checkpoint names must be a concrete name collection")
    completed = set(validated_completed_units)
    _require(len(completed) == len(validated_completed_units) and completed <= full_roster,
             "validated checkpoint names are not a unique measured subset")
    binding = {
        "schema": COMPLETED_SCHEMA, "plan_sha256": plan_sha256,
        "prepared_sha256": prepared_sha256, "units": sorted(completed)}
    binding_sha = hashlib.sha256(json.dumps(
        binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _require(type(a["validated_completed_units"]) is int
             and a["validated_completed_units"] == len(completed)
             and a["validated_completed_units_sha256"] in ((None, binding_sha) if not completed
                                                            else (binding_sha,)),
             "validated completed checkpoint binding differs")

    raw_windows = a["windows"]
    _require(type(raw_windows) is list, "window annotations must be an array")
    windows, covered = [], set()
    by_layer_count: dict[int, int] = {}
    tail = a["tail_retained_source_layers"]
    _require(type(tail) is list and tail and all(type(n) is int and n >= 0 for n in tail),
             "retained source tail annotation invalid")
    last_source = tail[-1]
    _require(last_source >= max(target_layers)
             and tail == list(range(max(0, last_source - 1), last_source + 1)),
             "retained source tail annotation differs")
    available = retained_budget.available_window_bytes(source_cap)
    for i, raw in enumerate(raw_windows):
        row = _object(raw, _WINDOW_KEYS, f"window[{i}]")
        layer = _int(row["layer"], f"window[{i}] layer")
        index = _int(row["window_index"], f"window[{i}] index")
        _require(layer in target_layers and index == by_layer_count.get(layer, 0),
                 f"window[{i}] has wrong layer or nonsequential index")
        by_layer_count[layer] = index + 1
        phase = f"cost_reverse_{layer:03d}_window_{index:03d}"
        _require(row["phase"] == phase, f"window[{i}] phase differs")
        full = row["original_full_target_names"]
        active = row["active_pending_names"]
        _require(type(full) is list and full and full == sorted(set(full))
                 and set(full) <= set(target_layers[layer]) and not set(full) & covered,
                 f"window[{i}] full names are invalid or repeated")
        _require(type(active) is list and active == [n for n in full if n not in completed],
                 f"window[{i}] pending names differ from validated checkpoints")
        covered.update(full)
        stats = _int(row["statistics_bytes"], f"window[{i}] statistics bytes", positive=True)
        renders = _int(row["render_file_upper_bound_bytes"],
                       f"window[{i}] render file upper bound", positive=True)
        candidates = _int(row["candidate_count"], f"window[{i}] candidate count", positive=True)
        _require(stats <= retained_budget.statistics_cap_bytes
                 and renders <= retained_budget.retained_render_cap_bytes
                 and stats + renders <= available
                 and index < retained_budget.max_windows_per_layer,
                 f"window[{i}] exceeds retained budget")
        windows.append(CostReadWindow(phase, layer, index, tuple(full), tuple(active),
                                      stats, renders, candidates))
    _require(covered == full_roster, "windows do not cover every measured target exactly")
    _require([w.layer for w in windows] == sorted([w.layer for w in windows], reverse=True),
             "window annotations are not reverse-layer ordered")
    partition = [[w.layer, w.window_index, list(w.original_full_target_names)] for w in windows]
    digest = hashlib.sha256(json.dumps(partition, separators=(",", ":"),
                                       ensure_ascii=False).encode()).hexdigest()
    _require(a["window_partition_sha256"] == digest, "window partition SHA-256 differs")
    expected_phases = ["cost_setup", "cost_head"]
    expected_phases += [f"cost_capture_{layer:03d}" for layer in range(last_source + 1)]
    expected_phases.append("cost_tail")
    for layer in range(last_source, -1, -1):
        expected_phases.append(f"cost_reverse_{layer:03d}_source")
        expected_phases.extend(w.phase for w in windows if w.layer == layer)
    _require(phases == tuple(expected_phases), "read-plan phase roster/order differs from COST")
    return tuple(windows)
