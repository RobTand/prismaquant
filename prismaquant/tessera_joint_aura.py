"""Research joint AURA over exact, completed Tessera campaign anchors.

The campaign's scalar MSE/interpolation is evidence of which wires were made,
never a joint price. Original decoded renders enter ProductionWeightCache;
Tessera's existing source/H/settings receipt and decoder qualify them before
ordinary streamed joint AURA consumes the exact per-Linear candidate roster.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import functools
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import os
import socket
import stat
import threading
import time
from types import SimpleNamespace

from .cost_stage_checkpoint import (
    MANIFEST_SCHEMA, _load_unit, atomic_write_bytes, canonical_json_sha256,
    canonical_json_sha256_normalized,
    prepare_journal, unit_path, write_unit,
    _drive_ordered_units as _drive_ordered_walk,
)
from .dev_mode import dev_mode_enabled, dev_stamp, dev_warning, seal_check
from .interned_json import load_json_file
from .joint_head_walk_quanta import check_quantum_for_roster
from .residency_map import (
    bind_residency_manifest, residency_report, residency_resolver,
)
from .schemas import Contract

SCHEMA = "prismaquant.tessera_joint_aura.plan.v1"
PREPARED_SCHEMA = "prismaquant.tessera_joint_aura.prepared.v3"
HISTORICAL_WIRE_VALIDATION = "historical_prepared_identity; current_bytes_require_export_gate"
RENDER_ORIGIN_SCHEMA = "prismaquant.tessera_joint_aura.render_origin.v1"
# Closed vocabularies. ``render_origin`` says where the decoded PWC shard on
# disk came from; ``render_comparison`` says what the ``torch.equal`` leg of
# ``verify_anchor_render`` established for that rung. They are two different
# facts and a record that collapses them claims verification it never had.
RENDER_ORIGINS = ("encoded", "synthesized_from_wire")
RENDER_COMPARISONS = ("independent_render_vs_wire", "wire_round_trip_only")
# An encoded render is an independently produced tensor, so comparing it with
# the decoded wire is evidence about the encode. A synthesized render was
# written by decoding that same wire, so the comparison can only establish
# that the ``.pt`` still round-trips to the bytes it was written from.
RENDER_COMPARISON_BY_ORIGIN = {"encoded": "independent_render_vs_wire",
                               "synthesized_from_wire": "wire_round_trip_only"}
# The producer's encoder source seal hashes every python/cuda/cpp file of the
# installed Tessera package, the unmeasured branches included
# (``tessera.cached_unit.encoder_source_sha256``). A campaign priced under an
# older package therefore cannot pass a newer package's own seal, even when
# every byte the encoder actually read -- source weight, H, calibration,
# recipe, unit, fixture and wire blob -- is identical. A plan may name such a
# historical digest explicitly, one 64-hex value at a time; that is an
# admission that those wire bytes are reused without re-deriving the encoder's
# own seal, and every receipt carries the admission. It is not a licence for
# any other identity field to differ, and there is no wildcard spelling.
HISTORICAL_ENCODER_REUSE_SCHEMA = "prismaquant.tessera_joint_aura.historical_encoder_reuse.v1"
ENCODER_REUSE_STATUS = "unverified_encoder_reuse"
HISTORICAL_ENCODER_REUSE_ENTRY_FIELDS = frozenset(
    {"encoder_source_sha256", "reason", "evidence", "recorded_unix", "recorded_by"})
CAMPAIGN_SCHEMA = "prismaquant.tessera_campaign_cost.v1"
CURRENCY = "output_mse_under_route_activation_contract"
STAGE = "Tessera campaign"
# The head walk's own durable journal (#754). Every restart re-paid the whole
# roster walk -- the withdrawn campaign night paid it three times -- so the
# walk banks each verified unit under the checkpoint machinery every other
# stage of this campaign already uses (``prepare_journal``/``write_unit``),
# and a resume re-verifies the banked prefix against the bytes it was banked
# from before trusting one row of it.
HEAD_WALK_STAGE = "joint head walk"
HEAD_WALK_STATE_SCHEMA = "prismaquant.tessera_joint_aura.head_walk_unit.v1"
HEAD_WALK_JOURNAL_SCHEMA = "prismaquant.tessera_joint_aura.head_walk.v1"
#: The campaign inputs the head walk reads, each bound by digest in a plan's
#: ``inputs`` block (``load_measured_anchor_input``).
HEAD_WALK_INPUT_KEYS = ("campaign_plan", "census", "campaign_receipts", "merged_cost",
                        "merged_checkpoint")


def merged_checkpoint_parts(path) -> Path:
    """The directory of the merged anchor checkpoint's per-unit part files."""
    path = Path(path)
    return path.with_name(path.name + ".parts")


def head_walk_read_set(inputs) -> tuple[frozenset[str], tuple[str, ...]]:
    """``(files, directories)`` the head walk reads, from a plan's ``inputs``.

    The files are the bound inputs; the directory is the merged checkpoint's
    parts, whose per-unit files the walk loads. Stage A takes its head from
    the prepared completion instead of walking (PQ #1051), so a Stage A data
    manifest declares none of these (``stage_a_head.drop_head_walk_reads``).
    A key the plan does not bind names no read: the walk itself refuses such
    a plan, and a catalog extension's plan binds other inputs.
    """
    bound = {key: inputs[key] for key in HEAD_WALK_INPUT_KEYS if key in inputs}
    files = frozenset(os.path.normpath(str(item["path"])) for item in bound.values())
    parts = (() if "merged_checkpoint" not in bound else (os.path.normpath(
        str(merged_checkpoint_parts(bound["merged_checkpoint"]["path"]))),))
    return files, parts


def is_head_walk_read(path, read_set) -> bool:
    """Whether ``path`` is one of the head walk's reads (``head_walk_read_set``)."""
    files, directories = read_set
    path = os.path.normpath(str(path))
    return path in files or any(path.startswith(directory + os.sep)
                                for directory in directories)


# The walk's worker pool is the CPU set PrismaBuild assigned this container
# (``os.sched_getaffinity``: PB applies its allocation with taskset before
# exec and the container inherits it) -- never a guessed core. The cap keeps
# a whole-box reservation from minting a thread per unit; the env knob
# lowers it or forces the serial path for an A/B measurement.
HEAD_WALK_WORKERS_ENV = "PRISMAQUANT_HEAD_WALK_WORKERS"
HEAD_WALK_MAX_WORKERS = 16
# Units are banked on a time cadence rather than one fsynced envelope per
# unit: a crash loses at most one interval of verified work, and the cadence
# matches the progress contract's own clock (#741).
HEAD_WALK_BANK_INTERVAL_S = 60.0
#: Freed-but-cached decode blocks a cuda head walk tolerates before the
#: retained pool is returned to the driver. Synthesis keeps no decoded
#: tensor by reference (the shard is written to the render file), but the
#: caching allocator retains every freed block for reuse and per-cell shapes
#: vary enough that the pool only grows across a long walk; on shared-system
#: GPUs that retained pool is charged to the action's GPU budget and a long
#: walk is reaped for memory it no longer uses (#823). The gap keeps a small
#: steady walk from ever paying the reclaim.
HEAD_WALK_RECLAIM_GAP_BYTES = 2 * 1024 ** 3
# Synthesis decodes through the bound Tessera reader on the reserved device,
# and that reader is proven single-threaded (the qualification walk's own
# wire pool is max_workers=1), so the rare missing render is synthesized
# under one lock while the common pure-read verification overlaps freely.
_HEAD_WALK_SYNTHESIS_LOCK = threading.Lock()


_require = Contract(ValueError).require


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _stat_signature(value):
    """The identity fence for a wire byte read, including its file type."""
    return (value.st_mode, value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _read_verified_wire_blob(cell):
    """Read one receipt-sized regular wire and bind the bytes actually read.

    This is deliberately an ephemeral one-cell buffer, rather than a cache.
    The caller may overlap its read with qualification of the preceding cell,
    but must pass these exact bytes to ``verify_cached_unit`` and the decoder.
    """
    record, wire = cell["record"], Path(cell["wire"])
    size = record.get("blob_bytes")
    _require(type(size) is int and size > 0,
             f"{wire}: wire receipt needs positive blob_bytes")
    expected = record.get("blob_sha256")
    # PrismaBuild's stage tier, when it holds this wire. The receipt already
    # names the digest, so the staged copy is admitted only if its bytes hash
    # to the same value the pool copy would have to; a refused entry reads the
    # declared path and is recorded. Unset environment, no stage, no change.
    resolver = residency_resolver()
    staged = (None if resolver is None
              else resolver.staged_read(wire, expected_sha256=expected))
    from .staged_tier_policy import policy_is_active, refuse_pool_bulk_read
    strict = policy_is_active()
    if staged is not None and not strict:
        # The ram copy first, the stage copy second, the declared path last.
        # PrismaBuild's ram tier (#640) promotes a staged range onto a tmpfs
        # and the resolver offers the copy only while the epoch the map dates
        # it with is the one the pool's tier record announces, so a dead tmpfs
        # never reaches this loop. A ram read that refuses -- released between
        # the stat and the open, or bytes that do not hash to the receipt --
        # is a miss on the ram half alone: the stage copy the map vouches for
        # serves next, and only its refusal reads the declared path.
        copies = ([("ram", staged["ram_path"])] if "ram_path" in staged else []) \
            + [("stage", staged["stage_path"])]
        for half, copy in copies:
            try:
                blob, digest = _read_wire_bytes(Path(copy), size,
                                                expected=expected, staged=True)
            except _StagedWireRefused as refusal:
                if half == "ram":
                    resolver.record_ram_fallback(wire, str(refusal))
                    continue
                resolver.record_fallback(wire, str(refusal))
            else:
                # The open fence passed: serving tier recorded at open,
                # before these payload bytes are trusted.
                resolver.record_serving_tier(wire, half)
                if half == "ram":
                    resolver.record_ram_read(wire, len(blob))
                else:
                    resolver.record_stage_read(wire, len(blob))
                return blob, digest
    elif staged is not None:
        # Lifetime-pinned window (one per blob); see the shard reader for
        # the RAM/SSD discipline shared here.
        from .staged_lease import LeaseRefused, acquire_entry_window
        window, key = acquire_entry_window(resolver, wire, staged)
        with window:
            try:
                fd, serving = window.open(key)
            except LeaseRefused as refusal:
                resolver.record_fallback(wire, str(refusal))
                raise
            tier = window.serving_tier or "stage"
            resolver.record_serving_tier(
                wire, tier, pin_id=str(serving.get("pin_id") or ""),
                range_ref=str(serving.get("range_ref") or ""))
            first = os.fstat(fd)
            parts = []
            remaining = size + 1
            offset = 0
            while remaining > 0:
                block = os.pread(fd, min(remaining, 8 << 20), offset)
                if not block:
                    break
                parts.append(block)
                offset += len(block)
                remaining -= len(block)
            blob = b"".join(parts)
            last = os.fstat(fd)
            if (len(blob) != size or first.st_size != size
                    or (last.st_ino, last.st_size, last.st_mtime_ns)
                    != (first.st_ino, first.st_size, first.st_mtime_ns)):
                # Changed under its pin: integrity fails clear, no alternate.
                raise LeaseRefused("lease-open-size-changed", kind="integrity")
        # The window closed (descriptor shut, exact ref released) before
        # these bytes are bound to the receipt.
        digest = hashlib.sha256(blob).hexdigest()
        if digest != expected:
            raise refuse_pool_bulk_read(
                str(wire), "content-corruption:staged wire bytes differ "
                           "from the receipt digest")
        if tier == "ram":
            resolver.record_ram_read(wire, len(blob))
        else:
            resolver.record_stage_read(wire, len(blob))
        return blob, digest
    elif strict:
        raise refuse_pool_bulk_read(
            str(wire), "readset-not-staged" if resolver is None
            else "staged-not-serving")
    blob, digest = _read_wire_bytes(wire, size, expected=expected, staged=False)
    if resolver is not None:
        resolver.record_pool_read(wire, len(blob))
    return blob, digest


class _StagedWireRefused(Exception):
    """The staged wire failed its identity check; read the declared path."""


class _StagedWireCorrupt(_StagedWireRefused):
    """The staged wire's content compares corrupt (digest mismatch).

    Availability failures (unreadable, resized, raced) may fall through to
    the next permitted copy; content corruption fails clear under policy —
    never a silent adoption of an unchecked alternate.
    """


def _read_wire_bytes(wire, size, *, expected, staged):
    """One fenced read of ``size`` bytes, digested and bound to the receipt.

    A staged read refuses instead of failing: PrismaBuild recomposes the map
    after every egress, so a staged copy can be released between the
    resolver's stat and this open, and every fence below is then a reason to
    read the declared wire rather than a reason to stop. The declared wire's
    own refusals are unchanged.
    """
    try:
        before = wire.lstat()
        if staged and not stat.S_ISREG(before.st_mode):
            raise _StagedWireRefused('staged wire is not a regular file')
        _require(stat.S_ISREG(before.st_mode),
                 f"{wire}: wire must be a regular file, not a symlink")
        if staged and before.st_size != size:
            raise _StagedWireRefused('staged wire size differs from the measured receipt')
        _same(before.st_size, size, f"{wire}: wire size differs from measured receipt")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(wire, flags)
        try:
            with os.fdopen(fd, "rb", closefd=True) as handle:
                fd = None
                if staged and _stat_signature(os.fstat(handle.fileno())) != _stat_signature(before):
                    raise _StagedWireRefused('staged wire changed before its content read')
                _same(_stat_signature(os.fstat(handle.fileno())), _stat_signature(before),
                      f"{wire}: wire changed before its content read")
                blob = handle.read(size + 1)
                after_open = os.fstat(handle.fileno())
        finally:
            if fd is not None:
                os.close(fd)
    except OSError as error:
        if not staged:
            raise
        raise _StagedWireRefused(
            f'staged wire is unreadable: {error.strerror}') from None
    if staged and (len(blob) != size
                   or _stat_signature(after_open) != _stat_signature(before)
                   or _stat_signature(wire.lstat()) != _stat_signature(before)):
        raise _StagedWireRefused('staged wire changed during its content read')
    _require(len(blob) == size, f"{wire}: wire changed during its content read")
    _same(_stat_signature(after_open), _stat_signature(before),
          f"{wire}: wire changed during its content read")
    _same(_stat_signature(wire.lstat()), _stat_signature(before),
          f"{wire}: wire changed during its content read")
    digest = hashlib.sha256(blob).hexdigest()
    if staged and digest != expected:
        raise _StagedWireCorrupt('staged wire bytes differ from the receipt digest')
    _same(digest, expected, f"{wire}: wire checksum")
    return blob, digest


def _bound(record, label):
    _require(isinstance(record, dict) and set(record) == {"path", "sha256"},
             f"{label}: independently bound path/SHA256 required")
    path = Path(record["path"])
    _require(_sha(path) == record["sha256"], f"{label}: artifact checksum changed")
    return path


def _same(actual, expected, label):
    _require(actual == expected, f"{label}: identity mismatch")


def _json(path, value):
    atomic_write_bytes(Path(path), (json.dumps(value, indent=2, sort_keys=True,
                                              allow_nan=False) + "\n").encode())


def _require_sha256(value, where):
    _require(isinstance(value, str) and len(value) == 64 and
             all(character in "0123456789abcdef" for character in value),
             f"{where}: 64-hex digest required")
    return value


def normalize_historical_encoder_reuse(value, *, where="historical encoder reuse"):
    """Validate a plan's explicit allowance for a historical encoder seal.

    Closed by construction: a versioned schema and a non-empty list of named
    64-hex digests, each with its own reason and provenance. An empty list, a
    prefix, a glob or a missing block all mean the same thing -- no reuse -- so
    there is no spelling that admits an unnamed hash.
    """
    if value is None:
        return None
    _require(isinstance(value, dict) and set(value) == {"schema", "allowlist"},
             f"{where}: requires exactly schema and allowlist")
    _same(value["schema"], HISTORICAL_ENCODER_REUSE_SCHEMA, f"{where} schema")
    entries = value["allowlist"]
    _require(isinstance(entries, list) and bool(entries),
             f"{where}: a non-empty explicit allowlist is required")
    seen, normalized = set(), []
    for index, entry in enumerate(entries):
        label = f"{where} allowlist[{index}]"
        _require(isinstance(entry, dict) and set(entry) == HISTORICAL_ENCODER_REUSE_ENTRY_FIELDS,
                 f"{label}: requires exactly {sorted(HISTORICAL_ENCODER_REUSE_ENTRY_FIELDS)}")
        digest = _require_sha256(entry["encoder_source_sha256"],
                                 f"{label} encoder_source_sha256")
        _require(digest not in seen, f"{label}: duplicate encoder_source_sha256 {digest}")
        seen.add(digest)
        for key in ("reason", "evidence", "recorded_by"):
            _require(isinstance(entry[key], str) and bool(entry[key].strip()),
                     f"{label}: non-empty {key} required")
        recorded_unix = entry["recorded_unix"]
        _require(type(recorded_unix) in (int, float) and math.isfinite(recorded_unix)
                 and recorded_unix >= 0,
                 f"{label}: finite non-negative recorded_unix required")
        normalized.append(dict(entry))
    return {"schema": HISTORICAL_ENCODER_REUSE_SCHEMA, "allowlist": normalized}


def _encoder_reuse_entry(policy, digest):
    for entry in (policy or {}).get("allowlist", ()):
        if entry["encoder_source_sha256"] == digest:
            return entry
    return None


def resolve_encoder_source_reuse(recorded, current, policy, *, where):
    """Decide whether one recorded encoder seal may be reused, and say so.

    The installed package's own seal is the only one a run can re-derive, so
    equality needs no permission and is reported as no reuse at all. Any other
    recorded digest is refused unless the plan named that exact value. A named
    value is carried forward with both observed digests, so no downstream
    report can read the result as a re-derived encoder seal.
    """
    _require_sha256(recorded, f"{where} recorded encoder source")
    _require_sha256(current, f"{where} current encoder source")
    if recorded == current:
        return None
    entry = _encoder_reuse_entry(policy, recorded)
    _require(entry is not None,
             f"{where}: recorded encoder_source_sha256 {recorded} is not the installed "
             f"package's {current} and the plan does not name it in "
             "historical_encoder_reuse; refusing unverified encoder reuse")
    return {"schema": HISTORICAL_ENCODER_REUSE_SCHEMA, "status": ENCODER_REUSE_STATUS,
            "recorded_encoder_source_sha256": recorded,
            "observed_current_encoder_source_sha256": current,
            "allowlist_entry": dict(entry)}


def require_encoder_source_reuse_record(value, *, where):
    """Validate one carried reuse record before it authorizes an identity swap."""
    if value is None:
        return None
    fields = {"schema", "status", "recorded_encoder_source_sha256",
              "observed_current_encoder_source_sha256", "allowlist_entry"}
    _require(isinstance(value, dict) and set(value) == fields,
             f"{where}: malformed encoder reuse record")
    _same(value["schema"], HISTORICAL_ENCODER_REUSE_SCHEMA, f"{where} reuse schema")
    _same(value["status"], ENCODER_REUSE_STATUS, f"{where} reuse status")
    _require_sha256(value["recorded_encoder_source_sha256"], f"{where} recorded encoder source")
    _require_sha256(value["observed_current_encoder_source_sha256"],
                    f"{where} current encoder source")
    _require(value["recorded_encoder_source_sha256"] !=
             value["observed_current_encoder_source_sha256"],
             f"{where}: a reuse record must name two different seals")
    # The carried entry is re-validated by the plan's own normalizer, so a
    # hand-built record cannot smuggle a field shape the plan grammar refuses.
    policy = normalize_historical_encoder_reuse(
        {"schema": HISTORICAL_ENCODER_REUSE_SCHEMA, "allowlist": [value["allowlist_entry"]]},
        where=f"{where} carried allowlist")
    _same(policy["allowlist"][0], value["allowlist_entry"],
          f"{where}: carried allowlist entry is not its normalized form")
    _same(value["allowlist_entry"]["encoder_source_sha256"],
          value["recorded_encoder_source_sha256"], f"{where}: allowlist entry names another seal")
    return dict(value)


@dataclass
class MeasuredAnchorInput:
    inputs: dict
    payload: dict
    manifest: dict
    census: dict
    campaign_plan: dict
    cells: dict
    formats_by_qname: dict
    # A read restricted to ``sorted(names)[lo:hi]``, or None for the whole
    # roster. Only the standalone synthesis stage produces a scoped read, and
    # ``execute`` refuses one: a partial roster is not the campaign's input.
    unit_scope: "tuple | None" = None
    # Where the synthesized shards were written when they were NOT written to
    # the campaign's own row caches (the measure mirror), or None.
    render_mirror_root: "str | None" = None
    # How many shards THIS read synthesized. Distinct from the per-origin
    # census, which counts what is on disk however it got there.
    synthesized_now: int = 0
    # The cumulative count this read last reported to PrismaBuild, and zero
    # when it reported nothing. The later stages continue from it rather than
    # restarting, because a counter that goes backwards renews no allowance
    # (RobTand/prismaquant#678).
    progress_committed: int = 0
    # ``None`` when the checkpoint's recorded encoder source seal is the
    # installed package's own. Otherwise the explicit record that this read
    # verifies wire identities against a historical seal the plan named, with
    # both observed digests. ``verify_anchor_render`` refuses to swap the seal
    # without it, so no cell can be qualified under a seal nothing observed.
    encoder_source_reuse: "dict | None" = None
    # The worker count this walk fanned out over (the PB-assigned affinity by
    # default) and how many units a resume re-verified from the head journal
    # instead of re-walking (#754). Zero when no journal was given.
    head_walk_workers: "int | None" = None
    head_walk_resumed_units: int = 0

    @property
    def total_render_bytes(self):
        return sum(2 * math.prod(self.census["unit_shapes"][name]) for name, _ in self.cells)

    def layer_render_bytes(self, layer_for_name):
        sizes = defaultdict(int)
        for name, _fmt in self.cells:
            sizes[layer_for_name(name)] += 2 * math.prod(self.census["unit_shapes"][name])
        return dict(sizes)


def _render_origin_marker_path(render):
    return Path(render).with_name(Path(render).name + ".render_origin.json")


def render_origin_census(origins):
    """Count each closed-vocabulary value, including the ones nobody used.

    A census that omits the zero keeps a reader from telling "no synthesized
    renders" apart from "this report does not say".
    """
    counts = {value: 0 for value in RENDER_ORIGINS}
    comparisons = {value: 0 for value in RENDER_COMPARISONS}
    for origin in origins:
        _require(origin in RENDER_ORIGINS, f"unknown render origin {origin!r}")
        counts[origin] += 1
        comparisons[RENDER_COMPARISON_BY_ORIGIN[origin]] += 1
    return {"render_origins": counts, "render_comparisons": comparisons}


def cell_render_census(cells):
    """The census of a cell mapping, from the field every cell must carry."""
    origins = []
    for pair, cell in sorted(cells.items()):
        _require(isinstance(cell, dict) and "render_origin" in cell,
                 f"{pair}: cell carries no render_origin")
        origins.append(cell["render_origin"])
    return render_origin_census(origins)


def _decode_wire(blob, *, reader, device="cpu"):
    """The one decode seam: the bound reader's, or the module-level decoder."""
    if reader is not None:
        return reader.read_unit_artifact(blob, device=device)
    from tessera.unit_artifact import read_unit_artifact

    return read_unit_artifact(blob, device=device)


#: The phase the standalone synthesis stage declares (``--progress
#: synthesize=<stall>``), and therefore the default a bare intake reports
#: under. ``execute`` overrides it with the joint prepare's own ``head``:
#: this loader spells no phase its caller has not declared.
SYNTHESIS_PHASE = "synthesize"


_DEV_SOURCE_SHA256_MEMO: str | None = None


def _progress_dev_source_sha256():
    """The executing package's actual tree digest, for the dev stamps.

    Lazy so importing this module never pulls ``aura_cost``; only a dev-mode
    progress commit that opted into the stamp pays for the hash -- and it pays
    it **once**: the digest is memoized after the first commit because a
    progress line fires per durable unit and a walk commits tens of thousands
    of them. Measured live
    on stage A (2026-09-20, action 398c81b4): the un-memoized form re-walked
    and re-hashed the whole package tree on every unit, holding the head
    walk to ~0.3 units/s of pure pathlib with zero IO -- the dev stamp is an
    identity, and the executing tree's identity does not change mid-run.
    """
    global _DEV_SOURCE_SHA256_MEMO
    if _DEV_SOURCE_SHA256_MEMO is None:
        from .aura_cost import _aura_source_sha256
        _DEV_SOURCE_SHA256_MEMO = _aura_source_sha256()
    return _DEV_SOURCE_SHA256_MEMO


def _pb_commit(units, phase, unit=None):
    """Report cumulative durable units to PrismaBuild; a no-op elsewhere.

    Held byte for byte against the published submission skill's snippet
    (``skills/prismabuild/SKILL.md``, ``pb-progress-snippet``) so an action
    inside a container that cannot import PrismaBuild still reports. It is a
    no-op when the action was not admitted under the progress contract, so it
    is called unconditionally rather than by testing how we were launched.

    Under ``PRISMAQUANT_DEV_MODE=1`` the record may carry the dev stamp in
    its metadata -- **opt-in** via ``PRISMAQUANT_DEV_PROGRESS_STAMP=1``.
    The stamp is provenance ceremony; Rob's standing campaign directive
    (2026-09-13) is that dev-mode campaign runs incur no sealing overhead,
    and the per-line stamp measurably did: before the memo it re-hashed the
    whole executing tree on every durable unit (2026-09-20, #826). The
    run's identity is already recorded where it belongs -- once, in the
    results record's top-level dev stamp and the startup implementation
    line -- so the default progress record stays byte-identical to the
    certified shape. The worker's ``ProgressWatch`` reads the fields it
    knows and ignores the rest either way.
    """
    path = os.environ.get("PRISMABUILD_ACTION_PROGRESS_PATH")
    token = os.environ.get("PRISMABUILD_ACTION_PROGRESS_TOKEN")
    if not path or not token:
        return False
    record = {"schema": "prismabuild.action_progress.v1", "token": token,
              "phase": phase, "units_completed": units, "unit": unit,
              "reported_unix": time.time()}
    if (dev_mode_enabled()
            and os.environ.get("PRISMAQUANT_DEV_PROGRESS_STAMP") == "1"):
        record.update(dev_stamp(_progress_dev_source_sha256()))
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return True


def _head_walk_worker_count(requested=None, environ=None):
    """Resolve the head walk's worker count from PrismaBuild's own placement.

    The default is the CPU set PB assigned this container, capped so a
    whole-box reservation cannot mint a thread per unit. An explicit request
    wins, the env knob is the operator's A/B lever, and neither may exceed
    the assignment: cores PB did not assign are never guessed
    (``file_hash_workers`` refuses the same way one screen up).
    """
    environ = os.environ if environ is None else environ
    try:
        assigned = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        assigned = 1
    ceiling = max(1, assigned)
    if requested is not None:
        _require(type(requested) is int and 0 < requested <= ceiling,
                 f"head walk workers {requested} exceed the PB-assigned CPU affinity ({ceiling})")
        return requested
    raw = environ.get(HEAD_WALK_WORKERS_ENV)
    if raw is None or raw == "":
        return min(ceiling, HEAD_WALK_MAX_WORKERS)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{HEAD_WALK_WORKERS_ENV}={raw!r} is not a worker count") from exc
    _require(0 < value <= ceiling,
             f"{HEAD_WALK_WORKERS_ENV}={raw!r} is not a worker count within "
             f"the PB-assigned CPU affinity ({ceiling})")
    return value


def _open_head_journal(root, *, resume, identity, qnames, workers=1):
    """Open the head walk's journal, setting aside state that cannot be trusted.

    The machinery is the campaign's own (``prepare_journal``: identity-bound
    manifest, sha256-checked envelopes, ``--resume`` to reuse what a prior
    attempt banked). A journal that fails its own checks -- a key from
    another input set, a corrupt envelope -- is moved aside rather than
    refused, because the walk it describes is a pure function of inputs that
    are digest-bound on every load: the safe recovery is to redo the walk,
    and the honest one is to say so in the log. A journal this run was not
    told to resume is a refusal, exactly as for every other checkpoint.
    """
    root = Path(root)
    try:
        return prepare_journal(root, stage=HEAD_WALK_STAGE, resume=resume,
                               identity=identity, qnames=qnames, unit_workers=workers)
    except RuntimeError as exc:
        if not resume:
            raise
        import shutil
        stale = root.with_name(root.name + ".stale")
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()
        root.rename(stale)
        print(f"tessera_joint_aura: discarding head-walk checkpoint ({exc}); "
              "restarting the walk from the roster's start", flush=True)
        return prepare_journal(root, stage=HEAD_WALK_STAGE, resume=True,
                               identity=identity, qnames=qnames, unit_workers=workers)


def parse_unit_scope(spec, count=None):
    """Read ``lo:hi`` as a half-open slice of the census's ``sorted(names)``.

    The stage fans out over the one order every reader already walks, so a
    row's range can be cut from the census alone and two rows with disjoint
    ranges never contend for a cell.
    """
    if spec is None:
        return None
    _require(isinstance(spec, str) and spec.count(":") == 1,
             "unit scope must be spelled lo:hi over the sorted census roster")
    low, _, high = spec.partition(":")
    start = 0 if not low.strip() else int(low)
    stop = (count if not high.strip() else int(high))
    _require(stop is not None, "an open-ended unit scope needs the roster size")
    _require(type(start) is int and type(stop) is int and 0 <= start < stop,
             f"empty or reversed unit scope {spec!r}")
    return (start, stop)


def _synthesize_render_from_wire(render, *, wire, record, name, fmt, shape, reader, device="cpu"):
    """Write the missing decoded PWC shard from the verified wire blob.

    A rung this campaign adopted rather than encoded has its wire but no
    ``.pt``. Re-encoding it costs GPU-hours on the critical path; decoding
    the wire costs an I/O pass. What the decode cannot buy is evidence about
    the encode, so the marker below is written FIRST: a crash between the two
    writes leaves a marker with no shard, which the next load re-synthesizes,
    whereas the other order would leave a shard that reads as ``encoded``.

    ``device`` is the decoder's, not the shard's. Tessera's decoder is device
    parameterized and the same pure-torch reconstruction runs wherever it is
    pointed; the published bytes are the canonical CPU BF16 tensor either way
    (``_canonical_rendered_weight_tensor``). A caller that has already
    reserved a GPU decodes on it rather than leaving it at idle while one
    Python thread walks the wire (measured 0.064 s vs 0.92 s an expert cell).
    """
    import torch
    from .production_weight_cache import _store_rendered_weight_entry

    blob = Path(wire).read_bytes()
    _same(hashlib.sha256(blob).hexdigest(), record["blob_sha256"],
          f"{name}@{fmt}: wire checksum before synthesizing its render")
    try:
        decoded = _decode_wire(blob, reader=reader, device=device).to(torch.bfloat16)
    except Exception as exc:
        raise ValueError(f"{name}@{fmt}: original decoded PWC shard missing and "
                         f"its wire does not decode: {exc}") from exc
    _require(isinstance(decoded, torch.Tensor) and decoded.dtype == torch.bfloat16 and
             list(decoded.shape) == list(shape) and bool(torch.isfinite(decoded).all()),
             f"{name}@{fmt}: decoded wire is not the census BF16 render")
    marker = _render_origin_marker_path(render)
    Path(render).parent.mkdir(parents=True, exist_ok=True)
    _json(marker, {"schema": RENDER_ORIGIN_SCHEMA, "render_origin": "synthesized_from_wire",
                   "unit": name, "format_name": fmt, "wire_sha256": record["blob_sha256"],
                   "wire_file": Path(wire).name})
    _store_rendered_weight_entry(weights={}, cache_dir_path=Path(render).parent,
                                 qname=name, fmt=fmt, tensor=decoded,
                                 weight_dtype=torch.bfloat16, durable=True)
    _require(Path(render).is_file(), f"{name}@{fmt}: synthesized render was not published")
    return "synthesized_from_wire"


def _resolve_render_origin(render, *, wire, record, name, fmt, shape, reader, device="cpu"):
    """Name where this rung's decoded PWC shard came from, never guess it.

    The campaign journals fresh and resumed wires through one receipt grammar
    (``_checkpoint_wire_record``), so nothing in the record distinguishes an
    adopted rung from an encoded one. The marker beside the shard is the only
    place that fact can live, and its absence beside an existing shard is the
    campaign's own render.
    """
    render, marker = Path(render), _render_origin_marker_path(render)
    if render.is_file():
        if not marker.is_file():
            return "encoded"
        stamp = json.loads(marker.read_text())
        _same(stamp.get("schema"), RENDER_ORIGIN_SCHEMA, f"{name}@{fmt}: render origin schema")
        _same(stamp.get("render_origin"), "synthesized_from_wire",
              f"{name}@{fmt}: marked render origin")
        _same(stamp.get("unit"), name, f"{name}@{fmt}: marked render unit")
        _same(stamp.get("format_name"), fmt, f"{name}@{fmt}: marked render format")
        _same(stamp.get("wire_sha256"), record["blob_sha256"],
              f"{name}@{fmt}: synthesized render names another wire")
        return "synthesized_from_wire"
    _require(Path(wire).is_file(), f"{name}@{fmt}: original decoded PWC shard missing")
    return _synthesize_render_from_wire(render, wire=wire, record=record, name=name,
                                        fmt=fmt, shape=shape, reader=reader, device=device)


def _reclaim_head_walk_allocator(synthesis_device, *, force=False):
    """Return the cuda head walk's retained decode pool to the driver (#823).

    Called at unit-commit boundaries, on the committing thread; a worker
    mid-synthesis holds its live tensors as allocations, which the allocator
    never frees, so a concurrent decode is unaffected. A reclaim that fails
    for any reason is logged-not-raised: the walk's correctness does not
    depend on it, only its memory charge does.
    """
    if synthesis_device != "cuda":
        return
    import torch

    if not torch.cuda.is_available():
        return
    try:
        reserved = torch.cuda.memory_reserved()
        allocated = torch.cuda.memory_allocated()
        if not force and reserved - allocated < HEAD_WALK_RECLAIM_GAP_BYTES:
            return
        torch.cuda.empty_cache()
        after = torch.cuda.memory_reserved()
        print(f"tessera_joint_aura: head-walk allocator reclaim: reserved "
              f"{reserved // (1024 ** 2)} MiB -> {after // (1024 ** 2)} MiB "
              f"(allocated {allocated // (1024 ** 2)} MiB)", flush=True)
    except Exception as exc:  # noqa: BLE001 - never fail the walk on a reclaim
        print(f"tessera_joint_aura: head-walk allocator reclaim skipped: {exc!r}",
              flush=True)


def load_measured_anchor_input(inputs, *, file_hash_workers=1, verify_payloads=True,
                               defer_render_hashes=False, reader=None,
                               synthesis_device="cpu", unit_scope=None,
                               render_mirror_root=None, log_every=100,
                               require_existing_renders=False,
                               historical_encoder_reuse=None,
                               progress_phase=SYNTHESIS_PHASE,
                               head_checkpoint=None, head_resume=False,
                               head_walk_workers=None, head_walk_quantum=None):
    """Read a complete merged journal and select only its measured wire cells.

    The default hashes all payload files. Preparation may explicitly defer
    payload reads to the existing PWC loader and wire verifier; metadata and
    roster gates still run here. Tensor/source/encoder verification occurs in
    ``prepare_cache`` using actual source weights and the original capture.
    Interpolated menu rows are deliberately excluded rather than converted.

    The merged checkpoint is read by ``interned_json.load_json_file``: the
    standard library's own reader, plus one ``object_pairs_hook`` that makes the
    identity block's 205,243,544 menu-name occurrences cost one ``str`` per
    distinct name instead of 12.4 GiB. Values, refusals, exact types and the
    canonical seal are the stdlib decoder's, and the seal below is recomputed
    from the parsed graph (``tests/test_interned_json.py``). Measured on the
    real campaign: 13.812 GiB parse peak, 15.830 GiB for this whole metadata
    intake, inside the 21 GiB CPU envelope.

    A prepared COST run uses metadata intake with existing renders required.
    Its wire identities describe PREPARE's authenticated bytes; COST consumes
    no wire bodies. PWC verifies each consumed render against PREPARE's SHA.
    Selected export must authenticate its current wire bytes independently.
    The optional strict wire-only scan remains available to other callers.

    A rung this campaign adopted has its wire but no decoded PWC shard. The
    shard is synthesized from that wire here and every cell carries the
    resulting ``render_origin``, so no later report can read "verified" as
    "independently compared". ``reader`` is the same bound Tessera consumer
    ``verify_anchor_render`` uses; there is one decode seam, not two.

    ``synthesis_device`` is where that decode runs. It changes no published
    byte -- the shard is the canonical CPU BF16 tensor either way -- so a
    caller holding a GPU reservation passes it rather than spending the
    reservation on one CPU core.

    ``unit_scope`` restricts the per-cell walk to ``sorted(names)[lo:hi]``.
    Every roster, seal and fanout gate above still runs over the whole
    census; only the cells are cut, which is what makes the standalone
    synthesis stage fannable. The result carries the scope, and a scoped
    read is refused wherever the complete campaign input is required.

    ``render_mirror_root`` publishes synthesized shards under that root
    instead of the campaign's row caches, so a measuring run can compare its
    bytes against the campaign's without being able to replace them.

    ``require_existing_renders`` is for metadata-only export handoff readers:
    it refuses a missing prepared shard instead of synthesizing into the
    campaign workspace. No export manifest builder may repair frozen inputs.

    ``log_every`` prints a cumulative count and rate every N synthesized
    shards. Silence is the defect this phase was reported for: it ran for
    hours at 2.6 cells/s saying nothing. The default is chosen against that
    measured rate rather than rounded -- 100 shards is ~38 s there and ~16 s
    on the GPU, inside the two minutes a silent phase is a defect after.

    ``progress_phase`` is the PrismaBuild phase this walk reports under, or
    ``None`` to report nothing. The count is **units resolved**, committed
    after each unit's journal shard has been read and authenticated and every
    one of its cells has a proven render origin -- not only after a shard this
    walk had to write. A resume writes nothing, so the old synthesis-only
    report left the whole walk silent and the watchdog reaped an action that
    was demonstrably working (RobTand/prismaquant#678); the same silence hit a
    retried synthesis row whose range was already complete. The unit is the
    boundary every other loop in this file commits on -- the journal shard,
    the replay part -- so the run's counter has one currency throughout, and
    this census reports 36,423 times rather than 197,990. The caller names the
    phase because only it knows what its submission declared: a name outside
    the declared set grants no continuation.

    ``historical_encoder_reuse`` is the plan's explicit allowance for a
    recorded encoder source seal the installed package cannot re-derive. The
    checkpoint's recorded digest is compared against the installed package's
    own here, once, before any render origin is resolved or any wire decoded:
    an unlisted digest is refused in the intake rather than after the
    per-cell origin walk. An admitted reuse is carried on the returned input
    so ``prepare_cache`` hands the same record to every cell.

    ``head_checkpoint`` banks the walk itself, unit by unit, under the
    campaign's existing checkpoint machinery (``prepare_journal``/
    ``write_unit``, identity-bound manifest and sha256-checked envelopes), so
    a restarted action resumes mid-roster instead of re-paying the whole
    walk (#754). The journal's identity is the exact input set -- plan,
    census, receipts, merged cost and checkpoint digests, the checkpoint
    seal and the roster order -- plus the options that change banked rows;
    a journal keyed to anything else is moved aside and the walk starts
    fresh, never reused. ``head_resume`` is the same ``--resume`` every other
    checkpoint of this campaign requires: without it an existing journal is
    a refusal, not an overwrite. A resume re-verifies each banked unit
    against the bytes it was banked from -- the campaign journal shard's
    file digest, and each render/marker/wire stat fence -- before trusting
    it, truncating the cursor at the first drift and re-walking from there:
    unverified state is discarded, never trusted. Banked units are re-walked
    never; their reverified prefix is reported once at its final cumulative
    count, so the resumed run's counter and final state are identical to a
    fresh walk's without rewriting progress for every already-durable unit.
    Units are banked on a time cadence, so
    an interruption loses at most one interval of verified work.

    ``head_walk_workers`` is the walk's worker count. The default is the CPU
    set PrismaBuild assigned this container (``os.sched_getaffinity``), the
    env knob ``PRISMAQUANT_HEAD_WALK_WORKERS`` lowers it or forces the
    serial path with ``1``, and no value may exceed the assignment. However
    many workers run, commitment -- banking, reporting, cell insertion --
    stays in the roster's one deterministic order, so parallel and serial
    walks produce identical downstream state and identical durable
    sequences. The fan-out is threads: the per-unit work is file I/O and
    digesting (the GIL is released through the syscalls and by hashlib for
    the buffers that dominate), and the one thread-unsafe step left -- a
    missing render's synthesis through the bound reader -- holds a lock.
    The same bounded driver also restores resume envelopes and rechecks
    independent banked-unit fences; only a verified roster prefix is retained.

    ``head_walk_quantum`` is the opt-in distributed-quantum half of #765: a
    descriptor from ``joint_head_walk_quanta.head_walk_quanta`` naming this
    action's ``[lo, hi)`` slice of the sorted census roster. It requires
    ``unit_scope == (lo, hi)`` and a ``head_checkpoint`` to bank the slice
    into, and the slice journal is bound to the FULL roster digest plus the
    descriptor itself -- never to the slice alone -- so per-quantum journals
    join (``joint_head_walk_quanta.join_head_walk_journals``) into the one
    journal a resumed walk consumes unchanged. Without a descriptor a scoped
    read still cannot bank, exactly as before. Fleet execution of quanta
    waits on the D45 measurement gate (checksum-bound versus read-bound at
    full-core fan-out); until then this parameter is the producer/consumer
    contract under test, not the production prepare path, which passes
    nothing here.
    """
    from .production_weight_cache import _cache_weight_filename
    from tools.dispatch_tessera_campaign import _require_receipts

    reuse_policy = normalize_historical_encoder_reuse(historical_encoder_reuse)
    _require(type(verify_payloads) is bool, "verify_payloads must be an explicit boolean")
    _require(type(defer_render_hashes) is bool, "defer_render_hashes must be boolean")
    _require(not defer_render_hashes or verify_payloads,
             "deferred render hashes require complete wire verification")
    _require(not defer_render_hashes or require_existing_renders,
             "deferred render hashes require existing prepared renders")
    _require(not defer_render_hashes or unit_scope is None,
             "deferred render hashes require the complete candidate roster")
    _require(type(file_hash_workers) is int and file_hash_workers > 0,
             "positive file_hash_workers required")
    _require(type(log_every) is int and log_every >= 0, "non-negative log_every required")
    _require(progress_phase is None or (type(progress_phase) is str and progress_phase),
             "progress_phase must be a declared phase name or None")
    _require(type(require_existing_renders) is bool, "require_existing_renders must be boolean")
    _require(head_checkpoint is None or isinstance(head_checkpoint, (str, Path)),
             "head_checkpoint must be a directory path or None")
    _require(type(head_resume) is bool, "head_resume must be an explicit boolean")
    _require(head_walk_quantum is None or isinstance(head_walk_quantum, Mapping),
             "head_walk_quantum must be a quantum descriptor or None")
    # A partial roster is not the campaign's input, and a journal over part
    # of one would be name-gated reuse of the wrong thing; the standalone
    # synthesis stage stays checkpoint-less exactly as it is. A distributed
    # quantum (#765) is the one exception: its descriptor binds the slice to
    # the full roster, and the slice journal it banks joins with its peers.
    _require(head_checkpoint is None or unit_scope is None or head_walk_quantum is not None,
             "a scoped read cannot bank a head-walk journal")
    # Hashing only part of a roster does not verify that roster, so the two
    # options are refused together rather than quietly producing a record
    # that reads as a verified campaign input.
    _require(unit_scope is None or not verify_payloads,
             "a scoped read cannot also verify the complete campaign payload")
    paths = {key: _bound(inputs[key], key) for key in HEAD_WALK_INPUT_KEYS}
    census = json.loads(paths["census"].read_text())
    plan = json.loads(paths["campaign_plan"].read_text())
    _same(plan.get("schema"), "prismaquant.tessera_campaign_plan.v1", "campaign plan schema")
    _same(Path(plan["census"]).resolve(), paths["census"].resolve(), "campaign census path")
    _same(paths["campaign_receipts"].resolve(),
          (paths["campaign_plan"].parent / "receipts.json").resolve(), "campaign receipt path")
    rows = plan["rows"]
    _require(len({row["row_id"] for row in rows}) == len(rows), "duplicate campaign row")
    _require_receipts(paths["campaign_plan"].parent, len(rows))
    owners, groups = {}, {}
    for row in rows:
        for name in row["members"]:
            _require(name not in owners, f"duplicate campaign unit {name}")
            owners[name] = Path(row["dir"])
        for group in row["groups"]:
            _require(group not in groups, f"duplicate campaign group {group}")
            groups[group] = row["row_id"]
    names = set(census["unit_shapes"])
    _same(set(owners), names, "complete census roster")
    _same(set(groups), set(census["anchor_groups"]), "complete census groups")
    _same(len(names), inputs["required_source_units"], "declared full source unit count")
    _same(len(groups), inputs["required_campaign_groups"], "declared full campaign group count")
    for group, members in census["anchor_groups"].items():
        owner = next(row for row in rows if row["row_id"] == groups[group])
        _require(set(members) <= set(owner["members"]), f"campaign group membership changed: {group}")

    payload = pickle.loads(paths["merged_cost"].read_bytes())
    _same(payload.get("schema"), CAMPAIGN_SCHEMA, "campaign cost schema")
    _same(payload.get("currency"), CURRENCY, "campaign scalar currency")
    _same(set(payload["costs"]), names, "complete merged cost roster")
    provenance = payload["provenance"]
    _same(provenance.get("cost_mode"), "production-render-score", "campaign cost mode")
    _same(provenance.get("model"), census["model"], "campaign model")
    _same(plan["model"], census["model"], "planned model")
    _require(provenance.get("stopped_early") is False, "campaign stopped before completing anchors")
    _same(provenance.get("campaign_fanout", {}).get("rows"),
          {row["row_id"]: sorted(row["groups"]) for row in rows}, "complete merged fanout")

    manifest = load_json_file(paths["merged_checkpoint"])
    _same(manifest.get("schema"), MANIFEST_SCHEMA, "campaign checkpoint schema")
    _same(manifest.get("stage"), STAGE, "campaign checkpoint stage")
    identity = manifest["identity"]
    declared_seal = manifest.get("identity_sha256")
    if dev_mode_enabled():
        # Rob's dev-mode directive (2026-09-19): the seal returns at the
        # artifact gate, not the run gate. This recompute is the run gate's
        # most expensive step on the real campaign -- 302.653 s of a 765.6 s
        # in-process profile on action 282c61140ba7 (2026-09-20) over the
        # 7.2 GB merged checkpoint -- and the digest it would produce is
        # already declared by the manifest. The declared value is checked for
        # 64-hex shape and RECORDED, never silently trusted: the walk still
        # hands it to every unit envelope below (``_load_unit``), and the run
        # stays ``dev_uncertified`` through the existing stamp.
        _require_sha256(declared_seal, "campaign checkpoint seal")
        dev_warning(
            "campaign checkpoint seal not recomputed under dev mode; using "
            f"the manifest's declared identity_sha256 {declared_seal} "
            "(recorded, not gated)")
        seal = declared_seal
    else:
        # The checkpoint is parsed from JSON, so its identity is already
        # normalized (string keys, dict/list containers, JSON scalars) and the
        # seal can stream the canonical bytes into the digest. The generic
        # helper normalizes first, which holds the encoded text, a second full
        # graph and the second encoded text at once; on a checkpoint this size
        # that is the difference between fitting a bounded envelope and being
        # killed by it. Same digest -- held by
        # tests/test_canonical_json_normalized.py.
        seal = canonical_json_sha256_normalized(identity, where="joint anchor input")
        _same(seal, declared_seal, "campaign checkpoint seal")
    _same(identity.get("campaign_schema"), CAMPAIGN_SCHEMA, "checkpoint campaign schema")
    _same(identity.get("currency"), CURRENCY, "checkpoint scalar currency")
    _same(set(identity["units"]), names, "complete checkpoint identity roster")
    listed = [row["qname"] for row in manifest["units"]]
    _require(len(listed) == len(names) and set(listed) == names, "incomplete checkpoint unit roster")
    for key in ("prismaquant_source_sha256", "encoder_source_sha256"):
        value = identity.get(key)
        _require(isinstance(value, str) and len(value) == 64 and
                 all(c in "0123456789abcdef" for c in value), f"missing checkpoint {key}")
    # The one encoder identity a run can re-derive is the installed package's
    # own. Refuse anything else here -- before the per-cell origin walk reads
    # or decodes a wire -- unless the plan named that exact digest.
    from . import tessera_campaign as tc
    encoder_source_reuse = resolve_encoder_source_reuse(
        identity["encoder_source_sha256"], tc._checkpoint_identity_api().encoder_source_sha256(),
        reuse_policy, where="joint anchor checkpoint encoder source")
    parts = merged_checkpoint_parts(paths["merged_checkpoint"])
    for row in manifest["units"]:
        _same(parts / row["file"], unit_path(parts, row["qname"]), "canonical checkpoint unit path")

    cells, formats = {}, {}
    wire_dir = Path(provenance["wire_dir"])
    # Resolve once, not per cell (#711): every cell of the campaign shares
    # this one wire directory, so resolving it per cell re-walks the same ~12
    # NFS path components ~198k times (~62 LOOKUPs/cell measured on the
    # GLM-5.3 run). The per-cell escape check below still stats the wire
    # itself; the recorded path is joined from the resolved root.
    wire_dir_resolved = wire_dir.resolve()
    # One resolved root per distinct row directory, for the same reason: the
    # recorded render is joined from the resolved root instead of resolved
    # per cell, and the join reaches the same file the walk verifies.
    owner_roots = {str(directory): Path(directory).resolve()
                   for directory in set(map(str, owners.values()))}
    mirror_root = None if render_mirror_root is None else Path(render_mirror_root)
    roster = sorted(names)
    if unit_scope is not None:
        low, high = unit_scope
        _require(type(low) is int and type(high) is int and 0 <= low < high <= len(roster),
                 f"unit scope {unit_scope} is outside the {len(roster)}-unit census roster")
        roster = roster[low:high]
    quantum_bounds = None
    if head_walk_quantum is not None:
        # The descriptor is checked against the live census roster, not
        # against the slice: a quantum from another campaign or roster
        # revision is refused before it banks anything. A quantum that
        # banks nothing is a refused quantum -- the shard is its output.
        quantum_bounds = check_quantum_for_roster(head_walk_quantum, sorted(names))
        _require(tuple(unit_scope or ()) == tuple(quantum_bounds),
                 "head-walk quantum descriptor does not name this scoped read")
        _require(head_checkpoint is not None,
                 "a head-walk quantum banks its slice journal or refuses")
    resolved, synthesized, started = 0, 0, time.time()
    walk_workers = _head_walk_worker_count(head_walk_workers)

    # -- the walk banks its prefix and fans out (#754) ----------------------
    # Every action restart used to re-pay this whole loop. Now each verified
    # unit is journalled under the campaign's own checkpoint machinery, and
    # a resume re-verifies the banked prefix against the bytes it was banked
    # from before trusting a single row of it. The journal is opt-in: every
    # other caller of this loader sees exactly the walk that ran before.
    head_root = head_seal = None
    banked = []
    if head_checkpoint is not None:
        head_identity = {
            "schema": HEAD_WALK_JOURNAL_SCHEMA,
            "inputs": {key: inputs[key]["sha256"] for key in (
                "campaign_plan", "census", "campaign_receipts",
                "merged_cost", "merged_checkpoint")},
            "required_source_units": inputs["required_source_units"],
            "required_campaign_groups": inputs["required_campaign_groups"],
            "checkpoint_seal": seal,
            "roster_sha256": hashlib.sha256("\n".join(roster).encode("utf-8")).hexdigest(),
            "render_mirror_root": render_mirror_root,
            "encoder_source_reuse_sha256": (None if encoder_source_reuse is None else
                canonical_json_sha256(encoder_source_reuse, where="head walk encoder reuse")),
        }
        if head_walk_quantum is not None:
            # The shard is bound to the full roster, never to the slice
            # alone: the digest is the verified descriptor's (checked against
            # the live census above), and the descriptor itself rides in the
            # identity so the collector can match each shard to the PB action
            # it was assigned to. The default path is untouched -- without a
            # descriptor a scoped walk still cannot bank, so `roster` here is
            # the whole roster there.
            head_identity["roster_sha256"] = head_walk_quantum["roster_sha256"]
            head_identity["quantum"] = dict(head_walk_quantum)
        head_root, head_seal, completed = _open_head_journal(
            Path(head_checkpoint), resume=head_resume, identity=head_identity, qnames=roster,
            workers=walk_workers)

        def _banked_unit_still_binds(name, state):
            """Re-verify one banked unit against the very bytes it names.

            Nothing is trusted because it is journalled: the campaign shard's
            file digest, and every render, marker and wire fence, must still
            answer for the rows the walk banked before the cursor advances
            over this unit. The global gates above already re-bound the
            whole input set on this load, so the per-unit fences are the
            only bytes that can have moved since.
            """
            try:
                if not isinstance(state, dict) or state.get("schema") != HEAD_WALK_STATE_SCHEMA:
                    return False
                rows, files = state.get("cells"), state.get("unit_files")
                if (not isinstance(rows, dict) or not isinstance(files, dict)
                        or set(rows) != set(files)
                        or not isinstance(state.get("formats"), list)
                        or not isinstance(state.get("journal_file_sha256"), str)):
                    return False
                if _sha(unit_path(parts, name)) != state["journal_file_sha256"]:
                    return False
                for fmt, fence in files.items():
                    row = rows[fmt]
                    if not isinstance(row, dict) or not isinstance(fence, dict):
                        return False
                    if Path(row["wire"]).stat().st_size != fence.get("wire_size"):
                        return False
                    if _stat_signature(Path(row["render"]).stat()) != tuple(
                            fence.get("render_signature") or ()):
                        return False
                    marker = _render_origin_marker_path(row["render"])
                    digest = (None if not marker.is_file()
                              else hashlib.sha256(marker.read_bytes()).hexdigest())
                    if digest != fence.get("marker_sha256"):
                        return False
                return True
            except (OSError, ValueError, TypeError, KeyError):
                return False

        # The cursor is a roster prefix: re-verify in the walk's own order and
        # truncate at the first unit that no longer answers for itself. The
        # stale suffix is discarded -- its units are re-walked, never reused.
        class _PrefixEnded(Exception):
            pass

        def verify_banked(name):
            state = completed.get(name)
            return state if state is not None and _banked_unit_still_binds(name, state) else None

        def retain_banked(name, state):
            if state is None:
                raise _PrefixEnded
            banked.append((name, state))

        # Independent read-only authentication may finish out of order, but
        # only the roster prefix is reusable. The existing bounded driver
        # cancels queued suffix work and joins active reads before recovery.
        try:
            _drive_ordered_walk(roster, verify_banked, retain_banked, workers=walk_workers)
        except _PrefixEnded:
            pass
        for name in roster[len(banked):]:
            if name in completed:
                unit_path(head_root, name).unlink(missing_ok=True)
        # Every banked unit has passed its original fences before this loop.
        # Assemble that durable prefix in roster order, then publish its one
        # final cumulative count. Rewriting the same atomic progress file for
        # every replayed unit paid 36,423 serial NFS writes in the GLM head
        # without establishing any additional durable work (#822). Nothing
        # it synthesized counts as written now -- a resume writes nothing.
        for name, state in banked:
            for fmt, row in state["cells"].items():
                cells[name, fmt] = row
            formats[name] = tuple(state["formats"])
            resolved += 1
        if banked and progress_phase is not None:
            _pb_commit(resolved, progress_phase, unit=banked[-1][0])

    def walk_one(name):
        """Verify one unit end to end; the rows, the fences, the events."""
        state = _load_unit(unit_path(parts, name), stage=STAGE, qname=name, identity_sha256=seal)
        _require(isinstance(state, dict) and set(state) - {"unservable"} == {"anchors", "wire_records"},
                 f"{name}: incomplete measured anchor journal")
        anchors = {anchor["format_name"]: anchor for anchor in state["anchors"]}
        _require(anchors and len(anchors) == len(state["anchors"]) and
                 set(anchors) == set(state["wire_records"]), f"{name}: anchor/receipt coverage differs")
        measured = {fmt for fmt, row in payload["costs"][name].items()
                    if row.get("output_mse_measured") is True}
        _same(set(anchors), measured, f"{name}: measured payload/journal coverage")
        unit = identity["units"][name]
        _same(unit["weight"]["shape"], census["unit_shapes"][name], f"{name}: census source shape")
        unit_cells, fences, events = {}, {}, []
        for fmt, anchor in sorted(anchors.items()):
            row = payload["costs"][name][fmt]
            _require(fmt in unit["menu"] and anchor["qname"] == name, f"{name}: anchor outside exact menu")
            _require(row.get("cost_source") == "tessera_campaign_measured" and
                     row.get("tessera_provenance") == "measured" and row.get("currency") == CURRENCY,
                     f"{name}@{fmt}: interpolated or foreign measured row")
            for target, source in (("output_mse", "dloss"), ("tessera_family", "family"),
                    ("tessera_body_rate_q256", "body_rate_q256"), ("activation_contract", "activation_contract"),
                    ("activation_quantized", "activation_quantized"), ("wire_bytes", "wire_bytes"),
                    ("input_global_scale", "input_global_scale")):
                _same(row.get(target), anchor.get(source), f"{name}@{fmt}: measured {target}")
            _require(type(anchor["dloss"]) in (int, float) and math.isfinite(anchor["dloss"])
                     and anchor["dloss"] >= 0, f"{name}@{fmt}: invalid measured value")
            _same(row["hessian_identity"].get("applied"), anchor["hessian_applied"], f"{name}: H applicability")
            for key in ("supplied", "capture_sha256", "text_sha256", "fit_ids_sha256", "fit_tokens"):
                _same(row["hessian_identity"].get(key), provenance["hessian"].get(key), f"{name}: measured H {key}")
            if anchor.get("input_global_scale") is not None:
                _same(anchor["input_global_scale"], unit.get("input_global_scale"), f"{name}: checkpoint scale")
                _same(anchor["input_global_scale"], provenance["activation_static_scales"]["units"].get(name),
                      f"{name}: merged static scale")
            record = state["wire_records"][fmt]
            recorded = record["identity"]
            _same(recorded.get("unit"), name, f"{name}: wire unit")
            _same(recorded.get("source"), unit["weight"], f"{name}: recorded source")
            _same(recorded.get("encoder_source_sha256"), identity["encoder_source_sha256"], f"{name}: encoder source")
            _same(recorded["recipe"].get("q256"), anchor["body_rate_q256"], f"{name}: wire rung")
            if anchor["hessian_applied"]:
                _same(recorded["calibration"]["hessian"], unit["hessian"], f"{name}: recorded H")
            else:
                _same(recorded.get("calibration"), None, f"{name}: unexpected recorded H")
            filename = record["file"]
            _require(isinstance(filename, str) and Path(filename).name == filename and
                     filename not in {".", ".."}, f"{name}: escaping wire filename")
            # The filename is a validated leaf, so the wire's lexical parent
            # is the wire directory itself: the escape check is the symlink
            # test on the wire, and the directory side is resolved once above
            # instead of per cell (PrismaQuant #711).
            wire = wire_dir / filename
            _require(not wire.is_symlink(), f"{name}: escaping wire path")
            wire_stat = wire.stat()
            _same(wire_stat.st_size, record["blob_bytes"], f"{name}: wire size")
            render = owner_roots[str(owners[name])] / "cache" / _cache_weight_filename(name, fmt)
            # The mirror keeps the render's absolute path under the mirror
            # root so a cell's two copies stay comparable by name and a
            # measuring run can never replace the campaign's own bytes.
            target = (render if mirror_root is None
                      else mirror_root / render.relative_to(render.root))
            present = Path(target).is_file()
            if require_existing_renders and not present:
                raise ValueError(f"{name}@{fmt}: prepared render is missing; selected cache will not synthesize it")
            if present:
                origin = _resolve_render_origin(target, wire=wire, record=record, name=name,
                                                fmt=fmt, shape=census["unit_shapes"][name],
                                                reader=reader, device=synthesis_device)
            else:
                # The one thread-unsafe step in the walk: synthesis decodes
                # through the bound reader, which is proven single-threaded
                # (the qualification walk's wire pool is max_workers=1), so
                # the rare missing render waits on one lock while pure-read
                # verification overlaps freely around it.
                with _HEAD_WALK_SYNTHESIS_LOCK:
                    origin = _resolve_render_origin(target, wire=wire, record=record, name=name,
                                                    fmt=fmt, shape=census["unit_shapes"][name],
                                                    reader=reader, device=synthesis_device)
                events.append(fmt)
            unit_cells[fmt] = {"anchor": anchor, "record": record, "wire": str(wire_dir_resolved / filename),
                               "render": str(target), "render_origin": origin,
                               **({} if render_mirror_root is None
                                  else {"campaign_render": str(render)})}
            # The resume fence for this cell: enough identity to prove on the
            # next load that these bytes are the bytes this row was banked
            # from, at stat-and-digest cost rather than a re-walk.
            marker = _render_origin_marker_path(target)
            fences[fmt] = {"wire_size": wire_stat.st_size,
                           "render_signature": list(_stat_signature(Path(target).stat())),
                           "marker_sha256": (None if not marker.is_file()
                                             else hashlib.sha256(marker.read_bytes()).hexdigest())}
        formats_row = (*sorted(anchors), "BF16")
        return {"formats": formats_row, "cells": unit_cells, "synthesized_events": events,
                "bank": {"schema": HEAD_WALK_STATE_SCHEMA,
                         "journal_file_sha256": _sha(unit_path(parts, name)),
                         "formats": list(formats_row), "cells": unit_cells,
                         "unit_files": fences}}

    pending_bank = []
    last_bank = time.monotonic()

    def _flush_bank():
        for banked_name, bank in pending_bank:
            write_unit(head_root, stage=HEAD_WALK_STAGE, qname=banked_name,
                       identity_sha256=head_seal, state=bank)
        pending_bank.clear()

    def commit_one(name, result):
        nonlocal resolved, synthesized, last_bank
        for fmt, row in result["cells"].items():
            cells[name, fmt] = row
        formats[name] = tuple(result["formats"])
        for fmt in result["synthesized_events"]:
            synthesized += 1
            if log_every and synthesized % log_every == 0:
                elapsed = max(time.time() - started, 1e-9)
                print(f"tessera_joint_aura: synthesized {synthesized} renders "
                      f"in {elapsed:.0f}s ({synthesized / elapsed:.2f} cells/s), "
                      f"at {name}@{fmt}", flush=True)
        resolved += 1
        # After the unit's shards are durable and its origins proven, never on
        # entering the loop.
        if progress_phase is not None:
            _pb_commit(resolved, progress_phase, unit=name)
        if head_root is not None:
            pending_bank.append((name, result["bank"]))
            if time.monotonic() - last_bank >= HEAD_WALK_BANK_INTERVAL_S:
                _flush_bank()
                last_bank = time.monotonic()
        # Unit boundary: the decoded shards for this unit are durable and the
        # tensors themselves are dead, so whatever the caching allocator is
        # still holding for them is retention, not working set (#823).
        _reclaim_head_walk_allocator(synthesis_device)

    try:
        # However many workers fan the verification out, commitment -- cell
        # insertion, reporting, banking -- stays in the roster's one order,
        # so the journal holds a prefix and the durable sequence means what
        # it meant serially.
        _drive_ordered_walk(roster[len(banked):], walk_one, commit_one, workers=walk_workers)
    finally:
        # The interruption window: units that committed inside the last
        # cadence interval are banked before the failure leaves this
        # function, so a restart resumes from the walk's true frontier.
        if head_root is not None and pending_bank:
            _flush_bank()
    if log_every and synthesized:
        elapsed = max(time.time() - started, 1e-9)
        print(f"tessera_joint_aura: synthesized {synthesized} renders in {elapsed:.0f}s "
              f"({synthesized / elapsed:.2f} cells/s) on {synthesis_device}", flush=True)
    scoped = dict(unit_scope=unit_scope, synthesized_now=synthesized,
                  progress_committed=(0 if progress_phase is None else resolved),
                  render_mirror_root=None if render_mirror_root is None else str(render_mirror_root),
                  head_walk_workers=walk_workers, head_walk_resumed_units=len(banked))
    result = MeasuredAnchorInput(dict(inputs), payload, manifest, census, plan, cells,
                                formats, encoder_source_reuse=encoder_source_reuse, **scoped)
    if inputs.get("candidate_overlay") is not None:
        from .joint_catalog_extension import attach_candidate_overlay
        attach_candidate_overlay(result, inputs["candidate_overlay"])
    if not verify_payloads:
        return result

    def verify_files(item):
        pair, cell = item
        wire, render = Path(cell["wire"]), Path(cell["render"])
        # Metadata is only a race detector around the actual content hash.
        # Every byte is still hashed; neither timestamps nor a previous run
        # authorize reuse. Existing per-consumption render checks remain below.
        def signature(path):
            stat = path.stat()
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        before = [signature(p) for p in (wire, render)]
        _same(_sha(wire), cell["record"]["blob_sha256"], f"{pair}: wire checksum")
        digest = None if defer_render_hashes else _sha(render)
        after = [signature(p) for p in (wire, render)]
        _same(after, before, f"{pair}: input files changed while hashing")
        return pair, digest

    if file_hash_workers == 1:
        verified_files = map(verify_files, cells.items())
        for pair, digest in verified_files:
            if digest is not None:
                cells[pair]["render_file_sha256"] = digest
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=file_hash_workers, thread_name_prefix="anchor-file-hash") as workers:
            for pair, digest in workers.map(verify_files, cells.items()):
                if digest is not None:
                    cells[pair]["render_file_sha256"] = digest
    return result


def calibrated_maxima(data, profile):
    """Reuse the producer's full-census fused scale policy; never invert G."""
    from . import tessera_campaign as tc
    from .nvfp4_activation_contract import unify_fused_sibling_max_abs

    positive = {name: float(value) for name, value in data.census["max_abs"].items()
                if float(value) > 0.0}
    maxima = unify_fused_sibling_max_abs(positive, profile=profile, tolerate_profile_errors=True)
    scales, policy = tc._static_input_scales(data.census["max_abs"], profile=profile)
    stamped = data.payload["provenance"]["activation_static_scales"]
    _same(policy, stamped["policy"], "campaign static scale policy")
    _same(scales, stamped["units"], "campaign fused static scales")
    return maxima, scales


def verify_anchor_render(cell, source_weight, rendered_weight, *, calibration_source,
                         projected_unit, static_scales, bound_unit=None, reader=None,
                         release_file_pages=False, wire_blob=None, wire_sha256=None,
                         encoder_source_reuse=None):
    """Re-derive encoder inputs from actual source/H and compare decoded bytes.

    Two legs, and they do not establish the same thing. ``verify_cached_unit``
    checks the wire against an encoder identity re-derived from the streamed
    source weights and H; it is independent of anything the cache holds and it
    is what qualifies an adopted rung at all. The ``torch.equal`` leg compares
    the decoded wire with the render on disk: for an ``encoded`` rung that is
    an independent render/wire agreement, and for a ``synthesized_from_wire``
    rung the render was written by decoding that same wire, so it can only
    establish that the shard still round-trips -- a corruption check between
    the write and this read, not evidence about the encode. The returned
    record names both facts so a reader never has to infer which one it holds.

    ``encoder_source_reuse`` is the intake's explicit record for a checkpoint
    priced under a historical encoder package. The freshly derived identity is
    compared against it -- the run's own package must be the one that record
    observed -- and then exactly one field, the encoder source seal, is
    replaced by the recorded value so the historical wire receipt can be
    checked against the historical identity it was written under. Both
    digests are returned: ``current_encoding_identity_sha256`` over the
    identity this package derives, and ``encoding_identity_sha256`` over the
    identity actually compared. Nothing in the wire or its record is
    rewritten, and the substitution is refused without a validated record.
    """
    import torch
    from . import tessera_campaign as tc
    from .production_weight_cache import _cb_cache_tensor_identity

    anchor = tc.CampaignAnchor(**cell["anchor"])
    name, fmt = anchor.qname, anchor.format_name
    render_origin = cell.get("render_origin")
    _require(render_origin in RENDER_ORIGINS,
             f"{name}@{fmt}: cell carries no closed-vocabulary render_origin")
    render_comparison = RENDER_COMPARISON_BY_ORIGIN[render_origin]
    _require(source_weight.dtype == rendered_weight.dtype == torch.bfloat16 and
             source_weight.ndim == 2 and rendered_weight.shape == source_weight.shape,
             f"{name}@{fmt}: source/render BF16 shape differs")
    source_receipt = (None if bound_unit is None else bound_unit.source_receipt(source_weight))
    _require((bound_unit is not None or bool(torch.isfinite(source_weight).all())) and
             bool(torch.isfinite(rendered_weight).all()), f"{name}@{fmt}: source/render is nonfinite")
    expected = tc._checkpoint_anchor_identity(anchor,
        weights={name: source_weight}, menus={name: [SimpleNamespace(format_name=fmt)]},
        calibration_source=calibration_source, static_scales=static_scales,
        projected_units={} if projected_unit is None else {name: projected_unit},
        **({} if bound_unit is None else {"bound_unit": bound_unit}))
    reuse = require_encoder_source_reuse_record(
        encoder_source_reuse, where=f"{name}@{fmt} encoder reuse")
    current_encoding_identity_sha256 = None
    if reuse is not None:
        # The identity this run just derived is the only thing that can say
        # which package the substitution is a substitution from.
        _same(expected.get("encoder_source_sha256"),
              reuse["observed_current_encoder_source_sha256"],
              f"{name}@{fmt}: installed encoder source seal")
        current_encoding_identity_sha256 = canonical_json_sha256(
            expected, where="joint anchor current encoding identity")
        expected["encoder_source_sha256"] = reuse["recorded_encoder_source_sha256"]
    wire_path = Path(cell["wire"])
    wire_stat = wire_path.stat() if release_file_pages else None
    if wire_blob is None:
        blob, actual_wire_sha256 = _read_verified_wire_blob(cell)
    else:
        _require(isinstance(wire_blob, bytes), f"{name}@{fmt}: wire reader returned non-bytes")
        _same(len(wire_blob), cell["record"].get("blob_bytes"),
              f"{name}@{fmt}: read-ahead wire size differs from receipt")
        blob = wire_blob
        if wire_sha256 is None:
            actual_wire_sha256 = hashlib.sha256(blob).hexdigest()
        else:
            # The read-ahead reader already fenced this exact buffer -- size,
            # stat signatures and the receipt digest -- and the handoff is one
            # process reference the reader thread no longer touches, so a
            # third hash here is a second full pass over the same bytes
            # (PQ #725). Bind the reader's digest to the receipt instead.
            actual_wire_sha256 = _require_sha256(
                wire_sha256, f"{name}@{fmt} read-ahead wire digest")
        _same(actual_wire_sha256, cell["record"].get("blob_sha256"),
              f"{name}@{fmt}: read-ahead wire checksum")
    verifier = tc._checkpoint_identity_api() if reader is None else reader
    verifier.verify_cached_unit(blob, cell["record"], expected)
    decoded = _decode_wire(blob, reader=reader,
                           device=str(rendered_weight.device)).to(torch.bfloat16)
    # Run on both origins. On a synthesized render it cannot fail as evidence
    # about the encode, and it is still live evidence that the shard on disk
    # decodes to the bytes it was written from.
    _require(torch.equal(decoded, rendered_weight),
             f"{name}@{fmt}: decoded wire differs from original PWC render"
             if render_origin == "encoded" else
             f"{name}@{fmt}: synthesized PWC render no longer decodes from its wire")
    del decoded
    if release_file_pages:
        from .perturbed_x_cache import release_activation_cache_file_pages
        release_activation_cache_file_pages(wire_path, expected_stat=wire_stat)
    return {"source_weight": (_cb_cache_tensor_identity(source_weight)
                              if source_receipt is None else source_receipt),
            "rendered_weight": _cb_cache_tensor_identity(rendered_weight),
            "encoding_identity_sha256": canonical_json_sha256(expected, where="joint anchor encoding"),
            # Present only when the encoder source seal was substituted. The
            # digest this run's own package derives is retained beside the one
            # actually compared, so the receipt never reads as a re-derivation
            # of the historical seal.
            **({} if reuse is None else {
                "current_encoding_identity_sha256": current_encoding_identity_sha256,
                "encoder_source_reuse": reuse}),
            "wire_sha256": actual_wire_sha256,
            "render_file_sha256": cell["render_file_sha256"],
            "render_origin": render_origin, "render_comparison": render_comparison}


def _live_targets(runner, names):
    from .aura_cost import _target_linears
    from .routed_experts import profile_declared_packed_expert_projections

    targets = _target_linears(runner.model, include_routed_experts=True, profile=runner.profile)
    packed = profile_declared_packed_expert_projections(runner.model, runner.profile)
    targets.update({member.qname: member for member in packed})
    _require(set(names) <= set(targets), "census units are absent from the actual streamed source")
    return {name: targets[name] for name in names}


def _prepare_file_read_bound(data, *, max_render_bytes):
    """Refuse an oversized later donor before any layer allocates read buffers."""
    maximum = max(Path(cell["render"]).stat().st_size for cell in data.cells.values())
    _require(0 < maximum <= max_render_bytes,
             "original render shard exceeds the declared PWC read buffer budget")
    return maximum


def _prepare_wire_read_bound(data):
    """Reserve the one bounded, receipt-sized wire buffer used by qualification."""
    sizes = [cell["record"].get("blob_bytes") for cell in data.cells.values()]
    _require(bool(sizes) and all(type(size) is int and size > 0 for size in sizes),
             "measured wire receipts need positive blob_bytes")
    return max(sizes)


QUALIFICATION_WINDOW_SCHEMA = "prismaquant.joint_anchor_qualification.v1"
QUALIFICATION_STAGE = "Tessera joint anchor qualification"
QUALIFICATION_CELLS_SCHEMA = "prismaquant.joint_qualification_cells.v1"


def normalize_qualification_window(config):
    if config is None:
        return None
    fields = {"schema", "max_capture_resident_bytes", "max_load_buffer_bytes",
              "workspace_reserve_bytes"}
    _require(isinstance(config, dict) and set(config) == fields and
             config.get("schema") == QUALIFICATION_WINDOW_SCHEMA,
             "joint anchor qualification requires a complete v1 window policy")
    for key in fields - {"schema"}:
        _require(type(config[key]) is int and config[key] > 0,
                 f"qualification window requires positive finite {key}")
    return dict(config)


def _qualification_capture_sizes(data, identity, policy):
    """Validate the whole roster before a first unit's X/H can be loaded."""
    sizes = {}
    for name in data.formats_by_qname:
        columns = data.census["unit_shapes"][name][1]
        rows = min(data.census["counts"][name], identity["max_act_rows"])
        _require(type(columns) is int and columns > 0 and type(rows) is int and rows >= 0,
                 f"{name}: invalid canonical capture geometry")
        sizes[name] = 4 * (columns * columns + rows * columns)
        _require(sizes[name] <= policy["max_capture_resident_bytes"],
                 f"{name}: canonical capture exceeds qualification budget")
    return sizes


def _qualification_file_sha(path):
    """Hash one held regular file; reject symlinks and path/descriptor drift."""
    path = Path(path)
    def signature(value):
        return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
                value.st_mtime_ns, value.st_ctime_ns)
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode), f'qualification input is not a regular file: {path}')
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise RuntimeError(f'qualification input changed before its read: {path}') from error
    with os.fdopen(fd, 'rb') as handle:
        _same(signature(os.fstat(handle.fileno())), signature(before),
              f'qualification input changed before its read: {path}')
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        _same(signature(os.fstat(handle.fileno())), signature(before),
              f'qualification input changed while hashing: {path}')
    _same(signature(path.lstat()), signature(before),
          f'qualification input changed while hashing: {path}')
    return digest


def _qualification_replay(data, manifest, completed, *, sealed=None, committed=0):
    """A journal envelope alone does not authenticate files that remain live.

    The roster is walked in the order sealed at submission, one complete unit
    at a time: its X/H capture, then each measured rung's wire and render. That
    is the order the data manifest declares for these bytes and the reason the
    resumed pass can announce a consumed prefix at all (RobTand/prismaquant#607).
    ``sealed`` is the submission's own read order and phase table, checked
    against this journal before the first read, so a part is announced exactly
    when the action reaches its first unit -- with the bytes in front of it
    already read -- and the count advances only after a unit's own reads finish.

    ``committed`` is where the run's counter already stands -- the head walk's
    resolved units -- because a count that restarts at zero here would go
    backwards and renew no allowance (RobTand/prismaquant#678).

    Without ``sealed`` this walk reports nothing, and that is not an oversight
    it can repair: ``prepare_cache`` refuses a resumed pass that carries sealed
    phases without its sealed frontier, so an unsealed replay only happens when
    the submission declared no phases at all -- there is no name to report
    under and no watchdog listening.
    """
    from .perturbed_x_cache import activation_cache_filename
    from .joint_prewarm_phases import HEAD_PHASE
    from .joint_replay_frontier import (
        PHASE_START_UNITS_KEY, ROSTER_KEY)

    # The immutable manifest seals the full roster. Only a unit whose
    # qualification is actually skipped needs its X/H bytes re-authenticated
    # here; unfinished units pass the usual verified capture loader later.
    root = Path(data.payload['provenance']['calibration_cache']['path']).parent
    order = sorted(completed) if sealed is None else list(sealed[ROSTER_KEY])
    starts = None if sealed is None else sealed[PHASE_START_UNITS_KEY]
    verified = {}
    current = HEAD_PHASE
    for index, name in enumerate(order):
        if starts is not None and starts[name] != current:
            # The previous part is read to its end, so everything in front of
            # this one has been consumed: PB releases only that prefix.
            current = starts[name]
            _pb_commit(committed + index, current, unit=name)
        entry = manifest['entries'][name]
        expected = str(Path('inputs') / activation_cache_filename(name))
        _same(entry.get('path'), expected, f'{name}: canonical X/H entry')
        _same(_qualification_file_sha(root / expected), entry['sha256'],
              f'{name}: canonical X/H bytes changed')
        state = completed[name]
        rows = state.get('verified_cells')
        _require(isinstance(rows, dict) and set(rows) == set(data.formats_by_qname[name]) - {'BF16'},
                 f'{name}: incomplete qualification journal cells')
        _require(isinstance(state.get('prefetch'), list), f'{name}: missing qualification prefetch')
        for fmt, record in rows.items():
            cell = data.cells[name, fmt]
            required = {'source_weight', 'rendered_weight', 'encoding_identity_sha256',
                        'wire_sha256', 'render_file_sha256', 'render_origin',
                        'render_comparison', 'activation'}
            # ``_verify_cell`` adds these two *only* when the encoder source
            # seal was substituted, and says so at its own return. They are
            # extra attestation, never a substitute for a required field, so
            # the check is "every required field is present and nothing
            # unknown is" -- not set equality, which refused every receipt the
            # writer produced under an encoder-reuse allowlist.
            optional = {'current_encoding_identity_sha256', 'encoder_source_reuse'}
            _require(isinstance(record, dict) and required <= set(record)
                     and set(record) <= required | optional
                     and isinstance(record['activation'], dict),
                     f'{name}@{fmt}: incomplete qualification receipt')
            # The writer emits the pair together or not at all, and the
            # journalled record is bound to THIS run's intake, not merely
            # shape-checked: a journal written under one allowlist entry must
            # not be replayed under another.
            _same('encoder_source_reuse' in record,
                  'current_encoding_identity_sha256' in record,
                  f'{name}@{fmt}: half a reuse receipt')
            _same(record.get('encoder_source_reuse'),
                  getattr(data, 'encoder_source_reuse', None),
                  f'{name}@{fmt}: journalled encoder reuse')
            _same(record.get('render_origin'), cell['render_origin'],
                  f'{name}@{fmt}: journal render origin')
            _same(record.get('render_comparison'), RENDER_COMPARISON_BY_ORIGIN[cell['render_origin']],
                  f'{name}@{fmt}: journal render comparison')
            _same(record.get('wire_sha256'), cell['record']['blob_sha256'],
                  f'{name}@{fmt}: journal wire receipt')
            _same(_qualification_file_sha(cell['wire']), record['wire_sha256'],
                  f'{name}@{fmt}: upstream wire bytes changed')
            _same(_qualification_file_sha(cell['render']), record.get('render_file_sha256'),
                  f'{name}@{fmt}: upstream render bytes changed')
            cell['render_file_sha256'] = record['render_file_sha256']
            verified[name, fmt] = record
        if starts is not None:
            # The unit is durable at this point; the count is what renews the
            # watchdog's allowance while a part is still being re-read.
            _pb_commit(committed + index + 1, current, unit=name)
    return verified


def _qualification_cells_sha256(cells):
    """Seal exact cell fields without a second, full-roster JSON allocation.

    The parent campaign checkpoint, cost and census are independently bound.
    Origin markers and resolved render/wire paths can still differ, so hash
    their complete sorted cell records here. Length framing keeps adjacent
    variable-sized JSON rows unambiguous; one row is the largest live buffer.
    """
    digest = hashlib.sha256(QUALIFICATION_CELLS_SCHEMA.encode() + b"\n")
    for name, fmt in sorted(cells):
        cell = cells[name, fmt]
        row = json.dumps((name, fmt, {
            'anchor': cell['anchor'], 'record': cell['record'],
            'render': cell['render'], 'wire': cell['wire'],
            'render_origin': cell['render_origin'],
        }), sort_keys=True, separators=(',', ':'), ensure_ascii=False,
            allow_nan=False).encode('utf-8')
        digest.update(len(row).to_bytes(8, 'big'))
        digest.update(row)
    return digest.hexdigest()


def prepare_cache(runner, data, *, capture, max_render_bytes, reader=None, file_load_workers=4,
                  qualification_window=None, capture_load_policy=None,
                  source_capture_compatibility=None, source_authentication=None,
                  qualification_guard=None, qualification_journal=None,
                  qualification_resume=False, qualification_identity=None,
                  prewarm_phase_starts=None, sealed_replay=None,
                  prewarm_phases=None, progress_base=0):
    """Qualify original per-layer inputs and return the existing PWC object.

    Only the original calibration/PWC/source prefetch mechanisms own tensors.
    PWC's LRU records absolute donor paths so compact/release stays reversible
    even though the merged renders have more than one original directory.

    ``progress_base`` is the cumulative count the caller's earlier stages have
    already reported -- the measured anchor intake's resolved units. Replay and
    the layer walk continue from it, so the run's counter is monotone across
    every phase it declares (RobTand/prismaquant#678).
    """
    import torch
    from contextlib import nullcontext
    from . import tessera_calibration_cache as cc, tessera_hessian as th, tessera_campaign as tc
    from .joint_aura import activation_identity, prefetch_joint_cache
    from .joint_prewarm_phases import HEAD_PHASE, phase_name
    from .memory_management import reserve_allocation
    from .production_weight_cache import ProductionWeightCache
    from .routed_experts import PackedExpertProjection, refresh_packed_expert_projections
    from . import format_registry as fr

    _require(type(max_render_bytes) is int and max_render_bytes > 0, "positive PWC residency budget required")
    _require(type(progress_base) is int and progress_base >= 0,
             "progress_base must be the non-negative count the run already reported")
    policy = normalize_qualification_window(qualification_window)
    from .perturbed_x_cache import normalize_verified_activation_load
    capture_load_policy = normalize_verified_activation_load(capture_load_policy)
    if capture_load_policy is not None:
        _require(policy is not None, 'verified capture loading requires explicit qualification windows')
    guard = qualification_guard
    if policy is not None and str(runner.device).startswith('cuda'):
        import os
        from .autoscale import require_bounded_capture_environment
        from .memory_management import CaptureMemoryGuard
        require_bounded_capture_environment(os.environ)
        if guard is None:
            guard = CaptureMemoryGuard(runner.device)
        guard.check('before_joint_qualification_identity')
    capture_path = _bound(capture, "canonical capture")
    stamped_capture = data.payload["provenance"].get("calibration_cache")
    _same(capture, stamped_capture, "priced canonical capture")
    manifest = cc.require_capture_contract(capture_path, expected_sha256=capture["sha256"])
    from .glm_capture_compatibility import require_capture_compatibility
    require_capture_compatibility(source_capture_compatibility, capture=capture, model=runner.model)
    recorded = manifest["identity"]
    # This verifies recorded canonical capture provenance plus current source
    # bytes/runtime. It does not pretend a from-config streaming skeleton is an
    # ordinary from_pretrained model or manufacture an initialization witness.
    expected = cc.capture_identity(data.inputs["census"]["path"],
        calibration=data.payload["provenance"]["hessian"]["calibration_identity"],
        max_act_rows=recorded["max_act_rows"],
        model_load_contract=data.census["model_load_contract"],
        attention_implementation=data.census["attention_implementation"],
        **({'source_authentication': source_authentication}
           if source_authentication is not None else {}),
        **(dict(resource_check=None if guard is None else guard.check,
                release_read_pages=True) if policy is not None else {}))
    _same(expected, recorded, "current source/canonical capture")
    _same(data.manifest["identity"]["calibration"], recorded["calibration"], "journal/canonical draw")
    metadata_owner = (None if policy is None else cc.open_capture_metadata(
        capture_path, expected_identity=expected, expected_sha256=capture['sha256']))
    capture_sizes = (None if policy is None else
                     _qualification_capture_sizes(data, expected, policy))
    _require(qualification_journal is None or
             (policy is not None and isinstance(qualification_identity, dict)),
             'qualification journal requires explicit windows and run identity')
    journal = journal_sha = None
    completed = {}
    if qualification_journal is not None:
        identity = dict(qualification_identity, schema='prismaquant.joint_qualification_journal.v1',
            inputs=data.inputs, campaign_checkpoint_sha256=data.manifest['identity_sha256'],
            capture=capture, capture_identity=expected,
            cells_digest_schema=QUALIFICATION_CELLS_SCHEMA,
            cell_count=len(data.cells), cells_sha256=_qualification_cells_sha256(data.cells),
            qualification_window=policy, capture_load_policy=capture_load_policy,
            source_capture_compatibility=source_capture_compatibility,
            max_render_bytes=max_render_bytes)
        journal, journal_sha, completed = prepare_journal(
            qualification_journal, stage=QUALIFICATION_STAGE,
            resume=qualification_resume, identity=identity,
            qnames=sorted(data.formats_by_qname))
        if sealed_replay is not None:
            from .joint_replay_frontier import ROSTER_KEY, require_replay_matches
            _require(prewarm_phase_starts is not None,
                     "a sealed replay frontier travels with its sealed phase table")
            # Before a single replayed byte is read: a journal, checkpoint or
            # unit set that moved after submission describes reads this action
            # will not make, and PB would release bytes nothing consumed.
            require_replay_matches(
                sealed_replay, completed=completed,
                cells_by_unit={name: [fmt for fmt in data.formats_by_qname.get(name, ())
                                      if fmt != "BF16"]
                               for name in sealed_replay[ROSTER_KEY]},
                journal_identity_sha256=journal_sha)
        else:
            # No sealed phases, no phase is reported, so a resume that reports
            # nothing stays as honest as it was before #607. A resume that
            # *does* carry a sealed phase table has to carry the frontier those
            # phases were derived from.
            _require(prewarm_phase_starts is None or not completed,
                     "a resumed pass was submitted with sealed phases but "
                     "without its sealed replay frontier; re-submit so the "
                     "read order and the journal identity are bound before "
                     "the run reads anything")
        if completed:
            replayed = _qualification_replay(data, manifest, completed,
                                             sealed=sealed_replay,
                                             committed=progress_base)
        else:
            replayed = {}
    _require(prewarm_phase_starts is None or journal is not None,
             "ARC read-frontier progress requires a durable qualification journal")
    capture_load_execution = cc._load_execution(capture_load_policy, expected)
    if capture_load_policy is not None:
        cc.preflight_verified_capture_entries(capture_path.parent, manifest['entries'],
            names=sorted(data.formats_by_qname), policy=capture_load_policy,
            census=data.census, max_rows=expected['max_act_rows'])
    maxima, scales = calibrated_maxima(data, runner.profile)
    cache = ProductionWeightCache(
        weights={pair: cell["render"] for pair, cell in data.cells.items()},
        levers={"tessera_campaign": True}, activation_max_abs=maxima,
        metadata={"schema": PREPARED_SCHEMA, "inputs": data.inputs,
                  "reader_identity": None if reader is None else reader.identity})
    cache.enable_lru(max_render_bytes)
    max_file_bytes = _prepare_file_read_bound(data, max_render_bytes=(max_render_bytes
        if policy is None else min(max_render_bytes, policy["max_load_buffer_bytes"])))
    max_wire_read_bytes = _prepare_wire_read_bound(data)
    cache.enable_file_load_receipts(max_file_bytes=max_file_bytes)
    targets = _live_targets(runner, data.formats_by_qname)
    if prewarm_phase_starts is not None:
        _same(set(prewarm_phase_starts), set(targets),
              "sealed prewarm phase/qualification roster")
        _require(prewarm_phases is not None,
                 "sealed prewarm phases require the manifest's own phase names")
    layers = defaultdict(list)
    for name in targets:
        layers[runner.layer_index_for_qname(name)].append(name)
    # The layers this source can read, in walk order: every body layer, or a
    # source scope's own (``SelectedSource.source_layers``, PQ #1338). The
    # prefetch window and its settlement run over this walk, so a scope never
    # installs, prefetches or settles a layer it does not hold.
    walk = tuple(runner.source_layers)
    _require(walk and set(layers) <= set(walk), "census units lie outside the source's layers")
    projected = {name: unit for units in (data.census.get("expert_projection") or {}).get("stacks", {}).values()
                 for name, unit in units.items()}
    renders = {name: tuple(fmt for fmt in fmts if fmt != "BF16")
               for name, fmts in data.formats_by_qname.items()}
    verified, telemetry = replayed if completed else {}, []
    committed_units = progress_base + len(completed)
    current_prewarm_phase = HEAD_PHASE

    def layer_phase(layer, walk_names):
        """The sealed phase this layer's reads belong to, or ``None``.

        Installing a layer and scheduling its prefetch is what reads the phase
        it declares: the source extents of this layer and of the ones the
        prefetch window opens. That phase is ``phase_name(layer, 0)``, and a
        layer whose units are ALL replayed still has one -- but no unit starts
        in it, so the unit table alone cannot name it. When the manifest does
        not declare it, nothing is read for this layer that a phase describes
        and a unit-driven lookup is the right answer instead.
        """
        if prewarm_phases is not None:
            candidate = phase_name(layer, 0)
            if candidate in prewarm_phases:
                return candidate
        if walk_names:
            return prewarm_phase_starts[walk_names[0]]
        return None

    if prewarm_phase_starts is not None:
        # A replayed unit was announced by its replay phase inside
        # ``_qualification_replay``, before the walk. Layer 0's phase comes
        # first here because the prefetch window this walk opens before any
        # install reads exactly the source extents that phase declares -- even
        # when every one of the layer's own units is already qualified.
        first = [name for name in sorted(layers.get(walk[0], ())) if name not in completed]
        phase = layer_phase(walk[0], first)
        if phase is not None:
            current_prewarm_phase = phase
            _pb_commit(committed_units, phase, unit=first[0] if first else None)
    for depth in walk[:runner.prefetch_lookahead + 1]:
        runner.context.schedule_prefetch(depth)
    for position, layer in enumerate(walk):
        names = sorted(layers.get(layer, ()))
        walk_names = [name for name in names if name not in completed]
        if prewarm_phase_starts is not None:
            phase = layer_phase(layer, walk_names)
            if phase is not None and phase != current_prewarm_phase:
                # The previous phase is read to its end, so everything in front
                # of this one has been consumed: PB releases only that prefix.
                _pb_commit(committed_units, phase, unit=walk_names[0] if walk_names else None)
                current_prewarm_phase = phase
        runner.context.install(layer, require_prefetched=runner.require_prefetched_residency)
        ahead = position + runner.prefetch_lookahead
        # Past the walk's end this is ``layer + lookahead``, which the context
        # ignores as it always has, so the body's call sequence is unchanged.
        runner.context.schedule_prefetch(walk[ahead] if ahead < len(walk)
                                         else layer + runner.prefetch_lookahead)
        members = [targets[name] for name in names if isinstance(targets[name], PackedExpertProjection)]
        try:
            targets.update({member.qname: member for member in refresh_packed_expert_projections(members, runner.profile)})
            if not names:
                continue
            if policy is not None:
                runner.context.settle_prefetched_layers(
                    walk[position + 1:position + 1 + runner.prefetch_lookahead])
            capture_windows = [names] if policy is None else [(name,) for name in names]
            layer_stats = []
            for unit_names in capture_windows:
                if len(unit_names) == 1 and unit_names[0] in completed:
                    # Replayed before the walk: its bytes are authenticated and
                    # its replay phase was announced there, so this unit draws
                    # no transition here.
                    name = unit_names[0]
                    layer_stats.extend(completed[name]['prefetch'])
                    if capture_load_execution is not None:
                        partial = completed[name].get('capture_load_execution')
                        _require(isinstance(partial, dict), f'{name}: missing capture load execution')
                        cc.merge_load_execution(capture_load_execution, partial)
                    continue
                if prewarm_phase_starts is not None:
                    phase = prewarm_phase_starts.get(unit_names[0])
                    _require(phase is not None, f"{unit_names[0]} has no sealed prewarm phase")
                    if phase != current_prewarm_phase:
                        # The previous unit has already been journaled. This
                        # phase names bytes we are ABOUT to read; PB releases
                        # only the preceding manifest prefix.
                        _pb_commit(committed_units, phase, unit=unit_names[0])
                        current_prewarm_phase = phase
                stats_start = len(layer_stats)
                acts = hessians = calibration_source = source_weight = None
                resident = rendered = bound_unit = None
                unit_load_execution = {}
                try:
                    if guard is not None:
                        # TWO BUDGETS, ONE UNIT'S FUTURE. The unit's capture
                        # payload, its serialized load buffers, the wire blobs
                        # read ahead of the render and the verified-load scratch
                        # are bytes THIS PROCESS holds, so they are charged to
                        # the cgroup cap the kernel enforces. The unit's X/H
                        # (moved to the device by the capture prefetch), the
                        # render being verified and the projection's workspace
                        # are device bytes, charged to the plan's envelope. The
                        # render is charged to BOTH on purpose and is the one
                        # term this adds to the old mixed number: PWC's backing
                        # storages are the cgroup's, and the tensor handed to the
                        # verifier is a copy of it on the device, so a bound that
                        # named the render once was counting one of the two.
                        # Everything else is the same total, re-labelled:
                        # adding the device half to the 21 GiB cap is what
                        # refused the row's first unit.
                        capture_bytes = capture_sizes[unit_names[0]]
                        reserve_allocation(
                            guard.check,
                            'before_joint_qualification_unit:' + unit_names[0],
                            cpu_bytes=(
                                capture_bytes + max_render_bytes +
                                policy['max_load_buffer_bytes'] +
                                2 * max_wire_read_bytes +
                                (0 if capture_load_policy is None else
                                 2 * capture_load_policy['max_buffer_bytes'] +
                                 capture_load_policy['max_scratch_bytes'])),
                            device_bytes=(
                                capture_bytes + max_render_bytes +
                                policy['workspace_reserve_bytes']))
                    (acts, hessians, _counts, _maxima), _receipt = cc.prefetch_capture(capture_path,
                        expected_sha256=capture["sha256"],
                        **({"expected_identity": expected} if metadata_owner is None else
                           {"metadata_owner": metadata_owner}),
                        census=data.census, names=unit_names, device=runner.device,
                        **(dict(resource_check=None if guard is None else guard.check,
                                release_file_pages=True) if policy is not None else {}),
                        **(dict(verified_load_policy=capture_load_policy,
                                load_execution=unit_load_execution) if capture_load_policy is not None else {}))
                    if capture_load_execution is not None:
                        cc.merge_load_execution(capture_load_execution, unit_load_execution)
                    calibration_source = th.activation_source(hessians, expected["calibration"])
                    if policy is None:
                        layer_stats.append(prefetch_joint_cache(cache, unit_names, renders,
                            max_resident_bytes=max_render_bytes, max_workers=file_load_workers))
                    for name in unit_names:
                        source_weight = targets[name].weight.detach()
                        anchors = [tc.CampaignAnchor(**data.cells[name, fmt]["anchor"]) for fmt in renders[name]]
                        keys = tuple((name, fmt) for fmt in renders[name])
                        # ONE WINDOW, THE BYTES THE GUARD ALREADY RESERVED.
                        # The reservation above prices this unit's residency at
                        # ``max_render_bytes`` and its serialized load buffers at
                        # ``policy['max_load_buffer_bytes']``, as two separate
                        # terms; nothing it charges scales with how many keys a
                        # quantum holds (the capture payload is per unit, and the
                        # wire reader is one lookahead, hence its fixed 2x). So a
                        # quantum planned against exactly those two budgets spends
                        # bytes this unit was already admitted -- or refused -- for,
                        # and the planner splits before it can exceed either one.
                        # The old ``min()`` charged the serialized buffer cap
                        # against residency, which is the wrong budget for it and,
                        # with the loader-count cap, left a real quantum at 12.5%
                        # of the residency it holds (#693).
                        windows = ((keys,) if policy is None else cache.plan_resident_windows(keys,
                            max_resident_bytes=max_render_bytes,
                            max_load_buffer_bytes=policy['max_load_buffer_bytes'],
                            max_workers=file_load_workers))
                        with tc.bind_checkpoint_unit_identity(anchors, source_weight=source_weight,
                                calibration_source=calibration_source, projected_unit=projected.get(name),
                                static_scales=scales) as bound_unit:
                            for window in windows:
                                owner = (nullcontext() if policy is None else cache.resident_window(window,
                                    max_resident_bytes=max_render_bytes, max_workers=file_load_workers,
                                    max_load_buffer_bytes=policy['max_load_buffer_bytes'], release_file_pages=True))
                                with owner as window_receipt:
                                    if window_receipt is not None:
                                        layer_stats.append(dict(unit=name, **window_receipt))
                                    # One ephemeral wire blob is read ahead while the GPU
                                    # verifies its predecessor.  Keep its worker inside the
                                    # resident window so an exception waits for/cancels it
                                    # before PWC releases the tensors it could overlap.
                                    from concurrent.futures import ThreadPoolExecutor
                                    wire_reader = ThreadPoolExecutor(max_workers=1,
                                                                     thread_name_prefix="joint-wire-read")
                                    pending = None
                                    try:
                                        for index, (_, fmt) in enumerate(window):
                                            cell = data.cells[name, fmt]
                                            if pending is None:
                                                pending = wire_reader.submit(_read_verified_wire_blob, cell)
                                            # Consume before borrowing the resident tensor: a
                                            # failed wire never leaves a background read after
                                            # this window's PWC entries have been released.
                                            blob, wire_sha256 = pending.result()
                                            pending = (None if index + 1 == len(window) else
                                                       wire_reader.submit(_read_verified_wire_blob,
                                                                          data.cells[name, window[index + 1][1]]))
                                            resident = (cache.get(name, fmt) if policy is None else cache.get_resident(name, fmt))
                                            receipt = cache.file_load_receipt((name, fmt), resident)
                                            if "render_file_sha256" in cell:
                                                _same(receipt["sha256"], cell["render_file_sha256"], f"{name}: original render file changed")
                                            cell["render_file_sha256"] = receipt["sha256"]
                                            rendered = resident.to(runner.device)
                                            record = verify_anchor_render(cell, source_weight, rendered,
                                                calibration_source=calibration_source,
                                                projected_unit=projected.get(name), static_scales=scales,
                                                bound_unit=bound_unit, reader=reader, wire_blob=blob,
                                                wire_sha256=wire_sha256,
                                                # ``MeasuredAnchorInput`` always carries this;
                                                # a caller that hands this seam a duck-typed
                                                # roster is treated as naming no reuse, which
                                                # fails closed in ``verify_cached_unit`` rather
                                                # than admitting a seal nothing named.
                                                encoder_source_reuse=getattr(
                                                    data, "encoder_source_reuse", None),
                                                **({'release_file_pages': True} if policy is not None else {}))
                                            activation = activation_identity(fr.get_format(fmt), cache.activation_max_abs, name)
                                            _same(activation["input_global_scale"], cell["anchor"].get("input_global_scale"),
                                                  f"{name}@{fmt}: joint/campaign static scale")
                                            record["activation"] = activation
                                            verified[name, fmt] = record
                                            resident = rendered = blob = None
                                    finally:
                                        if pending is not None:
                                            pending.cancel()
                                        wire_reader.shutdown(wait=True, cancel_futures=True)
                                    if guard is not None:
                                        guard.check('after_joint_qualification_window:' + name)
                finally:
                    acts = hessians = calibration_source = source_weight = None
                    resident = rendered = bound_unit = None
                if guard is not None:
                    guard.check('after_joint_qualification_unit:' + unit_names[0])
                if journal is not None:
                    name = unit_names[0]
                    write_unit(journal, stage=QUALIFICATION_STAGE, qname=name,
                        identity_sha256=journal_sha,
                        state={'verified_cells': {fmt: verified[name, fmt] for fmt in renders[name]},
                               'prefetch': layer_stats[stats_start:],
                               **({'capture_load_execution': unit_load_execution}
                                  if capture_load_execution is not None else {})})
                    committed_units += 1
                    _pb_commit(committed_units, current_prewarm_phase if
                               prewarm_phase_starts is not None else 'qualification', unit=name)
            stats = layer_stats[0] if policy is None else {'windows': layer_stats}
            telemetry.append({"layer": layer, **stats})
            print(json.dumps({"qualified_layer": layer, "qualified_cells": len(verified),
                              "total_cells": len(data.cells), "prefetch": stats}), flush=True)
        finally:
            cache.compact_for_pickle()
            runner.context.unload(layer)
            targets.update({member.qname: member for member in refresh_packed_expert_projections(members, runner.profile)})
    _same(set(verified), set(data.cells), "complete qualified wire/render roster")
    cache.disable_file_load_receipts()
    census_of_renders = cell_render_census(data.cells)
    for pair, record in verified.items():
        _same(record["render_origin"], data.cells[pair]["render_origin"],
              f"{pair}: qualified render origin")
    _same(render_origin_census(record["render_origin"] for record in verified.values()),
          census_of_renders, "qualified render origin census")
    cache.metadata.update({"verified_cells": verified, "prefetch": telemetry,
        **census_of_renders,
        **({'capture_load_execution': capture_load_execution} if capture_load_execution is not None else {}),
        **({"qualification_window": policy, "capture_resident_bytes": capture_sizes,
            "qualification_memory_guard": None if guard is None else guard.snapshot()}
           if policy is not None else {})})
    return cache


def _source_prefetch(config):
    prefetch = config.get("source_prefetch")
    fields = {"max_cache_slots", "prefetch_workers", "prefetch_lookahead",
              "cache_headroom_gb", "prefetch_min_available_gb",
              "require_prefetched_residency"}
    _require(isinstance(prefetch, dict) and set(prefetch) == fields,
             "explicit complete source_prefetch settings required")
    _require(prefetch["require_prefetched_residency"] is True,
             "source_prefetch must require prefetched residency")
    for name in ("max_cache_slots", "prefetch_workers", "prefetch_lookahead"):
        _require(type(prefetch[name]) is int and prefetch[name] > 0,
                 f"source_prefetch requires positive {name}")
    _require(prefetch["prefetch_lookahead"] < prefetch["max_cache_slots"],
             "source_prefetch lookahead must fit the declared cache slots")
    for name in ("cache_headroom_gb", "prefetch_min_available_gb"):
        _require(type(prefetch[name]) in (int, float) and
                 math.isfinite(prefetch[name]) and prefetch[name] > 0,
                 f"source_prefetch requires positive finite {name}")
    return dict(prefetch)


def _planned_source_window(config):
    """The prefetch note's bound from the sealed plan (PQ #1134), as kwargs.

    A plan with retained operator windows budgets the replay after the head
    phase inside its cap; only its source reserve is the source window. A
    plan without one gives no bound and the note reads the measured budget.
    """
    from .joint_retained_window_plan import planned_source_window_bytes
    try:
        window = planned_source_window_bytes(
            config.get('execution', {}).get('retained_operator_windows'))
    except Exception as exc:  # noqa: BLE001 - a log note must never stop a run
        print(f"prefetch note: the plan's source window is unreadable ({exc!r}); "
              f"the note reads the measured budget", flush=True)
        return {}
    return {} if window is None else {'planned_source_window_bytes': window}


def recommend_source_prefetch(*, cache_bytes, layer_bytes, cpu_count,
                              cache_headroom_gb, prefetch_min_available_gb):
    """Derive explicit ``source_prefetch`` numbers from measured budgets.

    The sealed plan still carries the explicit six fields -- nothing here
    changes what ``execute`` admits, and a seal over these numbers keeps the
    exact bytes it has today (PQ #737). What changes is where the numbers
    come from: instead of a pinned ``max_cache_slots: 2 /
    prefetch_workers: 1`` carried across seals, the operator seals the depth
    the measured budget admits -- ``cache_bytes // layer_bytes`` slots and
    up to four readers bounded by CPUs and slots, with the lookahead the
    slot count fits. The headroom and minimum-available floors stay operator
    policy: they are passed through, not derived. The result is validated
    through :func:`_source_prefetch`, so a recommendation that cannot run
    refuses here instead of inside the action.
    """
    for label, value in (("cache_bytes", cache_bytes), ("layer_bytes", layer_bytes),
                         ("cpu_count", cpu_count)):
        _require(type(value) is int and value > 0,
                 f"recommended source_prefetch requires positive {label}")
    slots = max(2, int(cache_bytes // layer_bytes))
    workers = max(1, min(4, slots, int(cpu_count)))
    lookahead = max(1, min(workers, slots - 1))
    recommendation = {
        "max_cache_slots": slots,
        "prefetch_workers": workers,
        "prefetch_lookahead": lookahead,
        "cache_headroom_gb": cache_headroom_gb,
        "prefetch_min_available_gb": prefetch_min_available_gb,
        "require_prefetched_residency": True,
    }
    return _source_prefetch({"source_prefetch": recommendation})


def _operator_window_policy(config):
    from .joint_statistics_replay import normalize_operator_windows
    policy = normalize_operator_windows(config['execution'].get('operator_windows'))
    if policy is not None:
        # The body replays Stage A's exact boundaries. The MTP scope has no
        # chain: its inputs are M1's final-hidden entries (PQ #1353).
        _require(config['execution'].get('boundary_storage') is not None
                 or config.get('source_scope') == 'mtp',
                 'operator-window campaign requires explicit exact boundary storage')
        _require(policy['max_render_resident_bytes'] <= config['max_render_bytes'],
                 'operator-window PWC cap exceeds campaign render admission')
    from .joint_retained_window_plan import normalize_retained_execution
    normalize_retained_execution(config['execution'].get('retained_operator_windows'),
        operator_windows=policy, boundary_storage=config['execution'].get('boundary_storage'))
    return policy


def _admit_candidate_phase(command, config, data, layer_bytes):
    """Keep legacy whole-layer admission; explicit windows admit each donor."""
    policy = _operator_window_policy(config)
    if command == 'run' and policy is not None:
        _prepare_file_read_bound(data, max_render_bytes=min(
            policy['max_render_resident_bytes'], policy['max_load_buffer_bytes']))
    elif command != 'prepare' or config.get('qualification_window') is None:
        _require(max(layer_bytes.values()) <= config['max_render_bytes'],
                 'largest measured candidate layer exceeds explicit PWC budget')
    return policy


def _load_plan(path, digest, *, projection_runtime=True, defer_pool_reads=False):
    """Load and admit a joint anchor plan.

    ``defer_pool_reads`` is for a caller whose readset is not bound yet (a
    Stage B layer quantum, PQ #1024): the plan's other inputs are admitted by
    shape only, and nothing but the plan itself is read. The source identity
    cache is then digest-checked where the caller reads it -- the head
    slice's declared entry, or ``_seed_source_identity_cache`` on the legacy
    walk. The boundary directory is never resolved here in either mode: the
    admission discards it, and its owner resolves it when it opens storage.
    """
    path = _bound({"path": str(path), "sha256": digest}, "joint anchor plan")
    config = json.loads(path.read_text())
    _same(config.get("schema"), SCHEMA, "joint anchor plan schema")
    if config.get("source_identity_cache") is not None:
        binding = config["source_identity_cache"]
        if defer_pool_reads:
            _require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}
                     and isinstance(binding["path"], str)
                     and isinstance(binding["sha256"], str),
                     "source identity cache: independently bound path/SHA256 required")
        else:
            _bound(binding, "source identity cache")
    # A plan that names a historical encoder seal is the only place one may be
    # admitted; the strict default is the same as before this field existed.
    normalize_historical_encoder_reuse(config.get("historical_encoder_reuse"))
    _source_prefetch(config)
    execution = config["execution"]
    from .glm_source_derivative import normalize_source_derivative
    normalize_source_derivative(execution.get('source_derivative'))
    normalize_qualification_window(config.get("qualification_window"))
    _require(config.get("source_scope") is None or
             (type(config["source_scope"]) is str and config["source_scope"]),
             "a plan's source_scope names a profile-declared scope")
    from .perturbed_x_cache import normalize_verified_activation_load
    if normalize_verified_activation_load(config.get('capture_load_policy')) is not None:
        _require(config.get('qualification_window') is not None,
                 'verified capture loading requires explicit qualification windows')
    from .joint_projection_backend import normalize_projection_backend, require_qualified_environment
    selector = normalize_projection_backend(execution.get("projection_backend"))
    if projection_runtime and selector["name"] != "torch":
        # Step 3a loads this plan inside the campaign's own container spec, so
        # the identity read here is the executing image's. Every capture-free
        # axis is compared -- torch, cuda, machine, ATen headers, compiler and
        # the image the launcher stamped. The ``device`` block needs
        # ``torch.cuda.get_device_properties``, which a ``--cpu-only``
        # preflight container does not have, so it is compared only when CUDA
        # is present here and is otherwise refused by the first gate in
        # ``execute`` -- seconds into the pass, before any render is written.
        require_qualified_environment()
    from .cost_streaming import check_boundary_storage
    check_boundary_storage(execution.get("boundary_storage"))
    _operator_window_policy(config)
    if config.get('joint_eval') is not None:
        from .tessera_joint_eval_panel import validate_panel_descriptor
        validate_panel_descriptor(config['joint_eval'],
            n_samples=execution['n_calib_samples'], seqlen=execution['calib_seqlen'],
            artifact_sha256=config['calibration_input']['sha256'])
        _require(config['execution'].get('operator_windows') is not None,
                 'diagnostic joint evaluation requires operator observation windows')
        _require(Path(execution['boundary_storage']['directory']).resolve().is_relative_to(
                 Path(config['output_root']).resolve()),
                 'diagnostic joint boundaries must be owned by pilot output root')
    _require(type(config.get("file_hash_workers", 1)) is int and config.get("file_hash_workers", 1) > 0,
             "positive file_hash_workers required")
    for name, minimum in (("n_calib_samples", 1), ("calib_seqlen", 1),
                          ("probe_microbatch", 1), ("n_probes", 2)):
        _require(type(execution.get(name)) is int and execution[name] >= minimum,
                 f"explicit positive {name} required")
    _require(type(execution.get("seed_base")) is int, "explicit probe seed required")
    _same(execution.get("token_scope"), "all", "full-draw joint token scope")
    _same(execution.get("temperature"), 1.0, "joint probe temperature")
    _same(execution.get("production_act_scales"), "0", "campaign optional activation clipping")
    _require(config.get("profile_tool") in {"cprofile", "py-spy"},
             "explicit supported full-duration profiler required")
    for name in ("max_render_bytes", "max_gpu_bytes"):
        _require(type(config.get(name)) is int and config[name] > 0, f"positive {name} required")
    _require(type(config.get("min_free_gib")) in (int, float) and config["min_free_gib"] >= 0,
             "nonnegative memory floor required")
    return config


def _io_counters():
    from .io_spans import read_proc_io

    return read_proc_io()


ACTIVATION_SCALE_ENV = "PRISMAQUANT_PROD_ACT_SCALES"


def _prepare_source_owner(config, data, *, resource_check=None):
    """Bind the complete capture before any streamed source tensor is read.

    The selected-source descriptor owner already authenticates a shard once,
    at its first payload read, and checks replacement for the whole lease.
    Preparation uses that same owner for its full roster: the late completion
    gate authenticates any source file no streamed layer needed.
    """
    from . import tessera_calibration_cache as cc

    capture = config["canonical_capture"]
    manifest = cc.require_capture_contract(capture["path"],
                                           expected_sha256=capture["sha256"])
    owner = cc.authenticate_selected_capture_source(
        config["inputs"]["census"]["path"], capture["path"],
        expected_sha256=capture["sha256"], model=config["model"],
        max_act_rows=manifest["identity"]["max_act_rows"],
        attention_implementation=data.census["attention_implementation"],
        resource_check=resource_check, release_read_pages=True)
    try:
        identity_cache = Path(config["output_root"]) / "prepare/source-identity.json"
        if identity_cache.is_file():
            adopted = owner.adopt_streamed_identity_cache(identity_cache)
            print(f"tessera_joint_aura: adopted {adopted} full source SHA proofs "
                  f"from {identity_cache}", flush=True)
        return owner
    except BaseException:
        owner.close()
        raise


def _adopt_built_source_identity(owner, identity_cache_path):
    """Give the capture owner the identity proof this pass just wrote.

    ``_prepare_source_owner`` adopts a cache only if one is already in the
    output root. With none bound, ``build_streamed_model_identity`` hashes
    every shard and writes one, and the owner then hashed every shard a second
    time in ``authenticate_complete_source``: two serial passes over about
    640 GB for the GLM-5.3 MTP prepare (PQ #1363). Adoption runs the owner's
    own checks, so a shard that changed since it was hashed still refuses.
    Returns the count adopted, or 0 when a cache was already adopted or none
    was written.
    """
    if (identity_cache_path is None or not Path(identity_cache_path).is_file()
            or owner.adopted_identity_cache_sha256 is not None):
        return 0
    adopted = owner.adopt_streamed_identity_cache(identity_cache_path)
    print(f"tessera_joint_aura: adopted {adopted} full source SHA proofs "
          f"this pass wrote to {identity_cache_path}", flush=True)
    return adopted


def _seed_source_identity_cache(config, root):
    """Carry an explicitly bound old digest record into this pass's cache slot.

    A new output root otherwise makes ``build_streamed_model_identity`` hash
    the whole source again before it can compare the live model. This copies
    only its existing identity JSON, never a weight or a render, and refuses
    any pre-existing different local cache rather than mixing two proofs.
    """
    destination = Path(root) / "source-identity.json"
    binding = config.get("source_identity_cache")
    if binding is None:
        return destination
    source = _bound(binding, "source identity cache")
    if source.resolve() == destination.resolve():
        return destination
    if destination.exists():
        _same(_sha(destination), binding["sha256"],
              "existing output source identity cache")
    else:
        atomic_write_bytes(destination, source.read_bytes())
        _same(_sha(destination), binding["sha256"],
              "seeded source identity cache")
    return destination


#: The prepared-record bindings that name DIGESTS of things a dev iteration
#: legitimately changes: which plan the prepare ran under, and which producer
#: package made it. In dev mode (the default since PQ #1147) these are
#: records -- ``seal_check`` prints them and the run continues -- while every
#: other prepared field (the model identity, calibration, roster, backend,
#: reader) stays a wall even in dev mode: a stale record naming a different
#: measurement is stale whatever the mode.
_DEV_RECORDED_PREPARED_KEYS = ("plan_sha256", "implementation_sha256")


def _prepared_digest_recorded(key, stored, expected):
    """Whether a prepared-record mismatch on ``key`` is recorded instead of gated.

    One reader for both prepared-equality sites (the startup preflight and the
    post-intake loop), so dev mode cannot admit a mismatch in one place that
    the other still refuses. Returns ``False`` in certified mode for every
    field, so each caller raises its own ``_same`` refusal, byte for byte.
    Dev mode (the default since PQ #1147) prints the recorded keys through
    ``seal_check`` and returns ``True``.
    """
    if stored == expected or key not in _DEV_RECORDED_PREPARED_KEYS or not dev_mode_enabled():
        return False
    return not seal_check(f"prepared {key}", expected, stored,
                          where="prepared completion versus the running pass",
                          refusal=lambda: ValueError(f"prepared {key}: identity mismatch"))


def require_prepared_digests(completion, *, plan_sha256, implementation_sha256):
    """Compare a prepared completion's plan and implementation digests.

    For callers outside this module that re-check a prepared completion
    after intake (the Stage B quantum runtime). Certified mode refuses a
    mismatch; dev mode records it through ``_prepared_digest_recorded``, the
    same reader the startup preflight uses.
    """
    for key, value in (("plan_sha256", plan_sha256),
                       ("implementation_sha256", implementation_sha256)):
        if not _prepared_digest_recorded(key, completion.get(key), value):
            _same(completion.get(key), value, f"prepared {key}")


def _preflight_run_prepared(prepared, *, plan_sha256, implementation_sha256,
                           reader_identity, projection_backend):
    """Refuse a stale small completion before reading the campaign metadata.

    Runtime/source/model and exact cell checks still run after input intake;
    this early gate checks only fields already independently known at startup.
    """
    _require(prepared is not None, "cost execution requires independently bound prepared inputs")
    completion = json.loads(_bound(prepared, "prepared anchors").read_text())
    return check_prepared_completion(completion, plan_sha256=plan_sha256,
                                     implementation_sha256=implementation_sha256,
                                     reader_identity=reader_identity,
                                     projection_backend=projection_backend)


def check_prepared_completion(completion, *, plan_sha256, implementation_sha256,
                              reader_identity, projection_backend):
    """The startup gates of :func:`_preflight_run_prepared` on parsed bytes.

    The Stage B head slice (PQ #1010) reads the prepared completion once,
    as a declared head entry, and checks those bytes here instead of
    re-reading the pool path.
    """
    _same(completion.get("schema"), PREPARED_SCHEMA,
          "prepared v3 schema required; legacy preparation requires fresh prepare and recompute")
    _same(completion.get("status"), "complete", "prepared completion")
    for key, value in (("plan_sha256", plan_sha256),
                       ("implementation_sha256", implementation_sha256),
                       ("reader_identity", reader_identity),
                       ("projection_backend", projection_backend)):
        if not _prepared_digest_recorded(key, completion.get(key), value):
            _same(completion.get(key), value, f"prepared {key}")
    return completion


def _config_device_envelope(config, command):
    """The device envelope a joint command declares, read before any device.

    ``max_gpu_bytes`` is what ``_load_plan`` requires of every admitted plan,
    and the envelope is the FIRST thing a command does that reaches the CUDA
    allocator. Reading it with ``config["max_gpu_bytes"]`` therefore turned a
    config the admission gate would have refused into a ``KeyError`` raised
    after the device had already been touched -- indistinguishable, to a
    reader, from a plan that was admitted and failed later. Stating it here
    keeps the cheap input refusal cheap, and keeps the allocator touch to the
    one place that owns it.
    """
    if "max_gpu_bytes" not in config:
        raise ValueError(
            f"joint {command}: the plan declares no max_gpu_bytes; every "
            "admitted plan carries the device envelope its row is bounded by")
    return config["max_gpu_bytes"]


def _apply_device_envelope(device, device_bytes, *, where):
    """Set the CUDA allocator envelope, through the one seam that names it.

    A module-level indirection so the CPU suites that drive ``execute`` for its
    preflight refusals can state the envelope without a device: those tests
    patch ``gpu_guard.require_cuda_hot_path`` because a refusal that happens
    before any allocation must be reachable without CUDA, and the envelope --
    which is the FIRST thing that touches the allocator -- is the same shape of
    seam. On a real box this is
    :func:`prismaquant.memory_management.enforce_device_envelope` unchanged.
    """
    from .memory_management import enforce_device_envelope

    return enforce_device_envelope(device, device_bytes, where=where)


def _restores_activation_scale_env(function):
    """Scope ``execute``'s activation-scale write to the call that makes it.

    ``execute`` sets ``PRISMAQUANT_PROD_ACT_SCALES`` from the admitted plan so
    the render path it drives reads the campaign's value.  As a process entry
    point that is right; called in-process it leaves the value behind.  Every
    admitted plan carries ``"0"`` (``_load_plan``), and that is the input
    which turns the render scorer's activation clip OFF for everything that
    runs afterwards (``production_weight_cache.py``, in
    ``_local_forward_render_score``).  Plenty of code outside ``execute``
    reads the key -- the render scorer is exactly that code, which is why the
    leak bites -- but nothing needs THIS command's value to still be set after
    ``execute`` has returned.  So restoring it on the way out leaves the
    campaign byte-identical and leaves the process as it was found.
    """
    absent = object()

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        import os

        prior = os.environ.get(ACTIVATION_SCALE_ENV, absent)
        try:
            return function(*args, **kwargs)
        finally:
            if prior is absent:
                os.environ.pop(ACTIVATION_SCALE_ENV, None)
            else:
                os.environ[ACTIVATION_SCALE_ENV] = prior

    return wrapper


@_restores_activation_scale_env
def execute(command, config, *, plan_sha256, prepared=None, resume=False,
            source_transition=None, prewarm_manifest=None, cost_read_manifest=None, plan_path=None,
            data_manifest_sha256=None):
    """Execute one admitted preparation or one dependent cost action."""
    if source_transition is not None:
        from .joint_aura_transitions import load_transition
        _require(command == "run" and resume, "source transition requires run --resume")
        source_transition = load_transition(
            source_transition, config=config, plan_sha256=plan_sha256,
            prepared=prepared, checkpoint_dir=Path(config["output_root"]) / "checkpoints",
        )
    import cProfile
    import io
    import os
    import pstats
    import socket
    import torch
    from .aura_cost import compute_aura_cost_streamed, _aura_source_sha256
    from .calibration_data import load_calibration_input
    from .cost_streaming import build_streamed_causal_lm, build_streamed_model_identity
    from .joint_aura import source_execution_identity, validate_joint_aura_entry
    from .joint_projection_backend import executing_image, prewarm_projection_backend
    from .model_profiles import detect_profile
    from .production_weight_cache import ProductionWeightCache
    from .gpu_guard import require_cuda_hot_path
    from .joint_prewarm_phases import HEAD_PHASE
    from .tessera_reader import load_declared_reader

    require_cuda_hot_path("tessera_joint_aura", "cuda")
    # THE ENVELOPE'S OWN CONFIG IS READ HERE, as a pure input refusal: a plan
    # with no ``max_gpu_bytes`` is refused, by name, before this process has
    # touched the allocator -- read with a subscript it was a ``KeyError``
    # raised after a device allocation, indistinguishable from an admitted
    # plan that failed later. The value's own validation (a positive byte
    # count) and the allocator touch both belong to ``_apply_device_envelope``
    # further down. ``synthesize`` is the one CPU command and never reaches
    # this function.
    #
    # WHAT IS *NOT* PROMISED: the prepared-completion preflight compares the
    # record against the prewarmed projection backend's identity, so it cannot
    # run before the prewarm, and the prewarm is what allocates. A stale
    # prepared record therefore refuses just after the envelope is set rather
    # than before it. Both the pure refusal and the ordering are measured on a
    # CPU-only box, with no mocked device, by
    # ``test_a_missing_device_envelope_is_a_pure_input_refusal`` and
    # ``test_the_declared_envelope_reaches_the_allocator_unchanged``.
    declared_device_bytes = _config_device_envelope(config, command)
    if (config.get("qualification_window") is not None
            or config["execution"].get("retained_operator_windows") is not None):
        from .autoscale import require_bounded_capture_environment
        require_bounded_capture_environment(os.environ)
    os.environ[ACTIVATION_SCALE_ENV] = config["execution"]["production_act_scales"]
    torch.set_num_threads(1)
    from .matmul_arithmetic import pin_matmul_arithmetic
    pin_matmul_arithmetic()
    execution = config["execution"]
    root = Path(config["output_root"]) / command
    root.mkdir(parents=True, exist_ok=True)
    device_envelope = None
    prewarm_phase_starts = None
    sealed_replay = None
    prewarm_phases = None
    if prewarm_manifest is not None:
        _require(command == "prepare" and config.get("qualification_window") is not None,
                 "sealed prewarm phases require windowed preparation")
        from .joint_prewarm_phases import load_prepare_read_set
        prewarm_phase_starts, sealed_replay, prewarm_phases = load_prepare_read_set(
            prewarm_manifest["path"], prewarm_manifest["sha256"], plan_sha256)
    cost_schedule = None
    if cost_read_manifest is not None:
        _require(command == 'run' and execution.get('retained_operator_windows') is not None
                 and prepared is not None and plan_path is not None,
                 'COST read schedule requires a retained run and exact plan/prepared paths')
        from .joint_cost_read_schedule import load_joint_cost_read_schedule
        from .joint_retained_window_plan import RetainedWindowBudget
        retained = execution['retained_operator_windows']
        cost_schedule = load_joint_cost_read_schedule(
            manifest_path=cost_read_manifest['path'], manifest_sha256=cost_read_manifest['sha256'],
            manifest_bytes=cost_read_manifest['bytes'], plan_path=str(plan_path), plan_sha256=plan_sha256,
            prepared_path=prepared['path'], prepared_sha256=prepared['sha256'],
            retained_budget=RetainedWindowBudget.from_dict(retained['budget']),
            source_owner_cap_bytes=retained['source_reserve_bytes'],
            n_probes=execution['n_probes'],
            progress_callback=lambda phase, units: _pb_commit(units, phase))
        cost_schedule.enter_phase('cost_setup', 0)
    # Bind the read set this pass was submitted with, so PrismaBuild's stage
    # tier can only answer for these bytes. A residency map declares the
    # manifest it was composed for; one naming a different manifest is refused
    # whole and recorded. A pass that seals no manifest binds nothing and gets
    # no redirect, which is the same behaviour as having no stage at all.
    # A pass that seals a read schedule already names its manifest; one that
    # does not (a fresh prepare, or a run whose schedule is not sealed) is
    # told the digest explicitly by the submitter that asked for the stage.
    bind_residency_manifest(
        data_manifest_sha256
        or (cost_read_manifest or prewarm_manifest or {}).get('sha256'))
    identity_cache_path = _seed_source_identity_cache(config, root)
    result = {"schema": "prismaquant.tessera_joint_aura.execution.v1", "command": command,
              "plan_sha256": plan_sha256, "env": {"host": socket.gethostname(),
                  "started_epoch": time.time(), "torch": str(torch.__version__),
                  "cuda": torch.version.cuda, "affinity": sorted(os.sched_getaffinity(0))},
              "phases": [], "passed": False, "device_envelope": device_envelope}
    if dev_mode_enabled():
        # The dev stamp is TOP LEVEL and lands on every results.json this
        # command writes, including a refusal path's -- the finally block
        # below publishes it whatever happened above. A dev result is
        # identifiable at a glance and grep-able, and can never masquerade
        # as a certified one (Rob, 2026-09-19: the seal returns at the
        # artifact gate, not the run gate).
        result.update(dev_stamp(_aura_source_sha256()))
    result["env"]["container_content_sha256"] = executing_image()
    profile_tool = config.get("profile_tool", "cprofile")
    profiler = cProfile.Profile() if profile_tool == "cprofile" else None
    result["profile_tool"] = profile_tool
    if profiler is None:
        session_path = Path(os.environ.get("PRISMAQUANT_SAMPLER_SESSION", ""))
        _require(session_path.is_file(), "sampling must run through the checked profiler launcher")
        session_bytes = session_path.read_bytes()
        session = json.loads(session_bytes)
        _same(session.get("schema"), "prismaquant.profiled_command_start.v1", "sampler session schema")
        _same(session.get("wrapper_pid"), os.getppid(), "actual sampler child parent")
        _same(session.get("command", [])[1:4],
              ["-m", "prismaquant.tessera_joint_aura", command], "observed joint command")
        result["sampling_session"] = {"path": str(session_path),
                                      "sha256": hashlib.sha256(session_bytes).hexdigest()}
    runner = source_authentication = qualification_guard = None
    completion_path = completion = output = payload = None
    started, before_io = time.time(), _io_counters()
    if profiler is not None:
        profiler.enable()
    try:
        file_hash_workers = config.get("file_hash_workers", 1)
        _require(type(file_hash_workers) is int and 0 < file_hash_workers <= len(os.sched_getaffinity(0)),
                 "file_hash_workers exceeds PB-assigned CPU affinity")
        # Every capture-free identity gate runs first: an unqualified runtime,
        # kernel source digest, build flag or binary sha256 is refused in
        # seconds rather than after hours of measured anchor input (#553).
        # The reader is bound first of the input owners: synthesizing an
        # adopted rung's missing render decodes its wire, and that decode must
        # come from the same bound consumer the qualification leg uses, not a
        # second one.
        reader = load_declared_reader(config.get("reader"))
        reader_identity = None if reader is None else reader.identity
        implementation = (_aura_source_sha256() if source_transition is None
                          else source_transition.measurement_source_sha256)
        # THE PLAN DIGEST THE PREPARED RECORD MUST CARRY. Normally the running
        # plan's: a prepared record made against another plan is a stale record.
        # An admitted transition may state another one, and exactly one kind
        # does -- the retained-budget transition, whose own proof holds the two
        # plans byte-identical outside the budget keys its contract enumerates,
        # which is what makes every other prepared field the checks below
        # re-derive from the plan still the same field. The dispatcher answers
        # per capability type from a literal table, so a transition that was
        # never taught this refuses rather than silently reusing the run's.
        prepared_plan_sha256 = plan_sha256
        if source_transition is not None:
            from .joint_aura_transitions import transition_prepared_plan_sha256
            prepared_plan_sha256 = transition_prepared_plan_sha256(
                source_transition, plan_sha256=plan_sha256)
        # THE DEVICE ENVELOPE IS APPLIED HERE, after the refusals that need no
        # device -- the envelope's own config and the declared reader -- and
        # before the first thing that allocates on the device (the projection
        # prewarm below). The prepared-completion preflight is the one refusal
        # that cannot precede it: it compares the record against the prewarmed
        # backend's identity, and the prewarm is what allocates. Both GPU
        # commands take it, because both allocate, and ``max_gpu_bytes`` was
        # otherwise compared with ``max_memory_allocated`` only after a window
        # had run, which on a unified-memory box is a report about memory
        # already spent.
        device_envelope = _apply_device_envelope(
            "cuda", declared_device_bytes, where=f"joint {command}")
        result["device_envelope"] = device_envelope
        projection_backend = prewarm_projection_backend(execution.get("projection_backend"), device="cuda")
        result["projection_backend"] = projection_backend.identity
        if command == "run":
            _preflight_run_prepared(prepared, plan_sha256=prepared_plan_sha256,
                implementation_sha256=implementation, reader_identity=reader_identity,
                projection_backend=projection_backend.identity)
        # The command holds a CUDA reservation (``require_cuda_hot_path``
        # above), so any shard it still has to synthesize decodes on that
        # device rather than on one CPU core beside an idle GPU. The standalone
        # ``synthesize`` stage normally leaves nothing to do here.
        # The head walk reports under the one phase every joint pass manifest
        # declares -- prepare and run both open on ``head``. A run whose read
        # schedule is sealed separately (the V2 cost read plan) declares
        # ``cost_setup``/``cost_head`` instead and no ``head``, so intake there
        # reports nothing rather than under a name that schedule did not
        # declare and the worker would refuse.
        #
        # Why this is not cosmetic: on ``ad8803aa`` the run's head resolved its
        # 512-entry anchor roster between the 12:10:14 claim and the 16:24:17
        # capture line -- 4 h 14 min in which the loop knew its own count at
        # every step and committed none of it, so the residency window had
        # nothing to advance on before the capture had even started.
        data = load_measured_anchor_input(config["inputs"], reader=reader,
            synthesis_device="cuda",
            progress_phase=(None if cost_read_manifest is not None else HEAD_PHASE),
            # The head walk banks its verified units under the command's own
            # root, so a restarted action resumes mid-roster instead of
            # re-paying the whole walk (#754). Per-command because prepare
            # and run must never read each other's journal even under one
            # output root: the stage name binds the manifest, and so does
            # the directory.
            head_checkpoint=root / "head-walk",
            head_resume=resume,
            **({} if file_hash_workers == 1 else {"file_hash_workers": file_hash_workers}),
            **({} if config.get("historical_encoder_reuse") is None else
               {"historical_encoder_reuse": config["historical_encoder_reuse"]}),
            **({"verify_payloads": False} if command == "prepare" else
               {"verify_payloads": False, "require_existing_renders": True}))
        _require(data.unit_scope is None and data.render_mirror_root is None,
                 "joint execution requires the complete campaign roster in its own caches")
        result["file_hash_workers"] = file_hash_workers
        result["head_walk_workers"] = data.head_walk_workers
        result["head_walk_resumed_units"] = data.head_walk_resumed_units
        result["reader_identity"] = reader_identity
        result["encoder_source_reuse"] = data.encoder_source_reuse
        # Per-run, not per-origin: the census says what is on disk, this says
        # how much of it this run had to write.
        result["renders_synthesized_now"] = data.synthesized_now
        render_census = cell_render_census(data.cells)
        # Stated whether or not this command reaches a completion: a run that
        # dies still says how many of its renders were only ever round-tripped.
        result.update(render_census)
        _same(config["model"], data.census["model"], "requested source model")
        _same(data.census["attention_implementation"], "eager", "qualified source attention")
        ids, calibration = load_calibration_input(config["calibration_input"]["path"],
            expected_sha256=config["calibration_input"]["sha256"],
            n_samples=execution["n_calib_samples"], seqlen=execution["calib_seqlen"])
        original_draw = data.payload["provenance"]["hessian"]["calibration_identity"]
        for name in ("fit_ids_sha256", "text_sha256", "nsamples", "seqlen", "seed"):
            _same(calibration["provenance"].get(name), original_draw.get(name), f"original full draw {name}")
        result["calibration_input"] = calibration
        from .tessera_joint_eval_panel import select_panel
        eval_ids, eval_panel = select_panel(ids, calibration, config.get('joint_eval'))
        if eval_panel is not None:
            result['joint_eval'] = eval_panel
        if command == "prepare":
            if config.get("qualification_window") is not None:
                from .memory_management import CaptureMemoryGuard
                # TWO BUDGETS, TWO ENFORCEMENTS. The cgroup cap the spec declares
                # is a CPU-accounted hard limit; the plan's ``max_gpu_bytes`` is
                # the device envelope, already applied above before this process
                # touched the device. The guard holds the first, that cap holds
                # the second, and the aggregate the submission reserved from
                # PrismaBuild is their sum -- so a row that holds 80 GiB of
                # device residency beside a 21 GiB CPU cap is bounded rather
                # than refused by its own arithmetic.
                qualification_guard = CaptureMemoryGuard(
                    "cuda", device_bytes=config["max_gpu_bytes"])
                qualification_guard.check("before_joint_source_authentication")
            source_authentication = _prepare_source_owner(
                config, data, resource_check=(None if qualification_guard is None
                                              else qualification_guard.check))
        source_prefetch = _source_prefetch(config)
        if cost_schedule is not None:
            cost_schedule.enter_phase('cost_head', 0)
        runner = build_streamed_causal_lm(config["model"], device=torch.device("cuda"),
            dtype=torch.bfloat16, offload_folder=str(root / "offload"),
            profile=detect_profile(config["model"]), attn_implementation="eager",
            **({'source_derivative': execution['source_derivative']} if execution.get('source_derivative') is not None else {}),
            **({'source_authentication': source_authentication}
               if source_authentication is not None else {}),
            # A plan over a profile-declared source scope (GLM's MTP layer,
            # PQ #1338) builds the same runner over that scope's layers only.
            **({'source_scope': config['source_scope']}
               if config.get('source_scope') is not None else {}),
            **_planned_source_window(config),
            **source_prefetch)
        from .glm_capture_compatibility import require_capture_compatibility
        require_capture_compatibility(config.get('source_capture_compatibility'),
                                      capture=config['canonical_capture'], model=runner.model)
        result["source_prefetch"] = source_prefetch
        source = build_streamed_model_identity(runner, config["model"],
                                               identity_cache_path=identity_cache_path)
        if source_authentication is not None:
            _adopt_built_source_identity(source_authentication, identity_cache_path)
        source_execution = source_execution_identity(runner.model)
        layer_bytes = data.layer_render_bytes(runner.layer_index_for_qname)
        operator_policy = _admit_candidate_phase(command, config, data, layer_bytes)
        if operator_policy is not None:
            _require(operator_policy['prefetch_workers'] <= len(os.sched_getaffinity(0)),
                     'operator-window prefetch workers exceed PB-assigned CPU affinity')
        result.update(source_model_identity=source, source_execution=source_execution,
                      units=len(data.formats_by_qname), measured_cells=len(data.cells),
                      layer_render_bytes=layer_bytes)
        if source_transition is not None:
            result["source_transition"] = source_transition.execution_provenance
        if command == "prepare":
            _require(prepared is None, "preparation does not consume a prepared cache")
            completion_path = root / "prepared.json"
            _require(not completion_path.exists(), "prepared completion already exists; use its bound record")
            cache = prepare_cache(runner, data, capture=config["canonical_capture"],
                                  max_render_bytes=config["max_render_bytes"], reader=reader,
                                  file_load_workers=file_hash_workers,
                                  qualification_window=config.get("qualification_window"),
                                  source_authentication=source_authentication,
                                  qualification_guard=qualification_guard,
                                  qualification_journal=(root / 'qualification' if
                                                         config.get('qualification_window') is not None else None),
                                  qualification_resume=resume,
                                  prewarm_phase_starts=prewarm_phase_starts,
                                  sealed_replay=sealed_replay,
                                  prewarm_phases=prewarm_phases,
                                  progress_base=data.progress_committed,
                                  qualification_identity={
                                      'plan_sha256': plan_sha256,
                                      'source_model_identity': source,
                                      'source_execution': source_execution,
                                      'implementation_sha256': implementation,
                                      'calibration_input': calibration,
                                      'reader_identity': reader_identity,
                                      'projection_backend': projection_backend.identity},
                                  **({'capture_load_policy': config['capture_load_policy']}
                                     if config.get('capture_load_policy') is not None else {}),
                                  **({'source_capture_compatibility': config['source_capture_compatibility']}
                                     if config.get('source_capture_compatibility') is not None else {}))
            source_receipt = source_authentication.authenticate_complete_source()
            cache.metadata["source_authentication"] = source_receipt
            result["source_authentication"] = source_receipt
            cache.metadata.update(plan_sha256=plan_sha256, source_model_identity=source,
                                  source_execution=source_execution, implementation_sha256=implementation,
                                  projection_backend=projection_backend.identity)
            cache.compact_for_pickle()
            cache_path = root / "production.pkl"
            atomic_write_bytes(cache_path, pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))
            completion = {"schema": PREPARED_SCHEMA, "status": "complete", "plan_sha256": plan_sha256,
                "implementation_sha256": implementation, "source_model_identity": source,
                "reader_identity": reader_identity, "projection_backend": projection_backend.identity,
                "source_execution": source_execution, "calibration_input": calibration,
                "production_cache": {"path": str(cache_path), "sha256": _sha(cache_path)},
                "formats_by_qname": data.formats_by_qname, "measured_cells": len(data.cells),
                "encoder_source_reuse": data.encoder_source_reuse,
                **render_census}
        else:
            _require(prepared is not None, "cost execution requires independently bound prepared inputs")
            completion = json.loads(_bound(prepared, "prepared anchors").read_text())
            _same(completion.get("schema"), PREPARED_SCHEMA,
                  "prepared v3 schema required; legacy preparation requires fresh prepare and recompute")
            _same(completion.get("status"), "complete", "prepared completion")
            for key, value in (("plan_sha256", prepared_plan_sha256), ("implementation_sha256", implementation),
                               ("source_model_identity", source), ("source_execution", source_execution),
                               ("calibration_input", calibration), ("measured_cells", len(data.cells)),
                               ("reader_identity", reader_identity),
                               ("encoder_source_reuse", data.encoder_source_reuse),
                               ("render_origins", render_census["render_origins"]),
                               ("render_comparisons", render_census["render_comparisons"]),
                               ("projection_backend", projection_backend.identity)):
                if not _prepared_digest_recorded(key, completion.get(key), value):
                    _same(completion.get(key), value, f"prepared {key}")
            _same(completion["formats_by_qname"], {n: list(v) for n, v in data.formats_by_qname.items()},
                  "prepared exact candidate roster")
            cache = pickle.loads(_bound(completion["production_cache"], "qualified PWC").read_bytes())
            _require(isinstance(cache, ProductionWeightCache), "prepared cache is not ProductionWeightCache")
            _same(cache.metadata["inputs"], data.inputs, "prepared source bindings")
            _same(cache.metadata.get("reader_identity"), reader_identity, "prepared reader identity")
            _same(cache.metadata.get("projection_backend"), projection_backend.identity, "prepared backend identity")
            _same(set(cache.metadata["verified_cells"]), set(data.cells), "prepared verified cell coverage")
            _same(cache.weights, {pair: cell["render"] for pair, cell in data.cells.items()}, "prepared original render paths")
            for key in ("render_origins", "render_comparisons"):
                _same(cache.metadata.get(key), render_census[key], f"prepared cache {key}")
            for pair, cell in data.cells.items():
                _same(cache.metadata["verified_cells"][pair]["render_origin"], cell["render_origin"],
                      f"{pair}: qualified render origin changed")
                _same(cache.metadata["verified_cells"][pair]["wire_sha256"], cell["record"]["blob_sha256"],
                      f"{pair}: historical qualified wire identity changed")
            expected_renders = {pair: cache.metadata["verified_cells"][pair]["render_file_sha256"]
                                for pair in data.cells}
            # The prepared receipt bound the original serialized shard on
            # PREPARE's necessary load. COST verifies it on its own necessary
            # PWC read, before a changed tensor can reach AURA.
            cache.require_file_load_sha256(expected_renders,
                max_file_bytes=_prepare_file_read_bound(data,
                    max_render_bytes=config["max_render_bytes"]))
            result["wire_validation"] = HISTORICAL_WIRE_VALIDATION
            _live_targets(runner, data.formats_by_qname)
            formats = list(dict.fromkeys(fmt for values in data.formats_by_qname.values() for fmt in values))
            expected_rows = data.formats_by_qname
            if config.get("source_scope") == "mtp":
                # The MTP layer is priced on its own head's self-KL, not the body's
                # end KL (PQ #1353), from the same qualified cache.
                from .glm_mtp_quantum import priced_formats, run_mtp_scope
                payload = run_mtp_scope(runner, config=config, data=data, production_cache=cache,
                    calibration_ids=eval_ids, calibration=calibration, source_model=source,
                    projection_backend=projection_backend, operator_windows=operator_policy,
                    device_bytes=declared_device_bytes)
                expected_rows = {name: priced_formats(fmts) for name, fmts in data.formats_by_qname.items()}
            else:
                payload = compute_aura_cost_streamed(runner, eval_ids.to(runner.device), formats,
                    n_probes=execution["n_probes"], probe_microbatch=execution["probe_microbatch"],
                    seed_base=execution["seed_base"], token_scope="all", temperature=1.0,
                    production_cache=cache, require_production_cache=True, joint_activation=True,
                    cost_read_schedule=cost_schedule,
                    # The count PrismaBuild accepts is cumulative across phases, so
                    # the capture continues from what the head already committed
                    # rather than restarting at zero, which is a regression and
                    # buys no time.
                    progress_base=data.progress_committed,
                    prepared_render_identities={pair: cache.metadata["verified_cells"][pair]["rendered_weight"]
                                                for pair in data.cells},
                    joint_projection_backend=projection_backend,
                    boundary_storage=execution.get("boundary_storage"),
                    **({"operator_windows": operator_policy} if operator_policy is not None else {}),
                    **({"source_transition": source_transition} if source_transition is not None else {}),
                    include_routed_experts=True, include_lm_head=False, dw_dtype="float32",
                    min_free_gib=config["min_free_gib"], formats_by_qname=data.formats_by_qname,
                    checkpoint_dir=Path(config["output_root"]) / "checkpoints", resume=resume,
                    model_identity=source, profile=runner.profile,
                    **({'retained_operator_windows': execution['retained_operator_windows'],
                        'device_envelope_bytes': declared_device_bytes}
                       if execution.get('retained_operator_windows') is not None else {}),
                    checkpoint_identity_extra={"tessera_joint_anchor_plan_sha256": plan_sha256,
                        "prepared_anchor_sha256": prepared["sha256"], "calibration_input": calibration,
                        **({'joint_eval': eval_panel} if eval_panel is not None else {}),
                        "reader_identity": reader_identity})
            _same(set(payload["costs"]), set(expected_rows), "complete joint output roster")
            for name, rows in payload["costs"].items():
                _same(set(rows), set(expected_rows[name]), f"{name}: joint output candidates")
                for row in rows.values():
                    _require(validate_joint_aura_entry(row), f"{name}: invalid measured joint cost")
            payload["provenance"]["tessera_joint_anchors"] = {
                "plan_sha256": plan_sha256, "prepared": prepared, "inputs": data.inputs,
                "calibration_input": calibration, "measured_cells": len(data.cells),
                **({'joint_eval': eval_panel} if eval_panel is not None else {}),
                "wire_validation": HISTORICAL_WIRE_VALIDATION,
                **render_census}
            if eval_panel is not None:
                _same(payload['provenance'].get('joint_eval'), eval_panel,
                      'pilot observation status and evaluation identity')
            output = root / "joint-cost.pkl"
        torch.cuda.synchronize()
        result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        _require(result["peak_gpu_bytes"] <= config["max_gpu_bytes"], "observed GPU allocation exceeds declared budget")
        # A completion is published only after the source residency owner has
        # shut down successfully as well as after the allocation gate passes.
        completed_runner, runner = runner, None
        completed_runner.shutdown()
        if source_authentication is not None:
            source_authentication.close()
            source_authentication = None
        if command == "prepare":
            _json(completion_path, completion)
            result["prepared"] = {"path": str(completion_path), "sha256": _sha(completion_path)}
        else:
            atomic_write_bytes(output, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            result["cost"] = {"path": str(output), "sha256": _sha(output)}
        result["passed"] = True
    finally:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(root / "profile.pstats"))
            text = io.StringIO()
            pstats.Stats(profiler, stream=text).sort_stats("cumulative").print_stats(100)
            (root / "profile.txt").write_text(text.getvalue())
        result["env"]["finished_epoch"] = time.time()
        result["phases"].append({"phase": command, "kind": "profile", "start_epoch": started,
                                 "end_epoch": result["env"]["finished_epoch"]})
        result["io_before"], result["io_after"] = before_io, _io_counters()
        # What PrismaBuild's tiers actually served, beside the process's own
        # read counters: the stage's bytes and, on a map of the ram-overlay
        # generation, the tmpfs's own. A stage with no reader is a copy nobody
        # reads, so this block is the closed loop: hits and bytes per tier when
        # the redirect worked, a named reason for every entry it refused. The
        # key is absent when no map was named, which keeps an unset run's
        # record identical to today's.
        residency = residency_report()
        if residency is not None:
            result["residency"] = residency
        _json(root / "results.json", result)
        try:
            if runner is not None:
                runner.shutdown()
        finally:
            if source_authentication is not None:
                source_authentication.close()
    return result


SYNTHESIS_SCHEMA = "prismaquant.tessera_joint_aura.render_synthesis.v1"


def synthesize_renders(config, *, plan_sha256, units=None, device="cpu", log_every=100,
                       mirror_root=None, authorized=False, receipt=None, compare=False):
    """Write the campaign's missing decoded PWC shards, as an independent quantum.

    The joint ``prepare`` pass has always synthesized an adopted rung's
    missing shard on the way past. Doing it there put an embarrassingly
    parallel, GPU-free, one-decode-per-wire pass on the critical path of a
    reservation it does not use: measured, 125,144 shards at 2.6 cells/s on
    one core while the reserved GB10 sat at 5 W of 140 W (#549).

    This is the same function, addressable on its own: a unit range, no
    model, no capture, no GPU required, and idempotent -- a cell whose shard
    exists is skipped, the origin marker is published before the shard, and
    staging names are unique per writer. PrismaBuild owns the fan-out; rows
    carry disjoint ``sorted(names)[lo:hi]`` ranges cut from the census, so no
    two rows ever address the same cell and a retried row re-reads rather
    than re-writes.

    ``mirror_root`` publishes into a mirror of the render paths instead of
    the campaign's row caches, and ``compare`` then byte-compares each
    mirrored shard against the campaign's own. That is a measurement, not a
    build: writing into the row caches needs ``authorized``.
    """
    from .tessera_reader import load_declared_reader

    _require(bool(mirror_root) or authorized,
             "publishing into the campaign row caches requires explicit authorization")
    _require(mirror_root is not None or not compare,
             "a byte comparison needs a mirror to compare against the campaign's shards")
    census = json.loads(_bound(config["inputs"]["census"], "census").read_text())
    scope = parse_unit_scope(units, len(census["unit_shapes"]))
    reader = load_declared_reader(config.get("reader"))
    started = time.time()
    data = load_measured_anchor_input(config["inputs"], reader=reader, verify_payloads=False,
                                      synthesis_device=device, unit_scope=scope,
                                      render_mirror_root=mirror_root, log_every=log_every,
                                      historical_encoder_reuse=config.get("historical_encoder_reuse"))
    record = {"schema": SYNTHESIS_SCHEMA, "plan_sha256": plan_sha256, "units": units,
              "unit_scope": scope, "device": device, "mirror_root": data.render_mirror_root,
              "host": socket.gethostname(), "pid": os.getpid(),
              "reader_identity": None if reader is None else reader.identity,
              "decoder_source": _decoder_identity(reader),
              "units_read": len(data.formats_by_qname), "cells": len(data.cells),
              "renders_synthesized_now": data.synthesized_now,
              "encoder_source_reuse": data.encoder_source_reuse,
              "seconds": time.time() - started, **cell_render_census(data.cells)}
    if compare:
        record["comparison"] = _compare_mirrored_renders(data)
    if receipt is not None:
        _json(receipt, record)
    return record


def _decoder_identity(reader):
    """Name the decoder that produced these bytes, never assume the pinned one.

    A shard is a pure function of (wire bytes, decoder source), so a receipt
    that does not name the decoder cannot say which source it is a function
    of -- and the installed Tessera is not always the pinned one.
    """
    if reader is not None:
        return {"kind": "bound_reader", "identity": reader.identity}
    import tessera

    return {"kind": "installed", "file": str(Path(tessera.__file__).resolve()),
            "version": str(getattr(tessera, "__version__", "unknown"))}


def _compare_mirrored_renders(data):
    """Byte-compare each mirrored shard with the campaign's own, where it has one."""
    counts = {"compared": 0, "byte_identical": 0, "differs": 0, "no_campaign_render": 0}
    differing = []
    for pair, cell in sorted(data.cells.items()):
        campaign = Path(cell["campaign_render"])
        if not campaign.is_file():
            counts["no_campaign_render"] += 1
            continue
        counts["compared"] += 1
        if _sha(campaign) == _sha(Path(cell["render"])):
            counts["byte_identical"] += 1
        else:
            counts["differs"] += 1
            differing.append("@".join(pair))
    return {**counts, "differing_cells": differing[:32]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "synthesize"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--prepared-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prewarm-manifest", type=Path)
    parser.add_argument("--prewarm-manifest-sha256")
    parser.add_argument("--cost-read-manifest", type=Path)
    parser.add_argument("--cost-read-manifest-sha256")
    parser.add_argument("--cost-read-manifest-bytes", type=int)
    parser.add_argument("--data-manifest-sha256",
                        help="the digest of the PrismaBuild data manifest this "
                             "pass was submitted with. It binds the read set a "
                             "residency map may answer for and nothing else; a "
                             "map composed for another manifest is refused.")
    parser.add_argument("--source-transition", type=Path)
    parser.add_argument("--source-transition-sha256")
    parser.add_argument("--units", help="synthesize: lo:hi over the sorted census roster. "
                        "Rows carry disjoint ranges; submit each with "
                        "--progress synthesize=SECONDS, because the stage reports its "
                        "cumulative durable count under exactly that phase name and an "
                        "undeclared phase grants no continuation")
    parser.add_argument("--device", default="cpu", help="synthesize: where the wire decode runs")
    parser.add_argument("--log-every", type=int, default=100,
                        help="synthesize: log a cumulative count and rate every N shards")
    parser.add_argument("--mirror-root", type=Path,
                        help="synthesize: publish into this mirror instead of the row caches")
    parser.add_argument("--compare", action="store_true",
                        help="synthesize: byte-compare each mirrored shard with the campaign's")
    parser.add_argument("--i-am-authorized", action="store_true",
                        help="synthesize: publish into the campaign row caches")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    if bool(args.source_transition) != bool(args.source_transition_sha256):
        parser.error("--source-transition and --source-transition-sha256 are required together")
    if bool(args.prepared) != bool(args.prepared_sha256):
        parser.error("--prepared and --prepared-sha256 are required together")
    cost_descriptor = (args.cost_read_manifest, args.cost_read_manifest_sha256, args.cost_read_manifest_bytes)
    if any(value is not None for value in cost_descriptor) and not all(value is not None for value in cost_descriptor):
        parser.error('all three --cost-read-manifest descriptor fields are required together')
    if args.cost_read_manifest is not None and args.command != 'run':
        parser.error('--cost-read-manifest applies only to run')
    if bool(args.prewarm_manifest) != bool(args.prewarm_manifest_sha256):
        parser.error("--prewarm-manifest and --prewarm-manifest-sha256 are required together")
    if args.data_manifest_sha256 is not None and (
            len(args.data_manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in args.data_manifest_sha256)):
        parser.error("--data-manifest-sha256 takes a 64-character lowercase digest")
    # ``synthesize`` constructs no lease and loads no backend: it decodes wires
    # and publishes the canonical CPU BF16 shard, whose bytes are measured
    # identical across x86/aarch64 and CPU/CUDA. It is the one command that
    # does not need the projection runtime, and refusing it here would refuse
    # the stage that exists to run off the qualified box.
    config = _load_plan(args.plan, args.plan_sha256,
                        projection_runtime=args.command != "synthesize")
    if args.command == "synthesize":
        record = synthesize_renders(config, plan_sha256=args.plan_sha256, units=args.units,
                                    device=args.device, log_every=args.log_every,
                                    mirror_root=args.mirror_root, compare=args.compare,
                                    authorized=args.i_am_authorized, receipt=args.receipt)
        print(json.dumps({key: record[key] for key in (
            "units_read", "cells", "renders_synthesized_now", "seconds", "render_origins")}))
        return 0
    result = execute(args.command, config, plan_sha256=args.plan_sha256,
        **({'cost_read_manifest': {'path': str(args.cost_read_manifest),
             'sha256': args.cost_read_manifest_sha256, 'bytes': args.cost_read_manifest_bytes},
             'plan_path': str(args.plan)} if args.cost_read_manifest is not None else {}),
        prepared=None if args.prepared is None else {"path": str(args.prepared), "sha256": args.prepared_sha256},
        resume=args.resume,
        data_manifest_sha256=args.data_manifest_sha256,
        **({"prewarm_manifest": {"path": str(args.prewarm_manifest),
                                   "sha256": args.prewarm_manifest_sha256}}
           if args.prewarm_manifest is not None else {}),
        **({"source_transition": {"path": str(args.source_transition),
                                  "sha256": args.source_transition_sha256}}
           if args.source_transition is not None else {}))
    print(json.dumps({key: result[key] for key in ("command", "passed", "units",
                                                   "measured_cells", "render_origins",
                                                   "render_comparisons")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
