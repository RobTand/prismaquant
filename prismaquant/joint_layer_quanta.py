"""Per-layer cost quanta for the distributed joint-AURA campaign.

Contract: ``docs/design/distributed_campaign_2026-09-19.md`` §§3–5, 7.
Pure data, stdlib only, torch-free (like ``joint_prewarm_phases``): a CPU
checkout builds the campaign. Same inputs always seal byte-identical records;
coverage proofs refuse gaps rather than shrinking the layer set.

Derivation decisions (the contract leaves these corners implicit; each is
documented where it is implemented):

D1. Slice phase tables name ``head`` as a zero-byte leading phase. ``chunks``
    must tile ``[0, total_bytes)`` (§3.1) and the 45 read sets must be
    pairwise disjoint (§2.3), so no parent-head entry and no other layer's
    source extent may be copied into a slice. The quantum loads its head
    inputs (plan, prepared, calibration, identity — digest-verified, ARC-warm)
    by path from the shared mount, exactly as boundary artifacts are
    receipt-addressed rather than manifest entries (§3.3); every tier-fed
    byte flows through a chunk phase.
D2. ``windows`` seals the ordered window-index slice of the plan's retained
    window partition (``windows_by_layer`` counts, copied verbatim). Per-window
    names and byte sizes are recomputed at runtime by the quantum through
    ``plan_joint_statistics_target_windows`` — deterministic from the same
    sealed budget — because per-target statistics bytes need module geometry
    the producer's declared inputs do not carry. Sealing an invented packing
    would assert false facts; sealing the count and order is what the
    coverage proof checks.
D3. ``adjoint.receipt_sha256`` is ``None`` until stage A seals
    ``adjoint-capture.json``. The dispatcher seals unbound records (nothing
    submits on them), then re-seals bound records once the receipt validates;
    binding is a new identity, never an edit. ``check_quantum_for_campaign``
    refuses an unbound record for a bound campaign.
D4. Slice ``argv`` annotations carry the §5.2 inner argv without
    ``--quantum-sha256``: that digest is a submission-time binding (like
    ``--data-manifest-sha256`` on the run manifest), since the identity it
    would name covers the manifest bytes carrying the argv.
D5. The joiner is NOT colocated here. The contract (§4.1/§7) sketched
    ``join_layer_quanta`` inside the producer module; #783 merged the joiner
    as ``prismaquant/joint_quanta_join.py`` first, and #787 kept that module
    (its CLI, its allocation reader, its coverage proof) and deleted this
    module's parallel joiner rather than adapting it -- two joiners that
    refuse each other's bytes is the failure #787 closed. The joiner imports
    this module's pure constructions (``roster_digest``, ``phase_ranges``,
    ``quantum_id``, ``qname_layer``) so the two sides of the wire share one
    spelling of every digest and id.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections.abc import Mapping, Sequence

from .cost_stage_checkpoint import canonical_json_bytes, canonical_json_sha256


LAYER_QUANTUM_SCHEMA = "prismaquant.joint_layer_quanta.v1"
PLAN_BLOCK_SCHEMA = "prismaquant.joint_layer_quanta.plan.v1"
COVERAGE_SCHEMA = "prismaquant.joint_layer_quanta.coverage.v1"
#: The stage-A capture schema, named here so ``bind_adjoint_receipt`` and
#: the records it digests cite one spelling.
ADJOINT_CAPTURE_SCHEMA = "prismaquant.joint_adjoint_capture.v1"
MANIFEST_SCHEMA_V1 = "prismaquant.prismabuild.data_manifest.v1"
MANIFEST_SCHEMA_V2 = "prismaquant.prismabuild.data_manifest.v2"

#: The tail leg's telemetry name. It is NOT a read-plan phase: published
#: ``manifest_phase_ranges`` drops cumulative==previous phases, so a
#: zero-byte tail would be absent from the sealed residency plan -- and
#: reporting a name the plan does not carry resets ``remaining`` to start.
#: The tail checkpoint work commits durable units under forward-last
#: instead; this string survives only for explicit runtime log lines.
ADJOINT_TAIL_PHASE = "tail"


def adjoint_forward_phase_name(layer: int) -> str:
    """The read-plan phase staging layer ``layer``'s source extents for the
    forward capture walk (ascending)."""
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"a forward phase needs a nonnegative layer, not {layer!r}")
    return f"forward-{layer:03d}"


def adjoint_chain_phase_name(layer: int) -> str:
    """The read-plan phase staging layer ``layer``'s source extents for the
    reverse chain walk (descending). One spelling shared by the builder, the
    capture's progress reports, and the tests -- two spellings of the chain
    convention is how a consumer ends up committing a name the window cannot
    match."""
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"a chain phase needs a nonnegative layer, not {layer!r}")
    return f"chain-{layer:03d}"


def adjoint_read_plan_phase_names(num_layers: int) -> tuple[str, ...]:
    """The frozen Stage A consumption order (PQ #837): head, forward
    ascending, reverse descending. The builder seals the manifest in this
    order, the capture reports in this order, and the dispatch lane
    declares progress in this order -- the published v2 linear-progress
    rule refuses anything else. There is deliberately no tail phase (see
    ``ADJOINT_TAIL_PHASE``)."""
    if type(num_layers) is not int or isinstance(num_layers, bool) or num_layers < 1:
        raise ValueError(f"a read plan needs a positive layer count, not {num_layers!r}")
    return (("head",)
            + tuple(adjoint_forward_phase_name(layer) for layer in range(num_layers))
            + tuple(adjoint_chain_phase_name(layer)
                    for layer in reversed(range(num_layers))))

SLICE_ENTRY_POINT = "prismaquant.joint_cost_quantum"
ADJOINT_ENTRY_POINT = "prismaquant.joint_adjoint_capture"

GIB = 1024 ** 3
CHUNK_MIN_BYTES = 8 * GIB
CHUNK_MAX_BYTES = 64 * GIB
DEFAULT_STRIDE = 8
DEFAULT_RAM_WINDOW_GIB = 160
DEFAULT_MAX_RESIDENT_CONSUMERS = 2

_QNAME_LAYER = re.compile(r"^.*\.layers\.(\d+)(?:\.|$)")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


# Canonical JSON: the one encoding (sort_keys, compact separators,
# unescaped UTF-8, no NaN), imported from cost_stage_checkpoint -- the one
# implementation in the tree (#787 B5). The thin wrappers keep this module's
# default ``where`` label; digests are byte-identical to the shared helpers
# (verified against all 45 sealed takeover records).

def canonical_bytes(value: object, *, where: str = "joint layer quanta") -> bytes:
    return canonical_json_bytes(value, where=where)


def canonical_sha256(value: object, *, where: str = "joint layer quanta") -> str:
    return canonical_json_sha256(value, where=where)


def seal_manifest_bytes(manifest: Mapping) -> bytes:
    """The exact bytes a slice/adjoint manifest file carries.

    Mirrors the run-manifest sealing (``dispatch_tessera_campaign``): compact
    JSON with insertion order, one trailing newline, one gzip member with
    ``mtime=0``. Insertion order is deterministic because every manifest dict
    is constructed key by key in this module.
    """
    if not isinstance(manifest, dict):
        raise ValueError("a data manifest must be a JSON object")
    try:
        decoded = json.dumps(manifest, separators=(",", ":"),
                             sort_keys=False).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise ValueError("a data manifest must be canonical JSON data") from exc
    return gzip.compress(decoded, mtime=0)


def quantum_id(layer: int) -> str:
    if type(layer) is not int or layer < 0:
        raise ValueError(f"a quantum layer must be a nonnegative integer, not {layer!r}")
    return f"layer-{layer:03d}"


def roster_digest(qnames: Sequence[str]) -> str:
    """``sha256`` of the sorted qname roster, one per line (#768's construction)."""
    ordered = sorted(qnames)
    if any(type(name) is not str or not name for name in ordered):
        raise ValueError("a unit roster holds nonempty qname strings")
    if len(set(ordered)) != len(ordered):
        raise ValueError("a unit roster holds unique qnames")
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()


def qname_layer(qname: object) -> int | None:
    """The layer a roster qname names, or None (§3.1's layer grammar).

    The single spelling of the grammar, shared with the joiner (#787 D5):
    a roster qname embeds its layer as ``…layers.N…``, which is how a gap
    names its quantum's units and how the producer partitions the roster
    per layer.
    """
    if type(qname) is not str:
        return None
    match = _QNAME_LAYER.match(qname)
    return None if match is None else int(match.group(1))


def _hex(value: object, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _bytes_int(value: object, label: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


# Parent-manifest phase tiling.

def phase_ranges(parent_manifest: Mapping) -> list[dict]:
    """Cut the parent entry list at its sealed phase boundaries.

    Returns one row per phase: ``name``, ``start_bytes``/``end_bytes`` in
    parent-absolute bytes, and the entry slice ``[entry_begin, entry_end)``.
    Refuses an entry straddling a boundary (misalignment), a phase total that
    misses its cumulative mark, and a manifest whose entries disagree with
    ``entry_count``/``total_bytes``. The run manifest's phases carry no start
    offsets; contiguity from zero is part of what is checked.
    """
    entries = parent_manifest.get("entries")
    annotations = parent_manifest.get("annotations")
    if not isinstance(entries, list) or not entries:
        raise ValueError("the parent manifest has no entries")
    if not isinstance(annotations, dict):
        raise ValueError("the parent manifest has no annotations")
    phases = annotations.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("the parent manifest has no phase table")
    if parent_manifest.get("entry_count") != len(entries):
        raise ValueError("the parent manifest entry_count disagrees with entries")
    total = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"parent manifest entries[{index}] is not an object")
        size = entry.get("bytes")
        if type(size) is not int or isinstance(size, bool) or size <= 0:
            raise ValueError(f"parent manifest entries[{index}] has no positive bytes")
        total += size
    if parent_manifest.get("total_bytes") != total:
        raise ValueError("the parent manifest total_bytes disagrees with entries")
    ranges = []
    start = 0
    cursor = 0
    running = 0
    for position, row in enumerate(phases):
        if not isinstance(row, dict):
            raise ValueError(f"parent manifest phases[{position}] is not an object")
        name = row.get("name")
        cumulative = row.get("cumulative_bytes")
        if type(name) is not str or not name:
            raise ValueError(f"parent manifest phases[{position}] has no name")
        if type(cumulative) is not int or isinstance(cumulative, bool) or cumulative <= start:
            raise ValueError(f"parent manifest phase {name!r} has no forward cumulative mark")
        entry_begin = cursor
        while running < cumulative:
            if cursor >= len(entries):
                raise ValueError(
                    f"parent manifest phase {name!r} runs past the entry list: gap")
            running += entries[cursor]["bytes"]
            cursor += 1
            if running > cumulative:
                raise ValueError(
                    f"parent manifest entry {cursor - 1} straddles phase {name!r}: "
                    f"entry-misaligned, refusing")
        ranges.append({"name": name, "start_bytes": start, "end_bytes": cumulative,
                       "entry_begin": entry_begin, "entry_end": cursor})
        start = cumulative
    if running != total:
        raise ValueError("the parent manifest phases do not cover the entry list: gap")
    names = [row["name"] for row in ranges]
    if len(set(names)) != len(names):
        raise ValueError("the parent manifest phase names are not unique")
    return ranges


# Chunk (§5.3) and stride (§3.4) derivations.

def derive_chunk_target_bytes(ram_window_gib: int | float,
                              max_resident_consumers: int) -> int:
    """``floor(window / (2 * consumers))``, clamped to [8 GiB, 64 GiB].

    A window that cannot support 8-GiB chunks (under 32 GiB at two consumers)
    is a plan error, not a smaller chunk: per-chunk movement overhead then
    dominates, so the plan refuses instead of shrinking.
    """
    if (isinstance(ram_window_gib, bool)
            or not isinstance(ram_window_gib, (int, float))
            or ram_window_gib <= 0):
        raise ValueError("ram_window_gib must be a positive number")
    if (type(max_resident_consumers) is not int or max_resident_consumers < 1):
        raise ValueError("max_resident_consumers must be a positive integer")
    window_bytes = int(ram_window_gib * GIB)
    floor_bytes = 2 * max_resident_consumers * CHUNK_MIN_BYTES
    if window_bytes < floor_bytes:
        raise ValueError(
            f"a {ram_window_gib} GiB ram window cannot support "
            f"{CHUNK_MIN_BYTES // GIB}-GiB chunks at {max_resident_consumers} "
            f"consumers: plan error, refusing")
    target = window_bytes // (2 * max_resident_consumers)
    return max(CHUNK_MIN_BYTES, min(CHUNK_MAX_BYTES, target))


def derive_chunks(quantum: str, layer_bytes: int, entry_ends: Sequence[int],
                  chunk_target_bytes: int) -> list[dict]:
    """Tile ``[0, layer_bytes)`` on entry boundaries (ceiling cuts).

    ``entry_ends`` are the slice-rebased cumulative entry boundaries in order,
    ending at ``layer_bytes``; cut ``k`` is the first boundary ``b`` with
    ``b * n >= k * layer_bytes`` for ``n = ceil(layer_bytes / target)``.
    A cut that lands on an already-cut boundary (one entry larger than its
    share) is refused rather than sealed as an empty chunk.
    """
    if type(quantum) is not str or not quantum:
        raise ValueError("a chunk table needs its quantum id")
    layer_bytes = _bytes_int(layer_bytes, "layer_bytes")
    if layer_bytes <= 0:
        raise ValueError("a chunk table needs a positive layer size")
    chunk_target_bytes = _bytes_int(chunk_target_bytes, "chunk_target_bytes")
    if chunk_target_bytes <= 0:
        raise ValueError("a chunk table needs a positive target")
    ends = list(entry_ends)
    if not ends or ends[-1] != layer_bytes:
        raise ValueError(f"quantum {quantum}: entry boundaries do not end at the layer size")
    if any(type(edge) is not int or isinstance(edge, bool) or edge <= 0 for edge in ends):
        raise ValueError(f"quantum {quantum}: entry boundaries must be positive integers")
    if any(later <= earlier for earlier, later in zip(ends, ends[1:])):
        raise ValueError(f"quantum {quantum}: entry boundaries are not strictly increasing")
    count = -(-layer_bytes // chunk_target_bytes)
    cuts = []
    for cut in range(1, count):
        threshold = cut * layer_bytes
        edge = next(edge for edge in ends if edge * count >= threshold)
        if cuts and edge <= cuts[-1]:
            raise ValueError(
                f"quantum {quantum}: an entry spans chunk cut {cut}: refusing")
        if edge >= layer_bytes:
            raise ValueError(
                f"quantum {quantum}: no entry boundary left for chunk cut {cut}: refusing")
        cuts.append(edge)
    bounds = [0, *cuts, layer_bytes]
    return [{"name": f"{quantum}-chunk-{index:03d}",
             "start_bytes": first, "end_bytes": second}
            for index, (first, second) in enumerate(zip(bounds, bounds[1:]))]


def derive_stride(num_layers: int, stride: int) -> dict:
    """Strided adjoint checkpoints: multiples of ``stride`` below the tail,
    plus the tail boundary itself (``ceil(num_layers / stride)`` checkpoints,
    at most ``stride - 1`` render-free backwards per quantum).
    """
    if type(num_layers) is not int or num_layers < 1:
        raise ValueError("a stride derivation needs a positive layer count")
    if type(stride) is not int or stride < 1:
        raise ValueError("a stride derivation needs a positive stride")
    checkpoints = [mark for mark in range(stride, num_layers, stride)]
    checkpoints.append(num_layers)
    return {"stride": stride, "num_layers": num_layers, "checkpoints": checkpoints,
            "num_checkpoints": len(checkpoints), "max_chain_layers": stride - 1}


def adjoint_binding(layer: int, checkpoints: Sequence[int]) -> tuple[int, list[int]]:
    """Nearest checkpoint boundary at or above L+1, chain descending below it."""
    if type(layer) is not int or layer < 0:
        raise ValueError(f"an adjoint binding needs a nonnegative layer, not {layer!r}")
    marks = sorted(checkpoints)
    if not marks or any(type(mark) is not int or isinstance(mark, bool)
                        for mark in marks):
        raise ValueError(f"no strided checkpoint covers layer {layer}: refusing")
    for mark in marks:
        if mark >= layer + 1:
            return mark, list(range(mark - 1, layer, -1))
    raise ValueError(f"no strided checkpoint covers layer {layer}: refusing")


# Producer.

def _plan_block(plan: Mapping) -> dict:
    block = plan.get("distributed_campaign", {})
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError("the plan distributed_campaign block must be an object")
    return block


def _canonical_partition_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _windows_by_layer(plan: Mapping, window_partition: Mapping | None = None) -> dict[int, int]:
    """The sealed retained-window partition, from the plan or an explicit input.

    The plan remains the source when it carries ``retained_window_budget_derivation``
    (its §4.1 home).  A plan that does not -- the single-run plan the prepared
    completion binds, whose re-plan sibling exists only to carry this partition --
    may name the same record explicitly as a derivation input.  The two sources
    never disagree silently: a plan that carries the block and an explicit input
    that differs from it refuses, and the derivation envelope (v2) records which
    source was used.
    """
    in_plan = plan.get("retained_window_budget_derivation")
    if in_plan is not None and window_partition is not None:
        if (_canonical_partition_bytes(in_plan)
                != _canonical_partition_bytes(window_partition)):
            raise ValueError(
                "the plan's sealed retained-window partition and the explicit "
                "window_partition input disagree: refusing to choose between "
                "two sealed sources")
        record = in_plan
    elif in_plan is not None:
        record = in_plan
    elif window_partition is not None:
        record = window_partition
    else:
        raise ValueError(
            "no sealed retained-window partition: the plan carries none and "
            "no explicit window_partition input was given")
    try:
        counts = record["windows_by_layer"]
    except (KeyError, TypeError) as exc:
        raise ValueError("the retained-window partition record has no windows_by_layer") from exc
    if not isinstance(counts, dict) or not counts:
        raise ValueError("the plan retained-window partition is not an object")
    ordered = {}
    for key, value in counts.items():
        try:
            layer = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"the window partition key {key!r} is not a layer") from exc
        if type(value) is not int or isinstance(value, bool) or value < 1:
            raise ValueError(f"the window partition for layer {layer} is not a positive count")
        ordered[layer] = value
    return ordered


def _layer_qnames(prepared: Mapping) -> dict[int, list[str]]:
    formats = prepared.get("formats_by_qname")
    if not isinstance(formats, dict) or not formats:
        raise ValueError("the prepared completion has no formats_by_qname roster")
    by_layer: dict[int, list[str]] = {}
    for qname in formats:
        match = _QNAME_LAYER.match(qname) if type(qname) is str else None
        if match is None:
            raise ValueError(f"roster qname {qname!r} names no layer")
        by_layer.setdefault(int(match.group(1)), []).append(qname)
    for names in by_layer.values():
        names.sort()
    return by_layer


def _argv_flag(argv: Sequence, flag: str) -> str | None:
    items = list(argv)
    for index, item in enumerate(items[:-1]):
        if item == flag and type(items[index + 1]) is str:
            return items[index + 1]
    return None


def layer_quanta(plan: Mapping, prepared: Mapping, parent_manifest: Mapping, *,
                 chunk_target_bytes: int | None = None,
                 stride: int | None = None,
                 output_root: str | None = None,
                 plan_path: str | None = None,
                 plan_sha256: str | None = None,
                 prepared_path: str | None = None,
                 prepared_sha256: str | None = None,
                 parent_manifest_sha256: str,
                 ram_window_gib: int | float | None = None,
                 max_resident_consumers: int | None = None,
                 window_partition: Mapping | None = None,
                 adjoint_receipt: Mapping | None = None) -> dict:
    """Cut the sealed campaign into per-layer quantum records (§4.1).

    Reads the plan, the prepared completion and the sealed run manifest;
    writes nothing (callers persist records, slices and the adjoint manifest
    at the paths the records name). Layers ascend; every list is sealed-order;
    only canonical JSON is digested.
    """
    if not isinstance(plan, dict) or not isinstance(prepared, dict):
        raise ValueError("layer_quanta needs the plan and prepared mappings")
    if not isinstance(parent_manifest, dict):
        raise ValueError("layer_quanta needs the parent manifest mapping")
    _hex(parent_manifest_sha256, "parent_manifest_sha256")
    annotations = parent_manifest.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError("the parent manifest has no annotations")
    scope = annotations.get("campaign_scope")
    if not isinstance(scope, dict) or not scope:
        raise ValueError("the parent manifest has no campaign scope")
    layers = annotations.get("layers")
    if (not isinstance(layers, list) or not layers
            or any(type(item) is not int or isinstance(item, bool) for item in layers)
            or sorted(layers) != list(range(min(layers), max(layers) + 1))):
        raise ValueError("the parent manifest layers are not a contiguous range")
    layers = sorted(layers)
    block = _plan_block(plan)
    if stride is None:
        stride = block.get("cotangent_checkpoint_stride", DEFAULT_STRIDE)
    if type(stride) is not int or isinstance(stride, bool) or stride < 1:
        raise ValueError("the cotangent checkpoint stride must be a positive integer")
    if ram_window_gib is None:
        ram_window_gib = block.get("ram_window_gib", DEFAULT_RAM_WINDOW_GIB)
    if max_resident_consumers is None:
        max_resident_consumers = block.get("max_resident_consumers",
                                           DEFAULT_MAX_RESIDENT_CONSUMERS)
    if chunk_target_bytes is None:
        chunk_target_bytes = derive_chunk_target_bytes(ram_window_gib,
                                                       max_resident_consumers)
    chunk_target_bytes = _bytes_int(chunk_target_bytes, "chunk_target_bytes")
    if chunk_target_bytes <= 0:
        raise ValueError("chunk_target_bytes must be positive")
    if output_root is None:
        output_root = plan.get("output_root")
    if type(output_root) is not str or not output_root.startswith("/"):
        raise ValueError("layer_quanta needs an absolute output_root")
    produced = parent_manifest.get("produced_by")
    argv = annotations.get("argv", [])
    if plan_path is None:
        plan_path = (produced.get("plan") if isinstance(produced, dict) else None)
    if type(plan_path) is not str or not plan_path:
        raise ValueError("layer_quanta needs the sealed plan path")
    if plan_sha256 is None:
        plan_sha256 = annotations.get("plan_sha256")
    _hex(plan_sha256, "plan_sha256")
    if prepared_path is None:
        prepared_path = _argv_flag(argv, "--prepared") if isinstance(argv, list) else None
    if type(prepared_path) is not str or not prepared_path:
        raise ValueError("layer_quanta needs the sealed prepared path")
    if prepared_sha256 is None:
        prepared_sha256 = _argv_flag(argv, "--prepared-sha256") if isinstance(argv, list) else None
    _hex(prepared_sha256, "prepared_sha256")

    roster = sorted(prepared.get("formats_by_qname", {}))
    if not roster:
        raise ValueError("the prepared completion has no unit roster")
    digest = roster_digest(roster)
    by_layer = _layer_qnames(prepared)
    partition_from_plan = plan.get("retained_window_budget_derivation") is not None
    counts = _windows_by_layer(plan, window_partition)
    ranges = {row["name"]: row for row in phase_ranges(parent_manifest)}
    entries = parent_manifest["entries"]
    if "head" not in ranges:
        raise ValueError("the parent manifest has no head phase")

    derived_stride = derive_stride(len(layers), stride)
    checkpoints = derived_stride["checkpoints"]
    receipt_sha = None
    if adjoint_receipt is not None:
        receipt_sha = bind_adjoint_receipt(adjoint_receipt, plan_sha256=plan_sha256,
                                           prepared_sha256=prepared_sha256,
                                           scope=scope, checkpoints=checkpoints)
    quanta_root = output_root.rstrip("/") + "/layer-quanta"
    adjoint_dir = quanta_root + "/adjoint"

    records = []
    slices = {}
    for layer in layers:
        name = f"layer-{layer}"
        row = ranges.get(name)
        if row is None:
            raise ValueError(f"the parent manifest has no phase {name!r}: gap, refusing")
        qid = quantum_id(layer)
        names = by_layer.get(layer, [])
        if not names:
            raise ValueError(f"the roster holds no unit for layer {layer}: gap, refusing")
        window_count = counts.get(layer)
        if window_count is None:
            raise ValueError(
                f"the window partition holds no count for layer {layer}: gap, refusing")
        layer_entries = entries[row["entry_begin"]:row["entry_end"]]
        layer_bytes = row["end_bytes"] - row["start_bytes"]
        if sum(entry["bytes"] for entry in layer_entries) != layer_bytes:
            raise ValueError(f"phase {name!r} bytes disagree with its entries")
        ends = []
        running = 0
        for entry in layer_entries:
            running += entry["bytes"]
            ends.append(running)
        chunks = derive_chunks(qid, layer_bytes, ends, chunk_target_bytes)
        checkpoint, chain = adjoint_binding(layer, checkpoints)
        windows = [{"window_index": index} for index in range(window_count)]
        manifest_path = f"{quanta_root}/manifests/{qid}.data-manifest.json.gz"
        record_path = f"{quanta_root}/records/{qid}.json"
        space = f"{quanta_root}/{qid}"
        argv_sealed = ["python3", "-m", SLICE_ENTRY_POINT, "--quantum", record_path,
                       "--output-root", output_root]
        slice_manifest = {
            "schema": MANIFEST_SCHEMA_V1,
            "produced_by": {"tool": "prismaquant/joint_layer_quanta.py",
                            "entry_point": SLICE_ENTRY_POINT,
                            "plan": plan_path, "plan_sha256": plan_sha256},
            "mount_prefix": parent_manifest.get("mount_prefix", "/mnt/shared"),
            "entries": [dict(path=entry["path"], offset=entry["offset"],
                             bytes=entry["bytes"], sha256=entry.get("sha256"))
                        for entry in layer_entries],
            "entry_count": len(layer_entries),
            "total_bytes": layer_bytes,
            "annotations": {
                "entry_point": SLICE_ENTRY_POINT,
                "quantum_id": qid,
                "parent_manifest_sha256": parent_manifest_sha256,
                "plan_sha256": plan_sha256,
                "prepared_sha256": prepared_sha256,
                "campaign_scope": scope,
                "windows": windows,
                "argv": argv_sealed,
                # No zero-byte head phase (PQ #849): PB's phase rule (#594)
                # voids a table whose cumulative falls outside the entries'
                # own prefix sums, and 0 is not one -- the whole slice then
                # stages nothing. Startup/head progress stays separately
                # declared by the dispatch lane (quantum_argv seals head=
                # explicitly; the payload reports enter_head under it).
                "phases": [
                    {"name": chunk["name"],
                     "bytes": chunk["end_bytes"] - chunk["start_bytes"],
                     "cumulative_bytes": chunk["end_bytes"]} for chunk in chunks],
            },
        }
        blob = seal_manifest_bytes(slice_manifest)
        record = {
            "schema": LAYER_QUANTUM_SCHEMA,
            "quantum_id": qid,
            "layer": layer,
            "campaign": {
                "plan_path": plan_path,
                "plan_sha256": plan_sha256,
                "prepared_path": prepared_path,
                "prepared_sha256": prepared_sha256,
                "read_manifest_sha256": parent_manifest_sha256,
                "campaign_scope": scope,
                "unit_roster_sha256": digest,
            },
            "read_set": {
                "manifest_path": manifest_path,
                "manifest_sha256": hashlib.sha256(blob).hexdigest(),
                "entry_count": len(layer_entries),
                "total_bytes": layer_bytes,
                "source_phase": {"name": name, "start_bytes": row["start_bytes"],
                                 "end_bytes": row["end_bytes"]},
            },
            "chunks": chunks,
            "windows": windows,
            "adjoint": {
                "checkpoint_boundary": checkpoint,
                "chain_layers": chain,
                "boundary_artifacts": adjoint_dir,
                "receipt_sha256": receipt_sha,
            },
            "output_space": {
                "root": space,
                "cost_payload": space + "/cost.pkl",
                "results": space + "/results.json",
                "counters": space + "/counters.json",
                "checkpoint_dir": space + "/checkpoints",
            },
        }
        record["identity_sha256"] = canonical_sha256(
            record, where=f"quantum record {qid}")
        records.append(record)
        slices[qid] = slice_manifest

    adjoint_manifest = build_adjoint_manifest(
        plan, parent_manifest, ranges,
        plan_path=plan_path, plan_sha256=plan_sha256, prepared_path=prepared_path,
        prepared_sha256=prepared_sha256, parent_manifest_sha256=parent_manifest_sha256,
        output_root=output_root)
    coverage = verify_quanta_coverage(records, parent_manifest, plan=plan,
                                      window_partition=window_partition)
    return {"records": records, "slice_manifests": slices,
            "adjoint_manifest": adjoint_manifest, "coverage": coverage,
            "derivation": {
                "schema": "prismaquant.joint_layer_quanta.derivation.v2",
                "window_partition": {
                    "source": "plan" if partition_from_plan else "explicit",
                    "sha256": hashlib.sha256(_canonical_partition_bytes(
                        plan.get("retained_window_budget_derivation")
                        if partition_from_plan else window_partition)).hexdigest(),
                    "windows_total": sum(counts.values()),
                },
                "chunk_target_bytes": chunk_target_bytes,
                "ram_window_gib": ram_window_gib,
                "max_resident_consumers": max_resident_consumers,
                "stride": derived_stride["stride"],
                "num_layers": derived_stride["num_layers"],
                "checkpoints": checkpoints,
                "num_checkpoints": derived_stride["num_checkpoints"],
                "max_chain_layers": derived_stride["max_chain_layers"],
            }}


def slice_layer_manifest(parent_manifest: Mapping, layer: int, *,
                         chunk_target_bytes: int | None = None,
                         quantum: str | None = None,
                         output_root: str = "/mnt/shared",
                         manifest_path: str | None = None,
                         record_path: str | None = None,
                         plan_path: str = "",
                         plan_sha256: str | None = None,
                         prepared_sha256: str | None = None,
                         parent_manifest_sha256: str | None = None,
                         ram_window_gib: int | float = DEFAULT_RAM_WINDOW_GIB,
                         max_resident_consumers: int = DEFAULT_MAX_RESIDENT_CONSUMERS,
                         windows: list[dict] | None = None) -> dict:
    """Build one layer's standalone slice manifest (§4.3).

    The entries are exactly the parent manifest's entries in the layer phase's
    byte range, re-based to start at zero, order preserved. The phase table
    is the chunk names tiling the slice -- and nothing else: a zero-byte
    ``head`` would void the whole table under PB's phase rule (#594, PQ
    #849), so startup/head progress stays in the dispatch lane, never here.
    """
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"a slice needs a nonnegative layer, not {layer!r}")
    qid = quantum or quantum_id(layer)
    annotations = parent_manifest.get("annotations", {})
    scope = annotations.get("campaign_scope")
    if not isinstance(scope, dict) or not scope:
        raise ValueError("the parent manifest has no campaign scope")
    name = f"layer-{layer}"
    rows = {row["name"]: row for row in phase_ranges(parent_manifest)}
    row = rows.get(name)
    if row is None:
        raise ValueError(f"the parent manifest has no phase {name!r}: gap, refusing")
    if chunk_target_bytes is None:
        chunk_target_bytes = derive_chunk_target_bytes(ram_window_gib,
                                                       max_resident_consumers)
    layer_entries = parent_manifest["entries"][row["entry_begin"]:row["entry_end"]]
    layer_bytes = row["end_bytes"] - row["start_bytes"]
    ends = []
    running = 0
    for entry in layer_entries:
        running += entry["bytes"]
        ends.append(running)
    chunks = derive_chunks(qid, layer_bytes, ends, chunk_target_bytes)
    if windows is None:
        windows = []
    return {
        "schema": MANIFEST_SCHEMA_V1,
        "produced_by": {"tool": "prismaquant/joint_layer_quanta.py",
                        "entry_point": SLICE_ENTRY_POINT,
                        "plan": plan_path, "plan_sha256": plan_sha256 or ""},
        "mount_prefix": parent_manifest.get("mount_prefix", "/mnt/shared"),
        "entries": [dict(path=entry["path"], offset=entry["offset"],
                         bytes=entry["bytes"], sha256=entry.get("sha256"))
                    for entry in layer_entries],
        "entry_count": len(layer_entries),
        "total_bytes": layer_bytes,
        "annotations": {
            "entry_point": SLICE_ENTRY_POINT,
            "quantum_id": qid,
            "parent_manifest_sha256": parent_manifest_sha256,
            "plan_sha256": plan_sha256,
            "prepared_sha256": prepared_sha256,
            "campaign_scope": scope,
            "windows": windows,
            "argv": ["python3", "-m", SLICE_ENTRY_POINT, "--quantum",
                     record_path or "", "--output-root", output_root],
            # No zero-byte head phase (PQ #849): PB's phase rule (#594)
            # voids a table whose cumulative falls outside the entries'
            # own prefix sums, and 0 is not one -- the whole slice then
            # stages nothing. Startup/head progress stays separately
            # declared by the dispatch lane (quantum_argv seals head=
            # explicitly; the payload reports enter_head under it).
            "phases": [
                {"name": chunk["name"],
                 "bytes": chunk["end_bytes"] - chunk["start_bytes"],
                 "cumulative_bytes": chunk["end_bytes"]} for chunk in chunks],
        },
    }


def build_adjoint_manifest(plan: Mapping, parent_manifest: Mapping,
                           ranges: Mapping[str, dict] | None = None, *,
                           plan_path: str, plan_sha256: str, prepared_path: str,
                           prepared_sha256: str, parent_manifest_sha256: str,
                           output_root: str) -> dict:
    """The stage-A read manifest: head plus per-layer ``chain_`` phases.

    Stage A walks the source backward render-free, so chain phases carry each
    layer phase's source-extent entries only (paths under the plan's model
    dir); the shared head prefix is tiled verbatim. Renders never enter this
    manifest.

    The phase table is the v2 ``read_plan`` in true consumption order --
    head, forward ascending, reverse descending -- with ``entry_indices``
    into the one entries list, so the repeated forward and reverse reads
    reference the same entries twice instead of duplicating bytes. v2
    forbids ``annotations.phases``. There is no tail phase: the tail
    checkpoint work commits under forward-last (see ``ADJOINT_TAIL_PHASE``).
    The entries list, its digests, and every record the producer seals are
    untouched by the table.
    """
    model = plan.get("model")
    if type(model) is not str or not model:
        raise ValueError("the plan names no source model dir")
    prefix = model.rstrip("/") + "/"
    annotations = parent_manifest.get("annotations", {})
    scope = annotations.get("campaign_scope")
    if not isinstance(scope, dict) or not scope:
        raise ValueError("the parent manifest has no campaign scope")
    rows = ranges if ranges is not None else {
        row["name"]: row for row in phase_ranges(parent_manifest)}
    head = rows.get("head")
    if head is None:
        raise ValueError("the parent manifest has no head phase")
    entries = parent_manifest["entries"]
    head_entries = entries[head["entry_begin"]:head["entry_end"]]
    layers = sorted(int(name.split("-", 1)[1]) for name in rows
                    if name.startswith("layer-"))
    manifest_entries = [
        dict(path=entry["path"], offset=entry["offset"],
             bytes=entry["bytes"], sha256=entry.get("sha256"))
        for entry in head_entries]
    layer_index_runs: list[list[int]] = []
    for layer in layers:
        row = rows[f"layer-{layer}"]
        group = [entry for entry in entries[row["entry_begin"]:row["entry_end"]]
                 if type(entry.get("path")) is str and entry["path"].startswith(prefix)]
        if not group:
            raise ValueError(
                f"phase layer-{layer} holds no source extent: gap, refusing")
        start = len(manifest_entries)
        manifest_entries.extend(
            dict(path=entry["path"], offset=entry["offset"],
                 bytes=entry["bytes"], sha256=entry.get("sha256"))
            for entry in group)
        layer_index_runs.append(list(range(start, len(manifest_entries))))
    read_phases: list[dict] = []
    cumulative = 0

    def _seal_phase(name: str, indices: list[int]) -> None:
        nonlocal cumulative
        size = sum(manifest_entries[index]["bytes"] for index in indices)
        cumulative += size
        read_phases.append({"name": name, "entry_indices": list(indices),
                            "bytes": size, "cumulative_bytes": cumulative})

    _seal_phase("head", list(range(len(head_entries))))
    for layer, run in zip(layers, layer_index_runs):
        _seal_phase(adjoint_forward_phase_name(layer), run)
    for layer, run in zip(reversed(layers), reversed(layer_index_runs)):
        _seal_phase(adjoint_chain_phase_name(layer), run)
    names = [phase["name"] for phase in read_phases]
    if names != list(adjoint_read_plan_phase_names(len(layers))):
        raise ValueError(
            "the adjoint read plan is not the frozen consumption order")
    unique_bytes = sum(entry["bytes"] for entry in manifest_entries)
    return {
        "schema": MANIFEST_SCHEMA_V2,
        "produced_by": {"tool": "prismaquant/joint_layer_quanta.py",
                        "entry_point": ADJOINT_ENTRY_POINT,
                        "plan": plan_path, "plan_sha256": plan_sha256},
        "mount_prefix": parent_manifest.get("mount_prefix", "/mnt/shared"),
        "entries": manifest_entries,
        "entry_count": len(manifest_entries),
        "total_bytes": unique_bytes,
        "annotations": {
            "entry_point": ADJOINT_ENTRY_POINT,
            "plan_sha256": plan_sha256,
            "prepared_sha256": prepared_sha256,
            "parent_manifest_sha256": parent_manifest_sha256,
            "campaign_scope": scope,
            "argv": ["python3", "-m", ADJOINT_ENTRY_POINT, "--plan", plan_path,
                     "--plan-sha256", plan_sha256, "--prepared", prepared_path,
                     "--prepared-sha256", prepared_sha256,
                     "--output-root", output_root],
        },
        "read_plan": {"phases": read_phases, "read_bytes": cumulative},
    }


# Coverage and campaign binding (§3.2).

def verify_quanta_coverage(records: Sequence[Mapping], parent_manifest: Mapping, *,
                           plan: Mapping | None = None,
                           window_partition: Mapping | None = None) -> dict:
    """Prove the quanta tile the parent run manifest exactly once.

    Checks unique quantum ids covering every parent layer, one shared campaign
    block, gapless entry-aligned tiling of the layer phases, per-record chunk
    tables tiling their slices, and — when the plan is given — the window
    union against the sealed partition. A lost layer, a duplicated byte, a
    misaligned edge or a foreign record is a refusal naming the gap, never a
    silent hole.
    """
    rows = list(records)
    annotations = parent_manifest.get("annotations", {})
    layers = annotations.get("layers", [])
    if not rows:
        raise ValueError("no quantum records to cover the campaign: gap, refusing")
    for record in rows:
        if not isinstance(record, dict) or record.get("schema") != LAYER_QUANTUM_SCHEMA:
            raise ValueError("a quantum record has a foreign schema: refusing")
    ids = [record["quantum_id"] for record in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("quantum ids are not unique: refusing")
    if sorted(record["layer"] for record in rows) != sorted(layers):
        raise ValueError("quantum layers do not cover the parent manifest layers: "
                         "gap, refusing")
    for record in rows:
        if record["quantum_id"] != quantum_id(record["layer"]):
            raise ValueError(
                f"quantum id {record['quantum_id']!r} does not name its layer: refusing")
    first_campaign = canonical_bytes(rows[0]["campaign"])
    for record in rows[1:]:
        if canonical_bytes(record["campaign"]) != first_campaign:
            raise ValueError("quantum records span more than one campaign: mixed "
                             "campaign, refusing")
    ranges = {row["name"]: row for row in phase_ranges(parent_manifest)}
    entries = parent_manifest["entries"]
    by_start = sorted(rows, key=lambda record: record["read_set"]["source_phase"][
        "start_bytes"])
    previous_end = ranges["head"]["end_bytes"] if "head" in ranges else 0
    total = parent_manifest.get("total_bytes")
    layer_table = []
    for record in by_start:
        phase = record["read_set"]["source_phase"]
        name = phase.get("name")
        start = phase.get("start_bytes")
        end = phase.get("end_bytes")
        row = ranges.get(name) if type(name) is str else None
        if row is None or (start, end) != (row["start_bytes"], row["end_bytes"]):
            raise ValueError(
                f"quantum {record['quantum_id']} cites no sealed parent phase: "
                f"gap, refusing")
        if start != previous_end:
            kind = "overlap" if start < previous_end else "gap"
            raise ValueError(
                f"quantum tiling has a {kind} at {name!r} "
                f"({previous_end} != {start}): refusing")
        previous_end = end
        group = entries[row["entry_begin"]:row["entry_end"]]
        if len(group) != record["read_set"]["entry_count"]:
            raise ValueError(f"quantum {record['quantum_id']} entry count disagrees "
                             f"with its phase: refusing")
        layer_bytes = end - start
        if record["read_set"]["total_bytes"] != layer_bytes:
            raise ValueError(f"quantum {record['quantum_id']} size disagrees with its "
                             f"phase: refusing")
        ends = []
        running = 0
        for entry in group:
            running += entry["bytes"]
            ends.append(start + running)
        chunks = record.get("chunks", [])
        if not chunks:
            raise ValueError(f"quantum {record['quantum_id']} has no chunk table: "
                             f"refusing")
        bounds = [chunk["start_bytes"] for chunk in chunks] + [chunks[-1]["end_bytes"]]
        if bounds[0] != 0 or bounds[-1] != layer_bytes:
            raise ValueError(f"quantum {record['quantum_id']} chunks do not tile "
                             f"[0, {layer_bytes}): gap, refusing")
        for first, second in zip(chunks, chunks[1:]):
            if first["end_bytes"] != second["start_bytes"]:
                raise ValueError(f"quantum {record['quantum_id']} chunks are not "
                                 f"contiguous: gap, refusing")
        rebased = [edge - start for edge in ends]
        for chunk in chunks:
            if chunk["start_bytes"] not in [0, *rebased] or chunk["end_bytes"] not in rebased:
                raise ValueError(
                    f"quantum {record['quantum_id']} chunk {chunk['name']!r} cuts "
                    f"through an entry: entry-misaligned, refusing")
        layer_table.append({"quantum_id": record["quantum_id"], "layer": record["layer"],
                            "source_phase": {"name": name, "start_bytes": start,
                                             "end_bytes": end},
                            "entry_count": len(group), "total_bytes": layer_bytes,
                            "chunks": [chunk["name"] for chunk in chunks],
                            "windows": len(record.get("windows", []))})
    if previous_end != total:
        raise ValueError(f"quantum tiling stops at {previous_end} of {total}: "
                         f"gap, refusing")
    window_total = sum(item["windows"] for item in layer_table)
    if plan is not None or window_partition is not None:
        counts = _windows_by_layer(plan or {}, window_partition)
        for item in layer_table:
            if counts.get(item["layer"]) != item["windows"]:
                raise ValueError(
                    f"quantum {item['quantum_id']} windows do not match the sealed "
                    f"partition: refusing")
        if window_total != sum(counts.values()):
            raise ValueError("quantum windows do not cover the sealed partition: "
                             "gap, refusing")
    proof = {"schema": COVERAGE_SCHEMA, "quantum_ids": sorted(ids),
             "layers": sorted(record["layer"] for record in rows),
             "parent_total_bytes": total,
             "quanta": layer_table, "window_total": window_total,
             "campaign_sha256": hashlib.sha256(first_campaign).hexdigest()}
    proof["coverage_sha256"] = canonical_sha256(
        {key: proof[key] for key in ("quantum_ids", "layers", "parent_total_bytes",
                                     "quanta", "window_total", "campaign_sha256")},
        where="quanta coverage proof")
    return proof


def check_quantum_for_campaign(record: Mapping, campaign: Mapping) -> None:
    """Refuse a record built for any other campaign revision (fail closed)."""
    if not isinstance(record, dict) or record.get("schema") != LAYER_QUANTUM_SCHEMA:
        raise ValueError("a quantum record has a foreign schema: refusing")
    if not isinstance(campaign, dict):
        raise ValueError("a campaign binding must be an object")
    headless = {key: value for key, value in record.items() if key != "identity_sha256"}
    if canonical_sha256(headless, where="quantum record") != record.get("identity_sha256"):
        raise ValueError(
            f"quantum {record.get('quantum_id')!r} identity does not recompute: "
            f"edited record, refusing")
    bound = record.get("campaign", {})
    for field in ("plan_sha256", "prepared_sha256", "read_manifest_sha256",
                  "unit_roster_sha256"):
        expected = campaign.get(field)
        if expected is None:
            raise ValueError(f"the campaign binding names no {field}: refusing")
        if bound.get(field) != _hex(expected, field):
            raise ValueError(
                f"quantum {record.get('quantum_id')!r} was sealed for another "
                f"{field}: refusing")
    expected_scope = campaign.get("campaign_scope")
    if not isinstance(expected_scope, dict):
        raise ValueError("the campaign binding names no campaign_scope: refusing")
    if canonical_bytes(bound.get("campaign_scope")) != canonical_bytes(expected_scope):
        raise ValueError(
            f"quantum {record.get('quantum_id')!r} was sealed for another scope: "
            f"refusing")
    expected_receipt = campaign.get("adjoint_receipt_sha256")
    if expected_receipt is not None:
        actual = record.get("adjoint", {}).get("receipt_sha256")
        if actual is None:
            raise ValueError(
                f"quantum {record.get('quantum_id')!r} is unbound (pre-A): re-seal "
                f"against the stage-A receipt before publishing, refusing")
        if actual != _hex(expected_receipt, "adjoint_receipt_sha256"):
            raise ValueError(
                f"quantum {record.get('quantum_id')!r} binds another stage-A "
                f"receipt: refusing")


def bind_adjoint_receipt(receipt: Mapping, *, plan_sha256: str, prepared_sha256: str,
                         scope: Mapping, checkpoints: Sequence[int]) -> str:
    """Digest a stage-A receipt after checking it answers for this campaign."""
    if not isinstance(receipt, dict):
        raise ValueError("a stage-A receipt must be an object")
    if receipt.get("schema") != ADJOINT_CAPTURE_SCHEMA:
        raise ValueError("a stage-A receipt has a foreign schema: refusing")
    identity = receipt.get("run_identity", receipt)
    for field, expected in (("plan_sha256", plan_sha256),
                            ("prepared_sha256", prepared_sha256)):
        if identity.get(field) != expected:
            raise ValueError(f"the stage-A receipt answers for another {field}: refusing")
    if canonical_bytes(identity.get("campaign_scope")) != canonical_bytes(scope):
        raise ValueError("the stage-A receipt answers for another scope: refusing")
    sealed = receipt.get("checkpoints", [])
    marks = sorted(item["boundary"] for item in sealed) if sealed else []
    if marks != sorted(checkpoints):
        raise ValueError("the stage-A receipt checkpoints differ from the stride "
                         "derivation: refusing")
    return canonical_sha256(receipt, where="stage-A receipt")


