"""Per-layer cost quanta for the distributed joint-AURA campaign.

Contract: ``docs/design/distributed_campaign_2026-09-19.md`` §§3–5, 7.
Pure data, stdlib only, torch-free (like ``joint_prewarm_phases``): a CPU
checkout builds the campaign. Same inputs always seal byte-identical records;
coverage proofs refuse gaps rather than shrinking the layer set.

Derivation decisions (the contract leaves these corners implicit; each is
documented where it is implemented):

D1. Slice phase tables contain only chunk phases (PQ #849); a zero-byte
    leading ``head`` would invalidate the table for PrismaBuild. ``chunks``
    tile ``[0, total_bytes)`` (§3.1) and the 45 slices are pairwise disjoint
    (§2.3), so no parent-head entry or another layer's source extent may be
    copied into a slice. The separately bound executable readset declares
    calibration, checkpoint, chain/own source and boundary consumption;
    its read phases do not change the slice's chunk tiling.
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
import os
import re
import struct
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
#: A stage-A or executable quantum manifest's record of additions to the
#: parent's layer extents and the source-coverage gate (PQ #898 / #900).
SOURCE_COMPLETION_SCHEMA = "prismaquant.joint_layer_quanta.source_completion.v1"

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


def _canonical_control_root(value: object, *, label: str = "metadata_root") -> str:
    """A control-metadata root, validated once with the same discipline the
    bound-readset binders enforce per path: absolute, already normalized, no
    ``..`` traversal, no trailing slash. Control metadata (slice manifests,
    record files, bound readsets) may live in a generation-specific
    namespace beside the immutable data tree; loose strings and symlinks are
    not a placement contract, so the root is checked, never reinterpreted.
    """
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{label} must be an absolute path: refusing")
    import os
    stripped = value.rstrip("/")
    if not stripped or os.path.normpath(stripped) != stripped \
            or ".." in stripped.split("/"):
        raise ValueError(f"{label} must be a canonical absolute path "
                         f"(normalized, no '..' segments): refusing")
    return stripped


def layer_quanta(plan: Mapping, prepared: Mapping, parent_manifest: Mapping, *,
                 chunk_target_bytes: int | None = None,
                 stride: int | None = None,
                 output_root: str | None = None,
                 metadata_root: str | None = None,
                 plan_path: str | None = None,
                 plan_sha256: str | None = None,
                 prepared_path: str | None = None,
                 prepared_sha256: str | None = None,
                 parent_manifest_sha256: str,
                 ram_window_gib: int | float | None = None,
                 max_resident_consumers: int | None = None,
                 window_partition: Mapping | None = None,
                 adjoint_receipt: Mapping | None = None,
                 layer_source_spans: Mapping[int, Sequence] | None = None) -> dict:
    """Cut the sealed campaign into per-layer quantum records (§4.1).

    Reads the plan, the prepared completion and the sealed run manifest;
    writes nothing (callers persist records, slices and the adjoint manifest
    at the paths the records name). Layers ascend; every list is sealed-order;
    only canonical JSON is digested.

    ``metadata_root`` (optional, PQ #884) separates the DATA output root from
    an explicit immutable control-metadata generation root. The data root
    (``output_root``, default the plan's) keeps owning what execution writes
    and what stage A seals: ``output_space`` stays
    ``{output_root}/layer-quanta/{qid}`` and
    ``adjoint.boundary_artifacts`` stays ``{output_root}/layer-quanta/adjoint``
    — the consumer's identity gate pins exactly those derivations. The
    control root owns what THIS producer seals: slice manifests land at
    ``{control_root}/manifests/…`` and the record path sealed into the slice
    argv is ``{control_root}/records/{qid}.json``. With no ``metadata_root``
    the control root is the historical ``{output_root}/layer-quanta`` and
    every sealed path is byte-identical to the previous layout (existing
    calls and defaults unchanged).

    ``layer_source_spans`` (optional, PQ #898) is handed to
    ``build_adjoint_manifest``, which completes and gates the stage-A manifest
    against it. The per-layer slices and the records that seal them are not
    completed here: they tile the parent byte for byte (§3.1), so a parent
    that dropped a layer's tail is carried into that layer's slice unchanged.
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
    # PQ #884: control metadata (slice manifests, record paths) may live in
    # an explicit generation namespace; the data derivations above never
    # move. Default control root = the historical layout, so the seam is
    # invisible to every existing caller.
    control_root = (_canonical_control_root(metadata_root)
                    if metadata_root is not None else quanta_root)
    control_manifest_dir = control_root + "/manifests"
    control_record_dir = control_root + "/records"

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
        manifest_path = f"{control_manifest_dir}/{qid}.data-manifest.json.gz"
        record_path = f"{control_record_dir}/{qid}.json"
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
        output_root=output_root, layer_source_spans=layer_source_spans)
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
                # PQ #884: provenance for an explicit control-metadata
                # generation namespace, present ONLY when one was given --
                # the default-layout derivation bytes stay exactly what the
                # previous producer sealed. Nothing consumes an exhaustive
                # key set here (the joiner cites derivation v2 for windows
                # only); the field is additive provenance, not identity
                # ceremony: the records and manifests already bind their
                # actual paths.
                **({"control_metadata_root": control_root}
                   if metadata_root is not None else {}),
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


#: A safetensors header is a u64 length and that many bytes of JSON. The bound
#: is ``layer_streaming``'s own, so the two readers refuse the same files.
_SAFETENSORS_HEADER_MAX_BYTES = 100_000_000


def read_layer_source_spans(model_dir: str, num_layers: int, *,
                            checkpoint_layers_prefix: str,
                            ) -> dict[int, list[tuple[str, int, int]]]:
    """What stage A reads from the source, per layer, as file spans.

    ``{layer: [(shard path, start, end), ...]}`` in absolute file offsets, for
    every checkpoint tensor named ``{checkpoint_layers_prefix}{layer}.*`` with
    ``0 <= layer < num_layers``. This is the streamed reader's own selection
    (``layer_streaming._read_layer_to_device`` takes every name under the
    layer's prefix), read from the same two places it reads: the checkpoint
    index and each shard's header. Stdlib only, like the rest of this module.

    It exists because a read manifest is a claim about what a reader will
    read, and until PQ #898 nothing compared the two.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    pattern = re.compile(rf"^{re.escape(checkpoint_layers_prefix)}([0-9]+)\.")
    wanted: dict[str, list[tuple[int, str]]] = {}
    for name, shard in weight_map.items():
        match = pattern.match(name)
        if match is None or int(match.group(1)) >= num_layers:
            continue
        wanted.setdefault(shard, []).append((int(match.group(1)), name))
    spans: dict[int, list[tuple[str, int, int]]] = {
        layer: [] for layer in range(num_layers)}
    for shard in sorted(wanted):
        path = os.path.normpath(os.path.join(model_dir, shard))
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError(f"{path} is too short to be a safetensors file")
            (length,) = struct.unpack("<Q", raw)
            if not 0 < length <= min(_SAFETENSORS_HEADER_MAX_BYTES, size - 8):
                raise ValueError(f"{path} has an invalid safetensors header length")
            header = json.loads(handle.read(length))
        base = 8 + length
        for layer, name in wanted[shard]:
            if name not in header:
                raise ValueError(f"{path} does not hold tensor {name!r}, which "
                                 "the checkpoint index places in it")
            begin, end = header[name]["data_offsets"]
            if (type(begin) is not int or type(end) is not int
                    or not 0 <= begin <= end <= size - base):
                raise ValueError(f"{path} tensor {name!r} has an invalid span")
            if end > begin:
                spans[layer].append((path, base + begin, base + end))
    empty = [layer for layer, rows in spans.items() if not rows]
    if empty:
        raise ValueError(
            f"no source tensor is named {checkpoint_layers_prefix}{empty[0]}.*: "
            "wrong prefix or wrong layer count, refusing")
    return {layer: sorted(rows) for layer, rows in spans.items()}


def uncovered_source_spans(entries: Sequence[Mapping],
                           spans: Sequence[tuple[str, int, int]],
                           ) -> list[tuple[str, int, int]]:
    """The ``spans`` that no single one of ``entries`` covers outright.

    One entry has to cover a tensor's span whole: the staged reader serves a
    span from the one staged range that contains it and reads a straddling
    span from the pool (``residency_shard_reader``), which the strict tier
    policy refuses. Two entries that together cover a span do not cover it.
    """
    by_path: dict[str, list[tuple[int, int]]] = {}
    for entry in entries:
        by_path.setdefault(os.path.normpath(entry["path"]), []).append(
            (entry["offset"], entry["offset"] + entry["bytes"]))
    return [(path, start, end) for path, start, end in spans
            if not any(low <= start and end <= high
                       for low, high in by_path.get(os.path.normpath(path), ()))]


def complete_source_extent(entries: Sequence[Mapping],
                           spans: Sequence[tuple[str, int, int]], *,
                           taken: set[tuple[str, int]], where: str) -> list[dict]:
    """The entries that make ``entries`` cover every one of ``spans``.

    Each run of uncovered tensors that touch or overlap becomes one entry, from
    the first tensor's own file offset to the last one's end. The offset is
    derived, not aligned: PrismaBuild refuses a manifest that repeats a
    ``(path, offset)``, and a run that begins in a shard's first MiB would
    otherwise land on the header entry an earlier phase already holds at
    ``(path, 0)``. That collision is how the 512 campaign's parent manifest
    lost the tails of layers 9, 19, 29 and 39 (PQ #898). ``taken`` is every
    ``(path, offset)`` the manifest already uses; it is updated here.
    """
    runs: list[list] = []
    for path, start, end in sorted(uncovered_source_spans(entries, spans)):
        if runs and runs[-1][0] == path and start <= runs[-1][2]:
            runs[-1][2] = max(runs[-1][2], end)
        else:
            runs.append([path, start, end])
    added = []
    for path, start, end in runs:
        if (path, start) in taken:
            raise ValueError(
                f"{where}: an entry already starts at {path}:{start} and does "
                "not cover the tensors that start there; refusing to repeat a "
                "(path, offset)")
        taken.add((path, start))
        added.append(dict(path=path, offset=start, bytes=end - start, sha256=None))
    return added


def build_adjoint_manifest(plan: Mapping, parent_manifest: Mapping,
                           ranges: Mapping[str, dict] | None = None, *,
                           plan_path: str, plan_sha256: str, prepared_path: str,
                           prepared_sha256: str, parent_manifest_sha256: str,
                           output_root: str,
                           layer_source_spans: Mapping[int, Sequence] | None = None,
                           ) -> dict:
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

    ``layer_source_spans`` (``read_layer_source_spans``) is what the reader
    will read. With it, each layer's extent is completed against those spans
    and the finished manifest is refused unless it covers all of them: the
    parent's layer phases are a claim, and a claim with a hole in it otherwise
    surfaces hours into a run, as a strict-tier refusal at the first layer
    whose tail the parent dropped (PQ #898). Without it nothing is added and
    nothing is checked, and the manifest is byte-identical to what this
    function always built.
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
    # Every (path, offset) the finished manifest will hold before completion:
    # the head's and every layer's, so an added entry cannot land on one a
    # later layer brings.
    # Spelled the way the reader-side spans are, so the collision check in
    # ``complete_source_extent`` compares like with like.
    taken = {(os.path.normpath(entry["path"]), entry["offset"])
             for entry in manifest_entries}
    taken.update(
        (os.path.normpath(entry["path"]), entry["offset"])
        for layer in layers
        for entry in entries[rows[f"layer-{layer}"]["entry_begin"]:
                             rows[f"layer-{layer}"]["entry_end"]]
        if type(entry.get("path")) is str and entry["path"].startswith(prefix))
    completed: list[dict] = []
    pending: list[list[dict]] = []
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
        if layer_source_spans is not None:
            if layer not in layer_source_spans:
                raise ValueError(
                    f"phase layer-{layer} has no source spans: gap, refusing")
            pending.append(complete_source_extent(
                group, layer_source_spans[layer], taken=taken,
                where=f"phase layer-{layer}"))
    # Added entries go after every entry the parent gave, so each of those
    # keeps the index it has without the reader's spans.
    for layer, run, completion in zip(layers, layer_index_runs, pending):
        run.extend(range(len(manifest_entries),
                         len(manifest_entries) + len(completion)))
        manifest_entries.extend(completion)
        completed.extend(
            dict(layer=layer, path=entry["path"], offset=entry["offset"],
                 bytes=entry["bytes"]) for entry in completion)
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
    if layer_source_spans is not None:
        for layer, run in zip(layers, layer_index_runs):
            missing = uncovered_source_spans(
                [manifest_entries[index] for index in run],
                layer_source_spans[layer])
            if missing:
                path, begin, end = missing[0]
                raise ValueError(
                    f"phase layer-{layer} leaves {len(missing)} source span(s) "
                    f"undeclared, first {path}:[{begin}, {end}): gap, refusing")
    unique_bytes = sum(entry["bytes"] for entry in manifest_entries)
    completion_block = {} if layer_source_spans is None else {
        "source_completion": {
            "schema": SOURCE_COMPLETION_SCHEMA,
            "layers_checked": len(layers),
            "source_spans": sum(len(layer_source_spans[layer]) for layer in layers),
            "added": completed}}
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
            **completion_block,
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


#: The quantum entry point consuming a boundary readset manifest.
QUANTUM_ENTRY_POINT = "prismaquant.joint_cost_quantum"


def quantum_boundary_read_phase_names(chain_layers: Sequence[int], layer: int,
                                      *, batch_windows: int, n_probes: int,
                                      replay_windows: int) -> tuple[str, ...]:
    """The frozen quantum bulk-read order (PQ #848): the checkpoint plane
    first (one whole-plane load, RAM-resident for the action), then each
    chain layer's boundary entries once per probe in batch windows
    (``render_free_layer_roll`` opens a fresh prefetch window set inside
    every probe pass), then the quantum's own boundary entries once per
    (replay window, probe) in batch windows (``replay_backward`` opens a
    fresh boundary iterator per active probe per retained window).

    Names are manifest-local: ``checkpoint``, ``chain-{boundary:03d}`` with
    ``-p{probe}-w{window}`` suffixes, ``replay-{window:02d}`` with
    ``-p{probe}-w{window}`` suffixes. Repeats across phases are the v2
    repeated-read mechanism -- a staging contract declares exactly this
    list, and resume only ever reads a subset of it (completed windows are
    skipped), never more.
    """
    chain = [int(c) for c in chain_layers]
    if any(type(c) is not int or isinstance(c, bool) for c in chain_layers):
        raise ValueError("chain layers must be integers, "
                         f"not {list(chain_layers)!r}")
    if len(set(chain)) != len(chain):
        raise ValueError("chain layers repeat: refusing")
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"a boundary readset needs a layer, not {layer!r}")
    for label, value in (("batch windows", batch_windows),
                         ("probe count", n_probes),
                         ("replay windows", replay_windows)):
        if type(value) is not int or isinstance(value, bool) or value < 1:
            raise ValueError(f"{label} must be positive, not {value!r}")
    names = ["checkpoint"]
    for boundary in chain:
        for probe in range(n_probes):
            names.extend(f"chain-{boundary:03d}-p{probe}-w{window:02d}"
                         for window in range(batch_windows))
    for window_index in range(replay_windows):
        for probe in range(n_probes):
            names.extend(
                f"replay-{window_index:02d}-p{probe}-w{window:02d}"
                for window in range(batch_windows))
    return tuple(names)


def _manifest_entry_from_exact(exact: Mapping, *, where: str) -> dict:
    """A v2 data-manifest entry for one sealed exact record.

    Carries the writer's path, wire byte length and digest -- the triple a
    staging contract admits and verifies without rehashing payloads. Any
    malformed record is a refusal, never a skipped file.
    """
    if not isinstance(exact, dict):
        raise ValueError(f"{where} is not an exact entry record: refusing")
    path = exact.get("path")
    digest = exact.get("sha256")
    size = exact.get("file_bytes")
    if type(path) is not str or not path:
        raise ValueError(f"{where} names no entry path: refusing")
    if type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{where} carries no entry digest: refusing")
    if type(size) is not int or isinstance(size, bool) or size <= 0:
        raise ValueError(f"{where} carries no entry byte length: refusing")
    return {"path": path, "offset": 0, "bytes": size, "sha256": digest}


def _collect_quantum_bulk_entries(*, receipt: Mapping,
                                  checkpoint_boundary: int,
                                  needed: Sequence[int]) -> dict:
    """The staged produced triples shared by the quantum readset builders.

    Returns checkpoint entry indices, per-boundary index runs, the sealed
    prefetch window, batch windows and the flat entries list -- one index
    space both the boundary readset and the executable manifest seal
    against. Entry order is deterministic: the checkpoint plane first,
    then needed boundaries ascending, batches ascending. A path that
    resolves twice, a boundary without entries, or uneven batch counts
    refuses; only callers decide the phase table laid over these indices.
    """
    boundary_table = receipt.get("boundary_entries", {})
    if not isinstance(boundary_table, dict):
        raise ValueError("the adjoint receipt carries no boundary entries: "
                         "refusing")
    storage_block = receipt.get("boundary_storage")
    policy = storage_block.get("policy", {}) \
        if isinstance(storage_block, dict) else {}
    prefetch_batches = policy.get("prefetch_batches")
    if type(prefetch_batches) is not int or isinstance(
            prefetch_batches, bool) or prefetch_batches < 1:
        raise ValueError("the receipt seals no prefetch batch window: refusing")
    manifest_entries: list[dict] = []
    seen_paths: set[str] = set()

    def _take(exact: Mapping, *, where: str) -> int:
        entry = _manifest_entry_from_exact(exact, where=where)
        if entry["path"] in seen_paths:
            raise ValueError(f"duplicate staged path {entry['path']}: refusing")
        seen_paths.add(entry["path"])
        manifest_entries.append(entry)
        return len(manifest_entries) - 1

    checkpoint_record = None
    for entry in receipt.get("checkpoints", []):
        if isinstance(entry, dict) and int(entry.get("boundary", -1)) \
                == checkpoint_boundary:
            checkpoint_record = entry
            break
    if checkpoint_record is None:
        raise ValueError(
            "the adjoint receipt does not carry the checkpoint boundary "
            f"{checkpoint_boundary}: refusing")
    checkpoint_indices: list[int] = []
    for exact in list(checkpoint_record.get("activation_entries", [])) \
            + list(checkpoint_record.get("shared_state_entries", [])):
        checkpoint_indices.append(_take(
            exact, where=f"checkpoint boundary {checkpoint_boundary}"))
    if not checkpoint_indices:
        raise ValueError("the checkpoint plane is empty: refusing")
    batch_counts = set()
    boundary_runs: dict[int, list[int]] = {}
    for boundary in needed:
        rows = boundary_table.get(str(boundary))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"the receipt carries no boundary {boundary} "
                             "entries: refusing")
        run = [_take(exact, where=f"boundary {boundary} entry {index}")
               for index, exact in enumerate(rows)]
        boundary_runs[boundary] = run
        batch_counts.add(len(run))
    if len(batch_counts) != 1:
        raise ValueError("needed boundaries hold different batch counts: "
                         "mixed campaign, refusing")
    batch_total = batch_counts.pop()
    return {"entries": manifest_entries,
            "checkpoint_indices": checkpoint_indices,
            "boundary_runs": boundary_runs,
            "prefetch_batches": prefetch_batches,
            "batch_total": batch_total,
            "batch_windows": (batch_total + prefetch_batches - 1)
                             // prefetch_batches}
    """A v2 data-manifest entry for one sealed exact record.

    Carries the writer's path, wire byte length and digest -- the triple a
    staging contract admits and verifies without rehashing payloads. Any
    malformed record is a refusal, never a skipped file.
    """
    if not isinstance(exact, dict):
        raise ValueError(f"{where} is not an exact entry record: refusing")
    path = exact.get("path")
    digest = exact.get("sha256")
    size = exact.get("file_bytes")
    if type(path) is not str or not path:
        raise ValueError(f"{where} names no entry path: refusing")
    if type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{where} carries no entry digest: refusing")
    if type(size) is not int or isinstance(size, bool) or size <= 0:
        raise ValueError(f"{where} carries no entry byte length: refusing")
    return {"path": path, "offset": 0, "bytes": size, "sha256": digest}


def build_quantum_boundary_readset(record: Mapping, receipt: Mapping, *,
                                   strided_boundaries: Sequence[int],
                                   n_probes: int) -> dict:
    """The quantum's real bulk readset as a NEW immutable v2 manifest.

    Derived post-capture from the completed adjoint receipt, the record's
    sealed ``chain_layers``/``layer``/``checkpoint_boundary``/``windows``,
    and ``n_probes`` -- which is caller-responsible: the post-capture regen
    passes the sealed plan value, never a knob, and the binder requires
    that same trusted count independently rather than trusting the
    manifest's attestation. The checkpoint plane (activation +
    shared-state entries -- file lease: one whole-plane load, released
    after decoded buffers exist; decoded tensors stay RAM-resident for the
    action), then each chain layer's boundary entries once per probe in
    batch windows (prefetch-lease per probe pass), then the quantum's own
    boundary entries once per (replay window, probe) in batch windows.
    Only a receipt whose status is ``complete`` derives anything here; a
    missing, running or failed capture refuses before any derivation.
    Every entry carries the sealed path/length/digest triple, so a staging
    contract admits and verifies the corpus with no payload rehash and no
    new cache. Repeats across phases are the v2 repeated-read mechanism;
    resume only ever reads a subset (completed windows are skipped).

    The receipt is validated through :func:`bind_adjoint_receipt` (campaign
    scope, plan/prepared digests, stride marks) and its canonical digest is
    sealed into the annotations; the record's chain is checked against the
    single ``chain_layers_for`` owner. Old unbound records, slices and the
    parent identity are untouched -- this manifest is a new generation with
    a fresh digest, never a mutation of a sealed action.
    """
    from .joint_adjoint_checkpoints import chain_layers_for

    if not isinstance(receipt, dict) or receipt.get("status") != "complete":
        raise ValueError("the adjoint receipt is not a completed capture: "
                         "this post-capture path derives nothing from a "
                         "missing, running or failed capture, refusing")
    if not isinstance(record, dict):
        raise ValueError("a quantum record must be an object: refusing")
    layer = record.get("layer")
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError("a quantum record names no layer: refusing")
    adjoint = record.get("adjoint")
    if not isinstance(adjoint, dict):
        raise ValueError("a quantum record carries no adjoint block: refusing")
    checkpoint_boundary = adjoint.get("checkpoint_boundary")
    if type(checkpoint_boundary) is not int or isinstance(
            checkpoint_boundary, bool):
        raise ValueError("a quantum record names no checkpoint boundary: "
                         "refusing")
    chain = adjoint.get("chain_layers")
    if not isinstance(chain, list) or any(
            type(c) is not int or isinstance(c, bool) for c in chain):
        raise ValueError("a quantum record names no chain layers: refusing")
    if tuple(chain) != chain_layers_for(checkpoint_boundary, layer):
        raise ValueError(
            f"quantum {record.get('quantum_id')!r} chain {chain!r} is not the "
            "sealed stride chain: refusing")
    campaign = record.get("campaign")
    if not isinstance(campaign, dict):
        raise ValueError("a quantum record carries no campaign block: refusing")
    for key in ("plan_path", "plan_sha256", "prepared_path", "prepared_sha256",
                "read_manifest_sha256", "campaign_scope"):
        if not campaign.get(key):
            raise ValueError(f"a quantum record seals no campaign {key}: "
                             "refusing")
    receipt_sha256 = bind_adjoint_receipt(
        receipt, plan_sha256=campaign["plan_sha256"],
        prepared_sha256=campaign["prepared_sha256"],
        scope=campaign["campaign_scope"], checkpoints=strided_boundaries)
    if type(n_probes) is not int or isinstance(n_probes, bool) \
            or n_probes < 1:
        raise ValueError("a boundary readset needs a sealed probe count, "
                         f"not {n_probes!r}")
    replay_windows = record.get("windows")
    if not isinstance(replay_windows, list) or not replay_windows:
        raise ValueError("a quantum record seals no replay windows: refusing")
    needed = sorted(set(chain) | {layer})
    bulk = _collect_quantum_bulk_entries(
        receipt=receipt, checkpoint_boundary=checkpoint_boundary,
        needed=needed)
    manifest_entries = bulk["entries"]
    checkpoint_indices = bulk["checkpoint_indices"]
    boundary_runs = bulk["boundary_runs"]
    prefetch_batches = bulk["prefetch_batches"]
    batch_windows = bulk["batch_windows"]
    read_phases: list[dict] = []
    cumulative = 0

    def _seal_phase(name: str, indices: list[int]) -> None:
        nonlocal cumulative
        size = sum(manifest_entries[index]["bytes"] for index in indices)
        if size <= 0:
            raise ValueError(f"read phase {name} is empty: refusing")
        cumulative += size
        read_phases.append({"name": name, "entry_indices": list(indices),
                            "bytes": size, "cumulative_bytes": cumulative})

    _seal_phase("checkpoint", checkpoint_indices)
    for boundary in chain:
        run = boundary_runs[boundary]
        for probe in range(n_probes):
            for window in range(batch_windows):
                _seal_phase(f"chain-{boundary:03d}-p{probe}-w{window:02d}",
                            run[window * prefetch_batches:
                                (window + 1) * prefetch_batches])
    own_run = boundary_runs[layer]
    for window_index in range(len(replay_windows)):
        for probe in range(n_probes):
            for window in range(batch_windows):
                _seal_phase(f"replay-{window_index:02d}-p{probe}-w{window:02d}",
                            own_run[window * prefetch_batches:
                                    (window + 1) * prefetch_batches])
    names = [phase["name"] for phase in read_phases]
    if names != list(quantum_boundary_read_phase_names(
            chain, layer, batch_windows=batch_windows, n_probes=n_probes,
            replay_windows=len(replay_windows))):
        raise ValueError("the boundary read plan is not the frozen reader "
                         "order: refusing")
    unique_bytes = sum(entry["bytes"] for entry in manifest_entries)
    return {
        "schema": MANIFEST_SCHEMA_V2,
        "produced_by": {"tool": "prismaquant/joint_layer_quanta.py",
                        "entry_point": QUANTUM_ENTRY_POINT,
                        "plan": campaign["plan_path"],
                        "plan_sha256": campaign["plan_sha256"]},
        "mount_prefix": "/mnt/shared",
        "entries": manifest_entries,
        "entry_count": len(manifest_entries),
        "total_bytes": unique_bytes,
        "annotations": {
            "entry_point": QUANTUM_ENTRY_POINT,
            "quantum_id": record.get("quantum_id"),
            "quantum_layer": layer,
            "checkpoint_boundary": checkpoint_boundary,
            "chain_layers": list(chain),
            "n_probes": n_probes,
            "batch_windows": batch_windows,
            "replay_windows": len(replay_windows),
            "receipt_sha256": receipt_sha256,
            "plan_sha256": campaign["plan_sha256"],
            "prepared_sha256": campaign["prepared_sha256"],
            "parent_manifest_sha256": campaign["read_manifest_sha256"],
            "campaign_scope": campaign["campaign_scope"],
        },
        "read_plan": {"phases": read_phases, "read_bytes": cumulative},
    }


def bound_readset_directory(output_root: str, *,
                            metadata_root: str | None = None) -> str:
    """The directory holding newly bound readset manifests (PQ #884).

    One derivation for both callers (binders and emitters), so a bound path
    can never disagree with the record that names it. Default (no
    ``metadata_root``): the historical
    ``{output_root}/layer-quanta/adjoint/bound-readsets``. With an explicit
    control-metadata generation root the bound readsets -- themselves
    control metadata, never stage-A data -- land at the SAME relative
    control layout inside that namespace:
    ``{metadata_root}/adjoint/bound-readsets``. The relative control layout
    is identical in both branches, so ``metadata_root ==
    {output_root}/layer-quanta`` reproduces the default paths exactly,
    while ``adjoint.boundary_artifacts`` (the stage-A DATA directory) keeps
    deriving from the data output root alone and never points here.
    """
    if metadata_root is not None:
        return _canonical_control_root(metadata_root) + "/adjoint/bound-readsets"
    if type(output_root) is not str or not output_root.startswith("/"):
        raise ValueError("an output root must be absolute: refusing")
    return output_root.rstrip("/") + "/layer-quanta/adjoint/bound-readsets"


def bind_quantum_boundary_readset(record: Mapping, receipt: Mapping, *,
                                  manifest: Mapping, manifest_path: str,
                                  manifest_sha256: str, output_root: str,
                                  strided_boundaries: Sequence[int],
                                  n_probes: int,
                                  metadata_root: str | None = None) -> dict:
    """Bind a sealed boundary readset manifest to a NEW record generation.

    Returns a deep copy of ``record`` carrying a ``boundary_readset`` block
    and a recomputed ``identity_sha256`` (the existing canonical owner, over
    every top-level field but the identity itself -- the same recomputation
    ``check_quantum_for_campaign`` and the consumer's
    ``verify_quantum_identity`` enforce, so a bound record passes both);
    the input record is never mutated.

    Refuses unless every identity binds exactly: the completed-capture
    receipt; the input record itself, reverified through the existing
    ``check_quantum_for_campaign`` owner against its own sealed campaign
    and bound receipt BEFORE any mutation (a tampered field with a stale
    identity refuses -- binding never blesses edits by recomputing); the
    record's schema, layer, producer-constrained quantum id, campaign
    digests/scope, and its already bound adjoint receipt (another receipt
    for the same layer refuses); the manifest path is exactly the
    producer-named bound path under the output root (the quantum id enters
    no free-form pathname). ``n_probes`` is an explicit trusted input --
    the regen passes the sealed plan value, and the manifest must attest
    that same count (never the manifest attesting to itself). The manifest
    itself is proven, not trusted:
    its schema, entries, phases, counts and schedule must equal what
    :func:`build_quantum_boundary_readset` derives from this record and
    receipt -- a different path/bytes/digest set with matching annotations
    and a consistent rehash still refuses, because the triples must
    originate from the bound receipt.
    """
    import copy
    import os
    if not isinstance(record, dict):
        raise ValueError("a quantum record must be an object: refusing")
    if record.get("schema") != LAYER_QUANTUM_SCHEMA:
        raise ValueError("a quantum record has a foreign schema: refusing")
    layer = record.get("layer")
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError("a quantum record names no layer: refusing")
    if record.get("quantum_id") != quantum_id(layer):
        raise ValueError(
            f"quantum id {record.get('quantum_id')!r} does not name layer "
            f"{layer}: refusing")
    adjoint = record.get("adjoint")
    if not isinstance(adjoint, dict):
        raise ValueError("a quantum record carries no adjoint block: refusing")
    bound_receipt = adjoint.get("receipt_sha256")
    if type(bound_receipt) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", bound_receipt):
        raise ValueError(
            f"quantum {record.get('quantum_id')!r} is unbound (pre-A): "
            "re-seal against the stage-A receipt before binding a readset, "
            "refusing")
    campaign = record.get("campaign")
    if not isinstance(campaign, dict):
        raise ValueError("a quantum record carries no campaign block: refusing")
    for key in ("plan_path", "plan_sha256", "prepared_path", "prepared_sha256",
                "read_manifest_sha256", "campaign_scope"):
        if not campaign.get(key):
            raise ValueError(f"a quantum record seals no campaign {key}: "
                             "refusing")
    if type(n_probes) is not int or isinstance(n_probes, bool) \
            or n_probes < 1:
        raise ValueError("a boundary readset needs a trusted sealed probe "
                         f"count, not {n_probes!r}")
    if type(output_root) is not str or not output_root.startswith("/"):
        raise ValueError("an output root must be absolute: refusing")
    expected_path = (
        bound_readset_directory(output_root, metadata_root=metadata_root)
        + f"/{quantum_id(layer)}.boundary-readset.json.gz")
    if manifest_path != expected_path or os.path.normpath(
            manifest_path) != manifest_path or ".." in manifest_path.split("/"):
        raise ValueError(
            f"a boundary readset path must be exactly {expected_path}: "
            "refusing")
    # The input record is reverified through the existing owner BEFORE any
    # mutation: a tampered field with a stale identity refuses here, and
    # binding never blesses edits by recomputing.
    check_quantum_for_campaign(
        record, {**campaign, "adjoint_receipt_sha256": bound_receipt})
    if not isinstance(manifest, dict):
        raise ValueError("a boundary readset manifest must be an object: "
                         "refusing")
    if manifest.get("schema") != MANIFEST_SCHEMA_V2:
        raise ValueError("a boundary readset manifest has a foreign schema: "
                         "refusing")
    annotations = manifest.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError("a boundary readset manifest has no annotations: "
                         "refusing")
    if annotations.get("receipt_sha256") != bound_receipt:
        raise ValueError(
            f"quantum {record.get('quantum_id')!r} binds another stage-A "
            "receipt: refusing")
    if annotations.get("n_probes") != n_probes:
        raise ValueError(
            f"the boundary readset attests another probe count "
            f"{annotations.get('n_probes')!r}: refusing")
    try:
        expected = build_quantum_boundary_readset(
            record, receipt, strided_boundaries=strided_boundaries,
            n_probes=n_probes)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise ValueError("the boundary readset does not derive from its "
                         f"record and receipt: refusing ({exc})") from exc
    if manifest != expected:
        raise ValueError(
            "the boundary readset entries, phases or counts do not "
            "originate from the bound receipt: refusing")
    wire = seal_manifest_bytes(manifest)
    if hashlib.sha256(wire).hexdigest() != manifest_sha256:
        raise ValueError("the boundary readset digest does not reproduce "
                         "from its manifest wire: refusing")
    fresh = copy.deepcopy(record)
    fresh["boundary_readset"] = {
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "entry_count": manifest["entry_count"],
        "total_bytes": manifest["total_bytes"],
        "read_bytes": manifest["read_plan"]["read_bytes"],
        "phases": [phase["name"]
                   for phase in manifest["read_plan"]["phases"]],
        "receipt_sha256": bound_receipt,
    }
    body = {key: value for key, value in fresh.items()
            if key != "identity_sha256"}
    fresh["identity_sha256"] = canonical_sha256(
        body, where=f"quantum record {fresh.get('quantum_id')}")
    return fresh


#: The executable-manifest phase staging the whole checkpoint plane load.
CHECKPOINT_LOAD_PHASE = "checkpoint-load"


def executable_source_phase_name(layer: int) -> str:
    """The executable-manifest phase staging one chain layer's source."""
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"a source phase needs a layer, not {layer!r}")
    return f"chain-{layer:03d}-source"


def executable_bound_phase_name(layer: int) -> str:
    """The executable-manifest phase staging one chain layer's boundary."""
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"a bound phase needs a layer, not {layer!r}")
    return f"chain-{layer:03d}-bound"


def executable_own_source_phase_name(layer: int) -> str:
    """The executable-manifest phase staging the quantum's own source."""
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"an own-source phase needs a layer, not {layer!r}")
    return f"own-{layer:03d}-source"


def executable_replay_phase_name(window_index: int | None, probe: int) -> str:
    """The executable-manifest phase for one replay probe pass.

    ``None`` names window zero: the zero-pending resume path still reads
    the own boundary once per probe with no window loop running, and those
    reads stage under the window-zero phases by this convention.
    """
    window = 0 if window_index is None else int(window_index)
    if window < 0 or type(probe) is not int or isinstance(probe, bool) \
            or probe < 0:
        raise ValueError(
            f"a replay phase needs a window and probe, not "
            f"{window_index!r}/{probe!r}")
    return f"replay-{window:02d}-p{probe}"


def quantum_executable_phase_names(chain_layers: Sequence[int], layer: int,
                                    *, n_probes: int,
                                    replay_windows: int) -> tuple[str, ...]:
    """The frozen executable staging order (PQ #862): calibration head,
    the checkpoint plane once, then per chain layer descending its source
    extents and its boundary entries, the quantum's own source extents,
    then per retained window and probe the replay boundary reads.

    Every name is reported by the runtime through the existing semantic
    reporter as its bytes are consumed -- a staging contract declares
    exactly this list. Chunk names are deliberately not staging phases:
    durable units keep their existing currency and commit under the read
    phase in effect when the journal lands.
    """
    chain = [int(c) for c in chain_layers]
    if any(type(c) is not int or isinstance(c, bool) for c in chain_layers):
        raise ValueError("chain layers must be integers, "
                         f"not {list(chain_layers)!r}")
    if len(set(chain)) != len(chain):
        raise ValueError("chain layers repeat: refusing")
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"an executable readset needs a layer, not {layer!r}")
    for label, value in (("probe count", n_probes),
                         ("replay windows", replay_windows)):
        if type(value) is not int or isinstance(value, bool) or value < 1:
            raise ValueError(f"{label} must be positive, not {value!r}")
    names = ["head", CHECKPOINT_LOAD_PHASE]
    for boundary in chain:
        names.append(executable_source_phase_name(boundary))
        names.append(executable_bound_phase_name(boundary))
    names.append(executable_own_source_phase_name(layer))
    for window_index in range(replay_windows):
        for probe in range(n_probes):
            names.append(executable_replay_phase_name(window_index, probe))
    return tuple(names)


def _is_source_model_path(path: object, model_root: str) -> bool:
    """Whether a parent entry path is checkpoint source (PQ #909).

    Component-boundary match against the sealed plan's source model
    directory: the root itself or anything under ``root/``, compared on
    normpath'd paths. A sibling such as ``root + "-evil"`` never matches,
    which a bare ``startswith`` would admit; this is the stage-A source
    selection (``build_adjoint_manifest``) spelled as the boundary it is.
    A non-string or empty path is never source.
    """
    if type(path) is not str or not path:
        return False
    candidate = os.path.normpath(path)
    base = os.path.normpath(model_root)
    return candidate == base or candidate.startswith(base + os.sep)


def _source_extent_entries(parent_manifest: Mapping, *,
                           layers: Sequence[int],
                           source_model_root: str | None = None
                           ) -> dict[int, list[dict]]:
    """The parent manifest's per-layer source-extent entries, re-based.

    Entry form is the parent's own ``{path, offset, bytes, sha256}`` --
    file coordinates, never readdressed. A layer phase missing from the
    parent table, or entries disagreeing with its byte range, refuses.

    With ``source_model_root`` (PQ #909), each layer phase keeps only the
    entries under the sealed plan's source model directory: rendered-cache
    files the parent also tiles stay under the produced-output lifecycle
    and never stage in a source phase. The agreement check still runs on
    the whole tiled group first, so the parent tiling itself is never
    reshaped here. A phase left with no source entry refuses. Without a
    root every entry is kept and the historical bytes reproduce unchanged.
    """
    if source_model_root is not None:
        if type(source_model_root) is not str \
                or not source_model_root \
                or not os.path.isabs(source_model_root):
            raise ValueError("a source model root must be an absolute "
                             "directory: refusing")
    rows = {row["name"]: row for row in phase_ranges(parent_manifest)}
    entries = parent_manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("the parent manifest has no entries: refusing")
    runs: dict[int, list[dict]] = {}
    for layer in layers:
        row = rows.get(f"layer-{int(layer)}")
        if row is None:
            raise ValueError(f"the parent manifest has no layer-{int(layer)} "
                             "phase: gap, refusing")
        group = entries[row["entry_begin"]:row["entry_end"]]
        if sum(entry["bytes"] for entry in group) != \
                row["end_bytes"] - row["start_bytes"]:
            raise ValueError(f"phase layer-{int(layer)} bytes disagree with "
                             "its entries: refusing")
        if not group:
            raise ValueError(f"phase layer-{int(layer)} holds no source "
                             "extent: gap, refusing")
        if source_model_root is not None:
            group = [entry for entry in group
                     if _is_source_model_path(entry.get("path"),
                                              source_model_root)]
            if not group:
                raise ValueError(
                    f"phase layer-{int(layer)} holds no source extent under "
                    f"{source_model_root}: gap, refusing")
        runs[int(layer)] = [
            dict(path=entry["path"], offset=entry["offset"],
                 bytes=entry["bytes"], sha256=entry.get("sha256"))
            for entry in group]
    return runs


def build_quantum_executable_manifest(
        record: Mapping, receipt: Mapping, parent_manifest: Mapping, *,
        strided_boundaries: Sequence[int], n_probes: int, calib: Mapping,
        render_prerequisite: Mapping,
        layer_source_spans: Mapping[int, Sequence] | None = None,
        source_model_root: str | None = None) -> dict:
    """ONE executable v2 read manifest for a quantum row (PQ #862).

    Derived post-capture from the completed adjoint receipt, the record's
    sealed chain/layer/checkpoint/windows, the sealed probe count
    (caller-responsible: the regen passes the sealed plan value), the
    parent manifest's source extents for the chain and own layers, one
    calibration entry, and the render prerequisite. Entries are deduplicated
    exact triples keyed by ``(path, offset)`` -- a key with contradictory
    bytes or digest refuses. Phases follow
    :func:`quantum_executable_phase_names` in true consumption order, with
    ``read_bytes`` counting staged bytes once per pinning phase; repeats
    across phases are the v2 repeated-read mechanism (the own boundary
    corpus repeats across replay windows because every window re-reads
    it). Resume only ever reads a subset.

    With ``layer_source_spans`` (PQ #900), complete and check every chain
    and own source phase against the actual reader's tensor spans, using
    the same completion rule as stage A. Coverage must hold in that phase
    and in one entry per tensor. Added entries follow the existing entries;
    the parent, slice tiling, chunks and campaign identity never change.
    Without spans the historical manifest bytes reproduce unchanged.

    With ``source_model_root`` (PQ #909), chain and own source phases keep
    only the parent entries under the sealed plan's source model directory
    (path-component boundary, the stage-A selection): rendered-cache files
    stay under the produced-output lifecycle and never stage here. Without
    it the historical manifest bytes reproduce unchanged.

    Rendered-weight bytes are NOT staged here: the PWC retained-window
    reads that need them name the PB732 produced-output scope in
    ``annotations.render_prerequisite`` (production pickle digest plus
    roster digest, both sealed inputs) -- never a silent HDD read.
    The annotation names the missing dependency; it does not implement
    staging and proves no capability. No accepted PB produced-output
    binding validator exists yet (the PB732/735 stacks are still
    unaccepted), so every manifest this builder seals carries
    ``binding: None`` and is sequencing-only: the dispatcher refuses all
    executable rows with a typed unsupported-binding refusal, even for a
    plausible-looking prerequisite dictionary. A non-None ``binding``
    input refuses here rather than sealing fiction. Only a receipt whose
    status is ``complete`` derives anything here.
    """
    from .joint_adjoint_checkpoints import chain_layers_for

    if not isinstance(receipt, dict) or receipt.get("status") != "complete":
        raise ValueError("the adjoint receipt is not a completed capture: "
                         "this post-capture path derives nothing from a "
                         "missing, running or failed capture, refusing")
    if not isinstance(record, dict):
        raise ValueError("a quantum record must be an object: refusing")
    layer = record.get("layer")
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError("a quantum record names no layer: refusing")
    adjoint = record.get("adjoint")
    if not isinstance(adjoint, dict):
        raise ValueError("a quantum record carries no adjoint block: refusing")
    checkpoint_boundary = adjoint.get("checkpoint_boundary")
    if type(checkpoint_boundary) is not int or isinstance(
            checkpoint_boundary, bool):
        raise ValueError("a quantum record names no checkpoint boundary: "
                         "refusing")
    chain = adjoint.get("chain_layers")
    if not isinstance(chain, list) or any(
            type(c) is not int or isinstance(c, bool) for c in chain):
        raise ValueError("a quantum record names no chain layers: refusing")
    if tuple(chain) != chain_layers_for(checkpoint_boundary, layer):
        raise ValueError(
            f"quantum {record.get('quantum_id')!r} chain {chain!r} is not the "
            "sealed stride chain: refusing")
    campaign = record.get("campaign")
    if not isinstance(campaign, dict):
        raise ValueError("a quantum record carries no campaign block: refusing")
    for key in ("plan_path", "plan_sha256", "prepared_path", "prepared_sha256",
                "read_manifest_sha256", "campaign_scope"):
        if not campaign.get(key):
            raise ValueError(f"a quantum record seals no campaign {key}: "
                             "refusing")
    if type(n_probes) is not int or isinstance(n_probes, bool) \
            or n_probes < 1:
        raise ValueError("an executable readset needs a sealed probe count, "
                         f"not {n_probes!r}")
    replay_windows = record.get("windows")
    if not isinstance(replay_windows, list) or not replay_windows:
        raise ValueError("a quantum record seals no replay windows: refusing")
    if not isinstance(calib, dict):
        raise ValueError("an executable readset needs its calibration entry: "
                         "refusing")
    calib_path = calib.get("path")
    calib_bytes = calib.get("bytes")
    calib_sha256 = calib.get("sha256")
    if type(calib_path) is not str or not calib_path.startswith("/"):
        raise ValueError("the calibration entry names no absolute path: "
                         "refusing")
    if type(calib_bytes) is not int or isinstance(calib_bytes, bool) \
            or calib_bytes <= 0:
        raise ValueError("the calibration entry carries no byte length: "
                         "refusing")
    if type(calib_sha256) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", calib_sha256):
        raise ValueError("the calibration entry carries no digest: refusing")
    if not isinstance(render_prerequisite, dict) or \
            render_prerequisite.get("scope") != "pb732" or \
            not render_prerequisite.get("production_pkl_sha256") or \
            not render_prerequisite.get("unit_roster_sha256"):
        raise ValueError("an executable readset names no PB732 render "
                         "prerequisite: refusing")
    if render_prerequisite.get("binding") is not None:
        raise ValueError(
            "an executable readset names a render binding, but no accepted "
            "PB produced-output binding validator exists (PB732/735 stacks "
            "unaccepted): refusing to seal fiction -- manifests are "
            "sequencing-only with binding None")
    prerequisite = {
        "scope": "pb732",
        "production_pkl_sha256": render_prerequisite["production_pkl_sha256"],
        "unit_roster_sha256": render_prerequisite["unit_roster_sha256"],
        "binding": None,
    }
    receipt_sha256 = bind_adjoint_receipt(
        receipt, plan_sha256=campaign["plan_sha256"],
        prepared_sha256=campaign["prepared_sha256"],
        scope=campaign["campaign_scope"], checkpoints=strided_boundaries)
    replay_windows = record.get("windows")
    if not isinstance(replay_windows, list) or not replay_windows:
        raise ValueError("a quantum record seals no replay windows: refusing")
    needed = sorted(set(chain) | {layer})
    bulk = _collect_quantum_bulk_entries(
        receipt=receipt, checkpoint_boundary=checkpoint_boundary,
        needed=needed)
    source_raw = _source_extent_entries(parent_manifest, layers=needed,
                                        source_model_root=source_model_root)
    manifest_entries: list[dict] = []
    by_coordinates: dict[tuple[str, int], dict] = {}

    def _take(entry: Mapping, *, where: str) -> int:
        if not isinstance(entry, dict):
            raise ValueError(f"{where} is not an entry: refusing")
        path = entry.get("path")
        offset = entry.get("offset", 0)
        if type(path) is not str or not path or type(offset) is not int \
                or isinstance(offset, bool) or offset < 0:
            raise ValueError(f"{where} names no file coordinates: refusing")
        key = (path, offset)
        known = by_coordinates.get(key)
        if known is not None:
            if known["bytes"] != entry.get("bytes") or \
                    known.get("sha256") != entry.get("sha256"):
                raise ValueError(f"contradictory entries stage {path} "
                                 f"offset {offset}: refusing")
            return manifest_entries.index(known)
        staged = dict(path=path, offset=offset, bytes=entry["bytes"],
                      sha256=entry.get("sha256"))
        if type(staged["bytes"]) is not int or isinstance(
                staged["bytes"], bool) or staged["bytes"] <= 0:
            raise ValueError(f"{where} carries no byte length: refusing")
        manifest_entries.append(staged)
        by_coordinates[key] = staged
        return len(manifest_entries) - 1

    head_index = _take(
        {"path": calib_path, "offset": 0, "bytes": calib_bytes,
         "sha256": calib_sha256}, where="calibration intake")
    # The bulk collector owns its own index space; remap it into this
    # manifest's unified space (bulk paths are unique, so order is kept).
    index_of: dict[int, int] = {}
    for bulk_index in bulk["checkpoint_indices"] + [
            index for boundary in needed
            for index in bulk["boundary_runs"][boundary]]:
        index_of[bulk_index] = _take(
            bulk["entries"][bulk_index], where="produced bulk entry")
    checkpoint_indices = [index_of[index]
                          for index in bulk["checkpoint_indices"]]
    boundary_runs = {
        boundary: [index_of[index]
                   for index in bulk["boundary_runs"][boundary]]
        for boundary in needed}
    source_runs = {
        boundary: [_take(entry, where=f"layer-{boundary} source extent")
                   for entry in source_raw[boundary]]
        for boundary in needed}
    completed: list[dict] = []
    if layer_source_spans is not None:
        taken = {(os.path.normpath(path), offset)
                 for path, offset in by_coordinates}
        for boundary in needed:
            spans = layer_source_spans.get(boundary)
            if not spans:
                raise ValueError(
                    f"phase layer-{boundary} has no source spans: gap, refusing")
            run = source_runs[boundary]
            added = complete_source_extent(
                [manifest_entries[index] for index in run], spans,
                taken=taken, where=f"phase layer-{boundary}")
            run.extend(_take(entry, where=f"layer-{boundary} completed source")
                       for entry in added)
            completed.extend(dict(layer=boundary, path=entry["path"],
                                  offset=entry["offset"], bytes=entry["bytes"])
                             for entry in added)
            missing = uncovered_source_spans(
                [manifest_entries[index] for index in run], spans)
            if missing:
                path, begin, end = missing[0]
                raise ValueError(
                    f"phase layer-{boundary} leaves {len(missing)} source "
                    f"span(s) undeclared, first {path}:[{begin}, {end}): "
                    "gap, refusing")
    read_phases: list[dict] = []
    cumulative = 0

    def _seal_phase(name: str, indices: list[int]) -> None:
        nonlocal cumulative
        size = sum(manifest_entries[index]["bytes"] for index in indices)
        if size <= 0:
            raise ValueError(f"read phase {name} is empty: refusing")
        cumulative += size
        read_phases.append({"name": name, "entry_indices": list(indices),
                            "bytes": size, "cumulative_bytes": cumulative})

    _seal_phase("head", [head_index])
    _seal_phase(CHECKPOINT_LOAD_PHASE, checkpoint_indices)
    for boundary in chain:
        _seal_phase(executable_source_phase_name(boundary),
                    source_runs[boundary])
        _seal_phase(executable_bound_phase_name(boundary),
                    boundary_runs[boundary])
    _seal_phase(executable_own_source_phase_name(layer), source_runs[layer])
    for window_index in range(len(replay_windows)):
        for probe in range(n_probes):
            _seal_phase(executable_replay_phase_name(window_index, probe),
                        boundary_runs[layer])
    names = [phase["name"] for phase in read_phases]
    if names != list(quantum_executable_phase_names(
            chain, layer, n_probes=n_probes,
            replay_windows=len(replay_windows))):
        raise ValueError("the executable read plan is not the frozen "
                         "consumption order: refusing")
    unique_bytes = sum(entry["bytes"] for entry in manifest_entries)
    return {
        "schema": MANIFEST_SCHEMA_V2,
        "produced_by": {"tool": "prismaquant/joint_layer_quanta.py",
                        "entry_point": QUANTUM_ENTRY_POINT,
                        "plan": campaign["plan_path"],
                        "plan_sha256": campaign["plan_sha256"]},
        "mount_prefix": "/mnt/shared",
        "entries": manifest_entries,
        "entry_count": len(manifest_entries),
        "total_bytes": unique_bytes,
        "annotations": {
            "entry_point": QUANTUM_ENTRY_POINT,
            "quantum_id": record.get("quantum_id"),
            "quantum_layer": layer,
            "checkpoint_boundary": checkpoint_boundary,
            "chain_layers": list(chain),
            "n_probes": n_probes,
            "replay_windows": len(replay_windows),
            "receipt_sha256": receipt_sha256,
            "plan_sha256": campaign["plan_sha256"],
            "prepared_sha256": campaign["prepared_sha256"],
            "parent_manifest_sha256": campaign["read_manifest_sha256"],
            "campaign_scope": campaign["campaign_scope"],
            "calib": {"path": calib_path, "bytes": calib_bytes,
                      "sha256": calib_sha256},
            "render_prerequisite": prerequisite,
            **({"source_completion": {
                "schema": SOURCE_COMPLETION_SCHEMA,
                "layers_checked": len(needed),
                "source_spans": sum(len(layer_source_spans[boundary])
                                    for boundary in needed),
                "added": completed}}
               if layer_source_spans is not None else {}),
        },
        "read_plan": {"phases": read_phases, "read_bytes": cumulative},
    }


def emit_quantum_boundary_readsets(receipt: Mapping,
                                   records: Sequence[Mapping], *,
                                   strided_boundaries: Sequence[int],
                                   n_probes: int,
                                   output_root: str,
                                   metadata_root: str | None = None) -> list[dict]:
    """The post-capture generation path: new records plus their manifests.

    For every record, derives the boundary readset manifest, seals it, and
    binds it to a new record generation under
    ``{output_root}/layer-quanta/adjoint/bound-readsets/`` -- or, with an
    explicit ``metadata_root`` (PQ #884), under that generation namespace's
    ``adjoint/bound-readsets/`` (the same relative control layout), leaving
    the immutable stage-A adjoint artifact tree untouched. Returns one
    ``{"record", "manifest", "manifest_path", "manifest_sha256"}`` per
    quantum, in record order. Emits nothing to disk and mutates nothing:
    the post-capture regen writes the returned bytes and adopts the
    returned records. Duplicate quantum ids or manifest paths refuse whole
    rather than binding half a campaign.
    """
    rows = list(records)
    if not rows:
        raise ValueError("no quantum records to bind: refusing")
    if type(output_root) is not str or not output_root.startswith("/"):
        raise ValueError("an output root must be absolute: refusing")
    bound_dir = bound_readset_directory(output_root, metadata_root=metadata_root)
    emitted: list[dict] = []
    seen: set[str] = set()
    for record in rows:
        manifest = build_quantum_boundary_readset(
            record, receipt, strided_boundaries=strided_boundaries,
            n_probes=n_probes)
        quantum_id = record.get("quantum_id")
        manifest_path = f"{bound_dir}/{quantum_id}.boundary-readset.json.gz"
        if quantum_id in seen or manifest_path in seen:
            raise ValueError(f"duplicate quantum binding {quantum_id!r}: "
                             "refusing")
        seen.add(quantum_id)
        seen.add(manifest_path)
        manifest_sha256 = hashlib.sha256(
            seal_manifest_bytes(manifest)).hexdigest()
        emitted.append({
            "record": bind_quantum_boundary_readset(
                record, receipt, manifest=manifest,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                output_root=output_root,
                strided_boundaries=strided_boundaries,
                n_probes=n_probes,
                metadata_root=metadata_root),
            "manifest": manifest,
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha256,
        })
    return emitted


def bind_quantum_executable(record: Mapping, receipt: Mapping,
                            parent_manifest: Mapping, *, manifest: Mapping,
                            manifest_path: str, manifest_sha256: str,
                            output_root: str,
                            strided_boundaries: Sequence[int], n_probes: int,
                            calib: Mapping,
                            render_prerequisite: Mapping,
                            metadata_root: str | None = None,
                            layer_source_spans: Mapping[int, Sequence] | None = None,
                            source_model_root: str | None = None) -> dict:
    """Bind a sealed executable read manifest to a NEW record generation.

    Returns a deep copy of ``record`` carrying an ``executable_readset``
    block and a recomputed ``identity_sha256`` (the existing canonical
    owner -- the same recomputation both real validators enforce); the
    input record is never mutated. The input record is reverified through
    ``check_quantum_for_campaign`` before any mutation. The manifest must
    equal what :func:`build_quantum_executable_manifest` derives from this
    record, receipt, parent and sealed inputs -- forged bytes with a
    consistent rehash refuse. The manifest path is exactly the
    producer-named bound path (no free-form pathname).
    """
    import copy
    import os
    if not isinstance(record, dict):
        raise ValueError("a quantum record must be an object: refusing")
    if record.get("schema") != LAYER_QUANTUM_SCHEMA:
        raise ValueError("a quantum record has a foreign schema: refusing")
    layer = record.get("layer")
    if type(layer) is not int or isinstance(layer, bool) or layer < 0:
        raise ValueError("a quantum record names no layer: refusing")
    if record.get("quantum_id") != quantum_id(layer):
        raise ValueError(
            f"quantum id {record.get('quantum_id')!r} does not name layer "
            f"{layer}: refusing")
    adjoint = record.get("adjoint")
    if not isinstance(adjoint, dict):
        raise ValueError("a quantum record carries no adjoint block: refusing")
    bound_receipt = adjoint.get("receipt_sha256")
    if type(bound_receipt) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", bound_receipt):
        raise ValueError(
            f"quantum {record.get('quantum_id')!r} is unbound (pre-A): "
            "re-seal against the stage-A receipt before binding a readset, "
            "refusing")
    campaign = record.get("campaign")
    if not isinstance(campaign, dict):
        raise ValueError("a quantum record carries no campaign block: refusing")
    for key in ("plan_path", "plan_sha256", "prepared_path", "prepared_sha256",
                "read_manifest_sha256", "campaign_scope"):
        if not campaign.get(key):
            raise ValueError(f"a quantum record seals no campaign {key}: "
                             "refusing")
    if type(n_probes) is not int or isinstance(n_probes, bool) \
            or n_probes < 1:
        raise ValueError("an executable readset needs a trusted sealed probe "
                         f"count, not {n_probes!r}")
    if type(output_root) is not str or not output_root.startswith("/"):
        raise ValueError("an output root must be absolute: refusing")
    expected_path = (
        bound_readset_directory(output_root, metadata_root=metadata_root)
        + f"/{quantum_id(layer)}.executable.json.gz")
    if manifest_path != expected_path or os.path.normpath(
            manifest_path) != manifest_path or ".." in manifest_path.split("/"):
        raise ValueError(
            f"an executable readset path must be exactly {expected_path}: "
            "refusing")
    check_quantum_for_campaign(
        record, {**campaign, "adjoint_receipt_sha256": bound_receipt})
    if not isinstance(manifest, dict):
        raise ValueError("an executable read manifest must be an object: "
                         "refusing")
    if manifest.get("schema") != MANIFEST_SCHEMA_V2:
        raise ValueError("an executable read manifest has a foreign schema: "
                         "refusing")
    annotations = manifest.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError("an executable read manifest has no annotations: "
                         "refusing")
    if annotations.get("receipt_sha256") != bound_receipt:
        raise ValueError(
            f"quantum {record.get('quantum_id')!r} binds another stage-A "
            "receipt: refusing")
    if annotations.get("n_probes") != n_probes:
        raise ValueError(
            f"the executable readset attests another probe count "
            f"{annotations.get('n_probes')!r}: refusing")
    try:
        expected = build_quantum_executable_manifest(
            record, receipt, parent_manifest,
            strided_boundaries=strided_boundaries, n_probes=n_probes,
            calib=calib, render_prerequisite=render_prerequisite,
            layer_source_spans=layer_source_spans,
            source_model_root=source_model_root)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise ValueError("the executable readset does not derive from its "
                         f"record, receipt and parent: refusing ({exc})") from exc
    if manifest != expected:
        raise ValueError(
            "the executable readset entries, phases or counts do not "
            "originate from the bound inputs: refusing")
    wire = seal_manifest_bytes(manifest)
    if hashlib.sha256(wire).hexdigest() != manifest_sha256:
        raise ValueError("the executable readset digest does not reproduce "
                         "from its manifest wire: refusing")
    fresh = copy.deepcopy(record)
    fresh["executable_readset"] = {
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "entry_count": manifest["entry_count"],
        "total_bytes": manifest["total_bytes"],
        "read_bytes": manifest["read_plan"]["read_bytes"],
        "phases": [phase["name"]
                   for phase in manifest["read_plan"]["phases"]],
        "receipt_sha256": bound_receipt,
    }
    body = {key: value for key, value in fresh.items()
            if key != "identity_sha256"}
    fresh["identity_sha256"] = canonical_sha256(
        body, where=f"quantum record {fresh.get('quantum_id')}")
    return fresh


def emit_quantum_executable_readsets(
        receipt: Mapping, records: Sequence[Mapping],
        parent_manifest: Mapping, *, strided_boundaries: Sequence[int],
        n_probes: int, calib: Mapping, render_prerequisite: Mapping,
        output_root: str, metadata_root: str | None = None,
        layer_source_spans: Mapping[int, Sequence] | None = None,
        source_model_root: str | None = None) -> list[dict]:
    """The post-capture generation path for executable read manifests.

    For every record, derives the executable manifest, seals it, and binds
    it to a new record generation under
    ``{output_root}/layer-quanta/adjoint/bound-readsets/`` -- or, with an
    explicit ``metadata_root`` (PQ #884), under that generation namespace's
    ``adjoint/bound-readsets/`` (the same relative control layout). Returns
    one ``{"record", "manifest", "manifest_path", "manifest_sha256"}`` per
    quantum, in record order. Emits nothing to disk and mutates nothing:
    the post-capture regen writes the returned bytes and adopts the
    returned records. Duplicate quantum ids or manifest paths refuse whole
    rather than binding half a campaign.
    """
    rows = list(records)
    if not rows:
        raise ValueError("no quantum records to bind: refusing")
    if type(output_root) is not str or not output_root.startswith("/"):
        raise ValueError("an output root must be absolute: refusing")
    bound_dir = bound_readset_directory(output_root, metadata_root=metadata_root)
    emitted: list[dict] = []
    seen: set[str] = set()
    for record in rows:
        manifest = build_quantum_executable_manifest(
            record, receipt, parent_manifest,
            strided_boundaries=strided_boundaries, n_probes=n_probes,
            calib=calib, render_prerequisite=render_prerequisite,
            layer_source_spans=layer_source_spans,
            source_model_root=source_model_root)
        quantum_id = record.get("quantum_id")
        manifest_path = f"{bound_dir}/{quantum_id}.executable.json.gz"
        if quantum_id in seen or manifest_path in seen:
            raise ValueError(f"duplicate quantum binding {quantum_id!r}: "
                             "refusing")
        seen.add(quantum_id)
        seen.add(manifest_path)
        manifest_sha256 = hashlib.sha256(
            seal_manifest_bytes(manifest)).hexdigest()
        emitted.append({
            "record": bind_quantum_executable(
                record, receipt, parent_manifest, manifest=manifest,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                output_root=output_root,
                strided_boundaries=strided_boundaries, n_probes=n_probes,
                calib=calib, render_prerequisite=render_prerequisite,
                metadata_root=metadata_root,
                layer_source_spans=layer_source_spans,
                source_model_root=source_model_root),
            "manifest": manifest,
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha256,
        })
    return emitted
