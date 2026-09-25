"""Deterministic join of per-layer joint-AURA cost quanta.

Implements §7 of ``docs/design/distributed_campaign_2026-09-19.md``: custody
→ coverage → per-row validation, merging the per-layer payloads the
``joint_cost_quantum`` runtime (§6) commits into the campaign's
``joint-cost.pkl`` shape (the allocation stage's pareto input).

The join is a disjoint union — each qname's rows come from exactly one
quantum, so completion order cannot change the merged bytes. Serialization
is canonical (sorted keys, the single run's pickle protocol), whatever order
the receipts arrive in.

Failure semantics (§7.2): a missing or ``gapped`` quantum does not fail the
join. The other layers complete, the merged payload carries
``status: "gapped"`` with the gaps named, and the exit code is 0 — retry is
free. What fails closed is *consumption*: :func:`load_joint_cost_for_allocation`
refuses a gapped payload, so a partial campaign can never be read as a score.

Shapes owned elsewhere (fixtures here, never constructed): the layer-quantum
record (§3, ``prismaquant.joint_layer_quanta.v1``, built by the producer) and
the per-quantum ``cost.pkl`` / ``status.json`` (§6.4, built by the runtime).
This module checks the contract's schemas and digests; it does not construct
producer or runtime records. The *constructions* it shares with the producer
-- the roster digest, the phase tiling, the quantum-id padding, the qname
layer grammar -- are imported from ``joint_layer_quanta`` so the two sides of
the wire cannot drift again (issue #787: they did, four ways, and every
genuine record refused).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import copy
import sys
import time
from pathlib import Path

from prismaquant.dev_mode import seal_check
from prismaquant.cost_stage_checkpoint import (
    atomic_write_bytes,
    canonical_json_bytes,
    canonical_json_sha256,
)
from prismaquant.joint_aura import (
    prepare_joint_aura_identities, validate_joint_aura_entry)
from prismaquant.joint_layer_quanta import (
    phase_ranges,
    qname_layer,
    quantum_id,
    roster_digest,
)

RECORD_SCHEMA = "prismaquant.joint_layer_quanta.v1"
STATUS_SCHEMA = "prismaquant.joint_layer_quantum.status.v1"
JOINED_RESULTS_SCHEMA = "prismaquant.joint_layer_quanta.joined_results.v1"

#: The producer's quantum-id spelling (§3.1: ``f"layer-{layer:03d}"``),
#: imported so the joiner's padding cannot drift from the sealed records.
quantum_id_for_layer = quantum_id

#: The single run seals ``joint-cost.pkl`` with this call
#: (``tessera_joint_aura.run``); the join seals the merged payload the same
#: way so readers see one pickle convention.
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

#: Exit code when the join refuses (custody, coverage-defect, or row defect).
#: Gapped campaigns still exit 0 — a gap is a state, not an error.
EXIT_REFUSED = 1

#: §6.4's payload provenance grammar, pinned by issue #787 (B4). The
#: contract names the contents -- "the campaign binding, the quantum
#: identity, the adjoint digest" -- not the keys; since PQ #993 the adjoint
#: digest is the quantum's stage-A slice digest, never a receipt's; these are the
#: blocks the §6 runtime (``prismaquant.joint_cost_quantum``) seals into
#: every ``cost.pkl``, and the joiner consumes what the producer side
#: seals. No implementation digest is promised in the payload: the
#: implementation is bound through ``prepared_sha256``, whose completion
#: seals ``implementation_sha256`` (the runtime re-checks that digest
#: against the prepared completion before measuring).
REQUIRED_PROVENANCE_BLOCKS = (
    "campaign_binding",
    "distributed_quantum",
    "adjoint_slice_sha256",
)

#: The campaign-binding block's keys (§6.4 "the campaign binding"; the
#: runtime copies them verbatim from the record's sealed ``campaign``).
CAMPAIGN_BINDING_KEYS = (
    "plan_sha256",
    "prepared_sha256",
    "read_manifest_sha256",
    "campaign_scope",
    "unit_roster_sha256",
)

#: The distributed-quantum identity block's keys (§6.4 "the quantum
#: identity"; the runtime seals them verbatim from the record).
DISTRIBUTED_QUANTUM_KEYS = (
    "quantum_id",
    "identity_sha256",
    "adjoint_slice_sha256",
    "checkpoint_boundary",
    "chain_layers",
    "windows",
    "chunks",
)


class JoinRefused(Exception):
    """The join failed closed: custody, coverage, or a row defect."""


class GappedPayloadRefused(Exception):
    """The allocation stage refused a gapped joined payload."""


def _load_json(path: Path, *, where: str) -> object:
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise JoinRefused(f"{where}: unreadable file at {path}: {exc}") from exc
    if blob[:2] == b"\x1f\x8b":
        # The parent run manifest is sealed as one gzip member
        # (``seal_manifest_bytes``: compact JSON + newline, mtime=0), so a
        # sealed ``.json.gz`` is read by its member. The digest check
        # elsewhere stays over the sealed file bytes, never the member.
        try:
            blob = gzip.decompress(blob)
        except (OSError, EOFError) as exc:
            raise JoinRefused(
                f"{where}: unreadable gzip member at {path}: {exc}") from exc
    try:
        return json.loads(blob.decode("utf-8"))
    except ValueError as exc:
        raise JoinRefused(f"{where}: unreadable JSON at {path}: {exc}") from exc


def _sha_file(path: Path, *, where: str) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise JoinRefused(f"{where}: unreadable file at {path}: {exc}") from exc


def _read_checked(path: Path, expected: str | None, *,
                  where: str) -> tuple[bytes, str]:
    """Read ``path`` once and return its bytes with their digest. The digest
    describes the bytes returned, so a caller that parses them never
    authenticates one read and parses another (PQ #1256)."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise JoinRefused(f"{where}: unreadable file at {path}: {exc}") from exc
    actual = hashlib.sha256(data).hexdigest()
    if expected is not None and actual != expected:
        raise JoinRefused(
            f"{where}: digest mismatch at {path}: expected {expected}, got {actual}")
    return data, actual


def _check_digest(path: Path, expected: str, *, where: str) -> None:
    _read_checked(path, expected, where=where)


def _scan_receipts(input_root: Path | None, *, records_dir: Path | None = None) -> list[dict]:
    """Build the receipt set from the campaign output root.

    The records are the sealed source of every path (§3.1 ``output_space``):
    each record names its quantum's ``cost.pkl`` and ``status.json``
    locations -- the same paths the §6 runtime writes (§6.4) -- so the joiner
    reads them from the record instead of guessing a layout. Record files are
    accepted from ``layer-quanta/records/`` (the producer's §4.1 layout) or
    flat in ``layer-quanta/`` (the sealed takeover layout the dispatcher
    consumed); custody re-checks each record's identity before its paths are
    trusted for content.
    """
    if records_dir is not None and input_root is not None:
        raise JoinRefused("coverage: choose records directory or historical input root, not both")
    quanta_root = records_dir if records_dir is not None else input_root / "layer-quanta"
    record_paths: list[Path] = []
    directories = (quanta_root,) if records_dir is not None else (quanta_root / "records", quanta_root)
    for directory in directories:
        record_paths = sorted(directory.glob("layer-*.json"))
        if record_paths:
            break
    if not record_paths:
        raise JoinRefused(f"coverage: no layer records under {quanta_root}")
    receipts = []
    for record_path in record_paths:
        quantum_id = record_path.stem
        where = f"records {quantum_id}"
        record = _load_json(record_path, where=where)
        if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
            raise JoinRefused(
                f"{where}: record schema is not {RECORD_SCHEMA!r}")
        if record.get("quantum_id") != quantum_id:
            raise JoinRefused(
                f"{where}: record names {record.get('quantum_id')!r}, "
                f"file names {quantum_id!r}")
        space = record.get("output_space")
        if not isinstance(space, dict) or not isinstance(space.get("root"), str):
            raise JoinRefused(f"{where}: record seals no output space")
        root = space["root"]
        receipts.append({
            "quantum_id": quantum_id,
            "record_path": str(record_path),
            "cost_path": space.get("cost_payload", f"{root}/cost.pkl"),
            "status_path": space.get("status", f"{root}/status.json"),
        })
    return receipts


def _check_record(record: object, receipt: dict, campaign: dict, *,
                  stage_a: tuple[str, dict[int, str]] | None = None) -> dict:
    """Custody step 1: the record answers for the receipt's identity and the
    campaign binding. Returns the record."""
    quantum_id = receipt["quantum_id"]
    where = f"custody {quantum_id}"
    if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
        raise JoinRefused(
            f"{where}: record schema is not {RECORD_SCHEMA!r}")
    if record.get("quantum_id") != quantum_id:
        raise JoinRefused(
            f"{where}: record names {record.get('quantum_id')!r}, "
            f"receipt names {quantum_id!r}")
    identity = canonical_json_sha256(
        {k: v for k, v in record.items() if k != "identity_sha256"},
        where=f"{where} record identity")
    if identity != record.get("identity_sha256"):
        raise JoinRefused(
            f"{where}: record identity digest does not verify "
            "(record edited after sealing?)")
    if receipt.get("identity_sha256") is not None and (
            receipt["identity_sha256"] != record["identity_sha256"]):
        raise JoinRefused(
            f"{where}: receipt identity {receipt['identity_sha256']!r} does not "
            f"match record {record['identity_sha256']!r} (retargeted receipt)")
    binding = record.get("campaign")
    if not isinstance(binding, dict):
        raise JoinRefused(f"{where}: record carries no campaign binding")
    # Run-gate seals (PQ #1147): certified mode refuses as before; dev mode
    # prints each mismatch and joins the record's data.
    for key in ("plan_sha256", "prepared_sha256"):
        seal_check(key, campaign[key], binding.get(key), where=where,
                   refusal=lambda key=key: JoinRefused(
                       f"{where}: record {key} {binding.get(key)!r} is not this "
                       f"campaign ({campaign[key]!r})"))
    seal_check("read_manifest_sha256", campaign["manifest_sha256"],
               binding.get("read_manifest_sha256"), where=where,
               refusal=lambda: JoinRefused(
                   f"{where}: record read-manifest digest is not this campaign's"))
    seal_check("campaign_scope", campaign["scope"], binding.get("campaign_scope"),
               where=where, refusal=lambda: JoinRefused(
                   f"{where}: record scope is not this campaign's scope"))
    # B1 (#787): the roster digest is the producer's #768 construction
    # (§3.1: "sha256 of the sorted qname roster, one per line") -- sorted,
    # no trailing newline -- imported from the producer so the joiner's
    # recomputation cannot drift from the seal.
    if binding.get("unit_roster_sha256") != roster_digest(campaign["roster"]):
        raise JoinRefused(f"{where}: record roster digest is not this roster")
    # §3.2's rule, mirrored from check_quantum_for_campaign (PQ #993): every
    # record binds exactly the slice the caller's stage-A proof gives its
    # layer -- an unbound (pre-A) record, or one binding a whole receipt,
    # refuses rather than joining. Fail closed per quantum.
    adjoint = record.get("adjoint", {})
    if adjoint.get("receipt_sha256") is not None:
        raise JoinRefused(
            f"{where}: record binds a whole stage-A receipt, not its slice")
    actual_slice = adjoint.get("slice_sha256")
    if actual_slice is None:
        raise JoinRefused(
            f"{where}: record is unbound (pre-A): re-seal against its "
            "stage-A slice before joining")
    if stage_a is not None:
        mode, expected_slices = stage_a
        expected_slice = expected_slices.get(record.get("layer"))
        if expected_slice is None:
            raise JoinRefused(
                f"{where}: the stage-A proof gives layer {record.get('layer')} no slice")
        if actual_slice != expected_slice:
            raise JoinRefused(
                f"{where}: record binds another stage-A slice than the "
                f"{mode} gives layer {record.get('layer')}")
    return record


def stage_a_expected_slices(campaign: dict) -> tuple[str, dict[int, str]]:
    """The slice digest the caller's stage-A proof gives every layer.

    Two modes (PQ #993). (a) ``adjoint_receipt``: the completed receipt;
    every layer's slice is recomputed from it. (b) ``adjoint_bands``: no
    complete receipt; the bands share one run header, every stride
    checkpoint from the tail to the lowest has one, and each band gives the
    slices of the layers it serves. Stage A's backward below the lowest
    checkpoint is an input to nothing, so mode (b) never waits for it.
    Returns ``(mode, {layer: slice_sha256})``; refuses any other proof.
    """
    from prismaquant.joint_adjoint_slices import (
        AdjointSliceRefused, adjoint_slice_sha256, band_set,
        missing_band_boundaries, nearest_checkpoint_boundary,
        stage_a_receipt_kind, stage_a_run_header, stage_a_slice)
    receipt = campaign.get("adjoint_receipt")
    bands = campaign.get("adjoint_bands")
    if (receipt is None) == (bands is None):
        raise JoinRefused(
            "coverage: the join needs exactly one stage-A proof: the completed "
            "receipt (adjoint_receipt) or the checkpoint bands (adjoint_bands)")
    try:
        if receipt is not None:
            if stage_a_receipt_kind(receipt) != "complete":
                raise JoinRefused("coverage: adjoint_receipt is not a completed receipt")
            mode, header = "completed receipt", stage_a_run_header(receipt)

            def source(boundary):
                return receipt
        else:
            indexed = band_set(bands)
            if not indexed:
                raise JoinRefused("coverage: an empty band set proves nothing")
            missing = missing_band_boundaries(indexed)
            if missing:
                raise JoinRefused(
                    f"coverage: stride checkpoints {missing} have no band")
            mode = "checkpoint bands"
            header = stage_a_run_header(next(iter(indexed.values())))

            def source(boundary):
                return indexed[boundary]
        boundaries = header["stride"]["boundaries"]
        layers = range(max(int(mark) for mark in boundaries))
        return mode, {layer: adjoint_slice_sha256(stage_a_slice(
            source(nearest_checkpoint_boundary(boundaries, layer)), layer))
            for layer in layers}
    except AdjointSliceRefused as exc:
        raise JoinRefused(f"coverage: stage-A proof refused: {exc}") from exc


def _proof_header_sha256(campaign: dict) -> str:
    from prismaquant.joint_adjoint_slices import stage_a_run_header, stage_a_run_header_sha256
    proof = campaign.get("adjoint_receipt") or campaign["adjoint_bands"][0]
    return stage_a_run_header_sha256(stage_a_run_header(proof))


def _check_stage_a_header(campaign: dict, records: dict[str, dict]) -> None:
    """The stage-A proof's run header answers for the campaign being joined.

    The identity half of the producer's check (``check_adjoint_run_identity``):
    the run's plan, prepared and scope, or, for a catalog extension, the
    original run the extension binds. Every record must carry one extension
    binding. With no record present there is nothing the header could admit.
    """
    from prismaquant.joint_adjoint_slices import (
        AdjointSliceRefused, stage_a_run_header)
    from prismaquant.joint_layer_quanta import canonical_bytes, check_adjoint_run_identity
    if not records:
        return
    extensions = {canonical_bytes(record.get("catalog_extension"))
                  for record in records.values()}
    if len(extensions) != 1:
        raise JoinRefused("custody: records carry different catalog extension bindings")
    proof = campaign.get("adjoint_receipt") or campaign["adjoint_bands"][0]
    try:
        header = stage_a_run_header(proof)
        # Each record binds the slice this header gives its layer
        # (stage_a_expected_slices), so the header's stride is the one the
        # producer bound; the joiner derives no stride of its own.
        check_adjoint_run_identity(
            header, plan_sha256=campaign["plan_sha256"],
            prepared_sha256=campaign["prepared_sha256"], scope=campaign["scope"],
            catalog_extension=next(iter(records.values())).get("catalog_extension"))
    except (AdjointSliceRefused, ValueError, OSError) as exc:
        raise JoinRefused(
            f"coverage: the stage-A proof does not answer for this campaign: {exc}") from exc


def _check_tiling(records: dict[str, dict], campaign: dict) -> None:
    """Coverage step 1: every present record's source phase cites its parent
    phase by name and byte range. Absent phases are gaps (handled by the
    caller), never silent holes: only names the parent manifest knows are
    admitted. The replay uses the producer's ``phase_ranges`` (entry-aligned,
    start/end derived from the sealed cumulative marks) so the joiner's
    tiling proof cannot drift from the seal-time cut (B2, #787: the sealed
    manifest's phase rows carry ``{name, bytes, cumulative_bytes}``, not
    start/end offsets)."""
    try:
        rows = {row["name"]: row
                for row in phase_ranges(campaign["parent_manifest"])}
    except ValueError as exc:
        raise JoinRefused(
            f"coverage: parent manifest phase table refuses: {exc}") from exc
    for quantum_id, record in sorted(records.items()):
        where = f"coverage {quantum_id}"
        source = record.get("read_set", {}).get("source_phase")
        if not isinstance(source, dict):
            raise JoinRefused(f"{where}: record cites no source phase")
        parent_phase = rows.get(source.get("name"))
        if parent_phase is None:
            raise JoinRefused(
                f"{where}: source phase {source.get('name')!r} is not in the "
                "parent manifest")
        for key in ("start_bytes", "end_bytes"):
            if source.get(key) != parent_phase.get(key):
                raise JoinRefused(
                    f"{where}: source phase range does not replay the parent "
                    "manifest")


def _read_status(receipt: dict) -> tuple[str, list]:
    """Read a quantum's terminal receipt. A missing receipt is a gap: the
    quantum never finished. Anything malformed fails closed."""
    quantum_id = receipt["quantum_id"]
    try:
        raw = Path(receipt["status_path"]).read_bytes().decode("utf-8")
    except OSError:
        return "gapped", [0, 0]
    try:
        status = json.loads(raw)
    except ValueError as exc:
        raise JoinRefused(
            f"custody {quantum_id}: unreadable status.json: {exc}") from exc
    if not isinstance(status, dict) or status.get("schema") != STATUS_SCHEMA:
        raise JoinRefused(
            f"custody {quantum_id}: status schema is not {STATUS_SCHEMA!r}")
    if status.get("quantum_id") != quantum_id or (
            status.get("identity_sha256") != receipt.get("identity_sha256")):
        raise JoinRefused(
            f"custody {quantum_id}: status receipt is not this quantum's")
    if status.get("status") not in ("complete", "gapped"):
        raise JoinRefused(
            f"custody {quantum_id}: unknown status {status.get('status')!r}")
    units = status.get("units", [0, 0])
    return status["status"], units


def _load_cost_payload(receipt: dict, record: dict,
                       campaign: dict) -> tuple[dict, str]:
    """Custody step 2: the payload's provenance equals the campaign binding
    and answers for exactly this record's sealed identity and adjoint
    binding; only per-layer content may differ. B4 (#787): the grammar is
    the §6 runtime's wire -- a ``campaign_binding`` block, a
    ``distributed_quantum`` identity block, and the adjoint receipt digest
    -- pinned here before the campaign's payloads land."""
    quantum_id = receipt["quantum_id"]
    where = f"custody {quantum_id}"
    cost_bytes, cost_sha256 = _read_checked(
        Path(receipt["cost_path"]), receipt.get("cost_sha256"), where=where)
    try:
        payload = pickle.loads(cost_bytes)
    except (ValueError, pickle.UnpicklingError) as exc:
        raise JoinRefused(f"{where}: unreadable cost payload: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(
            payload.get("costs"), dict) or not isinstance(
            payload.get("provenance"), dict):
        raise JoinRefused(f"{where}: cost payload shape is not §6.4")
    provenance = payload["provenance"]
    for block in REQUIRED_PROVENANCE_BLOCKS:
        if block not in provenance:
            raise JoinRefused(f"{where}: provenance lacks {block!r}")
    binding = provenance["campaign_binding"]
    identity = provenance["distributed_quantum"]
    if not isinstance(binding, dict) or not isinstance(identity, dict):
        raise JoinRefused(f"{where}: provenance blocks are not §6.4 objects")
    expected_binding = {
        "plan_sha256": campaign["plan_sha256"],
        "prepared_sha256": campaign["prepared_sha256"],
        "read_manifest_sha256": campaign["manifest_sha256"],
        "campaign_scope": campaign["scope"],
        "unit_roster_sha256": roster_digest(campaign["roster"]),
    }
    for key in CAMPAIGN_BINDING_KEYS:
        refusal = JoinRefused(
            f"{where}: payload campaign binding {key} is foreign to this campaign")
        if key == "unit_roster_sha256":
            # The roster names the units the payload holds: a wall.
            if binding.get(key) != expected_binding[key]:
                raise refusal
            continue
        # Run seals (PQ #1147): dev mode prints and joins the payload.
        seal_check(f"payload {key}", expected_binding[key], binding.get(key),
                   where=where, refusal=refusal)
    adjoint = record.get("adjoint", {})
    expected_identity = {
        "quantum_id": quantum_id,
        "identity_sha256": record["identity_sha256"],
        "adjoint_slice_sha256": adjoint.get("slice_sha256"),
        "checkpoint_boundary": adjoint.get("checkpoint_boundary"),
        "chain_layers": adjoint.get("chain_layers"),
        "windows": len(record.get("windows", [])),
        "chunks": [chunk.get("name") for chunk in record.get("chunks", [])],
    }
    for key in DISTRIBUTED_QUANTUM_KEYS:
        if identity.get(key) != expected_identity[key]:
            raise JoinRefused(
                f"{where}: payload distributed-quantum {key} does not "
                "answer for the record (retargeted receipt)")
    if provenance["adjoint_slice_sha256"] != adjoint.get("slice_sha256"):
        raise JoinRefused(
            f"{where}: payload adjoint slice digest does not answer for "
            "the record")
    if adjoint.get("slice_sha256") is None:
        # The §6.4 adjoint digest the contract names: an unbound (pre-A)
        # record never joins -- the same refusal check_quantum_for_campaign
        # makes for a bound campaign.
        raise JoinRefused(
            f"{where}: record is unbound (pre-A): re-seal against its "
            "stage-A slice before joining")
    # Every row of one payload carries the same probe identity object (pickle
    # keeps the shared reference), and on GLM-5.3 that identity is 8.9 MB of
    # model weight map. Validate and hash it once per payload instead of once
    # per row; every per-row check below still runs on every row (PQ #1256).
    prepare_joint_aura_identities(payload)
    return payload, cost_sha256


def _check_rows(costs: dict, quantum_id: str) -> None:
    """Coverage step 3: every row passes ``validate_joint_aura_entry``. A
    row that does not is a defect, not a gap — the qname is named."""
    for qname in sorted(costs):
        rows = costs[qname]
        if not isinstance(rows, dict) or not rows:
            raise JoinRefused(f"row {qname}: empty candidate set")
        for fmt in sorted(rows):
            try:
                valid = validate_joint_aura_entry(rows[fmt])
            except Exception as exc:
                raise JoinRefused(
                    f"row {qname}@{fmt} ({quantum_id}): invalid joint row: "
                    f"{exc}") from exc
            if not valid:
                raise JoinRefused(
                    f"row {qname}@{fmt} ({quantum_id}): not a joint row")


def _roster_by_layer(roster: list[str]) -> dict[int, list[str]]:
    """Partition the caller's roster by the qname layer grammar the producer
    refuses to seal unevenly (§3.1; ``joint_layer_quanta._layer_qnames``)."""
    by_layer: dict[int, list[str]] = {}
    for qname in roster:
        layer = qname_layer(qname)
        if layer is not None:
            by_layer.setdefault(layer, []).append(qname)
    for units in by_layer.values():
        units.sort()
    return by_layer


def _record_units(record: dict, roster_by_layer: dict[int, list[str]]) -> list[str]:
    """The quantum's roster units, by the record's sealed layer.

    B3 (#787): the producer seals windows index-only (D2, derivation v2) --
    no per-window names exist in a record -- so a gap's units are the
    roster's qnames for that layer, the same partition the producer refuses
    to seal unevenly. A record whose layer names no roster unit reports no
    units (only a foreign record can)."""
    layer = record.get("layer")
    if type(layer) is not int or isinstance(layer, bool):
        return []
    return list(roster_by_layer.get(layer, []))


def _expected_quantum_ids(campaign: dict) -> set[str]:
    """The quantum set the parent manifest declares. The manifest is caller
    input, so a lost record names its gap instead of shrinking the set.

    B2 (#787): ids are *constructed* through the producer's padding
    (§3.1: ``quantum_id`` is ``f"layer-{layer:03d}"`` over the plan's source
    layers), never read from the phase table -- the sealed run manifest's
    phases carry unpadded names (``layer-13``) and no ``quantum_id`` field,
    and non-layer phases (``head``) are not quanta. The declared layer set
    is the manifest's ``annotations.layers``; a manifest without it falls
    back to its ``layer-N`` phase names.
    """
    parent = campaign["parent_manifest"]
    annotations = parent.get("annotations")
    layers = annotations.get("layers") if isinstance(annotations, dict) else None
    if isinstance(layers, list) and layers:
        return {quantum_id_for_layer(int(layer)) for layer in layers}
    phases = parent.get("phases")
    if phases is None and isinstance(annotations, dict):
        phases = annotations.get("phases")
    ids: set[str] = set()
    for phase in phases if isinstance(phases, list) else []:
        name = phase.get("name") if isinstance(phase, dict) else None
        if not (isinstance(name, str) and name.startswith("layer-")):
            continue
        try:
            ids.add(quantum_id_for_layer(int(name[len("layer-"):])))
        except ValueError:
            continue
    if not ids:
        raise JoinRefused(
            "coverage: parent manifest declares no layer set "
            "(no annotations.layers, no layer-N phases)")
    return ids


def join_joint_quanta(*, receipts: list[dict] | None, campaign: dict,
                      output_dir: str | Path,
                      input_root: str | Path | None = None,
                      records_dir: str | Path | None = None,
                      now: float | None = None) -> dict:
    """Merge per-layer cost payloads into the campaign's results shape.

    ``receipts`` is one ``{quantum_id, record_path[, cost_path, cost_sha256,
    identity_sha256, status_path]}`` per quantum; when None the receipt set
    is scanned from ``input_root`` (required then) off each sealed record's
    ``output_space``. ``campaign`` is the caller-supplied binding --
    plan/prepared/manifest digests, scope, roster, ``formats_by_qname``, the
    parent manifest, and exactly one stage-A proof: the completed receipt
    (``adjoint_receipt``) or the checkpoint bands (``adjoint_bands``); see
    :func:`stage_a_expected_slices` -- never derived from the surviving
    shards. Every record must bind the slice that proof gives its layer.

    Returns ``{"status": "complete"|"gapped", "gaps": [...],
    "joint_cost_path": ..., "results_path": ..., "coverage_sha256": ...}``.
    Raises :class:`JoinRefused` on custody, coverage-defect, or row defects.
    """
    if receipts is None:
        if input_root is None and records_dir is None:
            raise JoinRefused("coverage: no receipts and no input root")
        receipts = _scan_receipts(Path(input_root) if input_root is not None else None,
                                 records_dir=Path(records_dir) if records_dir is not None else None)
    if not receipts:
        raise JoinRefused("coverage: empty receipt set")
    quantum_ids = [r["quantum_id"] for r in receipts]
    if len(set(quantum_ids)) != len(quantum_ids):
        raise JoinRefused("coverage: duplicated quantum ids in receipt set")
    for key in ("plan_sha256", "prepared_sha256", "manifest_sha256", "scope",
                "roster", "formats_by_qname", "parent_manifest"):
        if key not in campaign:
            raise JoinRefused(f"coverage: campaign binding lacks {key!r}")
    mode, expected_slices = stage_a_expected_slices(campaign)
    expected_layers = {int(qid.split("-")[-1]) for qid in _expected_quantum_ids(campaign)}
    if set(expected_slices) != expected_layers:
        raise JoinRefused(
            f"coverage: the stage-A {mode} covers layers "
            f"{sorted(set(expected_slices) ^ expected_layers)} unlike the campaign")

    records: dict[str, dict] = {}
    for receipt in sorted(receipts, key=lambda r: r["quantum_id"]):
        record = _load_json(Path(receipt["record_path"]),
                            where=f"custody {receipt['quantum_id']}")
        records[receipt["quantum_id"]] = _check_record(
            record, receipt, campaign, stage_a=(mode, expected_slices))
        receipt["identity_sha256"] = record["identity_sha256"]
    _check_stage_a_header(campaign, records)
    _check_tiling(records, campaign)
    roster_by_layer = _roster_by_layer(campaign["roster"])

    merged: dict[str, dict] = {}
    measured_payloads: dict[str, dict] = {}
    gaps: list[dict] = []
    per_layer: list[dict] = []
    for quantum_id in sorted(records):
        receipt = next(r for r in receipts if r["quantum_id"] == quantum_id)
        record = records[quantum_id]
        status, units = _read_status(receipt)
        if status != "complete":
            units_named = _record_units(record, roster_by_layer)
            gaps.append({"quantum_id": quantum_id,
                         "unit_count": len(units_named),
                         "units": units_named})
            per_layer.append({"quantum_id": quantum_id,
                              "identity_sha256": record["identity_sha256"],
                              "status": status, "units": units})
            continue
        payload, cost_sha256 = _load_cost_payload(receipt, record, campaign)
        measured_payloads[quantum_id] = payload
        costs = payload["costs"]
        for qname in costs:
            if qname in merged:
                raise JoinRefused(
                    f"coverage: {qname} answered by two quanta")
            if qname not in campaign["formats_by_qname"]:
                raise JoinRefused(f"coverage: {qname} is not on the roster")
            expected_formats = set(campaign["formats_by_qname"][qname])
            if set(costs[qname]) != expected_formats:
                raise JoinRefused(
                    f"coverage: {qname} candidate set {sorted(costs[qname])} "
                    f"is not the prepared {sorted(expected_formats)}")
        _check_rows(costs, quantum_id)
        merged.update(costs)
        per_layer.append({"quantum_id": quantum_id,
                          "identity_sha256": record["identity_sha256"],
                          "cost_sha256": cost_sha256,
                          "status": status, "units": units})

    # A quantum with no record at all is still a named gap, never a shrunk
    # layer set: the expected set comes from the parent manifest, supplied by
    # the caller, not from the surviving shards. B2 (#787): the ids are the
    # producer's padded spelling; a gapped layer's units are named from the
    # roster by layer (B3), so an absent record still accounts for its units.
    absent = sorted(_expected_quantum_ids(campaign) - set(records))
    for quantum_id in absent:
        layer = int(quantum_id[len("layer-"):])
        units_named = roster_by_layer.get(layer, [])
        gaps.append({"quantum_id": quantum_id, "unit_count": len(units_named),
                     "units": units_named, "record_absent": True})
        per_layer.append({"quantum_id": quantum_id, "status": "absent",
                          "units": [0, 0]})

    roster = list(campaign["roster"])
    if sorted(merged) != sorted(roster) and not gaps:
        missing = [q for q in roster if q not in merged]
        raise JoinRefused(
            f"coverage: complete campaign is short {len(missing)} roster "
            f"units, first {missing[:3]!r} (a complete quantum dropped rows)")
    if gaps:
        # Every gap named its units -- from its record's layer for a present
        # record (B3), from the roster by layer for an absent one (B2) -- so
        # roster units still missing are a complete quantum that dropped
        # rows: a defect, not a gap.
        named = set()
        for gap in gaps:
            named.update(gap["units"])
        unaccounted = [q for q in roster
                       if q not in merged and q not in named]
        if unaccounted:
            raise JoinRefused(
                f"coverage: gapped campaign is short {len(unaccounted)} "
                f"roster units outside the named gaps, first "
                f"{unaccounted[:3]!r}")

    status = "complete" if not gaps else "gapped"
    coverage_proof = {
        "schema": "prismaquant.joint_layer_quanta.coverage.v1",
        "quanta": per_layer,
        "gaps": [{"quantum_id": g["quantum_id"],
                  "unit_count": g["unit_count"]} for g in gaps],
        "roster_sha256": roster_digest(roster),
    }
    coverage_sha256 = canonical_json_sha256(
        coverage_proof, where="join coverage proof")
    ordered_costs = {qname: {fmt: merged[qname][fmt]
                             for fmt in sorted(merged[qname])}
                     for qname in sorted(merged)}
    joined = {
        "costs": ordered_costs,
        "provenance": {
            "plan_sha256": campaign["plan_sha256"],
            "prepared_sha256": campaign["prepared_sha256"],
            "read_manifest_sha256": campaign["manifest_sha256"],
            "campaign_scope": campaign["scope"],
            "coverage": {**coverage_proof, "coverage_sha256": coverage_sha256,
                         "status": status,
                         "gaps": gaps},
            "join_schema": JOINED_RESULTS_SCHEMA,
        },
    }
    extensions = [record.get("catalog_extension") for record in records.values()]
    if any(extension is not None for extension in extensions):
        if any(extension != extensions[0] for extension in extensions):
            raise JoinRefused("allocation: quantum catalog-extension bindings differ")
        joined["provenance"]["catalog_extension"] = copy.deepcopy(extensions[0])
    _preserve_allocation_payload(joined, measured_payloads, records)
    joined_unix = time.time() if now is None else now
    results = {
        "schema": JOINED_RESULTS_SCHEMA,
        "status": status,
        "coverage_sha256": coverage_sha256,
        "joined_unix": joined_unix,
        "distributed": {"per_layer": per_layer, "gaps": gaps,
                        "joined_unix": joined_unix,
                        "coverage_sha256": coverage_sha256},
        "campaign": {key: campaign[key] for key in
                     ("plan_sha256", "prepared_sha256", "manifest_sha256")},
        # Which stage-A proof admitted the join (PQ #993). The joined payload
        # does not depend on it: a band set and the completed receipt of one
        # run give every record the same slice.
        "stage_a": {"mode": mode, "run_header_sha256": _proof_header_sha256(campaign)},
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cost_path = out / "joint-cost.pkl"
    results_path = out / "results.json"
    atomic_write_bytes(cost_path,
                       pickle.dumps(joined, protocol=PICKLE_PROTOCOL))
    atomic_write_bytes(results_path, canonical_json_bytes(
        results, where="joined results.json") + b"\n")
    return {"status": status, "gaps": gaps,
            "joint_cost_path": str(cost_path),
            "results_path": str(results_path),
            "coverage_sha256": coverage_sha256}


def _preserve_allocation_payload(joined, payloads, records):
    """Retain measured stats and currency instead of dropping them at union.

    Old parser-only quantum fixtures have neither schema nor stats. They stay
    readable, but cannot acquire allocation metadata from this path. A mix of
    old and production payloads refuses rather than losing part of the probe.
    Per-quantum provenance is kept verbatim; unit-local run identities are not
    relabelled as a new whole-model measurement.
    """
    present = ["stats" in p or "schema" in p for p in payloads.values()]
    if not any(present):
        return
    if not all(present):
        raise JoinRefused("allocation: mixed measured and parser-only quantum payloads")
    from prismaquant.schemas import validate_probe_payload, validate_cost_payload
    from prismaquant.cost_currency import (
        CostCurrencyError, first_joint_probe_identity, probe_identity_walls_differ,
        require_run_currency)

    stats, provenance = {}, {}
    shared = shared_probe = None
    for quantum, payload in sorted(payloads.items()):
        try:
            validate_probe_payload(payload)
            validate_cost_payload(payload)
            currency = require_run_currency(payload)
        except (ValueError, CostCurrencyError) as exc:
            raise JoinRefused(f"allocation {quantum}: {exc}") from exc
        if payload.get("schema") != "prismaquant.aura_cost.v1":
            raise JoinRefused(f"allocation {quantum}: unsupported measured payload schema")
        if set(payload["stats"]) != set(payload["costs"]):
            raise JoinRefused(f"allocation {quantum}: statistics do not cover its cost rows")
        identity = {key: payload.get(key) for key in ("schema", "n_probes", "token_scope")}
        identity["probe_identity_sha256"] = currency.get("probe_identity_sha256")
        identity["served_activation_policy"] = payload["provenance"].get("served_activation_policy")
        identity["stage_b_resource_policy"] = payload["provenance"].get("stage_b_resource_policy")
        if not identity["probe_identity_sha256"]:
            raise JoinRefused(f"allocation {quantum}: complete joint currency required")
        probe = first_joint_probe_identity(payload["costs"])
        if shared is not None:
            # schema, n_probes and token_scope are the measurement's shape,
            # and the probe identity's calibration draw and probes are what
            # was measured: both stay a wall. The rest of the probe identity
            # (producer source, arithmetic) and the two policies are run
            # seals (PQ #1147): dev mode prints them and joins the rows.
            shape = ("schema", "n_probes", "token_scope")
            if (any(identity[key] != shared[key] for key in shape)
                    or probe_identity_walls_differ(shared_probe, probe)):
                raise JoinRefused(f"allocation {quantum}: probe or measurement identity differs")
            seal_check("probe identity", shared, identity, where=f"allocation {quantum}",
                       refusal=lambda: JoinRefused(
                           f"allocation {quantum}: probe or measurement identity differs"))
        shared, shared_probe = identity, probe
        for unit, stat in payload["stats"].items():
            if unit in stats:
                raise JoinRefused(f"allocation {unit}: duplicate statistic")
            stats[unit] = copy.deepcopy(stat)
        provenance[quantum] = copy.deepcopy(payload["provenance"])
    bindings = {(record["campaign"]["prepared_path"],
                 record["campaign"]["prepared_sha256"]) for record in records.values()}
    if not bindings:
        raise JoinRefused("allocation: quantum prepared bindings differ")
    ordered = sorted(bindings)
    seal_check("quantum prepared binding", ordered[0], ordered[-1], where="allocation",
               refusal=lambda: JoinRefused("allocation: quantum prepared bindings differ"))
    path, digest = ordered[0]
    joined.update({key: shared[key] for key in ("schema", "n_probes", "token_scope")})
    joined["stats"] = dict(sorted(stats.items()))
    joined["formats"] = sorted({fmt for rows in joined["costs"].values() for fmt in rows})
    joined["provenance"].update({
        "cost_mode": "aura", "joint_activation": True,
        "cost_currency": "joint_aura_predicted_dloss",
        "measurement_status": "research",
        "prepared": {"path": path, "sha256": digest},
        "quantum_provenance": provenance,
    })
    if shared["served_activation_policy"] is not None:
        joined["provenance"]["served_activation_policy"] = copy.deepcopy(shared["served_activation_policy"])
    if shared["stage_b_resource_policy"] is not None:
        joined["provenance"]["stage_b_resource_policy"] = copy.deepcopy(shared["stage_b_resource_policy"])
    # This also verifies the rows share one full probe/calibration identity.
    try:
        require_run_currency(joined)
    except (ValueError, CostCurrencyError) as exc:
        raise JoinRefused(f"allocation joined currency: {exc}") from exc


def load_joint_cost_for_allocation(path: str | Path) -> dict:
    """The allocation stage's reader: a gapped joined payload is refused so
    a partial campaign can never be read as a score."""
    try:
        payload = pickle.loads(Path(path).read_bytes())
    except (OSError, ValueError, pickle.UnpicklingError) as exc:
        raise GappedPayloadRefused(f"unreadable joined payload: {exc}") from exc
    coverage = payload.get("provenance", {}).get("coverage", {}) \
        if isinstance(payload, dict) else {}
    if coverage.get("status") != "complete" or coverage.get("gaps"):
        names = [g.get("quantum_id") for g in coverage.get("gaps", [])]
        raise GappedPayloadRefused(
            f"joined payload is gapped (gaps: {names}); retry the quanta and "
            "re-join before allocating")
    return payload


def _read_text_list(path: Path) -> list[str]:
    return [line for line in
            (line.strip() for line in path.read_text().splitlines()) if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Join per-layer joint-AURA cost quanta (§7).")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-root", help="historical root containing layer-quanta/records")
    inputs.add_argument("--records", help="exact records directory from regenerate_joint_quanta --metadata-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--scope", required=True,
                        help="JSON file with the sealed campaign scope block")
    parser.add_argument("--roster", required=True,
                        help="sorted qname roster, one per line")
    parser.add_argument("--formats-by-qname", required=True,
                        help="JSON mapping qname to its prepared format list")
    proof = parser.add_mutually_exclusive_group(required=True)
    proof.add_argument("--adjoint-receipt",
                       help="mode (a): the completed stage-A receipt; every "
                            "record's slice is recomputed from it")
    proof.add_argument("--adjoint-band", action="append",
                       help="mode (b): one sealed checkpoint band (repeat once "
                            "per stride checkpoint)")
    parser.add_argument("--adjoint-receipt-sha256",
                        help="file digest of --adjoint-receipt")
    parser.add_argument("--adjoint-band-sha256", action="append",
                        help="file digest of each --adjoint-band, in order")
    args = parser.parse_args(argv)
    if args.adjoint_receipt is not None and not args.adjoint_receipt_sha256:
        parser.error("--adjoint-receipt needs --adjoint-receipt-sha256")
    if args.adjoint_band is not None and len(args.adjoint_band_sha256 or []) != len(
            args.adjoint_band):
        parser.error("every --adjoint-band needs one --adjoint-band-sha256")

    try:
        _check_digest(Path(args.plan), args.plan_sha256, where="campaign plan")
        _check_digest(Path(args.prepared), args.prepared_sha256,
                      where="campaign prepared")
        _check_digest(Path(args.manifest), args.manifest_sha256,
                      where="campaign manifest")
        from prismaquant.joint_adjoint_slices import load_stage_a_receipt_like

        def _proof(path, digest):
            try:
                return load_stage_a_receipt_like(path, digest)
            except (OSError, ValueError, RuntimeError) as exc:
                raise JoinRefused(f"coverage: stage-A proof {path}: {exc}") from exc
        campaign = {
            "plan_sha256": args.plan_sha256,
            "prepared_sha256": args.prepared_sha256,
            "manifest_sha256": args.manifest_sha256,
            "scope": _load_json(Path(args.scope), where="campaign scope"),
            **({"adjoint_receipt": _proof(args.adjoint_receipt, args.adjoint_receipt_sha256)}
               if args.adjoint_receipt is not None else
               {"adjoint_bands": [_proof(path, digest) for path, digest in zip(
                   args.adjoint_band, args.adjoint_band_sha256)]}),
            "roster": _read_text_list(Path(args.roster)),
            "formats_by_qname": _load_json(Path(args.formats_by_qname),
                                           where="formats_by_qname"),
            "parent_manifest": _load_json(Path(args.manifest),
                                          where="parent manifest"),
        }
        result = join_joint_quanta(receipts=None, campaign=campaign,
                                   output_dir=args.output_dir,
                                   input_root=args.input_root, records_dir=args.records)
    except JoinRefused as exc:
        print(f"joint_quanta_join: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps({"status": result["status"],
                      "gaps": result["gaps"],
                      "coverage_sha256": result["coverage_sha256"],
                      "joint_cost_path": result["joint_cost_path"],
                      "results_path": result["results_path"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
