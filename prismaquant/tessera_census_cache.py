"""Close a Tessera census's measured wires for an assignment, without a joint handoff.

``tessera_export_lane.selected_cached_units_manifest`` closes the wires a
*joint* allocation selected, and reads its evidence from the joint handoff.
A census allocation (or a hand-written uniform control over a census) has no
joint handoff: its evidence is the census cost table, the census's checkpoint
seal, and the per-unit checkpoint journals that hold every measured wire
receipt -- dense receipts live only there.

This module keeps that function's refusals and adds the checks the census
evidence allows:

* the assignment covers the complete census roster, and any unit it names
  outside the roster passes through at BF16;
* the layer config carries the census's exact producer projection, wire
  directory and stack formats;
* every selected Tessera cell is a measured census row
  (``output_mse_measured is True``) whose journal receipt exists;
* each receipt's source, encoder and Hessian equal the checkpoint seal, its
  schema matches the unit kind, its source shape matches the census, its
  byte count matches the measured row, and a routed receipt equals both the
  cost table's and the layer config's;
* every referenced blob sits inside the census wire directory with its
  recorded byte count.

Blob content is not hashed here by default.  The exporter hashes every blob as
it reads it (``tessera.cached_unit.verify_cached_unit``, with the identity
recomputed from source and Hessian), so a build-time pass would read the whole
selected wire set (153.8 GB, about 2700 s of a 2978 s GLM-5.3 control build) only to
refuse earlier.  ``hash_blobs=True`` adds that pass back as an audit.

The seal is the checkpoint manifest's ``identity``, recomputed here from the
manifest's bytes and compared to its stored digest; every journal envelope is
loaded against that digest.  Nothing is encoded, interpolated or re-priced.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .cost_stage_checkpoint import MANIFEST_SCHEMA, _load_unit, unit_path
from .layer_config import LAYER_CONFIG_META_KEY
from .tessera_expert_projection import (
    EXPERT_WIRES_KEY, POPULATION_KEY, PROJECTION_KEY, STACK_FORMATS_KEY, WIRE_DIR_KEY,
    ExpertProjectionError, allocation_expert_projection_block, cached_units_manifest,
    carried_units, check_expert_wire_receipt, expand_stack_decision_assignment,
    locate_expert_wire, require_stack_uniform_assignment, verify_expert_wire_record,
)
from .tessera_joint_aura import STAGE

ROSTER_SCHEMA = "prismaquant.tessera_census_seal_roster.v1"
LAYER_CONFIG_META_SCHEMA = "prismaquant.layer_config_meta.v1"


class CensusCacheError(ValueError):
    """The census evidence does not close the selected wires."""


# ---------------------------------------------------------------------------
# The checkpoint seal
# ---------------------------------------------------------------------------
_STREAM_DEPTH = 2


def _canonical_chunks(value: Any, depth: int):
    encode = json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode
    if depth >= _STREAM_DEPTH or not isinstance(value, (dict, list)):
        yield encode(value)
        return
    if isinstance(value, dict):
        yield "{"
        for index, key in enumerate(sorted(value)):
            if not isinstance(key, str):
                raise CensusCacheError("checkpoint identity has a non-string key")
            yield ("," if index else "") + encode(key) + ":"
            yield from _canonical_chunks(value[key], depth + 1)
        yield "}"
    else:
        yield "["
        for index, item in enumerate(value):
            if index:
                yield ","
            yield from _canonical_chunks(item, depth + 1)
        yield "]"


def canonical_json_sha256_of_loaded(value: Any) -> str:
    """``cost_stage_checkpoint.canonical_json_sha256`` for data ``json.load`` produced.

    The checkpoint's own function encodes, decodes and re-encodes the whole
    value in memory -- three copies of a 7 GB identity.  For a value that is
    already JSON-decoded the round trip is the identity, so the same bytes
    are streamed into the digest instead, the outer two levels chunk by chunk.
    """
    digest = hashlib.sha256()
    for chunk in _canonical_chunks(value, 0):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def seal_roster(manifest: Mapping[str, Any]) -> dict:
    """The per-unit seal a census manifest commits to, after recomputing its digest."""
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("stage") != STAGE:
        raise CensusCacheError("not a Tessera campaign checkpoint manifest")
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise CensusCacheError("checkpoint manifest has no identity")
    actual = canonical_json_sha256_of_loaded(identity)
    if actual != manifest.get("identity_sha256"):
        raise CensusCacheError(
            f"checkpoint identity digest {actual} differs from its stored seal "
            f"{manifest.get('identity_sha256')}")
    units = identity.get("units")
    listed = [entry.get("qname") for entry in manifest.get("units", [])]
    if (not isinstance(units, Mapping) or len(set(listed)) != len(listed)
            or set(listed) != set(units)):
        raise CensusCacheError("checkpoint journal list differs from its sealed unit roster")
    for entry in manifest["units"]:
        if entry.get("file") != unit_path(Path("."), entry["qname"]).as_posix():
            raise CensusCacheError(f"{entry['qname']}: journal file is not its checkpoint path")
    roster = {}
    for name, unit in sorted(units.items()):
        if not isinstance(unit, Mapping) or "weight" not in unit or "hessian" not in unit:
            raise CensusCacheError(f"{name}: sealed unit has no weight and Hessian identity")
        roster[name] = {"weight": unit["weight"], "hessian": unit["hessian"]}
    return {"schema": ROSTER_SCHEMA, "identity_sha256": actual,
            "encoder_source_sha256": identity.get("encoder_source_sha256"),
            "units": roster}


# ---------------------------------------------------------------------------
# Assignments over a census
# ---------------------------------------------------------------------------
def _measured(cell: Any) -> bool:
    return isinstance(cell, Mapping) and cell.get("output_mse_measured") is True


def uniform_assignment(cost: Mapping[str, Any], fmt: str) -> dict[str, str]:
    """Every census unit at ``fmt``, refused unless every unit measured that cell."""
    costs = cost.get("costs")
    if not isinstance(costs, Mapping) or not costs:
        raise CensusCacheError("census cost table has no rows")
    unmeasured = sorted(name for name, rows in costs.items()
                        if not _measured((rows or {}).get(fmt)))
    if unmeasured:
        raise CensusCacheError(
            f"{len(unmeasured)} census units have no measured {fmt} cell "
            f"(first: {unmeasured[0]})")
    return {name: fmt for name in sorted(costs)}


def census_layer_config(cost: Mapping[str, Any], assignment: Mapping[str, str], *,
                        entry_for: Callable[[str], dict], target_profile: str) -> dict:
    """A layer config for a hand-written census assignment, with the allocator's block.

    ``entry_for`` is the format registry's ``autoround_config`` spelling, so a
    control's entries are shaped exactly like an allocation's.
    """
    costs = cost.get("costs") or {}
    if set(assignment) != set(costs):
        raise CensusCacheError("assignment does not name exactly the census units")
    config: dict[str, Any] = {name: entry_for(fmt) for name, fmt in sorted(assignment.items())}
    try:
        block = allocation_expert_projection_block(cost, assignment)
    except ExpertProjectionError as exc:
        raise CensusCacheError(f"expert projection: {exc}") from exc
    config[LAYER_CONFIG_META_KEY] = {"schema": LAYER_CONFIG_META_SCHEMA,
                                     "target_profile": target_profile, **block}
    return config


def plan_layer_config_projection(config: Mapping[str, Any],
                                 cost: Mapping[str, Any]) -> tuple[dict, list[str]]:
    """The layer config a Tessera planner reads: census units and ``__*`` keys.

    An allocator's layer config also names Linears kept outside the priced
    population (GLM-5.3 Flash: 124 visual-tower BF16 rows). The census has no
    wire for them and the planner's body projection does not name them, so the
    planner refuses them. Returns the projection and the sorted names dropped.
    Call it only after :func:`selected_census_assignment`, which refuses a
    dropped name that is not BF16 passthrough.
    """
    costs = cost.get("costs") or {}
    projected = {key: value for key, value in config.items()
                 if key in costs or key.startswith("__")}
    return projected, sorted(key for key in config if key not in projected)


def selected_census_assignment(assignment: Mapping[str, str], metadata: Mapping[str, Any],
                               cost: Mapping[str, Any]) -> tuple[dict[str, str], dict, dict, dict]:
    """``(selected, source, units, stack_of)`` after the roster and projection checks."""
    costs = cost.get("costs") or {}
    carried = (cost.get("provenance") or {}).get(PROJECTION_KEY)
    if carried is None or metadata.get(PROJECTION_KEY) != carried:
        raise CensusCacheError("selected cache needs the exact carried producer projection")
    try:
        source, units, stack_of = carried_units(carried)
        selected, _owners = expand_stack_decision_assignment(
            assignment, metadata.get(POPULATION_KEY), units=units, stack_of=stack_of,
            costs=costs)
        stack_formats = require_stack_uniform_assignment(
            {name: selected[name] for name in units if name in selected}, stack_of, units)
    except (ExpertProjectionError, KeyError) as exc:
        raise CensusCacheError(f"selected cache projection: {exc}") from exc
    if not set(costs) <= set(selected):
        raise CensusCacheError("selected cache assignment does not cover the full source roster")
    # An allocator's layer config also names the Linears it kept outside the
    # priced population (GLM-5.3 Flash: 124 visual-tower Linears at BF16). The
    # census holds no wire for them, so they may only pass through at BF16.
    outside = {name: fmt for name, fmt in selected.items() if name not in costs}
    wired = sorted(f"{name}@{fmt}" for name, fmt in outside.items() if fmt != "BF16")
    if wired:
        raise CensusCacheError(
            f"{len(wired)} selected unit(s) outside the census roster are not BF16 passthrough "
            f"(first: {wired[0]})")
    selected = {name: fmt for name, fmt in selected.items() if name in costs}
    if metadata.get(STACK_FORMATS_KEY) != stack_formats:
        raise CensusCacheError("selected cache stack formats differ from the assignment")
    return dict(selected), source, units, stack_of


def load_selected_wire_records(parts_root: Path, selected: Mapping[str, str], *,
                               identity_sha256: str, workers: int) -> dict[str, dict | None]:
    """Each selected cell's journal receipt, from envelopes bound to the seal digest."""
    def one(item):
        name, fmt = item
        try:
            state = _load_unit(unit_path(Path(parts_root), name), stage=STAGE, qname=name,
                               identity_sha256=identity_sha256)
        except (OSError, RuntimeError) as exc:
            raise CensusCacheError(f"{name}: {exc}") from exc
        records = state.get("wire_records")
        return name, (records.get(fmt) if isinstance(records, Mapping) else None)

    wanted = sorted((n, f) for n, f in selected.items() if f != "BF16")
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        return dict(pool.map(one, wanted))


def _verify_dense_blob(name: str, fmt: str, record: Mapping[str, Any], wire_dir: Path, *,
                       hash_blob: bool) -> None:
    path = wire_dir / record["file"]
    if path.is_symlink() or path.resolve().parent != wire_dir or not path.is_file():
        raise CensusCacheError(f"{name}@{fmt}: dense wire escapes the campaign directory")
    if not hash_blob:
        if path.stat().st_size != record["blob_bytes"]:
            raise CensusCacheError(f"{name}@{fmt}: dense wire differs from measured receipt")
        return
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
            size += len(block)
    if size != record["blob_bytes"] or digest.hexdigest() != record["blob_sha256"]:
        raise CensusCacheError(f"{name}@{fmt}: dense wire differs from measured receipt")


def census_selected_cached_units_manifest(
        assignment: Mapping[str, str], metadata: Mapping[str, Any], cost: Mapping[str, Any],
        roster: Mapping[str, Any], census_shapes: Mapping[str, Any],
        records: Mapping[str, Any], *, input_schema: str, encoding_input_schema: str,
        cache_schema: str, blob_workers: int = 8, hash_blobs: bool = False) -> dict:
    """The ``tessera.cached_units.v1`` bundle of exactly the census wires an assignment selected.

    Schemas are the producer's constants, passed in by a caller that imported
    them.  ``records`` is :func:`load_selected_wire_records`'s answer.  Each
    blob is located and sized on ``blob_workers`` threads; ``hash_blobs`` also
    hashes it (see the module docstring).
    """
    from .tessera_formats import parse_tessera_format_name

    costs = cost.get("costs") or {}
    if (roster.get("schema") != ROSTER_SCHEMA or set(census_shapes) != set(costs)
            or set(roster.get("units") or {}) != set(costs)):
        raise CensusCacheError("selected cache requires the complete census roster")
    selected, source, units, _stack_of = selected_census_assignment(assignment, metadata, cost)
    provenance = cost.get("provenance") or {}
    wire_dir = Path(provenance["wire_dir"]).resolve()
    if metadata.get(WIRE_DIR_KEY) != str(wire_dir) or provenance.get("wire_dir") != str(wire_dir):
        raise CensusCacheError("selected cache wire directory differs from the census")
    receipts = metadata.get(EXPERT_WIRES_KEY)
    if not isinstance(receipts, Mapping):
        raise CensusCacheError("selected cache expert receipts are missing")
    priced_wires = cost.get(EXPERT_WIRES_KEY) or {}

    checked: dict[str, dict] = {}
    blobs: list[Callable[[], Any]] = []
    for name, fmt in sorted(selected.items()):
        if fmt == "BF16":
            continue
        parsed = parse_tessera_format_name(fmt)
        if parsed is None:
            raise CensusCacheError(f"{name}: selected {fmt} is not a Tessera cached wire")
        family, q256 = parsed
        grid = family.payload_grid().name
        row = costs[name].get(fmt)
        record = records.get(name)
        if not _measured(row) or not isinstance(record, Mapping):
            raise CensusCacheError(
                f"{name}@{fmt}: selected rung has no exact measured census wire; "
                "acquire and qualify that wire before export")
        identity = record.get("identity") or {}
        sealed = roster["units"][name]
        if identity.get("source") != sealed["weight"]:
            raise CensusCacheError(f"{name}@{fmt}: selected wire source differs from checkpoint seal")
        if identity.get("encoder_source_sha256") != roster.get("encoder_source_sha256"):
            raise CensusCacheError(f"{name}@{fmt}: selected wire encoder differs from checkpoint seal")
        expected_hessian = (sealed["hessian"] if
                            (row.get("hessian_identity") or {}).get("applied") is True else None)
        if (identity.get("calibration") or {}).get("hessian") != expected_hessian:
            raise CensusCacheError(f"{name}@{fmt}: selected wire Hessian differs from checkpoint seal")
        if identity.get("schema") != (input_schema if name in units else encoding_input_schema):
            raise CensusCacheError(f"{name}@{fmt}: cached identity schema differs from unit kind")
        if list((identity.get("source") or {}).get("shape") or ()) != list(census_shapes[name]):
            raise CensusCacheError(f"{name}@{fmt}: wire source geometry differs from the census")
        if row.get("wire_bytes") != record.get("blob_bytes"):
            raise CensusCacheError(f"{name}@{fmt}: measured wire_bytes differs from the wire receipt")
        if name in units:
            if (priced_wires.get(name) or {}).get(fmt) != record:
                raise CensusCacheError(
                    f"{name}@{fmt}: cost table expert receipt differs from the checkpoint journal")
            if receipts.get(name) != record:
                raise CensusCacheError(f"{name}@{fmt}: selected expert receipt differs from measured wire")
            try:
                check_expert_wire_receipt(record, name=name, unit=units[name], q256=int(q256), grid=grid)
            except ExpertProjectionError as exc:
                raise CensusCacheError(f"{name}@{fmt}: {exc}") from exc
            blobs.append(lambda n=name, f=fmt, r=record, u=units[name], q=int(q256), g=grid:
                         _verify_routed_blob(n, f, r, u, q, g, wire_dir, hash_blobs))
        else:
            recipe = identity.get("recipe") or {}
            if identity.get("unit") != name or recipe.get("grid") != grid or recipe.get("q256") != int(q256):
                raise CensusCacheError(f"{name}@{fmt}: dense wire identity differs from selected rung")
            blobs.append(lambda n=name, f=fmt, r=record:
                         _verify_dense_blob(n, f, r, wire_dir, hash_blob=hash_blobs))
        checked[name] = dict(record)
    if not checked:
        raise CensusCacheError("selected cache has no selected Tessera wires")
    with ThreadPoolExecutor(max_workers=max(1, int(blob_workers))) as pool:
        for future in [pool.submit(check) for check in blobs]:
            future.result()
    try:
        return cached_units_manifest(source, checked, schema=cache_schema)
    except ExpertProjectionError as exc:
        raise CensusCacheError(f"selected cache: {exc}") from exc


def _verify_routed_blob(name, fmt, record, unit, q256, grid, wire_dir, hash_blob) -> None:
    # check_expert_wire_receipt already ran on this record in the caller.
    try:
        if hash_blob:
            verify_expert_wire_record(record, name=name, unit=unit, q256=q256, grid=grid,
                                      wire_dir=wire_dir)
        else:
            locate_expert_wire(record, name=name, wire_dir=wire_dir)
    except ExpertProjectionError as exc:
        raise CensusCacheError(f"{name}@{fmt}: {exc}") from exc
