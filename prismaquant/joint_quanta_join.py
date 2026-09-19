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

Shapes owned elsewhere (fixtures here, never imports): the layer-quantum
record (§3, ``prismaquant.joint_layer_quanta.v1``, built in parallel by the
producer) and the per-quantum ``cost.pkl`` / ``status.json`` (§6.4, built in
parallel by the runtime). This module checks the contract's schemas and
digests; it does not construct producer or runtime records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

from prismaquant.cost_stage_checkpoint import (
    atomic_write_bytes,
    canonical_json_bytes,
    canonical_json_sha256,
)
from prismaquant.joint_aura import validate_joint_aura_entry

RECORD_SCHEMA = "prismaquant.joint_layer_quanta.v1"
STATUS_SCHEMA = "prismaquant.joint_layer_quantum.status.v1"
JOINED_RESULTS_SCHEMA = "prismaquant.joint_layer_quanta.joined_results.v1"

#: The single run seals ``joint-cost.pkl`` with this call
#: (``tessera_joint_aura.run``); the join seals the merged payload the same
#: way so readers see one pickle convention.
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

#: Exit code when the join refuses (custody, coverage-defect, or row defect).
#: Gapped campaigns still exit 0 — a gap is a state, not an error.
EXIT_REFUSED = 1

#: Provenance keys every per-layer payload must carry (§6.4). The first four
#: are the shared campaign binding (identical across quanta); the last two
#: are the quantum's own identity.
REQUIRED_PROVENANCE_KEYS = (
    "plan_sha256",
    "prepared_sha256",
    "campaign_scope",
    "implementation_digest",
    "quantum_id",
    "identity_sha256",
)


class JoinRefused(Exception):
    """The join failed closed: custody, coverage, or a row defect."""


class GappedPayloadRefused(Exception):
    """The allocation stage refused a gapped joined payload."""


def quantum_id_for_layer(layer: int) -> str:
    return f"layer-{layer:03d}"


def _roster_sha256(roster: list[str]) -> str:
    return hashlib.sha256(("\n".join(roster) + "\n").encode()).hexdigest()


def _load_json(path: Path, *, where: str) -> object:
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise JoinRefused(f"{where}: unreadable JSON at {path}: {exc}") from exc


def _sha_file(path: Path, *, where: str) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise JoinRefused(f"{where}: unreadable file at {path}: {exc}") from exc


def _check_digest(path: Path, expected: str, *, where: str) -> None:
    actual = _sha_file(path, where=where)
    if actual != expected:
        raise JoinRefused(
            f"{where}: digest mismatch at {path}: expected {expected}, got {actual}")


def _scan_receipts(input_root: Path) -> list[dict]:
    """Build the receipt set from the campaign output root.

    Layout (the joiner's fixture of the producer/runtime layout, §3.1/§6.4):
    ``layer-quanta/records/layer-NNN.json`` records and
    ``layer-quanta/layer-NNN/{cost.pkl,status.json}`` quantum spaces.
    """
    records_dir = input_root / "layer-quanta" / "records"
    if not records_dir.is_dir():
        raise JoinRefused(f"coverage: no records directory at {records_dir}")
    receipts = []
    for record_path in sorted(records_dir.glob("layer-*.json")):
        quantum_id = record_path.stem
        space = input_root / "layer-quanta" / quantum_id
        receipts.append({
            "quantum_id": quantum_id,
            "record_path": str(record_path),
            "cost_path": str(space / "cost.pkl"),
            "status_path": str(space / "status.json"),
        })
    if not receipts:
        raise JoinRefused(f"coverage: no layer records at {records_dir}")
    return receipts


def _check_record(record: object, receipt: dict, campaign: dict) -> dict:
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
    for key in ("plan_sha256", "prepared_sha256"):
        if binding.get(key) != campaign[key]:
            raise JoinRefused(
                f"{where}: record {key} {binding.get(key)!r} is not this "
                f"campaign ({campaign[key]!r})")
    if binding.get("read_manifest_sha256") != campaign["manifest_sha256"]:
        raise JoinRefused(
            f"{where}: record read-manifest digest is not this campaign's")
    if binding.get("campaign_scope") != campaign["scope"]:
        raise JoinRefused(f"{where}: record scope is not this campaign's scope")
    if binding.get("unit_roster_sha256") != _roster_sha256(campaign["roster"]):
        raise JoinRefused(f"{where}: record roster digest is not this roster")
    return record


def _check_tiling(records: dict[str, dict], campaign: dict) -> None:
    """Coverage step 1: every present record's source phase cites its parent
    phase by name and byte range. Absent phases are gaps (handled by the
    caller), never silent holes: only names the parent manifest knows are
    admitted."""
    parent = campaign["parent_manifest"]
    phases = parent.get("phases")
    if isinstance(parent, dict) and phases is None and isinstance(
            parent.get("annotations"), dict):
        phases = parent["annotations"].get("phases")
    if not isinstance(phases, list) or not phases:
        raise JoinRefused("coverage: parent manifest carries no phase table")
    by_name = {}
    for phase in phases:
        if not isinstance(phase, dict):
            raise JoinRefused("coverage: parent manifest phase is malformed")
        by_name[phase.get("name")] = phase
    for quantum_id, record in sorted(records.items()):
        where = f"coverage {quantum_id}"
        source = record.get("read_set", {}).get("source_phase")
        if not isinstance(source, dict):
            raise JoinRefused(f"{where}: record cites no source phase")
        parent_phase = by_name.get(source.get("name"))
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


def _load_cost_payload(receipt: dict, record: dict, campaign: dict) -> dict:
    """Custody step 2: the payload's provenance block equals the campaign
    binding; only per-layer content may differ."""
    quantum_id = receipt["quantum_id"]
    where = f"custody {quantum_id}"
    cost_path = Path(receipt["cost_path"])
    if receipt.get("cost_sha256") is not None:
        _check_digest(cost_path, receipt["cost_sha256"], where=where)
    try:
        payload = pickle.loads(cost_path.read_bytes())
    except (OSError, ValueError, pickle.UnpicklingError) as exc:
        raise JoinRefused(f"{where}: unreadable cost payload: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(
            payload.get("costs"), dict) or not isinstance(
            payload.get("provenance"), dict):
        raise JoinRefused(f"{where}: cost payload shape is not §6.4")
    provenance = payload["provenance"]
    for key in REQUIRED_PROVENANCE_KEYS:
        if key not in provenance:
            raise JoinRefused(f"{where}: provenance lacks {key!r}")
    if provenance["quantum_id"] != quantum_id or (
            provenance["identity_sha256"] != record["identity_sha256"]):
        raise JoinRefused(
            f"{where}: payload answers for another quantum's identity "
            "(retargeted receipt)")
    expected = {"plan_sha256": campaign["plan_sha256"],
                "prepared_sha256": campaign["prepared_sha256"],
                "campaign_scope": campaign["scope"],
                "implementation_digest": campaign["implementation_digest"]}
    for key, value in expected.items():
        if provenance[key] != value:
            raise JoinRefused(
                f"{where}: payload provenance {key} is foreign to this campaign")
    return payload


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


def _record_units(record: dict) -> list[str]:
    units = []
    for window in record.get("windows", []):
        units.extend(window.get("names", []))
    return sorted(units)


def _expected_quantum_ids(campaign: dict) -> set[str]:
    """The quantum set the parent manifest declares. The manifest is caller
    input, so a lost record names its gap instead of shrinking the set."""
    parent = campaign["parent_manifest"]
    phases = parent.get("phases")
    if phases is None and isinstance(parent.get("annotations"), dict):
        phases = parent["annotations"].get("phases")
    return {phase.get("quantum_id") or phase.get("name") for phase in phases}


def join_joint_quanta(*, receipts: list[dict] | None, campaign: dict,
                      output_dir: str | Path,
                      input_root: str | Path | None = None,
                      now: float | None = None) -> dict:
    """Merge per-layer cost payloads into the campaign's results shape.

    ``receipts`` is one ``{quantum_id, record_path[, cost_path, cost_sha256,
    identity_sha256, status_path]}`` per quantum; when None the receipt set
    is scanned from ``input_root`` (required then). ``campaign`` is the
    caller-supplied binding — plan/prepared/manifest digests, scope,
    implementation digest, roster, ``formats_by_qname``, and the parent
    manifest — never derived from the surviving shards.

    Returns ``{"status": "complete"|"gapped", "gaps": [...],
    "joint_cost_path": ..., "results_path": ..., "coverage_sha256": ...}``.
    Raises :class:`JoinRefused` on custody, coverage-defect, or row defects.
    """
    if receipts is None:
        if input_root is None:
            raise JoinRefused("coverage: no receipts and no input root")
        receipts = _scan_receipts(Path(input_root))
    if not receipts:
        raise JoinRefused("coverage: empty receipt set")
    quantum_ids = [r["quantum_id"] for r in receipts]
    if len(set(quantum_ids)) != len(quantum_ids):
        raise JoinRefused("coverage: duplicated quantum ids in receipt set")
    for key in ("plan_sha256", "prepared_sha256", "manifest_sha256", "scope",
                "implementation_digest", "roster", "formats_by_qname",
                "parent_manifest"):
        if key not in campaign:
            raise JoinRefused(f"coverage: campaign binding lacks {key!r}")

    records: dict[str, dict] = {}
    for receipt in sorted(receipts, key=lambda r: r["quantum_id"]):
        record = _load_json(Path(receipt["record_path"]),
                            where=f"custody {receipt['quantum_id']}")
        records[receipt["quantum_id"]] = _check_record(
            record, receipt, campaign)
        receipt["identity_sha256"] = record["identity_sha256"]
    _check_tiling(records, campaign)

    merged: dict[str, dict] = {}
    gaps: list[dict] = []
    per_layer: list[dict] = []
    for quantum_id in sorted(records):
        receipt = next(r for r in receipts if r["quantum_id"] == quantum_id)
        record = records[quantum_id]
        status, units = _read_status(receipt)
        if status != "complete":
            units_named = _record_units(record)
            gaps.append({"quantum_id": quantum_id,
                         "unit_count": len(units_named),
                         "units": units_named})
            per_layer.append({"quantum_id": quantum_id,
                              "identity_sha256": record["identity_sha256"],
                              "status": status, "units": units})
            continue
        payload = _load_cost_payload(receipt, record, campaign)
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
                          "cost_sha256": hashlib.sha256(
                              Path(receipt["cost_path"]).read_bytes()
                          ).hexdigest(),
                          "status": status, "units": units})

    # A quantum with no record at all is still a named gap, never a shrunk
    # layer set: the expected set comes from the parent manifest, supplied by
    # the caller, not from the surviving shards.
    absent = sorted(_expected_quantum_ids(campaign) - set(records))
    for quantum_id in absent:
        gaps.append({"quantum_id": quantum_id, "unit_count": None,
                     "units": [], "record_absent": True})
        per_layer.append({"quantum_id": quantum_id, "status": "absent",
                          "units": [0, 0]})

    roster = list(campaign["roster"])
    if sorted(merged) != roster and not gaps:
        missing = [q for q in roster if q not in merged]
        raise JoinRefused(
            f"coverage: complete campaign is short {len(missing)} roster "
            f"units, first {missing[:3]!r} (a complete quantum dropped rows)")
    if gaps and not absent:
        # Every gap named its units from its record, yet roster units are
        # still missing: a complete quantum dropped rows — a defect, not
        # a gap.
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
        "roster_sha256": _roster_sha256(roster),
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
            "implementation_digest": campaign["implementation_digest"],
            "coverage": {**coverage_proof, "coverage_sha256": coverage_sha256,
                         "status": status,
                         "gaps": gaps},
            "join_schema": JOINED_RESULTS_SCHEMA,
        },
    }
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
                     ("plan_sha256", "prepared_sha256", "manifest_sha256",
                      "implementation_digest")},
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
    parser.add_argument("--input-root", required=True)
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
    parser.add_argument("--implementation-digest", required=True)
    args = parser.parse_args(argv)

    try:
        _check_digest(Path(args.plan), args.plan_sha256, where="campaign plan")
        _check_digest(Path(args.prepared), args.prepared_sha256,
                      where="campaign prepared")
        _check_digest(Path(args.manifest), args.manifest_sha256,
                      where="campaign manifest")
        campaign = {
            "plan_sha256": args.plan_sha256,
            "prepared_sha256": args.prepared_sha256,
            "manifest_sha256": args.manifest_sha256,
            "scope": _load_json(Path(args.scope), where="campaign scope"),
            "implementation_digest": args.implementation_digest,
            "roster": _read_text_list(Path(args.roster)),
            "formats_by_qname": _load_json(Path(args.formats_by_qname),
                                           where="formats_by_qname"),
            "parent_manifest": _load_json(Path(args.manifest),
                                          where="parent manifest"),
        }
        result = join_joint_quanta(receipts=None, campaign=campaign,
                                   output_dir=args.output_dir,
                                   input_root=args.input_root)
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
