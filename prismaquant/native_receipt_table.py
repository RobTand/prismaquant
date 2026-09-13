"""Assemble a ``prismaquant.measured_runtime_prices.v2`` table from native receipts.

One table row is one Tessera ``tessera.native_dense_operator_receipt.v1`` bound
to the panel it was frozen against, the joint AURA cost row that panel names,
and the runtime relation the receipt's runtime must belong to. Every binding
here is read off the receipts through the same consumer the loader reuses
(``native_operator_panel.consume_native_receipt``); nothing is restated from a
producer summary, and nothing is defaulted where the evidence is absent.

What this module does not do: it does not admit anything. The written table
is handed to ``measured_runtime_prices.load_measured_runtime_table``, whose
producer admission (``runtime_provenance.admit_runtime_provenance``) decides.
The emission report next to the table records that verdict verbatim, so a
table that exists on disk without an admitted verdict is never mistaken for a
price the allocator may read: ``build_runtime_resources`` refuses a v2 table
the loader did not admit.

Fixed resources are declared from PrismaQuant's own recomputation of the
full-engine resource report (``full_engine_resource_report``). A term the
report cannot express is declared ``0`` and recorded as unevidenced in the
emission report; ``admit_fixed_resources`` then refuses it by name, which is
the truthful state of the axis (PQ #237, #420) rather than a number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from .measured_runtime_prices import (
    PROVENANCE_CONTEXT_SCHEMA, PROVENANCE_IDENTITY_KIND, PROVENANCE_TABLE_SCHEMA,
    RuntimePriceError, _object, _string, identity_sha256, load_measured_runtime_table,
    parse_runtime_context,
)
from .runtime_provenance import FIXED_TERM_FIELDS, SCHEMA as RELATION_SCHEMA

EMISSION_SCHEMA = "prismaquant.native_receipt_table_emission.v1"
DENSE_PANEL_SCHEMA = "tessera.native_dense_panel.v1"
BINDING_FIELDS = ("unit", "format", "run_id", "panel", "receipt", "memory_trace")
PHASES = ("prefill", "decode")
#: Fields of the fixed charge this report schema carries no observation for at
#: all. They are declared ``0`` and named as unevidenced; the gate refuses them.
UNOBSERVED_FIXED_FIELDS = ("prefill_ms", "decode_ms", "serialized_bytes")


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _resolve(path: str | Path, base: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


def _table_path(path: Path, table_dir: Path) -> str:
    """Spell an artifact the way ``ArtifactReader`` resolves it from the table."""
    path = path.resolve()
    try:
        return str(path.relative_to(table_dir.resolve()))
    except ValueError:
        return str(path)


def _reference(path: Path, table_dir: Path, what: str) -> tuple[Path, dict]:
    if not path.is_file():
        raise RuntimePriceError(f"{what} is missing: {path}")
    return path, {"path": _table_path(path, table_dir), "sha256": file_sha256(path)}


def _json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimePriceError(f"{what} is not readable JSON: {path}: {exc}") from exc


def load_relation(path: Path) -> tuple[Mapping, dict]:
    """The relation the context names; its identity is the context's runtime_sha256.

    Only existence and schema are checked here. Whether the relation is a
    complete, exact, same-device account is ``load_runtime_relation``'s
    verdict, reported by the loader rather than pre-empted by this emitter.
    """
    if not path.is_file():
        raise RuntimePriceError(f"runtime provenance relation is missing: {path}")
    relation = _json(path, "runtime provenance relation")
    if not isinstance(relation, Mapping) or relation.get("schema") != RELATION_SCHEMA:
        raise RuntimePriceError(f"runtime provenance relation schema is not {RELATION_SCHEMA}")
    return relation, {"path": None, "sha256": file_sha256(path)}


def load_cost_payload(path: Path) -> tuple[Mapping, str]:
    """Our own pickled cost payload; the digest is over the exact file bytes.

    The allocator identifies its ``--costs`` file the same way
    (``hashlib.file_digest`` over the bytes), so a panel frozen against these
    bytes binds to this allocator input and to no other.
    """
    if not path.is_file():
        raise RuntimePriceError(f"cost payload is missing: {path}")
    with path.open("rb") as stream:
        try:
            payload = pickle.load(stream)
        except Exception as exc:  # pickle raises many concrete types
            raise RuntimePriceError(f"cost payload is not a readable pickle: {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("costs"), Mapping):
        raise RuntimePriceError(f"cost payload carries no costs mapping: {path}")
    return payload, file_sha256(path)


def bind_cost_row(panel: Mapping, cost_payload: Mapping, cost_sha256: str) -> Mapping:
    """The joint AURA row this panel was frozen against, or a named refusal."""
    from .joint_aura import validate_joint_aura_entry

    unit, fmt = panel["unit"], panel["format"]
    where = f"{unit}@{fmt}"
    row = cost_payload["costs"].get(unit, {}).get(fmt) if isinstance(cost_payload["costs"].get(unit), Mapping) else None
    if row is None:
        raise RuntimePriceError(f"cost payload has no row for {where}")
    try:
        joint = validate_joint_aura_entry(row)
    except (ValueError, TypeError, KeyError):
        joint = False
    if not joint:
        raise RuntimePriceError(f"cost payload row is not joint AURA currency: {where}")
    if panel["cost_sha256"] != cost_sha256:
        raise RuntimePriceError(f"panel cost_sha256 is not the digest of the supplied cost payload: {where}")
    if (panel["joint_operator_identity_sha256"] != row["joint_operator_identity_sha256"]
            or panel["joint_operator_identity"] != row["joint_operator_identity"]):
        raise RuntimePriceError(f"panel joint operator identity differs from the cost payload row: {where}")
    if panel["probe_identity_sha256"] != row["probe_identity_sha256"]:
        raise RuntimePriceError(f"panel probe identity differs from the cost payload row: {where}")
    return row


def bind_native_receipt(spec: Mapping, *, cost_payload: Mapping, cost_sha256: str,
                        manifest_dir: Path, table_dir: Path) -> dict:
    """One receipt binding -> one table row, one receipt binding, one observation.

    The receipt is consumed through ``consume_native_receipt`` against the
    panel as frozen, exactly as ``admit_native_rows`` will consume it again.
    """
    from .native_operator_panel import consume_native_receipt

    _object(spec, BINDING_FIELDS, "native receipt binding")
    unit, fmt, run_id = (_string(spec[key], "native receipt " + key) for key in ("unit", "format", "run_id"))
    where = f"{unit}@{fmt}"
    panel_path, panel_ref = _reference(_resolve(spec["panel"], manifest_dir), table_dir, f"native panel for {where}")
    receipt_path, receipt_ref = _reference(_resolve(spec["receipt"], manifest_dir), table_dir, f"native receipt for {where}")
    trace_path, trace_ref = _reference(_resolve(spec["memory_trace"], manifest_dir), table_dir, f"native memory trace for {where}")
    panel = _json(panel_path, "native panel")
    if not isinstance(panel, Mapping) or panel.get("schema") != DENSE_PANEL_SCHEMA:
        raise RuntimePriceError("unsupported native producer panel")
    if (panel["unit"], panel["format"]) != (unit, fmt):
        raise RuntimePriceError(f"native panel names {panel['unit']}@{panel['format']}, not {where}")
    cost_row = bind_cost_row(panel, cost_payload, cost_sha256)
    try:
        observation = consume_native_receipt(receipt_path, expected_sha256=receipt_ref["sha256"],
                                             expected_panel=panel, memory_trace_path=trace_path)
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimePriceError(f"native producer admission refused: {exc}") from exc
    scratch, activation, measurements = [], [], {}
    for phase in PHASES:
        actual = observation["phases"][phase]
        if actual["peak_scratch_bytes"] is None:
            raise RuntimePriceError(f"native row has an incomplete resource ledger: {where}")
        scratch.append(actual["peak_scratch_bytes"])
        activation.append(actual["input_bytes"])
        timing = actual["measurement"]
        measurements[phase] = {"method": timing["method"], "samples_ms": list(timing["samples_ms"]),
                               "warmup_iterations": timing["warmup_iterations"],
                               "receipt_path": receipt_ref["path"], "receipt_sha256": receipt_ref["sha256"]}
    route = panel["phases"]["prefill"]["expected_route"]["symbol"]
    row = {
        "unit": unit, "format": fmt,
        "binding": {"member_formats": {unit: fmt},
                    "member_operator_identity_sha256": {unit: panel["joint_operator_identity_sha256"]},
                    "member_shapes": {unit: list(panel["shape"])}, "operator_route": route},
        "resources": {"prefill_ms": float(statistics.median(measurements["prefill"]["samples_ms"])),
                      "decode_ms": float(statistics.median(measurements["decode"]["samples_ms"])),
                      "serialized_bytes": observation["serialized_unit_bytes"],
                      "resident_bytes": observation["resident_bytes"],
                      "peak_scratch_bytes": max(scratch), "activation_bytes": max(activation), "kv_bytes": 0},
        "prefill": measurements["prefill"], "decode": measurements["decode"],
    }
    binding = {"unit": unit, "format": fmt, "run_id": run_id,
               "panel": panel_ref, "receipt": receipt_ref, "memory_trace": trace_ref}
    return {"row": row, "binding": binding, "panel": panel, "observation": observation,
            "cost_row_identity_sha256": cost_row["joint_operator_identity_sha256"]}


def derive_context(panels: list[Mapping], *, relation: Mapping) -> dict:
    """The one workload/runtime context every bound panel was frozen under."""
    if not panels:
        raise RuntimePriceError("no native receipts were bound; a table needs at least one row")
    first = panels[0]
    for panel in panels[1:]:
        for what, key in (("runtime", lambda p: identity_sha256(p["runtime"])),
                          ("source model", lambda p: p["source_sha256"]),
                          ("calibration", lambda p: p["calibration_sha256"]),
                          ("prompt token count", lambda p: p["phases"]["prefill"]["m"])):
            if key(panel) != key(first):
                raise RuntimePriceError(f"native receipts were produced on more than one {what}")
    runtime = first["runtime"]
    execution = runtime["execution"]
    gpu = runtime.get("gpu")
    if not isinstance(gpu, Mapping) or not isinstance(gpu.get("uuid"), str) or not isinstance(gpu.get("capability"), list):
        raise RuntimePriceError("native runtime record names no GPU identity")
    major, minor = gpu["capability"]
    for phase in PHASES:
        expected = first["phases"]["prefill"]["m"] if phase == "prefill" else 1
        if any(panel["phases"][phase]["m"] != expected for panel in panels):
            raise RuntimePriceError(f"native {phase} panels do not all run {expected} token(s)")
    return {
        "schema": PROVENANCE_CONTEXT_SCHEMA, "runtime_identity_kind": PROVENANCE_IDENTITY_KIND,
        "serving_context": {"platform": f"sm_{major}{minor}", "structure": "dense",
                            "residency": execution["mode"], "runtime_image": runtime["image"],
                            "execution_mode": execution["execution_mode"]},
        "gpu_identity": gpu["uuid"], "runtime_sha256": identity_sha256(relation),
        "source_sha256": first["source_sha256"], "calibration_sha256": first["calibration_sha256"],
        "prompt_tokens": first["phases"]["prefill"]["m"], "batch_size": 1,
        "tensor_parallel": execution["tensor_parallel"], "graph_mode": execution["execution_mode"],
        "operator_routes": {},
    }


def derive_fixed_resources(report_path: Path, table_dir: Path) -> tuple[dict, dict, dict]:
    """Declare the fixed charge from this consumer's own recomputation of the report."""
    from .full_engine_resource_report import consume_full_engine_resource_report

    _, reference = _reference(report_path, table_dir, "full-engine resource report")
    try:
        verdict = consume_full_engine_resource_report({"path": str(report_path.resolve()), "sha256": reference["sha256"]},
                                                      root=report_path.resolve().parent)
    except RuntimePriceError as exc:
        raise RuntimePriceError(f"full-engine resource report refused: {exc}") from exc
    declared, evidence = {}, {}
    for term, field in FIXED_TERM_FIELDS.items():
        value = verdict.recomputed_terms.get(term)
        if type(value) is int and value >= 0:
            declared[field], evidence[field] = value, f"recomputed {term}"
        else:
            declared[field], evidence[field] = 0, f"no {term} is recomputable; declared 0 without evidence"
    for field in UNOBSERVED_FIXED_FIELDS:
        declared[field], evidence[field] = 0, "the report schema carries no observation for this field; declared 0 without evidence"
    return declared, evidence, {"reference": reference, "refusals": list(verdict.refusals),
                                "expressible_terms": list(verdict.expressible_terms),
                                "open_domains": list(verdict.open_domains)}


def emit_native_receipt_table(*, out: Path, table_id: str, costs: Path, relation: Path,
                              full_engine_report: Path, receipts: list[Mapping], manifest_dir: Path,
                              fixed_assignment: Mapping[str, str], valid_hours: float,
                              now: datetime | None = None) -> dict:
    """Write the table, its fixed-resource receipt and the emission report.

    Returns the emission report. The loader's verdict is inside it under
    ``admission``; a refused table stays on disk only because nothing reads a
    v2 table except through the loader that refused it.
    """
    out = Path(out)
    table_dir = out.parent
    table_dir.mkdir(parents=True, exist_ok=True)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise RuntimePriceError("now must have a timezone")
    relation_payload, relation_ref = load_relation(_resolve(relation, manifest_dir))
    relation_ref["path"] = _table_path(_resolve(relation, manifest_dir), table_dir)
    cost_payload, cost_sha256 = load_cost_payload(_resolve(costs, manifest_dir))
    if not isinstance(receipts, list) or not receipts:
        raise RuntimePriceError("receipt manifest must list at least one native receipt binding")
    bound, seen = [], set()
    for spec in receipts:
        item = bind_native_receipt(spec, cost_payload=cost_payload, cost_sha256=cost_sha256,
                                   manifest_dir=manifest_dir, table_dir=table_dir)
        key = item["row"]["unit"], item["row"]["format"]
        if key in seen:
            raise RuntimePriceError(f"duplicate native receipt for {key[0]}@{key[1]}")
        seen.add(key)
        bound.append(item)
    context = derive_context([item["panel"] for item in bound], relation=relation_payload)
    routes: dict[str, dict[str, str]] = {}
    for item in bound:
        routes.setdefault(item["row"]["unit"], {})[item["row"]["format"]] = item["row"]["binding"]["operator_route"]
    context["operator_routes"] = routes
    parse_runtime_context(context)
    fixed, fixed_evidence, report_verdict = derive_fixed_resources(_resolve(full_engine_report, manifest_dir), table_dir)
    receipt_path = out.with_name(out.stem + ".fixed-resources-receipt.json")
    receipt_path.write_text(json.dumps({"full_model_resources": report_verdict["reference"]},
                                       indent=1, sort_keys=True) + "\n")
    if not isinstance(fixed_assignment, Mapping):
        raise RuntimePriceError("fixed_assignment must be a unit -> format mapping")
    table = {
        "schema": PROVENANCE_TABLE_SCHEMA, "table_id": _string(table_id, "table_id"),
        "status": "proposal_data", "composition": "sequential_operator_sum", "context": context,
        "cost_sha256": cost_sha256,
        "measured_at": current.isoformat().replace("+00:00", "Z"),
        "valid_until": (current + timedelta(hours=float(valid_hours))).isoformat().replace("+00:00", "Z"),
        "fixed_assignment": {str(k): str(v) for k, v in sorted(fixed_assignment.items())},
        "fixed_resources": fixed,
        "fixed_resources_receipt_path": receipt_path.name,
        "fixed_resources_receipt_sha256": file_sha256(receipt_path),
        "rows": [item["row"] for item in sorted(bound, key=lambda i: (i["row"]["unit"], i["row"]["format"]))],
        "runtime_provenance": relation_ref,
        "native_receipt_bindings": [item["binding"] for item in sorted(bound, key=lambda i: (i["row"]["unit"], i["row"]["format"]))],
    }
    out.write_text(json.dumps(table, indent=1, sort_keys=True, allow_nan=False) + "\n")
    try:
        load_measured_runtime_table(out, expected_context=parse_runtime_context(context),
                                    expected_cost_sha256=cost_sha256, now=current)
        admission = {"status": "admitted", "refusal": None}
    except RuntimePriceError as exc:
        admission = {"status": "refused", "refusal": str(exc)}
    emission = {
        "schema": EMISSION_SCHEMA, "table_path": str(out), "table_sha256": file_sha256(out),
        "table_id": table["table_id"], "cost_path": str(_resolve(costs, manifest_dir)), "cost_sha256": cost_sha256,
        "runtime_provenance": relation_ref, "context": context,
        "fixed_resources": {"declared": fixed, "evidence": fixed_evidence, "report": report_verdict},
        "rows": [{"unit": item["row"]["unit"], "format": item["row"]["format"], "run_id": item["binding"]["run_id"],
                  "prefill_ms": item["row"]["resources"]["prefill_ms"], "decode_ms": item["row"]["resources"]["decode_ms"],
                  "samples": {phase: len(item["row"][phase]["samples_ms"]) for phase in PHASES},
                  "peak_scratch_bytes": item["row"]["resources"]["peak_scratch_bytes"],
                  "activation_bytes": item["row"]["resources"]["activation_bytes"],
                  "serialized_bytes": item["row"]["resources"]["serialized_bytes"],
                  "resident_bytes": item["row"]["resources"]["resident_bytes"],
                  "operator_route": item["row"]["binding"]["operator_route"],
                  "joint_operator_identity_sha256": item["cost_row_identity_sha256"],
                  "panel": item["binding"]["panel"], "receipt": item["binding"]["receipt"],
                  "memory_trace": item["binding"]["memory_trace"],
                  "unknown": list(item["observation"]["unknown"])}
                 for item in bound],
        "admission": admission,
    }
    out.with_name(out.stem + ".emission.json").write_text(json.dumps(emission, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return emission


def load_receipt_manifest(path: Path) -> list:
    manifest = _json(path, "receipt manifest")
    bindings = manifest.get("bindings") if isinstance(manifest, Mapping) else manifest
    if not isinstance(bindings, list):
        raise RuntimePriceError("receipt manifest must be a list of native receipt bindings")
    return bindings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="table path to write")
    parser.add_argument("--table-id", required=True)
    parser.add_argument("--costs", type=Path, required=True,
                        help="the allocator's --costs pickle; every panel must have been frozen against these bytes")
    parser.add_argument("--relation", type=Path, required=True,
                        help="prismaquant.runtime_provenance_relation.v1 document; its identity is the context's runtime_sha256")
    parser.add_argument("--full-engine-report", type=Path, required=True,
                        help="tessera.full_engine_resource_report.v1 the fixed charge is recomputed from")
    parser.add_argument("--receipts", type=Path, required=True,
                        help="JSON list of {unit, format, run_id, panel, receipt, memory_trace}; paths resolve against this file")
    parser.add_argument("--fixed-assignment", default="{}", help="JSON unit -> format mapping of the fixed members")
    parser.add_argument("--valid-hours", type=float, default=24 * 30)
    parser.add_argument("--now", default=None, help="ISO-8601 UTC instant (tests); default is the current time")
    args = parser.parse_args(argv)
    now = None
    if args.now is not None:
        now = datetime.fromisoformat(args.now.replace("Z", "+00:00"))
    try:
        fixed_assignment = json.loads(args.fixed_assignment)
    except ValueError as exc:
        raise RuntimePriceError(f"--fixed-assignment is not JSON: {exc}") from exc
    emission = emit_native_receipt_table(
        out=args.out, table_id=args.table_id, costs=args.costs, relation=args.relation,
        full_engine_report=args.full_engine_report, receipts=load_receipt_manifest(args.receipts),
        manifest_dir=args.receipts.resolve().parent, fixed_assignment=fixed_assignment,
        valid_hours=args.valid_hours, now=now)
    admission = emission["admission"]
    print(json.dumps({"table": emission["table_path"], "table_sha256": emission["table_sha256"],
                      "rows": len(emission["rows"]), "admission": admission}, sort_keys=True), flush=True)
    return 0 if admission["status"] == "admitted" else 2


if __name__ == "__main__":
    sys.exit(main())
