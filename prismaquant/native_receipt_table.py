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
There are **two** admitting gates and they answer about different objects, so
the emission report carries two verdicts and never one collapsed answer:
``admit_native_rows`` attests the per-row prices ``build_runtime_resources``
hands the DP, and ``admit_fixed_resources`` attests the whole-engine charge
the allocator adds once outside it. The loader raises on the first and
*returns* the second, so a caller that reads only the exception sees an
admitted table whose fixed charge is refused -- which is an admission the
emitter does not have. ``admission_report`` reads both flags and the refusal
text, and the CLI's exit code says which of the three answers it got:

=====  ================================================================
exit   what the table is
=====  ================================================================
``0``  both gates admitted. Unreachable while debt D37 stands, because
       ``admit_fixed_resources`` cannot pass for any v2 table.
``3``  the rows are admitted and priced; the fixed charge is refused,
       and ``admission.refusal`` is the gate's reason for it verbatim.
``2``  the loader refused the table outright, so no row is priced and
       the fixed-resource gate was never reached.
=====  ================================================================

Nonzero in both refusing cases, because an emitter must never certify an
admission it does not have; distinguishable, because a caller that can use
49 priced rows should not have to read "refused" the same way it reads
"this table prices nothing".

Fixed resources are declared by asking the admitting gate to recompute them:
``runtime_provenance.recompute_fixed_resources`` owns the only reader of the
full-engine resource report, so the numbers written here and the numbers
``admit_fixed_resources`` checks them against come from one implementation and
no second, drifting reader exists. A term the report cannot express is
declared ``0`` and recorded as unevidenced in the emission report; the gate
then refuses it by name, which is the truthful state of the axis (PQ #237,
#420) rather than a number.
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
from .runtime_provenance import SCHEMA as RELATION_SCHEMA, recompute_fixed_resources

EMISSION_SCHEMA = "prismaquant.native_receipt_table_emission.v1"
#: The CLI's three answers; see the module docstring for what each one means.
EXIT_ADMITTED = 0
EXIT_REFUSED = 2
EXIT_NATIVE_ROWS_ONLY = 3
EXIT_CODES = {"admitted": EXIT_ADMITTED, "refused": EXIT_REFUSED,
              "native_rows_only": EXIT_NATIVE_ROWS_ONLY}
DENSE_PANEL_SCHEMA = "tessera.native_dense_panel.v1"
BINDING_FIELDS = ("unit", "format", "run_id", "panel", "receipt", "memory_trace")
PHASES = ("prefill", "decode")

#: The one field of an attested native runtime record that a bound panel may
#: differ in, and the only one.
#:
#: Every other field is a *comparability* coordinate: the GPU and its driver,
#: the torch/CUDA build, the arithmetic flags, the image, the installed Tessera
#: package and runtime contract, the execution mode, the measurement collector.
#: Two rows measured under different values of any of them are not on the same
#: clock, and a table that priced them against each other would be comparing
#: two boxes. That equality is untouched.
#:
#: ``native_libraries`` is not a coordinate of the box. It is the set of shared
#: objects the row's own route actually dispatched to, and a route may load its
#: own: the fp4 route JIT-builds and ``dlopen``s a ``tessera_nvfp4_<hash>.so``
#: that no fp8 or bf16 route loads. Requiring the whole record to be equal
#: therefore put a route's kernel inside the box's identity and made a
#: mixed-route table -- the only kind a format menu spans -- impossible to
#: emit, over nine same-session cells whose records were otherwise identical.
#:
#: Moving it loosens nothing, because the binding that matters is per row and
#: already exists, twice over:
#:
#: * ``runtime_provenance.admit_native_rows`` requires each row's *whole*
#:   runtime record, ``native_libraries`` included, to equal the raw record of
#:   the run its own binding names (``runtime_provenance.py`` ``:771``), so a
#:   row cannot claim a runtime it was not measured on.
#: * ``runtime_provenance._load_runtime_relation`` requires every production
#:   library any native run loaded to exist in the full-engine run at the same
#:   digest (``:432``), so an extension one route loads is either shown to be
#:   the bytes the engine serves or the table is refused by name.
#:
#: It is also the model the relation already applies across its own runs
#: (``:414``): one box, one image, one package; library *sets* may differ
#: between runs, but a path present in both must carry the same bytes. This
#: emitter was the stricter of the two, and it was the one that refused.
PER_ROUTE_RUNTIME_FIELD = "native_libraries"

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
    from .native_operator_panel import consume_native_receipt, operator_route_identity

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
    route = operator_route_identity(panel["phases"]["prefill"]["expected_route"])
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


def _require_one_runtime(panels: list[Mapping]) -> None:
    """Refuse panels that are not on one clock; allow one route its own kernels.

    Three checks, and between them they are the whole-record equality this
    replaced, partitioned rather than relaxed. Each comparison is over the same
    canonical JSON digest the whole-record check used, so ``false`` and ``0``
    stay different values here as they were before.

    1. The runtime records must declare the same fields. A field one panel
       carries and another does not is refused, not exempted, so a future
       producer field cannot join :data:`PER_ROUTE_RUNTIME_FIELD` by accident.
    2. Every field except :data:`PER_ROUTE_RUNTIME_FIELD` must be equal. This
       is the comparability invariant and it is exactly as strong as it was.
    3. A library path that more than one panel loaded must carry the same bytes
       in each. One run may load *more* than another; it may never load a
       different ``libtorch.so`` and call the timings comparable.
    """
    first = panels[0]["runtime"]
    if not isinstance(first.get(PER_ROUTE_RUNTIME_FIELD), Mapping):
        raise RuntimePriceError("native runtime record declares no loaded libraries")
    for panel in panels[1:]:
        runtime = panel["runtime"]
        if set(runtime) != set(first):
            raise RuntimePriceError("native receipts were produced on more than one runtime: "
                                    "their runtime records declare different fields")
        for field in sorted(set(first) - {PER_ROUTE_RUNTIME_FIELD}):
            if identity_sha256(runtime[field]) != identity_sha256(first[field]):
                raise RuntimePriceError("native receipts were produced on more than one runtime: "
                                        f"{field} differs")
        libraries = runtime[PER_ROUTE_RUNTIME_FIELD]
        if not isinstance(libraries, Mapping):
            raise RuntimePriceError("native runtime record declares no loaded libraries")
        for path in sorted(set(libraries) & set(first[PER_ROUTE_RUNTIME_FIELD])):
            if libraries[path] != first[PER_ROUTE_RUNTIME_FIELD][path]:
                raise RuntimePriceError("native receipts were produced on more than one runtime: "
                                        f"one loaded library carries different bytes: {path}")


def unshared_native_libraries(panels: list[Mapping]) -> list[dict]:
    """Per panel, the loaded libraries that not every bound panel loaded.

    This is the emission report's answer to "which kernel did this row's own
    route dispatch to". It is read off each panel's own attested
    ``runtime.native_libraries`` and nothing else: no table here decides which
    route owns which ``.so``, because a producer-side guess about that is
    exactly the asserted runtime claim principle 14 refuses. On a table whose
    rows all loaded the same libraries every entry is empty.
    """
    shared = set.intersection(*(set(panel["runtime"][PER_ROUTE_RUNTIME_FIELD]) for panel in panels))
    return [{path: panel["runtime"][PER_ROUTE_RUNTIME_FIELD][path]
             for path in sorted(set(panel["runtime"][PER_ROUTE_RUNTIME_FIELD]) - shared)}
            for panel in panels]


def derive_context(panels: list[Mapping], *, relation: Mapping) -> dict:
    """The one workload/runtime context every bound panel was frozen under.

    "One runtime" means one box, one image, one package and one execution mode
    -- not one set of loaded kernels. See :data:`PER_ROUTE_RUNTIME_FIELD` for
    why those are two questions and where the second one is answered.
    """
    if not panels:
        raise RuntimePriceError("no native receipts were bound; a table needs at least one row")
    first = panels[0]
    _require_one_runtime(panels)
    for panel in panels[1:]:
        for what, key in (("source model", lambda p: p["source_sha256"]),
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
    """Declare the fixed charge the admitting gate recomputes from the report.

    The recomputation does not happen here. This function contributes the
    artifact reference -- the path the table spells and the sha256 of the bytes
    behind it -- and hands it to ``runtime_provenance``, which owns the reader
    of that partition and is the module whose numbers the admission checks.
    """
    _, reference = _reference(report_path, table_dir, "full-engine resource report")
    declared, evidence, verdict = recompute_fixed_resources(
        {"path": str(report_path.resolve()), "sha256": reference["sha256"]},
        root=report_path.resolve().parent)
    return declared, evidence, {"reference": reference, **verdict}


def admission_report(path: Path, *, expected_context, expected_cost_sha256: str,
                     now: datetime) -> dict:
    """Both admitting gates' verdicts, and the one status that follows from them.

    ``native_rows`` is ``refused`` whenever the loader raised, because then no
    row of the table is admitted for pricing -- whether it was the parse, the
    relation or ``admit_native_rows`` that said so -- and the fixed-resource
    gate is ``unreached`` rather than pretending to an answer it never gave.
    ``status`` is ``admitted`` only when both gates admitted, and the top-level
    ``refusal`` is ``None`` only then; otherwise it is the refusal that stands
    between this table and a full admission, verbatim from the gate that wrote
    it.
    """
    try:
        table = load_measured_runtime_table(path, expected_context=expected_context,
                                            expected_cost_sha256=expected_cost_sha256, now=now)
    except RuntimePriceError as exc:
        return {"status": "refused", "refusal": str(exc),
                "native_rows": {"status": "refused", "refusal": str(exc)},
                "fixed_resources": {"status": "unreached", "refusal": None}}
    native = {"status": "admitted" if table.native_rows_admitted else "refused",
              "refusal": None if table.native_rows_admitted else "the loader performed no producer admission"}
    fixed = {"status": "admitted" if table.fixed_resources_admitted else "refused",
             "refusal": table.fixed_resources_refusal}
    if native["status"] == "refused":
        return {"status": "refused", "refusal": native["refusal"],
                "native_rows": native, "fixed_resources": fixed}
    if fixed["status"] == "admitted":
        return {"status": "admitted", "refusal": None, "native_rows": native, "fixed_resources": fixed}
    return {"status": "native_rows_only", "refusal": fixed["refusal"],
            "native_rows": native, "fixed_resources": fixed}


def emit_native_receipt_table(*, out: Path, table_id: str, costs: Path, relation: Path,
                              full_engine_report: Path, receipts: list[Mapping], manifest_dir: Path,
                              fixed_assignment: Mapping[str, str], valid_hours: float,
                              now: datetime | None = None) -> dict:
    """Write the table, its fixed-resource receipt and the emission report.

    Returns the emission report. Both loader verdicts are inside it under
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
    admission = admission_report(out, expected_context=parse_runtime_context(context),
                                 expected_cost_sha256=cost_sha256, now=current)
    emission = {
        "schema": EMISSION_SCHEMA, "table_path": str(out), "table_sha256": file_sha256(out),
        "table_id": table["table_id"], "cost_path": str(_resolve(costs, manifest_dir)), "cost_sha256": cost_sha256,
        "runtime_provenance": relation_ref, "context": context,
        "fixed_resources": {"declared": fixed, "evidence": fixed_evidence, "report": report_verdict},
        "rows": [{"unit": item["row"]["unit"], "format": item["row"]["format"], "run_id": item["binding"]["run_id"],
                  "runtime_sha256": identity_sha256(item["panel"]["runtime"]), "unshared_native_libraries": unshared,
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
                 for item, unshared in zip(bound, unshared_native_libraries([item["panel"] for item in bound]))],
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
                        help="the full-engine resource report the fixed charge is recomputed from")
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
    return EXIT_CODES[admission["status"]]


if __name__ == "__main__":
    sys.exit(main())
