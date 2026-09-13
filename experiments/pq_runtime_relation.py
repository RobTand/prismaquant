#!/usr/bin/env python3
"""Assemble a ``prismaquant.runtime_provenance_relation.v1`` document.

The relation is the evidence :func:`prismaquant.runtime_provenance.
load_runtime_relation` reads to decide whether a set of native operator
receipts and one full-engine run are the SAME image, the SAME GPU, the SAME
plugin installation and the SAME production libraries. Nothing in the tree
wrote one before: the loader only read them, so the first real measured table
had no input to be refused on.

Every field here is derived from bytes on disk, never declared:

``package_source``
    one source archive plus the roster the installer excludes; the loader
    recomputes the installer's identity, the producer's source-tree seal and
    the installed seal from those bytes.
``production_dependencies``
    one entry per (native run, production library it actually loaded), bound
    to a full-engine library with the same bytes. A native library whose bytes
    the full-engine run never loaded is REPORTED by name, not dropped -- that
    is the #323 finding (generated Triton output differing between runs) and it
    has to stay visible.
``full_engine_extra_libraries``
    exactly the full-engine production libraries no native run needed.

The document is written and then read back through ``load_runtime_relation``
with the context the panels themselves derive, so this tool cannot emit a
relation it has not seen the loader's verdict on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.measured_runtime_prices import RuntimePriceError, parse_runtime_context  # noqa: E402
from prismaquant.runtime_provenance import SCHEMA, identity_sha256, load_runtime_relation  # noqa: E402

INSTRUMENTATION_ROLES = ("resource_collector", "blas_workspace_observer")


def sha256(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference(path) -> dict:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def read(path):
    return json.loads(Path(path).read_text())


def runtime_record(spec) -> dict:
    """The runtime record a run's own artifact publishes, read the loader's way."""
    raw = read(spec["runtime"]["path"])
    field = spec.get("runtime_field")
    return raw if field is None else raw[field]


def instrumentation_paths(record, scope) -> set:
    if scope == "native_operator":
        collector = record.get("resource_collector")
        return {collector["loaded_path"]} if collector and "loaded_path" in collector else set()
    return {value["loaded_path"] for key, value in record["instrumentation"].items()
            if key in INSTRUMENTATION_ROLES and value and "loaded_path" in value}


def declared_instrumentation(record, scope, *, libraries, build_receipt, sources):
    """The instrumentation block, derived from what the run actually observed.

    A native record names its collector by bytes only (the harness stamps
    ``library_sha256`` and the analysis source, not a path), so the loaded path
    is recovered from the library roster by those bytes rather than assumed.
    """
    observed = ({"resource_collector": record["resource_collector"]} if scope == "native_operator"
                else {key: value for key, value in record["instrumentation"].items()
                      if key in INSTRUMENTATION_ROLES and value is not None})
    entries = []
    for role, value in sorted(observed.items()):
        digest = value["library_sha256"]
        paths = sorted(path for path, sha in libraries.items() if sha == digest)
        if "loaded_path" in value:
            paths = [value["loaded_path"]]
        if len(paths) != 1:
            raise RuntimePriceError(
                f"{role} bytes {digest} match {len(paths)} loaded libraries; the relation cannot "
                "name which one was the instrument")
        entries.append({"role": role, "loaded_path": paths[0],
                        "artifact": reference(sources[role]["artifact"]),
                        "source": reference(sources[role]["source"]),
                        "build_receipt": reference(build_receipt)})
    python_sources = {}
    for key, digest in record["source"].items():
        if key in ("tessera_package_sha256", "runtime_contract_sha256"):
            continue
        python_sources[key] = digest
    if scope == "native_operator":
        python_sources["resource_analysis_source_sha256"] = record["resource_collector"]["analysis_source_sha256"]
    return entries, python_sources


def build(args) -> int:
    plan = read(args.plan)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    archive = Path(plan["package_source"]["archive"]).resolve(strict=True)
    package_source = {"archive": reference(archive), "prefix": plan["package_source"]["prefix"],
                      "excluded_files": sorted(plan["package_source"]["excluded_files"])}

    runs, records, production = {}, {}, {}
    for run_id, spec in sorted(plan["runs"].items()):
        scope = spec["scope"]
        record = runtime_record(spec)
        base = record["base"] if scope == "full_engine" else record
        libraries = base["native_libraries"]
        excluded = instrumentation_paths(record, scope)
        entries, python_sources = declared_instrumentation(
            record, scope, libraries=libraries, build_receipt=spec["instrumentation"]["build_receipt"],
            sources=spec["instrumentation"]["sources"])
        excluded |= {item["loaded_path"] for item in entries}
        runs[run_id] = {
            "scope": scope, "runtime": reference(spec["runtime"]["path"]),
            "runtime_field": spec.get("runtime_field"),
            "installation": reference(spec["installation"]),
            "post_core": reference(spec["post_core"]), "post_package": reference(spec["post_package"]),
            "instrumentation": {"libraries": entries,
                                "python_sources": {key: reference(path) for key, path
                                                   in spec["instrumentation"]["python_sources"].items()},
                                "artifacts": {key: reference(path) for key, path
                                              in spec["instrumentation"].get("artifacts", {}).items()}},
        }
        for key, digest in python_sources.items():
            declared = runs[run_id]["instrumentation"]["python_sources"].get(key)
            if declared is None or declared["sha256"] != digest:
                raise RuntimePriceError(
                    f"{run_id}: the run recorded {key}={digest}, and the plan names "
                    f"{'no file' if declared is None else declared['sha256']} for it")
        records[run_id] = record
        production[run_id] = {path: sha for path, sha in libraries.items() if path not in excluded}

    full_id = plan["full_engine_run_id"]
    full = production[full_id]
    by_bytes: dict[str, list[str]] = {}
    for path, digest in full.items():
        by_bytes.setdefault(digest, []).append(path)

    dependencies, unbound = [], []
    for run_id in sorted(production):
        if run_id == full_id:
            continue
        for path, digest in sorted(production[run_id].items()):
            candidates = by_bytes.get(digest, [])
            if not candidates:
                unbound.append({"native_run_id": run_id, "native_path": path, "sha256": digest})
                continue
            # Prefer the identical path; otherwise the one full-engine library
            # with these exact bytes. Two paths holding one blob are the same
            # dependency, and picking the first sorted one is not a choice the
            # verdict depends on.
            full_path = path if path in full and full[path] == digest else sorted(candidates)[0]
            dependencies.append({"native_run_id": run_id, "native_path": path,
                                 "full_engine_path": full_path, "sha256": digest})
    used = {item["full_engine_path"] for item in dependencies}
    extras = {path: {"sha256": digest, "scope": "full_engine"}
              for path, digest in sorted(full.items()) if path not in used}

    relation = {"schema": SCHEMA, "configuration": reference(plan["configuration"]),
                "image_manifest": reference(plan["image_manifest"]), "package_source": package_source,
                "runs": runs, "full_engine_run_id": full_id,
                "production_dependencies": dependencies, "full_engine_extra_libraries": extras}
    out.write_text(json.dumps(relation, indent=1, sort_keys=True, allow_nan=False) + "\n")

    report = {"schema": "prismaquant.runtime_relation_emission.v1", "relation_path": str(out),
              "relation_sha256": sha256(out), "relation_identity_sha256": identity_sha256(relation),
              "runs": {run_id: {"scope": spec["scope"],
                                "production_libraries": len(production[run_id]),
                                "instrumentation_libraries": len(spec["instrumentation"]["libraries"])}
                       for run_id, spec in sorted(runs.items())},
              "production_dependencies": len(dependencies),
              "full_engine_extra_libraries": len(extras),
              "unbound_native_libraries": unbound,
              "verdict": None}
    if unbound:
        # Named, counted and kept: a native library the full-engine run never
        # loaded is a finding about the two runs, not a row to quietly drop.
        report["verdict"] = {"status": "incomplete",
                             "error": f"{len(unbound)} native production libraries have no full-engine bytes"}
    else:
        try:
            panels = [read(item["panel"] if isinstance(item["panel"], str) else item["panel"]["path"])
                      for item in read(args.panels)] if args.panels else []
            if panels:
                from prismaquant.native_operator_panel import operator_route_identity
                from prismaquant.native_receipt_table import derive_context

                context = derive_context(panels, relation=relation)
                context["operator_routes"] = {panel["unit"]: {panel["format"]: operator_route_identity(
                    panel["phases"]["prefill"]["expected_route"])} for panel in panels}
                load_runtime_relation({"path": out.name, "sha256": sha256(out)},
                                      context=parse_runtime_context(context), root=out.parent)
                report["verdict"] = {"status": "loaded", "error": None}
            else:
                report["verdict"] = {"status": "unchecked",
                                     "error": "no panels supplied; the loader's verdict needs a context"}
        except RuntimePriceError as exc:
            report["verdict"] = {"status": "refused", "error": str(exc)}
    Path(str(out) + ".emission.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("relation_path", "relation_sha256", "production_dependencies",
                       "full_engine_extra_libraries", "verdict")}, indent=1, sort_keys=True), flush=True)
    if unbound:
        print(json.dumps({"unbound_native_libraries": unbound}, indent=1), flush=True)
    return 0 if report["verdict"]["status"] in ("loaded", "unchecked") else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True, help="JSON naming every run's evidence")
    parser.add_argument("--out", required=True, help="relation document to write")
    parser.add_argument("--panels", help="receipt manifest whose panels derive the checking context")
    args = parser.parse_args(argv)
    try:
        return build(args)
    except RuntimePriceError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
