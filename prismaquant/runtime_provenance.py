"""Admission for explicitly related native and full-engine measurements.

The relation has its own identity. Original run manifests are retained and
checked independently; no run is assigned another run's digest. This module
reads producer evidence and never imports the Tessera serving runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Mapping

from .measured_runtime_prices import (
    RuntimePriceError, _integer, _object,
    _sha, _string, identity_sha256,
)

SCHEMA = "prismaquant.runtime_provenance_relation.v1"


def _equal(actual, expected, where):
    # Python otherwise considers True == 1, including inside nested dicts.
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(actual) != set(expected):
            raise RuntimePriceError(f"{where}: evidence mismatch")
        for key in expected:
            _equal(actual[key], expected[key], where + " " + str(key))
    elif isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise RuntimePriceError(f"{where}: evidence mismatch")
        for left, right in zip(actual, expected):
            _equal(left, right, where)
    elif ((type(actual) is not type(expected) and isinstance(expected, (bool, int)))
            or actual != expected):
        raise RuntimePriceError(f"{where}: evidence mismatch")


def _mapping(value, where):
    if not isinstance(value, Mapping):
        raise RuntimePriceError(f"{where}: expected an object")
    return value


@dataclass
class ArtifactReader:
    root: Path

    def bytes(self, reference, where):
        _object(reference, ("path", "sha256"), where)
        path = Path(_string(reference["path"], where + " path"))
        if not path.is_absolute():
            path = self.root / path
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise RuntimePriceError(f"{where}: cannot read artifact {path}: {exc}") from exc
        _equal(hashlib.sha256(raw).hexdigest(), _sha(reference["sha256"], where),
               where + " artifact SHA-256")
        return path, raw

    def json(self, reference, where):
        path, raw = self.bytes(reference, where)
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise RuntimePriceError(f"{where}: duplicate JSON key {key!r}")
                result[key] = value
            return result
        def nonfinite(value):
            raise ValueError("nonfinite JSON number " + value)
        try:
            value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
        except (ValueError, UnicodeError) as exc:
            raise RuntimePriceError(f"{where}: invalid JSON artifact {path}: {exc}") from exc
        return path, _mapping(value, where)


def _source_digest(files):
    """Tessera's versioned source-byte seal, without importing its runtime."""
    digest = hashlib.sha256()
    for name in sorted(files, key=Path):
        raw = files[name]
        if Path(name).suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}:
            digest.update(name.encode() + b"\0" + raw + b"\0")
    return digest.hexdigest()


def _package_source(declaration, reader):
    _object(declaration, ("archive", "prefix", "excluded_files"), "package source")
    _, archive = reader.bytes(declaration["archive"], "original plugin source archive")
    prefix = _string(declaration["prefix"], "package archive prefix").rstrip("/") + "/"
    if Path(prefix).is_absolute() or ".." in Path(prefix).parts:
        raise RuntimePriceError("package archive prefix must be relative and normalized")
    files = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as source:
            for member in source:
                if not member.name.startswith(prefix) or member.isdir():
                    continue
                name = member.name[len(prefix):]
                if (not member.isfile() or name in files or Path(name).is_absolute()
                        or ".." in Path(name).parts or str(Path(name)) != name):
                    raise RuntimePriceError("package archive has duplicate, linked or unsafe source entries")
                files[name] = source.extractfile(member).read()
    except tarfile.TarError as exc:
        raise RuntimePriceError(f"invalid plugin source archive: {exc}") from exc
    excluded = declaration["excluded_files"]
    if (not files or not isinstance(excluded, list) or any(not isinstance(name, str) for name in excluded)
            or len(set(excluded)) != len(excluded) or not set(excluded) <= set(files)):
        raise RuntimePriceError("package source needs an exact archive and explicit excluded file roster")
    installed = {name: raw for name, raw in files.items() if name not in excluded}
    return {"archive_sha256": declaration["archive"]["sha256"],
            "source_tree_sha256": _source_digest(files), "installed_source_sha256": _source_digest(installed),
            "installed_files": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                                for name, raw in installed.items()}}


def _library_map(value, where):
    result = {}
    for path, digest in _mapping(value, where).items():
        _string(path, where + " library path")
        if not Path(path).is_absolute():
            raise RuntimePriceError(f"{where}: loaded library paths must be absolute")
        result[path] = _sha(digest, where + " library digest")
    if not result:
        raise RuntimePriceError(f"{where}: empty library observation")
    return result


def _instrumentation(run, raw, base, libraries, reader):
    declaration = _object(run["instrumentation"], ("libraries", "python_sources", "artifacts"), "instrumentation")
    if not isinstance(declaration["libraries"], list) or not declaration["libraries"]:
        raise RuntimePriceError("instrumentation libraries require explicit artifacts")
    observed = ({"resource_collector": raw["resource_collector"]}
                if run["scope"] == "native_operator" else {key: value for key, value in raw["instrumentation"].items()
                      if key in ("resource_collector", "blas_workspace_observer") and value is not None})
    if run["scope"] == "full_engine" and set(raw["instrumentation"]) - {
            "resource_collector", "blas_workspace_observer", "native_owner_rule"}:
        raise RuntimePriceError("unknown full-engine instrumentation must be explicitly supported")
    excluded, roles = set(), set()
    for item in declaration["libraries"]:
        _object(item, ("role", "loaded_path", "artifact", "source", "build_receipt"), "instrumentation library")
        role = item["role"]
        if role not in ("resource_collector", "blas_workspace_observer") or role in roles:
            raise RuntimePriceError("unknown or duplicate instrumentation role")
        roles.add(role)
        path = _string(item["loaded_path"], "instrumentation loaded path")
        if path in excluded or path not in libraries:
            raise RuntimePriceError("instrumentation library is duplicate or not actually loaded")
        # Measurement tools are separately built artifacts. Installed runtime
        # libraries cannot be removed from dependency matching by a role label.
        if path.startswith(("/usr/", "/lib/", "/lib64/", "/opt/venv/")):
            raise RuntimePriceError("installed production library cannot be declared instrumentation")
        reader.bytes(item["artifact"], "instrumentation binary")
        source_path, _ = reader.bytes(item["source"], "instrumentation source")
        _, build = reader.json(item["build_receipt"], "instrumentation build receipt")
        if "builds" in build:
            matches = [row for row in build["builds"] if row["name"] == Path(path).name]
            if len(matches) != 1:
                raise RuntimePriceError("instrumentation build must identify exactly one output")
            _equal(matches[0]["returncode"], 0, "instrumentation compiler exit")
            _equal(build["files"][Path(path).name]["sha256"], libraries[path], "built binary bytes")
            _equal(build["source_files"][source_path.name], item["source"]["sha256"], "built source bytes")
        else:
            _equal(build["source_sha256"], item["source"]["sha256"], "built source bytes")
            _equal(build["output_sha256"], libraries[path], "built binary bytes")
        _equal(item["artifact"]["sha256"], libraries[path], "instrumentation mapped bytes")
        _equal(observed[role]["library_sha256"], libraries[path], "observed instrumentation role")
        if "loaded_path" in observed[role]:
            _equal(observed[role]["loaded_path"], path, "observed instrumentation path")
        excluded.add(path)
    _equal(roles, set(observed), "instrumentation role coverage")
    source = base["source"]
    expected_sources = {key: value for key, value in source.items()
                        if key not in ("tessera_package_sha256", "runtime_contract_sha256")}
    if run["scope"] == "native_operator":
        expected_sources["resource_analysis_source_sha256"] = raw["resource_collector"]["analysis_source_sha256"]
    else:
        expected_sources.update(raw["source"])
    artifacts = _mapping(declaration["artifacts"], "instrumentation artifacts")
    expected_artifacts = ({"native_owner_rule": raw["instrumentation"]["native_owner_rule"]}
                          if run["scope"] == "full_engine" and raw["instrumentation"].get("native_owner_rule") else {})
    _equal(set(artifacts), set(expected_artifacts), "instrumentation artifact coverage")
    for key, reference in artifacts.items():
        reader.bytes(reference, "instrumentation artifact " + key)
        _equal(reference["sha256"], expected_artifacts[key]["sha256"], "observed instrumentation artifact")
    sources = _mapping(declaration["python_sources"], "instrumentation Python sources")
    _equal(set(sources), set(expected_sources), "instrumentation Python source coverage")
    for key, reference in sources.items():
        reader.bytes(reference, "instrumentation Python source " + key)
        _equal(reference["sha256"], expected_sources[key], "observed Python source " + key)
    return excluded


def _observe_run(run, *, reader, configuration, configuration_sha256, image_manifest, package_source, context):
    _object(run, ("scope", "runtime", "runtime_field", "installation", "post_core", "post_package", "instrumentation"), "runtime run")
    if run["scope"] not in ("native_operator", "full_engine"):
        raise RuntimePriceError("unsupported runtime observation scope")
    _, original = reader.json(run["runtime"], "original runtime artifact")
    if run["runtime_field"] not in (None, "runtime"):
        raise RuntimePriceError("unsupported runtime artifact field")
    raw = original if run["runtime_field"] is None else _mapping(original["runtime"], "embedded runtime")
    if run["runtime_field"] is not None:
        _equal(original["runtime_sha256"], identity_sha256(raw), "original embedded runtime digest")
    if run["scope"] == "full_engine":
        _equal(raw["schema"], "tessera.full_engine_runtime.v1", "full-engine runtime schema")
        base, execution = raw["base"], raw["actual_execution"]
        _equal(raw["configuration_sha256"], configuration_sha256, "actual full-engine configuration")
        for key in ("engine_args", "environment"):
            _equal(raw["execution"][key], configuration[key], "full-engine selected " + key)
    else:
        if raw["schema"] not in ("tessera.native_dense_runtime.v1", "tessera.native_moe_runtime.v1"):
            raise RuntimePriceError("unsupported native runtime schema")
        base, execution = raw, raw["execution"]
    image = context.serving_context.runtime_image
    _equal(base["image"], image, "runtime image")
    _equal(configuration["runtime_image"], image, "configuration image")
    _equal(base["gpu"]["uuid"], context.gpu_identity, "actual GPU UUID")
    capability = base["gpu"]["capability"]
    if not isinstance(capability, list) or len(capability) != 2:
        raise RuntimePriceError("actual GPU capability must contain major and minor")
    major, minor = (_integer(value, "GPU capability") for value in capability)
    _equal("sm_" + str(major) + str(minor), context.serving_context.platform, "actual GPU platform")
    _equal(context.graph_mode, execution["execution_mode"], "actual graph mode")
    _equal(execution["mode"], context.serving_context.residency, "actual residency")
    _equal(execution["execution_mode"], context.serving_context.execution_mode, "actual execution mode")
    _equal(execution["tensor_parallel"], context.tensor_parallel, "actual tensor parallelism")
    declared = base["image_declaration"]["record"]
    if declared["refused"] is not False or declared["present"] is not True or declared["gated"] is not True:
        raise RuntimePriceError("runtime image declaration refused or incomplete")
    for key in ("pinned", "resolved_reference", "requested"):
        _equal(declared[key], image, "declared image " + key)
    if image not in declared["repo_digests"]:
        raise RuntimePriceError("runtime image lacks actual RepoDigests evidence")
    _equal(declared["selection"]["configuration_sha256"], configuration_sha256, "launcher configuration")
    _, installation = reader.json(run["installation"], "runtime installation")
    _equal(installation["registry_base"], image, "installed image")
    _equal(installation["launcher_declared_image_id"], declared["local_id"], "actual image ID")
    if declared["local_id"] not in (image_manifest["manifest_digest"], image_manifest["config_digest"]):
        raise RuntimePriceError("actual image ID is neither the pinned manifest nor its config digest")
    core_sha = _sha(installation["core_manifest_sha256"], "stock core manifest")
    core_count = _integer(installation["core_files_unchanged"], "stock core count", 1)
    _, audit = reader.json(run["post_core"], "post-run stock core audit")
    if run["scope"] == "native_operator":
        _equal(audit["native_returncode"], 0, "native child exit")
        _equal(audit["manifest_sha256"], core_sha, "post-native core manifest")
        _equal(audit["stock_files_unchanged"], core_count, "post-native core files")
    else:
        for phase in ("before", "after"):
            _equal(audit["core_audit_" + phase]["manifest_sha256"], core_sha, "full-engine core manifest")
            _equal(audit["core_audit_" + phase]["unchanged_files"], core_count, "full-engine core files")
    _, package = reader.json(run["post_package"], "actual loaded package")
    _equal(package["schema"], "tessera.loaded_package_identity.v1", "loaded package schema")
    if package["package_files_unchanged_from_installer"] is not True:
        raise RuntimePriceError("loaded package files changed after installation")
    _equal(package["package_files"], installation["plugin_files"], "complete installed package file roster")
    _equal(installation["plugin_archive_sha256"], package_source["archive_sha256"], "original plugin archive bytes")
    _equal(package["package_files"], package_source["installed_files"], "installed package bytes from source archive")
    _equal(package["encoder_source_sha256"], package_source["installed_source_sha256"], "recomputed installed source seal")
    _equal(package["encoder_source_sha256"], base["source"]["tessera_package_sha256"], "installed package source digest")
    _equal(package["package_files"]["serving/runtime_contract.json"]["sha256"],
           base["source"]["runtime_contract_sha256"], "installed runtime contract")
    _equal(package["installer_evidence_sha256"], run["installation"]["sha256"], "loaded package installer evidence")
    _equal(package["module_identity_errors"], [], "loaded package module errors")
    package_path = Path(_string(package["package_path"], "loaded package path"))
    if not package_path.is_absolute() or ".." in package_path.parts:
        raise RuntimePriceError("loaded package path must be absolute and normalized")
    modules = _mapping(package["loaded_tessera_modules"], "loaded Tessera modules")
    if not {"tessera", "tessera.cached_unit"} <= set(modules):
        raise RuntimePriceError("required loaded Tessera modules are missing")
    for name, module in modules.items():
        if name != "tessera" and not name.startswith("tessera."):
            raise RuntimePriceError("foreign loaded Tessera module name")
        _object(module, ("file", "origin", "sha256"), "loaded module")
        filename = Path(_string(module["file"], "module file"))
        origin = _string(module["origin"], "module origin")
        if str(filename) != origin or ".." in filename.parts or not filename.is_relative_to(package_path):
            raise RuntimePriceError("loaded module origin is outside or differs from package")
        relative = str(filename.relative_to(package_path))
        if relative not in package["package_files"]:
            raise RuntimePriceError("loaded module absent from package roster")
        _equal(_sha(module["sha256"], "module SHA-256"), package["package_files"][relative]["sha256"], "loaded module bytes")
    if run["scope"] == "full_engine":
        _equal(raw["loaded_package"], package, "full-engine embedded loaded package")
    libraries = _library_map(base["native_libraries"], "runtime")
    excluded = _instrumentation(run, raw, base, libraries, reader)
    common = {"image": image, "image_identity": image_manifest, "gpu": base["gpu"],
        "versions": base["versions"], "arithmetic": base["arithmetic"],
        "package_sha256": base["source"]["tessera_package_sha256"],
        "contract_sha256": base["source"]["runtime_contract_sha256"],
        "core_manifest_sha256": core_sha, "core_files": core_count,
        "plugin_source_commit": installation["plugin_source_commit"],
        "plugin_archive_sha256": installation["plugin_archive_sha256"],
        "producer_source_tree_sha256": package_source["source_tree_sha256"],
        "plugin_files": installation["plugin_files"], "plugin_entrypoints": installation["plugin_entrypoints"]}
    return {"raw": raw, "base": base, "sha256": identity_sha256(raw), "common": common,
            "libraries": libraries, "instrumentation": excluded,
            "production": {path: sha for path, sha in libraries.items() if path not in excluded}}


def load_runtime_relation(reference, *, context, root):
    """Verify exact observations and an exhaustive, explicitly named relation."""
    try:
        return _load_runtime_relation(reference, context=context, root=root)
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimePriceError(f"runtime relation evidence is missing or malformed: {exc}") from exc


def _load_runtime_relation(reference, *, context, root):
    path, relation = ArtifactReader(Path(root)).json(reference, "runtime provenance relation")
    reader = ArtifactReader(path.parent)
    _object(relation, ("schema", "configuration", "image_manifest", "package_source", "runs", "full_engine_run_id",
                      "production_dependencies", "full_engine_extra_libraries"), "runtime relation")
    _equal(relation["schema"], SCHEMA, "runtime relation schema")
    _equal(identity_sha256(relation), context.runtime_sha256, "independent runtime derivation identity")
    _, configuration = reader.json(relation["configuration"], "selected serving configuration")
    configuration_sha256 = relation["configuration"]["sha256"]
    _, manifest = reader.json(relation["image_manifest"], "pinned image manifest")
    manifest_digest = "sha256:" + relation["image_manifest"]["sha256"]
    _equal(context.serving_context.runtime_image.rsplit("@", 1)[-1], manifest_digest, "pinned manifest bytes")
    _equal(manifest["schemaVersion"], 2, "image manifest schema")
    if manifest["mediaType"] not in ("application/vnd.docker.distribution.manifest.v2+json",
                                     "application/vnd.oci.image.manifest.v1+json"):
        raise RuntimePriceError("image provenance requires a concrete platform manifest")
    config_digest = _string(manifest["config"]["digest"], "image config digest")
    if not config_digest.startswith("sha256:"):
        raise RuntimePriceError("image config requires SHA-256 identity")
    _sha(config_digest.removeprefix("sha256:"), "image config digest")
    image_manifest = {"manifest_digest": manifest_digest, "config_digest": config_digest}
    package_source = _package_source(relation["package_source"], reader)
    runs = _mapping(relation["runs"], "runtime runs")
    full_id = _string(relation["full_engine_run_id"], "full-engine run ID")
    if len(runs) < 2 or full_id not in runs:
        raise RuntimePriceError("runtime relation needs full-engine and native observations")
    observed = {name: _observe_run(run, reader=reader, configuration=configuration,
                                 configuration_sha256=configuration_sha256, image_manifest=image_manifest,
                                 package_source=package_source, context=context)
                for name, run in runs.items()}
    _equal({name for name, run in runs.items() if run["scope"] == "full_engine"}, {full_id}, "full-engine observation coverage")
    full = observed[full_id]
    for name, run in observed.items():
        _equal(run["common"], full["common"], "common image/core/plugin/config/device coordinates")
        for path in set(run["libraries"]) & set(full["libraries"]):
            _equal(run["libraries"][path], full["libraries"][path], "same-path library bytes")
            _equal(path in run["instrumentation"], path in full["instrumentation"], "production/instrumentation role")
    relations = relation["production_dependencies"]
    if not isinstance(relations, list) or not relations:
        raise RuntimePriceError("production dependency relation must be explicit and nonempty")
    covered, used_full = set(), set()
    for item in relations:
        _object(item, ("native_run_id", "native_path", "full_engine_path", "sha256"), "production dependency")
        name, native_path, full_path = item["native_run_id"], item["native_path"], item["full_engine_path"]
        if name not in observed or name == full_id or (name, native_path) in covered:
            raise RuntimePriceError("unknown or duplicate native production dependency")
        digest = _sha(item["sha256"], "production dependency")
        _equal(observed[name]["production"].get(native_path), digest, "native production dependency")
        _equal(full["production"].get(full_path), digest, "missing or changed full-engine production dependency")
        covered.add((name, native_path)); used_full.add(full_path)
    _equal(covered, {(name, path) for name, run in observed.items() if name != full_id
                     for path in run["production"]}, "complete native production dependency coverage")
    extras = _mapping(relation["full_engine_extra_libraries"], "extra full-engine production libraries")
    _equal(set(extras), set(full["production"]) - used_full, "declared extra production library coverage")
    for path, item in extras.items():
        _object(item, ("sha256", "scope"), "extra production library")
        _equal(_sha(item["sha256"], "extra production library"), full["production"][path], "extra production bytes")
        _equal(item["scope"], "full_engine", "extra exercised production scope")
    return {"record": relation, "reference": reference, "reader": reader, "runs": observed,
            "full_engine_run_id": full_id, "configuration_sha256": configuration_sha256}


#: How the fixed-resource receipt names its evidence: one artifact reference,
#: in the existing ``{path, sha256}`` form :class:`ArtifactReader` rehashes, to
#: a ``tessera.full_engine_resource_report.v1`` document. This spelling is the
#: PrismaQuant side of the receipt and not a frozen producer field; the
#: producer emits neither this nor any other today. Anything else in that slot
#: -- an inline resource claim, a status flag, an opaque proof digest -- is
#: refused by name rather than read, because none of them is recomputable.
FIXED_RESOURCE_REPORT_REFERENCE = ("path", "sha256")

#: Which recomputed term carries the evidence for each declared fixed field.
#:
#: ``serialized_bytes`` has no entry: the serialized-byte partition is its own
#: rule in the design and this schema version emits no observation for it.
#: Neither ``prefill_ms`` nor ``decode_ms`` has one either -- the report has no
#: timing term at all -- and both refuse by name below, which is why wiring
#: this gate cannot by itself open the prefill axis.
#:
#: The candidate transients are deliberately absent as well. A native row's
#: scratch charge includes its returned output, while the full-engine
#: partition classifies bytes by lifetime inside the unit interval; the two are
#: differently bounded, so an equality between them would either refuse every
#: real report on a definitional gap or agree by coincidence. The design owes a
#: versioned boundary model first ("Set native/full-engine charge boundary").
FIXED_TERM_FIELDS = {"fixed_resident": "resident_bytes",
                     "fixed_activation": "activation_bytes",
                     "fixed_scratch": "peak_scratch_bytes",
                     "fixed_kv": "kv_bytes"}

#: What each observation the report names but leaves null costs this
#: admission. The absence is read from the report rather than assumed here, so
#: a capture that does supply one drops its entry and the terms it feeds become
#: reachable once their domain also closes.
OWED_EVIDENCE = {
    "worker_startup_records": "the fixed resident and activation charge",
    "kv_observations": "the fixed KV charge",
    "owner_views": "the fixed member roster and each candidate's retained weights",
    "timing_captures": "the fixed prefill and decode charge",
    "observer_qualification": "the observer impact any timing charge needs",
    "runtime_provenance_relation": "the report's own binding to this relation",
}


def admit_fixed_resources(table, relation):
    """Admit fixed resources only from a partition this consumer recomputes.

    Tessera observes and derives; PrismaQuant recomputes and admits only on
    agreement. Nothing here reads the report's ``derived`` block: the
    recomputation in :mod:`prismaquant.full_engine_resource_report` runs over
    ``observations`` and ``partition``, and a producer total that differs from
    it arrives as one of the refusals collected below. Reading ``derived`` and
    believing it is the ``qualified: true`` failure the design forbids.

    Refusals are collected and raised together so one call names all of them.
    There is no admission token and no return value: the v2 loader sets
    ``producer_admitted`` only because this raised nothing.
    """
    reader = ArtifactReader(Path(table.source_path).parent)
    receipt_path, receipt = reader.json({"path": table.fixed_resources_receipt_path,
                                         "sha256": table.fixed_resources_receipt_sha256},
                                        "fixed-resource receipt")
    claim = receipt.get("full_model_resources")
    if claim is None:
        raise RuntimePriceError("full-model fixed resource producer admission is incomplete")
    if not isinstance(claim, Mapping) or set(claim) != set(FIXED_RESOURCE_REPORT_REFERENCE):
        raise RuntimePriceError(
            "full-model fixed resources must reference one recomputable full-engine resource "
            "report as {path, sha256}; an inline claim, status flag or proof digest is not evidence")
    refusals = _fixed_resource_refusals(table, relation, claim, root=receipt_path.parent)
    if refusals:
        raise RuntimePriceError("no qualified recomputable full-engine resource partition: "
                                + "; ".join(refusals))


def _fixed_resource_refusals(table, relation, reference, *, root):
    """Every reason this table's declared fixed resources are not admissible.

    The import is function-local because the report consumer imports this
    module's :class:`ArtifactReader` and ``_equal``; the dependency runs one
    way at import time and the other way at call time.
    """
    from .full_engine_resource_report import (
        OWED_OBSERVATIONS, SUPPORTED_EXECUTION, consume_full_engine_resource_report,
        read_full_engine_resource_report,
    )
    for key in ("runs", "full_engine_run_id", "configuration_sha256"):
        if key not in relation:
            raise RuntimePriceError(
                "full-engine fixed resource admission requires the loaded runtime relation")
    context = table.context
    # The report was measured on other bytes unless it names the exact source
    # model, serving configuration, full-engine runtime manifest and device
    # this table was priced against. The relation supplies the last three: no
    # run is handed another run's digest, and the full-engine run keeps its own.
    full_run = relation["runs"][relation["full_engine_run_id"]]
    expected = {"model_sha256": context.source_sha256,
                "configuration_sha256": relation["configuration_sha256"],
                "runtime_manifest_sha256": full_run["sha256"],
                "device_uuid": context.gpu_identity}
    report = read_full_engine_resource_report(reference, root=root)
    verdict = consume_full_engine_resource_report(reference, root=root,
                                                  expected_run_identity=expected)
    refusals = list(verdict.refusals)

    # A synthetic capture admits nothing, on this gate's own authority rather
    # than on the report reader's: `fixture_provenance` travels from capture to
    # ledger to `identity`, so a fixture can never read as a measurement, and
    # a positive synthetic receipt proves the parser contract and nothing else.
    if verdict.fixture_provenance is not None:
        refusals.append(f"the report declares fixture provenance {verdict.fixture_provenance!r}, "
                        "and a synthetic capture admits no measured table")

    # The scalar device budget is TP1, one device, resident, eager, one
    # request. Anything else needs a versioned resource vector, never a rank
    # sum or a rank maximum written into these scalar fields.
    declared_execution = {"graph_mode": context.graph_mode,
                          "residency": context.serving_context.residency,
                          "topology": f"tp{context.tensor_parallel}"}
    for key, supported in SUPPORTED_EXECUTION.items():
        if declared_execution[key] != supported:
            refusals.append(f"the table's {key} is {declared_execution[key]!r}, and a recomputed "
                            f"partition covers only {supported!r}")
    if context.batch_size != 1:
        refusals.append(f"the table's batch size is {context.batch_size}, and a recomputed "
                        "partition covers only one request")

    # One measured assignment establishes no fixed charge under the others.
    menu = {(row.unit, row.fmt): row for row in table.rows}
    table_units = sorted({row.unit for row in table.rows})
    alternatives = sorted(unit for unit in table_units
                          if sum(1 for row in table.rows if row.unit == unit) > 1)
    if alternatives:
        refusals.append(f"the table prices more than one format for {alternatives}, and one "
                        "measured assignment establishes no invariant fixed charge under the others")

    # The reference must partition the roster this table independently supplies.
    census = report["reference"]["canonical_census"]
    census_units = None
    if not isinstance(census, Mapping) or set(census) != {"units"}:
        refusals.append("the reference carries no canonical census naming its units, so it "
                        "partitions no model roster")
    else:
        units = census["units"]
        if (not isinstance(units, list) or not all(isinstance(unit, str) for unit in units)
                or len(set(units)) != len(units)):
            refusals.append("the canonical census does not name a unique roster of units")
        else:
            census_units = sorted(units)
            if census_units != table_units:
                refusals.append(f"the canonical census names units {census_units} where this "
                                f"table prices {table_units}")
    selected = {}
    for index, row in enumerate(report["reference"]["selected_rows"]):
        if not isinstance(row, Mapping) or "unit" not in row or not set(row) <= {"unit", "format"}:
            refusals.append(f"selected row {index} is not a unit-and-format reference")
            continue
        unit = row["unit"]
        if unit in selected:
            refusals.append(f"unit {unit!r} carries more than one selected row, so the report "
                            "measures no single complete assignment")
        selected[unit] = row
        if "format" not in row:
            refusals.append(f"the selected row for unit {unit!r} names no format, so it binds to "
                            "no priced table row")
        elif (unit, row["format"]) not in menu:
            refusals.append(f"selected row {(unit, row['format'])} is not a row this table prices")
    if census_units is not None and sorted(selected) != census_units:
        refusals.append(f"the selected rows cover units {sorted(selected)} where the census "
                        f"names {census_units}")
    if table.fixed_assignment:
        refusals.append("the partition names no fixed member, so this table's fixed_assignment "
                        "binds to no observed allocation")

    # The workload the report was measured under must be this table's.
    calibration = report["workload"]["calibration"]
    if calibration is None:
        refusals.append("the workload names no calibration, so this table's calibration "
                        f"{context.calibration_sha256} is bound to nothing")
    elif (not isinstance(calibration, Mapping)
            or calibration.get("sha256") != context.calibration_sha256):
        refusals.append("the workload names another calibration than this table's")

    for name in OWED_OBSERVATIONS:
        if report["observations"][name] is None:
            refusals.append(f"the capture observes no {name}, so {OWED_EVIDENCE[name]} "
                            "has no evidence")

    # The keystone: the table's declared numbers against the recomputation.
    # "Not expressible" and "disagrees" are separate refusals -- a null term is
    # an absence of evidence, not a producer error -- and neither admits.
    fixed = table.fixed_resources
    for term, field in FIXED_TERM_FIELDS.items():
        recomputed = verdict.recomputed_terms[term]
        value = getattr(fixed, field)
        if recomputed is None:
            refusals.append(f"no {term} is recomputable, so this table's fixed {field} "
                            f"({value}) has no evidence")
        elif recomputed != value:
            refusals.append(f"this table declares fixed {field} {value} where the recomputed "
                            f"{term} is {recomputed}")
    resident = verdict.recomputed_terms["candidate_resident"]
    if resident is None:
        refusals.append("no candidate_resident is recomputable, so no priced row's resident "
                        "bytes has evidence")
    else:
        for unit, row in sorted(selected.items()):
            priced = menu.get((unit, row.get("format")))
            if priced is None:
                continue
            if unit not in resident:
                refusals.append(f"the partition charges unit {unit!r} no resident bytes")
            elif resident[unit] != priced.resources.resident_bytes:
                refusals.append(f"this table declares {unit!r} resident bytes "
                                f"{priced.resources.resident_bytes} where the recomputed "
                                f"candidate_resident is {resident[unit]}")
    refusals.append("the native-row and full-engine transient charge boundary is not versioned, "
                    "so no candidate activation or scratch term may be compared to a priced row")
    if fixed.serialized_bytes:
        refusals.append("the report partitions no serialized bytes, so this table's fixed "
                        f"serialized_bytes ({fixed.serialized_bytes}) has no evidence")
    for field in ("prefill_ms", "decode_ms"):
        value = getattr(fixed, field)
        if value:
            refusals.append("the report carries no timing partition, so this table's fixed "
                            f"{field} ({value}) has no evidence")
    return refusals


def admit_native_rows(table, relation):
    """Reuse exact same-panel producer gates before accepting v2 table rows."""
    from .native_moe_panel import consume_moe_receipt
    from .native_operator_panel import consume_native_receipt
    reader = ArtifactReader(Path(table.source_path).parent)
    bindings = table.native_receipt_bindings
    if not isinstance(bindings, (list, tuple)):
        raise RuntimePriceError("native receipt bindings must be an explicit list")
    by_key = {}
    for item in bindings:
        _object(item, ("unit", "format", "run_id", "panel", "receipt", "memory_trace"), "native receipt binding")
        key = item["unit"], item["format"]
        if key in by_key:
            raise RuntimePriceError("duplicate native row receipt binding")
        by_key[key] = item
    _equal(set(by_key), {row.key for row in table.rows}, "native row receipt coverage")
    for row in table.rows:
        binding = by_key[row.key]
        run_id = binding["run_id"]
        if run_id not in relation["runs"] or run_id == relation["full_engine_run_id"]:
            raise RuntimePriceError("native row requires its original native runtime")
        _, panel = reader.json(binding["panel"], "independent native panel")
        receipt_path, receipt = reader.json(binding["receipt"], "native receipt")
        trace_path, _ = reader.bytes(binding["memory_trace"], "native memory trace")
        run = relation["runs"][run_id]
        _equal(panel["runtime"], run["raw"], "original native panel runtime")
        _equal(panel["cost_sha256"], table.cost_sha256, "native panel cost payload")
        _equal(panel["source_sha256"], table.context.source_sha256, "native source model")
        _equal(panel["calibration_sha256"], table.context.calibration_sha256, "native calibration")
        if table.context.batch_size != 1:
            raise RuntimePriceError("native panel admission currently requires batch size one")
        if panel["schema"] == "tessera.native_moe_panel.v1":
            _equal(run["raw"]["schema"], "tessera.native_moe_runtime.v1", "native routed runtime scope")
            _equal(panel["serving_config_sha256"], relation["configuration_sha256"], "native panel configuration")
            wire_records = [member["wire"]["record"] for member in panel["members"]]
            consume = consume_moe_receipt
            expected_binding = panel["runtime_binding"]
        elif panel["schema"] == "tessera.native_dense_panel.v1":
            _equal(run["raw"]["schema"], "tessera.native_dense_runtime.v1", "native dense runtime scope")
            wire_records = [panel["wire"]["record"]]
            consume = consume_native_receipt
            expected_binding = {"member_formats": {panel["unit"]: panel["format"]},
                "member_operator_identity_sha256": {panel["unit"]: panel["joint_operator_identity_sha256"]},
                "member_shapes": {panel["unit"]: panel["shape"]},
                "operator_route": panel["phases"]["prefill"]["expected_route"]["symbol"]}
        else:
            raise RuntimePriceError("unsupported native producer panel")
        for record in wire_records:
            _equal(record["identity"]["encoder_source_sha256"],
                   run["common"]["producer_source_tree_sha256"], "original wire producer source-tree seal")
        try:
            observation = consume(receipt_path, expected_sha256=binding["receipt"]["sha256"],
                                  expected_panel=panel, memory_trace_path=trace_path)
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimePriceError(f"native producer admission refused: {exc}") from exc
        _equal(observation["unit"], row.unit, "native row unit")
        _equal(observation["format"], row.fmt, "native row format")
        _equal(expected_binding, row.binding.as_dict(), "native row operator binding")
        _equal(observation["serialized_unit_bytes"], row.resources.serialized_bytes, "native serialized bytes")
        _equal(observation["resident_bytes"], row.resources.resident_bytes, "native resident bytes")
        scratch, activation = [], []
        for phase, measurement in (("prefill", row.prefill), ("decode", row.decode)):
            if measurement is None:
                raise RuntimePriceError("native v2 row lacks a complete measured phase")
            actual = observation["phases"][phase]
            if actual["peak_scratch_bytes"] is None:
                raise RuntimePriceError("native row has an incomplete resource ledger")
            expected_path, _ = reader.bytes({"path": measurement.receipt_path, "sha256": measurement.receipt_sha256}, "table native phase receipt")
            _equal(expected_path.resolve(), receipt_path.resolve(), "native phase receipt path")
            _equal(measurement.receipt_sha256, binding["receipt"]["sha256"], "native phase receipt bytes")
            for key in ("method", "samples_ms", "warmup_iterations"):
                _equal(actual["measurement"][key], measurement.as_dict()[key], "native phase samples")
            _equal(panel["phases"][phase]["m"], table.context.prompt_tokens if phase == "prefill" else 1, "native phase token scope")
            scratch.append(actual["peak_scratch_bytes"]); activation.append(actual["input_bytes"])
        _equal(row.resources.peak_scratch_bytes, max(scratch), "native maximum phase scratch")
        _equal(row.resources.activation_bytes, max(activation), "native maximum phase input residency")


def admit_runtime_provenance(table):
    """The v2 loader calls this after its ordinary raw-receipt hash checks."""
    try:
        relation = load_runtime_relation(table.runtime_provenance, context=table.context,
                                         root=Path(table.source_path).parent)
        admit_native_rows(table, relation)
        admit_fixed_resources(table, relation)
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimePriceError(f"runtime producer evidence is missing or malformed: {exc}") from exc
