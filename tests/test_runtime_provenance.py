"""Synthetic admission contracts, never runtime or resource measurements."""
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import tarfile

import pytest

from prismaquant.measured_runtime_prices import (
    RuntimePriceError, identity_sha256, load_measured_runtime_table,
    parse_runtime_context,
)


def context_payload():
    return {
        "schema": "prismaquant.measured_runtime_context.v2",
        "runtime_identity_kind": "prismaquant.runtime_provenance_relation.v1",
        "serving_context": {"platform": "sm_121", "structure": "dense",
            "residency": "resident", "runtime_image": "example.invalid/runtime@sha256:" + "a" * 64,
            "execution_mode": "eager"},
        "gpu_identity": "synthetic-gpu", "runtime_sha256": "b" * 64,
        "source_sha256": "c" * 64, "calibration_sha256": "d" * 64,
        "prompt_tokens": 512, "batch_size": 1, "tensor_parallel": 1,
        "graph_mode": "eager", "operator_routes": {"layer": {"FP8": "synthetic"}},
    }


def test_relation_context_keeps_a_distinct_derivation_identity():
    payload = context_payload()
    assert parse_runtime_context(payload).as_dict() == payload

from types import SimpleNamespace
from prismaquant.runtime_provenance import (
    ArtifactReader, admit_fixed_resources, admit_native_rows, load_runtime_relation,
)
from prismaquant.measured_runtime_prices import (
    RuntimeBinding, RuntimeResources, OperatorMeasurement, MeasuredRuntimeRow,
    build_runtime_resources, parse_measured_runtime_table,
)
from prismaquant.native_operator_panel import operator_route_identity
from test_native_operator_panel import joined, receipt_fixture
from test_measured_runtime_prices import payload


class Evidence:
    def __init__(self, root):
        self.root = root

    def raw(self, name, raw):
        path = self.root / name
        path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    def put(self, name, value):
        return self.raw(name, json.dumps(value, sort_keys=True).encode())

    def get(self, reference):
        return json.loads((self.root / reference["path"]).read_bytes())

    def replace(self, reference, value):
        reference.update(self.put(reference["path"], value))


@pytest.fixture
def relation_fixture(tmp_path):
    evidence = Evidence(tmp_path)
    context = context_payload()
    manifest = evidence.put("image-manifest.json", {"schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {"digest": "sha256:" + "8" * 64}})
    image = "example.invalid/runtime@sha256:" + manifest["sha256"]
    context["serving_context"]["runtime_image"] = image
    image_id = "sha256:" + "8" * 64
    served_artifact = tmp_path / "served-artifact"
    served_artifact.mkdir()
    (served_artifact / "tessera_serving_manifest.json").write_text(json.dumps(
        {"modules": {"synthetic-module": {"family": "TESSERA_BF16"}}}))
    config = evidence.put("config.json", {"runtime_image": image, "engine_args": {}, "environment": {},
        "artifact": {"path": str(served_artifact), "scope": "synthetic served artifact"}})
    package_files = {name: {"sha256": hashlib.sha256(name.encode()).hexdigest(), "bytes": len(name)}
                     for name in ("__init__.py", "cached_unit.py", "serving/runtime_contract.json")}
    source_files = {name: name.encode() for name in package_files}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, raw in dict(source_files, **{"_dev/example.py": b"synthetic development source"}).items():
            info = tarfile.TarInfo("src/tessera/" + name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    archive_ref = evidence.raw("plugin.tar", buffer.getvalue())
    installed_digest = hashlib.sha256()
    for name, raw in sorted(source_files.items()):
        if name.endswith(".py"):
            installed_digest.update(name.encode() + b"\0" + raw + b"\0")
    installed_sha256 = installed_digest.hexdigest()
    installation = {"registry_base": image, "launcher_declared_image_id": image_id, "core_manifest_sha256": "1" * 64,
                    "core_files_unchanged": 42, "plugin_files": package_files, "plugin_source_commit": "2" * 40,
                    "plugin_archive_sha256": archive_ref["sha256"], "plugin_entrypoints": {"tessera": "tessera.serving:register"}}
    installer = evidence.put("installation.json", installation)
    package = {"schema": "tessera.loaded_package_identity.v1", "package_path": "/installed/tessera",
               "installer_evidence_sha256": installer["sha256"], "encoder_source_sha256": installed_sha256,
               "package_files": package_files, "package_files_unchanged_from_installer": True,
               "module_identity_errors": [], "loaded_tessera_modules": {
                   name: {"file": "/installed/tessera/" + filename, "origin": "/installed/tessera/" + filename,
                          "sha256": package_files[filename]["sha256"]}
                   for name, filename in (("tessera", "__init__.py"), ("tessera.cached_unit", "cached_unit.py"))}}
    post_package = evidence.put("package.json", package)
    runs = {}
    for name, scope in (("native", "native_operator"), ("engine", "full_engine")):
        binary = evidence.raw(name + "-observer.so", (name + " observer bytes").encode())
        source = evidence.raw(name + "-observer.cpp", b"synthetic collector source")
        build = evidence.put(name + "-build.json", {"source_sha256": source["sha256"], "output_sha256": binary["sha256"]})
        harness = evidence.raw(name + "-harness.py", b"synthetic harness source")
        analysis = evidence.raw(name + "-analysis.py", b"synthetic analysis source")
        loaded_path = "/measurement/" + name + "-observer.so"
        base = {"schema": "tessera.native_dense_runtime.v1", "image": image,
                "execution": {"mode": "resident", "execution_mode": "eager", "tensor_parallel": 1},
                "gpu": {"uuid": "synthetic-gpu", "capability": [12, 1]},
                "versions": {"torch": "synthetic"}, "arithmetic": {"tf32": False},
                "source": {"tessera_package_sha256": installed_sha256,
                           "runtime_contract_sha256": package_files["serving/runtime_contract.json"]["sha256"],
                           "harness_sha256": harness["sha256"]},
                "image_declaration": {"record": {"refused": False, "present": True, "gated": True,
                    "pinned": image, "required": image, "reason": "pinned",
                    "resolved_reference": image, "requested": image,
                    "repo_digests": [image], "local_id": image_id, "selection": {"configuration_sha256": config["sha256"]}}},
                "native_libraries": {"/usr/lib/libtorch.so": "5" * 64, loaded_path: binary["sha256"]}}
        observer = {"library_sha256": binary["sha256"], "loaded_path": loaded_path, "analysis_source_sha256": analysis["sha256"]}
        sources = {"harness_sha256": harness}
        if scope == "native_operator":
            base["resource_collector"] = observer
            raw = base
            sources["resource_analysis_source_sha256"] = analysis
            audit = {"native_returncode": 0, "manifest_sha256": "1" * 64, "stock_files_unchanged": 42}
        else:
            raw = {"schema": "tessera.full_engine_runtime.v1", "base": base,
                   "actual_execution": base["execution"], "configuration_sha256": config["sha256"],
                   "loaded_package": package, "execution": {"engine_args": {}, "environment": {}},
                   "source": {"full_engine_worker_sha256": harness["sha256"]},
                   "instrumentation": {"resource_collector": observer}}
            sources["full_engine_worker_sha256"] = harness
            base["native_libraries"]["/usr/lib/engine-extra.so"] = "6" * 64
            audit = {"core_audit_" + phase: {"manifest_sha256": "1" * 64, "unchanged_files": 42}
                     for phase in ("before", "after")}
        runs[name] = {"scope": scope, "runtime_field": None, "runtime": evidence.put(name + "-runtime.json", raw),
                      "installation": copy.deepcopy(installer), "post_core": evidence.put(name + "-audit.json", audit),
                      "post_package": copy.deepcopy(post_package),
                      "instrumentation": {"artifacts": {}, "libraries": [{"role": "resource_collector", "loaded_path": loaded_path,
                          "artifact": binary, "source": source, "build_receipt": build}], "python_sources": sources}}
    relation = {"schema": "prismaquant.runtime_provenance_relation.v1", "configuration": config, "image_manifest": manifest, "runs": runs,
                "package_source": {"archive": archive_ref, "prefix": "src/tessera", "excluded_files": ["_dev/example.py"]},
                "full_engine_run_id": "engine", "production_dependencies": [{"native_run_id": "native",
                    "native_path": "/usr/lib/libtorch.so", "full_engine_path": "/usr/lib/libtorch.so", "sha256": "5" * 64}],
                "full_engine_extra_libraries": {"/usr/lib/engine-extra.so": {"sha256": "6" * 64, "scope": "full_engine"}}}
    return evidence, relation, context


def relation_load(fixture):
    evidence, relation, context = fixture
    context = dict(context, runtime_sha256=identity_sha256(relation))
    return load_runtime_relation(evidence.put("relation.json", relation), context=parse_runtime_context(context), root=evidence.root)


def test_relation_preserves_distinct_raw_manifests_and_exhaustive_dependencies(relation_fixture):
    evidence, original, _ = relation_fixture
    admitted = relation_load(relation_fixture)
    assert admitted["record"] == original
    native, engine = admitted["runs"]["native"], admitted["runs"]["engine"]
    assert native["sha256"] != engine["sha256"] != identity_sha256(original)
    assert native["sha256"] == identity_sha256(evidence.get(original["runs"]["native"]["runtime"]))


@pytest.mark.parametrize("mutation", ["common_bytes", "missing_dependency", "undeclared_extra", "missing_mapping",
    "different_core", "different_package", "foreign_config", "foreign_gpu", "boolean_tp", "missing_origin",
    "foreign_origin", "module_bytes", "missing_module", "installer_drift", "missing_source", "binary_drift",
    "installed_instrumentation", "runtime_hash", "arithmetic"])
def test_relation_refuses_unproved_or_changed_coordinates(relation_fixture, mutation):
    evidence, relation, _ = relation_fixture
    run = relation["runs"]["native"]
    raw = evidence.get(run["runtime"])
    if mutation == "common_bytes":
        raw["native_libraries"]["/usr/lib/libtorch.so"] = "7" * 64
    elif mutation == "missing_dependency":
        raw["native_libraries"]["/usr/lib/missing.so"] = "7" * 64
    elif mutation == "undeclared_extra":
        relation["full_engine_extra_libraries"] = {}
    elif mutation == "missing_mapping":
        relation["production_dependencies"] = []
    elif mutation in ("different_core", "different_package", "installer_drift"):
        install = evidence.get(run["installation"])
        install["core_manifest_sha256" if mutation == "different_core" else "plugin_archive_sha256"] = "7" * 64
        evidence.replace(run["installation"], install)
    elif mutation == "foreign_config":
        raw["image_declaration"]["record"]["selection"]["configuration_sha256"] = "7" * 64
    elif mutation == "foreign_gpu":
        raw["gpu"]["uuid"] = "foreign"
    elif mutation == "boolean_tp":
        raw["execution"]["tensor_parallel"] = True
    elif mutation in ("missing_origin", "foreign_origin", "module_bytes", "missing_module"):
        package = evidence.get(run["post_package"])
        module = package["loaded_tessera_modules"]["tessera.cached_unit"]
        if mutation == "missing_origin":
            module["origin"] = None
        elif mutation == "foreign_origin":
            module["origin"] = "/elsewhere/cached_unit.py"
        elif mutation == "module_bytes":
            module["sha256"] = "7" * 64
        else:
            del package["loaded_tessera_modules"]["tessera.cached_unit"]
        evidence.replace(run["post_package"], package)
    elif mutation == "missing_source":
        run["instrumentation"]["python_sources"] = {}
    elif mutation == "binary_drift":
        run["instrumentation"]["libraries"][0]["artifact"]["sha256"] = "7" * 64
    elif mutation == "installed_instrumentation":
        library = run["instrumentation"]["libraries"][0]
        raw["native_libraries"]["/usr/lib/observer.so"] = raw["native_libraries"].pop(library["loaded_path"])
        library["loaded_path"] = "/usr/lib/observer.so"
    elif mutation == "runtime_hash":
        run["runtime"]["sha256"] = "7" * 64
    elif mutation == "arithmetic":
        raw["arithmetic"]["tf32"] = True
    if mutation != "runtime_hash":
        evidence.replace(run["runtime"], raw)
    with pytest.raises(RuntimePriceError):
        relation_load(relation_fixture)


@pytest.mark.parametrize("claim", [None, {}, {"status": "complete", "resident_bytes": 0}])
def test_relation_never_admits_unproved_fixed_resources(tmp_path, claim):
    """A receipt-level status flag is not evidence: `full_model_resources` must
    reference a full-engine resource report this consumer recomputes itself,
    and the surrounding `complete`/`qualified_complete` fields are read by
    nothing. What that recomputation then refuses lives in
    `tests/test_runtime_fixed_resource_admission.py`."""
    evidence = Evidence(tmp_path)
    ref = evidence.put("fixed.json", {"full_model_resources": claim, "full_model_fixed_resources_complete": True,
                                     "status": "qualified_complete", "closure": {"complete": True}})
    table = SimpleNamespace(source_path=str(tmp_path / "table.json"), fixed_resources_receipt_path=ref["path"],
                            fixed_resources_receipt_sha256=ref["sha256"])
    with pytest.raises(RuntimePriceError, match="incomplete|must reference one recomputable"):
        admit_fixed_resources(table, {})


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'[]'])
def test_artifacts_refuse_ambiguous_json(tmp_path, raw):
    ref = Evidence(tmp_path).raw("ambiguous.json", raw)
    with pytest.raises(RuntimePriceError):
        ArtifactReader(tmp_path).json(ref, "fixture")

@pytest.mark.parametrize("mutation", ["wrong_scope", "build_source", "build_output", "engine_config", "worker_source"])
def test_relation_refuses_unbound_observer_evidence(relation_fixture, mutation):
    evidence, relation, _ = relation_fixture
    run = relation["runs"]["engine"]
    if mutation == "wrong_scope":
        relation["full_engine_extra_libraries"]["/usr/lib/engine-extra.so"]["scope"] = "native_operator"
    elif mutation.startswith("build_"):
        ref = run["instrumentation"]["libraries"][0]["build_receipt"]
        build = evidence.get(ref)
        build["source_sha256" if mutation == "build_source" else "output_sha256"] = "7" * 64
        evidence.replace(ref, build)
    elif mutation == "worker_source":
        del run["instrumentation"]["python_sources"]["full_engine_worker_sha256"]
    else:
        raw = evidence.get(run["runtime"])
        raw["execution"]["engine_args"]["unexpected_override"] = True
        evidence.replace(run["runtime"], raw)
    with pytest.raises(RuntimePriceError):
        relation_load(relation_fixture)


def test_relation_keeps_original_preflight_envelope(relation_fixture):
    evidence, relation, _ = relation_fixture
    run = relation["runs"]["native"]
    raw = evidence.get(run["runtime"])
    envelope = {"schema": "tessera.native_dense_preflight.v1", "runtime": raw,
                "runtime_sha256": identity_sha256(raw), "original_other_field": "retained"}
    run["runtime"] = evidence.put("original-preflight.json", envelope)
    run["runtime_field"] = "runtime"
    result = relation_load(relation_fixture)
    assert result["runs"]["native"]["raw"] == raw
    assert evidence.get(run["runtime"]) == envelope


@pytest.fixture
def native_intake(relation_fixture, joined):
    evidence, relation, context = relation_fixture
    inputs, preflight, joint = joined
    raw = evidence.get(relation["runs"]["native"]["runtime"])
    inputs["runtime_image"] = raw["image"]
    source_tree_sha = relation_load(relation_fixture)["runs"]["native"]["common"]["producer_source_tree_sha256"]
    inputs["wire"]["record"]["identity"] = {"encoder_source_sha256": source_tree_sha}
    preflight["operator"]["wire_record_sha256"] = identity_sha256(inputs["wire"]["record"])
    raw["execution"] = copy.deepcopy(preflight["runtime"]["execution"])
    evidence.replace(relation["runs"]["native"]["runtime"], raw)
    preflight["runtime"] = raw
    preflight["runtime_sha256"] = identity_sha256(raw)
    panel, receipt, trace = receipt_fixture(joined, complete=True)
    route_identity = operator_route_identity(panel["phases"]["prefill"]["expected_route"])
    trace["capture"]["collector_library_sha256"] = raw["resource_collector"]["library_sha256"]
    receipt["resources"]["trace_sha256"] = identity_sha256(trace)
    context.update(prompt_tokens=1, source_sha256=panel["source_sha256"],
                   calibration_sha256=panel["calibration_sha256"],
                   runtime_sha256=identity_sha256(relation),
                   operator_routes={panel["unit"]: {panel["format"]: route_identity}})
    panel_ref, receipt_ref, trace_ref = (evidence.put(name, value) for name, value in (
        ("panel.json", panel), ("receipt.json", receipt), ("trace.json", trace)))
    measurement = OperatorMeasurement.from_dict({"method": "cuda_events", "samples_ms": [3., 1., 2.],
        "warmup_iterations": 4, "receipt_path": receipt_ref["path"], "receipt_sha256": receipt_ref["sha256"]})
    binding = RuntimeBinding.from_dict({"member_formats": {panel["unit"]: panel["format"]},
        "member_operator_identity_sha256": {panel["unit"]: panel["joint_operator_identity_sha256"]},
        "member_shapes": {panel["unit"]: panel["shape"]}, "operator_route": route_identity})
    row = MeasuredRuntimeRow(panel["unit"], panel["format"], binding,
        RuntimeResources(prefill_ms=2., decode_ms=2., serialized_bytes=42, resident_bytes=64,
                         peak_scratch_bytes=128, activation_bytes=8, kv_bytes=0), measurement, measurement)
    table = SimpleNamespace(source_path=str(evidence.root / "table.json"), rows=(row,),
        context=parse_runtime_context(context), cost_sha256=panel["cost_sha256"],
        native_receipt_bindings=[{"unit": row.unit, "format": row.fmt, "run_id": "native",
            "panel": panel_ref, "receipt": receipt_ref, "memory_trace": trace_ref}])
    return evidence, table, relation_load(relation_fixture)


def test_native_rows_reuse_original_same_run_numerical_and_resource_gates(native_intake):
    _, table, relation = native_intake
    admit_native_rows(table, relation)


@pytest.mark.parametrize("mutation", ["foreign_runtime", "foreign_cost", "foreign_source", "foreign_calibration",
                                     "wrong_scope", "missing_row", "wrong_tokens", "foreign_trace", "wrong_samples", "installed_source_relabel"])
def test_native_rows_refuse_cross_run_relabel_or_scope_drift(native_intake, mutation):
    evidence, table, relation = native_intake
    binding = table.native_receipt_bindings[0]
    panel = evidence.get(binding["panel"])
    if mutation == "installed_source_relabel":
        panel["wire"]["record"]["identity"]["encoder_source_sha256"] = relation["runs"]["native"]["common"]["package_sha256"]
    elif mutation == "foreign_runtime":
        panel["runtime"] = relation["runs"]["engine"]["raw"]
    elif mutation in ("foreign_cost", "foreign_source", "foreign_calibration"):
        panel[mutation.removeprefix("foreign_") + "_sha256"] = "0" * 64
    elif mutation == "wrong_scope":
        binding["run_id"] = "engine"
    elif mutation == "missing_row":
        table.native_receipt_bindings = []
    elif mutation == "wrong_tokens":
        table.context = parse_runtime_context(dict(table.context.as_dict(), prompt_tokens=512))
    elif mutation == "foreign_trace":
        trace = evidence.get(binding["memory_trace"])
        trace["capture"]["collector_library_sha256"] = "0" * 64
        evidence.replace(binding["memory_trace"], trace)
    else:
        receipt = evidence.get(binding["receipt"])
        receipt["phases"]["decode"]["measurement"]["samples_ms"] = [4., 1., 2.]
        evidence.replace(binding["receipt"], receipt)
    evidence.replace(binding["panel"], panel)
    with pytest.raises(RuntimePriceError):
        admit_native_rows(table, relation)


def test_v2_parsing_cannot_supply_unadmitted_allocation_resources(payload):
    payload["schema"] = "prismaquant.measured_runtime_prices.v2"
    payload["context"].update(schema="prismaquant.measured_runtime_context.v2",
        runtime_identity_kind="prismaquant.runtime_provenance_relation.v1")
    payload.update(runtime_provenance={"path": "relation.json", "sha256": "1" * 64}, native_receipt_bindings=[])
    table = parse_measured_runtime_table(payload, expected_context=parse_runtime_context(payload["context"]),
        expected_cost_sha256=payload["cost_sha256"], now=datetime(2026, 9, 5, tzinfo=timezone.utc))
    assert table.as_dict()["schema"] == payload["schema"]
    assert table.as_dict()["context"] == payload["context"]
    with pytest.raises(RuntimePriceError, match="producer admission"):
        build_runtime_resources(table, {}, expected_bindings={})


def test_image_manifest_explicitly_relates_manifest_and_config_local_id_types(relation_fixture):
    evidence, relation, _ = relation_fixture
    run = relation["runs"]["native"]
    raw = evidence.get(run["runtime"])
    raw["image_declaration"]["record"]["local_id"] = "sha256:" + relation["image_manifest"]["sha256"]
    evidence.replace(run["runtime"], raw)
    installation = evidence.get(run["installation"])
    installation["launcher_declared_image_id"] = raw["image_declaration"]["record"]["local_id"]
    run["installation"] = evidence.put("native-installation.json", installation)
    package = evidence.get(run["post_package"])
    package["installer_evidence_sha256"] = run["installation"]["sha256"]
    run["post_package"] = evidence.put("native-package.json", package)
    admitted = relation_load(relation_fixture)
    assert admitted["runs"]["native"]["common"]["image_identity"] == {
        "manifest_digest": "sha256:" + relation["image_manifest"]["sha256"], "config_digest": "sha256:" + "8" * 64}
    assert admitted["runs"]["native"]["raw"] == raw


@pytest.mark.parametrize("mutation", ["unrelated_id", "changed_manifest", "changed_config"])
def test_image_relation_refuses_unproved_identity_types(relation_fixture, mutation):
    evidence, relation, _ = relation_fixture
    if mutation == "unrelated_id":
        run = relation["runs"]["native"]
        raw = evidence.get(run["runtime"])
        raw["image_declaration"]["record"]["local_id"] = "sha256:" + "7" * 64
        evidence.replace(run["runtime"], raw)
    else:
        ref = relation["image_manifest"]
        manifest = evidence.get(ref)
        manifest["config"]["digest"] = "sha256:" + "7" * 64
        if mutation == "changed_manifest":
            evidence.replace(ref, manifest)
        else:
            (evidence.root / ref["path"]).write_text(json.dumps(manifest))
    with pytest.raises(RuntimePriceError):
        relation_load(relation_fixture)


def _repoint_image(evidence, relation, context):
    """Move every image reference the runs bind to the manifest's digest."""
    image = "example.invalid/runtime@sha256:" + relation["image_manifest"]["sha256"]
    context["serving_context"]["runtime_image"] = image
    config = evidence.get(relation["configuration"])
    config["runtime_image"] = image
    evidence.replace(relation["configuration"], config)
    configuration_sha256 = relation["configuration"]["sha256"]
    for name in ("native", "engine"):
        run = relation["runs"][name]
        raw = evidence.get(run["runtime"])
        base = raw["base"] if run["scope"] == "full_engine" else raw
        base["image"] = image
        record = base["image_declaration"]["record"]
        for field in ("pinned", "required", "resolved_reference", "requested"):
            record[field] = image
        record["repo_digests"] = [image]
        installation = evidence.get(run["installation"])
        installation["registry_base"] = image
        evidence.replace(run["installation"], installation)
        installer_sha256 = run["installation"]["sha256"]
        if run["scope"] == "full_engine":
            raw["configuration_sha256"] = configuration_sha256
            loaded = raw.get("loaded_package")
            if isinstance(loaded, dict):
                loaded["installer_evidence_sha256"] = installer_sha256
        if "selection" in record:
            # The shared base template stamps a launcher selection on both
            # records; every binding to the old configuration digest moves.
            record["selection"]["configuration_sha256"] = configuration_sha256
        evidence.replace(run["runtime"], raw)
        # The repointed installation has a new digest: rebind the package
        # evidence that cites it, the way ``_set_local_id`` does below.
        # (The native run is rebound again there onto its own file.)
        package = evidence.get(run["post_package"])
        package["installer_evidence_sha256"] = installer_sha256
        evidence.replace(run["post_package"], package)
    return image


def _index_relation_fixture(fixture):
    """Re-point the concrete fixture at an index pinning the same config.

    The platform manifest carries the fixture's config digest; the index
    carries the platform manifest (arm64) beside an unrelated amd64 entry.
    Every image reference the runs bind -- configuration, base image,
    declaration, installation -- moves to the index digest, the way a real
    multi-arch pull names the index.
    """
    evidence, relation, context = fixture
    platform = evidence.put("image-platform-manifest.json", {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"digest": "sha256:" + "8" * 64}})
    index = {"schemaVersion": 2,
             "mediaType": "application/vnd.oci.image.index.v1+json",
             "manifests": [
                 {"mediaType": "application/vnd.oci.image.manifest.v1+json",
                  "digest": "sha256:" + platform["sha256"], "size": 512,
                  "platform": {"architecture": "arm64", "os": "linux"}},
                 {"mediaType": "application/vnd.oci.image.manifest.v1+json",
                  "digest": "sha256:" + "9" * 64, "size": 513,
                  "platform": {"architecture": "amd64", "os": "linux"}}]}
    evidence.replace(relation["image_manifest"], index)
    relation["image_platform_manifest"] = platform
    _repoint_image(evidence, relation, context)
    return evidence, relation, context


def _set_local_id(evidence, relation, local_id):
    """One run reporting another honest container ID, carried through."""
    run = relation["runs"]["native"]
    raw = evidence.get(run["runtime"])
    raw["image_declaration"]["record"]["local_id"] = local_id
    evidence.replace(run["runtime"], raw)
    installation = evidence.get(run["installation"])
    installation["launcher_declared_image_id"] = local_id
    run["installation"] = evidence.put("native-installation.json", installation)
    package = evidence.get(run["post_package"])
    package["installer_evidence_sha256"] = run["installation"]["sha256"]
    run["post_package"] = evidence.put("native-package.json", package)


@pytest.mark.parametrize("local_id", ["config", "index", "platform"])
def test_image_index_admits_every_honest_container_id(relation_fixture, local_id):
    """PrismaQuant #723: the pin names what was pulled, the runs name what ran.

    The attested serving image is an OCI index; the loader binds the pinned
    index digest, resolves the platform manifest for the entry it carries,
    and records the index digest, the platform digest and the config digest
    together. A run may then report any of the three as its container ID.
    """
    evidence, relation, _ = _index_relation_fixture(relation_fixture)
    identities = {
        "config": "sha256:" + "8" * 64,
        "index": "sha256:" + relation["image_manifest"]["sha256"],
        "platform": "sha256:" + relation["image_platform_manifest"]["sha256"],
    }
    _set_local_id(evidence, relation, identities[local_id])
    admitted = relation_load(relation_fixture)
    assert admitted["runs"]["native"]["common"]["image_identity"] == {
        "manifest_digest": identities["index"],
        "platform_manifest_digest": identities["platform"],
        "platform_architecture": "arm64", "platform_os": "linux",
        "config_digest": identities["config"]}
    assert admitted["runs"]["engine"]["common"]["image_identity"] == (
        admitted["runs"]["native"]["common"]["image_identity"])


@pytest.mark.parametrize("mutation", ["missing_platform", "platform_not_indexed", "entry_not_concrete",
    "media_mismatch", "platform_beside_concrete", "unrelated_id", "unknown_media",
    "entry_without_platform", "config_drift"])
def test_image_index_refuses_unproved_resolution(relation_fixture, mutation):
    evidence, relation, context = _index_relation_fixture(relation_fixture)
    if mutation == "missing_platform":
        del relation["image_platform_manifest"]
    elif mutation == "platform_beside_concrete":
        manifest = {"schemaVersion": 2,
                    "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                    "config": {"digest": "sha256:" + "8" * 64}}
        evidence.replace(relation["image_manifest"], manifest)
        _repoint_image(evidence, relation, context)
    elif mutation == "platform_not_indexed":
        index = evidence.get(relation["image_manifest"])
        index["manifests"] = [entry for entry in index["manifests"]
                              if entry["digest"] != "sha256:" + relation["image_platform_manifest"]["sha256"]]
        evidence.replace(relation["image_manifest"], index)
        _repoint_image(evidence, relation, context)
    elif mutation == "entry_not_concrete":
        index = evidence.get(relation["image_manifest"])
        index["manifests"][0]["mediaType"] = "application/vnd.oci.image.index.v1+json"
        evidence.replace(relation["image_manifest"], index)
        _repoint_image(evidence, relation, context)
    elif mutation == "media_mismatch":
        index = evidence.get(relation["image_manifest"])
        index["manifests"][0]["mediaType"] = (
            "application/vnd.docker.distribution.manifest.v2+json")
        evidence.replace(relation["image_manifest"], index)
        _repoint_image(evidence, relation, context)
    elif mutation == "entry_without_platform":
        index = evidence.get(relation["image_manifest"])
        del index["manifests"][0]["platform"]
        evidence.replace(relation["image_manifest"], index)
        _repoint_image(evidence, relation, context)
    elif mutation == "unknown_media":
        index = evidence.get(relation["image_manifest"])
        index["mediaType"] = "application/vnd.example.unknown+json"
        evidence.replace(relation["image_manifest"], index)
        _repoint_image(evidence, relation, context)
    elif mutation == "config_drift":
        platform = evidence.get(relation["image_platform_manifest"])
        platform["config"]["digest"] = "sha256:" + "7" * 64
        evidence.replace(relation["image_platform_manifest"], platform)
        index = evidence.get(relation["image_manifest"])
        for entry in index["manifests"]:
            if entry["digest"] != "sha256:" + "9" * 64:
                entry["digest"] = "sha256:" + relation["image_platform_manifest"]["sha256"]
        evidence.replace(relation["image_manifest"], index)
        _repoint_image(evidence, relation, context)
    else:
        run = relation["runs"]["native"]
        raw = evidence.get(run["runtime"])
        raw["image_declaration"]["record"]["local_id"] = "sha256:" + "7" * 64
        evidence.replace(run["runtime"], raw)
    with pytest.raises(RuntimePriceError):
        relation_load(relation_fixture)


def test_nested_boolean_arithmetic_cannot_equal_numeric_flag(relation_fixture):
    evidence, relation, _ = relation_fixture
    run = relation["runs"]["native"]
    raw = evidence.get(run["runtime"])
    raw["arithmetic"]["tf32"] = 0
    evidence.replace(run["runtime"], raw)
    with pytest.raises(RuntimePriceError, match="arithmetic"):
        relation_load(relation_fixture)


def test_actual_multi_build_receipt_binds_observer_source_and_output(relation_fixture):
    evidence, relation, _ = relation_fixture
    library = relation["runs"]["native"]["instrumentation"]["libraries"][0]
    name = "native-observer.so"
    evidence.replace(library["build_receipt"], {"builds": [{"name": name, "returncode": 0}],
        "files": {name: {"sha256": library["artifact"]["sha256"]}},
        "source_files": {"native-observer.cpp": library["source"]["sha256"]}})
    relation_load(relation_fixture)


def test_source_tree_and_installed_package_have_distinct_recomputed_roles(relation_fixture):
    admitted = relation_load(relation_fixture)
    common = admitted["runs"]["native"]["common"]
    assert common["producer_source_tree_sha256"] != common["package_sha256"]


@pytest.mark.parametrize("mutation", ["undeclared_omission", "unknown_omission", "archive_drift", "source_relabel"])
def test_package_archive_refuses_unproved_installed_subset(relation_fixture, mutation):
    evidence, relation, _ = relation_fixture
    declaration = relation["package_source"]
    if mutation == "undeclared_omission":
        declaration["excluded_files"] = []
    elif mutation == "unknown_omission":
        declaration["excluded_files"].append("not_in_archive.py")
    elif mutation == "archive_drift":
        declaration["archive"]["sha256"] = "0" * 64
    else:
        run = relation["runs"]["native"]
        package = evidence.get(run["post_package"])
        package["encoder_source_sha256"] = "0" * 64
        evidence.replace(run["post_package"], package)
    with pytest.raises(RuntimePriceError):
        relation_load(relation_fixture)


@pytest.mark.parametrize("coordinate", ["platform", "graph_mode"])
def test_relation_checks_device_and_execution_against_context(relation_fixture, coordinate):
    _, _, context = relation_fixture
    if coordinate == "platform":
        context["serving_context"]["platform"] = "sm_100"
    else:
        context["graph_mode"] = "full"
    with pytest.raises(RuntimePriceError, match="actual GPU platform|actual graph mode"):
        relation_load(relation_fixture)


# --------------------------------------------------------------------------- #
# The three shapes a real Tessera run has and the synthetic fixture did not.
# Each is the producer's own spelling, read off the #399 Qwen3-0.6B full-engine
# capture and the frozen d403cc5a31 producer tree; the loader refused all three.
# --------------------------------------------------------------------------- #

def _records(evidence, relation):
    """Every run's image-declaration record, with a writer back to evidence."""
    for run in relation["runs"].values():
        raw = evidence.get(run["runtime"])
        base = raw if run["scope"] == "native_operator" else raw["base"]
        yield run, raw, base["image_declaration"]["record"]


def test_relation_reads_the_gated_reference_not_tesseras_packaged_pin(relation_fixture):
    """A lane image is `required`; `pinned` names Tessera's packaged default.

    `serving/runtime_image.resolve` sets `required = requested` and
    `reason = "explicit_digest"` whenever the requested repository is not the
    pinned one, which is every artifact served out of a lane image. The
    #399 capture records exactly that, and reading `pinned` refused it.
    """
    evidence, relation, _ = relation_fixture
    image = None
    for run, raw, record in _records(evidence, relation):
        image = record["requested"]
        record["pinned"] = "vllm/vllm-openai@sha256:" + "9" * 64
        record["reason"] = "explicit_digest"
        evidence.replace(run["runtime"], raw)
    admitted = relation_load(relation_fixture)
    assert set(admitted["runs"]) == {"native", "engine"}
    assert admitted["runs"]["engine"]["base"]["image_declaration"]["record"]["required"] == image


@pytest.mark.parametrize("mutation", ["ungated_reason", "foreign_required", "pinned_reason_other_pin"])
def test_relation_still_refuses_an_unproved_image_declaration(relation_fixture, mutation):
    evidence, relation, _ = relation_fixture
    for run, raw, record in _records(evidence, relation):
        if mutation == "ungated_reason":
            record["reason"] = "not_pinned_repository"
        elif mutation == "foreign_required":
            record["required"] = "example.invalid/other@sha256:" + "9" * 64
        else:
            record["pinned"] = "vllm/vllm-openai@sha256:" + "9" * 64
        evidence.replace(run["runtime"], raw)
    with pytest.raises(RuntimePriceError):
        relation_load(relation_fixture)


def test_full_engine_binds_its_configuration_without_a_launcher_selection(relation_fixture):
    """The full-engine record carries `configuration_sha256`; its capture
    stamps no `selection`, and the #399 report has none."""
    evidence, relation, _ = relation_fixture
    run = relation["runs"]["engine"]
    raw = evidence.get(run["runtime"])
    del raw["base"]["image_declaration"]["record"]["selection"]
    evidence.replace(run["runtime"], raw)
    assert relation_load(relation_fixture)["full_engine_run_id"] == "engine"


@pytest.mark.parametrize("run_id,field,value", [
    ("native", None, None),
    ("engine", "configuration_sha256", "7" * 64),
])
def test_a_run_without_its_own_configuration_still_needs_the_launcher_selection(
        relation_fixture, run_id, field, value):
    """Dropping `selection` is admissible only where another binding exists.

    The native record has no configuration of its own, so its launcher stamp
    is the only one; the full-engine record keeps its own binding and a
    disagreeing one is still refused.
    """
    evidence, relation, _ = relation_fixture
    run = relation["runs"][run_id]
    raw = evidence.get(run["runtime"])
    base = raw if run["scope"] == "native_operator" else raw["base"]
    if field is None:
        del base["image_declaration"]["record"]["selection"]
    else:
        raw[field] = value
    evidence.replace(run["runtime"], raw)
    with pytest.raises(RuntimePriceError):
        relation_load(relation_fixture)


def _as_source_tree_install(evidence, relation, *, members=None, identity=None):
    """Restate the installation the way the source-tree installer records it."""
    from prismaquant.runtime_provenance import _source_tree_identity
    _, archive = ArtifactReader(evidence.root).bytes(
        relation["package_source"]["archive"], "fixture archive")
    tree = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as source:
        for member in source:
            if member.isfile():
                tree[member.name] = source.extractfile(member).read()
    computed, count = _source_tree_identity(tree)
    for run in relation["runs"].values():
        install = evidence.get(run["installation"])
        install.pop("plugin_archive_sha256", None)
        install["plugin_source_sha256"] = computed if identity is None else identity
        install["plugin_source_members"] = count if members is None else members
        evidence.replace(run["installation"], install)
        package = evidence.get(run["post_package"])
        package["installer_evidence_sha256"] = run["installation"]["sha256"]
        evidence.replace(run["post_package"], package)
    _refresh_embedded_package(evidence, relation)
    return computed, count


def _refresh_embedded_package(evidence, relation):
    """The full-engine record embeds the loaded package; keep the copies equal."""
    run = relation["runs"]["engine"]
    raw = evidence.get(run["runtime"])
    raw["loaded_package"] = evidence.get(run["post_package"])
    evidence.replace(run["runtime"], raw)


def test_a_source_tree_install_binds_through_its_recomputed_source_identity(relation_fixture):
    """`full_engine_plugin_install.py` declares no archive digest.

    It seals the tree as a SHA-256 over `{member: sha256}` for the build
    metadata plus every file under `src/`, and records the count. Requiring
    `plugin_archive_sha256` refused every source-tree install, which is how
    the #399 capture and every native run beside it are installed.
    """
    evidence, relation, _ = relation_fixture
    computed, count = _as_source_tree_install(evidence, relation)
    admitted = relation_load(relation_fixture)
    assert admitted["runs"]["native"]["common"]["plugin_source_sha256"] == computed
    assert admitted["runs"]["native"]["common"]["plugin_archive_sha256"] is None
    assert count == len(evidence.get(relation["runs"]["native"]["post_package"])["package_files"]) + 1


@pytest.mark.parametrize("mutation", ["foreign_identity", "wrong_member_count", "both_bindings", "neither_binding"])
def test_a_declared_plugin_source_is_recomputed_not_accepted(relation_fixture, mutation):
    evidence, relation, _ = relation_fixture
    if mutation == "foreign_identity":
        _as_source_tree_install(evidence, relation, identity="7" * 64)
    elif mutation == "wrong_member_count":
        _as_source_tree_install(evidence, relation, members=9999)
    else:
        _as_source_tree_install(evidence, relation)
        # Both runs share one installation artifact; mutate its bytes once.
        install = evidence.get(relation["runs"]["native"]["installation"])
        if mutation == "both_bindings":
            install["plugin_archive_sha256"] = relation["package_source"]["archive"]["sha256"]
        else:
            install.pop("plugin_source_sha256", None)
        for run in relation["runs"].values():
            evidence.replace(run["installation"], install)
            package = evidence.get(run["post_package"])
            package["installer_evidence_sha256"] = run["installation"]["sha256"]
            evidence.replace(run["post_package"], package)
        _refresh_embedded_package(evidence, relation)
    with pytest.raises(RuntimePriceError):
        relation_load(relation_fixture)


ROUTE_LIBRARY = "/cache/extensions/tessera_nvfp4_synthetic/tessera_nvfp4_synthetic.so"


def _native_loads_route_library(evidence, relation):
    """The native run loads a route-specific JIT library, as every fp4 cell of
    2026-09-13 loaded ``tessera_nvfp4_84439e84….so`` (#570)."""
    run = relation["runs"]["native"]
    raw = evidence.get(run["runtime"])
    raw["native_libraries"][ROUTE_LIBRARY] = "9" * 64
    evidence.replace(run["runtime"], raw)
    relation["production_dependencies"].append({
        "native_run_id": "native", "native_path": ROUTE_LIBRARY,
        "full_engine_path": ROUTE_LIBRARY, "sha256": "9" * 64})


def _full_engine_run_loading_route_library(evidence, relation, name):
    """A full-engine observation equal to ``engine`` in every common coordinate
    (image, GPU, package, contract, core, plugin) that also loaded the library."""
    run = copy.deepcopy(relation["runs"]["engine"])
    raw = evidence.get(run["runtime"])
    raw["base"]["native_libraries"][ROUTE_LIBRARY] = "9" * 64
    run["runtime"] = evidence.put(name + "-runtime.json", raw)
    return run


def test_one_full_engine_run_that_loaded_the_route_library_binds_it(relation_fixture):
    """#570 leg (b), option A: the coverage rule is satisfiable in the schema
    as it stands. One full-engine serve whose artifact exercises the route
    loads that route's library, and the native cell's bytes bind to it."""
    evidence, relation, _ = relation_fixture
    _native_loads_route_library(evidence, relation)
    relation["runs"]["engine"] = _full_engine_run_loading_route_library(evidence, relation, "engine")
    admitted = relation_load(relation_fixture)
    assert admitted["runs"]["engine"]["production"][ROUTE_LIBRARY] == "9" * 64
    assert admitted["runs"]["native"]["production"][ROUTE_LIBRARY] == "9" * 64


def test_a_route_library_the_full_engine_run_never_loaded_refuses_by_name(relation_fixture):
    """The 2026-09-13 refusal: ``engine-a5`` served a uniform ``TESSERA_FP8``
    artifact and never loaded the nvfp4 extension the fp4 cells did."""
    evidence, relation, _ = relation_fixture
    _native_loads_route_library(evidence, relation)
    with pytest.raises(RuntimePriceError, match="missing or changed full-engine production dependency"):
        relation_load(relation_fixture)


def test_a_second_full_engine_run_cannot_supply_the_missing_coverage(relation_fixture):
    """#570 leg (b), decided 2026-09-17: option B is refused as a design.

    ``common`` (image, GPU, versions, arithmetic, package, contract, core
    manifest, plugin files) is equal across every run by construction, so it
    cannot tell two full-engine runs apart; what a second run would bring is a
    library set no single serve produced, under a configuration the native
    cell was not launched against. The relation keeps ONE full-engine
    observation, so a native library either binds to the serve that priced the
    table or is reported unbound -- even when another engine run, equal in
    every common coordinate, did load those bytes."""
    evidence, relation, _ = relation_fixture
    _native_loads_route_library(evidence, relation)
    relation["runs"]["engine2"] = _full_engine_run_loading_route_library(evidence, relation, "engine2")
    with pytest.raises(RuntimePriceError, match="full-engine observation coverage"):
        relation_load(relation_fixture)
    # Naming the second run as THE full-engine run does not help either: the
    # first is then an undeclared full-engine observation, and the relation
    # refuses on the same rule rather than silently adopting whichever run
    # happens to cover the library.
    relation["full_engine_run_id"] = "engine2"
    with pytest.raises(RuntimePriceError, match="full-engine observation coverage"):
        relation_load(relation_fixture)


def test_native_rows_refuse_a_route_family_the_served_artifact_never_exercised(native_intake):
    """#570 leg (b) residual: byte coverage cannot see a route class that loads
    no library. The 2026-09-13 control's bf16 rows were admitted against
    engine-a5, which served a uniform-FP8 artifact and never ran a bf16 route;
    the served manifest must now carry every priced route's family, and the
    refusal names the family it is about."""
    evidence, table, relation = native_intake
    (evidence.root / "served-artifact" / "tessera_serving_manifest.json").write_text(json.dumps(
        {"modules": {"synthetic-module": {"family": "TESSERA_FP8"}}}))
    with pytest.raises(RuntimePriceError, match="TESSERA_BF16"):
        admit_native_rows(table, relation)


def test_native_rows_admit_a_served_manifest_spanning_every_priced_family(native_intake):
    """The same gate admits when the manifest exercises every priced route:
    the mixed artifact option A requires carries all three families in one
    manifest, and a row bound to any of them is a price for that serve."""
    evidence, table, relation = native_intake
    (evidence.root / "served-artifact" / "tessera_serving_manifest.json").write_text(json.dumps(
        {"modules": {name: {"family": family} for name, family in
                      (("bf16-module", "TESSERA_BF16"), ("fp8-module", "TESSERA_FP8"),
                       ("nvfp4-module", "TESSERA_NVFP4"))}}))
    admit_native_rows(table, relation)


def test_native_rows_refuse_an_unreadable_served_manifest(native_intake):
    """A manifest that cannot be read refuses rather than passing silently:
    an absent manifest is missing evidence, not an empty exercise roster."""
    evidence, table, relation = native_intake
    config = evidence.get(relation["record"]["configuration"])
    config["artifact"]["path"] = str(evidence.root / "no-such-artifact")
    evidence.replace(relation["record"]["configuration"], config)
    with pytest.raises(RuntimePriceError, match="cannot read"):
        admit_native_rows(table, relation)


@pytest.mark.parametrize("operator_route", [
    json.dumps({"symbol": "torch.mm"}),
    json.dumps({"policy": ""}),
    json.dumps({"policy": ":resident"}),
    "not json",
])
def test_a_binding_naming_no_route_family_is_not_a_served_price(operator_route):
    """The family is read off the binding's own declared route policy, never
    derived through a second mapping: a binding that names none is refused
    rather than assigned one."""
    from prismaquant.runtime_provenance import _require_served_route_family
    row = SimpleNamespace(unit="synthetic-unit",
                          binding=SimpleNamespace(as_dict=lambda: {"operator_route": operator_route}))
    with pytest.raises(RuntimePriceError, match="names no served route family"):
        _require_served_route_family(row, {"TESSERA_BF16"})
