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
import statistics
import tarfile
from pathlib import Path
from typing import Mapping

from .measured_runtime_prices import (
    OFF_STEP_FIELD, RuntimePriceError, _integer, _object,
    _sha, _string, identity_sha256, RankResources, RuntimeRankResources,
)
from .schemas import strict_json_loads

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
        return path, _strict_json(raw, path, where)


def _strict_json(raw, path, where):
    """Producer JSON with the same bar as every other artifact read here.

    Duplicate keys and nonfinite numbers are refused rather than resolved by
    last-wins or float parsing, so a manifest that carries a family's name
    twice or through a NaN cannot slip past the coverage check below.
    """
    try:
        value = strict_json_loads(
            raw, duplicate=lambda key: RuntimePriceError(f"{where}: duplicate JSON key {key!r}"),
            constant=lambda value: ValueError("nonfinite JSON number " + value))
    except (ValueError, UnicodeError) as exc:
        raise RuntimePriceError(f"{where}: invalid JSON artifact {path}: {exc}") from exc
    return _mapping(value, where)


def _source_digest(files):
    """Tessera's versioned source-byte seal, without importing its runtime."""
    digest = hashlib.sha256()
    for name in sorted(files, key=Path):
        raw = files[name]
        if Path(name).suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}:
            digest.update(name.encode() + b"\0" + raw + b"\0")
    return digest.hexdigest()


def _source_tree_identity(tree):
    """The source-tree installer's own identity, recomputed from the bytes.

    ``experiments/full_engine_plugin_install.py`` seals a source-tree install
    as a SHA-256 over the compact JSON map ``{archive member: sha256}`` of the
    build metadata plus every file under ``src/``, and records the map's size
    as ``plugin_source_members``. That installer emits no archive digest, so
    an archive-only binding refuses every source-tree install outright. This
    is Tessera's function recomputed here, the way :func:`_source_digest`
    already recomputes its source-byte seal: the producer's declared identity
    is checked against bytes this side holds, never accepted as stated.
    """
    members = {name: hashlib.sha256(raw).hexdigest() for name, raw in tree.items()}
    body = json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest(), len(members)


def _package_source(declaration, reader):
    _object(declaration, ("archive", "prefix", "excluded_files"), "package source")
    _, archive = reader.bytes(declaration["archive"], "original plugin source archive")
    prefix = _string(declaration["prefix"], "package archive prefix").rstrip("/") + "/"
    if Path(prefix).is_absolute() or ".." in Path(prefix).parts:
        raise RuntimePriceError("package archive prefix must be relative and normalized")
    files, tree = {}, {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as source:
            for member in source:
                if member.isdir():
                    continue
                name = member.name
                if (not member.isfile() or name in tree or Path(name).is_absolute()
                        or ".." in Path(name).parts or str(Path(name)) != name):
                    raise RuntimePriceError("package archive has duplicate, linked or unsafe source entries")
                tree[name] = source.extractfile(member).read()
                if name.startswith(prefix):
                    files[name[len(prefix):]] = tree[name]
    except tarfile.TarError as exc:
        raise RuntimePriceError(f"invalid plugin source archive: {exc}") from exc
    excluded = declaration["excluded_files"]
    if (not files or not isinstance(excluded, list) or any(not isinstance(name, str) for name in excluded)
            or len(set(excluded)) != len(excluded) or not set(excluded) <= set(files)):
        raise RuntimePriceError("package source needs an exact archive and explicit excluded file roster")
    installed = {name: raw for name, raw in files.items() if name not in excluded}
    source_identity_sha256, source_identity_members = _source_tree_identity(tree)
    return {"archive_sha256": declaration["archive"]["sha256"],
            "source_identity_sha256": source_identity_sha256,
            "source_identity_members": source_identity_members,
            "source_tree_sha256": _source_digest(files), "installed_source_sha256": _source_digest(installed),
            "installed_files": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                                for name, raw in installed.items()}}


def _installed_from_declared_source(installation, package_source):
    """Bind the installer's declared plugin source to the archive's bytes.

    Exactly one binding, never a choice the evidence can decline to make: an
    archive installer declares ``plugin_archive_sha256`` and a source-tree
    installer declares ``plugin_source_sha256`` with its member count. An
    installation that declares both, or neither, names no source.
    """
    archive = installation.get("plugin_archive_sha256")
    declared = installation.get("plugin_source_sha256")
    if (archive is None) == (declared is None):
        raise RuntimePriceError(
            "plugin installation must declare exactly one of plugin_archive_sha256 "
            "(archive install) or plugin_source_sha256 (source-tree install)")
    if archive is not None:
        _equal(archive, package_source["archive_sha256"], "original plugin archive bytes")
        return
    _equal(_sha(declared, "declared plugin source"), package_source["source_identity_sha256"],
           "recomputed plugin source-tree identity")
    _equal(_integer(installation["plugin_source_members"], "declared plugin source members", 1),
           package_source["source_identity_members"], "recomputed plugin source-tree member count")


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


#: Image manifest media types the loader accepts as the PINNED bytes.
#: A concrete platform manifest (what the pin named through contract v29),
#: or a multi-platform index/list (what the ``sm_121`` serve image pin has
#: been since the attested image became one: an OCI index whose digest never
#: equals any platform manifest's). PrismaQuant #723.
CONCRETE_IMAGE_MANIFEST_TYPES = (
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
)
INDEX_IMAGE_MANIFEST_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
)


def _resolve_image_index_entry(index, platform_digest, where):
    """The index entry the resolved platform manifest must be.

    The relation carries the platform manifest BYTES (content-addressed, so
    its digest is certain); this checks those bytes are what the pinned index
    points at for exactly one entry: the entry's digest names them, its
    media type is a concrete platform manifest and matches the bytes' own,
    and it carries the platform the entry was resolved for.  The executing
    host's architecture is not re-derived here -- no run record states it --
    so the entry's platform is RECORDED into the identity, never compared.
    A resolvable-but-wrong-arch entry fails at serve time, not at intake;
    an entry that is not in the index fails here.
    """
    entries = index.get("manifests")
    if not isinstance(entries, list) or not entries:
        raise RuntimePriceError(f"{where}: expected a non-empty manifests list")
    matches = [entry for entry in entries
               if isinstance(entry, Mapping)
               and entry.get("digest") == platform_digest]
    if len(matches) != 1:
        raise RuntimePriceError(
            f"{where}: expected exactly one entry for {platform_digest}, "
            f"found {len(matches)}")
    entry = matches[0]
    media = entry.get("mediaType")
    if media not in CONCRETE_IMAGE_MANIFEST_TYPES:
        raise RuntimePriceError(
            f"{where}: indexed entry {platform_digest} is not a concrete "
            f"platform manifest (mediaType {media!r})")
    platform = entry.get("platform")
    if not isinstance(platform, Mapping):
        raise RuntimePriceError(
            f"{where}: indexed entry {platform_digest} names no platform")
    architecture = platform.get("architecture")
    system = platform.get("os")
    if (not isinstance(architecture, str) or not architecture.strip()
            or not isinstance(system, str) or not system.strip()):
        raise RuntimePriceError(
            f"{where}: indexed entry {platform_digest} names no "
            "architecture/os platform")
    return {"mediaType": media, "architecture": architecture, "os": system}


def _image_local_ids(image_manifest):
    """The container IDs one pinned image may honestly report.

    Docker reports either the manifest it pulled or the config it runs;
    behind an index pull that is the pinned index digest, the resolved
    platform manifest's digest, or the config digest the platform manifest
    names.  Anything else is unrelated evidence and refuses.
    """
    identities = (image_manifest["manifest_digest"], image_manifest["config_digest"])
    if "platform_manifest_digest" in image_manifest:
        identities = (image_manifest["manifest_digest"],
                      image_manifest["platform_manifest_digest"],
                      image_manifest["config_digest"])
    return identities


def _observe_run(run, *, reader, configuration, configuration_sha256, image_manifest, package_source, context):
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
    # ``required`` is the reference the resolver's gate demanded, and it is the
    # field to read: Tessera's ``serving/runtime_image.resolve`` sets it to the
    # contract pin only when the requested repository IS the pinned repository
    # (``reason`` "pinned"), and to the explicitly requested digest otherwise
    # (``reason`` "explicit_digest"), leaving ``pinned`` naming Tessera's own
    # packaged default. Reading ``pinned`` demanded that every artifact serve
    # out of that default repository, which no pinned lane image does.
    for key in ("required", "resolved_reference", "requested"):
        _equal(declared[key], image, "declared image " + key)
    if declared["reason"] not in ("pinned", "explicit_digest"):
        raise RuntimePriceError("runtime image declaration was not gated on a digest")
    if declared["reason"] == "pinned":
        _equal(declared["pinned"], image, "declared image pinned")
    if image not in declared["repo_digests"]:
        raise RuntimePriceError("runtime image lacks actual RepoDigests evidence")
    # A native operator record carries no configuration of its own, so its
    # launcher stamps ``selection`` (``_pb_native_moe_measure/launch.py``) and
    # that stamp is the only binding. The full-engine record binds its own
    # configuration (``configuration_sha256``, checked above) and its capture
    # stamps no ``selection``; requiring one there refused every real report.
    selection = declared.get("selection")
    if selection is None:
        if run["scope"] != "full_engine":
            raise RuntimePriceError("native runtime image declaration requires its launcher selection")
    else:
        _equal(selection["configuration_sha256"], configuration_sha256, "launcher configuration")
    _, installation = reader.json(run["installation"], "runtime installation")
    _equal(installation["registry_base"], image, "installed image")
    _equal(installation["launcher_declared_image_id"], declared["local_id"], "actual image ID")
    if declared["local_id"] not in _image_local_ids(image_manifest):
        raise RuntimePriceError("actual image ID is none of the pinned image, its platform manifest, or its config digest")
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
    _installed_from_declared_source(installation, package_source)
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
        "plugin_archive_sha256": installation.get("plugin_archive_sha256"),
        "plugin_source_sha256": installation.get("plugin_source_sha256"),
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
    relation_fields = ("schema", "configuration", "image_manifest", "package_source", "runs", "full_engine_run_id",
                       "production_dependencies", "full_engine_extra_libraries")
    if "image_platform_manifest" in relation:
        relation_fields = relation_fields + ("image_platform_manifest",)
    _object(relation, relation_fields, "runtime relation")
    _equal(relation["schema"], SCHEMA, "runtime relation schema")
    _equal(identity_sha256(relation), context.runtime_sha256, "independent runtime derivation identity")
    _, configuration = reader.json(relation["configuration"], "selected serving configuration")
    configuration_sha256 = relation["configuration"]["sha256"]
    _, manifest = reader.json(relation["image_manifest"], "pinned image manifest")
    manifest_digest = "sha256:" + relation["image_manifest"]["sha256"]
    _equal(context.serving_context.runtime_image.rsplit("@", 1)[-1], manifest_digest, "pinned manifest bytes")
    _equal(manifest["schemaVersion"], 2, "image manifest schema")
    media = _string(manifest["mediaType"], "image manifest media type")
    if media in CONCRETE_IMAGE_MANIFEST_TYPES:
        if "image_platform_manifest" in relation:
            raise RuntimePriceError(
                "image provenance carries a platform manifest beside a concrete pinned manifest")
        config = _mapping(manifest.get("config"), "image config")
        config_digest = _string(config["digest"], "image config digest")
        image_manifest = {"manifest_digest": manifest_digest, "config_digest": config_digest}
    elif media in INDEX_IMAGE_MANIFEST_TYPES:
        if "image_platform_manifest" not in relation:
            raise RuntimePriceError(
                "image provenance pins an image index and names no platform manifest")
        _, platform = reader.json(relation["image_platform_manifest"], "pinned platform manifest")
        platform_digest = "sha256:" + relation["image_platform_manifest"]["sha256"]
        entry = _resolve_image_index_entry(manifest, platform_digest, "pinned image index")
        _equal(platform.get("mediaType"), entry["mediaType"], "resolved platform manifest media type")
        config = _mapping(platform.get("config"), "platform image config")
        config_digest = _string(config["digest"], "platform image config digest")
        image_manifest = {"manifest_digest": manifest_digest,
                          "platform_manifest_digest": platform_digest,
                          "platform_architecture": entry["architecture"],
                          "platform_os": entry["os"],
                          "config_digest": config_digest}
    else:
        raise RuntimePriceError(
            f"image provenance requires a concrete platform manifest or a multi-platform index, not {media!r}")
    if not image_manifest["config_digest"].startswith("sha256:"):
        raise RuntimePriceError("image config requires SHA-256 identity")
    _sha(image_manifest["config_digest"].removeprefix("sha256:"), "image config digest")
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
#: real report on a definitional gap or agree by coincidence. They are compared
#: under the versioned boundary instead (``transient_charge_boundary``, D37):
#: an identity of ownership on the one measured assignment, never a value.
FIXED_TERM_FIELDS = {"fixed_resident": "resident_bytes",
                     "fixed_activation": "activation_bytes",
                     "fixed_scratch": "peak_scratch_bytes",
                     "fixed_kv": "kv_bytes"}

#: Fields of the fixed charge this report schema carries no observation for at
#: all -- not a term that failed to recompute, but an axis the capture does not
#: observe. They are declared ``0`` and named as unevidenced, and the gate
#: refuses them by name below.
UNOBSERVED_FIXED_FIELDS = ("prefill_ms", "decode_ms", "serialized_bytes")


#: A sealed per-rank partition of a full-engine capture: the one document an
#: admitted per-rank fixed charge is read from. It asserts no charge of its own
#: -- each rank row references **that rank's own sealed capture** (whose run
#: identity names the rank, the world and the runtime), and the consumer
#: recomputes that rank's four fixed terms from it. The declared terms are a
#: claim to check against that recomputation, never a source: a partition that
#: merely redistributes a world total between ranks is refused term by term,
#: because the numbers it would redistribute do not come from either rank's own
#: evidence. The whole-engine report reference is an **optional** cross-check
#: (that the per-rank terms still sum to the world's own recomputed terms), and
#: it is never a substitute for the per-rank evidence.
RANK_PARTITION_SCHEMA = "prismaquant.full_engine_rank_partition.v1"
RANK_PARTITION_RULE = (
    "sealed_per_rank_captures_recomputed_terms_with_optional_whole_engine_cross_check")
_RANK_PARTITION_FIELDS = ("schema", "world_size", "rule", "full_engine_report", "ranks")
_RANK_PARTITION_RANK_FIELDS = ("rank", "world_size", "runtime_manifest_sha256",
                               "capture_sha256", "report", "terms")
#: Identity coordinates that belong to a *world*, not to one rank of it. Two
#: ranks of one world must name the same source model, the same measured
#: assignment, the same canonical unit roster, the same serving configuration,
#: the same runtime manifest and the same workload: a shared runtime digest
#: alone would let rank 0 observe a small model and rank 1 a different one under
#: one world number, and the charge would then price neither.
WORLD_IDENTITY_FIELDS = ("assignment_sha256", "canonical_units_sha256",
                         "configuration_sha256", "model_sha256",
                         "runtime_manifest_sha256", "workload_sha256")


@dataclass(frozen=True)
class RankFixedCharge:
    """The verdict of recomputing one per-rank fixed charge, and its evidence.

    Only this object may become an admitted ``RankDeviceBounds``
    (``measured_runtime_prices.RankDeviceBounds.recomputed``): the charge is a
    result here rather than a field a caller supplies beside a claim.
    """

    world_size: int
    charge_per_rank: tuple[int, ...]
    per_rank_terms: tuple
    evidence: Mapping
    partition_sha256: str


def recompute_rank_fixed_charge(reference, *, root, expected_run_identity=None,
                                where="rank fixed charge") -> RankFixedCharge:
    """Each rank's fixed charge, recomputed from **that rank's own** capture.

    Every rank row references its own sealed full-engine report, whose run
    identity names the rank, the world and the runtime manifest, and this
    consumer recomputes that rank's four fixed terms from that report's own
    observations. The terms the partition declares are a claim to check against
    the recomputation, not a source of it: an arbitrary redistribution of one
    world total between ranks -- which is what summing a scalar report and
    splitting it lets through -- is refused term by term, because neither
    rank's own report recomputes the number that rank was handed.

    The whole-engine report is optional and is only a cross-check: when the
    partition carries one, the per-rank terms must still sum, term by term, to
    the world's own recomputed terms, so a partition may allocate the fixed
    charge across ranks but may never create it. It is never a substitute for
    the per-rank evidence, which is why it is not required.

    Every rank's report must also name the *same world*: the source model, the
    measured assignment, the canonical unit roster, the serving configuration,
    the runtime manifest and the workload are world coordinates, and a shared
    runtime digest alone would let rank 0 observe a small model and rank 1 a
    different one under one world number. ``expected_run_identity`` additionally
    binds those coordinates to the consumer's own independently supplied ones --
    the table's context, where the existing scalar gate passes the relation's --
    so a charge measured on other bytes cannot price this table.

    A term no rank's capture can recompute refuses by name instead of being
    charged as zero.
    """
    from .full_engine_resource_report import (
        consume_full_engine_resource_report, read_full_engine_resource_report,
    )

    root = Path(root)
    reader = ArtifactReader(root)
    _, document = reader.json(reference, where)
    fields = _object(document, _RANK_PARTITION_FIELDS, where)
    if fields["schema"] != RANK_PARTITION_SCHEMA:
        raise RuntimePriceError(
            f"{where}: unknown per-rank partition schema {fields['schema']!r}")
    if fields["rule"] != RANK_PARTITION_RULE:
        raise RuntimePriceError(
            f"{where}: per-rank partition rule {fields['rule']!r} is not "
            f"{RANK_PARTITION_RULE!r}")
    world = _integer(fields["world_size"], where + " world size", 1)
    rows = fields["ranks"]
    if not isinstance(rows, list) or len(rows) != world:
        size = len(rows) if isinstance(rows, list) else "no"
        raise RuntimePriceError(
            f"{where}: a per-rank partition carries one record per rank: world size {world} "
            f"against {size} records")
    per_rank, sums, runtime = [], {term: 0 for term in FIXED_TERM_FIELDS}, None
    world_identity: dict = {}
    for expected_rank, row in enumerate(rows):
        row_where = f"{where} rank {expected_rank}"
        row = _object(row, _RANK_PARTITION_RANK_FIELDS, row_where)
        _equal(row["rank"], expected_rank, row_where + " rank")
        _equal(row["world_size"], world, row_where + " world size")
        row_runtime = _sha(row["runtime_manifest_sha256"], row_where + " runtime manifest")
        if runtime is None:
            runtime = row_runtime
        elif row_runtime != runtime:
            raise RuntimePriceError(
                f"{where}: ranks disagree about the runtime manifest they were measured "
                "under, so they are not one world's ranks")
        capture = _sha(row["capture_sha256"], row_where + " capture digest")
        report = read_full_engine_resource_report(row["report"], root=root)
        run = report["identity"]["run"]
        # The report must be *this rank's* own capture of *this* world's
        # runtime. A rank row that references a peer's report, a scalar report,
        # or a capture of another runtime refuses here rather than having its
        # declared terms read.
        _equal(run.get("rank"), expected_rank, row_where + " report rank")
        _equal(run.get("world_size"), world, row_where + " report world size")
        _equal(run["runtime_manifest_sha256"], runtime, row_where + " report runtime manifest")
        _equal(report["identity"]["capture_sha256"], capture, row_where + " report capture digest")
        # ...and of *one* world: every world coordinate is joined across ranks,
        # so two ranks cannot be two different models, assignments, unit
        # rosters, configurations or workloads wearing one world number.
        for key in WORLD_IDENTITY_FIELDS:
            first = world_identity.setdefault(key, run[key])
            if run[key] != first:
                raise RuntimePriceError(
                    f"{where}: rank {expected_rank} names {key} {run[key]} where rank 0 names "
                    f"{first}; the ranks of one world observe one model, assignment, unit "
                    "roster, configuration, runtime and workload")
        for key, expected in (expected_run_identity or {}).items():
            _equal(run.get(key), expected, f"{row_where} report {key}")
        verdict = consume_full_engine_resource_report(row["report"], root=root)
        terms = _object(row["terms"], tuple(FIXED_TERM_FIELDS), row_where + " terms")
        resolved = {}
        for term in FIXED_TERM_FIELDS:
            recomputed = verdict.recomputed_terms.get(term)
            if type(recomputed) is not int:
                refusals = "; ".join(verdict.refusals)
                raise RuntimePriceError(
                    f"{row_where}: this rank's own sealed report recomputes no {term}, so a "
                    f"per-rank fixed charge has no evidence for it"
                    + (f" ({refusals})" if refusals else ""))
            value = terms[term]
            if type(value) is not int or value < 0:
                raise RuntimePriceError(
                    f"{row_where}: {term} is {value!r}; an absent or non-integer term is a "
                    "missing charge rather than a zero one")
            if value != recomputed:
                raise RuntimePriceError(
                    f"{row_where}: the partition declares {term} {value} where this rank's own "
                    f"sealed report recomputes {recomputed}; a partition allocates the charge "
                    "across ranks from each rank's own capture and may not restate it")
            resolved[term] = recomputed
            sums[term] += recomputed
        per_rank.append(resolved)
    whole_reference = fields["full_engine_report"]
    if whole_reference is not None:
        whole = consume_full_engine_resource_report(whole_reference, root=root)
        for term in FIXED_TERM_FIELDS:
            recomputed = whole.recomputed_terms.get(term)
            if type(recomputed) is not int:
                raise RuntimePriceError(
                    f"{where}: the sealed whole-engine report cross-check recomputes no {term}, "
                    "so it cannot check the per-rank partition's own terms")
            if sums[term] != recomputed:
                raise RuntimePriceError(
                    f"{where}: the per-rank {term} sums to {sums[term]} where the sealed "
                    f"whole-engine report recomputes {recomputed}; a partition allocates the "
                    "charge across ranks and may not create it")
    return RankFixedCharge(world_size=world,
                           charge_per_rank=tuple(sum(terms.values()) for terms in per_rank),
                           per_rank_terms=tuple(per_rank),
                           evidence={"full_engine_report": whole_reference,
                                     "per_rank_partition": True},
                           partition_sha256=_sha(reference["sha256"], where + " partition digest"))


def recompute_fixed_resources(reference, *, root):
    """The fixed charge a table may declare, recomputed by the gate that admits it.

    An emitter calls this to fill ``fixed_resources``; ``admit_fixed_resources``
    then recomputes the same partition from the same report and refuses on any
    disagreement. Both sides run one implementation on purpose. A producer that
    recomputed the partition with a second reader of its own would be checked
    against a copy of itself, and the two readers would drift apart silently --
    which is the "``derived`` is a claim" failure one module further out. The
    emitter supplies the artifact reference; every number below is this
    consumer's own.

    Returns ``(declared, evidence, verdict)``: the fields a table declares, one
    evidence string per field naming the term it came from or why it has none,
    and the consumer's refusals and domain state for the emission report.
    """
    from .full_engine_resource_report import consume_full_engine_resource_report

    try:
        verdict = consume_full_engine_resource_report(dict(reference), root=root)
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
    return declared, evidence, {"refusals": list(verdict.refusals),
                                "expressible_terms": list(verdict.expressible_terms),
                                "open_domains": list(verdict.open_domains)}

#: What each observation the report names but leaves null costs this
#: admission. The absence is read from the report rather than assumed here, so
#: a capture that does supply one drops its entry and the terms it feeds become
#: reachable once their domain also closes. ``worker_startup_records`` and
#: ``kv_observations`` are deliberately absent: their shapes are defined now
#: (``full_engine_resource_report``), so the domains they close are the refusal
#: a capture without them receives, not a missing producer field.
OWED_EVIDENCE = {
    "owner_views": "the fixed member roster and each candidate's retained weights",
    "timing_captures": "the fixed prefill and decode charge",
    "observer_qualification": "the observer impact any timing charge needs",
    "runtime_provenance_relation": "the report's own binding to this relation",
}


def admit_fixed_resources(table, relation):
    """Admit fixed resources only from a partition this consumer recomputes.

    The obligation a placement has to satisfy is
    ``max(scalar_budget_bytes, non_step_transient_peak_bytes)``: the seven
    composition terms price one engine step, and a row live during none of them
    is priced beside them rather than inside them. Both sides are recomputed,
    and a report that expresses only one of them admits nothing.

    Tessera observes and derives; PrismaQuant recomputes and admits only on
    agreement. Nothing here reads the report's ``derived`` block: the
    recomputation in :mod:`prismaquant.full_engine_resource_report` runs over
    ``observations`` and ``partition``, and a producer total that differs from
    it arrives as one of the refusals collected below. Reading ``derived`` and
    believing it is the ``qualified: true`` failure the design forbids.

    Refusals are collected and raised together so one call names all of them.
    There is no admission token and no return value: the v2 loader sets
    ``fixed_resources_admitted`` only because this raised nothing.
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
        OWED_OBSERVATIONS, REFERENCE_BOUNDARY_FIELD, SUPPORTED_EXECUTION,
        consume_full_engine_resource_report, read_full_engine_resource_report,
    )
    from .transient_charge_boundary import (
        boundary_identity_refusals, require_boundary, reservation_slack_refusals,
        route_class_coverage_refusals,
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

    # The boundary both sides name, or the named reason there is none. Under
    # it, one measured assignment establishes the fixed charge for every
    # assignment whose formats lie in the route classes the run exercised
    # (route_class_coverage_refusals, below); without it, a multi-format menu
    # is refused through require_boundary, so no path admits one without v1.
    menu = {(row.unit, row.fmt): row for row in table.rows}
    table_units = sorted({row.unit for row in table.rows})
    boundary = None
    try:
        boundary = require_boundary(context.transient_charge_boundary,
                                    partition_schema=report["partition"]["schema"],
                                    report_boundary=report["reference"].get(REFERENCE_BOUNDARY_FIELD))
    except RuntimePriceError as exc:
        refusals.append(str(exc))

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
        else:
            # v2 carries it; this consumer defines no recomputation over it
            # yet, and a carried record nobody recomputes from is evidence of
            # nothing here (`full_engine_resource_report.OWED_OBSERVATIONS`).
            refusals.append(f"the capture carries {name}, and this consumer recomputes "
                            f"{OWED_EVIDENCE[name]} from no observation at "
                            f"{report['schema']}, so it has no evidence")

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
    # The placement obligation, not the per-step budget alone. A table admitted
    # on the smaller of the two numbers is admitted against a box that still has
    # to hold the larger one, and on unified memory that is an OOM rather than a
    # spill. Both sides move it: a larger per-step composition raises it, and so
    # does a larger off-step peak, which is why neither side is defaulted to
    # zero when it is not expressible.
    if verdict.recomputed_placement_obligation_bytes is None:
        if verdict.recomputed_non_step_transient_peak_bytes is None:
            refusals.append("no off-step transient peak is recomputable, so this report prices "
                            "nothing the engine holds while no engine step is running")
        refusals.append("no placement obligation is recomputable, so this table's fixed "
                        "resources are admitted against no device extent")
    # The off-step peak the table declares, against the one the report
    # recomputes. The gate demanded the obligation and nothing carried it, so
    # the DP pruned on the per-step composition alone; the table now declares
    # the other half and it is checked here like every other fixed term.
    declared_off_step = fixed.non_step_transient_peak_bytes
    recomputed_off_step = verdict.recomputed_non_step_transient_peak_bytes
    if declared_off_step is None:
        refusals.append(f"this table declares no {OFF_STEP_FIELD}, so its fixed resources price "
                        "nothing the engine holds while no engine step is running")
    elif recomputed_off_step is None:
        refusals.append(f"no off-step transient peak is recomputable, so this table's declared "
                        f"{OFF_STEP_FIELD} ({declared_off_step}) has no evidence")
    elif recomputed_off_step != declared_off_step:
        refusals.append(f"this table declares {OFF_STEP_FIELD} {declared_off_step} where the "
                        f"recomputed off-step transient peak is {recomputed_off_step}")

    resident = verdict.recomputed_terms["candidate_resident"]
    if resident is None:
        refusals.append("no candidate_resident is recomputable, so no priced row's resident "
                        "bytes has evidence")
    else:
        for unit, row in sorted(selected.items()):
            priced = menu.get((unit, row.get("format")))
            if priced is None:
                continue
            if isinstance(priced.resources, RuntimeRankResources):
                # The recomputed partition is one scalar per unit. A per-rank
                # row has one value per rank, and the reduction that would make
                # the two comparable is the reduction this table refuses.
                refusals.append(
                    f"unit {unit!r} prices its resources per rank, and the recomputed "
                    "candidate_resident is one scalar per unit")
            elif unit not in resident:
                refusals.append(f"the partition charges unit {unit!r} no resident bytes")
            elif resident[unit] != priced.resources.resident_bytes:
                refusals.append(f"this table declares {unit!r} resident bytes "
                                f"{priced.resources.resident_bytes} where the recomputed "
                                f"candidate_resident is {resident[unit]}")
    # The candidate transients, under the boundary: an identity of ownership
    # on the one measured assignment (design §2.3), the route-class scope of
    # the fixed charge (§4.3) and the reserved-extent witness (§3.7). None of
    # it compares a native peak to a partition term by value.
    if boundary is not None:
        refusals.extend(route_class_coverage_refusals(boundary, table, report=report,
                                                      selected_rows=selected, menu=menu))
        refusals.extend(boundary_identity_refusals(boundary, report, selected_rows=selected,
                                                   menu=menu))
        refusals.extend(reservation_slack_refusals(boundary, report))
    if fixed.serialized_bytes:
        refusals.append("the report partitions no serialized bytes, so this table's fixed "
                        f"serialized_bytes ({fixed.serialized_bytes}) has no evidence")
    for field in ("prefill_ms", "decode_ms"):
        value = getattr(fixed, field)
        if value:
            refusals.append("the report carries no timing partition, so this table's fixed "
                            f"{field} ({value}) has no evidence")
    return refusals


#: The latency scope a whole-owner receipt must declare for its samples to
#: price one apply of the module rather than a local matmul: the producer's own
#: spelling, checked here rather than summarized.
LATENCY_SCOPE_KIND = "one_whole_owner_apply"
#: Both halves of that claim, and the second half is the one that decides it:
#: the runtime's *declaration* (``runtime.collective``: which reduction, at
#: which callsite, required by this owner, never skipped) and the *count* of
#: calls at that callsite during each priced phase
#: (``collective_calls_per_phase``). The producer counts calls at the runner's
#: own imported symbol and derives ``includes_output_collective`` from that
#: count, so a declaration alone -- "the timed region includes a collective" --
#: is not evidence that one ran, and is refused here.
_LATENCY_SCOPE_FIELDS = ("kind", "per_rank", "includes_output_collective", "collective_callsite",
                         "collective_calls_per_phase", "collective_evidence", "collective",
                         "never")
PHASES = ("prefill", "decode")
#: The runtime's own final reduction, which a tensor-parallel owner's timed
#: region must contain. The producer refuses a config that skips it; this side
#: refuses a receipt that does not say so.
RUNTIME_COLLECTIVE_OP = "tensor_model_parallel_all_reduce"
#: The module whose attribute the producer's probe wraps, and the runner method
#: that calls it. The callsite string is spelled from that module rather than
#: from an import path a reader may remember, and it is pinned here too: a
#: receipt may not name a callsite other than the one this contract counts.
RUNTIME_COLLECTIVE_MODULE = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
RUNTIME_COLLECTIVE_METHOD = "_maybe_reduce_final_output"
RUNTIME_COLLECTIVE_SITE = f"{RUNTIME_COLLECTIVE_MODULE}:{RUNTIME_COLLECTIVE_METHOD}"
_COLLECTIVE_FIELDS = ("op", "site", "required_by_this_owner",
                      "runtime_declares_skip_final_all_reduce", "world_size")
#: One rank's resource identity as the producer publishes it: the record's own
#: digest plus the rank and world it belongs to. ``peers`` is that same record
#: for every other rank, gathered before the timed region.
_RANK_IDENTITY_FIELDS = ("rank", "world_size", "bound_sha256")


def routed_rank_bound(resources):
    """The digest a routed owner's own resource record carries.

    The producer computes it over ``receipt["resources"]`` *before* its roster
    is attached, so the record it describes has no ``self`` key and holds the
    ``peers: None`` the producer had written at that moment. Recomputing it
    here is what makes a rank's claim checkable by the rank that gathered it
    and by this consumer, which holds both records and may not take either
    rank's word for the other's bytes.
    """
    record = {key: value for key, value in resources.items() if key != "self"}
    record["peers"] = None
    return identity_sha256(record)


def _check_routed_scope(receipt, *, world_size, where):
    """The producer's own statement that these samples price one whole apply.

    The claim is accepted on the runtime's *counted* reduction, not on its
    declared configuration: ``collective_calls_per_phase`` is the number of
    times the runner's own reduction ran during each priced phase, and it must
    be exactly what this world needs -- once per phase at a world above one,
    never at a world of one. The declaration says which reduction and where, so
    a count of calls to *something else* cannot stand in for it.
    """
    scope = receipt.get("latency_scope")
    if not isinstance(scope, Mapping) or set(scope) != set(_LATENCY_SCOPE_FIELDS):
        raise RuntimePriceError(
            f"{where}: the routed receipt declares no whole-owner latency scope")
    if scope["kind"] != LATENCY_SCOPE_KIND or scope["per_rank"] is not True:
        raise RuntimePriceError(
            f"{where}: routed latency scope is {scope['kind']!r} per_rank "
            f"{scope['per_rank']!r}, not one whole-owner apply measured per rank")
    runtime = receipt.get("runtime")
    collective = runtime.get("collective") if isinstance(runtime, Mapping) else None
    if not isinstance(collective, Mapping) or set(collective) != set(_COLLECTIVE_FIELDS):
        raise RuntimePriceError(f"{where}: the routed runtime declares no output collective")
    required = world_size > 1
    if (collective["op"] != RUNTIME_COLLECTIVE_OP
            or collective["site"] != RUNTIME_COLLECTIVE_SITE
            or collective["runtime_declares_skip_final_all_reduce"] is not False
            or type(collective["required_by_this_owner"]) is not bool
            or collective["required_by_this_owner"] != required
            or type(collective["world_size"]) is not int or collective["world_size"] != world_size):
        raise RuntimePriceError(
            f"{where}: the routed runtime's collective is not this world's own final "
            "all-reduce inside the timed region")
    counts = scope["collective_calls_per_phase"]
    if not isinstance(counts, Mapping) or set(counts) != set(PHASES):
        raise RuntimePriceError(
            f"{where}: a whole-owner latency scope counts the runtime's own reduction once per "
            f"priced phase ({list(PHASES)}), and this one counts "
            f"{sorted(counts) if isinstance(counts, Mapping) else counts!r}")
    expected_calls = 1 if required else 0
    for phase in PHASES:
        count = counts[phase]
        if type(count) is not int or count != expected_calls:
            raise RuntimePriceError(
                f"{where}: {phase} priced the runtime's own output reduction {count!r} time(s), "
                f"and this owner needs exactly {expected_calls} at world size {world_size}")
    # The runtime's own `site` was pinned to this same constant above, so what
    # this binds is the site the calls were COUNTED at.
    _equal(scope["collective_callsite"], RUNTIME_COLLECTIVE_SITE,
           f"{where} latency scope callsite")
    # Read off the counted calls rather than off the declaration or the world:
    # "these samples include the output collective" is true when every priced
    # phase counted one, and false at a world of one, where nothing was reduced.
    observed_inclusion = all(counts[phase] == 1 for phase in PHASES)
    _equal(scope["includes_output_collective"], observed_inclusion,
           f"{where} latency scope includes_output_collective")
    _string(scope["collective_evidence"], f"{where} collective evidence")
    _equal(scope["collective"], collective["op"], f"{where} latency scope collective")
    _string(scope["never"], f"{where} latency scope never")


def _check_routed_rank_identity(receipt, *, rank, world_size, bounds, where):
    """One rank's record, checked against the ranks that gathered it."""
    resources = receipt.get("resources")
    if not isinstance(resources, Mapping):
        raise RuntimePriceError(f"{where}: the routed receipt records no resources")
    for key in ("rank", "world_size", "peers", "self"):
        if key not in resources:
            raise RuntimePriceError(
                f"{where}: a whole-owner resource claim at a world above one carries every "
                f"rank's own bound, and this record has no {key!r}")
    _equal(resources["rank"], rank, f"{where} resource rank")
    _equal(resources["world_size"], world_size, f"{where} resource world size")
    identity = resources["self"]
    if not isinstance(identity, Mapping) or set(identity) != set(_RANK_IDENTITY_FIELDS):
        raise RuntimePriceError(f"{where}: this rank's resource identity is not a rank record")
    _equal(identity, {"rank": rank, "world_size": world_size,
                      "bound_sha256": routed_rank_bound(resources)},
           f"{where} own resource identity")
    peers = resources["peers"]
    if not isinstance(peers, (list, tuple)):
        raise RuntimePriceError(f"{where}: a gathered peer roster must be an explicit list")
    for entry in peers:
        if not isinstance(entry, Mapping) or set(entry) != set(_RANK_IDENTITY_FIELDS):
            raise RuntimePriceError(f"{where}: a gathered peer record is not a rank identity")
    expected = [{"rank": other, "world_size": world_size, "bound_sha256": bounds[other]}
                for other in sorted(bounds) if other != rank]
    _equal(list(peers), expected, f"{where} gathered peer bounds")


def routed_owner_rank_resources(panel, ranks, *, where):
    """The per-rank resource vector one whole routed owner apply prices.

    ``ranks`` is one ``(rank, receipt, observation)`` per rank of the world the
    owner declares, and the observations are what
    ``native_moe_panel.consume_moe_receipt`` already admitted for each rank's
    own receipt. Nothing here reads a producer summary: each rank's bytes come
    from that rank's own observed ledger, each rank's own resource digest is
    recomputed, and every other rank's digest is checked against the one this
    rank's receipt gathered before it was allowed to time anything.

    The timing is one whole-owner apply per rank, each including the runtime's
    own final all-reduce, and the row prices the slowest rank's median while
    retaining every rank's. That is a maximum over ranks of one interval, not a
    sum of leaf timings and not a world-wide average.
    """
    world_size = _native_world_size(panel)
    roster = sorted(ranks, key=lambda item: item[0])
    covered = [rank for rank, _receipt, _observation in roster]
    if covered != list(range(world_size)):
        raise RuntimePriceError(
            f"{where}: a routed owner row needs every rank's own receipt, ranks 0..{world_size - 1} "
            f"once each and in order, and its receipts name {covered}")
    bounds = {}
    records, medians = [], {"prefill": [], "decode": []}
    for rank, receipt, observation in roster:
        rank_where = f"{where} rank {rank}"
        _check_routed_scope(receipt, world_size=world_size, where=rank_where)
        resources = receipt["resources"]
        bounds[rank] = routed_rank_bound(resources)
    for rank, receipt, observation in roster:
        rank_where = f"{where} rank {rank}"
        _check_routed_rank_identity(receipt, rank=rank, world_size=world_size, bounds=bounds,
                                   where=rank_where)
        scratch, activation = [], []
        for phase in ("prefill", "decode"):
            actual = observation["phases"][phase]
            if actual["peak_scratch_bytes"] is None:
                raise RuntimePriceError(f"{rank_where}: the routed row has an incomplete "
                                        "resource ledger")
            samples = actual["measurement"]["samples_ms"]
            if not isinstance(samples, (list, tuple)) or not samples:
                raise RuntimePriceError(f"{rank_where}: native {phase} measurement carries "
                                        "no samples")
            medians[phase].append(float(statistics.median(samples)))
            scratch.append(actual["peak_scratch_bytes"])
            activation.append(actual["input_bytes"])
        records.append(RankResources(
            rank=rank,
            resident_bytes=_integer(observation["resident_bytes"], f"{rank_where} resident bytes"),
            peak_scratch_bytes=max(scratch), activation_bytes=max(activation),
            workspace_resident_bytes=_integer(observation["workspace_resident_bytes"],
                                              f"{rank_where} workspace resident bytes"),
            workspace_sha256=_sha(observation["workspace_sha256"], f"{rank_where} workspace"),
            bound_sha256=bounds[rank]))
    # The wire extent belongs to the module, not to a rank: the producer frames
    # one canonical whole-module container and shards it locally, so every rank
    # holds a view of the same artifact. It is counted once, from the frozen
    # panel's own member wire records, and the digest binds those identities --
    # summing one rank's view per rank would double count the same bytes.
    wire_records = [member["wire"]["record"] for member in panel["members"]]
    wire_bytes = sum(_integer(member["wire"]["blob_bytes"], f"{where} member wire bytes")
                     for member in panel["members"])
    vector = RuntimeRankResources(
        prefill_ms=max(medians["prefill"]), decode_ms=max(medians["decode"]),
        world_size=world_size, rank_medians_ms=medians, wire_bytes=wire_bytes,
        wire_sha256=identity_sha256(wire_records), ranks=tuple(records))
    return vector.as_dict()


def routed_slowest_rank(vector, phase):
    """Which rank's median a routed row priced, and the samples to cite.

    Ties go to the lowest rank so one table always cites the same receipt for
    the same numbers.
    """
    medians = vector["rank_medians_ms"][phase]
    return medians.index(max(medians))


def _native_world_size(panel):
    """The world a native panel was measured in, from its own two records.

    ``runtime.execution.tensor_parallel`` is the box's world; a routed owner's
    ``execution.tensor_parallel`` is the world its priced member shapes were cut
    for. They are different facts, and a panel stating two different worlds is
    refused rather than resolved in whichever direction a reader checks first.
    """
    runtime = panel["runtime"]
    execution = runtime.get("execution") if isinstance(runtime, Mapping) else None
    if not isinstance(execution, Mapping) or "tensor_parallel" not in execution:
        raise RuntimePriceError("native runtime record names no execution world size")
    world = execution["tensor_parallel"]
    owner = panel.get("execution")
    if isinstance(owner, Mapping) and "tensor_parallel" in owner:
        _equal(owner["tensor_parallel"], world, "native owner/runtime world size")
    return world


def _served_artifact_families(relation):
    """Route families the full-engine run's served artifact exercises.

    D39 leg (b) residual (#570): byte coverage cannot see a route class that
    loads no library -- the bf16 rows of the 2026-09-13 control were admitted
    against engine-a5, which served a uniform-FP8 artifact and never ran a
    bf16 route. The relation's configuration names the served artifact, so its
    serving manifest must carry every priced route's family.

    The manifest is read, not bound: the configuration bytes naming it are
    digest-bound, and this check only adds refusals -- a manifest that listed
    a family the serve never exercised would degrade to today's byte-coverage
    behavior, never weaker. A missing configuration field, a missing or
    unreadable manifest, or a module naming no family refuses rather than
    passing silently.
    """
    record, reader = relation["record"], relation["reader"]
    _, configuration = reader.json(record["configuration"], "selected serving configuration")
    artifact = configuration.get("artifact")
    if not isinstance(artifact, Mapping):
        raise RuntimePriceError("selected serving configuration names no served artifact")
    path = artifact.get("path")
    if not isinstance(path, str) or not path.strip():
        raise RuntimePriceError("selected serving configuration names no served artifact path")
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = reader.root / manifest_path
    manifest_path = manifest_path / "tessera_serving_manifest.json"
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise RuntimePriceError(
            f"served artifact manifest: cannot read artifact {manifest_path}: {exc}") from exc
    manifest = _strict_json(raw, manifest_path, "served artifact manifest")
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping) or not modules:
        raise RuntimePriceError("served artifact manifest names no modules, so it exercises no route family")
    served = set()
    for name, module in modules.items():
        family = _mapping(module, f"served artifact module {name!r}").get("family")
        if not isinstance(family, str) or not family.strip():
            raise RuntimePriceError(f"served artifact module {name!r} names no route family")
        served.add(family)
    return served


def _require_served_route_family(row, served):
    """A row may only price a route family the served artifact exercised.

    The family is read off the row binding's own declared route policy
    (`TESSERA_NVFP4:resident`, `TESSERA_FP8:resident`, ...) -- the contract's
    route vocabulary, shared with the manifest's per-module `family` field --
    never derived through a second mapping in this repo. A binding that names
    no family, or one the manifest does not carry, is a price for a serve
    nobody ran and is refused by name.
    """
    try:
        route = json.loads(row.binding.as_dict()["operator_route"])
    except (ValueError, TypeError, KeyError) as exc:
        raise RuntimePriceError(
            f"native row {row.unit} names no served route family in its binding operator_route") from exc
    policy = route.get("policy") if isinstance(route, Mapping) else None
    family = policy.split(":")[0] if isinstance(policy, str) else ""
    if not family:
        raise RuntimePriceError(
            f"native row {row.unit} names no served route family in its binding operator_route")
    if family not in served:
        raise RuntimePriceError(
            f"native row {row.unit} prices route family {family!r}, which the served artifact "
            f"never exercised (manifest families: {sorted(served)})")


def admit_native_rows(table, relation):
    """Reuse exact same-panel producer gates before accepting v2 table rows."""
    from .native_moe_panel import consume_moe_receipt
    from .native_operator_panel import consume_native_receipt, operator_route_identity
    reader = ArtifactReader(Path(table.source_path).parent)
    bindings = table.native_receipt_bindings
    if not isinstance(bindings, (list, tuple)):
        raise RuntimePriceError("native receipt bindings must be an explicit list")
    by_key = {}
    for item in bindings:
        # A dense or single-device row prices one process, so it has no peer
        # roster and keeps the binding shape it has always had. A ranked row
        # carries every other rank's receipt, and a ranked row without one is
        # refused rather than priced from the one rank that happened to write.
        fields = ("unit", "format", "run_id", "panel", "receipt", "memory_trace")
        if "peer_receipts" in item:
            fields += ("peer_receipts",)
        _object(item, fields, "native receipt binding")
        key = item["unit"], item["format"]
        if key in by_key:
            raise RuntimePriceError("duplicate native row receipt binding")
        by_key[key] = item
    _equal(set(by_key), {row.key for row in table.rows}, "native row receipt coverage")
    cohort_panels = []
    for row in table.rows:
        binding = by_key[row.key]
        run_id = binding["run_id"]
        if run_id not in relation["runs"] or run_id == relation["full_engine_run_id"]:
            raise RuntimePriceError("native row requires its original native runtime")
        _, panel = reader.json(binding["panel"], "independent native panel")
        cohort_panels.append(panel)
        receipt_path, receipt = reader.json(binding["receipt"], "native receipt")
        from .native_execution_binding import resolve_native_receipt_view
        receipt = resolve_native_receipt_view(receipt, panel)
        trace_path, _ = reader.bytes(binding["memory_trace"], "native memory trace")
        run = relation["runs"][run_id]
        _equal(panel["runtime"], run["raw"], "original native panel runtime")
        _equal(panel["cost_sha256"], table.cost_sha256, "native panel cost payload")
        _equal(panel["source_sha256"], table.context.source_sha256, "native source model")
        _equal(panel["calibration_sha256"], table.context.calibration_sha256, "native calibration")
        # The priced world is a coordinate of the row, not a label: member
        # shapes, per-rank bytes and the timing that includes the collectives
        # all belong to the world the panel was measured in.
        _equal(_native_world_size(panel), table.context.tensor_parallel, "native panel world size")
        if table.context.batch_size != 1:
            raise RuntimePriceError("native panel admission currently requires batch size one")
        if panel["schema"] == "tessera.native_moe_panel.v1":
            _equal(run["raw"]["schema"], "tessera.native_moe_runtime.v1", "native routed runtime scope")
            _equal(panel["serving_config_sha256"], relation["configuration_sha256"], "native panel configuration")
            wire_records = [member["wire"]["record"] for member in panel["members"]]
            consume = consume_moe_receipt
            expected_binding = panel["runtime_binding"]
            # A world above one is priced from every rank's own receipt, so the
            # binding roster is read as a roster: each peer entry's receipt is
            # rehashed, must carry the same frozen panel, and must declare the
            # rank the roster places it at.
            roster = [(receipt["resources"]["rank"], receipt_path, receipt, trace_path,
                       binding["receipt"]["sha256"])]
            for peer in binding.get("peer_receipts", ()):
                _object(peer, ("rank", "receipt", "memory_trace"), "native peer receipt binding")
                peer_path, peer_receipt = reader.json(peer["receipt"], "native peer receipt")
                peer_receipt = resolve_native_receipt_view(peer_receipt, panel)
                peer_trace, _ = reader.bytes(peer["memory_trace"], "native peer memory trace")
                _equal(peer_receipt["panel"], panel, "native peer receipt panel")
                _equal(peer_receipt["resources"]["rank"], peer["rank"], "native peer receipt rank")
                roster.append((peer["rank"], peer_path, peer_receipt, peer_trace,
                               peer["receipt"]["sha256"]))
            roster.sort(key=lambda entry: entry[0])
        elif panel["schema"] == "tessera.native_dense_panel.v1":
            _equal(run["raw"]["schema"], "tessera.native_dense_runtime.v1", "native dense runtime scope")
            wire_records = [panel["wire"]["record"]]
            consume = consume_native_receipt
            roster = [(0, receipt_path, receipt, trace_path, binding["receipt"]["sha256"])]
            expected_binding = {"member_formats": {panel["unit"]: panel["format"]},
                "member_operator_identity_sha256": {panel["unit"]: panel["joint_operator_identity_sha256"]},
                "member_shapes": {panel["unit"]: panel["shape"]},
                "operator_route": operator_route_identity(panel["phases"]["prefill"]["expected_route"])}
        else:
            raise RuntimePriceError("unsupported native producer panel")
        cited_paths = {rank: path for rank, path, _receipt, _trace, _sha in roster}
        cited_paths_sha256 = {rank: sha for rank, _path, _receipt, _trace, sha in roster}
        for record in wire_records:
            _equal(record["identity"]["encoder_source_sha256"],
                   run["common"]["producer_source_tree_sha256"], "original wire producer source-tree seal")
        observations = {}
        for rank, rank_path, _rank_receipt, rank_trace, rank_sha256 in roster:
            try:
                observations[rank] = consume(rank_path, expected_sha256=rank_sha256,
                                             expected_panel=panel, memory_trace_path=rank_trace)
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimePriceError(f"native producer admission refused: {exc}") from exc
        observation = observations[roster[0][0]]
        _equal(observation["unit"], row.unit, "native row unit")
        _equal(observation["format"], row.fmt, "native row format")
        _equal(expected_binding, row.binding.as_dict(), "native row operator binding")
        ranked = isinstance(row.resources, RuntimeRankResources)
        if ranked:
            # The row's per-rank vector is re-derived here from every rank's own
            # receipt, not re-read from the row: the emitter used this same
            # function to fill it, so what this compares is two emissions of one
            # implementation rather than a number that travelled and was trusted.
            _equal(row.resources.as_dict(),
                   routed_owner_rank_resources(
                       panel,
                       [(rank, rank_receipt, observations[rank])
                        for rank, _path, rank_receipt, _trace, _sha in roster],
                       where=f"native row {row.unit}"),
                   "native row per-rank resources")
        else:
            _equal(observation["serialized_unit_bytes"], row.resources.serialized_bytes, "native serialized bytes")
            _equal(observation["resident_bytes"], row.resources.resident_bytes, "native resident bytes")
        scratch, activation = [], []
        for phase, measurement in (("prefill", row.prefill), ("decode", row.decode)):
            if measurement is None:
                raise RuntimePriceError("native v2 row lacks a complete measured phase")
            # The row cites the receipt its own samples came from: at a world
            # above one that is the slowest rank's, because that is the median
            # the row priced.
            cited = roster[0][0] if not ranked else routed_slowest_rank(
                row.resources.as_dict(), phase)
            actual = observations[cited]["phases"][phase]
            if actual["peak_scratch_bytes"] is None:
                raise RuntimePriceError("native row has an incomplete resource ledger")
            expected_path, _ = reader.bytes({"path": measurement.receipt_path, "sha256": measurement.receipt_sha256}, "table native phase receipt")
            _equal(expected_path.resolve(), cited_paths[cited].resolve(), "native phase receipt path")
            _equal(measurement.receipt_sha256, cited_paths_sha256[cited], "native phase receipt bytes")
            for key in ("method", "samples_ms", "warmup_iterations"):
                _equal(actual["measurement"][key], measurement.as_dict()[key], "native phase samples")
            _equal(panel["phases"][phase]["m"], table.context.prompt_tokens if phase == "prefill" else 1, "native phase token scope")
            scratch.append(actual["peak_scratch_bytes"]); activation.append(actual["input_bytes"])
            if not ranked and row.resources.output_bytes is not None:
                # A row that names its returned output names the observation's,
                # phase by phase; the escape check reads it as a witness.
                _equal(row.resources.output_bytes.get(phase), actual["output_bytes"],
                       f"native {phase} returned output bytes")
        if not ranked:
            _equal(row.resources.peak_scratch_bytes, max(scratch), "native maximum phase scratch")
            _equal(row.resources.activation_bytes, max(activation), "native maximum phase input residency")
    if getattr(table.context, "native_cohort", None) is not None:
        from .native_runtime_cohort import bind_cohort
        _equal(bind_cohort(cohort_panels), table.context.native_cohort,
               "actual native operator contexts and shared runtime cohort")
    # D39 leg (b) residual (#570): byte coverage cannot see a route class that
    # loads no library, so the served artifact's manifest must carry every
    # priced route's family. Read-only -- this adds refusals, never admission.
    served = _served_artifact_families(relation)
    for row in table.rows:
        _require_served_route_family(row, served)


def admit_runtime_provenance(table):
    """The v2 loader calls this after its ordinary raw-receipt hash checks.

    Two gates, two answers, and only one of them is fatal to the table. The
    relation and the native rows attest the per-row prices the DP consumes: if
    those do not hold, the table prices nothing and the load fails. The
    fixed-resource gate attests a different object -- the whole-engine charge
    added once outside the DP -- so its refusal is returned rather than raised,
    and the consumer that reads `fixed_resources` spends it. Folding the two
    into one raise discarded a native attestation that had passed.

    Returns the fixed-resource refusal text, or ``None`` when that gate passed.
    """
    try:
        relation = load_runtime_relation(table.runtime_provenance, context=table.context,
                                         root=Path(table.source_path).parent)
        admit_native_rows(table, relation)
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimePriceError(f"runtime producer evidence is missing or malformed: {exc}") from exc
    try:
        admit_fixed_resources(table, relation)
    except (KeyError, TypeError, IndexError) as exc:
        return f"runtime producer evidence is missing or malformed: {exc}"
    except RuntimePriceError as exc:
        return str(exc)
    return None
