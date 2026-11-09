"""Run an admitted campaign quantum in its declared rootless Podman container.

The native Flux job owns placement and its CPU allocation. This adapter
authenticates that admission, maps the sealed checkout and explicit data
mounts into one owned container, and cleans up only the container it created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys

import runpy

# Authenticate before importing PQ: its initializer needs the producer stack,
# which the small host interpreter deliberately does not install (#601).
_runtime_identity = runpy.run_path(str(
    Path(__file__).resolve().parents[1] / "prismaquant/container_runtime_identity.py"))
# The spec contract planners import too: stdlib-only package knowledge, loaded
# by path for the same host-side reason as the identity helper above.
_spec_contract = runpy.run_path(str(
    Path(__file__).resolve().parents[1] / "prismaquant/campaign_container_spec.py"))
SAFE_PATH_ENV = _spec_contract["SAFE_PATH_ENV"]
CHECKOUT_COMMIT_ENV = _spec_contract["CHECKOUT_COMMIT_ENV"]
BOUNDED_CAPTURE_ENV = _spec_contract["BOUNDED_CAPTURE_ENV"]
GPU_RUNTIME_FLAGS = _spec_contract["GPU_RUNTIME_FLAGS"]
DEFAULT_GPU_RUNTIME = _spec_contract["DEFAULT_GPU_RUNTIME"]
container_memory_budget_gb = _spec_contract["container_memory_budget_gb"]
validate_container = _spec_contract["validate_container"]

from tools.tessera_campaign_namespace import (
    establish_namespace_temporaries, namespace_adapter_request,
    refuse_path_symlinks as _refuse_scratch_symlinks,
    validate_namespace_request,
)


#: The transitional residency map name (deleted by unit 4d with the resolver
#: map): where this action's declared inputs were staged.  The consumer half is
#: ``prismaquant.residency_map`` (``ENV_VAR`` there); the literal is repeated
#: here so the host adapter imports no package.  ``podman create``
#: inherits nothing from the launcher, so a map that stops at this boundary
#: leaves the resolver inside inert and every read on the pool -- which is
#: exactly what the GLM-5.3-Flash joint run did for its whole life on
#: 2026-09-18 (PQ #732): the launcher held the variable, its argv named it
#: nowhere.
RESIDENCY_MAP_ENV = "PRISMABUILD_RESIDENCY_MAP"
#: The key of the map whose value is the directory every staged copy lives
#: under, bound read-only into the container at the same path.
STAGE_ROOT_KEY = "stage_root"


#: The owner's rapid-iteration switch (Rob, 2026-09-19): when the submitting
#: environment holds ``PRISMAQUANT_DEV_MODE=1``, the provenance gates inside
#: the container -- the joint pass's source-proof, prepared-digest and
#: checkpoint-lineage gates -- stamp and record instead of refusing. The
#: submitter (``dispatch_tessera_campaign._submit_gpu_action``) seals it into
#: the spec env it hands this launcher, which is exactly how
#: ``PRISMAQUANT_LAYER_READ_THREADS`` travels: a hand-sealed spec that
#: declares it reaches the container the same way, with no dispatch-side
#: merge. The launcher itself neither requires nor interprets the value; the
#: gates in ``prismaquant.dev_mode`` do. The name is held together with
#: ``prismaquant.dev_mode.DEV_MODE_ENV`` by
#: ``tests/test_dev_mode_provenance_gates.py``.
DEV_MODE_ENV = "PRISMAQUANT_DEV_MODE"


def _read_residency_map(path: str) -> dict:
    """The map's JSON object, or raise ``RuntimeError`` with the reason."""

    try:
        with open(path) as stream:
            payload = json.load(stream)
    except OSError as exc:
        raise RuntimeError(
            f"the PrismaBuild residency map {path} cannot be read on this host "
            f"({exc.strerror}); the launcher named it, so this row's staged inputs "
            "cannot be located and it would read the pool for its whole life") from exc
    except ValueError as exc:
        raise RuntimeError(
            f"the PrismaBuild residency map {path} is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"the PrismaBuild residency map {path} is not a JSON object")
    return payload


def _stage_root_mounted(stage_root: str) -> bool:
    """Whether the stage root is a mounted directory here, triggering autofs.

    The Sparks reach ``/stage/prewarm`` through an automount; binding an
    untriggered placeholder hands the container an empty directory and every
    staged read falls back to the pool with the map looking perfectly healthy.
    Listing the directory triggers the mount; ``ismount`` is the fact.
    """

    try:
        os.listdir(stage_root)
    except OSError:
        return False
    return os.path.ismount(stage_root)


def residency_environment(spec: dict, environ) -> "tuple[dict, list[dict]]":
    """The residency map this container reads, and the stage mount it needs.

    Returns the environment to forward and the mounts to add.  Both are empty
    when the launcher named no map: an action outside PrismaBuild's residency
    contract keeps a byte-identical ``podman create``.

    Refused at launch, never dropped, when the map is not inside a declared
    mount (the resolver inside opens it by this same path), when it cannot be
    read or names no absolute ``stage_root``, or when that root is not a
    mounted directory on this host.  A row whose stage is invisible reads the
    pool for hours while its receipts say ``resident``; Rob's ruling for the
    campaign is stage-fed only, so a launch that cannot be is a refusal, like
    the progress channel's.  Per-file misses stay the resolver's own fallback.
    """

    named = environ.get(RESIDENCY_MAP_ENV)
    if not named:
        return {}, []
    map_path = PurePosixPath(named)
    if not map_path.is_absolute():
        raise RuntimeError(
            f"the PrismaBuild residency map path {named!r} is not absolute")
    mounts = spec["container"].get("mounts", [])
    inside = None
    for mount in mounts:
        source = PurePosixPath(mount["source"])
        if map_path == source or source in map_path.parents:
            inside = PurePosixPath(mount["target"]) / map_path.relative_to(source)
            break
    if inside is None:
        raise RuntimeError(
            f"the PrismaBuild residency map {named} is not inside any declared "
            "container mount, so the resolver inside the container could not open "
            "it and every staged input would be read from the pool; declare a mount "
            "covering it")
    payload = _read_residency_map(named)
    stage_root = payload.get(STAGE_ROOT_KEY)
    if not isinstance(stage_root, str) or not PurePosixPath(stage_root).is_absolute():
        raise RuntimeError(
            f"the PrismaBuild residency map {named} names no absolute "
            f"{STAGE_ROOT_KEY!r}, so the staged copies it points at cannot be "
            "mounted into the container")
    if not _stage_root_mounted(stage_root):
        raise RuntimeError(
            f"the stage root {stage_root} named by the PrismaBuild residency map "
            "is not a mounted directory on this host; binding it would hand the "
            "container an empty directory and every staged read would fall back "
            "to the pool")
    root = PurePosixPath(stage_root)
    extra: list[dict] = []
    if not any(PurePosixPath(m["target"]) == root or PurePosixPath(m["target"]) in root.parents
               for m in mounts):
        extra.append({"source": stage_root, "target": stage_root, "readonly": True})
    return {RESIDENCY_MAP_ENV: str(inside)}, extra


def _canonical_absolute(value: object) -> "str | None":
    """The value when it is a canonical absolute POSIX path, else ``None``.

    Same spelling rule as container mount sources and targets: absolute,
    normalized, no parent references, no NUL byte.
    """

    if (not isinstance(value, str) or not value.startswith("/")
            or "\x00" in value or ".." in PurePosixPath(value).parts
            or str(PurePosixPath(value)) != value):
        return None
    return value


COTANGENT_SCRATCH_ENV = ("PRISMAQUANT_STAGE_B_COTANGENT_ROOT",
                         "PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES")

PRODUCED_SPOOL_ENV = ("PRISMABUILD_PRODUCED_SPOOL_ROOT",
                      "PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES")

#: The Stage B one-pass replay spill (PQ #994); the names
#: ``prismaquant.joint_replay_spill.SPILL_ENV`` reads.
STAGE_B_SPILL_ENV = ("PRISMAQUANT_STAGE_B_SPILL_ROOT",
                     "PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES")


def cotangent_scratch_environment(spec: dict, environ) -> dict:
    """Forward only an explicitly sealed, identity-mounted local workspace."""
    return _bounded_local_environment(spec, environ, COTANGENT_SCRATCH_ENV,
                                      "cotangent scratch")


def stage_b_spill_environment(spec: dict, environ) -> dict:
    """Forward the Stage B replay spill through a writable identity bind.

    Declared exactly like the cotangent scratch. The root must also not be
    under ``/mnt/shared``: the spill is local-NVMe scratch, never NFS and
    never the ZFS pool behind it. The process refuses any other non-local
    filesystem when it opens the scratch.
    """
    forwarded = _bounded_local_environment(spec, environ, STAGE_B_SPILL_ENV,
                                           "Stage B spill")
    root = forwarded.get(STAGE_B_SPILL_ENV[0])
    if root is not None and PurePosixPath("/mnt/shared") in PurePosixPath(root).parents:
        raise RuntimeError("Stage B spill root must not be under /mnt/shared")
    return forwarded


def produced_spool_environment(spec: dict, environ) -> dict:
    """Forward the sealed local-spool declaration through an identity bind.

    The source-host exporter runs outside the container. A remapped pathname
    would name different bytes there, so the most specific covering mount must
    expose the same host path read-write. No host directory is mounted here.
    """
    return _bounded_local_environment(spec, environ, PRODUCED_SPOOL_ENV,
                                      "local output spool")


#: PrismaBuild's sealed list of an action's bounded-local pairs (PB #911):
#: ``ROOT_ENV:MAX_ENV`` items separated by commas. pbrun charges each pair's
#: ceiling, rounded up to whole GiB, to the executing box's ``spool_gb``.
LOCAL_SCRATCH_PAIRS_ENV = "PRISMABUILD_LOCAL_SCRATCH_PAIRS"
LOCAL_SCRATCH_IO_ENV = "PRISMABUILD_LOCAL_SCRATCH_IO"

#: Every bounded-local scratch kind a campaign container may write on the
#: executing box's disk, in the order its pair is listed. A kind added here
#: is forwarded and priced by the same call (``local_scratch_environment``).
#: The produced spool is not listed: PrismaBuild charges its window through
#: ``PRISMABUILD_PRODUCED_SPOOL_HOST_WINDOW`` and refuses its pair in the list.
CONTAINER_CACHE_SCRATCH_ENV = (
    "PRISMAQUANT_CONTAINER_CACHE_ROOT", "PRISMAQUANT_CONTAINER_CACHE_MAX_BYTES")


def container_cache_scratch_environment(spec: dict, environ) -> dict:
    """Validate an explicitly priced compilation-cache root, not its lifetime."""
    names = CONTAINER_CACHE_SCRATCH_ENV
    if not any(name in environ or name in spec.get("env", {}) for name in names):
        return {}
    maximum = environ.get(names[1])
    if (not isinstance(maximum, str) or re.fullmatch(r"[0-9]+", maximum) is None
            or int(maximum) <= 0):
        raise RuntimeError("container cache needs a sealed positive byte ceiling")
    root = _canonical_absolute(environ.get(names[0]))
    if root is not None:
        cache = PurePosixPath(root)
        for other_names, _ in LOCAL_SCRATCH_KINDS:
            other = environ.get(other_names[0])
            if other_names == names or _canonical_absolute(other) is None:
                continue
            path = PurePosixPath(other)
            if cache == path or cache in path.parents or path in cache.parents:
                raise RuntimeError("container cache root must be disjoint from other scratch")
    declared = _bounded_local_environment(spec, environ, names, "container cache")
    _refuse_scratch_symlinks(declared[names[0]])
    return declared


LOCAL_SCRATCH_KINDS = (
    (COTANGENT_SCRATCH_ENV, cotangent_scratch_environment),
    (STAGE_B_SPILL_ENV, stage_b_spill_environment),
    (CONTAINER_CACHE_SCRATCH_ENV, container_cache_scratch_environment),
)


def local_scratch_environment(spec: dict, environ) -> dict:
    """Every declared bounded-local scratch pair, plus PrismaBuild's pair list.

    Each kind in :data:`LOCAL_SCRATCH_KINDS` is validated by its own
    function, so a root without a sealed positive byte ceiling is refused
    here, as it is at launch. The pairs that are declared are listed in
    :data:`LOCAL_SCRATCH_PAIRS_ENV`, so PrismaBuild charges every byte
    bound it forwards. With no scratch declared this is ``{}`` and the
    sealed request carries nothing new.
    """

    if LOCAL_SCRATCH_PAIRS_ENV in spec.get("env", {}):
        raise RuntimeError(
            f"spec env {LOCAL_SCRATCH_PAIRS_ENV} is derived from the declared "
            "scratch kinds, not declared by a spec")
    forwarded: dict = {}
    pairs = []
    for names, environment in LOCAL_SCRATCH_KINDS:
        declared = environment(spec, environ)
        if declared:
            forwarded.update(declared)
            pairs.append(names)
    if not pairs:
        return {}
    roots = [forwarded[root] for root, _ in pairs]
    if len(set(roots)) != len(roots):
        raise RuntimeError(
            "two bounded-local scratch kinds name the same root; each needs "
            "its own root so PrismaBuild can charge each ceiling")
    forwarded[LOCAL_SCRATCH_PAIRS_ENV] = ",".join(
        f"{root}:{maximum}" for root, maximum in pairs)
    return forwarded


def local_scratch_request_environment(spec: dict) -> dict:
    """Project sealed spec declarations into the outer PB admission envelope.

    Unlike payload forwarding, admission never consults the coordinator's
    environment. Keep preflight and envelope construction on this same path.
    """
    forwarded = local_scratch_environment(spec, spec.get("env", {}))
    forwarded.update(local_scratch_io_environment(spec, forwarded))
    return forwarded


def local_scratch_io_environment(spec: dict, scratch: dict) -> dict:
    """Validate PB v1 traffic independently of occupancy; preserve wire bytes.

    ``scratch`` is the complete validated capacity projection, not a selected
    root. PB owns profile comparison, freshness at placement and scheduling.
    """
    env = spec.get("env", {})
    if LOCAL_SCRATCH_IO_ENV not in env:
        return {}

    def refuse(reason):
        raise ValueError(f"{LOCAL_SCRATCH_IO_ENV}: {reason}")

    def unique_object(items):
        result = {}
        for key, value in items:
            if key in result:
                refuse(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def nonfinite(value):
        refuse(f"nonfinite JSON value {value}")

    wire = env[LOCAL_SCRATCH_IO_ENV]
    if not isinstance(wire, str) or not wire:
        refuse("scratch I/O declaration must be a nonempty JSON string")
    try:
        intent = json.loads(wire, object_pairs_hook=unique_object,
                            parse_constant=nonfinite)
    except ValueError as exc:
        refuse(f"invalid scratch I/O JSON: {exc}")
    fields = {"schema", "write_bytes", "read_bytes", "profile_contract",
              "max_profile_age_s"}
    if not isinstance(intent, dict) or set(intent) != fields:
        refuse("scratch I/O declaration requires exactly the PB v1 fields")
    if intent["schema"] != "prismabuild.local_scratch_io.v1":
        refuse("unsupported scratch I/O schema")
    for direction in ("write_bytes", "read_bytes"):
        if type(intent[direction]) is not int or intent[direction] < 0:
            refuse(f"{direction} must be a nonnegative JSON integer")
    if not (intent["write_bytes"] or intent["read_bytes"]):
        refuse("at least one traffic direction must be positive")
    contract = intent["profile_contract"]
    if not isinstance(contract, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", contract) is None:
        refuse("profile_contract must be a nonempty token [A-Za-z0-9_.-]+")
    age = intent["max_profile_age_s"]
    # PB checks finiteness as a float, including for JSON integer ages.
    # Unrepresentable integers must refuse, not escape preflight as overflow.
    try:
        valid_age = (type(age) in (int, float) and age > 0
                     and math.isfinite(float(age)))
    except OverflowError:
        valid_age = False
    if not valid_age:
        refuse("max_profile_age_s must be a positive finite JSON number")
    pairs = scratch.get(LOCAL_SCRATCH_PAIRS_ENV, "").split(",")
    if len(pairs) != 1 or not pairs[0]:
        refuse("scratch I/O requires exactly one scratch pair")
    return {LOCAL_SCRATCH_IO_ENV: wire}


def _forwarded_local_scratch_environment(spec: dict, environ) -> dict:
    """Forward every registered pair, never the outer admission control."""
    forwarded = local_scratch_environment(spec, environ)
    forwarded.pop(LOCAL_SCRATCH_PAIRS_ENV, None)
    return forwarded


#: The container's write caches and PrismaQuant's temp parent, each with the
#: subdirectory it gets under a declared scratch root (PQ #1072, #1014 item f).
#: Left unset, each one lands in the container's writable overlay, which is
#: bounded by nothing and invisible to PrismaBuild.
CONTAINER_CACHE_ENV = (
    ("HF_HOME", "hf"),
    ("TRITON_CACHE_DIR", "triton"),
    ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
    ("XDG_CACHE_HOME", "xdg"),
    ("PRISMAQUANT_TMPDIR", "tmp"),
)
#: The directory under the scratch root that holds the caches.
CONTAINER_CACHE_DIRNAME = "container-cache"
#: Container paths that are always the overlay, whatever the spec mounts.
_OVERLAY_TEMP_ROOTS = (PurePosixPath("/tmp"), PurePosixPath("/var/tmp"))


def _on_writable_mount(spec: dict, value: str) -> bool:
    path = PurePosixPath(value)
    if any(path == root or root in path.parents for root in _OVERLAY_TEMP_ROOTS):
        return False
    covering = [mount for mount in spec.get("container", {}).get("mounts", [])
                if (PurePosixPath(mount["target"]) == path
                    or PurePosixPath(mount["target"]) in path.parents)]
    if not covering:
        return False
    nearest = max(covering, key=lambda item: len(PurePosixPath(item["target"]).parts))
    return not nearest.get("readonly", False)


def container_cache_environment(spec: dict, scratch: dict) -> "tuple[dict, list[str]]":
    """Default cache roots under the declared scratch, and the overlay pins.

    ``scratch`` is the forwarded bounded-local environment
    (:func:`local_scratch_environment` or the per-kind calls). For legacy
    declarations without a separate compilation-cache pair, each variable
    left unset gets ``<root>/container-cache/<subdir>``. That root is the
    first declared legacy kind: cotangent scratch, then Stage B spill. It sits on a writable
    identity bind, which the kind's own check has already required.

    A variable the spec sets keeps its value, so the default never overrides
    a declaration. The second return value names each variable the spec
    points at ``/tmp``, ``/var/tmp`` or a path that no writable mount
    covers, that is, the overlay. The launcher only lists them in its
    preamble; the joint dispatcher's spec check refuses them unless the spec
    names an ``overlay_cache_reason`` (PQ #1129).

    An explicit compilation-cache pair instead selects its own charged root.
    Four cache variables must stay there; TMPDIR must explicitly name another
    charged workspace. This declaration reserves bytes, not a filesystem quota
    or durable cleanup. Without it, the legacy defaults below are unchanged.
    """

    declared = spec.get("env", {}) if isinstance(spec, dict) else {}
    pinned = [name for name, _ in CONTAINER_CACHE_ENV
              if isinstance(declared.get(name), str)
              and not _on_writable_mount(spec, declared[name])]
    cache_root = scratch.get(CONTAINER_CACHE_SCRATCH_ENV[0])
    if cache_root:
        cache = PurePosixPath(cache_root)
        defaults = {}
        for name, subdir in CONTAINER_CACHE_ENV:
            if name == "PRISMAQUANT_TMPDIR":
                value = _canonical_absolute(declared.get(name))
                if value is None:
                    raise RuntimeError("TMPDIR needs an explicit separate charged workspace")
                workspaces = [PurePosixPath(scratch[names[0]])
                              for names, _ in LOCAL_SCRATCH_KINDS
                              if names != CONTAINER_CACHE_SCRATCH_ENV
                              and scratch.get(names[0])]
                path = PurePosixPath(value)
                if (not _on_writable_mount(spec, value)
                        or path == cache or cache in path.parents
                        or not any(path == root or root in path.parents
                                   for root in workspaces)):
                    raise RuntimeError("TMPDIR needs an explicit separate charged workspace")
            else:
                value = _canonical_absolute(declared.get(name, str(cache / subdir)))
                if value is None:
                    raise RuntimeError(f"{name} must stay within the charged cache root")
                path = PurePosixPath(value)
                if (not _on_writable_mount(spec, value)
                        or not (path == cache or cache in path.parents)):
                    raise RuntimeError(f"{name} must stay within the charged cache root")
                if name not in declared:
                    defaults[name] = value
            _refuse_scratch_symlinks(value)
        return defaults, []
    root = next((scratch[names[0]] for names, _ in LOCAL_SCRATCH_KINDS
                 if scratch.get(names[0])), None)
    if root is None:
        return {}, pinned
    base = PurePosixPath(root) / CONTAINER_CACHE_DIRNAME
    defaults = {name: str(base / subdir) for name, subdir in CONTAINER_CACHE_ENV
                if name not in declared}
    return defaults, pinned


def _bounded_local_environment(spec, environ, names, label):
    declared = spec.get("env", {})
    if not any(environ.get(name) or declared.get(name) for name in names):
        return {}
    root, maximum = (environ.get(name) for name in names)
    if _canonical_absolute(root) is None or root in ("/", "/mnt/shared"):
        raise RuntimeError(f"{label} needs a sealed canonical absolute root")
    if not isinstance(maximum, str) or not maximum.isdecimal() or int(maximum) <= 0:
        raise RuntimeError(f"{label} needs a sealed positive byte ceiling")
    for name, value in zip(names, (root, maximum)):
        if name in declared and declared[name] != value:
            raise RuntimeError(f"{label} spec disagrees with the sealed launch")
    path = PurePosixPath(root)
    candidates = [mount for mount in spec["container"].get("mounts", [])
                  if (PurePosixPath(mount["target"]) == path
                      or PurePosixPath(mount["target"]) in path.parents)]
    if not candidates:
        raise RuntimeError(f"{label} requires an explicit writable identity bind")
    mount = max(candidates, key=lambda item: len(PurePosixPath(item["target"]).parts))
    translated = PurePosixPath(mount["source"]) / path.relative_to(mount["target"])
    if str(translated) != root or mount.get("readonly", False):
        raise RuntimeError(f"{label} requires an explicit writable identity bind")
    # A nested bind can replace individual payload groups after the root
    # check. Refuse it rather than sealing ambiguous source-host paths.
    if any(path in PurePosixPath(mount["target"]).parents
           for mount in spec["container"].get("mounts", [])):
        raise RuntimeError(f"{label} cannot contain nested container mounts")
    return dict(zip(names, (root, maximum)))


def host_path(container_path: str, *, cwd: str, mounts: list) -> "Path | None":
    """The host path the container engine binds behind one absolute container path.

    The sealed checkout is bound at ``/workspace``; every other visible path
    comes from a declared mount. The longest matching target wins, so a
    ``/producer/src`` entry resolves through a ``/producer`` mount.
    """

    target = PurePosixPath(container_path)
    if not target.is_absolute():
        return None
    best: "tuple[int, Path] | None" = None
    candidates = [(PurePosixPath("/workspace"), Path(cwd))]
    candidates += [(PurePosixPath(mount["target"]), Path(mount["source"]))
                   for mount in mounts]
    for prefix, source in candidates:
        if target != prefix and prefix not in target.parents:
            continue
        remainder = target.parts[len(prefix.parts):]
        if best is None or len(prefix.parts) > best[0]:
            best = (len(prefix.parts), source.joinpath(*remainder))
    return None if best is None else best[1]


def import_search_roots(spec: dict, *, cwd: str, safe_path: bool) -> list:
    """The host directories the launched ``python -m`` searches, in order.

    ``sys.path[0]`` is the working directory unless safe-path mode is active;
    the ``PYTHONPATH`` entries follow it. An empty or relative entry means the
    working directory, which is why safe-path mode alone does not decide the
    question: a ``.`` written into ``PYTHONPATH`` reaches the same tree.
    Entries that name nothing this launcher can see are dropped, since Python
    would find no package there either.
    """

    mounts = spec.get("container", {}).get("mounts", [])
    entries = [] if safe_path else [""]
    raw = spec.get("env", {}).get("PYTHONPATH", "")
    entries += [entry for entry in raw.split(":")] if raw else []
    roots = []
    for entry in entries:
        resolved = (Path(cwd) if entry in ("", ".")
                    else host_path(entry, cwd=cwd, mounts=mounts))
        if resolved is not None:
            roots.append((entry, resolved))
    return roots


def _source_package(root: Path) -> bool:
    """The supported source-package candidate at one mounted search entry."""
    return (root / "prismaquant" / "__init__.py").is_file()


def _native_import_candidates(directory: Path, stem: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir()
                  if path.name == stem + ".so"
                  or (path.name.startswith(stem + ".") and path.name.endswith(".so")))


def _refuse_import_shadow(root: Path) -> None:
    """Conservative opt-in candidates, never an interpreter/ABI simulation.

    A source package wins over same-entry source/sourceless modules. Native
    package initializers may supersede its __init__.py; any native-looking
    initializer is ambiguous without the container's suffix authority.
    """
    try:
        package = root / "prismaquant"
        native_initializers = _native_import_candidates(package, "__init__")
        candidates = [(path, "ambiguous native initializer")
                      for path in native_initializers]
        if not _source_package(root):
            candidates += [(package / "__init__.pyc", "sourceless package shadow"),
                           (root / "prismaquant.py", "source module shadow"),
                           (root / "prismaquant.pyc", "sourceless module shadow")]
            candidates += [(path, "ambiguous native module shadow")
                           for path in _native_import_candidates(root, "prismaquant")]
        for path, reason in candidates:
            if path.is_file():
                raise RuntimeError(f"namespace guarded imports {reason}: {path}")
    except OSError as exc:
        raise RuntimeError(f"namespace guarded imports cannot inspect root: {root}") from exc


def _package_root(roots: list, *, refuse_shadows: bool = False) -> tuple[str, Path] | None:
    for entry, root in roots:
        if refuse_shadows:
            _refuse_import_shadow(root)
        if _source_package(root):
            return entry, root
    return None


def guarded_import_root(spec: dict, *, cwd: str,
                        require_checkout: bool = False) -> tuple[str, Path] | None:
    """Replay safe-path imports; opt-in ownership requires known checkout resolution."""
    roots = import_search_roots(spec, cwd=cwd, safe_path=True)
    guarded = _package_root(roots, refuse_shadows=require_checkout)
    if require_checkout:
        if guarded is None or guarded[1].resolve() != Path(cwd).resolve():
            raise RuntimeError("namespace guarded imports do not resolve to the executed checkout")
        known_entries = {entry for entry, _ in roots}
        for entry in spec.get("env", {}).get("PYTHONPATH", "").split(":"):
            if entry not in known_entries:
                raise RuntimeError("namespace guarded imports have an unknown earlier root")
            if entry == guarded[0]:
                break
    return guarded


def pinned_source_root(spec: dict, *, cwd: str) -> "tuple[str | None, Path, bool]":
    """The PrismaQuant tree this launch is expected to run, and how it was chosen.

    Returns the entry that declared it, its host path, and whether it was
    defaulted. A declared tree is the first ``PYTHONPATH`` entry that names a
    mount the spec declares and that holds a PrismaQuant package;
    ``/workspace`` is the sealed checkout, not a declared mount, so an entry
    resolving into it is not a candidate.

    When no declared mount holds a PrismaQuant package there is nothing for
    the sealed checkout to shadow, so the checkout is the tree to run and is
    returned with ``pinned_by_default`` set. Every container ``PYTHONPATH``
    recorded in this repository has that shape: the 2026-09-08 census
    invocations name ``/workspace`` and then Tessera source trees, which hold
    no ``prismaquant`` package. Refusing them would refuse the launch shape
    the campaign actually uses.
    """

    mounts = spec.get("container", {}).get("mounts", [])
    workspace = PurePosixPath("/workspace")
    raw = spec.get("env", {}).get("PYTHONPATH", "")
    for entry in raw.split(":") if raw else []:
        target = PurePosixPath(entry)
        if not target.is_absolute() or target == workspace or workspace in target.parents:
            continue
        root = host_path(entry, cwd=cwd, mounts=mounts)
        if root is not None and _source_package(root):
            return entry, root, False
    return None, Path(cwd), True


def verify_pinned_import(spec: dict, *, cwd: str) -> dict:
    """Refuse a launch whose import would resolve outside the pinned mount.

    111 completed rows of ``extension-r1024-02`` executed the sealed checkout
    rather than the pinned tree, because ``python -m`` puts the working
    directory ahead of every ``PYTHONPATH`` entry and the working directory
    carries its own ``prismaquant`` package (#519). The guard the launcher now
    sets removes that entry; this replays the interpreter's search rules over
    the launched environment and working directory and compares what would be
    imported against the pinned mount, byte for byte, using the same package
    digest the row stamps as ``prismaquant_source_sha256``.

    The launcher runs before the container, so these digests are a prediction
    from the launched environment, not an observation of the executed process.
    The row's own stamped digest remains the observation, and the two agreeing
    is what closes the loop.

    The refusal is narrow on purpose. It fires when a declared mount holds a
    PrismaQuant package and the import resolves to something else, which is
    #519 exactly: the reseal named a tree and the row ran another. It does not
    fire when no declared mount holds a PrismaQuant package, because nothing
    is being shadowed -- the sealed checkout is the only PrismaQuant there is,
    and its digest already enters the action key. That case is not silent: the
    receipt names the checkout as ``pinned_source_root`` and sets
    ``pinned_by_default``, so a reader or a later gate can tell a defaulted
    root from a declared one and catch an operator who meant to pin a tree and
    mistyped the path. The launcher states the fact and does not guess intent.

    The question only arises for a launch that can import PrismaQuant at all.
    When the guarded search reaches no package, the guard has already removed
    the working directory from the search, so there is no tree to shadow and
    nothing pinned to compare against; the launch proceeds and the receipt
    records that nothing was pinned. One route stays outside the replay either
    way: a ``PYTHONPATH`` entry that exists only inside the image, such as a
    pip-installed package under ``dist-packages``, maps to no declared mount,
    and the row's stamped digest is what catches that after the fact.
    """

    guarded = guarded_import_root(spec, cwd=cwd)
    unguarded = _package_root(import_search_roots(spec, cwd=cwd, safe_path=False))
    shadow_sha = (None if unguarded is None
                  else _runtime_identity["prismaquant_source_sha256"](unguarded[1] / "prismaquant"))
    if guarded is None:
        return {"pinned_source_entry": None, "pinned_source_root": None,
                "pinned_source_sha256": None,
                "pinned_by_default": False,
                "import_resolution_source_sha256": None,
                "import_resolution_root": None,
                "working_directory_source_sha256": shadow_sha,
                "safe_path_guard_is_load_bearing": shadow_sha is not None}
    entry, pinned, by_default = pinned_source_root(spec, cwd=cwd)
    pinned_sha = _runtime_identity["prismaquant_source_sha256"](pinned / "prismaquant")
    resolved_sha = _runtime_identity["prismaquant_source_sha256"](guarded[1] / "prismaquant")
    if resolved_sha != pinned_sha:
        raise RuntimeError(
            "the launched environment imports PrismaQuant from "
            f"{guarded[1]} ({resolved_sha}), not from the pinned mount "
            f"{entry} -> {pinned} ({pinned_sha}); a PYTHONPATH entry ahead of "
            "the pinned mount reaches another tree, and safe-path mode does "
            "not remove it")
    return {"pinned_source_entry": entry, "pinned_source_root": str(pinned),
            "pinned_source_sha256": pinned_sha,
            "pinned_by_default": by_default,
            "import_resolution_source_sha256": resolved_sha,
            "import_resolution_root": str(guarded[1]),
            "working_directory_source_sha256": shadow_sha,
            "safe_path_guard_is_load_bearing": shadow_sha != pinned_sha}


PODMAN = "/usr/bin/podman"
ADMISSION_SCHEMA = "prismaquant.campaign_container_admission.v1"
EXECUTION_SCHEMA = "prismaquant.campaign_container_execution.v1"
#: Exit codes owned by this adapter. A payload's own return code passes through;
#: these name a clean payload whose container cleanup failed, and a container
#: that could not be created or inspected.
EXIT_CLEANUP_FAILED = 91
EXIT_CREATE_FAILED = 92
#: The only ambient names this adapter reads. Nothing else crosses from the
#: launching process into the container; a spec's own ``env`` is application data.
AMBIENT_SETTINGS = ("FLUX_JOB_ID", "OUT", "TMPDIR", RESIDENCY_MAP_ENV,
                    *COTANGENT_SCRATCH_ENV, *STAGE_B_SPILL_ENV,
                    *CONTAINER_CACHE_SCRATCH_ENV, *PRODUCED_SPOOL_ENV)
_CID = re.compile(r"[0-9a-f]{64}")


def native_admission(environ) -> dict:
    """Authenticate the running native job; refuse any launch outside one.

    The saved launch, original jobspec, event log, cgroup scope and R allocation
    are checked by ``native_results.execution_context`` (through
    ``flux_jobs.verify_current_job``). Nothing here is read from a PB context.
    """
    job_id, out = environ.get("FLUX_JOB_ID"), environ.get("OUT")
    if not job_id or not out:
        raise RuntimeError("a container launch requires an admitted native job "
                           "(FLUX_JOB_ID and OUT); an unadmitted launch is refused")
    if os.environ.get("FLUX_JOB_ID") != job_id or os.environ.get("OUT") != out:
        raise RuntimeError("the launch environment differs from the running native job")
    from prismaquant import native_results
    context = native_results.execution_context()
    from prismaquant.digests import bytes_sha256hex
    raw = (Path(context["record_dir"]) / "launch.json").read_bytes()
    if bytes_sha256hex(raw) != context["launch_sha256"]:
        raise RuntimeError("the saved native launch changed during admission")
    saved_profile = json.loads(raw)["profile"]
    resources = context["resources"]
    return {"schema": ADMISSION_SCHEMA, "job_id": context["job_id"],
            "launch_sha256": context["launch_sha256"], "source": context["source"],
            "record_dir": context["record_dir"],
            "affinity": sorted(os.sched_getaffinity(0)),
            "cores": resources["cores"], "memory_gib": resources["memory_gib"],
            "gpus": resources["gpus"], "walltime_s": resources["walltime_s"],
            "profile": {key: saved_profile[key]
                        for key in ("profile", "config_sha256", "model_sha256")}}


def _require_admission(admission: dict) -> None:
    fields = {"schema", "job_id", "launch_sha256", "source", "record_dir", "affinity",
              "cores", "memory_gib", "gpus", "walltime_s", "profile"}
    if (not isinstance(admission, dict) or set(admission) != fields
            or admission["schema"] != ADMISSION_SCHEMA
            or not isinstance(admission["affinity"], list) or not admission["affinity"]
            or any(type(cpu) is not int or cpu < 0 for cpu in admission["affinity"])
            or admission["affinity"] != sorted(set(admission["affinity"]))):
        raise RuntimeError("container admission is missing or malformed")


def gpu_attachment(admission: dict) -> tuple:
    """Whether to attach a GPU, decided only by the native admission.

    A CPU admission withholds every device. A GPU admission refuses: the GPU
    arm is held for unit 4E, so this adapter never attaches a device. An empty
    ``CUDA_VISIBLE_DEVICES`` is not admission and is not consulted.
    """
    _require_admission(admission)
    if admission["gpus"] == 0:
        return False, "native admission reserved no GPU"
    raise RuntimeError("the container GPU arm is held for unit 4E; "
                       "native admission reserved a GPU and this adapter attaches none")


def podman_command(spec: dict, command: list[str], *, cwd: str, image_id: str,
                   admission: dict, cidfile, content_sha256=None, environ=None,
                   bounded=False, checkout_commit=None) -> list[str]:
    validate_container(spec, bounded=bounded)
    gpu_attachment(admission)
    environ = environ if environ is not None else {}
    budget_gb = container_memory_budget_gb(spec)
    if budget_gb is not None and budget_gb > admission["memory_gib"]:
        raise RuntimeError(
            f"the container memory cap {budget_gb:g} GiB exceeds the admitted "
            f"{admission['memory_gib']} GiB; refusing rather than running a larger cap than was reserved")
    argv = [PODMAN, "create", "--pull=never", "--cidfile", str(cidfile),
            "--cgroups=disabled", "--network=none", "--env-host=false",
            "--cpuset-cpus", ",".join(map(str, admission["affinity"])),
            "--entrypoint", "", "--workdir", "/workspace"]
    if os.uname().machine == "aarch64":
        argv += ["--userns=keep-id", "--user=1000:1000"]
    else:
        argv += ["--user=0:0"]
    argv += ["--mount", f"type=bind,src={cwd},dst=/workspace,readonly"]
    residency_env, residency_mounts = residency_environment(spec, environ)
    for mount in [*spec["container"].get("mounts", []), *residency_mounts]:
        value = f"type=bind,src={mount['source']},dst={mount['target']}"
        if mount.get("readonly", False):
            value += ",readonly"
        argv += ["--mount", value]
    bounded_defaults = BOUNDED_CAPTURE_ENV if bounded else {}
    scratch = _forwarded_local_scratch_environment(spec, environ)
    cache_defaults, _pinned = container_cache_environment(spec, scratch)
    forwarded = {SAFE_PATH_ENV: "1", **cache_defaults, **spec.get("env", {}),
                 **bounded_defaults, **residency_env, **scratch,
                 **produced_spool_environment(spec, environ)}
    for key, value in sorted(forwarded.items()):
        argv += ["--env", f"{key}={value}"]
    if content_sha256 is not None:
        argv += ["--env", "PRISMAQUANT_CONTAINER_CONTENT_SHA256=" + content_sha256]
    if checkout_commit is not None:
        argv += ["--env", f"{CHECKOUT_COMMIT_ENV}={checkout_commit}"]
    return [*argv, image_id, *command]


def _save_record(path: Path, record: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(record, indent=1) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _cleanup_container(record: dict) -> None:
    """Reap only the captured CID, keeping every error even when removal fails."""
    path = Path(record["cidfile"])
    record["removed"] = False
    try:
        if path.is_symlink():
            raise ValueError("owned cidfile replaced by symlink")
        cid = path.read_text().strip() if path.exists() else record.get("cid")
        if cid is None:
            raise ValueError("container creation scope unknown: no captured CID")
        if _CID.fullmatch(cid) is None:
            raise ValueError("invalid captured CID")
        if record.get("cid") not in (None, cid):
            raise ValueError("owned cidfile replaced")
        record["cid"] = cid
        for args in (["stop", "--time", "2", cid], ["rm", "--force", cid]):
            done = subprocess.run([PODMAN, *args], capture_output=True, text=True, timeout=15)
            if done.returncode:
                record["cleanup_errors"].append(args[0] + ": " + done.stderr.strip())
        gone = subprocess.run([PODMAN, "container", "exists", cid], capture_output=True, timeout=15)
        record["removed"] = gone.returncode == 1
        if gone.returncode not in (0, 1):
            raise ValueError("container absence query failed")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        record["cleanup_errors"].append(str(exc))


def run_owned_container(spec: dict, command: list[str], *, cwd: str, image_id: str,
                        content_sha256, admission: dict, record_dir, environ,
                        bounded=False, checkout_commit=None, image=None) -> int:
    """Create, run and reap one container owned by captured CID (the G01 controls)."""
    _require_admission(admission)
    record_dir = Path(record_dir)
    cidfile, path = record_dir / "container.cid", record_dir / "container.json"
    if path.exists() or path.is_symlink() or cidfile.exists() or cidfile.is_symlink():
        raise RuntimeError("container record already exists; refusing to reuse an owned handle")
    image = image or {}
    record = {"schema": EXECUTION_SCHEMA, "job_id": admission["job_id"],
              "launch_sha256": admission["launch_sha256"], "source": admission["source"],
              "image": {"Id": image_id, "Architecture": image.get("Architecture"),
                        "Os": image.get("Os"), "content_sha256": content_sha256},
              "cidfile": str(cidfile), "cid": None, "affinity": admission["affinity"],
              "memory_gib": admission["memory_gib"], "gpus": admission["gpus"],
              "creation_returncode": None, "inspection": None, "returncode": None,
              "interrupted_signal": None, "cleanup_errors": [], "removed": False}
    _save_record(path, record)  # the cidfile handle is registered before the first create

    class Interrupted(BaseException):
        pass

    def interrupt(signum, frame):
        record["interrupted_signal"] = signum
        raise Interrupted

    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    attached = None
    code = EXIT_CREATE_FAILED
    try:
        argv = podman_command(spec, command, cwd=cwd, image_id=image_id, admission=admission,
                              cidfile=cidfile, content_sha256=content_sha256, environ=environ,
                              bounded=bounded, checkout_commit=checkout_commit)
        made = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        record["creation_returncode"] = made.returncode
        (record_dir / "container.create.stdout").write_text(made.stdout)
        (record_dir / "container.create.stderr").write_text(made.stderr)
        if made.returncode:
            raise RuntimeError("rootless container create failed: " + made.stderr.strip())
        cid = cidfile.read_text().strip()
        if _CID.fullmatch(cid) is None:
            raise RuntimeError("create returned no captured CID")
        record["cid"] = cid
        inspected = subprocess.run([PODMAN, "inspect", cid], capture_output=True, text=True, timeout=30)
        if inspected.returncode:
            raise RuntimeError("created container inspect failed")
        actual, = json.loads(inspected.stdout)
        host = actual["HostConfig"]
        record["inspection"] = {"Id": actual["Id"], "Image": actual["Image"],
                                "CpusetCpus": host["CpusetCpus"],
                                "Devices": host.get("Devices") or []}
        if (actual["Id"] != cid
                or actual["Image"].removeprefix("sha256:") != image_id.removeprefix("sha256:")
                or host["CpusetCpus"] != ",".join(map(str, admission["affinity"]))
                or record["inspection"]["Devices"]):
            raise RuntimeError("actual container id, image, cpuset or devices differ from admission")
        _save_record(path, record)
        attached = subprocess.Popen([PODMAN, "start", "--attach", cid])
        record["returncode"] = attached.wait()
        code = record["returncode"]
    except Interrupted:
        code = 128 + record["interrupted_signal"]
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        record["cleanup_errors"].append("launch: " + str(exc))
        code = EXIT_CREATE_FAILED
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if attached is not None and attached.poll() is None:
                attached.terminate()
                try:
                    attached.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    attached.kill()
                    attached.wait(timeout=3)
            if record["cid"] is not None or cidfile.exists():
                _cleanup_container(record)
            else:
                record["removed"] = True  # nothing was ever created
        except (OSError, subprocess.SubprocessError) as exc:
            record["cleanup_errors"].append(str(exc))
        finally:
            _save_record(path, record)
            from prismaquant.digests import bytes_sha256hex
            digest = bytes_sha256hex(path.read_bytes())
            print(f"campaign container evidence {path} sha256={digest}", flush=True)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    clean = record["removed"] and not [e for e in record["cleanup_errors"] if not e.startswith("launch: ")]
    if code == 0 and not clean:
        return EXIT_CLEANUP_FAILED
    return code


def checkout_commit(cwd: str) -> str | None:
    """The commit of the git checkout at ``cwd``, or None when it is not one.

    Read on the host, where git exists, for a container that has none. A
    checkout whose tracked ``prismaquant/`` bytes differ from HEAD is refused:
    the variable this value is handed over as also disables the joint pass's
    own exactness check, so the launcher is where that check now happens, and
    a commit the package bytes do not match must not be stamped on a checkpoint.
    Untracked files are not part of a commit and are not compared here; the
    package digest the pass stamps (``prismaquant_source_sha256``) covers them.
    """
    root = Path(cwd)
    if not (root / ".git").exists():
        return None

    def _git(*args: str) -> str:
        done = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                              text=True, timeout=60)
        if done.returncode != 0:
            raise RuntimeError(
                f"the working directory {root} has a .git entry but git cannot read "
                f"it ({' '.join(args)}: {done.stderr.strip() or done.returncode})")
        return done.stdout.strip()

    commit = _git("rev-parse", "HEAD").lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError(f"the working directory {root} resolves to no commit: {commit!r}")
    changed = _git("status", "--porcelain", "--untracked-files=no", "--", "prismaquant")
    if changed:
        raise RuntimeError(
            f"the working directory {root} is at {commit} but its tracked prismaquant/ "
            f"files differ from that commit; commit them before launching:\n{changed}")
    return commit


def inspect_or_load(container):
    requested = container["image"]
    found = subprocess.run([PODMAN, "image", "inspect", requested], capture_output=True, text=True)
    if found.returncode:
        bound = container.get("archive")
        if bound is None:
            raise RuntimeError("declared container image is unavailable: " + found.stderr)
        with Path(bound["path"]).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != bound["sha256"]:
            raise RuntimeError("declared image archive bytes changed")
        subprocess.run([PODMAN, "load", "--input", bound["path"]], check=True)
        found = subprocess.run([PODMAN, "image", "inspect", requested], capture_output=True, text=True, check=True)
    rows = json.loads(found.stdout)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("the container engine returned no unique image inspection")
    image_id = rows[0].get("Id")
    if isinstance(image_id, str):
        rows[0]["Id"] = "sha256:" + image_id.removeprefix("sha256:")
    return rows


def validate_namespace_launch(spec: dict, command: list[str], *, cwd: str, environ) -> str | None:
    """Opt-in ownership refusal before image inspection or campaign entry."""
    if "namespace_binding" not in spec:
        if "namespace_profile" in spec:
            raise RuntimeError("namespace profile requires bound published ownership")
        return None
    row = namespace_adapter_request(spec, command, environ)
    commit = checkout_commit(cwd)
    if commit is None:
        raise RuntimeError("namespace requires a committed executed checkout")
    validate_namespace_request(row, executed_commit=commit)
    # The legacy helper checks package bytes only. Namespace ownership is over
    # the full source, including this adapter and the preparation contract.
    try:
        changed = subprocess.run(["git", "-C", cwd, "status", "--porcelain",
                                  "--untracked-files=normal"],
                                 capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("namespace cannot inspect executed source") from exc
    if changed.returncode != 0 or changed.stdout.strip():
        raise RuntimeError("namespace full executed source differs from its commit")
    guarded_import_root(spec, cwd=cwd, require_checkout=True)
    establish_namespace_temporaries(row)
    return commit


def require_namespace_native_profile(spec: dict, admission: dict) -> None:
    """A namespace row's sealed native profile must be the one that admitted this job."""
    binding = spec.get("namespace_binding")
    if binding is not None and binding["provenance"]["native_profile"] != admission["profile"]:
        raise RuntimeError("namespace native profile differs from the admitting launch")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--cpu-only", action="store_true",
                        help="accepted for old callers; native admission decides the device")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    spec = json.loads(args.spec)
    forwarded_env = spec.get("env") if isinstance(spec.get("env"), dict) else {}
    bounded = all(name in forwarded_env for name in BOUNDED_CAPTURE_ENV)
    validate_container(spec, bounded=bounded)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a container command is required")
    namespace_commit = validate_namespace_launch(spec, command, cwd=str(Path.cwd()), environ=os.environ)
    admission = native_admission(os.environ)
    require_namespace_native_profile(spec, admission)
    requested = spec["container"]["image"]
    inspected = inspect_or_load(spec["container"])
    if not isinstance(inspected, list) or len(inspected) != 1 or not isinstance(inspected[0], dict):
        raise RuntimeError("the container engine returned no unique image inspection")
    image_id = inspected[0].get("Id")
    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise RuntimeError("the container engine returned no immutable image ID")
    content_digest = _runtime_identity["image_content_sha256"](inspected[0])
    declared = spec["container"].get("content_sha256")
    if declared is not None and declared != content_digest:
        raise RuntimeError(f"container image content differs for {requested!r}: "
                           f"expected {declared}, observed {content_digest}")
    with_gpu, gpu_reason = gpu_attachment(admission)
    imports = verify_pinned_import(spec, cwd=str(Path.cwd()))
    commit = namespace_commit if namespace_commit is not None else checkout_commit(str(Path.cwd()))
    print(json.dumps({"schema": "prismaquant.tessera_campaign_container.v1",
                      "requested_image": requested, "image_id": image_id,
                      "image_content_sha256": content_digest,
                      "declared_content_sha256": declared,
                      "uid": os.getuid(), "gid": os.getgid(),
                      "gpu_attached": with_gpu, "gpu_decision": gpu_reason,
                      "checkout_commit": commit,
                      "overlay_pinned_caches": container_cache_environment(
                          spec, _forwarded_local_scratch_environment(spec, os.environ))[1],
                      **imports}), flush=True)
    return run_owned_container(spec, command, cwd=str(Path.cwd()), image_id=image_id,
                               content_sha256=content_digest, admission=admission,
                               record_dir=admission["record_dir"], environ=os.environ,
                               bounded=bounded, checkout_commit=commit, image=inspected[0])


if __name__ == "__main__":
    raise SystemExit(main())

