"""Run an admitted campaign quantum in its declared Docker environment.

PB owns placement, CPU affinity and container containment. This adapter only
maps the worker's sealed checkout and explicit data mounts into the container.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

from tools.container_runtime_identity import (
    image_content_sha256, prismaquant_source_sha256)


# These are PrismaBuild's action environment contract, deliberately kept in
# this host-side adapter rather than imported from ``prismaquant``.  Importing
# the package runs its production initialisation before this launcher reaches
# the qualified container, so a worker that only has the small host Python
# runtime could fail before Docker starts (#601).  The container-side campaign
# continues to use ``prismaquant.prismabuild_progress`` to write the record;
# this adapter only carries the two sealed channel values across the boundary.
PATH_ENV = "PRISMABUILD_ACTION_PROGRESS_PATH"
TOKEN_ENV = "PRISMABUILD_ACTION_PROGRESS_TOKEN"


#: Python's safe-path mode, which drops the implicit ``sys.path[0]`` entry that
#: ``python -m`` sets to the working directory.  The container's working
#: directory is the PB sealed checkout, which carries its own ``prismaquant``
#: package, so without this the pinned mount named first in ``PYTHONPATH``
#: never wins and the campaign runs the sealed checkout's code (#519).
SAFE_PATH_ENV = "PYTHONSAFEPATH"


#: The environment a bounded capture row's process must have been started with,
#: per ``prismaquant/autoscale.py``: torch wheels may statically link mimalloc,
#: whose delayed purge otherwise retains completed H/X after every owner is
#: gone, and the release-source-pages policy is the other half of the same
#: bounded phase plan. ``prismaquant.tessera_joint_aura`` calls
#: ``require_bounded_capture_environment`` at its first bounded step, which is
#: well into the loader on this campaign -- so a spec that omits a name is not a
#: late refusal, it is a dead pilot.
#:
#: Carried here for the same reason as ``PATH_ENV`` and ``SAFE_PATH_ENV``:
#: importing ``prismaquant`` on the worker to read the contract runs the
#: package's production initialisation before the qualified container starts
#: (#601), and this adapter is host-side. The two constants are held together by
#: ``tests/test_tessera_campaign_container.py``, which imports both and fails if
#: they drift -- a second copy is only a defect when nothing compares them.
#:
#: The launcher SUPPLIES these for a BOUNDED row rather than requiring the spec
#: to restate them, and refuses a bounded spec that declares a different value.
#: A bounded row cannot start without them, and refusing a sealed spec for
#: omitting one would turn a fixable launch into a re-seal. They are the
#: bounded path's environment and nothing else's: a legacy row that declares
#: ``MIMALLOC_PURGE_DELAY=10`` on purpose keeps 10 in the container it starts.
#: The launcher states which contract it is under from the row's own sealed
#: environment (``main``), because that is the only marker the host-side
#: adapter sees -- it is deliberately importable without ``prismaquant``.
BOUNDED_CAPTURE_ENV = {
    "PRISMAQUANT_RELEASE_SOURCE_PAGES": "1",
    "MIMALLOC_PURGE_DELAY": "0",
}


#: How a container reaches the GPU it was admitted for, per GPU runtime.
#:
#: ``--gpus all`` is the NVIDIA container runtime's flag and nothing else's. A
#: WSL2 ROCm box has no amdgpu driver and no NVIDIA runtime: the device is the
#: Windows GPU paravirtualization node ``/dev/dxg`` and the userspace half of
#: the driver lives on the host under ``/usr/lib/wsl/lib``, so a container
#: there needs both and is refused the device by the other flag. The mapping
#: is per runtime rather than per image because it is a property of the box.
GPU_RUNTIME_FLAGS = {
    "nvidia": ("--gpus", "all"),
    "rocm-wsl": ("--device", "/dev/dxg",
                 "--mount", "type=bind,src=/usr/lib/wsl/lib,dst=/usr/lib/wsl/lib,readonly"),
}
DEFAULT_GPU_RUNTIME = "nvidia"


def container_memory_budget_gb(spec: dict) -> float | None:
    """The cgroup memory cap this row runs under, or ``None`` when it declares none.

    A joint row cannot be bounded without one. ``CaptureMemoryGuard`` refuses at
    construction rather than running without a budget ("bounded capture requires
    a finite cgroup memory budget") and holds its physical margin back from this
    cap on every later check, so the cap is what makes the guard exist at all.
    It is NOT the aggregate physical bound: on GB10 the cgroup does not charge
    device memory (measured -- a 16 GiB ``HostConfig.Memory`` container held
    78.87 GiB of model and 6 GiB of KV), so this cap bounds the CPU side and the
    aggregate is ``cap + device envelope + external headroom``, carried by the
    plan's ``aggregate_memory_bytes`` and checked against the box by
    ``memory_management.require_aggregate_budget``.

    TWO FIELDS, TWO MEANINGS. ``cpu_memory_gb`` is this cap: what the container's
    cgroup may charge, which on GB10 is the CPU side. ``box_memory_gb`` is the
    box's total unified capacity, the number ``dispatch_tessera_campaign``
    refuses to derive a row's admission demand above -- and a PrismaBuild
    reservation for a row that holds 80 GiB of device residency beside a 34 GiB
    CPU cap has to be the combined physical demand, not the CPU cap. Reading the
    cap out of ``box_memory_gb`` conflated those and would have either refused
    the pilot or under-reserved the box.

    ``box_memory_gb`` remains the fallback so a spec sealed before
    ``cpu_memory_gb`` existed keeps the invocation it was sealed with. A spec
    that declares neither keeps the previous behaviour, which is what every row
    that ran before either field existed relies on.
    """
    raw = spec.get("cpu_memory_gb", spec.get("box_memory_gb"))
    if raw is None:
        return None
    if (isinstance(raw, bool) or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw)) or raw <= 0):
        raise RuntimeError(
            "the container memory cap must be a positive number of GiB when a spec "
            f"declares one, got {raw!r}")
    return float(raw)


def validate_container(spec: dict, *, bounded: bool = False) -> None:
    """Check a container spec, with the bounded-capture env gate opt-in.

    ``bounded`` is the caller stating that this row is a bounded capture row.
    The environment contract below belongs to THAT path -- a legacy row that
    declares ``MIMALLOC_PURGE_DELAY=10`` on purpose is not a bounded capture
    row, and refusing it here would break unrelated work for a rule it does
    not fall under. The dispatch path already knows which it is building and
    passes it.
    """
    container = spec.get("container")
    if not isinstance(container, dict) or set(container) - {"image", "mounts", "content_sha256", "archive", "gpu_runtime"}:
        raise RuntimeError("container must declare image and optional mounts/content_sha256/archive/gpu_runtime only")
    if "gpu_runtime" in container and container["gpu_runtime"] not in GPU_RUNTIME_FLAGS:
        raise RuntimeError(
            "container.gpu_runtime must be one of "
            f"{sorted(GPU_RUNTIME_FLAGS)}; a runtime with no declared flags "
            "would silently start the container without its device")
    image = container.get("image")
    if not isinstance(image, str) or not image or image.startswith("-"):
        raise RuntimeError("container.image must name a Docker image")
    if "content_sha256" in container:
        digest = container["content_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError("container.content_sha256 must be a lowercase SHA256 digest")
    if 'archive' in container:
        bound = container['archive']
        if (not isinstance(bound, dict) or set(bound) != {'path', 'sha256'} or
                not isinstance(bound.get('path'), str) or not bound['path'].startswith('/') or
                str(PurePosixPath(bound['path'])) != bound['path'] or '..' in PurePosixPath(bound['path']).parts or
                not isinstance(bound.get('sha256'), str) or re.fullmatch(r'[0-9a-f]{64}', bound['sha256']) is None or
                'content_sha256' not in container):
            raise RuntimeError('container archive requires canonical path/SHA256 and image content digest')
    mounts = container.get("mounts", [])
    if not isinstance(mounts, list):
        raise RuntimeError("container.mounts must be a list")
    targets = set()
    for mount in mounts:
        if not isinstance(mount, dict) or set(mount) - {"source", "target", "readonly"}:
            raise RuntimeError("container mount must declare source, target and optional readonly")
        for field in ("source", "target"):
            value = mount.get(field)
            if (not isinstance(value, str) or not value.startswith("/")
                    or "," in value or "\x00" in value
                    or str(PurePosixPath(value)) != value or ".." in PurePosixPath(value).parts):
                raise RuntimeError(f"container mount {field} must be a canonical absolute path")
        target = PurePosixPath(mount["target"])
        workspace = PurePosixPath("/workspace")
        if target == workspace or target in workspace.parents or workspace in target.parents:
            raise RuntimeError("container mount cannot hide /workspace sealed source")
        if str(target) in targets:
            raise RuntimeError(f"duplicate container mount target: {target}")
        targets.add(str(target))
        if not isinstance(mount.get("readonly", False), bool):
            raise RuntimeError("container mount readonly must be boolean")
    runtime = container.get("gpu_runtime", DEFAULT_GPU_RUNTIME)
    for flag, value in zip(GPU_RUNTIME_FLAGS[runtime], GPU_RUNTIME_FLAGS[runtime][1:]):
        if flag == "--mount":
            target = value.split("dst=", 1)[1].split(",", 1)[0]
            if target in targets:
                raise RuntimeError(
                    f"container mount target {target} is the one the "
                    f"{runtime} GPU runtime supplies; declaring it twice is a "
                    "duplicate bind Docker refuses at launch")
    env = spec.get("env", {})
    if not isinstance(env, dict) or any(
            not isinstance(k, str) or not k or "=" in k or "\x00" in k
            or not isinstance(v, str) or "\x00" in v for k, v in env.items()):
        raise RuntimeError("container env must map environment names to strings")
    if 'PRISMAQUANT_CONTAINER_CONTENT_SHA256' in env:
        raise RuntimeError('actual container content is supplied by the inspected launcher')
    if SAFE_PATH_ENV in env:
        raise RuntimeError('the import guard is supplied by the launcher, not by a spec')
    for name, expected in BOUNDED_CAPTURE_ENV.items():
        # Declaring it is optional -- the launcher supplies it -- but a spec may
        # not weaken it, and the refusal names the field so a reader of the spec
        # does not have to diff the container's environment to find out.
        if bounded and name in env and env[name] != expected:
            raise RuntimeError(
                f"spec env {name}={env[name]!r} contradicts the bounded capture "
                f"contract ({name}={expected!r}); the launcher would have to "
                "override the spec to run the row, and a bounded row that starts "
                "with the wrong value is refused by the pass long after the "
                "loader has read the model")


def gpu_attachment(spec: dict, *, cpu_only: bool, environ) -> tuple:
    """Whether to attach the GPU, and the declaration that decided it.

    ``--gpus all`` maps the whole device into the container, and until this
    function existed the only thing that withheld it was a caller remembering
    ``--cpu-only``.  A PrismaBuild row that reserved no GPU therefore ran with
    the device attached and PrismaBuild's GPU tokens unspent, so its admission
    arithmetic could seat a GPU-reserving row beside it and a power reading
    taken next door had a second owner it could not see (#430).

    So the grant decides, not the flag's absence.  ``pbrun`` sets
    ``CUDA_VISIBLE_DEVICES`` to the empty string in the action's environment
    exactly when it granted no GPU slots, and a spec may say the same thing
    about its payload; either declaration withholds the device.  Neither is
    a substitute for this check on its own: an empty ``CUDA_VISIBLE_DEVICES``
    hides the device from CUDA inside the container, it does not stop the
    runtime attaching and initialising it, which is the distinction
    ``require_pool`` already refuses to treat as an exemption.

    Unset is not a declaration.  An interactive run outside ``pbrun`` has no
    grant to read, and it keeps the behaviour it had; ``--cpu-only`` is still
    the way to say no there.
    """

    if cpu_only:
        return False, "--cpu-only"
    for source, value in (("container spec env", (spec.get("env") or {}).get("CUDA_VISIBLE_DEVICES")),
                          ("CUDA_VISIBLE_DEVICES", environ.get("CUDA_VISIBLE_DEVICES"))):
        if value == "":
            return False, f"{source} declares no visible device"
    return True, "no declaration withheld the device"


def progress_environment(spec: dict, environ) -> dict:
    """The PrismaBuild progress channel this container needs, if any.

    The container is launched with the declared ``env`` and nothing else, so
    without this the row inside it cannot report advancement, PB sees a silent
    action and ends it within the startup allowance -- a stall watchdog killing
    exactly the working rows it was added to save (PB #480).

    Refused rather than dropped when the file's directory is not inside a
    writable declared mount.  A report written into the container's own
    ephemeral filesystem is invisible to the worker and indistinguishable from
    not reporting at all, and failing at launch is far cheaper than failing an
    hour into a pricing round.
    """

    path, token = environ.get(PATH_ENV), environ.get(TOKEN_ENV)
    if not path or not token:
        return {}
    directory = PurePosixPath(path).parent
    for mount in spec["container"].get("mounts", []):
        target = PurePosixPath(mount["target"])
        if (directory == target or target in directory.parents) and not mount.get("readonly", False):
            return {PATH_ENV: str(path), TOKEN_ENV: str(token)}
    raise RuntimeError(
        f"the PrismaBuild progress file {path} is not inside any writable "
        "container mount, so this row could not report the anchors it commits "
        "and would be ended as a stall; declare a mount covering it")


def host_path(container_path: str, *, cwd: str, mounts: list) -> "Path | None":
    """The host path Docker binds behind one absolute container path.

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


def _package_root(roots: list) -> "tuple[str, Path] | None":
    for entry, root in roots:
        if (root / "prismaquant" / "__init__.py").is_file():
            return entry, root
    return None


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
        if root is not None and (root / "prismaquant" / "__init__.py").is_file():
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

    guarded = _package_root(import_search_roots(spec, cwd=cwd, safe_path=True))
    unguarded = _package_root(import_search_roots(spec, cwd=cwd, safe_path=False))
    shadow_sha = (None if unguarded is None
                  else prismaquant_source_sha256(unguarded[1] / "prismaquant"))
    if guarded is None:
        return {"pinned_source_entry": None, "pinned_source_root": None,
                "pinned_source_sha256": None,
                "pinned_by_default": False,
                "import_resolution_source_sha256": None,
                "import_resolution_root": None,
                "working_directory_source_sha256": shadow_sha,
                "safe_path_guard_is_load_bearing": shadow_sha is not None}
    entry, pinned, by_default = pinned_source_root(spec, cwd=cwd)
    pinned_sha = prismaquant_source_sha256(pinned / "prismaquant")
    resolved_sha = prismaquant_source_sha256(guarded[1] / "prismaquant")
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


def docker_command(spec: dict, command: list[str], *, cwd: str,
                   uid: int, gid: int, image_id: str, content_sha256=None,
                   with_gpu=True, environ=None, bounded=False) -> list[str]:
    validate_container(spec, bounded=bounded)
    gpu_flags = GPU_RUNTIME_FLAGS[
        spec["container"].get("gpu_runtime", DEFAULT_GPU_RUNTIME)]
    argv = ["docker", "run", "--rm", *(gpu_flags if with_gpu else []), "--ipc=host",
            "--user", f"{uid}:{gid}", "--workdir", "/workspace",
            "--entrypoint", "", "--mount",
            f"type=bind,src={cwd},dst=/workspace,readonly"]
    budget_gb = container_memory_budget_gb(spec)
    if budget_gb is not None:
        # THE HARD, KERNEL-ENFORCED LIMIT -- on what the cgroup ACCOUNTS, which
        # is not the same thing as "everything this row allocates".
        #
        # Without it ``CaptureMemoryGuard`` cannot even be constructed ("bounded
        # capture requires a finite cgroup memory budget"), so a bounded capture
        # row was refused rather than bounded. With it the guard refuses at
        # ``cap - MARGIN_BYTES`` and on its host floor, and it also adds the
        # whole CUDA reservation to the cgroup charge -- the conservative sum
        # that covers a driver which does NOT charge device memory here.
        #
        # That sum is a refusal at the guard's CHECK POINTS, not a physical
        # bound: between checks a large allocation can overshoot, and a cgroup
        # cap a driver does not charge bounds the CPU side alone. So this cap is
        # the CPU-accounted hard limit, and a row that needs an aggregate bound
        # still needs its risky allocations preceded by ``check(reserve_bytes=..)``
        # or a bounded CUDA allocator. See docs/ARCHITECTURE.md and
        # docs/design note in the joint-aura-resume directory; do not read this
        # line as "GPU + CPU is bounded by box_memory_gb".
        #
        # The value is the spec's own ``box_memory_gb``: the same declaration the
        # demand derivation already refuses to exceed, so the number priced and
        # the number enforced are one number. ``--memory-swap`` equal to the
        # limit stops the container growing into swap instead of failing.
        argv += ["--memory", f"{budget_gb:g}g", "--memory-swap", f"{budget_gb:g}g"]
    for mount in spec["container"].get("mounts", []):
        value = f"type=bind,src={mount['source']},dst={mount['target']}"
        if mount.get("readonly", False):
            value += ",readonly"
        argv += ["--mount", value]
    # The bounded capture defaults are the BOUNDED path's environment, so they
    # are merged only for a row the caller says is bounded. Merging them for
    # every row overrode a legacy spec's own declaration one layer below the
    # row-env check that already keeps it: a legacy row sealed with
    # ``MIMALLOC_PURGE_DELAY=10`` reached the container with ``0``.
    bounded_defaults = BOUNDED_CAPTURE_ENV if bounded else {}
    forwarded = {SAFE_PATH_ENV: "1", **spec.get("env", {}),
                 **bounded_defaults,
                 **progress_environment(spec, environ if environ is not None else {})}
    for key, value in sorted(forwarded.items()):
        argv += ["--env", f"{key}={value}"]
    if content_sha256 is not None:
        argv += ['--env', 'PRISMAQUANT_CONTAINER_CONTENT_SHA256=' + content_sha256]
    return [*argv, image_id, *command]


def inspect_or_load(container):
    requested = container['image']
    found = subprocess.run(['docker', 'image', 'inspect', requested], capture_output=True, text=True)
    if found.returncode:
        bound = container.get('archive')
        if bound is None:
            raise RuntimeError('declared container image is unavailable: ' + found.stderr)
        with Path(bound['path']).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != bound['sha256']:
            raise RuntimeError('declared image archive bytes changed')
        subprocess.run(['docker', 'load', '--input', bound['path']], check=True)
        found = subprocess.run(['docker', 'image', 'inspect', requested], capture_output=True, text=True, check=True)
    rows = json.loads(found.stdout)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError('Docker returned no unique image inspection')
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument('--cpu-only', action='store_true', help='Run admitted CPU checks without requesting a GPU')
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    spec = json.loads(args.spec)
    # The spec the launcher forwards carries the bounded capture environment
    # only for a bounded row (``dispatch_tessera_campaign._row_is_bounded``
    # merges it), so the row's own env is the marker that says which contract
    # this is: a legacy spec that names a different purge delay is not held to
    # a contract it never declared, and is not handed the bounded defaults
    # either. Both readers of the marker -- the validation below and the
    # argv built later -- are stated from this one value.
    forwarded_env = spec.get("env") if isinstance(spec.get("env"), dict) else {}
    bounded = all(name in forwarded_env for name in BOUNDED_CAPTURE_ENV)
    validate_container(spec, bounded=bounded)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a container command is required")
    requested = spec["container"]["image"]
    inspected = inspect_or_load(spec['container'])
    if not isinstance(inspected, list) or len(inspected) != 1 or not isinstance(inspected[0], dict):
        raise RuntimeError("Docker returned no unique image inspection")
    image_id = inspected[0].get("Id")
    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise RuntimeError("Docker returned no immutable image ID")
    content_digest = image_content_sha256(inspected[0])
    declared = spec["container"].get("content_sha256")
    if declared is not None and declared != content_digest:
        raise RuntimeError(f"Docker image content differs for {requested!r}: "
                           f"expected {declared}, observed {content_digest}")
    with_gpu, gpu_reason = gpu_attachment(spec, cpu_only=args.cpu_only, environ=os.environ)
    imports = verify_pinned_import(spec, cwd=str(Path.cwd()))
    print(json.dumps({"schema": "prismaquant.tessera_campaign_container.v1",
                      "requested_image": requested, "image_id": image_id,
                      "image_content_sha256": content_digest,
                      "declared_content_sha256": declared,
                      "uid": os.getuid(), "gid": os.getgid(),
                      "gpu_attached": with_gpu, "gpu_decision": gpu_reason,
                      **imports}), flush=True)
    docker = docker_command(spec, command, cwd=str(Path.cwd()),
                            uid=os.getuid(), gid=os.getgid(), image_id=image_id,
                            content_sha256=content_digest, with_gpu=with_gpu,
                            environ=os.environ, bounded=bounded)
    os.execvp(docker[0], docker)
    return 1  # exec never returns


if __name__ == "__main__":
    raise SystemExit(main())
