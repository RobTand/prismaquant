"""The container spec a campaign row declares, and the image PB admits it on.

Stdlib-only on purpose. Planners, dispatchers and row builders import it; the
host-side launcher ``tools/tessera_campaign_container.py`` loads this file by
path, as it loads ``container_runtime_identity.py``, so the launcher validates
the same contract without importing ``prismaquant`` before Docker starts
(#601).
"""
from __future__ import annotations

import math
from pathlib import PurePosixPath
import re


#: PB's protected reader identity (RobTand/prismabuild#730): the resource_exec
#: proxy injects these from the exact launch identity -- the 64-hex action key
#: it already ran under, the launch-bound nonce/scope pair the reader SDK
#: checks against the live claim row, and the sealed helper generation root
#: readers import from. The container-side SDK (PQ #846) appends ``/src`` to
#: the helper root itself, so this names the generation directory, never the
#: ``src`` tree beneath it. The literals are repeated here rather than
#: imported from ``prismabuild``, whose initialisation must not run on the
#: host before Docker starts.
ACTION_KEY_ENV = "PRISMABUILD_ACTION_KEY"
ACTION_NONCE_ENV = "PRISMABUILD_ACTION_NONCE"
ACTION_SCOPE_ENV = "PRISMABUILD_ACTION_SCOPE"
READER_HELPER_ROOT_ENV = "PRISMABUILD_READER_HELPER_ROOT"
QUEUE_ROOT_ENV = "PRISMABUILD_QUEUE_ROOT"
#: Every name the launcher -- never the spec -- may supply for reader context.
READER_CONTEXT_ENV = (ACTION_KEY_ENV, ACTION_NONCE_ENV, ACTION_SCOPE_ENV,
                      READER_HELPER_ROOT_ENV, QUEUE_ROOT_ENV)


#: Python's safe-path mode, which drops the implicit ``sys.path[0]`` entry that
#: ``python -m`` sets to the working directory.  The container's working
#: directory is the PB sealed checkout, which carries its own ``prismaquant``
#: package, so without this the pinned mount named first in ``PYTHONPATH``
#: never wins and the campaign runs the sealed checkout's code (#519).
SAFE_PATH_ENV = "PYTHONSAFEPATH"


#: The commit the sealed checkout is at, read on the host and handed to the
#: container. The campaign image ships no git binary, and the joint pass binds
#: every unit checkpoint to ``aura_provenance.checkpoint_git_commit()``, which
#: resolves the commit with git unless this variable names it. Inside the
#: container the variable also skips that function's own exactness check
#: (``git diff`` of the declared AURA checkpoint owners against HEAD), so the
#: launcher checks the whole ``prismaquant/`` package here instead, refusing
#: a checkout whose tracked package bytes differ from HEAD.
CHECKOUT_COMMIT_ENV = "PRISMAQUANT_IDENTITY_GIT_COMMIT"


#: The environment a bounded capture row's process must have been started with,
#: per ``prismaquant/autoscale.py``: torch wheels may statically link mimalloc,
#: whose delayed purge otherwise retains completed H/X after every owner is
#: gone, and the release-source-pages policy is the other half of the same
#: bounded phase plan. ``prismaquant.tessera_joint_aura`` calls
#: ``require_bounded_capture_environment`` before metadata intake and device
#: allocation; the joint submission adapter supplies missing names from the
#: plan's qualification-window or retained-execution requirement.
#:
#: Carried here rather than imported from ``prismaquant.autoscale`` because
#: this module stays stdlib-only: the host launcher loads it by path before
#: the qualified container starts (#601). The two constants are held together by
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
    if CHECKOUT_COMMIT_ENV in env:
        raise RuntimeError('the checkout commit is supplied by the inspected launcher, not by a spec')
    for name in READER_CONTEXT_ENV:
        # Sealed reader identity is the launcher's to forward from the exact
        # launch environment, exactly like PB's own core refuses a sealed
        # variable of any of these names rather than overwriting it: a spec
        # that names one is forged or conflicting, never a default.
        if name in env:
            raise RuntimeError(
                f'the {name} is supplied by the launcher, not by a spec')
    for name, expected in BOUNDED_CAPTURE_ENV.items():
        # Declaring it is optional -- the launcher supplies it -- but a spec may
        # not weaken it, and the refusal names the field so a reader of the spec
        # does not have to diff the container's environment to find out.
        if bounded and name in env and env[name] != expected:
            raise RuntimeError(
                f"spec env {name}={env[name]!r} contradicts the bounded capture "
                f"contract ({name}={expected!r}); the launcher would have to "
                "override the spec to run the row, and a bounded row that starts "
                "with the wrong value is refused by the pass before metadata intake")


#: The PrismaBuild option that declares the container image a submission
#: requires present on its claiming box before it runs (RobTand/prismabuild#714).
#: PrismaBuild does not pull, load or transfer an image: the declaration is a
#: placement requirement over the box's own Docker, so the reference has to be
#: an immutable identity the receipt can be held against (``sha256:<64 hex>``
#: for a local image ID or ``repository@sha256:<64 hex>`` for a manifest
#: digest).  The spelling lives here because this module owns the spec's image
#: contract and both PB submitters must agree on the flag.
CONTAINER_IMAGE_FLAG = "--container-image"


def admission_image_reference(
    spec: object, *, portable_refusal: type[Exception] = RuntimeError,
) -> str | None:
    """The image reference PrismaBuild must admit this spec against, or ``None``.

    One reader for every submission path (the joint dispatcher's stage-A and
    quantum rows, the campaign dispatcher's direct commands and its generated
    manifest rows), so the image a submission declares is the image its own
    launcher spec will inspect, and it comes from the same parsed document
    that is serialized into ``--spec``.

    An explicit ``container_admission_reference`` takes precedence: the
    joint dispatcher's existing portable spelling is ``content:sha256:<hex64>``
    and requires an inspected ``container.content_sha256``. ``portable_refusal``
    preserves that caller's error type; container validation keeps its own.
    This also preserves
    an explicit declaration beside an archive; without an override,
    ``None`` -- declare nothing -- applies in exactly two cases:

    * the spec runs no container at all; and
    * the spec binds a validated ``container.archive``.  The launcher's
      ``inspect_or_load`` verifies the archive's bytes against its declared
      digest, loads it and then checks the inspected content against
      ``container.content_sha256``; that path establishes the image *inside*
      the action, on whichever worker claims it, so a local-presence
      prerequisite would refuse the claim before the loader ever ran and
      strand the load-from-archive route (PrismaBuild #714).  The loader
      remains the responsible party for that spec.

    The spec is validated with :func:`validate_container`, which the launcher
    also runs,
    before anything is read from it, so a malformed archive or container
    refuses here rather than silently skipping the declaration. Without an
    explicit portable reference, a mutable tag is returned as declared -- the
    reference the launcher will
    resolve -- and PrismaBuild's admission refuses it, because a tag is not an
    identity that can be sealed into an action key.
    """
    if not isinstance(spec, dict):
        return None
    container = spec.get("container")
    if container is not None:
        validate_container(spec)
    admission = spec.get("container_admission_reference")
    if admission is not None:
        if (not isinstance(admission, str)
                or re.fullmatch(r"content:sha256:[0-9a-f]{64}", admission) is None
                or not isinstance(container, dict) or "content_sha256" not in container):
            raise portable_refusal("explicit portable image admission requires content SHA and inspected scientific image identity")
        return admission
    if container is None or "archive" in container:
        return None
    return container["image"]
