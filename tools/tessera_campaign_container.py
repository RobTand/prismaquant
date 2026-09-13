"""Run an admitted campaign quantum in its declared Docker environment.

PB owns placement, CPU affinity and container containment. This adapter only
maps the worker's sealed checkout and explicit data mounts into the container.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

from tools.container_runtime_identity import (
    image_content_sha256, prismaquant_source_sha256)
from prismaquant.prismabuild_progress import PATH_ENV, TOKEN_ENV


#: Python's safe-path mode, which drops the implicit ``sys.path[0]`` entry that
#: ``python -m`` sets to the working directory.  The container's working
#: directory is the PB sealed checkout, which carries its own ``prismaquant``
#: package, so without this the pinned mount named first in ``PYTHONPATH``
#: never wins and the campaign runs the sealed checkout's code (#519).
SAFE_PATH_ENV = "PYTHONSAFEPATH"


def validate_container(spec: dict) -> None:
    container = spec.get("container")
    if not isinstance(container, dict) or set(container) - {"image", "mounts", "content_sha256", "archive"}:
        raise RuntimeError("container must declare image and optional mounts/content_sha256/archive only")
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
    env = spec.get("env", {})
    if not isinstance(env, dict) or any(
            not isinstance(k, str) or not k or "=" in k or "\x00" in k
            or not isinstance(v, str) or "\x00" in v for k, v in env.items()):
        raise RuntimeError("container env must map environment names to strings")
    if 'PRISMAQUANT_CONTAINER_CONTENT_SHA256' in env:
        raise RuntimeError('actual container content is supplied by the inspected launcher')
    if SAFE_PATH_ENV in env:
        raise RuntimeError('the import guard is supplied by the launcher, not by a spec')


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
                   with_gpu=True, environ=None) -> list[str]:
    validate_container(spec)
    argv = ["docker", "run", "--rm", *(["--gpus", "all"] if with_gpu else []), "--ipc=host",
            "--user", f"{uid}:{gid}", "--workdir", "/workspace",
            "--entrypoint", "", "--mount",
            f"type=bind,src={cwd},dst=/workspace,readonly"]
    for mount in spec["container"].get("mounts", []):
        value = f"type=bind,src={mount['source']},dst={mount['target']}"
        if mount.get("readonly", False):
            value += ",readonly"
        argv += ["--mount", value]
    forwarded = {SAFE_PATH_ENV: "1", **spec.get("env", {}),
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
    validate_container(spec)
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
                            environ=os.environ)
    os.execvp(docker[0], docker)
    return 1  # exec never returns


if __name__ == "__main__":
    raise SystemExit(main())
