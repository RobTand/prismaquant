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

from tools.container_runtime_identity import image_content_sha256
from prismaquant.prismabuild_progress import PATH_ENV, TOKEN_ENV


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
    forwarded = {**spec.get("env", {}),
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
    print(json.dumps({"schema": "prismaquant.tessera_campaign_container.v1",
                      "requested_image": requested, "image_id": image_id,
                      "image_content_sha256": content_digest,
                      "declared_content_sha256": declared,
                      "uid": os.getuid(), "gid": os.getgid(),
                      "gpu_attached": with_gpu, "gpu_decision": gpu_reason}), flush=True)
    docker = docker_command(spec, command, cwd=str(Path.cwd()),
                            uid=os.getuid(), gid=os.getgid(), image_id=image_id,
                            content_sha256=content_digest, with_gpu=with_gpu,
                            environ=os.environ)
    os.execvp(docker[0], docker)
    return 1  # exec never returns


if __name__ == "__main__":
    raise SystemExit(main())
