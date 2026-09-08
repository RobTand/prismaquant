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


def docker_command(spec: dict, command: list[str], *, cwd: str,
                   uid: int, gid: int, image_id: str, content_sha256=None, with_gpu=True) -> list[str]:
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
    for key, value in sorted(spec.get("env", {}).items()):
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
    print(json.dumps({"schema": "prismaquant.tessera_campaign_container.v1",
                      "requested_image": requested, "image_id": image_id,
                      "image_content_sha256": content_digest,
                      "declared_content_sha256": declared,
                      "uid": os.getuid(), "gid": os.getgid()}), flush=True)
    docker = docker_command(spec, command, cwd=str(Path.cwd()),
                            uid=os.getuid(), gid=os.getgid(), image_id=image_id,
                            content_sha256=content_digest, with_gpu=not args.cpu_only)
    os.execvp(docker[0], docker)
    return 1  # exec never returns


if __name__ == "__main__":
    raise SystemExit(main())
