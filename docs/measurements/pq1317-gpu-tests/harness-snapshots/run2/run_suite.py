#!/usr/bin/env python3
"""Host side of the pq1317 qualification harness. PrismaBuild runs it on a GB10 box.

It fetches the candidate Tessera commit into a fresh git checkout, starts image X
with that checkout mounted read-only, and gathers what the run left behind.
``--mode collect`` runs without a GPU (the D38 preflight). ``--mode run`` runs the
suite on the GPU under ``--strict-cuda``. The container side is ``container_entry.py``.

Everything the run needs comes from the package directory above this one:
``candidate.json`` (commit, image, contract digest) and ``roster.json`` (the nodes).
The script refuses to run when the checkout is not exactly the candidate commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = Path("/mnt/shared/tessera-measurements/pq1317-gpu-tests")
DEFAULT_SOURCE_URL = "https://github.com/RobTand/tessera.git"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, check=True, text=True, **kwargs)


def captured(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


def fetch_source(url: str, commit: str, destination: Path) -> dict:
    """A fresh git checkout of exactly ``commit``; refuses anything else."""
    destination.mkdir(parents=True)
    git = ["git", "-C", str(destination), "-c", "advice.detachedHead=false"]
    run(["git", "init", "-q", str(destination)])
    run([*git, "remote", "add", "origin", url])
    run([*git, "fetch", "-q", "--depth", "1", "origin", commit])
    run([*git, "checkout", "-q", "--detach", "FETCH_HEAD"])
    head = captured([*git, "rev-parse", "HEAD"])
    if head != commit:
        raise SystemExit(f"fetched {head}, the candidate is {commit}")
    if captured([*git, "status", "--porcelain=v1", "--untracked-files=all"]):
        raise SystemExit("the fresh checkout is not clean")
    return {"url": url, "head": head, "tree": captured([*git, "rev-parse", "HEAD^{tree}"]),
            "tracked_files": len(captured([*git, "ls-files"]).splitlines())}


def docker_command(args, out: Path, source: Path, candidate: dict, nodes_path: Path) -> list[str]:
    cpus = ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
    environment = {
        "TMPDIR": f"{out}/tmp", "HOME": f"{out}/home", "TORCH_EXTENSIONS_DIR": f"{out}/torch-ext",
        "MAX_JOBS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": f"{source}/src:{out}/test-deps:{out}/harness",
    }
    command = ["docker", "run", "--rm"]
    if args.mode == "run":
        command += ["--gpus", "all"]
    else:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    command += ["--cpuset-cpus", cpus, "--user", f"{os.getuid()}:{os.getgid()}", "--shm-size", "1g",
                "-v", f"{source}:{source}:ro", "-v", f"{out}:{out}:rw", "-w", str(source)]
    for key, value in environment.items():
        command += ["-e", f"{key}={value}"]
    command += [candidate["image_digest"], "python3", "-u", f"{out}/harness/container_entry.py",
                "--suite", args.suite, "--mode", args.mode, "--out", str(out), "--source", str(source),
                "--candidate", candidate["tessera_commit"], "--contract-sha256", candidate["contract_sha256"],
                "--nodes-json", str(nodes_path)]
    return command


def native_libraries(out: Path) -> list[dict]:
    libraries = [{"path": str(p.relative_to(out)), "sha256": sha256_file(p)}
                 for p in sorted((out / "torch-ext").rglob("*.so"))]
    (out / "native-so.sha256").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in libraries), encoding="utf-8")
    return libraries


def triton_kernels(out: Path) -> dict:
    """Kernels Triton compiled during the run: the run's HOME started empty."""
    names: dict[str, int] = {}
    targets: set[str] = set()
    for path in (out / "home" / ".triton" / "cache").glob("*/*.json"):
        if path.name.startswith("__grp__"):
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(meta, dict) and meta.get("name"):
            names[meta["name"]] = names.get(meta["name"], 0) + 1
            targets.add(json.dumps(meta.get("target"), sort_keys=True))
    inventory = {"kernels": dict(sorted(names.items())), "targets": sorted(targets)}
    (out / "triton-kernels.json").write_text(json.dumps(inventory, indent=1) + "\n", encoding="utf-8")
    return inventory


def image_identity(image: str) -> dict:
    """What the box's Docker says about the image the action named."""
    try:
        text = captured(["docker", "image", "inspect", "--format",
                         "{{.Id}}|{{json .RepoDigests}}|{{.Created}}|{{.Architecture}}", image])
    except (OSError, subprocess.CalledProcessError) as error:
        return {"reference": image, "error": repr(error)}
    image_id, digests, created, architecture = text.split("|", 3)
    return {"reference": image, "id": image_id, "repo_digests": json.loads(digests),
            "created": created, "architecture": architecture}


def gpu_identity() -> dict:
    query = "name,uuid,driver_version,compute_cap,memory.total"
    try:
        text = captured(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
    except (OSError, subprocess.CalledProcessError) as error:
        return {"query": query, "error": repr(error)}
    return {"query": query, "rows": text.splitlines()}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite", choices=("moe", "dense"), required=True)
    parser.add_argument("--mode", choices=("collect", "run"), required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    args = parser.parse_args(argv)

    candidate = json.loads((PACKAGE / "candidate.json").read_text(encoding="utf-8"))
    roster = json.loads((PACKAGE / "roster.json").read_text(encoding="utf-8"))
    if roster["candidate"] != candidate["tessera_commit"]:
        raise SystemExit("roster.json and candidate.json name different candidates")
    nodes = [node["id"] for node in roster["nodes"] if node["suite"] == args.suite]
    if len(nodes) != roster["suites"][args.suite]["node_count"]:
        raise SystemExit("roster.json: node list and node_count disagree")

    args.out_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f"{args.mode}-{args.suite}.", dir=args.out_root))
    for name in ("tmp", "home", "test-deps", "torch-ext"):
        (out / name).mkdir()
        (out / name).chmod(0o777)
    out.chmod(0o777)
    shutil.copytree(HARNESS, out / "harness", ignore=shutil.ignore_patterns("__pycache__"))
    nodes_path = out / "expected-nodes.json"
    nodes_path.write_text(json.dumps(nodes, indent=1) + "\n", encoding="utf-8")
    print(f"PQ1317_OUT {out}", flush=True)

    started = time.time()
    source = out / "tessera-source"
    manifest: dict = {
        "schema": "pq1317.run.v1", "mode": args.mode, "suite": args.suite, "host": socket.gethostname(),
        "candidate": candidate["tessera_commit"], "image": candidate["image_digest"], "out": str(out),
        "harness_sha256": {p.name: sha256_file(p) for p in sorted(HARNESS.glob("*.py"))},
        "package_sha256": {name: sha256_file(PACKAGE / name) for name in ("candidate.json", "roster.json")},
        "closure_files": {p.name: p.read_text(encoding="utf-8")
                          for p in sorted(Path.cwd().glob(".pbrun-closure.*.json"))},
        "started_unix": started,
    }
    manifest["source"] = fetch_source(args.source_url, candidate["tessera_commit"], source)
    manifest["gpu"] = gpu_identity()
    manifest["image_inspect"] = image_identity(candidate["image_digest"])

    command = docker_command(args, out, source, candidate, nodes_path)
    manifest["docker_command"] = command
    print("+ " + " ".join(command), flush=True)
    returncode = subprocess.run(command).returncode
    print(f"DOCKER_RC={returncode}", flush=True)

    manifest.update(docker_returncode=returncode, finished_unix=time.time(),
                    native_libraries_built=native_libraries(out), triton=triton_kernels(out))
    (out / "run-manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("PQ1317_RUN " + json.dumps({k: manifest[k] for k in (
        "mode", "suite", "host", "candidate", "out", "docker_returncode")}, sort_keys=True), flush=True)
    return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
