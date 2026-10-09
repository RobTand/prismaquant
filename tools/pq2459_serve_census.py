"""PQ #2459 GPU census entry point: TP2 eager route census at one Tessera tree.

This module is the single entry point for both the D38 CPU dry run and
the GPU qualification action. The dry run exercises this parser, the
fixture metadata, the artifact binding, and the submission manifest
without CUDA. The GPU action runs the same parsed arguments through the
same driver path. Both record the immutable qualified source identity
through tessera.package_source.v1, never the provisioner checksum.

Usage (dry run, CPU):
  python tools/pq2459_serve_census.py --mode dry-run --profile tr3_batch ...

Usage (GPU census, inside the stock vLLM image with the Tessera plugin):
  python tools/pq2459_serve_census.py --mode census --profile tr3_batch ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

QUALIFIED_SERVING_COMMIT = "9eef9fea6edce32f4e64abf87f0058b11dab2287"
QUALIFIED_PRODUCER_COMMIT = "9eef9fea6edce32f4e64abf87f0058b11dab2287"
QUALIFIED_SERVING_SOURCE_SHA256 = (
    "a9b7bf32563ce874f45956dd4e5ff4b4c43de73459f9aa40dee1977b9b152330"
)
QUALIFIED_SOURCE_FILES = 128
QUALIFIED_ALGORITHM = "tessera.package_source.v1"
QUALIFIED_CONTRACT_SHA256 = (
    "840607e75b212e77aaa889803abf4e66ac7a7ef1815995db8c4a63d82523bbd7"
)
RUNTIME_IMAGE = (
    "localhost/prismaquant/spark-vllm-nccl230@sha256:"
    "5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a"
)
ARTIFACT = (
    "/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported"
)
FIXTURE_PROFILES = (
    Path(__file__).resolve().parent.parent
    / "docs/results/pq2471_fixture_profiles_2026-10-09.json"
)
TP_DEGREE = 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("dry-run", "census"))
    ap.add_argument("--model", default=ARTIFACT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runtime-image", default=RUNTIME_IMAGE)
    ap.add_argument("--tessera-commit", default=QUALIFIED_SERVING_COMMIT)
    ap.add_argument("--tensor-parallel-size", type=int, default=TP_DEGREE)
    ap.add_argument(
        "--profile",
        default="all",
        choices=("all", "tr3_batch", "speed_batch", "speed_decode"),
    )
    ap.add_argument("--tessera-src", default=None)
    return ap


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_profiles() -> dict:
    return json.loads(FIXTURE_PROFILES.read_text())["profiles"]


def check_artifact(model: Path) -> dict:
    cfg_path = model / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no config.json under {model}")
    cfg = json.loads(cfg_path.read_text())
    quant = cfg.get("quantization_config", {})
    if quant.get("quant_method") != "tessera":
        raise SystemExit("artifact is not a Tessera artifact")
    groups = quant.get("config_groups", {})
    if not groups:
        raise SystemExit("artifact names no config groups")
    index_path = model / "model.safetensors.index.json"
    return {
        "config_sha256": _sha256_file(cfg_path),
        "index_sha256": _sha256_file(index_path) if index_path.exists() else None,
        "groups": len(groups),
    }


def check_image(image: str) -> None:
    if "@sha256:" not in image:
        raise SystemExit("runtime image must name repository@sha256:digest")
    if image != RUNTIME_IMAGE:
        raise SystemExit(f"unpermitted runtime image {image}")
    digest = image.rsplit(":", 1)[1]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise SystemExit("runtime image digest is malformed")


def check_commit(commit: str) -> None:
    if commit != QUALIFIED_SERVING_COMMIT:
        raise SystemExit(
            f"unqualified serving commit {commit}; "
            f"this entry point serves only {QUALIFIED_SERVING_COMMIT}"
        )


def check_profile(profile: str, profiles: dict) -> dict:
    if profile == "all":
        return profiles
    if profile not in profiles:
        raise SystemExit(f"unknown fixture profile {profile}")
    return {profile: profiles[profile]}


def serving_source_sha256_of(src: Path) -> tuple[str, int]:
    """Compute the v1 digest without importing the Tessera package."""
    sys.path.insert(0, str(src / "src"))
    try:
        from tessera.serving.source_identity import (  # noqa: E402
            SOURCE_IDENTITY_ALGORITHM,
            serving_source_files,
            serving_source_sha256,
        )
    finally:
        sys.path.pop(0)
    if SOURCE_IDENTITY_ALGORITHM != QUALIFIED_ALGORITHM:
        raise SystemExit("identity algorithm mismatch")
    files = serving_source_files(src / "src")
    return serving_source_sha256(src / "src"), len(files)


def run_dry_run(args: argparse.Namespace, out: Path) -> int:
    profiles = load_profiles()
    wanted = check_profile(args.profile, profiles)
    check_image(args.runtime_image)
    check_commit(args.tessera_commit)
    if args.tensor_parallel_size != TP_DEGREE:
        raise SystemExit("this entry point serves TP 2 only")
    artifact = check_artifact(Path(args.model))
    manifest = {
        "schema": "prismaquant.pq2459_serve_census_dry_run.v1",
        "mode": "dry-run",
        "model": args.model,
        "artifact": artifact,
        "runtime_image": args.runtime_image,
        "tessera_commit": args.tessera_commit,
        "producer_commit": QUALIFIED_PRODUCER_COMMIT,
        "serving_source_sha256": QUALIFIED_SERVING_SOURCE_SHA256,
        "source_files": QUALIFIED_SOURCE_FILES,
        "algorithm": QUALIFIED_ALGORITHM,
        "tensor_parallel_size": args.tensor_parallel_size,
        "profiles": {
            name: {
                "regime": prof["regime"],
                "token_rows": prof["token_rows"],
                "dense_nk": prof["dense_nk"],
                "routed_moe_nk": prof["routed_moe_nk"],
            }
            for name, prof in wanted.items()
        },
        "execution_mode": "eager",
        "residency": "resident",
        "entry_point": "tools/pq2459_serve_census.py",
        "census_tool": "/home/rob/tessera/tools/tessera_route_census.py",
        "qualified_cells": 0,
    }
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=1, sort_keys=True))
    return 0

def run_census(args: argparse.Namespace, out: Path) -> int:
    if out.exists():
        out.unlink()
    check_image(args.runtime_image)
    check_commit(args.tessera_commit)
    if args.tensor_parallel_size != TP_DEGREE:
        raise SystemExit("this entry point serves TP 2 only")
    profiles = load_profiles()
    wanted = check_profile(args.profile, profiles)
    artifact = check_artifact(Path(args.model))
    src = Path(args.tessera_src or os.environ.get("TS", "/home/rob/tessera"))
    head = subprocess.run(
        ["git", "-C", str(src), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if head.returncode != 0 or head.stdout.strip() != QUALIFIED_SERVING_COMMIT:
        raise SystemExit(
            "Tessera tree is not the qualified serving commit "
            f"{QUALIFIED_SERVING_COMMIT}: got {head.stdout.strip()!r}"
        )
    digest, nfiles = serving_source_sha256_of(src)
    if digest != QUALIFIED_SERVING_SOURCE_SHA256 or nfiles != QUALIFIED_SOURCE_FILES:
        raise SystemExit(
            f"source identity mismatch: {digest} over {nfiles} files; "
            f"qualified is {QUALIFIED_SERVING_SOURCE_SHA256} "
            f"over {QUALIFIED_SOURCE_FILES} files"
        )
    census_tool = src / "tools/tessera_route_census.py"
    if not census_tool.exists():
        raise SystemExit(f"no census tool at {census_tool}")
    if os.environ.get("TESSERA_SERVE_MODE", "resident") != "resident":
        raise SystemExit("TESSERA_SERVE_MODE must be resident")
    env_image = os.environ.get("TESSERA_CENSUS_RUNTIME_IMAGE")
    if env_image is not None and env_image != args.runtime_image:
        raise SystemExit("launcher image declaration differs from --runtime-image")
    cmd = [
        sys.executable, str(census_tool), args.model, str(out),
        "--runtime-image", args.runtime_image,
        "--tessera-commit", args.tessera_commit,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--distributed-executor-backend", "ray",
    ]
    proc = subprocess.run(cmd, capture_output=False, text=False)
    if proc.returncode != 0:
        raise SystemExit(f"census tool exits {proc.returncode}")
    receipt = json.loads(out.read_text())
    ranks = receipt.get("ranks", [])
    if len(ranks) != TP_DEGREE:
        raise SystemExit(f"census covers {len(ranks)} ranks, not TP 2")
    for rank in ranks:
        stamped = (rank.get("header") or {}).get("serving_source_sha256")
        if stamped != QUALIFIED_SERVING_SOURCE_SHA256:
            raise SystemExit(
                f"rank {rank.get('rank')} stamps {stamped}, "
                f"not the qualified digest"
            )
    envelope = {
        "schema": "prismaquant.pq2459_serve_census_receipt.v1",
        "artifact": artifact,
        "runtime_image": args.runtime_image,
        "tessera_commit": args.tessera_commit,
        "producer_commit": QUALIFIED_PRODUCER_COMMIT,
        "serving_source_sha256": QUALIFIED_SERVING_SOURCE_SHA256,
        "contract_sha256": QUALIFIED_CONTRACT_SHA256,
        "algorithm": QUALIFIED_ALGORITHM,
        "tensor_parallel_size": TP_DEGREE,
        "execution_mode": "eager",
        "residency": "resident",
        "profiles": sorted(wanted),
        "receipt": receipt,
    }
    out.write_text(json.dumps(envelope, indent=1, sort_keys=True) + "\n")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    if args.mode == "dry-run":
        return run_dry_run(args, out)
    return run_census(args, out)


if __name__ == "__main__":
    sys.exit(main())
