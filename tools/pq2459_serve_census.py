"""PQ #2459 TP2 census entry point, gang role, and dry run.

This module is the single entry point for the D38 CPU dry run and
the GPU qualification census. Both modes use the same parser and
the same driver arguments. They differ only in the role they run.

Roles: ``head`` starts the ray head and runs the census inside the
stock vLLM image. ``worker`` joins the ray cluster and holds the
second rank. ``dry-run`` checks the arguments and the fixture
metadata and writes the exact argv the head would run, with no CUDA.

The member launcher («tools/pq2459_gang_member.sh») runs this module
in its declared role. PrismaBuild admits both members of the gang
together, so each rank holds a real resource admission.

No identity check in this module refuses. Recorded-versus-running
code and image comparisons go through ``prismaquant.dev_mode.seal_check``.
Certified mode refuses there; dev mode stamps ``[DEV-MODE]`` and
continues. Byte integrity, fixture shape, topology, and mode checks
stay hard walls in both modes.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from prismaquant.dev_mode import seal_check  # noqa: E402

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
    REPO / "docs/results/pq2471_fixture_profiles_2026-10-09.json"
)
TP_DEGREE = 2
RAY_PORT = 6379

#: Stock single-node vLLM serve env names that must not leak into a
#: two-node serve. Each would pin a single-rank master or backend.
FORBIDDEN_SINGLE_NODE_ENV = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "VLLM_DP_MASTER_IP",
    "VLLM_DP_MASTER_PORT",
    "VLLM_HOST_IP",
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=("dry-run", "head", "worker"))
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
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--ext-dir", default=None)
    ap.add_argument("--container", default=None)
    ap.add_argument("--peer", default=None)
    ap.add_argument("--head-addr", default=None)
    ap.add_argument("--ray-port", type=int, default=RAY_PORT)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=2049)
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2049)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    ap.add_argument("--kv-cache-dtype", default="fp8_ds_mla")
    ap.add_argument("--moe-backend", default="triton")
    ap.add_argument("--kernel-config", default='{"enable_flashinfer_autotune": false}')
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--language-model-only", action="store_true", default=True)
    ap.add_argument("--draft-routes", action="store_true", default=False)
    ap.add_argument("--speculative-config", default=None)
    return ap


def load_profiles() -> dict:
    """Read the committed fixture profile packet."""
    payload = json.loads(FIXTURE_PROFILES.read_text())
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise SystemExit("fixture packet names no profiles")
    return profiles


def check_artifact(model: Path) -> dict:
    """Verify the artifact identity. Hard wall in both modes."""
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
    if not index_path.exists():
        raise SystemExit(f"no safetensors index under {model}")
    import hashlib

    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "config_sha256": _sha256(cfg_path),
        "index_sha256": _sha256(index_path),
        "groups": len(groups),
    }


def check_topology(args: argparse.Namespace) -> None:
    """Verify the TP degree. Hard wall in both modes."""
    if args.tensor_parallel_size != TP_DEGREE:
        raise SystemExit("this entry point serves TP 2 only")


def check_mode() -> None:
    """Verify resident mode. Hard wall in both modes."""
    if os.environ.get("TESSERA_SERVE_MODE", "resident") != "resident":
        raise SystemExit("TESSERA_SERVE_MODE must be resident")


def check_profile(profile: str, profiles: dict) -> dict:
    """Select the fixture profiles. Hard wall in both modes."""
    if profile == "all":
        return profiles
    if profile not in profiles:
        raise SystemExit(f"unknown fixture profile {profile}")
    return {profile: profiles[profile]}


def check_engine_scope(args: argparse.Namespace) -> None:
    """Verify engine scope values. Hard wall in both modes."""
    if args.prompt_tokens != 2048:
        raise SystemExit("prompt tokens must be 2048 for this census")
    if args.max_model_len != 2049:
        raise SystemExit("max model length must be 2049 for this census")
    if args.max_num_seqs != 1:
        raise SystemExit("max sequences must be 1 for this census")
    if args.max_num_batched_tokens != 2049:
        raise SystemExit("max batched tokens must be 2049 for this census")
    if args.gpu_memory_utilization != 0.5:
        raise SystemExit("GPU memory use must be 0.5 for this census")
    if args.kv_cache_memory_bytes != 1073741824:
        raise SystemExit("KV cache bytes must be 1 GiB for this census")
    if args.kv_cache_dtype != "fp8_ds_mla":
        raise SystemExit("KV cache dtype must be fp8_ds_mla for this census")
    if args.moe_backend != "triton":
        raise SystemExit("MoE backend must be triton for this census")
    if json.loads(args.kernel_config) != {"enable_flashinfer_autotune": False}:
        raise SystemExit("kernel config must disable flashinfer autotune")
    if not args.trust_remote_code:
        raise SystemExit("trust remote code must stay on for this census")
    if not args.language_model_only:
        raise SystemExit("language model only must stay on for this census")
    if args.draft_routes:
        raise SystemExit("draft routes stay off; the decode attestation needs the one-row forward")
    if args.speculative_config is not None:
        raise SystemExit("speculative config stays unset without draft routes")


def check_no_single_node_env() -> None:
    """Refuse leaked single-node rendezvous settings. Hard wall."""
    leaked = [name for name in FORBIDDEN_SINGLE_NODE_ENV
              if os.environ.get(name)]
    if leaked:
        raise SystemExit(
            "single-node rendezvous leaks into the TP2 serve: "
            + ", ".join(sorted(leaked)))


def seal_image(image: str) -> bool:
    """Compare the declared image with the permitted one. A D32 seal."""
    return seal_check(
        "runtime image", RUNTIME_IMAGE, image,
        where="PQ #2459 census runtime image",
        refusal=lambda: SystemExit(f"unpermitted runtime image {image}"))


def seal_commit(commit: str) -> bool:
    """Compare the named commit with the qualified one. A D32 seal."""
    return seal_check(
        "serving commit", QUALIFIED_SERVING_COMMIT, commit,
        where="PQ #2459 census serving commit",
        refusal=lambda: SystemExit(
            f"unqualified serving commit {commit}; "
            f"this entry point serves only {QUALIFIED_SERVING_COMMIT}"))


def census_argv(args: argparse.Namespace, *, trace_path: str) -> list[str]:
    """The exact census argv the head runs inside its container."""
    argv = [
        "python3", "tools/tessera_route_census.py", args.model, args.out,
        "--runtime-image", args.runtime_image,
        "--tessera-commit", args.tessera_commit,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--distributed-executor-backend", "ray",
        "--prompt-tokens", str(args.prompt_tokens),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes),
        "--kv-cache-dtype", args.kv_cache_dtype,
        "--moe-backend", args.moe_backend,
        "--kernel-config", args.kernel_config,
    ]
    if args.trust_remote_code:
        argv.append("--trust-remote-code")
    if args.language_model_only:
        argv.append("--language-model-only")
    if args.draft_routes:
        argv.append("--draft-routes")
        argv.extend(["--speculative-config", args.speculative_config or ""])
    return argv


def head_env(args: argparse.Namespace, runs: Path) -> dict[str, str]:
    """Environment of the head census, including the trace path."""
    trace = runs / "trace-head.json"
    env = dict(os.environ)
    env["TESSERA_ROUTE_TRACE"] = str(trace)
    return env


def run_dry_run(args: argparse.Namespace, out: Path) -> int:
    """Check every argument and emit the exact head argv. No CUDA."""
    profiles = load_profiles()
    wanted = check_profile(args.profile, profiles)
    check_topology(args)
    check_mode()
    check_engine_scope(args)
    check_no_single_node_env()
    image_ok = seal_image(args.runtime_image)
    commit_ok = seal_commit(args.tessera_commit)
    artifact = check_artifact(Path(args.model))
    runs = Path(args.runs_dir or "/tmp/pq2459-runs")
    trace = runs / "trace-head.json"
    argv = census_argv(args, trace_path=str(trace))
    manifest = {
        "schema": "prismaquant.pq2459_serve_census_dry_run.v2",
        "mode": "dry-run",
        "model": args.model,
        "artifact": artifact,
        "runtime_image": args.runtime_image,
        "runtime_image_sealed": image_ok,
        "tessera_commit": args.tessera_commit,
        "serving_commit_sealed": commit_ok,
        "producer_commit": QUALIFIED_PRODUCER_COMMIT,
        "serving_source_sha256": QUALIFIED_SERVING_SOURCE_SHA256,
        "source_files": QUALIFIED_SOURCE_FILES,
        "algorithm": QUALIFIED_ALGORITHM,
        "tensor_parallel_size": args.tensor_parallel_size,
        "engine_scope": {
            "prompt_tokens": args.prompt_tokens,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "kv_cache_dtype": args.kv_cache_dtype,
            "moe_backend": args.moe_backend,
            "kernel_config": json.loads(args.kernel_config),
            "trust_remote_code": args.trust_remote_code,
            "language_model_only": args.language_model_only,
            "draft_routes": args.draft_routes,
            "speculative_config": (json.loads(args.speculative_config) if args.speculative_config else None),
        },
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
        "census_tool": "tools/tessera_route_census.py",
        "head_argv": argv,
        "head_trace": str(trace),
        "qualified_cells": 0,
    }
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=1, sort_keys=True))
    return 0


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, **kwargs)


def _remove_container(name: str) -> None:
    _run(["docker", "rm", "-f", name],
         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _ray_alive(container: str, tries: int = 60) -> bool:
    for _ in range(tries):
        proc = _run(["docker", "exec", container, "ray", "status"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc.returncode == 0:
            return True
        listed = _run(["docker", "ps", "-q", "-f", f"name={container}"],
                      capture_output=True, text=True)
        if not listed.stdout.strip():
            return False
        time.sleep(5)
    return False


def _cluster_size(container: str) -> int:
    proc = _run(
        ["docker", "exec", container, "python3", "-c",
         "import ray; ray.init(address='auto'); "
         "print(sum(1 for n in ray.nodes() if n['Alive']))"],
        capture_output=True, text=True)
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return 0


def run_head(args: argparse.Namespace, out: Path) -> int:
    """Start the ray head, run the census, stop the container."""
    check_topology(args)
    check_mode()
    check_engine_scope(args)
    check_no_single_node_env()
    seal_image(args.runtime_image)
    seal_commit(args.tessera_commit)
    profiles = load_profiles()
    check_profile(args.profile, profiles)
    check_artifact(Path(args.model))
    if not args.container:
        raise SystemExit("head mode needs --container")
    if not args.head_addr:
        raise SystemExit("head mode needs --head-addr")
    env = head_env(args, Path(args.runs_dir or "/tmp/pq2459-runs"))
    trace = env["TESSERA_ROUTE_TRACE"]
    argv = census_argv(args, trace_path=trace)
    quoted = " ".join(f"'{word}'" for word in argv)
    inner = (
        f"TESSERA_ROUTE_TRACE='{trace}' "
        f"TESSERA_CENSUS_RUNTIME_IMAGE='{args.runtime_image}' "
        f"{quoted}"
    )
    try:
        if not _ray_alive(args.container):
            raise SystemExit("ray head never answered in its container")
        for _ in range(60):
            if _cluster_size(args.container) >= TP_DEGREE:
                break
            time.sleep(5)
        else:
            raise SystemExit("the gang worker never joined the cluster")
        proc = _run(["docker", "exec", args.container, "bash", "-c", inner])
        if proc.returncode != 0:
            kept = Path(str(out) + ".refused.json")
            try:
                kept.write_bytes(out.read_bytes())
            except OSError:
                pass
            raise SystemExit(f"census tool exits {proc.returncode}")
    finally:
        _remove_container(args.container)
    return _wrap_receipt(args, out, trace)


def _wrap_receipt(args: argparse.Namespace, out: Path, trace: str) -> int:
    receipt = json.loads(out.read_text())
    ranks = receipt.get("ranks", [])
    if len(ranks) != TP_DEGREE:
        raise SystemExit(f"census covers {len(ranks)} ranks, not TP 2")
    envelope = {
        "schema": "prismaquant.pq2459_serve_census_receipt.v2",
        "runtime_image": args.runtime_image,
        "tessera_commit": args.tessera_commit,
        "producer_commit": QUALIFIED_PRODUCER_COMMIT,
        "serving_source_sha256": QUALIFIED_SERVING_SOURCE_SHA256,
        "contract_sha256": QUALIFIED_CONTRACT_SHA256,
        "algorithm": QUALIFIED_ALGORITHM,
        "tensor_parallel_size": TP_DEGREE,
        "execution_mode": "eager",
        "residency": "resident",
        "profiles": [args.profile],
        "trace": trace,
        "receipt": receipt,
    }
    out.write_text(json.dumps(envelope, indent=1, sort_keys=True) + "\n")
    return 0


def run_worker(args: argparse.Namespace, out: Path) -> int:
    """Join the ray cluster and hold the second rank."""
    check_topology(args)
    check_mode()
    seal_image(args.runtime_image)
    seal_commit(args.tessera_commit)
    if not args.container:
        raise SystemExit("worker mode needs --container")
    if not args.head_addr:
        raise SystemExit("worker mode needs --head-addr")
    try:
        proc = _run(["docker", "exec", args.container, "bash", "-c",
                     f"ray start --address='{args.head_addr}:{args.ray_port}' --block"],
                    timeout=25 * 60)
        return proc.returncode
    except subprocess.TimeoutExpired:
        return 0
    finally:
        _remove_container(args.container)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    if args.mode == "dry-run":
        return run_dry_run(args, out)
    if args.mode == "head":
        return run_head(args, out)
    return run_worker(args, out)


if __name__ == "__main__":
    sys.exit(main())
