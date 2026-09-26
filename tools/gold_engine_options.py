"""Stock-vLLM topology arguments for the in-process gold coordinator.

Remote nodes use the stock headless CLI. This module starts no processes.

Three things live here, and the reason they share a file is that all three
describe ONE engine: the topology rank 0 builds its `LLM` with, the argv the
peer rank must be launched with to join it, and the collective fabric the
operator asked for. Typing any of them a second time somewhere else is how a
receipt comes to disagree with the run it describes.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Mapping

try:  # package mode (`python -m tools.measure_vllm_full_kl`)
    from .serve_fingerprint import NCCL_FABRIC_ENV
except ImportError:  # script mode (`python /repo/tools/measure_vllm_full_kl.py`)
    from serve_fingerprint import NCCL_FABRIC_ENV  # type: ignore

#: The collective-library environment a gold measurement's fabric is REQUESTED
#: through, re-exported from `serve_fingerprint` so that the names this module
#: RECORDS and the names the fingerprint's `SERVER_ENV_ALLOWLIST` PROJECTS are
#: one tuple and cannot drift apart. They are NCCL's own variable names, read
#: from this process's environment and recorded as an operator request -- never
#: as an observation of what NCCL then did. The observation is NCCL's
#: `NET/Socket` vs `NET/IB` line on the serving process's own log, which this
#: tool does not read; principle 14 forbids turning one into the other by
#: naming.
__all__ = [
    "NCCL_FABRIC_ENV",
    "GOLD_FABRIC_SCHEMA",
    "add_gold_engine_arguments",
    "gold_engine_kwargs",
    "gold_fabric_request",
    "gold_provenance",
    "headless_peer_argv",
    "validate_gold_engine_arguments",
]

GOLD_FABRIC_SCHEMA = "prismaquant.gold_fabric_request/1"


def add_gold_engine_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("stock gold-engine topology")
    group.add_argument("--tensor-parallel-size", type=int, default=1)
    group.add_argument("--nnodes", type=int, default=None)
    group.add_argument("--node-rank", type=int, default=None,
                       help="gold runs on rank 0; launch other nodes with stock vllm serve --headless")
    group.add_argument("--master-addr", default=None)
    group.add_argument("--master-port", type=int, default=None)
    group.add_argument("--distributed-executor-backend", choices=("mp",), default=None)
    group.add_argument("--data-parallel-backend", choices=("mp",), default=None)
    group.add_argument("--moe-backend",
                       choices=("auto", "triton", "flashinfer_cutlass"), default=None,
                       help="explicit stock-vLLM MoE backend; the menu carries "
                            "the names this tool will pass through, not a claim "
                            "that a serving path requires one")
    group.add_argument("--kv-cache-memory-bytes", type=int, default=None,
                       help="explicit KV cache byte bound per rank; omitted means vLLM "
                            "sizes KV from --gpu-memory-utilization")


def gold_engine_kwargs(args: argparse.Namespace) -> dict:
    """Validate before loading; omission preserves the original TP1 kwargs.

    The explicit positive KV byte bound and the widened backend menu are
    selections this tool will pass through and validate; whether a given serving
    path needs one is not decided here and is not qualified by this validation.
    Every absent option stays absent rather than being stated as `None`, so an
    omitted-argument run produces exactly the kwargs it produced before they
    existed.
    """
    result = {"tensor_parallel_size": getattr(args, "tensor_parallel_size", 1)}
    for name in ("nnodes", "node_rank", "master_addr", "master_port",
                 "distributed_executor_backend", "data_parallel_backend", "moe_backend",
                 "kv_cache_memory_bytes"):
        value = getattr(args, name, None)
        if value is not None:
            result[name] = value
    tp, nodes = result["tensor_parallel_size"], result.get("nnodes", 1)
    for name, value in (("tensor_parallel_size", tp), ("nnodes", nodes)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if tp % nodes:
        raise ValueError("nnodes must evenly divide tensor_parallel_size (DP/PP/PCP stay 1)")
    rank = result.get("node_rank", 0)
    if type(rank) is not int or rank != 0:
        raise ValueError("gold runs only on node_rank 0; use stock headless workers for other nodes")
    if "master_addr" in result and (
            not isinstance(result["master_addr"], str) or not result["master_addr"].strip()):
        raise ValueError("master_addr must be a nonempty host address")
    if "master_port" in result and (
            type(result["master_port"]) is not int or not 1 <= result["master_port"] <= 65535):
        raise ValueError("master_port must be an integer in 1..65535")
    for name in ("distributed_executor_backend", "data_parallel_backend"):
        if name in result and result[name] != "mp":
            raise ValueError(f"{name} must be mp when explicitly selected")
    if "moe_backend" in result and result["moe_backend"] not in (
            "auto", "triton", "flashinfer_cutlass"):
        raise ValueError("moe_backend must be auto, triton or flashinfer_cutlass")
    if "kv_cache_memory_bytes" in result and (
            type(result["kv_cache_memory_bytes"]) is not int
            or result["kv_cache_memory_bytes"] <= 0):
        raise ValueError("kv_cache_memory_bytes must be a positive integer byte bound")
    if nodes > 1:
        required = ("master_addr", "master_port", "distributed_executor_backend", "data_parallel_backend")
        missing = [name for name in required if name not in result]
        if missing:
            raise ValueError(f"multi-node gold requires explicit {', '.join(missing)}")
    return result


def validate_gold_engine_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        gold_engine_kwargs(args)
    except ValueError as exc:
        parser.error(str(exc))


def gold_fabric_request(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The collective fabric this measurement was ASKED for, from its own env.

    A TP>1 number is not readable without its fabric: on the Spark pair RoCE
    `ibv_reg_mr` fails with ENOMEM often enough that quality evidence is run on
    sockets and labelled as such, and a receipt that omits which one ran cannot
    be told apart from one that did the other. This records the request so the
    label travels with the number.

    It is deliberately a REQUEST and says so. NCCL decides its transport at
    init and announces it on the serving process's log (`NET/Socket` vs
    `NET/IB`); this function reads environment variables instead, which is an
    operator intent. Principle 14: a claim about what another runtime *did* is
    attested from that runtime's own output or it is not made, so the field is
    named and typed as intent rather than promoted to an observation.

    `transport` is derived only from `NCCL_IB_DISABLE`, whose two meaningful
    values are documented by NCCL itself; anything else is reported verbatim as
    `declared_other` rather than being mapped onto one of the two by guess.
    """
    values = dict(os.environ if environ is None else environ)
    selected = {name: values[name] for name in NCCL_FABRIC_ENV if name in values}
    disable = selected.get("NCCL_IB_DISABLE")
    if disable is None:
        transport = "unset_nccl_default"
    elif disable.strip() == "1":
        transport = "sockets"
    elif disable.strip() == "0":
        transport = "ib_or_roce"
    else:
        transport = "declared_other"
    return {
        "schema": GOLD_FABRIC_SCHEMA,
        "observation": "requested_not_observed",
        "transport": transport,
        "values": selected,
        "observed_by": (
            "NCCL's own NET/Socket or NET/IB line on the serving process log; "
            "this block is the environment request, not that observation"
        ),
    }


#: How each engine kwarg rank 0 builds its `LLM` with is spelled on the stock
#: `vllm serve` command line. Only keys listed here can be forwarded to a peer;
#: `headless_peer_argv` REFUSES an unknown one rather than dropping it, because
#: a peer silently missing one of the coordinator's engine arguments joins with
#: a different engine and measures neither configuration.
_PEER_FLAG_SPELLING = {
    "tensor_parallel_size": "--tensor-parallel-size",
    "nnodes": "--nnodes",
    "master_addr": "--master-addr",
    "master_port": "--master-port",
    "distributed_executor_backend": "--distributed-executor-backend",
    "data_parallel_backend": "--data-parallel-backend",
    "moe_backend": "--moe-backend",
    "dtype": "--dtype",
    "kv_cache_dtype": "--kv-cache-dtype",
    "kv_cache_memory_bytes": "--kv-cache-memory-bytes",
    "max_model_len": "--max-model-len",
    "max_num_seqs": "--max-num-seqs",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "max_logprobs": "--max-logprobs",
    "quantization": "--quantization",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "logprobs_mode": "--logprobs-mode",
}

#: Boolean kwargs whose stock spelling is a bare flag when true. `False` emits
#: the negated spelling where vLLM publishes one, so "off" is stated rather
#: than left to the peer's default.
_PEER_BOOLEAN_SPELLING = {
    "trust_remote_code": ("--trust-remote-code", None),
    "enforce_eager": ("--enforce-eager", None),
    "disable_log_stats": ("--disable-log-stats", None),
    "language_model_only": ("--language-model-only", None),
    "enable_prefix_caching": (
        "--enable-prefix-caching", "--no-enable-prefix-caching"),
    "enable_chunked_prefill": (
        "--enable-chunked-prefill", "--no-enable-chunked-prefill"),
}

#: Kwargs that are rank 0's alone and must never be forwarded: the model is a
#: positional, and the coordinator's own rank is not the peer's.
_PEER_POSITIONAL_OR_LOCAL = frozenset({"model", "node_rank"})


def headless_peer_argv(
    kwargs: Mapping[str, Any],
    *,
    node_rank: int,
) -> list[str]:
    """The stock `vllm serve --headless` argv for a peer of THIS coordinator.

    Derived from the very `kwargs` dict rank 0 passes to `LLM(**kwargs)`, so
    the peer cannot drift from the coordinator by being retyped. Returns argv
    WITHOUT the leading `vllm`, ready for `tools/gold_headless_peer.py`.

    Fails closed on a kwarg it has no published spelling for: forwarding a
    guess would be worse than refusing, and dropping it silently is how the
    two ranks come to run different engines.
    """
    if isinstance(node_rank, bool) or not isinstance(node_rank, int) or node_rank < 1:
        raise ValueError(
            "a headless peer is rank >= 1; rank 0 is the in-process coordinator")
    model = kwargs.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("peer argv requires the coordinator's model path")

    argv = ["serve", model, "--node-rank", str(node_rank), "--headless"]
    for name in sorted(kwargs):
        if name in _PEER_POSITIONAL_OR_LOCAL:
            continue
        value = kwargs[name]
        if value is None:
            continue
        if name in _PEER_BOOLEAN_SPELLING:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool, got {value!r}")
            on, off = _PEER_BOOLEAN_SPELLING[name]
            if value:
                argv.append(on)
            elif off is not None:
                argv.append(off)
            continue
        flag = _PEER_FLAG_SPELLING.get(name)
        if flag is None:
            raise ValueError(
                f"no published stock-CLI spelling for engine kwarg {name!r}: "
                "add it to _PEER_FLAG_SPELLING once the pinned vLLM's flag is "
                "known, rather than launching a peer without it"
            )
        argv.extend([flag, str(value)])
    return argv


def gold_provenance(tool: str, args: argparse.Namespace, *, engine_kwargs,
                    spec_decode_detected, producer_identity, self_manifest,
                    serve_image) -> dict:
    """Serving-stack + code provenance for one gold result dict (R15).

    The one owner for both in-process gold tools (PQ #1302). Each tool passes
    its own module globals and helpers, so a test that replaces one of them
    on the tool still reaches this body.

    A TP>1 number is not readable without the fabric it crossed, and the
    fabric is an environment request rather than an engine argument, so it
    rides beside the topology instead of inside it. On a single node there is
    no collective to label, but the block is still recorded: "this ran on one
    box" is the honest reading of an absent fabric, and omitting the key would
    make a TP1 receipt and an unlabelled TP2 receipt look alike.
    """
    producer = producer_identity(tool)
    topology = gold_engine_kwargs(args)
    extra = {
        "measurement_tool": tool,
        "producer_identity": producer,
        "gold_engine_configuration": topology,
        "gold_fabric_request": gold_fabric_request(),
    }
    if int(topology.get("nnodes", 1)) > 1 and engine_kwargs is not None:
        # The peer argv this coordinator's own kwargs imply. Recorded so the
        # launcher that started rank 1 can be checked against the engine rank 0
        # actually built, rather than trusted because both were typed by hand.
        extra["headless_peer_argv"] = headless_peer_argv(
            engine_kwargs, node_rank=1)
    manifest = self_manifest(
        extra=extra,
        image=serve_image(args),
    )
    return {
        "git_commit": producer["git_commit"],
        "serve_fingerprint": manifest["serve_fingerprint"],
        "serve_manifest": manifest,
        "spec_decode_detected": spec_decode_detected,
    }
