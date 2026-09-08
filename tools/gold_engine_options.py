"""Stock-vLLM topology arguments for the in-process gold coordinator.

Remote nodes use the stock headless CLI. This module starts no processes.
"""
from __future__ import annotations

import argparse


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
    group.add_argument("--moe-backend", choices=("auto", "triton"), default=None)


def gold_engine_kwargs(args: argparse.Namespace) -> dict:
    """Validate before loading; omission preserves the original TP1 kwargs."""
    result = {"tensor_parallel_size": getattr(args, "tensor_parallel_size", 1)}
    for name in ("nnodes", "node_rank", "master_addr", "master_port",
                 "distributed_executor_backend", "data_parallel_backend", "moe_backend"):
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
    if "moe_backend" in result and result["moe_backend"] not in ("auto", "triton"):
        raise ValueError("moe_backend must be auto or triton")
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
