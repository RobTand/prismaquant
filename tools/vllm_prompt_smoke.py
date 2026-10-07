#!/usr/bin/env python3
"""Small vLLM prompt smoke for BF16 or compressed-tensors artifacts."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--distributed-executor-backend", choices=("mp", "ray"))
    parser.add_argument("--cpu-preflight", action="store_true",
                        help="Validate runtime arguments without importing vLLM or using a device.")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--quantization", default=None,
                        help="Optional vLLM quantization argument.")
    parser.add_argument("--no-enforce-eager", action="store_true",
                        help="Allow graph/compiled execution.")
    parser.add_argument("--output-json", default=None,
                        help="Optional path for a compact JSON result.")
    args = parser.parse_args()
    if args.tensor_parallel_size < 1 or args.max_model_len < 1 or args.max_new_tokens < 1:
        parser.error("parallel size, model length and token count must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be greater than zero and less than one")

    t0 = time.time()
    kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "enforce_eager": not args.no_enforce_eager,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": 1,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
    }
    if args.distributed_executor_backend:
        kwargs["distributed_executor_backend"] = args.distributed_executor_backend
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.cpu_preflight:
        print(json.dumps({"status": "cpu_preflight_only", "llm_kwargs": kwargs}, indent=2))
        return 0
    from vllm import LLM, SamplingParams

    llm = LLM(**kwargs)
    init_seconds = time.time() - t0

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    t1 = time.time()
    outputs = llm.generate([args.prompt], sampling)
    generate_seconds = time.time() - t1
    texts = [item.outputs[0].text for item in outputs]
    result = {
        "model": args.model,
        "prompt": args.prompt,
        "outputs": texts,
        "init_seconds": init_seconds,
        "generate_seconds": generate_seconds,
        "mode": "graph" if args.no_enforce_eager else "eager",
        "quantization": args.quantization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "distributed_executor_backend": args.distributed_executor_backend,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
