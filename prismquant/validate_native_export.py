#!/usr/bin/env python3
"""validate_native_export.py — load a compressed-tensors checkpoint via
vLLM and do a single forward + greedy decode. Binary check: either
vLLM accepts the format and produces tokens, or it doesn't.

Usage (from inside a vllm-node container):
    python -m prismquant.validate_native_export \\
        --model dq-runs-new/qwen36-fresh/exported \\
        --prompt "The capital of France is" \\
        --max-new-tokens 16

The script can optionally upgrade the container's flashinfer to a
specific version before loading; this is needed for some vLLM builds
that ship with a flashinfer that can't dispatch the NVFP4 MoE backend
on Blackwell. Pass `--no-flashinfer-upgrade` to skip.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def maybe_upgrade_flashinfer(version: str) -> None:
    """Upgrade flashinfer-python and flashinfer-cubin to `version` and
    set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass the AOT-cache pin
    that lags behind PyPI. No-op if already at the target version.
    """
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    try:
        import flashinfer
        if getattr(flashinfer, "__version__", "0.0") == version:
            return
    except ImportError:
        pass
    print(f"[validate] upgrading flashinfer-python + flashinfer-cubin "
          f"to {version}", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade", "-q",
        f"flashinfer-python=={version}",
        f"flashinfer-cubin=={version}",
    ])


def summarize_quantization_config(cfg_path: Path) -> None:
    cfg = json.load(open(cfg_path))
    qc = cfg.get("quantization_config", {})
    print(f"[validate] quant_method: {qc.get('quant_method', '<missing>')}")
    print(f"[validate] format:       {qc.get('format', '<missing>')}")
    for gn, g in qc.get("config_groups", {}).items():
        w = g.get("weights", {})
        print(f"[validate]   {gn}: bits={w.get('num_bits')} "
              f"strategy={w.get('strategy')} group={w.get('group_size')} "
              f"format={g.get('format')} n_targets={len(g.get('targets', []))}")
    print(f"[validate]   ignore: {len(qc.get('ignore', []))} entries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Compressed-tensors checkpoint directory.")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--flashinfer-version", default="0.6.8.post1",
                    help="Pinned flashinfer-python + flashinfer-cubin "
                         "version installed via pip before loading vLLM.")
    ap.add_argument("--no-flashinfer-upgrade", action="store_true",
                    help="Skip the flashinfer pre-flight upgrade.")
    ap.add_argument("--speculative-config", default=None,
                    help="JSON string for vLLM SpeculativeConfig. Use this "
                         "to exercise MTP heads, e.g. "
                         "'{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": 3, "
                         "\"model\": \"<same model dir>\"}'.")
    args = ap.parse_args()

    if not args.no_flashinfer_upgrade:
        maybe_upgrade_flashinfer(args.flashinfer_version)

    model_dir = Path(args.model)
    summarize_quantization_config(model_dir / "config.json")

    print(f"[validate] starting vLLM ...", flush=True)
    from vllm import LLM, SamplingParams
    spec = None
    if args.speculative_config:
        spec = json.loads(args.speculative_config)
        # If caller omitted "model", default to the same checkpoint — MTP
        # weights travel with the target model.
        if "model" not in spec:
            spec["model"] = str(model_dir)
        print(f"[validate] speculative config: {spec}", flush=True)
    llm = LLM(
        model=str(model_dir),
        quantization="compressed-tensors",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        speculative_config=spec,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    out = llm.generate([args.prompt], sp)
    print(f"[validate] generated:", flush=True)
    for o in out:
        print(f"  prompt: {o.prompt!r}", flush=True)
        print(f"  output: {o.outputs[0].text!r}", flush=True)


if __name__ == "__main__":
    main()
