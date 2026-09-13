#!/usr/bin/env python3
"""GB10 / sm121 hardening, lifted verbatim from the MiaAI-Lab overlay Dockerfile.

Both targets were confirmed present in this image before porting.
  1. FlashInfer's sparse-MLA autotune warmup wedges rank 0 on SM121, and the
     fused_moe gemm autotune does the same; skip both.
  2. PDL lowering races the KDA state kernels on SM12x; gate PDL to Hopper/SM10x.
"""

from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
applied = []

warmup = SITE / "model_executor/warmup/kernel_warmup.py"
text = warmup.read_text()
old_sparse_warmup = (
    "    flashinfer_sparse_mla_decode_autotune_warmup(worker)\n"
    "    deepseek_v4_sparse_mla_attention_warmup(worker)\n"
)
if text.count(old_sparse_warmup) != 1:
    raise SystemExit("expected one FlashInfer sparse-MLA warmup target")
text = text.replace(
    old_sparse_warmup,
    "    # GLM53_SKIP_FI_SPARSE_WARMUP: SM120 autotune wedges rank 0 on GB10.\n"
    "    deepseek_v4_sparse_mla_attention_warmup(worker)\n",
)
old_autotune = "    from flashinfer.autotuner import AutoTuner, set_autotune_process_group\n"
if text.count(old_autotune) != 1:
    raise SystemExit("expected one FlashInfer autotuner import")
text = text.replace(
    old_autotune,
    '    logger.info_once("Skipping FlashInfer autotune on SM121")\n'
    "    return\n"
    "    from flashinfer.autotuner import AutoTuner, set_autotune_process_group\n",
)
warmup.write_text(text)
applied.append("skip_flashinfer_autotune")

platform = SITE / "platforms/cuda.py"
text = platform.read_text()
old_pdl = "            return False\n        return major >= 9\n"
if text.count(old_pdl) != 1:
    raise SystemExit("expected one PDL capability gate")
platform.write_text(
    text.replace(
        old_pdl,
        "            return False\n"
        "        # PDL lowering races KDA state kernels on SM12x (GB10).\n"
        "        return major in (9, 10)\n",
    )
)
applied.append("pdl_gate_9_10")
print("PATCHED gb10 hardening:", ", ".join(applied))
