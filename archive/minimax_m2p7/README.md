# MiniMax-M2.7 prismaquant artifact track (archived 2026-04-24)

This folder preserves the MiniMax-M2.7-specific code and documentation
from the 2026-04-21 through 2026-04-24 work that produced the
`/models/pq-minimax-m2p7-v2/exported/` artifact at 3.25 bpp.

## Why archived

The MiniMax track required a custom vLLM kernel path (NVINT2/NVINT3
grouped-expert GEMM in `spark-vllm-fresh/mods/draft-mxfp6-nvint/`)
because sub-4-bit MoE quantization has no first-class support in vLLM's
stock compressed-tensors dispatch. That kernel reached 3.78 tok/s end-
to-end on MiniMax-M2.7 decode after a 3-stream byte-aligned load
rewrite (7.5× over the per-expert baseline), but the path carries a
significant maintenance burden:

- ~1500 lines of Triton + MoE dispatch patches that live outside
  upstream vLLM, requiring re-application with every image rebuild.
- "NVINT" is a non-standard format name that collides conceptually
  with NVFP4 and doesn't have vendor-backed hardware acceleration —
  the kernel falls back to bf16 Tensor Cores, leaving Blackwell's
  FP4 TC throughput on the table.
- Memory for MiniMax at NVFP4-only was just over GB10's 120 GB budget
  (~151 GB), which is why sub-4-bit was attempted in the first place.

The joint-optimizer track (REAP-style expert saliency pruning composed
with standard-format quantization) supersedes this for most practical
uses: it keeps the model in hardware-accelerated formats (NVFP4 +
MXFP8 + FP8_SOURCE + BF16) and fits inside the same memory budget by
dropping low-saliency experts. See the `prismaquant.allocator` changes
landing alongside this archive commit for the new path.

Custom sub-4-bit kernels remain valuable when expert pruning isn't
acceptable (e.g., applications sensitive to the non-uniform quality
degradation pruning introduces), but they are no longer the default
recommendation for memory-constrained MoE deployment.

## Contents

- `minimax_m2.py` — the MoE profile that routes MiniMax-M2/M2.7 to the
  NVINT-specific streaming + export paths. Moved from
  `prismaquant/model_profiles/minimax_m2.py`. To restore: copy back to
  `model_profiles/` and re-add the import + `_REGISTERED` entry in
  `model_profiles/registry.py`.
- `handover_2026-04-24.md` — the end-of-session handover doc from codex
  describing artifact state, known vLLM compat fixes applied, and
  performance findings through the grouped-kernel milestone.

## Related commits (kept in mainline, not archived)

These changes were motivated by MiniMax but generalize to any
FP8-sourced MoE (DeepSeek-V3 family, NVIDIA FP8 releases, etc.) and
remain in `prismaquant/` mainline:

- **FP8_SOURCE passthrough format + exporter path**
  (`format_registry.py`, `export_native_compressed.py`) — lossless
  copy of `.weight` + `.weight_scale_inv` pairs; used whenever the
  allocator picks FP8_SOURCE.
- **Streaming FP8-native checkpoint loader** (`streaming_model.py`,
  `layer_streaming.py`) — dequants source `weight × weight_scale_inv`
  on the per-layer streaming path so downstream RTN cost+probe see
  true BF16 values. Contains a `_minimax_native_fp8_checkpoint()`
  helper that remains because the same load pattern applies to
  any checkpoint whose remote modeling file bypasses transformers'
  standard FP8 integration.
- **Passthrough-integrity allocator filter** (`allocator.py`
  `PASSTHROUGH_SOURCE_REQUIREMENTS`) — prevents the DP from picking
  FP8_SOURCE on BF16-source Linears (or BF16 on FP8-source Linears).
  General-purpose safety net for any mixed-source checkpoint.

## Related spark-vllm-fresh commits (separate repo; active)

The custom Triton kernel still lives at
`spark-vllm-fresh/mods/draft-mxfp6-nvint/` on the main branch. Final
commits for reference:

- `cfb104e` grouped-expert MoE GEMM (5× decode speedup)
- `48f999d` CUDA-graph compat (`scatter_add` over `bincount`)
- `ec49c4f` 3-stream byte-aligned NVINT3 load (+2.5× kernel)
- `525940d` apply 3-stream to dense NVINT3 kernel too
- `13d716b` cosmetic byte-aligned rewrite of NVINT2 kernels

## Final MiniMax-M2.7 measurements (for posterity)

- Source: `MiniMaxAI/MiniMax-M2.7`, 160 GB FP8 native
- Artifact: `/models/pq-minimax-m2p7-v2/exported/` at 3.25 bpp
- Mix: 22,295 NVINT3 + 18,649 NVINT2 + 4,613 NVFP4 + 2,308 FP8_SOURCE
- End-to-end decode (GB10, vLLM + CUDA graphs):
  - Per-expert legacy apply: 0.5 tok/s
  - Grouped-expert kernel (cfb104e): 2.2 tok/s
  - Grouped + 3-stream NVINT3 load (ec49c4f + 525940d): **3.78 tok/s**
- GPU residency: 92.3 GiB weights + 6.4 GiB KV cache + ~1 GiB CUDA
  graph pool = ~100 GiB on GB10's 120 GB budget.

The artifact + serve wrapper scripts remain at their original paths:
- `/home/rob/spark-vllm-fresh/tools/serve-minimax-pq-a16-safe.sh`
- `/home/rob/spark-vllm-fresh/tools/serve-minimax-pq-a16-profile.sh`
