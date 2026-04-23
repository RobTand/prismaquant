# Vectorization Refactor Milestone

This document tracks the large-scale vectorization work needed for MiniMax-M2.7 and other MoE-heavy checkpoints. The current pipeline is correct enough to debug exports, but the process is still too Python-heavy and file-I/O-heavy for a model with tens of thousands of Linear-equivalent tensors.

## Goals

- Use one grouped execution path for all model sizes. Small models should run through the same code as large models; the knobs should tune chunk size and memory pressure, not select a separate implementation.
- Keep GPU work coarse-grained. Avoid per-Linear CUDA synchronizations, repeated allocator cache flushes, and tiny matmuls in Python loops.
- Keep native source formats intact. Source FP8 and BF16 passthrough must remain byte/semantic-preserving when the allocator selects `FP8_SOURCE` or `BF16`.
- Make MiniMax debugging easier before cleanup. Prefer explicit logs, simple correctness tests, and fail-fast checks over clever abstractions while the model is not yet known-good.

## Current Bottlenecks

- Probe: MiniMax routed experts can devolve into Python `ModuleList` expert loops. The fast-MoE path fixes non-target layers, but target-layer Fisher collection still uses hook-style per-Linear accounting.
- Cost: grouped weight and activation tensors already exist, but result extraction used per-Linear `.item()` calls and `torch.cuda.empty_cache()` inside the format loop. That serializes GPU progress and starves larger chunks.
- Activation cache: probe writes thousands of small `.pt` files. Cost and export then perform many small CPU loads instead of reading per-layer bundles.
- Export: raw activation loads are now lazy, but final tensor materialization still grows a large in-memory output dict before sharded safetensors writing.

## Refactor Plan

1. Cost hot path: keep grouped quantization, move metrics back to CPU once per format/chunk, and avoid inner-loop cache flushes.
2. Export hot path: group same-format/same-shape quantization within each layer and stream shards instead of retaining the whole checkpoint dict.
3. Activation store: replace per-Linear files with per-layer bundles that can be prefetched and sliced cheaply.
4. Probe hot path: collect routed-expert Fisher statistics as grouped tensors for MiniMax-style packed and unpacked MoE modules.
5. Validation: keep golden tests for scalar equivalence, passthrough FP8/BF16 behavior, activation lazy loading, allocator schema, and toy MiniMax routing.

## Open Cleanup Note

The current process still needs major optimization after MiniMax successfully runs end-to-end. The immediate priority is debuggable correctness; once the export is validated, the activation bundle and streaming-writer work should become the next milestone.
