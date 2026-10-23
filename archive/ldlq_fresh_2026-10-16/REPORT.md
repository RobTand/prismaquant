# Fixed-CB LDLQ genuinely-fresh-text validation

Date: 2026-08-03

GPU: NVIDIA GB10, torch 2.11.0+cu130

## Verdict

**VALIDATED-PENDING-SERVED.** Mean fresh-text reduction is 74.06% versus 94.93% in-sample, retaining 78.02% of the fit-set gain. The same-capture CAL32/HOLDOUT32 reference averages 80.27%. Cell verdicts: 9 validated-pending-served, 0 partial, and 0 distribution-fragile.

The fixed rule is applied to fresh-gain retention: more than 50% is `VALIDATED-PENDING-SERVED`, less than 10% is `DISTRIBUTION-FRAGILE`, and the interval between is `PARTIAL`.

## Fresh capture and disjointness

The production probe is identified by its stored metadata and reproduced loader hash `2682b690de20c091568729be8f78baf7`: `diverse-v1.jsonl`, 16 samples, sequence length 512, calibration seed 42. For a local JSONL, the historical seed-42 branch preserves file order. All first 16 text records are longer than 512 checkpoint-tokenizer tokens, so those records (source lines 2-17, text indices 0-15) are exactly the records production consumed. The fresh corpus contains the next 16 records only (source lines 18-33, text indices 16-31). Their full-text SHA-256 set has zero intersection with the production set; the per-record hashes and source-file hash are in [`text_manifest.json`](text_manifest.json). This proves record-level text disjointness rather than merely selecting different windows from the same text. Re-running the checkpoint tokenizer and loader reproduces the production hash exactly and gives fresh token-window hash `7d62ff03e53f7a2b7b0bfe413bae3f0c`.

Two one-layer probe captures (`--start-layer 20 --end-layer 21` and `--start-layer 40 --end-layer 41`) used the established incremental-probe path, 16x512 fresh tokens, bf16, and a 64-row cap; no production activation cache was opened for writing. Target row counts are `layers.40.attn.wq_b`=64, `layers.40.experts.81.up_proj`=64, `layers.20.experts.63.up_proj`=64. The measured probe hot sections total 518.0 seconds. The three retained cache records are under [`act/`](act/), with hashes and capture metadata in [`capture_manifest.json`](capture_manifest.json).

## Fresh vs held-out vs in-sample

For every row below, plain and LDLQ weights were re-fit from the original production 64-row activation tensor exactly as in the merged pilot: production codebook and two-tier scales are frozen, the plain arm is the production field assignment, and the LDLQ arm adds 64-column block feedback from the 1%-damped fit Hessian. Both frozen weights are evaluated on the new activation rows. Held-out is the mean of the three prior CAL32/HOLDOUT32 splits and is included as the estimation-level reference; its fit size differs, so it is not used for the fresh verdict.

| Linear | K | in-sample reduction | held-out reduction | fresh LDLQ/plain | fresh reduction | fresh gain retained | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `layers.40.attn.wq_b` | 12 | 90.73% | 82.40% | 0.1282 | 87.18% | 96.08% | VALIDATED-PENDING-SERVED |
| `layers.40.attn.wq_b` | 15 | 91.48% | 82.34% | 0.1235 | 87.65% | 95.81% | VALIDATED-PENDING-SERVED |
| `layers.40.attn.wq_b` | 18 | 91.29% | 82.08% | 0.1258 | 87.42% | 95.76% | VALIDATED-PENDING-SERVED |
| `layers.40.experts.81.up_proj` | 12 | 95.09% | 61.04% | 0.4191 | 58.09% | 61.09% | VALIDATED-PENDING-SERVED |
| `layers.40.experts.81.up_proj` | 15 | 96.61% | 61.66% | 0.4078 | 59.22% | 61.30% | VALIDATED-PENDING-SERVED |
| `layers.40.experts.81.up_proj` | 18 | 96.92% | 60.74% | 0.4197 | 58.03% | 59.88% | VALIDATED-PENDING-SERVED |
| `layers.20.experts.63.up_proj` | 12 | 97.37% | 97.37% | 0.2067 | 79.33% | 81.47% | VALIDATED-PENDING-SERVED |
| `layers.20.experts.63.up_proj` | 15 | 97.53% | 97.48% | 0.2310 | 76.90% | 78.85% | VALIDATED-PENDING-SERVED |
| `layers.20.experts.63.up_proj` | 18 | 97.37% | 97.28% | 0.2724 | 72.76% | 74.73% | VALIDATED-PENDING-SERVED |

Raw SSE values, ratios, tensor shapes, damping, and encoder provenance are in [`results.json`](results.json). The capture/evaluation commands are preserved in [`run_capture.sh`](run_capture.sh).

## What the served A/B still adds

The planned Qwen smoke and served A/B remain necessary because this test freezes one Linear at a time and measures only local output SSE. Serving tests the materialized serialization and kernel path, accumulation across many simultaneously changed Linears, router/expert-selection changes, activation quantization, and the resulting model-level KL/PPL and generation behavior; it also supplies load/generation correctness and latency evidence that an activation-only screen cannot. Fresh-text retention therefore clears the pre-serving distribution gate, but it does not promote LDLQ to a production default.
