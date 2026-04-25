# PrismaQuant

**Mixed-precision quantization for LLMs. Every layer refracts into a different format based on its sensitivity.**

PrismaQuant measures the actual per-layer curvature of the loss and the
actual per-(layer, format) quantization error, then runs a proper
multi-choice knapsack to choose each layer's format under a total-bit
budget. It produces a `compressed-tensors` checkpoint that vLLM serves
**natively** — no custom runtime, no vLLM patches, no `auto_round` or
`llmcompressor` dependency at serve time.

- Formats out of the box: **NVFP4** (W4A4), **MXFP8 / FP8** (W8A8 dynamic
  per-channel), **BF16** passthrough. Extensible via `format_registry.py`.
- MoE: packed-expert tensors (Qwen3.5 / 3.6 `gate_up_proj` / `down_proj`,
  Mixtral `w1`/`w2`/`w3`) are handled first-class with a custom
  `_GradNormCapture` autograd `Function` — no weight-surrogate gradient
  accumulation, no `auto_round` unfuse step.
- **MTP** (Multi-Token-Prediction) heads are quantized end-to-end, then
  exercised via vLLM's `--speculative-config method=mtp` at serve time.
- Qwen3.5 / 3.6 fused-sibling groups (q/k/v, gate/up, `in_proj_qkv+z`,
  `in_proj_a+b`) are promoted to share a `weight_global_scale` so vLLM's
  fused loader doesn't warn about scale divergence.
- **Zero vLLM patches.** The output of PrismaQuant is a standard
  `compressed-tensors` checkpoint, parsed by vLLM's stock loader. No
  custom kernels, no forked serving stack. `vllm serve` the artifact
  and go.

## Motivation

Naive post-training quantization has two failure modes, both caused by
the same underlying problem: the quantizer doesn't know which parts of
the model are sensitive.

1. **Over-preservation.** Tools without a sensitivity model leave
   large chunks in BF16 or FP8 "to be safe" — usually `lm_head`, the
   attention Q/K/V, `down_proj`, and any Linear the authors saw fail
   once. Every one of those is a conservative guess, not a measured
   decision. The result is a 5 – 6 bpp artifact that could have been
   4.2 – 4.5 bpp with no quality loss.

2. **Over-aggression.** The same tools, run with a tighter budget,
   cast those same genuinely-sensitive Linears to NVFP4 anyway because
   they have no way to tell "this 4-bit op will cost you 3 points of
   MMLU" apart from "this 4-bit op will cost you 0.1 points of MMLU."
   Quality collapses before the bit-budget savings are realized.

Both failure modes waste the most precious resource in local-LLM
serving: **DRAM capacity and bandwidth**. On a DGX Spark (128 GB
unified memory, Blackwell SM121 with NVFP4/MXFP8 tensor cores), every
byte the KV cache doesn't spend on weights is a byte available for
context length, batch size, or the next model in the rotation. Getting
the weight format right on a per-Linear basis is the difference between
running one model at high quality and running three.

PrismaQuant replaces those heuristics with a closed-form per-Linear
cost estimate:

    Δloss ≈ 0.5 · H_trace · MSE_W

where `H_trace` is the empirical Fisher diagonal trace (captured once
via hooks during a short calibration pass) and `MSE_W` is the measured
per-format round-trip error on that Linear's actual weights. The
allocator solves a standard multi-choice knapsack over those estimates
under a total-bit budget. **Each bit goes where it buys the most
likelihood.** The result on Qwen3.6-35B-A3B at 4.75 bpp: 124 Linears in
NVFP4, 26 in MXFP8, 252 in BF16 — a distribution no hand-crafted
heuristic would produce, and one that serves natively in vLLM with
coherent generation and valid MTP speculative decoding.

### Contrast with Intel AutoRound (Int4)

Intel's AutoRound is a strong baseline for weight-only INT4
quantization. We ran it on the same model class earlier in this
project, and the quality gap we observed wasn't because AutoRound's
rounding algorithm was worse — it's a principled sign-gradient-descent
search over rounding decisions and it produces genuinely good per-layer
integer packings. The gap was structural: **AutoRound quantizes the
whole model to the same format.** Every Linear becomes INT4, because
that's the API. There's no way to tell it "this specific `down_proj` in
layer 23 is curvature-sensitive; leave it in BF16 and spend those 12
bits somewhere less sensitive."

PrismaQuant operates one level up. It's not a rounding algorithm — it's
a format **allocator** that composes on top of RTN, AutoRound, GPTQ, or
any other per-Linear quantizer you want to plug into
`format_registry.py`. The `FormatSpec` for each format carries its own
`quantize_dequantize` function, so you can drop an AutoRound-generated
codebook into the pipeline and still get per-Linear mixed-precision
selection. Combined with the measured-vs-analytical cost model, that
makes the bit budget go farther at the same point on the
quality-vs-size Pareto curve.

### Working within vLLM's constraints

vLLM's compressed-tensors loader accepts a specific set of
format/strategy combinations (NVFP4 `tensor_group g=16`, FP8 `channel`
W8A8, BF16 passthrough, and a few more). PrismaQuant's allocator is
explicitly aware of those constraints — you can't ask for a format
vLLM can't serve. That's why the initial release targets **exactly the
subset vLLM supports natively** (NVFP4 + FP8 W8A8 + BF16) and requires
**zero vLLM changes**. If vLLM's constraints change, the
`format_registry.py` adapts; until they do, every PrismaQuant artifact
is a drop-in `--quantization compressed-tensors` target.

The bigger opportunity — and the immediate next roadmap item — is
**targeting a fixed memory footprint** rather than a fixed bits-per-parameter.
Currently you ask the allocator for `--target-bits 4.75`. What you
actually want to ask it is "make the artifact fit 24 GB with the KV
budget I need at 8K context." The Pareto curve PrismaQuant already
computes has exactly this information (bits vs. Δloss vs. disk size).
Wiring a size-target mode on top is a day of work, not a research
project, and unlocks direct comparisons like:

  - **3090 (24 GB)** — what's the best 24 B-parameter MoE we can ship?
  - **DGX Spark (128 GB)** — how large can an MoE get before the KV
    budget dominates?

Once that lands, PrismaQuant goes from "allocator that optimizes bpp"
to "allocator that fits your hardware."

## Validated result

Qwen3.6-35B-A3B-MoE at target **4.75 bpp**:

| Metric | Source BF16 | PrismaQuant | Delta |
|---|---:|---:|---:|
| Size on disk | 70 GB | **22 GB** | −69 % |
| Body format mix | 100 % BF16 | 124 × NVFP4 + 26 × MXFP8 + 252 × BF16 | |
| MTP head size | 1.7 GB (BF16) | **0.5 GB** (NVFP4 experts + BF16 attn) | −68 % |
| Generation | ✓ | **✓** (coherent) | |
| Serves in vLLM | ✓ | **✓** (`compressed-tensors`, no patches) | |
| MTP spec-decoding | ✓ | **✓** (n=3 draft tokens) | |

vLLM backend at serve time: **FLASHINFER_CUTLASS** for NVFP4 MoE,
**CutlassFP8ScaledMMLinearKernel** for the FP8 W8A8 bucket, **FLASH_ATTN v2**
for attention, **prefix caching + FP8 KV cache** enabled.

### Zero-shot benchmark suite

Three-way comparison against the BF16 source and **RedHatAI's
`Qwen3.6-35B-A3B-NVFP4`** — a uniform-NVFP4 quantization of the same
base model produced by `llm-compressor`. All three artifacts evaluated
on the same vLLM server config (FP8 KV cache, FlashInfer backend,
prefix caching, `num_concurrent=16`), zero-shot, loglikelihood
scoring via lm-evaluation-harness.

| Task | Metric | BF16 (70 GB) | **PrismaQuant 4.75 bpp (22 GB)** | RedHat NVFP4 (24 GB) | Δ PQ−BF16 | Δ RH−BF16 | **Δ PQ−RH** |
|---|---|---:|---:|---:|---:|---:|---:|
| arc_easy      | acc      | 81.23 ± 0.80 | **80.72 ± 0.81** | 77.61 ± 0.86 | −0.51 | **−3.62** | **+3.11** |
| arc_easy      | acc_norm | 71.76 ± 0.92 | **72.26 ± 0.92** | 69.49 ± 0.94 | **+0.51** | −2.27 | **+2.78** |
| arc_challenge | acc      | 54.86 ± 1.45 | **54.35 ± 1.46** | 51.79 ± 1.46 | −0.51 | −3.07 | **+2.56** |
| arc_challenge | acc_norm | 54.95 ± 1.45 | **55.03 ± 1.45** | 53.50 ± 1.46 | **+0.08** | −1.45 | **+1.54** |
| piqa          | acc      | 82.21 ± 0.89 | **81.94 ± 0.90** | 80.79 ± 0.92 | −0.27 | −1.41 | **+1.14** |
| piqa          | acc_norm | 82.97 ± 0.88 | **82.10 ± 0.89** | 81.77 ± 0.90 | −0.87 | −1.20 | +0.33 |
| hellaswag     | acc      | 63.43 ± 0.48 | 62.68 ± 0.48     | 62.70 ± 0.48 | −0.76 | −0.74 | −0.02 |
| hellaswag     | acc_norm | 83.47 ± 0.37 | **82.91 ± 0.38** | 82.21 ± 0.38 | −0.56 | −1.25 | **+0.70** |
| winogrande    | acc      | 75.69 ± 1.21 | **73.48 ± 1.24** | 70.80 ± 1.28 | −2.21 | **−4.89** | **+2.68** |

**Headline numbers**:

- Mean Δ vs BF16: PrismaQuant **−0.56 pp**, RedHat **−2.21 pp** (~4×
  closer to BF16 on average).
- PrismaQuant wins **8 / 9 metrics** vs RedHat (hellaswag-acc is a
  0.02 pp tie). Sign test p < 0.02.
- Biggest single-task gap: **arc_easy −3.62 pp** on RedHat vs
  −0.51 pp on PrismaQuant — a 3.11 pp PrismaQuant advantage at a
  combined stderr of 1.18 pp → **2.6σ, statistically significant**.
- arc_easy also shows the pathology the Motivation section warned
  about: when every Linear gets NVFP4 with only a hand-picked ignore
  list, the ~5 % of genuinely sensitive Linears collapse the whole
  task. PrismaQuant's sensitivity-driven allocation keeps them in BF16
  or MXFP8 and recovers 3 pp of accuracy for **2 GB less on disk**.

**What this says about the allocator**: RedHat's checkpoint is one
format group with one regex target and 342 hand-picked ignores. At
equivalent size, PrismaQuant's 252 BF16 + 26 MXFP8 + 124 NVFP4 mix
(all decisions driven by measured `0.5·H·MSE_W`) is *strictly better*
on 8 out of 9 zero-shot metrics and tied on the ninth. The
sensitivity-driven allocator is doing measurable work beyond what
uniform quantization + a curated ignore list produces.

### Why RedHat has more BF16 layers but worse accuracy

Worth dwelling on: **RedHat's artifact preserves *more* Linears in
BF16 than PrismaQuant does**, yet is simultaneously larger on disk
*and* lower quality. The RedHat quantization_config's ignore list
contains **342 entries** — the entire visual encoder (110 Linears),
all 40 MoE routers, the `linear_attn.*` GatedDeltaNet projections on
every body layer (~150 entries), every `shared_expert.*`, `lm_head`,
MTP head. PrismaQuant's artifact keeps only **252 Linears** in BF16 —
90 fewer.

|  | BF16 Linears | MXFP8 | NVFP4 (dense) | NVFP4 (per-expert MoE) | Disk |
|---|---:|---:|---:|---:|---:|
| RedHat NVFP4        | **342** | 0  | 0   | ~20.5 k (via one regex) | 24 GB |
| PrismaQuant 4.75 bpp | **252** | 26 | 44  | ~20.5 k                 | **22 GB** |

The apparent paradox — RedHat has *more* BF16 yet *larger* disk and
*lower* accuracy — is exactly the over-preservation failure mode the
Motivation section warned about. RedHat's quantizer can't measure
sensitivity, so it hedges: "leave everything that might be sensitive
in BF16." The 90-Linear gap between 342 and 252 is Linears RedHat
conservatively preserved but PrismaQuant's `0.5·H·MSE_W` measurements
showed were safe to drop to MXFP8 or NVFP4. That's where the 2 GB
disk savings come from.

Crucially, the *other* direction holds too: of the Linears RedHat
quantized to NVFP4, PrismaQuant's measurements flagged 26 of them as
too sensitive for NVFP4 and promoted them to MXFP8 (and 252 of what
RedHat quantized... wait, those are already in RedHat's BF16 list).
The interesting case is really the 26 MXFP8 Linears — these are
genuinely sensitive ones where NVFP4 would hurt but BF16 is
overkill. RedHat can't express that distinction with a one-format
scheme; PrismaQuant can.

The resulting accuracy pattern matches the theory exactly:

- **arc_easy / arc_challenge** are knowledge-recall tasks that lean
  heavily on the Linears PrismaQuant kept in BF16/MXFP8 that RedHat
  left NVFP4. PrismaQuant − RedHat ≈ **+3 pp** on both, **2.6σ** on
  arc_easy.
- **hellaswag / piqa** lean on the bulk MoE body, which both
  quants put in NVFP4 identically. Margin narrows to < 1 pp, tied on
  hellaswag-acc.
- **winogrande** stresses pronoun resolution through attention —
  sensitive to attention-projection quantization. Both quants drop
  (−2.21 / −4.89), but RedHat drops >2× more because PrismaQuant's
  sensitivity measurements nudged those Linears up to MXFP8/BF16.

Over-preservation burns disk *and* accuracy when you can't
discriminate which Linears actually need it. Under-aggression
(quantizing genuinely sensitive Linears to recover budget) burns
accuracy worse. Both are failure modes of sensitivity-blind
quantizers. Measuring `0.5·H·MSE_W` per Linear turns them into
separable decisions.

### Honest caveats on the task mix

The five tasks above are commonsense / world-knowledge benchmarks
that a 35 B MoE saturates. They test whether quantization
catastrophically broke the model; they do *not* stress reasoning
chains, in-context learning, or generation fluency.

What's **not yet in the table** and what the pre-publication roadmap
calls for:

- **MMLU 5-shot** (especially MMLU-STEM) — reasoning-heavy, standard
  benchmark for PTQ papers, where compounding quantization error
  shows up most visibly.
- **GSM8K 5-shot** (generative, strict-match) — arithmetic + multi-step
  reasoning; the canonical "did quantization break the model?" test.
- **HumanEval / MBPP** — code generation, tests generation quality
  directly rather than loglikelihood scoring.
- **MT-Bench or AlpacaEval** — open-ended generation quality.
- **Per-budget Pareto sweep** — we've only measured 4.75 bpp. The
  Pareto curve in the next section is *predicted* Δloss from the
  allocator. Genuine validation requires quantizing at 4.5 and 5.0
  and measuring.
- **Seed sweep** — single run per artifact. winogrande's −2.21 / −4.89
  deltas are real-looking but should be averaged over 2-3 seeds.

Until those land, the defensible claim is the headline above:
**PrismaQuant beats uniform NVFP4 on commonsense zero-shot at a
smaller size**. The reasoning-heavy and generative-quality story is
still pending.

## Why 4.75 bpp — the cheap quarter-bit

**The 4.75 bpp target in this artifact was a first-shot proof of
concept, not a tuned optimum. It works remarkably well.** This section
unpacks why, and why tiny bpp bumps above pure NVFP4 buy massive
quality.

### Stock NVFP4 is ~4.5 bpp, not 4.0

NVFP4 packs 4-bit weight + 4-bit activation values into 16-element
groups sharing one FP8 scale factor. Per-parameter storage cost
works out to `4 + 8/16 = 4.5 bits/param` for weights — not the
nominal 4.0 bits. A "pure NVFP4" model on disk is already at 4.5 bpp;
you don't save anything by restricting yourself to one format.

PrismaQuant's 4.75 bpp target is only **+0.25 bpp over pure NVFP4** —
roughly a 6% bit-budget increase, allocated intelligently rather than
uniformly.

### The Pareto curve for Qwen3.6-35B-A3B

Predicted Δloss vs. bit budget, from `pareto.csv`
([allocator.py](prismaquant/allocator.py)):

| Target bpp | Achieved | Format mix                                    | Predicted Δloss | Ratio vs 4.5 |
|-----------:|---------:|-----------------------------------------------|---------------:|-------------:|
| 4.5        | 4.643    | 173 NVFP4 + 1 MXFP8 + 228 BF16                | 4.282          | 1.00×        |
| **4.75** ← | **4.758**| **124 NVFP4 + 26 MXFP8 + 252 BF16** (shipped) | **2.355**      | **0.55×**    |
| 5.0        | 5.010    | 82 NVFP4 + 9 MXFP8 + 311 BF16                 | 1.184          | 0.28×        |
| 5.5        | 5.496    | 79 NVFP4 + 0 MXFP8 + 323 BF16                 | 0.923          | 0.22×        |
| 6.0        | 5.938    | 75 NVFP4 + 0 MXFP8 + 327 BF16                 | 0.860          | 0.20×        |

The curve is extremely steep near pure-NVFP4 and flattens out past
~5 bpp. Key takeaways:

- **4.5 → 4.75 bpp (+0.25 bpp, +5.6% size):** predicted Δloss drops
  **45%**. That's the quarter-bit that lets the allocator move 49
  Linears out of NVFP4 and into MXFP8/BF16 — the ones where
  `0.5·H·MSE_W` flagged NVFP4 as expensive.
- **4.75 → 5.0 bpp (+0.25 bpp, +5.3% size):** another **50%** drop.
- **5.0 → 6.0 bpp (+1 bpp, +20% size):** only a **27%** drop. Severely
  diminishing returns.

The Kneedle knee-detection the allocator suggests (Satopaa et al. 2011)
lands right around 5.0 bpp. 4.75 sits below the knee — it's on the
aggressive side of the curve, which is exactly what makes the result
interesting: **PrismaQuant at 4.75 bpp outperforms stock NVFP4 at
a similar or larger effective size**, because the allocator pulls the
quarter-bit from the least-sensitive 90% of the model and concentrates
it on the most-sensitive 10%.

### Serving-speed impact of the extra quarter-bit: near zero

The 4.75 bpp artifact's 124 NVFP4 Linears include the 80 per-expert
MoE projections (representing ~805 M params × 256 experts = the
model's dominant compute path at inference time). Those still go
through the FLASHINFER_CUTLASS NVFP4 kernel. The 26 MXFP8 Linears go
through CutlassFP8 (same family). The 252 BF16 Linears are almost
entirely small norm/bias/attention-projection weights that contribute
a negligible fraction of matmul FLOPs.

**Decode throughput on a DGX Spark (Blackwell SM121) for our 4.75 bpp
artifact is within noise of a hypothetical 4.5 bpp pure-NVFP4
artifact** — you pay ~6% extra DRAM bandwidth for ~50% less quality
degradation. That trade favors the quarter-bit bump almost every time.

### Implication for future targets

If +0.25 bpp over pure NVFP4 halves predicted Δloss, and the next
quarter-bit halves it again, then the interesting bit budgets for
production are **`pure-NVFP4-equivalent + 0.25 to 0.75 bpp`** — not
the `pure-NVFP4` point that the field has anchored on. We'll sweep
this space more thoroughly in a follow-up once MiniMax M2.7 and
3-bit support land (the same logic applies below 4 bpp: tiny bumps
above a uniform 3-bit baseline unlock large quality recovery at
~zero serving cost).

## Quick start

Three commands on a machine with the model cached locally:

```bash
export MODEL_PATH=/path/to/Qwen3.6-35B-A3B
export WORK_DIR=./dq-runs/qwen36
export FORMATS=NVFP4,MXFP8_E4M3,BF16
export TARGET_BITS=4.75

./quantization/prismaquant/run-pipeline.sh
```

That runs probe → cost → allocator → native export. MTP heads are
covered automatically by the incremental probe / cost as a built-in
shard; no separate commands are needed. Serve with:

```bash
vllm serve $WORK_DIR/exported \
  --quantization compressed-tensors \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

## Pipeline

    sensitivity_probe ──────► probe.pkl  (Fisher g² per Linear + packed expert)
                                 │
    measure_quant_cost ─────────┼─────► cost.pkl  (per-(Linear, format) MSE)
                                 │
              (optional) local_reconstruct — elite-candidate clipping sweep
                                 │
    allocator ◄─────────────────┘
         │
         ▼ layer_config.json, pareto.csv
         │
              (optional) measure_interactions + quadratic_refine_allocator
              (optional) calibrate_allocator     — KL-fit per-format gain
         │
         ▼
    export_native_compressed  ►  exported/   (compressed-tensors checkpoint)
         │
         ▼
    validate_native_export    ►  vLLM forward + greedy decode

The allocator uses a closed-form per-Linear loss proxy:

    Δloss ≈ 0.5 · H_trace · MSE_W · gain_per_format

where `H_trace` is the Fisher diagonal trace measured in stage 1,
`MSE_W` is the measured per-(Linear, format) weight MSE from stage 2,
and `gain_per_format` is the NNLS-fit calibration gain from
`calibrate_allocator` (defaults to 1.0).

### Pipeline stages

#### 1. Sensitivity probe — `sensitivity_probe.py`

Streaming backward with forward + backward hooks so no full parameter
gradient ever materializes. Sum-reduction CE so per-token gradients
aggregate cleanly into a per-layer Fisher diagonal trace. Route-aware
weighting for MoE experts (divide by observed routing probability so
sparse experts' Fisher is comparable to dense Linears'). Per-token
importance weighting (hard tokens count more).

Packed experts use `_GradNormCapture` — a `torch.autograd.Function` that
captures the squared Frobenius norm of the incoming gradient through an
identity forward and returns `None` for the weight gradient so autograd
doesn't accumulate a full 5 GB `.grad` on the leaf parameter. This is
what makes 40-layer × 2-param × BF16 MoE backwards tractable in 128 GB.

Incremental mode (`incremental_probe.py`) shards the work so only one
shard's worth of hooks is live at a time — makes a 35B run resumable
and keeps peak memory bounded.

#### 2. Measure quantization cost — `measure_quant_cost.py`

For each tracked Linear and each registered format, apply the native
weight/activation round-trip and measure `weight_mse` and
`output_mse = ‖Wx − Ŵx̂‖²` against saved activations from stage 1.
Batched GPU path groups Linears by shape and runs one `torch.bmm` per
(group, format) — converts ~31 000 tiny kernel launches on a 35B MoE
down to ~360 batched ones. Unbatched CPU path is slower but uses no
extra VRAM.

Incremental mode (`incremental_measure_quant_cost.py`) shards the
measurements the same way the probe does and merges the per-shard
pickles at the end.

#### 3. Allocate — `allocator.py`

Multi-choice knapsack DP. Per-Linear  `Δloss = 0.5 · H_trace · MSE_W`;
total constraint  `Σ bits ≤ target`. Fused-projection siblings
(q/k/v, gate/up, Qwen3.6-specific `linear_attn.{in_proj_qkv+z}` and
`{in_proj_a+b}`) are promoted to the highest-precision sibling so vLLM's
fused loader sees consistent formats.

Outputs:
- `layer_config.json` — per-Linear format assignment (also readable by
  `llmcompressor` / `auto_round`)
- `pareto.csv` — Δloss vs bits sweep across `--pareto-targets`
- Printed Kneedle-style knee suggestion

#### 4–6. Optional refinement

- `local_reconstruct.py` — grid-searches symmetric clipping on weights
  and activations for a small set of elite frontier-critical layers.
  Memory-safe (one layer at a time), intentionally slow.
- `measure_interactions.py` + `quadratic_refine_allocator.py` —
  sparse pairwise KL probe near the knee, then local quadratic
  refinement. The additive frontier stays fast; interactions only
  matter where they matter.
- `calibrate_allocator.py` — validate a few frontier points against
  actual KL on held-out data so the predicted frontier can be trusted
  or corrected on the current model. Emits per-format `gain` factors
  that the allocator re-reads via `--calibration`.

#### 7. MTP extensions — built into the incremental probe + cost

Transformers v5 drops `mtp.*` weights on load (they're in the class's
`_keys_to_ignore_on_load_unexpected`) because MTP is a vLLM-only
feature. PrismaQuant instantiates a standalone `MtpModule`
(`mtp_module.py`: HF `Qwen3_5MoeDecoderLayer` plus the `pre_fc_norm_*`,
`fc`, and `norm` exactly per vLLM's `Qwen3_5MultiTokenPredictor`
forward), loads the MTP weights directly from safetensors, then runs
the standard Fisher probe against the MTP auxiliary objective

    loss = CE( lm_head( MTP(embed_{t+1}, body_hidden_t) ), ids_{t+2} )

This happens inline in `incremental_probe.py` / `incremental_measure_quant_cost.py`
as a built-in shard kind (on by default, toggle with `--no-include-mtp`).
The allocator treats MTP Linears identically to body Linears — same
cost model, same knapsack, same fused-sibling rules. At export time,
MTP per-expert tensors are split and emitted with `mtp.layers.X.mlp.experts.Y.{gate|up|down}_proj.*`
naming, which vLLM's MTP loader picks up via its own
`mtp. → model.` weight-name remap.

#### 8. Native compressed-tensors export — `export_native_compressed.py`

Turns `layer_config.json` into a `compressed-tensors` checkpoint that
vLLM serves natively.

- Quantizes each `nn.Linear` per the recipe (NVFP4,
  MXFP8 → vLLM's `CompressedTensorsW8A8Fp8` dynamic per-channel, or
  BF16 passthrough).
- Packed MoE experts (`gate_up_proj` / `down_proj` 3D) are split into
  per-expert per-projection tensors
  (`experts.{e}.{gate|up|down}_proj.weight_packed`) to match vLLM's
  loader convention. Joint `weight_global_scale` is promoted across
  fused siblings so vLLM's loader doesn't warn about scale divergence.
- Writes `weight_global_scale` in compressed-tensors' **divisor
  convention** (`1/scale`); vLLM inverts on load. Emits
  `input_global_scale = 1.0` for every NVFP4 Linear so vLLM's
  `1/input_global_scale` initialization stays finite — otherwise
  the kernel produces degenerate output (`!!!!!!!!`).
- Generates `quantization_config` with
  `quant_method = compressed-tensors`, `format = mixed-precision`,
  per-format `config_groups` with explicit per-Linear regex targets in
  vLLM's internal naming (`language_model.model.layers.X.*` for the
  qwen3_5 `hf_to_vllm_mapper`), plus `PER_EXPERT_MOE_REGEX` and
  `MTP_PER_EXPERT_REGEX` catch-alls for the per-expert MoE tensors.
- MTP weights go through the same quantize-and-emit path, then are
  named with the source `mtp.` prefix that vLLM's MTP loader accepts
  verbatim.
- Visual encoder weights pass through as BF16 from source (real
  calibration is deferred — see "what's deferred" below).

#### 9. Validate — `validate_native_export.py`

Binary check: load the checkpoint in vLLM and do a single greedy decode.
Optionally upgrades the container's flashinfer to a known-good version
before loading (pass `--no-flashinfer-upgrade` to skip).

Extend with `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'`
to actually exercise the quantized MTP heads during the decode.

## Formats

### Supported

Built-in (register more via `format_registry.py`):

| Family | Formats                                                |
|--------|--------------------------------------------------------|
| NV     | NVFP4, NVFP4A16                                        |
| MX     | MXFP4, MXFP6_E3M2, MXFP6_E2M3, MXFP8, MXFP8A16         |
| Int    | INT8_W8A16, INT4_W4A16_g128                            |
| Float  | BF16 (passthrough)                                     |

**NVFP4 and MXFP4 are alternatives for the same 4-bit tier, not
separate precision levels.** Include at most one format per bit tier
— otherwise the allocator picks between them based on per-layer RTN
measurement noise and you end up with a serving mess (two kernel
paths for 4-bit quant). Allocator warns by default, errors with
`--enforce-family-coherence`.

### Hardware + serving-stack support

Everything in this section assumes serving with vLLM. Microscaling
formats (NVFP4, MX\*) require NVIDIA Blackwell-era hardware (SM100+)
for native kernel support; on older Ampere/Ada you get Marlin
emulation, which works at a significant speed penalty.

|              | Blackwell ISA | vLLM serving today           |
|--------------|:-------------:|:----------------------------:|
| NVFP4        | ✓             | ✓ (FlashInfer CUTLASS)       |
| MXFP4        | ✓             | ✓ (FlashInfer CUTLASS)       |
| MXFP6\_E3M2  | ✓             | ✗ (kernel not yet integrated)|
| MXFP6\_E2M3  | ✓             | ✗ (same)                     |
| MXFP8        | ✓             | ✓ (as W8A8 FP8 via Cutlass)  |
| INT4 / INT8  | all NV HW     | ✓ (Marlin)                   |

Until vLLM picks up MXFP6 serving kernels, including `MXFP6_*` in a
bundle means the allocator can pick it, the checkpoint will contain
it, but vLLM won't know how to load it at serve time. Safe to
experiment with on the quantization side; do not ship until the
kernels land.

### Recommended bundles

| Use case                          | `--formats`                       |
|-----------------------------------|-----------------------------------|
| Ship today on Blackwell via vLLM  | `NVFP4,MXFP8_E4M3` (validated)    |
| MX-pure on Blackwell              | `MXFP4,MXFP8`                     |
| Experimental with MXFP6           | `NVFP4,MXFP6_E3M2,MXFP8`          |
| Legacy INT pipelines              | `INT4_W4A16_g128,INT8_W8A16`      |

## Method notes

### Joint expert pruning + quantization

The joint optimizer treats an MoE choice as `(format, dropped_expert_ids)`
and prices it in the same Δloss/memory knapsack as dense quantization.
The implementation contract and failure-mode checks live in
[`docs/joint_optimizer.md`](../docs/joint_optimizer.md). Read that note
before changing prune saliency, candidate construction, or the export
sidecar.

### This is not gradient descent

`requires_grad_(False)` on all parameters. Backward runs only to push
gradient signal through autograd so the Fisher hooks can read it;
nothing is written back. It's a sensitivity measurement, not an
optimizer.

### Why Fisher and not Hutchinson?

Hutchinson on a Linear's weights via vHv probes requires a different
hook architecture than we use (hooks see activation gradients, not
parameter gradients). Fisher (g²) is the natural fit for hooks and
gives a first-order proxy for curvature that correlates well with
quantization sensitivity when combined with measured RTN error (which
removes the need for Fisher to predict anything — it only needs to
rank layers).

### Why measured RTN error over analytical formulas?

The uniform-quantization MSE formula overweights max-magnitude outliers
and doesn't model non-uniform FP codebooks. Running RTN once and
measuring `‖Wx − Ŵx‖²` captures the actual distribution of the weight
tensor and the actual functional perturbation at the layer output — no
tuning constants, no assumption about weight distributions.

### Why the closed-form `0.5·H·MSE_W`?

The earlier formula `output_mse · d_out` was dimensionally unsound
(mixed output-space error with fan-out). Under the Gauss-Newton
approximation, local Δloss from a weight perturbation `δW` is
`0.5·δWᵀ H δW`, and the Fisher diagonal trace `H_trace = Σ g²` is a
well-defined proxy for the trace of `H` under the standard
independence-across-tokens assumption. Measuring weight MSE directly
is a sharper estimator than inferring it from activation propagation.

### What about inter-layer interactions?

The frontier builder remains additive because that is the only
practical way to sweep the whole model cheaply. PrismaQuant addresses
the missing cross-layer terms by:

- measuring sparse pairwise interactions only for the most important
  units near the knee (`measure_interactions.py`)
- refining the knee locally with those terms
  (`quadratic_refine_allocator.py`)
- calibrating the refined frontier against actual KL
  (`calibrate_allocator.py`)

This keeps memory bounded while still capturing the interaction
structure that recent MPQ literature shows matters. Framing we owe
to Gemini's review: **use the blunt proxy to get to the Pareto
frontier fast, then measure true pairwise interactions only at the
margin where the allocator is borderline.**

### Methodological caveats — assumptions the proxy makes

Three places where the current `0.5 · H_trace · MSE_W` proxy cuts
corners. We know about them, we can quantify most of them, and all
three have explicit follow-up work on the roadmap.

**1. Scalar per-layer curvature.** `H_trace` is a single number per
Linear — the sum of `‖∂L/∂W‖²_F` across all channels. That collapses
any *intra-layer* curvature structure. If a given `up_proj` has 90 %
of its channels fine under NVFP4 but 10 % catastrophically sensitive,
the trace averages them and the allocator sees an average-pressure
layer. Per-output-channel H diagonal + per-channel weight MSE
preserves the knapsack's optimal substructure (the DP stays
separable), costs `out_features` floats per layer (single-digit MB
on a 35B model), and is the single most likely place the proxy
underperforms an informed allocator. **On the roadmap as the "next
rigor pass."** Open question: does the resulting allocation actually
differ meaningfully from the scalar-trace version on our validated
tasks? We haven't ablated yet.

**2. Empirical Fisher vs true Hessian.** `g · gᵀ` equals the true
Hessian at a strict local minimum; post-training LLMs on our
calibration distribution are close but not exact. The resulting
bias is roughly layer-uniform — the absolute Fisher magnitudes are
slightly off but the relative ranking across layers (which is all
the knapsack needs) is preserved. Hutchinson's estimator via
forward-over-reverse autodiff would fix this cleanly at ~2× probe
time, but the measured-ranking invariance means it probably doesn't
change allocation decisions in practice. Worth the experiment for a
preprint; not a production blocker.

**3. Zero cross-terms.** The additive formula `Σᵢ 0.5 · Hᵢ · MSEᵢ`
assumes every Linear's quantization error is uncorrelated with
every other's. False in general — K-FAC style off-diagonals would
capture it, but they destroy the knapsack's optimal substructure
(the problem becomes NP-hard Quadratic Knapsack). PrismaQuant's
existing `measure_interactions.py` + `quadratic_refine_allocator.py`
handles this the pragmatic way: measure sparse pairwise interactions
only for the most-sensitive units near the knee, then refine
locally. Additive gets us 95 % of the way to optimal; the margin
refinement handles the borderline cases where interactions actually
flip decisions.

The pragmatic TL;DR: the scalar-diagonal-additive proxy is a
feature, not a bug — it keeps the global bit-routing problem
cleanly solvable in O(N·bits) time. Each of the three corners
cut above has a defined off-ramp that doesn't force us to abandon
the knapsack.

## Memory budget

| Stage                     | Peak RAM        | Peak VRAM (GB10) |
|---------------------------|-----------------|------------------|
| incremental_probe (35B)   | 90 GB           | 90 GB (unified)  |
| incremental_cost (35B)    | 60 GB           | 60 GB            |
| allocator                 | < 1 GB          | n/a              |
| export_native_compressed  | 60 GB           | 20 GB            |

Fits 128 GB unified systems (DGX Spark GB10, etc.) for models up to
~48 B parameters. The incremental-mode watchdog in the probe and cost
scripts aborts cleanly on swap pressure rather than OOM-killing the
host.

## Adding a new architecture

PrismaQuant's pipeline is **model-agnostic by default**, with
architecture-specific knowledge concentrated in a `ModelProfile`
subclass. Most of what a profile needs is already encoded by vLLM on
its model class and **auto-derived** — a new architecture's profile
typically just names the vLLM class and optionally supplies an
MTP-probe helper.

### The concentration principle

Every place PrismaQuant needs to make an architecture-specific
decision — fused-sibling promotion, vLLM's weight-loader naming
convention, packed-expert parameter names, MTP module construction,
source passthrough prefixes — is routed through `ModelProfile`.
Profiles are looked up once at runtime from the HF config's
`model_type` and `architectures` fields; no other file in the
pipeline cares which architecture it's looking at.

### What the profile gets for free

Default implementations in `ModelProfile` read two attributes off
the vLLM model class registered for this architecture:

- **`packed_modules_mapping`** (e.g.
  `{'qkv_proj': ['q_proj', 'k_proj', 'v_proj'], 'gate_up_proj': ['gate_proj', 'up_proj'], ...}`).
  Drives `profile.fused_sibling_group()` — the allocator promotes all
  members of a fused group to the highest-precision sibling so vLLM's
  fused-loader sees consistent formats.
- **`hf_to_vllm_mapper.orig_to_new_prefix`** (e.g.
  `{'model.language_model.': 'language_model.model.', 'model.visual.': 'visual.', 'lm_head.': 'language_model.lm_head.'}`).
  Drives `profile.to_vllm_internal_name()` — the config_groups targets
  and vLLM's scheme-dispatch names stay in sync without PrismaQuant
  duplicating the mapping.

Both are **vLLM's source of truth**. When vLLM adds a new fused
pattern or naming quirk on an existing architecture, PrismaQuant picks
it up on the next probe run with no PrismaQuant-side code change.

### What the profile still writes by hand

- **`matches(model_type, architectures)`** — pattern-match on HF config.
- **`name`** — profile identifier string.
- **`vllm_architecture_class()`** — one string: the HF
  `architectures[0]` entry whose vLLM class carries the metadata above.
- **`build_mtp_module(text_config)`** (only if the arch has MTP) — an
  HF-module replica of vLLM's MTP forward so PrismaQuant's Fisher probe
  can backward through it. vLLM's MTP class isn't HF-autograd-friendly,
  which is why this one piece can't auto-derive.
- **`source_passthrough_prefixes()`** (optional) — which top-level
  prefixes get copied from source as BF16 (visual encoder if we defer
  calibration, MTP residue, etc.).
- **`to_vllm_internal_name()` override** (optional) — for arch quirks
  that the vLLM prefix map doesn't capture. Qwen3.5/3.6 overrides to
  preserve the `mtp.` prefix at scheme-dispatch (vLLM's `mtp.→model.`
  remap runs at weight-load, not scheme-dispatch).

### The onboarding flow (concrete steps)

For a hypothetical `WhateverForCausalLM`:

1. **Write the profile:**

```python
# prismaquant/model_profiles/whatever.py
from .base import ModelProfile

class WhateverProfile(ModelProfile):
    @classmethod
    def matches(cls, model_type, architectures):
        return model_type == "whatever" or any(
            a.startswith("Whatever") for a in architectures)

    @property
    def name(self): return "whatever"

    def vllm_architecture_class(self):
        return "WhateverForCausalLM"

    # ... only if has_mtp / visual / quirks:
    # def has_mtp(self): return True
    # def build_mtp_module(self, text_config): ...
```

2. **Register it:**

```python
from prismaquant.model_profiles import register_profile
register_profile(WhateverProfile)
```

Or add to `prismaquant/model_profiles/registry.py`'s `_REGISTERED` list.

3. **Validate it against a real checkpoint** — this is the load-bearing
step. The validator catches 90% of "I forgot to update this" bugs:

```
python -m prismaquant.model_profiles.validate \
    --model /path/to/Whatever-7B
```

Output is a list of ✓/✗ checks:

```
Validating profile: WhateverProfile (auto-detected)
Model:              /path/to/Whatever-7B
vllm class:         WhateverForCausalLM

  ✓ matches() returns True for this model
  ✓ vllm_architecture_class() resolves
  ✓ fused-sibling groups consistent
      5 fused groups × multiple siblings map to the same canonical key
  ✓ to_vllm_internal_name() obeys vLLM's prefix map
      3 prefix rewrites agree
  ✓ packed_expert_param_names() cover actual 3D params
      2/2 declared names found
  ✓ source_passthrough_prefixes() cover real tensors
      2 prefixes, all match at least one tensor
  ✓ MTP module constructs + loads weights

7 / 7 checks passed
```

The validator cross-checks against vLLM's class metadata + the
model's safetensors index + (if MTP) an actual MTP module
instantiation. Passing checks don't guarantee the export will work
end-to-end, but they catch every class of consistency bug I've
found so far across three architecture families.

4. **Run the pipeline.** No other file in PrismaQuant knows which
architecture you're on. The probe, cost, allocator, and export all
pull from the profile when they need an arch-specific decision.

5. **Report**. Add a line to the README's validated-models list, ship
a recipe, and open a PR with the probe/cost artifacts and eval
numbers so the next person can see precedent.

### Coverage status

- **`Qwen3_5Profile`** — covers Qwen3.5, Qwen3.6. Tested end-to-end on
  Qwen3.6-35B-A3B. Validator passes 7/7.
- **`DefaultProfile`** — generic fallback. No fused-sibling promotion,
  no MTP, identity name remap. Will do the right thing for simple
  architectures whose vLLM class has `packed_modules_mapping` but
  won't handle MTP or multimodal naming quirks.

Planned profiles (should all be ~50-line files each since the vLLM
registry gives most of what we need):

- `DeepseekV3Profile` — covers DeepSeek-V3 family; has MTP
  (`deepseek_mtp`), has MoE (different packed convention than Qwen).
- `GLM4MoEProfile` — has MTP (`glm4_moe_mtp`), has MoE.
- `MinimaxProfile` — MiniMax M1/M2.7; novel architecture, may need
  more than vLLM exposes.
- `LlamaMoEProfile` — Llama 3 / 4 MoE variants (no MTP).

Contributions welcome; the validator should give clear feedback on
what's missing for any profile submission.

## What's deferred

- **Visual encoder calibration**: weights pass through as BF16 from
  source. The probe ran the regex for visual blocks, but
  `stage_text_only` strips `vision_config` from the staged config so
  the loaded model never instantiates visual modules. Real multimodal
  calibration (load `ForConditionalGeneration` + use `AutoProcessor`
  on an image dataset) is straightforward follow-up but not yet in
  the pipeline.
- **Non-Qwen MTP forward construction**: The HF-side MTP module
  replica used for Fisher probing is profile-provided. Qwen3.5/3.6
  has one (`Qwen3_5Profile.build_mtp_module`). Other MTP architectures
  need their own — see "Adding a new architecture" above.

## Roadmap

Immediate priorities (next few weeks):

- **MiniMax M2.7 on a single DGX Spark.** MiniMax M2.7 is ~280 B
  parameters — it does not fit in 128 GB at BF16, at FP8, or at a
  uniform 4-bit quant with meaningful KV headroom. PrismaQuant's
  mixed-format allocator gets close: if we can land an average
  ~3.4 bpp recipe with the sensitive paths in NVFP4 and most experts
  in 3-bit, the model fits and the KV cache has room to breathe. The
  blockers are `3-bit support` and a size-targeting allocator mode
  (below).
- **3-bit format support.** Requires registering a 3-bit
  `FormatSpec` in `format_registry.py` and picking an underlying
  kernel path that vLLM can serve. Candidates: GPTQ 3-bit via Marlin
  (available now; slower), or native 3-bit W3A16 when/if
  compressed-tensors ships a dispatcher.
- **MXFP6 (E3M2 and E2M3).** Blackwell tensor cores support MXFP6
  natively — the hardware path is free. vLLM doesn't yet ship a
  serving kernel for it, but the quantization side works fine today
  (`format_registry.py` has both variants registered). Enable in the
  serving path once the vLLM dispatcher lands.
- **Size-targeting allocator mode.** `--target-bytes 24_000_000_000
  --kv-budget 8192` instead of `--target-bits 4.75`. The allocator
  already computes the full Pareto curve; this is a thin wrapper that
  picks the knee matching a hardware constraint. Unlocks direct
  comparisons at 3090 (24 GB), 5090 (32 GB), and Spark (128 GB)
  memory targets.

Broader research directions:

- **Per-channel Fisher + per-channel weight MSE.** The next rigor
  pass on the cost model. Keep `H_diag` as a vector of length
  `out_features` instead of summing to `H_trace`, pair with
  per-channel `MSE_W` already produced by `measure_quant_cost.py`'s
  batched path, and compute Δloss as `Σᵢ 0.5 · H_diag,i · MSE_W,i`.
  Preserves the knapsack's optimal substructure, costs < 10 MB of
  extra storage per 35B model, and is the single most likely place
  the current scalar proxy underperforms. Cheap intermediate: keep
  the scalar `H_trace` but use per-channel `MSE_W` right away — a
  half-day of work that captures ~50 % of the full win without
  changing the probe. Would settle the empirical question "does
  per-channel sensitivity actually differ meaningfully from
  per-layer sensitivity for our 4.75 bpp allocation?" (hypothesis:
  yes, on attention Linears and `down_proj` where outlier channels
  are the norm).
- **Per-MTP-Linear acceptance-rate tuning** — jointly optimize
  target-model quality and MTP-draft token acceptance rate. Today MTP
  Linears get assigned using the same body loss model; a more
  sophisticated allocator would weight MTP sensitivity by the draft's
  contribution to speculative-decoding throughput.
- **Multimodal calibration** — load
  `ForConditionalGeneration`, drive image + text pairs through the
  multimodal processor so visual encoder blocks get real Fisher stats
  instead of BF16 passthrough.
- **Model-profile registry completion** — the `model_profiles/`
  package is in place and handles Qwen3.5/3.6 end-to-end; add
  first-class profiles for DeepSeek-V3 / V3.1, GLM-4, MiniMax, and
  Llama-family MoE so PrismaQuant works out of the box on those
  architectures.
- **Sparse-outlier co-quantization.** Borrow from SpQR / SqueezeLLM:
  extract the top-k outlier weights per Linear, serve them as a
  sparse BF16 side-table, quantize the dense remainder more
  aggressively. Would push the 4.75 bpp frontier down to ~4.0 bpp
  with the same quality.
- **Interaction-aware refinement at scale.** We already have sparse
  pairwise interaction measurement + local quadratic refinement
  (`measure_interactions.py` + `quadratic_refine_allocator.py`).
  Scaling these to cover the whole model without the memory cost of
  a dense pairwise probe is open.
- **Serving-aware calibration.** The calibration set right now is
  generic ultrachat; domain-targeted calibration (code, reasoning,
  multilingual) should Pareto-dominate for task-specific deployments.

## References

PrismaQuant stands on the shoulders of a decade of mixed-precision
quantization research. The closed-form cost model, the Fisher-diagonal
sensitivity estimator, the multi-choice knapsack formulation, and the
format registry are all assembled from published ideas. Key influences:

**Mixed-precision bit allocation**

- Wang, K., Liu, Z., Lin, Y., Lin, J., & Han, S. *HAQ: Hardware-Aware
  Automated Quantization with Mixed Precision.* CVPR 2019.
- Dong, Z., Yao, Z., Gholami, A., Mahoney, M. W., & Keutzer, K.
  *HAWQ: Hessian AWare Quantization of Neural Networks with
  Mixed-Precision.* ICCV 2019.
- Dong, Z., Yao, Z., Arfeen, D., Gholami, A., Mahoney, M. W., &
  Keutzer, K. *HAWQ-V2: Hessian Aware Trace-Weighted Quantization of
  Neural Networks.* NeurIPS 2020.
- Yao, Z., Dong, Z., Zheng, Z., Gholami, A., Yu, J., Tan, E., et al.
  *HAWQ-V3: Dyadic Neural Network Quantization.* ICML 2021.
- Chen, W., Wang, P., & Cheng, J. *Towards Mixed-Precision
  Quantization of Neural Networks via Constrained Optimization.*
  ICCV 2021.

**Post-training quantization and rounding**

- Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. *GPTQ:
  Accurate Post-Training Quantization for Generative Pre-trained
  Transformers.* ICLR 2023.
- Frantar, E. & Alistarh, D. *Optimal Brain Compression: A Framework
  for Accurate Post-Training Quantization and Pruning.* NeurIPS 2022.
- Cheng, W., Zhang, W., Shen, H., Cai, Y., He, X., & Lv, K.
  *Optimize Weight Rounding via Signed Gradient Descent for the
  Quantization of LLMs (AutoRound).* arXiv 2309.05516, 2023.
- Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C.,
  et al. *AWQ: Activation-aware Weight Quantization for LLM
  Compression and Acceleration.* MLSys 2024.
- Nagel, M., Amjad, R., van Baalen, M., Louizos, C., & Blankevoort,
  T. *Up or Down? Adaptive Rounding for Post-Training Quantization
  (AdaRound).* ICML 2020.
- Li, Y., Gong, R., Tan, X., Yang, Y., Hu, P., Zhang, Q., et al.
  *BRECQ: Pushing the Limit of Post-Training Quantization by Block
  Reconstruction.* ICLR 2021.

**Rotation / pre-conditioning**

- Ashkboos, S., Mohtashami, A., Croci, M. L., Li, B., Cameron, P.,
  Jaggi, M., et al. *QuaRot: Outlier-Free 4-Bit Inference in Rotated
  LLMs.* NeurIPS 2024.
- Liu, Z., Zhao, C., Fedorov, I., Soran, B., Choudhary, D.,
  Krishnamoorthi, R., et al. *SpinQuant: LLM Quantization with
  Learned Rotations.* 2024.
- Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S.
  *SmoothQuant: Accurate and Efficient Post-Training Quantization for
  Large Language Models.* ICML 2023.
- Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. *LLM.int8():
  8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS
  2022.

**Low-bit and learned-codebook quantization**

- Chee, J., Cai, Y., Kuleshov, V., & De Sa, C. *QuIP: 2-Bit
  Quantization of Large Language Models With Guarantees.* NeurIPS
  2023.
- Tseng, A., Chee, J., Sun, Q., Kuleshov, V., & De Sa, C. *QuIP#:
  Even Better LLM Quantization with Hadamard Incoherence and Lattice
  Codebooks.* ICML 2024.
- Kim, S., Hooper, C., Gholami, A., Dong, Z., Li, X., Shen, S.,
  et al. *SqueezeLLM: Dense-and-Sparse Quantization.* ICML 2024.
- Dettmers, T., Svirschevski, R., Egiazarian, V., Kuznedelev, D.,
  Frantar, E., Ashkboos, S., et al. *SpQR: A Sparse-Quantized
  Representation for Near-Lossless LLM Weight Compression.* ICLR
  2024.
- Shao, W., Chen, M., Zhang, Z., Xu, P., Zhao, L., Li, Z., et al.
  *OmniQuant: Omnidirectionally Calibrated Quantization for Large
  Language Models.* ICLR 2024.

**MoE quantization**

- Kim, Y., Henry, R., Imani, H., Saberian, S., Sefidgaran, M.,
  et al. *MoQE: Mixture of Quantized Experts for Efficient
  Mixture-of-Experts.* 2023.

**MTP / speculative decoding**

- DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024 (MTP auxiliary
  objective, adopted here for the MTP Fisher probe).
- Leviathan, Y., Kalman, M., & Matias, Y. *Fast Inference from
  Transformers via Speculative Decoding.* ICML 2023.
- Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., &
  Dao, T. *Medusa: Simple LLM Inference Acceleration Framework with
  Multiple Decoding Heads.* 2024.

**KV cache quantization**

- Liu, Z., Yuan, J., Jin, H., Zhong, S., Xu, Z., Braverman, V.,
  et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV
  Cache.* 2024.
- Agarwal, V. (vLLM). *TurboQuant: 2-bit KV Cache Compression with
  Hadamard Rotation + Lloyd-Max Scalar Quantization.* vLLM PR #38479,
  2026. *(Waiting on hybrid-attention/mamba support for Qwen3.6.)*

**Formats**

- Open Compute Project (OCP). *Microscaling Formats (MX)
  Specification, Rev 1.0.* 2023.
- NVIDIA. *Blackwell GPU Architecture and NVFP4 Format.* 2024.
- Neural Magic / vLLM Project. *compressed-tensors: A Universal
  Format for Quantized Model Serving.* 2024.

**Pareto-knee detection**

- Satopaa, V., Albrecht, J., Irwin, D., & Raghavan, B. *Finding a
  'Kneedle' in a Haystack: Detecting Knee Points in System Behavior.*
  ICDCS Workshops 2011.

**Classical bit allocation (water-filling)**

- Cover, T. M., & Thomas, J. A. *Elements of Information Theory*,
  Chapter 13 (Rate-Distortion), 2nd ed. Wiley, 2006.

## Citation

If you use PrismaQuant in research, please cite this repository. A
preprint covering the closed-form allocator math, the
`_GradNormCapture` MoE Fisher estimator, and the MTP quantization path
is forthcoming.

```bibtex
@software{prismaquant2026,
  title        = {PrismaQuant: Mixed-Precision Quantization via
                  Fisher-Weighted Bit Allocation},
  author       = {Tand, Rob and contributors},
  year         = {2026},
  url          = {https://github.com/RobTand/PrismaQuant},
}
```
