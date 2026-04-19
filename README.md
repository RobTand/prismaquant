# PrismQuant

**Mixed-precision quantization for LLMs. Every layer refracts into a different format based on its sensitivity.**

PrismQuant measures the actual per-layer curvature of the loss and the
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

## Validated result

Qwen3.6-35B-A3B-MoE at target **4.75 bpp**:

| Metric | Source BF16 | PrismQuant | Delta |
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

## Quick start

Three commands on a machine with the model cached locally:

```bash
export MODEL_PATH=/path/to/Qwen3.6-35B-A3B
export WORK_DIR=./dq-runs/qwen36
export FORMATS=NVFP4,MXFP8_E4M3,BF16
export TARGET_BITS=4.75

./quantization/prismquant/run-pipeline.sh
```

That runs probe → cost → allocator → native export.  Extend the probe
to cover MTP heads:

```bash
python -m quantization.prismquant.mtp_probe \
  --model $MODEL_PATH --nsamples 4 --seqlen 256 \
  --activation-cache-dir $WORK_DIR/act_mtp \
  --output $WORK_DIR/artifacts/mtp_probe.pkl

python -m quantization.prismquant.mtp_cost \
  --model $MODEL_PATH \
  --activation-cache-dir $WORK_DIR/act_mtp \
  --output $WORK_DIR/artifacts/mtp_cost.pkl
```

Merge the MTP probe / cost into the body artifacts (see
`run-pipeline.sh`), re-run the allocator and exporter, then serve:

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

#### 7. MTP extensions — `mtp_probe.py`, `mtp_cost.py`

Transformers v5 drops `mtp.*` weights on load (they're in the class's
`_keys_to_ignore_on_load_unexpected`) because MTP is a vLLM-only
feature. PrismQuant instantiates a standalone `MtpModule` (HF
`Qwen3_5MoeDecoderLayer` plus the `pre_fc_norm_*`, `fc`, and `norm`
exactly per vLLM's `Qwen3_5MultiTokenPredictor` forward), loads the
MTP weights directly from safetensors, then runs the standard Fisher
probe against the MTP auxiliary objective

    loss = CE( lm_head( MTP(embed_{t+1}, body_hidden_t) ), ids_{t+2} )

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
practical way to sweep the whole model cheaply. PrismQuant addresses
the missing cross-layer terms by:

- measuring sparse pairwise interactions only for the most important
  units near the knee (`measure_interactions.py`)
- refining the knee locally with those terms
  (`quadratic_refine_allocator.py`)
- calibrating the refined frontier against actual KL
  (`calibrate_allocator.py`)

This keeps memory bounded while still capturing the interaction
structure that recent MPQ literature shows matters.

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

## What's deferred

- **Visual encoder**: weights pass through as BF16 from source. The
  probe ran the regex for visual blocks, but `stage_text_only` strips
  `vision_config` from the staged config so the loaded model never
  instantiates visual modules. Real multimodal calibration (load
  `Qwen3_5MoeForConditionalGeneration` + use `AutoProcessor` on an
  image dataset) is straightforward follow-up but not yet in the
  pipeline.
- **Non-qwen3.5/3.6 MTP support**: The `MtpModule` currently mirrors
  qwen3_5's forward exactly. Other MTP-bearing architectures
  (deepseek_mtp, glm4_moe_mtp, etc.) need their own forward construction.
- **Fused-sibling promotion for arbitrary model families**: The fused
  patterns are hand-listed in `allocator.py`. A model-profile registry
  (see "Roadmap") will make this derivable from vLLM's model registry.

## Roadmap

- **Model-profile registry** — auto-derive fused-sibling patterns,
  packed-expert names, and layer-type info from vLLM's model registry
  + HF config, so PrismQuant works on arbitrary architectures without
  hand-editing allocator patterns.
- **Multimodal calibration** — load `ForConditionalGeneration`, run
  image+text pairs through a multimodal processor, so visual encoder
  blocks get real Fisher stats.
- **MXFP6 serving** — enable once vLLM kernels land.
- **Speculative-decoding acceptance-rate tuning** — per-MTP-Linear
  format allocation that jointly optimizes target model quality and
  MTP-draft acceptance rate.

## Citation

If you use PrismQuant in research, please cite this repository. A
preprint covering the closed-form allocator math, the `_GradNormCapture`
MoE Fisher estimator, and the MTP quantization path is forthcoming.
