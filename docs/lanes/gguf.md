# GGUF lane — llama.cpp / vLLM-GGUF serving

**Deprecated 2026-10-17, per Rob.** The existing code is retained without further
fixes or enhancements. The implementation notes, commands, measurements and
limitations below remain historical reference, not a development roadmap.

*Added 2026-07-06 (branch `claude/gguf-lane-a`); updated 2026-07-30. Status:
historically enabled end-to-end. The two items this header used to call open work — the
GPTQ-into-k-quant rounder and MoE expert stacking — have both landed
(`prismaquant/gguf_gptq.py`, `prismaquant/export_gguf_direct.py`); see
"Known limitations / open work" for what is left. Trust the code and the
measured tables over this prose (prime directive applies).*

## What it is

A second export container. The allocator chooses per-Linear among the GGUF
k-quants — **Q2_K 2.625 / Q3_K 3.4375 / Q4_K 4.5 / Q5_K 5.5 / Q6_K 6.5625 /
Q8_0 8.5 bpw** (fixed bpw: all scales live inside the superblocks) plus BF16
passthrough — and the artifact is a single `.gguf` that llama.cpp serves
natively and vLLM serves via its GGUF path (in-tree ≤0.19, the official
`vllm-gguf-plugin` on current vLLM). No custom kernels anywhere; the 2–3 bpw
regime this unlocks has no NVIDIA-native alternative (NVFP4 is the floor of
the compressed-tensors stack).

## Subsystem map

| Concern | File |
|---|---|
| Formats: field quantizers, emulation QDQ, byte packers | `prismaquant/gguf_formats.py` |
| IQ family (IQ2_XXS..IQ4_NL): grid/codebook quantizers, recon, packers | `prismaquant/gguf_iq_formats.py` (+ grid data `prismaquant/data/iq_grids.pt`, generator `scripts/gen_iq_grids.py`) |
| Registry entries (family `"gguf"`) | `format_registry.py` (GGUF block at the end) |
| Serving constraints (menu + `%256` shape rules) | `serving_profile_specs/gguf.json` |
| Exporter (skeleton requantizer) | `prismaquant/export_gguf.py` |
| Batched cost path (`family == "gguf"` branch + imatrix) | `measure_quant_cost.py` |
| Pipeline stage (`EXPORT_CONTAINER=gguf`) | `run-pipeline.sh` |
| Tests (bit-exactness, profiles, batched==unbatched) | `tests/test_gguf_formats.py` |

## Design invariants

1. **One math path.** Each format has a single field quantizer whose output
   feeds *both* the registry emulation (`quantize_dequantize`, what cost
   measurement scores) and the export byte packer. `gguf-py`'s
   `dequantize(pack(w))` is pinned **bit-identical** to the emulation in
   tests — measured cost and shipped bytes cannot diverge.
2. **Reference-parity scale selection.** The quantizers port llama.cpp's
   `make_qkx2_quants` / `make_qx_quants` (weighted grid + weighted-LS refit,
   sign-aware symmetric search), vectorized in torch, GPU-first. Verified at
   parity: their preset mix re-rendered by our packers measures within ~2.5%
   KL of their own artifact.
3. **imatrix in lockstep.** `PRISMAQUANT_GGUF_IMATRIX` (default **on**)
   applies activation weighting (per-column mean squared activation, llama.cpp
   composition `qw·sqrt(sigma2+x²)`) in the batched *cost* path, and the
   pipeline passes `--imatrix-from-act-cache` to the exporter under the same
   flag — same calibration corpus, same rendering, both sides.
4. **Fail fast, never coerce.** The exporter hard-errors on assignments
   containing non-GGUF formats (allocate with `--target-profile gguf`), and
   on assignment entries that match no skeleton tensor.
5. **Container correctness is delegated.** The exporter requantizes a
   *skeleton* produced by llama.cpp's own `convert_hf_to_gguf.py --outtype
   bf16` — their converter owns metadata/tokenizer/arch/naming; we own only
   tensor bytes. Provenance (git commit, assignment sha256, per-tensor format
   map) is baked into `prismaquant.*` KV metadata.

## Running it

```bash
EXPORT_CONTAINER=gguf TARGET_PROFILE=gguf \
FORMATS=Q2_K,Q3_K,Q4_K,Q5_K,Q6_K,Q8_0,BF16 \
TARGET_BITS=2.95 PRODUCTION_CACHE=0 \
  ./run-pipeline.sh
```

Cost-objective note: use the M6 objective (`h_trace × weight_mse`) for this
lane — `h_trace × output_mse` allocation *lost to llama.cpp's hand heuristic*
at matched size (KLD 3.96 vs 2.73 on the 0.6B screen) while `weight_mse` beat
it (2.33). Embedding/head policy (`GGUF_TOKEN_EMBEDDING_FORMAT`,
`GGUF_OUTPUT_FORMAT`) matters for size-matched comparisons: llama.cpp presets
quantize `token_embd`/`output` (their Q2_K preset uses Q2_K/Q6_K).

Evaluation on the llama.cpp serving metric:

```bash
# once: save base logits from the bf16 skeleton
llama-perplexity -m skeleton.gguf -f wiki.test.raw \
  --kl-divergence-base base_logits.bin --chunks 64 -ngl 99
# per artifact
llama-perplexity -m exported.gguf --kl-divergence-base base_logits.bin \
  --kl-divergence --chunks 64 -ngl 99
```

## Measured status (Qwen3-0.6B, all arms 347 MB, 64-chunk KL-vs-BF16)

| arm | mean KLD | top-1 |
|---|---|---|
| llama.cpp Q2_K preset | 2.728 | 32.1% |
| their mix, our packers (render parity check) | 2.796 | 31.8% |
| **ours: M6 allocation, no imatrix** | **2.327** | **35.0%** |
| llama.cpp preset + imatrix (same corpus) | 0.913 | 55.6% |
| ours: M6 allocation + imatrix, fully consistent | 1.061 | 53.5% |
| **ours: M6 allocation + imatrix + GPTQ** (`--gptq`, research) | **0.890** | **56.9%** |

Measured allocation beats the hand heuristic by −14.7% KL at matched bytes
(and independently rediscovers its v/o/down-get-more shape). The imatrix-RTN
arm loses by +16% (llama.cpp's imatrix-mode `quantize_row_*_impl` paths carry
extra refinement beyond the reference path ported here) — and the
**GPTQ-into-k-quant rounder answers it**: full-Hessian OBS propagation under
the same frozen scales beats their best stack outright on their own harness.
Caveats before any public claim: this is a single-corpus 0.6B screen; the
GPTQ export is not yet scored by the cost stage (allocation was optimized on
imatrix-RTN cost, so a GPTQ-aware cost mode should only widen the margin);
house promotion rules (repeats, held-out corpora, downstream tasks) apply.

## Consistency guards (added after the 2026-07-06 review pass)

- `run-pipeline.sh` **fails fast** when `EXPORT_CONTAINER=gguf` is combined
  with `PRODUCTION_CACHE!=0` or `TARGET_PROFILE!=gguf`. The third condition
  used to be `COST_MODE!=local`; **since 2026-07-30 (re-vet R3) it is a
  render-faithfulness assertion instead.** The contract that matters is
  measured-cost-render == shipped-bytes-render, which is a property of the
  *render*, not of the objective — and it is keyed off the format family, as
  `measure_quant_cost._cost_render_uses_imatrix` always did. A `cached-menu`
  cost render on this lane is now built with `--col-weights`, so it applies the
  same imatrix the exporter does; `PRODUCTION_CACHE=0` still holds because
  `export_gguf` requantizes the skeleton and never reads the export cache.
- `export_native_compressed` **hard-fails** on GGUF formats in an assignment
  (previously they became newly reachable by the silent BF16-coercion branch).
- The exporter **hard-fails** when an imatrix is requested but empty, weights
  zero tensors, or mismatches a tensor's column count; weighted/fallback
  counts plus the imatrix sha256 (a deterministic digest of the calibration
  activations) and the embedding/head policy are baked into `prismaquant.*`
  KV metadata.
- Dead calibration columns (all-zero imatrix weight for a whole sub-block)
  fall back to the format's unweighted weighting instead of erasing real
  weights with a zero scale.
- The `PRISMAQUANT_GGUF_IMATRIX` truthiness parse is identical in the shell
  and Python readers (set-but-empty = default on; `0/false/no/off` any case
  = off).
- The skeleton stage writes-then-renames (no truncated-file skip-gate trust)
  and stamps `MODEL_PATH` in its settings manifest; the pipeline ends with a
  llama.cpp load+greedy-generate smoke on the exported artifact.

## 4B scale check (Qwen3-4B, byte-identical 1669.5 MB arms, 64-chunk KL)

| arm (byte-matched, r = activation rows feeding imatrix+GPTQ) | mean KLD | top-1 |
|---|---|---|
| llama.cpp Q2_K preset + imatrix | 0.461 | 74.3% |
| their mix, our render, r256 | 0.552 | 72.3% |
| our allocation (r256 cost), our render, r256 | 0.644 | 70.1% |
| their mix, our render, **r1024** | 0.497 | 74.2% |
| **ours fully consistent at r1024 (cost+alloc+render)** | **0.510** | **73.5%** |

The initial +40% loss decomposed into input starvation, not structural
deficits: with 1024-row activations feeding the Hessian, the imatrix, AND the
cost measurement, the full stack lands at +10.6% KLD / −0.8pp top-1 vs
llama.cpp's best — of which ~+7.7% is residual render (1024 rows is still
10.5% rank at 4B; try more) and ~+2.6% is allocation vs their hand mix
(validated-frontier selection is the answer there). The M6 objective is
roughly vindicated at 4B once fed proper inputs. Causes isolated on the way:

1. **Tied embeddings**: Qwen3-4B ties `token_embd` to the output head, and
   llama.cpp's preset ships it Q6_K — a Q2_K embedding policy silently gives
   the artifact a 2.6-bit output head (+2.9pp top-1 recovered by matching
   Q6_K at equal bytes). Embedding/head policy must become a measured
   decision, tied-aware; until then, default Q6_K for tied models.
2. **Rank-starved GPTQ Hessian — CONFIRMED**: the act cache held 256 rows —
   25% of a 1024-dim H at 0.6B but 2.6% of a 9728-dim H at 4B, so the damp
   term dominated and GPTQ degenerated toward RTN exactly where llama.cpp's
   rank-agnostic sub-block refinement keeps working. Re-probing with
   `ACTIVATION_ROWS_LIMIT=1024` and re-running the byte-identical render
   cell: **0.552 → 0.497 KLD, top-1 74.19% = parity with their best stack**
   (74.27%). The gguf pipeline lane now defaults to 1024 activation rows;
   the residual +7.7% KLD gap is the next render investigation (more rows —
   1024 is still 10.5% rank at 4B — and/or their imatrix-mode refinements).

The allocation gap is the classic surrogate-mis-ranking regime —
validated-frontier real-KL selection (via a llama.cpp evaluator) is the house
answer. No public quality claim until both land and repeat.

## Known limitations / open work

- ✅ **MoE expert stacking** — SHIPPED in `prismaquant/export_gguf_direct.py`
  (streaming direct-from-HF exporter; experts stacked into one 3-D tensor per
  (layer, projection), one quant type per stacked tensor = experts uniform per
  layer; allocator lookups keyed by the LIVE packed param name,
  `export_gguf_direct.py:210-216,252-253`). Standalone CLI
  (`export_gguf_direct.py:173,380`), not a `run-pipeline.sh` stage.
- ✅ **GPTQ-into-k-quant rounder** — SHIPPED in `prismaquant/gguf_gptq.py`:
  the two-tier scales (fp16 super-scale + quantized sub-scales/mins) are frozen
  from the imatrix-weighted reference search and GPTQ re-decides only the
  integer `q` under them, reusing the NVFP4 lane's Hessian prep and the
  promoted fixed damp 1.0. Covers the six uniform k-quants
  (`gguf_gptq.py:30-43`, `GPTQ_SUPPORTED`); wired at
  `export_gguf.py:322-334` via `--gptq-act-dir`. Still true: the IQ family has
  no GPTQ rounder (grid/codebook indices, not a uniform level), and the GGUF
  **GPTQ** path lives in the exporter, not in `ProductionWeightCache`, so cost
  measurement scores the imatrix-RTN render rather than the GPTQ one. That is a
  rendering-confound risk if allocation ever proves sensitive to it. What
  *changed* on 2026-07-30 (re-vet R3): `_format_supports_render_mechanism` now
  returns True for the gguf family on the `weighted_vq` mechanism, so a
  production-cache render of a GGUF format applies `col_weights` — the imatrix
  half of the confound is closed, the GPTQ half is not.
- **IQ formats** (IQ2_XXS 2.06 / IQ2_XS 2.31 / IQ2_S 2.56 / IQ3_XXS 3.06 /
  IQ3_S 3.44 / IQ4_XS 4.25 / IQ4_NL 4.5 bpw): **implemented** end-to-end
  (`gguf_iq_formats.py`) — registry FormatSpecs, layer_config/serving allow-list
  (IQ4_NL is the only block-32 rung, usable when `in_features % 256 != 0`),
  batched imatrix cost path (routes via `family == "gguf"`), and both exporters
  (`gguf_pack` dispatch). Grid quantizers do an **exhaustive** weighted grid
  argmin (not llama.cpp's neighbour heuristic): emulation == gguf-py-decoded
  bytes bit-exactly for all 7 (`tests/test_gguf_iq_formats.py`), and on real
  Qwen3-0.6B tensors with a matched imatrix the grid types **beat** llama-quantize
  on weighted-MSE (IQ3_XXS ratio ~0.87, all tensors won) while IQ4_XS matches it
  (~1.000). Still **research/candidate**: the vLLM/llama.cpp serving path uses
  MMVQ/Triton (no CUDA MMQ) and has not cleared a perf gate; GPTQ-into-IQ is not
  implemented (IQ renders via imatrix-RTN, `GPTQ_SUPPORTED` excludes them).
- **Selection**: the lane now HAS a KL evaluator behind the
  `validate_assignments_kl` interface —
  `prismaquant/gguf_kl_evaluator.py:measure_assignment_kl` wraps
  `llama-perplexity --kl-divergence-base` and returns
  `(mean, per_sequence, stats)` under the gold lane's key names
  (`kl_mean`/`kl_p99`/`kl_max`/`nll_mean`), so a GGUF row and a native row are
  comparable (re-vet R16, declared in `prismaquant/lane_specs/gguf.json`).
  Honest limits: `per_sequence` is EMPTY and `kl_tail_domain="aggregate"`,
  because llama-perplexity reports token-domain quantiles rather than
  per-sequence values — a tail-veto reading `kl_p99` here is reading a
  different domain than the resident evaluator's. Parsing is unit-tested
  against canned output from both spellings llama.cpp has shipped; the live
  path (invoking the binary) is integration and has not been run.
  `run-pipeline.sh` still runs `SELECTION_MODE=surrogate` on this lane — the
  evaluator exists, the frontier loop is not wired to it — and the ~3 bpw
  cliff is exactly where measured selection should pay.
- **Gold metric**: the tables above are the llama.cpp KL harness (the serving
  metric *for this lane's runtime*); vLLM-GGUF serving of the same artifacts
  was smoke-verified on the 0.19.2 venv but not KL-measured there.
- **Ship gate**: the pipeline's llama.cpp smoke proves load+generate only. The
  *declaration* now exists (`prismaquant/lane_specs/gguf.json`, re-vet R16):
  llama-server exposes an OpenAI-compatible endpoint and
  `validate_quantized_model.py` is endpoint-agnostic, so the same PPL + p99
  per-prompt NLL gate applies unchanged — it has simply not been RUN on this
  lane. Gates are **advisory** (recorded in `shipcard.json`); whether they
  become blocking is deferred to Robert.
- **imatrix vector mismatch (minor)**: the cost path derives per-column weights
  from the chunk-truncated activation rows it bmm's with, the exporter from
  the full cached rows — same estimator, slightly different sample counts.
  Unify if allocation ever proves sensitive to it (0.6B was not).
- ✅ **Packed-MoE expert imatrix** — SHIPPED in `prismaquant/moe_imatrix.py`:
  `synthesize_packed_expert_col_weights` replays the routed forward from the
  CHECKPOINT tensors (no model load) to produce the two entries the activation
  cache structurally cannot hold (`<qn>.gate_up_proj`, `<qn>.down_proj`), from
  ONE shared source for both the exporter and the packed-expert cost — closing
  the rendering confound. Wired at `run-pipeline.sh:625,990,1561`.
- **Embedding/head formats are an operator policy**, not a measured
  allocator decision — a principle-2 debt; the flags are recorded in
  provenance KVs so size-matched claims stay auditable.

## Hardware targets — Strix Halo canceled 2026-07-31

Access to the only gfx1151 system was lost, so all Strix validation and ROCm
kernel work is canceled. The unqualified Gridbook HIP prototype was removed;
there is no supported or passively tracked Gridbook-on-Strix lane. Earlier
measurements remain in dated audit/history documents only. Reopening this target
requires new hardware and a fresh qualification plan; it is not part of the
current GGUF or CB release backlog.
