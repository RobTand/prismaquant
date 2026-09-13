# Loading the device: where fp4 x fp4 crosses fp8 x fp8 — Qwen3-0.6B, Tessera layer 0

Status: **measurement record.** 2026-09-13, branch
`claude/load-sweep-the-fp4-route`, PQ #568. Stacks on PQ #563, which produced
§8 of `docs/measurements/prefill-frontier-qwen3-0.6b-2026-09-13.md`; that §8 is
amended by this record in three places.

Receipts root (`$LS` throughout):

```
/mnt/shared/tessera-runs/receipts/fp4-load-sweep-qwen3-0.6b-20260913
```

`$NR` (#563's root) and `$R` (#237's root) are frozen and were not written to.

---

## 0. Outcome, first

1. **fp4 does not cross fp8 on the operator sum, at any M that fits.** Swept
   512 → 131 072 tokens, all intervals disjoint at every M, the fp4 route costs
   **1.052x** fp8 at M = 512 and **1.295x** at M = 131 072, peaking at **1.350x**
   at M = 32 768. It gets *worse* with load, not better.
2. **The device was genuinely loaded.** The fp8 arm holds **0.586–0.644** of the
   140 W envelope at every M and the fp4 arm **0.368–0.481**; Netdata's
   box-level view of the whole action reads mean **61.3 W** (0.438) with a peak
   of **94 W** (0.671). #563's arms were measured at 0.058. The hypothesis is
   **tested**, not left untested.
3. **Rob's expectation is right about the arithmetic.** The fp4
   `torch._scaled_mm` kernel — `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue…`,
   a genuine sm_120-family block-scaled schedule, not a fallback — costs
   **0.60x to 0.72x** the fp8 GEMM's device time at **every** M measured. fp4
   x fp4 is 1.39x to 1.66x faster than fp8 x fp8 on this hardware.
4. **The route loses to one kernel the fp8 route does not have.**
   `nvfp4_route.py:238` applies `y = y * layer.tessera_epilogue_scale` — a
   **Python float**, built at `:207` as `float(prepared.global_scale) / gs` — as
   a separate full-output elementwise pass. At M = 8192 it costs **913.7 us**
   summed over the three units, *more than the fp4 GEMM's own 737.6 us*, and it
   runs at about **243 GB/s**, i.e. bandwidth-bound on the unified pool.
5. **fp4 already wins work per joule, at every M: 1.15x to 1.46x.** 1718 against
   1176 GFLOP/J at M = 512; 1255 against 1096 at M = 131 072.
6. **One unit does cross, and the crossing is exactly where the mechanism says
   it should be.** `mlp.down_proj` (N = 1024, a third of the other two units'
   output width, so a third of the epilogue) is **faster in fp4 than in fp8**
   at M = 1024 (**0.810x**), 2048 (0.939x), 4096 (0.875x) and 8192 (0.940x),
   all with disjoint intervals, and loses again from M = 16 384 once the
   epilogue saturates bandwidth. **The crossing exists; it is a property of the
   output width, and the wide units never reach it.**
7. **Axis 2 (unit size) is untested.** No larger model's Linears are encoded at
   both formats, and no export campaign was started to make one (§2).

The short version: **#563's headline is not retracted, but its scope is.** fp4
loses at the *route*, on one elementwise pass, not at the *arithmetic* — and it
loses under load, not only under dispatch overhead.

## 1. The question, and why #563 could not answer it

#563 measured the fp4 route class against fp8 and bf16 on the three layer-0
Qwen3-0.6B MLP units and found fp4 **slower** at prefill: 0.133104 ms against
0.127616 ms, about 1.04x, disjoint intervals. It recorded the expectation it
disappointed — that fp4 x fp4 `torch._scaled_mm` should beat fp8 x fp8 on
Blackwell — and it recorded the reason the finding was narrow: at M = 512 the
same three units make **1.01x** as much prefill work as decode work across a
512-fold change in M (fp8 1.13x, bf16 3.31x), which is the signature of a cost
that does not scale with M.

That reading is confirmed here and given a number. **At M = 512 the apply is
about half dispatch.** The CUDA-event median of one fp4 three-unit apply is
0.134928 ms; the sum of every kernel's self device time in the same apply is
0.064326 ms. The gap — 0.070602 ms, **52%** — is launch and Python, not
arithmetic. By M = 65 536 the same gap is 0.035 ms against 17.64 ms of apply,
**0.2%**. **A route comparison taken at M = 512 is a comparison of two dispatch
paths that happen to end in a GEMM.**

One correction to #563 §8.1 before anything else. That section reports a mean of
**8.15 W** over the `28afcdc891db` action window and reads it as "the box is not
loaded". The number is right; the inference needed a caveat it did not carry.
Netdata's `nvidia_smi.gpu_power_draw` collector on sparklina runs at
`update_every = 10` (tier 0), so a 324-second action window holds about 32 real
samples, almost all taken while the container was importing torch, installing
the plugin, preparing 9 cells and writing evidence — 40 applies of about 44 us
each is under 2 ms of GPU work inside 324 seconds of action. **8.15 W is the
action's mean, not the apply's.** Measured with an in-process sampler around a
*sustained* apply loop, the same M = 512 applies on the same units and the same
GPU draw 51.5 W (fp4) and 90.2 W (fp8). #563's structural finding stands; its
power sentence is superseded by §4.

## 2. What this sweeps, and what it does not

**Axis 1 — token count M.** One process, all nine cells resident, arms
byte-matched, M swept 512, 1024, 2048, 4096, 8192, 16 384, 32 768, 65 536,
131 072 over the same three layer-0 MLP units. **M outer, arms inner**, so the
three arms at one M share clock and thermal state.

**The arms.** The same nine cells #563 §8.4 priced: `mlp.gate_proj`,
`mlp.up_proj`, `mlp.down_proj` at `TESSERA_E2M1_K2_R896` (fp4 x fp4, contract
`e2m1_group16_ue4m3_static`), `TESSERA_E4M3_K1_R1006` (fp8 x fp8,
`fp8_per_token_dynamic`) and `TESSERA_BF16_K1_R1792` (`bf16_unquantized`), all
in `TESSERA_SERVE_MODE=resident`. The fp4 and fp8 arms are byte-matched to
0.45% (4 722 278 B against 4 701 232 B), so this is a route-class comparison at
one rate. The three attention units #563 found `numerical_refused` on the
activation gate are **not** in this sweep; no tolerance was widened and no
refused cell was priced.

**Axis 2 — unit size. Not available, and not started.** A second axis was
allowed only if a larger model's Linears were *already* encoded at both
`TESSERA_E4M3_K1_*` and `TESSERA_E2M1_K2_R896` and reachable through the same
cell harness. They are not. `/mnt/shared/tessera-runs/receipts/` holds four
Tessera receipt trees and every one is Qwen3-0.6B
(`399-qwen3-0.6b-2026-09-13`, `399-qwen3-0.6b-20260913`,
`frontier-qwen3-0.6b-20260913`, `frontier-fp4-qwen3-0.6b-20260913`). Producing a
larger unit means running `pq_frontier_native_cells.py prepare`, which renders
and **encodes wires on a GPU** from the model
(`$NR/stage/control/g1_prepare.sh:49-52`) — an export campaign. It was not
started. **Axis 2 is untested and this record makes no claim about unit size.**
§6 does, however, show that the *output width* N moves the answer, which is the
part of the size axis that mattered here.

**How M above 512 is built, and what that costs the claim.** A prepared cell's
`tensors.safetensors` stores exactly the 512-row prefill activation the
calibration produced, and `prepare_native_inputs`
(`prismaquant/native_operator_panel.py`) refuses `prefill_rows` above
`activation_rows.shape[0]` (2048 with the 4x512 calibration). Larger M is
therefore built by **tiling** the frozen 512-row input, `base.repeat(reps, 1)`.
Three gates make that honest, and all three ran at every one of the 81 points:

* **the anchor.** At M = 512 every arm is compared against the cell's own frozen
  `prefill.reference_output` and `prefill.reference_qdq` at the panel's declared
  `atol = rtol = 0.015625`. Nine of nine passed; nothing was widened.
* **the head.** At every M the first 512 output rows of the tiled apply are
  compared against that same frozen reference at the same tolerance.
* **replica equality.** The tiled output is reshaped to `(reps, 512, N)` and
  `(blocks - blocks[0:1]).abs().max()` must be **exactly 0.0**. It was, at every
  point, on all three arms. A tiled input is a repeated input; if a route's
  numerics depended on where a row sat in the batch, this is where it would show.

Every point also re-checks the emitted route: `policy`, `symbol`, `decoder`,
`contract`, `state == "served"`, `reason is None` and the shape string
`M<m>:N<n>:K<k>`. No arm fell back at any M.

What tiling does **not** establish: a real M-token prefill has distinct rows
whose per-token dynamic fp8 scales and group-16 fp4 block scales vary more than
a 512-row block repeated. The kernel work is identical — the same shapes, the
same tiles, the same bytes moved — but the *value* distribution is periodic.
Read every number below as the cost of an M-row GEMM on this hardware, not as a
claim about a particular corpus.

## 3. The instruments

Two, because they answer different questions (principle 15).

**In-process, fine.** `pynvml` sampled at 100 ms **inside the measuring
process**, on `GPU-b1eceeea-fec7-371e-2cf3-cd10f2e7b705` — the same GPU #563 and
#237 used, 4801 samples over the run. Each of the 81 points runs a **sustained
apply loop for 5 s** and the reported watts are the mean over that window after
a 1.5 s settle prefix is discarded. A single apply is microseconds long and no
power meter can see it; the sustained loop is what makes the number mean
anything. `torch.profiler` supplies the per-kernel device time
(`total_self_device_us` over `replays`) in a separate final pass, and the chrome
traces are kept at `$LS/sweep-L1/profile/`.

**Box-level, coarse.** Netdata `nvidia_smi.gpu_power_draw` on node `sparklina`,
`update_every = 10`, so 49 real samples over the 8-minute action. It cannot
resolve a 5-second arm window and is reported in §4 as the run-level check it
is.

**Bootstrap.** Each unit's own samples (32 CUDA-event samples per point, 8
warmup) resampled with replacement, re-reduced by median, 10 000 draws, seed
20260913; the three unit medians are summed per draw, so the interval on a
route's sum is an interval on the sum — the same estimator #563 §8.4 and #237
§3.2 use.

**One deviation to record.** The power sampler thread was running during the
CUDA-event timing pass as well as during the sustained loop. It is a 100 ms
NVML poll on a separate thread, and the M = 512 event medians reproduce #563's
independent measurement to under 0.6% (0.134928 against 0.133104 for fp4;
0.128272 against 0.127616 for fp8), so it did not measurably perturb the timing.
It is recorded rather than assumed away.

## 4. The sweep

| M | arm | ms (median of sums) | 95% CI | vs fp8 | sustained wall ms | profiler device ms | dispatch gap ms | W | frac 140 W | TFLOP/s | GFLOP/J | SM MHz |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | fp4 | 0.134928 | [0.133808, 0.135648] | 1.052x | 0.109269 | 0.064326 | 0.070602 | 51.5 | 0.368 | 71.6 | 1718 | 2366 |
| 512 | fp8 | 0.128272 | [0.127504, 0.128496] | 1.000x | 0.091222 | 0.081547 | 0.046725 | 90.2 | 0.644 | 75.3 | 1176 | 2349 |
| 512 | bf16 | 0.291840 | [0.291008, 0.292912] | 2.275x | 0.268237 | 0.263325 | 0.028515 | 72.1 | 0.515 | 33.1 | 585 | 2278 |
| 1024 | fp4 | 0.225120 | [0.224768, 0.226272] | 1.142x | 0.177344 | 0.169600 | 0.055520 | 67.3 | 0.481 | 85.9 | 1719 | 2333 |
| 1024 | fp8 | 0.197072 | [0.196064, 0.197888] | 1.000x | 0.168739 | 0.162023 | 0.035049 | 90.2 | 0.644 | 98.1 | 1275 | 2223 |
| 1024 | bf16 | 0.569232 | [0.567776, 0.570384] | 2.888x | 0.546361 | 0.538197 | 0.031035 | 70.9 | 0.506 | 34.0 | 589 | 2301 |
| 2048 | fp4 | 0.468928 | [0.468336, 0.469888] | 1.232x | 0.434954 | 0.425346 | 0.043582 | 60.7 | 0.433 | 82.4 | 1538 | 2359 |
| 2048 | fp8 | 0.380768 | [0.379664, 0.381840] | 1.000x | 0.363088 | 0.352018 | 0.028750 | 86.8 | 0.620 | 101.5 | 1227 | 2245 |
| 2048 | bf16 | 1.348864 | [1.345632, 1.350528] | 3.542x | 1.340013 | 1.312977 | 0.035887 | 64.9 | 0.463 | 28.7 | 508 | 2357 |
| 4096 | fp4 | 1.024640 | [1.023120, 1.026384] | 1.264x | 0.997054 | 0.961637 | 0.063003 | 55.9 | 0.399 | 75.5 | 1438 | 2377 |
| 4096 | fp8 | 0.810464 | [0.808288, 0.812784] | 1.000x | 0.832330 | 0.767153 | 0.043311 | 82.1 | 0.586 | 95.4 | 1146 | 2277 |
| 4096 | bf16 | 2.809600 | [2.806784, 2.814160] | 3.467x | 2.830043 | 2.786259 | 0.023341 | 63.8 | 0.456 | 27.5 | 516 | 2344 |
| 8192 | fp4 | 2.201888 | [2.198960, 2.204944] | 1.264x | 2.168624 | 2.159094 | 0.042794 | 53.7 | 0.384 | 70.2 | 1361 | 2392 |
| 8192 | fp8 | 1.742528 | [1.741536, 1.743616] | 1.000x | 1.732034 | 1.674587 | 0.067941 | 82.7 | 0.591 | 88.7 | 1081 | 2322 |
| 8192 | bf16 | 5.812128 | [5.798128, 5.821888] | 3.335x | 5.784162 | 5.728213 | 0.083915 | 62.5 | 0.446 | 26.6 | 485 | 2364 |
| 16384 | fp4 | 4.530768 | [4.527072, 4.534864] | 1.316x | 4.500162 | 4.454167 | 0.076601 | 53.7 | 0.383 | 68.3 | 1303 | 2403 |
| 16384 | fp8 | 3.443712 | [3.440288, 3.446688] | 1.000x | 3.454978 | 3.421662 | 0.022050 | 83.1 | 0.594 | 89.8 | 1079 | 2325 |
| 16384 | bf16 | 11.416624 | [11.403616, 11.427472] | 3.315x | 11.407283 | 11.324702 | 0.091922 | 64.0 | 0.457 | 27.1 | 484 | 2362 |
| 32768 | fp4 | 9.170976 | [9.164416, 9.177888] | 1.350x | 9.148400 | 9.133869 | 0.037107 | 54.4 | 0.389 | 67.4 | 1267 | 2403 |
| 32768 | fp8 | 6.791040 | [6.786176, 6.793088] | 1.000x | 6.861204 | 6.761276 | 0.029764 | 82.5 | 0.589 | 91.1 | 1094 | 2314 |
| 32768 | bf16 | 22.661920 | [22.629568, 22.693472] | 3.337x | 22.662953 | 22.655179 | 0.006741 | 65.6 | 0.468 | 27.3 | 480 | 2384 |
| 65536 | fp4 | 17.642208 | [17.629376, 17.656528] | 1.304x | 17.641773 | 17.606896 | 0.035312 | 56.4 | 0.403 | 70.1 | 1257 | 2403 |
| 65536 | fp8 | 13.526400 | [13.519760, 13.540576] | 1.000x | 13.715353 | 13.500329 | 0.026071 | 82.7 | 0.591 | 91.4 | 1092 | 2307 |
| 65536 | bf16 | 45.038145 | [45.002944, 45.113952] | 3.330x | 45.166554 | 45.031040 | 0.007104 | 65.8 | 0.470 | 27.5 | 474 | 2359 |
| 131072 | fp4 | 35.239103 | [35.210272, 35.274784] | 1.295x | 35.260605 | 35.298907 | -0.059804 | 56.5 | 0.404 | 70.2 | 1255 | 2405 |
| 131072 | fp8 | 27.209920 | [27.144176, 27.235216] | 1.000x | 27.365608 | 26.845940 | 0.363980 | 82.6 | 0.590 | 90.9 | 1096 | 2308 |
| 131072 | bf16 | 90.103119 | [90.037566, 90.193647] | 3.311x | 90.222927 | 89.813437 | 0.289682 | 65.0 | 0.464 | 27.5 | 488 | 2380 |

**Reading the columns.** "ms" is the median of the per-draw sum of the three
units' CUDA-event medians, with its bootstrap interval. "sustained wall ms" is
the independent wall-clock per apply inside the 5-second power loop — it agrees
with the event median throughout, which is the cross-check that the power window
is measuring the same work. "profiler device ms" is the sum of every kernel's
self device time in one apply. "dispatch gap" is the first minus the third: the
part of the apply that is not a kernel. The two negative/large gaps at
M = 131 072 are the profiler's own overhead at 90 ms applies, not a measurement
of dispatch; the gap column is only meaningful while it is positive and small
relative to the apply.

**The device is loaded.** Every point in this table is between **0.37 and 0.64**
of the 140 W envelope. #563's arms sat at 0.058. The box-level Netdata view of
the whole `4001d2a67bad` action window (2026-09-13T18:55:41Z → 19:03:50Z, 49
real samples at 10 s) reads mean **61.3 W** = **0.438** of the envelope, min
5 W, max **94 W** = 0.671 — consistent with an in-process instrument that saw
51–90 W inside each arm's window and idle between them. **This is the answer to
"was it ever tested where it would win": yes, it was.**

**fp8 draws the most power and fp4 the least, at every M.** That is not fp4
being idle — it is fp4 spending a third to a half of its time in a
bandwidth-bound elementwise kernel, which draws less than a tensor-core GEMM.
It is also why fp4 wins work per joule while losing wall clock.

**Clocks.** SM clock means are recorded per window and sit at 2223–2405 MHz.
fp8 runs 17–114 MHz slower than fp4 at the same M, consistent with its higher
power draw. No arm was clock-limited into a different regime. The GEMM-only
achieved rate (kernel FLOP over kernel device time) is 182–234 TFLOP/s for fp4
and 129–146 TFLOP/s for fp8 across the sweep, with fp4's dipping to 182 around
M = 32 768 and recovering; that dip is recorded and not explained.

## 5. Where the time goes

The kernels, profiled, summed over the three units, in microseconds of device
time per apply. "quantize" is `cvt_fp16_to_fp4` / `fp8_quant`; "GEMM" is the
CUTLASS or nvjet matmul; "epilogue/other" is everything else, which on the fp4
arm is entirely the elementwise pass of §6.

| M | arm | GEMM us | activation quantize us | epilogue/other us | device total us | GEMM vs fp8 |
|---:|---|---:|---:|---:|---:|---:|
| 512 | fp4 | 42.9 | 10.8 | 10.6 | 64.3 | 0.631x |
| 512 | fp8 | 68.1 | 13.5 | 0.0 | 81.5 | 1.000x |
| 512 | bf16 | 140.9 | 0.0 | 122.5 | 263.3 | 2.069x |
| 1024 | fp4 | 82.8 | 21.1 | 65.7 | 169.6 | 0.602x |
| 1024 | fp8 | 137.4 | 24.6 | 0.0 | 162.0 | 1.000x |
| 1024 | bf16 | 308.8 | 0.0 | 229.4 | 538.2 | 2.248x |
| 2048 | fp4 | 180.9 | 96.8 | 147.6 | 425.3 | 0.621x |
| 2048 | fp8 | 291.4 | 60.6 | 0.0 | 352.0 | 1.000x |
| 2048 | bf16 | 624.5 | 0.0 | 688.4 | 1313.0 | 2.143x |
| 4096 | fp4 | 344.3 | 224.5 | 392.8 | 961.6 | 0.652x |
| 4096 | fp8 | 527.8 | 239.3 | 0.0 | 767.2 | 1.000x |
| 4096 | bf16 | 1192.6 | 0.0 | 1593.7 | 2786.3 | 2.259x |
| 8192 | fp4 | 737.6 | 507.9 | 913.7 | 2159.1 | 0.635x |
| 8192 | fp8 | 1162.1 | 512.4 | 0.0 | 1674.6 | 1.000x |
| 8192 | bf16 | 2368.3 | 0.0 | 3359.9 | 5728.2 | 2.038x |
| 16384 | fp4 | 1609.5 | 979.0 | 1865.6 | 4454.2 | 0.673x |
| 16384 | fp8 | 2391.3 | 1030.4 | 0.0 | 3421.7 | 1.000x |
| 16384 | bf16 | 4592.9 | 0.0 | 6731.8 | 11324.7 | 1.921x |
| 32768 | fp4 | 3392.8 | 1911.1 | 3830.1 | 9133.9 | 0.721x |
| 32768 | fp8 | 4708.8 | 2052.5 | 0.0 | 6761.3 | 1.000x |
| 32768 | bf16 | 9206.0 | 0.0 | 13449.2 | 22655.2 | 1.955x |
| 65536 | fp4 | 6076.9 | 3797.2 | 7732.8 | 17606.9 | 0.646x |
| 65536 | fp8 | 9402.6 | 4097.7 | 0.0 | 13500.3 | 1.000x |
| 65536 | bf16 | 18234.2 | 0.0 | 26796.8 | 45031.0 | 1.939x |
| 131072 | fp4 | 12282.5 | 7602.0 | 15414.4 | 35298.9 | 0.656x |
| 131072 | fp8 | 18726.8 | 8119.2 | 0.0 | 26845.9 | 1.000x |
| 131072 | bf16 | 36253.7 | 0.0 | 53559.7 | 89813.4 | 1.936x |

Three things fall out of this table, and none of them is what #563 guessed.

**(a) The fp4 GEMM wins, everywhere, by a lot.** 0.60x to 0.72x the fp8 GEMM's
device time at every M — 1.39x to 1.66x faster. This is the measurement Rob's
expectation was about, and it holds. The kernel is
`cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue…`: an sm_120
block-scaled tensor-core schedule, on sm_121 hardware, which is the native
family. Nothing here rides a fallback.

**(b) The activation quantize is *not* the difference.** #563 §8.4 named
"the shape of that work" — one fp32 scale per token for fp8 against group-16
UE4M3 block scales for fp4 — as the plausible mechanism, while explicitly
declining to claim it. Profiled, the two cost the same: at M = 8192, fp4's
quantize is **507.9 us** and fp8's is **512.4 us**; at M = 131 072, 7602.0
against 8119.2. **fp4's block-scale quantize is, if anything, marginally
cheaper.** The guess was wrong and is retracted by §8.4's amendment.

**(c) bf16 is not in the race and its GEMM is the smaller half of why.** The
bf16 route's own elementwise column (the fp32 row-scale it deliberately applies
on the GEMM's fp32 output, `bf16_route.py`) is *larger* than its GEMM at every
M — 53 560 us against 36 254 us at M = 131 072. Two of its three units ride
`cutlass_80_tensorop_s16816gemm_bf16_*`, an **sm80** schedule on sm_121
hardware, while `down_proj` gets `nvjet_sm121_tss_mma_*`. That is recorded here
because principle 9 cares about it; it is not this record's subject and no claim
is made about why the dispatcher splits that way.

## 6. The one kernel the fp8 route does not have

The fp4 apply, in the frozen producer tree
`producer-source-d403cc5a31`, `src/tessera/serving/nvfp4_route.py`:

```python
y = torch._scaled_mm(a_q, b.t(), scale_a=a_scale, scale_b=scale_b,
                     out_dtype=torch.bfloat16)          # :236-237
y = y * layer.tessera_epilogue_scale                    # :238
```

and `tessera_epilogue_scale` is built once, at load, at `:207`:

```python
layer.tessera_epilogue_scale = float(prepared.global_scale) / gs
```

`gs` is itself a Python float (`:171`). **So the fp4 route reads the entire
M x N bf16 output out of memory and writes it back, to multiply it by one
Python float.** The fp8 route's apply (`fp8_route.py:444-484`) has no
counterpart pass: its `_scaled_mm` result is reshaped and returned.

That pass is what the whole finding is made of, and it is bandwidth-bound:

| M | unit | epilogue us | bytes moved | GB/s |
|---:|---|---:|---:|---:|
| 512 | mlp.gate_proj | 4.4 | 6.0 MiB | 1429 |
| 512 | mlp.up_proj | 4.4 | 6.0 MiB | 1429 |
| 512 | mlp.down_proj | 1.8 | 2.0 MiB | 1154 |
| 1024 | mlp.gate_proj | 31.2 | 12.0 MiB | 403 |
| 1024 | mlp.up_proj | 31.3 | 12.0 MiB | 401 |
| 1024 | mlp.down_proj | 3.2 | 4.0 MiB | 1329 |
| 2048 | mlp.gate_proj | 65.7 | 24.0 MiB | 383 |
| 2048 | mlp.up_proj | 65.0 | 24.0 MiB | 387 |
| 2048 | mlp.down_proj | 16.9 | 8.0 MiB | 496 |
| 4096 | mlp.gate_proj | 170.8 | 48.0 MiB | 295 |
| 4096 | mlp.up_proj | 181.1 | 48.0 MiB | 278 |
| 4096 | mlp.down_proj | 40.9 | 16.0 MiB | 410 |
| 8192 | mlp.gate_proj | 407.2 | 96.0 MiB | 247 |
| 8192 | mlp.up_proj | 410.7 | 96.0 MiB | 245 |
| 8192 | mlp.down_proj | 95.7 | 32.0 MiB | 351 |
| 16384 | mlp.gate_proj | 822.6 | 192.0 MiB | 245 |
| 16384 | mlp.up_proj | 822.7 | 192.0 MiB | 245 |
| 16384 | mlp.down_proj | 220.4 | 64.0 MiB | 304 |
| 32768 | mlp.gate_proj | 1655.6 | 384.0 MiB | 243 |
| 32768 | mlp.up_proj | 1644.2 | 384.0 MiB | 245 |
| 32768 | mlp.down_proj | 530.2 | 128.0 MiB | 253 |
| 65536 | mlp.gate_proj | 3320.5 | 768.0 MiB | 243 |
| 65536 | mlp.up_proj | 3334.4 | 768.0 MiB | 242 |
| 65536 | mlp.down_proj | 1077.9 | 256.0 MiB | 249 |
| 131072 | mlp.gate_proj | 6625.9 | 1536.0 MiB | 243 |
| 131072 | mlp.up_proj | 6626.7 | 1536.0 MiB | 243 |
| 131072 | mlp.down_proj | 2161.9 | 512.0 MiB | 248 |

For the two wide units (N = 3072) the rate settles at **243–245 GB/s** from
M = 8192 upward and stays there across a 16-fold further increase in M — the
LPDDR5X ceiling of this box, reached by a kernel that does one multiply per
2 bytes read. The small-M rows read higher only because the pass is too short
to be bandwidth-bound there (at M = 512 it is 1.5 us and is measuring launch).

The consequence in one line: at M = 8192 the fp4 route pays **913.7 us** for
this pass and saves **424.5 us** on its GEMM against fp8. It cannot win.

**What would remove it is a hypothesis, not a result.** Neither `scale_a` nor
`scale_b` on the nvfp4 `_scaled_mm` call can absorb the scalar — both are
quantized UE4M3 planes, and rounding a global into them is not the same
arithmetic. Removing the pass therefore means either a kernel epilogue argument
that `torch._scaled_mm` does not currently take, or fusing the multiply into
whatever consumes `y`. **Neither was measured and neither is claimed.** What
*is* measured is the counterfactual's size: subtract the pass and the fp4 arm's
device total at M = 8192 would be 1245.4 us against fp8's 1674.6 — but that
subtraction is arithmetic on this table, not an experiment, and it is recorded
as such. `docs/ARCHITECTURE.md` debt **D40** carries the open question.

## 7. The crossing — per unit

The sum never crosses. One unit does, and it crosses exactly where §6 predicts:
`mlp.down_proj` has N = 1024 against 3072 for the other two, so its epilogue
costs a third as much for the same GEMM saving.

| M | unit | N | fp4 ms [95% CI] | fp8 ms [95% CI] | fp4/fp8 | intervals disjoint |
|---:|---|---:|---|---|---:|---|
| 512 | `mlp.down_proj` | 1024 | 0.04499 [0.04469, 0.04531] | 0.04467 [0.04397, 0.04483] | 1.007x | **no** |
| 512 | `mlp.gate_proj` | 3072 | 0.04485 [0.04403, 0.04522] | 0.04144 [0.04131, 0.04154] | 1.082x | yes |
| 512 | `mlp.up_proj` | 3072 | 0.04509 [0.04421, 0.04557] | 0.04216 [0.04203, 0.04224] | 1.069x | yes |
| 1024 | `mlp.down_proj` | 1024 | 0.05670 [0.05656, 0.05754] | 0.07002 [0.06987, 0.07050] | 0.810x **<** | yes |
| 1024 | `mlp.gate_proj` | 3072 | 0.08406 [0.08390, 0.08448] | 0.06310 [0.06224, 0.06386] | 1.332x | yes |
| 1024 | `mlp.up_proj` | 3072 | 0.08435 [0.08410, 0.08502] | 0.06395 [0.06333, 0.06406] | 1.319x | yes |
| 2048 | `mlp.down_proj` | 1024 | 0.11493 [0.11477, 0.11578] | 0.12237 [0.12211, 0.12250] | 0.939x **<** | yes |
| 2048 | `mlp.gate_proj` | 3072 | 0.17723 [0.17718, 0.17757] | 0.12909 [0.12835, 0.12957] | 1.373x | yes |
| 2048 | `mlp.up_proj` | 3072 | 0.17677 [0.17619, 0.17712] | 0.12931 [0.12867, 0.13032] | 1.367x | yes |
| 4096 | `mlp.down_proj` | 1024 | 0.25792 [0.25746, 0.25827] | 0.29486 [0.29421, 0.29586] | 0.875x **<** | yes |
| 4096 | `mlp.gate_proj` | 3072 | 0.38339 [0.38307, 0.38422] | 0.26275 [0.26182, 0.26374] | 1.459x | yes |
| 4096 | `mlp.up_proj` | 3072 | 0.38333 [0.38187, 0.38464] | 0.25285 [0.25136, 0.25446] | 1.516x | yes |
| 8192 | `mlp.down_proj` | 1024 | 0.58306 [0.58234, 0.58536] | 0.62056 [0.61990, 0.62091] | 0.940x **<** | yes |
| 8192 | `mlp.gate_proj` | 3072 | 0.81107 [0.80896, 0.81274] | 0.56077 [0.56035, 0.56157] | 1.446x | yes |
| 8192 | `mlp.up_proj` | 3072 | 0.80776 [0.80619, 0.80870] | 0.56120 [0.56059, 0.56157] | 1.439x | yes |
| 16384 | `mlp.down_proj` | 1024 | 1.23974 [1.23843, 1.24118] | 1.21920 [1.21664, 1.22045] | 1.017x | yes |
| 16384 | `mlp.gate_proj` | 3072 | 1.64947 [1.64736, 1.65266] | 1.11246 [1.11126, 1.11453] | 1.483x | yes |
| 16384 | `mlp.up_proj` | 3072 | 1.64155 [1.63933, 1.64354] | 1.11205 [1.11040, 1.11331] | 1.476x | yes |
| 32768 | `mlp.down_proj` | 1024 | 2.49026 [2.48910, 2.49470] | 2.39026 [2.38768, 2.39126] | 1.042x | yes |
| 32768 | `mlp.gate_proj` | 3072 | 3.33955 [3.33554, 3.34392] | 2.20408 [2.20174, 2.20547] | 1.515x | yes |
| 32768 | `mlp.up_proj` | 3072 | 3.34117 [3.33549, 3.34410] | 2.19670 [2.19366, 2.19819] | 1.521x | yes |
| 65536 | `mlp.down_proj` | 1024 | 5.02760 [5.02144, 5.03010] | 4.77006 [4.76554, 4.77997] | 1.054x | yes |
| 65536 | `mlp.gate_proj` | 3072 | 6.33445 [6.32602, 6.34886] | 4.38144 [4.37800, 4.38531] | 1.446x | yes |
| 65536 | `mlp.up_proj` | 3072 | 6.28016 [6.27392, 6.28563] | 4.37490 [4.37090, 4.38342] | 1.435x | yes |
| 131072 | `mlp.down_proj` | 1024 | 10.03298 [10.02726, 10.05248] | 9.57024 [9.56128, 9.59091] | 1.048x | yes |
| 131072 | `mlp.gate_proj` | 3072 | 12.63443 [12.62080, 12.66160] | 8.82438 [8.76859, 8.83734] | 1.432x | yes |
| 131072 | `mlp.up_proj` | 3072 | 12.57170 [12.55027, 12.59162] | 8.81530 [8.80560, 8.82442] | 1.426x | yes |

fp4 beats fp8 on `down_proj` at **M = 1024 (0.810x), 2048 (0.939x), 4096
(0.875x) and 8192 (0.940x)**, every one with disjoint 95% intervals, and loses
again from M = 16 384 once the epilogue is bandwidth-saturated. The wide units
move the other way throughout, from 1.07–1.08x at M = 512 to 1.43–1.52x.

**So the honest crossing statement is:** fp4 crosses fp8 on a real served unit,
in a real band of M, on this hardware — and the crossing is governed by output
width, not by token count alone. An allocator pricing the fp4 route by a single
number for all Linears is pricing the wrong thing.

## 8. The verdict

* **Does fp4 cross fp8?** On the three-unit operator sum #563 reports:
  **no, at no M up to 131 072**, with disjoint intervals at every M, and the gap
  *widens* with M (1.052x → 1.350x → 1.295x). On the narrow-output unit
  `mlp.down_proj` alone: **yes, for 1024 ≤ M ≤ 8192**, best 0.810x at M = 1024.
* **Was it tested where it would win?** **Yes.** 0.37–0.64 of a 140 W envelope
  in-process, 0.438 mean / 0.671 peak box-level, against #563's 0.058. This is
  not a "still under 30 W, hypothesis untested" outcome.
* **Is the expectation refuted?** **The kernel expectation is confirmed** — fp4
  x fp4 `_scaled_mm` is 1.39–1.66x faster than fp8 x fp8 at every M. **The route
  expectation is refuted**, and the cause is named: one full-output elementwise
  multiply by a Python float that the fp8 route does not perform.
* **Is fp4 the wrong choice?** Not on energy. fp4 delivers 1.15x–1.46x more
  work per joule than fp8 at every M measured, which on a 140 W box is a real
  axis. It is the wrong choice on prefill latency for wide-output units, as the
  route is written today.
* **Larger M?** Not the lever. The ratio has been flat to rising since
  M = 16 384 and the mechanism (a pass proportional to M x N, bandwidth-bound)
  does not improve with M. **No M would make the wide units cross.** The lever
  is the epilogue, not the token count.

## 9. Production runs, and the evidence

Every GPU action went through PrismaBuild at `--priority -10` on `--tag
sparklina`, each far under the 30-minute ceiling. Status read from
`pb-queue/{done,failed}/<key>.json` `detail.returncode`, never from a wrapper
message.

| Stage | PB key | Host | Outcome |
|---|---|---|---|
| smoke sweep (`sweep-s1`, M = 512, 8192) | `f44898093e4d` | sparklina | rc 0, 122.2 s, 18 points |
| load sweep (`sweep-L1`, M = 512 … 131 072) | `4001d2a67bad` | sparklina | rc 0, 495.1 s, 81 points |
| CPU tests `tests/test_prefill_load_sweep.py` | `acc5ae03a7cd` | dl380g10 | rc 0, 7 passed, 0 skipped |
| CPU tests (docs staleness + architecture + sweep) | `7b2e59a0e420` | dl380g10 | rc 0, 26 passed, 0 skipped |

The measurement ran inside the same container `full_engine_plugin_install.py`
builds for a #563 native receipt, launched by `$LS/stage/control/sweep_launch.py`
(adapted from `$NR/stage/control/g2_launch.py`), on the same image
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`,
the same GPU, the same seeded Triton cache from the #399 engine run, and the
same attested vLLM core manifest. `$LS/stage/control/pq_frontier_native_runner.py`
is #563's runner with one change — a `--module` argument whose default is the
Tessera harness it always ran — so the sweep module rides the same
`post-native-core.json` / `post-native-package.json` evidence path a receipt
does. The L1 run records `core_files_unchanged: 4956`,
`package_files_unchanged_from_installer`, and a Triton cache verification.

| Object | path |
|---|---|
| sweep record (schema `prismaquant.native_prefill_load_sweep.v1`) | `$LS/sweep-L1/sweep.json` |
| raw 100 ms power series | `$LS/sweep-L1/sweep-power-series.json` |
| chrome traces, 81 points | `$LS/sweep-L1/profile/` |
| host vitals (5 s nvidia-smi) | `$LS/sweep-L1.host/host-vitals.log` |
| smoke run | `$LS/sweep-s1/`, `$LS/sweep-s1.host/` |
| drivers (SHA256SUMS.txt) | `$LS/stage/control/` |
| frozen PQ tree the run measured | `$LS/stage/pq/` (`COMMIT`) |
| this document, as run | `$LS/stage/MEASUREMENT.md` |
| the renderer that produced every table here | `$LS/stage/analyze.py` |
| evidence digests | `$LS/EVIDENCE-SHA256SUMS.txt` |

| Object | sha256 |
|---|---|
| `sweep-L1/sweep.json` | `fb2e64e24bce20279f884712170d28bbccb642ac47a6acd011fee0620d0a6230` |
| `sweep-L1/sweep-power-series.json` | `bb24ebec65c2975d9da8e95b6137c5bb5d37670a91a54964a34fbfe90f14ab9f` |
| `sweep-s1/sweep.json` | `8e394efd45d41caa2871faf3bb4bab3d1dee65615600bcf186d360d94fcebcd0` |
| `sweep-L1/post-native-core.json` | `adcdee21163c459d4c52ce56134dd23eee4eba51f0c8e959b441fbd78cb86e9c` |
| `sweep-L1/post-native-package.json` | `7935ad05ab8dc7f33e9c10a1ef0567f8dad4dac651bd01f4d6a7fde7bf09a336` |
| `sweep-L1/triton-cache-verification.json` | `2f7d74b7a5f02bb51f1a87712ccb9327569121cdf212ef700dbdac97e0fd19a2` |

## 10. What this does not answer

* **Axis 2 (unit size) is untested.** No larger unit is encoded at both formats;
  see §2. §6–§7 show output width matters, on 0.6B units only.
* **The remedy is a hypothesis.** See §6. The Tessera tree is read-only here and
  nothing was changed in it.
* **A real corpus is not measured.** §2 says what tiling does and does not
  establish.
* **Decode was not swept.** The sustained loop and the M axis are a prefill
  instrument; #563 §8.4 holds the decode numbers.
* **The two structural gates still refuse these rows.** Nothing here changes
  debt D39: a `measured_runtime_prices.v2` table still cannot hold fp4 rows
  beside fp8 rows, and fp4 rows still bind to no full-engine bytes. This record
  is a price, not a table.
* **One box, one session, one GPU.** Every number came out of one container on
  sparklina on `GPU-b1eceeea-fec7-371e-2cf3-cd10f2e7b705`.

**Method note.** No `fable-<tier>` consultation was made for this record.
