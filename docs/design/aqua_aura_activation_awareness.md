# AQUA-AURA: activation-quantization awareness for the AURA cost

**Status: RESEARCH / design only.** No served evidence. Nothing here changes a
default. Every quantitative statement below is either (a) a static fact read off
the current code, cited inline, (b) a **measured** check marked ✅ with its
output pasted, or (c) a derivation marked as a prediction with the experiment
that would falsify it (§6). Per the promotion ladder this stays opt-in until it
has a served A/B.

**One experiment has been run.** T1 (§6) confirms the premise on the live
registry: NVFP4 and NVFP4A16 render weights **bit-identically** (max deviation
`0.0`) while differing by **9.42 % RMS activation error** (Gaussian input) — so
the allocator is provably indifferent between them today. The bit-identity is
distribution-independent and is the finding; the 9.42 % is illustrative and
distribution-dependent (§6). Everything else remains a prediction.

Author's note on provenance: written 2026-08-14 as the "spare time" item of the
overnight mandate. It answers Rob's framing directly — *"AURA is blind to
activation quantization, but it's possible we have the choice to quantize to a
W4A8 format or a W4A4 model… we need a way to balance that. I am guessing a
specifiable lever to pick between speed/quality. I would like to see if there's
some pareto frontier between the two to optimize against."*

---

## 0. The defect, established from code rather than asserted

AURA's cost is a KL-adjoint contracted with the production-rendered `dW`. It
prices **weight** perturbation and nothing else.

Read the format registry (`prismaquant/format_registry.py`):

| format | `weight_element_dtype` | `act_bits` | `activation_quantize_dequantize` |
|---|---|---|---|
| `NVFP4`    | `fp4_e2m1` | `4`    | `_make_rtn("fp4_e2m1", 16)` |
| `NVFP4A16` | `fp4_e2m1` | `None` | `lambda x: x` (identity) |

Identical weight element type, identical weight rendering, identical footprint.
They differ **only** in activation treatment.

**Consequence 1 — an expressive dead zone.** Since AURA's cost depends only on
`dW`, it assigns NVFP4 and NVFP4A16 *exactly* the same cost, and the footprint
model gives them the same bytes. The DP is therefore *exactly indifferent*
between W4A4 and W4A16. It cannot express a preference even in principle, so
NVFP4A16 can never be selected on merit — it is invisible, not merely
unattractive. This is the cleanest possible statement of the blindness.

**Consequence 2 — a directional bias on the *shipping* menu.** The default menu
is `NVFP4, FP8_DYNAMIC, BF16` — which in activation terms is W4**A4**, W8**A8**,
W16**A16**. Promoting a Linear NVFP4→FP8 improves weight precision 4→8 *and*
activation precision 4→8. AURA counts only the first. So AURA **systematically
under-counts the quality bought by promotion to FP8**, and the bias has a
predictable sign: the allocator should be under-promoting. This is a
falsifiable prediction, not a result — see T2 in §6.

The registry already exposes the predicate this needs
(`FormatSpec.act_bits is not None and act_bits < 16`), so the *plumbing* to
distinguish these formats exists; only the **cost** is blind.

---

## 1. The error decomposition

For a Linear computing `y = x Wᵀ`, with `δx = Q_a(x) − x` and `δW = Q_w(W) − W`:

```
Q_a(x) Q_w(W)ᵀ − x Wᵀ  =   x δWᵀ   +   δx Wᵀ   +   δx δWᵀ
                           └ E_w ┘     └ E_a ┘     └ E_wa ┘
```

- **E_w** — weight error through clean activations. *This is what AURA prices.*
- **E_a** — activation error through clean weights. **The blind spot.**
- **E_wa** — the cross term, second order.

Two structural facts make a cheap model plausible:

1. `E_a` does not involve `δW` at all. It is a property of the *activation
   distribution*, the *activation format*, and the **clean** weight.
2. `E_w` does not involve the activation format.

So to first order the two are **separable**, which is exactly what makes an
additive cost credible — and exactly what makes it testable (T3, §6). This
mirrors the existing AURA additivity gate rather than inventing a new discipline.

---

## 2. Why this forces an output-space sensitivity

AURA's currency is weight space. `E_a = δx Wᵀ` has no weight-space
representation — it is not a perturbation of `W` at all, for any choice of
`δW`. You therefore **cannot** fold activation error into the existing adjoint;
this is a currency mismatch, not a missing term. (Consistent with
[[aura_is_activation_quant_blind]]: *"'supersurrogate' is a CURRENCY claim, not
an error model"*, and with [[aqua_aura_activation_awareness]]: *"E_w+E_a needs
an OUTPUT-space sensitivity"*.)

Price everything in **output** space instead. Under the Gauss–Newton/Fisher
approximation to the KL objective, with `F_o` the per-output-channel diagonal
Fisher:

```
cost(ℓ, f)  =  E_x[ Δyᵀ H_y Δy ]  ≈  Σ_o F_o · E_x[ Δy_o² ]
```

This form is **format-agnostic**: hand it any `(weight-format, activation-format)`
pair and it returns a number in Δloss units, comparable across the whole menu.
That is precisely the property the mandate asks the artifact to have — *"the
artifact should be sufficiently descriptive so that an arbitrary collection of
formats can be assigned optimally by the optimizer."*

---

## 3. The SensitivityCard already carries everything required

Expand both terms under the diagonal approximation, writing `a_i = E[x_i²]`:

```
cost_w(ℓ,f) = Σ_o F_o Σ_i a_i · δW_oi(f)²          ← AURA today
cost_a(ℓ,f) = Σ_o F_o Σ_i e_i(f) · W_oi²           ← the new term
```

where `e_i(f) = E[δx_i²]` is the per-input-channel activation quantization MSE.

The inputs required are:

| symbol | meaning | source |
|---|---|---|
| `F_o` | per-output-channel Fisher | card **`g_sq_sum`** ✓ (**not** `fisher_row` — see below) |
| `a_i` | per-input-channel activation 2nd moment | card `act_sq_sum` ✓ |
| `W`   | clean weight | the checkpoint ✓ |
| `e_i(f)` | activation quant MSE for format `f` | §4 — needs `act_absmax` ✓ |

> **Correction, 2026-08-14 — this table originally named `fisher_row` as `F_o`,
> and that was a currency error of exactly the kind
> [[factorization-currency-dependent]] records.** Read the marginals off
> `incremental_probe.py::_marginal_chunk` and `sensitivity_card.py`:
>
> ```
> fisher_row[o] = Σ_i H[o,i] = Σ_t g[t,o]² · (Σ_i x[t,i]²)     # input energy FOLDED IN
> g_sq_sum[o]   =              Σ_t g[t,o]²                      # the OUTPUT-space Fisher
> ```
>
> `fisher_row` is output-*indexed* but not output-*space*: it already carries a
> `‖x_t‖²` factor. Both `cost_w` and `cost_a` above multiply `F_o` by a second,
> explicitly activation-derived factor (`a_i` and `e_i(f)` respectively), so
> substituting `fisher_row` would **double-count activation energy** — inflating
> exactly those units whose inputs are largest, which is the same failure
> signature as the ÷token-fraction bug removed in PR #14. Use `g_sq_sum`; it is
> already on the card, so the "no new probe pass" conclusion is unaffected.
> The card's own docstring states this distinction outright (`sensitivity_card.py`
> lines 31 and 43) — I had the field list right and the semantics wrong.
>
> **The error was confined to this table; the code is correct.** Do not "fix"
> `format_cost_protocol.py` to match the old prose. Checked line by line:
> `activation_dloss` (l.227) already uses `g_sq_sum` and carries an explicit
> NOTE — *"the sensitivity used here is `g_sq_sum` (output space), NOT `h_trace`
> (weight space). That distinction is the whole point."* And `weight_dloss`
> (l.196) uses `fisher_row` **legitimately**, because the W-side is not literally
> `Σ_o F_o Σ_i a_i δW²` in the implementation: it reconstructs the full Hessian
> rank-1 as `H ≈ outer(row, col) / h_trace_raw` and evaluates
> `(row @ δW² @ col) / h_trace_raw`, which needs *both* marginals and never forms
> `H`. So the two terms legitimately consume different fields — the W-side a
> rank-1 (row, col) pair, the A-side the output-space `g_sq_sum` — and §3's
> two-line `cost_w`/`cost_a` display is a simplification of the first, not a
> specification of it.

**This is the headline finding of the design pass: the contract built this
session already carries the exact state AQUA needs.** `act_sq_sum` and
`act_absmax` were specified into the AQUA tier, and they turn out to be
necessary and (for the analytic tiers) sufficient. No new probe pass, no new
capture, no re-instrumentation — AQUA is a *pricing* change over an artifact
that already exists. The card's tier name was better chosen than I knew when I
wrote it.

---

## 4. Modelling `e_i(f)` — three tiers that mirror the card's own tiers

**AQUA-0 (analytic, relative-error formats).** For a float format with `m`
mantissa bits the relative error is roughly uniform in the mantissa ULP, so
`e_i ≈ κ_f · a_i`. Then

```
cost_a(ℓ,f) = κ_f · [ Σ_o F_o Σ_i a_i W_oi² ]
```

The bracket is **one scalar per Linear**, precomputed once. *Any*
relative-error activation format is then priced by a single multiply. This is
essentially free and it is what makes a large menu tractable.

**AQUA-1 (block-scaled formats — NVFP4 group 16, MX group 32).** ✅ **IMPLEMENTED
2026-08-14** — `format_cost_protocol.block_scaled_act_quant_variance`, dispatched
by `analytic_act_quant_variance` on the descriptor's `act_group_size`.

Error is set by the **group's maximum**, not the element's magnitude:
`e_i ≈ (2·s_g / 2^b)² / 12`. The subtlety is *which* maximum, and this
document originally got it wrong: it specified `s_g` from **`act_absmax` over
the group**, which is a max over the whole calibration's **tokens**. That is the
right model for a *statically calibrated* activation grid and the wrong one for
what NVFP4/MX kernels actually do — they rescale **per token, per block**, so
the step follows the *local, per-token* block maximum, which is far below any
over-tokens maximum. Pricing a dynamic quantizer with a static step overstates
its error by **22.6×** on activations carrying massive-activation tokens
(measured against a simulated block quantizer), and the bias is directional: it
penalises exactly the W4A4 formats, which is the axis §5 exists to reason about.

So the shipped model computes `E_t[blockmax_t²]` from the **per-channel scales**
`σ_j² = act_sq_sum[j] / n_tokens`, exactly, via the tail-integral identity with
the folded-normal CDF:

```
E[M²] = ∫₀^∞ 2u (1 − Π_j erf(u / (σ_j √2))) du
```

This is exact for independent Gaussian channels **of differing scale** — the
per-channel heterogeneity is carried, not assumed away. Two cheaper routes were
tried and rejected *on measurement*:

| route | error vs 400k-sample simulation |
|---|---|
| `√(2 ln 2G)` asymptotic | **+52%** at G=16, +46% at G=32 |
| …with its Fisher–Tippett correction | **−14%** at G=16 |
| collapse each block to its RMS σ | **−47% to −71%** (1.9×–3.5× under-priced) |
| **exact tail integral (shipped)** | **≤0.1%** across all four regimes tested |

Against the *simulated quantizer* end-to-end (which adds the `step²/12` model's
own error on top of the order statistic) the shipped model lands at median
**1.04×** on clean Gaussian activations and **1.17×** with massive-activation
tokens, versus the old global-absmax model's **22.6×**.

Assumptions that remain, all stated in the docstring: Gaussian and independent
*within* a block; dynamic per-token scales (a statically-calibrated format
should decline to declare `act_group_size` and get the uniform model instead);
and the block scale treated as exact, ignoring NVFP4's FP8 snap and MX's pow2
snap as second order.

**AQUA-2 (measured).** Render `Q_a` on the calibration activations and measure
`e_i` directly. This is the fallback for formats the analytic model does not
cover, and the arbiter when AQUA-0 and AQUA-1 disagree — the same
measure-don't-model escape hatch the platform uses everywhere else.

> **Caveat stated up front, because it is the one most likely to bite.**
> Activation error is **outlier-dominated** in a way weight error is not: one
> channel with a large absmax inflates an entire group's scale. That asymmetry
> is the entire reason SmoothQuant/AWQ exist. So AQUA-0 will be *optimistic* on
> models with activation outliers, and **AQUA-1 is the honest default for
> block-scaled formats**. The per-channel ratio `act_absmax_i / sqrt(a_i)` is a
> direct outlier diagnostic and should be surfaced on the card rather than left
> implicit.

**Interface slot.** This lands on the existing `FormatCostPlugin` protocol as a
sibling of the weight cost — a plugin declares which AQUA tier it supports and
returns `cost_a` from `(unit, format, card)`. A plugin that declares no
activation model must **fail closed** for activation-quantized formats rather
than silently returning zero, or we reintroduce exactly the blindness this
document exists to remove.

---

## 5. The speed/quality lever, and why the frontier is real

The observation that makes Rob's intuition precise:

> **Activation format changes quality and speed but *not* bytes.**

W4A4 and W4A16 have identical weight footprints. So the activation choice is
**invisible to the existing byte-budget knapsack** — it is genuinely a *second
axis*, which is why it needs a new lever rather than a re-tuned budget. The
allocation problem becomes bi-constrained:

```
min Σ_ℓ cost(ℓ, f_ℓ)   s.t.   Σ bytes ≤ B   AND   Σ time ≤ T
```

a 2-D multi-choice knapsack (NP-hard). Dualize the *time* constraint with a
multiplier `λ ≥ 0`:

```
min Σ_ℓ [ cost(ℓ,f) + λ · time(ℓ,f) ]   s.t.   Σ bytes ≤ B
```

**This is the existing 1-D DP with a modified unary cost. No new solver, no
change to the knapsack, no change to union-find promotion.** That is the key
engineering result: AQUA is structurally free — it changes the *number* the DP
consumes, not the DP.

`λ` is the specifiable lever Rob guessed at, in units of **Δloss per unit
time**. `λ = 0` → pure quality (A16 wherever it is free); `λ → ∞` → pure speed
(A4 everywhere legal).

**And the house methodology extends cleanly, including its own graveyard.** The
rejected-methods table rejects Lagrangian λ-bisection *as a selector* — the
discrete frontier has non-convex pockets no λ selects — but explicitly keeps it
*as a candidate generator*. That is exactly its role here: **sweep λ to
generate candidates, then measure real served throughput and real held-out KL
on each, and take the empirical 2-D Pareto frontier under η-dominance.**
Surrogates generate, real measurement selects — now on two axes instead of one.
The lever does not need to be accurate; it needs to be *diverse*.

`time(ℓ,f)` must be **measured**, not modelled — a per-`(shape, format)` kernel
microbenchmark table. The entire premise is that A4 unlocks a different
tensor-core path whose speedup is shape- and kernel-dependent; modelling it
would repeat the mistake the platform already rejects (principle 1).

### 5a. Implementation status — the SELECTOR half landed 2026-08-14

The selection prototype described below was removed on 2026-10-14. Its
implementation remains in Git history. It implemented the *selection* half
of the proposal. The λ-sweep **candidate generator** is deliberately **not** implemented;
it is a modified unary cost fed to the existing DP, and it should be written
when there is a measured `time(ℓ,f)` table to sweep against.

What landed:

- `pareto_frontier` — non-dominated under (min Δloss, max speed).
- `select_by_speed_floor` / `select_by_quality_ceiling` — **the lever, as a
  constraint in the two directions an operator thinks in.** Never
  `α·quality + (1−α)·speed`. `test_frontier_keeps_a_point_no_weight_can_select`
  builds a non-convex pocket, sweeps 2001 weights, and shows **none** of them
  selects the middle point a speed floor reaches immediately — the same geometry
  that put λ-bisection in the graveyard as a selector. An unsatisfiable
  constraint returns `None`, never a quietly relaxed answer.
- `aggregate_speed_index` — a **params-weighted harmonic** mean (total work over
  total time), because throughput does not average arithmetically; an arithmetic
  mean lets a few fast units hide the slow one that sets the pace.

**Why AQUA-1 had to come first, with numbers.** Priced on the same unit with
identical weights, NVFP4 and NVFP4A16 come out at **identical `weight_dloss`
(0.182269) and identical bpp (4.500)** — the weights are 4-bit either way.
Without an A-side term they are *literally the same candidate* and the frontier
is one degenerate point. With AQUA-1, NVFP4 prices at **0.419871** vs
NVFP4A16's **0.182269**. That 2.3× is the quality the extra throughput costs,
and it is the entire frontier. Demonstrated end-to-end over
`{NVFP4, NVFP4A16, MXFP4, MXFP8_E4M3}`: 2 of 4 survive domination, and raising
the speed floor from 1.0 to 1.5 flips the selection from MXFP8_E4M3
(8.25 bpp, best quality) to NVFP4 (4.5 bpp, 2× declared speed).

**What this is NOT.** `speed_index` is a **declared per-format hint**, so the
aggregate is a *proxy*, not a serve measurement — blind to kernel dispatch,
bandwidth limits, batch shape, and to a mixed assignment falling off a fused
kernel entirely. This is the machinery; §5's "`time(ℓ,f)` must be measured"
requirement is **unchanged and unmet**. A measured W4A4-vs-W4A8 frontier needs
a served A/B and is not claimed. Two honesty properties are enforced in code
rather than documented: an assignment with any unit lacking a hint has
`speed_index=None` and is *excluded* from the frontier rather than defaulted,
and units that quantize activations but came back unpriced are counted and
printed by `frontier_table`, so a candidate flattered by a missing measurement
cannot pass as a win.

---

## 6. Falsification plan — ordered, cheap first, each one fatal

**T1 — Indifference check. ✅ EXECUTED 2026-08-14, premise CONFIRMED.**
Rendered both formats through the live registry on CUDA (256×512 bf16, seed 0,
**Gaussian synthetic input** — see the confidence split below):

```
weight renderings bit-identical : True
  max |dW_nvfp4 - dW_nvfp4a16|  : 0.0
activation handling identical   : False
  NVFP4    act rel-err (RMS)    : 9.4199e-02
  NVFP4A16 act rel-err (RMS)    : 0.0000e+00
```

The weight renderings are **bit-identical** (max deviation exactly `0.0`), so
AURA — which contracts the adjoint with `dW` alone — necessarily returns the
same cost for both. Meanwhile the two formats differ by **9.42 % RMS relative
error injected into the activations**. So the allocator is exactly indifferent
between a format that perturbs activations by ~9 % and one that does not
perturb them at all. That is the blindness, measured rather than argued.

**The two halves of this result carry different confidence, and they must be
quoted differently.** The **bit-identity is distribution-independent** — it
follows from the two `FormatSpec`s declaring the same
`weight_element_dtype="fp4_e2m1"` and the same group size, so no choice of
input can make the weight renderings differ. The blindness is therefore a
structural fact about the cost function, and that is the finding. The
**9.42 % is distribution-dependent** — it is one draw of Gaussian input, and
real activations are outlier-heavy in exactly the way that inflates block-scaled
quantization error, so the true figure on a calibrated model is plausibly
*worse*. Use 9.42 % to say "the gap is not negligible"; do not use it as the
magnitude of the A-side term, and re-measure on calibration activations before
it goes anywhere near a paper.

*(Still open in T1: confirming the same indifference end-to-end on a production
`cost.pkl`. The default menu does not contain `NVFP4A16`, so that requires a
menu-extended cost run — cheap, but not free.)*

**T2 — Underpricing direction (cheap, no serve).** Recompute an existing 4B
allocation with `cost_a` added; count FP8 promotions. *Predicted direction:
promotions increase.* Magnitude unknown — measuring it is the point.

**T3 — The additivity gate for E_w / E_a (the one that matters).** Measure true
end-KL for weight-quant-only, activation-quant-only, and both; check
`cost_wa ≈ cost_w + cost_a`. **In fp32** — per
[[cross_layer_additivity_fp32]], differencing in bf16 manufactures
non-additivity and has already fooled this project once. If the cross term
`E_wa` is material, the cheap separable model dies and AQUA needs the third
term.

**T4 — Does the frontier actually exist?** Sweep `λ` at a fixed byte budget on
0.6B and 4B, render, and measure *served* KL against *served* throughput. If
every `λ` lands on the same allocation, there is no tradeoff to optimize and the
lever is pointless — a real and publishable negative result. If a genuine
frontier appears, that is the deliverable.

**T5 — Only then** 27B and MoE, where routed experts add the route-flip
blindness already documented in [[aura_expert_routeflip_floor_confirmed]].

---

## 7. What this does *not* claim

- **No served evidence exists.** This is derivation plus one static registry
  fact. Research status, per the ladder.
- The diagonal-Fisher and per-channel-independence approximations are
  assumptions, not results. T3 tests the one that matters.
- `E_wa` is dropped at first order and is untested.
- "AURA underprices NVFP4 vs FP8" is a **directional prediction from
  structure**, not a measurement. T2 is what would turn it into a result. It
  must not be quoted as a finding before then.
- Nothing here changes a default, a menu, or a shipped artifact.
- **The analytic fallback no longer degrades silently for block-scaled formats
  — ✅ CLOSED 2026-08-14 by AQUA-1 (§4).** Recorded here because the shape of
  the defect is worth keeping: `price` preferred a plugin-supplied
  `activation_error_variance` (AQUA-2, measured) and fell back to
  `uniform_act_quant_variance`, which models a uniform grid over **per-channel
  absmax**. Every activation-quantizing format in the shipped registry is block
  scaled (NVFP4 `act_gs=16`, MXFP4/MXFP8 `act_gs=32`), so the fallback was wrong
  for **100% of the formats it could ever fire on**, and it produced a
  *plausible but wrong* number rather than a refusal — an asymmetry against the
  discipline one function over, where a card lacking `g_sq_sum` returns `None`,
  never `0.0`. It is now dispatched on the declared `act_group_size` and priced
  exactly. **Two things still deserve the None-discipline and have it:**
  `block_scaled_act_quant_variance` returns `None` when the card carries no
  `act_sq_sum` (it needs per-channel *scales*; `act_absmax` alone would smuggle
  the global maximum back in), and `CostComponents.quantizes_activations` is
  carried as data so a consumer can tell "this format leaves activations alone,
  `None` is correct" from "this format quantizes them and we failed to price it,
  `None` is a hole" — `speed_quality_frontier` counts and prints the latter.
  **Still research-tier**: the measured plugin variance remains preferred and
  takes precedence, and none of this changes a default or a shipped artifact.
- **The remaining modelling gap is the grid *shape*, not the grid *scale*.**
  AQUA-1 prices the step from the correct (per-token, per-block) maximum, but
  still applies the uniform-quantizer `step²/12` variance. NVFP4's E2M1 levels
  are exponentially, not uniformly, spaced within a block. Empirically this is
  the residual: `E[blockmax²]` matches simulation to ≤0.1% while the end-to-end
  variance sits at 1.04–1.17×, so the `step²/12` model is the whole discrepancy.
  It is small, and it is *conservative* (over-prices), which is the safe
  direction — but it is the next thing to fix if AQUA-1 is ever promoted past
  research.

## 7b. Relationship to `activation_fair_pricing.py` (the existing partial answer)

This is **not** a greenfield gap — `prismaquant/activation_fair_pricing.py`
already exists and already concedes the problem in its own docstring: on the
weight-only branch the "activation-contract difference is *structurally
invisible*". It prices a whole **family** with one multiplicative penalty

```
penalty(family) = exp( mean_i ln( d_measured_i / d_weight_only_i ) )
```

and its own text calls this an **"estimator-transfer calibration, not an
isolated A-side term."** It carries a documented bias that matters here: the
**A-side error is rung-independent while the W-side shrinks with `k`**, so the
true ratio *grows* along a CB ladder and a single scalar per family cannot
track it.

AQUA's relationship to it is therefore precise: `activation_fair_pricing` is a
*family-level correction factor applied to `E_w`*; AQUA is an *additive,
per-unit, per-format `E_a` term*. The second subsumes the first if T3 holds —
but **superseding it is a promotion decision, not a refactor**. Leave that
module alone until AQUA has served evidence, and do not run both corrections at
once or the A-side gets counted twice.

## 8. Relationship to the rest of the stack

- Consumes the **SensitivityCard** AQUA tier (`act_sq_sum`, `act_absmax`) — no
  new probe pass required.
- Extends **`FormatCostPlugin`** with an activation-cost sibling that must fail
  closed rather than default to zero.
- Requires **no change** to `allocator_solver.py`: λ enters as a modified unary
  cost.
- Would make `NVFP4A16` (today research/"rarely chosen") a *meaningful* rung for
  the first time, and is the prerequisite for any W4A8-class rung.
