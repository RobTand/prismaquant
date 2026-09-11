# W2: probe reduction for the GLM-5.3-Flash Tessera anchor campaign

Worker: W2 (Fable 5.1). Worktree `/home/rob/tmp/pq-glm-probe-reduction`, branch
`study/glm-probe-reduction` from 9753a5b7c5. Date 2026-09-10/11.

Study script (committed on the branch, not pushed):
`/home/rob/tmp/pq-glm-probe-reduction/experiments/glm_probe_reduction_regret.py`
(commits `bb8dcb19b8`, `9b2cd23d63`).

## 1. Outcome in one paragraph

A routed MoE layer's expert stack is ONE rate decision (all 288 experts x
gate/up/down = 864 Linears at one q256), so the census's 2,592 encodes per
layer serve a single three-way choice. The lever is therefore sampling
experts per stack, not a cross-family transfer law; the code already has the
sampling path (`--stack-sample`, PPS draw + Horvitz-Thompson) and this
campaign did not use it (`plan.json` `stack_sample.size: null`). Scored
offline against the 28 completed census stacks with an exact multi-choice
knapsack, the schedule **"960 census + a 3% dual-rate (832/1088) expert
sample with a per-stack intercept transfer law"** reproduces the omniscient
allocation at the 4.0709 bpp target with mean regret 0.004% (uniform
weights) / 0.001% (routed-count weights), p90 0.000% / 0.001%, max 0.074% /
0.012% over 50 draws, and cuts routed-expert anchor-encode GPU time by 66%
(44.4 of 66.9 GPU-h for all 42 stacks) before the selective encode of the
winners (net ~45%, section 5). Gate caveat: b_law@3 passes the proposed
p90 <= 0.1% regret gate at +0% and +2.5% but fails it at -2.5% (p90 0.168)
and -5% (0.117) under uniform weights; s=10% meets it at every
non-saturated budget under all three weightings for 3.2 GPU-h more encode
(section 3.6). The cheapest zero-sampling alternative,
endpoints 832/1088 plus a pooled curvature prior for 960 (`c2`), has 0.000%
regret at the target under both weightings and saves 32% (21.1 GPU-h). All
of this is scalar route-MSE regret on the census currency; it is not a KL,
joint-AURA, or served-quality claim (section 8).

## 2. Step 1: the decision unit, from code

Question: is a routed MoE stack one rate decision or per-expert, and does the
Tessera wire allow per-expert rates in a packed stack?

Answer: **one decision per packed stack; per-member rates are licensed only
inside vLLM fused dense modules (q/k/v, gate/up), never inside a packed expert
stack.**

Evidence (worktree paths, commit 9753a5b7c5):

- `prismaquant/allocator_candidates.py:3622` `aggregate_packed_serving_groups`
  turns a packed group into ONE multi-choice DP item whose cost is the sum of
  member `predicted_dloss` and whose formats are the ones legal for every
  member. `prismaquant/model_profiles/specs/glm5_next.json:109`
  `packed_experts.format_groups` includes `["gate_proj","up_proj","down_proj"]`,
  so the group is all three projections of all experts in a layer (the census
  `s:` groups have 864 members; 42 such groups).
- `prismaquant/allocator_solver.py:735-770` `_promote_group_components`: "a
  packed or untagged component takes the uniform path even when its max-rank
  format is a Tessera rung"; the per-member relaxation
  (`_resolve_family_group`, `:672`, `fused_licence.is_per_member("q256")` at
  `:710`) is entered only for components made solely of fused groups. A
  fused+packed mix refuses (issue #140).
- `prismaquant/tessera_runtime_contract.py:263-267`: the pinned contract's
  `fused_module` block licenses `q256: per_member`, `rows: per_member`,
  `mixed_rung_receipt: False`. The two `routed_moe` cells (`:512`, `:590`)
  publish no per-expert rate licence.
- `prismaquant/tessera_formats.py:1678` `format_promotion_class` returns the
  family for Tessera rungs; the code warns this is not a claim that the rate
  is free.
- `prismaquant/tessera_campaign.py:719-760`: "A routed MoE stack is ONE
  decision: vLLM loads a packed [E, M, N] expert tensor under a single
  quantization scheme"; the stack row is the h-weighted mean expert MSE so
  that `0.5 * h_stack * output_mse` reproduces the summed member dloss
  (`_stack_cost_rows`, `:1069`, row `cost_source:
  tessera_campaign_measured_stack_sample` at `:1147`).
- `prismaquant/tessera_menu.py:1629-1648`: `ANCHOR_FRESH_ENCODE`; every
  anchor is a distinct encode, no truncated or nested wire exists (matches
  Astra's row-0081 audit: 2,592 distinct wire cells).

Consequence: with one decision per stack the census spends 3 x 864 encodes to
resolve a choice that a stack-level estimate of three numbers resolves. The
existing lever is `tools/dispatch_tessera_campaign.py:484`
`sample_stack_groups` (`--stack-sample`, `:1460`; `--stack-sample-seed`,
`:1464`; `--audit-rate`, `:1467`; `--probe`, `:1470`) calling
`tessera_campaign.draw_stack_sample` (`:2699`, randomized systematic PPS
without replacement with a take-all stratum, sizes = per-expert Fisher
`h_trace_per_expert`) and `_horvitz_thompson_stack` (`:961`). It requires a
probe with `h_trace_per_expert`; no GLM probe exists (`plan.json`
`stack_sample.probe: null`), which is why the census was run instead.

The one-anchor transfer law (Gridbook precedent) is still useful, but at the
stack level: it predicts the two missing rates of a stack from its measured
960 census plus a small dual-rate sample (schedule `b_law` below).

## 3. Step 2: offline regret study

### 3.1 Setup

- Inputs: 118 completed `cost.pkl` payloads under
  `BASE/first-proof-anchor-preparation-05/workspace/rows/` (measured cells
  only, `output_mse_measured == True`, `cost_source ==
  tessera_campaign_measured`) plus `workspace/census.json`. Input-list SHA-256
  `e9ba432446d1c8e9bbf8488795791cb078b54a7ed29be64444ad77ec1911472d` (per-file
  hashes in the result JSON). Nothing under the workspace was modified.
- Population: 28 expert stacks (layers 7, 10-14, 16-34, 36, 38, 44; 288
  experts x 3 projections x 3 rates each), 135 dense units (45 fused gate/up
  pairs + 45 down; E4M3 x {832, 960, 1088}, BF16 x {832, 960, 1088}, E2M1_K2
  @896). Dense units are a census in every arm; only the stack predictions
  differ between schedules.
- Currency: `output_mse_under_route_activation_contract`. Bytes:
  `wire_bytes` (cost rows carry no `memory_bytes`).
- Objective. The production DP prices `0.5 * h_trace * output_mse`
  (`allocator_solver.py:434`). No GLM Fisher probe exists, so three weightings
  are reported: `uniform` (unweighted sum of `output_mse`, the convention of
  `tessera_rate_surface.allocation_regret`), `counts` (per-expert routed-token
  fraction `counts/262144` from `census.json`, a stated proxy for `h_trace`,
  never labeled as it), and `lognormal_h` (a hypothesis sensitivity: h ~
  lognormal, CV 1.5, rank-correlated 0.5 with log counts, 3 draws; calibrated
  to the LFM2.5 probe's per-expert h CV 1.0-1.75). Routed-count CV per stack:
  median 1.12.
- Allocator: exact multi-choice knapsack DP (numpy, 256 KiB byte
  resolution, bytes rounded up so every DP-feasible choice is byte-feasible),
  identical for the predicted and omniscient arms. Fused gate/up pairs get a
  joint family x rate_gate x rate_up menu (per-member q256 inside one family,
  as the contract licenses).
- Budget: 155,668,854,528 B over 305,915,756,544 params = 4.070895 bpp,
  applied to the measured subset (204,447,154,176 params) = 104,035,354,901 B,
  at -10 / -5 / -2.5 / 0 / +2.5 / +5 / +10 %. Menu span for the subset:
  min 83.61 GB (all 832), max 109.16 GB (all 1088). **+5% (109.24 GB) and
  +10% (114.44 GB) exceed the menu maximum: both arms select all-1088 and
  regret is 0 by construction. Those two columns are degenerate, not
  evidence.** -2.5% and +2.5% were added so the table has informative points
  on both sides of the target.
- Regret = (truth objective of the allocation decided on predicted stack
  costs - omniscient truth objective) / omniscient, in percent. Sampling arms
  are repeated over 50 seeds (20 per lognormal draw) and summarized as mean /
  p90 / max; `ag` is the fraction of the 28 stack decisions that agree with
  the omniscient allocation. Hold-out is by layer: every pooled quantity
  (slope, curvature prior) is fitted on the other 27 stacks.
- Sampling reuses the campaign's own code: `tessera_campaign.draw_stack_sample`
  (sizes = the arm's weights; the study cannot draw PPS on the true h) and
  `_horvitz_thompson_stack` (which refuses m < 2 random draws). Confirmed by
  `sampling_source: prismaquant.tessera_campaign` in the PB result.

### 3.2 Schedules

| tag | schedule | stack prediction |
|---|---|---|
| a | three anchors, census (today) | truth |
| c | endpoints 832/1088 census | 960 by the per-unit log-linear chord |
| c2 | endpoints + other-layer curvature prior | chord + median per-projection (log2 mse(960) - chord) from the other 27 layers |
| b_ratio@s | 960 census + s% dual-rate sample | HT ratio: T(960) x HT(r)/HT(960) |
| b_law@s | same sample | per-(stack, projection) intercept, slope pooled from other layers: log2 mse_e(r) = a + b_r log2 mse_e(960), summed over all 288 experts |
| b_diff@s | same sample | b_law + HT-weighted residual correction (model-assisted difference estimator) |
| d@k | k experts per stack at all three rates | HT total (the code's `--stack-sample` path) |

s in {1, 3, 5, 10}% -> n = {3, 9, 14, 29} experts; k in {8, 32, 96, 216}.

### 3.3 Regret at the byte target and neighbours (percent; mean / p90 / max over draws; ag = stack decision agreement)

Uniform weights:

| schedule | -10% | -5% | -2.5% | **+0%** | +2.5% | +5% (sat.) |
|---|---|---|---|---|---|---|
| a | 0 | 0 | 0 | 0 | 0 | 0 |
| c | 0 | 0 | 0.102 (ag .93) | **0** | 0.003 (ag .93) | 0 |
| c2 | 0 | 0 | 0 | **0** | 0.003 (ag .93) | 0 |
| b_ratio@1 | .042/.026/.771 | .138/.241/1.42 | .055/.180/.309 | **.047/.156/.377** ag .97 | .058/.127/.365 | 0 |
| b_law@1 | .030/0/.405 | .140/.329/.604 | .122/.309/.645 | **.050/.156/.521** ag .97 | .047/.126/.276 | 0 |
| b_diff@1 | .100/.032/3.17 | .186/.218/4.79 | .179/.169/5.88 | **.172/.156/7.00** ag .98 | .196/.105/7.94 | 0 |
| b_ratio@3 | 0 | .035/.074/.129 | .013/0/.326 | **.037/.057/.694** ag .99 | .023/.057/.107 | 0 |
| b_law@3 | 0 | .046/.117/.277 | .050/.168/.265 | **.004/0/.074** ag .996 | .018/.074/.167 | 0 |
| b_diff@3 | .062/0/3.08 | .118/.056/4.80 | .013/0/.326 | **.030/0/.694** ag .99 | .046/.033/.955 | 0 |
| b_law@5 | 0 | .028/.071/.128 | .029/.168/.265 | **.009/.074/.074** ag .99 | .005/.004/.074 | 0 |
| b_law@10 | 0 | .022/.071/.128 | .013/0/.168 | **0** | .003/.003/.033 | 0 |
| d@8 | .625/1.19/1.65 | .519/.967/1.45 | .609/.957/1.68 | **.448/.785/1.33** ag .87 | .298/.722/1.76 | 0 |
| d@32 | .039/.236/.351 | .173/.379/.778 | .070/.180/.333 | **.090/.211/.277** ag .94 | .079/.181/.383 | 0 |
| d@96 | 0 | .064/.131/.213 | .006/0/.102 | **.017/.074/.156** ag .98 | .025/.064/.126 | 0 |
| d@216 | 0 | .012/.039/.071 | 0 | **.001/0/.055** ag .999 | .008/.033/.059 | 0 |

Routed-count weights (`counts`):

| schedule | -10% | -5% | -2.5% | **+0%** | +2.5% | +5% (sat.) |
|---|---|---|---|---|---|---|
| c | .042 (ag .93) | .030 (ag .93) | 0 | **0** | 0 | 0 |
| c2 | 0 | .030 (ag .93) | 0 | **0** | 0 | 0 |
| b_ratio@1 | .008/.042/.042 | .013/.048/.155 | .010/.038/.109 | **.018/.058/.313** ag .97 | .013/.049/.095 | 0 |
| b_law@1 | 0 | .006/0/.155 | 0 | **.006/.012/.102** ag .99 | .005/0/.128 | 0 |
| b_diff@1 | .002/0/.042 | .002/0/.059 | 0 | **.001/0/.012** ag .99 | .003/0/.049 | 0 |
| b_law@3 | 0 | .001/0/.030 | 0 | **.001/.001/.012** ag .99 | 0 | 0 |
| b_diff@3 | .181/0/3.02 | 0 | 0 | **.000/0/.012** ag .999 | 0 | 0 |
| b_law@5 | 0 | .001/0/.030 | 0 | **.001/0/.012** ag .997 | 0 | 0 |
| b_law@10 | 0 | 0 | 0 | **.000/0/.012** ag .999 | 0 | 0 |
| d@8 | .086/.293/.506 | .133/.338/.506 | .099/.250/.416 | **.135/.314/.713** ag .90 | .138/.327/.419 | 0 |
| d@32 | .012/.042/.098 | .050/.121/.227 | .026/.042/.183 | **.031/.088/.266** ag .95 | .036/.070/.145 | 0 |
| d@96 | .003/0/.042 | .003/.003/.030 | .009/.038/.042 | **.003/.012/.012** ag .98 | .001/0/.049 | 0 |
| d@216 | 0 | 0 | 0 | **0** | 0 | 0 |

Lognormal-h sensitivity (hypothesis, 3 h-draws x 20 seeds): the ranking is
unchanged (b_law@3 max 0.000% at +0%, d@8 max 0.38%, c max 0.055%, c2 0);
b_diff@1 shows the same heavy tail as under uniform (max 11.3%). Full table
in the result JSON (`summary.lognormal_h`) and in
`/home/rob/tmp/glm-perf-20260910/w2/pbrun-regret-02.log`.

Per-layer: the disagreements concentrate on the stacks whose adjacent
marginal prices straddle lambda* (layers 12, 24, 25 under uniform; 13, 16
under counts; see `omniscient.*.marginal_price_diag`, which reports
log10(marginal price / lambda*) per stack and rate step). Layers 26-44 sit
0.3-0.7 decades above lambda* at the target; at +0% their stack agreement is
1.0 under every schedule except d@8 (layers 26, 27: 0.96, 0.98) and
b_diff@1 (layer 44: 0.98) under uniform weights, and 1.0 for all schedules
under counts (`summary.*.per_layer[*]["agree_+0%"]`). Pointwise stack-cost
error (max |log2 pred/true| over draws, per held-out layer): c 0.03-0.04,
b_law@1 0.10-0.21, b_law@3 0.02-0.14 (uniform; mean |log2| 0.012) and
0.01-0.05 (counts; mean 0.002), b_law@10 <= 0.08 / <= 0.03, d@8 1.65-1.80,
d@32 0.56-0.69, d@96 0.17-0.29. That is the "few chunky decisions"
structure: the grid hides pointwise error unless a stack's marginal price
lies within its error of lambda*.

Budget sweep (8 seeds, 21 points from 3.27 to 4.27 bpp; `sweep` in the JSON):
c has isolated regret spikes up to 0.10% (uniform) / 0.16% (counts) where the
chord's 960 error crosses lambda*; c2 stays at 0 except 0.003% (uniform) /
0.03% (counts) at two points; b_law@1 reaches mean 0.17% / p90 0.40%
(uniform) between 3.9 and 4.0 bpp; d@8 runs at 0.3-0.7% mean throughout;
d@96 below 0.09% mean.

### 3.4 What the numbers say

1. The three-way stack decision is insensitive to everything except the
   stacks near lambda*. Even the plain chord (`c`) reproduces the omniscient
   allocation at the target under both weightings; its risk is a 0.03-0.04
   log2 pointwise error crossing a marginal price at some other budget
   (spikes in the sweep). Adding the pooled curvature prior (`c2`) removes
   those spikes on this data.
2. Among sampled schedules, the per-stack intercept law (`b_law`) beats the
   HT ratio at every s under both weightings, and beats the difference
   estimator (`b_diff`) on the tail: with only 3 random draws the HT residual
   correction scales one expert's residual by ~96 and produces 3-8% regret
   outliers (max 7.0% uniform, 11.3% lognormal). Do not use `b_diff` at s <=
   3%.
3. Pooled slopes are stable across layers (sd over 28 per-layer fits 0.02-0.03
   for both 832 and 1088, all projections), so the layer hold-out costs
   nothing; the intercept from 3-9 experts is the whole error.
4. Pure HT stack sampling (`d`) needs k >= 96 experts per stack (33%) to
   match `b_law@3` under uniform weights, and is worse on the tail at every k
   under uniform; under counts weights d@96 and b_law@3 are comparable. Its
   attraction is that it is the code's existing path.
5. The weighting changes magnitudes (counts weights make every sampled
   schedule look better because the routed-count PPS draw then matches the
   objective) but not the ranking `b_law@3 <= d@96 < b_ratio@1 ~ b_law@1 <
   d@32 < d@8`, nor the verdict on `c2`.

### 3.5 Encode accounting

Measured `encode_seconds`; expert cells use `batch_wall_time_divided_by_batch_size`
(batch 8). Measured subset (28 stacks): 832: 24,192 cells / 52,621 s; 960:
24,192 / 50,589 s; 1088: 24,192 / 57,357 s; adaptive extras 5,184 cells /
10,891 s; dense (all families) 952 cells / 24,640 s.

| schedule | expert cells (28 stacks) | GPU-h (28) | GPU-h (42 stacks, extrapolated) | saving vs 3-anchor (42) |
|---|---|---|---|---|
| a, today as actually run (with adaptive extras) | 77,760 | 47.6 | 71.4 | -4.5 |
| a, three anchors | 72,576 | 44.6 | 66.9 | 0 |
| c / c2, endpoints | 48,384 | 30.6 | 45.8 | 21.1 (32%) |
| b @1% (n=3) | 24,696 | 14.4 | 21.6 | 45.4 (68%) |
| **b @3% (n=9)** | 25,704 | 15.0 | 22.5 | **44.4 (66%)** |
| b @5% (n=14) | 26,544 | 15.5 | 23.3 | 43.6 (65%) |
| b @10% (n=29) | 29,064 | 17.1 | 25.7 | 41.2 (62%) |
| d k=8 | 2,016 | 1.2 | 1.9 | 65.0 (97%) |
| d k=32 | 8,064 | 5.0 | 7.4 | 59.5 (89%) |
| d k=96 | 24,192 | 14.9 | 22.3 | 44.6 (67%) |
| d k=216 | 54,432 | 33.5 | 50.2 | 16.7 (25%) |

Encode time is the anchor-encode wall time only; capture/Hessian/scoring and
row overheads are not included, and the 42-stack figures extrapolate the
per-stack average (stack shapes are identical across layers).

### 3.6 Recommendation

- Routed stacks: **schedule b_law@3%** (960 census + 9 experts per stack
  encoded at 832 and 1088, per-(stack, projection) intercept with pooled
  slopes) gated by **p90 regret over >= 50 draws at the run's actual
  budget <= 0.1%**, with the +-2.5% points reported as a diagnostic curve,
  not a gate. The 0.1% threshold is the policy constant carried over from
  the Gridbook one-anchor regret gate (`oneanchor_alloc`, ~0.1% regret), not
  a value derived from this objective; it is a choice to record, not a
  result. On this data it is 0.004% mean / 0.000% p90 / 0.074% max
  (uniform) and 0.001 / 0.001 / 0.012% (counts), saving 66% of routed-expert
  anchor-encode time. Caveat, from the same tables: b_law@3 passes the
  p90 <= 0.1% gate at +0% and +2.5% but fails it at -2.5% (p90 0.168) and
  -5% (p90 0.117) under uniform weights, and the subset's "+0%" column does
  not pin where the full 42-stack campaign sits relative to lambda*. The
  smallest s that meets the gate at every non-saturated budget on this data
  under all three weightings is **s=10%** (uniform p90 <= 0.071, counts
  <= 0.012, lognormal <= 0.004; max 0.168), at 3.2 GPU-h more encode (41.2
  vs 44.4 GPU-h saved). So: s=3% if the gate is applied at the run's actual
  budget and passes there; s=10% if the gate must hold across the +-5%
  range. If the sampling machinery is not wanted in the next
  campaign, **c2** (endpoints + pooled curvature prior) is 0 regret at the
  target under both weightings for a 32% saving and needs no sample and no
  probe.
- s=1% is not recommended: n=3 gives max 0.5% (uniform) and the sample can
  hit `draw_stack_sample`'s `remaining == 1` refusal when the take-all
  stratum absorbs two experts (the study bumps n to 4 in that case; the PB
  run shows `n_used == 3` and `m_sampled == 3` on every stack, seed and
  arm at s=1% (`raw[*].sample_info`), so it did not fire here, but it will
  with a more dispersed h).
- Dense units: keep the three anchors (3 x 135 x 2 families = 810 cells,
  6.8 GPU-h in this campaign); the saving is small and the dense allocation is
  where families actually compete (section 4).
- Do not use d@8 or d@32 for the first-proof: 0.45% / 0.09% mean regret with
  13% / 6% of stack decisions wrong (uniform).
- Sizes for the PPS draw: with no probe, `counts` is the only per-expert
  size available. It is a routed-token proxy, not `h_trace`; the planner
  today refuses to run without Fisher weights (`sample_stack_groups`
  requires `h_trace_per_expert`), so an explicit `--stack-sample-sizes
  counts` option (or a probe with `h_trace_per_expert`) is a prerequisite.

## 4. Step 3: dense cross-family screen

Leave-one-LAYER-out (45 layers, 135 units): predict log2 BF16 mse at each of
832/960/1088 from log2 E4M3 mse at the same rate + projection indicator +
log2 shape + layer index + rate indicator, OLS on the other 44 layers.

- Pointwise |log2 error|: mean 0.039, median 0.019, p90 0.090, max 0.495.
- Regret on the dense-only allocation (fused pairs + down units, families
  E4M3/BF16/E2M1@896, dense budget scaled from 4.0709 bpp): 0.000% at
  -5 ... +10% under both weightings; 0.0014% at -10% (agreement 0.978). The
  allocation picks BF16 for all 90 items at every budget, so the family
  choice is never close and the regret is trivially small. E2M1_K2 has one
  anchor (896) and contributes no surface. This is a screen on 135 units in
  one narrow band; it says the BF16 curve is predictable from E4M3 to ~0.04
  log2 typical / 0.5 max, and that the dense allocation on this menu does not
  exercise the prediction. It does not license dropping BF16 anchors.

## 5. Step 4: integration sketch (no code change made)

Anchor schedule per unit class:

- Dense (`g:` fused pairs, `d:` down units): unchanged, three anchors per
  family (`round_one_rates` in `tessera_campaign.py` ~5090-5130; rate band
  ends + midpoint).
- Routed stacks (`s:` groups): all experts at 960 (one fresh encode per unit,
  `_anchor_batches` `:698` batching unchanged), plus a sample of n=9 experts
  per stack at 832 and 1088. Round 1 becomes per-class: `round_one_rates`
  returns `[960]` for members of `s:` groups outside the sample and
  `[832, 960, 1088]` for sampled members. The adaptive rounds (`anchor_budget`,
  LOO gate `:5140-5160`) apply to sampled members only.

Sample selection: `tools/dispatch_tessera_campaign.py:484`
`sample_stack_groups` already draws per packed parameter and persists
`inclusion_probability`, `audit_experts`, and the draw; extend it with a size
source (`--stack-sample-sizes {probe,counts}`) so it can run without a probe,
writing `design: "pps_wor_counts"` and the size SHA into `units.json`
(`UNITS_SCHEMA_V2`). The plan's `stack_sample` block (`plan.json`) already
carries `size`, `seed`, `audit_rate`, `probe`.

Prediction feeding the allocator: a new `tessera_rate_surface.py` helper
`fit_stack_transfer_law(stacks_measured, hold_out)` fits pooled slopes b_r per
projection over all other stacks and per-stack intercepts from the sample,
and `predict_stack_rates(stack)` returns the two predicted stack costs plus a
model-error field; `_stack_cost_rows` (`tessera_campaign.py:1069`) writes the
960 row as `tessera_campaign_measured` (a census) and the 832/1088 rows as
`PROVENANCE_INTERPOLATED` with `cost_source: tessera_campaign_interpolated`
(the constant at `allocator_candidates.py:1153`) and a `transfer_law` block
(slopes, intercepts, sample ids, n, residual sd) instead of the HT
`dloss_stderr` / `estimator: horvitz_thompson` fields, which must not be
stamped on a model-predicted row. `TesseraRateSurface` (`:160-250`) stays as
is for dense units.

Selective encode of allocated winners: `tessera_joint_aura.py:81-174`
`load_measured_anchor_input` requires every allocated cell to be an exact
measured wire (`cost_source == tessera_campaign_measured`, `provenance ==
measured`, receipt coverage equal to measured payload coverage). So after the
scalar allocation picks a rate per stack, every stack whose winner is 832 or
1088 needs its remaining experts encoded at that one rate (one fresh encode
per unit, same `_measure_anchor_batch` path) before joint AURA runs. Expected
volume at the target (`omniscient.*.stack_rates` at +0%): 17 of 28 stacks
choose 1088, 11 (uniform) / 10 (counts) choose 960, 0 / 1 choose 832. Under
b_law@3 the 960 winners are already fully measured (census) and each
832/1088 winner needs its remaining 279 experts x 3 projections = 837 cells
at one rate: 17 x 837 x 2.37 s (1088; ledger 57,357 s / 24,192 cells) =
33,700 s, plus 0-1 x 837 x 2.18 s (832) = 0-1,800 s: 33,700-35,500 s =
9.4-9.9 GPU-h on the 28 stacks, ~14-15 GPU-h extrapolated to 42. Net saving
for b_law@3 including the selective pass: 44.4 - ~14.5 = ~30 GPU-h of 66.9
(45%). Under c2 the endpoints are
already measured and only the 960 winners need a pass: 11 x 864 x 2.09 s
(960; 50,589 s / 24,192) = ~19,900 s = 5.5 GPU-h on 28, ~8.3 on 42; net
21.1 - 8.3 = ~13 GPU-h (19%). State these in the plan rather than the 66%
sampling-only figure.

Repair loop (Gridbook `oneanchor_alloc.repair` pattern): after the selective
encode, replace predicted stack rows with the now-measured ones, re-solve the
scalar DP on truth-where-known, and encode any newly promoted stack; iterate
until the allocation is fixed (order-independent; in the 27B rehearsal one
pass flipped 0.8%). Here at most 28 items flip, so the loop is one or two
short passes.

Regret gate: before the selective encode, run the study's regret estimate on
the campaign's own data (bootstrap the intercept sample within each stack,
>= 50 draws, at the run's byte budget): `--max-regret-pct 0.1` on p90; if
it fails, fall back to the full three-anchor schedule for the failing stacks
(the driver already knows how to add rates in later rounds). Report the
budget sensitivity curve, not gate on it.

Files that change:
- `prismaquant/tessera_campaign.py`: `round_one_rates`/round-1 planning
  (~5090-5130) per unit class; `_stack_cost_rows` (1069-1245) to write
  transfer-law rows; a selective-encode round after allocation.
- `prismaquant/tessera_rate_surface.py`: stack transfer law + regret gate
  (`allocation_regret` at 465 is greedy and unweighted; the gate needs the
  exact DP and the packed-group item, so add a `stack_allocation_regret`).
- `tools/dispatch_tessera_campaign.py`: `--stack-sample-sizes`, per-class
  anchor schedule in the plan, and a `selective-encode` subcommand that reads
  the allocation and emits rows for the missing cells.
- `prismaquant/tessera_joint_aura.py`: unchanged if the selective encode
  produces exact measured cells; it must keep refusing interpolated rows.

Checkpoint / identity implications: the campaign identity binds the anchor
set (`anchor_placement`, `anchors_round_one`, `anchor_budget`,
`unit_selection_sample` in `provenance`), so a sampled schedule is a new
campaign identity, not a resume of the census; seed-checkpoint adoption
(`_adopt_seed_checkpoint`, `:2028`) is unaffected. Rows encoded in the
selective pass are new `cost.pkl` rows with their own receipts; `wire_dir`
gets only the winners' wires, which is the point. Any PQ source change forces
the seed rebind on all rows (handover); this design is for the next campaign,
not for the 16 remaining rows.

## 6. PrismaBuild actions, receipts, hashes

- `8e830f456333d39bb815bc889a917ab95ddf945a351b7ffc79d23aa187342abd`:
  first submission, dl380g10, x86, 16 CPU / 24 GiB, priority -10. **Failed:
  timeout at 3579 s** (re-solving the 90-item dense menu inside every
  evaluation). Record
  `/mnt/shared/prismabuild-fleet/pb-queue/failed/8e830f456333d39bb815bc889a917ab95ddf945a351b7ffc79d23aa187342abd.json`.
  No result was produced or used; its output dir was removed.
- `c56ff05dcbe38d6cf66a888c1a0e64027a3da640977f8ad34fbd3624d129e729`:
  resubmission after folding the dense items into one DP prefix (commit
  `9b2cd23d63`). Executed on dl380g10 in 1390 s, rss 1.6 GB, exit 0. Record
  `/mnt/shared/prismabuild-fleet/pb-queue/done/c56ff05dcbe38d6cf66a888c1a0e64027a3da640977f8ad34fbd3624d129e729.json`
  (`receipt_sha256 9fba43a1cc8d06eca97a88b34d783c48e00f463f548124337ad95712630dd87f`,
  result payload
  `/mnt/shared/prismabuild-fleet/cas/blobs/ff/ff79b6f86b6534c7fd2bcbdd6cab74c5b1b033fda9cc7ee0650c4f5ef041cdc6`).
  pbrun noted `--timeout-s 14400` exceeds the box's 3600 s ceiling; the run
  needed 1390 s.
- Command: `pbrun.py --cwd /home/rob/tmp/pq-glm-probe-reduction --cpus 16
  --demand mem_gb=24 --tag x86 --priority -10 --timeout-s 14400 --wait-s
  14500 --env PYTHONPATH=. --env OMP_NUM_THREADS=1 --
  /home/rob/venvs/pq-cpu312/bin/python experiments/glm_probe_reduction_regret.py
  --out BASE/probe-reduction-regret-20260910-02 --workers 16 --seeds 50
  --sweep-seeds 8 --h-seeds 3` (log: `w2/pbrun-regret-02.log`).
- Result: `BASE/probe-reduction-regret-20260910-02/regret_study.json`
  (copied to `/home/rob/tmp/glm-perf-20260910/w2/regret_study.json`),
  SHA-256 `4336ceac82af47e93928372eb0cadb328aa555151a073e0862ddb27b46594860`;
  per-group partials under `.../partial/`. Local smoke runs (60 rows, 2
  seeds) under `w2/smoke-local*/` are dev checks, not results.

## 7. Consultations

- `advisor` (session reviewer), before writing: flagged budget saturation
  above +2.5%, the PB symlink refusal (none present on this checkout),
  writing outputs under BASE, reusing the campaign estimator, two weightings
  plus a lognormal sensitivity, exact MCKP, seed-quantile reporting, actual
  encode accounting. All adopted.
- `fable-high` (one consult): recommended the model-assisted difference
  estimator for (b) and warned that the intercept law has no design
  correction; recommended a p90 gate at the actual budget with the +-x% sweep
  as a diagnostic; predicted weighting could flip the (b)/(d) ranking;
  suggested the marginal-price diagnostic; listed the four Gridbook
  non-transfer scope limits (currency, decision unit, wire, weighting).
  Verified against the code (`_stack_cost_rows` writes HT fields only;
  `draw_stack_sample` `remaining == 1` refusal; `_horvitz_thompson_stack`
  m < 2 refusal). Outcome: the difference estimator was implemented and
  **measured worse on the tail at s <= 3%** (section 3.4 point 2), so the
  recommendation is `b_law`, not `b_diff`; the rest was adopted.

## 8. What is NOT established

- No KL, joint-AURA, or served-quality claim. The campaign's final selection
  objective is joint AURA (`tessera_joint_aura.py`, `compute_aura_cost_streamed`,
  GPU KL-adjoint over exact measured cells); the code itself calls the scalar
  MSE "evidence of which wires were made, never a joint price". The regret
  here is scalar route-MSE regret on the menu the schedule produces.
- The true objective weights (`h_trace`) are unknown for GLM; `uniform` and
  `counts` are stated proxies and `lognormal_h` is a hypothesis. The
  schedule ranking was stable across all three, the magnitudes were not.
- 28 of 42 stacks (the completed rows), 118 of 132 rows. Layers 0-6, 8-9,
  15, 35, 37, 39-43 are not in the study.
- +5% and +10% budget columns are degenerate (menu saturated).
- The 42-stack GPU-hour figures extrapolate per-stack encode wall time; the
  selective-encode and repair costs in section 5 are estimates from this
  study's allocations, not measurements.
- The dense cross-family screen never exercised a family decision (BF16 wins
  every item on this menu and budget). That is partly by construction of
  this study's objective: at the same rate BF16 costs only 0.06-0.48% more
  `wire_bytes` than E4M3 (405/405 matched pairs; a ~16 KB fixed plane) and
  has lower `output_mse` on 405/405 pairs, so a scalar route-MSE-per-byte
  knapsack always prefers it. This study's objective consumes only
  `output_mse` and `wire_bytes`; no serving-time or route field enters the
  knapsack (a serialized payload row contains no `prefill_ms`, `decode_ms`,
  `latency`, `route_status` or `activation_contract` key, checked on
  `rows/row-*` payload 0). The activation-contract difference between the
  families (section 2 citations) is therefore outside the objective, and
  the family axis is degenerate here, not settled.
- The study draws PPS on the arm's weights; a production draw on `counts`
  when the objective is Fisher-weighted is unbiased but carries the product's
  spread (as `draw_stack_sample`'s docstring says); that variance was not
  simulated except through the lognormal arm.
- No code change to the campaign was made; the sketch names the seams only.
