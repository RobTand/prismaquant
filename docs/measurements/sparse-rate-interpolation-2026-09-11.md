# Two-anchor interpolation on the completed GLM scalar census

Date: 2026-09-11. Research result; **the global accuracy target failed**.
This is a reproducible interpolation implementation and a bounded allocation
proxy replay. It does not qualify joint AURA, served KL, a new serving rung, or
a production change to the campaign's interpolation policy.

## Implementation and declared evaluation

The existing `anchored_shape` core now fits endpoint-conditioned curvature.
A small shared pilot learns a degree-one or degree-two polynomial of endpoint
log level, endpoint log ratio and normalized rate. A held-out unit supplies
only its two endpoints, R832 and R1088 (3.25–4.25 body bits/parameter).
Binding expands that polynomial once; subsequent queries use Horner arithmetic.
Exact anchors remain exact. Invalid values and extrapolation refuse.

`tools/tessera_surface_replay.py` exposes the method through the existing
receipt-bound replay, using the optional segment `shape_model` described in
[the design](../design/tessera_anchored_replay.md). The same importer, currency
and recipe checks, separate audits, measured overlays and group measurement
requests apply. In that validation interface audits are additional measured
points; it is not a campaign mode that automatically stops all units at two.

The compact dataset was extracted from all 132 planned completed `cost.pkl`
files in `first-proof-anchor-preparation-05`. It contains 36,693 unit/family
rows and 125,144 explicitly measured observations; 68,573 unmeasured or
interpolated entries were excluded. The underlying model has 36,423 units:
36,288 routed expert units with E4M3 only, plus 135 dense units with three
families. Family names describe reconstruction alphabets, not stored bits.
E2M1 has only R896 in this dataset, so it has no two-anchor curve to validate.

The measured rates are 832, 864, 880, 896, 912, 928, 944, 960 and 1088.
R832/R960/R1088 have broad coverage; intermediate coverage is sparse and was
adaptively selected. **There is no measured R1024, no upper-half interior
observation above R960, and no evidence here for the full reader span or the
BF16 dense serving rung R1792.** The currency is
`output_mse_under_route_activation_contract`, not AURA or KL.

Before model selection, final layers were fixed to `layer % 5 == 3`:
3, 8, 13, 18, 23, 28, 33, 38 and 43. Development used whole-layer fourfold
splits on the remaining layers. Pilot selection used the smallest SHA256 of
unit names within each fitting segment, capped at 256 units before reading
interior costs. Segments keep family, recorded activation, dense/expert role,
projection, rows and columns separate.

The final choice was frozen in `development-04/choice.json`, including hashes
of the dataset, manifest and numerical implementation. It selected
`value_surface1` for two anchors and `one_ridge2` for one anchor. The latter
uses R960 and remains a failed comparator. The declared scalar screen was
p99 relative error <=1%, maximum <=5%, complete predictions, and at least 100
held-out observations per segment. This screen is not a production gate.

## Accuracy: untouched layers

Errors below are absolute relative errors of scalar MSE predictions, excluding
measured anchor points. They are not percentages of whole-model quality loss.

| Method or segment | Audited points | Mean error | p99 error | Maximum error |
|---|---:|---:|---:|---:|
| Log-linear endpoint baseline, all eligible segments | 12,886 | 2.953% | 12.743% | 34.789% |
| Conditional two-anchor model, all eligible segments | 12,886 | 0.956% | 11.650% | 34.638% |
| Two-anchor, expert gate | 4,256 | 0.215% | 0.787% | 3.033% |
| Two-anchor, expert up | 4,256 | 0.211% | 0.827% | 2.521% |
| Two-anchor, expert down | 4,320 | 2.420% | 17.218% | 34.638% |
| One-anchor comparator, covered points only | 19,116 | 2.006% | 13.632% | 59.670% |

The two-anchor model covered every eligible final observation. The one-anchor
comparator missed 1,600 additional observations because its per-rate prior had
insufficient training coverage. Its covered-point error must not hide that gap.
Dense segment counts were only nine observations each in the final split.
They do not satisfy the declared minimum evidence count.

Expert gate and up pass the declared scalar screen. Expert down fails it.
The global p99 is much worse than its development value of 4.065%, which is
why the untouched-layer split matters. Its development maximum was also bad:
73.357%. Do not turn the improved average into a claim that every curve is
accurate. More context from sibling endpoints did not solve the development
tail and added substantial preprocessing cost; those candidates are retained
as negative research comparisons, not selected inference paths.

## Allocation proxy

The audit makes each complete routed layer one atomic rate choice over all
288 experts and all three projections. Candidate costs use summed measured
scalar MSE, either uniform or weighted by recorded routed-token counts.
Candidate bytes are sums of measured wire bytes. No predicted byte count,
family fallback, partial group or invented quality price is admitted.

An exact byte DP compares the fully measured oracle with predicted-cost
selection at budgets 3.50, 3.75, 4.00 and 4.25 bits per quantizable parameter.
The nominal 3.25 budget is below the minimum actual wire footprint and is
excluded, rather than silently discarding overhead. The DP fails on its
explicit state/transition limits instead of approximating.

The selected two-anchor model reproduces every oracle choice at all four
budgets under both weightings, on both the 33 development routed layers and
the nine final routed layers: observed scalar objective regret is zero.
On development, the log-linear baseline has maximum count-weighted regret
0.102597%, and the one-anchor comparator 0.012486%. All three reproduce the
oracle choices on the smaller final set.

This is only a three-rate, single-family, whole-layer scalar proxy at four
budgets. It does not establish every budget, cross-family activation rankings,
per-expert selectable rates, joint AURA ranking or served quality. In
particular, the allocation result does not override the failed down-projection
curve screen.

## CPU evaluation cost

A separate PrismaBuild **measurement** action on dl380g10 compared direct
standardized polynomial evaluation at each rate with binding once and using
Horner evaluation. Both arms used the same 28,728 actual development endpoint
pairs and 17 rates from 832 through 1088 at step 16: 488,376 scalar queries.
The degree-two log2 coefficients were deterministic synthetic values. This
isolates numerical evaluation cost; it is not an accuracy measurement of the
selected degree-one value model.

Three interleaved AB/BA repeats, one admitted CPU and native threads bounded
to one, gave these unprofiled medians:

| CPU operation | Seconds |
|---|---:|
| Direct polynomial evaluation | 1.5293 |
| Bind plus evaluate | 0.7383 |
| Binding alone | 0.2565 |
| Evaluation after binding | 0.4751 |

The complete numerical operation is 2.07x faster; evaluation after binding is
3.22x faster. Maximum cross-arm relative difference is 2.50e-15; endpoints
are exact. Every repeat has a stable per-arm checksum. Both arms also have
cProfile and pstats artifacts. The direct profile attributes substantial time
to rebuilding polynomial features and summing them at each query; the bound
profile replaces that work with one expansion per unit and scalar evaluation.
Profiler-instrumented times are not used in the table.

The PB scope reports 22.53 CPU seconds, peak memory 848,875,520 bytes, and
24.999 seconds wall for the whole benchmark action, including loading,
repeats, equivalence checks and separate profiles. Netdata CPU/RAM series from
**dl380g10 and both Sparks** cover the window. PB's dl380g10 CPU series averages
3.00% host busy, peaks at 4.39%, and reports maximum PSI some avg10 of 0.45.
Its absent pqteld CSV is recorded; this was CPU-only work and carries no GPU
saturation or energy claim. CPU-source staging and submission were performed
on dl380g10 so measurement attestation matched the executing platform.

## Measurement count and remaining work

The final shared fit used 966 pilot unit/family pairs and 3,167 measured cells:
1,235 measurements beyond those pilots' two endpoints. A hypothetical
all-two-anchor search over 36,558 endpoint-eligible pairs would therefore use
74,351 points including this pilot, versus 109,674 for three points per pair.
That is a **nominal 32.2% search-point reduction**, before validation, selective
extra measurements, unsupported geometries or the chosen artifact's wire
encodes. The failed accuracy screen prevents claiming it as an achieved
campaign saving. Encode times vary, so point counts are not wall-time savings.

The selected artifact still needs real encoded bytes and receipts. In the
current campaign, R1024 wires are absent for the routed body; interpolation
cannot remove that required encoding pass. Runtime attestation also applies
to exact rungs, never to interpolated admission. The R1024 extension can provide
prospective interior evidence that this historical dataset does not contain.

A hard two-point cap across every segment is not supported by this result.
Accuracy-first operation needs selective additional measurements or a better
validated model for down projections, plus independent joint-AURA and serving
checks before any production use. The new replay retains explicit measurement
requests and the existing production defaults.

## Reproduction and receipts

Evidence root:
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/sparse-rate-20260911`.
`receipt-summary.json` binds the principal artifacts, verified terminal records,
CAS result hashes, checkout snapshots and resource profiles.

- Dataset: `dataset-01/manifest.json` and `sparse_rate_dataset.npz`.
  NPZ SHA256 `3547254ab482ed60f20ec738d2aaae23c2d2ca82cb2b9a53fddf756f52df855e`.
- Development: `development-02` (full training comparator), `development-03`
  (capped pilot and context comparisons), and `development-04` (shared fitter,
  frozen choice). Final: `final-01`. Each includes reports and per-model profiles.
- Exact allocation audits: `allocation-development-04` and `allocation-final-01`.
- Timing and profiles: `inference-bench-02`; host series:
  `telemetry/inference-bench-02.json`.
- CPU regression plus compile checks: PB
  `efd4b0cd4ecd9b7c1cecb48bd6f2c5a966c3ed4c40deadd9154b89156b2bd5f4`,
  **100 passed**, no skips, 112 Torch deprecation warnings. It covers the
  numerical core, both replay modes, extractor, study, allocation audit and
  architecture/document contracts. Compilation covered the touched Python modules.
- Frozen final evaluation and allocation: PB
  `cb691ee731f86beacd9f5f9e9a70175d525f2de0215b41ac4a8d2530a9ec7827`, exit 0.
- Controlled timing: PB
  `35bf707fbd32dbe4db905f053e47be73e71aa686b4ac9f9786f5d93e58b83760`, exit 0.

All tests and numerical replay/benchmark actions ran through PrismaBuild.
The CPU interpreter was `/home/rob/venvs/pq-cpu312/bin/python` on dl380g10;
the benchmark records Python, NumPy, platform and checksum details.

For a new reproduction, use new output paths. A development submission is:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /path/to/this/checkout --tag x86 --cpus 4 --demand mem_gb=8 \
  --priority -10 --env PYTHONPATH=. --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 -- \
  /home/rob/venvs/pq-cpu312/bin/python experiments/sparse_rate_models.py \
  --dataset /path/to/dataset-01 --out /new/development-output \
  --workers 4 --pilot-units-per-segment 256 \
  --models log_chord,value_surface1,value_surface2,one_ridge2
```

The final invocation uses the same submission envelope with `--stage final`,
`--choice /new/development-output/choice.json`, a new `--out`, and no
`--models`. It refuses a changed dataset or numerical implementation. The
existing final observations have now been examined; rerunning them is a
reproduction, not a fresh holdout for selecting another model. The inference
benchmark uses `experiments/sparse_rate_inference_bench.py --dataset ... --out ...`
with one CPU and 4 GiB, submitted from dl380g10 with `--measurement`.

Retained failed attempts: the first parallel profiler run collided with
CPython's cProfile monitoring slot (fixed with separate processes); an
independent exponential test initially used an absolute tolerance smaller than
one floating-point ULP (corrected to a relative tolerance); and an x86
measurement submitted from the ARM host was correctly refused at toolchain
preflight (resubmitted from dl380g10). None counts as a pass or a performance
sample. No final truth was used to revise the selected model after `final-01`.

## Follow-up: Gridbook's logarithmic interpolation

After the final results had been examined, Rob requested a comparison with the
earlier Gridbook method. The original artifacts and frozen choice above remain
unchanged; subsequent model comparisons are exploratory.

`tools/dsv4_afast_campaign.py:89-133` records the hierarchical law
`log D(K) = log G(K) + a + b*K`, where the shared projection-specific shape
`log G(K) = a0 + a1*K + phi[K % 4]` came from a pilot. One per-unit anchor
fixes its level; two fix level and tilt. With two anchors, this is exactly
the endpoint log chord plus the part of `log G` that departs from its own
endpoint chord. A purely linear shared log shape therefore reduces to the
log-chord baseline already evaluated here. The useful information is the
shared nonlinearity, not the logarithm's base. The archive also records
measured audits and fallback, including down-projection failures; it did not
establish an unconditional two-point guarantee.

`prismaquant/expert_empirical_cost.py:894-996` carries a different Gridbook
proposal: `D(K) = F + C*R(K)`, where `R(K)` sums the rate factors of the actual
ceil-first subtable split. It models a nonzero floor and the split-phase
variation that a smooth exponential misses. The smooth three-parameter floor
law needs three anchors unless the decay shape is supplied independently.
Neither Gridbook's modulo-four terms nor its fitted coefficients describe
Tessera's encoding grammar.

The transferable idea is to use Tessera's own rate geometry. The campaign's
resealed producer, `identity-reseal-20260911/producer-source-d403cc5a31`, uses
`tessera/grammar.py:279-324` and `tessera/export.py:1786`: scalar column rates
mix the two neighboring integers in exact proportions. This gives a known
change of mixture at R1024 in the present band. It motivates a shared log
shape with a hinge there and a floor-law comparison using the mean of
`2**(-beta*column_rate)` for fixed beta hypotheses. This coordinate comes from
the schedule; the proposed distortion law still needs measured validation,
because LDLQ, activation error and output cross terms need not follow it.

The bounded comparison reused the same development folds, segment definitions,
pilot cap and endpoint eligibility. The two-anchor comparisons each have
39,007 interior observations, 38,989 predictions and 18 unsupported cases.
No final-layer fit, scoring or model selection was performed in this follow-up.

| Development method | Mean relative error | p99 | Maximum |
|---|---:|---:|---:|
| Endpoint log chord | 2.765% | 6.908% | 62.699% |
| Shared log shape with Tessera R1024 hinge, two anchors | 1.624% | 5.712% | 66.407% |
| Previously selected conditional value model | 0.512% | 4.065% | 73.357% |
| Integer-rate mixture floor law, beta=2 | 0.486% | 4.041% | 73.756% |

The rate-only shared log shape numerically reproduces the log chord, as the
algebra predicts. The one-anchor shared-hinge model has p99 error 26.123%
and maximum 87.957%, with 36 missing predictions among 67,735 observations.
Fixed beta values 1, 3 and 4 do not improve the two-anchor result. Beta=1
also gives negative inferred floors for most fitted units; these are counted,
not clamped into apparently exact anchors. Beta=2 has nonnegative fitted
floors on its covered population but still fails the accuracy screen badly.

**Disposition: reuse the existing shared-shape machinery; do not adopt another
model from this comparison.** The slight beta=2 mean/p99 difference versus the
selected conditional model does not resolve its extreme tail. No additional
model search or production change followed. The comparison and its cProfile
artifacts are retained in `gridbook-transfer-development-02`. The earlier
`-01` is superseded because its truth population did not exclude units lacking
the required anchors. This is an accuracy comparison, not a timing claim.

PrismaBuild action
`770db46a523a50fdac43ba8375d2e22e274179c7fa7eee3cac8749efec2899d1`
completed the corrected replay. Report SHA256:
`f81f58bfe9dbc2a9be4ab11dc7cd7d9c3d4937dfa2b818efb76f24c7a29aa0ef`.
Focused transfer, shared-shape and study tests: **30 passed**, no skips,
14 Torch deprecation warnings, action
`aa9f82e3b7ddda0938c844a37b8d936975c24e7547a1e1bee5d683c69192a706`.

## Allocation evidence integrity follow-up

Review found that the original allocation-audit reader verified the dataset
but did not bind the prediction archives to the study report and frozen
choice. A replay of the original snapshot demonstrated acceptance of wrong
currency and a 100-fold altered endpoint while its endpoint truth was masked
(PB `54a59141c19eac352b53d3ff059b1469657b35e5f6dd7e9333015c32267ffb8c`).

The audit now requires a separate post-run evidence seal covering dataset,
plan, report, model roster, declared anchors and prediction archives. Final
sealing verifies the frozen-choice and development-report identities; measured
anchor predictions are rechecked against the dataset. Reports preserve the
exact measurement currency and label their scalar aggregation separately.
Post-run sealing establishes integrity of the recorded bytes, not historical
proof that a choice preceded a measurement.

The original studies were preserved. `post-run-seals-01` binds them, and
`allocation-sealed-development-04` / `allocation-sealed-final-01` reproduce
the original numerical allocation results exactly. Focused integrity checks:
**9 passed**, no skips, 14 Torch deprecation warnings, PB
`e9e55c855b0d6f55c81689b7c8cef63b8a16a796eb27f70494ae36cc7a16d9fb`.
The two sealed audits ran as independent PB campaign actions
`6fb605ed901c6a3808c8bcec7a12345732b5697b0047f68f66b3ea385c4543d6`
and `f874d8609e88cdabc1ae78a9e24705adc1c1d16d4f715bfa2e94688d77e5c700`.
These follow-up receipts and artifact hashes are in `followup-receipts-01.json`.

After merging main's separate routed-stack transfer-law work (#499), four PB
test shards passed **85 tests**, with no skips and 56 Torch deprecation
warnings. They cover both research replay modes, the stack sampling/transfer
contracts and architecture staleness. `post-merge-tests-01.json` contains the
submission results and `post-merge-verified-01.json` records independently
checked terminal/CAS receipts. The frozen numerical core and study-script
hashes are unchanged by this merge.

## Prospective R1024 predictions

`prospective-r1024-03` records 7,830 predictions from the already frozen
`value_surface1` coefficients and the same final-layer endpoint measurements.
It covers nine routed layers and eligible final dense curves; the three E2M1
segments are explicitly omitted because they lack endpoint curves. No R1024
cost or extension journal was read to generate these predictions. The artifact
is prospective only at this new rate, not a claim that other final observations
remain unseen. Future comparisons still need matching calibration and currency.

PrismaBuild action
`46b93fcdb1d2b4aca94ddc599e0b56e459ceea2a22598598ba350866f20089ea`
completed with exit zero. The seal checks the frozen choice, report, dataset,
core and study-script hashes, exact anchors, positive finite predictions and
unique keys; it hashes `predictions.json` and records `measured=false` and
`serving_qualified=false`. Prediction SHA256:
`3e0fccfb597c840b9ff18b332d2df2d794f5d8b9e221850a66e41a040a8f1a77`.
`verified-receipt.json` holds the checked terminal/CAS evidence. The standalone
producer and prior withdrawn-attempt dispositions are archived with the
artifact; neither earlier attempt produced an accepted artifact.
