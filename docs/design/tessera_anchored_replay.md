# Sparse Tessera curve replay and the activation/prefill axis

Status: opt-in research replay, updated 2026-09-11. This capability emits a report and
measurement requests, not allocator costs or serving qualification. The normal
Tessera campaign retains its piecewise log-linear interpolation and adaptive
measurement policy.

## Recovered mechanism

The former Gridbook campaign used a shared projection-level error curve and a
per-expert correction. Two measurements fitted both a level and a slope;
separate audit measurements decided whether to use predictions or measure the
full declared grid. The retained implementation is in
`tools/dsv4_afast_campaign.py` and `tools/dsv4_afast_burn.py`, originally adopted
in `0b34145c17f76a2ec715af13e18715b25b0007af`. The earlier study survives at
`629cfb79958a4d4400471d8044b5df93b061908f`, under `interp-diagnosis/` and
`cost-ldlq/interp-diagnosis/`.

That history supports recovering the mechanism, not transferring its constants
or declaring all ranges predictable. The K28–38 median study recorded worst
rung errors of 3.50%, 5.87% and 6.34% on held-out L0 gate/up/down projections.
Broader ranges and heterogeneous experts produced larger misses. The archived
480-solve regret study used a weight-MSE proxy with uniform sensitivity, not
served KL or AURA. Its tolerances are not Tessera promotion gates.

`prismaquant.anchored_shape` now supplies the numerical mechanism shared with
the existing strict AURA fitter. Given a declared segment shape G and a
centered/scaled rate coordinate x:

```text
log10 D_i(x) = log10 G(x) + a_i + b_i*x
```

The pilot fit removes each pilot unit's own log-cost mean before solving its
feature coefficients. One anchor sets b to zero; two fit level and slope.
An explicit later fit can use more observations. Real measured points remain
exact. Invalid costs, unsupported keys and insufficient design rank refuse;
there is no fabricated epsilon loss or inherited Gridbook modulo-four term.
The strict `anchored_cost` AURA wrappers keep their currency, render-receipt
and segment checks.

## Two-endpoint conditional curvature

`fit_endpoint_curvature` fits a shared, degree-one or degree-two polynomial
from pilot measurements. For a unit's positive endpoint costs L and H and
normalized rate t, the value-mode predictor is

```text
z = standardize([log2(L), log2(H)-log2(L), t])
D(t) = (1-t)*L + t*H + L*t*(1-t)*P(z)
```

Standardization and five ridge/Huber fitting iterations use pilot observations
only. Every pilot unit supplies both endpoints and at least one interior;
the fit needs at least twelve interior observations. Log2 mode applies the
same endpoint-vanishing correction to the log2 chord. No loss floor is
invented. `EndpointCurvatureModel.bind(L, H)` expands P once into three
coefficients in t; the resulting frozen curve uses Horner arithmetic for
arbitrary in-range queries. It returns actual endpoints exactly and refuses
extrapolation, nonpositive predictions and unrepresentable arithmetic.

This is an opt-in alternative to the centered-log shape. A replay segment
selects it with `shape_model: {kind: endpoint_curvature, mode: value, degree: 1}`
and omits `features_by_key`, since this model owns its feature basis. Its
held-out anchors must be the two declared domain endpoints, and
`refit_after_audit` must be false. Existing currency/segment checks, independent
audits, monotonicity checks, measured overlays and fused-group measurement
requests apply unchanged. A rate measured in pilots is held out only by unit;
an unobserved pilot rate is held out by both unit and rate.

The offline study tools `experiments/sparse_rate_dataset.py` and
`experiments/sparse_rate_models.py` bind a compact measured-only dataset to
the original files and calibration contract. The study caps shared pilot
units before reading their interior values, uses whole-layer development
folds, and reserves `layer % 5 == 3` for a separately invoked frozen-choice
evaluation. Saved fitted models expose reusable coefficients. These artifacts
are scalar output-MSE evidence, not joint AURA, exact wire predictions or
serving admission. Sparse interior measurements from an adaptive campaign
are not a uniform sample of its full rate span.

The [2026-09-11 frozen-layer evaluation](../measurements/sparse-rate-interpolation-2026-09-11.md)
passes its declared scalar screen for expert gate/up projections and fails it
for down projections. It does not support a universal two-measurement cap or
an achieved campaign-time reduction.

## Adaptive acquisition on a complete measured curve

`AdaptiveAnchoredCurve` accepts a finite, strictly increasing integer roster
and positive, strictly decreasing measured values. Its immutable state exposes
the next requested coordinate and its prediction before that value is known.
Callers obtain the measurement through the existing campaign machinery, then
record that exact coordinate. The sampler has no access to unmeasured truth.

Each interval gets either a midpoint check or two approximately third-point
checks. These are compared with the parent interval's frozen value/log2 PWL
prediction. A failed interval splits at its measured checks; an accepted
interval retains those checks as empirical evidence. The measurement budget
includes endpoints and every check. Unresolved intervals remain explicit at
the budget limit, and invalid or nonmonotone measurements refuse rather than
being smoothed into a plausible curve. Sentinel agreement cannot prove a
bound on every unseen rate.

`--exhaustive-rate-grid` with an explicit `--rate-band` provides the complete
measured curve for a bounded research experiment. It uses the existing encoder,
cache and journal and does not extend any family's legal or serving domain.
`experiments/collect_complete_rate_curve.py` binds that journal and its wire
receipts to a premeasurement plan. `experiments/sparse_rate_adaptive.py` replays
a frozen matrix of acquisition policies and compares each with a geometric
widest-gap schedule at the same measurement count. Accuracy is scored on
never-revealed rungs, with separate common holdouts for direct policy comparisons.
Families, activation contracts and recipe segments are independent curves.

`experiments/sparse_rate_curve_oracle.py` uses all measured truth to compute the
fewest anchors that meet a chosen maximum relative error with value/log2 PWL
interpolation. This exact finite-grid lower bound distinguishes a difficult
curve from an inefficient acquisition policy. Its selected anchors cannot be
counted as an achieved measurement saving: the oracle already read every rung.
An infeasible strictly decreasing path reports refusal explicitly.

## Replay inputs and output

Run from the repository with the same Python dependencies as the campaign:

```bash
python tools/tessera_surface_replay.py \
  --costs /run/cost.pkl \
  --checkpoint /run/anchors.json \
  --plan /run/replay-plan.json \
  --out /run/replay-report.json
```

The input is a trusted local campaign pickle plus the existing checkpoint
manifest, its `.parts` journal, and its recorded wire directory. The plan
binds the exact payload SHA-256 and checkpoint identity SHA-256. The importer
reuses the journal reader, checks measured rows against recorded anchors and
source/encoder/Hessian/static-scale identities, and verifies wire hashes and
sizes. It does not re-read model tensors or re-run producer wire validation.
Its report states that boundary explicitly; recorded identity replay is not
fresh source or serving attestation.

The plan schema is `prismaquant.tessera_anchored_replay.plan.v1`. It declares:

- `input.payload_sha256`, `input.checkpoint_identity_sha256`, and `currency`;
- a list of `segments`, each with an `id` and a `descriptor` identifying
  family, explicit profile role, activation contract, geometry, wire recipe
  without the rate, and Hessian applicability;
- `features_by_key` and integer rate `coordinates`, keyed by format name;
- `pilot`, mapping pilot unit names to measured format keys;
- `heldout`, mapping distinct unit names to one/two `anchors` and separate
  `audit` keys;
- `max_absolute_log10_error` and optional `refit_after_audit`.

The exact example construction and executable cases live in
`tests/test_tessera_anchored_surface.py`. Roles are explicit plan declarations;
the importer does not certify that a declared transfer class generalizes.
Actual family, activation, geometry, recipe and Hessian boundaries must agree.
The pilot covers the target rate envelope; held-out units are distinct from
pilot units, and audit rungs are distinct from their fit anchors.

Audit predictions are frozen before audit truth can enter an optional refit.
Reports retain the pre-refit errors, measured/predicted provenance and a
piecewise log-linear baseline where two anchors bracket the audit rung.
Missing evidence, failed audits or nonmonotone curves produce deterministic
measurement requests. Requests propagate to members of existing campaign
anchor groups. They are requests to the existing measurement machinery, not
newly encoded wires. Packed experts never become selectable from this report.

Replaying identical inputs is deterministic. A changed report cannot overwrite
an existing output pathname; choose another path. The plan hash binds the split,
features and policy. No speedup, full-model quality improvement, or production
qualification is implied by a successful replay or its numerical tests.

## AURA and overlapping activation families

The importer currently accepts only
`output_mse_under_route_activation_contract`. That is the actual campaign
currency. AURA import requires its own attested measurement path; MSE cannot
be relabeled as AURA. Interpolate within activation/recipe segments, then
compare candidates in a common validated downstream quality currency.

Tessera output-MSE scoring includes the local joint weight/activation residual
but discards its direction before scalar sensitivity weighting. The separate
streamed AURA producer's opt-in `--joint-activation` path projects the complete
joint residual through AURA's output cotangent G:

```text
deltaY = Xhat What.T - X W.T
       = X deltaW.T + deltaX W.T + deltaX deltaW.T
a[k] = <G[k], deltaY>
predicted_loss = 0.5 * mean_k(a[k]**2)
```

That implementation preserves signed cancellation across terms and repeated
invocations before squaring, persists aligned signed samples, and reuses the
production weight and activation caches. `tessera_joint_aura` prepares these
inputs from completed measured campaign anchors. See
[joint AURA and runtime allocation](joint_aura_runtime_allocation.md) for the
current implementation and its measurement gates. Its existence does not
convert this importer's scalar MSE predictions into measured joint prices.

Quality and runtime form distinct axes. With a complete measured runtime table,
the opt-in allocator preserves faster same-byte alternatives and searches the
discrete bytes/quality/prefill frontier, with optional decode and device-memory
limits (`measured_runtime_prices` and `solve_runtime_frontier`). Legacy prefill
SLO flags alone still filter after bytes/quality search. The frontier requires
joint quality inputs and complete whole-operator runtime bindings; the scalar
replay here supplies neither. A weighted penalty sweep or convex hull cannot
represent every nonconvex feasible choice.

Use measured kernel/workload costs to propose assignments and end-to-end
serving measurements to qualify them. Activation width, encode time and weight
bytes do not determine prefill time. Serialized weights, resident terminal
weights, activation scratch and KV memory also require distinct accounting.

Validate close cross-family choices using paired probe differences and shared
held-out calibration, then whole-assignment/close-swap checks against the fixed
teacher. Probe covariance is not layer-error additivity. Re-centering a Fisher
square on a quantized assignment alone changes its reference and omits the
fixed-teacher first-order term. Current historical AURA validation does not
establish these overlapping families' rankings or cross-layer additivity.

This joint AURA/runtime extension and its required current measurements are
tracked in [PrismaQuant #237](https://github.com/RobTand/prismaquant/issues/237).
The separate opt-in grouped-uncertainty defect is
[#236](https://github.com/RobTand/prismaquant/issues/236). Production defaults,
serving admission and the existing uniform-control/quality gates do not change
through research replay.
