# Activation quality versus prefill: proposed experiment

Date: 2026-09-11. **Unmeasured plan, not a completed frontier.**

Rob proposed plotting activation error against prefill speed and using the
knee to guide allocation. The active branch already has an exact discrete
runtime frontier. The missing input is matched runtime evidence for the
candidate recipes; the current scalar-MSE study supplies no prefill timings.

## One joint allocation, staged measurement

The proposed decision is a single assignment over all mutable serving units.
For each unit, enumerate the legal tuples `(weight family, weight rate,
activation policy, execution route)`. The runtime's supported combinations
define the menu; an arbitrary Cartesian product of formats is not legal.
Treat fused projections, expert stacks and other coupled serving units as
atomic options when their execution contract requires it.

For assignment `c`, the constrained objective is:

```text
minimize    predicted joint-AURA loss(c)
subject to  serialized mutable bytes(c) <= byte budget
            resident + peak scratch/activation + fixed/KV bytes(c) <= device budget
            measured operator prefill proposal(c) <= prefill budget
            optional decode constraint and all serving compatibility checks
```

Sweep the prefill budget to obtain the frontier. Weight and activation choices
remain free in every solve; a screening stage must not silently freeze one
axis before optimizing the other. The exact solver can find the optimum for
its retained discrete menu and additive cost/resource model. That is not a
proof of the best possible whole-model assignment: screening can omit useful
choices, downstream loss can be non-additive, and mixed execution can alter
runtime. Validate complete assignments near the proposed optimum and retain
the evidence needed to expand or correct the menu/model if rankings change.

The stages below control the cost of acquiring evidence. They are not separate
greedy weight, activation and format allocation passes.

## Quality pricing and selection

1. Use AQUA's cheaper activation-aware approximation to screen a broad menu
   and identify layers where higher activation precision may buy useful
   quality. Preserve the approximation's uncertainty when screening.
2. Use joint activation/weight AURA to price the surviving options with the
   same source, calibration, actual rendered weights and served activation
   contract. Its full output residual includes weight error, activation error
   and their interaction before projection and squaring. Do not add the
   separate AQUA activation term to a joint-AURA price a second time.
3. Combine those quality prices with measured operator runtime and memory
   resources. Sweep prefill budgets at a fixed byte/device budget using the
   existing exact frontier. Keep coupled serving operators atomic.
4. Serve candidate assignments around the apparent knee and compare actual
   latency and held-out quality. Local prices and sequential operator sums
   propose assignments; they do not establish whole-model KL or p95 TTFT.

An activation-only ablation can hold the rendered weights fixed and measure
the output change caused by activation quantization. A deployable recipe
comparison must score the actual combined weight/activation behavior. This
distinction matters for Tessera families, whose reconstruction alphabets can
change along with their activation contracts.

The existing complete-rate scalar MSE curves are useful for economical
candidate measurement and interpolation research. Their cost currency is
`output_mse_under_route_activation_contract`, not `joint_aura_predicted_dloss`.
They must not be relabeled or added directly to joint-AURA rows.

## Workload proposal

Prepare two separate views, subject to the final serving workload freeze:

| View | Proposed prompt tokens | Batch/concurrency | Purpose |
|---|---:|---:|---|
| Interactive | 2,048 | 1 / 1 | Single-request prefill and TTFT |
| Long context | 32,768 | 1 / 1 | Longer prefill work |

These are planning assumptions, not measured workloads. Bind prompt content,
tokenization, prefix-cache policy, prefill chunking, TP, graph mode, residency,
KV dtype/budget and the exact runtime manifest before acquisition. Keep them
fixed across the candidate assignments within each view. If the selected
runtime cannot execute a proposed context or topology, revise the workload
before measurement and record that revision.

Measure prefill duration directly where the engine exposes the phase, and
report prompt tokens divided by that duration. Report client TTFT separately;
it can include queueing, setup and first-token work. Preserve repeats and
uncertainty instead of treating one timing sample as a price. Measure resident
weights, activation/scratch peaks and fixed/KV costs along with serialized
bytes. Weight bpp excludes immutable, unquantizable parameters.

## Frontier and knee

The points should be mixed per-Linear assignments, including conservative
endpoints, rather than only three uniform activation formats. Retain the
selected A16/A8/A4 distribution, actual weight recipes, bytes and workload
identity on every point. Unknown or unsupported runtime prices remain gaps.

Use the empirical non-dominated set first. Display both the quality/pre-fill
latency view used by the budget solver and a tokens/second view. A knee is a
candidate operating region where the marginal quality benefit from additional
prefill time changes sharply. Its appearance depends on axis scaling and
measurement uncertainty: declare the display/normalization before scoring,
show neighboring points, and report an ambiguous or absent knee honestly.

Convert the selected region into an explicit prefill budget or quality ceiling
for deterministic allocation. Do not replace the discrete frontier with a
weighted sum that can skip non-convex feasible choices. Whole-model validation
must include the knee's neighbors; a locally predicted knee can move after
serving or quality measurement.

## Existing implementation and present gaps

- `prismaquant/aqua_activation_cost.py`: approximate activation-side pricing.
- `prismaquant/joint_aura.py`: joint residual/projection and cost currency.
- `prismaquant/measured_runtime_prices.py`: strict measured whole-operator
  prices and workload/runtime identity, distinct from served-SLO evidence.
- `prismaquant/allocator_solver.py::solve_runtime_frontier`: exact discrete
  search under byte, prefill and optional decode/device limits.
- `prismaquant/allocator.py`: measured-runtime-table integration and checks
  that serving expansion preserves the measured operator assignment.

The older `speed_quality_frontier.py` composes declared speed hints. Those
hints cannot substitute for measured prefill prices in this experiment.

The reviewed v22 Tessera pin in this branch qualifies routed E4M3 K1 R1024;
it does not qualify routed A16 or A4 alternatives. Research operator timing
can investigate such alternatives, but each point must retain its qualification
status and distinguish a kernel measurement from a served result. The reviewed
GLM evidence contains quality/bytes rows and unrelated phase diagnostics, not
a matched activation-quality/prefill table. A measurement producer and exact
supported candidate/workload matrix remain to be specified before acquisition.

Non-vLLM operator measurements run through PrismaBuild, with a complete paired
experiment inside each isolated measurement action. vLLM serving and its
benchmarks use the explicit vLLM exemption. Collect in-process profiles and
host telemetry for any performance comparison; on GB10 use power and useful
work per joule, not utilization percentage, to assess resource use.
