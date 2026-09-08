# Six GLM recipe intake — 2026-09-08

The six requested outputs can share one original capture, one measured candidate
union and one exact joint cost handoff. Hardware and quality objectives change
allocation/validation/export inputs; they do not justify another source capture
or probe. This is a source-cited intake map, not six completed recipes. Byte
ceilings, latency budgets, serving workload and qualified runtime prices remain
unbound. No new GPU, probe, export or serving run was performed.

Source basis: PrismaQuant `058bcc32a7966066a15a24aa744c14ec1bb1598a`, with the
metadata refusal in `26b97c6a4a`. PR #415's explicit source-derivative path remains
an integration dependency of the planned joint run. Earlier preparation is
retained at `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/reuse-plan-01/`;
its blocked native commands are not executable authorization. This report
updates the downstream recipe path rather than restarting that preparation.

## Shared immutable inputs

| Input shared by all six | Existing owner / admission |
|---|---|
| Original source census: SHA `b63f7bf6c4320714b4ceb38fbd6996e032e0f0c9b82ac2a30a8337d846e358fd`, 36,423 units, 132 groups | `tessera_joint_aura.py:80` binds census, campaign plan and exact complete roster; `tessera_calibration_cache` owns complete-capture/source authentication. |
| Original token artifact: SHA `9cd1fa129f249abd80d22efaeb8bc7e8b2d3b4252f173a8c6f2b2e496a4f8329`, 512 × 512, seed 0, fit IDs `6b6a0c4283de3aae633fd2bf00f74ac80928a8c389c42aa6d5101b765115532e` | `tessera_joint_aura.py:678` reads original int64 tokens and compares fit IDs/text/sample/sequence/seed provenance. No redraw. |
| Original complete capture manifest and source witness | Pending original completion and root seal; no partial manifest may substitute. The derivative path requires its explicit compatibility receipt; it does not create a new capture. |
| One measured candidate union: campaign plan, receipts, merged cost/checkpoint, wire and rendered-file identities | `tessera_joint_aura.py:80-165` requires complete receipts and selects measured cells only. Interpolated rows are excluded. Candidate scope must cover the union of all six target menus before measuring. |
| One prepared `ProductionWeightCache`, exact joint plan and joint output | `tessera_joint_aura.py:347` qualifies original capture/source/producer/render identity using existing PWC; `:669-814` executes the existing joint path. Topology, derivative and operator-window native gates remain prerequisites. |
| One allocation handoff pickle and receipt | `tessera_joint_allocation.py:157-208` binds joint output, plan, prepared completion/cache and historical wires. Joint currency and fields stay unchanged. Pass this same pickle as both allocator `--probe` and `--costs`. |
| One accounting convention and held-out validation contract | Quantizable-only bpp; immutable BF16 is excluded from that denominator but included in device/artifact requirements. All six need the same held-out teacher/draw and validation settings for comparisons. This held-out gate is separate from the original fitting draw. |

## What differs

These labels describe intent, not already selected thresholds or achieved speed.
No 8-bpp cap or allocator default Pareto grid is approved by this report.

| Recipe | Hardware input | Selection intent expressed by existing controls |
|---|---|---|
| `spark1_low_prefill_high_accuracy` | Explicit single-Spark profile and runtime context | Minimum measured/predicted loss under the chosen artifact/device budgets and a separately specified, relatively loose prefill bound. |
| `spark1_high_prefill_lower_accuracy` | Same hardware class; exact workload/context still required | Minimum loss under a stricter operator-sum prefill budget. A measured held-out quality ceiling must separately accept the result. |
| `spark1_balanced` | Same hardware class | Explicit intermediate constraints or measured-KL saturation among feasible points; no hidden weighted score. |
| `spark2_low_prefill_high_accuracy` | Explicit dual-Spark placement/topology, profile and runtime context | Same quality objective, with independently priced two-device resources and routes. |
| `spark2_high_prefill_lower_accuracy` | Same declared two-device topology | Stricter prefill constraint using its own measured context, followed by held-out quality acceptance. |
| `spark2_balanced` | Same declared two-device topology | Explicit constraints or measured-KL saturation among its feasible points. |

Two machines do not automatically mean TP=2. The profile's
`tensor_parallel.world_size` and sharding rules (`serving_profiles.py:750,937`)
must agree with the measured-runtime context's `tensor_parallel`, full runtime
configuration and export/launch topology. No allocator `--tp` option exists.
Per-unit dense/routed-MoE structure comes from model/probe facts, independently
of the target's platform (`tessera_serving_scope.py:1-45`). Exact context includes
residency, runtime-image digest, eager/compiled and graph mode, GPU identity,
source/calibration hashes, prompt tokens, batch size and operator routes
(`measured_runtime_prices.py:99`). An image digest alone is insufficient.

## Existing executable interfaces

The following are templates; capitalized values must be bound and hashed first.
They name existing consumers, not a new dispatcher. GPU work uses PrismaBuild;
vLLM execution is exempt. The prior plan supplies PB campaign admission and
fanout, so there is no machine-specific manual sharding here.

```text
python -m prismaquant.tessera_joint_aura prepare --plan JOINT_PLAN --plan-sha256 SHA
python -m prismaquant.tessera_joint_aura run --plan JOINT_PLAN --plan-sha256 SHA \
  --prepared PREPARED_COMPLETION --prepared-sha256 SHA
python -m prismaquant.tessera_joint_allocation \
  --joint-cost JOINT_COST --joint-cost-sha256 SHA \
  --plan JOINT_PLAN --plan-sha256 SHA --output SHARED_ALLOCATION_COST
```

After the shared measured handoff exists, each variant uses the existing
allocator. The following template includes the optional measured-runtime
predictor; its runtime flags require the additional admission described below.
The common cost handoff and ordinary byte/quality allocation do not require it:

```text
python -m prismaquant.allocator \
  --probe SHARED_ALLOCATION_COST --costs SHARED_ALLOCATION_COST \
  --model-override ORIGINAL_MODEL --target-profile VARIANT_PROFILE \
  --formats EXPLICIT_MEASURED_UNION_FORMATS \
  --target-disk-gb VARIANT_DECIMAL_GB \
  --artifact-overhead-reserve-bytes DERIVED_NON_TENSOR_RESERVE \
  --pareto-targets EXPLICIT_REVIEWED_BPP_GRID \
  --layer-config VARIANT_LAYER_CONFIG --pareto-csv VARIANT_PARETO_CSV \
  --pareto-output-dir VARIANT_PARETO_DIR \
  --tessera-platform PLATFORM --tessera-runtime-image REPOSITORY_AT_SHA256 \
  --tessera-execution-mode MODE --tessera-residency RESIDENCY \
  --measured-runtime-table QUALIFIED_TABLE --measured-runtime-context EXACT_CONTEXT \
  --slo-prefill-p95-ttft-ms OPERATOR_SUM_PREFILL_BUDGET \
  --serve-device-budget-bytes DEVICE_BUDGET \
  --serve-kv-bytes DERIVED_KV --serve-peak-scratch-bytes DERIVED_SCRATCH
```

Source: `allocator.py:1677-1753,1913-1950,2082-2155,3625-3674,3852-3906`.
`--target-disk-gb` is decimal GB and overrides the default `--target-bits`;
reported bpp still excludes pinned BF16. Bind `--formats` to the measured union
and record `PRISMAQUANT_TESSERA_MENU` explicitly: its unset production default is
`attested`; prior research planning used `readable`, which does not grant serving
qualification (`tessera_menu.py:175-200`). Do not silently substitute either menu. Optional
`--slo-decode-p95-itl-ms` is a separate proposal bound. Despite their historical
flag names, measured operator sums certify neither p95 nor end-to-end SLOs.
`--slo-decode-p05-tps` is refused in this mode. Legacy dispatch/workload-mix
flags are mutually exclusive with the measured table. The solver minimizes
loss among feasible candidates; it has no direct measured-throughput maximizer
or quality-ceiling CLI. Those acceptance choices must be explicit inputs.

`speed_quality_frontier.py:44,93,176,191` composes declared per-format speed
hints as a parameter-weighted harmonic index. Its speed-floor/quality-ceiling
functions are useful research selectors, but do not provide measured prefill,
route-aware latency or a production allocator CLI. They cannot fill absent
runtime table cells.

`validate_assignments_kl` produces assignment-bound measured KL. Then
`select_validated_frontier --validation-json VALIDATION --mode budget
--target-disk-gb LIMIT --output-layer-config LAYER_CONFIG --output-assignment
ASSIGNMENT --output-summary SUMMARY` selects minimum measured KL under a
whole-artifact upper bound. `--mode saturation` selects a lower-bpp point
within the measured uncertainty band; it needs repeated measurements for a
useful noise estimate. Neither mode selects by serving speed. A nonuniform
Tessera pick still needs the existing byte-matched uniform control; acknowledging
an outstanding control permits candidate construction, not shipping.

**Metadata limit:** allocator Pareto payloads (`allocator.py:4379-4470`) omit
selected Tessera wire/scale/source provenance; validator resolved payloads
(`validate_assignments_kl.py:2006-2025`) also omit it. The fixed selector refuses
to inherit another assignment's destination claims before writing any outputs
(#421). Full final allocator output (`allocator.py:5613-5765`) already applies
`priced_static_scales` and `allocation_expert_projection_block`; retain each
individually produced exact recipe and its own measured validation result.
Do not overwrite it with a different Pareto pick and assume provenance followed.
Publishing arbitrary validated Pareto picks still needs a selected-metadata
producer/validator handoff; this report does not implement that larger path.

Exporter intake remains the existing gate:

```text
python -m prismaquant.tessera_export_lane --model ORIGINAL_MODEL \
  --assignment VARIANT_LAYER_CONFIG --target-profile VARIANT_PROFILE \
  --hessian BOUND_EXPORT_H --input-scales BOUND_INPUT_SCALES \
  --tessera-platform PLATFORM --tessera-runtime-image REPOSITORY_AT_SHA256 \
  --tessera-execution-mode MODE --tessera-residency RESIDENCY \
  --write-build-json VARIANT_BUILD --print-build-sha256 --write-cached-expert-units
```

`preflight` (`tessera_export_lane.py:1443`) checks producer/runtime pins,
assignment scope, priced H and scale identities, and selected wire receipts.
The build anchors the exact selected cached-wire bundle for Tessera's own
producer tools (`lane_specs/tessera.json`, invoked by the existing pipeline's
export section). This command does not itself write a served checkpoint.
`artifact_collection.py:553,602` can bind the six target profiles and immutable
common inputs into collection records; it is an offline record owner, not a
runtime scheduler.

Do not use `tessera_materialization.run` as a GLM recovery shortcut: its current
`AutoModel.from_pretrained`/calibration path (`tessera_materialization.py:223-240`)
does not preserve this bounded original-capture contract. Measuring all selected
wires in the common union avoids that fallback. The top-level validated-surrogate
pipeline also prepares/recalibrates its own menu cache; use the explicitly bound
consumers above until that wrapper has a reuse contract for these artifacts.

## Direct served selection without runtime-v2 prediction

Runtime-v2 admission is an optional prediction path, not a prerequisite for
comparing actual served artifacts. A simpler route is to allocate candidate
recipes from the shared measured quality/cost table under explicit hardware
byte budgets, retain each allocator-produced recipe's metadata, validate each
candidate on the same held-out draw, and measure its actual prefill/decode in
the target engine on the same workload. Select the accuracy, prefill and
balanced outputs from those observed tradeoffs. Omit the measured-runtime/SLO
flags when using this route; do not attach a producer-admitted resource or
latency prediction to it. Device fit, serving correctness and performance still
need actual evidence for each selected topology. Candidate allocation and
served comparisons reuse the source probe; they do not require another probe.

## Remaining gates and evidence

1. Original COMPLETE capture, native wire-screen qualification, source derivative
   compatibility and full original-graph joint qualification remain owned by the
   active root campaign. None is replaced by this intake audit.
2. For the optional measured-runtime predictor, the qualified full-engine
   resource partition is unimplemented:
   `runtime_provenance.py:375-389` unconditionally refuses v2 fixed resources.
   Native operator rows do not close this gate. Filed #420, linked #237/#267/#323,
   because implementing and measuring the full producer exceeds this bounded
   metadata repair. The existing refusal must remain until real evidence exists.
3. Exact dense+MoE prefill/decode cells, fixed resources and end-to-end measurements
   for each declared one/two-Spark topology and workload are unavailable. The
   historical readiness inventory is retained by hash in recipe-intake.json;
   Tessera PR #427 has reviewed head `4cefd8d5f214b648aaed57846b5f5baaf6b212f4`,
   with no new native06 TP2+MoE measurement yet. Historical selected-expert
   controls are not full trained-model serving qualification.
4. Full export-H handoff currently accumulates all row H tensors in
   `dispatch_tessera_campaign.py:864-934`. Metadata-only PB arithmetic gives
   1,834,991,222,784 logical bytes, largest group 43,486,543,872 bytes. This is
   sum(4*K*K) over the existing 36,423 census entries, with no deduplication,
   serialization or process-memory claim. Root assigned the bounded canonical
   mapping/producer-reader repair independently; this branch edits neither owner.
5. Explicit budgets/SLOs/workload/topology and fixed-teacher, downstream quality,
   byte-matched uniform, final recursive-byte and unforked-vLLM ship gates remain
   necessary. No candidate is called accurate, fast, fitted or shipping here.

Sizing helper PB `2f047451fde4b09bd1dadc40e434ae4249c87032e073a34c8045b5d6ffa34adb`
ran on dl380g10 CPU 1 / 4 GiB, no GPU, actual exit 0; native CAS receipt/result
and source bundle verified. `hessian-sizes.json` SHA is
`7e0e365e6fc4e55ffd275a90d995eafd2ed2b6fb7ff670b3e0f9b4c3ff56c6e5`.
The metadata repair's red/green/compile records are in the sibling
`frontier-metadata-owner-20260908/` report. A separate comment-only commit fixes
two stale allocator statements about development overrides and retired Gridbook
routing; it changes no behavior and needed no additional test.
