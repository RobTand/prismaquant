# Joint Expert-Prune + Quantization Optimizer

This note describes the current PrismaQuant joint optimizer: the code path
that composes expert pruning with mixed-precision allocation. It is meant to
be read before changing the allocator modules:

- `prismaquant/allocator.py`: CLI orchestration and backwards-compatible
  re-exports.
- `prismaquant/allocator_solver.py`: candidate type, DP solver, achieved-bit
  accounting, and fused-format promotion.
- `prismaquant/allocator_candidates.py`: per-Linear candidate construction,
  passthrough-source gating, source dtype scanning, and fused-sibling
  aggregation.
- `prismaquant/allocator_prune.py`: MoE aggregation, prune candidates, global
  packed-MoE ratio rewrites, assignment expansion, and prune manifests.
- `prismaquant/schemas.py`: validation for probe, cost, layer_config, and
  prune-manifest handoff files.

## Objective

The allocator still solves a multi-choice knapsack:

```text
minimize   Σ unit_cost(choice)
subject to Σ unit_memory(choice) <= target_bits
```

The difference is that an MoE choice can now be a pair:

```text
(quantization_format, dropped_expert_ids)
```

For each candidate:

- quantization cost is the existing predicted Δloss for the kept weights;
- prune cost is `alpha * S_j` for every dropped expert `j`;
- memory is the physical bytes emitted by export for that candidate.

`S_j` is not the older `mean(g * ||f||)` ranking score. The probe stores
REAP dropout loss:

```text
S_j = (1 / T_cal) * Σ_t g_j(t) * ||f_j(t)||_2^2
```

That value is already in Δloss units, so the allocator does not square it
again and does not multiply by `h_trace / n_params`.

## Candidate Construction

### Nested / Per-Expert MoE

`aggregate_moe_candidates` groups per-expert Linears into synthetic
super-Linears. When pruning is enabled and saliency is complete for every
expert in a group, each format gains positive-ratio DROP variants. A DROP
variant:

- removes the lowest-saliency experts first;
- drops `floor(R * E)` experts for ratio `R` over `E` experts, so the
  requested ratio is a cap rather than a value rounded upward;
- sums quantization Δloss over kept experts only;
- adds `alpha * S_j` for dropped experts;
- scales memory and effective bits by the kept-expert fraction.

If the saliency map is incomplete, positive prune variants are not emitted.
Missing saliency is an observer or cache bug, not evidence that an expert is
dead. Dead experts must appear explicitly with `S_j = 0.0`.

### Packed-3D MoE / vLLM Uniform Expert Count

For `target_profile=vllm_qwen3_5_packed_moe`, vLLM/HF config uses a single
expert-count scalar. The allocator therefore sweeps global prune ratios. For
each ratio `R`, `apply_global_prune_ratio` replaces every packed entry's
candidate list with exactly one DROP variant per format. This makes uniform
`num_experts_kept` part of the DP problem instead of a post-processing guess.

If a packed entry has incomplete saliency, it is not rewritten for that ratio.
The main allocator refuses to emit a uniform-kept sidecar if the manifest pass
would need to add unscored drops afterward.

### Nested MoE With Scalar Expert Count

MiniMax-M2.7 stores experts as nested per-expert Linears, but its HF/vLLM
config still carries a single scalar expert-count field. For that profile the
allocator uses the same global-ratio discipline: after `aggregate_moe_candidates`
emits DROP candidates, `apply_nested_global_prune_ratio` filters every MoE
super-Linear to the ratio currently being swept. This keeps the DP honest:
format choices remain per group, but expert-count choices are globally uniform
and the manifest can be exported without post-hoc, unscored drops.

## Invariants

These are correctness constraints, not style preferences:

1. A pruned assignment must resolve to an exact candidate by both format and
   dropped-expert set. Format-only fallback is forbidden for pruned entries.
2. `compute_achieved` must use the chosen candidate's `memory_bytes` for
   pruned assignments; no-prune `_memory_bytes_by_format` is not valid.
3. Predicted Δloss for a final assignment must be the sum of chosen
   candidates' `predicted_dloss`; do not rederive it from `stats × costs`.
4. The prune manifest must describe the same drop set the DP scored. Default
   manifest construction never adds drops. `uniform_kept=True` is only for the
   packed global-ratio path and is treated as an invariant check.
5. Passthrough formats remain source-driven. A super-Linear can choose BF16 or
   FP8_SOURCE only if every member source dtype supports that passthrough.
6. Runtime shape masks propagate through aggregation. A fused MoE or fused
   sibling group can choose a format only if every member with candidates can
   legally use that format.

## Tests To Run After Changes

Fast loop:

```bash
python3 -m pytest -q \
  tests/test_schema_validation.py \
  tests/test_allocator_prune_candidates.py \
  tests/test_allocator_sibling_aggregation.py \
  tests/test_expert_saliency_observer.py \
  tests/test_incremental_probe.py \
  tests/test_exporter_prune.py
```

Full local suite:

```bash
python3 -m pytest -q
pytest -q
```

The plain `pytest` run matters because this checkout may be used from a user
environment whose `pytest` executable points at another virtualenv. The repo's
`pytest.ini` pins `pythonpath = .` so collection works there too.

## Known Limits

- The objective is still a proxy. It needs empirical validation at multiple
  budgets against BF16 and a strong uniform-quant baseline.
- Projection-granularity pruning uses consensus intersection across
  gate/up/down projections. This under-prunes relative to individual
  projection candidates. Use packed global-ratio pruning for serveable uniform
  expert-count artifacts.
- Export tests cover helper-level prune plumbing. A tiny synthetic MoE
  end-to-end export/load test would be the next high-value addition.
