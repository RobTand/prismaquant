# Pluggable Refactor Contract

This refactor separates four concerns that were previously easy to couple:

1. **Model structure** lives in `prismaquant/model_profiles/specs/*.json`.
   These specs describe live-to-recipe naming, source/vLLM naming,
   fused-sibling groups, packed experts, packed-expert projection splits,
   legacy packed-expert module class names, passthrough prefixes, and pinned
   tensors. They also describe text-only staging keys, shard regex prefixes,
   MTP sidecar Linears, the default serving profile, production-speed kernel
   package requirements, and probe hook skip class names for the model family.
   `ModelProfile` still owns executable behavior, but low-risk profile answers
   now read these specs.

2. **Serving/runtime constraints** live in
   `prismaquant/serving_profile_specs/*.json`. These specs describe backend
   format menus, static kernel shape rules such as MXFP8/NVFP4 alignment, and
   optional runtime validator probes such as FlashInfer's MXFP8 problem-size
   check. Shape and runtime-validator rules may also include name conditions
   so a backend constraint can target experts, dense Linears, or a module
   family without adding Python branches. Runtime validators are dotted
   callables declared in the JSON profile; the allocator asks
   `serving_profiles.py` and does not call backend validators directly.
   Serving profiles can also carry runtime package pins. The vLLM packed-MoE
   profile owns the FlashInfer package version used by
   `validate_native_export.py`, while the CLI still accepts an override for
   ad hoc container testing.

   Entry points resolve the target profile as: explicit CLI/API override,
   then the detected model profile's configured default, then `research`.
   That same resolver is used by allocator candidate construction, decision
   units, export audits, and KL/L3 format selection.

   Layer-config parsing is runtime-neutral. Research formats such as
   `MXFP8_E5M2` and `FP8_E5M2` may be parsed and carried through the research
   stack; production legality is enforced later by the selected serving
   profile and exporter runtime-coercion gate.
   `vllm_packed_moe` is the generic packed-MoE vLLM target; the older
   `vllm_qwen3_5_packed_moe` id remains as a compatibility alias only.

3. **Pipeline artifacts and gates** live in `prismaquant/pipeline.py`.
   `PipelineSpec` describes stages, typed artifacts, metric gates, and cache
   ownership. The current production flow is exposed by
   `default_production_pipeline_spec()` as a contract first. `run-pipeline.sh`
   writes the configured contract to
   `${WORK_DIR}/artifacts/pipeline_spec.json` before launching the GPU-bound
   stages, so each run records the render mechanisms, target profile, formats,
   and cache policy it executed.

4. **Optimization units** come from `ModelGraph.optimization_units()`.
   This is the typed replacement for scattered coupling rules: individual
   tensors, fused-sibling groups, and packed expert groups are explicit units
   with constraints.

## Cache Rule

Rendered weights remain owned by `ProductionWeightCache`. Activation replay
remains owned by `PerturbedActivationCache` or the established streaming
activation cache. Pipeline validation rejects alternate owners for those
resources so new plugins cannot accidentally introduce a parallel cache.

## Adding a Component

Add model-specific naming/decomposition in a model-structure spec. Add
backend legality in a serving-profile spec. Add local numerical transforms as
render mechanisms using `render_score.py` and gate them with `MetricGateSpec`.
Only then wire the component into production execution, reusing
`ProductionWeightCache` / `PerturbedActivationCache`.

When adding a multimodal or MoE model, keep source-checkpoint naming and export
emission naming separate. `ModelProfile.source_tensor_name()` is used for
source lookup and shape audits; `ModelProfile.export_tensor_name()` is used for
keys PrismaQuant writes. They are usually identical, but Gemma 4 keeps export
expert keys in recipe form because vLLM performs its own `.experts` to
`.moe.experts` remap during loading.

Packed-expert `split_for_formats` is also part of the model-structure spec.
Qwen MoE profiles set it to `["*"]` so mixed-format artifacts never emit a
blend of unsplit BF16 packed tensors and per-expert quantized tensors, which
trips vLLM's packed-expert loader state.

Pinned names are likewise profile data. Export and production-cache CLI
defaults read `ModelProfile.pinned_names()` instead of assuming `lm_head`,
which keeps DeepSeek-style `head` and future runtime head constraints
configured at the model boundary.

Text-only staging and incremental probe shard schedules are profile data as
well. Qwen3.5/3.6, Gemma 4, and DeepSeek declare their strip keys, visual
layer prefixes, MTP prefixes, MTP sidecar names such as `mtp.fc`, and head
name in JSON. The shard scheduler resolves the model profile from
`config.json`, so cache/prefetch scheduling no longer assumes
`model.layers`, `mtp.layers`, `model.visual.blocks`, or `lm_head`.

Architecture-specific fast-kernel requirements are profile data too. The
Qwen3.5/3.6 specs declare the `causal_conv1d` and `fla` imports required to
avoid the slow linear-attention fallback; `_fast_kernel_guard` **(walled
2026-07-30, `archive/orphans_2026-07-30/` — it has had no caller since
2026-05-15; see ARCHITECTURE §12 D28)** asked the profile
instead of parsing model path substrings, with a string fallback only for
remote IDs that do not have a local `config.json`.

`python -m prismaquant.model_profiles.validate --model <path>` now checks the
resolved serving profile as well as the model profile: the target profile must
load, and every configured runtime-validator callable must be importable before
a long GPU/export run starts.

## Shelved Cross-Layer Research

CLADO and SMRF are archived, not registered pipeline components. The current
ports and run notes live under `archive/cross_layer_2026-05-09/`:

- `prismaquant/research_components/block_clado*.py`
- `prismaquant/research_components/smrf*.py`
- `tests/test_block_clado_runtime.py`
- `tests/test_smrf_runtime.py`
- `docs/qwen3_4b_clado_assignment_optimization.md`
- `docs/qwen3_4b_smrf_optimization.md`
- `docs/qwen3_27b_smrf_validation.md`

`prismaquant.pipeline` has no component-composition or stage-registry surface:
the empty `PipelineComponentSpec` registry and its composition API were deleted
on 2026-09-19 (`astra/simplify-redundant-machinery-20260919`) once it was clear
that no live component existed and the only callers were synthetic tests.
Reviving a shelved method is a new research effort on an explicit opt-in
research path, not a registry toggle: port it from archive, keep rendered weights in
`ProductionWeightCache`, keep activation replay in `PerturbedActivationCache`,
and clear the measured KL/PPL/log-likelihood/ToolEvalBench gate before any
candidate can feed export.
