# #2974 research and acceptance evidence

## Decision and boundaries

The cause is a missing in-repository scientific consumer, not an absent profile adapter. `prismaquant/pact_frontier_profile.py` and the current ModelProfile PACT methods already own bands, cohort, TP and hidden layout. Main has no `experiments/pact_replay` directory. Import the actual W1-approved harness there and thread those owners through its existing entry functions; do not implement a small replacement that only satisfies these tests. Verdict: **as asked**.

These changes contain only acceptance tests, a synthetic fixture and this evidence note. They do not import or implement the harness. The existing adapter regression file is unchanged. Tests deliberately assert the intended source deliverable exists before delayed imports, so missing harness files produce assertion failures, not collection/import errors.

No native admission, campaign execution, quality, performance, GPU arithmetic, export or serving qualification is claimed. Metadata setup is not the same as `actual_cpu` or a complete replay. The setup fixtures intentionally have no source/render tensor bodies: that API prepares metadata and token inputs only. Do not let these fixtures authorize a missing-source fallback at a later consuming API.

## Existing work and unresolved handoff

Supplied open issues already allocate W1 baseline/caller/input pinning to #2973 and exhaustive caller migration/public-entry equivalence to #2975. #3008 tracks the existing producer handoffs, including #2973–#2975 and #2960–#2961. Do not duplicate those packages, change active campaign copies, or treat historical source reports as a new migration approval. Rob's #2705 instructions concern already-shipped old PrismaBuild API tests; this package must not reopen them.

Research recovered the historical source and inventory, but **did not establish delivery of an owner-approved W1 migration manifest**. This remains an existing #2973 dependency, not a reason to manufacture a baseline or split W2. Do not assign final per-file imports until the W1 manifest, canonical oracle and source selection are available. Tests must not read historical Git objects: admitted checkouts have no pre-base history.

### Historical inventory, not fresh W1 authorization

Recovered immutable revision: `c9260081766ec8a47e7a0f357432833a4d3b1cb6`.

Its `experiments/pact_replay/IMPORT_PROVENANCE.json` hashes to `3e50ec0afe04f04ea6d1db90a035aa80ab42d8aff824740665446a839ce76da2`; all 31 listed imported entries were readable and matched during source research. This means 30 Python files plus API_CONTRACT.json:

```
accepted/__init__.py accepted/g3_lib.py accepted/g3_residency.py
assemble_bands.py band_continuation.py band_gain_reduce.py band_replay.py
bounded_profile.py build_manifest.py calibrate_sample.py canonical_weight.py
chord_diagnostics.py energy_consumer.py energy_inputs.py energy_math.py
footprint.py frontier_replay.py menu_contract.py mixed_injection.py
mixed_manifest.py multi_stream_replay.py panel_stack.py plan_panels.py
profile_entry.py reduce_prices.py run_a958_proofs.py smoke_integration.py
stage1a_energy.py strict_leases.py tail_replay.py API_CONTRACT.json
```

The directory also contains IMPORT_PROVENANCE.json, DEPENDENCY_MANIFEST.json and D38_IMPORTS.json. The dependency manifest identifies 94 inferred external source files; 7 G3 and 58 PQ-owner files were readable and matched, while reads of the 29 pinned Tessera sources were denied. This is not proof of a completely independently verified closure or of all callers. Do not vendor a historical dependency tree over current main's installed contracts.

Concrete historical consumer limitations:

- `multi_stream_replay.resolve_cli_run`: graph extent and teacher stop 45, fixed prefixed cohort, 3500 bounded deadline ceiling after the canonical correction.
- `stage1a_energy.setup`: RAW/PREFIXED literals, fixed draw geometry, cohort/sample/prefix checks and source-layer bounds.
- `energy_inputs.ExistingRenders` / `menu_contract.extend_rows`: fixed role/layer assumptions and manifest population handling.
- `frontier_replay.load_frontier`: 45-layer range and only initial hidden sample/token axes, not complete profile layout.
- `energy_consumer.SequenceEnergy.consume`: default down-projection TP=2 and original_tokens=512 literals.
- `build_manifest`: fixed layer, draw and first-sample assumptions that must follow resolved configuration too.

Existing ownership to reuse includes ModelProfile/profile_from_config/detect_profile, pact_frontier_profile bridge, model structure and routed projection owners, ProductionWeightCache, PerturbedActivationCache/exact-entry owner, source streaming/prefetch, ReadPlan and strict resident-input leases. Do not add a model-name switch, parallel registry/cache/preload or relaxed reader.

### Deadline generations must not be conflated

The approved correction commit is `7672bace107fe09b9907747a9a70ba758af59b32`. The recovered corrected `multi_stream_replay.py` at c9260081 hashes to `c661efc8228545b91e97b6c5d50bf5d35f922a1ca47e3b0e00ff081c142a1eaf`. It accepts an explicit normal-CLI deadline of 3500; its parser default remains 1700. The issue forbids restoring the old **1700 ceiling**, not evidence-based preservation of the historical default. The tests assert 3500 acceptance and invalid bound refusals, not a new default.

The shared `eng-pact-band-20261009/pq2482-canonical-a8/multi_stream_replay.py` still has the historical ceiling and hashes to `6e6f2101ed991a468f5aaaf1be7258a463c6818a52b767c6badf6108f2eb25ec`. Do not silently use it as the corrected entry.

A separate later stall generation, `ab6bfb843b85ac61b96693d13bc7e9463f004046`, includes replay_progress.py and records different stall/backstop control. It is evidence for reconciling the existing W1 handoff, not authority to select another duration policy in W2. Preserve the W1-selected canonical stall/progress/control implementation; these CPU tests do not independently prove that supervision.

## Acceptance API and fixture choices

`tests/test_pact_harness_consumer_2974.py` enters the actual imported functions:

- `stage1a_energy.setup(args)` -> existing `(renders, cohort, batches, rows)` tuple.
- `multi_stream_replay.cli_parser()` and `resolve_cli_run(args, receipt, raw, roster)` -> existing run mapping, augmented with JSON-safe `bands`, `cohort`, `hidden_layout` resolved metadata.
- `frontier_replay.load_frontier(args, cohort, inputs, cpu_proof=True)` with actual exact-cache reference descriptors.
- `energy_consumer.SequenceEnergy` and `LiveRows`, and `band_continuation.bind_teacher_content`.

The local API extensions specified by the tests are optional `profile=None, config=None` keyword arguments on setup, CLI resolution and frontier loading; SequenceEnergy accepts the resolved `profile`. Explicit call configuration wins over the profile's captured config and disk config, using existing profile lookup precedence (including text_config). Without overrides, source-root config selects the existing profile. Wire these same APIs from normal main/setup/replay paths; a disconnected exported test resolver is not sufficient.

Fixtures are synthetic metadata and small tensors, **not W1's canonical source/equivalence oracle**. GLM retains its real 45-layer bands, prefixed64 sample/token contract, vocabulary and four-stream layout, but has explicitly configured 8-wide fixture geometry. Qwen uses the existing qwen3 profile, 12 layers, two experts, hidden width 8, vocabulary 32, two sequences of four original tokens, explicit prefix IDs and complete PACT cohort scope. No production-sized weight tensors are created. GLM's frozen draw is only a 512x512 int64 token fixture. Hidden and energy tensors are tiny and written under pytest's per-run scratch.

The fixture metadata was independently checked against the current bridge: both cohort mappings and band lists match; layouts resolve to GLM `(4,8)` and Qwen `(1,8)`. That successful fixture-validation probe is not a passing acceptance claim for the absent harness.

## Observed checks

Final command:

```
python -m pytest tests/test_pact_harness_consumer_2974.py tests/test_pact_frontier_profile_adapter.py -q --basetemp="$TMPDIR/pytest"
```

Observed: **34 acceptance failures, 26 unchanged adapter regression passes**, exit 1. The retained stdout was read back. All new cases currently fail on the deliberate missing-harness assertion before consumer execution. This proves the unmet import deliverable; it does not establish additional runtime failures deeper in an implementation that does not yet exist. After importing, the same cases will exercise cohort/roster/dimension resolution, absent/invalid scope, configuration precedence, deterministic CLI state, hidden layouts, TP wiring, prefix exclusion, original-token energy normalization, teacher-content refusals and deadline bounds. A reviewer must also inspect the complete source import, canonical oracle preservation, main-path wiring and the preserved cache/control/scientific contracts.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"59698e6a9ed5b1940bf415ddf424e2c6897d5d7097389ea9483cd6f307455b70","size":50912}},"operation":"0fa7ffbebba1b1da70df2a0b28bb7761eeb6adc72202406c7b1944f77775a2bc","receipt":"55a168a51abcfbbbd9044bcc660a6e24495c842e4d359e04c5545e5fddd84b0a"}]

Fixture-validation probe observed exit 0, printing matching GLM and Qwen cohorts/bands and their resolved layouts:

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"07d3b6bedf1f5878bd4e7d586f4542fd0646c2327d737d84830080571c900eda","size":151}},"operation":"3c68d9b7c8179d6a3faa6a9b14901de45349119b9d24d6ff1c98819aabddff07","receipt":"f553819e8039ea9d93efc9bf549ce67e0e1e19fb17e1d67692ce5614d1e58b8b"}]
