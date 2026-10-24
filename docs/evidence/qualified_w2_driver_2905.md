# #2905: production-menu replay and qualified-input boundary

## Status and reviewer repairs

`prismaquant.interpolation_regret.replay_three_arms` is a portable, surrogate-only
adapter, not W2 qualification or an export/serving validator. The explicit-input
driver is `experiments/qualified_w2_replay_2905.py`. Existing allocator, digest-owner,
duplication-guard and acceptance-test files are unchanged.

This repair addresses the production defects found in review:

* Canonical chords are interpolation. Filtering recognizes the production
  `CANONICAL_CHORD_SOURCE` through both Candidate activation pricing and effective
  row `cost_source`, in addition to legacy band/output-MSE interpolation. Thus
  chords count toward nondegeneracy and cannot enter meas or measured-only units.
* Replay consumes **effective `built.cost_rows`**, not the original supplied
  fractional values. Immutable byte snapshots separate supplied, build and replay
  copies. Explicit two-stage provenance binds replacements to roster-qualified
  anchors; a changed non-chord row refuses.
* Identity documents are no longer merely hashed/discarded. Actual supplied and
  effective scientific coordinates must match their contents, and must align
  across units under production scope/probe owners.
* The local `_sha` recipe is removed. Core candidate/assignment identities and
  driver canonical digests use the existing `DIRECT_ASCII_STRICT` owner profile;
  raw artifact/code digests use `bytes_sha256hex`. This preserves the former
  compact sorted JSON, ASCII escaping, strict finite numbers, UTF-8 and SHA-256
  byte recipe. No digest owner or guard baseline was extended.

**No real-input driver success, qualified W2, completed campaign, validated export
or serving-quality result is claimed.** Output retains `qualification_claim:
false`. Hashes bind inputs; they neither authenticate a producer nor prove the
truth of measurements. The numeric campaign denominator is still unavailable.

## Production mechanisms and restrictions

The driver uses `build_candidates`, producer-qualified scientific pricing,
`qualified_cost_scope`, scoped `load_rung_allowability`, matching kernel builds and
the configured runtime contract. It requires every supplied qualified cell to
survive production admission; no scope refusal is silently downgraded or retried.
Candidate and effective-row grids must equal the supplied qualified grid.

The portable seam filters before production packed-then-fused grouping, whose
fused folding invokes contract-licensed `tessera_group_composites`. It expands all
synthetic group names before accounting/scoring. Only surviving legal atomic
options count toward eligible interpolation. There is no parallel knapsack or
new cross-product admission mechanism: selection uses `solve_allocation`.

`context_by_unit_from_stats` supplies profile-checked topology. The driver
conservatively marks **all routed-MoE units** measured-only, unioned with explicit
operator additions. An empty operator list cannot enable interpolation on
wire-backed routed experts. This is stricter than the exact campaign projection,
not a claim to reconstruct that projection; it can exclude legal non-wire routed
interpolation or refuse a degenerate replay. Exact measured-wire/export preflight
is still a downstream requirement. No interpolation proposal is called a validated
wire or checkpoint.

Successful layer configurations use the existing format registry's
`autoround_config()`, as `AllocationResult.to_layer_config` does. They are assignment
configurations only: conversion, required calibration/assets, group metadata,
export validation and serving launch remain separate obligations.

## Explicit v2 handoff

```sh
python -m experiments.qualified_w2_replay_2905 --inputs /operator/inputs.json \
  --output /operator/new-report.json
```

Paths are configuration, not deployment defaults. Relative artifact and
allowability paths resolve against the input file. Optional `--output` is
exclusive-create; an existing file refuses. A complete JSON report also goes to
stdout. Validation/replay refusal produces no report.

Input schema: `prismaquant.o5_explicit_w2_inputs.v2`. Artifact entries are
`{ "path": "...", "sha256": "<raw-file-sha256>" }`. Required artifacts:

* `stats`, `costs`, `model_config`, `source_manifest`, `kernel_builds`.
* `source_identity`: `source_weights` maps exactly the mutable units to actual
  `quality_scope.source_weight` objects. Each weight identity must match its row,
  and its shape must match production stats geometry. Unit weights may differ.
* `teacher_identity`: full `prismaquant.streamed_model.identity.v1`, checked by
  `validate_streamed_model_identity`. Its content digest must match both the row
  teacher coordinate and the owner-validated actual probe source-model identity.
* `probe_identity`: production projection
  `{probe_identity, probe_identity_sha256, probe_ids}`. The nested object is the
  actual scientific identity, not this document's path/hash. Its digest uses
  `joint_identity_sha256`; sample/probe alignment uses `require_probe_alignment`,
  including the owner's D32 scientific/provenance policy.
* `calibration_identity`: `calibration_sha256`, `calibration_shape`, `token_scope`,
  `temperature`. These must match actual row calibration/window and actual probe
  calibration. A content digest is not the raw-file digest of this document.
* `measured_overhead`: schema `prismaquant.measured_overhead.v1`, named nonnegative
  integer-byte `components`, and nonblank `measurement_reference`. Empty components
  explicitly mean zero. Do not count tensor spans/serialization twice. The driver
  cannot independently verify the referenced measurement or detect double counts.
* `qualification`: the complete producer decision roster described below.

Mandatory configuration includes detected `profile`, explicit `target_profile`
(or null), one `currency`, unique Tessera `formats`, positive `tensor_parallel`,
production `serving_context`, `allowability_root`, positive `bit_precision`,
nonnegative integer `overhead_reserve_bytes`, and `measured_only_units` (unique known
units; additions only). Source manifest has `bytes`, `spans`, `total_bytes`; byte
and span coverage must agree. Stats and cost units must match.

Supplied **and effective** rows must match the documents' actual coordinates.
Across units, `require_quality_scope_alignment(..., same_unit=False)` and production
probe alignment require common scientific context (including family, currency,
calibration, teacher, window, objective and samples). Per-unit weight/shape or
activation coordinates are not spuriously required to be identical. The adapter
neither remeasures nor authenticates these artifacts. Duplicate JSON keys,
nonfinite constants and overflowing JSON floats refuse.

### Qualification and effective-row provenance

Roster schema: `prismaquant.w2_qualification_roster.v1`.

* `artifact_sha256` exactly maps artifact names to raw digests, excluding the
  qualification file and input file to avoid cycles.
* `replay_config_sha256` is the canonical digest of inputs excluding the
  qualification entry. It binds formats, currency, precision, TP/context, reserve,
  paths and artifact references. Compute this before finalizing the roster.
* `identities` has exactly `source`, `probe`, `calibration`, `teacher`, `scope`.
  The first four are identity-document file digests; scope is the canonical digest
  of sorted mutable unit names. These bindings supplement, not replace, the
  scientific-coordinate checks above.
* `cells` covers every mutable unit × declared format exactly once. Qualified
  cells have a matching canonical `row_sha256`; refused cells have a nonblank
  reason and no supplied row. Missing/duplicate cells, changed rows, extra cost
  coverage and unknown statuses refuse. Refusals become an immutable `frozenset`.

The roster certifies supplied rows, not replacements hidden under the same keys.
Production may rebuild a fractional canonical chord. `effective_row_provenance`
(schema `prismaquant.o5_effective_rows.v1`) records supplied/effective complete-table
hashes and, per cell, supplied/effective row hashes, `replaced`, `effective_origin`
(`supplied` or `production_canonical_chord`), and anchor format/supplied-row hashes.
Every embedded effective anchor must be the exact supplied roster-qualified row.
Replacements must pass production scientific/chord validators; non-chord changes
refuse. The supplied hash is never relabeled as an effective-row hash. This is an
explicit production derivation, not a claim the producer signed the replacement.

Result schema remains `prismaquant.o5_explicit_w2_result.v2`, with bound roster,
input/code/contract identities, effective-row provenance, measured-only policy,
overhead provenance, replay report and per-successful-arm configurations.

## Budget, accounting and interpretation

Frozen N is the sum of mutable leaf `n_params`, recorded as
`frozen_mutable_n_params`. The immutable targets are 3.5, 4.5 and 6.0, each with
`target_bytes=floor(Decimal(target_bpp)*N/8)`. No floor/reachable bpp replaces a
requested target. Infeasibility returns exact all-in menu floor/gap without a
fabricated assignment or fullm comparator.

`assignment_artifact_bytes` includes selected tensor payload serialization and
retained source spans. Measured non-tensor overhead and conservative reserve are
reported separately and both count in the ceiling. A reserve is not measured
physical-file equality. DP bins conservatively round up exact group byte deltas
from each arm's floor using quantum `max(1, ceil(bit_precision*N/8))`; available
bins round down and the production solver's extra bin is removed. This is not a
claim of exact knapsack optimality at coarse precision. Expanded picks are
independently accounted and checked against both additivity and the exact ceiling.

`fullm` receives meas's achieved **integer all-in bytes**, never reconstructed bpp.
Reach is `meas loss / full loss` at the common nominal target. Common-budget
comparison is `meas loss / fullm loss` at meas's achieved ceiling. Reports state
arm names, numeric numerator/denominator, currency, signed achieved-byte gaps and
equal-size status. Zero denominator yields JSON null (`undefined_null`), not 1.
Nonzero gaps are not ranking-only/equal-size claims. All scores use one immutable
leaf Candidate reference currency; eligible reference prices affect candidate
hashes, independent of input ordering. All conclusions remain surrogate-only
pending W4.

## Missing input: existing #3044 / #2902 / #2903

No immutable qualified W2 handoff was supplied for this repair. The earlier
campaign-01 inspection found the measured-only reconstruction and README's
explicit withdrawal/no-interpolation/scope-refusal notes. Research's v7 table
covered three dense units only. Neither is a substitute for the qualified
full dense-plus-packed payload. No exhaustive claim about all fleet artifacts
is made, and no real W2-loader round trip has been demonstrated.

The existing owners must supply the immutable payload, qualification/refusal
roster, matching frozen scope/probe/calibration/source-byte identities,
contract/D41 publications/builds and overhead accounting. Until then **there is no
real campaign denominator, three-budget campaign table, real regret result or
validated real export configuration**. O6 and real-input O5 validation remain
blocked through existing #3044, not a new duplicate issue. No GPU acquisition,
W4, joint-currency audit, export or served run was performed.

## Verification after repair

```sh
python -m pytest tests/test_interpolation_regret_canonical_2905.py tests/test_interpolation_regret_replay_2905.py experiments/test_qualified_w2_driver_2905.py tests/test_canonical_quality_quantity.py tests/test_canonical_consumer_join.py tests/test_rung_allowability.py tests/test_duplication_baseline.py tests/test_duplication_alias_scope_audit.py tests/test_allocator_byte_budget_selection.py tests/test_allocator_packed_group_menu_reduction.py tests/test_allocator_packed_group_units.py tests/test_fused_fold_enumeration_2287.py tests/test_incomplete_fused_group.py tests/test_incomplete_fused_ignore.py -q --basetemp="$TMPDIR/pytest"
```

Observed **263 passed, 33 subtests passed**, 14 upstream Torch/Python deprecation
warnings, exit 0. Retained stdout verified with `read_test_artifact`. This includes
all eight unchanged acceptance cases, both reported duplication guards, canonical
owner/consumer APIs and existing byte/grouping regressions. `git diff --check`
emitted no whitespace errors. The full repository suite was not run.

The new canonical replay regression reuses existing canonical-quality fixtures
and the real production build/effective-row path. It verifies both provenance
labels, meas and measured-only exclusions, refusal/nondegeneracy, and input reuse.
Its measured anchor is above all fixed budgets: it explicitly checks infeasibility
rather than inventing a successful fullm run. Actual selections remain covered by
the immutable portable tests. Driver regressions exercise real canonical rebuilding
with explicit test-only serving/admission/replay seams; they are boundary/API
evidence, not actual W2 qualification or a real-input driver success. Rehashed
unrelated documents, cross-unit scientific drift, altered anchors/non-chord rows,
source geometry mismatch, mutable provenance and admission-grid drift refuse.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"143c162f2b9b44343b9f328031e3f7130d0d4c2e80de6c6a416c16ff8d12ccbc","size":1140}},"operation":"fc1d402fc75a019f283f821263cef21a959fd0a7960911a3a8e0a942615bf915","receipt":"cd7874adeb00c9a2396d5f51f125f089540218b7e7ca4e198945625005a298a4"}]

### Rechecks and failed development assertions

Before repair, the two duplication guards reproduced the reported helper/raw
recipe failures (2 failed, 30 passed). The new canonical API regression reproduced
misclassification before repair (4 failed). An added fullm-success assertion later
failed because this production fixture's measured anchor cannot fit any nominal
budget (3 failed, 260 passed); the fixture now explicitly asserts infeasibility,
without weakening production budgets or behavior. Driver development also exposed
fixture protocol/schema/aliasing assumptions and one nonexistent test filename;
those iterations were not counted as successful checks. Final verification above
uses the corrected paths and current files. Detailed diagnostics remain in retained
computation outputs, not duplicated as source fixtures.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"84479fec47adfa1e4e7ab0e706e1739c75d4e18b3eeca189c6320707f3f24c5c","size":3424}},"operation":"06222fd875b22252b6cc73cba01f419cb2d86764db049328b5d56b3af4ae7349","receipt":"6f11dbd520558b95d66b09fec30d2a44b0537be81e8dbec76c25a6e5c55e879a"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"873778fc1f802f083230eaef5c53215a683b05ef6ee098710215e2bae011e31a","size":26226}},"operation":"e34ff03bec6f40ba68bacf91be67cc5de6754afafa2c3fe7d6f50aa524f455ac","receipt":"0088859aa977a00e3df7296473f824a56434a082d36e7be7b657665fe4051713"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"7ccd4b1d1b7315d513fd7eb0e434377e7cb41a0587cf548fc6396ad71ce30295","size":7403}},"operation":"611a40e2acb181304e9838dcd55b64faf1630934d3d4373dc9250c4120fb8f4d","receipt":"3ca996d84ccaff8f808b060c90323ebd9f25a234edb54ea998f581d7095ae17a"}]
