# #2905 O5/O6 explicit-input driver — strengthened handoff, not a campaign

## Status

`experiments/qualified_w2_replay_2905.py` remains a fail-closed adapter to the
`prismaquant.interpolation_regret.replay_three_arms`, added in this work.
Existing allocator and acceptance-test files are unchanged. The portable seam now exists
and its eight CPU acceptance tests pass in the combined run below. Earlier notes
that the module was absent describe a superseded check, not the current state.

**No W2 qualification, completed O5/O6 campaign, served-quality result, or successful
real-input driver run is claimed.** Successful output retains
`qualification_claim: false`. Test prices and manifests are explicitly synthetic.
Hashes make decisions reproducible/tamper-evident relative to the input digest;
they neither authenticate a producer nor prove that its measurements are true.

## Production API evidence

Inspected project instructions and production implementations:

* `allocator_candidates.build_candidates` returns `CandidateBuild`. The driver
  uses production runtime/shape/TP/D41 gates, explicit serving contexts, no Fisher
  pricing, a retained runtime frontier, and requires every supplied row to survive
  admission. A rejected qualified row refuses the entire handoff; it is not silently
  downgraded or reclassified.
* `complete_scientific_price` / `qualified_cost_scope` check producer-qualified
  scientific prices/anchors and bind unit, format and currency. Legacy interpolation
  with a locally asserted `predicted_dloss` is insufficient. Each supplied row is
  additionally pinned by the qualification roster's canonical row digest.
* `rung_allowability.load_rung_allowability` uses scoped publications, pinned
  contract family rules and explicit kernel builds; unscoped boolean publications
  are not accepted. Permission to run is distinct from quality qualification.
* `tessera_campaign._measured_projected_units` derives wire-backed measured-only
  units from `_tessera_expert_projection.population.stacks` / `present_expert_ids`.
  `tessera_expert_projection.carried_expert_projection` validates carried
  projections. That campaign helper is private and depends on a bound campaign;
  this adapter does not instantiate a fake campaign or invent its projection.
* Instead, `context_by_unit_from_stats` supplies production profile-checked
  topology. The adapter conservatively marks **every routed-MoE unit** measured-only,
  unioned with explicit operator additions. Thus an empty supplied list cannot
  re-enable interpolation on wire-backed routed experts. This is intentionally
  stricter than the exact campaign projection and can exclude otherwise legal
  non-wire routed interpolation or refuse a degenerate replay. It is not a claim
  of exact wire-population reconstruction. Exact projection integration remains
  future work if that less-conservative behavior is needed.
* The portable seam filters arms before packed/fused production aggregation,
  expands assignments, accounts source-bound integer bytes and supplies common
  reference regret. Tests cover this seam; the adapter does not duplicate its DP.
* Existing `allocator.AllocationResult.to_layer_config` maps assignments using
  `format_registry.get_format(fmt).autoround_config()`. The adapter uses that same
  registry configuration generation, without constructing a fake AllocationResult.

## Explicit v2 input contract

```sh
python -m experiments.qualified_w2_replay_2905 --inputs /operator/inputs.json \
  --output /operator/new-report.json
```

Paths are operator configuration, not deployment defaults. Relative artifact and
allowability paths resolve against the input file. `--output` is optional and
**exclusive-create**: an existing file is refused. The complete JSON report also
goes to stdout. No report is produced when validation or replay fails.

Input schema is `prismaquant.o5_explicit_w2_inputs.v2` (v1 is refused because it
lacks mandatory qualification/lineage/accounting bindings). Every artifact entry
is `{ "path": "...", "sha256": "<raw-file-sha256>" }`. Mandatory entries:

* Existing `stats`, `costs`, `model_config`, `source_manifest`, `kernel_builds`.
* `source_identity`, `probe_identity`, `calibration_identity`, `teacher_identity`:
  actual producer identity JSON documents, hashed as bytes, not bare unverifiable
  names. Their content is retained by reference/digest, not independently audited
  or schema-interpreted here. They must describe the actual producer lineage;
  row scope/anchor qualification still comes from the production validators.
* `measured_overhead`: schema `prismaquant.measured_overhead.v1`, `components`
  mapping named components to nonnegative integer byte counts, and nonblank
  `measurement_reference`. Empty components explicitly mean zero. Count only
  measured non-tensor overhead **not already represented** by source spans or
  format shape accounting. The adapter cannot independently detect double counts
  or verify the referenced measurement. The measured total and reserve are passed
  separately to the portable API and reported separately; both count in all-in
  bytes in every arm.
* `qualification`: immutable producer decision roster, detailed below.

Other mandatory fields retain their existing meaning: `profile` (must equal
production profile detection), explicit `target_profile` (or null), `currency`,
unique Tessera `formats`, positive integer `tensor_parallel`, production
`serving_context`, `allowability_root`, `bit_precision`, nonnegative integer
`overhead_reserve_bytes`, and `measured_only_units` (unique known units; additions
only). Kernel builds must match each family publication. Source manifest contains
`bytes`, `spans`, `total_bytes`; byte and span coverage must agree. Stats and cost
unit sets must match, with at least one qualified row per unit.

Fixed targets remain 3.5/4.5/6.0; no alternate solver or substitute W2 is created.

### Qualification roster

Schema: `prismaquant.w2_qualification_roster.v1`.

* `artifact_sha256`: exact mapping of all loaded artifact names to raw SHA-256,
  excluding `qualification` itself and `inputs` (avoids digest cycles).
* `replay_config_sha256`: canonical JSON digest of the entire input object except
  its `qualification` entry. It binds formats, currency, precision, TP, context,
  reserve, paths and all artifact references. Produce this **before** adding the
  qualification reference, then hash the roster file, then finalize inputs.
* `identities`: exactly `source`, `probe`, `calibration`, `teacher`, `scope`.
  The first four equal the corresponding identity document's raw SHA-256. `scope`
  is the canonical JSON SHA-256 of the sorted mutable unit names. Full stats,
  model config, row scopes and source spans are separately artifact-bound.
* `cells`: exactly one entry for every mutable unit x declared format. Entry keys
  include `unit`, `format`, `status`. A `qualified` cell requires `row_sha256` equal
  to the canonical JSON digest of the supplied production cost row. A `refused`
  cell requires a nonblank `reason` and **no supplied cost row**. Missing cells,
  duplicates, changed rows, extra cost coverage, or unknown statuses refuse the
  handoff. Qualified coverage must exactly equal supplied cost coverage.

Canonical digests use Python JSON `sort_keys=True, separators=(",", ":"),
allow_nan=False`, UTF-8, SHA-256 (the driver's `_sha`). Raw artifact hashes are not
canonical JSON hashes. Duplicate JSON keys and nonfinite constants are rejected.
Refusal pairs become a `frozenset` passed to replay; the hashed full roster,
including reasons, is also included in the report. A producer must actually
publish the decisions: the driver never synthesizes missing refusal records.

## Output and precise limitations

Result schema is `prismaquant.o5_explicit_w2_result.v2`. Alongside replay results
and artifact/input hashes it contains the full bound roster, derived measured-only
roster/policy, measured overhead provenance and per-successful-arm layer configs.
Configs are keyed by target/arm, exclude infeasible arms, and come from the existing
format registry. This is **assignment configuration output only**, not an exported
checkpoint: conversion, export preflight, wire source manifests, group metadata,
calibration assets and serving launch remain separate production obligations.
No export or serving workload was run here.

Real-input end-to-end driver execution still requires the actual qualified W2
bundle, matching contract/D41 publications, builds, profile/TP/context, lineage
identity documents, source-byte manifest and measured-overhead evidence. None was
fabricated. We have not shown that a supplied real W2 grid passes admission or that
canonical chord labels from that grid yield a nondegenerate replay. The actual
qualified rows must be checked by a retained real-input run before claiming O5/O6
completion. Conservative routed filtering may require revisiting exact projection
integration for such a run. The full repository/GPU suites were not run.

## Portable replay and accounting convention

The frozen denominator is the sum of `n_params` over the explicitly supplied
mutable leaf scope, recorded as `frozen_mutable_n_params`. Requested budgets are
exactly `floor(Decimal(target_bpp) * N / 8)` for 3.5, 4.5 and 6.0. The real
campaign's numeric denominator **cannot yet be declared** without its frozen W2
scope. No synthetic denominator is substituted for it.

Each arm independently uses production packed-then-fused grouping. Fused folding
calls the existing contract-licensed `tessera_group_composites` internally; this
adapter does not create cross-products bypassing that gate. Only surviving legal
atomic options count as eligible interpolation. Choices expand back to leaves
before scoring, byte accounting and configuration generation.

`assignment_artifact_bytes` includes selected payload serialization and retained
source spans. Measured non-tensor overhead and reserve are separate quantities,
both included in the all-in ceiling. The reserve is conservative accounting, not
proof of priced-versus-written physical-file equality. Exact measured-wire/export
preflight is still required downstream; interpolation remains a proposal.

The existing production `solve_allocation` receives conservative integer-byte
delta bins relative to each arm's exact group floor. Quantum is
`max(1, ceil(bit_precision*N/8))` bytes. Options round **up**, the available budget
rounds **down**, and the solver's extra bin is explicitly removed. This preserves
the ceiling, not a claim of globally exact knapsack optimality at coarse precision.
The complete expanded assignment is independently re-accounted and checked against
the exact ceiling and additive option prices. `fullm` receives `meas`'s achieved
integer bytes directly, never achieved bpp. Infeasible targets retain their requested
budget, menu floor and gap, with no fabricated assignment or fullm comparator.

Reach is `meas loss / full loss` at the common nominal target. Common-budget
comparison is `meas loss / fullm loss` at meas's achieved all-in ceiling. Every
ratio states arm names, numeric numerator/denominator and declared currency;
a zero denominator produces JSON null (`undefined_null`), never a fabricated 1.
Both comparisons report signed achieved-byte gaps and equal-size status. A nonzero
gap is not a ranking-only/equal-size claim. All scores use immutable leaf Candidate
reference prices; hashes include eligible reference rows and prices, independent
of input ordering.

## Missing-input disposition (O6 and real-input O5 validation)

No immutable qualified W2 handoff was supplied to this run. The inspected
`/mnt/shared/tessera-measurements/pq275-2026-09-06/campaign-01/` listing still has
`cost_measured_from_journal.pkl`, not the required qualified full payload. Its
README lines 229–249 explicitly describe withdrawal and no interpolated table,
and lines 299–301 describe scope refusal/fallback. It is not a substitute result.
The research-identified v7 table covers three dense units only; it is not accepted
as the required qualified dense-plus-packed campaign. We did not claim an
exhaustive discovery of all fleet artifacts.

Existing owners #2902/#2903 must deliver the immutable qualified full payload,
complete qualification/refusal roster, matching frozen scope/probe/calibration/
source-byte identities, runtime contract/D41 publications and builds, and relevant
overhead accounting. Until then there is **no real three-budget campaign table,
real denominator, real regret result or validated real export configuration**.
O6 remains blocked on that handoff; this change prepares the portable capability,
input contract, machine-readable output and registry configurations without
inventing scientific qualification. No GPU acquisition, W4, joint-currency audit
or served run was performed.

## Final integrated verification

```sh
python -m pytest tests/test_interpolation_regret_replay_2905.py experiments/test_qualified_w2_driver_2905.py tests/test_allocator_byte_budget_selection.py tests/test_allocator_packed_group_menu_reduction.py tests/test_allocator_packed_group_units.py tests/test_fused_fold_enumeration_2287.py tests/test_incomplete_fused_group.py tests/test_incomplete_fused_ignore.py -q --basetemp="$TMPDIR/pytest"
```

Observed **83 passed, 15 subtests passed**, 14 upstream deprecation warnings;
stdout verified with `read_test_artifact`. This includes all eight immutable
acceptance cases, the explicit-input boundary tests and existing byte/grouping
regressions. No acceptance files changed. Full suite, GPU, real W2 replay and
export/serving validation were not run. Untracked new-file diffs were inspected;
`git diff --no-index --check` emitted no whitespace diagnostics (exit 1 denotes
the new-file difference).

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b730e1542aaae31fbd8c574706fed4d7c218684775d1630ef71de3e1e345c018","size":917}},"operation":"9475bde641cc1aee2de696d083ea0ce384f930ededdc5dfc8325fcf42c75a8e2","receipt":"631c51bdd769d74ad69b8e5ffe1c36c21c2eeacffe60925ca015fcc9a3ac8f35"}]

## Earlier development checks (all through run_tests)

Initial strengthened boundary run: **1 failed, 12 passed** because the synthetic
configuration test used unregistered lowercase `fp16`; corrected fixture to the
registered `BF16`, not application behavior.

After the initial passing combined run, bumping the input schema to v2 exposed
one stale v1 digest-mismatch fixture: **1 failed, 21 passed**. Updated the fixture
to v2 so it again reaches the intended digest guard. Final combined rerun:

```sh
python -m pytest -q experiments/test_qualified_w2_driver_2905.py \
  tests/test_interpolation_regret_replay_2905.py --basetemp="$TMPDIR/pytest"
```

**22 passed, 15 subtests passed**, 14 upstream Torch/Python deprecation warnings.
Driver tests cover roster completeness/immutability/refusal resurrection, changed
row/config/artifact/lineage bindings, topology-based restrictions even with an empty
operator list, integer measured overhead, real portable-seam separate-overhead
accounting on synthetic prices, registry configs, and exclusive report writing.
This is feasible CPU/API evidence, not W2 evidence.

Final test evidence:

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e3eccf29bbf2891aeacf1320c41fe4461092afa96f30cf4220046bd69093c7a5","size":826}},"operation":"7f0465d9304818a9da65b052d241acd49925e7922b82811f8610f3bf55c283f3","receipt":"cf631ea1839e306eb03ad60758f44b55eacb0a20f66aef7ca4a91438db5ca375"}]

Schema-fixture failure evidence:

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"a25f0e676bd48276443923b1f148920a22923aa7bb624969204fda1f1d2ba191","size":2204}},"operation":"e965ac2725d0755478ab7032ee80ef14446a6a92a1aed54255419b17098ea5bd","receipt":"34717aad761592dc6063c46f185e97290d1fac96e23f813c6fa881eddccac2af"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"446de2f3589edefc35bb045131221703c89d04127c1cc6786641837848524c74","size":4659}},"operation":"e4048fa307000647a96677dbedd3ae40ea1dc3e364d14dc31787aa7b7591653f","receipt":"a189fa3667ef69a2d42fcfc3e96e6ea0aee22044286569a49070c04d3a67186f"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"921a4c3df55e6d9baa3382d6a70311b398e9737308c293794f441bd0f59b6967","size":826}},"operation":"992911aee5d3973eb026321fc6c3863d9c008b0bf7ecca1d4e1c0911bb34b4c1","receipt":"6829385189adad1db2d6362a3d1ace75dbe7ef65013746738c18fea3853ca71c"}]
