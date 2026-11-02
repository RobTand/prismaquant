# Historical graph-impact replay

The [proof bundle](graph-impact-replay-fb655551.tar.gz) contains the full report, all JSON files named below, both graph cores, the frozen manifests, native Bazel evidence, and the static replay driver. SHA-256: `9d55cfedab679c93891ee8761275637e48543465cfb3e570e1ce6ac12e23eef4`.


Frozen revised core: `fb6555514e2f43727ab73d056f82d8ad34ab64e3`.

The old core is the pre-review `72af52b5` implementation. Each policy uses
the same base/head graph and a baseline measured at that range’s base.
Changed test bodies are detected by content hashes. Baseline gain recording
is enabled so this replay judges regressions rather than baseline maintenance.

PrismaQuant uses the current generator’s evidence and its hosted gate population.
Revisions predating the hosted selector use all test files. Tessera uses its own
#1590 generator evidence at `23909243b94bb308dc19c80c9f91567412aaa855`,
including `src/tessera` module resolution and its conftest closure rule.
Historical application code is parsed, never executed. Pure parser work is
memoized; Tessera facts are invalidated when source bytes or file inventory change.

All base/head SHAs, full diagnostics under both policies, and every added edge
are retained in `replay/pq-results.json` and `replay/tessera-results.json` in the bundle.
The frozen 30 first-parent merge ranges are in the `*-ranges.json` manifests.
TL3 is excluded from this worker’s replay by the lead’s explicit correction.

## pq

Old failures: 16/30. Revised failures: 13/30.

| PR | Old failure reasons | Revised failure reasons |
|---|---|---|
| #2461 | pass | pass |
| #2490 | pass | pass |
| #2495 | new test_to_test_imports edge ×4; test reach budget | new test_to_test_imports edge ×4; existing source fanout grew; existing test reach grew |
| #2504 | new test_to_test_imports edge ×2; new conftest_source_imports edge; test reach budget ×2 | new test_to_test_imports edge ×2; new conftest_source_imports edge |
| #2523 | pass | pass |
| #2525 | pass | pass |
| #2610 | conftest_source_modules rose; new test_to_test_imports edge; SCC grew or merged (2 nodes) | new test_to_test_imports edge; existing test reach grew; SCC grew or merged (2 nodes) |
| #2689 | u_pct rose ×2; loc_u_pct rose ×2; scc_max rose; modules_over_half rose; conftest_source_modules rose; new test_to_test_imports edge ×2; test reach budget ×8; SCC grew or merged (200 nodes) | scc_max rose; new test_to_test_imports edge ×2; test reach budget; existing source fanout grew; existing test reach grew; SCC grew or merged (200 nodes) |
| #2690 | pass | pass |
| #2691 | pass | pass |
| #2696 | u_pct rose ×2; modules_over_half rose; conftest_source_modules rose; new test_to_test_imports edge ×104; test reach budget ×680; SCC grew or merged (184 nodes) | new test_to_test_imports edge ×104; existing source fanout grew; existing test reach grew; SCC grew or merged (184 nodes) |
| #3058 | new test_to_test_imports edge | new test_to_test_imports edge; existing source fanout grew; existing test reach grew |
| #3142 | pass | pass |
| #3179 | pass | pass |
| #3195 | new test_to_test_imports edge ×3; test reach budget ×2 | new test_to_test_imports edge ×3 |
| #3193 | test reach budget | pass |
| #3199 | new test_to_test_imports edge ×3; test reach budget ×4; SCC grew or merged (2 nodes) | new test_to_test_imports edge ×3; SCC grew or merged (2 nodes) |
| #3198 | new test_to_test_imports edge ×7; test reach budget ×2 | new test_to_test_imports edge ×7 |
| #3192 | new test_to_test_imports edge; test reach budget ×3 | new test_to_test_imports edge; test reach budget ×3; existing test reach grew |
| #3201 | pass | pass |
| #3065 | pass | pass |
| #3130 | new test_to_test_imports edge; new conftest_source_imports edge | new test_to_test_imports edge; new conftest_source_imports edge |
| #3129 | pass | pass |
| #3131 | new test_to_test_imports edge | new test_to_test_imports edge |
| #3204 | pass | pass |
| #3208 | new test_to_test_imports edge | new test_to_test_imports edge |
| #3211 | pass | pass |
| #3218 | pass | pass |
| #3219 | test reach budget ×2 | pass |
| #3222 | test reach budget | pass |

### Revised failures and added edges

#### #2495

new test_to_test_imports edge; existing source fanout grew; existing test reach grew.

4 added graph edges named by diagnostics; 30 total added graph edges. Full details: `pq-2495-failure.json`.

```text
tests/test_allocator_byte_budget_selection.py -> tests/test_footprint.py
tests/test_calibration_split_row_consumer.py -> tests/test_tessera_campaign_packed.py
tests/test_indomain_split_capture.py -> tests/test_glm5_next_streamed_forward_parity.py
... 1 further edges in the proof bundle
```

#### #2504

new test_to_test_imports edge; new conftest_source_imports edge.

3 added graph edges named by diagnostics; 21 total added graph edges. Full details: `pq-2504-failure.json`.

```text
tests/conftest.py -> prismaquant/staged_lease.py
tests/test_pb_sdk5_pin_2152.py -> tests/test_band_serial_dispatch.py
tests/test_pb_sdk5_pin_2152.py -> tests/test_quantum_executable_readset.py
```

#### #2610

new test_to_test_imports edge; existing test reach grew; SCC grew or merged (2 nodes).

54 added graph edges named by diagnostics; 103 total added graph edges. Full details: `pq-2610-failure.json`.

```text
tests/g3job/test_g3.py -> prismaquant/__init__.py
tests/g3job/test_g3.py -> prismaquant/layer_streaming.py
tests/g3job/test_g3.py -> tools/g3job/g3_lib.py
... 51 further edges in the proof bundle
```

#### #2689

scc_max rose from 194 to 200 (tolerance 0); new test_to_test_imports edge; tests/g3job/test_g3_residency_windows.py reach 279 exceeds budget 275 (package tests/g3job); existing source fanout grew; existing test reach grew; SCC grew or merged (200 nodes).

12 added graph edges named by diagnostics; 32 total added graph edges. Full details: `pq-2689-failure.json`.

```text
prismaquant/joint_dispatch_pilot.py -> prismaquant/stage_inputs.py
prismaquant/model_profiles/base.py -> prismaquant/allocator.py
prismaquant/stage_b_prep_io.py -> prismaquant/io_engine.py
... 9 further edges in the proof bundle
```

#### #2696

new test_to_test_imports edge; existing source fanout grew; existing test reach grew; SCC grew or merged (184 nodes).

320 added graph edges named by diagnostics; 2213 total added graph edges. Full details: `pq-2696-failure.json`.

```text
prismaquant/allocator.py -> prismaquant/assignment_space.py
prismaquant/allocator.py -> prismaquant/joint_aura_rows.py
prismaquant/allocator.py -> prismaquant/priced_cell.py
... 317 further edges in the proof bundle
```

#### #3058

new test_to_test_imports edge; existing source fanout grew; existing test reach grew.

1 added graph edges named by diagnostics; 1 total added graph edges. Full details: `pq-3058-failure.json`.

```text
tests/test_produced_output_spool.py -> tests/test_streamed_boundary_artifacts.py
```

#### #3195

new test_to_test_imports edge.

3 added graph edges named by diagnostics; 28 total added graph edges. Full details: `pq-3195-failure.json`.

```text
tests/g3job/test_g3_source_access.py -> tests/g3job/test_g3_readset.py
tests/test_tautology.py -> tests/test_no_tautological_tests.py
tests/test_tessera_compiled_allocation.py -> tests/test_tessera_scope_endpoints.py
```

#### #3199

new test_to_test_imports edge; SCC grew or merged (2 nodes).

5 added graph edges named by diagnostics; 43 total added graph edges. Full details: `pq-3199-failure.json`.

```text
experiments/diagnose_exl3_grouped_down_replay.py -> experiments/exl3_replay_cli.py
experiments/exl3_replay_cli.py -> experiments/diagnose_exl3_grouped_down_replay.py
tests/test_exl3_grouped_down_replay_integration.py -> tests/test_exl3_grouped_down_replay.py
... 2 further edges in the proof bundle
```

#### #3198

new test_to_test_imports edge.

7 added graph edges named by diagnostics; 52 total added graph edges. Full details: `pq-3198-failure.json`.

```text
tests/test_held_producer_release_gate_2826.py -> tests/test_held_producer_sdk5_smoke_2152.py
tests/test_keyed_egress_validation_retry_2789.py -> tests/test_keyed_egress_owner_2789.py
tests/test_vision_alloc_failclosed_1921.py -> tests/test_visual_allocator_roots_1921.py
... 4 further edges in the proof bundle
```

#### #3192

new test_to_test_imports edge; tests/test_gate_run.py reach 532 exceeds budget 524 (package tests); tests/test_gate_workflow.py reach 532 exceeds budget 524 (package tests); tests/test_stage_b_fp8_source_strict_reads_1219.py reach 532 exceeds budget 524 (package tests); existing test reach grew.

16 added graph edges named by diagnostics; 29 total added graph edges. Full details: `pq-3192-failure.json`.

```text
tests/test_gate_actions.py -> tools/ci/gate_actions.py
tests/test_gate_budget.py -> tools/ci/gate_run.py
tests/test_gate_evidence.py -> tools/ci/gate_evidence.py
... 13 further edges in the proof bundle
```

#### #3130

new test_to_test_imports edge; new conftest_source_imports edge.

2 added graph edges named by diagnostics; 35 total added graph edges. Full details: `pq-3130-failure.json`.

```text
tests/conftest.py -> prismaquant/tessera_lane.py
tests/test_graph_shipcard_numerical.py -> tests/test_graph_shipcard_receipt.py
```

#### #3131

new test_to_test_imports edge.

1 added graph edges named by diagnostics; 5 total added graph edges. Full details: `pq-3131-failure.json`.

```text
tests/test_own_process_measurement.py -> tests/test_own_process_isolation.py
```

#### #3208

new test_to_test_imports edge.

1 added graph edges named by diagnostics; 6 total added graph edges. Full details: `pq-3208-failure.json`.

```text
tests/test_allocator_menu_floor_3132.py -> tests/test_footprint.py
```

## tessera

Old failures: 16/30. Revised failures: 13/30.

| PR | Old failure reasons | Revised failure reasons |
|---|---|---|
| #1275 | pass | pass |
| #1289 | new test_to_test edge ×3; test reach budget ×5 | new test_to_test edge ×3 |
| #1295 | test reach budget | pass |
| #1298 | pass | pass |
| #1302 | new test_to_test edge | new test_to_test edge |
| #1306 | pass | pass |
| #1307 | test reach budget | pass |
| #1309 | new test_to_test edge | new test_to_test edge |
| #1364 | pass | pass |
| #1393 | new test_to_test edge; test reach budget | new test_to_test edge; existing source fanout grew; existing test reach grew |
| #1409 | new test_to_test edge ×2 | new test_to_test edge ×2 |
| #1546 | pass | pass |
| #1560 | pass | pass |
| #1430 | pass | pass |
| #1480 | pass | pass |
| #1497 | new test_to_test edge; test reach budget ×2 | new test_to_test edge |
| #1506 | new test_to_test edge | new test_to_test edge |
| #1513 | new test_to_test edge ×3 | new test_to_test edge ×3 |
| #1536 | test reach budget | pass |
| #1558 | pass | existing source fanout grew; existing test reach grew |
| #1564 | new test_to_test edge ×3; test reach budget ×3 | new test_to_test edge ×3 |
| #1565 | new test_to_test edge ×2 | new test_to_test edge ×2 |
| #982 | pass | pass |
| #1498 | pass | pass |
| #1573 | new test_to_test edge ×12; test reach budget ×6 | new test_to_test edge ×12 |
| #1578 | pass | pass |
| #1579 | new test_to_test edge ×3; test reach budget | new test_to_test edge ×3; existing source fanout grew; existing test reach grew |
| #1582 | pass | pass |
| #1584 | test reach budget | pass |
| #1586 | pass | pass |

### Revised failures and added edges

#### #1289

new test_to_test edge.

3 added graph edges named by diagnostics; 11 total added graph edges. Full details: `tessera-1289-failure.json`.

```text
tests/test_stageprev_793_s7_contract.py -> tests/test_stageprev_793_operative.py
tests/test_stageprev_793_s7_contract.py -> tests/test_stageprev_793_prepare.py
tests/test_stageprev_delegated_observer.py -> tests/test_stageprev_793_prepare.py
```

#### #1302

new test_to_test edge.

1 added graph edges named by diagnostics; 2 total added graph edges. Full details: `tessera-1302-failure.json`.

```text
tests/test_t4_prepare_inputs_release_gate.py -> tests/test_t4_fused_qualify.py
```

#### #1309

new test_to_test edge.

1 added graph edges named by diagnostics; 3 total added graph edges. Full details: `tessera-1309-failure.json`.

```text
tests/test_issue1253_mla_collection.py -> tests/test_serving_mla_mask_runtime_gpu.py
```

#### #1393

new test_to_test edge; existing source fanout grew; existing test reach grew.

18 added graph edges named by diagnostics; 63 total added graph edges. Full details: `tessera-1393-failure.json`.

```text
tests/test_collection_probe.py -> tests/test_pq2459_fixture_export.py
tests/test_comparison_intake.py -> tools/pq2459_export_progress.py
tests/test_comparison_intake.py -> tools/pq2459_fixture_export.py
... 15 further edges in the proof bundle
```

#### #1409

new test_to_test edge.

2 added graph edges named by diagnostics; 4 total added graph edges. Full details: `tessera-1409-failure.json`.

```text
tests/test_value_prefetch_public_sdk.py -> tests/test_t4_prefetch_staged_qualification.py
tests/test_value_prefetch_public_sdk.py -> tests/test_value_prefetch_numeric.py
```

#### #1497

new test_to_test edge.

1 added graph edges named by diagnostics; 17 total added graph edges. Full details: `tessera-1497-failure.json`.

```text
tests/test_graph_attest_batch_kl_binding.py -> tests/test_graph_attest_batch_kl.py
```

#### #1506

new test_to_test edge.

1 added graph edges named by diagnostics; 9 total added graph edges. Full details: `tessera-1506-failure.json`.

```text
tests/test_issue1253_mla_collection.py -> tests/test_serving_mla_mask_registration.py
```

#### #1513

new test_to_test edge.

3 added graph edges named by diagnostics; 19 total added graph edges. Full details: `tessera-1513-failure.json`.

```text
tests/test_graph_owner_hooks.py -> tests/test_glm53_graphs.py
tests/test_graph_owner_lifecycle.py -> tests/test_graph_owner_hooks.py
tests/test_graph_owner_publication.py -> tests/test_graph_owner_hooks.py
```

#### #1558

existing source fanout grew; existing test reach grew.

1 added graph edges named by diagnostics; 8 total added graph edges. Full details: `tessera-1558-failure.json`.

```text
experiments/t4_code/bench_geometry_e2m1.py -> experiments/t8r_speed/rung_quality.py
```

#### #1564

new test_to_test edge.

3 added graph edges named by diagnostics; 15 total added graph edges. Full details: `tessera-1564-failure.json`.

```text
tests/test_forest_grid_roster_determinism.py -> tests/test_forest_grid_roster.py
tests/test_frozen_eager_fixture.py -> tests/test_graph_attest_eager_levers.py
tests/test_frozen_eager_fixture.py -> tests/test_piece_major_reader_boundaries.py
```

#### #1565

new test_to_test edge.

2 added graph edges named by diagnostics; 19 total added graph edges. Full details: `tessera-1565-failure.json`.

```text
tests/test_piece_major_preparation.py -> tests/test_piece_major_comparison_protocol.py
tests/test_piece_major_real_bytes.py -> tests/test_piece_major_preparation.py
```

#### #1573

new test_to_test edge.

12 added graph edges named by diagnostics; 76 total added graph edges. Full details: `tessera-1573-failure.json`.

```text
tests/test_gate_contract.py -> tests/test_matched_reach.py
tests/test_gate_selection.py -> tests/test_attest_receipt.py
tests/test_gate_selection.py -> tests/test_audit_sec2.py
... 9 further edges in the proof bundle
```

#### #1579

new test_to_test edge; existing source fanout grew; existing test reach grew.

3 added graph edges named by diagnostics; 3 total added graph edges. Full details: `tessera-1579-failure.json`.

```text
tests/test_gate_selection.py -> tests/test_census_cell_agreement.py
tests/test_gate_selection.py -> tests/test_lfm_measured_cells.py
tests/test_gate_selection.py -> tests/test_route_census_regimes.py
```
