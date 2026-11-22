# #2806 test pruning ledger

Decision: keep the existing case matrices and the live guards below. Delete no tests. No further deletion qualifies on this evidence.

## Evidence

- Historical source: `docs/measurements/2806-test-line-ranges-afac07dfbcfe.tsv.gz`. The issue attributes it to `afac07dfbcfe`. This run did not verify the generating revision. The historical query used `python tools/coverage_containment.py query` with that file, the test paths or class selectors below, and `--lines`. It exited with code 0 and wrote counts, ranges and containing node IDs to `$TL3B_OUT/2806-historical.tsv`. The baseline has 3,295 unmeasured nodes.
- Current source: the unchanged test and production files at leaf base `bcd621b0b6b5`. The source citations below use current line numbers. The historical counts do not identify current source lines.
- The query joins setup, call and teardown lines. The first consumer can get lines from a shared fixture. The tool does not measure child-process work (`tools/coverage_containment.py:2-17`). A containing node can execute the same lines with different inputs or assertions. An unmeasured or unknown node does not establish redundancy (`tests/test_coverage_containment.py:12-35`).

## Existing consolidations: retain all cases

The historical query already returns the consolidated IDs. Each row maps the historical node suffix to the same suffix now. It does not reconstruct the IDs or coverage of tests before consolidation.

### Aura reuse identity

Selector: `tests/test_stage_settings_guard.py`. The query found 21 measured and 4 unmeasured cases in this file. Use `tests/test_stage_settings_guard.py::test_aura_streaming_and_checkpoint_directory_are_reuse_identity` as the prefix in both revisions.

| Historical suffix → current suffix (same ID) | Historical lines | Containing nodes | Setting that must cause refusal |
| --- | ---: | ---: | --- |
| `[streaming-aura-cost]` | 96 | 1 | `AURA_COST_STREAMING` |
| `[checkpoint-directory-aura-cost]` | 94 | 3 | `AURA_COST_CHECKPOINT_DIR` |
| `[streaming-aura-hybrid-cost]` | 96 | 1 | `AURA_COST_STREAMING` |
| `[checkpoint-directory-aura-hybrid-cost]` | 94 | 3 | `AURA_COST_CHECKPOINT_DIR` |

One test parameterizes both stages and both settings (`tests/test_stage_settings_guard.py:135-164`). Each case sets the baseline identity and writes a separate artifact under its own `tmp_path`. It changes one setting, checks exit code 2 and checks that the diagnostic names that setting. The parameter ID names the stage and setting on failure. Both stages use the common settings (`prismaquant/pipeline.py:326-362`). The live reuse guard looks up the manifest and refuses incompatible reuse (`prismaquant/pipeline.py:749-866`). Line containment does not show that another test checks the same stage and changed setting.

### Export cache deferral

Selector and node prefix in both revisions: `tests/test_prismaquant_export_native_compressed.py::TestProductionCacheExportPath`. The query found nine measured nodes for this class.

| Historical method → current method (same ID) | Historical lines | Containing nodes | Input |
| --- | ---: | ---: | --- |
| `test_mxfp8_scale_sweep_cache_defers_to_export_recompute` | 194 | 0 | `MXFP8_E4M3`, `scale_sweep` |
| `test_fp8_scale_sweep_cache_defers_to_export_recompute` | 195 | 1 | `FP8_E4M3`, `scale_sweep` |
| `test_fp8_gptq_cache_defers_to_export_recompute` | 196 | 0 | `FP8_E4M3`, `gptq` |
| `test_mxfp8_e4m3_gptq_cache_defers_to_export_recompute` | 195 | 0 | `MXFP8_E4M3`, `gptq` and `joint_scale_opt` |

A direct historical `query --lines` for the FP8 `scale_sweep` node confirmed that the FP8 `gptq` node contains its lines. The query reports 196 lines for the containing node. The query wrote the ranges to `$TL3B_OUT/2806-fp8-scale-sweep.tsv`.

The first three `unittest` methods have distinct names and use one helper (`tests/test_prismaquant_export_native_compressed.py:2630-2666`). Each call builds its own CPU cache entry with the specified render levers and sets an activation marker. It checks that `_pack_production_cached_2d` returns `None`. This result tells export to recompute. The helper restores `_PRODUCTION_WEIGHT_CACHE` and `_CACHED_ACTIVATIONS` in `finally`. The live packer has separate MXFP8 and FP8 branches. FP8 tests `scale_sweep` and `gptq` separately in its condition (`prismaquant/export_native_compressed.py:1317-1352`). The fourth method uses different levers and is not part of the requested consolidation (`tests/test_prismaquant_export_native_compressed.py:2668-2690`). Keep all cases. The same result does not make the inputs equivalent.

## Other retention decisions

### Gridbook

- Archive: no deletion. `AGENTS.md:93-101` records removal of the Gridbook pin, exporter, serving profiles and ship gates. It gives `archive/gridbook_lane_2026-09-02/` as the archive path. The archive is absent from this checkout. Current default collection uses `tests` (`pytest.ini:2`). The issue's historical collection description also excludes the archive. No archive test was available to inspect or remove.
- Live readers: retain `tests/test_retired_codebook_refusals_1304.py`. The historical query found 82 measured nodes. The current reader matrix checks `RetiredFormatError` and the archive diagnostic for old names (`tests/test_retired_codebook_refusals_1304.py:187-233`). For example, `test_reader_refuses_a_retired_codebook_rung[format_registry.get_format-NVFP4_CB_K16]` covered 1,239 historical lines and had two containing nodes. The live registry checks retirement before it raises `KeyError` (`prismaquant/format_registry.py:1508-1535`). The name pattern and archive path remain in `prismaquant/schemas.py:35-38`.
- Error type: retain `test_retired_refusal_is_not_a_key_error` in that file. Its six historical lines had 7,311 containing nodes. Its assertions require a `ValueError` subtype that is not a `KeyError` subtype (`tests/test_retired_codebook_refusals_1304.py:236-240`). A reader that catches `KeyError` must not ignore a retired name as an unknown name. The matrix and this type check protect live readers, not the archived implementation.

### Process, timeout and memory controls

| Historical selector | Historical query | Decision and assertion inspection on current source |
| --- | --- | --- |
| `tests/test_own_process_isolation.py` | 22 unmeasured | Retain. The shared-session test checks that the selected cases run in one child, that the shared case runs in the parent, and that a deselected case does not run. The timeout cases check that only the hanging test fails, with a timeout diagnostic, in both child and in-process runs. An unavailable timeout plugin must cause a failure, not a skip (`:113-254`). |
| `tests/test_own_process_child_exit_audit.py` | 8 unmeasured | Retain. The exit audit checks that a nonzero child exit fails its proxy even after all phases reported success. It checks that the separate shared case does not fail (`:68-95`). |
| `tests/test_own_process_measurement.py` | 13 unmeasured | Retain. The measurement checks compare recorded phase sums with child reports. The child-only coverage check requires the child context to survive coverage combination (`:70-172`). |
| `tests/test_io_engine_sealed_memory.py` | Unknown selector | Retain. In its own cgroup, the test pauses reads during its comparisons. It checks that reclaim and drop reduce shared memory by the expected pages, and that a take alone does not (`:55-114`). |
| `tests/test_reclaim_readings_gb10.py` | Unknown selector | Retain. The cases compare observed drops and declared readings for the render stream, CUDA render cache and spill replay. These cases need CUDA and a separate cgroup (`:89-95`, `:134-220`). |

A shared cgroup includes allocations from sibling processes. This can invalidate the memory comparisons (`tests/cgroup_readings.py:1-16`). Do not remove isolation or timeout controls because the historical query did not measure their work.

### Path-sealed row stream

Selector: `tests/test_tessera_row_stream.py`. Retain the file. The historical query found 23 measured nodes. These cases use different inputs and check different outcomes:

| Node suffix | Historical lines | Containing nodes | Current assertion inspection |
| --- | ---: | ---: | --- |
| `test_a_killed_stream_row_adopts_its_journalled_anchors_on_relaunch` | 4,697 | 0 | Force an interruption. Check which anchors the relaunch adopts and which it encodes. Compare the published bytes with a clean run (`:381-426`). |
| `test_a_unit_whose_inputs_changed_is_refused_not_adopted` | 4,052 | 0 | Change the journalled weight receipt. Require the W/X/H receipt refusal (`:478-491`). |
| `test_a_corrupt_late_entry_refuses_before_any_identity_bound_write` | 3,957 | 0 | Corrupt the late activation file. Require a checksum refusal and no identity-bound outputs. Check that only the earlier wires exist (`:500-523`). |
| `test_an_entry_whose_receipt_is_not_the_manifests_is_refused_by_name[journalled]` | 4,271 | 1 | Change the manifest's Hessian receipt and re-sign the manifest and shards. Require a refusal that names the journalled unit. Check that resume does not encode or publish the cost (`:676-706`). |
| `test_an_entry_whose_receipt_is_not_the_manifests_is_refused_by_name[pending]` | 4,386 | 0 | Use the pending unit instead of the journalled unit. Require the same checks for that unit (`:676-706`). |

The file also checks that a reference stand-in without a sealed receipt fails (`:825-833`). These tests protect input identity and publication, not just line execution.

## Deletion accounting

No subject or node was removed. There are no prune-specific before and after node mappings or covered-line sets. There is no permitted coverage loss to list. The historical query is not a same-source before and after measurement. Report no runtime saving for the existing consolidations or for the uncollected archive.

## Current case results

The latest targeted CPU check used `python -m pytest -v --tb=short` on `tests/test_2806_test_pruning_ledger.py`, the aura node prefix and the four export methods above. It collected nine cases. All nine passed. The eight existing cases used unchanged test and production source at `bcd621b0b6b5`. The new evidence test ran the historical queries again. It checked the aura and cache line counts, FP8 containment, retirement type-check count, unmeasured process nodes, unknown memory selectors and row-stream node count. It wrote the query output to `$TL3B_OUT/2806-retention-query.tsv`.

| Case suffix or method | Result |
| --- | --- |
| `[streaming-aura-cost]` | PASS |
| `[streaming-aura-hybrid-cost]` | PASS |
| `[checkpoint-directory-aura-cost]` | PASS |
| `[checkpoint-directory-aura-hybrid-cost]` | PASS |
| `test_mxfp8_scale_sweep_cache_defers_to_export_recompute` | PASS |
| `test_fp8_scale_sweep_cache_defers_to_export_recompute` | PASS |
| `test_fp8_gptq_cache_defers_to_export_recompute` | PASS |
| `test_mxfp8_e4m3_gptq_cache_defers_to_export_recompute` | PASS |

The prior check reported 1/1 coverage-tool tests, 171/171 export tests, 82/82 retirement tests, 25/25 stage-setting tests and 23/23 row-stream tests as passing. That check reported no new test. The leaf now adds `test_retention_evidence_matches_supplied_baseline`. The latest targeted run above includes its passing result.

This leaf changes no existing tests or production code. It adds one evidence test and this ledger. GPU and served qualification do not apply to this retention-only scope.
