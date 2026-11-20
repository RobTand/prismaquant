# PQ #3045 MTP census acceptance

## Current issue acceptance

> - [ ] Repository: prismaquant. Remaining objective: O4, specifically legitimate regeneration of an MTP census usable by the existing pinned-calibration campaign reader. Continue the retained implementation; O1/O2 restoration, strict 710-key binary64 comparison, immutable prepared-plan forwarding, installed backward integration, and existing CLI argument repairs are complete and must not be repeated. At retained HEAD 1f6ff7fa, the adjacent/integration/recipe CPU suite reports 145 passed, 3 skipped; this is not historical numerical equivalence.
> - [ ] Preserve the actual bound base-census calibration_input receipt through MTP census production, using the existing identity/census mechanisms. Do not fabricate a receipt, redraw tokens, weaken the reader, change builder admission, edit sealed artifacts, or extend this to unrelated capture scopes. Preserve legacy behavior when the base census has no optional receipt. Add CPU-only regression coverage in tests/test_glm_mtp_census_calibration.py using synthetic inputs: exact receipt preservation, successful require_census_draw with the original identity, refusal for a changed/dropped receipt or different token digest, unchanged legacy no-receipt behavior, and no mutation of caller metadata. Exercise the real mtp_census and reader rather than mocks of the missing propagation. The historical route must still use the exact 512x512 seed-0 calibration artifact hash 9cd1fa129f249abd80d22efaeb8bc7e8b2d3b4252f173a8c6f2b2e496a4f8329; probe seed_base 7000 is unrelated.
> - [ ] Shared interface agreement: retain the existing calibration_input receipt schema and existing capture manifest filename capture_manifest.json; no CLI/schema or documentation changes belong here. The sibling recipe correction does not depend on this implementation and this repair must be independently testable without executing that recipe. Do not modify either user-protected acceptance file. Run python -m pytest tests/test_glm_mtp_census_calibration.py tests/test_glm_mtp_capture.py -q --basetemp="$TMPDIR/pytest" with run_tests; report skips honestly. Run the existing objective acceptance suite as regression coverage if feasible. No GPU, fleet artifacts, full source authentication, regeneration, prepare, or Arm-A match is claimed. Keep #1406/#2661 as attribution; do not reopen shipped #2705 or create dependency work.

## Coverage on `origin/main` at `6b3fa6953`

| Criterion | Verdict | Evidence on main |
| --- | --- | --- |
| 1. O4 authenticated regeneration on the pinned draw | PARTLY | `tests/test_glm_mtp_capture.py::test_both_phases_publish_a_capture_a_selected_consumer_accepts:652` and `tests/test_glm_mtp_source_scope.py::test_campaign_row_prices_mtp_units_from_the_mtp_capture:287` cover synthetic capture and pricing on a draw without a receipt. |
| 1. Prior O1/O2 and adjacent work | MOOT | The issue records completion at retained head `1f6ff7fa7`. `b5215773a` restores the reference and prepared-plan path. `1f6ff7fa7` adds backward integration and CLI checks. |
| 2. Exact receipt, original identity, and caller metadata | NOT COVERED | `tests/test_tessera_campaign_calibration_input.py::test_census_and_capture_reuse_keep_saved_draw_identity:259` covers the body census. `prismaquant/glm_mtp_capture.py:462` selects two digests and drops the receipt in `mtp_census`. Lines 533–534 apply to `attention_census`. |
| 2. Changed or dropped receipt and different token digest | PARTLY | `tests/test_tessera_campaign_calibration_input.py::test_census_and_capture_reuse_keep_saved_draw_identity:259` rejects a changed body receipt. `tests/test_tessera_campaign_fanout.py::test_a_census_of_another_draw_is_refused:299` rejects a different token digest on a hand-written census. |
| 2. Legacy census without a receipt | PARTLY | `tests/test_glm_mtp_source_scope.py::test_campaign_row_prices_mtp_units_from_the_mtp_capture:287` accepts a legacy draw. An assertion for the absent receipt key is missing. |
| 2. Historical artifact and draw | NOT COVERED | `tests/test_tessera_campaign_fanout.py::test_a_derived_census_names_its_base_census_s_draw:257` uses synthetic hashes. The historical artifact hash is recorded in `docs/design/glm_mtp_option_b_protocol_2771.md:36`. |
| 3. Receipt schema and manifest filename | COVERED | `tests/test_tessera_joint_eval_panel.py::test_fresh_draw_descriptor_and_load_round_trip:277` checks the schema. `tests/test_tessera_calibration_cache.py::test_cli_capture_then_reuse_never_repeats_forward:503` checks the published filename. |
| 3. Independent CPU regression and requested run | NOT COVERED | Main lacks the named census module. The retained objective suite is absent from main. |

## Row 110 changes and verification

`tests/test_glm_mtp_census_calibration.py` uses a real receipt from a synthetic eight-token safetensors file.
It checks receipt equality, draw acceptance, and input preservation at `test_mtp_census_preserves_bound_calibration_input:55`.
`test_mtp_census_reader_refuses_changed_or_dropped_receipt:66` checks both receipt failures.
`test_mtp_census_reader_refuses_different_draw:79` checks both digest failures.
`test_mtp_census_without_optional_receipt_keeps_legacy_draw:89` checks draw acceptance, the absent key, and input preservation.

The lead must regenerate BUILD for the new test file and run both requested modules on CPU in Flux. The new cases need one worker and process a 64-byte token tensor per case.
The test-only commit is expected to fail when it checks the missing receipt. No test ran in this worktree.
Historical regeneration still needs the pinned 512 by 512 seed-0 artifact, source authentication, and the original GPU route.
