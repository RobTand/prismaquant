# Dispatch digest handoff evidence

## Scope and provenance

Only application files `tools/dispatch_joint_quanta.py` and `tools/dispatch_stage_a_split.py` were changed in this slice. Owner APIs, shared census, baseline/acceptance files, frozen registry, remote package rosters, and other agents' package-builder files were not edited.

`provenance.json` records frozen registry calls from `tests/fixtures/digest_parent_completion_1301.json`, source commit `6d8288adc94b3c4b624c143ed372fb6513f32784`. The two `*.original.txt` files now preserve only needed executable excerpts (synthetic wrappers for isolated JSON expressions, real handoff derivations without docstrings, and `_BAND_SET`). They total **6,795 bytes**, down from **254,731 bytes** (97.3% reduction). Full historical Git blob identities remain provenance only; tests verify excerpt hashes, source line mappings, and registry call AST identities, not a false whole-file blob match. `original_outcomes.json` captures original rendering outcomes. `ledger_dispatch.json` covers all 14 registry calls in scope plus the separately audited embedded `_BAND_SET`; **seven calls are replaced, seven remain eligible unresolved API gaps, and one is a separate justified deployment retention**. Among the 14 registry calls the split is 7 replaced / 7 unresolved / 0 justified retention; the embedded fifteenth site is the single deployment retention. Every ledger test reference is an unparametrized, AST-resolvable function node.

## Implementation, pending API gaps and deployment retention

Seven sorted JSON recipes now call existing finished profiles: compact strict progress-grace wire; compact lax handoff identity; indented lax handoff payload, dry-run report and split comparison report; spaced lax publish/wait reports. Existing encoding, newline, digest, publication, reads, ordering and dispatch behavior remain in their original locations.

**Pending blocker: `pending-owner-recipes`.** The existing `prismaquant.digests` owners lack both exact recipes needed by these seven eligible callers:

- **6 unsorted JSON sites:** `_assert_container_image` spec clone; `build_round` template clone; four split CLI output sites (`init`, `init-forward`, `print-argv`, `dispatch`). Preserve insertion order, `ensure_ascii=True`, default separators, lax NaN, no default hook, and native exceptions. Existing stdout classification does not exempt supplemental JSON.
- **1 sorted pretty `default=str` site:** `check_source_coverage`, with `sort_keys=True`, `indent=2`, `ensure_ascii=True`, lax NaN and fallback conversion. Existing indent2/no-default owners are not equivalent.

These need exact reviewed owner recipes before routing. Missing recipes are **not justified retention**. Owners were not modified; do not substitute sorting or remove `default=str`. The seven unresolved callers are intentionally unchanged pending that external owner work.

`target_deployment_witness.json` describes the isolated target witness. The actual unchanged `_BAND_SET` payload executes with its existing import roster and **without digests installed in the isolated target package**; its test checks argv read order, first-band header choice, sorted boundaries, exact stdout and newline. This is not a claim that an inspected live fleet lacks the owner: the frozen source checkout actually contains `prismaquant/digests.py`. The deployment witness precludes routing against that protected target roster: adding the owner would introduce an unavailable target dependency. The test explicitly verifies `find_spec("prismaquant.digests") is None` before executing the real embedded payload. Unsorted JSON or a missing owner recipe alone would NOT justify retention. This witness is scoped to the isolated protected deployment contract, not a newly observed live fleet. No new remote dependency was introduced.

The handoff filesystem test uses a deterministic boundary-template SDK builder seam, while exercising actual dispatcher template derivation, identity/path construction, byte publication, idempotence and conflicting-file refusal. This avoids an installed SDK generation mismatch (SDK_VERSION 6 versus required 5); it is not live SDK compatibility evidence.

## Replaced-call consumer coverage review

Every replacement now has actual consumer behavioral coverage, not just a comparison of a frozen expression to an owner profile:

| Replaced call | Actual consumer check |
| --- | --- |
| `quantum_argv` strict grace JSON | Real slice/receipt/manifest fixture through dispatch and emitted `--progress-grace-derivation` argv; exact frozen original bytes. Separate strict NaN rejection test. |
| `handoff_template_id` compact JSON | Current ID derivation equals frozen hash recipe with Unicode, NaN and ignored `template_id`. |
| `handoff_template` indent2 JSON | Real derivation/filename/publication versus both frozen handoff functions; exact bytes, newline, idempotence and conflict refusal. SDK builder seam remains explicit. |
| `_dispatch` dry-run JSON | Real `main`/`_dispatch` branch; exact frozen stdout, Unicode and newline, no gateway publication. |
| `_dispatch` published JSON | Real gateway loop; exact frozen stdout and newline, submitted argv and persisted action state. |
| `_dispatch` pending JSON | Real band-pending branch; exact frozen stdout and subsequent empty-publication message, no submission or state write. |
| split `main` compare JSON | Actual CLI branch; exact frozen stdout/newline and return code. |

The report-branch tests stub external plan/admission/gateway inputs, not the serialization or output branches. Separate ledger tests enforce the 7/7/1 classification and pending blocker.

## Correction validation

The first two correction runs found only new test setup mistakes (4 failed/11 passed, then 3 failed/12 passed: incorrect CLI seam/flag and incomplete fixture inputs). Those mistakes were corrected without changing application behavior. The corrected focused run passed **15 tests**:

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"7a8cedd99fbbc9f2d279ebc09c26413fb05f3e8d8a1f8db01c6db1e9b0feac86","size":773}},"operation":"258513886225412f282ff3826d0ce051e77f3580c24d463a8447a062648d93df","receipt":"c17c4b0f4975303b48a43f6ee22caf60eafacb2f48cdb1b4b849e03d71e61583"}]

Final correction validation (no exclusions):

```
python -m pytest -q -rs tests/test_digest_1301/joint-handoff/test_dispatch.py tests/test_dispatch_stage_a_split.py tests/test_dispatch_joint_quanta.py tests/test_compute_phase_grace_1165.py --basetemp="$TMPDIR/pytest"
```

**160 passed, 1 skipped, 14 warnings.** The skip is Stage B spill qualification, requiring administrator-provisioned private scratch via `PQ_STAGE_B_SPILL_TEST_ROOT`; that qualification remains incomplete. The two placement tests that failed in the earlier environment both passed in this run. No live-deployment or full-repository validation is claimed. `git diff --check` also passed. Correction edits touched only this slice's dispatch test, ledger, provenance, two excerpt fixtures and evidence document; no owners or other agents' files were edited.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"dcc58dbd283b97fe38520928e968557cd2f3b0ab4a380ef8d8d5b58826442202","size":1260}},"operation":"bd9cde663ad1f8908aa84404ca7d6bd0feebd5c1ee9505958ae204e4b1ed6a82","receipt":"b8f5ebd65130884e4a2f606737eb4ecfcd017891de0a3f83aa1efd5fe277c14b"}]

## Earlier checks (before this correction)

All tests were run through `run_tests`. Final focused command:

```
python -m pytest -q tests/test_digest_1301/joint-handoff/test_dispatch.py tests/test_dispatch_stage_a_split.py tests/test_dispatch_joint_quanta.py -k 'not test_two_stage_a_rows_whose_windows_exceed_a_box_are_not_both_placed and not test_a_stage_b_row_charges_spill_cotangent_and_spool_to_the_box' --basetemp="$TMPDIR/pytest"
```

Result: **141 passed, 2 deselected**, 14 Torch/Python 3.14 deprecation warnings.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"0c401adf53ca1cc71795a955eef1f061e65458295b74261467b88b708e14c5f2","size":917}},"operation":"0ea25edb2c81eb31d116d4b317ab7ec5176aa7aaaaee601b3433ab07075bb785","receipt":"490c07a4189ab37827f74e4c83af4a20aacb7e9e79e18e6c9b7e23a2e85e3a5e"}]

The initial unfiltered run found two existing installed-PrismaBuild placement failures (`result['first']` is None), alongside three subsequently corrected new-test setup failures. Its result was 137 passed, 5 failed. No application behavior was changed to address test setup or external placement failures.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b8f84b4bfb5851e274e32fbd798b57962d09d3e303531e9d4b2c39862fdbe943","size":10863}},"operation":"32dd1fa5743302bb8dc3a451c5c5dcf557efaf550b8cf2a0cea4bc735d76935d","receipt":"5055259a407fb4f5f20ab292749675b08882d5e4ffa477bc83c89f7f5fd72823"}]

A subsequent isolated new-test run exposed the SDK mismatch described above (10 passed, 1 failed); the builder seam corrected that setup dependency.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"fcf008b3314cd729645a41c0339f6a4a8c873d2744ba1ae0bd58097f8881e414","size":5272}},"operation":"d25e98cdf01129b1587a942bf6bc3a4c2270c99b604e3105e82fab765faafad2","receipt":"50a53c81e4f3bd5d5e801b338230f432c1e77abba713c556ef8ddded61a354c9"}]

To distinguish a regression from an environment/baseline issue, the two placement tests were rerun with `dispatch_joint_quanta` loaded from the frozen original source fixture (no worktree source restoration). Both failed identically: **2 failed**. Their external placement behavior remains unresolved and is outside the assigned slice.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"c488727047ec9ae11b222bc2640da6dcb4bc244f228942819e64d6b93a3c835e","size":4790}},"operation":"045a66b168321ea52503c8a6b98bfab0395814e8dbc8dff5b9c7cb5fc54d44e2","receipt":"304b362426288b773b231269ffda9fbdc16cceea90b646100ddc3e209a8452e1"}]

`git diff --check` completed successfully. No live deployment or full-suite validation is claimed.
