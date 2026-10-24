# Report route evidence

Original verbatim consumer blocks come from commit `6d8288adc94b3c4b624c143ed372fb6513f32784`; ledger.json identifies each original call, source blob, route, fixture and test node. Only the four assigned application files were changed by this task. Existing unrelated worktree modifications were left alone. No owners, census or baseline files changed.

CPU tests execute AST-extracted current print blocks, refusal Counter/comparison statements, recheck write_text publication statement and waiter main, comparing them against the original block fixtures. Quantizer tensor/device inputs are CPU adapters; tests do not execute or qualify scientific/GPU work. Coverage includes sorted spaced ASCII output, lax nonfinite output, strict NaN rejection, waiter refusal stderr/status, refusal reordering/multiplicity/content, and exact raw-byte SHA-256 within the unchanged recheck report format.

## Explicit gaps

- The unsorted indent=2 recheck JSON report plus newline is a recipe GAP, not a retention justification. Its JSON serialization remains unchanged; only its nested raw-byte hash operation migrates.
- No GPU batch qualification, registered-op execution, admission checks, campaign execution, snapshot subprocess deployment or end-to-end merge/diff runs were performed. CPU publication tests are not evidence for those capabilities.
- No unconditional quantizer --help execution was attempted.

## Observed tests

`python -m pytest -q tests/test_digest_1301/quality-calibration/test_reports.py tests/test_digest_profiles_1301.py --basetemp="$TMPDIR/pytest"`

103 passed, 14 upstream torch.jit deprecation warnings, exit 0 (CPU environment).

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"01b93dbfe0bc76c56ab52727c3df83da2f3758aad8211c87c60f0a78ae0ef9a4","size":903}},"operation":"8767265d4f28d8c42cc3dc8a64eafd52ad3ff6f91ed62c276dc4ca288827f283","receipt":"7a1fd27594d0abdf7ad2ff1f08de4bf902f0172ccb3360419814b153ebd22972"}]
