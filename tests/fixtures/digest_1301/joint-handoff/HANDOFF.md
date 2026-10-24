# Joint-handoff partial implementation

## Disposition

This slice routes 41 of the ten assigned files' 56 frozen registry calls through finished owner APIs. `ledger.json` merges the per-contract ledgers for serial integration into the shared census. It is not a closure declaration. Fifteen calls remain eligible but unresolved because exact finished encoding APIs are absent. There is one separately audited embedded deployment retention, `_BAND_SET`; it is outside the ratchet population. No owner, shared census, baseline, scanner, immutable acceptance fixture/test, runtime pin or deployment roster was edited.

Required owner-recipes handoff: insertion-preserving unsorted JSON (plain and indent2; including clone round trips); sorted ASCII indent1 lax and strict with no fallback; sorted ASCII indent2 lax with `default=str`. Existing profiles cannot reproduce these contracts. Missing recipes are NOT justified retentions. Exact per-call witnesses are in the contract ledgers. Owner-recipes must provide the APIs before the remaining consumers can be completed under Agreement S.

Independent original function excerpts/transaction blocks, original outcomes, and provenance are committed locally. Behavioral tests compare actual consumers, or actual main transaction blocks where running full main would require external execution, against those originals. Main block tests use dependency doubles and do not certify real authorization policy, native execution or fleet submission. Original reference source does not call the replacement owners. The framing tests include component-wise Path ordering, binary content, Unicode name byte lengths and native surrogate refusal. Package tests include NaN/Infinity/-Infinity publication parity; pretty package JSON remains distinct from compact identity JSON. Readset tests cover hash-before-decode and malformed-input/refusal boundaries.

Review found and corrected an initial strict-versus-lax pretty-JSON substitution before final validation. No such strictness change remains. Protected embedded `_BAND_SET` source is unchanged; target-layout evidence and isolated execution are recorded separately. That evidence does not claim to have queried live fleet contents.

## Verification

Final combined command:

`python -m pytest tests/test_digest_1301/joint-handoff tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py tests/test_duplication_baseline.py -q --basetemp="$TMPDIR/pytest"`

Observed **260 passed, 1 failed**, 14 upstream torch deprecation warnings. The sole failure is `test_primitive_digest_sites_only_shrink`: no growth assertion failed; 25 genuinely removed scopes require the serial integrator to shrink the shared baseline. This slice is forbidden to edit that baseline. It is an integration blocker, not an impossible objective or a passing ratchet claim.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"77a542dde7572db6ec5cd97fe71fbd2db6503199d901f74cf464d188d451d831","size":4605}},"operation":"05f715bbac87a30741fd03d9d276549977ed7cb1d207cd4702f4e33ab55a8a36","receipt":"6a111ce9580acfcb129c6733679fdc4c458d902c8480dba23e80635ecc84e184"}]

After adding the local integration-ledger validation:

`python -m pytest tests/test_digest_1301/joint-handoff/test_ledger.py -q --basetemp="$TMPDIR/pytest"`

Observed **1 passed**. This checks exact frozen registry membership for these ten files, unique IDs, pinned original blobs, fixture existence and executable AST-resolvable test references. Metadata validity is not itself compatibility proof.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"91a036ea195bf2a0490a514c4305f91d731c79cdf1a766897175f3b49cf66cf7","size":268}},"operation":"089430de21c12668cc77469454348ace245dfb546d5d6c78bb3817bc25eb6002","receipt":"5a6f83245151bfac8d5909ecc610c180e02a2165187a28ff1fb669d7a6923f44"}]

Focused existing consumer checks and independent review receipts are in the three contract evidence documents. Two installed-SDK placement cases failed equivalently against the original source in dispatch's broader exploratory run; no policy workaround was applied. A private-scratch spill test skipped in that exploratory run; no native spill qualification is claimed. Final combined command above contains no skips. `git diff --check` passed. An additional shell-only census print attempt timed out after ten seconds; the completed pytest ratchet output, not that attempt, is the baseline evidence.

No GPU, remote/native operation, whole-suite, performance or parent acceptance completion is claimed. Parent current-census and baseline integration remain outside this piece. Supplemental audit is bounded to the documented ten-file population and embedded source witness, not an assertion that automated enumeration exhausts every possible dynamic alias/generated/shell deployment.
