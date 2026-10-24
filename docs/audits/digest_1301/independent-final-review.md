# Digest #1301 independent final-review prerequisite check

## Verdict: BLOCKED — final integrated tree not supplied

Inspected HEAD: `8bc0b5af8c69af352b559dccda99bd4e6fa8b418`.
Research base: `6d8288adc94b3c4b624c143ed372fb6513f32784`.

This is **not an independent closure approval**. The assigned review is a FINAL SERIAL PHASE after implementation and census closure. The supplied checkout does not meet that prerequisite. `git diff --name-only <research-base> HEAD` lists only:

- `tests/test_digest_parent_completion_1301.py`
- `tests/fixtures/digest_parent_completion_1301.json`
- `tests/fixtures/digest_allocator_pareto_1301.py`

The initial working tree was clean. All implementation, owner, scanner, baseline, deployment, qualification and historical census files are byte-identical to the research base. There are no integrated consumer or census-closure changes to review. No present competing-PR ownership is inferred from historical issuegraph notes. No implementation, census, fixture or acceptance file was edited by this review.

## Blocking return to implementation/integration owners

1. **final-tree (O6):** Supply the integrated implementation/census-closure tree before final review. The full per-retention, classification, fixture-independence and beyond-syntax review is deliberately not conducted concurrently on this unfinished tree.
2. **current-census (O1–O3):** `docs/audits/digest_site_census_pq1301_2026-10-04.json` has no `current_1301` key and no current completion ledger. Historical `refresh_2540` has 420 primary rows and 275 supplemental rows; it is not a current snapshot. Do not overwrite it to satisfy current membership. Acceptance failures independently confirm stale membership, unresolved eligible calls and an empty completion ledger. Resolve current eligibility using actual witnesses, not the historical dispositions; in particular the RED listing of GGUF is not authorization to refactor deprecated GGUF.
3. **allocator-route (O4):** `allocator.main` still constructs raw SHA-256 for Pareto publication. Its acceptance test reaches the final owner-routing assertion only after complete publication/hash-feed/stdout, duplicate suppression, label-width and mixed-key-refusal comparisons pass. Preserve those policies and publication structure when implementing the route.

These are unfinished implementation/integration work, not acceptance-test defects. No objective is declared impossible as written.

## Independent bounded enumeration

A read-only Python enumeration parsed every file yielded by `tools.duplication_inventory._files`, traversed `(call, scope, aliases)`, resolved callees and counted individual raw hashlib calls, literal sorted-JSON calls, and separately other `json.dump`/`json.dumps`/`json.JSONEncoder` calls. It calculated Git blob identities from raw file bytes; it did not derive membership from the baseline. Observed:

| Population | Count |
| --- | ---: |
| Primary calls, including owners | 378 |
| Primary scopes, including owners | 294 |
| Supplemental JSON calls | 300 |
| Gated scopes, excluding owners | 277 |
| Committed baseline scopes | 277 |
| Missing / stale baseline scopes | 0 / 0 |

Enumerated primary scope membership equals `digest_sites()` membership. These counts are distinct populations, not completion evidence. Exact current census membership/source-identity comparison cannot succeed because the current snapshot is absent. No completed research-base call correspondence or independent fixture coverage is certified.

Scanner source inspection confirms `SCANNED=("prismaquant", "tools")`, owners `prismaquant/digests.py` and `prismaquant/tensor_digests.py`, exclusions `vendored`, `archive`, `__pycache__`, `serving_runtime_patches`, and three-value traversal. Git comparison confirms the entire scanner, baseline and historical census are unchanged from research base. Supplemental enumeration did not enlarge the ratchet. No baseline shrink occurred in this checkout.

## Observed executable checks

Command (O1, O2, O3's acceptance assertion, O4 and O5 combined):

```sh
python -m pytest tests/test_digest_parent_completion_1301.py tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py tests/test_duplication_baseline.py -q --basetemp="$TMPDIR/pytest"
```

Observed exit 1: **4 failed, 202 passed, 14 torch deprecation warnings**, 29.91 seconds. The four failures are the four parent-completion tests. Existing owner/profile, guarded/scientific retention, standalone and ratchet checks passed in this focused CPU run. Verified retained stdout with `read_test_artifact`; this is observed execution, not command-submission evidence alone.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"2adbf3a21e6f1676f6d2a07d2a9c6c33faf5a86c942f600be4f2a65c4e00f6a1","size":11545}},"operation":"6afe76aa31e18d7cd9ac55e8eb81da3999eed157e45d202c04a9251405defeba","receipt":"abe8a136be11a77d3af282dffd702623ebabd7553d1a7ea5d2a14fb4bfdffd7b"}]

The requested O3 ledger-driven consumer invocation was also attempted separately, using the brief's `python -c` command to select all `current_1301.completed_calls[*].tests` nodes. It exited 1 with `KeyError: 'current_1301'` before pytest could execute consumer nodes. No ledger-driven consumer suite passed.

[Test evidence {"artifacts":{"stderr":{"digest":"8f41bc2bcd94613a401e4e806240cc36fcb116bd04e7134e29f73f56afbb0c80","size":670},"stdout":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0}},"operation":"33500115578c730e4dadfd7625184d0dd1a089594243667207ac571259ddf298","receipt":"ba43df210eda94a7b0b3e21f9fbd766a3ed6e6e523f4f1da888783a1d9a9dbfa"}]

An initial reviewer-transcribed variant had an extra closing parenthesis and exited with SyntaxError. That was a review-command error, not a repository defect or compatibility result; the corrected invocation above establishes the actual missing-ledger failure.

[Test evidence {"artifacts":{"stderr":{"digest":"14be581838337746c5251cf4cd17d897b386f1872ed6643f41948d504e36ec3a","size":863},"stdout":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0}},"operation":"7bd76e2ffa4c40619c113055364be7e8953e7c8088f705d06b0e97b6c585aaf9","receipt":"7fd52dbb70daa50439ce1aa55a512c18252fdd1c8302bc651af4c2b046514f07"}]

## Explicit review boundary / required resumption

See `supplemental-audit-boundaries.json`. No assertion is made that automated scanning exhausts dynamic or supplemental populations. Actual supported file-spec publication of `io_spans.ReadRateReporter._emit`, individual current witnesses for all six #2637 contracts, protected source rosters, source-qualified kernels/tools, generated Python, scripts/workflows and deployment fragments still require the specified independent review on the integrated final tree. The #2637 test file did execute successfully, but this is not a substitute for the requested per-retention current-witness audit.

AGENTS.md and CLAUDE.md's GGUF deprecation policy was inspected; GGUF is retention/deprecation accounting, not a new refactoring target. Archived, vendored and serving-runtime excluded populations have not been independently classified as supported or unsupported shipping scope in this bounded check. Their scanner exclusion alone proves neither irrelevance nor exhaustive coverage.

Unchanged blobs prove no implementation/default/pin/roster change relative to research base in the supplied tree. They do not prove that the unfinished parent meets identity, framing, memo-sharing, Unicode/NaN/fallback, result width/type, descriptor/read/fence/page-release or publication-order objectives after future migrations. Those final claims remain unverified. No whole-suite, GPU, serving, numerical or performance qualification is claimed. Resume O6 only after the implementation and current census/ledger are integrated; rerun focused checks and the ledger-selected consumer tests, and audit every replacement and retention then.
