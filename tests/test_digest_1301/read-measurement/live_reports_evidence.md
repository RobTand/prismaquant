# Live reports subset — Agreement S

## Scope and result

Only the four assigned application tools were edited. Existing settled owners were used without changing shared code or adding owner recipes:

- `joint_prepare_startup_probe.py`: plan bytes -> `bytes_sha256hex` (1 site).
- `profile_stage_b_head.py`: record bytes -> `bytes_sha256hex`; final sorted report -> `DIRECT_ASCII_SPACED_LAX.text` (2 sites).
- `band_serial_handoff_live_pair.py`: producer and consumer reports -> `DIRECT_ASCII_SPACED_LAX_DEFAULT_STR.text`; derive report -> `DIRECT_ASCII_SPACED_LAX.text` (3 sites).
- `stagea_produced_live_cycle.py`: `_digest` -> `bytes_sha256hex` (1 site).

The ledger records all 23 original census IDs in scope: 7 replaced, 16 blocked. `original_id` is the exact stable census identifier, not a newly generated ID using shifted line numbers. Source blob IDs and snippet line numbers refer to commit `6d8288adc94b3c4b624c143ed372fb6513f32784`. Snippets were extracted from `git show <commit>:<path>`, scoped to their original consumers. No complete original source files are stored. Both original consumer SOURCE statements and the edited consumers are executed in tests.

## Ordering and coverage

Application edits retain the existing read calls, digest comparison positions, JSON load positions, checks, print calls, flush arguments, return statuses and publication code. Owner imports are local at existing execution boundaries where appropriate.

Observed tests cover binary/empty/raw JSON bytes, hash-before-parse refusal, environment-before-plan-read and plan-check-before-config-load, the no-digest startup path, JSON parsing acceptance/refusal, sorted ASCII output, non-finite numbers, `default=str`, circular errors, unsupported-object errors, derive CLI dispatch, and the profile main report with stubbed intake. Producer/consumer terminal report blocks execute from actual edited function bodies with success/failure checks; these are isolated report-boundary tests, not end-to-end producer publication tests. The existing canonical-read audit additionally exercises the real consumer role. Existing bootstrap and settled owner tests also pass.

## Blockers (not retention)

No nearby profile was substituted for a missing exact recipe:

- 1 raw binary SHA256 `.digest()` site in band preparation: no settled raw digest-bytes owner.
- 3 sorted ASCII spaced, indent=1, no-default JSON sites (startup report; two band prepare writes): no settled exact owner.
- 12 insertion-order ASCII spaced lax, no-default JSON sites (2 profile guard reports; 10 live-cycle reports): no settled exact owner.

Each is explicitly marked `blocker` in `live_reports_ledger.json`, with its original ID and original SOURCE snippet. These are not retention claims. No GPU, fleet, real staged publication, or full live-cycle run was performed. Producer success-path publication was not exercised end-to-end by the new tests.

## Provenance repair

Corrected 16 shifted IDs in ledger rows, fixture keys and ledger fixture
fragments using the frozen parent registry. All 23 IDs now match the registry.
Independently compared all 23 original expression ASTs and all 23 executable
SOURCE-statement ASTs against `git show` at base
`6d8288adc94b3c4b624c143ed372fb6513f32784`, not modified working-tree source.
Added regression checks for registry completeness, syntax hashes, source blobs,
line numbers and exact fixture references. No application source changed in
this repair.

The repair's first test run failed because the test snapshot lacks the base
Git paths (230 passed, 1 failed). The Git-backed check now explicitly skips
when those objects are unavailable; the independent frozen-registry AST/hash
check always runs. The host Git-object AST audit completed successfully.
This snapshot limitation is a validation gap, not retention. End-to-end live
publication remains untested, not retention.

Re-ran the command below after the repair: **231 passed, 1 skipped**, 14 Torch
deprecation warnings, exit 0. The skip is solely the Git-object check described
above. The older 230-pass receipt below predates the repair.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"37029fdbefb0c01e811f6fb0a4cb16981bbb0623aff4ab71649923780d13ec90","size":1074}},"operation":"a1f9c14f0063ef185d6e62d4be60208452d49676cc4b69c1ebb1044b7f7a2f7e","receipt":"7befd03d42355f37e973b85425ff18a065cedee51ad81b2589fa55bfa222822f"}]

## Checks (before provenance repair)

Final command:

```sh
python -m pytest -q tests/test_digest_1301/read-measurement/test_live_reports.py tests/test_band_pair_canonical_read_audit.py tests/test_profile_digest_bootstrap_1899.py tests/test_digest_hex_1457.py tests/test_spaced_json_profile_routing_1884.py --basetemp="$TMPDIR/pytest"
```

Result: **230 passed**, 14 existing torch Python 3.14 deprecation warnings, exit 0.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"01130d659ec996e65ff21485b553c04db01b4be8bd8d419bad8802294b9aeeca","size":1062}},"operation":"68281748bf1e4b358ad56a326ea637d9296e1429e3419622deeffd0bf4ffb1f4","receipt":"a186da2f29f05ab3279a2dec01c2dc88165565cd630ede6f224d9ab10ad32f1b"}]

Earlier checks, not hidden: the first new-test run had 5 test expectation/selector failures (incorrect refusal wording, assumed schema validation, ambiguous main-snippet selector), corrected without changing application behavior; an intermediate run passed 18 tests. One expanded-suite invocation named a nonexistent test file and collected no tests (exit 4); corrected to `test_spaced_json_profile_routing_1884.py` for the successful final run. Fixture generation printed completion but exceeded one shell timeout; the subsequent scoped extraction completed successfully and the fixtures were consumed by passing tests.

Pre-existing changes in other tools and other fixture/test subsets were left untouched. No workflow transition or publication was performed.
