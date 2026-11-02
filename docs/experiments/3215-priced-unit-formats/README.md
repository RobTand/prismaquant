# Parsed formats in priced unit selection

The original cold file passed all 25 cases in 128.34 seconds on baseline
`9c8beebde2266c78ea5910bc83b30e90cb669d74`. Its real 11,232-row,
718-pick fill test took 121.648 seconds. Of 12,613 sampled stacks,
10,064 included `_grid_rung` and 8,778 included
`parse_tessera_format_name`. Sampling locates repeated work; these counts
are not CPU-time percentages. Fixture setup took 0.027 seconds.

## Behavior and ownership

Each selector call memoizes its existing grid/rung parser locally with
`functools.cache`. Distinct format spellings still pass through the same
parser and rate validator on first use. The memo ends with that selector
call, so later calls resolve their formats again. Parsed results contain
only a grid string and integer rung, or `None` for another format domain.

The allocator still recomputes every marginal after every single pick.
Price validation, candidate iteration, loss arithmetic, byte deltas, tie
ordering, eligible-row counts, refusals and final spend receipts are unchanged.
There is no persistent metadata memo or change to numerical formats,
serving admission, allocator defaults, or export gates.

All 65 original test-module statements and all 114 original assertions remain
unchanged and in order. The real CSV, its SHA-256, all 11,232 units and the
718-pick expectation remain intact. The new regression runs the real allocator
on the existing six-unit fixture, independently requires picks 4, 5, then 2,
300 bytes spent and 50 bytes left, and then requires one parse per spelling.
Its second selection requires fresh parsing, which prevents a global memo.

## Evidence

- Cold before, terminal OK, 25 passed, zero skips:
  https://app.buildbuddy.io/invocation/d7c051d7-2ffb-41d4-a981-7fa7edf09b6e
- Red commit `2faa4f1005ddd7394988abafcb269d0d805a6f31`, terminal Failed
  as intended. Real picks, their order and the independent spend expectations
  passed before the parser-count assertion failed. The two spellings were
  parsed 48 and 30 times instead of once each:
  https://app.buildbuddy.io/invocation/9c6541bb-c4f6-4738-89d4-99afd98119e8
- After measurement and full-case comparison use source commit
  `963debc5570e1216faf6da5be4f0edc668396964`:
  https://app.buildbuddy.io/invocation/9cceaf51-b4fd-4efe-8278-85156f5e182c

Cold after: 26 passed, zero skips, 21.70 pytest seconds / 21.674 JUnit
seconds, including the new regression. The baseline JUnit time was 128.324
seconds. Its 106.650-second reduction projects to 76.788 static seconds per
merge at the frozen 72% selection rate, and 25.596 observed dynamic seconds
per merge at 24%. Dynamic coverage is incomplete. Sampled stacks containing
`_grid_rung` fell from 10,064 to one, and parser stacks from 8,778 to one.

The initial comparison invocation ended Failed after both complete file runs
passed. Three hashes differed because the recorder hashed mapping values for
key-membership assertions. The CLI metadata contains measured `solver_seconds`,
which the original test already removes in its separate content comparison.
The recorder now observes the complete key set for membership in a builtin
`dict`, preserving the actual operand and operator. Equality still records
complete values. Full canonical membership containers remain in a separate
audit ledger, so this projection can be inspected rather than hiding fields.
This change applies only to the diagnostic recorder, not the tests.

Corrected full-case comparison, terminal OK, 11 CAS artifacts verified:
https://app.buildbuddy.io/invocation/c19e81f5-e517-4bc6-a4f3-61030b9f7b29
All 25 original cases passed on both revisions. All 114 original source
assertions executed; 228 asserted operand records and 45,324 evaluations
matched. The complete-container audit compared all 12 dictionary-membership
observations and found changes only in unasserted `solver_seconds` fields.
The compressed full ledgers and `audit_membership.py` are retained here.

Related price-domain, workload-bound, lane-protocol, allocator-projection and
expert-projection tests: 79 passed, four existing skips. All four skip because
`/home/rob/venvs/pq2131-producer-b770727c-20261003/bin/python` is absent in the
remote VM. They exercise the separate real projection producer, not the
changed priced-unit selector. Exact skip names and reasons are retained in
`verification-summary.json`. Compilation of the changed source and test passed.

## Reproduction and limits

Run tests and profiling only on an execution backend allowed by Rob's policy.
Submission scripts are retained under
`/home/rob/tmp/bazel-wave2/pq/notes/heavytests/run/`. Every runner step starts
with `unset BUILDBUDDY_API_KEY`. The coordinator's API key is used only in
its API authentication header. Artifacts are scanned before upload and after
retrieval; CAS receipts check their SHA-256 and byte sizes.

The cold before and after runs use separate fresh x64 Ubuntu 24.04 VMs,
four requested CPUs and 16 GB RAM. Native thread counts are one. The image is
`gcr.io/flame-public/rbe-ubuntu24-04@sha256:5f0c01e5f6d79e58d7a996330de33aa38ce599bf8431eaf27521e7602daadf50`.
Dependencies follow the checked-out gate constraints and pinned Tessera resolver.
The environment reports the optional PrismaBuild SDK absent; no target case
requires it. Fleet Netdata is unavailable in these free remote VMs, so this
comparison establishes neither fleet load nor GPU performance. This is one
cold pair without a noise bound.

In a disposable remote checkout with the pinned test dependencies:

1. Scrub the key at the start of each step and bound native threads.
2. Put this directory and the checkout on `PYTHONPATH`.
3. Set `PQ_TIMING_REPORT`, then run the whole file in a fresh process with
   `python -m pytest -q -p timing_probe tests/test_tessera_per_unit_rates.py
   --junitxml=RESULT.xml`. Reduce the retained `timing.json` with `summarize.py`.
4. Set `PYTHONHASHSEED=0` and `PQ_ASSERTIONS_OUT`. Instrument the file with
   `assertion_probe.py instrument tests/test_tessera_per_unit_rates.py`.
5. Run the whole file with `python -m pytest -q -p assertion_probe
   tests/test_tessera_per_unit_rates.py --basetemp=OWNED_TEMP_DIRECTORY`.
6. Restore the source and repeat on the other revision with the same explicit
   temporary path. Run `assertion_probe.py compare BEFORE.json AFTER.json`.
7. Scan all artifacts before upload.

The recorder retains original objects and short-circuit behavior, records
canonical operand hashes and evaluation counts for every original case,
and refuses unsupported values. Builtin dictionary membership records all
keys, while equality retains complete values. It does not consume iterators. Sampling
runs independently from assertion recording. Static source preservation is
reproducible without executing tests:

```sh
python docs/experiments/3215-priced-unit-formats/source_preservation.py \
  9c8beebde2266c78ea5910bc83b30e90cb669d74 tests/test_tessera_per_unit_rates.py
```

Inline review checked the per-call memo lifetime and unchanged marginal loop.
Independent review remains for the lead because this worker was instructed
to start no subagents. Documentation commits leave measured source and tests
unchanged.

Audit the retained full dictionaries without executing tests:

```sh
python docs/experiments/3215-priced-unit-formats/audit_membership.py \
  docs/experiments/3215-priced-unit-formats/assertions-before.json.gz \
  docs/experiments/3215-priced-unit-formats/assertions-after.json.gz \
  /tmp/priced-unit-membership-audit.json
```

## Delivery dependency and loader-axis correction

This PR is stacked on #3255. The initial hosted gate passed all tests but
rejected `test_with_no_contract_pinned_the_axis_leg_adds_no_refusal` at
30.909 seconds against its 30-second budget. The initial hypothesis was that
#3255's immutable-verdict-copy fix would reduce that research-menu cost.

The follow-up measurement does not support that hypothesis. In invocation
`4e27bb2a-d6a2-435c-8242-9b1ac0db7f66`, the unchanged node took 20.097
JUnit seconds on main and 20.149 on the combined source. Call phases were
19.632 and 19.777 seconds; setup was 0.004 and 0.003 seconds. Of 2,070 and
2,053 sampled stacks, 1,257 and 1,258 included exact footprint calculation;
only 13 and 16 included `deepcopy`. These samples identify work but are not
CPU-time percentages. The later green hosted gate does not prove the budget
flake is fixed. No loader-axis test or budget changes here.

The combined-source full-case proof ended OK: all 25 original cases, all
114 assertions, 228 asserted operands and 45,324 evaluations match against
#3255's head. Only unasserted `solver_seconds` container values differ.
The related files passed 79 tests with the same four missing-interpreter
skips. All 19 downloaded artifacts match their CAS identities.

The `stack-*` files retain the complete assertion ledgers, membership audit,
phase and sampling summary, verification summary and CAS receipts. The
original cold figures above isolate the parser memo against main; do not
sum overlapping shared-code savings. Each stack layer retains one changed
test file in its own diff. Final hosted validation is in the PR body.
