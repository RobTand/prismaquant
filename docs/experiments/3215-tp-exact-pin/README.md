# Runtime verdict tuple traversal

The cold TP exact-pin file passed all 14 original cases in 223.79 seconds
on baseline `69eaea55a201355ab02f4cad646efcf739acd914`. Its JUnit duration
was 223.743 seconds, with 223.326 seconds in pytest phases. Fixture setup
accounted for 0.085 seconds. Of 21,798 sampled stacks, 17,050 included
`RuntimeContractSnapshot.native_rung_verdict`, and 17,110 included tuple
copying. Sampling locates repeated work; its counts are not CPU-time percentages.

The cold optimized file passed all 16 cases, including the two added
regressions, in 59.68 seconds in the pytest summary. These two summary
measurements differ by 164.11 seconds. At the frozen planner selection rates
of 70% static and 21% observed dynamic, the measured reduction projects to
114.877 and 34.463 seconds per merge. Dynamic coverage is incomplete.
This is one cold pair, without a noise bound or cross-machine load series.

## Copy behavior

A verdict still comes from the same native-cell query and still receives a
defensive deep copy. The copy memo retains six parsed integer-tuple fields:
`rungs`, `rungs_q256`, `covered_rungs_q256`, `run_tables`,
`launch_rungs_q256`, and `launch_covered_rungs_q256`.

The parser constructs these from validated integer rungs and integer run
tables. Ordinary deepcopy already returns each immutable tuple unchanged,
after traversing its elements. Retaining those identities in the memo avoids
that traversal without changing the resulting object graph. Verdict and cell
objects still receive copies, and mutable predicate values still receive
independent copies. The memo is local to one query. Pin checks, context
selection, evidence gates, shape predicates, and numerical accounting retain
their existing owners.

All 24 original test-module statements and all 25 original assertions remain
unchanged and in order. The new traversal regression first requires native
BF16 R1024 admission and an equal verdict, then requires zero immutable-tuple
visits. A second regression changes a returned predicate list and requires the
next native E2 R896 query to retain the original list.

## Evidence

- Cold before, terminal OK, 14 passed and zero skips:
  https://app.buildbuddy.io/invocation/8c30db51-b5cd-49ea-b4fa-36660ef332fb
- Initial diagnostic, terminal Failed: the E2 witness was native but had no
  run tables, so it failed its fixture precondition before counting copies.
  This is not the accepted red counter evidence. The ownership control passed.
  https://app.buildbuddy.io/invocation/3618c480-9652-4a7e-91ab-e5fe97009844
- Corrected red commit `f8faa1f430242bcd1b0493ba52cdb245d775266e`, terminal
  Failed as intended: native BF16 admission and verdict equality passed, then
  the counter saw ten traversals, including a 3,329-entry coverage tuple,
  instead of zero. The ownership control passed.
  https://app.buildbuddy.io/invocation/aa3fd711-e3ff-4716-829a-b2e89e37e2a6
- After profile and full-case comparison, source commit
  `5bd2e0e0d5668a546cc8b4da7a043d25a3d185ad`:
  https://app.buildbuddy.io/invocation/0df60c50-a361-440b-ae9b-a43abbf090c1
  Terminal OK, 15 CAS artifacts verified. Cold: 16 passed in 59.68 seconds,
  59.645 JUnit seconds. Full-case equivalence: all 14 original cases,
  65 recorded assertion operands and 5,187 evaluations identical. Related
  runtime-snapshot, admission, contract, lane and loader-axis tests: 88 passed
  in 25.373 seconds, zero skips, failures or errors.

The before and red artifact receipts and baseline profile summary are retained
beside this document. Receipt validation checks each CAS SHA-256 and byte size
and scans raw and decompressed artifacts for the local API key without printing
it. The after measurement includes both new regressions. No original case,
assertion, parametrization, or budget changed.

## Reproduction

Run tests and profiling only on an execution backend allowed by Rob's policy.
The coordinator's submission scripts are retained under
`/home/rob/tmp/bazel-wave2/pq/notes/heavytests/run/`. Every runner step begins
with `unset BUILDBUDDY_API_KEY`. The API key remains on the coordinator and
appears only in its API authentication header.

Both cold runs use fresh x64 Ubuntu 24.04 VMs, four requested CPUs, 16 GB RAM,
and native threads limited to one. The pinned image is
`gcr.io/flame-public/rbe-ubuntu24-04@sha256:5f0c01e5f6d79e58d7a996330de33aa38ce599bf8431eaf27521e7602daadf50`.
Dependencies follow the checked-out gate constraints and pinned Tessera resolver.
Fleet Netdata is unavailable in these free remote VMs. These measurements
establish neither fleet load nor GPU performance.

In a disposable remote checkout with the pinned test dependencies:

1. Remove `BUILDBUDDY_API_KEY` before the runner step and bound native threads.
2. Put this directory and the checkout on `PYTHONPATH`.
3. Set `PQ_TIMING_REPORT` and run the whole file in a fresh interpreter with
   `python -m pytest -q -p timing_probe tests/test_tessera_tp_exact_pin.py
   --junitxml=RESULT.xml`.
4. Reduce the retained `timing.json` with `summarize.py`.
5. Set `PQ_ASSERTIONS_OUT`, instrument the test file with
   `assertion_probe.py instrument tests/test_tessera_tp_exact_pin.py`, and run
   the entire file in a fresh process with `python -m pytest -q -p assertion_probe
   tests/test_tessera_tp_exact_pin.py --basetemp=OWNED_TEMP_DIRECTORY`.
6. Restore the test source, repeat on the other revision with the same explicit
   temporary path, and run `assertion_probe.py compare BEFORE.json AFTER.json`.
7. Scan all artifacts for credential material before upload.

The assertion recorder retains original objects and short-circuit behavior.
It records canonical operand digests and evaluation counts for every original
case, and refuses unsupported values. It does not consume iterators. Timing
uses the sampler alone, independently of assertion recording.

Check source preservation without executing tests:

```sh
python docs/experiments/3215-tp-exact-pin/source_preservation.py \
  69eaea55a201355ab02f4cad646efcf739acd914 tests/test_tessera_tp_exact_pin.py
```

Inline review checked the parser-owned tuple types, deepcopy identity behavior,
and preservation of mutable predicate copying. Independent review remains for
the lead because this worker was instructed to start no subagents. Later
measurement-documentation commits leave the measured production source and
tests unchanged.
