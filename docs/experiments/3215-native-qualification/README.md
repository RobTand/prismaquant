# Native qualification read cost

The cold legal-domain file spent 741.747 seconds in pytest phases on the
baseline `f1335b0333749c1f400934b9e0f6ac06a99cf706`. It passed 62 tests and
skipped the existing check for the unavailable GLM config mount. Of 97,182
sampled stacks, 78,415 included `installed_runtime_snapshot` and 62,708
included `native_qualification_set`. The walk captured the installed contract separately for every cell query.
Sampling counts locate repeated work; they are not CPU-time percentages.

The qualification walk now captures one admission runtime and passes it to
the existing route-admission function for every original cell and serving
context. Each later walk captures current bytes again. Empty cell populations
avoid capture, and capture failures retain the empty qualification result.

## Reproduction

All execution belongs on BuildBuddy or another execution backend allowed by
Rob's policy. These tools do not submit work or supply credentials. The
retained submission driver and scripts are under
`/home/rob/tmp/bazel-wave2/pq/notes/heavytests/run/` on the coordinator.

In a disposable remote checkout with the pinned Tessera and test dependencies:

1. Remove `BUILDBUDDY_API_KEY` from the runner environment before each step.
2. Set native thread limits to one and use at most 16 concurrent workers.
3. Put this directory on `PYTHONPATH` together with the checkout.
4. Set `PQ_TIMING_REPORT` to an artifact path. Run the complete file in a fresh
   process with `python -m pytest -q -p timing_probe
   tests/test_tessera_legal_domain.py --junitxml=RESULT.xml`.
5. Run `summarize.py` against the downloaded artifact directory containing
   `timing.json`. This only reduces recorded measurements.
6. For equivalence, set `PQ_ASSERTIONS_OUT`, run `assertion_probe.py instrument
   tests/test_tessera_legal_domain.py`, then run the complete file with
   `python -m pytest -q -p assertion_probe tests/test_tessera_legal_domain.py`.
   Use the same explicit `--basetemp` for both revisions and fresh processes.
7. Restore the instrumented file. Repeat the equivalence step for the other
   revision. Run `assertion_probe.py compare BEFORE.json AFTER.json`.
8. Scan every artifact for credential material before upload.

Check source preservation without executing tests:

```sh
python docs/experiments/3215-native-qualification/source_preservation.py \
  f1335b0333749c1f400934b9e0f6ac06a99cf706 tests/test_tessera_legal_domain.py
```

The recorder wraps evaluated assertion operands and returns the original
objects. It preserves chained comparisons, boolean short circuiting, and
`all`/`any` generator evaluation. It never consumes an iterator to record it.
Values are canonicalized and streamed into per-assertion SHA-256 digests,
with evaluation counts. Comparison requires every original node, phase
outcome, operand digest, and evaluation count to match. Unsupported operand
types make comparison fail. Source assertion ASTs are checked separately.

Timing runs use the sampling plugin alone. Operand recording runs separately.
The sampling interval is 5 ms, with setup, call, teardown, and collection
recorded by phase. Fresh remote VMs use the same pinned image and dependency
setup for before and after. Their recorded dependency freezes are identical.
Fleet Netdata is unavailable inside these VMs;
these measurements do not establish fleet load or GPU performance.

## Evidence

- Before: https://app.buildbuddy.io/invocation/bf0ce335-298b-4f23-a4ed-d1ff3c74e59e
  Terminal OK, 62 passed, one existing skip, 742.246 seconds JUnit file time.
- Red: https://app.buildbuddy.io/invocation/58035f6c-4e14-441d-9acc-026c9588f45c
  Commit `968b5f965aac51e9978746b2933f04ec857e6c12`. The two real pinned
  triples matched, then the capture-count assertion failed with `2 == 1`.
  The capture-error refusal control passed. Terminal Failed as intended.
- After and full-case equivalence:
  https://app.buildbuddy.io/invocation/71abbbe1-eed3-426c-ae0d-58f6299ae3a5
  Source commit `73e9c040151226bb164b23350e88bf033434169c`.
  Cold file: 64 passed and the same existing skip, 693.379 seconds JUnit
  time and 692.874 seconds of pytest phases. Both instrumented full-file
  runs also passed. The invocation ended Failed because the recorder did
  not support a `dict_values` membership operand. The recorded 394 operand
  digests and 58,051 evaluation counts matched, but that unsupported value
  prevents an equivalence claim. Direct callers were not reached.
- Corrected full-case equivalence and direct callers:
  https://app.buildbuddy.io/invocation/35f76834-5b01-4d3a-8c83-d373ebe6aa87
  The recorder now snapshots dictionary values without consuming an iterator.
  Its terminal verdict and artifacts carry the corrected comparison and
  direct-caller results. The current result summary is in
  https://github.com/RobTand/prismaquant/pull/3242.

The cold JUnit reduction is 48.867 seconds. At the frozen static selection
rate of 70%, this measured reduction projects to 34.207 seconds per merge.
The incomplete dynamic selection gives 10.751 observed seconds per merge.
These use the measured delta directly; the historical ranking uses older
alone timings. This is one cold pair on identically configured remote VMs,
with no cross-machine load series available.

After sampling recorded 72,857 stacks, of which 12,925 included
`installed_runtime_snapshot` and 12,445 included `native_qualification_set`.
The remaining work includes snapshot verdict deep copies, rate schedules,
and repeated live-pin checks. This change removes one measured repeated
read; it does not eliminate all legal-domain cost. Per-test phase totals
and both sampling summaries are retained beside this document.

All 133 original source assertions are unchanged. The added regression also
changes the contract bytes between walks and requires the next walk to refuse
the unpinned document. The original full qualification-roster and inventory
checks remain in place.

## Runtime-weighted order

The frozen planner replay covers 100 merges. Costs below use its selection
masks and alone runtimes. Static and dynamic totals reproduce
`9986.22663` and `5563.24853` seconds per merge respectively, including
the planner's `0.231`-second imputation for its two unmeasured files.
The retained `rank.py` generator is beside the planner notes.

| Test file | Static s/merge | Dynamic observed s/merge |
|---|---:|---:|
| `test_tessera_legal_domain.py` | 494.760 | 155.496* |
| `test_run_pipeline_additivity_identity_audit.py` | 190.540 | 106.158 |
| `test_tessera_menu.py` | 157.780 | 96.922 |
| `test_tessera_tp_exact_pin.py` | 155.610 | 46.683* |
| `test_select_validated_frontier.py` | 140.000 | 72.000 |
| `test_run_pipeline_frontier_identity_audit.py` | 135.240 | 77.280 |
| `test_native_receipt_table_routed.py` | 98.770 | 74.783 |
| `test_pq1588_predispatch_packet.py` | 87.430 | 43.715 |
| `test_tessera_per_unit_rates.py` | 82.944 | 27.648* |
| `test_quality_prefill_dry_run.py` | 70.210 | 30.090 |

*Incomplete dynamic coverage. The observed value is the lower bound and
static selection is the upper bound for that file. These are replay costs,
not new measurements of the optimized implementation.

Frozen inputs on the coordinator are under
`/home/rob/tmp/bazel-wave2/pq/notes/plan/`, with alone timings in
`notes/inventory/alone.tsv`. The plan revision is
`f1335b0333749c1f400934b9e0f6ac06a99cf706`.

## Import-leak candidate added by the lead

The follow-up candidate is
`tests/test_prismabuild_import_leak_1281.py::test_source_consumers_restore_the_public_plugin_import_graph`.
Its required base is #3223, branch `prismaquant-3216-pbsdk`, or main after
that PR merges. The observed branch head when recording this candidate was
`fd6f39e8868c5ff0b3aa8ad9f6f8428c4ad39a88`.

The lead supplied 92–116 seconds on hosted x64 and 59 seconds on arm64.
The 232-second budget is a limit, not a measured runtime. In the frozen
replay its file was selected on 27 of 100 merges, statically and dynamically.
Applying that historical selection to the supplied x64 node runtime gives
24.84–31.32 seconds per merge. This is a provisional additional candidate,
not a replacement for the frozen file measurement of 9.1 seconds, which
lacked SDK-dependent execution. Selection at the new base is unmeasured.

A 16-second control omitted seven of the fourteen groups. It establishes no
equivalent after time. If consolidating interpreter isolation preserved all
fourteen groups and reached that time, the conditional saving would be
20.52–27.00 seconds per merge. Proven savings are still unknown. The
comparison ceilings for the originally assigned TP exact-pin and per-unit
files are their full historical costs, 155.610 and 82.944 seconds per merge;
their attainable reductions also require measurement. The candidate stays
behind those two pending equivalent full-case measurements. The original
three-file delivery limit is unchanged.

## Loader-axis fixture candidate added by the lead

`tests/test_tessera_tp_loader_axes.py` is number 52 in the frozen file-cost
ranking. Its 32.1-second alone measurement and selection on 70 static and
31 dynamic merges give 22.470 and 9.951 seconds per merge, respectively.

The lead reports 30.226 seconds of module-fixture setup attributed to
`test_a_whole_unit_is_never_refused_on_an_axis_it_does_not_cut` in #3223's
hosted shard 7, and 14 to 22 seconds in other runs. The first test executed
owns the setup timing; moving that attribution does not remove the cost.
At the frozen selection rates, that setup accounts for 9.800 to 21.158
static seconds per merge and 4.340 to 9.370 dynamic seconds per merge.
These are conditional ceilings on savings from eliminating the setup,
not measured improvements. Full-file cold profiling and an equivalent
after run over every original case are still required.

## Review scope

Inline review checked that the existing admission implementation still owns
every per-cell decision and serving context, and that capture lifetime is
limited to one walk. The original cell roster and test statements remain.
No numerical kernel, format menu, plugin contract, ship gate, or architecture
default changes. The regression uses two independently pinned triples and
changes actual contract bytes between walks. Independent review remains for
the lead because this worker was instructed to start no subagents.

The source and tests used for both measurements are commit
`73e9c040151226bb164b23350e88bf033434169c`. Later commits in this PR add only
measurement notes and reproduction tools.
