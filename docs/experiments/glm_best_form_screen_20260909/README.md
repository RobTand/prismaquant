# Best-form trellis: independently checked evidence

The candidate carries each predecessor class's minimum between steps instead
of storing the full front. Its smaller resident buffers permit a wider
internal column batch under the existing L2 budget. The producer's
`TESSERA_WINDOW_BEST_FORM` switch remains off by default. This note records
the bounded kernel screen; actual GLM encode/decode/pricing validation is
separate and still pending at this checkpoint.

Evidence root: `/mnt/shared/tessera-measurements/window-best-form-ab-20260909`.
Producer worktree: `/home/rob/tmp/tessera-fusion`, branch
`claude/window-viterbi-two-step-fusion`. The reviewed runtime differs from
the frozen original `07ad344c3275bb2fa7ce2432f93d89945d66f4c2` only in
`src/tessera/window_viterbi.py`.

## Correctness and source verification

Six PB GPU actions passed 184 tests, zero skipped or uncollected. Of these,
132 tests actually allocated on CUDA; the remaining tests do not establish
additional device coverage. Environment: GB10, torch 2.11.0+cu130.

| Test module | Passed | Allocated CUDA | Action prefix |
|---|---:|---:|---|
| test_window_viterbi_best_form.py | 66 | 59 | 0d25f0b35e23 |
| test_window_viterbi_fast.py | 52 | 52 | ee400325d0f1 |
| test_window_graph.py | 20 | 20 | de91a9570767 |
| test_window_body.py | 30 | 1 | d6345d0465cc |
| test_audit_doc_claims.py | 10 | 0 | e12d64f0421f |
| test_e4m3_ladder.py | 6 | 0 | 1fa92fa73e0b |

Root checked actual terminal exits, cleanup, CAS payload hashes, source
snapshots and logs. Runtime/test source matches `3f872bad8`; differences are
the experiment harness and PB closure files. Full keys and source comparisons
are in `root-native-tests-profile-cas-source-audit.json`. Best-form tests
compare states and SSE against both existing fused and reference paths,
including weighted inputs, ties, graph replay and partial column batches.

## v3 timing and recovered continuous power

PB `c836406e4a6bf9ca76697db54bd757e5bea2c1eabaaefb2c3b99dffb1fc7ee67`
ran an isolated measurement on Sparky. Same seeded representative targets,
actual E4M3 window table, L=14, arity 1, weights, and 4096 rows. R3 has 192
columns; R4 has 64. Front and best-form states and SSE match exactly.
These are not captured GLM residuals. Blocks run approximately five seconds;
both plans are warmed before interleaved timing.

The original 1 Hz sampler did not retain raw samples and did not enforce
coverage at interval endpoints. Its energy figures are superseded here by
separately recovered continuous 2 Hz pqteld data, with full boundary coverage
and gaps below one second. Both hosts' historical Netdata CPU/power series
are retained under `v3/root-recovered-host-series/`; they are explicitly
separate from the original measurement stream.

| Rate | Front ms/call | Best ms/call | Throughput ratio | Front → best mean W | Work/joule ratio |
|---|---:|---:|---:|---:|---:|
| R3 | 109.5460 | 41.7141 | 2.6261 | 52.3651 → 85.1873 | 1.6143 |
| R4 | 37.5276 | 21.7971 | 1.7217 | 62.1085 → 69.6596 | 1.5350 |

Time is total measured seconds / total calls. Energy integrates the continuous
power series over each actual block interval, then divides total useful work
by total estimated joules. No cross-host or cross-runtime speed comparison
is implied. The 90–100 W mean operating target remains unmet.

The `best@32` attribution arm is **not an isolated internal-width control**:
it changed outer `chunk` from 512 to 32, repeating the final reduction and
traceback and changing SSE summation association. Its states match, but SSE
differs. Do not attribute its delta solely to reduced front stores or derive
a separate width speedup from it. A corrected control is pending.

Verification PB
`9ae94f89b3b35f75fcd24780641961f192c07f35690afb11f4ba33b46f34a449`
compared the external result against actual native CAS stdout and checked
continuous telemetry coverage. The result is
`v3/root-timing-recovered-telemetry-audit.json`. Its first attempt expected
the live allmetrics `idle` dimension in the historical chart and failed;
the corrected reader sums the historical non-idle CPU dimensions, matching
PrismaBuild's existing `box_window` interpretation.

## v3 profiler: R4 retained, R3 overwritten

Profile action
`9d7135d5c659d8344bb9fd4954a483d8903e356020958dd1b7a17c4b37bc8f98`
ran both arms on Sparklina through PB torch mode. The harness exported R3
and then R4 to the same output path. Consequently its attested raw blob
`98cfc5f5df269115db2d77fe34aaa044643d8b98564634faed5753118b902080`
contains R4 only: 251,979 compressed bytes, 11,969,348 decoded bytes,
41,399 events. R3 printed aggregates survive, but its raw trace does not.

Root verified R4 markers, kernel counts, boundaries, duration plausibility
and zero dependent-step overlaps. Front has 8192 steps averaging 4.41875 us;
best has 4095 averaging 4.94479 us. Each has one traceback, 2.0535 and
1.9903 ms respectively. The flawed `best@32` control has two tracebacks,
directly demonstrating its extra epilogue work.

Compiled-kernel measurements report R3 front/best registers 40/40 with no
spills; R4 reports 40/37 with no spills. The smaller logical intermediate
does not establish a universal reduction in compiled register pressure.

Root verification PB
`f7657a545d17c7974bcd3d03b28bc88d627ac41511da8cdf0a37012505fff455`
produced `v3/root-r4-profile-audit.json`, explicitly marked
`R4_PROFILE_VERIFIED_R3_RAW_TRACE_MISSING`. CAS/source checks for both root
audits are retained in `v3/root-verification-cas-source-audit.json`.
Before accepting the mechanism attribution, preserve distinct raw profiles
for both rates and correct the width control. Before adopting the candidate,
measure the actual GLM path in its known-good torch 2.13.0+cu130 container.

## Corrected control and distinct traces, 2026-09-09 04:00 UTC

The corrected control holds outer `chunk=512` for all arms and overrides
only the candidate's internal L2 budget to admit width 32. All three arms
now have identical states and SSE and exactly one outer traceback. The
control isolates the combined recurrence and final-front changes; it does
not measure a single store instruction in isolation.

| Rate | Front/best throughput | Front/best work per joule | Front/best at width 32 throughput |
|---|---:|---:|---:|
| R3 | 2.6280 | 1.5125 | 1.4843 |
| R4 | 1.7196 | 1.5582 | 0.9921 |

These figures come from v6 timing action
`63eae68972d28d587080fd258ce65bbb2e8c536d05b951efac5cdea6839033b6`.
Raw 1 Hz power samples are now retained and bracket every interval. Root
recomputed their integrals, respecting timestamp/energy serialization
precision. This table uses this run's own power series, not the recovered
v3 data above.

Distinct v5 profiles retain both rates: R3 action
`748070a53a9685c9fa92149619b4449a8b949659c87c2a51885606be3ecef64a`
and R4 action
`d2da1454fefaf994a1d992d81837d3a4ec8a33cdf511bee5ad7a37ebfc95ab02`.
Root checked marker boundaries, expected step counts, no dependent-step
overlap, actual CAS blobs and retained file hashes. R3 has front/best/control
step counts 24576/4095/24570; R4 has 8192/4095/8190. The R3 fixed-width
control includes one 193.759 microsecond step outlier; it is retained and
does not violate the execution boundaries or dependencies.

The audit is `experiments/glm_best_form_corrected_audit.py`, PB
`b7dfd2b9e7270e2f49a6021935598747760081befd14e68d1623ed34fe0a2423`,
with result `v6/root-corrected-control-profile-audit.json` and source/CAS
verification `v6/root-corrected-and-real-audit-cas-source.json`.

## Actual GLM pricing, 2026-09-09 04:15 UTC

The fixed-B8 comparison used the first 16 actual layer-4 expert down
projections, shape 4096×2048, at `TESSERA_E4M3_K1_R832`. This artifact rung
mixes recurrence rates 3 and 4; B8 means eight experts per encode call.
The original 864-entry calibration capture was prefetched, with no source
forward execution. Resident source data was 49,526,341,632 bytes. Producer
commit `61da00740c0a47c43319340a030db1f618555a31` has package identity
`e6fe414581b7c51f76c0c3f75fd461365136949339b5082d8fe4de189034f8d4`.
The measured source predates the subsequent empty-input guard; that guard
does not change this positive-input kernel path.

PB action
`569ee819458cb3cf1c91a687ff66f1edbcd51a87a4f31813e36dc9fb2c572c2b`
completed on Sparky in the qualified torch 2.13.0+cu130 container, with
8 reserved CPUs and 104 GiB host memory. Two warm-ups were excluded, then
the arms ran front/best/best/front. All six arms returned identical wire
hashes and pricing scores. Root hashed all 16 actual wire files and also
confirmed exact parity with the earlier original-07ad batch screen.

| Arm | Seconds for 16 experts | Estimated joules | Mean GPU W |
|---|---:|---:|---:|
| Front, mean of two | 77.4095 | 4516.05 | 58.34 |
| Best, mean of two | 37.3955 | 3078.12 | 82.31 |

Throughput improved **2.0700×** and work per joule **1.4671×**. Each actual
arm interval is fully bracketed by the retained 2 Hz pqteld series; maximum
sample gap is 0.501 seconds. Both hosts' Netdata samples are retained.
Candidate power peaks were 91.44 and 91.52 W, but its 82.31 W mean remains
below the requested sustained 90–100 W target.

The CUDA instrumentation is only partially usable. Both front traces have
impossible summed durations and overlapping dependent steps despite fresh
profiler contexts; both candidate windows pass those physical checks.
There is no accepted paired kernel-time attribution from this run. Complete
wall intervals, continuous power, Python stacks and actual output parity
remain independently checkable. A separate complete-call profile without
dynamic collection toggling is pending; the full A/B is not being repeated.

Sampled process `read_bytes` deltas were zero in all four measured arms,
with 7.7–54.3 KB of `rchar`. Writes were 162–260 MB in the candidate arms.
Samples exclude interval tails, and Linux process counters alone cannot
establish zero NFS latency. Startup, warm-up and trace-export overhead are
excluded from the table and remain part of total job cost. This is a bounded
16-expert result, not a completed pricing row or model.

Evidence lives under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02/performance-best-form-ab-01/`.
The audit is `experiments/glm_real_best_form_audit.py`, PB
`12e3377464c00cc51d75e9b9f25a40c81631771dada9e92ec33c921dbcadcc4c`,
result `root-real-wall-energy-parity-audit.json`, status
`LIMITED_CUDA_TRACES_REJECTED`. Raw rejected traces are retained. Pricing
and EXL3 qualifier09 remain paused at this checkpoint; no producer default,
serving pin or completed receipt was changed.

## Complete-call profile accepted, 2026-09-09 04:25 UTC

The replacement capture uses a complete Viterbi call with no dynamic
collection toggling. It intercepts the first real campaign residual after
normal resident prefetch and Hessian preparation: weighted 4096×192 targets,
the actual 16384×1 E4M3 table, L14/R3, outer chunk 512. Each implementation
is warmed three times before its own fresh CPU/CUDA profiler context.
Both return identical states and SSE (70,974,832.0).

| Actual R3 call | Front | Best |
|---|---:|---:|
| Graph replays | 6 | 1 |
| Step kernels | 24576 | 4095 |
| Total step-kernel time, ms | 98.4966 | 34.5666 |
| Whole kernel span, ms | 103.8845 | 37.4797 |
| Traceback count | 1 | 1 |
| Traceback time, ms | 2.15972 | 2.15976 |

The raw traces pass timestamp boundaries, expected step/replay counts and
serial-duration checks. A single front-step overlap is below the audit's
one-microsecond tolerance and negligible aggregate threshold; candidate
steps have none. No graph construction occurs inside either profile. This
establishes before/after attribution on the actual torch 2.13 workload;
the separate full ABBA experiment above remains the source for throughput
and energy. R4 is covered by the earlier distinct representative profile,
not this actual-residual capture.

Native PB action
`c86c6920b91f65c45dc9f25d6a9c19e17a47159314dfb436206ffbabb74364f6`
completed on Sparky with cleanup confirmed. The two retained compressed
traces hash to
`08f429306345f2a1e9d555841167816140107e64d2fb51b6d970f74b36ac69e3`
(front) and
`7590f7fc61c3422d2a5c50153907736c2d5f92a1974a648221aba84c5df2f11d`
(best, also PB's primary profile). The audit
`experiments/glm_real_call_profile_audit.py` ran as PB
`15a5ba58ec04f64841bf6cc280f4ace747e45bdb3e28396a8f6e245f4cae0c99`.
Its `PASS` result and source/CAS checks are
`performance-best-form-call-profile-01/root-complete-call-profile-audit.json`
and `root-native-and-audit-cas-source.json` alongside the preceding evidence
directory. Both hosts' observer series are retained there too.

Final producer test intake also verifies six modules after the empty-input
fix: **209 passed, 156 allocated CUDA, zero skipped or uncollected**. A
separate CPU-reference prototype has 68 passing tests and zero device
allocations. The broader six-action encoder suite has 1453 passed, 5 skipped,
1 expected failure and 404 device-allocating tests, with best-form unset;
it checks the default front path on the earlier source before the empty
guard. The five skips are one unpublished E2M1 reader range and four
unsupported 4-way/8-way column cuts. These receipts do not add candidate
coverage. Full keys, source differences and logs are retained in
`window-best-form-ab-20260909/root-post-empty-fix-tests-cas-source-audit.json`.
The isolated zero-row reproduction caught and printed the pre-fix CUDA
fault while exiting 0; the post-fix output is empty states and zero SSE.
The fault text, rather than the wrapper exit status, is the regression
evidence in `root-empty-regression-terminal-audit.json`.
