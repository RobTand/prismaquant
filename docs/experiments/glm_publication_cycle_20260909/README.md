# Publication-cycle qualification, 2026-09-09

The full-cycle harness drives the ordinary campaign CLI with a bounded prefix
of the original first-round batch list. It preserves original selected-row
residency, source/capture guards, scoring, file publication, receipt readback,
checkpoint publication and final return. Measurement artifacts are partial
evidence, never a completed model or pricing row. Its implementation compiled
through PB `834c402b572cb75d10cfa97ad2787c26da584ad2e9a9402029a6251f82576f8a`;
the actual empty compiler output, exit, cleanup, receipt and executed source
were checked. Source differed only by the PB closure. Full native publication
qualification remains pending at this checkpoint.

## Retained recurrence counter diagnostic

PB `96e5dd4e55b6000974d689d76b4e2263da3e6581c3d0b3052eb7093f20ed4b1e`
ran on Sparky with the qualified torch 2.13+cu130 producer container and frozen
b1eb1dccc source. It used the retained actual R4/B16 residual, 4096×384,
SHA256 `3ae53d839ad565ce15871556fee432cda5a7606ad55b79e6472dd040cb40370d`.
Nsight Compute 2025.3.1 profiled four middle graph nodes, ten replay passes
each, while the complete call retained exact states and SSE 38,035,888.0.

The 64,4,2 tile launched 1,536 blocks on 48 SMs, 64 threads/block, 40
registers/thread, no register spills and 1.33 waves/SM. Achieved occupancy
averaged 73.34%; approximate SM throughput was 31.36%, L1 throughput 58.41%,
L2 throughput 16.83% and memory busy 24.24%. These are intrusive replay
counters, **not unprofiled speed or saturation measurements**. Clock and cache
control were disabled to preserve the workload's conditions; NCU warns of
possible inconsistency. One reported L2 hit rate exceeds 100%, which must not
be read as a physical rate. Instrumented durations are not comparable to the
ordinary call timings.

Native result, report, raw CSV, summary, both-host observation, recovered power
and root CAS/source verification live under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02/performance-best-r4-ncu-01/`.

## Smaller tile hypothesis: retain the negative result

A 32,4,2 tile was screened against 64,4,2 on unchanged actual residuals and the
same immutable producer, with two warmed plans and three ABBA cycles of
five-second blocks. Every block preserved exact states/SSE and built zero
plans. Complete before/after traces have one graph, 4,095 steps, one traceback
and zero overlaps. No dynamic profiler toggling was used.

| Actual call | Shape | 64 tile ms/call / mean W | 32 tile ms/call / mean W | Throughput | Work/J |
|---|---|---|---|---|---|
| R3, Sparky | 4096×192 | 34.2640 / 88.90 | 32.4469 / 93.41 | 1.05600x | 1.00506x |
| R4, Sparklina | 4096×384 | 58.0897 / 89.97 | 63.6024 / 94.73 | 0.913326x | 0.867469x |

The R3 energy difference is too small to call a decisive improvement from
this screen. R4 becomes slower and less efficient despite drawing more power.
The grid-wave hypothesis therefore does not justify a blanket tile change.
The 64,4,2 opt-in remains the publication experiment's control. Power here is
for repeated calls on one retained residual, not full GLM encoding.

PB native keys: R3
`b633da0827937d70eea27563cf8ce15245cd871a604f2239a70a67b8b993abd4`, R4
`f6b880773d59bf18e6d0333b5775ac9921ae7026d5878be07fc042544664c25c`.
CPU audit `b6563acf4f6419a51e8c947d8f3840d9555b6e5b8bea6a2747cff8d2c9e9d557`
checks actual CAS output, input hashes, all 12 blocks per rate, complete traces
and fully bracketed 2 Hz power integrals (gaps below one second). Both-host
Netdata has 13 samples per host in each run; the other rate was concurrent on
the other host. No cross-host comparison is implied. Root separately checked
native/audit terminal exits, cleanup, payload/receipt hashes and source
snapshots. Trace durations/dependencies are checked; the harness did not
record individual profile wall brackets, so none are claimed.

Evidence under the preceding shared preparation root:
`performance-actual-r{3,4}-tile32-ab-01/`, `root-actual-tile32-audit.json` and
`root-actual-tile32-cas-source.json`. The PB manifest is
`actual-tile32-screen-campaign.json`. Native/audit source matches retained
commit 4e14b58f35 apart from PB closures and the subsequently added audit file.

## Publication correctness intake and retained failed run

The final upstream cleanup at `4ef7ef9654`, integrated with the merged b1
producer pin in root `d6f99fa01d`, passed 51 focused CPU tests with no skips.
They covered publisher behavior, the actual campaign CLI, architecture and
staleness. Four independent PB shards used the scoped b1 producer CPU
Python environment on dl380g10; actual terminal exits, cleanup, CAS payloads,
receipt hashes and source snapshots were verified. Source differences were
only PB closure files. Evidence:
`/mnt/shared/tessera-measurements/glm-publication-final-cleanup-tests-20260909{,-cas-source}.json`.
The broader earlier integration had 120 passes and one CUDA-only skip; final
upstream regression had 209 passes and three skips in its older CPU producer
environment. These are CPU policy/correctness results, not native performance.

Seven deliberately damaged implementations failed their intended tests.
Root fetched their actual CAS source bundles and read failed terminal output.
M1 forces synchronous publication despite a publisher; M2 journals before
publication; M3 disconnects the publisher from the campaign call; M4 removes
unwind journalling; M5 admits oversized jobs; M6 stages before reservation;
M7 makes unwind flushing conditional on newly completed receipts, losing
previously applied dirty rows. M7 fails specifically in the eager-completion
case. The initial receipt index mislabeled M1 and swapped the meanings of M5
and M6; actual patches take precedence. Failed actions have no successful CAS
receipt: their logs are retained terminal output, while their source bundles
are CAS hash checked. Full keys, patches and logs:
`/mnt/shared/tessera-measurements/glm-publication-mutants-20260909-cas-source.json`.

Native PB `1e1f6decc655af9327d57e54afd6f6b00a4a5ddc7470809394dea514eefa5307`
on Sparky ended with exit 1 after the first synchronous warm arm produced 64
actual wires and matching journal scores. The next arm refused the checked
phase plan against its remaining memory budget. There was no cgroup OOM,
and cleanup completed. The first arm's 188.389 seconds includes warm-up and
is **not an A/B result**. Its source was `b8f2c3f66d`, before final cleanup.
Retain `performance-publication-cycle-r1088-ab-01/` as bounded failure evidence.

The comparison harness now records the process/cgroup/CUDA floor before and
after Python collection and unused CUDA allocator cleanup between arms,
outside timed intervals; cached live plans remain. It records admission and
completed guard snapshots and releases the loaded cost payload before the
next arm. The primary trace is preserved after either warm profile so a later
failure does not discard the first trace. The safety guard is unchanged.
The retry reserves 104 GiB, including 6 GiB over the rounded 98 GiB phase plan
for the observer and retained state; 110 GiB was refused before submission
because both fleet workers declare 104 GiB. No workload ran for that refused
submission. Native R1088/64-unit and R832/32-unit A/B qualification is pending.

## Completed qualification and disposition, 08:06 UTC

The final PR465 head `83d265aa560f54e8430e7d8da4c0ea783fccb1b7`
passed full CPU CI: 7,352 passed, 204 skipped, three xfailed and 192 subtests
in 1,265.05 seconds. Root retrieved the actual job log, SHA256
`d33a48df4a7cbd33e40cf8590dce21f0b2a55dd84112d11be4755a179f2620f9`;
`/mnt/shared/tessera-measurements/glm-publication-pr465-ci-20260909-audit.json`
binds the job and head. CUDA is covered separately by the bounded native runs.

The R832 retry completed all six arms over the same fully resident 864-unit
row, with 32 actual encoded units per arm, B8 and the qualified 64,4,2 tile.
All 192 wires and scores matched exactly. Two unprofiled synchronous arms took
237.048103 seconds total; the interleaved asynchronous arms took 231.405995
seconds: **1.02438x throughput** for this bounded full publication cycle.
GPU energy was 12,346.45 J versus 12,282.54 J (1.00520x work/J), too small a
difference to claim decisive energy improvement. Mean GPU power was 52.08 W
versus 53.08 W. Main-thread publication time fell from 3.72–3.75 seconds to
0.105–0.123 seconds. Capture hashing still dominates the first prepare at
43.77–47.11 seconds; this is measured separately in the next fix.

Before/after CPU/CUDA traces, 443 Netdata samples on each host and 4,473 power
samples accompany the result. CPU audit verifies actual files, receipts,
checkpoint scores, queue budget/order and complete recurrence traces with
one graph and 4,095 steps. PB native:
`55a9fbd847d34329793c1eb04e3b8a95b392a55ea298d28c1554d4d426b64056`;
audit: `3c2158f07380a41b59bf23136abd2c9ca129904aa970b048a828b9dc19c8fedd`.
Evidence: `performance-publication-cycle-r832-ab-02/` and
`root-publication-lower-audit.json` under the preparation root above; root
receipt/source checks are `glm-publication-lower-{native,audit}-cas-source-20260909.json`
under `/mnt/shared/tessera-measurements/`.

The R1088 retry completed four arms (64 wires/scores exact) before the next
arm refused the memory phase guard. Its one completed unprofiled A/B pair
was 177.695322 versus 173.413246 seconds, 1.02469x throughput. Energy was
8,924.49 versus 8,905.47 J, again inconclusive. This is **partial evidence**,
not a completed ABBA comparison. No OOM occurred and cleanup completed.
PB native failed action:
`92f904bb18f41dc68f0c1a46146e2fa8c9d10431819ad346ae30b60f16df9a3e`;
completed-prefix audit:
`d2544955fe4817846690543fe2f4f720352f1e6008bbf8230d3127e6502a36f1`.
The audit deliberately verifies the failed terminal output instead of claiming
a successful native receipt. Evidence: `performance-publication-cycle-r1088-ab-03/`,
`root-publication-upper-prefix-audit.json`, and
`/mnt/shared/tessera-measurements/glm-publication-upper-{failed-native-source,audit-cas-source}-20260909.json`.

Between-arm diagnostics identified retained unused pinned-host allocator pages.
The final R832 harness releases these outside measurement, while preserving
live plans and the production guard. Earlier failed runs remain failure
evidence. No further publication tuning or R1088 retry is planned: the gain is
modest, and the default remains synchronous (`--publication-overlap-bytes=0`).
The experiment does not establish full-model throughput or served quality.
