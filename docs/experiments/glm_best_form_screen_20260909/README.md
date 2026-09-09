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
