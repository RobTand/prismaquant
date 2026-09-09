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
