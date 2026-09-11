# GLM recurring publication pause

The live campaign showed repeated 3–5 second low-power intervals inside a
continuously running action. A 120-second `/proc` observation aligned with the
existing power recorder measured 19.9% of Sparklina's interval and 16.7% of
Sparky's below the descriptive 45 W threshold. Each complete interval carried
roughly 154 MiB of writes and 26 MiB of readback. CPU activity continued during
the pause. These observations identify a batch boundary, but do not establish
which function consumes it or equate CPU activity with useful CPU work.

Evidence: `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/live-gap-review-20260909-1958/`.
Both hosts' Netdata series cover the observation interval. The issue record is
RobTand/prismaquant#283. GPU utilization percentage is not a saturation measure
on GB10; no speedup is claimed from these observations.

The controlled comparison uses the existing campaign's opt-in bounded
publication writer, budgeted at 256 MiB, against its synchronous default. It
retains the complete selected expert group's resident source weights and
original H/X capture. A stable permutation of complete compatible batches
places the gate/up shape first; each arm measures the same bounded 64-anchor
prefix. This is explicitly a prefix measurement, never a completed cost table.
No production scheduling, cache, encoder, capture or numerical code changes.

The four fresh-process arms run synchronously, overlapped, overlapped,
synchronously under one PrismaBuild measurement admission. Compiler caches
are private per arm; shared filesystem cache remains part of the environment.
Each arm records the same CUDA profile window, main-thread stacks, nested
publication/identity/checkpoint wall and thread CPU spans, and both hosts'
Netdata series. Phase spans are inclusive and may overlap across threads;
they cannot be added as independent elapsed time. The first profiled batch is
excluded from steady-state timing. Power integration uses the existing host
recorder, aligned with each arm's actual timestamps.

Acceptance requires exact wire bytes, wire receipts, checkpoint identity and
scores across arms, completed journal state for every measured anchor, a
reduction in steady-state elapsed time and joules, and a verified admission,
exit, cleanup and CAS receipt. Any campaign rollout must preserve the frozen
source and original capture, account for staging memory and resume only through
the existing checkpoint gates. Production defaults remain synchronous unless
broader qualification or an explicit decision changes that contract.

Initial status: comparison prepared; native results were pending. Earlier batch-width
profiling that failed its trace-size cap remains negative evidence and is not
used as an A/B result here.


## Completed comparison and disposition, 2026-09-09

All four arms completed with identical 64-anchor wires, receipts, checkpoint
identity and scores. After excluding the first profiled batch and including
final checkpoint completion, the 56-anchor steady intervals were:

| Saving mode | Seconds | GPU joules | Time below 45 W |
|---|---:|---:|---:|
| Synchronous A0 | 143.497 | 8611.08 | 20.40% |
| Background B1 | 137.150 | 8487.03 | 16.66% |
| Background B2 | 137.019 | 8468.12 | 16.22% |
| Synchronous A3 | 142.836 | 8561.60 | 19.20% |

Mean throughput improved 4.44%; GPU energy per anchor fell 1.27%. The measured
work remains dominated by the producer; background saving removes only part
of the recurring pause. The 45 W threshold describes the trace, not GPU
idleness or saturation. The existing 0.5-second power recorder covered every
steady interval with a maximum sample gap of 0.501 seconds. Both hosts' Netdata
series and main-thread stack samples accompany each arm. Main-thread samples
still show tensor hashing, wire bit packing and producer column construction;
those samples do not attribute every remaining pause to those functions.

Evidence root:
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/publication-gap-optimization-02/`.
`steady-summary.json` SHA256:
`c67b650c839e6d89a9fbb0ee572982388950733df60a7e12547d77c253206545`.
Native action:
`cb9ee1ab995a968f8b07af0e11f5b6f4af2c08144a639b821cbcc4dda418fc67`.
CAS receipt:
`922748ed9a64687409b64727f8d533bfc22e73be9debac1c4a12352708fe8652`.
Exit 0, completed resource cleanup, receipt and payload hashes, actual source,
all four observer results and produced output were checked. The native
snapshot differs from benchmark commit `ec980688ec50` only by PB's closure
record. CPU validation passed 43 tests with no skips; its combined action
failed the original preparation memory guard. The corrected preparation and
compile action succeeded separately. See `cpu-evidence.json` for both outcomes.

Rob explicitly deferred further optimization and requested main-campaign
resumption. The remaining work is recorded in
[PrismaQuant #283](https://github.com/RobTand/prismaquant/issues/283).
The 34-row overlap candidate manifest was prepared with existing resource
accounting, but never submitted or activated. Main-campaign actions retain
their original configuration, frozen production/producer sources and original
512×512 capture. At resumption, both GPU workers had original campaign actions;
81 actions were audited complete and 49 were ready. Prepared manifests and
profiles are retained as bounded evidence for the later fix. No full-model KL,
serving measurement or six-variant export is claimed by this comparison.
