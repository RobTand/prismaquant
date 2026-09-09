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

Status: comparison prepared; native results pending. Earlier batch-width
profiling that failed its trace-size cap remains negative evidence and is not
used as an A/B result here.
