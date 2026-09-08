# Adaptive anchor progress and durable retry — 2026-09-08

Fixes [PrismaQuant #428](https://github.com/RobTand/prismaquant/issues/428).
An interior-rung failure in one fused member previously rescheduled all members,
including successful ones, and could repeat without progress when `--max-rounds 0`.
The scheduler now submits only missing member/rung pairs. An adaptive round with
pending work and zero successes flushes its journal and raises a refusal. A retry
with all endpoint anchors already journaled proceeds to adaptive selection and
reuses original successful prices and wire records.

The regression uses two CPU Linear modules and controlled producer failures;
it exercises the real scheduler and journal, not native producer encoding.
Before the fix, PB action `64cd567f15eada1025215be0094e5ae9641c5829b2a3c8762c1f8b4a1c4bfde2`
failed the expected-refusal assertion and showed the already successful member
encoded at the same interior rung in rounds 2, 3 and 4. A test-only round bound
made that negative result terminate. The production campaign receives no new
round bound, sampling limit, deadline or budget reduction.

Final PB CPU validation on `dl380g10`, Python 3.12, native threads bounded to one:

| Check | Result | PB action |
|---|---|---|
| Progress and existing campaign-resume tests, 4 workers | 31 passed, no skips | `2caaba804d2fee3aaac53aef414d2bf94ca7780ab4b7cd939e5958c3d3f26b60` |
| Architecture and documentation-staleness tests, 2 workers | 19 passed, no skips | `c947a6e982aef7f444ba87f0518203f6bc82cb64232885f5a2f136fa8e730d28` |
| Compile touched runtime and regression module | exit 0 | `3596c742d91e8aef33e7b6ba03670c45a64c33544dadca2bf71ad3d573bca5e3` |

Assertions cover permanent rejection, durable partial successes, successful
missing-only retry, a second retry with zero encodes, unchanged successful
refinement, and immediate Hessian/activation contract errors. Existing resume
tests retain their malformed/stale-wire and initial-unpriced-unit coverage.
These are behavioral tests; no GPU throughput or memory improvement is claimed.

Every final action had terminal `done`, exit 0, attributable stdout, and CAS
payload/hash checks passing. Exact tested source bytes for the runtime, new test
and architecture document were independently compared against each sealed Git
snapshot. CAS verification does not assert a verified signer attestation.
Evidence, commands and receipts are retained under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/full-anchor-preparation-01/`:
`progress-cpu-campaign.json`, action-prefix `.stdout.log`, `.pb-action.json`,
`.cas-verification.json`, and `verified-progress-actions.json`.
