# GLM menu pricing through Tessera extents — 2026-09-08

The PrismaQuant footprint accountant uses Tessera’s paired extent builders when
both are present. Older installed producers keep the existing writer-based
path. Exact arithmetic remains owned by Tessera. Runtime pins, menu eligibility,
calibration and wire recipes do not change (PrismaQuant issue #367; Tessera PR #421).

## Paired measurement

The workload is the complete readable menu for a GLM routed gate projection,
shape `(2048,4096)`, TP=1, no parallel split: 5,635 rows. One PrismaBuild CPU
measurement on DL380 ran fresh processes in before/after/after/before order,
with cProfile in every arm. Affinity was `[0,1]`, native threads one, physical
reservation 8 GiB; the action’s observed cgroup peak was 536,391,680 bytes.
The interpreter was CPython 3.12.11. Baseline PrismaQuant is `42e9f19a58`,
optimized source is `341b4345da`; both use the same Tessera producer `af2c20360`.

| Arm | Seconds |
| --- | --- |
| Before 1 | 183.7279 |
| After 1 | 86.3785 |
| After 2 | 85.9021 |
| Before 2 | 182.5634 |

Mean time fell from 183.1456 to 86.1403 seconds (52.97%, 2.126×). All four
serialized menus have SHA256
`6c896618f982c0c0a6a573e5d7c5adc5a745847a5a37f7af65ae6bb7b179ab67`.
SHA256 calls fell from 56,353 (95.039/95.013 seconds of self time) to 5,638
(0.018/0.017 seconds). The remaining hashes have purposes outside placeholder
payload pricing. This is menu construction, not capture, encoding, allocation
quality, served latency or GPU energy efficiency.

Netdata covers the exact 555-second paired window on both Sparks and DL380.
DL380 averaged 1.611% user CPU, 0.411% system CPU and 0.043% I/O wait across
its CPUs, with no swap I/O. Sparky/Sparklina averaged 4.151/3.964 W GPU power;
these GPUs did not execute the timed menu. Their raw swap, memory, CPU and load
series are retained, including background swap reads on the Sparks.

**Retained failure:** action `a24c36f97753` completed all four arms, then exited
one when the Netdata hostname lookup failed. It has no success CAS receipt.
The historical telemetry was recovered separately over the exact arm timestamps
using explicit endpoints (`192.168.1.180`, `.110`, `.107`). Timings were not
repeated. The helper now requires explicit endpoints before starting a paired
measurement, so the source of each host series is reviewable.

## Validation and evidence

The two new consumer regressions failed before implementation (`46cd6bb2d677`)
and passed afterward. The initial targeted footprint/architecture run passed all
57 cases (`62cc58b5838e`). A broader menu run passed 154 cases and failed four
allocation cases because the historical table named a source model directory
missing on DL380 (`e2550f91fc6c`). The existing complete Qwen3-0.6B checkpoint
was staged at that recorded path; all ten tests in the affected file then passed
(`60cad850c2c8`). These are separate executions, with no missing collection or
skips in their reported populations. CI with the repository’s older pinned
producer supplies the compatibility lane; its final result is recorded on the PR.
The comparison test also calls the legacy writer fallback with the modern
producer and requires the exact same footprint report.

Evidence root: `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/`.

- `extents-menu-root-audit-01.json`: failed terminal and cleanup, full measured
  PrismaQuant snapshot, 1,141 producer file bytes/modes, four exact menus,
  profiles, source-module hashes and telemetry recovery.
- `extents-menu-profile-01/`: original four arm outputs and cProfile files.
- `extents-menu-telemetry-recovery-01.json`: explicit URLs, raw historical host
  series, exact window and descriptive statistics.
- `extents-consumer-tests-root-audit-01.json` and
  `extents-consumer-correction-root-audit-01.json`: canonical success CAS
  receipts/payloads and complete snapshot comparisons for the passing tests.

The upstream Tessera extent change separately qualified 43 identical byte-audit
rows and the complete native suite; see Tessera’s
`docs/measurements/layout-extents-2026-09-08.md`. No runtime release pin is moved
by this consumer change.
