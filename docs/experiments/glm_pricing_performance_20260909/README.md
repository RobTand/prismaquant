# GLM selected-pricing performance screens, September 9

Pricing was paused after the selected expert profile showed a roughly 4 us
trellis step and low board power. The requested 90–100 W operating target is
still unmet. These bounded experiments preserve the original capture and
encoder; neither changes production defaults or establishes a served quality
result. Existing completed pricing receipts remain usable.

Evidence root (abbreviated `PREP` below):
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02`.

## Batch width: real first 16 down projections

PB `ed30dec36df89ca9fd477cbdfbd7f5c9563e1947df32cebaf2e6c1582d37d715`
ran on Sparklina GB10, with the selected group's 864 X/H entries resident
(49,526,341,632 bytes, zero cache misses). The tested inputs are 16 actual
4096 x 2048 expert down projections at E4M3_K1_R832. Both shapes were warmed,
then four arms ran B8/B16/B16/B8 with identical observation. All six arms
produced identical wire hashes and pricing scores. No source forwards ran.

| Batch | Mean seconds / 16 units | Mean GPU watts | Estimated GPU joules / 16 units |
|---|---:|---:|---:|
| 8 | 77.658544 | 53.314841 | 4140.297006 |
| 16 | 76.675099 | 54.164605 | 4153.097604 |

B16 throughput was 1.283% higher; estimated energy was 0.309% higher.
Increasing batch width did not materially resolve the measured limitation.
Energy integrates the 2 Hz pqteld series with interpolated boundaries;
Netdata series from both hosts and 1 Hz main-thread stacks are retained.

**Profiler limitation:** the original harness toggled CUDA collection four
times inside one profiler context. Only the first half-second CUDA window
has valid dependent-step timings (106,496 steps, mean 4.0908 us). Later
windows have physically impossible durations and are rejected. Full-arm
wall times, stack samples and power remain usable; there is no accepted
paired CUDA-timing comparison for batch width. The initially permissive audit
is retained as `root-profile-power-artifact-audit-02-rejected-cuda-timing.json`.
The corrected audit explicitly returns `LIMITED_INVALID_LATER_CUDA_WINDOWS`.

The current experiment harness uses a fresh profiler context per arm. That
revision has not been rerun on these real pricing inputs. It must receive
the same physical timing checks on any future run.

Primary artifacts under `PREP/performance-batch-ab-01/`:
`evidence/result.json`, `evidence/observer/`, both `*-pqteld.csv` files,
`root-profile-power-artifact-audit.json`, and the root CAS/source audits.
The PB CUDA blob is
`12eae3c02cc3d9373d49b1e5bf24341ddc5768e3a88130390a1987fd1a021eab`.
Corrected artifact verification ran through PB
`b3cea76a6097448214b51f83c7ab6479d6cf99a9db8fc4f2d1bfd1d99f23e2e7`.

## Tile geometry: representative kernel screen

PB placed two independent campaign actions: R3 on Sparky
`6c56f4de7979e2f9af1822d10af70604dd3b2c8a8649b9abf034f957b60b146c`
and R4 on Sparklina
`93f3669a34f04ca170d36f83b0b3becaec3856e11ca9447aec0d60bc2c11bbed`.
Each requested four CPUs and 24 GiB. Inputs use the actual pinned E4M3 table
and seeded representative targets: 4096 x 192 at R3 and 4096 x 64 at R4,
L=14, 128 calls per arm. They are not captured GLM residuals. Each geometry
has a separate ABBA comparison; compare within a row, not across hosts.

| Rate | Candidate | Throughput ratio | Work/joule ratio | Before → after watts |
|---|---|---:|---:|---:|
| R3 | cols1_w4 | 1.0920 | 0.7584 | 47.02 → 67.70 |
| R3 | cols1_w2 | 0.9723 | 0.9560 | 50.26 → 51.12 |
| R3 | cols4_w4 | 1.0440 | 0.9166 | 51.91 → 59.13 |
| R4 | cols1_w4 | 1.0571 | 0.8519 | 42.96 → 53.30 |
| R4 | cols1_w2 | 0.9969 | 0.9826 | 47.03 → 47.72 |
| R4 | cols4_w4 | 1.0573 | 1.1268 | 46.98 → 44.08 |

Every arm has exact states and SSE. Fresh profiler contexts produced 24
independently checked traces; occasional sub-microsecond CUPTI overlaps are
retained explicitly (maximum 0.256 us), unlike the invalid batch trace above.
R3 cols1_w4 changed the launch grid from [16,16] to [16,32], registers/thread
from 36 to 34, and mean step time from 4.0654 to 3.6652 us. This improves
throughput while reducing work per joule. R4 cols4_w4 is a candidate worth
retaining, but neither geometry is promoted without actual-pricing evidence.

Artifacts are under `PREP/window-tile-screen-01/`: manifest, per-rate raw
results and traces, both host recorders, and
`root-profile-power-artifact-audit.json`. Artifact verification PB
`ddea39c6020c770fe4641f8ce714b019da1ec15c81ff5b5c228eb4f3770620e3`
checked raw hashes and timing bounds. Both native action terminal records
and CAS receipts were verified independently, including resource cleanup.

## Startup and I/O bounds

Existing samples cover 226.18 seconds of the 227.28-second startup: process
`rchar` increased 81.23 GB and `read_bytes` 111.06 GB. Of 225 main-thread
samples, 72 ended in tensor identity hashing, 66 in verified activation-cache
loading, and 23 in a wait. Inclusive samples identify selected-capture
prefetch (87), checkpoint identity (45), and Hessian-reference export (33).
These overlapping stack counts are not additive function timers. Both-host
Netdata mean non-idle CPU was 6.86% on Sparklina and 1.84% on Sparky.

Each measured encode arm has zero `read_bytes` growth between its first and
last sample; `rchar` growth was only 15.8–64.5 KB. Output writes remain:
142.1–283.6 MB of `write_bytes` per sampled arm. These exclude boundary tails,
and Linux process counters alone do not establish NFS traffic, server disk
latency, or zero I/O stalls. No startup optimization is claimed.

PB `8e74b952d709a21cc834ceb6e4cd1fd0c584611d4e01d2440ef02a7d71eca6b3`
audited the existing records without new GPU work; actual exit 0, cleanup and
CAS payload hash were checked. Its result is
`PREP/performance-batch-ab-01/root-io-phase-audit.json`. The command was
`python3 -m experiments.glm_pricing_io_profile_audit` through pbrun, one CPU,
2 GiB; the client selected its local-host default. Future portable CPU
commands should explicitly use `--anywhere`.

All five retained performance harness/audit modules passed `py_compile` in PB
`6b1eb0c2aaec6107acad9c5b34c8331e2fa5e900a183315b82698c9d6a29da6c`
on DL380 CPU; the actual terminal, empty CAS output and source were checked.

Read-only configuration observation at 03:19 UTC is retained separately as
`PREP/root-network-memory-readonly-audit.json`. Both clients use NFS 4.2,
RDMA, 1 MiB reads/writes; each client link and both server links report
100000 Mb/s, full duplex, up. DL380 reports 308,829,708 KiB total RAM;
ZFS ARC maximum is 257,698,037,760 bytes (240 GiB), current ARC size
156,034,159,728 bytes. The shared dataset has 1 MiB records, primary/secondary
cache set to all, and compression enabled. This is neither a transfer
benchmark nor proof that a particular selected group is warm in ARC. The
initial server memory query lacked `rg`; a successful `cat /proc/meminfo`
fallback is recorded alongside that error. No network or storage settings
were changed.

Reproduction commands and container/environment identities for the GPU runs
are preserved in each directory's manifest/invocation, submission log and PB
request. Retain these screens as bounded evidence; do not use them as a
full-model throughput, KL/bpp, or shipping qualification.
