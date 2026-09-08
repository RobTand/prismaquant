# Planned-size joint statistics qualification, 2026-09-08

Two consecutive planned-size `JointOperatorStatisticsLease` windows passed on
Sparklina (NVIDIA GB10, PyTorch 2.13.0+cu130, CUDA 13.0). Each window retained
**34,326,183,936 bytes (31.96875 GiB)** in 1,023 distinct actual FP32 GW/GA
backings, completed 682 controlled candidate projections, and expired every
statistics backing. The complete 33,315,763,888-byte source stand-in remained
resident and unchanged across both windows. The second admission reclaimed
31.992 GiB of inactive allocator reservation without releasing the live source.

This is an operator storage, sampled-contraction, cotangent and lifetime gate.
It is not an original GLM weight/routing replay, model-quality measurement,
served gate, end-to-end fit claim, or performance qualification. No production
runtime code, defaults, format menu, serving gate, or tolerance changed.

## Frozen inputs and execution

- Helper commit: `a833c18db12c734bf981c4d16880f449669879d4`.
  Runtime under test: `34388d2bfeea0e76830645aab92d15132f766bcf`.
- Actual census: `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/workspace/census.json`,
  SHA-256 `b63f7bf6c4320714b4ceb38fbd6996e032e0f0c9b82ac2a30a8337d846e358fd`.
- Resource ledger: `reuse-plan-01/resource-ledger.v2.json` under that shared
  measurement root, SHA-256
  `dfc21a3a9988ec89db74db0f0e3502279477abfcfe97ea9a26fb45ae5a989e0c`.
  Its original blocked planning status is retained; this bounded gate does not
  certify the full ledger runnable.
- CPU planning derives all 867 layer-three Linear targets from the actual
  census: 578 shapes `(2048, 4096)` and 289 shapes `(4096, 2048)`. Existing
  registry specs `FP8_E4M3` and `NVFP4` retain their two dynamic activation QDQ
  groups, with no global-scale attestation override. Planned windows have
  341/341/185 targets and 34,326,183,936 / 34,326,183,936 / 18,622,709,760
  statistics bytes. Only the first two windows execute in this gate.
- All 867 seeded BF16 Linear weight planes own 14,545,846,272 bytes. A filled
  uint8 geometry remainder owns 18,769,917,616 bytes. These initialized shape
  stand-ins do not contain original checkpoint weights. Their names do not
  imply that packed expert routing executed.
- Eight-row BF16 inputs/cotangents and one resident 32 MiB FP32 controlled dW
  backing are used. Each candidate reshapes that same backing to the required
  orientation; there is no all-target candidate plane or full-matrix oracle.
- PB reserves 4 CPUs, 104 GiB physical memory and a 92 GiB GPU subset, with
  measurement isolation, a 900-second limit and native thread count 1.
  Existing guards reserve 16 GiB workspace and 256 MiB replay allowance;
  these are future reservations, not realized allocations in this fixture.
- The pinned producer container content is
  `eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd`;
  actual image ID is
  `sha256:9f9b9f05b17531399ba66dc6415b054cf5d68c82270626d0e9150e75c808435f`.
  Exact command, environment, file hashes and input digests are in
  [native-invocation-01.json](native-invocation-01.json).

## Numerical and owner checks

| Measured check | Window 0 | Window 1 |
| --- | ---: | ---: |
| Targets / actual FP32 matrices | 341 / 1,023 | 341 / 1,023 |
| Simultaneous statistics bytes | 34,326,183,936 | 34,326,183,936 |
| Sampled entries compared with FP64 | 65,472 | 65,472 |
| Maximum absolute sample error | 6.9849193e-9 | 5.5879354e-9 |
| Maximum relative error where reference exceeds atol | 5.6735979e-6 | 6.0075575e-6 |
| Exact complete input-gradient hashes vs unobserved Linear | 341 | 341 |
| Finite nonzero weight / activation / mixed terms | 682 / 682 / 682 | 682 / 682 / 682 |
| Expired statistics backing owners after projection | 1,023 | 1,023 |
| Peak candidate backing bytes | 33,554,432 | 33,554,432 |

Each full FP32 matrix has its own 33,554,432-byte storage. All matrices were
present simultaneously, and their `StorageWeakRef` identities were distinct.
The oracle independently contracts 64 paired row/column samples in FP64 from
the exact eight BF16/FP32 operand rows. GA references reuse the existing QDQ
primitive; the contraction is independent, the QDQ implementation is not.
The fixed comparison is `math.isclose(rel_tol=3e-5, abs_tol=2e-9)`. The maximum
absolute error can exceed atol while satisfying the relative bound. Samples
cover all matrices, not every matrix element; full matrix values are not
claimed independently verified.

The complete source storage identity, pointer, version, shape, stride and
sampled-value hashes are checked after each window. All source weights remain
frozen and have no gradients. After the fixture drops its owners, all **869**
source/remainder/candidate storage weak references expire. CUDA still reports
67,109,376 allocated bytes and 85,983,232 reserved bytes. The enumerated fixture
owners all expired; this fixture does not identify the remaining backing.

## Physical memory and profiles

With either window fully live, CUDA allocated 67,809,655,296 bytes and reserved
67,828,187,136 bytes. The action's cumulative CUDA peaks were 67,880,975,360
allocated bytes and **67,891,101,696 reserved bytes (63.229 GiB)**, below the
92 GiB subset. This is realized fixture memory, not the ledger's total future
reservation. PB cgroup peak was 3,481,288,704 bytes, with zero OOM/kill events;
on GB10 that cgroup number does not include all CUDA/UVM physical residency.

Before window 1 admission, inactive CUDA reservation was still present:
67,782,049,792 reserved bytes. Existing `check_operator_allocation` reduced it
to 33,430,700,032 bytes, a release of **34,351,349,760 bytes (31.992 GiB)**.
Allocated bytes stayed exactly 33,416,428,032 across the admission. This is
actual coupled operator turnover following the earlier
[allocator proxy](../glm-joint-allocator-turnover-20260908/README.md).

Raw CPU+CUDA profiler traces cover each complete window, including baseline
cotangent construction, oracle sampling, observations, diagnostics, candidate
projections and cleanup. Shapes and memory-event profiling are disabled to
avoid retaining tensors for the observer. Window 0 has 28,473 kernel events
and 1,754.172 ms summed kernel duration; window 1 has 27,849 and 1,689.724 ms.
FP32 multiply and reduction kernels account for the largest summed GPU event
durations. Window 0 includes a 2,549.128 ms Dynamo compilation event. Window
wall intervals were 6.833 and 3.674 seconds; these different target sets and
cold/warm states are not an A/B speed comparison. CPU events overlap and their
summed durations are not self time.

Netdata has 13 samples from **each** GB10 host during the native gate, with no
monitor errors. Whole-host CPU non-idle means/peaks are 8.91%/18.01% on Sparky
and 6.78%/7.93% on Sparklina. PB's 27.953-second container window has 56 pqteld
power samples, mean 10.415 W and peak 24.2 W, or 17.3% of the 140 W reference
envelope at peak. That window includes startup and observer work. Unified
memory available reached a minimum of 53,075,279,872 bytes. GPU utilization
is not used to infer saturation. There is no comparative work-per-joule or
production throughput claim from this single correctness fixture.

## Validation and retained evidence

All execution ran through PB. CPU tests passed **8 tests, 0 skips**; the real
census plan action exited 0 without CUDA. Native action
`18ad980463cae0f7c11a3948c3dc5d623e17882aa7df56234d0bbf399bb49026`
ran on Sparklina and exited 0 with cleanup complete. No native retry occurred.
All CPU/native and two trace-analysis terminal statuses, CAS result bytes,
local payload claims and source checkout closures were independently audited.
The source closure diffs contain only PB's generated closure files.
The coordinator independently repeated the native receipt/source audit and
checked saved numerical rows, matrix rosters, guard arithmetic and telemetry.
Full cotangent hashing, backing uniqueness and weak-owner expiry checks were
asserted inside the native run; the saved artifact is not a second execution
of those native checks.

- [Coordinator native PB audit](root-native-pb-audit.json) and
  [coordinator artifact audit](root-native-artifact-audit.json).
- [CPU test audit](cpu-tests-audit.json), [CPU plan audit](cpu-plan-audit.json),
  [native audit](native-audit-01.json), [artifact summary](artifact-summary.json).
- [Window 0 profile summary](window-0-profile-summary.json) and
  [window 1 profile summary](window-1-profile-summary.json), with corresponding
  [window 0 audit](window-0-profile-audit.json) and
  [window 1 audit](window-1-profile-audit.json).
- Shared raw artifact root:
  `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/joint-statistics-scale-01/native-01/`.
  `result.json` is 12,622,055 bytes, SHA-256
  `58e237a0d2df0864981fa94efbb8050a755bdbd0a9f18f49a51af2344a59f389`.
  The result retains sampled values, all projected terms, target rosters,
  cotangent counts, source identities, snapshots and telemetry references.
  Trace sizes/hashes and the two-host Netdata hash are in the artifact summary.

Trace extraction initially hit PB's source-closure refusal because the parser
was outside the repository. The same bounded parser was recorded as
`experiments/chrome_trace_summary.py` in commit `bc294c6a83`, then both CPU
analysis quanta passed. Nothing bypassed admission. The first receipt reads
preceded shared CAS directory visibility; subsequent audits verified the
published bytes without repeating either analysis or native execution.

The coordinator also checked both CPU gates and both trace-analysis receipts
against their exact source snapshots. Each saved trace summary equals its
CAS-backed JSON output and matches the native result's trace identity. See
[root CPU audit](root-cpu-pb-audit.json),
[root trace PB audit](root-trace-pb-audit.json) and
[root trace artifact audit](root-trace-artifact-audit.json).
