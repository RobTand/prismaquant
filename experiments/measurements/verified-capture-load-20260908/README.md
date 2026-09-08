# Verified capture loading — 2026-09-08

The explicit `prismaquant.verified_activation_load.v1` mode reads each sealed
capture file once into an admitted private byte buffer, validates and decodes
those same bytes, then releases the buffer before CUDA transfer. It remains
opt-in. On the final instrumented native comparison, prefetch elapsed fell
21.6%, while replay was 4.3% slower and seal was 12.5% slower. This is a
bounded two-unit loading result, not a production throughput or model-fit claim.

## Contract and scope

The existing activation/capture owners implement this route. No second cache,
artifact format or canonical identity is introduced. Closed policy fields
`max_buffer_bytes` (F) and `max_scratch_bytes` (M, at least 1 MiB) price private
serialized storage and metadata separately. A second full F term prices source
file contents that the kernel may retain despite best-effort page advice;
page rounding/bookkeeping remain in the physical guard's runtime margin.
Existing CPU tensor storage S remains separately charged. Materialization,
seal and joint preparation include these terms; model forward and source
validation retain their prior phase terms. Default capture, exact-boundary
prefetch and the qualification-v1 PWC buffer retain their existing behavior.

A regular nonsymlink descriptor is bound to device/inode/size/mtime/ctime. Each
up-to-16 MiB read uses a view of admitted F, checks memory and file identity,
hashes the consumed bytes and advises eligible consumed pages. Before Torch
allocation, bounded ZIP-directory and inert pickle checks refuse oversized
rosters, unsupported archives, sparse memo/frame allocations, unknown owners,
forged backing storage and views outside their individual storage records.
Meta then CPU reconstruction validate canonical geometry and unique backing
storage. CPU finite validation uses two scalar extrema with explicit
contiguity, dtype, device and thread/scratch checks. File changes, unsupported
copying reads and cap violations refuse. Canonical manifest, journal and
payload bytes stay unchanged; load execution receipts are separate metadata.

## Native comparison

PrismaBuild admitted each complete legacy/verified/verified/legacy experiment
on Sparklina's GB10 with **6 GiB physical, 2 GiB GPU subset, two CPUs, one native
math thread and a 300-second deadline**. Both used the content-sealed producer
image with Torch 2.13.0+cu130 (`cf30153c4c13`), CUDA 13.0 and driver 595.84.
Each arm replayed, sealed and prefetched the same independent, journal-verified
copies of real GLM dense `layers.0.mlp.down_proj` and expert
`layers.3.mlp.experts.0.gate_proj`. Their full storage sizes were 600 and 72 MiB;
serialized sizes were 629,147,557 and 75,499,429 bytes. The subset envelope is
explicitly diagnostic. Frozen source files were neither changed nor advised.

All operations collected cProfile, Torch CPU/CUDA traces, process I/O and
Netdata on both hosts. Expected CPU tensors were retained only during replay.
Timing includes instrumentation; artifact/tensor hashing after each operation
is outside that interval. Each table cell is the mean of two arms, not a
statistical confidence interval.

| Version and operation | Legacy seconds | Verified seconds | Verified elapsed change |
| --- | ---: | ---: | ---: |
| Initial completed run 02: replay | 1.4332 | 2.0349 | +42.0% |
| Initial completed run 02: seal | 1.0610 | 1.6854 | +58.8% |
| Initial completed run 02: prefetch | 1.3125 | 1.5114 | +15.2% |
| Final run 03: replay | 1.4408 | 1.5029 | +4.3% |
| Final run 03: seal | 1.0193 | 1.1464 | +12.5% |
| Final run 03: prefetch | 1.2870 | 1.0091 | −21.6% |

Every verified operation in both completed runs read exactly **704,646,986
source bytes in two opens**, versus **1,409,300,910 bytes in four opens** for
legacy. Final prefetch process `read_bytes` was 704,651,264 per verified arm
versus 1,409,294,336 per legacy arm. Replay/seal physical reads were largely
unchanged: fewer logical reads do not generally imply half the physical I/O.

The before profile identified CPU finite masks and frequent guard callbacks.
In verified prefetch, Torch's main-thread finite scope changed from 672
`isfinite` calls / 0.4453 seconds to four `aminmax` calls / 0.02575 seconds.
Source reads no longer tie their view size to metadata scratch. Guard calls
fell from 686 / 0.2210 cumulative seconds to 55 / 0.02950 seconds. SHA work
remains, and both meta and CPU reconstruction remain. Four pageable HtoD
copies took approximately 0.028 seconds before and after; these scopes contain
no GPU compute kernels. This establishes no GPU-kernel speedup. Nested Torch
and cProfile durations overlap, and cProfile also observes background
Netdata/thread waits; their totals must not be added as exclusive wall causes.

The final run passed all 12 actual phase preflights against the **4 GiB
cap-minus-margin threshold**, including cumulative source-page exposure and
conservative CUDA/transfer overlap. The smallest preflight slack was
732,098,742 bytes. Cgroup peak was 4,063,555,584 bytes; cleanup completed with
zero live processes and no OOM. All tensor hashes matched, canonical bytes
remained unchanged, and every return/transfer boundary had zero serialized
owners. The largest copying buffer read was 523 bytes; bulk decode used
`readinto`. The source-derived 104 GiB full-capture phase maximum remains
unchanged because forward dominates, but that arithmetic neither qualifies a
full capture nor authorizes changing the frozen run.

Sparklina's whole-action power averaged 8.36 W and peaked at 11.74 W against
the 140 W SoC reference; host CPU busy averaged 6.48% in PB's Netdata window.
Both hosts' raw Netdata series are retained; Sparky had unrelated ongoing
work. GPU utilization is not used as a saturation measure. Netdata GPU power
updates every ten seconds, and only a few 2 Hz pqteld observations fall inside
each short operation. Per-operation work/joule ranking is therefore
unestablished, and host/CPU energy was not measured. Exploratory interpolated
energy calculations are superseded by the timestamped observations in the
final analyses; they are not accepted results.

## Validation and retained failures

The final runtime passed **170 CPU tests** on Torch 2.10.0+cpu and **70 CPU
tests** inside the pinned producer image, with no skips in those populations.
The latter also ran the exact integrated scalar helper over empty, finite,
NaN and positive/negative infinity cases at first/middle/last positions.
Special-value tests include FP32 extrema, signed zero and subnormals. Torch
CPU allocations were eight bytes for 4, 64 and 600 MiB inputs. Pinned ATen
source inspection confirms contiguous allreduce with scalar/vector partials;
non-Torch thread-pool bookkeeping remains runtime overhead, not a claimed
Torch allocation measurement. All touched modules passed a portable PB
compile. Full-suite integration validation is a separate receipt.

New read-window/page-pricing tests failed twice on the pre-change source, then
137 focused tests passed before scalar integration. Earlier regressions cover
single reading, pre-allocation ZIP/pickle guards, per-storage extent, alias
accounting, source mutation, FIFO/symlink refusal and failure cleanup.
The first native attempt failed before completing its ABBA: the harness
unnecessarily retained 672 MiB of reference tensors during prefetch and hit
the guard by about 9.3 MiB. It had no OOM and clean termination. The reference
ownership correction was applied equally to both arms; its five completed
operations are not a completed comparison.

## Evidence

[Receipt audit](receipts.json), [run 02 analysis](native-02-analysis.json),
[run 03 analysis](native-03-analysis.json), [exact final invocation](native-03-invocation.json),
[scalar probe](finite-cpu-probe.json), and
[phase arithmetic](unchanged-capture-cap-arithmetic.json) are retained here.
[The analyzer](analyze_native.py) derives summaries from the raw artifacts;
it does not interpolate per-operation energy. Root independently verified
terminal exits, CAS result bytes, canonical receipts and executed source.

Full traces, individual operation records, Netdata/pqteld samples and failed
attempt evidence are under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/verified-capture-load-native-{01,02,03}/`;
CPU regressions, reviewed invocations, pinned ATen source and audits are under
`verified-capture-load-design-01/` beside those directories. Runtime changes
are `65f1e1b01` (read windows/page exposure) and `063ab5a05a` (scalar finite
validation); the final native harness is `9dfde2ec1b`.
