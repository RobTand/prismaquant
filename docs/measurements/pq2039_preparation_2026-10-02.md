# Bounded projected-source preparation — 2026-10-02

Refs #2039; implementation PR #2041. Exact measured implementation:
`dd76f0e3ae59730a100e9615975fdd57e8688736`.

The opt-in two-worker preparation window reduced this original 864-unit
projected-check latency from **2.972780 s to
1.617907 s**, a **45.5760%** local reduction
(1.8374x). The predefined >=2% full-check screen passed.
Every original comparison and all 1,729 copy operators remain. Aggregate copy
self CPU increased: the gain comes from overlap, not from eliminating copies.

The default remains serial. This is one resident original-layer control;
it does not qualify the higher CPU-material source provider for GPU transfers,
change automatic capture admission or serving gates, or establish a full
campaign, GPU-saturation or energy result. Clock alignment and energy remain HOLD.

## Matched original-layer result

One admitted PB measurement action ran an ABBA sequence: serial, prepared,
prepared, serial. Both arms use the same resource guard, source decoder,
immutable source deliveries, live BF16 views and residency. Both set
`release_source_pages=False` to retain resident source pages for this control;
source-release callbacks are independently covered by the CPU controls below.
The prepared arm explicitly selects `parallel_preparation=True` and a 64 MiB
private byte cap. It uses the existing two-thread layer-read pool and four
credits; CUDA comparisons and refusal order stay on the calling stream.

| Phase | Complete checks | Steady seconds | Mean seconds/check | Copy self CPU seconds/profile |
| --- | ---: | ---: | ---: | ---: |
| check-0-serial | 11 | 32.064 | 2.914876 | 1.078393 |
| check-1-prepared | 19 | 30.733 | 1.617490 | 1.270925 |
| check-2-prepared | 19 | 30.749 | 1.618324 | 1.269247 |
| check-3-serial | 10 | 30.307 | 3.030683 | 1.055980 |

The claim is the ratio of the two arm means, not the old recovery timing.
The earlier 2.113 s recovery result omitted the per-unit resource guard and
is not a matched baseline for this guarded comparison. Warmup and profiling
are outside the steady timings. Each timed call includes CUDA completion.
This is one ABBA screen, not a repeated-run variance study.

## Where the copy work went

The CPU/CUDA profiler uses `profile_all_threads=True`; a main-thread-only
profile would omit the preparation copies. Every profile records 1,729 copy
operators. The serial profiles place them on the main thread. Each prepared
profile records 432 direct CPU copies on each of two preparation threads and
865 transfer/verdict copy operators on the main thread. The independent trace
ancestry/correlation reconstruction found no crossing CPU intervals.

Direct source-copy self CPU was 1.067286 / 1.044955 s for serial and
1.264265 / 1.262312 s for prepared. These 864 direct copies have no GPU memcpy
correlation. The other 865 copies correlate to 864 pinned H2D transfers and
one ordered verdict D2H transfer. Each profile transfers 14,495,514,624 source
bytes plus 864 verdict bytes. Correlated GPU memcpy activity sums are about
0.276 s in every arm; summed activity is not GPU wall time.

No duplicate source unit within the pass was found. Existing render-cache
identity does not establish that an arbitrary live CUDA view equals its
producer source. The implementation preserves all comparisons and private
copies rather than adding a second cache or memoizing live/source equality.
The original attribution and current reconstruction scripts and results are
bound in the artifact manifest.

## Source and execution contract

The workload is GLM-5.3-Flash-BF16 layer 20: 288 experts, hidden 4096,
intermediate 2048, top-k 8, 864 BF16 expert projections. The four publisher-bound
whole shards total 21,463,213,160 bytes and contain 866 source tensors including
the router and correction bias. Actual live GPU weight bytes are 14,548,206,720.
The publisher is the unchanged
`a6c167b62691b2bac901344b65cb651a70f53e43` revision. Source/config/index/capture
coordinates are preserved from the independently verified recovery binding.

The existing canonical shared-input capture has 512 x 4096 FP32 values that
are exact BF16 values, with the original wikitext2/train contract,
512 samples, sequence length 512 and seed 0. This check-only control uses the
real resident MoE layer and captured inputs; it executes no extra full-model
forward, routing/Hessian capture, quantization or serving run. The experiment
produced no quantization assignment or export artifact; KL, bpp and served
runtime were not measured.

The measurement adapter holds the existing kernel-sealed whole-file
`SealedBuffer` deliveries, decoded through their exact retained memfds.
It contributes no new source authority. The declared manifest has nine
entries; six sealed source/config/capture deliveries were consumed through
staged leases. The resolver reports 21,538,718,528 staged bytes, zero pool
bytes and zero fallback to pooled source. Three optional RAM copies were
unavailable; the supported staged tier supplied those deliveries. Source
buffers closed after the final CUDA synchronization inside their ExitStack.

Execution: Sparklina NVIDIA GB10, UUID
`b1eceeea-fec7-371e-2cf3-cd10f2e7b705`, driver 595.91.07,
Torch 2.13.0+cu130 / CUDA 13.0. The same known image used by recovery is pinned
by PB content `a0b85c050cdd73a00488f46e1f5a436d5fd31abbf09a0c23b3e51be54a176918`
and PQ launcher content
`8334f06b9a47e0ee398750e8f9a2c21e389792c25b622fc690c90025f3db86ba`
(the two digests use different schemes). The reviewed noneditable installed
PB95 and TesseraB40 Git provenance and RECORDs passed, 43 and 102 files.
The producer capture records Transformers 5.16.1; the reused frozen runtime
image records 5.17.0. The check compares original tensor values and layout.

PB demand: 80 GiB aggregate, 24 GiB GPU subset, four assigned CPUs (5,6,7,8),
native threads one, existing read pool two, exclusive GPU use, GB10 measurement
class, one attempt. The actual platform-keyed request preserves the measured
GB10 ABI/driver/model signature. Its hard execution deadline is 900 s; ordered
startup/measure/publish stall allowances are 300/300/60 s. Each cumulative
phase unit is reported after durable file and directory fsync. Final publish
unit is six. No request was extended or automatically retried.

The conservative guard peaked at 58,473,381,888 bytes; PB cgroup peak was
43,960,340,480 bytes. Final phase CUDA reservation was 14,615,052,288 bytes.
These fit the unchanged declared budgets, including whole deliveries, consumer
pagecache, pending private CPU data, temporary GPU data and context/allocator
allowance. Source subviews did not reduce whole-file charges.

## Telemetry and limits

Four trace files and four raw per-phase Netdata API files were hashed and
checked. Every phase has nonempty CPU, I/O, RAM, pressure and GPU power series
for both Sparky and Sparklina, with no request errors. The existing continuous
observer completed 42 both-host rounds and 209 main-thread stack samples with
no instrument errors. The all-thread Torch profiler, not the stack sampler,
carries the preparation-thread copy attribution.

Local unqualified power samples averaged 8.70 / 9.77 / 9.92 / 8.91 W in ABBA
order (121/116/116/115 samples). The PB whole-action profile reports mean
8.81 W, peak 10.09 W and mean host CPU busy 8.60%; startup is included in that
whole-action window. The power samples are about 6-7% of the standing ~140 W
reference envelope, not evidence of GPU saturation. PB's own profile uses a
100 W declared fallback, a different reference. These records do not qualify
cross-host alignment, power agreement or work per joule. No GPU utilization
percentage is used to diagnose saturation.

The 45.58% claim concerns the complete guarded 864-unit check in this
configuration. It is not a full-campaign gain, less aggregate CPU work,
automatic source-provider admission, a default change, or a serving result.

## Ownership and causal validation

The accepted CPU family has 44 passes and two CUDA skips: twelve new
preparation/order/cap/cancellation/fence/release controls, nineteen existing
projected-check passes, and thirteen unchanged architecture checks. It used
Python 3.14.4 / Torch 2.11.0+cpu and four assigned CPUs/native one through PB.
The causal serial barrier failed before the implementation. Root review then
found skipped source release on failure/mismatch: reviewed-source release
controls failed four cases and passed success; the fix makes every acquired
callback run once after clearing CPU aliases. All five then passed. Touched
modules compiled at exact dd76 through PB. No CPU token test is claimed as
real CUDA evidence.

Small CUDA action `0b9668` passed equality/refusal/partial-enqueue/cancellation/
cap controls but all 24 completion queries were already true. Its pending-DMA
limit was retained rather than promoted into a lifetime claim. Additional
admitted action `887a2fe97bcaf47c0dfb8ff605f2c7a813965358f3812440d20469fd170c52d9`
used a bounded CPU readiness callback on a producer stream and a real waiting
consumer stream. The callback makes no CUDA or tensor calls; it creates no
artificial GPU kernels. Existing allocators/kernels/events were warmed on the
same consumer stream before the gate.

That control observed four false completion queries in success and three in
cancellation, with four private owners / 16 MiB held and no fifth comparison
launch while held. Cancellation's real device fence blocked before gate
release. After completed stream joining, all private owners retired and source
memfds restored. It passed exit0/CAS on Sparky in 10.61 s within declared
aggregate16/GPU4 GiB/four CPU/native1 limits; conservative peak847,364,096 bytes.
It qualifies this bounded ownership control, not the higher source provider.

## Receipts, commands and negative results

Original action:
`cd5b0d0d95c30bb7f6a3a8b683df31f49afe2348e195c7a784ece8f9f70563c0`.
Actual exit0, one attempt, 220.337 s. Snapshot
`34625038e15fbc8c84d101042b343bbf1c8dd5e7`, parent dd76,
input SHA256 `4f0ccf15522a26f6d04314b69669a122335a68b3744b9dea1921b0f883de4920`.
Payload SHA256 `96063a55b4f3afe786fb672aa2434523c735914c6028054a23c590aa90935699`;
claim `d24cdd1adae72951174ddd21688ac388e61532056d4631385d21c00e3928103a`.
Published `PrismaBuildCAS.lookup` verified the actual full request/receipt;
independent claim verification also hashed the actual payload. All 864 records
were equal and a changed source-corresponding live bit gave identical first
refusals. Runtime pin records and all phase stdout match durable outputs.

[The checked artifact manifest](artifacts/pq2039-preparation-2026-10-02/manifest.json)
binds 19 immutable shared output files plus source programs, exact submission
argv, control inputs, trace attribution and receipt packets. Exact argv JSON
retains the complete official launcher/spec/embedded pin guard/program;
program SHA256 is `8eddbc9bf77b544ad35ff10b0c8bc06e2ce892028d5ac0b926a2c8ffa481dca7`,
argv SHA256 `2b3355f2b80f4b9db1084935b5553bf136261735541083e6aae82de7f783c799`.
Shared outputs are under
`/mnt/shared/tessera-measurements/pq2039-performance-20261002/original-check-dd76-v1/`.
Review packets are on both hosts under
`/home/rob/tmp/astra-resume-20261002/pq2039_performance/`.
Replay requires a new output namespace and an explicitly bound new request;
the recorded output directory intentionally refuses reuse.

The earlier CLI omitted `--exclusive`; review corrected it before execution.
The first corrected client attempt was refused before sealing/publication
because this agent supplied a string for `produced_by`, which requires an
object. No action key or original-source read was created. The invalid control
and log are retained; immutable manifest-v2 changes only that metadata shape,
with the same nine entries. After Root approved the corrected control,
official published PB parsing passed before submission. Manifest SHA256:
`3247d0b61ef91be8a23c3650c079fcd7b35731fa301173aa6c00c8880c0fc957`;
binding SHA256:
`122f3ee6352aaf5916023624a11f2b232f7bc2fdf1b994003e9c4dda38310829`.
No failed action was relabeled as qualification or mutated to extend it.

Retain the bounded source/test worktrees and these evidence namespaces.
The previous recovery branch, original outputs and 44-artifact manifest are
unchanged. Root owns final acceptance, native issue closure and merge.
