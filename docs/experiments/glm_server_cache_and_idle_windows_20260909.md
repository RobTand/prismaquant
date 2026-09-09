# GLM server cache and GPU gaps — 2026-09-09

A bounded warm-up read 66.089 GiB of an existing queued group's source and
calibration files through dl380g10's ZFS cache. A subsequent read of the same
four source shards transferred 19.985 GiB to Sparklina in 5.801 s: **3.445 GiB/s**
while normal pricing continued. Client NFS READ counters increased by
21.470 GB, consistent with the 21.459 GB diagnostic plus concurrent traffic;
this was not merely a client page-cache read. The four shards read locally on
the server in 6.219 s. These are observed warm-data rates, not an isolated
before/after speedup or a claim about full capture-prefetch time.

The server uses a four-disk RAIDZ1 pool, an NVMe L2ARC and a ZFS ARC configured
with a 240 GiB maximum. Before the warm-up, a 20.044 s observation found
10.006 GB of L2ARC reads and approximately zero HDD reads. That observation
identifies an NVMe-backed interval; the aggregate demand-cache hit percentage
alone would hide data supplied by predictive prefetch. It does not establish
that every slow read has the same cause.

The warm-up selected the existing `row-0045` layer-10 expert group: 864 capture
files (46.104 GiB) and four complete source shards (19.985 GiB). It read each
file once on the storage server with four 8 MiB buffers. It created no new
weight/cache format and changed no quantization input. It took 183.240 s;
ARC size changed from 192.843 to 186.539 GiB as other cache activity continued.
Reading a file warms the existing cache but does not pin it or guarantee its
continued residency. The warm-up overlapped other pricing activity and may
have competed with it for storage. No live cache limit was changed.

## Observed GPU gaps

The recent fifteen-minute Sparklina window had 32.06% of samples below 15 W,
but included preparation before its first batch. The row logged resident
calibration at 13:28:57 UTC and completed its first eight anchors at 13:31:13.
Over the following 595 s covered by the recorder, power averaged 68.36 W;
0.084% of samples were below 15 W and 0.504% reported zero GPU activity.
Sparky's separate fifteen-minute window averaged 73.39 W and had no samples
below 15 W. Low-power thresholds are descriptive screens, not a proof that
no kernel ran, and GB10 utilization percentage is not a saturation measure.

A separately recorded 64-anchor expert prefix spent 241.049 s before its first
anchor and 157.229 s in its eight batches and finalization. The six unprofiled
batches took approximately 17.6–17.9 s each, with about 0.3 s between calls.
Their sampled low-power beginnings often coincide with a main thread waiting
for the native Viterbi GPU work, so power alone cannot label those intervals
CPU idle. Occasional tail samples showed render/wire publication. This prefix
is not a completed pricing group and does not establish an end-to-end idle
fraction.

The proposed eight-versus-sixteen batch comparison did not qualify: the first
control completed 64 anchors, but its second two-second CUDA trace exceeded
its 128 MiB export limit (283,292,151 bytes). No width-16 arm ran. Analysis of a
retained representative trace showed CUDA/runtime activity ending near the
requested two seconds; there was no evidence that the timer retained an
entire call. No batch-width change was promoted. Two attempts to attach an
external sampler to a live contained row failed on privilege/ptrace restrictions;
no permissions or containment settings were weakened. Attribution above uses
the existing in-process samples, actual container timestamps, NFS counters and
host telemetry.

## Reproduction and evidence

All file-warming and transfer-diagnostic execution went through PrismaBuild. The warm-up reserved
four CPUs and 70 GiB for possible server-cache footprint; its attempt scope peaked at
108 MiB. The local four-shard follow-up reserved four CPUs and 24 GiB. The
client diagnostic reserved four CPUs and 2 GiB, retaining only four 8 MiB
buffers and advising consumed client pages away. It ran on Sparklina, with
its current pricing job present; it did not reserve or execute GPU work.

The warm-up helper is `experiments/server_cache_prime.py`. Exact file plans,
commands, per-file results, `/proc` counters, Netdata series for all three
hosts, and CAS/source audits are under:

`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/server-read-investigation-01/`

PB actions are `6871ff44168539853c2d3573133823f4648996fd24f900dbb91808daa50ebb9c`
(warm-up), `d388343d6dfdb0be5697c8dd1432e26dbe2e7c7bb8003a6de1b02fb5febdfae1`
(local follow-up), and
`9367379a579236259b4407fdf0eb4f26ab7b7959a75c3e91add5230884bfa24a` (client).
Actual actions exited zero, cleanup completed, and result/source CAS was
verified. The warm-up's py-spy process exited 1 after its child ended but
retained a usable 4,014-sample profile; the payload's successful exit is
independently recorded. The two follow-up profilers exited zero. The deployed
PB terminal process-I/O aggregate has the known exited-child omission (#451),
so this report uses the workload's own counters and NFS evidence.

GPU-window artifacts are under `live-pricing-gap-observation-02/` and
`expert-batch-width-optimization-01/` beside that directory. Source weights,
the original calibration capture, production batch width, and the 104 GiB
Spark limits remain unchanged. All original pricing rows were returned to the
ordinary PB queue after the bounded measurement window.
