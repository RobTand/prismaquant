# /mnt/shared was depth-bound at one RPC, not bandwidth-bound

Measured 2026-09-11 on both Sparks with
`experiments/nfs_rdma_read_sweep.py`: O_DIRECT reads so the client page cache
cannot answer, disjoint file slices per worker, disjoint file windows per arm,
and a single-reader warm pass first so every arm is served from the server's
ARC rather than from spindles.

## Before

`/mnt/shared` is NFS 4.2 over RDMA, `rsize=wsize=1048576`, `port=20049`, and had
**no `nconnect`**. One transport means one in-flight read:

    workers 1 (cold)   286 MB/s   queue 0.06 ms   rtt 2.99 ms
    workers 1 (warm)   642 MB/s   queue 0.05 ms   rtt 1.09 ms

`queue_ms` is ~0 on every arm. Nothing was queued behind anything; the pipe was
empty. Throughput was exactly `rsize / rtt` with depth 1. The campaign's own
8-reader prefetch got **192 MB/s** -- *worse than one reader* -- because the
eight readers serialised against each other on the single transport.

## After `nconnect=16`

| workers | MB/s | Gb/s | queue | rtt |
|--------:|-----:|-----:|------:|----:|
| 1  |  642 |  5.1 | 0.047 ms | 1.088 ms |
| 2  | 1392 | 11.1 | 0.043 | 0.985 |
| 4  | 2888 | 23.1 | 0.036 | 0.966 |
| 8  | 5317 | 42.5 | 0.035 | 1.092 |
| 16 | 8004 | 64.0 | 0.035 | 1.576 |
| 24 | 8145 | 65.2 | 0.036 | 1.869 |
| 32 | 8927 | 71.4 | 0.035 | 3.039 |
| 48 | 9011 | 72.1 | 0.038 | 3.702 |

Sparky reproduced lina's curve within 2% at every arm across a kernel change
(6.17 -> 7.0), so this is a property of the mount, not of one box.

Plateau is ~9 GB/s = **72% of the 100 Gb line rate**. RTT doubles from 32 to 48
workers for +0.9% throughput: past 32 the arms only queue. `nconnect` itself is
capped at 16 by the kernel, but more *threads* than connections keeps helping --
16 -> 32 workers buys +21% on the same 16 connections.

RoCE `port_rcv_data` tracks wall throughput within 1.5% at every arm.

## The cache was never the problem

During the 192 MB/s runs, dl380g10 served **99.78% of demand data from ARC**
(3929 hits/s against 8.7 misses/s), L2ARC was untouched and the spindles were
idle. Prewarming resident bytes cannot help; the loss was entirely client-side
RPC depth. This retires the prewarm direction for this workload and supersedes
the reasoning in `pb_prewarm_spindle_reads_reset_nfs_rdma`.

## Topology

dl380g10 has exactly two 100 Gb RoCE ports, one dedicated per Spark
(`ens5f0np0` = 10.100.99.3 to sparklina, `ens5f1np1` = 10.100.98.3 to sparky), so
the Sparks do not contend and each is capped at 100 Gb. The Sparks' own 200 Gb
ports face each other, not storage.

## Applying the option

`nconnect` is honoured only when the transport is **created**. Four scripted
attempts to cycle the mount failed, each silently:

1. `mount -o remount,nconnect=16` is accepted, returns 0, and does nothing.
2. `ls /mnt/shared` does not trigger a *stopped* automount -- it reads an empty
   directory and succeeds, leaving the box with no mount at all.
3. A surviving `nfs_client` for that server (`/proc/fs/nfsfs/servers`) makes the
   new mount reuse the old transport; it comes up without `nconnect` and reports
   success. The tell is `clientaddr=0.0.0.0` instead of the real client IP.
4. Holders are not only `cwd`. On sparky the PB workers are respawned by **cron**
   (`*/5 supervise.py --ensure`), not by `prismabuild-supervisor.service`, so
   stopping units alone lets cron refill the mount within five minutes.

**A reboot did it first try on both counts.** On an idle box whose fstab is
already correct, reboot rather than scripting the cycle.

## What this does not establish

Reads only; no write-path measurement. One file-size distribution (~21 MB
`.pt` entries). The 72 Gb/s plateau is not attributed -- RDMA and RPC overhead
are the presumption, not a measurement. Whether 72 Gb/s can be raised was not
tested beyond 48 workers.
