# Capture prefetch: 16 readers, and the prefetch is now CPU-bound

Measured 2026-09-11 on sparky (GB10, kernel 7.0.0-1019-nvidia) against
`row-0081` of the GLM canonical census: 864 routed units, 48.9 GB of source
bytes, through `experiments/capture_prefetch_ab_container.py`, which drives the
real `tessera_campaign` prefetch inside the row's own sealed container. The only
variable is `PRISMAQUANT_CAPTURE_READ_THREADS`.

Two changes landed the same day and the second is only legible after the first:
`/mnt/shared` gained `nconnect=16` on both Sparks, which took the NFS/RDMA mount
from one in-flight RPC to sixteen (see
`glm_nfs_rdma_nconnect_2026-09-11.md` for that half).

## The numbers

| readers | cold | warm | MB/s (cold) | decode_thread | consumer |
|--------:|-----:|-----:|------------:|--------------:|---------:|
| 8       | 72.03 s | 22.80 s |  680 | 13.08 s | 9.47 s |
| **16**  | **21.37 s** | **21.47 s** | **2291** | 30.77 s | 16.59 s |
| 32      | 21.87 s | 21.70 s | 2238 | 32.80 s | 16.91 s |

**8 -> 16 is 3.4x on the cold read.** 16 -> 32 returns nothing: it is 0.5 s
slower and spends twice the cores, which on a 20-core box contend with the
encode that follows. 16 is the operating point.

## Why it stops at ~2250 MB/s

The mount itself sustains ~9000 MB/s (72 Gb/s of a 100 Gb link) on a
concurrency sweep with no verification. The prefetch stops at a quarter of that
because `decode_thread_seconds` (30.8 s at 16 readers) already exceeds the
21.4 s wall: the readers are waiting on decode and per-unit `tensor_identity`
hashing, not on the network. **The transport has stopped being the constraint
for this stage.** Raising reader count further only adds contention.

This is the measurement a synthetic read sweep cannot make -- the sweep reads
bytes and verifies nothing, so it reports the transport ceiling and calls it the
prefetch ceiling. The two differ by 4x here.

## Cold and warm converge

At 16 and 32 readers cold and warm are within 0.5% of each other (21.37 vs
21.47; 21.87 vs 21.70). The cold-start penalty is gone. Two consequences:

* Prewarming the storage host's cache buys nothing for this stage. dl380g10 was
  already serving 99.78% of demand data from ARC during the slow runs, and the
  penalty that remained was client-side RPC depth, not residency.
* The 8-reader row is the only one where cold and warm differ (72.03 vs 22.80),
  which is the signature of depth starvation, not of a cold cache.

## What this does not establish

No claim about the encode stages, which were not run here. No claim about
`PRISMAQUANT_LAYER_READ_THREADS` (a different knob, a different working set,
left at 4). The `cpu: 10` per-row demand that shipped with the 8-reader resume
manifest was recorded in that manifest's README as an inference, not a
measurement; 16 readers wants a matching re-derivation of that number, which is
owed. One row, one box, one repetition per arm.

## Where the value is set

`PRISMAQUANT_CAPTURE_READ_THREADS` is injected into the container env by
`build_restart_manifest.py` (`CAPTURE_READ_THREADS`), not by the campaign
dispatcher and not by any default -- `capture_read_threads()` returns 1 unless
the variable is set, deliberately, because 1 is the byte-identical serial read.
A campaign built without that injection runs the loader serially at 641 MB/s.
