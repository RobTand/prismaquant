# Separately charged container compilation caches

CPU accounting slice: #1820. Lifetime and GPU acceptance: #1091 and PB #1360.

## Opt-in declaration

A new sealed container spec may declare both:

- `PRISMAQUANT_CONTAINER_CACHE_ROOT`: a canonical absolute host path on an
  explicit writable identity mount;
- `PRISMAQUANT_CONTAINER_CACHE_MAX_BYTES`: an ASCII positive integer byte
  ceiling chosen by the operator, not an inferred production default.

The existing `LOCAL_SCRATCH_KINDS` registry validates and forwards this pair
through `PRISMABUILD_LOCAL_SCRATCH_PAIRS`. PB rounds each declared ceiling to
GiB separately. For example, cotangent=1 byte, spill=1 GiB, and cache=1 GiB+1
byte declare 1+1+2=4 GiB, not the rounded aggregate. The control list belongs to
the outer admission request and is not forwarded into Docker.

The cache root must be disjoint from every other registered scratch root.
Equal, ancestor, and descendant roots refuse. A remapped, readonly, unmounted,
or nested mount refuses. Existing symlink ancestors and non-directory
ancestors refuse; validation neither creates nor deletes any path. Checks
are configuration/launch-time inspection, not a race-free filesystem sandbox.

## Routing

The explicitly charged cache root contains four defaults:

| Variable | Default |
|---|---|
| `HF_HOME` | `<cache-root>/hf` |
| `TRITON_CACHE_DIR` | `<cache-root>/triton` |
| `TORCHINDUCTOR_CACHE_DIR` | `<cache-root>/inductor` |
| `XDG_CACHE_HOME` | `<cache-root>/xdg` |

An explicit pin within that root is preserved. An escaped or overlay pin
refuses, even when `overlay_cache_reason` is supplied: that waiver does not
make an opt-in cache declaration account for bytes outside its root.

`PRISMAQUANT_TMPDIR` must be explicit and within a different declared charged
workspace, such as the cotangent or spill root. It cannot use the cache root
or an uncharged directory. No attempt/generation suffix is guessed here.

Legacy specs without the new pair retain their previous defaults and overlay
admission policy. No existing sealed spec is rewritten or silently selected
into this policy. The launcher, dry-run command builder and preamble use the
same registry-driven forwarding path.

## Limits and acceptance

This declaration is a reservation, **not** a measured size, filesystem quota,
or durable lifetime. Persistent compilation caches and temporary workspace
must not be described as crash-cleaned. PB #1360 owns the public generation-
bound lifetime contract; PrismaQuant must bind that contract and qualify its
selected runtime before claiming #1091 complete. No private cleanup hook,
local sweeper, `finally` cleanup, or produced-output retirement is substituted.

CPU tests cover declaration/pricing, command forwarding, containment,
conflicting/partial/ASCII bounds, mount safety, symlink refusal and unchanged
legacy cases. PQ #2463 measured the workload-specific ceiling: two GPU rows
of a complete Stage B quantum peaked at 3596288 bytes each, and the
fixture `tests/fixtures/container_cache_ceiling_2463.py` derives the 1 GiB
ceiling from that peak with declared headroom. Receipts live in
`docs/measurements/container_cache_quantum_row1_2463.json`,
`container_cache_quantum_row2_2463.json` and
`container_cache_quantum_row1_netdata_2463.json`; the report is
`docs/measurements/container_cache_peak_2463.md`. Recovery from
worker/launcher loss through the eventual public lifetime contract, quota
enforcement and cleanup still need proof under #1091 and PB #1360. No
work-per-joule result or speedup follows from this measurement.

This extends the existing container/scratch adapter. It adds no rendered-
weight/activation cache, model arithmetic, wire format, serving gate, runtime
pin, kernel, production recipe default, stage graph or agent scheduler.
