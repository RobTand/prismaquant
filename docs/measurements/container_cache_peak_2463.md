# PQ #2463: container-cache peak on a complete Stage B quantum row

Part of PQ #1091. The cache ROOT/MAX pair declares an operator-chosen
reservation. Two GPU rows ran a complete Stage B quantum through
PrismaBuild with cache writes observed. Peak bytes are recorded with
immutable runtime evidence. The declared ceiling derives from the peak.

## Workload and scope

`tools.measure_container_cache_quantum_row` runs in the known-good
campaign container (image `sha256:c0e532d28a78b3bf425bbbc0d862e2840ba624249162aedfadd09748a6c68c37`,
content `d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026`;
torch 2.13.0+cu130, triton 3.7.1, transformers 5.16.1, NVIDIA GB10,
capability 12.1). The row runs Stage A adjoint capture on a tiny
two-layer bf16 fixture model, then one Stage B layer quantum for that
capture's slice (unit `model.layers.1.proj`, formats FP8_E4M3/NVFP4A16/
BF16, two probes, seed 7000), then the served NVFP4 RTN activation
quantiser compile (`torch.compile`, fixed tensor 64x256, seed 2463) and
the six KDA capture kernels (`triton.jit` launches, probe digest
`844e6ad2f98c481dcb6de63099428b7d49fa6b464da2f89dc4edf2afd6235f45`).
The quantum never quantizes a fixture input, so the compile probe runs
the served path the fixture leaves cold; both write under the routed
cache root. Caches route under `PRISMAQUANT_CONTAINER_CACHE_ROOT` (hf,
triton, inductor, xdg). `PRISMAQUANT_TMPDIR` sits in a separate charged
cotangent workspace. Each row starts from a cleared empty cache root.
The sampler forks a child with its own GIL, walks the root every
250 ms, and keeps the largest observed sample.

## Measured result

| Row | PB action | Peak bytes | Samples | Valid | Receipt digest |
|---|---|---:|---:|---|---|
| 1 | `628ceda204fb2e1b5e126c66926f9da49d8da3945006d540d058ba0880bac078` | 3596288 | 27 | true | `074ba9750361c0288f463b3fecb66b26365431023181942ef104e78a14f7134c` |
| 2 (repeat) | `02da0e85bb7472fc6f2ba0a5df2940fd046449475cbc5977585b8dc72cc94a4e` | 3596288 | 27 | true | `a52a2c5e341020f0876ec6bd81691a7fdb173502aa66fcdd143484adc5e08c61` |

Both peaks are identical. Each row grew 4096 to 3596288 bytes over
6.54 s (88 files, 23 dirs at peak; the peak sits at the last sample).
Initial state is the cleared root (4096 bytes, 1 dir, 0 files), captured
before the sampler starts. No scan errors, no gaps, no incomplete scans.
The repeat declared the 1 GiB ceiling and charged spool_gb 10 (cache 2
+ cotangent 8 on row 1; cache 1 + cotangent 8 on row 2); both peaks fit.
Sealed demand: gpu=1, mem_gb=32, gpu-mem 20 GiB, 4 CPUs, 1500 s timeout,
progress phases row/head/layer-001-chunk-000 at 600 s.

Receipts: `container_cache_quantum_row1_2463.json`,
`container_cache_quantum_row2_2463.json`. Both-Spark Netdata for the row
1 window: `container_cache_quantum_row1_netdata_2463.json` (sparky power
mean 10.6 W, max 12 W; sparklina idle 4 W). In-process cProfile tops ride
in each receipt (`profile_top`; full text on the worker). Row 1 in-process
power: 76.36 GPU joules, peak 13.25 W.

## Derivation

`C = G * ceil((P + H) / G)` with `G = 1073741824`, `P = 3596288`,
`H = 4341760`. `H` is four times the largest single 250 ms interval growth
(1085440 bytes): one more unseen jump past the final sample (the peak sits
at the last sample), one concurrent compile burst, scan-gap margin.
`P + H = 7938048`, so `C = 1073741824` (1 GiB). Fixture:
`tests/fixtures/container_cache_ceiling_2463.py` binds both digests, both
peaks, the inputs and the ceiling; its tests rehash both receipts and
check the headroom against the recorded samples.

## Limits

This ceiling covers this workload, runtime, concurrency and initial cache
state only. It proves no quota enforcement, cleanup or crash recovery.
Those stay with PQ #1091 and PB #1360.

## Sampler lifecycle repair: 2026-10-09

The review found that a failed child could leave only a final parent scan.
That scan could hide cache growth during the row.
The repair removes this fallback.
Entry now waits for the first complete child scan.
Exit requires a final child scan after the row ends and a zero child exit code.
The receipt records both row boundaries, sample phases, and the child exit code.

An early exit, signal, failed final scan, or row exception invalidates the result.
An active sampler also reports an incomplete result.
Both row commands return status one for an invalid receipt.
The original GPU receipts and ceiling fixture remain unchanged.
They predate these lifecycle fields; this repair does not certify their missing fields.
No new GPU measurement ran for this repair.

### Failure evidence

PB action `5f1fc6b124075a531680de520984d2bb979b19f1ba732db3503fd5dbf6e2e3ef`
ran the new regressions before the fix.
Eight lifecycle cases failed; ten existing cases passed.
The cases covered delayed startup, startup failure, early exits, active results, row failure, and a fast row.

An intermediate CPU smoke, action
`33aa8f9aec48b8b7ff9174e5b4f1d02aacc1b30b7e146b2d2ad8c5e11535c109`,
timed out after 302 seconds during the child-kill scenario.
The intermediate event-based stop could wait on a lock after the child died.
The final repair uses a control pipe, not that shared event.
A permanent regression checks the external child-kill path.

The first test submission, action
`049e4d16e061db244926f171f748fd87f2a5c9dbfcbf5f6f845870fc5a11c25c`,
stopped before pytest because its interpreter had no installed PB SDK.
This action has no test verdict.
The later submissions used the qualified interpreter below.

### Final verification

All jobs used priority zero, class `x86`, two CPUs, four GiB memory, and a 300-second execution limit.
The interpreter was `/home/rob/venvs/pq-pin-fca4c6ce0/bin/python`.
The test jobs used `/tmp` for scratch.
PB verified the installed PrismaBuild and Tessera commits before pytest.

| Test file | PB action | Result |
|---|---|---|
| `tests/test_container_cache_peak_2463.py` | `be4cd0f4a8dc0ef9908f44ac1e1ad1f6eff96254bffa025084db5ff71d30cd4c` | 22 passed |
| `tests/test_container_cache_charge_1091.py` | `dcb1c772ecace07b59d345670e15f37a9217c94605905e7685391f92d435dc66` | 30 passed |
| `tests/test_container_cache_roots_1072.py` | `0b8faa87e63c4c654ec69640da6935f3f1e50c5bdcc23a91d62e8740ebfdd25c` | 10 passed |

The test command was:

```sh
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbtest.py \
  --checkout /home/rob/wt/ig/prismaquant-2463 \
  --python /home/rob/venvs/pq-pin-fca4c6ce0/bin/python \
  --tmpdir /tmp --tag x86 --priority 0 --mem-gb 4 \
  --timeout-s 300 --wait-s 1800 \
  --json /home/rob/fleet/ceo/exec/ig-pq-2463-l1-a4/final-pb.json \
  tests/test_container_cache_peak_2463.py \
  tests/test_container_cache_charge_1091.py \
  tests/test_container_cache_roots_1072.py
```

PB action `4583b60809bc92bac35974083885ec0140e6bb8be0bbc49f0941a1a98cdad007`
ran the temporary lifecycle smoke.
It checked compile syntax, cache growth followed by deletion, external child death, and the complete CPU quantum command.
The growth peak exceeded the final size.
The killed child produced an invalid, incomplete receipt with exit code minus nine.
The complete CPU quantum produced 110 gap-free samples and a zero child exit code.
Its receipt bound the invoked command.

The smoke command was:

```sh
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /home/rob/wt/ig/prismaquant-2463 --tag x86 --cpus 2 \
  --demand mem_gb=4 --priority 0 --timeout-s 300 --wait-s 1800 \
  --env TMPDIR=/tmp --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 --env PRISMAQUANT_DEV_MODE=1 \
  --env PRISMAQUANT_DETERMINISTIC=1 -- \
  /home/rob/venvs/pq-pin-fca4c6ce0/bin/python -u \
  -m tools._container_cache_lifecycle_smoke
```

The PB snapshot retains the temporary smoke source; the branch removes it after verification.
The smoke CAS receipt digest is
`f0a1af21957c476263a8e40b0b399485fe32f3b6249204cc798188030c9457c3`.
The smoke payload digest is
`34bc5edeca968dcec41308e6de00e96d8c0109cd4493cc41e928a3946716e21f`.
The CPU quantum receipt digest is
`71964ec31db17ca2877644ea4c8663c522fb09c253ddb3953ef3fc19f2b5b373`.
The CAS payload contains that receipt and its process profile.

This CPU smoke proves lifecycle behavior, not the GPU ceiling.
The CPU runtime has no Triton or NVIDIA power tool.
The repair changes no production default, cache declaration, quota, cleanup contract, or crash-recovery claim.
Keep PQ #2463 open for the pipeline's independent check.
