# Fused per-token FP8 QDQ before/after — 2026-10-10

**Historical evidence, selectively recovered from `ecaf0114` for #2867.**
The original method, tables and action keys below describe the earlier v1
kernel, not repaired-v2 performance. The corrective receipts are also inherited
history from that delivery, not current-worktree qualification or receipts
verified in this work package. See [Current qualification](#current-qualification-for-2867)
for the pre-run decision and fresh CPU/GB10 qualification receipts. Historical machine/image
names below identify evidence, not deployment requirements.

PQ #1398 fuses the joint-statistics hook's per-token FP8 activation QDQ
(about ten torch ops per observed expert invocation) into one Triton launch
(`prismaquant/kernels/fp8_per_token_qdq.py`, kernel id
`prismaquant.triton_fp8_per_token_qdq.v1`). The fused leg is the default path
on eligible CUDA inputs; `fp8_dynamic.fp8_qdq_reference` serves CPU and
unsupported inputs and is the oracle baseline.

## Method

QDQ-only, following `fp8-native-activation-parity-2026-09-07.md`: call only
`fp8_dynamic_activation_qdq_vllm(...).dequant` on GPU-resident bf16 inputs
(256 rows, seeded randn with sparse 40x outliers), time with CUDA events
(20 warmup, 11 rounds of 32 calls), count kernels from an exported profiler
trace in a fresh process per arm, and sample device power over a sustained
8 s phase per arm plus a Netdata GPU-power pull per power window. The before
arm forces the reference through `PRISMAQUANT_DISABLE_FP8_FUSED_QDQ=1`; the
after arm uses the fused default. Arms interleave per width in one exclusive
GPU job, so both arms share the box and the inputs.

Both arms run in one snapshot (`e8871bdb1f`); the before arm forces the
reference through the kill switch, so no code drifts between the arms.
No numerics change: the oracle
(`tests/test_fp8_per_token_qdq_oracle.py`) holds bitwise on both legs,
including -0, midpoint ties, zero rows, saturation, E5M2 and NaN/Inf.

Environment: NVIDIA GB10 (compute 12.1), exclusive GPU on sparklina, torch
2.13.0+cu130, Triton 3.7.1, image
`prismaquant-glm-producer:content-qualified-20260908` (the 2026-09-07
source-reference image). Bench: `experiments/fp8_per_token_qdq_bench.py`;
Netdata pull: `experiments/fp8_netdata_pull.py`.

## Results

PB action `597669b33f088655f7abc1a9ef5e532f4a70f2466eaaf5161151e0c191241ac4`
(executed on sparklina in 126 s). Power means are the in-band nvidia-smi
phase means (38 samples per arm); Netdata means cover the same windows
(9 samples per arm, chart `nvidia_smi.gpu_*_power_draw`).

| K | ref ms/call | fused ms/call | speedup | ref kernels | fused kernels | ref dev µs | fused dev µs | ref W (smi) | fused W (smi) | ref W (netdata) | fused W (netdata) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 0.0606 | 0.0289 | 2.10x | 12 | 1 | 17.1 | 6.7 | 22.0 | 26.9 | 24.4 | 27.0 |
| 1536 | 0.0621 | 0.0289 | 2.15x | 12 | 1 | 28.5 | 7.5 | 33.9 | 30.2 | 19.4 | 25.3 |
| 2048 | 0.0610 | 0.0291 | 2.10x | 12 | 1 | 34.2 | 6.7 | 38.7 | 30.8 | 40.7 | (see receipt) |
| 4096 | 0.1081 | 0.0289 | 3.74x | 12 | 1 | 106.8 | 8.0 | 44.8 | 34.1 | (see receipt) | (see receipt) |
| 16384 | 0.6173 | 0.1287 | 4.80x | 12 | 1 | 657.1 | 127.1 | 41.1 | 39.0 | (see receipt) | (see receipt) |

Full JSON (per-call minima, power samples, calls/joule, kernel names, phase
epochs) is in the action log (`SUMMARY_JSON`) and its CAS payload. The fused
trace shows exactly one kernel per call; the reference trace shows the twelve
expected ops (upcast, abs, amax, div, clamp_min, div, clamp, two casts, scale
multiply, scale fill, denominator fill).

## Reading

The win is launch-bound at small K (fused wall stays near 0.029 ms while
device time is under 8 µs) and grows with K (4.8x at K=16384). Device power
sits at 22–45 W against the 140 W envelope on both arms, so neither arm
saturates the box; this matches the 2026-09-07 observation for this helper
and establishes no serving throughput claim. Fused power reads slightly
higher than reference at small K because the faster arm retires more calls
per second in the sustained phase; calls/joule in the receipt is the fair
comparison.

## Limits

The action keys and table above are historical results for the stated
snapshot, not independent GPU qualification of subsequent repairs. Recorded
numerical qualification is GB10/sm121 only. The fused kernel has no autograd
backward; grad-enabled inputs requiring gradients use the reference. A grid
spy counts Triton invocations, not ancillary torch GPU work; actual kernel
counts come from the separate profiler trace method above. The profiled
historical implementation used two input passes for K > 4096; see the v2
corrective implementation below for the single-read path.

This is QDQ-scoped evidence, not a full-model or hook-level number. The
`fe9730f7` full-backward profile on the MTP harness and its Netdata series
remain open (child issue): that harness serves the higher-priority
statistics-accumulate fuse, needs a tessera-quant GPU image plus the M5
plan/inputs/model, and its arms compare accumulate variants, not QDQ legs.

## Checks

| Check | PB action | Actual result |
| --- | --- | --- |
| Oracle + parity, x86 CPU | `65870d66fadf625d6333dbd4b5718979b8fc591b8012ce0207eb9003bb1d2e00` | 19 passed, 60 CUDA-tier skips |
| Parity, x86 CPU | `d66d97a9d6775794215c0faf25ebd1bf8573397c8b9fc78c69552af92c879c8e` | 2 passed, 2 CUDA skips |
| Format registry, x86 CPU | `7736916ae613c47ede9a45e3dfc28060d0d1221bb9bf9616d1a6bf7cd392d4eb` | 15 passed |
| Oracle + parity, GB10 CUDA | `ff9dc584ee5b15f8cdda6356317105bad19768c662db7690f42597cb89c142fd` | 98 passed |
| QDQ before/after + power | `597669b33f088655f7abc1a9ef5e532f4a70f2466eaaf5161151e0c191241ac4` | table above |
| Netdata reachability | `6429ce4f2082ccb6bc7fc57ab8cd4b3f00ea5f168370b2dc88648125d86c4c11` | worker-local Netdata answers |

## Historical corrective implementation and qualification (`ecaf0114`)

This section records the repaired-v2 delivery as reported at `ecaf0114`;
its pass counts and receipt strings are preserved verbatim as historical
attribution, not a claim that the current checkout has been run or qualified.

The v2 kernel replaces the wide-row reload loops with a whole-row register
tile, bounded to K <= 16384. Each eligible row is loaded once; larger rows use
the reference. The compiled-code regression checks one logical input load,
PTX loads before stores, no backward load loop and no register spills at
K=4, 5000 and 16384, for both formats with and without dx. A grid spy alone
would not establish those properties. Noncontiguous inputs may still require
a torch materialization before the fused launch.

On the admitted torch 2.11.0+cu130 / Triton 3.6.0 GB10 environment, the old
native packed FP8 conversions produced correct stored codes but corrupt
in-register dequant values for narrow rows. A full-width integer-register
FP32-to-FP8 RTNE conversion and exact code-bit decoding avoid that lowering.
New regressions exercise every finite code, zero signs and subnormals at
packing boundaries for BF16/FP16/FP32, plus both formats' NaN/Inf and dx.
Logical negative views and forward-AD duals use the reference, as do
noncanonical element maxima (preserving torch cast overflow semantics and
the final scale clamp). The kernel also retains the final `2**-127` floor.

The timings/power above remain the historical implementation's measurements,
not measurements of v2. They are not a renewed performance claim. Final
corrective test receipts are recorded below; a CPU skip is not CUDA evidence.

Corrective verification commands (repository root, scratch outside checkout):

```sh
# GB10, torch 2.11.0+cu130 / Triton 3.6.0
python -m pytest tests/test_fp8_fused_repair.py tests/test_fp8_per_token_qdq_oracle.py tests/test_fp8_dynamic_native_parity.py tests/test_joint_aura_one_squaring.py tests/test_joint_aura_packed.py tests/test_joint_replay_aggregate_guard.py -q --basetemp="$TMPDIR/pytest"
# x86 CPU, including docs, duplication and affected-caller guards
python -m pytest tests/test_fp8_fused_repair.py tests/test_fp8_per_token_qdq_oracle.py tests/test_fp8_dynamic_native_parity.py tests/test_joint_aura_one_squaring.py tests/test_joint_aura_packed.py tests/test_joint_replay_aggregate_guard.py tests/test_format_registry.py tests/test_docs_staleness.py tests/test_architecture_doc.py tests/test_docs_entrypoints.py tests/test_duplication_baseline.py tests/test_autoscale_fp8_resident.py tests/test_layer_streaming_fp8.py tests/test_mxfp8_ue8m0.py -q --basetemp="$TMPDIR/pytest"
```

Observed: **232 passed, 4 skipped on GB10** (25.80 s); the four skips require
two CUDA devices. **179 passed, 145 skipped on CPU** (24.85 s); CUDA tests
skip there. The four named regression files also passed before the repair:
24 passed on GB10, 22 passed / 2 skipped on CPU. Their files and the supplied
oracle are unchanged. No new power/profile measurements were run, and
cross-device restoration on a two-GPU host remains unverified.

GB10 final qualification:

```text
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f02b48cc28f53d587cee690ed9ada8475bad2dd78c5be22d3550838498c7dd80","size":1777}},"operation":"646248f8bd71047fa6cf01f2216caee73965b08c31a5da4a4b2475be8211ece9","receipt":"acb179a9605ed2a1dd1ef951756b9b7b653eabbf1f304b2493f980993a33589c"}]
```

CPU final qualification:

```text
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"492c82c750979b7c912b446985ef4ab9a305dacbdee1c40aaa788dca2e9b2157","size":1539}},"operation":"563f65e48cc7825665977db979a5f2eb197e28a4c5de2b87f410f6fa7c9feb77","receipt":"614cba463783716478ecabe0d6f9c8fddedf92e6ba9527de78c37d9f3b04dc01"}]
```

Before-repair regression baselines (the GB10 action first reproduces the
four oracle failures, then runs the passing regressions; its final shell
exit status alone is not a passing oracle):

```text
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"a684611e542b75ad2cf93f2cbd97de02c297c670e17fb19f48ed756c723c42ac","size":14397}},"operation":"4084ea8af0af5fbfeb4391d8a118f11509ad21e16e19c689c53d763192aba52a","receipt":"16fe482f8946a5ad2ccacca7ccea2592f0b0f21d6ae6f8d41a59ed3125d79f39"}]
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"70465d12e93d1d06c50849263074291a50c4f2c4db702b4cacff4fd0c90c8294","size":832}},"operation":"8b88a9a44308763640f7361be65c64d5115e5bfa51f04ec7f7ae7f2fc72f02f4","receipt":"7ed080acc7b00887232e7883478b5b86a7d1bc309f097f680dfba8b970dd89ef"}]
```


## Current qualification for #2867

Status (2026-10-24): **Integrated qualification passed on CPU and admitted GB10.**
The PRE-RUN decision below preceded the single GPU run. No latency, power or
repaired-v2 performance claim is made.

### Bounded delivery contract

The recovered repaired-v2 contract is BF16/FP16/FP32 two-dimensional rows with
1 <= K <= 16384 and canonical E4M3FN/E5M2 maxima 448/57344. Offset eligibility
must include power-of-two tile padding within signed-int32 indexing, not merely
M*K. Wider rows, noncanonical maxima, logical negative views, unsupported
inputs, CPU inputs, reverse-mode inputs requiring gradients when grad is
enabled, and forward-AD duals (including under `no_grad`) use
`fp8_dynamic.fp8_qdq_reference`. Preserve the final `2**-127` scale floor.
Noncontiguous inputs can require a separate torch materialization; empty row
batches launch nothing. Guard the input device, use that device’s current
stream, and restore the caller’s device; host constants are rounded on CPU,
independent of torch’s default device. Eligible CUDA import/compiler failures
propagate, rather than silently passing through the reference.

Both public per-token entries return codes, FP32 scale and FP32 dequant; the
`_with_dx` entry optionally emits FP32 `dx = dequant - input.float()` in the
same launch. This is not a joint-hook roundtrip or fused hook backward: the
joint-hook dx subtraction and weight QDQ are unchanged. The one-row register
tile and full-width FP8 code conversion/decoding are the repaired design, not
proof furnished by counting launches. **One launch is not single-load proof**:
qualification must inspect compiled TTIR/PTX for one logical input load,
loads before stores, no backward reload loop, and no local-memory/register
spills, for both formats with and without dx at K=4, 5000 and 16384.

### PRE-RUN decision

- Decision unlocked: whether the recovered #2867 repair meets bitwise public
  CUDA parity and the bounded compiled single-read contract without regressing
  existing FP8/joint callers. This is a correctness admission, not performance
  promotion or cross-architecture qualification.
- Cheapest decisive check: **one admitted GPU command**, combining **two
  acceptance files** (`test_fp8_fused_repair.py`,
  `test_fp8_per_token_qdq_oracle.py`) and **four regression files**
  (`test_fp8_dynamic_native_parity.py`, `test_joint_aura_one_squaring.py`,
  `test_joint_aura_packed.py`, `test_joint_replay_aggregate_guard.py`).
  Budget approximately **10 GPU minutes**, including compilation/setup;
  this is an estimate, not observed runtime or a reservation receipt.
- Why GPU: CPU checks cannot prove CUDA bitwise parity or compiled TTIR/PTX
  properties; CUDA skips on CPU are not acceptance. Use the admitted GPU arm
  and record its actual device, torch/Triton versions and environment. The
  inherited GB10 evidence does not qualify other GPUs or this snapshot.
- Focus: narrow packing boundaries (K=1,2,3,4,7,8,16,48), canonical finite-code
  rows including signed zero/subnormals in BF16/FP16/FP32, both FP8 formats,
  NaN/Inf and optional dx, plus compiled K=4,5000,16384 in both dx modes.
  Preserve layout, offset, device/current-stream and AD fallback coverage;
  two-device restoration is unverified if the admitted host lacks two GPUs.
- Admission/stop: the main agent must use the project’s GPU admission and
  execution policies, an isolated configured environment and explicit durable
  `TMPDIR` outside the checkout. Run only after the recovered implementation
  and acceptance files are present. Retain failures/skips and investigate
  rather than replacing failed CUDA coverage with CPU success. No benchmark,
  full-model probe or additional GPU command is authorized by this note.

Single command recorded before execution (repository root, admitted GPU arm):

```sh
python -m pytest tests/test_fp8_per_token_qdq_oracle.py tests/test_fp8_fused_repair.py tests/test_fp8_dynamic_native_parity.py tests/test_joint_aura_one_squaring.py tests/test_joint_aura_packed.py tests/test_joint_replay_aggregate_guard.py -q --basetemp="$TMPDIR/pytest"
```

### Current-run results and receipts

Reconciled adapter/kernel snapshots exactly match `ecaf0114`, including its
intervening `a0300a1a` protections. Git blob IDs: adapter
`2326cc2ea262064fc0806ee78b3152c8e80c2211`; kernel
`5084648572dd5ea3c1ac93bce31fc9214f4b3324`. Acceptance files and joint callers
were not modified. Test fixtures are self-contained; `_real_activations` is
seeded synthetic data, not captured activations. Canonical-row constants retain
their independent LFM capture provenance and diagnostic SHA256 in the oracle;
E4M3 code/scale constants are captured, E5M2/dx expectations reference-derived.

CPU combined command (the command above, default arm): **107 passed, 153
skipped**, exit 0, 3.57s; Python 3.14.4, torch 2.11.0+cpu. Skips do not
qualify CUDA. Fresh admitted `arm="gpu"` run of that same command:
**302 passed, 4 skipped**, exit 0, **30.43s pytest time** (not total action
wall time), within the approximately ten-minute estimate. The arm supplies one
GB10 sm_121 GPU, Python 3.14.4 and torch 2.11.0+cu130. The plain pytest output
does not print the Triton version. Retained stdout was read back after execution.

All twelve compiled-code cases passed: K=4,5000,16384, E4M3/E5M2, dx off/on.
They inspect the admitted compiled kernel's TTIR for one logical `tt.load`,
PTX for global loads before stores, absence of backward branches and local
loads/stores, and `n_spills == 0`. Public CUDA parity and one `(M,)` launch /
zero-row no-launch assertions passed independently. This establishes only the
tested bounded correctness/compiled properties, not performance or every
compiler/architecture. Four two-device guard cases remain skipped on the
single-GPU host; no TP2 qualification or new TP2 dependency is claimed.

CPU combined receipt:

```text
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"beebaa031a3b631996a99e887536081aa679a4e2ad9b2f511635816828406bc5","size":1458}},"operation":"ee9bc1798b077daa426d50bee5139624a772eaf2579b4955f52b1cb2db007210","receipt":"cc3f70836bbd1d961cf730592f582569d006f14d2e9610f970090296a26a29c5"}]
```

Fresh admitted GB10 receipt:

```text
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"874b3c3972c27d2b3608d5c4f4ea96bbeb29c3f51a870557c88755df719b3a6a","size":1857}},"operation":"061c613dd2523e5863e92cd94ef5232339e6408bec6b4e2fa494481e118d2fc0","receipt":"3b6b141db273b8bbf50de3697e3976fe7547c4cdf61425bcc6f2e978016d91fc"}]
```

The separately repeated CPU regression command and rename contract were:

```sh
python -m pytest tests/test_fp8_dynamic_native_parity.py tests/test_joint_aura_one_squaring.py tests/test_joint_aura_packed.py tests/test_joint_replay_aggregate_guard.py -q --basetemp="$TMPDIR/pytest"
python -m pytest tests/test_fp8_per_token_qdq_oracle.py::test_no_symbol_still_says_fallback -q --basetemp="$TMPDIR/pytest"
```

Results: **22 passed, 2 CUDA skips**, then **1 passed**, both exit 0,
matching the supplied before-integration regression baseline. `git grep`
for the retired symbol had no tracked-text hits; `git diff --check` passed.

```text
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f393ba9a26a81623d612653140f396835739ecb1910cc9cf905641d202a2bc36","size":1100}},"operation":"1ef96abc2d2c3189b5ccb732c0d50255a5eb13b34b2047d1b97f5ffa8875eb2c","receipt":"f8a919ca5f0fef61112235662e6949f3fff14edf3099e23f4c970dc36b932eea"}]
```

W2/#2667 instrumentation and W3/#2668 campaign measurements, publication and
workflow transitions remain outside this reconciliation.
