# Retained-render qualifier: sealed CPU decode, 2026-09-30

Refs PQ #1295. This is the qualifier-render host-IO slice, not completion of
that consolidation parent or a model, codec, wire, GPU or serving qualification.

## Change and unchanged gates

`tools/qualify_t4_overlay.py:read_cell` requests the existing
`io_engine.read_file(..., sealed=True)` buffer and reuses
`ProductionWeightCache._decode_file_tensor`'s private mmap decoder.
`contextlib.closing` closes the descriptor on successful decoding and decoder
failure; the returned tensor owns its mapping. A type check confirms the
engine's requested sealed-buffer contract.

First qualification has no prior render digest. It still uses the existing
bounded reader's staged readiness, PB copy-time digest, lease, stat and tier
accounting checks; it does not switch to the digest-required `load_file` gate.
Wire verification, GPU finishing, tensor equality and qualification identity
are unchanged. No new reader, cache, pool, format, codec, numerical method,
runtime pin or serving gate is introduced.

## Controlled experiment

The existing campaign PWC allocation driver was extended with optional
`PQ_CPU_PROFILE_CONSUMER=qualifier`, preserving its default PWC behavior.
BEFORE and AFTER use the same driver and orchestration:

- BF16 tensor `[2048,65536]`, 268435456 tensor bytes and 268437033 file bytes;
  four actual render-read/decode/drop cycles.
- A separate creator exits before the fresh measured process. The real
  `read_cell` render/hash/finite/identity leg runs; wire verification uses the
  existing CPU fixture's controlled four-byte doubles. No `finish_cell`, codec,
  GPU decode or numerical qualification runs.
- Both actions ran on sparky, local ext2/ext3, Python 3.12.3, Torch
  2.11.0+cu130, one native/interop thread. CUDA was hidden and uninitialized.
  The exact current b40c93cb pin gate passed. BEFORE affinity was `[18]`;
  AFTER affinity was `[9]`. These different cores/times and external load are
  not an isolated throughput comparison.
- Shared package source digest was
  `6ce10d61f7c66c30ea9970cc2f858a00fc9bdab07802e9445fd0f6c156aa5014`.
  Qualifier tool digests were
  `9f4d8e800f2c9ec9b25ad94de786ac698b0f3ccc64bc5e85bc0b4fa5e672bf03`
  BEFORE and
  `772c192da93bf7025e19d6779d2b21c3598a124048d62153271329cd3a448659`
  AFTER. The tool digest is independent of the package closure digest.

### Process memory (bytes)

The first pre-load anonymous RSS was 566951936 BEFORE and 568070144 AFTER.
These samples include qualifier temporary allocations, not only decoded tensor
storage. Each entry below is measured after the real load or drop.

| Cycle | BEFORE live RssAnon | BEFORE dropped RssAnon | AFTER live RssAnon | AFTER dropped RssAnon | AFTER live/drop RssShmem |
|---|---:|---:|---:|---:|---:|
| 1 | 837574656 | 567042048 | 568143872 | 568143872 | 268439552 / 0 |
| 2 | 1378672640 | 1108140032 | 702394368 | 702394368 | 268439552 / 0 |
| 3 | 1515020288 | 1515020288 | 1239396352 | 1239396352 | 268439552 / 0 |
| 4 | 1515020288 | 1515020288 | 1239494656 | 1239494656 | 268439552 / 0 |

BEFORE live/drop RssShmem was zero in all four cycles. Maximum sampled
cumulative VmHWM was 2134990848 BEFORE and 1865646080 AFTER.

The live tensor pages now appear as shmem and return to zero on every drop.
Anonymous growth remains after drops: `read_cell` also performs `isfinite` and
tensor-identity work with temporary allocations. These measurements do not
attribute all anonymous growth to the decoder, show zero residency, prove
cgroup charge drops, or establish that all retained-memory behavior is fixed.

### In-process profile and system telemetry

| cProfile frame | Calls BEFORE/AFTER | Cumulative seconds BEFORE | Cumulative seconds AFTER |
|---|---:|---:|---:|
| `gc.collect` | 8 / 8 | 1.424640365 | 0.879567480 |
| `io_engine.read_file` | 4 / 4 | 0.814291080 | 0.781712183 |
| `SealedBuffer.fill` | absent / 4 | — | 0.054266810 |
| `_cb_cache_tensor_identity` | 4 / 4 | 0.459451973 | 0.434507642 |
| `torch.isfinite` | 4 / 4 | 0.447029128 | 0.542972170 |

Driver wall was 3.849571287s BEFORE and 3.225562066s AFTER. It includes GC
and import overhead; nested frame times must not be added. Expected parent
`read_cell` and `_decode_file_tensor` entries are absent from the captured
function lists. This coverage limitation is retained: the profile does not
establish complete parent-level time attribution, and these walls are not
published as a load speedup.

Aligned Netdata captured five CPU/IO/RAM/pressure series on each of sparky,
sparklina and dl380g10, with zero capture errors. The exact windows were
1790803667.3075428–1790803671.157117 BEFORE and
1790808892.703931–1790808895.9294972 AFTER. Four CPU samples per host overlap
BEFORE; three overlap AFTER. Mean CPU user/system percentages in those windows:

| Host | BEFORE user/system | AFTER user/system |
|---|---:|---:|
| sparky | 18.975746 / 9.437264 | 16.601918 / 4.648119 |
| sparklina | 16.531943 / 4.233961 | 9.301909 / 4.359679 |
| dl380g10 | 5.549697 / 7.978329 | 15.548837 / 3.628308 |

Box load differs. No isolation, GPU work-per-joule, model/NFS throughput or
production speedup is inferred. No GPU was requested or initialized.

## Commands and durable evidence

All execution used published PB clients, priority -10, explicit timeout and
an independently checked WINDOW_ACTIVE. Tests used x86/current-main b40,
one CPU/thread/shard and 4GiB. CPU profiles used eligible GB10 workers for
the aarch64 CUDA-build allocator dependency, one CPU/thread,4GiB,120s, with
no GPU demand or `--measurement`. PB preserved placement/affinity.

Campaign root: `/home/rob/tmp/claude-campaign-20260926/tmp/p2p3/prismaquant`.

```bash
bash /home/rob/tmp/claude-campaign-20260926/tmp/p2p3/prismaquant/qualification-sealed-red.sh
bash /home/rob/tmp/claude-campaign-20260926/tmp/p2p3/prismaquant/qualification-sealed-before.sh
# Only after consuming actual RED and BEFORE, apply the source fix.
bash /home/rob/tmp/claude-campaign-20260926/tmp/p2p3/prismaquant/qualification-sealed-green.sh
bash /home/rob/tmp/claude-campaign-20260926/tmp/p2p3/prismaquant/qualification-sealed-compile.sh
bash /home/rob/tmp/claude-campaign-20260926/tmp/p2p3/prismaquant/qualification-sealed-before.sh after
```

The scripts contain the exact published-client commands and resource envelopes;
the admitted payloads preserve the actual inline profiler bytes. Logs and
selected evidence JSONs are `qualification-sealed-before/after.log`,
`qualification-sealed-before/after-evidence.json` and
`qualification-sealed-netdata-before/after.json` under the campaign root.

| Phase | PB action | Result |
|---|---|---|
| RED | `62de5db9eb22176dd2019da3aaf4b2efa8550ecc2ea2b12da2389844e77902c0` | dl380g10,rc1,2failed/0passed/0skip,2collected/ran; no reconciliation problems |
| BEFORE | `6ba51a473e97d1d2dae78c0661621b532bfb817152e15f800b061b8edb43f795` | sparky,public executed/rc0 |
| AFTER | `cb49d2be2e165dc40c021c66f00f18666695b3f6d847ec77ee7964da1547a791` | sparky,public executed/rc0 |

BEFORE receipt:
`8658bbb7b2894764e663b908882cb9f155594244d5ca62ef7ec5cfef23ed5f68`,
149582-byte payload SHA
`fcb1813832496e39e814505f4b1e9f59dbc3edcdbf18b7a5eb7fff59caef6c22`.
AFTER receipt:
`c14753f010ded442a25b21d1edfc59e403012e0e8d56c8df2bb490e37d1261d5`,
152920-byte payload SHA
`b278787ba88bdbb98f96e16fb0d6999b2c15124552593d21936a9f2beefd95b5`.
Actual action/worker binding, payload lengths/SHA and public terminal return
codes were independently checked. Public local-result claims
`2563058c6958a1f297cadb7241c28811f1839a1572c7cc5184e6b0e12dce6356`
and `1fedc083efe72014b04cb7ed6472acdac1b4e167f519bd2da2604280dd6c5be8`
passed claim/receipt/payload checks. Full worker attestation was not verified
by that tool. Failed RED has no success CAS receipt; the initial malformed
pytest-argv client rejection was not a test result.

## Appended CPU validation receipt

The submitted source snapshot's GREEN action
`2262a845f8506e10824c5a42d50dcd5e9c0a3114162de4f2cde4b1e74212df4b`
executed on dl380g10 with rc0: **132 passed,0 failed,0 skipped**,
132 collected/ran, no outcome-reconciliation problems,132.92s. Receipt
`ae417c05c6d9c0357ff7f42385157c14e89fb7117e49c5d0334588df16ecb242`
binds a 30716-byte result with SHA
`b284c6b419d984951d08ead497de4748fd7c47af9d28037b479facc1e1eff633`.
Two-module compile action
`345c726da492110dd2074fe782af6efef233c2a84ca493892f5c04d95bfa49c0`
executed on dl380g10 with rc0; receipt
`1e934e857adead516ceaf5885a1d1400860a29198dd46c40e0739f83d08b906e`,
29-byte result SHA
`d90428eb6b8f9ce1af1670e0fbfd3aec46910ea03056f13ad4372733649faa05`.
Actual payload lengths/SHA, producer/action/worker binding, public terminal
rc0 and public local-claim checks were independently verified. These actions
precede this appended evidence and architecture link; they do not contain
future receipts. No full worker-attestation claim is made.

Evidence: `qualification-sealed-green-evidence.json` under the campaign root.
CPU regression doubles do not qualify the untouched GPU/wire finishing gate.
The broader #1295 parent remains open.
