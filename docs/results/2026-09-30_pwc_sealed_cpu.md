# Bounded PWC sealed-load CPU profile

Refs #1295. This is a host-I/O slice, not closure of the I/O-consolidation
umbrella. No kernel, quantization math, export bytes, serving lane or runtime
pin changed.

## Change and safety boundary

Bounded `ProductionWeightCache._load_file_tensor` requests the existing
`io_engine.load_file(..., sealed=True)` path. The existing cache decoder maps
the sealed pages with private `torch.load(..., mmap=True)`; the engine closes
the buffer after decode while the tensor retains its mapping. No new cache,
preloader, decoder or memory-residency owner is introduced.

The unbounded legacy path is unchanged. Digest binding, staged-tier refusal,
resident-window signatures, serialization checks and tensor-mutation guards
remain mandatory. Other nonstream callers are not migrated by this slice.

## Experiment

The same driver creates a BF16 `[2048, 65536]` tensor (268435456 bytes), stores
one 268437033-byte Torch shard, then performs four real PWC get/drop iterations
in a fresh measured process. The creator exits before measurement. Both runs
use Python 3.12.3, Torch 2.11.0+cu130, one native and one interop thread, the
current b40c93cb Tessera pin, and local ext2/ext3 action scratch. CUDA is hidden
and remains uninitialized.

This is an artificial local-file fixture, not a model, NFS workload, cgroup
residency test or GPU/serving qualification. PB selected different admitted
workers and cores: BEFORE sparklina `[6]`, AFTER sparky `[19]`. External load
also differs. These are not isolated throughput measurements.

## In-process measurements

Values below are bytes from the existing `/proc` status reader. The first
iteration distinguishes the live tensor's allocation from memory after drop.

| Measurement | BEFORE: bytes/non-mmap | AFTER: sealed/private mmap |
| --- | ---: | ---: |
| First pre-load RssAnon | 554369024 | 553857024 |
| First live RssAnon | 824954880 | 553885696 |
| First dropped RssAnon | 554422272 | 553885696 |
| First live RssShmem | 0 | 268439552 |
| First dropped RssShmem | 0 | 0 |
| Last dropped RssAnon | 554422272 | 553885696 |
| Maximum sampled VmHWM | 1441992704 | 1171218432 |

The live anonymous-memory increment is 270585856 bytes before and 28672 bytes
after. The mapped tensor is still resident memory: the AFTER run reports it
as shared-memory RSS, which disappears after drop. It is not a zero-residency
load.

**Negative result:** the BEFORE fixture did not reproduce retained anonymous
memory after drop. Do not describe the change as fixing such retention on the
basis of this fixture. It demonstrates the decoder boundary and the live
allocation class, not that broader production symptom.

Both runs include cProfile, rather than inferring where time went from logs.

| cProfile function | BEFORE calls / cumulative seconds | AFTER calls / cumulative seconds |
| --- | ---: | ---: |
| `gc.collect` | 8 / 5.082983 | 8 / 0.882422 |
| `io_engine.read_file` | 4 / 1.701141 | 4 / 0.777440 |
| `SealedBuffer.fill` | 0 | 4 / 0.049487 |
| Whole profiled driver | 7.334018 | 1.831304 |

The driver wall includes garbage collection. Different workers, cores and
external load prevent an isolated speedup conclusion. No production,
model, NFS or GPU throughput improvement is claimed.

## Box telemetry

The bounded Netdata capture includes CPU, I/O, RAM, CPU pressure and I/O
pressure on both Sparks and dl380g10: 15 series per run, zero capture errors.
Eight samples per host overlap the exact BEFORE window; two per host overlap
AFTER. Sparse AFTER sampling limits any load or isolation inference.

| Host | BEFORE mean CPU user/system/iowait (%) | AFTER mean CPU user/system/iowait (%) |
| --- | --- | --- |
| sparklina | 12.555698 / 6.313191 / 0.222645 | 6.214842 / 4.718937 / 0.735642 |
| sparky | 22.026823 / 13.790248 / 0.215935 | 33.765505 / 6.126178 / 0.101130 |
| dl380g10 | 5.243864 / 2.616414 / 0.102541 | 8.113764 / 2.635279 / 0.088067 |

BEFORE window: 1790787503.905653–1790787511.2396727.
AFTER window: 1790788998.753672–1790789000.5849774.

## PB evidence

All runs use PB, priority -10, explicit timeouts, one reserved CPU/native
thread and 4 GiB memory. The sentinel is checked immediately before each
submission. The GB10 CPU-only profiles require the aarch64 CUDA-build Torch
allocator; neither requests a GPU nor uses `--measurement`.

| Run | Action key | Outcome |
| --- | --- | --- |
| Original behavioral RED | `6656f9e3dedae924f3a73bc5c92d1d0014645a47d7cc3eea8fc3ab1ff1461462` | dl380g10, rc1; 2 failed, 0 passed, 0 skipped; 2 collected/ran, reconciled |
| BEFORE profile | `dd9ce32e42cabf4d9340e8c5041f20a5f2be1f769cdcf9570e6cd8d21ddf26e4` | sparklina, public terminal executed, rc0 |
| AFTER profile | `3c33c93414333b1b197c641f79af558e5f4afa32d608fc7a76f6cdd87fd3c0d3` | sparky, public terminal executed, rc0 |
| Corrected nine-file GREEN | `3edd119fb9710333759de296cfc3afa99391acddac507d30447ad89cf84a8fe2` | dl380g10, rc0; 124 passed, 0 failed, 0 skipped; 124 collected/ran, reconciled |

The RED loads returned correct tensors and then failed
`AssertionError: bounded PWC load copied verified bytes`: the decoder received
bytes rather than the existing sealed buffer. A failed action has no success
CAS receipt. The first GREEN snapshot had 122 passed and two failed legacy
buffered-read spies; it predates the observers for the sealed fill path.
The corrected snapshot passes all 124 cases. Its public terminal is
executed/rc0, and its CAS receipt is
`a66b812db9fdb049a791f09f0efa6cc24f64e7f34194e477009d5ef9165a4cd4`;
29026-byte result SHA-256
`5185fddbac315c53ede2b44fd4c24ab327f47448c0852e278c322d34ce6641e5`.
The public local-result claim verifier passed all checks including payload
hash; it does not verify the full worker attestation. Action/worker binding
and actual payload bytes/hash were independently checked. Compile evidence
will be appended when verified.

BEFORE CAS receipt:
`8928ac0bdec74fbf5a7ea56ab286b207083f4854738ec5a5595f4534d3c7186e`;
97403-byte result SHA-256
`1fdd12a9f3611404339c175eaf35040b3c32ebaed846ce882eefa1f50805315d`.
AFTER CAS receipt:
`3c804cc5824a1521e01f7373010b2e1a0677095a8ad5643d10d73a4f045a1bbd`;
99853-byte result SHA-256
`a58d2ad3283220f191245b9036e7d6853f09cef62ea0c8dc507fcb8eff943411`.
Producer action/worker and payload lengths/digests were independently checked;
public `pbwait --wait-s 0` confirms both executed/rc0.

## Reproduction and raw artifacts

The exact PB recipes, frozen profiler and orchestration live under
`/home/rob/tmp/claude-campaign-20260926/tmp/p2p3/prismaquant/`:

```bash
export TMPDIR=/home/rob/tmp/claude-campaign-20260926/tmp
bash "$TMPDIR/p2p3/prismaquant/pwc-sealed-red.sh"
bash "$TMPDIR/p2p3/prismaquant/pwc-sealed-profile-before.sh"
# Apply only the bounded sealed-load change, then:
bash "$TMPDIR/p2p3/prismaquant/pwc-sealed-profile-after.sh"
```

These historical recipes specify the task worktree and are evidence, not an
instruction to repeat completed measurements. Each profiling action uses the
same `pwc-sealed-profile.py` and `pwc-sealed-profile-orchestrate.py`, with
`--tag gb10 --priority -10 --timeout-s 120 --cpus 1 --demand mem_gb=4`, bounded
OMP/MKL/OpenBLAS threads, and the main-matching interpreter
`/home/rob/venvs/pq-pb059953bc-tessera-b40c93cb/bin/python`.

Raw profile logs, complete cProfile functions and status/I/O snapshots,
verified result metadata and full Netdata series are retained as
`pwc-sealed-profile-before.log`, `pwc-sealed-profile-after.log`,
`pwc-sealed-before-evidence.json`, `pwc-sealed-after-evidence.json`,
`pwc-sealed-netdata-before.json` and `pwc-sealed-netdata-after.json` under that
artifact directory. The profile plan records the unchanged fixture and
acceptance boundary in `pwc-sealed-profile-plan.md`.

## Delivery verification update

The source was rebased onto main
`20f66d140fef208fc78a32f4dbbcbe54bf12bd91`; core/test head
`31f2dcf043c24df5aee8a6b32272bbdefec9e5bc` then passed all nine files:
PB action `ba0bb9567fd0ed4464affaf03b8091e6effc2a1a5f5fa758d899c2f657a706f6`,
dl380g10, executed/rc0, **124 passed, zero failed/skipped**, 124 collected/ran,
no reconciliation problems, 120.29 s pytest. CAS v3 receipt
`7a5ed7487d513d7cdc37eda197f86140eda88c328e9af7ebeb5a5789e496240c`;
29026-byte result SHA-256
`856ac9abc0b36f732e142d4305b4ea856373834b7d2a5cef59b6dd945fb84846`.

Three-module compile action
`62298183dc460556b936849cfff57cb76f69d2dcee120ad36009530a9c488966`
executed/rc0 on dl380g10; receipt
`e24ebe86cae3b5bfd0b50078a05c464fb0eb1de459742db4808d50ddb3f5ce2d`;
27-byte result SHA-256
`7ed73160ccc8b883f82917d4047587809fee8060fc4460964e50094b70c9bc9b`.
Public terminals, producer action/worker bindings and actual payload lengths
and SHA-256 were independently checked. These checks do not verify the full
worker attestation. The evidence-only appendix is a later documentation commit,
not a claim that the earlier action included its own future receipt.

Recipe and reconciled/evidence artifacts:
`pwc-sealed-final-green.sh`, `pwc-sealed-final-green.json`,
`pwc-sealed-final-compile.log`, `pwc-sealed-final-evidence.json`, under the
campaign artifact directory above. Both clients independently checked the
serve-window sentinel and used priority -10, one reserved CPU/native thread,
the current b40-pinned x86 interpreter, 240 s tests/120 s compile deadlines,
and 4 GiB tests/2 GiB compile. No GPU or model qualification was performed.
