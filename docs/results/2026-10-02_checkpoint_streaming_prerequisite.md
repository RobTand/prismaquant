# Checkpoint streaming executable prerequisite

PQ #2118 implements the bounded executable integration under #1366. The work is
stacked on frozen PR #2110 (`0ad5fe714ca38e4bdae3e39139ba2e303b32ee57`). Root owns
review, integration and the original GLM GPU comparison. No GPU workload,
default promotion, PB/Tessera pin change or runtime deployment occurred.

The existing incoming adapter now has an explicit immutable readset selection.
It moves original checkpoint rows to their consuming phases, binds that selection
to the protected PB claim through the shared authenticated metadata owner, and
keeps existing exact-reader leases/digests/residency and destination resource
checks. The dispatcher checks the real slice-derived incoming counts before
subtracting them from compute units. The [contract and finite two-arm protocol](../design/checkpoint_streaming_1366.md)
records supported regimes and the remaining original-row gates.

## PB CPU evidence

All tests used Python3.14.4, CPU torch2.11.0, transformers5.17.0, pytest9.1.1 on
`dl380g10`, admitted CPU1/memory4 GiB per shard and native threads1, priority -10,
300-second action ceilings. PB chose placement. Every shard verified noneditable
PB Git/RECORD `dc4803daaf09b6426083d2d36bd2a2da3d6832fe` and unchanged Tessera
`b40c93cb73745097e57a1ba4cf5b9eee166c759a` before pytest.

| Scoped checks | Passed / skipped | Action(s) |
|---|---:|---|
| New selection, phase/mode/claim negatives, real first-chain order, bitwise operands/final plane/costs/journal, default/grouped complete and partial resume | 29 / 0 | `76ab0410fcd7fed5d7a3160a84b2fc431c364c0036603d16d11dc78abd3feb16` |
| Existing CPU research behavior/refusals | 12 / 0 | `d03d446924b1733a44724d27e6dc2ace1763a2a96625bb91a4d1aeb03a1ae580` |
| Existing spill-phase plan | 8 / 0 | `67db75886e816af0b1ab94ea7f88e958f820c5f7d8dd5d781b3ed1eccfd57def` |
| Docs staleness / architecture | 19 / 0 | `f9cf6dfaea5f51b8fc7a24809c7b009a5d6766b9362ca5a8fc1623026e07a8d4`, `fab13f98398d627c2ccaace49bf57a4df0a7897927ce156537dbb525fc871837` |
| Compute phase/tail, existing streamed handoff, executable readsets, load-phase allowances | 79 / 2 | `278ec0e78ec821b5842e5dbf2392fdecf33586f54c8f97b28b17e5d0821a0b2b`, `1e8df08fd78bc9dfb79986ae9c6355343f0e92145243825e7cffeb2f1e3fd89a`, `2cd701583e8ebb535eaeb49163e2ec34e2aaef126b6ec059f6682e1010a8b89a`, `b3a20412a7b67cd37c4160d7649f3ce381944f5e3c507962bf1e731f4c88ee3c`, `d2e5600e104ba0fc0991bded6250f33f20e3393a9840f714894769e46eb0ef7a` |

The aggregate is **147 passed, 2 skipped**. Both skips are existing numeric
compute-progress/tail cases requiring an 8 KiB statx DIO grid unavailable on this
x86 worker. Their receipts retain that reason; they are not passes. New numeric
cases keep real direct IO with an explicit fixture grid-report double; protected
input identity and source-context doubles are separately labelled. Neither this
fixture nor its fake source installs qualify real PB residency/open lifetimes or
GPU device behavior.

Compile action `2df74b1d359dc9ba4ed4314efc982d201b4e6a2d349510936ee004a402d3925f`
exited0 on sparky, CPU1/memory2 GiB, no GPU request, using `/usr/bin/python3 -m
py_compile` over all five production/CLI modules and both test files. The client
automatically chose its submitting-host contract; that compile is not a device
probe. An earlier invocation naming the remote scoped interpreter was rejected
before publication because it was absent on the submitter; no compile was inferred
from that rejection.

The shared PrismaQuant package digest for every final qualification action is
`8c2cb03a1c131ebef89c775a284bf1d3d09ece6761118f04c6621b62f1b942c8`.
Read-only inspection fetched each exact PB input Git bundle, checked its SHA-256
and byte length, archived the named commit, matched touched production/CLI files
to the delivered source, hashed result payloads, matched CAS receipts and verified
attempt log lengths/hashes. Earlier regression shards retain the previous new-test
bytes; that file alone changed to compare grouped complete resume with the matching
baseline resume, and its final 29-case shard carries the delivered file.
This byte inspection does not independently attest the worker.

## Original-row metadata preparation

CPU-only action `88ee89c7abfdf79ebb464ad13d7fa8ac8fc88a7cca4fb764966df61cbd14fe11`
exited0 on dl380g10, CPU1/memory4 GiB. It checked the original hash-bound GLM plan,
layer-7 row, Stage A slice and input manifest; validated the 2,048-row grid through
`CheckpointIncoming`; derived incoming membership through the shared owner;
re-sealed a labelled metadata analysis and validated it through the public PB
decoder. It also checked the original capture4/operator-GEMM regime, render-only
window5 profile configuration and the existing observer wrapper's preservation
of the child argv. It launched no workload child or profiler.

| Phase | Original declared bytes / entries | Incoming analysis declared bytes / entries |
|---|---:|---:|
| checkpoint-load | 34,367,170,838 / 2,050 | 3,023,126 / 2 |
| each spill-pP | 8,591,036,928 / 512 | 17,182,073,856 / 1,024 |

Every input entry, phase name and total read byte count was unchanged. These are
**declared metadata counts**, not measured IO, memory savings or elapsed-time
improvements. Analysis wire SHA-256
`cbb1c268d64d4a54ef0e970bad7725f1af3d06c786c3e96440b79f830b18b82d`,
394,559 bytes; result payload SHA-256
`4648c4be575baa0791b79b56782b12a74c5c9f8117dd8768b8d4829d5f61ed46`,
1,375 bytes; receipt SHA-256
`0bb8d68d653b0d8fcfb8c94196a92370ee9a5a041f3a274c0ff4337812202602`.
This is incoming-membership analysis, not a regenerated production row or
qualification of its complete source/invocation closure or protected leases.

## Negative evidence and artifacts

Causal RED `f524d6834008f7a98ff79a27565bf828ea85f7677116ea91de32b787da7c2811`
failed because original cotangents were still declared in checkpoint-load.
Intermediate failed actions remain failed: an invalid fixture mount was refused
by PB's decoder; an incorrectly placed edit was caught by the existing builders;
capture4 refused a fixture's read window2; and a comparison of grouped fresh
capture with single-batch complete resume correctly found different grouping.
Complete resume skips spill when no prices remain, so the valid comparison is
baseline-resume against candidate-resume. The original metadata preflight's
first attempt `dd11ea7732422ccc424b3a1eaa54eea821b3021fc608b59701ef27c17cf4507f`
refused string-valued integers passed as a mapping; the corrected action uses the
existing regime parser on the original canonical string.

The owned evidence root is
`/home/rob/tmp/astra-resume-20261002/pq_spill/checkpoint-1366/`:
`final-causal.json`, `final-targeted.json`, `final-regressions.json`, corresponding
logs, `compile-portable.log`, `original-preflight-fixed.log`, the immutable inline
preflight command/source, 23 typed action records under `receipts/`, and
`checkpoint-source-evidence.json`. The ledger preserves negative actions and
source differences. Those bounded records and the previous prerequisite's frozen
artifacts are retained; no other worker's branch, cache or scratch was changed.

Each test command used the published `pbtest.py --checkout <checkpoint-worktree>
--python /home/rob/venvs/pq-pbdc4803da-tessera-b40c93cb/bin/python --priority -10
--mem-gb 4 --cpus-per-shard 1 --threads-per-shard 1 --timeout-s 300 --json <ledger>`
with the file sets listed above. No full suite or GPU performance claim is implied.
