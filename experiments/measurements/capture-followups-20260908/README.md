# Capture writer and page-release integration — 2026-09-08

This integrates PRs #393, #397 and #399 on the operator-window runtime merged
by #402. Completed anchor and Hessian-sidecar page release no longer performs
a discarded full-file hash pass. The existing descriptor identity check,
durability fence and page advice remain. A tensor-bearing Hessian archive that
produces no stable `data/*` record now refuses publication and removes its
temporary payload, retaining the previously published pair. Successful archive
bytes and content seals are unchanged.

The selected-source CLI test now supplies the existing bounded-capture
environment before invoking the CUDA-aware gate. This is a test correction,
not a relaxation of the production gate. A separate comment correction removes
an unsupported claim that completed anchor files had already been sealed in
memory; no consumer used the removed hash result.

## Verification

The integrated source at `623fc3a4ce027a01cc2f0155f8714a410f1ac19c` passed
**154 CPU tests with four skips**, through PrismaBuild on DL380 with Torch
2.10.0+cpu. PB distributed eight independent focused files, one CPU and 4 GiB
per action, and one original-layout GLM file, one CPU and 16 GiB. Native math
threads were one. The focused files passed 142 tests; the streamed GLM file
passed all 12 tests in 238.40 seconds. Each of the bounded-sidecar,
selected-source, priced-export-input and materialization files reported one
skip. The first two are the explicit native writer profile and CUDA memo gate;
the latter files have producer-dependent checks, whose exact skip reasons were
not retained by this invocation's report options. Skips are not coverage. All
nine requested files collected tests. The touched runtime module also passed
a portable PB compile.

The root independently checked terminal exits, cleanup, canonical CAS receipts,
result bytes and executed Git snapshots. Every integrated snapshot differs from
the reviewed commit only by its PB source-closure file. Small audit records are
retained beside this report. Full commands, logs and test populations are under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/capture-followups-integration-01/`.

The original regression receipts were checked independently as well:

| Fix | Before | After, before integration |
| --- | --- | --- |
| Discarded hash readbacks | Both new readback tests failed; 32 other tests passed | 34 passed on source `045691cf2824cd568376c13e9192d63754035f79` |
| Missing stable writer records | Refusal test did not raise; 9 passed, 1 skipped | 10 passed, 1 skipped on source `d8cae8744533f93467d56984e6c10cbe8ce0bd8d` |
| CUDA fixture environment | Selected-source test failed on the real CUDA gate; 10 passed | 11 passed on CUDA source `54d0b6c8de5de2ff054ab48f5f4a9b4845d06d41` |

The CUDA receipt establishes that test's environment correction on its original
branch. The combined integration was checked on CPU; no new GPU execution is
implied by the combined count. Existing native archive-byte and source gates
remain separate evidence.

Two initial client invocations refused the unsupported `pbtest --pytest-args`
option `-q` before admission. The corrected invocations used the supported
empty option list. An initially queued compile inherited a local host tag; it
was withdrawn from `ready`, releasing zero tokens, and resubmitted portably.
No executed test was repeated for these submission corrections.

This removes a known extra logical read; it makes no latency, physical-I/O,
energy or full-model fit claim. The original full-capture action remains on its
frozen source. The separate verified capture-loader work and per-row source
hashing investigation are not implementations in this integration.
