# Independent review of PR #413

Reviewed head: `d163aa8ac1eb54587438aad8759336269a1104dd`.
Integration base: `57543bd59937ccfb41679c065db0920f5235c755`.
Tested integration: `2f7fdea9b56c573878a11984fec801b9ed3d8b96`.

**No blocking findings for the completed-anchor page-release regression.**
The exact PR diff is `reviewed-pr.patch`: one 294-line test and an architecture
stamp. The isolated integration preserves the current source-authentication and
verified-capture-load stamps. No production Python or tool changes were needed.
The author branch was not edited, and this review performs no delivery merge.

The test actually runs `campaign.main` for census, capture and the selected
source row. Its tiny frozen token draw avoids downloading calibration data;
after capture, `_collect_activations` is replaced with a failure sentinel and
`source_forward_count == 0` is asserted. The capture must be complete. The
real planner consumes a packed probe and writes the sampled `s:` selection,
which the selected CLI consumes through the current source authenticator.

The menu wrapper calls the real menu expansion before restricting this small
correctness fixture to two shared dynamic rungs. Encoder, decoder, scoring,
wire publication, render publication and release operations are unchanged.
The release wrapper calls the real helper and compares each expected stat
against device/inode/size/mtime/ctime from the actual file. Memo/measurement
wrappers call through and only record diagnostics.

The selected output has exactly the two packed parameter cost rows; its expert
wire mapping and observed measurement set equal the sampled-member/rung set.
All 24 expected rendered/wire files exist and each has exactly one recorded
release. The source selection and checkpoint stack identity are checked. This
proves the expected artifact/release roster; it does not claim wire/render byte
parity, forbid every possible unrelated file on disk, or prove physical eviction
from a best-effort page-advice syscall.

## Integrated GPU result

PB action `1931f45f9fc1451ee64c92812fcf70ed8262b2ffa7b1ea1951c5fec49b30075c`
ran once on Sparklina CUDA in the content-qualified GLM producer container
(`eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd`).
It passed: **1 passed, 0 skipped, 14 Torch deprecation warnings**. Pytest reported
35.41 seconds; PB reported 38.50 seconds. These are execution observations, not
an optimization comparison. The test was unchanged from the author head.

PB selected the eligible GB10. Reservation: three CPUs (main, one source reader,
one prefetcher), one thread per native library, 12 GiB physical guard budget,
4 GiB GPU subset, priority -10, 1,200-second deadline and one attempt. Scope
memory peaked at 2,908,307,456 B. The frozen original Sparky capture was untouched;
shared inputs were mounted read-only. No native/full-model measurement was run.

The logs record six sampled members, `TESSERA_BF16_K1_R256` and
`TESSERA_BF16_K1_R257`, 12 anchors and 24 anchor files. The 31 total release calls
also include the Hessian writer's `hessian_capture.pt.tmp` and
`hessian_capture.pt`. Memo diagnostics were hits=6, misses=6, maxsize=1 and a
unit-major order. Batch-size-above-one regrouping and the reorder half of #389
remain outside this regression.

`integrated-invocation.json` is the exact submission argv.
`integrated.stdout.log` is the actual CAS result payload.
`integrated-receipt.json` records source, terminal, CAS, resource and cleanup
checks. The sealed source bundle's bytes/hash were verified and fetched: its
only tree addition over the tested integration is PB's closure metadata.
The local claim digest, receipt digest, payload bytes/hash and both attempt-log
bytes/hashes passed independently. This audit does not independently repeat
PB's full-manifest worker-attestation verifier. Cleanup is complete, the scope
was released, live processes are zero, and both OOM counters are zero.

## Author evidence audit

All ten cited action records, source bundles and attempt logs were read and
hash-checked. Seven successful actions have self-consistent success receipts
and matching CAS payload bytes/hashes. The mutated action and two whole-module
skip actions correctly have no success CAS receipt. Exact keys, paths, hashes,
exit statuses and cleanup records are in `author-receipt-audit.json`.

- `12cce758`: original x86 test passes, one test, 56 warnings.
- `5fbc49d9`: original x86 mutation fails at test line 286; an expected anchor
  wire has zero release calls instead of one. The only production-code delta
  is `if False and ...` around `_finish_anchor`'s release branch.
- `4a833667` / `2b870378`: driver integration passes, five tests and one skip.
- `ab39e299` / `c138637b`: architecture checks pass, 13 tests each.
- `6a555f47` / `87681a3e`: staleness checks pass, six tests each.
- `695f07fe` / `6cc26a1b`: Sparklina venv collection skips the module because
  `transformers.models.glm5_next` is absent, pytest exit 5. These are not passes.

The original green snapshot's repository files match the reviewed head; its
additional PB closure metadata means the complete Git trees are not literally
identical. The mutated snapshot differs by that one production line, the older
architecture stamp placement, and its own closure metadata. The test blob is
identical in the author green, author mutation and integrated run:
`63cc445ba3d6414d252ad73001b8998210a23290`.
All ten author attempts have completed cleanup, released scopes, zero live
processes and zero OOM counters.

## Coordinator verification and integration

The coordinator independently checked the integrated native CAS payload and
source bundle, the author's original positive CPU result, and the mutated
negative. The mutation changes exactly the release condition to `if False and`
and preserves the test bytes; its failure reports zero releases for expected
wires, against one required release. Those three root audits are copied here.
The integration changes only the test's introductory prose to limit its claim
to page-advice calls and file identity. Executable test behavior is unchanged.
The dispatcher partition and failed-fit publication changes have their own CPU
regressions; the native receipt predates those planner changes.
