# Original copy/identity CPU preparation — 2026-10-02

Refs PQ #2119/#2010/#2008, draft PR #2121. This record is CPU preparation
only. Original CUDA and automatic-capture predicates remain closed. All 32
real-CUDA harness nodes are unrun and explicitly skipped in this CPU action.
No original GLM bulk payload was read, no GPU action was submitted, and no
primal/g6, complete-provider, performance, price or serving acceptance follows.

The dormant existing layer/head/direct-scale paths retain native CPU source
aliases and conversion/stack staging through their actual current-stream event.
CPU controls use the real source owner, sealed memfd decoder and native Torch
StorageWeakRef accounting with explicitly identified CPU CUDA spies. Failed
event proofs retain actual owners and credit; successful copies precede head
installation and cancelled readers drain before refusing output. The earlier
CPU regression failure is retained in `original-cuda-lifetime-red-r2.json`;
20 lifecycle controls and the neighboring integration passed (152 outcomes)
before the final combined integration below. Test harness setup failures and
the inadequate arithmetic spy's failure are retained as harness failures,
not reported as production GPU defects.

The existing model/source-checkpoint identity serializers now consume the
explicit original owner's independently bound expected descriptors and owned
config/complete-index facts. Exact resolved config/live/checkpoint/shard maps
must agree. Legacy source identity/digest cache inputs refuse and Stage A does
not seed or consult them. Actual decoder/readset receipts are separate from
expected whole-file identity. The initial 13 identity controls all failed before
this extension (`original-identity-red.json`), then passed. The Stage A cache
controls reproduced six failures before wiring (`original-stage-a-identity-red.json`),
with one extra teardown error from an error-frame owner; the corrected wiring
passed all 21 outcomes. Same-path/same-signature pool mutation, forged owner,
missing auxiliary proof, incomplete scopes and config/index/roster divergence
are controls. The runner-free original identity equals the unchanged legacy
source-checkpoint identity for the same fixture bytes, preserving v1 semantics.

The final published PB integration used Python 3.14.4, Torch 2.11.0+cpu,
Transformers 5.16.1, PB 95a59051d48cda82eea7927f31870c6c862d7174 and Tessera
b40c93cb73745097e57a1ba4cf5b9eee166c759a on dl380g10. It collected 318 outcomes:
286 passed, 32 explicit CUDA skips, zero missing and eight green shards.
Native threads were bounded to one; each shard reserved four CPUs and 8 GiB
aggregate memory. PB owned partitioning/placement; no GPU demand was declared.
Compile checks passed separately with one CPU and 2 GiB.

Exact commands and complete reconciliation are retained in
`/home/rob/tmp/astra-resume-20261002/quality-next/original-cuda-identity-final.json`.
The command used published `tools/pbtest.py`, `--shards 8 --workers-per-shard 4
--threads-per-shard 1 --cpus-per-shard 4 --mem-gb 8 --timeout-s 480
--test-timeout-s 120 --wait-s 1200`, and the exact interpreter
`/home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python`. The report names all
19 test files and every skip reason. Compilation used published `pbrun.py`
and `python -m py_compile` for the four touched production modules and five
touched/new test modules (action 41fd02a9b49c9a1a757f5aa7251ba4a2b36d8f0bbdff99e828f3ca7549f9c275).

All nine final actions have terminal exit 0, present CAS receipts, complete
process telemetry, zero live processes and zero OOM events. The sanitized
terminal/resource evidence is retained at
`/home/rob/tmp/astra-resume-20261002/quality-next/ORIGINAL-CUDA-IDENTITY-CPU-RECEIPTS.json`.
CPU shard peak memory was 2,111,315,968–2,742,341,632 bytes, within reservation.
Host profiles contain Netdata CPU series and explicitly report no pqteld CSV
for dl380g10. That missing series is preserved; these CPU resource profiles
provide no GPU or performance claim.

Every tested checkout bundle was authenticated by its declared SHA256/length
and imported for exact Git comparison. All nine contain source identical to
candidate 83745030d92943d5d41db775980da2f52e7ca976; only PB's delivery closure
files differ. The bounded audit is
`/home/rob/tmp/astra-resume-20261002/quality-next/ORIGINAL-CUDA-IDENTITY-SOURCE-COMPARISON.json`.
This results document is an additional prose record after that code freeze.
Refactor, dormant lifetime and identity changes are separate commits, with
normative architecture/contracts updated alongside behavior. Remaining actual
CUDA lifetime, original a6 source/readset/envelope and fresh g6 acceptance remain
open in the parent task.
