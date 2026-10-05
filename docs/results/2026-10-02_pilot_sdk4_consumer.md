# SDK4 pilot consumer: CPU source and result qualification

PQ #2106 completes the bounded consumer portion of #1293. The dispatcher now
binds counter bytes to an explicitly selected PB execution's verified CAS
result and sealed request. The capture binder proves the standard wrapper;
PQ inspects the authenticated payload command, original quantum record and
independently reviewed source contract. The contract binds the snapshot
input descriptor and complete selection (schema, commit, parent, subdirectory,
refs and input), plus entry, image and environments.
The same bundle can contain older commits, so descriptor equality alone is
insufficient. Cross-output namespace equivalence and the stamped override
remain covered. The normative contract and finite measurement protocol are
in `docs/design/joint_dispatch_pilot.md`.

## Qualified environment and mode

All tests and compile actions ran through the published PrismaBuild tools,
on dl380g10, in `/home/rob/venvs/pq-pbdc4803da-tessera-b40c93cb/bin/python`.
The environment is Python 3.14.4, CPU Torch 2.11.0, Transformers 5.17.0,
pytest 9.1.1, PB dc4803daaf09b6426083d2d36bd2a2da3d6832fe and unchanged
Tessera b40c93cb73745097e57a1ba4cf5b9eee166c759a. Each shard's preflight
verified both non-editable Git installs, import ownership and RECORD bytes.
The shared connected-fixture source pin resolves one read-only SDK4 archive
with the complete file union; this is source qualification. The deployed
worker generation remains d028dfee920b-1790960385-1d815cff72d1.

CPU tests reserve one CPU and 4 GiB per shard, with one native thread,
priority -10 and CUDA disabled. The finite compile reserves one CPU and 2 GiB.
The original B31 interpreter was retained. Initial attempts failed for missing
pytest and later xxhash; PB provisioning actions installed the scoped tooling.
Those failures remain failed evidence, not qualifying runs. The earlier
unauthorized local smoke in SDK4-CHECKPOINT.json is excluded.

## Causal and selected-result controls

| Report/action prefix | Actual outcome | Interpretation |
| --- | --- | --- |
| `sdk4-consumer-causal-red.json` / `204ba21fd3b6` | 6 failed, 2 passed, 22 deselected | Draft admitted foreign entry/plan/prepared/adjoint/regime; real binder path incorrectly compared wrapper to command |
| `sdk4-source-causal-control.json` / `f896f1c44540` | 1 failed, 7 passed | Removing only the reviewed source-descriptor comparison admits the resealed foreign-source control; pre-selected-commit contract variant |
| `sdk4-commit-causal-control.json` / `363b52ac4d00` | 1 failed, 8 passed | Removing only the selected Git-commit comparison admits an older tree from the same reviewed bundle |
| `sdk4-consumer-green-reviewed.json` | 111 passed, 111 collected/ran, 0 skipped | Pre-full-selection candidate Python bytes: dispatcher 37, counter reference 20, real selected-result controls 9, quantum runtime 38, duplication 7 |

The real controls use private CPU Git bundles, actual sealed requests, CAS
input/result receipts and PoolQueue publish/claim/finish records, read through
the public SDK4 and its actual capture binder. Source, older selected commit,
entry, plan, environment, wrapper, legacy result and wrong attempt each refuse.
The matching control admits. GPU numerical/power/wait measurements and the
fixture source-review relation are controlled doubles; they are not a genuine
joint GPU pilot or approval of a production source contract.

Pre-full-selection green actions, all rc 0:

- `51da58dd0459c40b1fb67f59a7bc8e7048399ae377718504f01440d7bd886839` — dispatcher 37.
- `ff99ee206357070ad3be226d65e184dc159c17cdf1eb24b859928e1a2913a9af` — counter reference 20.
- `84baca214434a3643fc76db8f4da5f382c3742e590b53a283c5587686e251567` — real selected-result controls 9.
- `3f4e98c3e274756a1087b8b88f5663222b9bf0971df23b0784d0675aa24b74b8` — runtime 38.
- `659a64d0417f92ac358207b6e9b08fb88cd39d95ed444d2bb40e48ac46be2585` — duplication 7.

The pin/differential run passed 22 cases with no skips. Earlier caller
regressions passed launch 8, metadata 27 and catalog extension 65 cases; the
quantum runtime's missing-xxhash failures were superseded by the final 38-case
run. Architecture 13 and docs-staleness 6 passed. Duplication initially found
new primitive digest sites; these were replaced with the existing byte-digest
owner, and final duplication 7 passed. Supporting runs retain their own source
identities rather than being relabelled as the final consumer source.

Pre-full-selection five-module compile action
`3bbe2e7c01610aa06de8ea618922b9afcb4d50b5b74cd37180d0403bfc3b0f91`
returned 0, receipt
`c3c14b693bcc4735492e4c88bfd0f7129b2b55ed785ffc9e015ea2e35d86851f`.

## Attributable artifacts and commands

Reports and logs are under `/home/rob/tmp/astra-resume-20261002/pq_spill/`.
`sdk4-receipts/*-action.json` holds complete typed PB action observations;
`sdk4-source-evidence.json` binds them to raw source bundle SHA/length, selected
Git commit, package/file digests, exact terminal status, receipt/result bytes
and attempt-log hashes/lengths. The final admission and compile source matches
the delivered quantum, pilot, staged-lease, dispatcher and experiment Python
files. The final full durable package digest is
`cc961090340184dd58533f4f84c0825720030fba214a637cd6516248305f027f`.
The preceding 111-case candidate package was
`694b30b349cb2cf5e652a2d25453b302fdd5c8fefaa1d0abbcb9d022870d668c`.
The evidence records also retain earlier variants and failed controls.

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbtest.py \
  --checkout /home/rob/tmp/astra-resume-20261002/pq_spill/pilot-worktree \
  --python /home/rob/venvs/pq-pbdc4803da-tessera-b40c93cb/bin/python \
  --priority -10 --mem-gb 4 --cpus-per-shard 1 --threads-per-shard 1 \
  --timeout-s 300 --json sdk4-consumer-green-reviewed.json \
  tests/test_joint_dispatch_pilot.py tests/test_pilot_counter_reference_1293.py \
  tests/test_pilot_verified_result_1293.py tests/test_joint_cost_quantum_runtime.py \
  tests/test_duplication_baseline.py
```

The compile used published `pbrun.py --cpus 1 --demand mem_gb=2 --priority -10
--timeout-s 120 --anywhere`, with the explicit SDK4 interpreter and `-m
py_compile` on the five modules above. The interpreter path supplies the
placement dependency; PB selected the worker. Completed profiles report honest
one-CPU demand and sub-GiB peak residency for the final test actions. Netdata
CPU series are present; missing pqteld coverage is retained as an observation.
No speed, energy, work-per-joule, GPU residency or serving claim follows.

The genuine #1293 pilot and matched #1291 before/after driver-receipt
acceptance remain open gates. Current-source original-model GPU input/lifetime
qualification, exact input/image/resource seals and root resource GO are
required before the finite protocol can be launched. Historical unbound
receipts remain negatives and cannot be retro-stamped.


## Full snapshot-selection follow-up

Root's final contract freezes the complete authenticated
`params.checkout_snapshot`, including schema, commit, parent, subdirectory,
refs and input. It is compared as one exact expected record; PQ adds no Git
validator. This supersedes the descriptor-plus-commit contract variant above.

`sdk4-full-selection-green.json`: 48 collected/ran/passed, zero skips,
dispatcher 37 plus real selected-result controls 11. Actions are
`392901db1ec82eac42d6392e8e260431ef124e5d67b6cb54a803d4fbb5c812b1`
and `2e44e03c61516ced25dcff3cab1635cd1b806918a35414e3db58e4f72c7f185c`.
The fixture then made its private Git branch name explicit and passed all 11
again in `sdk4-selected-fixture-final.json`, action
`cec8bcc6646407c8175c470ce6e9023927f403ca31336bf15eb285926144ea59`.
The matching result admits; wrong source, older commit, subdirectory, refs,
entry, plan, environment, wrapper, legacy result and attempt refuse.

Removing only complete-selection equality gives two failures (older commit
and refs) and nine passes, action
`9f3c9067f7179ee8ea0d764fa591d8f8e4b3dc3eb40ba46f287398888816d4b3`.
The unsupported subdirectory remains refused by the invocation contract.
This is a deliberate failed mutation control, not a qualified execution.

Final five-module compile action
`3a58a106651d166c8c04790bdc5ecd7ec34d7dc1a9641be988a395064c25a2dc`
returned 0, receipt
`2b3e5a1e9454564f8b53d74e97c8f8e39d80c50aafa8fdf2d0e4ca9a184e0403`.
Architecture/docs checks passed 19 cases again in `sdk4-docs-final.json`.
The source ledger retains all 24 action observations and source variants;
current-file comparisons were refreshed after this follow-up. The final
source-package hash and compile/admission byte matches are recorded above.


## Current-main composition and production Gateway smoke

Continuation composes current main `736fd56aa68` without duplicating the existing
SDK4 implementation or producer prerequisite. Merge commit
`63e8ebda5677c7720413db6cd7f1b53d6e656a77` preserves both additive architecture
sections. Source commit `7a7f1bdc31de9b503944186506035da4180eb5ab` adds five
production `Gateway` dry-run cases. The sealed SDK4 root resolver, public result
reader and capture binder are unmodified; only the queue owner's default points
to the private real CAS/queue fixture. Matching evidence emits three receipt
stamps; foreign source, refs, wrapper and attempt refuse without dispatch state
or fleet publication. Producer telemetry and source-review relations remain
controlled CPU doubles, not a genuine GPU pilot. No pins, defaults or runtime
activation changed in this continuation.

Fresh PB results on dl380g10, SDK4 interpreter and d028 runtime as above:

| Action | Outcome | Receipt SHA-256 |
| --- | --- | --- |
| `545407d96c266541ba49f6a8c865307dd915a9ffb01e3bd71b62e9c18611502d` | 5 passed, 0 skipped, 11 deselected | `e4299f2224bdcccd287fce3a46d39df13a42cf6be312231a613c6c60dd99a692` |
| `0e7ced75b3a4dd045e8a87b581f15c4fce17ed29c6b795849ac3e5048204a577` | 13 architecture passed, 0 skipped | `6d484fe840d289e9940b04ed59bc5a35eb2203d7c033509b0008a02658c394a1` |
| `a62b788119dd31d4a2709e47feb384e5a58912157f2e125b96648bb62f3abce2` | 6 docs passed, 0 skipped | `1445a5db16cf1fd1246e7c77210aed3a8f14fd1774dc1b77f9b360e2b49850f0` |
| `354f70d27e216809bd029e6e37d00b7b3d8cf2187cd46705bd5dd0b29a121d4e` | targeted test compile rc 0 | `60908b06d34c3da3eb90fe820478bc2628f2abd5f2201eb40f122c6e2d0bf398` |

All four actions are terminal executed/rc 0. Actual logs and CAS payloads were
read, and each worker local-result claim passed payload hashing and receipt
binding verification. New smoke source snapshot
`cf38f949cef4cf996ec39ef210bbbaed15b59ed1` has parent equal to source commit
`7a7f1bdc31de9b503944186506035da4180eb5ab`; its only added path is PB's generated
`.pbrun-closure.f370bb0f1d4ae2fa.json`. The delivered test and snapshot have the
same Git blob `456aaca287c5bf298f8aea9d7ac4fed20acdc9c2`. Raw source bundle
SHA-256 is `2fefb105d1a9cf3bc1ce1b67a1bb43ab044c136e60aef05416d99cd24764f34f`
(41045499 bytes). Exact requests, terminal records, log hashes, result/receipt
hashes and successful claim checks are retained in
`/home/rob/tmp/astra-resume-20261002/pq_spill/consumer-completion-evidence.json`.
Reports are `consumer-production-gateway.json` and `consumer-composition-docs.json`.

The smoke used the published `pbtest.py` command above with this isolated
continuation checkout, `--pytest-args '["-k", "production_gateway"]'`,
`--timeout-s 300 --wait-s 600`. Docs used the same admitted CPU envelope on
`tests/test_architecture_doc.py tests/test_docs_staleness.py`. Targeted compile
used published pbrun with one CPU, 2 GiB, native threads 1 and 120 seconds, on
`tests/test_pilot_verified_result_1293.py`. An initial submission with non-JSON
pytest arguments was rejected before publication and is not test evidence.

Earlier complete qualification is reused under its original source identities,
not rerun or relabelled. Root critical review/merge remains required. The genuine
#1293 pilot and matched #1291 driver acceptance, original CUDA source/lifetime
qualification, fleet activation and all energy/work-per-joule claims remain
independent unopened gates. PR #2058's producer commits are already included in
#2110; no second producer implementation or duplicate PR was created.

## Pilot gateway collection isolation

The production gateway test owns the authenticated source graph through the
existing `pinned_pb_source` fixture, matching the SDK4 pin tests. This fixture
detaches preimported PrismaBuild modules and sibling fleet tools for the test,
then restores their identities and path order. It does not weaken the production
resolver or replace the result reader, capture binder or gateway.

`tests/test_original_cuda_control_artifacts.py` imports `PrismaBuildCAS` from
the installed distribution at module collection. Module and function import
restore fixtures start after collection and therefore preserve this installed
graph as their entry state. Adding the sealed source to `sys.path` in
`require_paths()` cannot replace a module already in `sys.modules`; the gateway
test previously failed there before entering `reader_sdk_bound()`. The reader
binding names a helper root, not a fresh canonical import graph.

The minimal pair fails even with the gateway file listed first and all artifact
tests deselected: action
`f7d476fa074ac6b24b4db4b54208b39da81b3f1227f3c14d333f5f663b3a854c`
records five failures and 28 deselections on dl380g10. This establishes
collection-time contamination, not a requirement that an artifact test execute
first or share a particular work-stealing schedule. Production code is unchanged.

Before the fix, source-guided bisection reduced the 69 companion files to
35, then to the single artifact test file. The complementary 34 files were
collected with only the five gateway cases selected and did not reproduce it.
The minimal pair runs all 33 cases in one worker, with the artifact file first.

| Control | PrismaBuild action | Actual outcome |
| --- | --- | --- |
| Main-equivalent original 70-file grouping, four workers | `2087a67cbf4c2c143c891360111feb7eda1e68f44751860ee1da75d29ea98e7a` | 5 failed, 971 passed, 30 skipped |
| First 35 companion files and gateway file, one worker | `3cfaacb7c1a87f42d2214e92af810a9a1674b7a7e44eab7c18628c6249e32a8e` | 5 failed, 703 passed, 18 skipped |
| Complementary 34 companion files and gateway file, one worker | `a337c9a2db3885e691d6a53e0871ad41ff80b657d9f5885fa31af0694da414e5` | 5 passed, 291 deselected |
| Minimal pair before the fix, one worker | `27c3c4eb81f1ae7ce3c1ac3627f64e0d17d3b475dbee3d6ce1264e23199d6eb4` | 5 failed, 28 passed, 0 skipped |

These are CPU-only actions on dl380g10, with one native thread per worker.
The original 70-file grouping uses four CPUs and 8 GiB; serial controls use
one CPU and 8 GiB. The unchanged interpreter argument is
`/home/rob/venvs/pq-pbdc4803-tessera-b40c93cb/bin/python`. The main-equivalent
grouping and the earlier passing file-only control are reused from the supplied
receipts, not claimed as new runs of this branch.

Post-fix source was submitted for the same minimal pair and the exact original
70-file grouping, respectively:

- `fdc600eed049d4d96250bf53347e3f0895f92e73169152d49bb4329bf1beab67`
- `92f457bc9fe3a05e1e9d9612f7f1be62b3c2b2fb04f566cbc4440eb3d427f60a`

Both sealed source bundles contain the fixed test bytes with SHA-256
`76d2faebea96d054ff6ed344a4bd4d63b066ba9da46568a117c6fdea94ec392d`.
No post-fix terminal outcome or receipt was observed at handoff. The pair was
still ready with zero attempts at the 03:06Z diagnostic; completion clients
remain attached. This is a proven collection cause and a submitted fixture
fix, not a claim that the pending fix proofs passed. No branch was pushed.
No complete repository suite, graphics processor work, real model validation,
live deployment or performance measurement was performed for this fix.

## Completed pilot gateway isolation proof

The pending statements above describe the initial handoff. The actions then
finished on dl380g10 without withdrawal or resubmission. Original full-group
coordinator clients reached their tool deadlines while the pool kept running;
a published `pbwait.py` completion client recovered both worker endings.
Coordinator timeouts are not treated as worker test failures.

| Control | PrismaBuild action | Actual outcome |
| --- | --- | --- |
| Minimal pair after the fix, one worker | `fdc600eed049d4d96250bf53347e3f0895f92e73169152d49bb4329bf1beab67` | 33 passed, 0 skipped |
| Unfixed original 70-file grouping, four workers | `899348347907b7ff7ae108781859751577591b859de38675473609094362f04f` | 5 failed, 971 passed, 30 skipped |
| Fixed original 70-file grouping, four workers | `92f457bc9fe3a05e1e9d9612f7f1be62b3c2b2fb04f566cbc4440eb3d427f60a` | 976 passed, 30 skipped |

Both full-group source snapshots have parent
`62fd01ad964d55a23a2fdd8ae891c78dd9a691b6`, use the exact original ordered
70-file selection and collect 1006 tests with no ignored files. The five
gateway cases are the only failures before the fix and all five pass afterward.
Both full-group runs use four CPUs, work stealing, 8 GiB and one native thread
per worker, with graphics processor visibility disabled.

Successful receipt SHA-256 values are
`16212d3a2165f4fd28ef92433532c8be4d0c3b484a0c072fb2e2c9e6f5b4e6ef`
for the pair and
`67958f1502db27f95bb1c0699fc4704e229bd2f2cb73980c985f04daee9d9492`
for the full grouping. Their durable result payloads were read and their
hashes and byte lengths verified. The failed full-group worker log was also
read and verified against its recorded hash and byte length; no successful
receipt is claimed for that failure.

Complete action, source, receipt and skip-reason records are retained in
`/home/rob/tmp/pq-pbimport-triage-evidence.json`. The 30 skips include graphics
processor and Triton paths, an absent DeepSeek source configuration, optional
real-model validation, the installed Gemma4 shared-key/value limitation, and
the filesystem direct-input/output prerequisite. This proof establishes test
isolation only, not a complete repository suite, model or graphics processor
qualification, live deployment, throughput or energy.
