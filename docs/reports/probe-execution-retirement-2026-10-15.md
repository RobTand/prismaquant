# Probe execution retirement and Fisher composition checks

Provenance: 2026-10-14 PrismaQuant architecture refactor, base 138069b7993.
This unit starts at integration 9a80849241711e78e7968e992cd06ab081b854c8.

The unreachable sample-parallel execution model is archived. Its five manual
commands are also explicitly retired. The recorded artifact readers remain live.
The ordinary body, MTP and visual algorithms retain their separate objectives,
tapes and source lifetimes. A generic sweep with schedule callbacks was rejected
because it would conceal these differences behind a larger argument list.

## Ownership and caller migration

`archive/sample_parallel_probe_2026-10-15/` contains the complete old producer,
merger and priority producer contract, complete incremental and sensitivity
source snapshots, their two old suites and operator runbook. These are historical
files, not a supported import path. The old resident `run_probe_pass` had zero
calls in the live package, tools, scripts and tests. Five offline producer commands
were real manual entry points, not proven unused commands: prepare-calibration,
prepare-worker-source-cache, merge-importance, build-cover and merge.

The AST inventory found 34 imports in seven files. All live imports of the three
archived modules are gone. `sample_parallel_artifacts` now contains the dependency
closure needed for source-census, dense-stat and activation-priority admission.
Its 22 moved function/class declarations retain identical ASTs apart from the
canonical helper's unique owner name. Schema strings, digest profiles, error
messages, finite/shape/token checks and marginal tolerances are unchanged. It
cannot prepare a worker, publish a receipt or merge workers. Existing incremental
shard, pooled-row and precompute admission checks continue using these recipes.
ActivationIndex retains its independent same-byte/stamp checks.

The CLI still refuses global sample calibration before model reads. Its dead
deep worker branch and ordinary-path sample arguments are removed. The three
ordinary scientific operation expression multisets are identical for importance
weighting off and on: precompute has 21/25 operations, body 46/46 and MTP 4/6.
This source proof supplements the actual numerical checks below.

The test disposition ledger records all 59 historical test functions. Live refusal,
priority vectors, pooled-cache disagreement, head scored-token rows and full
precompute payload checks moved into current behavioral tests. Full-schema
ID/routed-map mutations replace old partial-payload checks that refused before
reaching their named condition. The ordinary MTP collector still emits its legacy
payload under either marginal environment value. Retired-only publication, merge,
source-cache preparation and sample-MTP cases remain in the archive.

The source-only MTP dispatch signature check and retired runbook shell-string
checks are removed with their execution. A new actual body fixture calls both
production precompute and shard runners on a saved two-layer Qwen3 checkpoint.
It compares body Fisher scalars/marginals against end-to-end autograd, and head
statistics against the explicit per-token gradient formula, for both weighting
modes. It uses the real source cache and prefetch/install/unload interface on CPU.
The KV source-wiring safety assertion remains because this Qwen3 fixture does not
prove borrowed Gemma K/V custody. Existing shared-state behavioral tests remain.
This unit does not claim a full MTP/visual numerical owner migration.

## Admitted verification

All execution used PB from immutable checkouts. CPU shards reserve one core and
six GiB with native threads one. Compile reserves one core and two GiB. Portable
pbtest submissions carry no manually requested host tag. The actual strict
interpreter is `/home/rob/venvs/pq-pin-d2e98b3c-pb20261015-git/bin/python`;
PB verifies exact Tessera d2e98b3 and PB 027103d9 RECORD/Git provenance.

The pre-edit baseline `d841ae38fc7a` used the separately scoped wheel-byte
environment, not the later strict environment. It returned 792 passes, eight
skips and 20 failures. All failures were a stale file-digest table reference to
deleted `lane_eligibility._sha256`; its repair is a separate commit. The first
baseline submission named a nonexistent former test file and ran zero cases.
Both results are retained.

The structural result has 684 distinct passing cases and seven documented skips
across 19 files. Final changed cohorts include `c206eac50923` and the final
thread-bounded body rerun; `0d40c8154331`, `c7d206bdbbc5`, `15c31d879308` and
`d4cef35c9532` supply the other latest per-test outcomes. The two old failures in
the latter tensor cohort were corrected by moving only portable GoldenTable
labels to the public tensor owner, retaining every historical byte/error outcome.
The frozen original random-input table remains unchanged. Eleven changed Python
files compile under PB `64bdb8017816`.

Six skips require CUDA for generic tensor-digest tables. One Gemma4 order case
cannot run because the installed Transformers marks layer 6 as shared but exposes
no `kv_shared_layer_index`. The archived producer-only CUDA priority test accounts
for the eighth baseline skip. The actual GPU body comparison below is separate
coverage and does not erase these skips or qualify a Gemma/MTP/visual model.

The new oracle passed unchanged pre-edit source in `470f0b10cb67` and post-edit
source in `523bafe68155`. Initial fixture attempts used an empty exclusion regex,
then incorrectly expected the general Fisher collector to measure `lm_head`,
which it intentionally skips. These failures are retained. A source-edit helper
initially treated U+2028 inside a frozen string as a physical line boundary;
compilation caught the corruption. The helper now counts LF exactly, the original
fixture bytes were restored, and the compile/byte checks passed. The duplication
inventory has no new semantic helper group, no near-duplicate pair, and primitive
digest sites decrease from 293 to 290. Its first admitted generator exited via
the CLI before copying its output; the corrected generator and its artifact are
recorded separately.

## Paired real GPU characterization

PB `2d6cd5385dab23a05dbbc97bdf3f5b561c1f7d8e63687ad949ebbde1f27b30d1`
finished on Sparklina with exit zero, settled cleanup and all four child exits
zero. Its independently read/hashed CAS payload is
`42bcd2525773becb7322178c7458c8254abef0b3f973a241f91266a718cd46d9`.
The before/after/after/before experiment uses the same existing Qwen3-0.6B source,
explicit 2x16 token draw, BF16 eager attention, max two cache slots, one prefetch
worker and one layer-read/activation-write/native thread. Both weighting modes
run full precompute plus the top body layer and terminal head. Fisher probing
uses original weights through the existing source cache; it does not render
weights or introduce a second residency mechanism.

Each arm produces eight Fisher-stat rows per weighting mode, 29 boundary tensors,
the tail cotangent, eight activation records and eight H-detail records. All
scientific values, calibration/source bytes, read multiplicities, installation
flags and unload coordinates match exactly. Each arm records 60 source reads,
58 installs, 58 unloads, 4,436 CUDA kernels and a 2,654,889,472-byte peak CUDA
allocation. PB affinity `[5, 6]` is retained in every child. Measurement admission
reserves two cores, 16 GiB host memory and a six-GiB GPU subset, exclusively, using
class-qualified portable placement and the exact existing image.

| Arm | Profiler self CPU | Profiler self CUDA |
|---|---:|---:|
| Before 0 | 3.365 s | 225.193 ms |
| After 1 | 3.248 s | 224.623 ms |
| After 2 | 3.250 s | 226.184 ms |
| Before 3 | 3.151 s | 224.170 ms |

Each arm exports torch.profiler and cProfile artifacts. Netdata records 35 samples
from each Spark box, without collection errors. Sparky power remains four watts;
Sparklina samples span four to 12 watts against the roughly 140-watt envelope.
These short diagnostic arms establish numerical and lifecycle parity. They make
no throughput, production bottleneck, saturation or energy-ranking claim.

Image identity is
`sha256:44426342348a2fb1ae1191e09e5d4e19ab1231ab6dbdf0e17e8192bfdbeff025`,
content `ac918444aa2d16d1d19771510963d138981fdd468db44a2bf10f9d627b3af12a`.
Torch is 2.13.0+cu130/CUDA 13.0. The immutable capsule includes the reviewed d2e
metadata wheel package, SHA-256
`30748bc3df5380475fe6dfbf3b49aa95d4b73467fe7ed8680938fe940382aa39`,
in a scoped import directory. It imports no serving runtime and modifies no
Tessera checkout. No format or serving admission is inferred from this fixture.

Raw evidence is under
`/mnt/shared/pq-refactor-20261014-native/p4-retirement-gpu-pair-20261015/`.
The path/size/SHA manifest binds all traces, pstats, scientific artifacts, logs and
both-box telemetry. PB `aa4694d07c36` publishes it with CAS
`873dfc6ebfefbd1b551b139028808126541f027b6ba5a44d0e906e5b10677d09`.
Local recipes, exact commands, caller/source inventories, failed attempts,
terminal records and independently checked receipts are retained under
`/home/rob/tmp/pq-refactor-20261014/reports/p4-*`.

## Findings kept separate

The stale Tessera digest-test caller is repaired separately. An ordinary shard
reuse gap was also found: absent sample fields were omitted from expected metadata,
allowing a recorded sample-stamped shard with otherwise matching fields to enter
ordinary reuse. Its reproduction and fail-closed fix follow this structural unit
in a separate commit. Neither finding changes the numerical refactor above.

### Separate digest-test caller repair

The stale `lane_eligibility._sha256` test row now calls its actual shared owner,
`digests.file_sha256hex`, with the same original default block size and Path
argument. It keeps the frozen byte/error outcomes and call ordinals. No dead
helper or compatibility export was restored. PB
`7737ce196413b85c37a56bccbd379eb52ab40fe43450a63fc80937f9c259c999`
passes all 45 file/byte digest cases with no skips. Its terminal and CAS payload
were independently checked. Combined latest coverage is now 729 passes and seven
skips across 20 files; the failed pre-edit baseline remains recorded above.

### Separate ordinary-shard resume repair

The P1 metadata-admission finding is fixed separately from the refactor. A real
ordinary shard is first proved reusable, then each of the five recorded sample
fields is stamped at top level or under `incremental_shard`. All ten cases failed
before the fix because the public reader accepted the stamp when expected metadata
omitted it. Pooled-row reuse already refused that disagreement.

`sample_parallel_artifacts.SAMPLE_PARALLEL_META_FIELDS` now supplies the existing
field set to both readers. Individual-shard reuse requires each recorded value to
equal the request, including absence. Probe writers and artifact bytes do not
change. This check runs while the CLI examines existing pickles before creating
its streaming context; it adds no probe, kernel, cache-fill or per-layer work.

The admitted RED check returned ten failures and 15 passes. GREEN PB
`6ad183dbfbf3e6f4510d0beba26b36b02c557d3a40745e94ef2e2f8d9907ffe9`
and `42dc97aa7d33d2aacc1d64a9e61392b2621ab41304e8334c2b20bc6559e85a90`
returned 90 passes and no skips across seven affected files. Three files compile
under PB `e3f3b9aa25c1`. All terminal exits, settled cleanup and CAS payloads were
checked. Combined latest coverage is 739 passes and seven skips across 20 files.

### Separate source cleanup

A final cosmetic commit removes specialization parentheses/empty signature lines,
wraps identical CLI help literals and corrects comments naming retired owners.
Both changed Python modules have identical complete executable ASTs after excluding
only docstrings. No additional tests were dispatched for this comment/format-only
unit; the source invariant is retained in `reports/p4-cosmetic-ast-invariants.json`.

### Historical activation-reader coverage restored after independent review

Two archived cases exercised the retained ActivationIndex consumer rather than
the retired producer. Their original test names now live in
tests/test_retained_activation_reader.py. A fixed plain tensor/identity fixture
replaces the former merge-bundle producer setup. The cases retain post-validation
replacement refusal and pinned-directory prefetch custody across pathname
replacement, including a later corrupted tensor. The disposition ledger records
these two corrections. No producer, merger or manual CLI is restored, and no
production implementation or admission policy changes in this test-only unit.
