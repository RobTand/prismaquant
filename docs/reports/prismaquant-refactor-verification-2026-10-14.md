# Refactor integration checks

Base: `138069b7993`. Branch: `sol/refactor-simplify-20261014`.

The first composed check, PB
`2b4e6d708fc24bed2441ce093a4944e4dc4207cfef00b0501504aa3516872d36`,
passed 590 cases and failed one duplication guard, with no skips. The ledger
had copied the producer's digest validator while leaving the producer copy.
The ledger now owns `require_quantum_digest`; the producer uses it. The exact
string rule, lowercase digest predicate and refusal text remain the same.
The shared digest owner supplies the text predicate.

PB `1d77dee590e483c34e86ae5ca588bcd4231b6e452508e374899ea4b1ceeab24e`
passed all 174 focused producer, ledger, dispatcher, duplication and migration
checks with no skips. Its exit was zero and its result CAS digest is
`2fbdfbdbe8254a61bb8f1756bea6a23979a31bf932639731c89e107fcab241a6`.

The prior submission `f8d059b780094051e275920246d53dbb607520190a3db194e97dc663334becd8`
ran no tests because the command named a nonexistent test file. It failed;
the corrected composed command above records the actual coverage.

Additional verified units, 2026-10-14 23:50 UTC:

| Unit | PB action | Outcome | Result CAS SHA-256 |
| --- | --- | --- | --- |
| Probe inventory and declaration refusals | `9e563546b2c8d0b9dc0dfc813929aae1902572b470373aec79c6204d209c489e` | 173 passed, no skips; three modules compiled | `b013dac66f968e58601a12f5ae338a63fb2ada56708b0e71337e00cebf8dd969` |
| Shared device-placement composition | `72a7b128d8bfb0f1767ddc48719fa047c9cb87dbe5c44e44c12b9d67733b1a37` | 126 passed, no skips; three modules compiled | `e6884cc8cf6d8c528fee745ced87d0fa0b8287be817679f635f02150c52af1ae` |
| Off-step memory obligation | `10911b7ba97a432f155469da26d7ae3eb3737f035227e990ce2ff3d8a3dfe248` | 129 passed, no skips; three modules compiled | `153d21f80886dbc5c43b5ddeaf99593bd2d367ec6f60e4eed0fc74002c9284be` |
| Spool alias removal and snapshot tools | `f639aeeb10ee4b315492fa7aa96d1e066455b5d6749604d671966819178ca3ee` | 22 passed, no skips; three tools compiled | `29ce3c835ab0534bec79387ec10a366281730e83734f0d547d37934b7d5fa7a6` |
| Seven unused prototypes retired | `48153da90d271e164340323efa94b93be22e3400c37167b83fa27950d52b0f35` | 133 passed, no skips; retirement tool compiled | `c27d121eca98f3c8d2c1448f80a2851a88fe9ab8ae9bf289389d098255e6e51e` |
| Corpus sampler ownership | `d5cd7e4126e981ce8d647e5dbe5c323636c02321b62853305fa3d2c89a51b2a1` | 44 passed, no skips; four modules compiled; sampler AST identical | `7ff633617670867258d4471d5c0d820a2b1fde0339952ac78d562a65ea24d259` |

These are CPU behavior and ownership checks, with exit 0 and terminal/CAS
records inspected. They establish no GPU speed, served quality or new lane
admission. All used the qualified PB interpreter
`/home/rob/venvs/pq-pin-fca4c6ce0-pb027103d9/bin/python`, with OMP, MKL and
OpenBLAS threads limited to one. Four-way checks reserved four CPUs and 16 GiB;
the spool check reserved two CPUs and 8 GiB.

The admitted command is `tools/verify_refactor_unit.py --compile ... --tests ...
--workers N`, with the unit's source capsule supplied as `pbrun --cwd`. Corpus
ownership also ran `--proof tools/refactor_calibration_owner.py --base f80ae9a186`.
Exact submissions are retained at
`/home/rob/tmp/pq-refactor-20261014/reports/*-submit.json`; logs and terminal
records are under `/mnt/shared/prismabuild-fleet/pb-queue/{done,failed}/ACTION.json`.
The CAS receipts are `actions/v3/XX/ACTION.json`, where XX is the first two
characters of the action key. Source capture uses
`tools/freeze_refactor_checkout.py --cwd WORKTREE --output NEW_CAPSULE`.

Expected-failure records retain the before behavior: four inventory refusals
failed at `fe9dc7e5dfe5a0a8966d9bf7b1cbec681e8ef6bb0add8afb396b6af5ea0fd245`;
one off-step memory case failed at
`0de80d9108190e6261819a912788b360c797ab564b8ada98f918f1204ba839dd`.
A first inventory correction rejected an optional undeclared packed-profile
method; the final correction distinguishes absent methods from declared
methods that fail. The original inventory move proved five unchanged bodies
and passed 163 behavior cases, with three GPU-only streaming skips, while the
composed duplication guard still reported unrelated journal-name collisions.
The broader composed check `8209cd0e9589e0693922c06521a4abddc5cf8463207fb46ca742861b959a91a2`
passed 225 cases and 59 subtests but failed duplication guards. Those failures
remain explicit until the journal corrections and final composed check land.

Off-task fixes encountered during these moves: the existing pin was omitted
from wheel data; mutable metadata views could alter later admission facts;
source-snapshot capture could race ongoing edits. The pin packaging, owned
metadata views and immutable verification capsules address those findings.

Additional verified units, 2026-10-15:

| Unit | PB action | Outcome | Result CAS SHA-256 |
| --- | --- | --- | --- |
| Relocated publication allowance and dead streaming imports | `db2ebbe46bbf49c9a454d4d25ab7c02888eb5ab94a62e11d31a1009e65257e0f` | 27 passed, no skips | `ae7592a5707ee80b41b54c934c91d3a72b25c6552432594a7f7514c2acac6f73` |
| Typed calibration request and draw | `6af3ecf1c2c0f02ffa7f0bb002a5c81673250a98a00f312e416efe2d30406795` | 16 passed, no skips; two modules compiled | recorded terminal receipt |
| Checked calibration prefix and captured lazy fit samples | `615fd7e68ce4864fd9f9a5af882c3df1207c614e124a2172ca75becddc316346` | 32 passed, no skips; two modules compiled | `b330d60c465775ff8e62111ab72a4dab75105fa33c3885062af2b5f80cc23c63` |
| Explicit seed zero and isolated sampling RNG | `6de56ed06f534062958565eb9e99761422390d8b391fa0cb6f7a122ab7ca4ea2` | 34 passed, no skips; two modules compiled | `0a04143b030b9aca3b62d44baca47f3e71636e3c18393df14d46c78a5ad67699` |
| Composed quantum phase, codec and calibration changes | `3c9a75ea793d501fa2abae373ce232d869cc2222f87ce51cae94d5f7557ddebc` | 447 passed, 59 subtests passed, three skips; package and tools compiled | `8aaf348723610508de898c6786fd919d0df5ec9004cf88dc2e296b1bf1fa18a0` |

The composed check's skips were two Stage B spill cases requiring
administrator-provisioned private scratch in `PQ_STAGE_B_SPILL_TEST_ROOT`,
whose qualification remains incomplete, and one inline packed-expert render
requiring a GPU. These CPU checks grant no new spill or serving qualification.

The original shuffled-corpus prefix assertion failed in PB
`a14d7a7f53e645a9166952313793191a28821e69cc40f1d6c8fb96030f8ad8f4`.
Two actual calibration regressions, explicit zero seed and preservation of
caller RNG state, failed in
`0100fdf485536e91fccf6028e1b7c2588f39e1c5942ec30ec12583c21bda0c1d`.
Their corrected units above passed. The old holdout test's assumed loader was
replaced by actual corpus sampling; equality of legacy draw hashes is not a
proof of corpus-position disjointness across arbitrary draw dimensions.

The native codec extraction passed 436 CPU cases and 59 subtests, with four
explicit CUDA/Triton skips, at
`5942bde1e21b71a91f45e5d94f23dbb92a2ad913c51ab9b366dc7f7afe36d657`.
Its real CUDA before and after runs were
`a5850a70c36403926d5245d02f52cb65ca2ae3834466b2ba55bf09dd339312ef`
and `26908018f5156e947f50bc9f73c1d00dd386ef712f2af53783060e7f56e2ee4d`.
PB `d50fa0d877aab166d77785dfc3a87bfeadd6f039a528c4be02c805f894bd0426`
verified exact equality of 14 rendered tensors, 14 native wire packs, scores,
gates, calibration, assignment, render levers and prefetch results. The
comparison, profiler trace hashes and both-box Netdata hashes are recorded at
`/mnt/shared/pq-refactor-20261014-native/structure-comparison.json`.
There is no speed claim. These receipts do not qualify a new format or engine.

Two commands named nonexistent test files and failed without valid coverage:
`af449b26e0ffc5f9523984a7b60a4d750363f381ebf34bec77c87a6136561e20`
and `f78d42bedc91d2717ac295966b986990ecf416eaf744d9c33b438ec9f201f7ef`.
An absolute helper outside the frozen checkout was rejected before submission.
`tools/submit_refactor_unit.py` now validates every source/check path before
freezing and submitting, and invokes the helper inside the captured checkout.

Eight completed source capsules were moved intact to
`/mnt/shared/pq-refactor-20261014-source-capsules/`, preserving tracked edits,
untracked files and ignored files, with repaired Git worktree links. The
old/new mapping and preserved HEAD/status are recorded at
`/home/rob/tmp/pq-refactor-20261014/reports/source-capsule-archive.json`.
Scientific inputs, fleet queues, CAS objects and other agents' worktrees were
not deleted. This is retained source relocation, with no measured space claim.

Additional encountered fixes: CLI help now names the calibration owner, and
architecture text identifies sample-parallel probing as retired and unavailable.
