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
| Typed calibration request and draw | `6af3ecf1c2c0f02ffa7f0bb002a5c81673250a98a00f312e416efe2d30406795` | 16 passed, no skips; two modules compiled | `5ac5dd215742e4c9c05b66a7dfe8c4c327bc2dd58fa731447cde1a185a64d961` |
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

The first source/cost-row owner split passes 350 cases with six explicit skips in action `e90dfa7d9f2efb756ce63a37afbe22696c3edb593383bb9b569557a743e748ee`. The skips comprise one undeclared fleet-data census fixture and five CUDA Tessera encodes. Public-name-normalized source proof `b56722cfc38383673b6a5a08764b1ff82b66078af98c3c09002df9ee1a0bd909` preserves moved bodies, constants and source initialization guards. The initial duplicate-name guard failure remains recorded.

J2 bounded GPU characterization and S2 ordinary-block traversal have separate reports, `joint-aura-gpu-characterization-2026-10-15.md` and `streamed-block-walk-refactor-2026-10-15.md`. J2 retains the original failed diagnostic comparator and verifies its immutable four-arm outputs through corrected admitted CPU postprocessing. S2 compares actual Qwen logits and all 29 boundaries exactly across four CUDA arms. Both retain in-process profiles and both-box Netdata; neither claims a speed improvement or wider-chain/expert-fork throughput qualification.

The public Tessera metadata consumer uses the exact API-only backport `d2e98b3cb2fcb6fdc27a552442ea021ee02da8ff`. Contract v60 remains SHA `ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e`, with unchanged 22 cells and gates. Consumer action `7a430d5d` passes 278 cases without skips, and `a6faa7b0` passes six explicit metadata-boundary cases. Producer broad-suite failures reproduce on the exact old pin and are retained as limitations, not passes. Other serving imports found in campaign ownership/decoding and route-image replay require additional public APIs; handoff is `/home/rob/tmp/pq-refactor-20261014/reports/tessera-additional-producer-api-handoff.md`, mailbox message `1792027811152-b3102383`.

Environment provisioning action `6eb3e535adf797672c0b0c42cf98abf262b79d7520ddea6deed3cba8fc426e12` exits zero with CAS result `2ffb67329d0630a7d59319a00b8ec4deb6906513761b2ee9982153c5bf121e71`, receipt `14a8afb9d12236085b422bd26d40a0638b31c2b3482ab34c9c9969b721112313`. It verifies installed package bytes/origins and unchanged contract in `/home/rob/venvs/pq-pin-d2e98b3c-pb20261015`, borrowing the old qualified numerical dependencies without mutating that environment. Torch is 2.11 CPU; this is no device qualification. Two earlier bootstrap attempts were rejected before submission for a not-yet-existing output dependency and a concurrently changed checkout. The final immutable bootstrap checkout resolves both.

Composed runtime/admission/pricing action `91c2c7eb06422785425f0c50ad67bd079478a64dda1260484100fa67c22ff72d` records 484 passes and 13 task-suite skips, exit one. Its sole failure identifies a missing explicit model-runtime initialization in the native-render comparison tool. The export worker is fixing this actual loader boundary separately; the action is not counted as a composed pass.

Additional integration evidence, 2026-10-15:

| Unit | PB action | Outcome | Result CAS SHA-256 |
| --- | --- | --- | --- |
| Captured pricing policy | `6c229d4abe84186e5307391729ae4bd04ff3a6954c287c35b1e51fd3d3b92f1a` | 194 passed, 56 warnings in 68.54s (0:01:08) | `5534ca5efb02edbd503d157d6a5e66b58d21bfd862232b11a59966ff042aaa8b` |
| Explicit candidate/cost rows | `58046fcce445c161424aaab2e692bc135ac7d0b74042588576f5fbae25c53360` | 529 passed, 56 warnings in 64.32s (0:01:04) | `3377603d8ee81c74a4544711ba84a312b42eafa9f6c3e5e0aee30d45cdec2727` |
| Shared price basis | `afd0a6e8c95a435c10b8686492c81088e614cc0489b59db158b667aab3b4e735` | 604 passed, 2 skipped, 56 warnings in 92.39s (0:01:32) | `097d0e5590084d89a6eb7bc0391a0d86f161460c86712d45416f5d6fb7ee523f` |
| Unadjusted validation estimate | `7e1c1f318103f6b72079eddbb289f5bb6efd841cc2f0f8ae2392275cbe52b26b` | 74 passed, 2 skipped, 56 warnings in 58.08s | `022197c51a29900a981807e6eae91c11e64eca0dde66dde0f7ff7359587c7e11` |
| Campaign table owner | `82c627deb1c222ba1a9a6ffa934a61fa12ad4deb9d5efed2e9833571ea927ed0` | 615 passed, 5 skipped, 14 warnings in 357.17s (0:05:57) | `6ad89463742a78c0e10543ced08f4ab31ad630c4f7a303f939e12726eccff2aa` |
| Concrete layer-output oracles | `0380215628bfcbc1674db0294e19dd1a77a9c97c8ed265b16ade879c3351a06c` | 2 passed, 14 warnings in 6.58s | `368327471a3ad95accdc6474d34aba610e1ac621d07131a579429fd950c721bd` |
| Output corruption sensitivity | `0c2d11c1bdbab9e79d86e796292ec7579db1c8caf2704016d60fe2b4b0f0ac26` | exit 0; result bytes verified | `bb9892a0a7e4cb5e2a708b2eba3553ca57f39bbb39ca5f5131b5e0bd4c9864e9` |
| Helper digest owner | `6d55d04daf7c379bbbf5a5c1485c4094bb2e0a4a0672c7a2698b37d7e9a2ac3b` | 10 passed in 52.63s | `6ce17f924555a6cc12d663229287ce4ce98931ee0fd5d5bc00b8e5d7fc889733` |
| Actual environment revalidation | `9b476d50d6d3c6e1ca462b05b890f2124fcb3289caaddfc72f7b341ab22c68ea` | exit 0; result bytes verified | `b3117db2ccfe00a37f6d1bf58368af114c019734e90a2310a4f002571ca09ecd` |
| Sample reuse metadata | `6ad183dbfbf3e6f4510d0beba26b36b02c557d3a40745e94ef2e2f8d9907ffe9` | 48 passed, 14 warnings in 46.38s | `f8ec2a497f95af14ad5d03ec4ce4d41b4f7cad179bf5f3bda3f3a6dc9f926680` |
| Pooled reuse metadata | `42dc97aa7d33d2aacc1d64a9e61392b2621ab41304e8334c2b20bc6559e85a90` | 42 passed, 14 warnings in 18.77s | `c6ef370b386c552457414375f585a6cdf35fab8d3d343de0a37a1293ba66f120` |
| G3 acquisition availability retry | `c8ca865f733870a8dbe5b887560e4c6ac56f29276b0a7599eb5123d3a0e018b5` | 16 passed in 3.07s | `0d6780b3fcd6e5ce8193aaa7bf3579e01dd0e50f0e25483b63c8860ceeb9cfab` |

The pricing and validation checks skip two historical DSv4 artifact directories
that are not mounted. The campaign-table check skips five CUDA encode cases;
its cold-import case refuses numerical runtimes and serving imports. These
counts describe their individual source capsules, not a whole-branch pass.

The original wheel-only environment had correct reviewed package bytes but
exposed duplicate Tessera distribution metadata and lacked real Git PEP610
provenance. Strict PB qualification rejected it before testing at `196bb124`.
The separate `pq-pin-d2e98b3c-pb20261015-git` prefix installs the exact real
producer Git commit and checks its package against the reviewed wheel. Strict
checks `0792327a` and `5621dfee` passed. The helper's retained receipt records its
initial phase, so its pending strict-gate field is historical rather than an
assertion that subsequent strict checks failed.

Combined check `bd1306e6` passed 584 cases and 59 subtests, skipped the two
historical artifact cases, and failed its new helper digest site. The helper
now calls the existing digest owner in both parent and child; actual published
prefix revalidation above also passed. This failure remains recorded. The
subsequent pbtest invocation rejected `-q` before testing; the corrected command
uses supported options. Combined table, Fisher and price checks are pending.

The old output-hash tests already failed on the pre-policy base. Their repair
keeps companion/Pareto/seed hashes and checks actual assignments, independent
byte counts, price sums and exact pin authority. Four deliberate output
corruptions each fail a concrete assertion. It does not establish an unexplained
historical metadata difference as harmless merely by replacing a hash.

Ordinary resume previously accepted sample-stamped shard metadata when the
request omitted those keys. PB `5d19aada` reproduced ten failing cases. Individual
and pooled reuse now compare the same owned sample metadata field set, including
requested absence. Writers and numerical statistics are unchanged.

PR #2669's acquisition fallback is incorporated locally with the current refusal
classifier. PB `f97f2c4d` reproduces one refusal, with 15 passing cases; the green
check above passes 16. The earlier 12-case action `624220ca` ran only existing
coverage after the initial patch failed to apply, and is not a regression proof.
No PR or issue was closed, merged, commented on or pushed.


The nineteen-file composed PB run completed 567 passing cases, ten skips and two
failures, with no collection/reconciliation gaps. Both failures are duplication
guards for the moved table helpers and routed source-proof tool. Follow-up table
owner `e5158e3458` and tool owner `14ebc1ca31` remove those sites without a waiver.
The subsequent integrated owner/guard check `ffb16121` passed 105 cases with no
skips and compiled the active package and tools. This later check precedes the
separate reseal staging correction, whose own 74-case check passed.

A source capsule was captured during an unresolved architecture merge. It was
never executed or submitted. PB `ca7f2384` reproduces the missing refusal;
`23ebf653` passes after the capture helper rejects an unmerged index before
creating or registering a target worktree. The invalid capsule and explanation
are retained separately from the explicitly named valid broad-test capsule.

Broad changed-test verification is now admitted from that valid source capsule.
The first attempt was refused before execution because the GLM census test is
marked fleet_data. Its exact two artifacts exist and total 1,454,802,228 bytes;
a separate manifest declares their known digests for that one offline validation.
The remaining 367 test files use a separate PB fanout. These submissions are
pending and are not recorded as passing coverage.


The separately declared GLM census test passed at
`0386b8141ee4636f12e69dedbc646456a8d019c60291290d9c1937d075aa8ab1`,
with one collected and one completed case, no skips. Result CAS SHA-256 is
`756cb881249eff7ac834e2a17a8ad7b96a687fca910d69d763675aca51cabe0b`.
It validates the actual digest-bound historical probe/census roster and raw
statistics. It grants no new served quality or speed qualification.


Further integration checks, 2026-10-15:

- Cold rendered-cache paths are shared through `render_cache_paths` in
  `2379886723`. PB `90ae0951766f` passed 373 cases with six unchanged CUDA skips;
  filenames, format spelling and collision refusals are preserved.
- DSv4 initialization repair `062cce9920` preserves packed expert class identity
  while selecting the probe constructor through `experts_type`. The original
  grouped Fisher failure and fresh initialization refusal are retained in
  `dsv4-initialization-2026-10-15.md`. Follow-up checks passed 95 cases without
  skips, including poisoned real parameters and initialization-context isolation.
- The broad changed-file suite found an unmigrated `CandidateBuild` consumer in
  the streamed Joint research harness and a digest spy aimed at the old streaming
  import alias. Fixes `d668b0939c` and `dc312bf438` pass 29 cases without skips in
  PB `579e7275606a2717d299c223449f835db4fdb0bb2b472b4969540129992e7d2b`.
  Its result is `7f34676a85163194cba6ee074ed1c99f3a34d336cd8bd4fa3f916a1e2ef70a1e`;
  receipt `6954ff86fbbf6c2f8eb10cef00c7707a879b8c393df874393e81012015cf9c6f`.
- Campaign seal classifications now follow their table owner in `c5301f9796`.
  Scanning that owner also finds the existing declared-size versus replayed-draw
  digest comparison. It is classified as byte integrity, with no new runtime
  check or refusal waiver. PB `fa98a351fe6538b544fba2c8791e8a5a67c2a26f119f039ebc66026d6092ccec`
  passes all 18 lint and fault-injection cases without skips. Result is
  `013c4ed7158360db0e60f3801b802cb2a7926d617bfeb1da62b4ef5d96625b65`;
  receipt `beb12339d5ac1e199d32cb079695b44049bff65e76af81219b70edadfdc4ac24`.
  Both result blobs above were read and their SHA-256 and byte counts verified.
- Real Transformers stays loaded in the corrected campaign test fixtures,
  `f5722ef4f5`. Only `AutoModelForCausalLM.from_pretrained` is patched. Independent
  fresh checks passed 79 cases without skips, rather than concealing imports
  behind a replacement module.

Independent review found two historical `ActivationIndex` reader tests were
incorrectly archived as producer-only. Their actual post-validation mutation and
prefetched-directory custody checks are being restored with a committed-blob
fixture. The separate table coordinate-validation path still imports Torch;
its scalar row-contract extraction is under implementation. These are pending
findings, not verified completed repairs. The broad suite remains in progress;
individual passing units do not establish a whole-branch pass.


The earlier frozen broad run is now terminal. Its 367 hermetic changed-test
files produced 7,790 passes, 144 skips, 45 failures and 70 passing subtests
across 16 PB shards, with no reconciliation gaps. This is a failed historical
run, not a final branch pass. Every failed node has a disposition in
`/home/rob/tmp/pq-refactor-20261014/reports/broad-failure-dispositions.tsv`.

Further verified repairs:

| Repair | Commit | PB action | Actual outcome |
| --- | --- | --- | --- |
| Real override registration before injected refusal | `4802a4459d` | `be9a53cb30bfd9c9749233ada6cab38c65e87607bdbd63f2e6d741bf0c6e49cb` | 43 passed, zero skips; fresh baseline `366b4aba` failed 17 cases |
| MTP schema import at its initialization owner | `44dfdbe8d7` | `e086db439d430d0f343a3706c995606dfa7555560dac590dd30fa56cd2b0084d` | 29 passed, zero skips; four broad failures reproduced in initial composed capture check |
| Ordered typed-rung fixture extension | `48ddd219cb` | `1f203eeee2540fa2111a77e3454c8d119ca451e156e811d2882e1e4aee937591` | Both original allocator entrypoint/overflow cases passed |
| Anchored limitation at the predicate's actual owner | `d68412a07a` | `355dbb3fd6b586ffa59947eb56a5e6e26d482d5415b76ab1c90807791f5cb97f` | 10 passed, zero skips |
| Controlled scheduler clock and real queue barriers | `84e5db56ec` | `b1ebbe742f73ae0fab503b6354131accd2c9e43d69524512bea8817e62e4fe58` | 14 passed, zero skips |
| Manual transition digest and native panel recipe imports | `06732f05f2`, `4a031cef88` | `3535839499722e92eb79baa05a7baf6f6eaedaafe65e5104ce042df5ab037f2e` | 36 related cases passed, zero skips; both callers compiled |

The scheduler case now advances a controlled monotonic clock while the actual
executor runs closures and barriers. It checks that a deferred task yields to
another optional task, stays deferred before its due time, runs when due,
completes once and records exactly two 0.25-second execution intervals excluding
a ten-second deferral. Real waits are failure watchdogs. No production scheduler
body changed, and no timing or throughput claim follows.

The historical activation-reader finding is fixed by `e4ffde062c`. Both original
mutation/custody tests now use a frozen plain committed-blob fixture and the real
`ActivationIndex` and chunk stream. PB `b4d31cde45fb` passed 44 cases without
skips. Deliberate incorrect content identities and pathname-based prefetch each
fail their restored check in admitted proof `67eb18d313d2`. The retired producer
and its CLI remain retired. The ship fixtures use actual attestation and captured
complete native recipes in `50981d14a0`; their checks pass 99 ordinary cases and
38 explicitly selected task-suite cases, with zero skips. Existing same-size
weight-content tampering refusal remains covered.

The manual caller inspection also checks relative internal imports, which the
earlier absolute-only scan missed. The rerunnable source scanner and bounded
output are `reports/check_internal_imports.py` and `reports/internal-import-review.json`
under the local refactor directory. Current inspected explicit internal imports
have no unresolved names. Dynamic exports are outside that static check.

Calibration capture and readsets are integrated in `7ecd85752b` and `b86a87a8f1`.
See `tessera-calibration-forward-qualification-2026-10-15.md` for actual inputs,
failed harness attempt, corrected CUDA equality and profiler/Netdata evidence.
The composed capture body has identical normalized AST to its immediate root
predecessor except the final tuple becomes `CalibrationStatistics`; both body
hashes are `8f031fc1af1df4f550c28e0c1796c129c2f7fdaf0fd10f08715c18825d363971`.
Existing routed helper ownership was retained during integration. The first
composed check has 130 passes, two explicit artifact skips, two subtests and four
MTP-import failures; the independently verified MTP fix above resolves those
four failures. This does not relabel that first action as passing.

The cold table row-contract extraction, wire/cell/request units, one obsolete
menu-test import expectation and additional public producer APIs remain under
work. No final whole-branch qualification is claimed.
