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


All 45 failures in that earlier broad snapshot now have targeted passing
follow-ups. The last obsolete contract test is repaired in `111207ac95`.
PB `fc70910ae59b89a7b534564edf43475e85fd415e55b78a1cf286db16e2c767bf`
passes 93 cases without skips. Its actual isolated interpreter blocks serving,
Torch, Transformers, Triton and vLLM imports while reading the exact installed
code/contract pin and all 22 published cell IDs through the public metadata API.
Result is `31daadbc3a15a6a1916aa4108238869f95a62ee2aaaf65d2b69e0f7526f76b1c`;
receipt is `a4ea45f234017d426f845dfbca5e27d5cb4f67b2907000bf5961fd8ea9c206ec`.
The failure ledger records the actual follow-up for each original node.

The cold scalar row-contract repair is integrated in `60e749c5dc`; its independent
review and unchanged-operation evidence are in `joint-row-contract-2026-10-15.md`.
Wire custody is integrated in `fa84d81d32`, and `6dc41ff8c9` repairs readset
producer provenance. Four actual copied dependency mutations previously left
that digest unchanged; each now changes it. The latter binds the shared filename
owner and three actual phase/replay owners, preserving digest framing and schemas.

The integrated source at `60e749c5dc` passed the 12-file owner/custody/guard check
in PB `fed2c0f3714cb1f6716fa74cb42889fae52f70b4fb19e995dda7af26a5d7db18`.
Actual outcome is 157 passed, three explicit CUDA skips, zero failures; all
`prismaquant`, `tools` and `experiments` Python sources compiled. Result is
`f5d756ea51b54ac73907c9b88bb44d45a28909c99654449a504087413953ccae`;
receipt is `7122f0c49e6d92d3c2529658a9ec97ca4a2b53b56de84c23905915c4e8822fc3`.
Both result blobs above were independently read and checked by SHA-256 and size.
The compile and guard check grant no CUDA or private direct-I/O qualification.
Typed requests, remaining cell measurement and the additional producer APIs
still require completion before final branch verification.

A further GitHub read-only refresh found the same 36 open PRs and unchanged
heads. No remote PR disposition was executed. The additional API handoff was
resent as mailbox message `1792036048215-f730d08d` with its shared context path;
no producer acknowledgement had arrived at this checkpoint.


The cold menu prerequisite is integrated in `ea5b9f4dc7`. Its real isolated
menu test blocks serving, Torch, Transformers, Triton and vLLM while checking
menu defaults, modes and refusals. Two unused solver reexports and their pointer
identity test are deleted; three actual DP behavior tests still call the solver.
PB `8f9db04d7906c503ffdc8fc94c688711228b09801a24d1d0a56ff166af469299`
passes 118 cases without skips and compiles four touched sources. Result is
`0bfeaba30c6da5e29c4eb8a11f586d9b14383ea2b537d4ccf520eff210409ae0`;
receipt is `53191c653d4632b7badb4e59c5ce75ffc031bdd2cfa7429467fe4f46cd747997`.

The T2 structural preparation/join unit is integrated in `d83314d6a4`.
PB `70e7c64ef5b513597b5efafcd32338890e149edce4c5fbd1d195d33f62477717`
passes 107 cases with two explicit CUDA skips and compiles six sources. Result
is `498ab5a7e92dcd784139baee2d90d2e2662234d950be82f59ad023fd32e3c048`;
receipt is `2ec0af4b82ebd8a2637a2ea4e2841e9082ae501709fe58a6fd5d5eab2021c96f`.
The coordinator independently checked both terminal outcomes, logs and actual
CAS bytes against their recorded sizes and SHA-256 identities. Both units used
CPU1, RAM4 GiB, native threads1 and portable admission with the exact qualified
Git interpreter. Their immutable source compositions are recorded locally in
`reports/tessera-menu-cold-composition.json` and
`reports/tessera-cli-bookends-composition.json`.

An independent read-only review of the integrated prep/join unit found no
missing request fields or changed refusal ordering. It traced nested census,
calibration, pinned-roster, selection and streamed-identity consumers. Preparation
keeps the saved calibration and source context through the synchronous identity
callback. Join retains its automatic-source refusal before I/O. The checkpoint
settings codec still consumes the original CLI Namespace on its existing path;
it does not serialize a request dataclass. Direct request copies retain seed0
and explicit false controls. This review grants no additional execution coverage.
Census/capture/pricing request units, the separately reproduced argv defect,
and the additional public producer APIs remain pending.


The real CLI resume-head test replaces source extraction/evaluation in
`225b62f925`. PB
`45f1d7a12b3d8fb5df858565c4d014ef4f022eb588b6da618d66a1d4178dc744`
passes all nine retained geometry/admission cases without skips and compiles the
test. It observes actual persisted head telemetry before source authentication
for absent, manifest and partial-directory checkpoints. Two deliberate source
mutations are caught: the obsolete checkpoint keyword causes six failures,
and forcing load-all on checkpoint presence causes four. The worker restores
the source after each fault. Result is
`3fd37ba1e57ffc8d0850377a59b0e93271d9a9dc48884fbc2c0f80701b5b3852`;
receipt is `bae278c7271d6106a019c535809b0e234fd7ac84bf8cab833aabccbea2b6e3d4`.

Census/capture requests and runners are integrated in `41d7417de3`.
`CensusRequest` carries the ten actual census fields. `CaptureRequest` carries
eighteen fields with an explicit monolith/quantum phase. The CLI retains source,
calibration and model lifetimes. Existing collection/callback operations compare
exactly after request-field/phase normalization. Five moved raw serializer calls
use existing named profiles, with separate byte oracles for Unicode, NaN,
Infinity, indentation and spacing. The obsolete private digest-site baseline
entry is removed; the remaining main-function entry is retained.

PB `8fcc4bacf2552cdd5d48bcfc5d82b579a2b9c44c7aff2f829d96b9a9426cfc18`
passes 220 cases with two CUDA skips, but fails the obsolete baseline entry
check. Its runtime/test/proof bytes equal the final source capsule across nine
recorded files; the coordinator independently checked all nine identities.
Final PB `ea183223eea5b3dd84e572cf65fc793c9bc0c13e16c6578111b5bc1fc827cd07`
passes 43 guard/oracle cases without skips, compiles seven sources, and verifies
the live primitive-site set against the scoped baseline removal. Result is
`3f0d12b1be8a115d3f88a38f2beb225c2f9dacc4566afa7b9f515a262b16eaf1`,
1130 bytes; receipt is
`b0488aa4db3192c0d964e32a4b95ba7191b1a5332cca996ae0bf0f0e2a777a82`.
Terminal outcomes, logs and successful payload/receipt identities were checked.

Earlier phase proof failures remain failed: `6c32be51` selected a nonexistent
test filename and ran no tests; `f2955250` exposed inherited raw serialization
sites; `db53ba20` required equality of a generated inventory's entire presentation
rather than its primitive-site set. No generated whole-file rewrite or new
allowance was adopted. These actions have no passing receipts. Final execution
uses CPU1, RAM4 GiB and native threads1 with portable eligibility.

A read-only explicit internal-import scan after `41d7417de3` finds no missing
absolute or relative imports in the package, tools, experiments and live tests.
It does not inspect dynamic exports or certify runtime behavior. Pricing and
additional public producer APIs remain pending; final composed verification
has not yet run.


The raw checkpoint-settings API is integrated in `e90583f2be`. The public
`campaign_checkpoint_identity` copies a mapping and retains all existing wire
filters and per-unit acquisition proof. All fourteen caller test modules migrate;
the private API is deleted. PB
`1bb969c75aa65e368892aed4232ed3a4e57e329e23337fa2ca2c787d4f3b9bcb`
passes 363 cases without skips and compiles seventeen sources. Result is
`913b3a5f621a884010a5aa4d27ba291dcdba6fadfdc2209e78ebe1d8c5c6d8c3`,
1234 bytes; receipt is
`26be667c2439a27ea58d260853137fba854148fad8370d70698fba7142ca5170`.
Actual terminal, logs, payload bytes and receipt were independently verified.

Its two-record historical fixture came from the old producer in PB `235dd63f`.
The candidate checks its actual current package-source seal, then normalizes
only that field to compare all remaining historical record content. It does
not claim that different source trees have identical whole identities. Raw
settings bytes independently preserve unknown keys, zero, false, null, Unicode
and acquisition unit proofs. The main-function snapshot follows all direct
acquisition stores; read-only AST inspection finds no later direct args stores
or setattr calls. Source-seal gates remain active.

The encoder-startup source-order test now observes actual main/model/seal/
producer events through its real captured-row fixture. Existing failure and
drain cases remain. One packed-module docstring-only assertion is deleted.
Independent census/capture review found no missing nested fields or changed
valid phase, refusal, source/guard or serialization behavior.

Encountered dynamic manual callers are repaired separately in `6d2adb9df3`.
The rebind timing harness instruments the public API while retaining historical
phase names. The profile reader counts old, new and recursive frames once under
the historical output key. PB
`29f6155f4feda38c288070fd17c9c5a97a4582fe3e4b736c236b300315a81ee2`
passes two cases without skips and compiles three sources. Result is
`d6ed4c4ffd8fcb85c8c478fc14af8a3c056716dfb6204c00074b9251622900e5`;
receipt is `40258850f8dce15ddbd695775e3dd8029990bec821b4ec8995631ab36b5cc20e`.
This CPU instrumentation case mocks the CUDA availability boundary and stops
before model execution; it does not qualify a CUDA rebind timing run.

The encountered sampler wall-clock test is repaired in `aa648ece2a`.
A real sampler thread and close/read barrier replace sleeping for 300 ms and
expecting at least four samples. Four literal readings independently specify
peak/minimum and final-read ordering, while one unmocked /proc read preserves
source parsing coverage. No production sampler behavior changes. PB
`b857b29d43eb780e98b8a26a10b4e3943079223f28f465edc15a24e0daff864a`
passes the full twelve-case module without skips and compiles the touched test.
Result is `fe152fbf0fa75fbabd16e334b2b49b2228edeb73de5d31a51eb35edf3f0539ad`;
receipt is `cd20a0c6aa89fd4fcc512e58a899f64aee0a07d3e2c6a4ba648c6f1803a33b0c`.
Both successful manual/sampler results were independently checked against their
actual terminal records, logs and CAS bytes. Each uses CPU1, RAM4 GiB, native
threads1 and portable eligibility with the qualified Git interpreter.


The sampler follow-up `7d2f623b4a` preserves explicit Linux kB-to-byte
conversion coverage with literal input and expected values. PB
`30c7cf210b05611ce18516d21fa31a7d454e2a5d5598da55de9210dbf74e2bd4`
passes the whole thirteen-case module without skips and compiles the test.
Result is `438d459033c4ce96062374c0d477adb205014a62adf9d42226a00a98116c79d1`;
receipt is `8d3ca14b6e687fa90256408bf8055883c5d59188a13de39ea057b769afa30b4c`.
The earlier twelve-case proof remains historical; no production sampler change
or timing improvement is claimed.

The scope resolver is integrated in `217271a66e`. Its nineteen-field request
returns twenty-three cohesive facts using the existing roster and population
structures. Whole-population and selected pricing views, source admission,
sampling and refusal order are preserved. The initial PB `0e5a292a` failed
after 308 passes and two CUDA skips because a formerly local import was missing.
It remains failed. The missing bindings were repaired before integration.

The corrected partition proof `d48a18ef` passes sixteen cases without skips.
The corrected scope/family/guard proof `c88966fa` passes eighty cases without
skips and compiles four sources. Its result is
`3395d2128231854c56ac56c47fafac4e2da21c6bd762be791a22717cb4597544`;
receipt is `f38cc48a732bcc6d57e67f66e74335d5ff80690ccca0d51c11d0a5b39a050a2e`.
The actual scope operation's sixty-two global reads have no missing bindings
in PB `ba6da1a3`. The durable global-binding tool also inspects nested code and
actual function globals/builtins; its negative cases and related guards pass
fifty cases without skips in PB `7e39db2f`. Result is
`d40c5596cea93a8a3b43b47fa411ebac26624ef6435f00354bb1eac807243b39`;
receipt is `dade9d08de49217574353a37098c03d6e0ce11615b6d8c253b019884ea07ddb6`.
These overlapping selections are not added into a unique test count.

Independent review caught one annotation mismatch: seed_checkpoint is one
optional string path, rather than a sequence. Commit `7a2ade0283` corrects it.
PB `73f4eac0` compiles the request, verifies actual parser-to-request preservation
of one spaced path and resolved annotation, and reruns actual global binding.
Result is `7db9a78a23ed0b115a6018accadd1dbfe464080043dc1c6283b28fa7d15b94c7`;
receipt is `0a09e91bf4f01452185f3e0b20cd8387f82fe2b1cc8267e386da4948045702c4`.
Terminal records, result payload sizes/hashes and embedded receipts for these
successful actions were independently checked. Resource reservations are
CPU1, RAM4 GiB and native threads1, with portable eligibility. Final composed
verification remains pending the menu and pricing operations.


The menu/projection operation is integrated in `71972608cb`. The actual
production, partition-test and global-binding-tool bytes equal the qualified
root-composed source capsule. The call remains after the existing activation
memo/cache setup and before acquisition scheduling and source closure. It
retains whole-partition meta geometry and narrows only priced consumer maps.
The partition test now calls the operation rather than extracting and executing
its source. A producer-controlled projected-dense case retains the real refusal.
Independent read-only review found no changed policy or ownership behavior.

PB `1acee643710085defd2a306248ae693b41cbe117694b075dd2a31cc70be4aa50`
passes 179 cases without skips and compiles three sources. Actual operation
body equivalence and scope/menu global-binding checks pass. Result is
`99d717148d1f5051749f4c25f41848fe5f1a9a9407ae7e6fb778965e599526d7`,
1186 bytes; receipt is
`3e6c6c4f6bcda06c2c7e4a6e29d1e5a1c64c1b00112627f028015d37295f929d`.
The preceding focused proof `2a632b8476dd2d1328a5b89baf931f0f4b30248b41ae24c53ac6d9f2f223643d`
passes 27 cases without skips and compiles five sources, including the next
acquisition-wire unit's real entry-point proof before its production change.
Its result is `40b7b56b55859f559d67c6706c32312188e851bb8194973c886eba73bb5f2dd4`;
receipt is `d0db53a52c3df13029a11f884d32699226d13e377f5ce8dcab0f915c791253c6`.
These overlapping selections are not a combined unique count. Both actions use
CPU1, RAM4 GiB and native threads1 with portable eligibility. Terminal records,
result payloads and embedded receipts were independently checked. Final
composed verification remains pending the raw-wire and loaded-campaign runner.


Raw acquisition settings are integrated in `4ec6bfc93a`. The CLI mapping is
copied before the three derived settings, updated at their existing policy
points and handed to checkpoint identity read-only. Typed requests never
serialize themselves into this historical wire. Production and new test bytes
equal the qualified root-composed capsule. The actual main fixture verifies
literal three-unit work, schedule and origin fields, per-unit source proofs,
input-map immutability and physical source replacement refusal before pricing.
The source-owner's existing held-descriptor and pathname checks enforce that
negative case; no new refusal or internal fact revalidation was introduced.

PB `876aeb6a2e0881f66570062a0ea0af76de9e42253b45c11880074749ee34070a`
passes 125 cases without skips and compiles three sources. Full guards,
operation global bindings and the three-store/provenance order proof pass.
Result is `3bc4154f44a70e5060fc3c119e69a5d249cd440db5dd30e3d593a8cc03f1df08`,
1316 bytes; receipt is
`e924f5fa1bbf21ba2ab5e9a45adada0cabe2d05ab836c8262e03398df1e1e4df`.
The earlier `1f0bc440` action failed before tests because its proof baseline
came from an older worker source. It remains failed. The composition tool now
retrieves baselines from its recorded root commit, records their hashes and
checks that manifest before body comparison. No unrelated ownership difference
is normalized away. CPU1, RAM4 GiB and native threads1 are reserved portably.
Terminal records and CAS payload/receipt bytes were independently verified.

A bounded audit of other changed tests found a span-accounting sleep that
could use the existing public clock injection. Commit `a1b69ab7ea` replaces the
20 ms sleep and minimum duration with a literal 0.125-second interval through
real IoSpanLog and QuantumCounters objects. Count, total, outside-phase wait
and recorded span duration are checked independently of wall-clock progress.
PB `a27d4f0dffc46dba4563e91f1c610eac651ebb404d7887dfc3f260ce17b3b5af`
passes the full twenty-three-case module without skips and compiles the test.
Result is `ec95fc81c815ba6a8398b12bd99ae484d0b7aedcfa606eae145d0551b0230dfd`,
750 bytes; receipt is
`208aba840eb6b270434df35073fd8d241ab571b18b07450f39e711032d69fe7f`.
Resource bounds are CPU1, RAM4 GiB and native threads1, with portable eligibility.
The actual diff, terminal and CAS result/receipt were independently checked.
No production telemetry code changes.

The package-runtime AST guard remains a structural explicit-initializer policy.
It can catch a missing local declaration, but lexical order does not prove
runtime order. Separate cold metadata-import and real streaming-skeleton tests
exercise their actual boundaries. The audit does not claim complete dynamic
coverage of every manual loader.


The bounded test audit's Stage B tail case is repaired in `87a75a1570`.
Actual IoSpanLog open/close events and public open_names replace inclusive
wall-clock timestamp comparisons. The final validation count must occur under
the payload span, after every window closes and before payload closes. Existing
row, unit, parent and outcome checks remain. Production code and shared runtime
fixtures are unchanged.
PB `f9c9deaaa43c449afec2df191a5d4b1dbfb17e74a929cdfb7a0f1e8c1e795ea6`
passes all seven module cases without skips, with fourteen warnings, and compiles
the test. Result is
`1a7f31b5b128f4bf41be507674de3820963ba544452d8ffdfb4c293e17b8b691`,
821 bytes; receipt is
`86409332e7dc04c55585c925b09db0e82fed2d65d7edef6001632c8715a2b33b`.
The actual diff, successful terminal and CAS payload/receipt were independently
checked. CPU1, RAM4 GiB and native threads1 are reserved with portable eligibility.


Typed loaded-campaign execution is integrated in `f92a4d8d33`. Ten actual
production/test/tool files equal the final qualified root-composed capsule.
The request is either loaded census or pricing; source and head variants,
placement, refinement, publication and outputs bind actual admitted controls.
The operation receives named calibration statistics and explicit runtime facts,
creates the existing cache/memo/row-stream/journal/publisher, and completes
synchronously inside its caller's authenticated source ExitStack. No Namespace
or serialized request becomes checkpoint policy. Requested identity threads
remain distinct from the affinity-capped execution count. Census retains H-off
and its existing constructor effects; resident capture provenance is retained.

Independent review found no implementation defect in the recorded source.
The final source-preserving function placement changes the campaign diff to
276 lines. All scheduling, priming and capture-publication helper callers move
in the same unit. Real census cases check constructor order and H-off under
both CLI modes; real resident/selected reuse retains capture evidence. A cold
subprocess constructs both request variants while refusing numerical imports.
No tests are removed in this unit.

The final four-shard PrismaBuild run selects 25 files and passes all 515
collected cases without skips. Per-action collected, executed and outcome
counts reconcile without gaps. Each action reserves CPU1, RAM4 GiB and native
threads1 with portable eligibility. The actual successful terminal records,
result bytes and embedded CAS receipt identities were independently checked.

| Action | Passed | Result SHA-256 | Embedded receipt SHA-256 |
| --- | ---: | --- | --- |
| `ec29cad4dba5836d4338057d52a191e0f54946cc61f2099681df3f54ce9ef269` | 169 | `b2bcb2664db2ad498168376db0d2d258cb4428b606964c6de1d1ffce2b25967b` | `2df282124d52b586a7689d3f616828676f2b55c03e6a298ca60b39357c31d05c` |
| `18b67acbe9495bf071df5265cdf9a5c4899f55f0ad629ea459e58415b75df6d8` | 87 | `fd073b1d4f1c8b41f21e289170d92c7498ead8cd2d63e3757fbc3c7d8eb2db08` | `a975d3932624c50370b33d4273a9d4f95280ea9441b3e53ced92dc1fd550e542` |
| `9c58eff190f011803d6d5a64b70d6a2e7da044f0b7b6f3a16d230fcc1edb1f1e` | 121 | `d4a50500dcf020442151ebf97f651773dc455feb22365e95623aa4faf3458ff8` | `8e4ba114e5c52861c55ed9f3ca400e0bcdc3440b412bce22a3057aec5c3ebfe6` |
| `c9e3a19e0dc3d261ea8ee86e27ad127e916b66ef4664879f10ce3ac3e21ab0da` | 138 | `b980e7068951743421db20f133017304c078c79290b8e7350f2506b752954657` | `2cfcaf209881d2243590d0d525c45ecdb6bcfe09c35282de55968a7b8a8bd76e` |

Final proof action
`475c6781e93c7222978b4422ea4d7993158680a71925e73fc489f52ddbd564b0`
compiles eleven sources, checks the actual three operation global closures,
and verifies the root-bound main prefix, static-scale/detachment/raw-copy
order, moved numerical/callback/publication body and three helper bodies.
Constructor, priming and operation-input bindings are checked separately.
Result is `56c665051b60fb8bec3fa33e66d0168bcb257cdfbc4d984da801cbd72a5bb686`,
450 bytes; receipt is
`847d31c9c367376400e2bc8989dfe6ef1a69a1c2c08e18826f4ec7e550f7dee6`.
It uses CPU1, RAM3 GiB and native threads1, with portable eligibility.

The corrected focused action `afd859745198c7bbc6dd762f9b5cb0cad00f8b84609c9bbaa36270feed41f71c`
passes 23 cases without skips and compiles ten sources. Its result is
`f37a056074d70e7adde00f46716452bf9c0ae7828eb705682c59177b38165eb6`;
receipt is `dd2ce7d63765b275bb5c952bba9cd5de9bed2f1b0fd7711ab28cfe4675c72a51`.
These cases overlap the final 515 and are not added into a unique count.
The initial `0535525b` action failed compilation because the test migration
placed an import before __future__. The helper now preserves future imports.
A later controller attempt refused unsupported reporting options before any
publication; the accepted run uses the supported empty pytest-options list.
Neither failed attempt is reported as a passing action.

The independent accepted implementation is complete. Final composed branch
verification is next. Additional producer APIs still block complete removal
of eleven serving imports in three files and the remaining T1/T3 work; no
Tessera source, runtime table, admission gate or serving default was changed
as a workaround.


## Final independent verification, 2026-10-15 06:42 UTC

The accepted CPU verification matrix has 8,881 passes, 144 case skips and one
collection-time skip for missing Triton, with 72 passing subtests recorded
separately. Its 9,025 collected cases reconcile across 441 files and sixteen
PrismaBuild shards. The original run remains failed with 8,878 passes, three
failures and 145 skip records. Three complete shard replacements supply the
accepted results below. Their file and collected-node populations exactly match
the originals. Neither the historical LIST nor its failed terminal records was
rewritten.

This is an explicit revision matrix, not one whole-suite execution on a single
final revision. Root source differences and qualified byte identities are
recorded for each step. No overlapping focused or companion pass counts are
added into a unique total.

| Source | Scope | Verified outcome |
| --- | --- | --- |
| `9c6164490d` | Original sixteen-shard CPU population | 8,878 passed, three failed, 145 skip records; preserved failed. |
| `e6c87281f6` | Complete original AQUA shard 9, 31 files | 642 passed, 33 skipped. Replaces the obsolete empty-row predicate mock with a validated Joint AURA row; no production changes. |
| `4f2db7494d` | Complete original materialization shard 11, 32 files | 541 passed, three skipped. Campaign grammar is applied only to declared campaign modules; real producer/translator and resource/container gates remain active. |
| `c00349318e` | Complete original thread-site shard 14, 32 files | 513 passed, one skipped and two passing subtests. The new qualification-tool thread uses the existing PeriodicSampler; allowlist unchanged. |
| `9c6164490d` | Declared GLM data lane | One passed, no skips, using the two-file 1,454,802,228-byte declared manifest. |
| `9ed23194c0` | Explicit four-file task-suite companion | 76 collected, ran and passed, no skips. |
| `9ed23194c0` | Seven guard/settings/runtime modules and all-source compile | 49 passed, no skips; package, tools and experiments compile. |
| `c00349318e` | Final nine-module guards, container and sampler checks, plus all-source compile | 108 passed, no skips; package, tools and experiments compile. |

The initial task companion `153da70de106bf8af2d92a6c5db2fee5777a2dbfd6294aded8cc7f109a8e0682`
remains failed with 72 passes and four failures. Commit `215e2f1fab` moves the
real HF-reader refusal before optional backend imports. Existing refusal cases
now explicitly block lm_eval imports while retaining real configurations and
reader checks. Commit `9ed23194c0` separately removes only trailing newlines in
six extracted owners; their syntax trees are unchanged. The valid task-engine
execution path and real quality inference are not qualified by these fixtures.

The materialization fix was independently qualified in a ten-module action with
230 passes, no skips and three compiles. Two new host/container cases preserve
literal command, environment and resource demand. Its real producer-to-translator
case is unchanged, and its plan check now reads the actual manifest instead of
mocking the row builder. The sampler fix was independently qualified with 44
passes, no skips and compilation. Five event-driven cases cover both hosts,
body failures, unavailable telemetry and an in-flight write after the bounded
join returns. The numerical/profiler loop is byte-identical at 1,821 bytes,
SHA-256 `58d9074d5c7160c18274516ec05711d047c03d7e863432731d487b3b9e8bd227`.
No numerical GPU rerun or speed claim is inferred from this tooling change.

The source-derived accounting tool covers all 443 surviving changed test files,
with no uncovered files or pending evidence. It checks 451 distinct selected
files across the recorded lanes. The frozen 441-file CPU population retains
its original selection provenance. The additional task-reader, container,
telemetry and declared-data modules have explicit separate coverage. Actual
sealed commands and runner bytes establish focused file selection. This is
file-selection accounting, not a claim that skipped cases ran or that every
source file has one same-revision broad execution.

The 145 CPU skip records remain explicit: 57 task cases qualified separately
by the current 76-case companion, 41 private Stage B scratch cases, six direct-I/O
cases, two unavailable DSv4 checkpoint cases, two closed-package campaign-env
cases, and 37 device or GPU-dependency cases. One of the latter is the collection
skip for missing Triton. Private scratch, device-dependent checks and missing
artifacts remain unqualified by this CPU run. No scratch fallback, dependency
stub or exclusion was introduced to make the matrix pass.

Each main shard reserves aggregate CPU4/RAM16 GiB with four pytest workers and
native threads1, with portable placement. Task, guard and compile lanes reserve
CPU1/RAM3 or RAM4 GiB with native threads1; the declared-data lane uses CPU2/RAM16.
The materialization focus uses CPU2/RAM8 and the sampler focus CPU1/RAM4.
PB-assigned affinity is preserved. All tests and compiles execute through the
published PrismaBuild client. The accepted matrix's 21 actual terminal records,
CAS action identities, result hashes/sizes, embedded receipts and both captured
log hashes/sizes were independently verified. Focused action evidence is verified
separately by the coverage accounting tool.

The rerunnable artifacts are under `/home/rob/tmp/pq-refactor-20261014/reports/`:
`final-verification-summary.json`, `final-verified-receipts.json`,
`final-shard-repairs.json`, `final-test-coverage.json`, source/difference records,
and the lane-specific skip records. `summarize_final_verification.py` reconciles
outcomes and receipts; `account_final_test_coverage.py` derives final changed
files from Git and checks their evidence. The original CPU result file's hash
and size are retained in `final-original-cpu-record.json`.

| Accepted action | Result SHA-256 | Embedded receipt SHA-256 |
| --- | --- | --- |
| `197577d1bc53589ce0fd92c5d0fcd2af031b81f13ae827053bd835939120587e` | `878eea474b2811c5c29b6a44bd472c7def7ecd72e850952f635123308f229f30` | `0436f8f094188679677f8341551afe98059a4d92ebad69963f8b5a6b8d3a1bb8` |
| `88af616e5806bd5c5c8fdccce4cfb8d58688a9018a0b4477c2279d7b2ae883eb` | `c144967a9199aa3511802168b23c07184a5c40e7a92ec55015340d151e3e913a` | `16f84d16a87de957301550633d6824ed22e805eb0a0a0055e13954be765d6d04` |
| `b9a48061cd3f8c0fcfaa176805129a92772e0cdf31bd725e625f6c717aab454f` | `93e2568ff638d3195d80c688569dfbe5e2b86d5ed7890e67d11d2ab2e7fdee2e` | `7d4df90dad73694227a74e4ab666f1154dc53d6a35b34bce62d51601412ae4e2` |
| `45a25542521e04aada1362b8947d7f35c67fc72296530751be307d5bfd7a796e` | `5c2e81b9a1b2cf908843dc3c8d7a61880b6817016adf9c1a2d1ef8ceeffff4ec` | `ec4d311b24c65311883767de3156b19821b9f6d26843e15b3ab8ae7b1f681cd1` |
| `1445439a47ad35d8dffa51bac75124b75bcebbc2340aa0caeb97a3b992cdc507` | `365a0e1e445e9373c89961123ebbf928794db8a6ca9ac652b975bb36beaec66b` | `934853ec6dd34207038eec40cc5e2d8a40d2348a023e2024d0193d5723710eba` |
| `7003e82c5682803e9bd843d8253746ec951b3788b5141cf0a0dcec70ab7e2c56` | `867403f6741ef87ac5df7c087f60a1b98bf2703cbad987f47176129bf275852f` | `c1144113c9e7d21d93addfd2daf7ef328e4c2a0dcd20ca279d494e84a44444c5` |
| `68c1950e6ec3d8eec539cec3086a7b2d0024d55a6072ae8ab870ba5ebb7d386f` | `6a656dc666f6c1d3bdfe91e3bf78600e13fd91739450e86203c01ec9fa4466b4` | `2861d25da55d284558f646daf624ce0ad581f7e95282154b23ee004b14897866` |
| `743190a292f3d6406685d57868e0027f58797432eb393b5f74b1066ee1c4e5fd` | `545bad91faed4ad435dde28a0bfc21cd3933857a8da38879c18d36613230ee59` | `9e8ab42376d1157faff017c20692b3d3330f6b4027e322d933b9a519576cf8cf` |
| `c53a98a3003e20db4eb7e8b66c7664f94f4a1eaf4213a868c4027e941546cf19` | `1e55c99d906a6aee343d7afaea1ccf7d984a253870a722e30a7b0e12ec515365` | `77e6dc1ff99dde7c7e9b903a5a22213c4fc164f1cbec4a73c8e5c5ebd955e31c` |
| `7a45a4edb07f0673a09f460fadb49ce1eda3f08e674586d6c9c84807b6a2bc69` | `16ea321848c6398ed7ab60511ba6ca723c961448b06593654c93f8b2e92c118c` | `a910cb2d3326d53e8214b13d350452586819872182221f65fa2a9e5e54307d50` |
| `76a09541422c85d31204870802c785d34dd82f2de7562a6cf0e8442f71b74132` | `4883f6bd7481a6f57a1ab5f4ec0886c377582eef6a3cd66f7102f37436cf083a` | `d1722de3d9584b1dd9f7298fd9ef4142827473bdb8da46d5b73345a39b5e78d0` |
| `af6e69ab81f0d820bf1d55e3b18bf504dd1a7bc66e8db434591714970cd87ddd` | `a8dc01885d6bd5b9e01fd20ec7f9a16d1919cbc81689d4b1cc291ae58ee189d9` | `03db1f49351d2c1443cd977cd2112e57ef0bc14af05fece7762fade1f1a7968a` |
| `7d35a187dc2f6c223495d2afb1dd3830813a1ad4ea51892117ef999f58bedf3d` | `efdee7a964d592d4b3fe433cf16c69199843c21f255751ad309f0ae4c4927b31` | `e37c78d33555fa0b64456cfab20e52b73844c7e736a8bd35d5a7678f5b039122` |
| `f206585055ea30a16bf95a255f54ac4d7acc2e599c37d6d0245e67ce2052b24d` | `41062d5a4f5ab2b06beb8f59eded41c2e3673749f5c98a492389a91b404c580d` | `883d8c0bef265d350a4ad15e6821e4ca7c100d75b80284f3067f9584f6a8fdd8` |
| `a1ccfbb36644bdf78e94b8aa3c58b0f22484f53e5c20df9e3e9ba7bccf23713d` | `94dc0ba8af524df145548d71fe87c432f30e605db676b262d29f4c74b27c2af0` | `48a97bcd65e49dddf3a98ae6526192a5c19828a745868624b3b49e2aacdd2822` |
| `6a1b117401a5a4678ac77ec2941340c71def43e4bdc08f1fa1154be722154041` | `fa0c8f926a45c0bd1c33224f05bf40d33efdc23108a200d46fd57f09e2d3597f` | `7d8fb4f63544d23ae95ce3fb6eb55ec843ec82aa962ea294f69b0431ef58907d` |
| `d8104473fe7cc65128028f4beae1a8adcd9ecd0cb38f8dcaa9017210b5c574d6` | `afa2878a8c3adf383ec176e00fb6fbe38997894053b51630951c1759b3bafa93` | `7ebf285204443316384d0dd5fa8d098fda0ee8dcb953271ac31be95767aae8a7` |
| `d21cbd04fc5b2573da775511a029b6a6ebbb6576a264974849f33e0da41eb1c6` | `2575182b66340976661daca78fba7d736ea57ab5f785d5b0f840dea486e6c028` | `4a9d1dfc5d7b8d8ff4f3f5fef98922c7bc5e6665a69a25c9f2436ead69049398` |
| `7599875dfab54299031e4dd1fe9666ae29d5b8858ef8d40f6d0711909e85c0f9` | `1e44fa9cf5e9b8dd215712d8639e1a824e1fbe2d54bd4fa6d8b65e76b75dfe1e` | `1f9c94c801c1977ad251a5f4dfd356ad64efa4add37f7e251fb5dfe9414dd31f` |
| `7107d48da14393d9dc2131a9f9a6120e46377965ee6330602cc6484591e5d400` | `237c2099fd2a892f4991988abd14de401f5337d378d0a03c4b2c4f0c96f2983d` | `a6185a5fbb1cd2c8fd47a993f668de808c9d0ed6364d048dd77542d93926c9a0` |
| `3d0161618cd41d888c00cc9f848e7a055cef47ebc2f73089dde27c92948c5ac0` | `d545f1f66798c6d2f89fa203a74e1f970338e8ab3f267af1b0654ea7512e0456` | `fcea94f678b8a7d3214aef00f250c395e7ca3bf4a285dc7a4db71d7059c5502a` |

The API-dependent T1/T3 remainder stays open. These checks do not extend the
reviewed v60/22-cell roster, serving/export gates, quality qualification or
performance claims. Historical failed runs and historical measurements above
retain their original status and source.


## T1/T3 completion verification, 2026-10-16

The accepted refactor is implemented through `9baa795565`. T1 consumes the public
Tessera metadata, projection and image receipt APIs. T3 owns resolved cell recipes,
measurement, batching, memoization, scoring, publication and resume checks in
`tessera_cell_measurement`. All live callers moved and the old APIs were deleted.
Static AST inspection of `prismaquant/` finds zero `tessera.serving` imports.
The [disposition report](prismaquant-refactor-dispositions-2026-10-15.md#t1-and-t3-implementation-completion-2026-10-16)
records the implementation and separately rejected proposals. The
[original review](../reviews/prismaquant-architecture-review-20261014.html)
is preserved byte-for-byte.

The new API-only pin is exact Tessera
`aaa417003f0ee690e8ace138acd18df7f9531369`. Code, JSON and development pin agree.
The packaged v60 contract remains
`ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e`, with 22 cells.
The separately provisioned CPU environment uses Python 3.14.4, Torch 2.11.0+cpu
and Transformers 5.17.0. PB verifies the installed Git commit, RECORD bytes and
absence of import shadows before pytest. Schema-v2 runtime admission binds the
reviewed contract bytes; the environment proof supplies installed Git provenance.
The delivered wheel hashes to
`5823979dbba0404b5e926dccacab9a2acd4ab15f32ace7db79022f541c32fae5`.
No active production installation was changed.

The accepted broad CPU matrix selects 1,149 files and records
19,940 passed, 520 skipped, 1 xfailed, 114 subtests passed. All collected nodes and selected files reconcile after replacing
each failed shard's complete file and case population. The original run retains
19,893 passes and
47 failures. It remains failed.
Two test names changed with their owners; their explicit one-to-one mappings
are recorded by `reconcile_final.py`. No failing cases were dropped.
The 20,458 collected cases exclude three collection-time skip
records included in the skip total: two missing Triton imports and one GLM
derivative check requiring its pinned Transformers 5.16.1 modeling source.

This is an explicit source revision matrix. It is not one full-suite pass on
one final SHA. Overlapping focused, companion and GPU counts are not summed.

| Source | Scope | Verified outcome |
| --- | --- | --- |
| `86a937ea4a` | Original sixteen-shard CPU population | Original failed run retained; complete replacements below supply accepted results. |
| `968b6aacda` | Original 72-file anchor shard | 1,021 passed, seven skipped, 13 passing subtests; complete typed anchor fixtures. |
| `88089d0ba5` | Original 72-file runtime/packet shard | 1,370 passed, 32 skipped; explicit runtime initialization and new dated packet. |
| `49409f2729` | Original two caller shards, 144 files | 2,593 passed, 164 skipped, 73 passing subtests; source binding before import, live owner tests and unchanged numerical goldens. |
| `785cbb74e7` | Original 72-file legal-domain/receipt shard | 1193 passed, 10 skipped, 61 warnings in 747.64s (0:12:27) |
| `9baa795565` | Original 71-file checkpoint-settings shard | 1207 passed, 6 skipped, 211 warnings in 330.33s (0:05:30); current source seals checked separately from the frozen identity oracle. |
| `86a937ea4a` | Focused 32-file cell/campaign/profiler CPU lane | 744 passed, 10 skipped. |
| Recorded data capsule | Declared GLM fleet-data lane | 1 passed, 0 skipped; two files and 1,454,802,228 declared bytes. |
| Recorded companion capsule | Explicit four-file task-suite lane | 76 passed, 0 skipped. |
| `785cbb74e7` | Final eight-file docs, boundary, pin, domain, receipt and provisioner guards | 140 passed, 0 skipped. |
| `785cbb74e7` | All-source compile | `prismaquant`, `tools`, `experiments` and `tests` compile successfully. |
| `968b6aacda` plus sealed qualification helpers | GB10 real encoder and memo parity | Six passed, no skips. |
| Same GPU capsule | Native-batch CUDA events and CUDA-only argument guard | Two passed, no skips; one native CUDA case and one ordinary guard. |
| Same GPU capsule, corrected selection | Real timed CUDA window excludes later kernels | One passed, no skips. |

Only two code files differ after the primary CPU source. The reseal proof driver
now binds its declared source tree before importing its measurement owner.
The legal-domain audit updates its paired code-commit literals to the API pin.
Its contract, export and grammar fingerprints, source-state match and historical
rate counts remain unchanged. Full replacement populations cover both changes.
The GPU-qualified campaign, materialization and measurement sources are unchanged
at the final implementation head. Later report additions change documentation only.

The CPU environment deliberately hides GPUs. Its skip records are retained in
`cpu-skips.json`; each record includes the actual node and reason. They include
device/dependency checks, private Stage B scratch, unavailable checkpoints and
separate task-suite cases. The declared-data and explicit task-suite lanes above
supply their own evidence. Five unchanged fleet-data files remain outside this
hermetic run. All 465 surviving changed test files have
selected-file coverage across the recorded lanes. A selected file does not mean
every skipped case in it executed.

Of the focused lane's ten CPU skips, six real encoder cases and two profiler
cases passed separately on the admitted GB10 worker. One deliberate pre-seal
API-unavailable control remains skipped. The full CLI batch/resume parity case
requires the frozen x86 producer interpreter and hashes in
`tests/projection_producer_environment.json`; that exact producer environment
cannot execute in the ARM GB10 container. It remains unqualified. No fixture,
hash or admission requirement was weakened to make it run.

GPU actions used the known image content
`ac918444aa2d16d1d19771510963d138981fdd468db44a2bf10f9d627b3af12a`
on sparklina with CUDA 13 and Torch 2.13.0+cu130. The exact delivered wheel was
verified and unpacked into an ephemeral overlay, preserving the image's CUDA
Torch installation. The six cases exercise actual wire encoding, scalar/batch
prices, weights-only behavior and bounded memo byte/price parity. The profiler
cases exercise CUDA event attribution and the timed window with the existing
Netdata/Python monitors. They establish behavior, not speed, work per joule,
served KL or new cell admission. No full model campaign or serving promotion ran.
The first observer action selected the ordinary CUDA-only argument guard beside
the native batch case. The real timed-window node was subsequently run in its
own admitted action. Its source is `test_native_timed_cuda_window_excludes_later_work`;
the earlier argument guard is not counted as native GPU evidence.

The initial source failures and disk failures remain recorded. The composed
checks caught duplicate memo scope, batch structure attribution, split clocks,
callback mutation of profiler identity, incomplete anchor fixtures and premature
proof-driver import. Each was fixed at its owner or stale caller. The subsequent
clean focused run passed all 744 cases. A local root-filesystem exhaustion caused
ENOSPC in an earlier retry. Only this task's inactive historical scratch capsules
were relocated to the shared evidence mount, preserving their original paths as
symlinks. `scratch-relocations.jsonl` records that repair. `t3-fixed.json` and
`compile.log` retain the disk-error observations. A retried action can replace its
canonical terminal record, so the captured observations are the historical record.

All execution used the published PrismaBuild client at
`/mnt/shared/prismabuild-fleet/repo/tools/`. Main CPU shards reserved aggregate
CPU4/RAM16 GiB, four pytest workers and one native thread each. The final guards
reserved CPU1/RAM4 GiB and compilation CPU1/RAM3 GiB. Work used portable placement,
preserved PB affinity and self-validation priority -10. The exact interpreter path
and container content were declared dependencies. `pbtest` owned partitioning.

The rerunnable evidence directory is
`/mnt/shared/prismaquant-refactor-evidence/20261016/final/`.
`reconcile_final.py` checks the frozen selection, complete case populations,
replacement mappings and changed-file coverage. `verify_pb_receipts.py` checks
actual successful terminal records, CAS action identity, result hash/size,
embedded receipt identity and both captured log hashes/sizes. All
28 accepted action receipts passed those checks. The machine-readable
`verification-summary.json`, `verified-receipts.json`, `final-composed-source.json`,
`commands.json`, original/replacement result files and per-lane skip records are
retained there. Raw overlapping counts remain separate. The original broad result
is 20,249,692 bytes with SHA-256
`30867a349a97e7dc8a89e58f53b703cc5b101372a6ea9a4ae9cdbaf78fec7e33`.

The API producer's selected integration still has its two known baseline
failures and CPU-only delivery evidence. Later Tessera architecture source
`53bcd17e3a88c93237465bd7be740f98cd2abc47`, archival head `c70188a791`, v69 contract
and review status remain separate. This consumer refactor does not adopt that
contract or establish a producer-wide pass.

Three encountered corrections are separate commits: the PR template accepts an
open parent reference, the pin docstring distinguishes contract binding from
installed Git provenance, and the provisioner reads contract resources without
initializing serving. The accepted implementation removes a net
21,693 live Python lines across `prismaquant`,
`tools` and `experiments` relative to the recorded main base. This is a source-size
observation, not a performance result. Historical artifacts and dated measurements
retain their original provenance. Incomplete native recipes require re-rendering.

The committed [qualification manifest](../results/prismaquant-refactor-qualification-2026-10-16.json) records the accepted actions, result digests, replacement revisions and limits.


The first publication, PR #2695, was refused by the trusted-base issue policy
because `sol/refactor-simplify-20261014` does not follow its required branch
spelling. Delivery uses `prismaquant-1295-refactor` from the same qualified
implementation. The original authoring branch is retained. This correction
changes report text and branch metadata only; no policy or source gate is relaxed.


## CI dependency publication, 2026-10-16 14:42 UTC

GitHub import CI initially stopped before imports because its checkout could not
fetch the delivered API commit `aaa417003f0ee690e8ace138acd18df7f9531369`.
The exact commit is now reachable in `RobTand/tessera` through
`prismaquant-1295-api-aaa417`. Publication used an isolated bare repository built
from the delivered API branch. The remote branch and GitHub commit API both
return the exact SHA, and the source contract retains its pinned digest. No
source file, existing producer branch, main branch or architecture-wave ref
changed. No Tessera PR or merge was performed.

An additional admitted import probe checks installed Git provenance, imports
`prismaquant`, and runs `python -m prismaquant.allocator --help`. It passed under
the qualified CPU prefix. Action
`8085ff7d35eac1908639cac215c9e914a2bb49dadb5ed5db693c539dde9cc08d`
has result `3c713fd9b7e9606d55665323caea10e61f82ffbe18fc2d206e7dd67f4e1820f2`
and receipt `ae0200462af2e52bcf08c1580739eb6dba669262f0ecda76c0580ee7bd58c119`.
The terminal, result bytes, receipt identity and both logs were verified. This
is one additional probe beyond the frozen 28-action qualification matrix, not
another broad-suite execution or a replacement for GitHub's Python 3.12 import
job. It reserves CPU1/RAM4 GiB and native threads1 through PrismaBuild.

Publication records, the original CI fetch failure, probe command and verified
receipt are archived in
`/mnt/shared/prismaquant-refactor-evidence/20261016/ci-followup/`.


## Claude checkpoint and slot corrections, 2026-10-16

Claude's custody worktree at `3c5bc40ae7` fixed STD-011/R1S-F02,
STD-015 and STD-018. Parent integration is `bad08c3e6a` through `d5260967e9`.
The 19 moved verifier definitions were independently checked against their
original bytes. The child CPU run completed 2,091 cases over 67 files,
with 1,951 passes, 139 skips and one existing expected failure. Its three
successful action keys and compile action, terminal records, CAS results,
receipts and both logs were independently verified in
`/home/rob/tmp/pq-claude-review-20261016/fix-custody/parent-verified-receipts.json`.
The original 19 failed regressions and intermediate failed checks remain in
that directory's report and evidence log; they are not counted as passing runs.

The parent also repaired the registered route-sweep config binding, malformed
falsy no-spec replay for gold and uniform-control arms, and the paired boundary
producer's missing slot-policy source owner. The explicit existing evidence
contracts govern these repairs; they add no format or serving qualification.
The initial seven-case run failed all seven tests. Five expose accepted
malformed evidence or missing artifact context; two fail because the new
source-fingerprint helper is absent on the old code, so those two are API red
checks rather than a recorded pre-fix endpoint measurement. Four additional
uniform-control cases demonstrate the same malformed evidence acceptance.
Their actual failed terminals and log hashes are independently verified in
`custody-red-terminal-proofs.json` under the same review root.

At exact source `5d51ccd6f42fcb9cc375f55c79cae4eb0cd2d37f`, the composed
67-file CPU run completes all 2,102 collected cases: 1,961 pass, one fails,
139 skip and one has the existing expected failure. The only failure is
`test_primitive_digest_sites_only_shrink`, because the removed inline boundary
hash call remains in the historical duplication inventory. Commit `e80396a921`
removes exactly that obsolete entry; a complete inventory-file rerun is being
qualified separately. The composed run remains recorded as failed. The new
behavioral regressions pass, and compilation passes at that source.
Two successful shards and compile are independently verified in
`custody-composed-receipts.json`; the failed shard's terminal and both logs
are verified in `custody-composed-failed-terminal-proof.json`.
Commands, immutable checkout paths and raw results are in `checkpoint.json`,
`custody-followup-{red,green}.json`, `custody-control-red.json` and their logs
under `/home/rob/tmp/pq-claude-review-20261016/`. These overlapping child and
parent runs are not added into a unique total. The full combined Claude
re-review and final official branch qualification remain pending.


The corrected inventory file passes all seven cases, without skips, at
`e80396a921fec664a4ff8b9642f23e2547d62bc5`. Action
`9e9cc8ccea3368eea08c76e2da068dd72452a8a5cd6c23138fa07748255f9840`
executes 0; its terminal, result, receipt and both logs are independently
verified in `custody-baseline-receipt.json`. This qualifies the complete failed
file after its one-entry inventory correction. It does not relabel the original
67-file run or qualify later source changes. All 67 selected file/case populations
now have passing evidence across those named revisions; final branch checks
remain pending.


### Claude review fixes and deeper test review, 2026-10-16

Custody and Tessera fixes are integrated. The three committed native-render
fixes were recovered from the child checkout when Claude reached its session
limit. Depth review A completed all 162 assigned entries. B read 8 of 163
entries and C read none before the same limit, so those reviews are incomplete.
The PR remains draft; the prior official green checks at `5af6092384` do not
qualify the current fixes.

The parent independently audited the worker terminal, result, CAS receipt and
actual log hashes. Evidence root:
`/home/rob/tmp/pq-claude-review-20261016/`.

| Source and scope | Actual result | Evidence |
| --- | --- | --- |
| `7ef4171c50`, 105 affected CPU files | 2,133 passed, 16 skipped, 59 passing subtests, 2 failures in new test changes | `composed-depth-a-tests.json` |
| `a4cdcbc477`, 33 complete corrected/loaded-campaign files | 686 passed, 6 skipped | `depth-a-corrections-tests.json` |
| `7ef4171c50`, package/tools/experiments/tests compile | exit 0 | action `b00f826ca1e7` |
| Base `138069b799`, enhanced allocator oracle under its fca4c6ce Git environment | 2 passed, both original layer digests retained | action `8c0ccd84fba1` |
| Current allocator oracle, in the composed CPU run | complete file passed with the same historical digests | action `9c234bb38bcb` |
| Two deliberate oracle mutations | all 9 selected cases fail as intended | action `4ab09074fdbc` |
| `6da8790036`, known CUDA image on GB10 | 89 pytest cases plus resident/streamed and inline-expert fresh-process bodies passed; no skips | action `4e7fa5eb2774` |

The CPU failures were an invalid class fixture and a stale test helper reference.
Both were corrected and their complete affected modules passed in the 33-file
follow-up. The original composed run remains failed; overlapping populations
are not added together. The follow-up also qualifies the loaded runner's removal
of 19 stable scope aliases. Three genuinely rebound temporaries remain local.
`loaded-scope-alias-proof.json` records the normalized AST comparison.

The allocator checks restore the full historical stock `layer_config.json` and
canonical Tessera `layer.json` hashes, plus all companion file hashes and the
newer field-level assertions. The inputs explicitly clear the optional answer
pin. The two revisions use their respective required installed Tessera Git pins.
An initial base submission correctly refused a mismatched installed pin before
pytest. Two subsequent attempts stopped at an incorrect new assertion that an
unset request was `None`; its actual value is an empty string. Those failed
records are retained. No historical digest was changed.

The GPU image lacked GGUF for two unchanged imatrix consumers in the worker's
first expanded run. The parent overlaid the pinned `gguf==0.19.0` wheel into a
temporary directory inside the admitted action, alongside the unchanged
aaa417 Tessera API wheel. The GGUF wheel SHA-256 is
`70bcd10edfe697fb2dad6e40af2234b9d8ece9a41a99761405121ebda1c3c1cd`.
No active installation changed. The final GPU action passed all 89 selected
cases and both fresh-process bodies. The existing body adapters replace only
the two tests' host-specific temporary roots; their actual assertions run.
The old resident/streamed body fails before the native fix in action
`a8fca7c6d67f`; the post-fix bodies pass. This is tiny-fixture correctness
evidence, with no serving or performance promotion and no full CLI batch/resume
qualification.

`depth-a-parent-receipts.json` audits the 10 successful parent actions.
`depth-a-parent-failed-terminal-proofs.json` audits seven failed records, including
the intended mutation failures and the new cache regression below. The original
native worker qualification has nine separately audited successful receipts in
`fix-native/parent-verified-receipts.json`. Initial node-selection and external
helper-path submissions were refused before publication; their diagnostics are
retained and corrected submissions used supported selection and sealed helper
bytes.

A further cache correctness gap remains open at this checkpoint. Existing
streamed JSO caches record the intended recipe but do not prove the fused global
actually used it. The exporter currently accepts such an ambiguous legacy cache.
The new test at `fa4646a631` reproduces the missing refusal through PrismaBuild: 6
cases pass and the new case fails with `DID NOT RAISE` in action `09f6e00e98cd`.
Producer and consumer need a shared witness or equivalent parity proof before
this branch can pass its final gate. The PR is not release-ready.


### Native legacy-cache fix qualification, 2026-10-16

The gap above is repaired by `83c68fe5dd` and `5ca499e660`. The latter's tree is
`670d16f26671d2b35d9400ac13053f14610bf712`. Each fused NVFP4 entry now carries
the rendering producer's joint-global record. Missing or differing records
refuse at export, including after resident resume, MTP append and union. Export
resume invalidates payloads produced before this check. The original regression
now removes that actual record and checks refusal for all five fused entries.

| PrismaBuild scope at `5ca499e660` | Result | Action |
| --- | --- | --- |
| 69 CPU files, one shard | 1,365 passed, 12 CUDA-only skips, 59 passing subtests | `09c8e8eb52b900012ac844b9e1d97e9584b8d3e4450ab74cf3338cd77644bc03` |
| Known CUDA image on GB10 | 100 pytest cases passed, zero skips, plus both fresh-process parity bodies | `775fbcff27e00bcf08492589dc740fc637926e0a03ac16cd5db22aa4e67db920` |
| Five deliberate gate/resume/append/union mutations | Exactly 10 intended failures, 64 passing controls | `70d46acbcabc19d965b0e0e765fb3d293c190fed9e2ecc7f915cae37772ad892` |

Evidence is under
`/home/rob/tmp/pq-claude-review-20261016/fix-native-legacy-r2/`.
`pb-evidence.json` binds the sealed snapshots to the tested tree and records
every attempt. The parent independently checked three successful terminal/CAS
receipts, result bytes and both logs in `parent-receipt-audit.json`. A separate
`parent-terminal-and-log-audit.json` verifies all 12 terminal records, including
the original regression and mutation failures. Overlapping runs are not added
into a unique total.

The first eight-shard attempt failed when the worker root filesystem filled;
seven shards never reached pytest, and a T3 restart cancelled its client. The
complete single-shard resubmission above passed. The failed records remain.
The worker also disclosed one local `py_compile` invocation contrary to the
mandatory PrismaBuild routing policy. That invocation is excluded from
acceptance evidence. Final compile checks must run through PrismaBuild.

An independent Claude review of this fix is running. Depth B has now completed
all 163 patches and its ten P3 findings are assigned for correction; depth C is
still in progress. The final combined review and official final-head checks
remain pending. These small-fixture checks do not qualify full CLI batch/resume
parity, serving, model quality or performance.
