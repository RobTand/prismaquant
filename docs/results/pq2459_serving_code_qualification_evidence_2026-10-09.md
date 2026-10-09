# PQ #2459 qualification record (attempt 4; corrected by attempt 5)

Refs #2459. Part of #1549. This record preserves the failed action history.
Attempt 5 withdraws the four E4M3 qualification claims from `b5fc8a0e0ac4`.
No cell qualifies. The authorized fixture roster remains incomplete.
The failed census has no retained rank traces or observed serving-code identity.
The live v2 pin, constants, reviewed answer, and frozen legal-domain state stay unchanged.
Keep #2459 and activation blocked until every required scope has valid evidence.

## Immutable identity

The Tessera owner publishes the qualified serving code on #2459
(tessera#1095, merged as Tessera PR #1096):

- serving_commit: `9eef9fea6edce32f4e64abf87f0058b11dab2287`.
- producer_commit: `9eef9fea6edce32f4e64abf87f0058b11dab2287`
  (same tree; identity differs by recipe).
- Algorithm: `tessera.package_source.v1`, computed through
  `tessera.serving.source_identity.serving_source_sha256`.
  It digests every source file in the package (`.py` and native
  sources) as relative paths and bytes. Data files, including
  `runtime_contract.json`, are not code.
- Digest: `a9b7bf32563ce874f45956dd4e5ff4b4c43de73459f9aa40dee1977b9b152330`
  over 128 files.
- Packaged contract at that commit: v57, lane schema v11, raw
  SHA-256 `840607e75b212e77aaa889803abf4e66ac7a7ef1815995db8c4a63d82523bbd7`.
- The same commit's `telemetry.py` stamps that digest into every
  route-trace header under identity version 1.
- Scope of that owner packet: construction-only. It proves the
  code identity, not GPU qualification.

The coordinator authorizes GPU qualification for the eight cells
below in writing (decision `dec-1009-061144-b3de`, delivered on
#2459 at 2026-10-09 08:32:02 UTC). The authorization permits
execution. It qualifies nothing. The serve evidence (run
`u4-R1-20261001T0058Z`, Tessera `bb088715`, image `5be13705`,
TP 2 over RoCE, resident) and its limits reside in
`pq2471_serve_evidence_authorization_2026-10-09.md` beside this
record. The historical serve commit differs from both the live
pin and the qualified serving commit. This record asserts no
source equivalence between those commits.

## Live pin (unchanged)

- Schema `prismaquant.tessera_serving_runtime_pin.v2`.
- Commit `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`.
- Contract v60, lane schema v11, raw SHA-256
  `ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e`.
- Producer and serving commits remain equal. The serving-source
  constant remains `None`. Under this v2 pin every serving-code
  check is a skip: cells that name no code still admit, and trace
  headers that carry a digest are not compared against it.

## Target coverage matrix

Common scope for all eight rows: image
`localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`,
vLLM `0.30.1rc1.dev336+gaf5b4857e.d20260929`, Torch `2.13.0+cu130`,
platform `sm_121`, execution eager, residency resident,
TP 2, plugin `tessera`, grade `device_qualified`.
A future qualification packet must record `runtime.tessera_commit`
and `runtime.serving_source_sha256` on each qualified cell.
Every trace rank must carry the same measured source digest.
Numeric profiles reside in `pq2471_fixture_profiles_2026-10-09.json`.

The failed census uses these engine limits: requested prompt 2048 tokens,
maximum model length 2049, one sequence, token batch 2049, and GPU memory fraction 0.5.
It uses a 1 GiB KV cache, `fp8_ds_mla`, and the triton MoE backend.
It disables flashinfer autotune and selects the language model with remote code.
It uses one Ray head and one worker.
These limits do not prove the eager scorer's dispatch geometry.

| # | Family | Structure | Regime | q256 rungs | Profiles | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | TESSERA_BF16_K1 | dense | batch | 832, 880, 960, 1024, 1088 | tr3_batch, speed_batch | unqualified: no BF16 dense module in the artifact |
| 2 | TESSERA_BF16_K1 | dense | decode | 832, 880, 960, 1024, 1088 | speed_decode | unqualified: no BF16 dense module in the artifact |
| 3 | TESSERA_BF16_K1 | routed_moe | batch | 1024 | tr3_batch, speed_batch | unqualified: the sole BF16 module never dispatches |
| 4 | TESSERA_BF16_K1 | routed_moe | decode | 1024 | speed_decode | unqualified: the sole BF16 module never dispatches |
| 5 | TESSERA_E4M3_K1 | dense | batch | 832, 960, 1024, 1088 | tr3_batch, speed_batch | unqualified: M461 is outside the profiles; rank traces are absent |
| 6 | TESSERA_E4M3_K1 | dense | decode | 832, 960, 1024, 1088 | speed_decode | unqualified: rank traces and code identity are absent; M2/M4 are unmeasured |
| 7 | TESSERA_E4M3_K1 | routed_moe | batch | 896, 928, 1024, 1088 | tr3_batch, speed_batch | unqualified: M461 is outside the profiles; rank traces are absent |
| 8 | TESSERA_E4M3_K1 | routed_moe | decode | 896, 928, 1024, 1088 | speed_decode | unqualified: rank traces and code identity are absent; M2/M4 are unmeasured |

The immutable candidate packet names these required runtime values.
The failed census did not observe them:

- `runtime.tessera_commit`: `9eef9fea6edce32f4e64abf87f0058b11dab2287`.
- `runtime.serving_source_sha256`:
  `a9b7bf32563ce874f45956dd4e5ff4b4c43de73459f9aa40dee1977b9b152330`.

Dense rank-local (N, K) pairs: `(12288,4096)`, `(2048,4096)`,
`(4096,1024)`, `(4096,6144)`. Routed MoE pair: `(2048,4096)`.
`tr3_batch`: M2048, M2049 (eager scorer, one sequence, max
length and token batch 2049). `speed_batch`: M512, M1024,
M1026, M1537, M2048 (fresh eager replay of L512/L2048/L8192 at
c1/c4, max length 8448, token batch 2048). `speed_decode`:
M1, M2, M4 (fresh eager decode, one token per sequence, up to
four sequences). M2049 is a complete scorer dispatch, not a
separate M1 decode fixture. L8192 does not permit M8192.

The failed census requests 2048 prompt tokens but encodes only 461 tokens.
The owner tool repeats a fixed text twenty times and truncates its token list.
It does not extend that list to the requested length.
The logs show M461 batch dispatches and M1 decode dispatches.
M461 belongs to neither authorized batch profile.
The logs do not supply the required code identities or raw rank traces.
All authorized cells remain unqualified.
The remaining batch rows and decode M2/M4 remain unmeasured.

## Actions and receipts

All actions run through PrismaBuild at priority 0. CPU actions
use tag `x86` and scratch `/tmp`. GPU members use tags
`sparky` and `sparklina` with the `gang-v1` capability. No
action runs on celestia. Each GPU member respects the
30-minute bound.

CPU entry tests (pass, dl380g10):

- `7007ac67e40e306c36029e7134aedf38fb0288bc03aac1d4d69f415b2b794a4f`
  (11 passed): the entry-point suite before the R1 scope move.
- `8002374e917d1c0a8afdeb4886848d15d824c8066689cda82d4ea394e4b40dec`
  (11 passed): the same suite with the R1 scope assertions.
- `f534296d0129c6dd6c3e446db7f1f54f275ddb6b8f19eb5a141fd843ea34c62b`
  (11 passed): the final suite with the refused-receipt path.

D38 dry runs of the real entry point (pass, dl380g10,
`PRISMAQUANT_DEV_MODE=0`):

- `cdfb1c43eb294cabf99079bbdbd10deb3255aadd94e8feb953b9ccf5f5065ed2`
  (`tools/pq2459_serve_census.py --mode dry-run --profile all`,
  v2 manifest, seals pass).
- `ca37def9a673a445a1ccd154d034f4f5403f6d4384bd89aec4833d4850c6cc96`
  (member launcher syntax plus the same dry run at `tr3_batch`).
- `2185d94d67ba56c873462d5ed5432ea6a09f68630de8081ab55594641461efbc`
  (dry run after the R1 scope move: 0.5, 1 GiB KV, triton).
- `388d5eccc04dd72351ab8b427ec0d0c8f356a1403515ff5d3d66c8198cee5460`
  (final dry run: draft routes off, no speculative config).

Native TP2 gangs (group, head on sparky, worker on sparklina,
8 CPUs and 100 GiB per member, 30-minute bound):

- Gang `9636db9245aa18526be3bda5557a2307`: head
  `f9724f5487b73c60efff418583c842d219e733066c555eef340aa2f76d35d96d`
  (FAILED, return code 125, 1 s): the launcher splices the
  container env into the docker argv as one word, so docker
  refuses with `invalid reference format`. Worker
  `42967fc01655f5ca665fed184157da7e6a44ee8fc889c9d06eeeef45d1c36207`
  withdraws with its gang. Fix: splice each `KEY=VALUE` line
  as its own `-e` pair (commit `fbeda33525`).
- Gang `95882bc3c885c9b913207861e57e53f7`: head
  `04e89accaa263b7d6a726100ab39bfc2f14f8e94826af606b7b19d36fe3390d5`
  (FAILED, return code 1, 611 s): both ranks load 81.61 GiB
  in 532-538 s, then the engine refuses with `No available
  memory for the cache blocks` at GPU memory 0.3. Available
  KV cache reads -49.76 GiB on rank 0 and -49.33 GiB on
  rank 1. History runs 0.5 with a 1 GiB KV cap. Worker
  `5e3ca232832dac8cba6d0463518ae37e4e2f3ff52d3d9b228c58cfd5c6c63a54`
  withdraws with its gang. Fix: adopt the R1 scope
  (commit `ea0d9e6a2e`).
- Gang `7fe9bcac1f9cc0bf81a4a26572e763e1`: head
  `33764c036b84d352636228705cf83060aeb8d439958542f41fe98508ac5d60ce`
  (return code 1, 523 s): the engine loads (81.05 GiB in
  434-437 s), reserves 1 GiB of KV cache, generates one
  token (11.03 s) and eight tokens (14.28 s), and writes
  its receipt. 112 of 113 declared modules serve in both
  phases: 28 dense and 84 routed MoE on
  `TESSERA_FP8:resident` with contract
  `fp8_per_token_dynamic`. Prefill shapes are M461 on all
  five (N, K) pairs; decode shapes are M1 on the same
  pairs. The sole refusal is the layer-45 draft target,
  which the non-speculative serve never dispatches (see
  below). The head exits 1 on that refusal. Its receipt
  stays inside the removed container; only the log
  survives. Worker
  `8364b42701b1b296a09e2b2590f543005c5e4ea8cda27982be7fc34c035c0678`
  withdraws with its gang. Fix: keep a copy of a refused
  receipt beside the output path (commit `0a79654b25`).
  Log SHA-256: stdout
  `51a4dd13c3e78477682615ef66569a34d6fa8936533bdbc133d94fdce0598ece`,
  stderr
  `c864fafe787791a80c535754dd22225ea27cb8f5080228097c8cf8f0a1e34c97`.
- Gang `81ae6752a445eb3cef878d98ff86d103`: head
  `4cb4a5f7b5c742d9a76a0ad4c5e2b068d66f906a09f9664660ae565351d35eb4`
  (return code 1, 527 s): it repeats the same serve. 112
  of 113 modules serve; the same layer-45 refusal exits 1.
  The kept refused-receipt copy lands inside the removed
  container with the receipt; only the log survives again.
  Worker `d8d66f4bb9ea74565463a385256c20ac6bd9052da0971fd631fc11e0313ec480`
  withdraws with its gang.
  Log SHA-256: stdout
  `627e71fddb7d676684ee90dc5d962ac2796317b59930267646589b874c4c3893`,
  stderr
  `c864fafe787791a80c535754dd22225ea27cb8f5080228097c8cf8f0a1e34c97`.

Equivalence suite (pass, dl380g10):

- `9c4dcd27675f13ed9aaf510c31ffab0609b59b083a3a20cbce200246bd406d9f`
  (20 scope admission cases),
  `2d0ba7a8e4ba409b5997cc3e1645062ee62a34aa74aeb4ee81019c47b29aad4b`
  (44 serving-code identity cases),
  `f3e5c983e9968d9ad71c51f6b256f028ea6766883f8aca29aa2c603d4c45c355`
  (11 split-pin drift cases), and
  `9656fed38a89296e92236c9405430db60f8391924846762f532c1a444649e60b`
  (63 legal-domain cases). These are CPU contract tests.
  They do not supply new GPU qualification or a before/after snapshot.
  Attempt 5 records the direct equivalence check below.

Attempt-3 history stays valid and is not repeated here. Its
final action `767f659b` ends with `memory_budget_exceeded`
(return code 137, 346 s), not with a timeout: 65,912,438,784
observed bytes exceed the 64,424,509,440-byte budget. The
attempt-4 gangs size each member at 100 GiB and no member
exceeds its budget.

## Failures and excluded scopes

- The layer-45 draft target
  (`model.language_model.layers.45.mlp.experts`, the sole
  `TESSERA_BF16` declaration) never dispatches under the
  non-speculative serve. The census therefore names it in
  four PROBLEM lines (both ranks, both phases) and exits 1.
  The historical R1 `route/verdict-tr3.json` excludes that target under its `body_scope` label.
  That diagnostic exclusion does not establish missing fixture coverage.
  The authorized artifact has no BF16 dense module.
  Its sole BF16 routed module does not dispatch in these actions.
  No BF16 cell qualifies.
- The logs show M461, not M2048.
  The owner tool's encoded text is shorter than its requested prompt length.
  All authorized batch rows remain unmeasured.
- The retained gang logs prove route activity only.
  Both failed gangs lose their census JSON and raw rank traces.
  Neither gang has a successful CAS receipt.
  Activation must not cite these logs as qualification evidence.
- The candidate serving commit `9eef9fea` names contract v57; the live pin names v60.
  The eight candidate cells have no `tessera_commit` or `serving_source_sha256` fields.
  The live v2 pin performs no code comparison.
  A future packet must bind each cell to its measured source and all rank traces.
  This record moves no pin and performs no GPU re-census.
- Excluded from this authorization: both other live image
  identities (`f8dbe1a0`, `61fc8a89`), the whole
  `TESSERA_E2M1_K2` family, platforms `gfx1151` and `gfx1201`,
  compiled execution, CUDA graph qualification, streamed
  residency, TP 1, TP above 2, MTP and speculative decode,
  synthetic CPU traces, mixed scheduler rows M7/M8/M25, and
  every shape and rung outside the mapped fixture profiles.
  No cell qualifies from CPU fixtures.
- The historical TR3 KL (`0.027885896312391557`) scores the eager scorer only.
  It does not score the graph serve.
  The historical speed traces use decode graphs and supply geometry only.
  Neither those traces nor the failed M1 census qualifies eager decode.

## Attempt 5: local corrections and fixture prerequisite

Attempt 5 fixes local custody and the D32 image seal.
The launcher mounts the output and trace directories into each member container.
The host exports raw census JSON, each member trace, and packaged contract bytes before container removal.
The artifact manifest records absent output explicitly, including failed actions.
The refused census retains its raw bytes.

The envelope reads the actual trace header and hashes the exported contract bytes.
It does not replace an absent source digest with the expected constant.
It separates declared commit identity from observed identity.
An archive-only installed tree has no observed Git commit; the envelope records null.
The envelope qualifies zero cells.
Tessera's telemetry computes the trace digest through its `serving_source_sha256` function.
PrismaQuant imports and vendors no Tessera serving implementation.

The launcher no longer calls Tessera's hard `runtime_image_require` refusal.
Its Python entry point reads Docker `RepoDigests` and applies `seal_check`.
A mismatch refuses in certified mode.
A mismatch stamps `[DEV-MODE]` and proceeds with the observed image in dev mode.
The runtime tests execute the actual launcher path in both modes.
Temporary config and index files replace the tests' external model dependency.
Runtime assertions replace the source-text launcher test.

The authorized A8S config has 57 groups: 56 FP8 groups and one BF16 routed group.
Every declared rung is q256 1024.
The config has no BF16 dense group.
The sole BF16 routed group targets layer 45, which the non-speculative body does not execute.
Thus, the fixture cannot cover the complete authorized family and rung roster.
Changing the output label or driver arguments cannot create the absent tensor bytes.

Tessera must publish an executable fixture roster for the missing BF16 and non-R1024 scopes.
That packet must bind each fixture to immutable producer bytes and its actual wire metadata.
The coordinator must authorize any replacement of the currently fixed A8S artifact.
The source owner must also name an entry point for each permitted M and concurrency scope.
Do not substitute the ordinary census for the TR3 scorer or speed replay.
No new GPU action runs in attempt 5.
No target scope has new GPU qualification.

## Attempt 5: CPU checks and retained evidence

All actions use PrismaBuild at priority 0, tag `x86`, and `/tmp` scratch.
The interpreter is `/home/rob/venvs/pq-pin-fca4c6ce0-pb027103d9/bin/python`.
The first submission used the generic CPU interpreter and failed its dependency guard before pytest.
That action is `bbc83053e6738c9dde724c0442642228c6107b3177bb46266d9f3ecf41d26f2b`.
It supplies no reproduction or test result.

- Reproduction `2b524870c42725126cb8033ad36b33e3e1d56fc51fa57538e35c429acc0c47dc`: three failures before the fix.
  The cases expose lost successful output, lost refused output, and invented identity.
- Entry suite `645ed5bdd08069d00c028225a54fe7dce26b3e617a34adaf3e4e3e363243c715`: 15 passed.
- Final entry suite `3e31d53a20eb149bb81995ce9a14eabd77a8bd2f58712de4215c0d8f9e5a735f`: 15 passed after the mount change.
- Scope admission `a553d23360d201fd3376a7060d7561901181f1d88796e9589fbbc57486c30784`: 20 passed.
- Serving identity `20fa5395cec20be9a03732b2afebfc24360f64d473179e67b1a698daa598118c`: 44 passed.
- Split-pin drift `78d1985a4662c321a5f936304e2306acd67ce917ed29b36a8ecf566af514cce8`: 11 passed.
- Legal domain `d9190ec79b6912e8f1a57d4b9f6e4711c0c2a2d74300e1e7811042afe670c4d2`: 63 passed.
- CPU Docker discovery `9850389f94fd043d7a21963ef7d72544c86fb540d88821b165161be7d2b4e91a`: passed.
- Docker custody and equivalence smoke `3b7e721ba8d0f3f657dc9f8a373f4be46a63839b82c0c123fd0267d207fcb1aa`: passed.

The real Docker smoke exports artifacts from a stopped CPU container and removes that container.
The host can still read the exact raw JSON and trace bytes afterward.
Its synthetic trace records an absent serving-source digest and qualifies zero cells.
The smoke uses `python@sha256:ddb0207ae1f0356c2b724d740769b0c5f5f51cc54a0525178f721825f78fe74c`.
It does not run vLLM, CUDA, or a Tessera serve.
The CPU custody proof does not qualify the GPU runtime.

The six successful test claims and the smoke claim pass `pb_verify_claim` with payload hashing.
The verifier checks receipt binding, payload presence, length, and SHA-256.
It does not verify the worker attestation.
Successful receipt paths use `cas/actions/v3/<key-prefix>/<full-key>.json` under the fleet CAS.

| Action | Local-result claim SHA-256 | Payload SHA-256 |
| --- | --- | --- |
| `645ed5bdd080` | `fdf49838d38c9aaa855a2ef9d3f435884895928da4e8db42a35b02b40f87c01c` | `724470da049546b78a2c83f1b807dc61bad3782b9a75eb7b17afb3ef6652307c` |
| `3e31d53a20eb` | `437dd22721e9ff1c229d0de8151bc1991be5cc08ea38219e5139ff227fa3eeb0` | `416f33bd53b234a70d849bed49dd6d1db0c1b6f21a29fe2e08d7e1be507e6e4d` |
| `a553d23360d2` | `75de8272abe2185cfcd5591d02ac584ccad3d791d8893d967766cd958e66f31b` | `840275d17c445779024b5e57428aeaec560df0d7841959fe2ef0fb2d161c9133` |
| `20fa5395cec2` | `ec4bf3ccbf005cee70034fd3ed72647ef6bee3b70af9bfdbb99ecb9f5a7f0208` | `4291a9f8df3561a90916f298cb8eee6f3d635c0568a105c52d556fafdb926286` |
| `78d1985a4662` | `3ed191bbc37d2f6c170da3ca6d2bacbfc6083608aed3aca957e82960c6e23e65` | `c09aceb16568b0e2f73831a73733fca57663ad05769e68a68393163753d3f546` |
| `d9190ec79b69` | `1a2fafbd6e03cb21f9b98dc0180fefee61846d69eac915906b289abbb251ebc4` | `a2139ab7f4ea3fc25d1e0d90fcb14c77ae9c63316f810dbf5984de22d45c97a7` |
| `3b7e721ba8d0` | `024ef997bec85b7c58e528757d72aac0bf2aef713232de2faddbfc46a39b95b7` | `2729a47d9c05a0963d1d2a99b16823fed27ff644a307ac268ebeffaeda3c182a` |

## Attempt 5: direct equivalence

The smoke compares the protected source bytes with `b5fc8a0e0ac48432d86e74dbc7548798e67e5494`.
It executes that revision's unchanged snapshot and domain modules against the pinned CPU environment.
It then executes the worktree modules and compares their full results.
The pin, constants, reviewed answer, and frozen-domain module hashes remain identical.
The full snapshots and legal projections match:

- 22 cells.
- 160 unit routes.
- Eight route verdicts on the committed trace fixtures.
- E4M3: 1,793 legal rates.
- BF16: 3,841 legal rates.
- The native-qualification projection and pin-drift report also match.

The retained root is `/mnt/shared/tessera-measurements/pq2459-attempt5-cpu-20261009`.
The smoke payload binds these file digests:

- `equivalence.json`: 2,682,508 bytes; SHA-256 `1135a74955896108fb46a18b25c29a5f7aa220965787cd7b63e86170c6a315ad`.
- `artifact-roster.json`: 12,955 bytes; SHA-256 `9bcae18d9b7c9c6f8111fb212b9341dc2591205d118b20e329ff802a0111fdef`.
- `smoke-raw.json`: 141 bytes; SHA-256 `b7183a13ad91462ca806b6a7af206fe62374dfbe88fe43292daa0c724eb618fd`.
- `smoke-trace.json`: 163 bytes; SHA-256 `853507278cd6be15f043e12440dc16138d3be589c427540481d98c3a1ebe8f7c`.
- `summary.json` records the protected file hashes and measured counts.

These are CPU evidence files, not GPU qualification artifacts.
