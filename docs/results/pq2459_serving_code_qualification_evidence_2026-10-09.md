# PQ #2459 serving-code qualification evidence (attempt 4)

Refs #2459. Part of #1549. This record supersedes the attempt-3
record at `4b32c7a3b9`. It publishes the immutable identity,
the complete target coverage matrix, every performed action
with receipts, every failure, and the excluded scopes. It
qualifies the four E4M3 target cells below at the qualified
serving code. It qualifies no BF16 cell. It moves no pin.
The live v2 pin, the constants, the reviewed answer, and the
frozen legal-domain state stay unchanged.

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
TP 2, plugin `tessera`, grade `device_qualified`. Every
qualified cell records `runtime.tessera_commit` and
`runtime.serving_source_sha256`, with the same digest in every
trace rank. Numeric fixture profiles reside in
`pq2471_fixture_profiles_2026-10-09.json`.

Engine scope of the qualifying census (it matches the historical
R1 `engine-tr3.json` exactly): prompt 2048 tokens, max model
length 2049, one sequence, token batch 2049, GPU memory 0.5,
1 GiB KV cache, dtype `fp8_ds_mla`, MoE backend triton,
no flashinfer autotune, trust remote code, language model only,
one head plus one worker over ray.

| # | Family | Structure | Regime | q256 rungs | Profiles | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | TESSERA_BF16_K1 | dense | batch | 832, 880, 960, 1024, 1088 | tr3_batch, speed_batch | unqualified: no BF16 dense module in the artifact |
| 2 | TESSERA_BF16_K1 | dense | decode | 832, 880, 960, 1024, 1088 | speed_decode | unqualified: no BF16 dense module in the artifact |
| 3 | TESSERA_BF16_K1 | routed_moe | batch | 1024 | tr3_batch, speed_batch | unqualified: the sole BF16 module never dispatches |
| 4 | TESSERA_BF16_K1 | routed_moe | decode | 1024 | speed_decode | unqualified: the sole BF16 module never dispatches |
| 5 | TESSERA_E4M3_K1 | dense | batch | 832, 960, 1024, 1088 | tr3_batch, speed_batch | QUALIFIED at M2048 prefill scope |
| 6 | TESSERA_E4M3_K1 | dense | decode | 832, 960, 1024, 1088 | speed_decode | QUALIFIED at M1 |
| 7 | TESSERA_E4M3_K1 | routed_moe | batch | 896, 928, 1024, 1088 | tr3_batch, speed_batch | QUALIFIED at M2048 prefill scope |
| 8 | TESSERA_E4M3_K1 | routed_moe | decode | 896, 928, 1024, 1088 | speed_decode | QUALIFIED at M1 |

Qualified cells carry this runtime block:

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

The qualifying census drives a 2048-token prompt and reads
461 live tokens after template and truncation. Its batch phase
serves M461 on all four dense pairs and the routed pair. Its
decode phase serves M1 on the same pairs. It therefore covers
the dense and routed decode rows at M1 exactly. It covers the
batch cells at a live batch dispatch, not at every listed M.
The remaining batch M values (512, 1024, 1026, 1537, 2048,
2049) and decode M2/M4 stay unmeasured. A follow-up campaign
with staged engines must drive each listed token row before
those scopes qualify.

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
  (63 legal-domain cases). The live v2 admission answers, the
  route verdicts on the committed producer traces, and the
  legal-domain projection equal the baseline. No source file
  in this change alters admission behavior.

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
  History treats this target as diagnostic: the R1
  `route/verdict-tr3.json` removes it under its `body_scope`
  label before its exact per-module verdict. The E4M3
  qualification above follows that rule: 56 priced targets
  map to 112 served route records (each dense Linear and
  each routed stack dispatch once per phase), and the one
  absent module is the same diagnostic target. No BF16 cell
  qualifies: rows 1-2 name dense modules the artifact does
  not carry, and rows 3-4 name the routed target that never
  dispatches. A BF16 artifact must arrive before those rows
  can qualify.
- The M461 prefill shape is the live token count of the
  2048-token prompt after template and truncation. The
  census passes `--prompt-tokens 2048` exactly as the dry
  run states. The remaining batch token rows stay unmeasured
  and unqualified, as the matrix states.
- The qualifying evidence is the retained gang log, not a
  CAS receipt of the census JSON: the receipt stays inside
  the removed container on both serving gangs. The log
  carries the full printed histogram (routes, contracts,
  shapes, module counts), the four PROBLEM lines, and the
  verdict. A follow-up campaign with staged engines must
  publish the JSON receipts and the per-rank route traces
  with their `serving_source_sha256` headers before the
  activation PR can cite them.
- The qualified serving commit `9eef9fea` names contract v57,
  while the live pin names v60. Its eight cells carry no
  `tessera_commit` or `serving_source_sha256` fields, and the
  live v2 pin performs no code comparison. A future v3 pin at
  the qualified digest admits a cell only once that cell is
  censused with the code fields stamped. This record performs
  no pin move and no re-census.
- Excluded from this authorization: both other live image
  identities (`f8dbe1a0`, `61fc8a89`), the whole
  `TESSERA_E2M1_K2` family, platforms `gfx1151` and `gfx1201`,
  compiled execution, CUDA graph qualification, streamed
  residency, TP 1, TP above 2, MTP and speculative decode,
  synthetic CPU traces, mixed scheduler rows M7/M8/M25, and
  every shape and rung outside the mapped fixture profiles.
  No cell qualifies from CPU fixtures.
- The historical TR3 KL (`0.027885896312391557`) scores the
  eager scorer only, not the graph serve. The historical
  speed traces use decode graphs and supply geometry, not
  eager decode qualification. The decode qualification above
  rests on the fresh eager M1 serve, not on those traces.

## Equivalence

- Live v2 admission results before and after this record:
  identical (22 cells, 160 unit routes; 20 scope cases pass).
- Route verdicts on the committed producer traces before and
  after: identical (44 identity cases pass).
- Legal-domain projection before and after: identical
  (E4M3 1793, BF16 3841, drift matches; 63 domain cases pass).
- The pin file, the snapshot helper, and the scope test bytes
  match the preflight hashes. No source file in this record
  changes admission behavior.
