# PQ #2459 serving-code qualification evidence (no cell qualifies)

Refs #2459. Part of #1549. This record publishes the immutable
identity, the complete target coverage matrix, the performed
actions with their receipts, the failure, and the excluded scopes.
It qualifies no cell and moves no pin. The live v2 pin, the
constants, the reviewed answer, and the frozen legal-domain state
stay unchanged.

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
qualified cell must record `runtime.tessera_commit` and
`runtime.serving_source_sha256`, with the same digest in every
trace rank. Numeric fixture profiles reside in
`pq2471_fixture_profiles_2026-10-09.json`.

| # | Family | Structure | Regime | q256 rungs | Profiles | (cell, rung, M, NK) scopes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | TESSERA_BF16_K1 | dense | batch | 832, 880, 960, 1024, 1088 | tr3_batch, speed_batch | 140 |
| 2 | TESSERA_BF16_K1 | dense | decode | 832, 880, 960, 1024, 1088 | speed_decode | 60 |
| 3 | TESSERA_BF16_K1 | routed_moe | batch | 1024 | tr3_batch, speed_batch | 7 |
| 4 | TESSERA_BF16_K1 | routed_moe | decode | 1024 | speed_decode | 3 |
| 5 | TESSERA_E4M3_K1 | dense | batch | 832, 960, 1024, 1088 | tr3_batch, speed_batch | 112 |
| 6 | TESSERA_E4M3_K1 | dense | decode | 832, 960, 1024, 1088 | speed_decode | 48 |
| 7 | TESSERA_E4M3_K1 | routed_moe | batch | 896, 928, 1024, 1088 | tr3_batch, speed_batch | 28 |
| 8 | TESSERA_E4M3_K1 | routed_moe | decode | 896, 928, 1024, 1088 | speed_decode | 12 |

Total: 410 scopes. Status of every scope: unqualified. No GPU
qualification action completed for this record.

Dense rank-local (N, K) pairs: `(12288,4096)`, `(2048,4096)`,
`(4096,1024)`, `(4096,6144)`. Routed MoE pair: `(2048,4096)`.
`tr3_batch`: M2048, M2049 (eager scorer, one sequence, max
length and token batch 2049). `speed_batch`: M512, M1024,
M1026, M1537, M2048 (fresh eager replay of L512/L2048/L8192 at
c1/c4, max length 8448, token batch 2048). `speed_decode`:
M1, M2, M4 (fresh eager decode, one token per sequence, up to
four sequences). M2049 is a complete scorer dispatch, not a
separate M1 decode fixture. L8192 does not permit M8192.

## Actions and receipts

All actions run through PrismaBuild at priority 0. CPU actions
use tag `x86` and scratch `/tmp`. No action ran on celestia.

- CPU preflight `cb56877aa9d17b4b3e0b4c31b1c980f78b89e69909f5f70a0f7ab76ef5faaf1d`
  (pass, dl380g10, 143 s): coordinator delivery present;
  live v2 baseline 22 cells and 160 unit routes; legal-domain
  projection E4M3 1793 and BF16 3841 rates with drift verdict
  `frozen pins match the pins the code reads`; protected hashes
  for the pin, the snapshot helper, and the scope test.
  Payload:
  `/mnt/shared/prismabuild-fleet/cas/blobs/76/7652707922700f1567a4189c2a97f48ee4fd26911739618a395fb302376bba0b`.
- D38 CPU dry run `b8d434721cc1e60135085dcb9c5812a96380bf45710659452b718d4650d4c120`
  (pass, dl380g10, 1 s): the GPU census entry point's argument
  parsing, fixture metadata, artifact check (57 config groups),
  and submission manifest against the A8S artifact, TP 2, image
  `5be13705`, and serving commit `9eef9fea`. No CUDA executes.
- GPU census `22758f6d05ce393cd75e05d6f59c1c77bd5f3ddf0f6e156545d281d0661e47ef`
  (FAILED, sparky, return code 2, 2.2 s, within the 30-minute
  bound): TP 2 eager census of the A8S artifact at serving
  commit `9eef9fea` on image `5be13705` through
  `tessera_plugin_served_tp.sh`. The driver refuses before any
  serve: `/home/rob/tessera` is at `1fe0c73bfb` on sparky and
  `a9eb572e` on sparklina. Two trees are two plugins, so the
  receipt could name only one. Neither tree equals the
  qualified serving commit `9eef9fea`. No trace, no log beyond
  the refusal, and no partial output exist to preserve.
- Equivalence suite (pass): `721132fdb4adc7582f1ec7141c7961a91554380036b4b77e5b4736f076cfea3a`
  (20 scope admission cases), `2b9a74337c74b38c0c1d28805fd6e7452b14e8bc6db9c5b2c21af80993124b79`
  (44 serving-code identity cases),
  `23aa36d4884eb9561674fede79407e6aa6489d453c84547c9cd245970928c5cf`
  (11 split-pin drift cases), and `e97fdda6e9848b94ca55a0699d933594274109821d157e6d55ff637dd3b942c2`
  (63 legal-domain cases). The live v2 admission answers, the
  route verdicts on the committed producer traces, and the
  legal-domain projection equal the baseline. Protected file
  hashes match the preflight record.

## Failures and excluded scopes

- The single GPU action fails closed on the two-box tree
  mismatch above. It qualifies nothing. A retry needs the same
  qualified Tessera tree at the same absolute path on both
  boxes before the serve starts.
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
  eager decode qualification. Fresh eager fixtures are still
  required before any decode cell can qualify.

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
