# GLM T-8 serve evidence and GPU authorization for PQ #2459

Refs #2459. Refs #2471. This packet records the evidence for the
coordinator decision `dec-1009-061144-b3de`, approved at 2026-10-09 06:14:02 UTC.
The decision authorizes only scopes with verified evidence.
The GPU hold stays until the coordinator records this packet and written authorization on #2459.
A repository merge alone does not release that hold.
Tessera #1095 remains a separate producer-evidence prerequisite.

## Serve evidence

The measured run is window `u4-R1-20261001T0058Z`, arm A8SESHMN.
It executed on 2026-10-01.
The eager TR3 trace spans 01:17:29 to 01:30:02 UTC.
The latency trace spans 01:31:35 to 01:52:03 UTC.
The run serves the A8S artifact:
`/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported`
(config.json sha256
`3f5c2c7381aae1c02d486c645ec6015cd1a60eb41faa5686541a15f523d79898`,
index sha256
`2990e8c051fac609e913d8619a1dea3bfda49fe69bfaf474a8ca802bdf8b9c84`).
It runs Tessera `bb088715f5a114fe0da33bb6f30b3f4e64b8661c` with
`TESSERA_FUSED_E4M3_MMA=e4m3` on image
`localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`
(vLLM `0.30.1rc1.dev336+gaf5b4857e.d20260929`, Torch `2.13.0+cu130`).
Topology is TP 2 over RoCE.
The latency serve uses `max_num_seqs 4`, `max_model_len 8448`, and `max_num_batched_tokens 2048`, with MTP off.
It passes `--compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}'`
and `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (tessera#774).
The control arm is A8SESH, window `u4-A8SESH-20260930T2132Z`, on 2026-09-30.
The control omits `"mode":"NONE"`.
The eager TR3 scorer uses a different profile: `max_num_seqs 1`, `max_model_len 2049`, and `max_num_batched_tokens 2049`.
Its `engine-tr3.json` sets `enforce_eager=true` and disables chunked prefill and prefix caching.

Measured against the control arm: L8192 c1 prefill 1558.5 to 1657.8
tok/s (+6.4%), TTFT 5256.3 to 4941.6 ms (-6.0%). The full panel is
L512, L2048 and L8192 at c1 and c4. L512 c4 TTFT rose while its
prefill rate rose 18.7%; that cell stays unresolved. The TR3 panel is
unchanged: full-vocabulary KL `0.027885896312391557` over 25 windows
(51,175 positions), with identical domain means in both arms. The TR3
serve is eager in both arms, so the panel shows the scorer did not
move. It does not score the graph serve.

The receipt root is:
`/mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/A8SESHMN-nightly-20260930/run`.
The control root is the sibling `A8SESH-nightly-20260930/run`.
The runtime observation resides at:
`head/tr3/full-vocabulary-kl.runtime-5fbf983a927983c15973e67ae0b9745802322af9d2c7c454be708172b1051d70.json`.
Its `runtime_binding_sha256` is
`5fbf983a927983c15973e67ae0b9745802322af9d2c7c454be708172b1051d70`.
Its raw file SHA-256 is
`3e6741d2bdc484fb5082b0207c43fda4e45b7bed21afd21b971200533e220545`.
The observation records `initialized_before_scoring`; it binds the runtime but does not prove completed scoring.
The result resides in `tr3/kl-summary.json` and `tr3/full-vocabulary-kl.json`.

The route verdict, `route/verdict-tr3.json`, covers 56 exact modules: 14 dense and 42 routed MoE modules, on `TESSERA_FP8` contracts.
It excludes the layer-45 draft target as diagnostic.
The eager traces are `head/route/tr3-rank0.json` and `route/tr3-rank1.json`.
The speed traces are `head/route/latency-rank0.json` and `route/latency-rank1.json`.
Each rank declares `torch.distributed`, world size 2, and `sm_121`.
The speed files are `speed/host-L512.json`, `speed/host-L2048.json`, `speed/host-L8192.json`, and `speed/summary.json`.
Tessera [#774](https://github.com/RobTand/tessera/issues/774) records the historical speed change.

Verified raw SHA-256 identities:

- `route/tr3-rank1.json`:
  `6311a0e76c6ddb32c606773f8a1b2c4ccc3885a6a5ae66859013c18274555252`
- `route/latency-rank1.json`:
  `edbd516fcb0aa1ac92e7ab5ef9c5108c34701872adea8201685038dde26e9174`
- `engine-tr3.json`:
  `85f5f1cbff18eaa9a2db3ec2ce4a9cdd3dd35d3f55e11598ad9b0eddac4afe78`

Limits of this evidence: decode under FULL graphs past `max_model_len`
2048 is not eager-equivalent (tessera#702 cause 2); the release serve
runs 8448. The lane TP 1 reference script cannot serve the GLM-5.3
artifact, so TP 1 has no usable fixture entry point on this artifact.
The MTP drafter under graphs is open (tessera#695). The
measured Tessera commit `bb088715f5a114fe0da33bb6f30b3f4e64b8661c`
predates the live pin below; the recipe evidence binds the image,
flags, topology and artifact class. Each qualification packet records
its own serving commit and code digest through
`tessera.package_source.v1`. The served trace stamps
`serving_source_sha256`
`e0f9b3433a40de2c90a40f288bbe2bb7fc68b1025aa52cd3ff5a2f7fbc585731`
for the `bb088715` tree.

## Live pin (unchanged by this record)

This authorization preserves the live v2 pin and admission behavior.
It moves no pin, constant, reviewed answer, or frozen legal-domain
state. The identities below are historical evidence. A later reviewed
pin move supersedes them without invalidating this record.

- Tessera commit: `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`
- Contract: v60, lane schema v11
- Contract SHA-256:
  `ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e`
- Pin schema: `prismaquant.tessera_serving_runtime_pin.v2`
- Producer and serving commits remain equal. The serving-source
  constant remains `None`. The release label remains advisory.
- The independent export and serve gates stay in force. The
  `cell_evidence_admits` status-only rule stays in force.
- Serving-code checks compare each packet trace header against the
  digest that packet qualifies, per `tessera_route_trace_gate`. This
  record names the measured digest above; it does not gate packets to
  the live pin identity. The Tessera #1095 producer packet names its
  own commit `9eef9fea6edce32f4e64abf87f0058b11dab2287` and digest
  `a9b7bf32563ce874f45956dd4e5ff4b4c43de73459f9aa40dee1977b9b152330`.

## Authorized cell scope

The coordinator authorizes 8 cells. Each is `device_qualified`,
requires plugin `tessera`, runs eager only, and names its permitted
stock vLLM runtime image as `repository@sha256`. Each row gives the
regime, the covered q256 rungs, and the permitted TP sizes. TP 2 rests
on the served TP2 receipt `glm53_a4_stub_tp2_sm121` and the R1 run at
world size 2. TP above 2 is refused by the closed-world ceiling.

Common runtime for all 8 authorized cells:
`localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`,
vLLM `0.30.1rc1.dev336+gaf5b4857e.d20260929`, Torch `2.13.0+cu130`,
execution mode eager, residency resident
(`TESSERA_SERVE_MODE=resident`), platform `sm_121`.

- `tessera_bf16_k1_dense_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | batch | rungs q256 832, 880, 960, 1024, 1088 | TP 2 | tr3_batch, speed_batch
- `tessera_bf16_k1_dense_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | decode | rungs q256 832, 880, 960, 1024, 1088 | TP 2 | speed_decode
- `tessera_bf16_k1_routed_moe_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | batch | rungs q256 1024 | TP 2 | tr3_batch, speed_batch
- `tessera_bf16_k1_routed_moe_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | decode | rungs q256 1024 | TP 2 | speed_decode
- `tessera_e4m3_k1_dense_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | batch | rungs q256 832, 960, 1024, 1088 | TP 2 | tr3_batch, speed_batch
- `tessera_e4m3_k1_dense_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | decode | rungs q256 832, 960, 1024, 1088 | TP 2 | speed_decode
- `tessera_e4m3_k1_routed_moe_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | batch | rungs q256 896, 928, 1024, 1088 | TP 2 | tr3_batch, speed_batch
- `tessera_e4m3_k1_routed_moe_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e`
  | decode | rungs q256 896, 928, 1024, 1088 | TP 2 | speed_decode

Each target cell permits fresh served traces on the named image, at its listed rungs, with TP 2, eager execution, and resident weights.
The numeric fixture profiles reside in `pq2471_fixture_profiles_2026-10-09.json`, beside this packet.
Each profile specifies token rows M and rank-local (N, K) pairs.
Use dense pairs `(12288,4096)`, `(2048,4096)`, `(4096,1024)`, and `(4096,6144)`.
Use routed MoE pair `(2048,4096)`.

The profiles separate the two historical serves:

| Profile | Target regime | Permitted M | Fixture and entry point |
| --- | --- | --- | --- |
| tr3_batch | batch | 2048, 2049 | A8S export; eager TR3 scorer; `engine-tr3.json`; max length and token batch 2049; one sequence |
| speed_batch | batch | 512, 1024, 1026, 1537, 2048 | A8S export; fresh eager replay of the R1 speed client; L512/L2048/L8192; c1/c4; max length 8448; token batch 2048 |
| speed_decode | decode | 1, 2, 4 | Same eager replay; one token per active sequence; up to four sequences; same N and K pairs |

The source `route/latency-rank1.json` records each speed profile's M values with all five structure-specific (N, K) pairs.
The eager scorer's `route/tr3-rank1.json` records M2048 and M2049 with those pairs.
M2049 describes a complete scorer dispatch, not a separate M1 decode fixture.
Prompt length L does not equal dispatch M under chunked prefill.
L8192 therefore does not permit M8192.

The historical speed serve uses decode graphs; its traces do not count CUDA graph replays.
Its M1, M2, and M4 records supply geometry, not eager decode qualification.
Run fresh eager decode fixtures before any decode cell can qualify.
Do not transfer the eager TR3 KL result to the graph serve.
The scheduler also records mixed M7, M8, and M25 dispatches.
Those rows have no isolated prefill or decode classification in this packet and remain outside per-cell qualification.
Record them in the coverage matrix as excluded, not as qualified decode rows.

All eight cells publish grade `route_only` and smoke `not_recorded`.
The unchanged status-only gate admits that evidence.
Those statuses do not prove these new fixture profiles.
TP 1 has no usable A8S fixture entry point and remains excluded.

Behavioral check: `tests/test_pq2471_t8_scope_admission.py` reads the
live pin and packaged contract through `lane_eligibility` and
`tessera_render`. It asserts each authorized cell admits at its stated
rungs and refuses the excluded scopes. Static review reads this
record; no substring test gates its prose.

## Excluded scopes

The authorization covers none of these. Cells with absent or
incompatible evidence stay outside it.

- Image `f8dbe1a0`
  (`localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`):
  no measured serve runs on this image. Excluded cells:
  `tessera_bf16_k1_dense_sm121_batch_resident`,
  `tessera_bf16_k1_dense_sm121_decode_resident`,
  `tessera_bf16_k1_routed_moe_sm121_batch_resident`,
  `tessera_bf16_k1_routed_moe_sm121_decode_resident`,
  `tessera_e2m1_k2_dense_sm121_batch_resident`,
  `tessera_e2m1_k2_dense_sm121_decode_resident`,
  `tessera_e2m1_k2_routed_moe_sm121_batch_resident`,
  `tessera_e2m1_k2_routed_moe_sm121_decode_resident`,
  `tessera_e4m3_k1_dense_sm121_batch_resident`,
  `tessera_e4m3_k1_dense_sm121_decode_resident`,
  `tessera_e4m3_k1_routed_moe_sm121_batch_resident`,
  `tessera_e4m3_k1_routed_moe_sm121_decode_resident`.
  The whole `TESSERA_E2M1_K2` family is therefore outside this
  authorization: no `TESSERA_E2M1_K2` cell names the `5be13705`
  image.
- Vanilla vLLM image
  (`vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`):
  no measured serve runs on this image. Excluded cells:
  `tessera_e4m3_k1_dense_sm121_batch`,
  `tessera_e4m3_k1_dense_sm121_decode`.
- Platforms gfx1151 (Strix Halo) and gfx1201 (RDNA4): no sm_121
  qualification transfers to them. gfx1151 ships no cell. The two
  gfx1201 BF16 dense cells from contract v24 are not in the live
  v60 reviewed answer and are not authorized.
- Execution mode compiled: all live cells run eager only.
- Residency streamed: no authorized cell publishes it. The two
  vanilla dense E4M3 cells publish streamed, but they are excluded
  above for lack of serve evidence.
- TP 1: no usable fixture entry point exists on the A8S artifact.
- TP above 2: refused by the closed-world ceiling.
- CPU fixtures: no cell qualifies from CPU fixtures.
- Tensor shapes outside the served set above: need coordinator
  review before they qualify.
- Synthetic fixtures: `tests/fixtures/tessera_route_trace_union/`
  carries synthesized speculative entries for layer 45 and empty real
  speculative traces (see its `PROVENANCE.md`). Those files support
  CPU unit tests only and supply no qualification evidence for any
  cell. The A8 window run `u4-A8-20260928T1809Z` (2026-09-28, GLM-5.3
  uniform T-8, MTP artifact `a8/body-mtp-v39/exported-r2`) is a
  separate fixture source, not the release serve.
- Families, structures, or rungs outside the live reviewed answer.

## Coordinator authorization

The coordinator authorizes GPU qualification for #2459 in writing,
for the 8 cells named above only, subject to every limit below. This
authorization does not qualify any cell or artifact. It records
permission to run, not a result.

- Execute through PrismaBuild only. Use priority 0.
- Bound each GPU action to 30 minutes or less.
- Require a D38 CPU dry run for every new or changed GPU entry point before its GPU action.
- Use the named stock vLLM runtime with the Tessera plugin. Do not add vLLM core patches.
- Require `runtime.tessera_commit` and
  `runtime.serving_source_sha256` on every target cell the packet
  qualifies. Compute the digest with
  `tessera.serving.source_identity.serving_source_sha256` under
  `tessera.package_source.v1`, not the provisioner checksum.
- Require the same qualified source digest in every trace rank.
- Publish the cell coverage matrix, raw traces, logs, action keys,
  and verified CAS receipts. Record failures and excluded scopes.
- Tessera #1095 remains a separate producer-evidence prerequisite.
  Resume the parent plan only after its applicable prerequisites pass.
- Equivalence: compare the live v2 admission results, route
  verdicts, and legal-domain projection before and after the
  evidence PR. Require identical results.

## CPU verification of this correction

These actions use PrismaBuild, tag `x86`, priority 0, and the pinned interpreter `/home/rob/venvs/pq-pin-fca4c6ce0/bin/python`.
They start no GPU work and qualify no cell.

- Red regression:
  `b20a3e8b26707e19e3d5a0b4b66e5614e673b37a721b07e48274389950365bec`.
  The retained M2048/M2049 restriction fails seven speed and decode cases.
  Three cases pass; ten existing admission cases are deselected.
- Corrected suite:
  `52bde44fc854c333a5990c88b206ca526720083c2c3bfc0509fad7fa85c32841`.
  All 20 cases pass; no case skips.
  The result file is `/home/rob/fleet/ceo/exec/ig-pq-2471-l1-a4/tests.json`.
  The log resides at the action's CAS receipt:
  `/mnt/shared/prismabuild-fleet/cas/actions/v3/52/52bde44fc854c333a5990c88b206ca526720083c2c3bfc0509fad7fa85c32841.json`.
- Initial CPU smoke:
  `f8b8a37349b99c31b2bd3288a6d9a6d878babee711f86c0315ab79fad55d675a`.
  The temporary harness incorrectly expects `backed`.
  The live owner returns `backed_with_serve_flag`, with `TESSERA_SERVE_MODE=resident`.
  This harness failure changes no production gate or test assertion.
- Corrected CPU smoke:
  `12580f3491d15b8744a5d3311cb2ed35cb8cf8c976f37917d1cf4701a7acdf8f`.
  The harness expands each numeric profile and calls the live `resolve_unit_route` owner for both structures and families.
  Each route requires the resident serve flag.
  The harness also compiles the changed test module on the worker.
  It exits successfully and reports `qualified_cells=0`.
  Its payload SHA-256 is `c9924b3974f8583a9e55c731eaf2253f3f313d7627b1f4e6014b6fc8e7690d8a`.
  The temporary harness is absent from the delivered branch.

The parent issue must receive the coordinator record before GPU qualification can start.
