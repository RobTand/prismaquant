# GLM-5.3 MTP BF16-grid R1024 fixed selection — 2026-09-26

Rob selected the closest Tessera compute-route analogue to the completed
EXL3 comparison checkpoint: compact Tessera BF16-grid R1024 for the 864 MTP
routed experts, source BF16 for the three shared-expert Linears. This leaves
the exported body allocation unchanged. The earlier E4M3-grid R1024 MTP
selection remains a valid research measurement but is superseded for this
comparison. Neither is a served quality or prefill measurement.

The existing M3 R1024 panel has all 864 original BF16-grid cached wires and
measured cost rows. Its routed wire total is 3,657,566,880 bytes; the three
source BF16 shared Linears add 50,331,648 bytes. The exact measured M6 cost
`m6/layer45/merged-cost.pkl` has SHA-256
`052adde6bb66ff9b32eb05fd2326ee0aacd7435b812a48c0c7c29841f7d2fbba`.
The fixed-group input is
`m6/bf16-fixed-20260926/fixed-formats.json`, SHA-256
`fadeb7af63418145b1b9b266fafa96932a13fd7322282067b9f2363951c65dad`.
The source body config is the exact completed export input, SHA-256
`b0cc6063d49ede76b5e5e0b1b00ca0c76a4f0b857dd108b47e38dbc03e14e657`.
These paths are under
`/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/`.

`tools/reselect_mtp_fixed.py` invokes the existing MTP group-product selector
with those three fixed formats intersected with the pinned runtime's cell
eligibility, then uses the existing bound-cost receipt join. PrismaBuild CPU
action `370bdce3b3c2b59c19bc6a01fedcae9ee257a9826d4ace3fa9a0b639691d7213`
passed (CAS receipt SHA-256
`d4d165eee8b6675aa48f138f19d5bd53f68866c022fff16465263b95963746e4`).
The final metadata-corrected layer config is
`m6/bf16-fixed-20260926/layer_config.routes.json`, SHA-256
`02f7ef17ce848f1590c7fb714d337d739317bbda3f001d47ccd03be59fd6143b`.
It selects 3,707,898,528 bytes and reports MTP-head self-KL proxy
`E=0.0005112509337531043`. The prior 4.068181779 body bpp and exact body
unit choices are unchanged. The original body plus MTP assignment has 37,414
units: 37,173 Tessera routes backed with a serve flag, 241 plain BF16; 36,302
W16A16 and 871 W8A8 activation contracts. This full census replaces an
intermediate body-only route histogram; that intermediate config is retained
as bounded evidence and is not the final handoff.

The new whole-artifact budget stamp uses the 175,642,157,752-byte EXL3
120-shard reference as a **strict all-file ceiling**. Its selected tensor
payload estimate is 175,542,306,328 bytes: the original body's
186,380,254,072-byte estimate minus 14,545,846,272 source BF16 MTP bytes
plus 3,707,898,528 selected MTP bytes. The completed body's observed
all-file inventory was 186,468,866,053 bytes, 88,611,981 above its payload
estimate. The reselect adds an explicit 4 MiB allowance for new metadata,
making the provisional reserve 92,806,285 bytes and upper estimate
175,635,112,613 bytes, 7,045,139 below the strict ceiling. This is a
selection estimate, not a claimed final file inventory; the exporter must
stat every output file and refuse if the actual total exceeds the ceiling.

The BF16 MTP original v1 child has 864 unchanged historical receipts, SHA-256
`5eea17d8aa7aa47094fcf7b514fcca51e65c00a16c17719bad46dbde939ad35e`
(PB `34184c72...`, CAS `62f5b280...`). The unchanged body v1 child plus
that MTP child form a disjoint 37,173-unit Tessera v3 bundle, SHA-256
`f4e18d18b0318ada48847cf7398f964ec7a65a116d49e353fef69b1e27069ee5`
(PB `23ce7b7a...`, CAS `eb228c04...`). The body/MTP Hessian collection is
unchanged, SHA-256 `15bcd34cb73e73d4ea43c4bea80c1cb08724a7e7dc393932bf3ea1b2c4331a43`.
PQ's full actual preflight under Tessera `d2a6455025...` passed PB
`518f529d13de05a46e5fe32e57bd2d0c5f888268fb63df533311ecc95686caf5`
(CAS `d42dc322...`), including the BF16 R1024 runtime cell and all selected
original receipts. Its build SHA is
`f90ac64acb973820303c799554a74550dedb5e75580d70029dc0dec171b9043b`.
Tessera's plan translator wrote the 584-entry plan at
`integration/bf16-fixed-20260926/tessera_plan.json`, SHA-256
`f94981e6296a7263d715ad91bf07e58477a69c39726ffd0e2fa7a9b9b6ec7d11`
(PB `d1898b42...`, CAS `5d10f8f9...`). The optional PrismaQuant charged-bpp
annotations in its provenance sidecar are null because the host-local PQ
worktree was unavailable on the dl380 worker; the plan itself and PQ
preflight have exact source names, rates and coverage.

The pin then moved to Tessera's MTP mapper commit `09d6559d7…`. The refreshed
full preflight at that pin passed PB
`0f2de41f0994757f80f820c56d93acdb4ba7bbb74d3b5c04b2367140a5f3ea93`
(CAS receipt SHA-256 `7532f773e9cd…`) with 37,173 scoped selected units and
the same build SHA, `f90ac64a…`. The pin tests passed in PB `9e1233ec…`.

## Combined export

PB `0e39638c2fc0ca4bc655b1d1f07e6100793373a263a9ede0cd44d5b06e351717` ran
Tessera `09d6559d7…`'s `experiments/export_tessera_serving.py` on CPU. Its
inputs were the plan, the `f90ac64a…` build, the Hessian collection and the
v3 cached-unit bundle above, and it wrote
`/mnt/shared/tessera-runs/moe/glm53-body-mtp-bf16-r1024-20260926/exported`.
The exporter finished: the last line of its `export.log` is the `elapsed` line
naming that output. **The action itself failed** (exit 1, no CAS receipt).
The inline gate that the sealed request runs after the exporter read
`cached_units.hessian_identity.established`. The manifest carries the
by-producer identity schema
(`tessera.cached_unit_hessian_identity.by_producer.v1`), which records
`established` per producer, so the gate raised `KeyError`. That gate was
campaign-local code in the request, not PrismaQuant code.

PB `014fb213d82f2413204aea96fc48fe6f13a82b5e60bdfae90c1ba511d5b1100e`
re-applied the same checks read-only, reading the per-producer schema. It
also **exited 1 with no CAS receipt**. Every check before the last one
passed:

- The exporter log ends with the finished line for this output.
- 37,173 planned units, with a 37,152-unit routed-expert intake.
- Two producers, both `established: committed`, including `0833671b…` and
  `da5805bc…`. Two cached cohorts and two producer packages.
- 120 `model-*-of-00120.safetensors` shards. The index shard roster is
  those 120.
- For each shard, the header extent equals the file size, and every header
  tensor is indexed to the shard that holds it. The header and index tensor
  name sets are equal.
- The index `total_size` equals the measured shard bytes,
  175,576,305,954 B.

The last check, the strict all-file ceiling, refused:
175,643,087,583 B > 175,642,157,752 B, over by **929,831 B**.
`tessera_serving_manifest.json` is written with `indent=2` (42,445,119 B).
The same object in compact JSON is 29,341,610 B, so the whitespace alone is
13,103,509 B (Tessera #635). By Rob's decision the artifact is **not
re-exported** over this miss. It stands as exported, and the ceiling
acceptance criterion in PR #1416 is waived by that decision, not met.

A separate read-only metadata check of the same directory (PB
`30cdff440a69…`, launcher `tools/check-mtp-artifact.py`) found 120 shards,
38,764 indexed tensors and 58 config groups. It also found
`model.language_model.layers.45.mlp.experts` declaring `TESSERA_BF16` at rung
1024, with no refusals.

### Size against the EXL3 reference

The reference is `brandonmusic/GLM-5.3-Flash-tr3-4bpw` at revision
`a5fee929cf4888b1824323e33e8a19b60129e025`. Both sides count every regular
file.

| Accounting | This export | Reference | Difference |
| --- | ---: | ---: | ---: |
| Safetensors shards (120 each) | 175,576,305,954 B | 175,642,157,752 B | 65,851,798 B smaller |
| All files | 175,643,087,583 B | 175,790,111,275 B | 147,023,692 B smaller |

This is a size comparison only. It makes no quality, speed or serving claim.

## TP2 MTP draft census: not run

The planned census was an eager TP2 vLLM serve with a one-step GLM MTP
speculative config. It used image
`localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0…e1b7f5` and Tessera
`09d6559d7`, with rank 0 on sparklina and rank 1 on sparky. It did **not run**:
the derived per-rank footprint does not fit sparky. Nothing was served, and
this document makes no generation, draft-route, acceptance or memory-fit claim
for the artifact.

The derivation uses this artifact's manifest totals and the measured x-picks
routed retention: 72.87 GiB per rank at TP2 from a 153.3 GB wire, which is
51.04% of the wire per rank (PB `ac0ff9cbfe14`).

| Term | Per rank |
| --- | ---: |
| Routed and MTP wire, 156,973,875,200 B × 0.5104 | 74.6 GiB |
| Passthrough, 18,567,773,048 B (sharded … replicated; unmeasured) | 8.6 … 17.3 GiB |
| KV cache (`--kv-cache-memory-bytes 268435456`) | 0.25 GiB |
| Runtime and activation peak (eager, `max-model-len 512`) | ~6.5 GiB |
| **Footprint** | **89.9 … 98.6 GiB** |
| + 16 GiB watchdog floor | 105.9 … 114.6 GiB |
| + 24 GiB launcher READY floor instead | 113.9 … 122.6 GiB |

With the GPU idle and PB drained, sparky's MemAvailable was 105.9 GiB at
2026-09-27 00:20Z, and sparklina's was about 116 GiB. The most optimistic
bound leaves sparky exactly at the 16 GiB watchdog floor, and no bound clears
the launcher's 24 GiB READY floor. A census on the two Sparks needs either a
smaller per-rank footprint or about 8 to 17 GiB more free memory on sparky.

The runtime-plus-activation term and passthrough sharding are estimates, not
measurements. A real TP2 load is still the only per-rank measurement.

## Path expansion correction — 2026-09-27

The relative `m6/...` and `integration/...` evidence paths above are under
`/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/ws-mtp-20260925/`,
not directly under `/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/`.
For example, the bound MTP cost is
`/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/ws-mtp-20260925/m6/layer45/merged-cost.pkl`.
This corrects path expansion only; it changes no historical digest, measurement,
selection, gate or runtime claim.
