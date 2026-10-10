# GLM-5.3 MTP layer 45: original capture and price qualification (PQ #2530)

Part of #1271. This is revision 3. It replaces revisions 1 and 2.

## Verdict

The original capture and the original Stage A and Stage B prices of layer 45 qualify for all 867 units.
The audit passes 181 of 181 checks. No check fails.
Four limits stay open. Section "Limits" lists them.
This report makes no performance claim.

Revisions 1 and 2 cited an audit script that was not in the repository.
This revision commits the script, its test and the complete record.
Every number in this report comes from a committed file.

## Scope

The layer holds 867 units.
864 are routed expert units: 288 experts times the down, gate and up projections.
3 are shared dense units: the shared expert down, gate and up projections.
Stage A is the Tessera campaign (workspace `m3`). Stage B is the joint AURA preparation and run (`m4`).
Each Stage B part prices two Tessera rungs for every unit. The r1024 part prices the R1024 rungs. The r896 part prices the R896 rungs.
The merged price (`m6`) joins the two parts.

This task changes no code owner, budget, default, menu, runtime pin or gate.
It changes no existing evidence file on the fleet mount. The audit wrote new files under `qualification-pq2530/` there.
It adds one audit tool, one test, this report, a manifest and the record files.

How to check this report:

- `tests/test_pq2530_mtp_layer45_record.py` checks the committed record offline. It needs no fleet mount.
- `tools/audit_glm_mtp_layer45.py` re-derives the record from the original files on the fleet mount.
- `python3 tools/audit_glm_mtp_layer45.py --verify-actions-only --extra-action audit=dc971e51bea2eaa6a43d099e1fd1db6fdcef8adad9ef6a432758b3465df0952b` re-checks the PrismaBuild chain of the audit action. It needs no torch.

## What this qualification ran

The accepted evidence passes the audit, so no capture, campaign, preparation or Stage B job needed a new run.
Only the audit ran. It ran on CPU as PrismaBuild action `dc971e51bea2eaa6a43d099e1fd1db6fdcef8adad9ef6a432758b3465df0952b`.
No GPU job ran for this qualification. No evidence is missing, so the audit lists no command for a new run.

| Planned job | Result |
| --- | --- |
| Audit capture, M3/M4 prices, prepared caches, wire blobs and receipts | Run. Action `dc971e51bea2`. 181 checks pass. |
| `tessera_joint_aura identity` (source digest cache) | Not needed. The identity file `4db808a769eb` exists and both Stage B parts bind it. The live shards match their recorded stat fingerprints. |
| `glm_mtp_capture --phase projection` | Not needed. Action `89369de5afb2` holds the 864 routed units. The audit checks it. |
| `glm_mtp_capture --phase final-hidden` | Not needed. Action `83dc1e8371bb` holds 512 layer-44 records. The audit checks them. |
| `glm_mtp_capture --phase capture` | Not needed. Action `7c78d20519be` holds 867 entries. The audit re-reads all of them. |
| `tessera_campaign` scoped to `mtp` | Not needed. Six campaign rows ran, three per rate. The audit checks every row. |
| `tessera_joint_aura prepare` | Not needed. Actions `c3dc42c2ae32` and `64969134bf06` ran. The audit checks both caches. |
| `tessera_joint_aura run` | Not needed. Actions `6d839e49e7c3` and `b91a4554c003` ran. The audit checks both prices. |
| Merge, receipt join and body refusal | Run on CPU inside the audit. The current merge owner reproduces the original merged prices. The receipt join binds every routed cell. |

Historical evidence counts only because it passed every coverage and identity check in this report.

## Criterion 1: the capture and price rosters hold exactly the 867 units

The authenticated census (`577bf157ad623aee`) lists 867 units.
Routed counts run from 722 to 36,716. Each shared unit has count 261,632.
Every unit has a positive count and a finite positive maximum. The maxima run from 3.09375 to 59.25.
The geometry is uniform. 288 down projections are 4096 by 2048. 288 gate and 288 up projections are 2048 by 4096.
The producer projection covers exactly the 864 routed units. The shared units have no projection entry, by design.

The census derives from authenticated source.
The base census (`b63f7bf6c4320714`) and the canonical capture manifest (`f4bcbf408d3aa81b`) match their recorded digests.
The capture read 5 shards and hashed 26,811,252,024 payload bytes. Those shard digests equal the producer seal.

The audit re-read all 867 capture entries through the capture owner.
For each entry it checked the Hessian, the activations, the count and the maximum against the census.
The entries hold 49,700,004,815 bytes.
`units.csv` lists every unit with its geometry, count, maximum, capture digest, Hessian shape and activation shape.

| Roster | Scope | Units | Census | Equal |
| --- | --- | --- | --- | --- |
| `capture_entries` | census | 867 | 867 | yes |
| `m3_r1024_costs` | census | 867 | 867 | yes |
| `m3_r1024_expert_wires` | routed | 864 | 864 | yes |
| `m4_r1024_price` | census | 867 | 867 | yes |
| `m3_r896_costs` | census | 867 | 867 | yes |
| `m3_r896_expert_wires` | routed | 864 | 864 | yes |
| `m4_r896_price` | census | 867 | 867 | yes |
| `merged_price` | census | 867 | 867 | yes |
| `merged_wire_bytes` | census | 867 | 867 | yes |
| `merged_params` | census | 867 | 867 | yes |

Scope `census` means all 867 units. Scope `routed` means the 864 routed units, because the shared units have no expert wire.
Every roster equals the census or its named subset. Every census unit is present. No other unit appears.

## Criterion 2: every offered rung has original Stage A and Stage B evidence

The merged price holds 3,468 priced cells: 867 units times four rungs.
Every cell has:

- a measured Stage A price and a Stage B price;
- a prepared render and a wire blob;
- a serialized byte count and a producer receipt.

`cells.csv` lists all 3,468 cells with these columns.

| Rung | Class | Priced | Offered now | Offered M6 | Expert receipts | Checkpoint receipts | Wire bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `TESSERA_BF16_K1_R1024` | routed | 864 | 864 | 864 | 864 | 864 | 3,657,566,880 |
| `TESSERA_BF16_K1_R1024` | shared | 3 | 3 | 3 | 0 | 3 | 12,699,885 |
| `TESSERA_E4M3_K1_R1024` | routed | 864 | 864 | 0 | 864 | 864 | 3,643,407,648 |
| `TESSERA_E4M3_K1_R1024` | shared | 3 | 3 | 3 | 0 | 3 | 12,650,721 |
| `TESSERA_BF16_K1_R896` | routed | 864 | 0 | 0 | 864 | 864 | 3,204,571,392 |
| `TESSERA_BF16_K1_R896` | shared | 3 | 3 | 0 | 0 | 3 | 11,126,984 |
| `TESSERA_E4M3_K1_R896` | routed | 864 | 864 | 864 | 864 | 864 | 3,190,412,160 |
| `TESSERA_E4M3_K1_R896` | shared | 3 | 3 | 0 | 0 | 3 | 11,077,820 |

Serialized bytes per rung over all 867 units:
`TESSERA_BF16_K1_R1024` 3,670,266,765, `TESSERA_E4M3_K1_R1024` 3,656,058,369, `TESSERA_BF16_K1_R896` 3,215,698,376, `TESSERA_E4M3_K1_R896` 3,201,489,980.

The count of 3,468 cells and the count of 3,456 expert wires differ by 12 cells.
3,456 is 864 routed units times four rungs. The Stage A cost carries an expert wire receipt for each of them.
The other 12 cells are 3 shared units times four rungs.
Stage A records their wires in the unit checkpoint, not in the expert wire table.
The audit checks these 12 cells like all others. Each wire passes the producer verifier. Each wire equals the prepared evidence. Each has the priced byte count.
They are not passthrough rungs. BF16 passthrough has no priced cell here.

Every cell passes four checks.
The pinned Tessera producer verifies the wire blob.
The wire digest equals the Stage A receipt and the prepared evidence.
The rendered shard digest equals the prepared evidence.
The file size equals the serialized byte count.
The audit re-hashed 3,468 wire blobs (13,743,513,490 bytes) and 3,468 rendered shards (58,192,191,178 bytes).

Two rules define "offered".
The repo-pinned Tessera contract (`ee065629b081`) offers 2,604 cells.
The menu that the M6 selection recorded offers 1,734 cells.
Every offered cell under either rule has all the evidence above.
The current contract does not attest `TESSERA_BF16_K1_R896` for the 864 routed units. The price exists anyway.
The M6 selection record (`40941983dc24`) names the audited merged price, both Stage B parts, the probe identity and the MTP objective.

Merge. The audit merges both Stage B parts again with the current owner (`merge_mtp_costs`).
The costs, wire bytes, parameter counts, source dtypes and groups equal the original merged price (`052adde6bb66ff9b`).
The original merge predates `wire_rungs` in its provenance. The current receipt join refuses it with `MTP M4 part 0 wire roster differs from merged cost`.
The re-merged payload (`8224c7e4154f3e38`) passes the join and binds 3,456 routed cells.
A selection that needs wire receipts must use a payload merged by the current owner.

Prepared caches. Each cache holds 1,734 measured cells: 867 units times two Tessera rungs.
Every cell has render origin `encoded`. Every render compares with an independent render of its wire.
The r1024 production cache digest is `a1af9cda72638ab1f302c6fcb2a1e8d8b881b789f4e732aa0f4e671c10e1b030`.
The r896 production cache digest is `0dcd84463c82a8ebe4bc97bc6c1e589cb3d223dbfcf0f8ec9142f8cbc07eb8b7`.
Both Stage B plans bind the Stage A cost, checkpoint, census, campaign plan and receipts by digest.

## Criterion 3: identities bind across all receipts, and MTP stays apart from body AURA

The table compares each identity value across the receipts that carry it.
"Seen in" counts the receipts and records that name the value. `audit.json` lists them with the full digests.

| Identity | Distinct | Value | Seen in |
| --- | --- | --- | --- |
| `source.identity_file_sha256` | 1 | `4db808a769eb` | 2 |
| `source.model_content_sha256` | 1 | `c45d2d35b452` | 5 |
| `source.model_path` | 1 | `/mnt/shared/models/GLM-5.3-Flash-BF16` | 7 |
| `source.shard_digests_sha256` | 1 | `0bd675c57d6f` | 6 |
| `calibration.artifact_sha256` | 1 | `9cd1fa129f24` | 5 |
| `calibration.calibration_sha256` | 1 | `dd5be8c887a4` | 5 |
| `calibration.fit_ids_sha256` | 1 | `6b6a0c4283de` | 9 |
| `calibration.hessian_capture_sha256` | 1 | `aa90b10ea6c5` | 2 |
| `calibration.text_sha256` | 1 | `aee724fa58bf` | 6 |
| `probe.calibration_sha256` | 1 | `dd5be8c887a4` | 2 |
| `probe.identity_sha256` | 1 | `eab33bfbfc74` | 3 |
| `probe.objective` | 1 | `mtp_head_self_kl` | 2 |
| `probe.producer_source_sha256` | 1 | `13863d663c58` | 2 |
| `producer.encoder_fixture_id` | 1 | `03bbc5b1c56d` | 2 |
| `producer.encoder_source_sha256` | 1 | `da5805bc707c` | 5 |
| `producer.m3_campaign_source_sha256` | 1 | `d17e1480db11` | 2 |
| `producer.prismaquant_source_sha256` | 2 | `13863d663c58`; `f8c794ba2fcb` | 6 |
| `producer.tessera_pin_mount` | 1 | `/mnt/shared/tessera-pins/07bfcc0e9b7da13276938cb722bc7dcd893e6c63` | 13 |
| `runtime.capture_torch` | 1 | `2.13.0+cu130` | 1 |
| `runtime.container_admission_reference` | 1 | `content:sha256:b5cfa805119c` | 7 |
| `runtime.container_content_sha256` | 1 | `d0256efb8329` | 17 |
| `runtime.cuda` | 1 | `13.0` | 4 |
| `runtime.gpu` | 1 | `NVIDIA GB10` | 4 |
| `runtime.projection_backend_binary_sha256` | 1 | `9305c183c521` | 4 |
| `runtime.projection_backend_qualification_sha256` | 1 | `935f83489314` | 4 |
| `runtime.torch` | 1 | `2.13.0+cu130` | 4 |

Headline identities, with full digests:

- Census `577bf157ad623aee11c28570a06f050a18d0cf5786e14f659c17857cfa2c728f`, base census `b63f7bf6c4320714b4ceb38fbd6996e032e0f0c9b82ac2a30a8337d846e358fd`, canonical capture manifest `f4bcbf408d3aa81b04c1fabd1d2d7457176a95dcccdd5ed368800de5c37e277c`, capture manifest `934f151727c6a370e8a9199a8594d17dae49b91e37401edbb9fc3da826b78eb1`.
- Source model `/mnt/shared/models/GLM-5.3-Flash-BF16`: identity file `4db808a769eb4fa4fa3df3c30c9857bd197200555355671e1c0c84ee8faefdfd`, content `c45d2d35b4520d729e1026b926dc1eb0de9d8fc905d46733c7edd322af2009a0`, shard roster `0bd675c57d6faceb57147dcd37f20eb6c3a9cf61a0fbb6d40af8109079afb263`.
- Calibration: token file `9cd1fa129f249abd80d22efaeb8bc7e8b2d3b4252f173a8c6f2b2e496a4f8329`, token bytes `dd5be8c887a48465fcd7df567b8aba876026339785da03f6642c7179413668e1`, text `aee724fa58bfbdeb3fc6803297fb6bab27b203d7c40b39ddef9b9770e5d52fe5`, fit ids `6b6a0c4283de3aae633fd2bf00f74ac80928a8c389c42aa6d5101b765115532e`, Hessian capture `aa90b10ea6c5f3b8b3952870f46f0afa81cf869e50b9f893f89176238f3b92b8`.
- Probe: identity `eab33bfbfc744aa46d7bbb5072ded168804492331fb46c2214cb10ff63703b9e`. Every Stage B row carries it. Both parts and the merged price share it. 4 probes, seed base 7000, token scope all.
- Producer: pinned Tessera tree `07bfcc0e9b7da13276938cb722bc7dcd893e6c63`, encoder source `da5805bc707c041efaeb27c5083fb7809fd8cdc51d501f41103b791270b569cb`, campaign source `d17e1480db118ef68210720eb83e785dea4596e052c201490901011760fcccf2`.
- Runtime: container content `d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026`, torch 2.13.0+cu130, CUDA 13.0, GPU NVIDIA GB10, projection backend qualification `935f83489314402f58d458afe066dd33c5ec0a8d7f11827cea118ab23a7272d1`.

One value splits, by design: `producer.prismaquant_source_sha256`.
The r1024 prepare ran from source tree `f8c794ba2fcb`.
The r1024 run and both r896 passes ran from tree `13863d663c58`.
The audit extracted both trees from their PrismaBuild snapshot bundles and reproduced both stamped digests.
The two trees differ in 4 of 315 files: `cost_streaming.py`, `tessera_calibration_cache.py`, `tessera_joint_aura.py`, `tessera_source_digest_adoption.py`.
These files change only how the source identity proof is built, reused and adopted (PQ #1363). They hold no render, cost or projection code.

Body-price admission refuses the MTP payloads.
`prismaquant/cost_currency.py:308` raises `joint AURA requires matching aura/joint provenance`. The allocator reaches it through `--costs`.
The audit offered four MTP payloads to `require_run_currency`: the r1024 part, the r896 part, the original merged price and the current merged price.
The gate refuses all four with that message.
`tests/test_pq2530_mtp_layer45_record.py::test_body_cost_admission_refuses_an_mtp_payload` runs the same refusal from the repository.

A second guard keeps MTP rows out of body tables: the probe identity wall (`probe_identity_walls_differ`).
`tests/test_glm_mtp_layer.py::test_mtp_rows_cannot_join_body_rows` pins it.
All rows of both parts share one probe identity. That identity names the objective `mtp_head_self_kl` (schema `prismaquant.joint_aura.mtp_objective.v1`, layer 45, 261,632 tokens).
The merged price shares it.

The first guard reads payload provenance, not the row objective.
The audit relabelled the same MTP rows with body-style provenance (`cost_mode` aura, `joint_activation` true). The first guard then answered: **admitted**.
The probe identity wall does not apply here, because the table holds MTP rows only.
Limit 4 records this. The pull request lists it as a non-blocking child.

## Criterion 4: every execution has a verified outcome and CAS receipt

The audit checked 14 accepted actions: the capture chain, six campaign rows, four Stage B passes and one support action.
For each action it read the queue record, the adopted attempt, the stdout log, the CAS payload, the CAS receipt and the sealed request.
It required all of these:

- status `executed` and return code 0;
- a payload that hashes to the receipt result;
- a receipt that hashes to its own digest;
- a request that hashes to the receipt manifest digest.

It also compared the command sealed into each action with the job its role names.

| Role | Action key | Host | Wall s | Peak GiB | Receipt | Payload |
| --- | --- | --- | --- | --- | --- | --- |
| capture-v2 | `7c78d20519be1a13b57daaf9c323ab08c0819c26360d9311c70bf7e995de7005` | sparklina | 922 | 72.1 | `6a15cd931d7e` | `273073faadc0` |
| projection | `89369de5afb2457ff188fff4692661a1f0acdce540e3cda626ec32078168a3c2` | sparklina | 2,139 | 6.0 | `1fc10924e282` | `38eb32f92a23` |
| final-hidden | `83dc1e8371bb8cc1efe1c5262d83858a2396db456cb7e769cb4e498c5a106667` | sparklina | 221 | 11.7 | `624dd662eaf5` | `375a744d5469` |
| m3-r1024-row-0000 | `ef8afe37ea2e17f92e7ad6ac31609d256ade03b224cbc3b586a87eb560136661` | sparklina | 89 | 3.5 | `c21b10462b49` | `8857c243a4c2` |
| m3-r1024-row-0001 | `0fe8686955508c0604712ffe67a214a900f9e19e665527a9b8d1533d1922be25` | sparklina | 3,239 | 30.5 | `c59008dcb96f` | `ad85cc1c9e10` |
| m3-r1024-row-0002 | `65f323e3764d39d6249e7d0269f66555102d1af9e6fed0ad09daeeda044257e3` | sparklina | 59 | 2.3 | `e066a69396a6` | `b38e29a9d332` |
| m3-r896-row-0000 | `b355042a99024c594bba2dbe8975ae48d08966c7bb8e521487aeb76c70d32198` | sparklina | 101 | 2.6 | `72d8b0c7ce6a` | `bc832e2fc08f` |
| m3-r896-row-0001 | `a994cdc5a3e802ffb0dde2a0ead0bf165612054c83819173842b938d5f2821af` | sparky | 4,529 | 30.4 | `fe364c7f6b10` | `ee563bcb8515` |
| m3-r896-row-0002 | `a5ad8cc0fdc0eff441d5760bbf41b35522c4c847b127183361414d734f3f3bf8` | sparklina | 75 | 2.5 | `be3393d690d2` | `322ef7604d93` |
| m4-r1024-prepare | `c3dc42c2ae32773fc516701dbf736b9b72df7e36c5ab1b4323c0bbd5f3db3f4b` | sparklina | 4,672 | 72.1 | `d1b6df5ddc7a` | `7ce0d6437388` |
| m4-r1024-run | `6d839e49e7c3d10e6b31665fbc1f048fd550753648cbd24746afd3d6fa551865` | sparky | 3,891 | 7.6 | `c59c0ce0f441` | `32c0d8bda799` |
| m4-r896-prepare | `64969134bf06a2eebcaeee8963d76acb66b87f3628c22d25f59ce4b3dea416ad` | sparky | 369 | 6.1 | `601bcccac132` | `f3b42f034ffc` |
| m4-r896-run | `b91a4554c0034915c3d133c073502b54602249e8e59e13d91dc124bf492a0898` | sparklina | 3,932 | 6.3 | `237f1657c53d` | `62cd2470e6cd` |
| reselect-owner-join | `370bdce3b3c2b59c19bc6a01fedcae9ee257a9826d4ace3fa9a0b639691d7213` | dl380g10 | 47 | 1.6 | `d4d165eee8b6` | `1235469e4984` |

Wall time and memory peak size future repeats. They are not performance claims.

The audit action itself follows the same chain.
Action `dc971e51bea2eaa6a43d099e1fd1db6fdcef8adad9ef6a432758b3465df0952b` ran on `dl380g10` for 1,198 s with return code 0.
Its CAS payload is `a5b69fbd9f5c846a3f2abdeacd4f0af3ed020bde78fd0a60bf53896d6aeed5f0` (128,600 bytes). This is `docs/measurements/pq-2530-mtp-layer45/audit.json`, byte for byte.
Its CAS receipt is `c80c0c53d67a3bf64d53ed256cf608d00aba11692f012dc21a564f3018e7c549`. The stdout log is `0f658c5b954882eb25ddc0dd22ae29b73096a0125f7f25bfa9b51e9f78d8fb40`. The sealed request manifest is `8434357faf16eac8962f270a725902301c19fb181fddb21d540e074531ae2fb6`.
The action ran from snapshot commit `133768d716ea60677b3b2c21b1a62b49928702dd` (parent `fa74f5bb7ee89530d29317c6aa446d2e05f1e4c6`).
The snapshot closure file `.pbrun-closure.9432ba2158eacec2.json` names head `fa74f5bb7ee89530d29317c6aa446d2e05f1e4c6` and `dirty_sha256` `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. That is the SHA-256 of empty input, so the tree had no uncommitted edit.
`python3 tools/audit_glm_mtp_layer45.py --verify-actions-only --extra-action audit=dc971e51bea2eaa6a43d099e1fd1db6fdcef8adad9ef6a432758b3465df0952b` repeats this check.

The read-only PrismaBuild MCP tool `pb_receipts` gives a second check of the same state. It returns verdict `green` for all 14 accepted actions and for the audit action.

Attempts that are not accepted. The audit records them and never counts them.

| Role | Action key | Queue state | Attempts | CAS receipt |
| --- | --- | --- | --- | --- |
| capture-v1-superseded | `d1e1681bd0711f6801b134d752af4cc9ab77fe292f7c8036238e69b7472708b9` | done+intent | executed rc 0 | yes |
| m4-r896-attempt1-withdrawn | `1f51251ce30ed6bcd8fdb2a8ab6db5c9c69560380cb3415c774a5e3f39fb9aba` | intent | none | no |
| m3-r1024-attempt1-derivative-guard | `4a68abb9fb0515755ddce8622c0aac8bb4d0d52ddf72423fa219d0fe3718f9a4` | failed+intent | failed rc 1 | no |
| m3-r1024-attempt1-derivative-guard | `3640fdaf805e9d8af015840004e3b0b5d6d95bf2ec32175520a2d0ec2f7fb284` | failed+intent | failed rc 1 | no |
| m3-r1024-attempt1-derivative-guard | `edd4019fb805988f119d58e4420a58add29335ee64bc514dc0bca75bb61a72fe` | failed+intent | failed rc 1 | no |
| m3-r1024-attempt2-no-datasets | `9c3225d8ebe23ecb4eeb6b0118ff444089f7da8448b1415b1c8da0d0c10d15ef` | failed+intent | failed rc 1 | no |
| m3-r1024-attempt2-no-datasets | `ce3299a0f32e6ccad4c884fb8bf1ed27756c24ba39841d270ca459df3e46b9e8` | failed+intent | failed rc 1 | no |
| m3-r1024-attempt2-no-datasets | `69901d3fe135c7eda75180f4f7e0cd8b1a77507e388deba4de71b09a8a93cb8b` | failed+intent | failed rc 1 | no |

Earlier audit runs. They used earlier versions of the tool. Their evidence sections equal this record.

| Action key | Tool commit | Note |
| --- | --- | --- |
| `4a811283fc0379f28ee8bca64908424a81fe79dde55810439634f24a406620ba` | none | Revision 1 audit script, 50 checks. The script was never committed. |
| `179ddafaa0f88101f3126a5409e8cea690531b08f3e377955b7d4134088525f4` | none | Revision 2 audit script, 104 checks. The script was never committed. |
| `65932b5915b02958a3955d419761beb009292ac7a0330487bfe3b46f0f6a26b3` | e2905ce1f0 | First committed tool, 178 checks. Every evidence section equals this record. |
| `2c736f7d8a05b44277e106b944c755b65a29ee5326fec71da9c586ffa04c22bc` | da64adf400 | Digest owners in place of raw hashing, 178 checks. Every evidence section equals this record. |
| `1a452539b99fd28719301983fd2dddeef7ad6e60a6df9fc2c80eff606cf9fe96` | 559e401332 | Light import of the digest owner, 178 checks. Every evidence section equals this record. |
| `35029e831cfbc518f660be88d86a0a4e8644f8ea7189c3dd9d53dc7deffb8db6` | da64adf400 | Dry run on 6 units without the seal recompute, 175 checks. It is not a record. |

Performance claims: none. `performance_claims` is empty in the record and in the manifest, and a test enforces it.
Before and after in-process profiles and aligned Netdata series from both GPU boxes apply to a speed, power or residency claim. This report makes none.

## Limits

The record states four limits. They do not change the verdict. The parent issue decides what each one needs.

### Limit 1: dev mode

13 of the 14 accepted actions ran in a container whose sealed spec sets `PRISMAQUANT_DEV_MODE=1`. `reselect-owner-join` has no container.
The four Stage B results carry `dev_uncertified: true`.
Dev mode turns provenance gates into stamps. The audit re-checks what the skipped gates would check.
It does not turn the runs into certified runs. The artifact gate still needs its certificate. That step belongs to the parent issue.

| Suspended gate | [DEV-MODE] lines | Actions | Resolution |
| --- | --- | --- | --- |
| `prepared_plan` | 2 | m4-r1024-run | explained |
| `prepared_implementation` | 2 | m4-r1024-run | explained |
| `checkpoint_seal` | 4 | m4-r1024-prepare, m4-r1024-run, m4-r896-prepare, m4-r896-run | verified_here |
| `projection_shape` | 4 | m4-r1024-run, m4-r896-run | limitation |
| `source_rehash` | 1 | m4-r1024-prepare | verified_here |

`explained` means that the audit shows why the difference exists. `verified_here` means that the audit performs the check that dev mode skipped.
`limitation` means that the check cannot pass on these runs (Limit 2).

### Limit 2: projection shapes

The routed expert shapes (4096, 2048) and (2048, 4096) are outside the packaged qualification of the fused projection kernel.
The original Stage B runs used the reference arithmetic for them. A run with `PRISMAQUANT_DEV_MODE=0` refuses these shapes.
PQ #1175 closed the dev-mode refusal. PQ #1178 stays open for the certified-mode resolution.

### Limit 3: dense wires

The 12 shared-unit cells have verified wires and checkpoint receipts.
The owner receipt join (`enrich_mtp_cost_wires`) binds routed cells only.
A selected dense Tessera rung is refused at selection (`tests/test_glm_mtp_priced_wires_1413.py`).
The M6 selection kept the shared down projection at `BF16`.
It selected `TESSERA_BF16_K1_R1024` for the shared gate and up projections.

### Limit 4: body admission by provenance

The refusal of MTP payloads does not read the row-level MTP objective. See Criterion 3.

## Files

All files are under `docs/measurements/pq-2530-mtp-layer45/` unless noted.

| File | SHA-256 | Bytes |
| --- | --- | --- |
| `audit.json` | `a5b69fbd9f5c846a3f2abdeacd4f0af3ed020bde78fd0a60bf53896d6aeed5f0` | 128,600 |
| `units.csv` | `64ce7611b5d7fbba2c39bd8b68f2a285fd15d1255eab05fb655a34a621807416` | 170,038 |
| `cells.csv` | `c8d30e8f73b2788af8b5813d7f496c35c18e0e79d080720064bbe0dca3c9c6b7` | 864,378 |
| `tools/audit_glm_mtp_layer45.py` | `415db9e3a82bf052966b3835f76ecffbfa195ac69ccfb68fbab9a34d666d59c2` | 100,094 |
| `docs/measurements/pq-2530-mtp-layer45-manifest.json` | this file binds the rows above |  |

`audit.json` also lists every file that the audit parsed. It holds the path, the SHA-256 and the size of each of its 45 inputs.
The audit passes 181 checks. This table counts them by family.

| Check family | Checks |
| --- | --- |
| action | 27 |
| calibration | 2 |
| capture | 5 |
| census | 12 |
| chain | 8 |
| dev_ledger | 1 |
| excluded | 1 |
| final_hidden | 2 |
| identity | 26 |
| io | 3 |
| offered | 4 |
| producer | 1 |
| projection | 3 |
| refusal | 4 |
| roster | 10 |
| selection | 3 |
| source | 1 |
| stage_a | 24 |
| stage_b | 38 |
| trees | 6 |

## Repair history

Revision 1 named the evidence by action key. It committed no audit tool and no record. It carried one wrong digest for the r896 production cache.
Revision 2 added numbers and corrected that digest. It also committed no tool and no record.
The review could not check either revision against the repository.

Revision 3 commits the tool, the test and the record.
The first committed tool broke the repository duplication ratchet (`tests/test_duplication_baseline.py`).
It added raw `hashlib` and sorted-JSON sites, and four helper names that other modules define.
The final tool calls the digest owners, and the ratchet passes.
