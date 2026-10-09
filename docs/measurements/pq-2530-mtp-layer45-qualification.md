# GLM-5.3 MTP layer-45 original capture and price qualification — PQ #2530

verdict: The original source and prices qualify for all 867 units.
This report serves part of PQ #1271 before Option B artifact production.
Revision 2 replaces the attempt-1 summary with action keys and numbers.

## Scope

This task publishes measurement only. It changes no code, budget,
default, menu, runtime pin, or gate. It makes no performance claim.
Resource figures below size repeats only. Historical files stay unchanged.

## Criterion 1 — capture and price rosters hold all 867 units

The census holds 867 unit shapes: 864 routed experts and three
shared units. The 864 routed units are 288 experts times down, gate,
and up projections. Counts run from 722 to 36,716, all positive.
All 867 maxima are finite and positive.

The v2 capture manifest is complete under the canonical schema. Its
867 entries equal the census roster exactly. The audit rehashed all
867 unit blobs against the manifest; all digests match. The capture
summary records 867 identity units, 864 routed units, and 864
projection-checked units. Its census digest equals the census file.

The producer projection covers exactly the 864 routed units. Geometry
is uniform: 288 down projections at 4096 by 2048, 288 gate and 288 up
projections at 2048 by 4096. Both M3 campaign costs carry the same
864 units through the producer projection.

Both price parts hold 867 units with two rungs each. The r1024 part
prices TESSERA_BF16_K1_R1024 and TESSERA_E4M3_K1_R1024. The r896 part
prices TESSERA_BF16_K1_R896 and TESSERA_E4M3_K1_R896. Row-level probe
validation passes on both parts: every row is a joint-AURA entry,
each operator names its own unit and rung, each params entry equals
its source shape product, and each row carries the MTP objective.

The merged payload holds 3,468 priced cells. Its costs, wires, params,
and groups equal a fresh merge of the two parts. The merged file
digest is 052adde6bb66ff9b32eb05fd2326ee0aacd7435b812a48c0c7c29841f7d2fbba.
The selection record binds this digest. Groups are 864 routed, two
shared gate-up, and one shared down unit.

## Criterion 2 — every offered rung has original Stage A/B evidence

Both prepared caches are complete with 1,734 measured cells each.
Each covers 867 units times three formats: two Tessera renders and
BF16. Both production cache digests match their prepared records.
The r1024 cache digest is a1af9cda72638ab1f302c6fcb2a1e8d8b881b789f4e732aa0f4e671c10e1b030.
The r896 cache digest is 0dcd84463c82a8ebe4bc97bc6c1e589cb3d223dbfcf0f8ec9142f8cbc07eb8b7.
Both prepare plans bind by file digest. The r1024 run plan binds too.

The M3 campaign census files equal the authenticated census on both
rates. All six M3 rows verify: four executed with return code zero,
two are CAS cache hits with receipts. Both M3 costs carry the campaign
schema and valid wire directories.

The wire join binds 3,456 routed cells to M3 wire receipts with exact
byte counts, and checks 3,456 wire receipts. The 12 shared dense cells
carry prepared evidence and byte counts but no expert wire, by design:
dense units sit outside the expert projection. Serialized byte sums
are 3,670,266,765 for BF16 at r1024, 3,656,058,369 for E4M3 at r1024,
3,215,698,376 for BF16 at r896, and 3,201,489,980 for E4M3 at r896.

## Criterion 3 — identities bind; MTP stays apart from body AURA

The source identity file digest is
4db808a769eb4fa4fa3df3c30c9857bd197200555355671e1c0c84ee8faefdfd.
It equals the prepared record. The capture identity binds the census
digest 577bf157ad623aee11c28570a06f050a18d0cf5786e14f659c17857cfa2c728f.
The calibration digest dd5be8c887a48465fcd7df567b8aba876026339785da03f6642c7179413668e1
appears in the final-hidden manifest, both prepared files, and the M4
qualification manifest, beside calibration artifact 9cd1fa129f249abd80d22efaeb8bc7e8b2d3b4252f173a8c6f2b2e496a4f8329.
Both price parts share probe identity
eab33bfbfc744aa46d7bbb5072ded168804492331fb46c2214cb10ff63703b9e
on objective schema prismaquant.joint_aura.mtp_objective.v1.
The capture container is prismaquant-glm-derivative:causal-exp-v1-20260908
with content digest d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026.

Body admission refuses all three MTP payloads, r1024, r896, and merged.
Each refusal reads exactly "joint AURA requires matching aura/joint
provenance". MTP self-KL never enters the body table.

## Criterion 4 — executions verify; no performance claim stands

All 15 historical actions show executed status with return code zero.
Capture v2 7c78d20519be ran on sparklina for 920 s. Projection
89369de5afb2 ran on sparklina for 2,137 s. Final-hidden 83dc1e8371bb
ran on sparklina for 219 s. Capture v1 d1e1681bd071 ran on sparky
for 823 s; it is superseded and does not qualify. M3 r1024 rows
ef8afe37ea2e, 0fe868695550, and 65f323e3764d ran on sparklina for
87 s, 3,238 s, and 58 s. M3 r896 rows a5ad8cc0fdc0, a994cdc5a3e8,
and b355042a9902 show two CAS cache hits with receipts and one
4,528 s execution on sparky. M4 r1024 runs 6d839e49e7c3 and
c3dc42c2ae32 ran for 3,890 s on sparky and 4,670 s on sparklina.
M4 r896 runs 64969134bf06 and b91a4554c003 ran for 367 s on sparky
and 3,930 s on sparklina. Reselect support 370bdce3b3c2 ran on
dl380g10 for 44 s.

The read-only check reports a terminal summary conflict on the 14
older receipts: their adopted detail uses the old six-field form
while the queue summary carries full detail. Each immutable attempt
record still reads disposition done, status executed, return code
zero, with log hashes. The new audit receipt uses the current format
and verifies clean.

The final audit action 179ddafaa0f88101f3126a5409e8cea690531b08f3e377955b7d4134088525f4
ran on dl380g10 for 260 s, exited zero, and passed 104 of 104 checks.
Its report payload digest is
9203e78d0d8eff4ce8414d577b34966c552f2e74b82026347d6af56f4d3416cb.
Its stdout receipt log is attempts/179ddafaa0f88101f3126a5409e8cea690531b08f3e377955b7d4134088525f4/3e9aa5df7bc8d65fbbb7c5d1e5d3175ba1e002bdbed7ab28dc66a5af443e0053/00000001.stdout.cd1e5a33b2b61c33963395183e0ec6e32fe44398f0a321a2fbf127c89b791a16.log
with digest cd1e5a33b2b61c33963395183e0ec6e32fe44398f0a321a2fbf127c89b791a16.
The log holds the compact report JSON between REPORT_JSON_BEGIN and
REPORT_JSON_END; it parses to the same object as the payload.

Seven owner test shards pass with 112 tests total and zero failures.
Capture shard 358ef3c83e99bf678cd759828a0fd2cae340dd8ea409e0391e8621cac680d8e7
ran 25 tests. Quantum shard 968c0b6d4cee193dffec63005e8fadceb023a08afd5560ceb328ad1502a58c59
ran 16 tests. Selection shard f2ee10d2db5f7071546c4c4394451747e482ddd92345f44bb8accb96c8190a3f
ran 23 tests. Currency shard bbf13a381169566242136967342c9b4a115dc9531858be6cae6bb80f3a8bbfc2
ran 10 tests. Anchored admission shard beb99df4ca2bbce4f4ee0d90a42725be2f858788a9c973fc277532a97717c5c2
ran 10 tests. Merge integrity shard 0f0b06d135070fd8ebf5616ab29e4827631cece0fd303189a722f57a1cdb8884
ran 15 tests. MTP layer shard b51595bd3ea161a78057554394d88ca44814cdddd86d44d3b4b91fe432c5ee60
ran 13 tests.

This report claims no speed, power, or serving result. Elapsed times,
memory peaks, and byte counts above size repeats only. With no
performance claim, before and after in-process profiles and aligned
Netdata from both GPU boxes do not apply.

## Repair history

The attempt-1 manifest carried one wrong digest: the r896 production
cache entry read 0dcd84463c82a8ebe4bc97bc6c1e58963b3d223dbfcf0f8ec9142f8cbc07eb8b7
but the file and its prepared record read
0dcd84463c82a8ebe4bc97bc6c1e589cb3d223dbfcf0f8ec9142f8cbc07eb8b7.
The re-audit found this; this revision corrects it. The attempt-1
audit action 4a811283fc0379f28ee8bca64908424a81fe79dde55810439634f24a406620ba
with 50 checks is superseded by the final audit with 104 checks.
The audit script needed four fixes during revision: the Tessera venv,
the projection stack parse, numeric expert ordering, CAS cache-hit
rows, the per-row objective location, and the wire binding key.
Nothing in the evidence files changed except the one manifest digest.

## Evidence

Manifest: `docs/measurements/pq-2530-mtp-layer45-manifest.json`.
Audit payload: `9203e78d0d8eff4ce8414d577b34966c552f2e74b82026347d6af56f4d3416cb`.
Audit receipt stdout: `cd1e5a33b2b61c33963395183e0ec6e32fe44398f0a321a2fbf127c89b791a16`.
Audit script: retained in the PrismaBuild snapshot under the audit action key.
