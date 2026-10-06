# Menu shard-geometry fixture cost (Refs #1929)

## Finding and bounded change

The retained current full-cohort action c080c71c034e8f64b52dfa2a3f7910ae7884c7f38e7bec745c9fc1019f1fc1a0
attributes 364.069 accumulated phase seconds to tests/test_tessera_menu.py.
Its E4M3 R1024 and R1023 real-encode geometry cases take 81.90 and 45.79
call seconds. That FAILED full cohort is ranking evidence, not qualification
or a matched performance baseline. This slice leaves its frozen94ce source
and all previous accepted packets untouched.

The seven shard-geometry fixtures now explicitly use the existing
scale_refit=0 control. They still call the real public encode_linear_planes
exporter and compare its real unit to the menu's independent geometry answer.
Seed0, all seven original tensors, (64,512) shape, family/rung list, plane and
body expectations, and every shard assertion are unchanged. No node or
assertion is added, removed or renamed. verify=False is unchanged. These
cases check structural geometry, not reconstruction error or quality; refits
change reconstructed values, not the fields the shard calculation consumes.
No production encoder option, pin, menu, numerical default or gate is changed.

This reuses the established forest-byte fixture convention and accepted
causal layout/refit work in docs/measurements/test-forest-bytes-2026-10-02.md
(#2098/#2103), rather than repeating that old proof. Its six paired cases
establish layout agreement, not byte/reconstruction equality and not the exact
seven inputs here. Fresh before/after runs below exercise all seven exact
geometry inputs; both retain real WINDOW/TCQ encoder work. The installed
public export.py AST independently declares scale_refit and forwards it to
encode_linears_planes; no private or newly invented encoder path is used.

## Frozen source, population and actual execution

Base: 736fd56aa6853c3ee06655eda6015b18bef763bd.
Test commit: 318b466d0e27e52ae534c6b04138f8dd32c4d542.
Changed test: tests/test_tessera_menu.py, one argument plus an explanatory comment.
Before test SHA-256: 3e45b80a4a5a49d46d001cc92bc3434d6b0857600fc8311bb2c76d70e55cd53a.
After test SHA-256: d42724e5363fb492a9aad862fb9abc15f0b9cdafe04138b387cde2e7bb242bf6.

Selected test_shard_granularity_matches_a_real_encoded_unit IDs:
E2M1_K1-R512, E2M1_K1-R448, E2M1_K2-R896, E2M1_K2-R700,
E4M3_K1-R1024, E4M3_K1-R1023, E4M3_K1-R900 (each TESSERA_ prefixed).
Both reconciled runs report 7 passed, zero skipped, 67 explicitly deselected
other cases and 14 existing Torch JIT warnings. Selected nodes are identical;
zero missing, duplicate-collected, never-ran or outcome problems. This is not
a fresh qualification of the other67 cases or a full suite.

Published pbtest.py seals one PB-owned file shard with -k selecting those
seven cases, explicit dl380g10 tag, one worker, native threads1, CPU2/mem4 GiB,
900-second deadline and priority-10. GPU visibility is disabled. Python3.14.4,
Torch2.11.0+cpu, Transformers5.16.1, PB SDK95a59051 and Tessera b40c93cb are
recorded and verified by the existing install/pin preflight in both logs.
Worker runtime is the admitted c437 generation; both preferred cpusets are
[0,1]. Neither action, profiler or observed test ending failed.

| Observed measurement | Before | After |
|---|---:|---:|
| Pytest wall |191.48s|67.97s|
| PB contained wall |198.909s|76.469s|
| Cgroup CPU |233.825s|86.601s|
| Cgroup memory peak |807796736 bytes|770117632 bytes|
| Observed write bytes |2339288|2339288|
| Observed read bytes |0|24576|
| Real geometry test inclusive sampled time |185.99s|61.01s|
| Public exporter inclusive sampled time |185.76s|60.85s|
| viterbi_window inclusive sampled time |180.18s|56.21s|
| py-spy100Hz samples |19396|7061|

Observed selected-case wall decreases123.51s (64.50%) and cgroup CPU falls
147.224s. This is one sequential profiled pair on a non-isolated machine;
background activity differs, so it is not a guaranteed isolated speedup or a
suite-wall claim. Inclusive sampled times overlap and must not be added.
The after profile still records genuine exporter and trellis work; no mock,
precomputed output, source-wording assertion or skip replaced execution.
No byte-identical reconstruction or quality improvement is claimed.

## Receipts, profiles and telemetry

Before action:92c074b39859a24cd5d8b47b3c5a69b6411620945a030e77fe1bf7e693ccbcd1.
Receipt:037949ba6a04137cd65c1469d5d33db78e7cf7d72139f0dd725120e3cb04da85.
Snapshot:122eed73f537661393f8ed30ed216423b1ea71e5; parent exact base.
Input:ae06d2c91e0fdd2354b58cffef84a62800382c9ee8ae696eb163cfa1c60d33d5.
Profile:245aa0b8773ee09b1be1009062d2c1bb9a282d49cfcc684ef3d7eb5910a32736.
After action:1e247e5638ee511bf02b6c8f26828dea92f74f1302ffb876f172b38590747202.
Receipt:8916f37c9561067f0e2096dc6e4f3e842c0f829acd3777c2cffd0a86645a0bff.
Snapshot:bacc1f253b7d01b374f979a4b8a9e3a590aed587; parent exact test commit.
Input:a0834527bfc47264fb7aba9a2839d750f8f64a80e4644baa53fe1a4a072859e4.
Profile:536b274ea743a07395e9d3e100efaf574c1a8caebcd0f676a08bc67f4839f3de.

Both profiles are successful py-spy0.4.2/100Hz CAS blobs. Raw reconciled
inventories, action/snapshot/receipt/resource records and all six raw Netdata
responses are /home/rob/tmp/pq1929-menu-evidence.json.

Average Netdata CPU busy excludes idle/iowait:

| Host | Before busy | After busy |
|---|---:|---:|
| dl380g10 |13.743%|13.298%|
| sparky |2.308%|1.727%|
| sparklina |1.779%|1.399%|

Requested queries were1791055646..1791055846 and1791055970..1791056047.
Captured before response view is1791055647..1791055846, with data timestamps
1791055648..1791055846 (100 buckets). Captured after view is1791055968..1791056047,
with data timestamps1791055969..1791056047 (40 buckets), including about2.5s
before the PB claim. The table means cover those full response buckets, not
claim-only intervals. PB scope windows are1791055646.640335..1791055845.549253
and1791055970.520300..1791056046.989485. No response has flagged/empty buckets.
PB's own DL window averages are13.836%/13.476%, with
explicit missing-pqteld-CSV diagnostics retained. No GPU, energy or monetary
claim is made. The designated parent reviews this exact source; independent
review and one compatible full-cohort integration qualification remain separate
mandatory gates. PQ1929 stays OPEN; a selected-case saving cannot close it.
