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
preflight have exact source names, rates and coverage. No checkpoint export
has been launched. The later exact pin retake to Tessera's MTP mapper commit
requires its own refreshed preflight build anchor before export.
