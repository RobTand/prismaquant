# Packed GLM profile and frozen producer source — 2026-09-08

The opt-in `glm_packed_research_sm121` profile restricts routed GLM logical
experts and aggregate stacks to legal Tessera E4M3 K1 rungs or plain BF16,
while retaining the inherited dense menu and the existing candidate gates.
The shared family filter and its CPU evidence are described in
[the profile report](../glm_packed_research_profile_20260908.md).

The development and serving pins now both identify Tessera
`07ad344c3275bb2fa7ce2432f93d89945d66f4c2`, reviewed and merged through
[Tessera #431](https://github.com/RobTand/Tessera/pull/431).
This source includes the explicit checkpoint research execution declaration,
strict parser, exporter/partition identity binding and ordinary plugin dispatch
into the existing packed expert owner. The raw packaged contract stays
`a688f8de244f936ec3a63a782e20af7985733e7a6fb0b4b981b5fe4c44112212`.
There is no new runtime cell, production default or full-model qualification.
The allocation profile is emulation-only, declares no export lane, retains TP1
and grants no per-role expert capability.

The producer archive is frozen before new anchor pricing:
`956bbafeab40fca1144672c3af35900daaeac693d777b0c3f243de73354d4c75`.
All 1,179 staged files were checked against their recorded bytes and SHA-256;
the archive had already been independently matched to the exact Git tree.
A PB CPU import of that source verified both runtime pin/contract agreement
and the actual encoder seal
`42783e5214ba1a56d020925155641f9348139ce3f1980ec6df016f71583ee4c3`.
The PrismaQuant production-cache source seal is
`5a507741716b7a7a7f50dd8f650852e2f5b910d245e9325244b9ccf736c64a53`.
Pricing and export must retain these identities; a later consumer-only change
must not silently relabel an existing priced wire.

Validation used PB CPU workers on `dl380g10`, Python 3.12 with Torch 2.10,
GPU visibility disabled and native thread counts bounded to one:

| Check | Result | PB action |
|---|---|---|
| Profile/candidate suite, 4 CPU / 8 GiB | 118 passed, no skips | `1f7de9ed8fd697f296640c82d77f3e99bf682727d3d7f66067e3bbe9fd14b128` |
| Profile architecture/docs checks, 2 CPU / 4 GiB | 19 passed, no skips | `5ebbead650a8e7a5165e8e0bf73c95d2bcc47322f81e09dbfe3c75f9ec6e43b5` |
| Cross-repository H handoff, bounded sidecar, pin, v4/v5 contract, profile, candidates and docs, 4 CPU / 8 GiB | 194 passed, 2 skipped; compile passed | `fb948aafb61ee9717e4439187a95bfea7991a4c197eab5009fc33fccb3533332` |
| Final staged source and pin identity, 1 CPU / 3 GiB | exit 0; 1,179 files verified | `b50e2e1a986bd02358d9a1487150bef363c38c0b1e350203d0a266ad0418db90` |

The cross-repository skips require the native writer measurement and an explicit
v5 contract fixture. There were 56 Torch deprecation warnings and no missing
collection. Commands and root CAS/source audits are retained beside this report.
The audits check terminal exit and cleanup, receipt body hash, payload bytes and
executed Git snapshot against the reviewed source. The final identity snapshot
differs from `fa5f29d8de` only by PB's closure metadata; the cross-repository
snapshot differs from `4142bef407` only by that metadata. Later integration of
the reviewed observer readiness fix changes tests/docs only. Signer attestation
verification is not asserted by these checks.

No calibration forward, encoding, model-quality, memory-fit or speed measurement
was performed by this integration. The original GLM capture remains the shared
input for the six requested variants. Full-model export, stock-engine generation,
TP1/TP2 fit and measured quality/prefill/decode gates remain outstanding.
