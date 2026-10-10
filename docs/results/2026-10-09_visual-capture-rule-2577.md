# Visual capture rule and non-body wire plan (prismaquant#2577)

Date: 2026-10-09.
Signer: CEO, decision fleetgraph#16.
Parent: https://github.com/RobTand/prismaquant/issues/1921.
Child: https://github.com/RobTand/prismaquant/issues/2577.
Owner route: the campaign lead owns #1921 per CEO decision dec-1004-185302-b12c (2026-10-04). The issue brief names the campaign lead or Rob as owner. The CEO records this decision for the parent thread.

## Rule

Use strict full-scope capture for all vision Linears and all merger Linears.
Each declared quantizable vision or merger Linear needs genuine measured capture under the same calibration contract.
An OOM event or an absent live target fails closed.
No uniform cost, rate, or text-only Fisher enters the DP, except the approved explicit source-native control for A8S BF16 vision.
A uniform fallback is a guessed cost. Principles 1 and 2 forbid a heuristic where a measurement exists.
Prior finding: VISION-COST-FALLBACK-FINDING.md and parent comment 5936717501 (https://github.com/RobTand/prismaquant/issues/1921#issuecomment-5936717501).

## Producer plan for true non-body wires

The external Tessera producer owns the decoder body and true non-body wire support.
The campaign lead owns the producer plan with that producer.
PrismaQuant vendors no Tessera serving runtime and imports none.

## Admission plan for true non-body wires

Admit only the exact pinned Tessera runtime.
The pin must equal the reader constants, and the installed runtime_contract.json hash must equal the pinned digest.
Serve only device_qualified native cells that declare requires_plugin tessera.
Keep all gates. The export lane keeps its refusal of low-bit non-body choices (prismaquant/tessera_export_lane.py:2312-2406). The GLM vision source-precision pin stays.
Remove no gate. Guess no serve use.

## Artifact plan for true non-body wires

Keep measured original-wire receipts through selected-cache intake and export.
Emit byte witnesses for picked wires.
Emit only names that the pinned contract admits, else refuse.
Write no synthetic wire proof.

## Budget rule (prismaquant#1271)

See https://github.com/RobTand/prismaquant/issues/1271.
Claim no joint body-plus-MTP selection without an explicit versioned common currency and budget policy.
Ban quiet relabel of self-KL, body-KL, or a routed sub-budget as joint per-Linear choice.

## Code change

None. This note changes no code, removes no gate, and guesses no serve use.
