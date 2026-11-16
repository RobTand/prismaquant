# Tessera pin move to v68, PQ #3677

This record replaces the earlier v67 preparation for this change.
Target: `8bf496abdb7d598a3132f6b4b7cb7d7e96199d13`, contract v68, lane v12, 18 cells.
The packaged contract gives SHA-256 `145a6dac54a63f9bac5ed3eda5d59b5c0f09c4233b4e95ffa63f9b4009a73be4`.
`literal-inventory.txt` and `consumer-files.txt` record the source searches.

## STEP 0 and OPTION A

| Default or policy | Coverage at v68 |
|---|---|
| `joint_served_activation.FORMAT` and its v1 policy | E2M1 R896, IMG-G only, TP1; dense admits, routed plan lacks required start state |
| `joint_catalog_extension.R13_ADDED_FORMAT` | Same scope; historical source records grant no extra scope |
| Campaign BF16/E2M1 source proofs | BF16 requires IMG-N; E2M1 requires IMG-G; old bytes need new encoding and prices |
| PQ2471 authorization | BF16 IMG-N TP2 loses its receipt; narrow to TP1; E4M3 keeps TP2 |
| Research profile | TP1 stays the default; no export lane or shipping grant |
| Pipeline and optional Qwen3 Tessera lane | Pipeline uses compressed-tensors; the optional vanilla image has no routed R768 cell |

The initial gaps were BF16 on IMG-G and BF16/E2M1 at TP2.
The CPU run also found that the routed E2M1 plan lacks the required start state.
The existing lane predicate refuses it. No default or policy can infer admission from the cell alone.
The lead chose OPTION A on 2026-11-16 at 02:54Z. No missing receipt is inferred.
PQ2471 adds no E2M1 IMG-N grant, image, rung, or world size.
IMG-G routed E4M3 R768 gains only upstream v66 evidence, not a PrismaQuant grant.
Its receipt is `experiments/results/glm53_r768_stub_base_tp2_eager_census.json` at the target commit.
Explicit research/custom targets can reach this cell. No current shipping default reaches it.

## Evidence

Flux evidence root: `/mnt/shared/flux-jobs/pq-3677-cpu-20261116T030014Z`.
The submit script is `submit-after-review.sh` in that directory.
Prepare job `f2ktKprQTrw` failed because the old parser lacked `graph_equality`; `f2ktyvV23hZ` passed after the parser fix.
The real domain walk found E4M3 1,793 rates and BF16 3,841 rates, with no holes.
BF16 table widths change at R3585 and R3841. The primary grammar fingerprint stays unchanged.
Requirements, reviewed answer, identity snapshot, and BUILD files use their existing generators.
The v60 shape-price fixture retains consumed historical bench rows. Historical qualification records stay unchanged.
The PR records final test jobs, verdict changes, and separate source/data line counts. The lead runs bbrun and review.
