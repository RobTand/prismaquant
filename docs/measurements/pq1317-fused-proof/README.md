# PQ-2546 allocation-bound fused-module proof

Scope: prismaquant#2546, part of prismaquant#1317. Measurement only.
No allocation, profile, menu, or export-gate change.

## Verdict

Three measured allocations carry zero mixed-rung fused modules.
The export gate passes each one with `result: null`.
The mixed-rung refusal stays active in five gate cases.

## Files

- `selectable_matrix.json`: every selectable family, structure, and rung.
  It derives from the packaged v60 contract cells and the three
  allocation menus, not from historical picks alone.
- `allocations.json`: per-allocation digests, menus, serving groups,
  fused census, and gate result.
- `bindings.json`: checkpoint, exported artifacts, profile, producer
  build, and packaged contract digests.
- `gate_cases.json`: the five retained gate results and their PB action.
- `route_gaps.json`: contract-dependent route gaps, kept apart from
  the grouping proof.

## Method

The probe canonicalizes each retained assignment. It then runs
`serving_groups_by_key`, `_fused_sibling_groups`, and
`require_fused_rung_coherence` under the detected GLM profile.
The retained record is
`/mnt/shared/tessera-measurements/pq2546-a1/source-probe.json`.
PB actions `a56a86a3` and `7680586a` hold the admitted runs.

The five gate cases run through PrismaBuild as action `e827ceeb`:
5 passed, 43 deselected, clean reconciliation.

## Limits

- A changed assignment needs a new proof. These digests bind this result.
- Route gaps (compiled mode, 880/912 census, grades, TP2 stub) do not
  alter the grouping result. They are recorded, not waived.
- Producer identities come from retained receipts. They were not re-hashed.
- The joint pricing pkl was not re-opened here. Its priced grid is prior
  evidence in `CELL-COVERAGE-2026-09-28.md`.
