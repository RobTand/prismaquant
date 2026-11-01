# Sampled whole-stack proposal validation helper, retired 2026-10-14

Commit `d91ed9c5dd` deleted `prismaquant/proposal_validation.py` and
`tests/test_proposal_validation.py`, together with six other prototypes that
only their own tests imported. This wall holds the last live version of both
files, byte for byte. It was added on 2026-10-16 for Claude review finding
R3B-03.

## Why the helper is kept

`prismaquant/tessera_sampled_stack_proposal.py` still emits an
`independent_selected_assignment_validation` block with
`required_for_production_promotion: true`, `status: pending` and
`validation_helper_schema: prismaquant.sampled_proposal_validation.v1`. This
module is the only implementation of that schema
(`SAMPLED_PROPOSAL_VALIDATION_SCHEMA`, `validate_sampled_proposal`). The
block's `validation_helper_archive` field names the file in this wall.

No live gate, exporter or allocator reads that block. The proposal stays a
research proposal whose required validation has not run, and no live code can
run it. This wall does not promote a proposal and does not make the
validation a gate.

## What the helper does

`validate_sampled_proposal` compares held-out, per-sequence losses of a
candidate selected assignment with those of an incumbent. It binds the
source, teacher, tokenizer, held-out population, both assignments, the
metric identity and the held-out/training overlap audit. It refuses unequal
sequence sets or scored-token counts. It reports an approximate paired
manifest-cluster bootstrap interval and `pass`, `inconclusive` or
`regression` against a caller-supplied margin. Failed or timed-out rows force
`inconclusive`. It cannot authorize export and makes no latency claim.

## Provenance

| File in this wall | Original path | Git blob | SHA-256 |
|---|---|---|---|
| `prismaquant/proposal_validation.py` | `prismaquant/proposal_validation.py` | `b8f4dc25d431db325feab8b7d310c9ba6ce02b78` | `b16c2fa3139cdc02c269d1c627326d5147ad19d4dfad0da7a757defeb2d9d750` |
| `tests/test_proposal_validation.py` | `tests/test_proposal_validation.py` | `9457837647e2b1bf2eb922f2b32a31de36c56a3f` | `184c4e2b3573fab41d5110989d57a1be9c35779dce31d978472e9beff4cb1e66` |

The bytes come from `f80ae9a186f7a32d949f1390286308267dfda587`, the parent of
the deleting commit. Both blobs equal the files at the refactor base
`138069b799323435fb5f0b3cec3ee339f6264a62`. The module was added in
`252fd9cefe` and last changed in `8c3f3c0491` (both 2026-09-13).

## To restore

1. Move `prismaquant/proposal_validation.py` back to `prismaquant/` and
   `tests/test_proposal_validation.py` back to `tests/`. The test imports
   `prismaquant.proposal_validation`, so it cannot run from this wall.
2. Expect `tests/test_duplication_baseline.py` to fail. Retirement removed
   three baseline entries for this module: the `hashlib` site in
   `_bootstrap_differences` and the same-name helpers `_quantile` and
   `_require_sha256`. The baseline only shrinks, so move those helpers to
   their owners (`prismaquant/digests.py` for the digest) or record a reviewed
   exception.
3. Re-add the gate-table row in `docs/ARCHITECTURE.md` that
   `d91ed9c5dd` removed ("Sampled whole-stack proposal comparison"), and
   point the proposal's `validation_helper_archive` field at the live module.

A static import check at the commit that added this wall found that the
module's only package import, `cost_stage_checkpoint.canonical_json_sha256`,
still resolves; `cost_stage_checkpoint` re-exports it from `digests`. That
check does not show unchanged behavior. Run the archived tests under
PrismaBuild after restoring.
