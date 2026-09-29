# Keep closed source-transition contracts separate

Decision: retain the three closed transition implementations. Share only
helpers with identical behavior; do not replace their proof, receipt, or
capability logic with a configurable base. This selects the explicit-retention
option in [PrismaQuant #1608](https://github.com/RobTand/prismaquant/issues/1608).

## Scope

This decision documents the current implementation. It changes no runtime
behavior, plan default, receipt schema, rewrite table, pinned digest, campaign
input, or artifact byte. It does not authorize a new source transition.

## Why the implementations are not interchangeable

Each transition proves a different contract:

| Owner | Contract-specific checks |
| --- | --- |
| `prismaquant/joint_aura_source_transition.py` | `_load_inputs(bindings, checkpoint_dir)` binds an existing checkpoint manifest, its identity, an inspection, and the preserved unit roster. `create_transition` accepts a predecessor; `_read_chain` validates that chain. `_actual_execution` requires a committed, clean producer package and refuses the legacy Git override. |
| `prismaquant/joint_aura_run_transition.py` | `_load_inputs(bindings)` binds the prepared record, plan, campaign identity, and production cache, without adopting a checkpoint roster. `source_proof` has an explicit development-mode recording branch. Its strict branch owns its allowed new files and reverse rewrites. |
| `prismaquant/joint_aura_retained_budget_transition.py` | `plan_difference(prepared_plan, run_plan)` permits only its enumerated retained-budget and record keys, checks exact integer caps, and requires the remaining parsed plans to match. The receipt carries both plans and the re-derived difference. This module has its own strict source proof. |

The verified capability types are also distinct: `VerifiedTransition`,
`VerifiedRunTransition`, and `VerifiedRetainedBudgetTransition`. Their loaders
and the transition router retain the corresponding admission and provenance
rules. A common implementation would need policy switches for checkpoint
adoption, predecessor traversal, plan substitution, development mode, and source
reconstruction. Those switches would obscure the closed contracts rather than
remove one identical rule.

In particular, source-proof file inventories and rewrite tables belong to the
contract that reconstructs the sealed package. They must not become mutable
shared configuration that lets a sibling's source changes alter that proof.
The campaign-specific bindings remain unchanged.

## What remains shared

`prismaquant/joint_aura_transition_base.py` owns the identical refusal, bound-file,
byte-identity, checkout-identity, and committed-package helpers. Its
`actual_execution(source_proof, repo_root)` helper takes the caller's proof and
repository root; the run and retained-budget transitions use that helper.
The older checkpoint-resume transition keeps its different execution check.
Digest serialization and hashing use `prismaquant/digests.py`.

Future deduplication must first establish identical behavior, including refusal
messages and failure paths. A new contract or admitted rewrite is a separate
change, not a consequence of deduplication.

## Verification

The existing regression suites remain the behavioral evidence:

- `tests/test_transition_twins_base_1607.py` checks the shared helpers against
  their previous spellings and refusal vocabulary.
- `tests/test_joint_aura_source_transition.py` checks checkpoint-resume admission,
  source reconstruction, chain handling, and verified capabilities.
- `tests/test_joint_aura_run_transition.py` checks the run contract and its
  strict and development-mode paths.
- `tests/test_joint_aura_retained_budget_transition.py` checks the budget-only
  plan difference and rejects changes outside that set.

Real-package campaign checks are explicitly environment-gated. A CPU fixture
pass does not attest a sealed campaign checkout, served numerics, or performance.
No regression test is added solely to restate this documentation decision.
