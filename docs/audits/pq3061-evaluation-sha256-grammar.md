# Evaluation SHA-256 grammar: #3317

## Changed sites

Two checks in `prismaquant/tessera_joint_eval_panel.py` call
`is_sha256hex` instead of a bare regex:
- `validate_panel_descriptor` checks `eval_ids_sha256` (`:57`).
- `_draw_sha256` checks its value (`:79`).
`prismaquant/digests.py:120` owns `is_sha256hex`.
It accepts string subclasses.
Return values, refusal messages, call order, structure checks, selection,
seal checks and dev-mode behavior stay.
Callers of `validate_panel_descriptor`: `select_panel` and
`evaluation_execution` in the module, `prismaquant/tessera_joint_allocation.py`
and `prismaquant/tessera_joint_aura.py`.
`validate_eval_draw_descriptor` calls `_draw_sha256` three times
(`:108`, `:112`, `:117`).
Line numbers are for main at `6ca8ada35`.

## Evidence

Merge `61e0ed131` (PR #4622, #4591) made the change.
Flux job `f3B3gLEtfrf` recorded 36 compatibility cases through the
unedited consumer.
`test_evaluation_descriptors_use_shared_sha256_grammar` failed by
assertion on base and passed on head.
Main record run 44303024917 on `dbb7f578d` passed
`//tests:test_tessera_joint_eval_panel` on amd64 and arm64.

## Residual scope

The module has no other inline SHA-256 grammar.
Sample populations, seeds, prices, admission and token hashes stay
unchanged.
The duplication inventory and its shrink-only baseline stay unchanged.
#3061 retains inventory integration and scanner gaps.
#1301 recipes stay unchanged.
This note makes no speed or serving-quality claim.
