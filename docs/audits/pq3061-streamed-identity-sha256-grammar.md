# Streamed identity SHA-256 grammar: #3319

## Changed sites

`prismaquant/streamed_model_identity.py::validate_streamed_model_identity`
checks the content digest (`:1247`) and each shard digest (`:1256`) with
`is_sha256hex` instead of a bare regex.
`prismaquant/digests.py:120` owns `is_sha256hex`.
It accepts string subclasses.
Both caller-side `str(...).lower()` conversions stay.
Every mapping, schema, shard, canonicalization and content-equality check
stays.
Callers: 13 Python files outside the module, `tests/` and `archive/`, for example
`prismaquant/aura_cost.py`, `prismaquant/joint_cost_stage_a.py`,
`prismaquant/streaming_production_cache.py` and
`tools/full_kl_teacher_payload.py`.
Line numbers are for main at `6ca8ada35`.

## Evidence

Merge `91c4613a6` (PR #4658, #4616) made the change.
Flux job `f3BCo9TzYK9` recorded 19 compatibility cases through the
unedited consumer.
`test_streamed_identity_validation_uses_shared_sha256_grammar` failed by
assertion on base and passed on head.
Main record run 44303024917 on `dbb7f578d` passed
`//tests:test_single_file_streamed_source_proof` on amd64 and arm64.

## Residual scope

Four inline regex checks stay outside this leaf:
`_read_source_checkpoint_digest_cache` (`:206`),
`validate_source_model_binding` (`:812`),
`_strict_streamed_digest_entries` (`:895`) and
`build_streamed_model_identity` (`:1134`).
The note makes no source-custody claim for #2989.
The duplication inventory and its shrink-only baseline stay unchanged.
#3061 retains inventory integration and scanner gaps.
#1301 recipes stay unchanged.
This note makes no speed or serving-quality claim.
