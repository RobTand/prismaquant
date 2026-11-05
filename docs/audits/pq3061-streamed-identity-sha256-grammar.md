# PQ #3319 (part of #3061): streamed identity SHA-256 grammar

## Changed sites

`prismaquant/streamed_model_identity.py::validate_streamed_model_identity`:

1. Content digest check: `str(identity.get("content_sha256", "")).lower()`.
2. Shard digest check: `str(shard.get("sha256", "")).lower()` per shard.

Both inline `re.fullmatch(r"[0-9a-f]{64}", ...)` checks now call the
existing owner `prismaquant.digests.is_sha256hex`. The caller keeps the
`str(...).lower()` conversion, every RuntimeError text, and the schema,
shard, canonical and content-equality checks. `digests.py` is unchanged.

Callers: `streamed_model_identity` (about lines 846, 1349, 1432),
`aura_cost`, `expert_empirical_cost`, `glm53_stock_reprice`,
`joint_adjoint_band`, `joint_aura_rows`, `joint_cost_quantum`,
`joint_cost_stage_a`, `native_moe_panel`, `prismasnap_checkpoint`,
`prismasnap_moe_checkpoint`, `prismasnap_validation`, `source_generation`,
`streaming_production_cache`, `union_production_cache`.

## Compatibility evidence

The owner grammar is `SHA256_HEX = re.compile(r"[0-9a-f]{64}")` by
fullmatch on a `str`. The caller always passes a `str`, so the grammar is
the same. Tests in `tests/test_single_file_streamed_source_proof.py`:

- `test_streamed_identity_validation_uses_shared_sha256_grammar` records
  one owner call for the content digest and one per shard digest.
- `test_streamed_identity_digest_grammar_outcomes_are_frozen` freezes
  acceptance or the full RuntimeError text for uppercase, nonhex, wrong
  length, newline suffix, non-string, malformed shard and resealed
  contradictory metadata cases. These pass on base and on head.

## Untouched residual sites

Inline regexes in the same file at about lines 210, 816, 899, 1138 and
1392 stay unchanged. `import re` stays.

## Scope

The parent #3061 keeps the repository-wide inventory and the scanner gaps.
This leaf makes no source-custody, #2989, speed or serving-quality claim.
