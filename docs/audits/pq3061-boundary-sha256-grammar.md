# pq3061: boundary_control SHA-256 grammar (#3313)

Owner: `prismaquant.digests.is_sha256hex` (str or str subclass,
`[0-9a-f]{64}` fullmatch). digests.py is unchanged.

## Changed sites (prismaquant/boundary_control.py)

1. `artifact_content_id`: `content["model_sha"]` check.
   Callers: `_validate` (same module), `tools/measure_boundary_control.py`,
   `experiments/pq87_paired_validation.py`, `experiments/pq87_physical_ab.py`.
2. `validate_stack_contract`: each role `artifact_id` check.
   Callers: `_paired_stack_payload` (same module),
   `experiments/pq87_paired_validation.py`.

The `isinstance(..., str)` checks, the error strings, the validation order
and the per-role loop stay. The digest input and identities do not change.

## Compatibility evidence

tests/test_boundary_stack_contract.py adds control tests. They pass on base
and on head. They capture the return value or the exact refusal type and
message for these inputs: lowercase, str subclass, uppercase, short, long,
empty, newline prefix/suffix, non-hex, space, bytes, None and integers. They
also cover field precedence (image before roles, artifact_id before
extensions, extension refusal before a later role, distinct-artifact check
last). `test_boundary_declarations_use_shared_sha256_grammar` records calls
to the owner. It fails on base and passes on head.

## Residual sites (unchanged)

- Image-reference grammar `[^\s@]+@sha256:[0-9a-f]{64}` in
  `validate_stack_contract`.
- Extension-basename grammar `[A-Za-z0-9_+.-]+\.so(?:\.[0-9]+)*`.
- Sites outside this module, for example
  `shipcard._closed_weight_content_manifest` and `schemas._SHA256_HEX`.

The parent #3061 keeps the repository-wide inventory and the scanner gaps.
This note makes no speed or serving-quality claim.
