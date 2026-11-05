# PQ3061: activation shard stamp SHA-256 grammar (#3314)

## Changed site

- `prismaquant/sample_parallel_artifacts.py::activation_cache_shard_stamp`.
  The inline `re.fullmatch(r"[0-9a-f]{64}", ...)` check is now
  `is_sha256hex(execution_digest)`.
- The existing `.digests` import now also imports `is_sha256hex`.

## Callers

- `prismaquant/incremental_probe.py::_validate_sample_activation_blob`
  rebuilds the expected stamp with this function.
- `tests/test_sample_parallel_artifacts.py` calls it directly.

## Existing owner

- `prismaquant/digests.py::is_sha256hex`. This change does not edit it.

## Compatibility evidence

- The caller keeps `str(execution_identity_sha256)`. The owner thus always
  gets a `str`, so its `isinstance` guard accepts every value that the
  inline regex saw. The owner uses the same pattern with `fullmatch`.
- The int conversion and cardinality refusal still come first.
- The refusal type is still `SampleParallelProbeError`, with the same text.
- The returned stamp keeps every field and value.
- Tests: `test_activation_shard_stamp_uses_shared_sha256_grammar` records
  owner calls. The controls cover valid, uppercase, bad-length, nonhex,
  None, numeric and `__str__` inputs, first-error order, and qname grouping.

## Untouched residual sites

- The same module keeps other inline `[0-9a-f]{64}` checks in the census,
  model-content and shard validators. The module `re` import also serves
  `_CALIBRATION_HASH_RE`.
- Hash constructors and JSON encoding (#1301) are unchanged.

## Scope

- The parent #3061 keeps repository-wide inventory integration and the
  scanner gaps outside this module. The parent stays open.
- This note makes no speed or serving-quality claim.
