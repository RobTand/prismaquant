# Joint handoff digest transitions — four-consumer slice

Scope remains the four prior transition consumers and their prior test/fixture/ledger files. No owner, registry, scanner or other agent files edited in this revision.

## Ledger status (not complete)

15 hash replacements; **zero justified retentions; four API-gap unresolved JSON sites**. `ledger_transitions.json` keeps `replacements` and `current_retentions` for integration, with an explicit `unresolved` array and incomplete `status_summary`. These sites remain inline pending exact owner APIs, not because pretty output or stdout is exempt.

Inspected finished `prismaquant.digests.JsonProfile`: its encoder always sorts keys. Exact blockers:

- `regenerate_joint_quanta._pretty`: sorted direct ASCII JSON, indent=1, default spaced separators, allow_nan=False, no fallback serializer. `DIRECT_ASCII_INDENT1_LAX_DEFAULT_STR` differs in both nonfinite handling and `default=str`; strict UTF-8 indent=2 differs in indentation and escaping.
- Both `stage_b_preparation_submission.write_submission` serializations: sorted direct ASCII JSON, indent=1, allow_nan=True, no fallback serializer. `DIRECT_ASCII_INDENT1_LAX_DEFAULT_STR` wrongly accepts unsupported objects instead of raising TypeError.
- `prepare_extended_joint_quanta.prepare` stdout serialization: insertion-order (not sorted) direct ASCII JSON, default spacing, allow_nan=True, no fallback. All finished profiles sort keys; `DIRECT_ASCII_SPACED_LAX` is not exact. stdout is eligible supplemental serialization.

The first three sites own UTF-8 encoding and final LF outside serialization; these must remain caller-owned. No owner changes were made and no approximate profile was substituted.

## Reduced provenance fixtures

Replaced four entire-source snapshots (2,397 lines combined) with only executable original functions needed for compatibility checks, plus the original stdout print statement. At extraction, each full input was verified byte-for-byte against `git cat-file blob` using the frozen source blob ID. Ledger `fixture_excerpts` records original 1-based inclusive source ranges, full-source blob identity and reduced-fixture SHA-256. Tests verify excerpt digests and registry identities; reduced excerpts are **not** claimed to hash to the full Git blob.

Original expression outcomes remain independent of owner implementation. Tests compare all 15 original/current hash routes against captured results, including package ordering and tree exclusions. Structural checks now cover only retained executable function excerpts (bind, _pretty, write_submission, _read_tree); they no longer claim whole-source or every-function AST equivalence. Existing focused consumer tests provide additional behavioral coverage.

## Behavioral gap follow-up

Added `test_transition_consumers.py` and independent frozen original whole
functions and enclosing transaction blocks, not full source copies.
`consumer_excerpts.json` records original commit/blob, exact line ranges, and
excerpt SHA-256. Extraction compared excerpts to the original Git source;
tests validate frozen hashes and registry provenance without requiring Git
history in test snapshots. Relevant ledger rows now reference behavioral tests.

Both live consumers and frozen originals are called for `_load_json`
(mismatch before gzip/UTF-8/JSON decoding, successful compressed-wire binding),
`_prior_slice_manifest` (mismatch before verified loader),
`_load_production_cache` (mismatch before unpickling), `_build_head_slices`
(actual emitted bytes bound by digest/length), `prepare_reads` (exact catalog
extension bytes), and `write_submission` (published manifest/template/submission
bytes, strict flags, conflicting-template refusal). Package tests cover Unicode,
astral/composed/decomposed names and identical surrogate encoding refusal.

All five regenerate `main` hash sites execute within actual live AST enclosing
`try` blocks against frozen original whole blocks in equivalent namespaces.
Assertions count one relocation and four publication hash calls. Traces compare
prior-verification/authorization order and refusal propagation. Publication tests
corrupt head, adjoint, slice and bound bytes independently, also remove slice or
bound files, and assert refusal without record publication. Both produced-output
contract settings are exercised. This completes the requested **block-level**
behavioral evidence, not full `main()` integration: intake/building, publication
backend, and authorization checker are dependency doubles. Real policy/backend
integration and the full suite were not run. Existing JSON API gaps remain
unresolved; overall migration status is still incomplete.

Final command:

```
python -m pytest -q tests/test_digest_1301/joint-handoff/test_transitions.py tests/test_digest_1301/joint-handoff/test_transition_consumers.py tests/test_prepare_extended_joint_quanta.py tests/test_generate_transition_rewrites.py tests/test_preparation_digest_routing_1850.py tests/test_stage_b_prepared_digest_dev_mode.py --basetemp="$TMPDIR/pytest"
```

**88 passed**, 14 upstream deprecation warnings, exit 0. Earlier harness runs
failed due to snapshot Git history, local-import mocking, missing namespace
bindings, and an incorrect expected dependency keyword; these harness defects
were corrected before the successful final run.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"810c612f94df140daf839eced9ae5135c8483f9700199dffe3ac3d1d3c1cd904","size":902}},"operation":"99a517253f5c5f6ae52e896d5edaae0204b533f9a80998af2c3809b7a5f265a2","receipt":"de29997780a2aacdedc8c711e02f15aa8ae4cefdf0fd306597154afeef9e96bb"}]

## Earlier verification

Command: `python -m pytest -q tests/test_digest_1301/joint-handoff/test_transitions.py tests/test_prepare_extended_joint_quanta.py tests/test_generate_transition_rewrites.py tests/test_preparation_digest_routing_1850.py tests/test_stage_b_prepared_digest_dev_mode.py --basetemp="$TMPDIR/pytest"`

Revised result: **63 passed**, 14 upstream Torch deprecation warnings, exit 0. Full suite not run. The new regression explicitly demonstrates default=str and stdout ordering incompatibilities.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"0d1549d3cddcf6028dddc0a21cd6e5ff8635156f2b48b082c1b3d3880af42b0b","size":821}},"operation":"0c049cd4d148a49132d12fc8b08b92fab310830208afe3e47d084c5e8ee8b5e7","receipt":"9fdda499e5afef6f8a510ccf3501148e96158317fa56f857ab648fe425e09578"}]
