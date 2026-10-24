# Plans serving evidence

Frozen research base: `6d8288adc94b3c4b624c143ed372fb6513f32784`, as recorded by `tests/fixtures/digest_parent_completion_1301.json`. `ledger_plans.json` records exact per-call original IDs, source commits/blobs, owner routes, independent fixtures, tests and compatibility `test_node_ids`, matching the completed calls in `ledger.json`. The historical `8bc0b5af8c69af352b559dccda99bd4e6fa8b418` attribution in `plans_original.py` describes snippet extraction provenance, not the frozen research base or completion identity. Whole-file owner-routed entries belong to population `owner-route`, not the primary raw scanner population.

`ledger_shipcard.json` likewise uses the frozen research registry identities and current call locations from `ledger.json`, including `_publish_shipcard`. Its detailed incremental-read recipe references the current source lines; historical snippet provenance in `shipcard_original.py` is not per-call registry provenance.

## Scope and decisions

Read AGENTS.md, design guidelines, architecture overview, digest owners, the historical census, the three consumers, and the existing external Hessian witness. Only tessera_export_lane.py and tessera_materialization.py required production changes; tessera_plan_writer.py was inspected and is unchanged. Owners/shared modules were not edited. The pre-existing glm_routing_replay.py change and other serving-evidence artifacts were left alone.

- Export build bytes use DIRECT_ASCII_INDENT2_LAX.encoded plus exactly one LF. In particular, using the strict file helper here would wrongly reject NaN/Infinity. Target display uses DIRECT_ASCII_SPACED_LAX.text.
- Materialization's sorted strict indent2 plus LF recipe already has an EXACT existing owner: digests.indent2_json_file_bytes. There is no strict indent2 JsonProfile and none was added. This helper preserves allow_nan=False, ASCII escaping, sorting, UTF-8, and LF; it is not a lax-profile approximation. Atomic writer and all request/plan/result callers are unchanged.
- Plan writer's two insertion-order indent2 calls (the disagreement diagnostic and provenance sidecar) are blocked eligible, NOT justified retentions. No insertion-order profile exists; sorting them or using a strict helper would change behavior. Existing plan/setup owner routes remain unchanged.
- Whole-file accounting lists all JSON encoders (primary and supplemental), existing owner routes, and decoders. Independent Hessian hashlib calls are separately recorded. No new profile was invented.

## Independent Hessian witness

The hessian_capture_sha256 function AST is identical to the frozen source. Its schema/identity JSON uses stdlib sorted, spaced ASCII JSON with default=str; tensor metadata is sorted spaced JSON, followed by NUL and host-contiguous tensor bytes in the per-unit hash. The outer seal frames sorted unit names with NULs and appends the unit hex seals. Its collection dispatch, roster validation, and runtime reference crosscheck remain intact. The existing test test_the_digest_rule_is_tesseras_capture_seal compares the independent digest with installed Tessera ActivationSource.capture_sha256 for a real saved torch payload, checks the allocation handoff, and demonstrates changed tensor bytes change the seal. That exact witness passed independently below.

Reservation, acquisition, and publication control flow is unchanged. Consumer tests exercise actual export main publication/hash/stdout/stderr and actual materialization atomic writing, including strict rejection preserving an existing publication. Original fixtures import no production owners.

## Metadata-only provenance repair checks

Focused rerun: `python -m pytest -q -rs tests/test_digest_1301/serving-evidence/test_serving_1301_plans.py tests/test_digest_1301/serving-evidence/test_serving_1301_shipcard.py --basetemp="$TMPDIR/pytest"` — **40 passed**.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9955ab9fc9c84ad4162604fdf5c9b70f3080dc96ca583011e18551caf5fb9ea3","size":269}},"operation":"07bf018a38876f7643bb5409798669525c78c798c630f45fab75196aa1961888","receipt":"2bc56b017f58b3ce03c213a5f7587253d1155e2fdc08beb099d2a3fb60ab5818"}]

Additional metadata assertions passed: three replacements exactly match `ledger.json` completed calls; four shipcard identities match the research registry and current ledger; recipe source ranges resolve to equivalent current AST nodes. An initial validation script incorrectly parsed a containing dictionary-entry line as a standalone statement and failed with SyntaxError; the corrected check resolves actual AST nodes (including embedded calls) and passed. No application or fixture change was needed for that validator correction. `git diff --check` passed. This repair changed only the two metadata ledgers and this evidence document; no tests, acceptance fixtures, `ledger.json`, or production files were modified by this repair.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"dcaeadc29d901f4c8334dd7356d7b5ec37c7fc6b70fdfd21ad9b5f84953dd6db","size":123}},"operation":"88429e55f47640d1303863c00f685e3a857164893d44e5e622064d5e1e8bebf2","receipt":"e92114ea5cdd529ee42dda7800ced10f0a1357e63ae832ac24e7237f0e433530"}]

## Historical implementation checks

Focused command:
`python -m pytest -q -rs tests/test_digest_1301/serving-evidence/test_serving_1301_plans.py tests/test_tessera_materialization.py tests/test_tessera_export_lane.py tests/test_tessera_plan_writer.py tests/test_tessera_priced_export_inputs.py --basetemp="$TMPDIR/pytest-plans"`

Final: **119 passed, 1 skipped**, 14 torch deprecation warnings. The skipped test covers the old runtime without capture_sha256; this installed Tessera publishes that seal. Two earlier runs failed because the new test's mocked preflight report omitted required display fields, corrected before the final run. The earlier runs reported 117 passed/2 skipped; the projection-producer availability skip was absent in the final run. No claim that earlier submissions passed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"aa6aa9eae78f054f7eb4c1edfb78d44b6ccdb821cee9d09aa36be1d4d6cef7c4","size":1117}},"operation":"94c69d4b9bc8ad9394c539a1d564c5969523c0877d9a3eae9f027176d864ad84","receipt":"0d196f0db8da0c798f9e8f81de866b698cf9c58b566c13601fbb9bc233abbafa"}]

External witness command:
`python -m pytest -q tests/test_tessera_priced_export_inputs.py::test_the_digest_rule_is_tesseras_capture_seal --basetemp="$TMPDIR/pytest-plans-witness"`

Result: **1 passed**, 14 torch deprecation warnings; not skipped or emulated.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b53eb5b52481d4f5e691d4eb6a166c8755aff58295946c5c600662f29cd4d5d1","size":760}},"operation":"af04fbca7ee4aa0e45db03fe051e6c48a3d6fbc08a3813568958adfc1a009a18","receipt":"b3286929f30ae63c2e754ac004f9653281c97aafbb09bacb1d6d4cba23e0108e"}]

`git diff --check` passed. No full repository suite or GPU qualification was run.
