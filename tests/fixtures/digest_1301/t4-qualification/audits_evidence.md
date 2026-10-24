# Audit consumer digest migration evidence

## Scope and accounting

Only the five assigned audit/inspection/pilot consumers, `test_audits.py`, and
`audits*` fixtures were changed. Pre-existing recovery/qualification/rebinding
changes were left untouched. No owner, shared implementation, census, baseline,
or deployment changes were made.

`audits_ledger.json` freezes six original raw SHA callsites against source commit
`8bc0b5af8c69af352b559dccda99bd4e6fa8b418` and their original Git blobs.
Each entry identifies its owner route, verbatim original main fixture, fixture
SHA, original main starting line, and executable parametrized test node. Tests
resolve original AST line/column identities and check migrated AST owner calls.
The render consumer retains its existing temporary-sys.path `digests` import,
extending that same import rather than introducing a package dependency.

All ten original unsorted `json.dumps` calls are unchanged. They are recorded
separately in `audits_deferred_json.json` as **deferred missing-owner eligible
gaps**, not retentions. Owner-recipes does not supply these unsorted JSON owners.

## Consumer evidence and limits

Five independent fixtures contain only verbatim original main functions, not
full source copies. Tests run each original main and the actual imported current
consumer main with identical inputs, compare exact stdout and publication bytes,
and check JSON/newline output. Inputs include non-ASCII metadata and binary wire
bytes. CPU/path/decoder adapters replace external campaign paths and unavailable
GPU/decoder dependencies; they do not replace the migrated digest owners except
for the render consumer's external historical proof-file digest gate. Real
render-file hashing and the existing JSON profile remain exercised. A separate
negative test uses the real proof gate and verifies refusal without publication.

Malformed envelope payloads are refused before payload unpickling in both the
original and migrated audit/pilot mains, with no publication. The selected-cache
loader/verifier and pilot tensor decoder are adapters, not qualification of those
dependencies. **Pilot CPU decoder tests are not GPU qualification**; real T4,
production campaign artifacts, and end-to-end external verification remain
unmeasured. No protected current source-binding pin was identified for these
consumers; a historical measured qualifier is not a current source pin. No
source-pin or deployment expansion was performed.

## Checks

Initial scoped run: 6 passed, 2 failed. The harness initially overwrote an unrelated
cost pickle during corruption setup and assumed Git objects were available in
test snapshots. Fixed by limiting corruption to unit envelopes and validating
frozen blocks via registry hashes and original AST coordinates without runtime
Git access. Subsequent scoped run: 8 passed. Added real proof-gate refusal test.

Final command:

```sh
python -m pytest tests/test_digest_1301/t4-qualification/test_audits.py tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py -q --basetemp="$TMPDIR/pytest"
```

Result: **143 passed**, 14 upstream Torch/Python 3.14 deprecation warnings.
`git diff --check` also passed. Full repository suite and GPU execution were not run.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9d1560470d1103eaf63066062a93bae4bc503f4de437b610700dbec85bea41af","size":902}},"operation":"d4b351b6a20339a1c64b74d65784a39b6dbac2acdef5e12aeb5e46a79ba5324c","receipt":"9ca2e9d12c2659eb26f1b2a5ad61f7b6ac49386bc4c0fefa6f5bd5c4980b2c33"}]
