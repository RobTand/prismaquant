# Image consumer digest routing evidence

## Scope and provenance

Only these production consumers changed:

- `tools/build_glm_derivative_image.py`
- `tools/build_prismasnap_27b_campaign.py`
- `tools/build_tessera_hessian_collection.py`
- `tools/build_tessera_mtp_cached_units.py`
- `tools/compose_tessera_cached_units.py`

No owner, profile, shared helper, baseline, census, or other consumer changed.
The five adjacent Python fixtures contain independent original source fragments,
not consumer copies or imports of the migrated implementations. They were
extracted with `git show` from commit
`6d8288adc94b3c4b624c143ed372fb6513f32784`. Each ledger row records the result
of `git rev-parse <source_commit>:<consumer_path>` as `source_blob`.

`ledger.json` records all 20 replaced base calls and the two retained supplemental
JSON calls in `calls`. `original_id` exactly matches the frozen registry in
`tests/fixtures/digest_parent_completion_1301.json`, including the `hashlib:`
prefix (not `raw-hashlib:`); `census_id` has the same spelling. Only routed
registry calls are candidates for `completed_calls`; the supplemental gap rows
are not completed migrations. The embedded-query witness is separate under
`auxiliary_witnesses`, not a registry call or completion entry.

`fixtures` contains plain repository file paths for acceptance `_repo_path`.
Function-qualified references are retained separately in `fixture_references`.
Every row names an AST-resolvable, non-parametrized test function in
`tests/test_digest_1301/tessera-supply/test_images.py`. The ledger test joins all
22 call IDs to the frozen registry, verifies exact consumer coverage and source
provenance, and validates evidence paths and function references.

## Compatibility and gaps

Existing owners perform the original sorted JSON recipes and byte/file SHA-256
operations. LF remains caller-owned for file formats. Campaign files retain
16 MiB reads, the final EOF read, regular-file/symlink refusal, canonical error
translation, and UTF-8/nonfinite policies. Eager `read_bytes()` and archive
`read()` boundaries remain eager; no path/stream substitutions were made there.
Publication, validation, no-clobber, and digest-before-selection order remain in
the original consumer bodies.

The supplemental JSON audit found two **unrouted** local calls in GLM `main`:

- base line 151: insertion-order, ASCII, indent-2, lax JSON result file;
- base line 154: insertion-order, ASCII, spaced, lax JSON stdout.

All existing JsonProfile owners sort keys. The available
`indent2_json_file_bytes` is sorted and strict, so it is not compatible with the
first call either. These calls remain unchanged and have gap rows, not fabricated
owner-retention claims. Tests compare their actual emitted bytes against original
source expressions.

The embedded `query` executes inside the original Docker base image, where no
settled owner is available. Its insertion-order JSON call is intentionally not
migrated. The fixture retains the exact original query string and the witness
test compares that string with the consumer AST. No Docker execution was claimed.

## Ledger repair validation

After repairing the registry IDs, plain fixture paths, and auxiliary witness
separation, the focused command was rerun:

```sh
python -m pytest -q -rs tests/test_digest_1301/tessera-supply/test_images.py --basetemp="$TMPDIR/pytest"
```

Result: **7 passed** (exit code 0). This includes the frozen-registry join and
all image compatibility witnesses. No production files were edited for this
ledger repair.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3fcc6f980c157c9b095cea391992746d9d6cef54849a7b6fa481baa7c4acfe70","size":268}},"operation":"02839859b47511b2ec527f6b6665444ee403f3dd172207a0d14eff2d9e37f29d","receipt":"1189f226c4fa55c0dabbb323b8cebed790a20d202203597f9a5fb79526bed91e"}]

## Prior compatibility checks

All tests were executed through `run_tests`.

Final command:

```sh
python -m pytest -q -rs tests/test_digest_1301/tessera-supply/test_images.py tests/test_build_prismasnap_27b_campaign.py tests/test_glm_derivative_reach.py tests/test_tessera_hessian_collection_handoff.py tests/test_glm_mtp_selection.py --basetemp="$TMPDIR/pytest"
```

Result: **45 passed, 1 skipped, 14 warnings**. The skip is the existing GLM reach
test because the installed modeling source does not match its pinned source; it
requests the `-tf516` environment. Warnings are existing Torch JIT deprecations on
Python 3.14. The seven new tests all pass. They exercise actual consumer function
bodies with injected domain boundaries, original independent fragments, Unicode,
nonfinite rejection/acceptance, serialization errors, file refusal and read
sizes, publication refusals, assignment digest refusal, Docker archive bytes and
readback, and ledger/test-node resolution. They do not build an actual image or
run a GPU campaign.

An initial harness run had 2 failures (wrong CLI invocation in the new tests),
then passed after fixing the harness; no failed check is represented as passing.
That initial run's operation was
`5e1849d1dd983e1e34f2a199dd02d82af22a9d1661cb4c28837b7f9326c72e3f`.
An intermediate expanded run passed 20 tests with the same existing skip.
`git diff --check` also passed.

Final retained test evidence:

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e3520b093d2edf2ee1915dc8392cab7ab0c9d1692ec8f2cc7c50c32dbf444999","size":1141}},"operation":"b6d19caea242c0d21106c884319bb5b22b249ad19bcdd93f76e6815f595153ce","receipt":"1c915733871942091a1b1e08fa30f47ddc6a60f557d998ac42d51d9af6455b08"}]
