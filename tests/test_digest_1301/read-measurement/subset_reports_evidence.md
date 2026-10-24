# Read-measurement subset reports

## Changes and provenance

Three base calls replaced, using only existing settled owners:

- `tools/benchmarks/joint_validation_pair.py::sha`: `file_digest_sha256hex`, on the same caller-owned handle, retaining stdlib `file_digest` acquisition.
- `tools/compare_joint_layer_gate.py::main`: `DIRECT_ASCII_INDENT2_LAX.text`.
- `tools/compare_stage_a_checkpoints.py::main`: `DIRECT_ASCII_INDENT2_LAX.text`.

The benchmark now resolves the checkout package using its source-relative root, matching the standalone tools' existing bootstrap convention. No environment-specific paths were introduced.

`subset_reports_ledger.json` records each original scanner ID, pinned source commit, git blob ID, owner route, independent fixture and literal test node. Fixtures contain complete original-consumer SOURCE functions taken from git commit `6d8288adc94b3c4b624c143ed372fb6513f32784`. Behavioral tests execute those originals independently alongside the migrated consumers; they do not use parent acceptance tests. They compare bytes, stdout and failure behavior, including Unicode escaping, nesting, nonfinite floats, terminal newlines, unsupported objects, publication refusal, binary/empty/multi-megabyte files and filesystem errors. GPU/model computation is stubbed in the report-writer tests; real GPU behavior is not claimed.

## Blockers and audit boundaries

All JSON calls in the eight assigned files were inventoried. Seventeen writers are blocked on exact settled profiles, not justified retentions. Fifteen readers are recorded separately outside the digest-encoding population; JSON parsing is not a consolidation target. Original SOURCE expressions and pinned provenance are included individually.

Missing writer profiles are insertion-order (unsorted) defaults with no indentation, indent 1 or indent 2; sorted indent 1 without `default=str`; and unsorted indent 2 with `ensure_ascii=False, allow_nan=False`. Using a sorted owner for unsorted output would reorder keys. Using `DIRECT_ASCII_INDENT1_DEFAULTSTR_LAX` for the sorted indent-1 writers would silently accept unsupported objects that currently raise. No custom profiles were instantiated and no owner, census or baseline was edited.

Consequently the five other assigned application files remain unchanged. Existing owner-routed file/text hashes and canonical domain hashes were retained. This subset is not an assertion that all JSON migration is complete: exact missing profiles require owner work outside this assignment.

## Verification

`python -m pytest -q tests/test_digest_1301/read-measurement/test_subset_reports.py --basetemp="$TMPDIR/pytest"`

Observed: **5 passed**, exit 0, CPU environment. `git diff --check` also exited 0. No parent acceptance tests were run.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"cb04fd598b2f2b46c80e5c105bcbdce949ba23169762d5759ce815f010af2c63","size":268}},"operation":"22c316b6b74abd89be2ce1d8d8d9eb567c2807fe2e00763a9f729cb44d28eb3e","receipt":"cf8aae5ddadda3f068e32f651eef50a476e5058eafbefb11fbaf92cba58547ea"}]
