# Operational consumer subset

## Result

No application changes are safe or necessary with the current settled owners:
26 existing primary digest/JSON/file-hash calls already route to settled owners.
All seven supplemental JSON calls in the frozen parent-completion registry are
blocked by absent insertion-order recipes. No owners, census, baseline, pins or
frozen registry were edited. No json.dump calls occur in these six consumers;
no streaming write or partial-failure behavior was replaced.

Missing recipes (ASCII escaping, lax nonfinite handling, insertion order):

- indent=2 stdout: autoscale module CLI and day0_model_intake.main.
- default-spaced stdout: ship_gates.main and release_receipts.main error path.
- default-spaced dumps/loads normalization: streaming_initialization's MTP
  validator and streaming_model's complete_selected / complete_prefix.

All settled JSON profiles sort keys, so using them is not exact. Changing the
normalization calls would also reorder nested records. These sites deliberately
retain their original expressions and are explicitly marked blocked with null
owner_route in ledger.json, rather than claiming completion.

## Provenance and coverage

ledger.json records each call, source commit
6d8288adc94b3c4b624c143ed372fb6513f32784, original Git blob, actual owner route or
blocker, independent original source excerpts, and executable AST-resolvable test
functions. Primary IDs are local provenance IDs for already-routed calls, not
new frozen-registry entries. Excerpts were captured with git show, not rebuilt
from owner implementations.

Tests exercise original/current consumer call expressions, original/current
consumer statements (stdout, insertion order, nonfinite values, exceptions,
Unicode escaping and normalization snapshots), the complete receipt CLI main
with a substituted ingestion boundary, and the actual MTP validator with valid
and invalid witnesses. These are CPU compatibility tests, not full GPU/model
integration tests. Source blob identity, frozen supplemental call syntax hashes,
fixture syntax and ledger test-node resolution are checked.

The first focused run had 12 failures caused by an incorrect test expectation
(default whitespace instead of the existing compact hash recipe); this was
corrected in tests only. The final focused run passed all 54 tests.

Command:
`python -m pytest -q tests/test_digest_1301/orchestration-runtime/test_operational.py --basetemp="$TMPDIR/pytest"`

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e7265107f897019dd1ab91b307cca86f80baf647e44c8339839e3b03706a0a1e","size":269}},"operation":"cccf4d31e991666423c11d0fe3ad7df272daabaf47f1fa8a6e18452337f561f5","receipt":"f048a53a2ef944eb2ca1d8bcfee88450f078d2952656214b8d9b64acd304f87e"}]
