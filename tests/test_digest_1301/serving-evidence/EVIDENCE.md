# Serving evidence slice: blocked eligible recipe gaps

## Scope and disposition

Inspected AGENTS.md, prismaquant/digests.py, docs/design/design_guidelines.md
and docs/ARCHITECTURE.md. All **eight insertion-order gaps are blocked eligible,
awaiting shared owner integration**. Missing owner recipes are not justified
retention and are not evidence of an unusual non-migratable profile.

The task requires replacement after owner integration, but that integration is
absent: the inspected JsonProfile encoder hard-codes sort_keys=True. Substituting
it would change insertion order and mixed-key acceptance. No replacement is
claimed or attempted. The ledger has disposition `blocked_eligible`,
`retained_calls: []`, `production_replacements: 0`, and null owner routes meaning
**unresolved**, not retained. No production replacements, owner, census, baseline,
or frozen-registry changes were made for these eight sites. This correction
changes only the prior ledger, test and this evidence file; other existing work
is untouched.

## Exact requested shared owner recipes and identities

All rows use source commit `6d8288adc94b3c4b624c143ed372fb6513f32784`.
The ledger records each complete `supplemental-json:prismaquant/...` original ID,
file, scope, source blob, fixture, executable test nodes and explicit requested
recipe. Integration is requested in `prismaquant/digests.py`; no new API name is
invented here.

Common options: insertion order (`sort_keys=False`), ensure_ascii=True,
check_circular=True, skipkeys=False, cls=None, default=None (no fallback).
Default-spaced recipes use indent=None and separators=(", ", ": "). Indented
recipes use separators=(",", ": "). Preserve stdlib exception behavior and,
for dump, partial stream writes. Preserve caller dict conversion and loads
roundtrips rather than replacing them with sorted canonical normalization.

| File / scope @ line:column | Source blob | Exact requested expression |
| --- | --- | --- |
| compressed_route_sweep.py / write_sweep @297:8 | 976f489cccdc658d9e997f2aca8b0fe0b1235e20 | `json.dump(record, handle, indent=1, sort_keys=False)`; allow_nan=True; caller UTF-8 file and final LF |
| glm_source_derivative.py / bind_source_derivative @210:22 | 013fb3311b014cef293c1f7872fed290f496d1c0 | `json.loads(json.dumps(identity))`; default-spaced, allow_nan=True |
| glm_source_derivative.py / normalize_source_derivative @72:22 | 013fb3311b014cef293c1f7872fed290f496d1c0 | `json.loads(json.dumps(value, allow_nan=False))`; default-spaced, strict nonfinite refusal |
| shipcard_slots.py / _verify_route_sweep_record @1044:27 | df7d8dd675d21fd0df1577e672de363ba39e08e7 | `json.dumps(row.get("sweep"))`; default-spaced, allow_nan=True |
| shipcard_slots.py / make_uniform_control_record @918:23 | df7d8dd675d21fd0df1577e672de363ba39e08e7 | `json.loads(json.dumps(dict(control_block)))`; default-spaced, allow_nan=True |
| shipcard_slots.py / make_uniform_control_record @942:38 | df7d8dd675d21fd0df1577e672de363ba39e08e7 | `json.loads(json.dumps(dict(control_arm)))`; default-spaced, allow_nan=True |
| tessera_route_receipt.py / _check_runtime_image_declaration @415:51 | 6c213081ca83e743eead89a9fd9d34d62164f4c8 | `json.dumps(dict(record))`; default-spaced, allow_nan=True |
| tessera_tp_audit.py / _write @250:11 | 6cd2c62f4c91f7f86247592bbe73af721ea7917a | `json.dumps(receipt, indent=2, sort_keys=False)`; allow_nan=True; caller UTF-8 file/stdout and final LF |

Whole-file raw JSON encoder accounting (dump/dumps/JSONEncoder, not decoders):

| File in prismaquant/ | Original | Current | Replaced | Blocked eligible |
| --- | ---: | ---: | ---: | ---: |
| compressed_route_sweep.py | 1 | 1 | 0 | 1 |
| glm_source_derivative.py | 2 | 2 | 0 | 2 |
| shipcard_slots.py | 3 | 3 | 0 | 3 |
| tessera_route_receipt.py | 1 | 1 | 0 | 1 |
| tessera_tp_audit.py | 1 | 1 | 0 | 1 |

## Independent fixtures and executable coverage

All eight behavioral fixtures are unchanged. They contain original enclosing
functions and expressions from frozen source blobs, not owner-generated output.
Tests verify AST SHA256 against the frozen registry, original/current behavior,
insertion order, Unicode and lone surrogates, whitespace, nested tuples, integer
and mixed keys, nonfinite values, cycles, unsupported values, invalid keys,
errors and partial stream writes. Writer bodies exercise temporary files/stdout,
bytes and final LF; normalization checks validation and independent nested copies.
The former noneligibility test is renamed to recipe-gap provenance; accounting
now asserts all eight blocked eligible, no retained calls and no replacements,
and checks each explicit requested recipe against its frozen expression.

These isolated CPU checks do not establish owner integration, migrated consumer
compatibility, end-to-end serving, GPU, model or runtime-image validation. After
shared owner integration, migrate these eligible consumers and rerun compatibility
checks against the integrated recipes before claiming completion.

## Observed results

Command: `python -m pytest -q tests/test_digest_1301/serving-evidence/ --basetemp="$TMPDIR/pytest"`

Result: **51 passed in 0.95s**, exit 0, including the existing plan and replay
checks in this directory. Prior retention-era results are superseded.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"5faa29e988a2671599e0ec780204124423b6f73ee580ef3f1d31960fab6150a4","size":269}},"operation":"2b8548685c19e62e608c2e363cc7f92509219995f6994b4b5101ee697d5b2193","receipt":"50a66e3b28305d76eeadd1c266353e1ca436922ad1bda4fa7b25dc8354d74039"}]
