# Exact SHA-256 regex validation (#1301, #1385, #1457)

This slice extends `prismaquant/digests.py`; it does not change any bytes or
acceptance rules. Source lines below refer to base `8fd6d931acc03a022e9858c16a1cef5b4de29851`.

| Site | Persistence / responsibility | Preserved behavior |
|---|---|---|
| `prismaquant/artifact_collection.py:85-88` | Load-bearing record/reference identities; `verify_record`, `make_reference` and `_validate_reference` check these fields. | `isinstance(str)`, underlying text fullmatch; own `ArtifactCollectionError` and text; returns the original object. |
| `prismaquant/prepriced_cost.py:45-48` | Load-bearing explicit priced-cost input bindings. | Same lexical check; own `ValueError` and text; returns input. |
| `prismaquant/prismasnap_checkpoint.py:405-408` | Load-bearing checkpoint/plan digest bindings. | Same lexical check; own `RuntimeError` and text; returns input. |
| `prismaquant/prismasnap_validation.py:167-170` | Load-bearing persisted validation evidence. | Same lexical check; its distinct `RuntimeError` text; returns input. |

`digests.is_sha256hex` owns that predicate. `digests.SHA256_HEX` owns the
compiled `[0-9a-f]{64}` pattern. The old `_SHA256` names bind that same object,
so inline users are consolidated without changing their coercions, branch
order or field reads:

- `prismaquant/prismasnap_checkpoint.py:541-547`: environment value is still
  lowercased before its fullmatch; no new coercion or acceptance rule.
- `prismaquant/prismasnap_validation.py:818,871,1242,1582`: the existing
  `str(...)` calls and surrounding evidence checks remain verbatim.

All these checks concern persisted identities; none is relabeled ephemeral.
They validate a digest's spelling, not its content, and compute no new digest.
The direct pattern's behavior on non-string inputs (including its TypeError)
is also frozen. Manual `len`/iteration validators, exact-`str` validators,
normalizing validators and protected/standalone modules remain separate.
A string subclass overriding `__len__` or `__iter__` makes a Python character
loop observably different from `re.fullmatch`; the shared owner deliberately
uses the latter.

## Evidence

`tests/test_digest_hex_1457.py` freezes 182 pre-change outcomes in
`tests/fixtures/digest_hex_1457.json`: four wrappers and three pattern names
across 26 inputs. It compares exception type/text/cause and accepted return
type/value/object identity. Cases include lengths 0/63/64/65, case, Unicode,
LF/NUL/space, non-string values and subclasses overriding length, iteration,
string conversion and truthiness. An added owner-binding test checks the
shared implementation and the important subclass separations directly.

Recording ran through PB action
`5ddebeb13f1f72f4cd0b71a4afd26c74714babde2bc533f881812d5f8b1a5f93`:
182 passed on the unmodified implementations. Claim
`0e13e9bb5d293102ca4c80df1af66a0f2cfb623381f60c444b24f930e47e9dfb`
passed payload/receipt checks; worker attestation was not independently checked.

No serialized value is rewritten. The load-bearing values remain untouched;
this fixture checks acceptance rather than a new wire or hash. No GPU,
rendering, calibration, residency, format, serving or ship-gate change is
claimed. The local wrappers remain because their error contracts differ, so
the same-name scanner may retain them even though their shared responsibility
has one implementation. Regeneration must still produce no baseline growth.
