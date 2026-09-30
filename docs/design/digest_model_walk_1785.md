# Model-walk metadata hash ownership (#1785, part of #1301)

Only three one-shot SHA-256 constructions move to `bytes_sha256hex`:

| Caller in `prismaquant/model_walk.py` | Existing input and result contract |
| --- | --- |
| `_claim_rules_digest` | Existing `claim_rules_to_json` list, `DIRECT_ASCII_LAX_DEFAULT_STR.text`, strict UTF-8, full lowercase digest |
| `_model_identity` | Config's existing lax ASCII/default-str JSON, strict UTF-8, full digest and unchanged type prefix; config failure/missing config falls back to parameter count, failed introspection to -1 |
| `_example_inputs_spec` | Existing sorted/flattened tensor shape/dtype description, strict UTF-8, first 12 digest characters, first 160 description characters; synthesized default description bypasses hashing |

No JSON profile, ordering, normalization, NaN/default-str behavior, type policy,
UTF-8 failure, exception boundary, input descriptor or provenance comparison
changes. This is metadata, not tensor contents or rendered weights. Other model
walk algorithms, export gates and cache/residency contracts remain unchanged.
There is no performance or serving-quality claim.

`tests/test_model_walk_digest_routing_1785.py` independently reconstructs legacy
JSON/UTF-8/hashlib outputs and records the exact shared-owner byte input. Cases
cover empty/Unicode/CRLF rules, NaN/default-str config, identity prefixes, sorted
and nested inputs, empty descriptions, long Unicode truncation, unpaired-surrogate
refusal, default-input bypass, and both model fallback paths. Routing RED must be
executed before source delegation; subsequent existing walk/export-gate and
ratchet checks run through PB. Expected inventory change: 547 to 544, only the
three `hashlib` function rows removed, all other baseline fields unchanged.
