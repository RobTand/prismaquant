# Reader SHA-256 grammar: #3316

## Changed sites

`prismaquant/tessera_reader.py::load_declared_reader` now calls
`digests.is_sha256hex` instead of its bare regex.
The digest import adds the predicate.
The file removes the unused `re` import.
The caller keeps `str(record['source_sha256'])` and its error.
`prismaquant/digests.py` owns the same full-match grammar.
It accepts string subclasses.
Dependency edges stay unchanged.
Callers remain in `g3_v2.py`, `joint_cost_quantum.py`,
`joint_cost_stage_a.py`, and `tessera_joint_aura.py` under `prismaquant/`.

## Evidence

`tests/test_tessera_reader_namespace.py` records real predicate calls.
Base reports `wrapper did not record the owner calls`.
Receipt: `8d292f90505cd91ab2545ac4a17977333c64d894ac57c717fcfb5e2f6be46d9a`.
Checking must confirm this advisory proof.

Controls pass before the production edit: 13 pass, zero skips.
They cover valid text, subclasses, uppercase, malformed text, length,
nonhex text, newlines, and string conversion.
They check early refusal, isolation, primary modules, and source refusals.
New tests remove only their own namespaces.
Both test modules pass on head: 22 pass, zero skips.
Receipt: `f0f725fc11e47ce19efe9bc5d2d008e69beab8fbbd82050ec94ec1dae7f67f91`.

## Residual scope

Source profiles, source-byte checks, v2 namespaces, memo ownership, cleanup,
caches, descriptors, defaults, pins, and gates stay unchanged.
The duplication inventory covers structural pairs, helper names, and primitive
digest sites, not bare SHA-256 regex checks.
Its shrink-only baseline stays unchanged.
The retained census stays historical.
#3061 retains inventory integration and scanner gaps.
#3061 stays open.
#1301 recipes and #2922 wire reuse remain unchanged.
