# Native-export digest slice: prerequisite not yet supplied

This is a **blocked handoff**, not parent-closure evidence. The inspected head is
`8bc0b5af` and all six inspected consumer blobs still match research base
`6d8288adc94b3c4b624c143ed372fb6513f32784`.

`ledger.json` records 16 eligible calls blocked on the owner-recipes prerequisite
and three read-only, supported-deprecated GGUF calls. Original call IDs and blob
identities come from the frozen research registry and were checked against the
current source with `git hash-object`. The registry is provenance, not an oracle.
No call has been replaced; consequently `completed_calls` is deliberately empty.
No independent compatibility fixtures or new consumer-test coverage are claimed.

## Required shared-owner work (Agreement S)

* Insertion-ordered, default-spaced ASCII/lax text JSON: two `_json.dumps(entry)`
  sites in the native renderer and the export-structure error record. Existing
  `JsonProfile` always sorts; sorting can reorder output and can reject mixed
  int/string keys that the original encoder accepts.
* Insertion-ordered indent-2 ASCII/lax streamed JSON publication: six native
  exporter sites and four RTN staging sites. These use `json.dump`, not a
  whole-document string. Preserve partial writes/refusals and publication policy,
  not only successful final bytes. No matching publication API exists here.
* Exact incremental NVFP4 u32-little-endian string framing: three stage/routed/
  target constructors. Existing u64-big-endian or NUL framing is incompatible.
  Scalar conversion, F32 representation, schema/policy fields and roster order
  must remain consumer-owned. Do not substitute a joined whole-document buffer.

These are blockers to the slice, **not retention justifications**. Shared owner
changes are outside this slice's permission. Resume after owner-recipes is
integrated, then freeze independent actual-consumer outcomes/source fixtures,
route calls, and record per-call executable test references. Existing native
fingerprint routes already owned on entry are accounted separately and must not
be counted as new completions.

## GGUF read-only accounting

`AGENTS.md:7-9` explicitly deprecates GGUF and prohibits fixing, enhancing or
extending it. Its assignment SHA at line 408, assignment serializer at line 409,
and incremental imatrix SHA at line 455 remain unchanged. The assignment digest
is embedded at lines 438-439; the imatrix digest is computed over sorted names
and F32 CPU numpy bytes at lines 455-459 and embedded at lines 462-463. This is
supported deprecation/no-refactor accounting, not an independent new workstream
or a claim that the recipes are intrinsically unmigratable.

## Verification

Executed on CPU:

```sh
python -m pytest tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_export_structure.py tests/test_build_rtn_cache.py tests/test_duplication_baseline.py -q --basetemp="$TMPDIR/pytest"
```

Observed: **195 passed**, 14 torch deprecation warnings, exit 0. This checks the
unchanged existing contracts, not migration completion. Parent acceptance and
unavailable new consumer compatibility tests were not run. No GPU, numerical,
rendering, serving or performance qualification is claimed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"54c052b7dba40e4d7c4b2fc777c7ca382af200c58617389d882b8a18727e59ac","size":983}},"operation":"261d7cf924193d4c177c28ad221c5af21487c60233654f30a337f40ded94b2bc","receipt":"5e04028394ecbf614142a026970840391dcb6ddbe1f9d027046b90cfa1431ec9"}]
