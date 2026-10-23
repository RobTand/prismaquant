# Package-digest acceptance repair (#2648)

## Cause and scope

The supplied Python 3.14 CPU runtime publishes
`tessera.metadata.METADATA_API_VERSION == "tessera.producer-metadata.v1"` and
`parse_smoke_record(record, *, control, where)`, but no longer publishes
`tessera.serving.contract._evidence_smoke_record`. The four reported selector
subprocess failures reproduced before editing.

The repair adapts these two producer entry points in `lane_eligibility.py`.
The legacy four-positional-argument signature remains supported; the public
parser is used only when the legacy callable is absent. All grammar validation
still belongs to Tessera. Missing or incompatible capabilities refuse; a
producer rejection is never retried through another parser. Runtime pins,
contract data, selector assertions and admission policy are unchanged.

The v9 boundary tests now patch the module that owns record grammar in the
installed producer. Added cases cover both call signatures, accepted records,
nulls, exception chaining, and unavailable/incompatible public capabilities.
The legacy signature is unit-tested; integration runs below use the supplied
public-API runtime. No acceptance module was edited during this repair.

The package-digest route, its 22 frozen rows, completed census and CHANGELOG
entry were preserved. Recomputed with
`python tools/duplication_inventory.py --write-baseline`: 303 gated primitive
digest sites, with no new baseline diff (the package-digest raw row stays removed).

## Final verification

Command, exit 0: **627 passed**, 14 Torch/Python deprecation warnings; no skips.
This includes every requested acceptance module, the six review guard modules,
and the repaired v9 boundary suite:

```sh
python -m pytest tests/test_digest_profiles_1301_files.py tests/test_generate_transition_rewrites.py tests/test_select_validated_frontier.py tests/test_prefill_frontier.py tests/test_joint_replay_frontier_607.py tests/test_campaign_digest_profiles_2174.py tests/test_stage_a_chain_resume.py tests/test_duplication_baseline.py tests/test_source_framing_1752.py tests/test_framed_digest_owner_1301.py tests/test_tool_imports_1304.py tests/test_digest_profiles_1301.py tests/test_docs_staleness.py tests/test_architecture_doc.py tests/test_tessera_lane_v9.py -q --basetemp="$TMPDIR/pytest"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"566d9a111cef3b2ec76f7019c89eadff5ba2af69134c7eb529ce71e3ebcf1a39","size":1474}},"operation":"979d97d81b6727650c1f15a57e6511440cda197928c4a6c651aa28dc23be528d","receipt":"5f7af6e41125e0e7c9829126b370873ca744ea2b403a72d1d8fa9ce7049a7142"}]

A separate CLI subprocess ran the generator from an unrelated temporary working
directory, using the small sealed/executing trees from the existing frozen test.
Its stdout had an empty unified diff against
`test_package_digest_generator_output_matches_frozen#1` in the JSON fixture.
Exit 0; sealed package SHA-256
`f5f9a8a75a78ab06fde1e0c22ee0fd388e6fe41d5e983c54e641effafd0278fa`;
one rewritten file, one hunk, one new file. Temporary inputs stayed under the
runner's `$TMPDIR`; no historical checkout was needed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"81f66e10a16ca61e20da811ecd7bdaa30a62dc57de935ac13c09a3112d07ca9e","size":187}},"operation":"be13ce36f1f9f67117bfbef918ae1c00f19f4c989b79617ba510fd861fd3c127","receipt":"fc43ad44d7a8963c2187771f7b6885cfccc828ffb34eb801e09fa3bf4affdd41"}]

After the final documentation edits, reran
`python -m pytest tests/test_docs_staleness.py tests/test_architecture_doc.py tests/test_tessera_lane_v9.py -q --basetemp="$TMPDIR/pytest"`:
**88 passed**, exit 0.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"920265ad07f7c9278c97baf13db951962a239af230706f62167e290f92c4ab64","size":901}},"operation":"739ca1bf96556885af8beb95e82c7258e50ed45a794860abc07e8e840b5fbe01","receipt":"f3ef7c8ee48fada5294bee71272eae3559e358917b8c9a804bb9858ea710e198"}]

## Earlier runs (not passes)

- Four reported selector tests, before editing: 4 failed, 55 deselected, exit 1;
  all stopped on the missing private producer validator.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3b5554d5ac9b80ee5013882ac0ad8763c13e88b7d84538550cd48f3173b5697d","size":31222}},"operation":"bb4ea7ef3d51ace7796714b6821572824fa83fa90b08afc1f1d8f19463f06a07","receipt":"b96aa573ef3ce42461a5e35bc883644df5b82b91a3b75805cf271acf64011f17"}]

- API inspection followed by `python -m pytest tests/test_tessera_lane_v9.py -q
  --basetemp="$TMPDIR/pytest"`, before editing: 52 failed, 8 passed, exit 1.
  This exposed the same missing-validator boundary and legacy-only test patches.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"86a4137df4075954519f379d8070819ba9d99b9413ec89403445bb55e967cfe0","size":117634}},"operation":"45838fb6111607985e0a23873e17fe8ae062c2027ce0c7bfd7ae077b513d97f7","receipt":"fe93170556c363643c8d1ae421a3ddd62fac7c6fb3506ab8d8c00e2a7d1ca8b2"}]

- Intermediate `python -m pytest tests/test_tessera_lane_v9.py
  tests/test_select_validated_frontier.py -q --basetemp="$TMPDIR/pytest"`:
  1 failed, 127 passed, exit 1. The selector tests passed, but the vocabulary
  ownership test still patched the old module. Moving that patch to the actual
  producer owner resolved it; its widening/refusal assertions were preserved.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"1e274582e29e1243c1084a3aa7c4ad652d74303d19d077ef89c54893d247603d","size":10547}},"operation":"9baba6489833d2eb16de3139c76861f3a01687fa715bd87db000f030bc80c5b0","receipt":"691514b748c2891c620550fd30b1e5a82b142c0a143a1bce0be15b63510c3db3"}]
