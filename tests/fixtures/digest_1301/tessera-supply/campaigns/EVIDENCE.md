# Remaining campaign owner routes

## Scope and provenance

Only the nine assigned tools and this campaign fixture/test package were changed.
Pre-existing delegated changes in the images/artifacts packages were not edited.
No digest owner, shared census, baseline, scanner, or acceptance files were changed.
The existing stdlib-only `prismaquant.digests` owner is used without new profiles.

`ledger.json` accounts for all 35 acceptance-registry call fragments in the nine
assigned tools: 19 replacements and 16 unresolved recipes. Every entry carries
the original acceptance ID (including supplemental-json IDs), source commit
`6d8288adc94b3c4b624c143ed372fb6513f32784`, source blob, and fixture path.
Replacement entries also identify the exact owner route and unparameterized
pytest function AST nodes.

The per-tool JSON fixtures contain verbatim original call expressions extracted
from that commit, not owner-generated expected values. The small Python fixtures
contain verbatim original function definitions for independent behavior checks.

## Independent-review follow-up

This follow-up changes only this campaigns fixture subtree and
`tests/test_digest_1301/tessera-supply/test_campaigns.py`; production is unchanged.
The scope and broad results below describe the earlier migration, not additional
production edits in this follow-up. Ledger references now use plural `fixtures`
arrays containing plain repository-relative paths.

Actual old/current `cmd_plan` execution now compares selection, data, resource,
plan and manifest publication bytes and sorted group membership; it checks data derivation precedes publication, model refusal and
read-set derivation refusal without output mutation. The prior census/capture
test remains attached only to those routes, not `cmd_plan`.

Actual old/current `build_tessera_census_cache.main` now compares projected config,
manifest, summary, hash links, stdout, publication order, existing-output refusal,
input-digest refusal and outside-roster refusal before projection publication.
Actual old/current `reselect_mtp_fixed.main` compares complete layer-config bytes,
body preservation, rebased budget metadata and stdout hash, and tests existing
output, mismatched input digest, publication-race and changed-body refusals.
Numerical/external dependencies are CPU stubs; these are small actual main/function
execution tests, not live cluster or full numerical integration tests.

New `.py` fragments were copied verbatim from the pinned original commit, not
reconstructed with current digest helpers. `function_sources.json` records source
blob and fragment SHA-256; a test checks fragment integrity. The **15 unsorted +
one indent1 strict** recipes remain explicitly unresolved, even where their
behavior is now covered.

### Follow-up check results

Focused command:
`python -m pytest -q tests/test_digest_1301/tessera-supply/test_campaigns.py --basetemp="$TMPDIR/pytest"`

Result: **11 passed**. The first follow-up run had 9 passes and one harness
failure because the old dispatch fixture did not yet include `cmd_plan`. The
verbatim old function was added; subsequent runs passed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"7739372e3583ea0223d9633d35b1d322740485767744e18bd6e1a40853311e50","size":269}},"operation":"6f7e92ef8291a5a2dd1d1d597599f44cb078cb7445fe0562bdece6f8c8408354","receipt":"20d363a518abaa1e83e017fe34891c684d46a5a41dee4fb37d9104c2b675a1be"}]

The expanded command listed under earlier migration checks was also rerun in this
follow-up: **161 passed**, 14 PyTorch deprecation warnings.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ae6ccbb31e9259162e07ea22e630ea64e4bf6c35cd3ad85c7491ce9f50131a3d","size":983}},"operation":"0247ed4f98fcefd2c65e306c59e6940505c6d12bdeb99baff80866b55027b6ca","receipt":"5f1848ffe2bc69c9ecab0aeafcb0a22adbf7eff47eea513b775ac85772cedcc5"}]

## Earlier migration checks

The seven new tests compare actual current AST call expressions against retained
original expressions for serialization bytes/text and errors, with payload-only
substitution. They cover Unicode, escaping, negative zero, NaN/infinity,
unsupported objects, mixed keys, and surrogate strings. Original/current function
execution additionally checks bound reads before unpickling, digest mismatches,
probe validation before publication, roster and probe sidecar contents/order,
replay nofollow publication/idempotency/refusals, census/capture resource files
before submission, capture model refusal, union dry-run/refusal, and PrismaSnap
strict output. External campaign/numerical dependencies are stubbed in these
small function tests; no live submission or GPU workflow was exercised.

Final command:

```sh
python -m pytest -q tests/test_digest_1301/tessera-supply/test_campaigns.py tests/test_build_tessera_selected_cache_bind.py tests/test_tessera_census_cache.py tests/test_tessera_anchored_surface.py tests/test_union_tessera_census_tables.py tests/test_tessera_campaign_cli.py tests/test_dispatch_digest_routing_1670.py tests/test_campaign_digest_profiles_2174.py --basetemp="$TMPDIR/pytest"
```

Result: **157 passed**, 14 existing PyTorch deprecation warnings. An earlier run
had one new-test harness failure because extracted production functions replaced
a submission stub; the harness now reapplies injected dependencies after loading.
The final rerun above passed. `git diff --check` also passed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"d228189099d234403e360a23692e73d24ded2b5995e3dd5093ac51c1e958c9c3","size":983}},"operation":"b9f4cfc41e41c6d1083f6011acbad9e7798cfcb376aa432947d78e843588f9d1","receipt":"65d5ff79487c057195527352b8bc3f33fa1fd97f2c1d04081519d20413727fc4"}]

## Unresolved, not retention

The ledger's `unresolved` list records **15 unsorted recipes** and **one sorted
indent1 strict recipe** (the sealed roster encoding). They are left unchanged
because no existing settled exact profile matches. This includes supplemental
unsorted publication/console recipes in census-cache building and dispatch,
and the unsorted strict compact reselect output. These are owner-profile gaps,
not claims that hashing/serialization must be retained locally. No acceptance
or ratchet relaxation was made. Full repository acceptance and live operational
workflows were not run.
