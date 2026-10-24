# T4 qualification/audit/recovery digest slice

## Verdict and integration boundary

Partial, blocked on the shared owner-recipes prerequisite. This is **not parent closure**.
`ledger.json` records 26 replaced research-base calls and executable independent-consumer evidence.
`residuals.json` records 24 exact current raw calls: 21 **eligible missing-owner gaps**, and
three retained independent malformed-checkpoint reference encodings. No eligible call is disguised
as a retention. Default scanner populations/exclusions, the historical census and the baseline
are unchanged. The aggregate ledger supersedes sub-slice ledgers for serial integration.

The 21 gaps require settled insertion-order JSON recipes (`sort_keys=False`): default-spaced
ASCII/lax JSON, its `default=str` variant, and two-space-indented ASCII/lax JSON with and
without `default=str`. This includes persisted summaries, not just stdout. The current owner
always sorts keys and has no matching recipes. Agreement S forbids this slice from editing
owners or constructing local profiles/wrappers. Return this exact gap to owner-recipes, then
finish these remaining consumer routes; do not use sorted profiles or round-trip normalization.
The source locations, explicit options and current blobs are in `residuals.json`.

The independent `tools/test_qualify_t4_overlay.py::test_partial_checkpoint_corruption_is_not_adopted`
retains its one sorted JSON call and two supplemental JSON calls. It constructs a valid cell
commitment independently, deliberately corrupts the verified-cell commitment, and checks that
adoption refuses without a manifest. The entire test source remains at its research-base blob;
`test_ledger.py` verifies that fact. It is not routed through the producer it checks.

Serial integration must consume `ledger.json` into the separate current census and shrink the
baseline for the completed scopes. The live scanner has **20 fewer primitive scopes and no
added primitive scopes**. Same-name `sha` membership shrinks by three actual definitions;
the fourth removed baseline member (`tools/benchmarks/joint_validation_pair.py`) is a surviving
singleton, not an edited file. This work deliberately does not change either shared record.

## Source-binding and deployment inspection

- `production_weight_cache.py::_production_cache_source_root`, `_production_cache_identity_paths`,
  and `_production_cache_source_profile` bind the package source closure, not sibling `tools/` by default.
  The four explicit sources in `aura_provenance.py` do not include these tools.
- `tessera_runtime/tessera_serving_runtime_pin.json` binds the Tessera repository/runtime
  contract, not these tool bytes. No runtime pin or qualified result is rewritten here.
- `build_t4_logical_request.py` emits the qualifier checkout and argv. Recovery preserves
  the request's executable and changes only its existing policy fields. Neither builder
  asserts a qualifier-source digest. No deployed file roster changes.
- `docs/results/2026-09-30_qualifier_sealed_cpu.md` records historical before/after qualifier
  tool hashes, separately from the package closure. Neither equals this research-base
  qualifier's SHA-256 (`34176faed30e1665861e758e77d42672dfde5475643eab10f1f6dfe887904198`).
  A historical measurement is not a current immutable-source retention witness. This slice
  does not repair that record or extend its qualification to new source bytes.
- Catalog source/H commitments, wire/render fences, reseal proof pins and qualified-result
  bindings are data contracts in `build_t4_overlay_catalog.py`, `rebind_t4_qualified_results.py`
  and `assemble_t4_overlay.py`; substitutions do not change their policies or values.
- `check_t4_overlay_prepare.py` still refuses source-model opens and declares its metadata-only
  boundary. Selected-cache audit still says source/H/encoder claims are not rederived.
- The formerly stdlib-only audit/inspection/recovery tools use the existing same-tree `digests`
  import pattern, without importing the PrismaQuant package. An isolated `-I -S` subprocess
  stages the tools and the existing owner, blocks package/torch/Tessera imports, verifies import
  safety and recovery byte hashing, and runs audit argument parsing. This is not evidence for
  an unobserved protected single-file deployment or authority to expand a deployment roster.

No current immutable source binding for the edited tools was found in inspected checkout
records. This is a scoped search, not an exhaustive audit of external deployments, generated
Python, shell populations, dynamic aliasing or qualification artifacts outside this checkout.
No GPU, serving, numerical, performance, or real campaign qualification is claimed.

## Independent evidence

Fixtures contain verbatim original consumer functions or their publication statements, not
expected bytes computed by a replacement owner. Original source blobs/commit come from the
frozen research registry. Tests do not require historical Git objects in test checkouts.
Tests exercise actual main functions or actual publication AST blocks; the latter deliberately
start with already-assembled small metadata records and do not claim to exercise all upstream
catalog science or the 864-qname census. The pilot decoder uses explicit CPU test adapters.

Covered behavior includes cell/verified-cell commitments, adoption without re-decoding,
malformed checkpoint and payload refusals, hash-before-unpickle, journal identity paths,
anchor-only rebinding, complete publication bytes and stdout, LF/pretty/compact distinctions,
Unicode/surrogates, lax nonfinite values, fallback objects in unchanged reports, mixed-key
refusal, exclusive publication and temporary/manifest behavior.

## Final checks

1. `python -m pytest tests/test_digest_1301/t4-qualification tools/test_qualify_t4_overlay.py tests/test_assemble_t4_overlay_original_pair.py tests/test_t4_overlay_catalog_anchors.py tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py -q --basetemp="$TMPDIR/pytest"`

   **296 passed**, 14 upstream Torch deprecation warnings.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6d20e245e765dbae7b95a2af48d5802531bb9b5f163fe0fcf80b1049c4b12da9","size":1142}},"operation":"e49017c4a967f9b2d4fa36574f11fba6640ad28c2214b3e78143b6ed94b973ab","receipt":"c2ead94c7ab57bf91437e1a76848b508f445cc5d7b5d082ebedaf754376fe672"}]

2. Execute the unique `tests` node IDs in this slice's `ledger.json.completed_calls` with
   `pytest.main`, using `--basetemp=$TMPDIR/consumer-pytest`: **13 executable node references,
   37 passed**, covering all 26 replacements. Parameterized functions are referenced without
   parameter suffixes.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"695cb7312fc59bbeecacef67b1b0681d5ad99066699b391a5487cc9a83fd699c","size":878}},"operation":"0e7cbe2df2e90f56c81809aeabaf89c58a75d6fea5e94adb979b53ba920b8a1d","receipt":"1b52ac01282a37bab8be54bf991c49ddab0f75abe71301e6807e6f04d96de4bd"}]

3. Full requested O5 command: `python -m pytest tests/test_digest_parent_completion_1301.py tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py tests/test_duplication_baseline.py -q --basetemp="$TMPDIR/pytest"`

   **200 passed, six failed**. Four parent acceptance assertions remain red (missing current
   census/completion evidence, unresolved calls and allocator routing outside this piece).
   Two baseline tests require the forbidden-to-this-slice serial shrink, not allowance growth.
   This is not a passing parent acceptance result.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f80b4924586918693d0b1362c84a75c442f6e8a0b5db84c4588f1b6f9ddb111b","size":16317}},"operation":"b3c5c39aeec91cae53e8ea89b1d56d35ecc6305344ac838038f86910694e43f3","receipt":"4815474875549eeb5fed0c6e99c5c62d4c34bc0836e69eb43da7c4e27115a63f"}]

4. Compiled all 12 edited tools and five new test modules using built-in `compile` without
   writing bytecode; scanner comparison confirms 20 primitive scope removals, zero additions,
   and reports the exact same-name removals for integration.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"63492c8e2f5c8f38c00cbb461bb30ef88e1b434e25e1dffdb124aded6c23171e","size":1401}},"operation":"269dc3c0611fcbb8d39ed11ca791d20c39ff290a93f6253c02a1bf97e161b137","receipt":"58eb0e9feb1055f0c6f5f4ab2815f57b6c36b647d84923f68ce1f901c543f25f"}]

Earlier sub-slice evidence and failed intermediate checks remain in their evidence records.
The initial combined run had 298 passes and two stale-baseline failures. A later standalone
import repair initially failed an alias-identity test and the import-safety test; both causes
were corrected before the successful final compatibility run. No full-suite run was made.
