# Quality/calibration digest slice — partial delivery

This extends only the existing digest owners, through consumer routes. No owner,
shared census, baseline, scanner, acceptance fixture, qualification record, pin,
or deployment roster was edited. The inspected research base is
`6d8288adc94b3c4b624c143ed372fb6513f32784`.

## Delivered and independently checked

* 21 exact owner substitutions in 13 tools: eight byte/text hashes and thirteen
  sorted JSON encodings. Unicode, NaN/Infinity, mixed-key and fallback refusals,
  result widths, tensor-byte interpretation, publication bytes/order and stdout
  are exercised where applicable by original consumer block fixtures.
* `ledger.json.completed_calls` has 22 research-base call records: the 21
  substitutions plus the enclosing recheck report call whose AST changed solely
  because its nested hash now delegates. That report's serializer remains a gap;
  it is **not** claimed as a completed JSON route.
* Each completed record has original ID, commit/blob, exact owner route,
  independent fixtures and executable repository test nodes. `test_ledger.py`
  compares the exact replaced-call multiset against the immutable research
  registry, checks test-node resolution and verifies all current local residual
  call locations, expressions and blobs. Metadata alone is not compatibility
  evidence: the referenced consumer tests execute original and current blocks.
* Corpus, EXL3, MTP and prompt-smoke tools use the existing stdlib source-owner
  loader, avoiding package/torch startup in lightweight import contexts.
* The eight G3 tools are individually audited in `g3/ledger.json`, including all
  23 native JSON encoders, their options, consumers and native launch constraints.
  Generated/delegated code and shell/deployment boundaries are recorded separately.

The original fixtures are small source blocks, not expected outputs generated
by the replacement owner. Full numerical/model computation is outside the
CPU serializer/publication tests; supplied records and engine doubles are
explicitly labeled. Corpus constants were not changed. Scientific calculations,
scoring windows, defaults and tensor conversion chains were left unchanged.

## Not complete: return to shared-owner integration

46 native JSON calls remain unresolved, individually listed with current source
identity in `ledger.json.unresolved_calls`. None is claimed as a retention.
The existing owner fixes `sort_keys=True`; it cannot represent these unsorted
recipes. Other missing combinations include sorted spaced UTF-8/lax and sorted
indent2 ASCII/strict. Missing unsorted recipes include ASCII and UTF-8, default
spaced, compact, indent1, indent2, and default=str distinctions. Five `json.dump`
writers additionally require streaming/partial-write/error-order compatibility,
not `fh.write(profile.text(value))`. The remaining tests demonstrate ordering,
nonfinite, fallback and stream-prefix differences. Agreement S prohibits adding
these owner recipes in this consumer slice. No local replacement owner or
per-consumer encoder abstraction was introduced as a workaround.

The serial integrator must supply independently demonstrated exact recipes,
finish these routes, and reconcile the shared `current_1301` census and baseline.
The local ledger lists exactly 18 removed scope rows. The unchanged baseline
therefore fails its exact shrink check; there are no new primitive scope rows.
Parent allocator/package closure is outside this assigned footprint.

## Independent review and qualification limits

A separate read-only review inspected all 13 tool diffs, original fixture
independence, imports, scientific-code preservation and recipe options. It found
no changed scientific calculations or observed byte incompatibility. It flagged
five historically reverted expressions in replay, both quantizer reports and
refusal normalization as requiring separate native/producer qualification rather
than interpreting CPU block tests as that qualification. The historical audit
records and old qualification artifacts were not rewritten. New tests establish
only original/current publication and refusal-block equivalence on CPU fixtures;
they do **not** renew those native/producer claims. In particular the unconditional
`check_stageb_a4_quantizer` module was not invoked with a fictitious help mode.
No real GPU, vLLM, checkpoint, Docker, campaign or performance qualification ran.

Supplemental native enumeration is exact for these 27 lexical source files,
not proof of exhaustive dynamic aliases, generated Python, shell commands,
external staged files or transitive imported implementations. See the G3 and
remaining audit fragments for explicit boundaries and current witnesses.

## Final verification

Passing compatibility run:

```
python -m pytest tests/test_digest_1301/quality-calibration tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py tests/test_duplication_baseline.py tests/test_exl3_grouped_down_replay.py -q -k 'not test_primitive_digest_sites_only_shrink' --basetemp="$TMPDIR/pytest"
```

Observed: **326 passed, 1 skipped, 1 deselected**, 14 upstream torch warnings.
The one deselection is the intentionally unmodified shared baseline's exact
shrink check, not hidden coverage removal. The skip is existing compatibility
coverage, not GPU evidence. All local consumer tests and the local per-call
registry check passed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"585a8e6bbf2eff5c2d3ef0adb1695c95edb39a6f5fe15b788e7aff13c671ec17","size":1168}},"operation":"e551893b162baf645e8a6eb2063617f9be7c9a47c434a8a4a97c0fa2660a7ac4","receipt":"b9c113c36294de891ef6762cc755e964f1e8b4e24c03beab1b7c8619544af0af"}]

Full parent/ratchet check was also run without deselection:

```
python -m pytest tests/test_digest_parent_completion_1301.py tests/test_duplication_baseline.py -q --basetemp="$TMPDIR/pytest"
```

Observed: **5 failed, 6 passed**. Four parent assertions still fail on shared
census, unresolved calls, shared completion ledger and allocator routing. The
fifth requires the shared baseline to shrink by the 18 completed removals.
These are integration/outstanding-work failures, not passing checks or
acceptance-test defects. Parent completion is not claimed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9d11ab24c88448f6c05acd371b6f2dd14ed22d1d3a78f8e8a28a54587e2eb90e","size":14397}},"operation":"e163a1167a8dd9eeef4e77e5694b0ea48ea24d454909e32b2c669d9a61cf0311","receipt":"8644b571be45098dbdc80addfd871c46be2e44d5c1336e8440d5ef06282f8601"}]

Earlier fragment evidence includes the failed development checks as well as
repairs. The combined development run first exposed a stale vLLM audit hash;
that test now uses the frozen original fixture after the vLLM migration.
The first local ledger check also exposed the changed enclosing recheck call;
its nested-only change is now explicitly accounted without claiming its JSON
serializer migrated. Final `git diff --check` passed. A separate admitted `compile()` check compiled
all 27 assigned tools and seven local test modules without writing generated
files (34 modules, exit 0).

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9906a76272aaf94e8003c9c82e44d56e3d59208d06cd919c464fc002b6975c15","size":58}},"operation":"90d7dd8c5b085d9b4db6eafc4192f1cfebafcf1bd884b5f0cb65703371298694","receipt":"507566963902d094d5574d66b07c027a63cf4e448d845cd4c26a8f2a8f1afc9e"}]
