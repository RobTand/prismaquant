# Tessera supply-chain digest slice — partial, not parent closure

Research source: `6d8288adc94b3c4b624c143ed372fb6513f32784`.

## Result and boundaries

- `ledger.json` joins all 89 frozen-registry calls in the 21 assigned files:
  **48 completed calls**, **39 unresolved exact-owner gaps**, and **2 retained
  descriptor-integrated publisher algorithms**. It also records all 41 current
  primary/supplemental calls in those files with current Git blob identities.
  Individual contract ledgers retain fixture/test details; the aggregate is the
  integration handoff, not an edit of the shared census.
- Replacements use settled owners only. Independent verbatim original consumer
  functions/blocks/expressions are committed as small fixtures. Executable tests
  compare actual publication paths as well as encodings, hashes, refusal/order
  behavior and descriptor scans. Ledger-resolution tests are metadata checks,
  **not** compatibility oracles. Original IDs always refer to the frozen base;
  live locations refer to the changed source.
- Existing profiles always sort keys. The remaining insertion-order recipes
  (including explicit `sort_keys=False`), sorted indent-1 strict JSON, and the
  provisioner's outer path/NUL/ASCII-content-hash/LF tree recipe cannot be routed
  to these owners without changing bytes. They remain explicitly unresolved,
  not reclassified as retentions. Agreement S forbids shared-owner edits here.
  Owner integration must supply those exact recipes before this slice can close.
- Publisher `_scan_file` and `_replay_frozen_digests` retain their incremental
  constructors because digest updates share descriptor-bound scan/replay work,
  inline capture, block/size accounting and post-upload verification. A pathname
  file owner would not preserve these contracts. Their specific witnesses are in
  `artifacts/audit.json` and the original-consumer tests.
- Embedded Python is a separate population: the original-image Docker query and
  isolated installed-target provisioner probe remain unchanged. Their execution
  environments do not supply the host owner. Source/layout witnesses are in the
  image/artifact evidence; the installed-target probe runs in an isolated CPU
  test. No new package is required in these target layouts.
- Host provisioner and progress reporter use the existing stdlib public-file
  loading pattern. Sparse checkout-layout `-I -S` tests prove no package
  initializer/Torch is needed. This does not claim copied-single-file deployment
  support. The provisioner acceptance wrapper resolves the host tool from its
  checkout root; the isolated target probe is a different environment.
- Current adapter delegation is preserved: no delegated package modules were
  edited, and no historical reseal/union algorithm was resurrected. The adapter
  reports with no exact owner are recorded individually as unresolved. The
  separately staged model worker is outside this slice's mutable footprint; its
  dispatcher's staging roster was not changed to add an owner.
- Supplemental boundaries: JSON parsing is not encoding; digest validators do
  not construct identities. Incremental constructors are separately dispositioned.
  Embedded/generated Python was inspected separately from AST call traversal.
  This bounded audit does **not** establish exhaustive dynamic-alias, generated
  code or shell coverage for the repository, nor independently close O6.
- No pins, qualification source, deployment rosters/authority, numerical policy,
  stored-identity widths, shared owners, scanner/exclusions, shared census,
  acceptance files or baseline were edited. No identity migration is claimed.

## Observed checks

Final focused check on the final source:

```sh
python -m pytest tests/test_digest_1301/tessera-supply tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py -q --basetemp="$TMPDIR/pytest"
```

**223 passed**, 14 Torch deprecation warnings, exit 0. This includes the actual
consumer compatibility tests referenced by every completed-call entry and the
sparse stdlib host-layout checks.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ccb27951df4ebd9789e68e8d20ef91e6e392ffa1115541f0bc945bb13660cc31","size":1063}},"operation":"6494de4d8a26282e4ccb83f4d4c8bed02bcdaf98be8ac1cfdfd2b0eee1a8cfec","receipt":"c20749937301cbb92bddfe690c25d2dd7b7b34b835a2e56314101fdaba0a42b9"}]

The broader parent command was also actually run (before the last stdlib-only
progress-reporter bootstrap adjustment):

```sh
python -m pytest tests/test_digest_parent_completion_1301.py tests/test_digest_1301/tessera-supply tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py tests/test_duplication_baseline.py -q --basetemp="$TMPDIR/pytest"
```

**228 passed, 5 failed**, 14 Torch deprecation warnings, exit 1. The failures are:

1. Parent current census not integrated.
2. Parent eligible calls unresolved.
3. Parent completed-call ledger not integrated.
4. Allocator Pareto owner routing (outside this supply slice).
5. `test_primitive_digest_sites_only_shrink`: the unchanged baseline still has
   34 now-removed gated scope rows. Its no-growth assertion passed; exact equality
   correctly refuses the stale baseline. Agreement S explicitly forbids editing
   the baseline here. The integrator must shrink only those genuine removals.

These are **not passing acceptance checks**, nor test defects/impossible
objectives. Shared census/baseline reconciliation belongs to the serial
integrator. Subtree evidence documents also retain broader affected-consumer
runs and their honest skips/harness corrections.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9cdd23ef614cc6af2c03a3cc3ee0ecd31ef2e38c86d981493dd5f87babcc94fa","size":15648}},"operation":"dd191ccca208e3e4d97dfa9dd9f1e4f21892004beff7397a328ff593b407e7c6","receipt":"6e40ca7362e86c561f6a75ab1dccd9b905550f4651cee531e7cec08f7e01f965"}]

`git diff --check` passed. A read-only independent review identified the initial
image ledger ID/path defects and insufficient plan/main behavioral coverage;
those were corrected and retested. Review also rejected treating the unusual
outer tree recipe as sufficient retention justification; it is now a gap.

No real image build, remote upload, fleet/campaign execution, package provision,
GPU/whole-model qualification or performance measurement was observed. Docker
and external/numerical dependencies in local behavioral checks are mocked or
stubbed; those checks establish only the stated CPU consumer contracts.
