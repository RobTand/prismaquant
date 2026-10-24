# Read-measurement leaf handoff (Agreement S)

This is partial implementation and integration evidence, **not parent closure**.
The existing digest owners lack several exact recipes required by this assigned
population. Agreement S forbids extending owners in this leaf; blocked calls
were not disguised as retentions or moved to nearly equivalent profiles.

## Current accounting and independent review

`ledger.json` is the consolidated handoff: **105 frozen-base calls**, partitioned
into **28 completed**, **4 narrowly retained reference/workload calls**, and
**73 blocked eligible calls** across the 21 assigned files. Primary and
supplemental encoding calls remain distinct. Original IDs/blobs refer to
`6d8288adc94b3c4b624c143ed372fb6513f32784`; residuals additionally record exact
current IDs/blobs. JSON readers are a separate audit boundary, not encoding
retentions. The frozen registry proves provenance only; expected compatibility
comes from original source fixtures and actual consumer behavior.

`test_ledger.py` checks unique base membership, per-call fixture/test references,
AST-resolvable literal test nodes, and exact current residual membership/source
blobs. This leaf adds no scanner population or exclusion. Independent read-only
review found and prompted correction of the file-digest acquisition route,
shifted live-report provenance IDs, and misleading blocker/retention wording.
All twelve frozen window-consumer function ASTs and the live-report snippets
were independently compared against available original Git objects during
implementation. Tests use committed fixtures/registry, never historical Git
objects. An exploratory Git-backed test was removed because execution snapshots
need not have that history. That earlier unavailability is not a source defect
or a compatibility pass.

## Contracts preserved

- `joint_validation_pair.sha` retains `path.open('rb')`, caller close ownership,
  and stdlib file_digest through `file_digest_sha256hex`, not a differently
  chunked file reader. Chain hashing retains explicit 1 MiB reads.
- Tensor view/memoryview and bounded conversion expressions stay with callers.
  No whole-tensor copying owner was substituted for a view or window.
- Setup writes/receipts, serialized metadata feeds followed by tensor bytes,
  highest-protocol journal payload/envelope pickling and memo sharing,
  checkpoint comparisons, output files, stdout markers/newlines and refusal
  order are exercised against independent original-consumer functions/blocks.
- Host-side window tools and the paired validator reuse the existing
  `pq_profile_digest` loader instead of introducing a new loader or importing
  the ML package for stdlib-only work.
- Four reference retentions are specific: fence `_read_digest` file_digest
  reader; fence `_busy_consumer` synthetic wire-sized zero-buffer CPU work;
  fence `main`'s four-iteration in-memory SHA throughput reference; wire-reader
  `_hash`'s unbuffered fixed readinto loop. CPU tests witness handles, reused
  memoryview buffers/partial reads/EOF, feed bytes and take/work/release order.
  None of these reasons covers setup hashing or JSON reports.
- No owner, shared census, baseline, scanner, acceptance fixture/test,
  deployment roster, qualification source, runtime pin, numerical policy or
  default was edited. No stored identity migration or width change.

## Outstanding work for shared-owner/integration owners

The 73 eligible gaps are individually listed in `ledger.json`, not deferred
under a generic retention label. Required exact recipes include insertion-order
JSON (spaced/indent-one/indent-two and strict UTF-8 cases), sorted indent-one
without default=str, actual `json.dump` streaming publications, raw binary
SHA256 output, and incremental unframed byte/metadata/tensor feeds. For example,
fence setup hashes each 1 MiB generated piece immediately after its write,
before stat/ctime drift: a post-write file pass is not compatible. The tensor
leaf prefix recipes differ from `tensor_digests.tensor_hash_update`, and
window/projection receipts must not gain a whole-tensor copy.

The serial integrator must import the completed-call ledger into a separate
current census and shrink exactly the 21 now-absent baseline scopes reported
by the ratchet. This leaf deliberately does not edit either shared file.
`refresh_2540` and historical records remain untouched. Remaining parent-wide
census/eligible/completion-ledger and allocator failures are outside this leaf.

## Final CPU verification

1. `python -m pytest tests/test_digest_1301/read-measurement tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py -q --basetemp="$TMPDIR/pytest"`

   **303 passed**, 14 upstream Torch deprecation warnings, no skips.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"7624514a50af2fa6e5cd6aa44023f06a9cde854dd708d379542e748084115131","size":1142}},"operation":"e5ffa16280f23705bec9ea01eb924880ed2fbf17e819f577ff7e0530cf3caea1","receipt":"5b59ef25a41495c3237742acf2f11eaefe2d6aee33d4767c344cff4dd8d54e56"}]

2. Executed the exact deduplicated literal test nodes from every
   `ledger.json.completed_calls[].tests` using the same pytest-node driver as
   O3, with the local ledger substituted for the shared census:
   **43 passed**, no skips.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"20f1d18b7e1180c165d8fd23dcf9d78c47aa96bdaab26fdb0aa468650e036dfc","size":269}},"operation":"45034f7dc76f5b2c7ac1f7c8e8b3598cd4a87492d6eef00a56d0fe54c023d7fe","receipt":"2d69fb577a4eab192259664d907b2dc1221aa20196322c5d4f1789e7fe7c3ba3"}]

3. Full requested O5 command:
   `python -m pytest tests/test_digest_parent_completion_1301.py tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py tests/test_duplication_baseline.py -q --basetemp="$TMPDIR/pytest"`

   **201 passed, 5 failed**, 14 warnings. Failures: the four existing parent
   completion gates (current census, unresolved consumers, completion ledger,
   allocator owner routing), plus the primitive ratchet's required removal of
   21 stale baseline rows. No new raw scope was reported. This is **not** O5
   green; serial shared-file reconciliation and other parent work remain due.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"df10936182b3b5d665a6a3019fe1974c33179545e63dc25cc2d0b02a96381f0e","size":14728}},"operation":"aa6805ddd77c5760c4138213fc3eb8a1f649aff7dcaf11e463682800fecd7be0","receipt":"da55bdc8f1af912e6d7ab6a12cb4e7935318a3da46982c4eb4c85288dd14e5be"}]

Earlier intermediate combined validation was 308 passed/1 expected stale
baseline failure; the first new ledger test then failed because it referenced
an unavailable private scanner helper. It was repaired to use the inspected
existing predicate, without changing the scanner. Final results above supersede
those attempts. `git diff --check` passed after final source review.

These are CPU contract checks, including real tiny tensors/files plus explicit
IO/telemetry/producer adapters. No GPU, fleet campaign, full workload,
qualification, numerical-equivalence or performance claim is made. Supplemental
aliases/dynamic options, validators, generated code, and shell/deployment
populations outside this fixed Python footprint are not claimed exhaustive.
