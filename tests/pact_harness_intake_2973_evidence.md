# #2973 acceptance-test preparation evidence

Research changed only this test note and `test_pact_harness_intake_2973.py`.
No application code, campaign source, frozen execution copy, runtime configuration,
or source-identity record was changed. Source inspection is not execution proof.

## Design decision before acceptance-test construction

Use a fixtures-only source-intake package, not another harness implementation.
The missing capability is an accepted, portable source/caller/input baseline.
The current repository's `prismaquant/pact_frontier_profile.py:1–18` is a profile
bridge and names external workspaces; it is not the actual replay harness.
Pin original comparison bytes, normalize their provenance separately, retain the
Campaign declaration, and run the original source through a small test driver.
The driver must not reconstruct the approved correction, replace scientific
functions with stubs, replay golden JSON, or call future migrated implementation.

Rejected alternatives: use the old snapshot as canonical (wrong entry guard);
apply the documented patch and declare the result authoritative (not source
custody); treat observed imports as all callers (different populations); import
the harness into application code now (owned by #2974/#2975); change deadlines
or migrate the price adapter (#2520 and #2960/#2961 already own those).

## Read/identity findings

The following are evidence paths, not runtime requirements for acceptance tests.

- `/mnt/shared/tessera-measurements/eng-pact-band-20261009/pq2482-canonical-a8`
  has 30 Python files. Entry SHA-256:
  `6e6f2101ed991a468f5aaaf1be7258a463c6818a52b767c6badf6108f2eb25ec`.
  Its parser defaults to 1700 and resolver rejects deadlines above 1700.
- The adjacent `pq2482-source-equivalence-a8.json` has SHA-256
  `76bb6c75f1d79c8cef81b848d7493b4f5fd35e18f5f8e336ab9e5852b5f294c2`.
  It reports CLI head `c9260081766ec8a47e7a0f357432833a4d3b1cb6`, the
  3500 correction, 31 imported files, and 95 observed project-module records.
  Those are historical report claims, not fresh executions during this research.
- **Recovered authentic Git bytes:** the provided PrismaQuant object store
  `/home/rob/.local/state/trustladder-live/prismaquant/repo.git` contains c926
  and `ab6bfb843b85ac61b96693d13bc7e9463f004046`. Research used `git cat-file`,
  `git ls-tree`, and `git show`; acceptance tests must never use those objects.
- At c926, `experiments/pact_replay/multi_stream_replay.py` hashes to
  `c661efc8228545b91e97b6c5d50bf5d35f922a1ca47e3b0e00ff081c142a1eaf`.
  The original `IMPORT_PROVENANCE.json` hashes to
  `3e50ec0afe04f04ea6d1db90a035aa80ab42d8aff824740665446a839ce76da2`.
  Its 31 imported files are **30 Python files plus API_CONTRACT.json**. This
  reconciles the apparent missing 31st Python file without inventing one.
  The old snapshot's `old-D38_IMPORTS.json` hashes to
  `48d1432271666974e28445ccc6f7648b74a2cb81d78fd5806e6360c0b66c9729`;
  its different entry identity must be explained, not overwritten.
- At ab6, the entry hashes to
  `2a4e67430a25e76c3d34b649f71239d5763f128f682347bdbf5ec24ea77855b4`.
  It adds `replay_progress.py` (SHA-256
  `a21127cd614770addcc2fa0949a62caecc26bf00e7c1436e2bbe1bbe0ae0cc67`).
  Its provenance explicitly separates historical import hashes from #2520
  post-import progress/stall changes. There are now 31 Python files.
- `/mnt/shared/tessera-measurements/eng-pact-band-20261010/sequential-ab6bfb84-01/approval-packet.json`
  hashes to `abcf93676a10b7a95d393168b46cc5f9ae270fe636e77adefd970e460e784faa`.
  It names ab6, application stall 1800, payload backstop 7200, PB phase grace
  1560, and no fixed application duration cap. This later generation is NOT
  the earlier default-1700 / approved-maximum-3500 deadline correction.
- The c926/ab6 dependency inventory distinguishes 94 inferred external source
  files from 95 observed project-module names. Declared dependency identities:
  G3 `a23165b979a90536e060a2c99ea26930d5445a59`, PQ-owner
  `931085dbdf9cc321e3e586c6a47cc62f970be978`, Tessera
  `b40c93cb73745097e57a1ba4cf5b9eee166c759a`. Retain their per-file records.
  Some historical Tessera-pin bytes were unreadable; do not claim a complete
  independent rehash of that mount or silently substitute main's installed pin.

Useful recovered test sources at ab6: `tests/test_pact_replay_cli.py`,
`tests/test_pact_replay_stall_2520.py`, `tests/test_capture_pact_replay_imports.py`,
and `tools/capture_pact_replay_imports.py`. At c926,
`tests/test_pact_replay_deadline_2482.py` contains the legacy/corrected resolver
fixture. These are intake material to inspect/reuse, not commands that current
shallow-checkout acceptance tests may recover from git.

A retained bounded CPU evidence candidate is
`/mnt/shared/tessera-measurements/pq2520-stall-a3-final/report.json`, SHA-256
`194114804d042c727e5e2e0e62b5d2231c70dcba8bb495f1879465d7eab48203`.
It records a compact three-stream synthetic CPU fixture, NOT the seven-stream
full GLM run. Its omitted generator `tools/_pq2520_smoke.py` is associated with
snapshot `658cba21f75134e0ffb5c79b026acbe8ed6e6337` and CAS bundle
`7c73d3061fc2e1d738c22393a3547192ae73b384cb2a8f74a74741bb02ccc318`.
Do not claim the retained report alone is an independently runnable oracle.

## Remaining handoff gap and duplicate boundary

A complete current Campaign declaration of harness sources and callers, the
chosen approved migration baseline, and exact frozen-copy/cutover dispositions
were not established by these reads. Known candidate callers include
`profile_entry.py` -> `bounded_profile.profile_run` -> `runpy.run_path(entry)`,
the public `multi_stream_replay.main`, the capture observer, omitted proof
scripts, and external launch wrappers. Do not infer that list is exhaustive.
Recovered source existence alone is not authority to select or cut over a run.

If still absent after obtaining the Campaign handoff, the precise #2484 report
is: supply the owner-declared caller/source/cutover manifest tied to the chosen
corrected revision, any omitted proof-generator/input closure needed for the
portable canonical public-entry CPU oracle, and baseline-selection authority
reconciling c926's 3500 deadline with ab6's later stall policy. Do NOT request a
nonexistent 31st Python file or re-create the correction from a nearby snapshot.

The supplied open issues already name #2999 for immutable-source delivery,
#3000 for exact corrected-source approval, and #3008 for existing producer
handoffs; coordinate with those existing owners rather than create duplicates.
Their titles alone do not establish completed delivery. #2974 owns harness
import/profile use, #2975 owns caller migration/equivalence, #2960/#2961 own the
price adapter. This package records those boundaries, not their implementation.
No issue comment was posted during research; publication belongs to TrustLadder.

## Fixture interface and review limits

The new tests define `tests/fixtures/pact_harness/manifest.json` with original
`deadline` and `stall` source generations, evidence descriptors, declared caller
mapping, dependency identities, frozen-copy inventory, approved policy selection,
and oracle cases. A normalized `campaign_handoff` record includes `authority`,
`source_evidence` (digest-bound raw handoff artifacts), `canonical_revision`,
`declared_callers`, and `frozen_execution_copies`; it must cite a real declaration,
not an implementer-authored completeness claim.

Case JSON contains `bindings` as {path, sha256} descriptors plus arguments and
source references. Each case's stable expected JSON is committed separately.
`oracle.py --case FILE --output-dir DIR` must run genuine original source and
emit `result.json`; result fields are documented/asserted in the test. The
acceptance test copies only the fixture bundle, removes its manifest and all
expected JSON before invoking the oracle, clears deployment/Python-path/progress
environment overrides, and checks original input/source bytes remain unchanged.
The GLM numerical dry-run is distinct from resolver cases. Keep any new bounded
fixture-generation provenance explicit; do not present synthetic or reduced
inputs as original full GLM measurement inputs, nor claim GPU qualification.

Hash consistency cannot prove authority, absence of undeclared callers, or lack
of scientific stubs in a new driver. The reviewer must inspect raw handoff
provenance, actual source invocation, golden generation and normalization, all
input/dependency bindings, portability, and the absence of production/frozen-copy
changes. No new application path constants, shared-profile implementation,
scheduler, deadline correction, or price-adapter migration belongs here.

## Executed acceptance check

Command (repository root):

```sh
python -m pytest tests/test_pact_harness_intake_2973.py -q --basetemp="$TMPDIR/pytest"
```

Final result: **4 failed**, all at the explicit missing committed
`tests/fixtures/pact_harness/manifest.json` delivery assertion. Collection and
imports succeeded. The original harness, deadline/stall execution, numerical
equivalence, GPU and frozen-run behavior were NOT freshly validated.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"a83b7fe77d9b05dc90b1c447ec6b3d44b8fbf33b1cab73b245fa3faab3e1b35b","size":5307}},"operation":"10f85a16a5e63e28a5d4be22ba6e419bbff66829f2a0a095cbda3a645fe6ae38","receipt":"e5ad15a9c08d67c04dcdd7f425b29b13ed9f4269d931e8e02fe3d0eae006d00d"}]
