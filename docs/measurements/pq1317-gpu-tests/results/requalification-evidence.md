# Fresh native-v2 requalification publication

This is the fresh evidence index and offline validation suite for candidate
`fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb` on image X after repairing the
native observer and numerical retention. It preserves the original GPU logs and
manifests without rewriting their historical identities. The top-level summaries
now point to this evidence; the current harness is bound by the hashes below,
not by a fabricated update to the historical run2 manifests.

## Final integrated verification

Current observer, numerical-retention and requalification-package tests, on the
explicit aarch64 CPU arm with normal repository conftest: **159 passed**.

```sh
python -m pytest tests/test_pq1317_native_identity.py tests/test_pq1317_bf16_identity_requalification.py tests/test_pq1317_native_v2_package.py -q --basetemp="$TMPDIR/pytest"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f2b73a45237775a7175a320ad4980888b6e8fd97c54b940d444d8b79c309016f","size":916}},"operation":"516b06d372590cbceb8dbf4a04bf1ccb9b890638566c8d2cdbdbc698ddf50f91","receipt":"deccb1ac14e5af776ee73fd836b94b12bfd19349d71b1ffccc77328d7bde4ff2"}]

The combined package/documentation/architecture/duplication/import subset gave
**377 passed, 1 failed** (14 Torch deprecation warnings). It is **not all green**:

```sh
python -m pytest tests/test_pq1317_native_identity.py tests/test_pq1317_bf16_identity_requalification.py tests/test_pq1317_native_v2_package.py tests/test_pq1317_gpu_qualification_package.py tests/test_docs_entrypoints.py tests/test_docs_staleness.py tests/test_architecture_doc.py tests/test_duplication_baseline.py tests/test_tool_imports_1304.py -q --basetemp="$TMPDIR/pytest"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"4b15f6a815cfb31382eccab5d7a46d88b3da1a60e6a7fdbd4aec5240cd8d90a4","size":2849}},"operation":"ad0e949457e4447eabbdc664263b51777778bac73fe2e9798822d8389a9af4a6","receipt":"a1ac6ff4459e21dc48a163b5d28b7249994355594f6a36c0f9ee02b7809941bd"}]

The sole failure is the unchanged legacy acceptance assertion at
`tests/test_pq1317_gpu_qualification_package.py:151`, in
`test_the_accepted_runs_used_a_clean_snapshot_of_the_recorded_harness`:

```text
AssertionError: container_entry.py differs from the harness that ran
historical: bde7261695af8f0a630e3720d7a866a85764ce65d0ad0461579e79dad67acfa9
current:    84d9391b3343c76eddd17a4fff06256d4aea9ba0aea46d9890afc1457830576d
1 failed, 377 passed
```

That test hard-codes the original run2 directories and requires their recorded
harness hashes to equal the current files, even after the required observer
repair. A legitimate new qualification cannot change which bytes ran in those
historical actions. The minimal fixture repair is to select current qualification
manifests through `results/requalification.json` for current-harness equality,
and keep historical manifests checked as historical evidence. The new package
tests already perform the former check against **all five current harness files**.
The legacy assertion also assumes a clean snapshot and equates Docker's config
ID to its repository digest; neither should be imposed on new sealed-snapshot
records. No acceptance test was edited and no historical manifest was falsified.
The PR's old claim of "30 package tests passed" is therefore not repeated for
the repaired tree.

An intermediate combined check produced 376 passes and two failures: the same
historical-harness assertion plus a result-manifest digest that had not yet been
refreshed after adding the requalification pointers. The generated manifest was
then refreshed; the final check above confirms only the historical-harness
failure remains. No candidate, image or measured harness changed after the GPU
runs.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f35e2059b36e95e64e9e51ebefb141a6996b4bc2d0578a53371718ef66f8fd7f","size":3628}},"operation":"199164adc19a1932692ede115cf4eec54b4aa7d1a137cabc2ef93e9faa2f6c9b","receipt":"aa72efbd9bdfb110bd8885836e82cb525e411d4ef9d72053152fddb8603caa49"}]

## Observed launches

Both original operation receipts were inspected with `read_test_artifact` and
report success, exit status 0. Operation identities are not PB action IDs.

- Runs: `a577b8d9db2049265c77f9c211f8a7994b45374c25575163ddc6310664be1cb7`.
  **One operation, four independent fresh-directory container launches.**
- CPU collection preflights:
  `5267a884d103f822f34a4ff5022bdd80bc6006db6f937d89c417547e02ef01cf`.
  `collect-dense.5393ejr1` and `collect-moe.x3u7l0i0`, CUDA unavailable, exact
  65/71-node collections, successful probe self-tests. Collection is not a
  CPU execution of the CUDA qualification suites.

Round assignment follows `run-manifest.json.started_unix`, not directory order:

| Round | Launch | started_unix | JUnit passed | BF16 nodes / measurements | Built native SOs |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | run-moe.4yrogwtx | 1792797955.4219651 | 71 | 0 / 0 | 0 |
| 1 | run-dense.8iph6q0c | 1792798009.1268075 | 65 | 19 / 23 | 4 |
| 2 | run-moe.lze3_4vo | 1792798146.412726 | 71 | 0 / 0 | 0 |
| 2 | run-dense.qg186gcx | 1792798193.4974463 | 65 | 19 / 23 | 4 |

Each round has the same **136 exact roster nodes**, all passed, with no failures,
errors, skips or xfails. Each covers the 9 issue-610 and 45 issue-611 correction
nodes, all ten correction elements, and all 4/39 original-failure mappings.
The strict-CUDA allocation floors are 46 dense and 38 MoE nodes per launch;
these floors are not substituted for per-node CUDA execution evidence.

The source verifier reports 2,005 verified files, source digest
`980a9718a3ff28714794ce2d3c04cea79e045318f9f16f566db361e9db42d627`,
and agreement across measurement entry/exit and both workers. Exact hashes of
all five harness files are recorded in the index and every retained manifest.
The local helper/harness bytes match those fresh-run hashes. Sealed snapshot
closure metadata is preserved without imposing a clean-worktree or
closure-HEAD-equals-candidate requirement.

Every launch reports `native_evidence_qualified: true`. In each dense launch,
49 test records have native mappings; test/session records retain 292 mapping
observations. All successful readers are `pathname` after recorded `map_files`
permission failures. Offline tests independently match each mapped device/inode
to backing metadata and the successful opened-file provenance, check module,
path and digest consistency, and match the entire mapped set to the built-SO
inventory and `native-so.sha256`. **MoE has empty built/mapped SO inventories**;
its true native gate is not proof of a native shared-library call. Triton CUDA
kernel inventories are retained for both suites.

Docker inspect fields remain distinct. The preflight configuration image ID is
`sha256:9e6882df27dd5274bcb8fe048a0c435c531bbd794247cfc964556cf7d60fbc7d`;
its image reference/RepoDigest ends in
`sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`.
Run inspect reports are preserved as observed, not normalized to assume that a
Docker configuration ID must equal a repository digest.

## Retention and index schema

`requalification.json` uses **`pq1317.native-requalification.v1`**:

- `candidate`, `inputs`: candidate identity and SHA-256/bytes of the unchanged
  candidate, roster and corrections documents.
- `harness_sha256`: exact shared launch hashes for `bf16_oracle.py`,
  `container_entry.py`, `native_probe.py`, `probe_selftest.py`, `run_suite.py`.
- `operations`: exact run/preflight operation identifiers, launch counts,
  successful exit statuses, and verified receipt stdout SHA-256/bytes.
- `preflights`: two CPU collection entries.
- `rounds`: two chronological rounds; each contains `round`, `started_unix`,
  `finished_unix`, `total_nodes`, and two `runs` entries.
- Every launch entry binds suite/mode, operation, timestamps, device/CUDA,
  checkout/source verification, unmodified image inspect, launch snapshot,
  roster count, artifact root, and the `artifacts` SHA-256/bytes map.
- Run entries additionally summarize JUnit `outcomes`, CUDA allocation floor,
  BF16 node/measurement counts, native provenance counts, and `raw_probe_tests`
  original path/SHA-256/bytes. `external_raw_jsonl` indexes each launch's
  original probe, collection-probe and self-test JSONL paths/digests/bytes.
- `correction_coverage_per_round`, `probe_tests_encoding`, `limitations`:
  correction coverage, reversible native-inventory encoding, and evidence
  boundaries. No fabricated per-launch PB action identifiers are present.

`logs/native-v2/` has **58 measurement artifacts, 1,211,764 bytes total**:
13 per run (the eleven requested files plus `probe-tests.json` and
`native-inventories.json`), and only collection/report/manifest per preflight.
All requested run artifacts other than `probe-tests.json` are unchanged copies.
The only additional log-tree configuration is a narrowly scoped `.gitignore`
exception retaining the four `pytest.log` files despite the root `*.log` rule.
No source, wheel, library binary, cache, or raw JSONL duplicates are included.
The temporary one-shot publication script is outside the worktree, not part of
the deliverable.

### Lossless probe record representation

Only `native_libraries_mapped` is replaced, using this explicit reference:

```json
{"native_libraries_mapped":{"$native_inventory":"<sha256>"}}
```

The sidecar schema is `pq1317.native-inventories.v1`, with
`inventories: {"<sha256>": <original native inventory>}`. The key is:

```python
hashlib.sha256(json.dumps(inventory, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")).hexdigest()
```

Resolve the field in every record from that sidecar, preserving every other
field and the record order. The **original raw bytes**, including whitespace,
are reconstructed by the serialization verified against the actual source:

```python
raw = (json.dumps(expanded_records, sort_keys=True, indent=1) + "\n").encode("utf-8")
```

Tests require every sidecar key to match its content, no unused inventories,
and the reconstructed bytes to match the original raw file's digest and size.
The four original SHA-256s are also independently pinned in the new test.
The original raw probe JSON files total 1,706,753 bytes; packed probe records
alone total 764,409 bytes. Each dense sidecar has five inventory snapshots;
each MoE sidecar contains the empty inventory. No record/assertion is dropped.

Offline checks validate retained measurements with the existing BF16 helper,
including CUDA labels, operand/shape coverage, finite statistics, actual bounds,
errors and zero violations. Negative tests damage real retained copies (missing
or duplicate comparisons, CPU labels, nonfinite/errors, bad native provenance,
or record/inventory tampering). These are refusal tests, not new measurements.
Tensors and external SO/source bytes are not included, so CPU tests do not
recompute GPU results or re-hash unavailable binary/source content.

## Test evidence

All pytest invocations used `run_tests` and `--basetemp="$TMPDIR/pytest"`.
The initial publication-only invocations below disabled plugin autoload and
used `--noconftest -p no:cacheprovider`. The final integrated checks at the end
use normal repository discovery and conftest. These CPU checks do not execute
CUDA or substitute for the measured GPU outcomes.

### Final fresh-package suite: 38 passed

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest --noconftest -q -p no:cacheprovider --basetemp="$TMPDIR/pytest" tests/test_pq1317_native_v2_package.py
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ed346c4c224ace7e0a218d14b43f2a025026aede15f9716088cbbfd6758882b1","size":99}},"operation":"29acb4dfa8f4ccf4dc85864a39b44dbc4effbb01607cce7d20fbc114ca64a54b","receipt":"6b0aa8a4765b143ca939d3d38fe622d04f1c96116b0dcbd371a2847ec7cd75c9"}]

### Fresh package plus unchanged native/BF16 helper regression tests: 152 passed

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest --noconftest -q -p no:cacheprovider --basetemp="$TMPDIR/pytest" tests/test_pq1317_native_v2_package.py tests/test_pq1317_native_identity.py tests/test_pq1317_bf16_identity_requalification.py
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"abcbfede2e8f1685b8dc64a0afd740ad3ef100c596dc2a7e1b6b421b1be3ac81","size":260}},"operation":"fdf44ba06a1492af7fc761c8d7bb4222fa1109b93a57d8121aae4f58403faf72","receipt":"202348504cee9a8a83ac8b48ae773190faea0c286ac3eae6f1b7a7f4b93ee13f"}]

### Historical suite is not green: 28 passed, 2 failed (left unchanged)

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest --noconftest -q -p no:cacheprovider --basetemp="$TMPDIR/pytest" tests/test_pq1317_gpu_qualification_package.py
```

- `test_the_accepted_runs_used_a_clean_snapshot_of_the_recorded_harness`
  compares the current repaired harness with historical-run harness hashes;
  `container_entry.py` differs. That harness mismatch already existed when this
  publication task began.
- `test_every_retained_log_matches_its_digest` requires the old log manifest to
  exhaustively describe the entire log tree, including the newly required
  `native-v2` subtree. It therefore fails after this additive publication.

Neither the old test nor old log manifest was rewritten to conceal these
incompatibilities. The new test serves fresh qualification, and no whole-suite
pass is claimed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"4e57d3f5fbe9969475a59eeaa9c2a11ac87898307ccce3e19f557c7b00755982","size":2855}},"operation":"01b95cb85e1885a7c5c7eb33f7ddbc38d23947f2d32ce949687914e7d40ae60e","receipt":"d041363e691c845f1d396097e06e61880992722cfccb75883c5e7c415cb8664e"}]

Earlier development checks are retained, not hidden: operation
`b8d02a86295c5a58cfccb2bc92421c2421dea13523444544b3d78defe317be44`
had 30 passed/8 failed because ignored `pytest.log` files were missing from the
snapshot. Operation
`e9d2d9cecbb7c6620bcc500cbabe84bbc58a4fdef916dc487ff5355a955b7dd3`
had 178 passed/4 failed: two new-test lexical-roster-order assumptions (fixed to
validate exact membership/multiplicity), plus the two historical failures above.
The final successful receipts supersede those fresh-test failures.

`git diff --check` also completed successfully. This publication adds only the
new index, this evidence document, the `native-v2` subtree and
`tests/test_pq1317_native_v2_package.py`.
