# Token-owner integration (#2638 / #3012)

## Repair on 2026-10-28

This repair adopts [PR #3022](https://github.com/RobTand/prismaquant/pull/3022)
at `58a2db30074498e0d9522ea00c6859289e82a622`, preserving its two original
TrustLadder commits and [repair adjudication](https://github.com/RobTand/prismaquant/pull/3022#issuecomment-6403753446).
The replacement is [PR #3158](https://github.com/RobTand/prismaquant/pull/3158),
branch `prismaquant-3012-pri52`, with non-closing references to #3012 and #2638.
No linked issue or parent obligation is closed by this slice.

Current GitHub main and local origin/main were both verified at
`05b62b4742793a2188abd4ed041fc4d46fd6c2c0`. All eight affected paths were
unchanged from the original integration base before adoption. A task-owned
isolated worktree applied the retained patch without changing production bytes,
acceptance tests, source pins, baseline or shared prose. The replacement preserves
the original commits and merges that main at
`a0c93c2f6296a32896d36286869102fe27f14a41`; its eight-file diff matches the
tested worktree. Only this repair section is added after that qualification.
The final PR head, including this documentation-only update, belongs in the
review handoff rather than in a self-referential source file.

The closed, unmerged [precursor #2645](https://github.com/RobTand/prismaquant/pull/2645)
was compared at `2ebf5b3226064a41b62b17e70ba640c7d415c0eb`. Its token recipe
matches the retained recipe's AST exactly. Its three other owner recipes and
Stage A/cache rewrites are excluded by #3012's explicit one-recipe scope;
their independent goldens remain as local-retention tests. No missing in-scope
token behavior was identified. Census work remains outside this repair.

Fresh local CPU command, from the isolated worktree:

```sh
TMPDIR="$PAPERCLIP_TASK_SCRATCH_DIR/cpu-tmp" \
XDG_CACHE_HOME="$PAPERCLIP_TASK_SCRATCH_DIR/cache" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
/home/rob/venvs/pq-pin-aaa41700-pb20261016-git/bin/python -m pytest \
  -c pytest.ini --basetemp="$PAPERCLIP_TASK_SCRATCH_DIR/cpu-tmp/pytest-pinned" \
  --timeout=120 --tb=short -q tests/test_tensor_digest_owner_recipes_2638.py \
  tests/test_tensor_digests_1384.py tests/test_guarded_read_digest_retention_2637.py \
  tests/test_tessera_campaign.py tests/test_duplication_baseline.py
```

Result: **126 passed, 14 skipped**, exit 0, 137.75s, 14 torch JIT warnings.
Python 3.14.4, torch 2.11.0+cpu, transformers 5.17.0; installed Tessera Git
identity matches `aaa417003f0ee690e8ace138acd18df7f9531369`.
Log SHA256: `db5a60f6cf4a30975827c3b1ecef9e7988c3f80595e639d098195d3f6940ee4c`.
The initial stale `/home/rob/venvs/pq-cpu312` run had **112 passed, 14 skipped,
14 failed** in 177.49s: every failure required the absent `tessera.metadata` API.
Its log SHA256 is `921e8d0e1b1dfc92efbbb84c2a533778afe3c2e7e85657920d285990c76fba4b`.
The retry changed to the pinned installed environment; no production workaround,
shared environment mutation or acceptance weakening was made.

Fresh `python tools/duplication_inventory.py --write-baseline` ran in a separate
inventory-only worktree of `a0c93c2f6296a32896d36286869102fe27f14a41`, exit 0:
0 near-duplicate pairs, 130 same-name helper groups, 276 primitive sites.
All regenerated sections match shipped membership recursively modulo list order;
the shipped baseline remains exactly main minus the token consumer scope,
277 to 276. Existing list order and the canonical identity below are preserved.
No scanner or sibling baseline changes were imported.

The retained CUDA wrapper receipt, stdout and quiescence hashes below were
recomputed from retained files. Terminal state is executed, exit 0, cleanup
complete, one attempt on sparklina. Underlying PrismaBuild action:
`49064446a5379690a9e0bf7ac9a39500cb60915726bffca2cdd19f3f2606f434`.
Its canonical CAS receipt digest is
`a7be40140b0a3dd36080d5dd3e5a21db3a6980a0ff38fcc3b97c5f584bf23681`;
verified result blob `55b6ffbe75ccd26fc5cd7a2305b1bdc8e5275acbb9ffa072bf06575508ac871b`
has code 0 and decodes to the recorded **3 passed, 11 deselected in 0.89s**.
Retained input owner bytes equal this repair's owner bytes. This is recovered
historical byte-transfer evidence, not a fresh GPU or model-scale qualification.
No new GPU job was submitted and no historical acceptance is promoted.

The linked-issue status passed on integration head `a0c93c2f6296a32896d36286869102fe27f14a41`
with `Refs open same-repository parent #3012`. Final-head checks remain required.
Local tests are evidence only. Integrated-SHA CI remains a GitHub/Mergify queue
gate; a skipped PR CPU job and historical PB receipts do not replace it.
Native independent review is assigned to Tactical Worker 29. That same-family
review does not satisfy the separate D52 cross-family/exact-head requirement;
manager routing is still required before queue integration. No merge is performed.

Inherited diagnostic history is retained below: one failed raw-order assertion,
two missing-working-directory normalization diagnostics, followed by successful
membership/order-preserving verification. The source PR had zero native reviews
and review threads when adopted. No new implementer/reviewer bounce occurred.
All following sections and Test evidence lines are the retained historical report,
not a claim that its old source base or old commands describe this repair run.

## Freshness and scope

Integration source base: `6d8288adc94b3c4b624c143ed372fb6513f32784`.
Working HEAD before implementation was `3b6f9d3c58adda6b3c0c13e07134c61a17b6e87b`,
which adds only the immutable research acceptance file to that base. The worktree
was clean. Git inspection of retained commit
`a9fce4a588dd41ebffcb77756166c8394a21b378` against the actual base showed only
the intended additive recipe and delegation in the two production files.
Their pre-port bytes matched base; historical source `30b6231d` was not used
as current-source evidence. Only those two production diffs were applied.

| File | Pre-port Git blob | Pre-port SHA256 |
| --- | --- | --- |
| `prismaquant/tensor_digests.py` | `5c364c5d7c8fadf78ad9917668e72e1262546278` | `98406374a7859e9a1b9e490671432372ad5dbe84a7a58c2b3c34377125af2037` |
| `prismaquant/tessera_hessian.py` | `55f1af290bdf84eeee156e04d16f13f12c372c6e` | `3eba21f2e10d374891c19ce52dc0188f5dee8eccf28a4c43cc63730e4db267ec` |

| File | Post-port Git blob | Post-port SHA256 |
| --- | --- | --- |
| `prismaquant/tensor_digests.py` | `0777758650869edc365cbf545280250e6fa175f2` | `8baeaf7541a23481b9d233a7c65ee5f679f4fcc248c004e83ad0ca16ea8ee50a` |
| `prismaquant/tessera_hessian.py` | `a800f254c0b619ff317485aeb8bffc3b877121d4` | `3fc303e9de4b3639700c42dc3bba2f0bbeab888a3cf5053e8cad33642d940d75` |

Identities were computed with `git hash-object` and `sha256sum` on final production
bytes. The owner is byte-for-byte the retained commit's owner.

## Route and preservation

Exactly one recipe is added: `tensor_digests.token_ids_int32_sha256`.
`tessera_hessian.token_ids_sha256` lazily imports it and delegates once with the
original batches object. Per-batch int32 conversion, CPU transfer, NumPy logical
C-order bytes and caller fold order are unchanged, without headers or delimiters.
The empty stream hashes zero bytes. The consumer owns no tensor-byte conversion.
No seed, sampling, signature or existing owner profile changes were made.

Only the owner's `CURRENT_SOURCE_BLOBS` override changes. Its historical tuple
in `MANIFEST_SOURCE_BLOBS`, snapshot/revalidation provenance, other current pins,
six retention contracts, stale-source rejection and negative controls remain
unchanged. Stage A payload, both production-cache identities and export capture
remain local. No census, metadata, compatibility or archived-module repair is
included. The supplied issue list identifies #3012 as this route's canonical
integration issue; census decisions remain with the retention-audit sibling.

## Baseline

`python tools/duplication_inventory.py --write-baseline` ran through `run_tests`
and reported 276 primitive sites (277 before integration). Inspection of the
retained generated output exposed pre-existing list-order differences involving
`tessera_census_union` entries: no site or helper membership changes beyond the
intended deletion. A follow-up diagnostic canonical-identity assertion against
raw regeneration failed (`813e13b66775b7ce1f751c4eed7d197294f446eb51b9d859284ddba582ac38d0`),
because of that ordering. The shipped baseline preserves base list order and
all other sections, deleting only
`hashlib:prismaquant/tessera_hessian.py::token_ids_sha256`.
Its canonical JSON SHA256 is
`78d379bc79f16c8529736bdcc272a085ee6c2c21754d0d1887a5f73fd59249c9`;
its raw SHA256 is `a3eb8d5200795210f01c58772ff0eab9ba6b08b248dbb5512f0835cbcc60fb71`.
No inventory rules changed. The acceptance assertion verifies the exact shipped
baseline identity. Two further standalone normalization diagnostics could not
read the baseline in their runner working directory (FileNotFoundError / cp
cannot stat), so are not claimed as successful checks.

## CPU qualification

Command: `python -m pytest --basetemp="$TMPDIR/pytest" -q tests/test_tensor_digest_owner_recipes_2638.py tests/test_tensor_digests_1384.py tests/test_guarded_read_digest_retention_2637.py tests/test_tessera_campaign.py tests/test_duplication_baseline.py`.
Result: **126 passed, 14 skipped**, 14 torch JIT deprecation warnings, 161.69s.
This includes the active campaign calibration-draw token oracle, independent
stream/draw goldens, empty/noncontiguous/repartitioned streams, owner surface,
delegation, exact baseline identity and guarded-read negative controls.
CPU skips do not qualify device-to-host bytes.

## Bounded CUDA qualification plan (recorded before dispatch)

Purpose: qualify post-port owner int64/int32 CUDA-to-host conversion and digest
bytes against both independent token goldens, rather than relying on research's
pre-route CUDA consumer result. CPU cannot decide device transfer behavior.
Smallest case: three CUDA-parametrized acceptance cases, tiny token tensors
(each uses a 13-token stream; the noncontiguous/repartitioned case also checks
empty input and an empty batch). The alternate draw is a fixed token fixture,
not a GPU sampling run. No model, download, benchmark sweep or training. Expected runtime is
under one GPU minute. One bounded GPU run is requested:
`python -m pytest --basetemp="$TMPDIR/pytest" -q tests/test_tensor_digest_owner_recipes_2638.py -k cuda`.

## Final qualification and review

The bounded post-port CUDA run completed: **3 passed, 11 deselected in 0.89s**
on the GPU arm (torch 2.11.0+cu130), exit 0. Both independent goldens and the
logical-stream cases are now qualified through the owner on CUDA. This is tiny
byte-transfer compatibility evidence, not model-scale performance evidence.

The explicit O1 selection (`python -m pytest --basetemp="$TMPDIR/pytest" -q
tests/test_tensor_digest_owner_recipes_2638.py -k 'not token_route'`) also passed:
**9 passed, 3 skipped, 2 deselected**, 7.00s. The three CPU CUDA skips are
separately covered by the GPU result, not counted as passes.

Final inventory verification regenerated the baseline again, compared every
section recursively modulo list ordering with the shipped snapshot, and
preserved existing order. It passed and retained the exact shipped bytes and
canonical identity above. This confirms the raw-regeneration diagnostic was
order-only; the scanner was not modified. Source/prose review confirmed the
six-file existing footprint plus this new audit, immutable acceptance tests
and all out-of-scope files unchanged. An independent read-only review caught
and corrected the CUDA fixture-size description and the CPU-suite label.

## Test evidence

Final regeneration, membership equivalence and ordering preservation (exit 0):

[Test evidence {"artifacts":{"outputs/tests/fixtures/duplication_baseline.json":{"digest":"a3eb8d5200795210f01c58772ff0eab9ba6b08b248dbb5512f0835cbcc60fb71","size":35357},"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"97b4eefdb66b702e0bd367c48819ab575f3ec88ac89a14424877ce5d40daf8c3","size":311}},"operation":"8fc41ed4c1224c969fe6fb6b0b2a4cff7d91acb6f19ef6aea9db101738801b42","receipt":"28f49aa6bff6714beeef00d39e1455f49f5231d1d17c4894bc29f34c727e16a3"}]

Post-port CUDA (exit 0):

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"2806c625f47cf5b23f623e13b82beded9cebe1ca5b70ccfc7a528c98ecc05140","size":287}},"operation":"d3ffcbdca46e4e1cc2361b1be9d369de74b8cd20c906fda970d80d64ee437207","receipt":"85c940f002f794805bafe54ada575b07ce59803e7d87b19b6c4298e72527bab0"}]

O1 selection (exit 0):

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"c2d253e454b52f7bf0b00532fab2c9182f1bbb760348d027dc2d8693443567f8","size":789}},"operation":"741beea0b0d68e210ba3a450b1919b0074f8ca5a71372fb6953f09780ba5675b","receipt":"fbd955a3862416a0404f9cb7b614c683e3308171ed1369b3c76489032d05a65e"}]

Inventory generation (exit 0):

[Test evidence {"artifacts":{"outputs/tests/fixtures/duplication_baseline.json":{"digest":"acfc49574228cd344829a73c088ed8290effe4e37eb563ee1210dbc966883377","size":35357},"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e977f841e6acedc059fba799c28b4d3e0068ede9ae035d1a0240a28b49135bc2","size":136}},"operation":"aa6f694bd1c2eecf54feded9d34030d8f5d8f9e49c4d45002ee32dd8553c14ac","receipt":"209868038e7aee1b9788efe77ea0aad86f746540787845f7e053d9e63e5d165f"}]

Raw regeneration diagnostic (exit 1; ordering assertion, not a route failure):

[Test evidence {"artifacts":{"outputs/tests/fixtures/duplication_baseline.json":{"digest":"acfc49574228cd344829a73c088ed8290effe4e37eb563ee1210dbc966883377","size":35357},"stderr":{"digest":"aafa9a20c655916fb2423e3950d4fb9109c0c5fd9543bf4390dea50a1fc24fe2","size":777},"stdout":{"digest":"c604813b2ab00f074f1336ddbf5b2561fd6f297f2d7cfb45447ea57a439b4d31","size":4014}},"operation":"7dc6bab2c047b41f4a436a4591d48427113087182a4f5d7870276e77c68afb7c","receipt":"8f94540e9e9465ea70d3829536bb81a1f0a5f436f8a5a4e410fd105a340179d8"}]

Selected CPU qualification suite (exit 0):

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e92bf4ce1e2f9d1d76b5396969feae0fe53e35d21603d7ed863eeedd9aca6024","size":870}},"operation":"70420bb167954dc4e3c067c074bb49df3074c5a31127eb1b485e029afd6dd99c","receipt":"b2f8e94f62e075ded4376991e0d98042379f46afa48833632d19b6dbe6c8e306"}]
