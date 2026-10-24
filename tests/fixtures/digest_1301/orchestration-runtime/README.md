# Orchestration/runtime digest slice

This is local integration evidence, not a replacement for `current_1301`, the shared census, or the shrink-only baseline. `ledger.json` partitions all **29 frozen research-registry calls** in the assigned files, including the two inspected standalone modules: **4 completed**, **13 blocked on exact owner APIs**, and **12 deployment retentions**. Existing owner calls recorded by the sub-ledgers predate this slice and are not counted as new completions.

## Completed substitutions

- Cluster run/status stdout: exact sorted spaced/indent-2 ASCII lax JSON profiles, through the existing package/direct-file import branches.
- Pipeline spec stdout: exact sorted indent-2 ASCII lax JSON profile.
- `SealedBuffer.seal`: existing byte owner, including empty buffers. Tests compare raw digest feeds, result width/type, released views, mapping closure before kernel seals, and descriptor lifecycle against independent original source.

The original consumer excerpts are fixtures, not expectations computed by the replacement owner. Tests execute actual consumer publication blocks, including refusals, whole output bytes and lifecycle traces. The root ledger stores executable unparameterized test nodes and fixture hashes. Each sub-ledger contains more detailed witnesses.

## Uncompleted prerequisite routes

The prerequisite owner-recipes extension is not present in this checkout. Agreement S prohibits this slice from editing shared owners. These gaps are **blocked migrations**, not retention justifications:

- Pipeline settings, unverified reuse, and validation output need sorted indent-1 ASCII lax JSON **without** `default=str` (3 calls).
- Dev-mode diagnostic rendering needs sorted spaced ASCII lax JSON with `default=repr` (1 call), keeping its original fallback and truncated suffix.
- Autoscale/day-0 stdout need insertion-order indent-2 ASCII lax JSON (2 calls).
- Gate/receipt stdout and initialization snapshot normalization need insertion-order default-spaced ASCII lax JSON (5 calls). Sorting would alter stdout or normalized mapping order.
- Pipeline file and atomic registry publications need an owner **streaming sink** API preserving `json.dump` write partitions and partial-write/refusal behavior (2 calls). Replacing these with `f.write(profile.text(...))` changes the partial output on failure. Independent tests compare partial destination/temp bytes, write traces, cleanup and flush/fsync/replace/LF order. Both original streaming implementations remain intact.

## Supported retentions

- `io_spans` (2 calls): `tools/pq_profile_source.py` loads the file by spec before package initialization. Tests execute both span closure and `_emit` with package imports prohibited. `_emit` exercises marker/bytes, strict nonfinite refusal, successful-line count, caught logger errors and warnings. The span encoder's strict `default=str` recipe stays distinct from the reporter's no-fallback recipe.
- `container_runtime_identity` (6 calls): current production `runpy.run_path` loaders in `tools/tessera_campaign_container.py` and `tools/build_glm_derivative_image.py`. Independent isolated operation tests cover framed source hashing, image identity, receipt/refusals, identity publication/resume and stdout.
- `prismabuild_progress` (4 calls): current standalone path-loading contract in existing progress tests and its supported standalone source contract. Tests exercise report, commit, staged/export wait publication and failure results. No additional production file-spec loader is claimed.

Neither standalone source file nor any deployment roster was edited. Their JSON loader population is separately documented in the standalone sub-ledger; it does not enlarge the encoder ratchet. This slice does not claim an exhaustive alias/dynamic-options/generated-code/shell audit beyond its inspected files or parent closure.

## Integration required

The shared baseline must remove exactly:

- `hashlib:prismaquant/io_engine.py::SealedBuffer.seal`
- `sorted-json:prismaquant/cluster_campaign.py::main`

Pipeline's mixed scope remains live. Shared baseline/census/scanner files were deliberately not edited. The integration owner must reconcile these changes serially and provide the missing recipes before the remaining routes can close. No allocator, numerical, policy, default, pin or protected source changes were made. No GPU, whole-suite, deployment or performance qualification is claimed.

## Verification

Final focused command:

```sh
python -m pytest tests/test_digest_1301/orchestration-runtime tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py tests/test_profile_observer_bootstrap_1942.py tests/test_container_runtime_identity.py -q --basetemp="$TMPDIR/pytest"
```

Observed: **334 passed**, 14 dependency deprecation warnings.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"1e48707930556fa6b8c34ce7e0f5b4e2760749f68cbab048057fb63ed7eea168","size":1142}},"operation":"32a059cab46efecee81b43e385c710c74326673ab4f84f90361a85fc456e26fc","receipt":"5239e509573d5bad31406e5ca397f8bbe18546f558e18826ffc46b9256c51480"}]

The same command **including** `tests/test_duplication_baseline.py` observed **340 passed, 1 failed**, 14 warnings. The sole failure is `test_primitive_digest_sites_only_shrink`: no new raw scopes, but the two completed scopes listed above must be removed from the baseline by its assigned owner. This is not a passing ratchet, nor a defect in the test.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f815fb6906652fdc1be002f5505f2884949d3f57ab130cc0e2f55351caa01a93","size":3258}},"operation":"aba715b587a2aac1f7bc5248fdbe5d941564417b0bec8ee7ce67e592cae8bd64","receipt":"bfa6d69027d79da3c10eb672dca01d94c333a3959c27044f2aa66297087e3b38"}]

`git diff --check` passed. Final production diff inspected: five files, only three stdout JSON substitutions, one byte-hash substitution, and explanatory comments on blocked/deployed recipes. Parent acceptance was not claimed; O1/O4/O6 are outside this assigned slice and O2/O5 still need the explicit integration/prerequisite work above.
