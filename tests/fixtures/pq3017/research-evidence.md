# #3017 acceptance-test research evidence

Base inspected: `9f35f1e45254a099fba7bc075aa887a87343257e`.
Only acceptance tests and their fixtures were added; no package repair or recovery
into the application was performed. Snapshot fixtures came from the supplied Git
objects as documented in README.md. Runtime tests do not read Git history.

## Design decision before writing tests

Use the requested local root-cause repair: preserve run2 and native-v2 snapshots,
add explicit recorded/current validation bindings, and distinguish reader
unavailability from integrity refusal. Do not rewrite measured launch manifests
or present retained replay as GPU execution of the repair. #3011 owns the newer
Tessera pin qualification-scope clarification; no parallel work is proposed.

## Final acceptance baseline

Command (CPU):

```sh
python -m pytest tests/test_pq3017_qualification_integration.py -q --basetemp="$TMPDIR/pytest"
```

Observed: **22 failed**, exit 1, no import/discovery/setup errors. Six assertions
require the missing package, independent validator, publication instructions and
recovered gate. Sixteen behavioral cases expose the preserved native-v2 observer
and downstream gates accepting detected integrity failures. The four real-mmap
cases reproduce one-time mutation after an emulated successful map_files open
of the same inode. Verified EPERM fallback and unchanged successful-map_files
positive controls execute within those same failing regressions. Retained
stdout was inspected using read_test_artifact.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6df13200f1a0a81aa2b8f3d0d68b99148c109e48c154daaf98ba76e220d387e4","size":60601}},"operation":"b963b9eaac25e98d0d3bd78cd270f6c30f35ffcdf3873295bc34cd43ea435a48","receipt":"08a0113e90f66be20105be950b99619cd4e204a04ecf9471530b568e7bb06240"}]

## Existing repository guards

```sh
python -m pytest tests/test_architecture_doc.py tests/test_docs_staleness.py tests/test_docs_entrypoints.py tests/test_doc_claims_driver_truth.py tests/test_duplication_baseline.py tests/test_duplication_alias_scope_audit.py tests/test_tool_imports_1304.py -q --basetemp="$TMPDIR/pytest"
```

Observed: **232 passed**, exit 0; 14 existing Torch Python-3.14 deprecation
warnings. Retained stdout was inspected using read_test_artifact. This is a
baseline guard result, not evidence that the missing qualification package has
been integrated. Rerun the guards after implementation.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"2d79326104183f473a3de2049000299d6be84159cffb2b5fe4a600589a417f32","size":991}},"operation":"a37ef1ef03a06bd0e5d566eb6f8cd94cdcfa35e01d37ca659319e85d1de9d7c1","receipt":"406020fd731ae7e30f95aad81ebae158379cb7173c7b9857c6e89a1e321a8378"}]

## GPU decision

No GPU execution was needed or attempted in research (estimated GPU minutes: 0).
The experiment under test is Linux backing-file refusal and offline validation;
CPU tests decide it. The measurement-bearing retained records were inspected,
not independently rerun. Their EPERM pathname provenance is not evidence of a
mutation during those runs, nor of execution of repaired observer bytes.

Implementation should publish the recorded fca4c6ce candidate/contract-v60/image-X
qualification separately from CPU validation of the new observer. If it instead
claims GPU execution of changed measured code, exact-image requalification is
required: write the decision and GPU-minute estimate first, run CPU preflight,
then the smallest sm_121 case in actual image X before any necessary expansion.
The supplied GPU arm's Torch 2.11 runtime is not image X; do not substitute it.
