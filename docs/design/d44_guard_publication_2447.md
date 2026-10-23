# D44 guard publication and owner integration (#2447)

## Ownership and source binding

`prismaquant/d44_guard/campaign_launch.py` retains the accepted standalone guard,
not a replacement numerical controller. The supplied source binding names head
`fa3775151f77dc713fd78882daf6e147ab471243`. The retained approved guard's exact
SHA-256 is `38ca21ef9156eaf0f706643dd14dd3258db658aa652a98a2832c440f1d99d2e9`.
The test fixture `tests/fixtures/d44_guard_2447/accepted_source_delta.json`
reconstructs those exact bytes from the delivered source and verifies that digest.
It stores only the reverse delta, avoiding a duplicate source archive. Tests need
neither Git history nor a developer's private temporary file.

The separate delivery changes only publication reporting and explicit runtime,
routing, and digest-owner wiring. It does not overwrite the frozen workspace,
change measured action `a435d3d42b75`, or rewrite its source-bound evidence.

## Runtime configuration

Help works before numerical owners are available:

```sh
python -m prismaquant.d44_guard.campaign_launch --help
python -m prismaquant.d44_guard.campaign_launch route-plan --help
```

Execution requires the **existing accepted standalone numerical tree**, supplied
by `--source-dir` or `D44_SOURCE_DIR`. The directory must contain `stage1.py` and
the selected stage's existing entry and dependencies. A missing tree refuses;
the guard does not generate substitute numerical owners. It loads `stage1.py`
from that explicit directory, rather than an incidental module on `PYTHONPATH`.
CPU execution uses that directory as the child cwd and selects its entry by
absolute path. Caller-relative root and batch paths are resolved before that
cwd change. CPU execution never loads the D30 container launcher or uses Docker.

For container execution, `--owners-dir` or `D44_DELIVERED_OWNERS_DIR` selects the
existing external D30/residency owner directory. If neither is supplied, the
accepted `stage1.OWNERS` remains authoritative. The guard loads and invokes that
owner's `v2_launch.py` by absolute path. `D44_CONTAINER_TEMPLATE`, when configured,
selects the existing launch template; otherwise the D30 owner's `TEMPLATE` is
unchanged. The numerical owners retain their own data/runtime configuration.
No GPU execution or scientific replay is part of this repair.

Run routing from the worker-visible repository delivery:

```sh
python -m prismaquant.d44_guard.campaign_launch route-plan plan.json --out routed.json
```

Routing selects **this delivered guard**, never an unchecked sibling named
`campaign_launch.py` in the frozen tree. It defaults the numerical source directory
to the original `encode_launch.py` directory, resolving a relative entry against
`common.cwd`. Explicit source/owner configuration can also be passed to
`route-plan`; the resulting argv carries absolute `--source-dir` and, when
configured, `--owners-dir` paths. The delivery and configured owners must be
readable at those paths on selected workers. Set any execution-only template
override in the request's existing `common.env`. No source is installed or copied
into the active execution tree. Native request validation uses the installed
PrismaBuild API, not a hard-coded fleet checkout.

## Publication and preserved gates

`_publish_batch_result` returns false without touching scientific receipts or
manifests on a dry run. It returns true only after the owned manifest save returns
successfully. The container caller assigns `published` from that Boolean. Child
failure, cancellation before the commit decision, unsuccessful cleanup, missing
receipts, and publication-save exceptions leave the flag false. A save failure
can leave partial bytes; their presence alone is not publication success.

Tests compare the accepted memory monitor, start/abort policy, cancellation,
commit decision, and process-group/container cleanup functions, plus the entire
container try/except/finally, against the digest-verified baseline. Behavioral
probes additionally exercise the start floor, cancellation, failed cleanup, and
publication failures before and after a simulated write. Launch-token generation
uses `prismaquant.digests.text_sha256hex` over the original `json.dumps(command)`
text, preserving ordering, spacing, ASCII escaping, and the 20-hex truncation.

## Bounded PrismaBuild CPU proof

All commands below use `run_tests`' admitted CPU actions. Numerical/container
processes are transport doubles: these checks prove guard reporting and runtime
wiring, not numerical or GPU qualification. Fresh interpreters exercise both
entry forms, explicit owner loading, and routed execution from another cwd.
Routed requests pass the **real installed PrismaBuild validator**. Tests never
copy the corrected guard into a selected destination.

Failing-before command (exact accepted baseline; expected **1 failed, 2 passed**):

```sh
D44_PROOF_APPROVED=1 python -m pytest tests/test_d44_guard_publication_2447.py -q -k 'container_dry_run_caller_reports_unpublished or container_real_run_caller_reports_published or container_failed_child_caller_reports_unpublished' --basetemp="$TMPDIR/pytest"
```

The dry-run invariant fails because the saved flag is true; the real-publication
and failed-child invariants pass on the same baseline.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b9fe05f6fdfdac46b2e0c538556fb4bb47b2667371e93e11f0841ed4cf7c7ce7","size":1912}},"operation":"abb41baa9818dad694ab76b561815360b87477158c98c0071ee8744b306e994e","receipt":"8de20ae3421cde93b36ff08bc3a47eab345310a65f3a95f6f637ef334aeecf33"}]

Corrected implementation and repository guards: **97 passed**, including the
previously failing primitive-digest guard. The same dry/real/failure invariants
run without `D44_PROOF_APPROVED` and all pass.

```sh
python -m pytest tests/test_d44_guard_publication_2447.py tests/test_d44_guard_integration_2447.py tests/test_duplication_baseline.py tests/test_architecture_doc.py tests/test_docs_entrypoints.py tests/test_docs_staleness.py tests/test_campaign_launcher_pinned_import.py -q --basetemp="$TMPDIR/pytest"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3d08e61951ee38727f91f7f7485c2a105e7f99c38d9c88730bc9ae64128326b3","size":840}},"operation":"5e51160b4f2dc2bf816bed19540716d0ea5c0bc5757f88e43befb8c4bc2f6b81","receipt":"3f9201e3b4dfa10d288cf76cc9d6688a58dc98fb98df93b0385cdfb22053d06a"}]

An additional, broader check was **not green**: running the five repository guard
files above together with `tests/test_tessera_campaign_pinned_roster.py` produced
**52 passed, 6 failed**. Those six pinned-roster failures import unavailable
`fused_module` or `direct_consumer_weight` APIs from the environment's installed
Tessera. Their traces do not enter the D44 guard. No pin, unrelated runtime code,
or those tests was changed to work around this integration mismatch.

```sh
python -m pytest tests/test_duplication_baseline.py tests/test_architecture_doc.py tests/test_docs_entrypoints.py tests/test_docs_staleness.py tests/test_campaign_launcher_pinned_import.py tests/test_tessera_campaign_pinned_roster.py -q --basetemp="$TMPDIR/pytest"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"81696e5768c71d64bdbd378b495955911e77972658247af9e80c73bb1b2b2d4a","size":10309}},"operation":"4e1093418d5d50dac73192a533da998c1bc2e217ca531414a2c780cb42c2b301","receipt":"73d21b249843b5bb3b3c9a74f88d36cfcc9760bad80765faf6d09d95a55b47af"}]
