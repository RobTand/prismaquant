# SDK4 pilot consumer: CPU source and result qualification

PQ #2106 completes the bounded consumer portion of #1293. The dispatcher now
binds counter bytes to an explicitly selected PB execution's verified CAS
result and sealed request. The capture binder proves the standard wrapper;
PQ inspects the authenticated payload command, original quantum record and
independently reviewed source contract. The contract binds both snapshot
input descriptor and selected Git commit, plus entry, image and environments.
The same bundle can contain older commits, so descriptor equality alone is
insufficient. Cross-output namespace equivalence and the stamped override
remain covered. The normative contract and finite measurement protocol are
in `docs/design/joint_dispatch_pilot.md`.

## Qualified environment and mode

All tests and compile actions ran through the published PrismaBuild tools,
on dl380g10, in `/home/rob/venvs/pq-pbdc4803da-tessera-b40c93cb/bin/python`.
The environment is Python 3.14.4, CPU Torch 2.11.0, Transformers 5.17.0,
pytest 9.1.1, PB dc4803daaf09b6426083d2d36bd2a2da3d6832fe and unchanged
Tessera b40c93cb73745097e57a1ba4cf5b9eee166c759a. Each shard's preflight
verified both non-editable Git installs, import ownership and RECORD bytes.
The shared connected-fixture source pin resolves one read-only SDK4 archive
with the complete file union; this is source qualification. The deployed
worker generation remains d028dfee920b-1790960385-1d815cff72d1.

CPU tests reserve one CPU and 4 GiB per shard, with one native thread,
priority -10 and CUDA disabled. The finite compile reserves one CPU and 2 GiB.
The original B31 interpreter was retained. Initial attempts failed for missing
pytest and later xxhash; PB provisioning actions installed the scoped tooling.
Those failures remain failed evidence, not qualifying runs. The earlier
unauthorized local smoke in SDK4-CHECKPOINT.json is excluded.

## Causal and selected-result controls

| Report/action prefix | Actual outcome | Interpretation |
| --- | --- | --- |
| `sdk4-consumer-causal-red.json` / `204ba21fd3b6` | 6 failed, 2 passed, 22 deselected | Draft admitted foreign entry/plan/prepared/adjoint/regime; real binder path incorrectly compared wrapper to command |
| `sdk4-source-causal-control.json` / `f896f1c44540` | 1 failed, 7 passed | Removing only the reviewed source-descriptor comparison admits the resealed foreign-source control; pre-selected-commit contract variant |
| `sdk4-commit-causal-control.json` / `363b52ac4d00` | 1 failed, 8 passed | Removing only the selected Git-commit comparison admits an older tree from the same reviewed bundle |
| `sdk4-consumer-green-reviewed.json` | 111 passed, 111 collected/ran, 0 skipped | Final delivered Python bytes: dispatcher 37, counter reference 20, real selected-result controls 9, quantum runtime 38, duplication 7 |

The real controls use private CPU Git bundles, actual sealed requests, CAS
input/result receipts and PoolQueue publish/claim/finish records, read through
the public SDK4 and its actual capture binder. Source, older selected commit,
entry, plan, environment, wrapper, legacy result and wrong attempt each refuse.
The matching control admits. GPU numerical/power/wait measurements and the
fixture source-review relation are controlled doubles; they are not a genuine
joint GPU pilot or approval of a production source contract.

Final green actions, all rc 0:

- `51da58dd0459c40b1fb67f59a7bc8e7048399ae377718504f01440d7bd886839` — dispatcher 37.
- `ff99ee206357070ad3be226d65e184dc159c17cdf1eb24b859928e1a2913a9af` — counter reference 20.
- `84baca214434a3643fc76db8f4da5f382c3742e590b53a283c5587686e251567` — real selected-result controls 9.
- `3f4e98c3e274756a1087b8b88f5663222b9bf0971df23b0784d0675aa24b74b8` — runtime 38.
- `659a64d0417f92ac358207b6e9b08fb88cd39d95ed444d2bb40e48ac46be2585` — duplication 7.

The pin/differential run passed 22 cases with no skips. Earlier caller
regressions passed launch 8, metadata 27 and catalog extension 65 cases; the
quantum runtime's missing-xxhash failures were superseded by the final 38-case
run. Architecture 13 and docs-staleness 6 passed. Duplication initially found
new primitive digest sites; these were replaced with the existing byte-digest
owner, and final duplication 7 passed. Supporting runs retain their own source
identities rather than being relabelled as the final consumer source.

Final five-module compile action
`3bbe2e7c01610aa06de8ea618922b9afcb4d50b5b74cd37180d0403bfc3b0f91`
returned 0, receipt
`c3c14b693bcc4735492e4c88bfd0f7129b2b55ed785ffc9e015ea2e35d86851f`.

## Attributable artifacts and commands

Reports and logs are under `/home/rob/tmp/astra-resume-20261002/pq_spill/`.
`sdk4-receipts/*-action.json` holds complete typed PB action observations;
`sdk4-source-evidence.json` binds them to raw source bundle SHA/length, selected
Git commit, package/file digests, exact terminal status, receipt/result bytes
and attempt-log hashes/lengths. Every final green and compile source matches
the delivered quantum, pilot, staged-lease, dispatcher and experiment Python
files. Their full durable package digest is
`694b30b349cb2cf5e652a2d25453b302fdd5c8fefaa1d0abbcb9d022870d668c`.
The evidence records also retain earlier variants and failed controls.

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbtest.py \
  --checkout /home/rob/tmp/astra-resume-20261002/pq_spill/pilot-worktree \
  --python /home/rob/venvs/pq-pbdc4803da-tessera-b40c93cb/bin/python \
  --priority -10 --mem-gb 4 --cpus-per-shard 1 --threads-per-shard 1 \
  --timeout-s 300 --json sdk4-consumer-green-reviewed.json \
  tests/test_joint_dispatch_pilot.py tests/test_pilot_counter_reference_1293.py \
  tests/test_pilot_verified_result_1293.py tests/test_joint_cost_quantum_runtime.py \
  tests/test_duplication_baseline.py
```

The compile used published `pbrun.py --cpus 1 --demand mem_gb=2 --priority -10
--timeout-s 120 --anywhere`, with the explicit SDK4 interpreter and `-m
py_compile` on the five modules above. The interpreter path supplies the
placement dependency; PB selected the worker. Completed profiles report honest
one-CPU demand and sub-GiB peak residency for the final test actions. Netdata
CPU series are present; missing pqteld coverage is retained as an observation.
No speed, energy, work-per-joule, GPU residency or serving claim follows.

The genuine #1293 pilot and matched #1291 before/after driver-receipt
acceptance remain open gates. Current-source original-model GPU input/lifetime
qualification, exact input/image/resource seals and root resource GO are
required before the finite protocol can be launched. Historical unbound
receipts remain negatives and cannot be retro-stamped.
