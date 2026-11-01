# Joint AURA measured-unit and public journal ownership

Provenance: 2026-10-14 PrismaQuant architecture refactor, base `138069b7993`.

`joint_statistics_replay.observe_and_project_retained_windows` already owns retained reverse replay. This unit leaves its incoming cotangent adapters, backward arithmetic, residency admission and GPU lifetimes in place. The repeated measurement completion around it now belongs to `joint_price_step.JointPriceStep`: diagnostic and signed-term accumulation, restored joint-row admission, zero-cost or measured row construction, and exact legacy state capture. `AuraUnitMeasurements` owns its accumulated vectors. Full streamed AURA and Stage B layer quanta use that owner.

The existing journal now exposes `AuraCheckpointJournal.open`, `resume_units`, `publish_group` and `publish_state`. It binds its root, digest and unit roster once. `UnitCheckpointWriter.publish` owns bounded encoding and atomic publication; codecs are private. Bounded host replay calls public `load_unit` with its already-owned envelope, preserving its one-read and retirement bound. No production, tool or experiment caller reaches a private journal helper. The independent asynchronous publication ledger retains staging credit, FIFO failure and durable acknowledgement. Persistence alone never advances its completed set. Full-run group completion still follows every write in that group.

Public producer provenance names are `producer_git_commit`, `producer_git_head_commit` and `producer_package_sha256`. `checkpoint_git_commit` retains its old refusal behavior. Every caller moved in the same unit; no compatibility aliases or free prepare/write API remain. The duplicate-function baseline shrinks only: the former AURA `_git_commit` member and the former two-member `_canonical_json_sha256` group are gone. Canonical journal bytes use the uniquely named private `_checkpoint_json_sha256`.

The measured objective, row component order, per-probe multiply samples, pickle protocol, envelope fields/order, error bytes, source-transition decisions, probe identities, zero-cost exemption and observation bookkeeping stay unchanged. Full rows use exact probe equality and their existing optional cache activation/render checks. Quantum rows preserve mathematical walls and D32 source run seals. Attribution refusals retain the adapter-specific exception.

## Test dispositions

The one-squaring source/AST assertion now feeds recorded signed samples through the real price owner and checks independently multiplied samples, accumulated second/fourth moments, row values and predicted loss. The bounded encoder's comparison against itself now compares public writer bytes to an independent literal envelope recipe and verifies the actual reader and oversize refusal before any file appears. A stale module-global certified-mode hook moved to the resume policy owner; its real partial-quantum resume proof remains. Goldens for bytes/refusals, file custody, held directory fsync, async acknowledgement, bounded staging, cyclic snapshot retirement and NoNewSeals remain.

New owner tests publish actual checkpoint bytes and exercise exact versus run-seal resume, calibration walls, optional cache activation binding, and a failed second write that leaves a durable first file but no group completion. No GPU or speed claim is made.

## Verification

All execution used published PrismaBuild clients and the qualified absolute interpreter `/home/rob/venvs/pq-pin-fca4c6ce0-pb027103d9/bin/python`. Portable shards reserved one CPU and 6 GiB each, with one native thread. The affinity-contract file reserved its required three CPUs. Compile reserved one CPU and 2 GiB. Native thread bounds and PB affinity were preserved. Priority was -10; no host tag was added. Source snapshots came from immutable detached capsules made by `tools/freeze_refactor_checkout.py`.

The pre-change numerical baseline passed 268 cases with zero skips across 15 files. The broader final caller/owner/docs/inventory coverage has 1,038 distinct passed nodeids and 46 skipped nodeids across 45 files, after replacing failed results with the corrected reruns. The last source-changing capsule rerun passed 552 cases with 32 skips across 30 files. The separate affinity cohort passed all 12 cases. All 61 touched Python files compiled. Every passing shard reports complete collection/outcomes/teardown and no completion problems; terminal records and CAS receipts were inspected.

The 46 skips retain their precise reasons in `/home/rob/tmp/pq-refactor-20261014/reports/joint-j2-step-final-skips.json`: 36 require administrator-provisioned private spill scratch, 6 require CUDA, 2 require the historical campaign package, 1 requires Triton, and 1 requires direct I/O unsupported on the 8-KiB temporary filesystem. These are missing qualification, not passes.

Final source capsule `/home/rob/tmp/pq-refactor-20261014/capsules/joint-j2-step-after3` has five passing test actions:

- `a926d65e720276ca12046ef7e86b72479440e64fe8e9c8f5f4b5f92a21bfd87b`:119 passed.
- `2dddc7a484483bb29046fb74d5c3c8658fe71562171677d3d0883cd62bc95919`:69 passed.
- `4ded4ac9ffa9489f563dc392c72666d140f7b2e16958b26054b15b4165d37c3b`:121 passed.
- `c3021f054511653b040698af6ce43ffa4d90e65a1fb96095756823412ba9ae72`:65 passed.
- `8de0d85e226ae1bcffbb2db415a422f001d887ca84155f6694f06080c336bd21`:178 passed,32 skipped.

Affinity action `779dca0e26f04161f78b730aec35128b9258daaac2bfb90b4e84ce21943f02d3` passes12. Compile action `7be4ae7de2cfddd0deff976c7b52c29c617dede848e93507fc4b6bd279bb45ea` executed with return code0 and CAS receipt `6254568884ec6357b7d01f1964b69f0ea6827a853c6e1d75bc2a8cb6eaa68d67`. The untouched caller cohorts on the preceding immutable capsule have successful receipts in `joint-j2-step-final-receipts.json`.

Full commands, file lists, outputs, capsule identities, failure logs, latest node counts and receipts are under `/home/rob/tmp/pq-refactor-20261014/reports/joint-j2-step-*`. First migration verification failed on one deleted payload local plus stale policy/fsync hooks. The next broader verification exposed a moved certified-mode hook and obsolete inventory membership. Their actual failed records remain; they have not been counted as success.

The rerunnable source migration recipes are `migrate_joint_price_step.py`, `migrate_price_callers.py`, `migrate_journal_writers.py`, `migrate_price_tests.py` and `rename_aura_provenance.py` in that reports directory. Apply their sequence to the documented pre-unit checkout; they are a migration recipe, not runtime compatibility. `joint-j2-step-final-inventory.json` records the final imports and confirms zero production private journal reaches.


## Checkpoint provenance correction, 2026-10-16

The earlier unit retained the single-file Git exactness check. Claude review
STD-011/R1S-F02 found that it omitted the new journal and price-step owners.
The separate correction `bad08c3e6a` checks `aura_cost.py`,
`aura_checkpoint_journal.py`, `joint_price_step.py` and `aura_provenance.py`
and names each changed owner. Certified refusal, development warnings, Git
error refusal and the explicit identity override retain their contracts. The
whole-package digest continues to bind every other package module.
