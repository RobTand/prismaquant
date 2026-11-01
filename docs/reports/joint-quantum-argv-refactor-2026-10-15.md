# Joint quantum invocation and demand ownership

Provenance: 2026-10-14 PrismaQuant architecture refactor, base `138069b7993`.

`joint_quantum_row` now constructs the unbound producer manifest invocation, the authenticated consumer file-pair command, optional band handoff and grace arguments, policy-derived CPU/host/GPU demand and ordered PrismaBuild options. These contracts complete the second J4 unit after phase ownership in `c250464a58`. Three existing production callers moved directly to this owner: the manifest producer, dispatcher and consumer parser. There is no legacy entry-point or grace-flag alias.

`QuantumFileBinding` carries a path and admitted digest. `QuantumRunInputs.argv` preserves the consumer's exact flag order. `QuantumBandArguments` preserves handoff, emit and grace ordering. `QuantumRowDemand.from_policy` preserves the existing byte-to-GiB formatting, defaults and maximum of source workers, operator workers and declared layer-read threads. `QuantumPBEnvelope.argv` preserves tags, manifest, residency, progress, demand, output-template, image, ordered environment pairs, dev mode and separator order. Duplicate environment keys remain ordered entries, as before. The manifest's unbound locator remains distinct from the dispatcher-authenticated executable invocation.

The dispatcher still authenticates every raw file, record identity, executable readset, prepared render, projection, scratch budget and cotangent plane. It still owns spec loading, container wrapping and the PB gateway. The row values contain no readers, readiness state, cache or runtime callback. This extraction runs during CPU dispatch preparation and does not change GPU arithmetic, residency or source-loading phases.

The independent literal command tests cover consumer file pairs, optional handoff flags, producer locator bytes, default and policy-derived demand, duplicate environment order and PB envelope options. Existing executable-readset, launch-contract, prepared-render, container, band-serial, compute/load-grace, spill-ceiling and safety-inventory checks remain. The old source-line assertion in `test_load_phase_grace` was removed. Its replacement loads real referenced checkpoint and handoff tensors and verifies that all entry waits share one deadline, with one read of the configured wait budget.

The immediate pre-unit checkout had already passed the first J4 final cohort, 683 passes and five recorded skips. This unit's latest distinct coverage is 427 passes and three skips over 20 test files. Every collection, outcome and teardown completes. The skips are the existing administrator-private Stage B spill scratch requirements in `test_compute_phase_grace_1165`, `test_compute_tail_grace_1190` and `test_stage_b_spill_ceiling_sealed`; missing qualification is not counted as a pass. All six changed Python files compile. The duplication baseline does not grow.

All execution used PrismaBuild and immutable capsules from `tools/freeze_refactor_checkout.py`. Tests reserve one CPU and 6 GiB per shard, one native thread, priority -10 and no requested host tag. Compile reserves one CPU and 2 GiB. The qualified absolute interpreter is `/home/rob/venvs/pq-pin-fca4c6ce0-pb027103d9/bin/python`; the published client derives its eligible placement. PB affinity is preserved.

The first broad run recorded the old line-number assertion failure, while its other 416 cases passed and three skipped. After replacement, the new checkpoint case's assertions passed but fixture cleanup incorrectly called a nonexistent `close`; its cleanup now uses the owner's existing context-exit API. The final load-grace action passes all 25 cases. Both failed actions remain in the evidence, and neither is presented as a passing action.

Final action evidence:

- `d117ae6bf5e85afd2b1054c73cccf26a33276e5af6131d2d81fcdaacd4567799`: 85 passed, one skipped.
- `a3bcf8bad7fe984172c5112635fb6eadee8b197e1ccbf28064517669b4c7f7c9`: 153 passed, one skipped.
- `266bed68e5a0632eb280eae39bf52a7a820c07a2266757b68e1c584dfbde9e61`: 82 passed, one skipped.
- `a9eed7b1456a932e83f0f9bd8baf3bc8a1fd397e337671beb6788fa9048d5151`: 60 passed.
- `d1a3ecc015b0bbc5815303e05c0736133b4e4ae171b0befe45f03a8b97b5df61`: 25 passed.
- Compile `86c2e7a855ba5e57e3e65364fddbdd0de8b56d91baad2a2b456d3b832261819d`: executed with exit 0.

The remaining unchanged cases passed within the two failed cohorts; the final outcome reconciliation, exact nodeids, commands and failures are in `/home/rob/tmp/pq-refactor-20261014/reports/joint-j4-argv-*`. `joint-j4-argv-final-receipts.json` records actual terminal and CAS state. `joint-j4-argv-final-outcomes.json` deduplicates repeated cases and excludes the retired source-line assertion.

The discovery also exposed stale line-number prose in `LOAD_PHASE_BOUND`. Its separate correction will cite the qualified reader symbols; it changes no grace value or admission rule. R2 and J3 remain explicit rejected wrapper proposals in `joint-r2-j3-disposition.md`: their lifecycle, file and readiness obligations are independent and already have owners.

The separate prose correction now cites `prismaquant.joint_adjoint_checkpoints.load_adjoint_checkpoint` and `prismaquant.joint_quantum_handoff.load_handoff_inputs` in the load-bound evidence. PB `f7e09a1543e09b08656bf8ebde0ade1adf2b1ffafba22c428c5c25aab09bc590` passes 43 load-grace and seal-inventory cases with no skips. Its terminal state is done and its CAS payload was read and SHA-256 checked. This off-task finding changes only prose carried in the progress stamp; grace numbers, read deadlines and admission rules are unchanged. Evidence remains in `joint-load-bound-prose-*`.
