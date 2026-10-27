# Joint quantum phase ownership

Provenance: 2026-10-14 PrismaQuant architecture refactor, base `138069b7993`.

`joint_quantum_row` owns joint phase spelling, Stage A forward/reverse and Stage B chain/own-source/render/replay/spill consumption order, checkpoint/handoff load names, typed quantum coordinates and per-phase compute work. The producer, runtime, coverage reader, handoff derivation, dispatcher and all tests import that owner directly. No legacy producer or dispatcher phase function remains. The producer still owns manifest construction and byte ranges; the handoff owner still authenticates and publishes handoffs.

Phase factories and ordered-plan builders moved verbatim, including existing integer guards, replay-window coercion, empty-pending window-zero convention and error strings. Their normalized source ASTs match the previous committed functions exactly in `joint-j4-phase-factory-source-invariant.json`. Runtime phase reporting introduces no new per-call data-class allocation. The dispatcher alone parses wire names into frozen ChainSourcePhase, ChainBoundaryPhase, OwnSourcePhase, RenderPhase, ReplayPhase, SpillPhase or NamedPhase values. Its compute-work code branches on those coordinates. Unknown/no-compute names retain the existing no-work or tail-only behavior; incoming-plane entries remain read bytes and do not invent capture groups.

The dispatcher retains measured ceiling/grace evidence, raw-file custody, prepared-render and cotangent admission, spec parsing, container wrapping and PB gateway. This bounded first J4 unit does not yet move the pure payload/PB argv and resource derivation. Its second target remains QuantumRunInputs over bound file pairs, QuantumRowDemand over already-verified policy values, and a pure ordered PB option builder. Moving the dispatcher wholesale into a session would hide independent gates and has been rejected.

Source inspection migrated 27 existing Python files plus the new owner and its behavioral tests. No phase test was deleted. Existing runtime/readset/progress, handoff, prepared-render, callback/ACK and safety inventory proofs remain. New cases parse real wire phase coordinates, assert literal windowed/spill plan order, preserve unknown-name behavior, count capture/replay/tail terms independently and subtract moved incoming entries before pricing work. The duplication baseline does not grow. Stage A naming stays in the same knowledge owner, instead of manufacturing a cross-module copy of its phase-index rule.

All tests, compile and standalone CLI checks ran through PrismaBuild. Immutable capsules were prepared with the root freeze helper, one CPU and 6GiB per ordinary shard, one native thread, priority-10 and no host tag. Compile and CLI each reserved one CPU and 2GiB, preserving PB affinity. The immediate pre-change seven-file cohort passed102 with4skips. The final33-file cohort passes683 with5skips, complete collection/outcomes/teardown. The five skips are one historical campaign-package reconstruction, one CUDA prepared-render check and three administrator-private spill-scratch qualifications. Missing qualification is not counted as a pass. All29 touched Python files compiled and `tools/dispatch_joint_quanta.py --help` executed through its actual standalone import path.

Final test actions:

- `1b1e4d11671fa48fbfcaea94878ad9edfa9736e4089f39e823da08f27cc7f7c4`:102passed,2skipped.
- `05cd691d136d3541f81152f89333fa8e6fa270eb07603d9e4db055b6d4a78b55`:131passed,2skipped.
- `cff661c4e8b3c8e5d0a0f26afa64fe2df6b25078f40db2d0fa90a0a2021801bf`:293passed.
- `3cf8f9f3c0cee4aee526bd107c590acc1cfc5e4b04836f1bde0437b6a2376b0f`:76passed.
- `9e685ccaead8472f34688513c5eeeeac2cf91aa06bab1ad56b5f2c8cf403e785`:81passed,1skipped.

Compile `13c675c6c0f6a6ffe324cfc6f0b2a3cd67794f3e1f8634e695f80f200842e681` and CLI `a5eb3dc0d5067aedccc3634f7cd6cc98a7dd56006b293da30d6e0eaf3a4cdff06` executed0. Full action keys and actual receipts are authoritative in `/home/rob/tmp/pq-refactor-20261014/reports/joint-j4-phase-final-receipts.json`; terminal state is done and a CAS receipt is present for every listed passing action. Full commands/results/failures and the exact skipped nodeids remain in `joint-j4-phase-*`.

The first broad run revealed the old band legacy-row fixture mutating executable_readset without resealing. It reproduced on the unchanged pre-phase capsule as PB `8cd8f2832ab10092d8479d95b9a89c3ecdea26c030ecf547e5df6c747a871071`,1failed and21deselected. Its independent correction is commit `36ae967f57`, which uses the ledger's existing seal operation so the test reaches its intended executable-only gate. The final full band-dispatch file passes. The first broad run also exposed phase naming divided across owners; completing ownership removed that duplication rather than increasing the baseline.

The two migration recipes `migrate_quantum_phases.py` then `migrate_adjoint_phase_owner.py` are under the reports directory, with caller inventories. Apply them to the documented pre-unit checkout, then keep the standalone tool import after its path bootstrap. R2 and J3 dispositions, exact caller obligations and rejected wrapper sketches live in `joint-r2-j3-disposition.md`.
