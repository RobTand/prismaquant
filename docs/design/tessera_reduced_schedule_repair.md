# Offline reduced-stack repair history

CPU slice #1776, Refs #495. This extends the [next-work contract](tessera_reduced_schedule.md); it does not execute the live research campaign.

## Inputs

`plan_reduced_schedule_repair(history, *, max_rounds, expected_previous_sha256=None)` consumes an ordered history of JSON objects using `prismaquant.stack_reduced_schedule_input.v1`. Each snapshot supplies its current assignment and scalar observations. The existing decoder and planner validate each snapshot; no solver callback or replacement allocator is allowed.

A root history has one snapshot and no previous digest. Each appended snapshot requires the exact `binding_sha256` of the preceding history. The caller supplies an explicit positive integer round cap. The cap is bound into the receipt; changing it starts a different root, not an extra round on the old experiment.

The declared stack roster, expert/projection frames, reference/target band, currency, expert weights, byte costs, byte budget and gate configuration remain fixed. This does not certify an unprovided calibration population, serving runtime or artifact identity.

## Transition checks

Before progressing to the next snapshot:

- Keep every previously covered cell.
- Complete every cell requested by the previous plan.
- Supply actual scalar rows for newly covered cells, including every requested completion. A ledger entry alone is insufficient.
- Preserve fixed fields and previously supplied observations as canonical JSON bytes, not Python numeric equality. Numerically equal integer/float substitutions cannot change their canonical identity. Append observations rather than rewriting them.
- Supply the newly re-solved assignment explicitly. Do not substitute the diagnostic gate's greedy assignment.

The wrapper reuses `plan_reduced_schedule`. A changed winner therefore requests only its missing cells when the global diagnostic gate passes; a failed global gate still requests the full missing band for all stacks. The wrapper does not invent per-stack failure verdicts.

`rounds_used` is the number of appended snapshots. Reject histories exceeding the cap, or reaching it with work still outstanding. Replaying identical history and configuration returns an identical receipt.

## Output boundary

The versioned receipt binds each validated input and plan, records winner changes and coverage additions, and carries the current next-work plan. Its states are `awaiting_measurements` or `scalar_coverage_only`.

Both states retain `wire_ready: false` and `production_solver_run: false`. A history checksum detects accidental replay/input drift; it is not an authenticated measurement receipt. Scalar coverage is not convergence, verified wire readiness, production-DP regret or a quality result.

There is no encoder, dispatcher, fleet placement, cache, measured-row replacement in the live cost store, or automatic re-solve. Actual execution and numerical/resource acceptance remain #495 and are HELD under the coordinator's 2026-09-29 23:25Z ruling. Do not modify running resumes or protected release inputs.
