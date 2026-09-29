# Offline reduced-stack next-work planning

This research-only transition connects the existing stack-transfer regret gate
and selective winner-cell planner. It does not schedule fleet work, encode a
weight, qualify a wire, or modify a live campaign. It is a CPU-verifiable slice
of #495, not completion of its measured repair and re-solve loop.

## Inputs

Supply typed `StackRateSample` records, the current allocator's explicit winners,
positive integer byte costs for the declared reference and target rungs, a byte
budget, and a measured-cell ledger. Every stack must declare the same currency,
reference rung, and target band. The ledger must include the measurements used
by each record; unknown stacks, experts, or rates are refused. Scalar evidence
is caller supplied. Its presence does not verify an encoded wire or a CAS
receipt.

The output binds those inputs and the explicit gate configuration with a
canonical SHA-256. A new ledger, winner assignment, or gate configuration is a
new input identity. There is no implicit resume, artifact discovery, or campaign
mutation.

## Transition

- If the existing global regret gate passes, request only missing cells at the
  supplied winners through `selective_encode_plan`.
- If the gate fails, request every missing reference/target cell across all
  supplied stacks. The existing gate has a global verdict, not a per-stack
  failure verdict, so this planner cannot select individual failed stacks.
- An empty request means scalar-cell coverage only. The output always reports
  `wire_ready: false`. Joint AURA must still use the existing exact measured-wire
  intake and its independent admission checks.

The gate compares its existing greedy diagnostic allocator on both arms. It
does not certify regret for the production group-knapsack DP. Budget sensitivity
is reported by the existing gate and does not affect admission. No quality,
serving, speed, energy, or GPU acceptance claim follows from this manifest.

## Execution boundary

Invoke `python -m prismaquant.tessera_reduced_schedule --input INPUT.json
--output NEXT.json`. The input schema is
`prismaquant.stack_reduced_schedule_input.v1`; its fields match the planner's
arguments. Record maps use decimal string keys for expert IDs and rates in JSON.
The output is published exclusively: an existing output path is refused.

The pending rows are data for the existing anchor path. PrismaBuild owns
placement, sharding, retry, and balancing; do not dispatch these rows to machines
in a second scheduler. Re-run the offline transition with explicit new evidence
and allocator winners after measurements. Live reduced scheduling, encoded
winner replacement, production-DP acceptance, and a bounded repair/re-solve loop
remain open in #495. GPU runs require coordinator approval and are listed in
`tmp/PENDING-GPU.md`.
