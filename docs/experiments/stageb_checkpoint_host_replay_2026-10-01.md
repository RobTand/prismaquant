# Stage B checkpoint host replay

## Scope

PQ #1367 moves retained-window AURA encoding, hashing and atomic publication off the consumer onto the existing bounded IO engine. The default remains synchronous. The explicit publication budget and optional positive job cap bound owned builtin snapshots; durable acknowledgements alone advance units and the ordered frontier. Success drains before finalization. Failure preserves its published prefix.

The coordinator approved replay of the existing 867 measured unit states as CPU/equivalence acceptance. This is not a fresh quantization, a GPU A/B, or evidence of a 22.6-second reduction. Representative before/after in-process profiles, next-window lag and both-box Netdata are carried under PQ #1253 to the next real Stage B production row. No default promotion or GPU speedup is claimed.

## Inputs and execution

- Read-only source: `/mnt/shared/tessera-measurements/ws-ra-1291/prof-1/layer-quanta/layer-007/checkpoints`.
- Source manifest SHA-256: `ef1802d5dee959f38f59d5ca0bdf72ca19f767c339edd2317fe5f52f024c698c`.
- Existing identity: `2ee6eb4f5919d46f863f05c0ea1e69d719a880f3b10318f6d3e31e0910e6e977`.
- Fresh outputs: `/mnt/shared/tessera-measurements/pq-stageb-1367-host-replay-20261001-03/{synchronous,shared-engine}`.
- Portable x86 PrismaBuild action, priority -10, CPU4/memory12 GiB, native threads1, deadline1200s. Actual worker `dl380g10`, Python3.14.4, torch2.11.0+cpu; Tessera immutable commit/RECORD preflight matches `b40c93cb73745097e57a1ba4cf5b9eee166c759a`.
- Source `27ac3f6c53fdcaab2e8c2dc9413809bb8cb012b2`; final test-only corrections preserve all five production/utility Git blobs exactly.

After compile checks and the targeted 153-case suite, the admitted action ran:

```sh
python experiments/stageb_checkpoint_host_replay.py \
  --source /mnt/shared/tessera-measurements/ws-ra-1291/prof-1/layer-quanta/layer-007/checkpoints \
  --destination /mnt/shared/tessera-measurements/pq-stageb-1367-host-replay-20261001-03 \
  --expected-units 867 --host-windows 14 \
  --budget-bytes 4294967296 --max-jobs 4
```

The factory reserves before construction, caps the file read and a conservative builtin decode allowance (`128 * encoded_bytes + 1 MiB`), and refuses reducers, external buffers and sparse memo allocations before decoding. Its maximum source file is6,560,466B, giving840,788,224B allowance beneath the1GiB slot. Existing AURA integrity decoding is shared, not duplicated. Scoped leases retire decoder-owned envelope/state cycles on every exit; independent publication clones retire before reusable credit. No cyclic GC or borrowed graph mutation is required.

## Observed equivalence and ownership

| Check | Result |
|---|---:|
| Existing source units / bytes |867 /5,681,078,212 |
| Historical-source/control/shared-engine file matches |867 /867 /867 |
| Consumer encoder calls in synchronous control |867 |
| Shared `pq-io` encoder calls in candidate |867 |
| Durable acknowledgements / idempotent resume skips |867 /867 |
| Final charged bytes / pending units / held jobs |0 /0 /0 |
| Peak charged publication bytes |2,147,483,648 |
| Pending-window metadata ceiling observed |2 |
| Source manifest and every source file unchanged |true |

`files.json` SHA-256 is `ac795330ccd7e8cdf2baf9f66355fe3adb4097c62521f7f288a7c3d6808a8d26`; it records every source and matching output digest. The historical record does not carry resolved unit membership, so the explicit balanced host partitions (`61`, then thirteen `62`s) are not claimed as the original GPU windows.

Action `aeef9cc599a90e60814e06c4cff4198ec886a7c816fb2083973dd3ba0f34c430` ended0/complete/unambiguous after966.718s, including setup, regressions and both replay arms. Its suite was153passed/0failed/0skipped. CAS receipt `ce655af8a602bb7e65f49c9d572772c074618fd2bd156526da8cfa257474d0ed`, result `a170d3b7553bad17d92bf18bd94d27394fbe71b26cba66d28cd39d994bade2d4`; local claim `9711839567b5ed5a66e08c64d91468586f2813f7d3eae0f6801a594d72ad6a4c` passed receipt binding and payload hashing. This check does not independently attest the worker. Logs: `/home/rob/tmp/claude-campaign-20260926/tmp/p2p3/prismaquant/pq-stageb-1367-source-lease-green.log`.

## Regression and review corrections

Genuine RED03581ab6... was7failed/3passed/0skipped: four GC-disabled clone lifetime outcomes and three predecode bounds. Source-cycle RED `cad493ada1e5ea1bca92d36af4cba221b34ace826da4b61c375257166ff662d3` was1failed/4passed/0skipped despite matching five-state output bytes and zero publication credit. Scoped source leases fix that lifetime gap.

Static review also found a test injected snapshot refusal before constructing its shared source and omitted nested unit files from its hash census. Test-only RED `6dbb9d3eb34ae3518a524afaa6f65d338fd3defa897209c3794b2cfa03c69d35` was1failed/0passed/0skipped (`assert 5 == 6`). The corrected test refuses at `snapshot_bound` after factory invocation, requires decode counts10/1/6 for success/synchronous failure/shared snapshot failure, and verifies all six source files recursively with GC disabled.

Final-source tests/compile/pin action `dea9a5560262512f3fbc612891ed5a1550f6fa8b4b0b34091ca3c92c3b1fcafd` ended0/complete/unambiguous:153passed/0failed/0skipped,49.135s action. CAS receipt `589537119a18b01f091f5b1acebccf3252902d889254273c87d9a44deb8c189e`, result `b252f7dc269c53b0cb0ad80d4c1f95c501da34814b88a3507994210b1416c3cd`. All production and utility blobs are unchanged from the 867-state action; the large replay was not repeated for test-only edits. Read-only same-role Sol review accepted the implementation and the final correction; it did not execute or independently authenticate these runs.
