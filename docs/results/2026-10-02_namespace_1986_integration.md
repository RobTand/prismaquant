# PQ1986 namespace/import correction integration

Base: current main `6707ee494a504195babd355e77d76343233d738d`.
Original PR2007 source `f31acaae0ca64bdce4411debac1786f89939a39e` and component
worktrees are preserved. This integration retains the original ordered commits
and adds mount/import components without changing their behavior. The only
original-main merge conflict was the architecture prologue; both complete
provenance sections were retained. Component test additions merged together.

## Components and historical causal evidence

- Mount refactor `78b5c117c647c501ba1b2ed13e65a050237857be`, correction
  `5d710072dfda7eed67dd1f37b86a442fb96e09bc`: one eager canonical mount view
  checks all sources/targets before output containment and writable identity
  coverage. PB RED `ac005fc48ad1cb1bd692e96f366dd73b6d151d123d15c71e2d0d4d802cc09336`
  observed6 failures/1control pass; GREEN
  `2c5d3df38ced36ee61d1e5c4b97cbae0c00e5ed945730426600f77c28f4481a1`
  observed22passes/no skips plus compile2.
- Import refactor `dbe7f2501d20cf4665c970e35f1701ac8ec36948`, correction
  `6076183be5965eb932e2c0af6e159fba05b10ad2`: the existing selection owner
  rejects earlier real mounted source/bytecode/native-looking shadows only for
  opt-in namespace admission. RED
  `ad21e1457489d6d41abc1568cbc9dd82459a5276e497947cf51d90f8be8c5449`
  observed1failure reaching Docker sentinel; GREEN
  `8e60d9c5e8156c950f07c4808ea8a13009b99d6378b9e26be18bc7e4757eab21`
  observed22passes/56deselected/no skips plus compile2.

These are component results, not a fresh integrated-head qualification. Both
successful claim/receipt/payload populations were independently read with all9
named checks passing. Failed causal actions have no success receipts. No valid
component action or historical full-stream timeout is repeated or called a pass.

## Boundaries and final gate

The one integrated gate covers full namespace/adapter modules plus the existing
IO/duplication/no-new-seals/architecture/staleness ratchets and touched-module
compile, through published PB from dl380 with pinned PB95a/Tesserab40 packages,
explicit aggregate CPU/memory, bounded native threads and retained PB affinity.
Its exact command, source snapshot, observed population, terminal and CAS
records are banked outside source at
`/home/rob/tmp/claude-campaign-20260926/pi/native-pq1986-integration/`.
The final observed result is recorded there, not inferred from submission.
Full suite remains owned by the central integration batch.

Namespace request/roster/readset/provenance expectations bind caller-supplied
metadata; preparation does not authenticate artifacts, derive runtime demand,
reconcile actual completed work or establish compatibility. Writable outputs
need canonical identity-mapped durable coverage, and temp creation retains the
existing descriptor/no-follow owner. The import guard is a static declared-root
admission check, not a general Python interpreter or container ABI qualification.
A read-only bind mount does not prevent host mutation and is not source-generation
immutability. PQ2010/PQ2008 immutable capture remains a separate correctness seam.
No source gate, format/default/pin/image/serving gate, numeric currency, production
cache or GPU/serving path is changed. No real campaign row is selected or rerun.

Only the bounded #1986 namespace/import criteria are proposed for closure after
Astra acceptance. #1588's actual reconciliation/compatibility/resource/missing14
prices/all42 qualification and #1842 remain open. Existing legacy absent-contract
behavior, seed/checkpoint/journal gates and all raw failed evidence are retained.
