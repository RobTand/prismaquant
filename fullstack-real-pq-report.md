# Full-stack real PQ chain — delivery report (2026-09-20, R3 repair)

Scope: new PQ-side harness only. No production file edited.
Branch: `test/fullstack-real-pq-integration-20260920`. Issue #847, PR #849.

Files (new, owned by this lane):

- `tests/fullstack_pb_generation.py` — published-PB resolver: resolves the
  fleet link ONCE to a full immutable root, derives generation identity
  from that root's own name, inserts its `src`/`tools/fleet`, verifies
  every published module's `__file__` sits under the same root, and prints
  `[pb-generation] <id> root=<...> core=<...>` into the admitted-action
  output. Candidate-pin integration still waits on root-qualified 730/846.
- `tests/test_fullstack_real_chain.py` — connected storage chain plus
  producer gap demo plus join gates (below).

## GREEN qualification (distinct — this lane only)

- PB entrypoint: `pbtest.py --checkout <this worktree> --python
  /home/rob/venvs/pq-cu130-tessera-cc739a55/bin/python --tag gb10
  --priority -10`, CPU torch env, no model compute.
- Evidence: `/home/rob/tmp/pqchain-r3red3-20260920.json`.
- Queued action `ec6e5b7c83d2`, executed on sparky, receipt
  `747f4e4f1f28b9ff1414474bea4778469a151e81b0759b8d8ee82c2d48a9c39b`,
  runtime generation `0467e9e2316c-1789881139-d2055704fb70` (published PB;
  tessera pin `cc739a55` verified).
- Result: 10 passed, 1 xfailed
  (`test_strict_slice_phases_are_stageable` — a real conditional
  assertion that flips red when the producer zero-head fix lands).

What passed, one connected path per leg sharing the same staged objects:

- Real `layer_quanta` producer on the 3-entry parent; PB `core`
  validation of the slice wire with the zero-head phase table
  demonstrated as `[]` (gap, not conformance; slice staging through the
  real row/planning interface is PENDING the producer worker's isolated
  fix — no hand-built plan is frozen to paper over it).
- Real `stage_move` (whole files + a nonzero source-offset `.pbrange`
  payload range) and `ram_promote` (wire+render+range window) with epoch;
  the PQ map is the real PB `compose` overlaid with the real RAM
  fragments plus the current epoch, with the ram tier record filed beside
  it for the reader's epoch check.
- Real readers over that map: whole-file source shards bit-identical
  (`fallback_count == 0`); the nonzero-offset range serving its tensor
  (`range_hits >= 1`); the wire blob through the real wire reader served
  from RAM (`ram_hits >= 1`); the `.pt` render through the real
  production weight cache served from stage (`hits >= 1`).
- Real join CLI on this producer's own records with cost signs derived
  from actual staged-read bytes: coverage accepted, the complete gate
  (`load_joint_cost_for_allocation`) accepting; the absent-record case
  exiting 0 with top-level `gapped` while the same gate refuses it
  (`GappedPayloadRefused` — exit 0 is the join contract, refusal lives
  downstream; no production join change made here); tampered and
  duplicate-qname payloads refusing (exit 1, no output artifact).
- Uncovered span: unmapped file reads the pool silently (`stages() is
  False`, bit-identical, `fallback_count == 0`, `hits == 0`) — named gap.
  The old always-failing strict-reader test is deleted; strict negatives
  with exact refusals belong to the PR #846 SDK-glue lane.

## RED history, honestly labeled (harness debugging, not regression)

- `pqchain-r3red1`: all setups erroring — `verify_quanta_coverage`
  refusing `tiling stops at 1536 of 5403` because the 6-entry manifest
  broke the producer's layer-phase tiling. Fixed by splitting the
  stageable storage manifest (6 entries) from the producer parent (3
  entries); producer coverage tiles exactly again.
- `pqchain-r3red2`: 1 failed — PWC `requires a tensor shard` because the
  fixture saved a dict. Fixed by saving a bare tensor.
- `pqchain-r3red3`: GREEN (above).

## NOT integrated — candidate pins (distinct, unaccepted)

- PQ PR #846 (`flash/strict-reader-tier-enforcement-20260920`,
  head `494e58017`): NOT merged, NOT integrated here. No placeholder
  tests for its SDK surface.
- PB PR #730 (`fix/pb-reader-lifetime-20260920`, head `a6e6b310a1`):
  NOT merged, NOT integrated here. Context/broker proof belongs to the
  PB worker.
- Reader-lifetime `acquire_for`/`open_pinned`/`release` integration waits
  on root coordination. This lane pins only the published generation.

## Producer correction (root review, recorded not claimed)

Own-generation activation is still unstaged: a read-only attach receipt
is not a lease. Post-capture bulk-read-set producer changes are next
(producer worker owns zero-head + #848) and are not in this report.

## Repro

`python3 /mnt/shared/prismabuild-fleet/repo/tools/pbtest.py --checkout
/home/rob/tmp/pq-fullstack-integration-20260920 --python
/home/rob/venvs/pq-cu130-tessera-cc739a55/bin/python --tag gb10 --priority
-10 --shards 2 --workers-per-shard 1 --threads-per-shard 1 --mem-gb 8
--wait-s 900 --json <path-outside-checkout>
tests/test_fullstack_real_chain.py`
