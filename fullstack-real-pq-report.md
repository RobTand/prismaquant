# Full-stack real PQ chain — delivery report (2026-09-20)

Scope: new PQ-side harness only. No production file edited.
Branch: `test/fullstack-real-pq-integration-20260920`. Issue #847.

Files (new, owned by this lane):

- `tests/fullstack_pb_generation.py` — published-PB resolver (immutable
  generation via readlink, fails loudly when unresolvable).
- `tests/test_fullstack_real_chain.py` — real producer → PB validation →
  movers → readers → join, plus gap posture.

## GREEN qualification (distinct — this lane only)

- PB entrypoint: `pbtest.py --checkout <this worktree> --python
  /home/rob/venvs/pq-cu130-tessera-cc739a55/bin/python --tag gb10
  --priority -10`, 1 shard.
- Evidence: `/home/rob/tmp/pqchain-green1-20260920.json`.
- Queued action `a110ea5dd841`, full key
  `a110ea5dd8413a67649f753afe09249b724c048075b555a1012eddecf1732d04`,
  executed on sparky in 5 s, receipt
  `012d22e50430f08dae86ebcf3018d0221346d20b477a989aa4eb63638e0f702c`,
  CAS blob `b2db3717e025cb1ece13d17588d025ece25fd2b02292056ba4b16abac7113bde`.
- Runtime generation `0467e9e2316c-1789881139-d2055704fb70` (published PB;
  tessera pin `cc739a55` verified, 93 files).
- Result: 7 passed, 2 xfailed (`test_strict_slice_phases_are_stageable`,
  `test_strict_reader_refuses_uncovered_span`), 14 warnings.

What passed: real `layer_quanta` producer on a 3-entry fixture;
PB `core.validate_data_manifest` + freeze; `stage_move` whole + manifest-split
+ `ram_promote`; PQ map composed from the real PB stage fragments
(`pb_map.compose`); `staged_shard_opener` bit-identical reads with
`fallback_count == 0`; join CLI coverage accepted, absent-record named gap,
tampered/duplicate CLI refusals (exit 1).

Gap posture (demonstrated, not conformance):

- Slice zero-byte head phases: `tiers.manifest_phase_ranges == []` with a
  strict xfail that flips red when either side repairs (cross-repo defect,
  filed separately).
- Uncovered span: unmapped file reads the pool silently — `stages() is False`,
  bit-identical, `fallback_count == 0`, `hits == 0`. Strict refusal is
  xfail(strict) pending the owning worker (PQ #845 / PR #846).

## NOT integrated — candidate pins (distinct, unaccepted)

- PQ PR #846 (`flash/strict-reader-tier-enforcement-20260920`,
  head `494e58017`): tier-policy worker passes (372+27) are that worker's
  evidence, not acceptance. NOT merged, NOT integrated here.
- PB PR #730 (`fix/pb-reader-lifetime-20260920`, head `a6e6b310a1`):
  atomic-ref milestone worker passes (123) are the PB worker's evidence,
  not acceptance. Broker proof and real context wiring belong to the PB
  worker. NOT merged, NOT integrated here.
- Reader-lifetime API brief
  (`pb-reader-lifetime-api-brief.md`: `injected_context` / `acquire_for` /
  `open_pinned` / `release`) is INTEGRATION direction only. This lane adds
  no placeholder/invented tests for those symbols; integration waits on
  root coordination after the PQ SDK worker and PB context/broker-proof
  worker land.

## Producer correction (root review, recorded not claimed)

Own-generation activation is still unstaged: a read-only attach receipt is
not a lease. Post-capture real bulk-read-set producer changes are next and
are not in this report.

## Repro

`python3 /mnt/shared/prismabuild-fleet/repo/tools/pbtest.py --checkout
/home/rob/tmp/pq-fullstack-integration-20260920 --python
/home/rob/venvs/pq-cu130-tessera-cc739a55/bin/python --tag gb10 --priority
-10 --shards 2 --workers-per-shard 1 --threads-per-shard 1 --mem-gb 8
--wait-s 900 --json <path-outside-checkout>
tests/test_fullstack_real_chain.py`
