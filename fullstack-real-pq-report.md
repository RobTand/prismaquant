# Full-stack real PQ chain — delivery report (2026-09-20, slice-connected)

Scope: new PQ-side harness only. No production file edited.
Branch: `test/fullstack-real-pq-integration-20260920` (rebased onto
`origin/main` at `c4e50bfdb8`, the #852 merge). Issue #847, PR #849.

Files (new, owned by this lane):

- `tests/fullstack_pb_generation.py` — published-PB resolver: resolves the
  fleet link ONCE to a full immutable root, derives generation identity
  from that root's own name, verifies every published module's `__file__`
  sits under the same root, and prints `[pb-generation]` identity into
  the admitted-action output. Candidate-pin integration still waits on
  root-qualified 730/846.
- `tests/test_fullstack_real_chain.py` — slice-connected chain (below).

## GREEN qualification (distinct — this lane only)

- PB entrypoint: `pbtest.py --checkout <this worktree> --python
  /home/rob/venvs/pq-cu130-tessera-cc739a55/bin/python --tag gb10
  --priority -10`, CPU torch env, no model compute.
- Evidence: `/home/rob/tmp/pqchain-slice1-20260920.json`.
- Queued action `de1824345ab7`, executed on sparky, receipt
  `14c8742add476775d7e9d6f1ae2b9794375b30de3ddc7ccdc1e040ea9bac44cf`,
  runtime generation `0467e9e2316c-1789881139-d2055704fb70` (published PB;
  tessera pin `cc739a55` verified).
- Result: 10 passed, 0 xfailed. The zero-head strict xfail is gone: its
  assertion is now a positive one.

What passed — one connected slice path, no substitute manifest:

- Real `layer_quanta` producer on the 6-entry parent (layer-0 holds its
  shard plus the wire blob; layer-1 its shard plus the render plus a
  nonzero-offset payload range). Producer #852 seals chunk-only slice
  phases; PB `core` validation plus `tiers.manifest_phase_ranges`
  returns the real ranges for both slices (names match, tile exact).
- Real `stage_move`: each slice in two entry-aligned legs under distinct
  keys (4 stage movers), and real `ram_promote` with epoch (wire span on
  layer-000, range span on layer-001). Each slice's PQ map is the real PB
  `compose` of its own fragments (selected by wire digest) overlaid with
  its real RAM fragment plus the current epoch, with the ram tier record
  filed beside it.
- Real readers over those maps: whole-file source shards bit-identical
  off both slices (`fallback_count == 0`); the nonzero-offset range
  serving its tensor (`range_hits >= 1`); the wire blob through the real
  wire reader served from RAM (`ram_hits >= 1`); the `.pt` render through
  the real production weight cache served from stage (`hits >= 1`).
- Real join CLI on this producer's own records with cost signs seeded
  from staged-read bytes and activation maxima measured off the staged
  tensor, rows sealed by the real `make_joint_aura_entry`: coverage
  accepted and the complete gate accepting; the absent-record case
  exiting 0 with top-level `gapped` while the same gate refuses it
  (`GappedPayloadRefused` — exit 0 is the join contract, refusal lives
  downstream; no production join change made here); tampered and
  duplicate-qname payloads refusing (exit 1, no output artifact).
- Uncovered span: unmapped file reads the pool silently (`stages() is
  False`, bit-identical, `fallback_count == 0`, `hits == 0`) — named gap.
  No always-failing qualifier remains anywhere in this file.

## Honest remaining gates (owning workers, not this lane)

- Strict SDK enforcement (PR #846) with real context/proof wiring: no
  placeholder tests for `acquire_for`/`open_pinned`/`release` are added
  here; integration waits on root coordination.
- Post-capture bulk-read-set producer work (#848) is not in this report.
- No file-reading activation loader with a residency seam exists in
  production (surveyed: residency is wired into source-shard, PWC-file,
  and wire-blob reads only). Activation coverage here is the measured
  max-abs sealed into real cost rows; a staged activation-file read is
  named as pending, not faked.

## RED history, honestly labeled (harness debugging, not regression)

- `pqchain-r3red1`: all setups erroring — `verify_quanta_coverage`
  refusing on the 6-entry single manifest. Fixed by splitting the
  stageable storage manifest from the producer parent (R3 era).
- `pqchain-r3red2`/`r3red3`: PWC dict-vs-tensor fixture fix, then GREEN
  10+1xfail on the substitute-manifest chain (superseded below).
- `pqchain-slice1`: GREEN 10 passed on the slice-connected chain after
  integrating merged #852 and deleting the substitute manifest.

## NOT integrated — candidate pins (distinct, unaccepted)

- PQ PR #846 (`flash/strict-reader-tier-enforcement-20260920`,
  head `494e58017`): NOT merged, NOT integrated here.
- PB PR #730 (`fix/pb-reader-lifetime-20260920`, head `a6e6b310a1`):
  NOT merged, NOT integrated here.
- This lane pins only the published generation `0467e9e2316c`.

## PB726 note

PB component tests merged upstream (`6deb388f`); the old PB worktree is
retained detached. No edits made or needed there.

## Repro

`python3 /mnt/shared/prismabuild-fleet/repo/tools/pbtest.py --checkout
/home/rob/tmp/pq-fullstack-integration-20260920 --python
/home/rob/venvs/pq-cu130-tessera-cc739a55/bin/python --tag gb10 --priority
-10 --shards 2 --workers-per-shard 1 --threads-per-shard 1 --mem-gb 8
--wait-s 900 --json <path-outside-checkout>
tests/test_fullstack_real_chain.py`
