# One owner for the atomic-write recipe (#1573)

Part of #1301 (epic #1295). Slice after the calibration-cache consolidation
(#1512): the duplication inventory's atomic-write family in
`prismaquant/`/`tools/` carried three module-local re-implementations of one
durability recipe.

## The recipe

"Publish bytes durably; a crash leaves either the old or the new file":

1. `path.parent.mkdir(parents=True, exist_ok=True)`
2. stage beside the target as `path.name + suffix`, open `"wb"` (no chmod —
   the replaced file carries the staging inode's mode)
3. `write` -> `flush` -> `fsync(file)`
4. `os.replace(staging, path)` — the atomic swap
5. `fsync(containing directory)` — so the rename itself survives a host reset

## Sites consolidated (main `39c8305054f3`)

| Site | Was | Now |
| --- | --- | --- |
| `prismaquant/cost_stage_checkpoint.py:69` `atomic_write_bytes` | the recipe, fork-safe suffix `.tmp{host}{pid}` | **owner** (unchanged) |
| `prismaquant/aura_cost.py:155` `_atomic_write_bytes` | same recipe, bare `.tmp` | imports the owner; both call sites (`manifest.json`, unit-checkpoint pickles) use it |
| `tools/reseal_campaign_identity.py:260` `atomic_write` | same recipe, bare `.tmp` | **retained standalone twin** (see below), now cross-referenced to the owner |

## The standalone twin that stays

The retired `must_differ` baseline entry recorded a deliberate boundary:
reseal keeps its row rewrites stdlib-only -- importing the package pulls in
torch, and the tool's single package touch is `consumer_merge`'s lazy import
(`tools/reseal_campaign_identity.py:960`, which does its own `sys.path`
insertion). Delegating its `atomic_write` to the owner would break that
design, so the twin stays local with a comment naming the owner and the
constraint; `tests/test_atomic_write_recipe_1573.py` pins the boundary
mechanically (every `prismaquant` import in the tool must sit inside a
function body).

## The one deliberate delta

The delegating site (`aura_cost`) staged under a bare `.tmp`; the owner
stages under `.tmp{host}{pid}` (`unique_temp_suffix()`, fork-aware, already
imported by `production_weight_cache` and `joint_adjoint_checkpoints`).
The staging name is ephemeral and never the published name, and the
published bytes are the caller's payload — nothing persisted changes.
Per-process staging is strictly safer: two concurrent writers to one target
no longer interleave on a single staging inode. Mode (`open("wb")`, no
chmod anywhere) and the fsync sequence are preserved exactly.

## Byte identity

This primitive does not transform bytes — the payload encoders are pinned
where they already were (`tests/test_digest_calibration_cache_1512.py`,
`tests/test_digest_profiles_1301*.py`). What this slice pins instead, in
`tests/test_atomic_write_recipe_1573.py`:

- the delegation identities (aura's spelling *is* the owner; the reseal tool
  calls through to it),
- the durability ordering: exactly one `os.replace`, file-fsync strictly
  before it, directory-fsync strictly after, no staging leftovers,
- the mode behaviour (replace installs the staging inode's mode — same as
  the copies did),
- the per-process staging suffix (the deliberate delta, stated above).

## Ratchet

Baseline (main `39c8305054f3`, after #1568): near-duplicate pairs >= 0.9: 5;
must-differ pairs: 5; same-name helper groups: 148 (412 definitions); gated
primitive digest sites: 691. After this slice: 4 / 4 / 148 (412) / 691 —
the `aura_cost <-> reseal` pair entries leave both sets (one member is
deleted); nothing else moves. The reseal twin keeps its local copy by the
recorded stdlib-only design above.

## Evidence

- PB byte-identity suite for the encoder refactor commit (commit 1 of this
  PR, `_stream_sha256`/`JsonProfile._encoder`): 6/6 shards green, 766
  passed (`r4-atomic-refactor-targeted.json`).
- PB baseline regen for this tree: `r4-atomic-baseline-regen.json`
  (bytes-pulled and verified against the committed file).
- PB targeted suite over every touched site's tests:
  `r4-atomic-trio-targeted.json`.
