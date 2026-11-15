# Atomic byte publication

`prismaquant.cost_stage_checkpoint.atomic_write_bytes` owns atomic byte publication.
The writer stages bytes beside the target and replaces the target after the file flush.
It then flushes the containing directory.
`aura_checkpoint_journal` uses the same writer.
The unused reseal writer and its publication test are removed.
`tests/test_atomic_write_recipe_1573.py` checks the shared writer's publication order and bytes.

## Evidence

- PB byte-identity suite for the encoder refactor commit (commit 1 of this
  PR, `_stream_sha256`/`JsonProfile._encoder`): 6/6 shards green, 766
  passed (`r4-atomic-refactor-targeted.json`).
- PB baseline regen for this tree: `r4-atomic-baseline-regen.json`
  (bytes-pulled and verified against the committed file).
- PB targeted suite over every touched site's tests:
  `r4-atomic-trio-targeted.json`.
