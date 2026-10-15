# Retired sample-parallel probe execution

Archived 2026-10-15 from integration 9a80849241711e78e7968e992cd06ab081b854c8.
Provenance: 2026-10-14 PrismaQuant architecture refactor, base 138069b7993.

The incremental worker has refused this mode since 2026-09-02. Its source census
producer retired with the Gridbook lane. This archive records the complete
producer, reducer, priority producer contract, pre-edit incremental implementation,
uncalled whole-model pass, tests and operator runbook. These files are history,
not an executable package. Their old imports are intentionally recorded verbatim.

The five manual offline commands prepare-calibration, prepare-worker-source-cache,
merge-importance, build-cover and merge are now removed from the live package.
No live tool or shell entry point invokes them. This is an explicit manual API
retirement, not an inference that nobody ever invoked a CLI.

Live readers use prismaquant.sample_parallel_artifacts for the unchanged recorded
schema/digest, source-census, dense-row and activation-priority contracts. The
incremental CLI retains its early refusal and cache admission validators. Existing
merged ActivationIndex reads retain their independent same-byte and row-stamp gates.
Restoration needs a lane-independent source census and a separately reviewed
execution contract; importing this archive is not a supported restoration.
