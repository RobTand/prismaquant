# Archived PrismaQuant experiments (2026-11-09, migration slice 6)

**This code is archived. It is not executable and nothing in the live tree imports, runs or tests it.** It drove finished experiments
(native-MoE launchers, the pq87/pq183 GPU A/B family, the LFM mixed-serving drivers, chained docker launchers, the doorway checks of the
retired execution system) and its scenario tests. Its history is in Git; the dated records of what it measured are under `docs/` and
`experiments/measurements/`.

The execution system these files submitted to is retired. Do not revive a file here: write the experiment again against the native
Flux stack (`prismaquant/flux_jobs.py`, `prismaquant/native_results.py`). Library behaviour that a test here pinned was ported to a test
of that library (`tests/test_serve_fingerprint_descendants.py::test_normalize_performance_argv_masks_only_arm_identity`).
