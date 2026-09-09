# Explicit campaign family restriction

Delivery for #449. Production source `36c6ba8cb2f4e4193b40067a88b546094967cd2e` adds opt-in dense/routed family scope to the existing campaign, checkpoint identity, seed intake and fanout merge. It uses the existing topology resolver and menu expander. The option grants no native or release qualification.

PB CPU action `b421bbcb08c55bdad581c3f704acb5eac723e35f96add3f239647d79707f5fa1` ran the four files in `cpu-invocation-03.json`: **86 passed, 2 skipped**, 28 deprecation warnings, 12.38 seconds. dl380g10, Python3.12, CUDA disabled, two reserved CPUs, 4GiB total memory, two pytest workers, native thread limits1. The captured summary does not identify the skipped node IDs; they are not passes. All four requested files existed and collected tests. `root-cas-source-audit.json` verifies terminal rc0, completed cleanup, CAS receipt digest, result bytes and source bundle; the snapshot differs from the production commit only by PB closure metadata.

Campaign compile action `d53784643e061eb89535ff126e866b450889c453a12276fe544f2069a505812e` exited0 with an empty successful output blob; its independently verified receipt/source is adjacent. Both compile and tests ran through PB. No GPU performance, quality or serving claim follows from these CPU checks.

Retained failed setup attempts: the first submission had no eligible interpreter without the x86 dependency tag and did not execute. Action `e055fa57d52ff54ef743a0b3e2c9e02d034ab4c85b079016c55e408654205323` then used a misspelled test path and exited5 with no tests collected. Neither is counted as validation. Raw commands and terminal evidence remain under `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/family-restriction-delivery-01/`.

The implementation was previously developed on the frozen first-release planning branch at `f9432892b920211e82d831cb981cb26875099db2`. This delivery cherry-picks those exact production/test bytes onto main `e4ed1cd863`; only the architecture provenance was reconciled with intervening documentation. Historical regression and broader receipts are retained under `docs/experiments/glm_first_release_rate_band_20260908/` on that planning branch. The fresh delivery result above is the validation claimed here.

Dispatcher compile action `4ff03fc8efaae1bc90aa4c5a01396d8d728dc49f43ca341d02362bd64e1e7bd7` also exited0; independently verified CAS/source audit is adjacent.
