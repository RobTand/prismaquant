# Polish Archive

Archived 2026-05-15 per user direction: "Polish, as well, as long as it's
not doing anything major." Confirmed it's not load-bearing:

- `run-pipeline.sh` does not invoke polish (only mentions it in a comment
  about what KL paths measure).
- `prismaquant/__init__.py` references `polish_from_assignment.py` in a
  docstring describing the workflow but does not import it.
- No other module in `prismaquant/` or `tests/` imports polish.

## Contents

- `prismaquant/polish.py` — measured single-flip coord-descent polish:
  for each pass evaluates every single-unit format flip and accepts the
  one with the lowest measured KL provided it strictly improves the
  metric. Fused-sibling aware. Pure measured KL gating, no surrogate.
- `prismaquant/polish_from_assignment.py` — CLI wrapper: loads a
  kneedle-style assignment, optional ProductionWeightCache, runs
  measured single-flip polish, writes polished assignment + trace JSON.

## Related (live in tree)

- `decision_units.py` — the fused-sibling unit / pair API both polish
  modules use. Stays in the live tree; load-bearing for
  `production_recache.py` and `production_weight_cache.py`.
  `weight_session.py` used it too until it was retired on 2026-10-14; it is
  archived at `archive/state_prototypes_2026-10-14/`.

## To resurrect

Move both files back to `prismaquant/`. Archiving them removed no live call
sites, but the live tree has changed since. A static AST check on 2026-10-16
(Claude review finding R3B-04) found two gaps:

- `polish.py:324` imports `prismaquant.weight_session` when weight-session
  mode is on. That module was retired on 2026-10-14; restore it from
  `archive/state_prototypes_2026-10-14/`.
- `polish_from_assignment.py:202` imports
  `prismaquant._fast_kernel_guard`, which was removed on 2026-07-30
  (`08612a4a46`), before the 2026-10 refactor. Remove or replace that guard.

All other `prismaquant` imports in these two files resolve in the live tree.
Resolution is not behavior; re-qualify polish before relying on it.
