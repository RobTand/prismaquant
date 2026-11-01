# Unused state prototypes, retired 2026-10-14

Commit `475ca1adb0` deleted `prismaquant/layer_state_cache.py`,
`prismaquant/weight_session.py` and their dedicated tests, because no live
module used them. Two older archive walls still import them:

- `archive/l3_propagated_2026-07-30/`: `kl_measurement_l3.py` and
  `kl_sensitivity_probe.py` import `LayerHiddenStateCache` at module level.
  `kl_sensitivity_probe.py` also imports `WeightSession` inside `run_probe`
  when weight-session mode is on.
- `archive/polish_2026-05-15/`: `polish.py` imports `WeightSession` inside
  `coord_descent_polish` when weight-session mode is on.

This wall holds the last live version of all four files, byte for byte, so
those walls can name what they need. It was added on 2026-10-16 for Claude
review finding R3B-04. Neither module is live. Neither is a supported cache,
residency or weight-swap mechanism: `AGENTS.md` principle 3 names the live
mechanisms.

## What the modules did

- `LayerHiddenStateCache` captured each decoder layer's input hidden state in
  one baseline forward. It then replayed only the layers after a changed
  Linear, for one-at-a-time coordinate descent.
- `WeightSession` materialized a baseline assignment once on the live model.
  It swapped one unit's format per trial and kept BF16 sources so a trial
  could be reverted, for measured polish.

## Provenance

| File in this wall | Git blob | SHA-256 | Added |
|---|---|---|---|
| `prismaquant/layer_state_cache.py` | `7827effcd9afb689e0b5ca4cb931856f1a71c183` | `ec90b280a1170dda9cce580584d3063a5d5d1e6dd43ad180056818dc3ae2296b` | `093b986acc` |
| `prismaquant/weight_session.py` | `c2005f77096d1f1d0e4dbe51a9ff3831ab8045b0` | `e2fffb4ea7939472bfa8dd5115807d8912f09525fd60f555713ffa6b17995374` | `cfdd300342` |
| `tests/test_layer_state_cache.py` | `41e0cfed19225e32c1fbde37bf72736c1ea59cde` | `ab13a385936286eb3c26e8b577fc654a02e00ce81baa4ee485ff41a694a20215` | `093b986acc` |
| `tests/test_weight_session.py` | `155564f07ec537e54b59bf7b318c3e514b92d234` | `e6f5df010dfbbac4f989ab4f90071fcf079e8386fb73fd937238d457558c57ca` | `0842e7c1c1` |

Each file keeps its original path inside this wall. The bytes come from
`3ad314f6baecda677a7742f945463e77f1b965e2`, the parent of the deleting
commit. All four blobs equal the files at the refactor base
`138069b799323435fb5f0b3cec3ee339f6264a62`. The modules last changed in
`d53a523996` (`layer_state_cache.py`, 2026-09-25) and `a1b7e2d1e5`
(`weight_session.py`, 2026-10-04).

The same commit also deleted tests from files that stay live: the
weight-session snapshot refusal in `tests/test_campaign_write_destinations_2231.py`
and the dead-vendored-override refusal in `tests/test_vendored_override_gate.py`.
Recover them from `3ad314f6ba` if the modules return.

## To restore

1. Move `prismaquant/*.py` back to `prismaquant/` and `tests/*.py` back to
   `tests/`. The tests import the live package, so they cannot run from this
   wall.
2. Add `weight_session` back to `CORE` in `tests/test_tessera_core_boundary.py`.
   It was removed on 2026-10-16 because the name no longer matched a file.
3. Expect `tests/test_duplication_baseline.py` to fail on the same-name helper
   `_replace_tensor_input`, which `layer_state_cache.py` and
   `perturbed_x_cache.py` both define. The baseline only shrinks, so
   consolidate it on `perturbed_x_cache` or record a reviewed exception.
4. Edit `prismaquant/weight_session.py:37`. It imports
   `production_weight_cache._cache_weight_leaf`, which `2379886723`
   (2026-10-15) moved to `render_cache_paths.cache_weight_leaf`.
5. Re-add the `docs/ARCHITECTURE.md` text that `475ca1adb0` removed, if the
   restored modules regain those roles.

A static AST check of this wall at the commit that added it found that gap
and no other: every other `prismaquant` import of the four files resolves in
the live tree. Resolution is not behavior. Run the archived tests under
PrismaBuild after restoring. The two consuming walls record their own
resurrection status in their READMEs.
