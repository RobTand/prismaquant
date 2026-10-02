# Profile-owned visual allocator namespaces: CPU regression

Refs #1921. Current main already retained complete measured visual/merger rows
in the DP, but its allocator classifier and source-header census recognized
only `model.visual.*` and `visual.*`. Gemma4's declared namespaces are
`model.vision_tower` and `model.embed_vision`. Its rows therefore missed the
visual completeness checks and were later mistaken for source-only fixed BF16
entries. With measured Tessera rows, the final candidate-membership gate then
refused that unintended restamping.

The source census first delegates classification to the existing allocator
predicate (refactor `d2b47ccc0fd32b43754d2d41b129c4b1d95d7023`). The predicate
now reads `ModelProfile.visual_root_prefixes`, with the historical source/recipe
`model.` alias. The detected profile reaches the measured-row partition, source
headers, explicit uniform control, metadata and final bit attribution. No new
name registry, guessed role namespace or second source reader is introduced.
Callers without declared roots retain the old visual control. Audio/body names
and loose root prefixes are refused as visual names. Source-header rank and
non-Linear exclusions are unchanged.

## Regression evidence

RED action `e77ce99345089b754b4cbafaf170fcade014e404802a59ef9076f8187f8aa4e9`
ran the four new real-allocator cases after the refactor but before the behavior
fix: four failures, zero skips, exit 1. They demonstrate measured choices being
restamped, missing-cost and source-shape checks being skipped, and the uniform
control receiving the wrong accounting domain. The tests reuse the existing
synthetic measured-row/contract fixture, changing only declared profile and
visual names; they do not fabricate production measurements.

GREEN action `f98272b05d5499e1204bf76df091c9942cc169197f3844dd635bc38180e38e03`
passed those same four case bodies plus five profile namespace/alias controls
and two actual safetensors header modes: 11 passed, zero skips, exit 0. CAS
result `24ee24cf97154f847f5947b451407b2d4e3ab9c0b9ed4577ba8fe8b7e109ade2`,
17,058 bytes; receipt
`224c909ea2ee4efe31e65a6cfe9faf6be33993c8b2bfbedbc7e3956b13f61c96`.

The related PB file fanout ran nine shards: 144 collected, 144 executed, 144
passed, zero skips, no missing collection. It covers the new cases, existing
visual allocator/uniform/Fisher paths, shard and ModelWalk scopes, architecture
staleness and duplication ratchets. Separate attribution/compile action
`58031bce5f1fa03cbc1dc98fabd6bd39d6aeecb5505a9cc55b3d5ffcc1fe8dc9`
passed seven bit-attribution tests and compiled three touched modules. Those
seven are additional to the 144; the earlier targeted 11 are not counted again.
Its CAS result is
`ab82208fae1057252ac61a28d6d6633164097ff754b334a8ba42ac19f1be26b5`, 4043 bytes;
receipt `a7d0d75ac91231c2b7e6058377906db8eb3201e01d7016426e1c6187cdbcc16e`.

All checks ran CPU-only on dl380g10 through published PB, priority -10,
one CPU/4 GiB per action, native threads one and a 180-second hard bound.
The executing interpreter was
`/home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python`: Python 3.14.4,
Torch 2.11.0+cpu, Transformers 5.16.1, pytest 9.1.1. Published preflight
verified PB `95a59051d48cda82eea7927f31870c6c862d7174` and Tessera
`b40c93cb73745097e57a1ba4cf5b9eee166c759a`, their RECORD bytes and imports.
PB preserved assigned CPU affinity. No GPU work was submitted for this slice.

| Final executable input | SHA-256 |
| --- | --- |
| `prismaquant/allocator.py` | `1515197c9b9fdd5ca89b8be49fa78b00ba7f2e9983a0a6a532f37e5c52394b55` |
| `tests/test_visual_allocator_roots_1921.py` | `a25e25a0e8aef6f0cebfe1cc63c8423d2ba1b7606be787b8b05125fdd8124125` |
| `tests/test_prismaquant_visual_format.py` | `2467a754cf0628835649f5da32b2a4ee1125a16d17e47260a8cb12c323673134` |
| Same-commit `docs/ARCHITECTURE.md` | `8911144088f837f61b6f976d0876320a9b8e756c6e72eff89de58e3addbded24` |

The attribution/compile CAS source bundle is
`bc4d84c3946d52a44ef91e89f6d3d11789d6b13f48f77d3c4ded35711f13f24a`.
Independent bundle readback reproduced the allocator, regression and
architecture digests. Receipt keys, reconciled populations, raw canonical logs,
JUnit and compiled modules are retained under
`/mnt/shared/tessera-measurements/pq-serving-instrumentation-20261002/`;
the fanout report is `vision-regressions.json`.

## Remaining qualification

This is CPU roster/accounting plumbing. It neither supplies empirical visual
prices nor admits a non-body wire or kernel. GLM's current BF16/source-format
vision restriction and the producer/cache/export/serving gates stay in force.
Representative multimodal quality/cost/rate panels, exact original-wire
coverage, native vision/projector recognition, and served quality against BF16
and the matched EXL3 reference remain required by #1921. No KL, byte-size,
throughput, residency or energy improvement is claimed.
