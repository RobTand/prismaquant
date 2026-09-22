# PQ #917 prepared-input completion

Implementation: `c53041847f`; current-main integration: `bfd92afc9c`;
integrated fixture and compatibility updates: `451c486757`.
PR: https://github.com/RobTand/prismaquant/pull/923. Closes #917.

Normal `regenerate_joint_quanta.py --executable-readsets` now loads the bound
production pickle once and derives each layer's retained-window contracts from
verified cells and existing planners. It seals complete candidate membership,
verified digests, current file sizes and whole-file offset zero. Preflight and
runtime window disagreement refuses instead of regrouping. Payloads are not
re-rendered or rehashed. The binder carries exact manifest entries to the
runtime; dispatch checks those entries and their prepared identity.

The actual production `before_window` calls `prepare_retained_window_read`:
enter the render phase, then use the existing bounded staged-span readiness
API before opening the retained PWC loading pool. Real lease checks remain
authoritative for reads. This adds no mover, scheduler, global barrier or HDD
fallback.

The new integrated regression begins at the normal generator CLI, then uses
the normal dispatcher CLI with only its submission transport replaced. Its
unchanged generated manifest becomes the private queue's CAS input. The test
uses real PB fragment/material writers, map composition, QuantumProgress,
production readiness and the strict leased PWC. The second window has neither
a staged file nor publication until its render phase has been entered. The
fixture proves a real readiness poll precedes pool loading, byte equality,
render-before-replay ordering, positive staged bytes, zero pool bytes, serving
pin IDs, and pin release. Tiny copies/publication run in an admitted private
queue fixture; this is CPU integration evidence, not a live fleet-mover or GPU
campaign completion claim.

## PrismaBuild validation

All runs: published `pbtest.py`, portable `--tag gb10`, priority -10, one CPU
and one native thread per shard, 4 GiB per shard, no GPU. The six-file
regression used PB's six independent shards, with aggregate peak declarations
of six CPUs / 24 GiB. Pytest and both pinned dependency guards ran; no skips or
missing tools. Torch deprecation warnings and expected CPU-only CUDA profiler
warnings are retained in logs.

| Suite | Passed | Action |
| --- | ---: | --- |
| prepared input bridge | 6 | `fbdd109d1853` |
| prepared render inputs | 15 | `90fc0f8b59d1` |
| executable readset | 19 | `91a74bc4ad03` |
| dispatcher | 33 | `ea2b3af8dc1b` |
| source/render exclusion | 8 | `871666e9b0e8` |
| source coverage | 10 | `1f246e261e0a` |
| metadata generation | 27 | `406f10d73d77` |
| quantum runtime | 30 | `ecd6a592bc9d` |

148 passed. All eight terminal records are `done`, rc 0, their stdout sizes
match terminal metadata, and all CAS claim/receipt/payload checks pass including
payload SHA256. Worker attestation verification is outside `pb_verify_claim`'s
scope; it reports that limitation explicitly. Both GB10 hosts executed work.
Observed cgroup peaks: 0.58–1.17 GB; wall times: 6.6–18.5 seconds. These are
qualification resource observations, not performance comparisons.

Evidence outside the sealed source tree:

- `/home/rob/tmp/stageb-astra-final.json`: both Stage B suites and receipts.
- `/home/rob/tmp/stageb-astra-regression.json`: six regression suites and receipts.
- `/home/rob/tmp/stageb-astra-verification.json`: full keys, CAS checks, terminal
  outcomes and bounded resource summaries.

Earlier REDs `8c7c53b534d1` / `5d8dd89f6ca7` retain the original missing-route
regression. Completion fixture attempts caught three fixture assumptions:
source paths outside its plan model directory, missing boundary zero in a
single-layer receipt helper, and local paths outside the generated manifest's
`/mnt/shared` mount. The final integrated fixture supplies all boundaries and
uses a unique, cleaned-up shared directory so it exercises PB's unchanged
manifest validator instead of rewriting the generated declaration.

## Campaign gate

No production metadata, calibration, prepared pickle, render payload or live
runtime was changed. Full-512 Stage A proof still precedes production Stage B
metadata regeneration. Production 360-window geometry and 197,990 renders /
36,423 qnames were not regenerated or measured by these CPU tests. Existing
records remain intact; the derivation refuses planner partition drift. The
original partial report is retained and explicitly marked superseded.
