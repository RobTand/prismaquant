# Resident Hessian commitment reuse, 2026-09-09

On the canonical GLM-5.3-Flash row of 864 resident Hessians (43,486,543,872
bytes), ordinary campaign publication already computes per-unit commitments.
The next capture seal redundantly hashed that population. Tessera PR441 binds
the existing resident tensors to its authenticated metadata owner, and
PrismaQuant issue466 uses that public API after export inputs are accepted.
Per-unit consumption still checks the tensor's digest. This removes no
per-unit validation and creates no additional cache.

## Native comparison

PB `216d26085bed9aea062e8747d6f4b289506001fbbbd570b6817d5dfb92de651a`
ran on Sparky, GB10, torch 2.13.0+cu130, with the qualified container and frozen
producer `7f8aef0dd1ba4ef6a6ff5cf00ee9c2bc06c73e5d`. Producer source digest is
`959a1a43b26865e5e04dfacbabd627634ade4537979568af9c00c76d9604f9ea`;
packaged runtime contract remains
`a688f8de244f936ec3a63a782e20af7985733e7a6fb0b4b981b5fe4c44112212`.
The native PQ snapshot matches `1addeb62f7` apart from PB's closure.

One ordinary selected-row campaign loads the original resident data, publishes
references, then compares six fresh sources: profile plain/resident and timed
plain/resident/resident/plain. Construction and capture sealing are timed
together. All six seals are identical. Plain sources hash all 864 tensors;
bound sources hash zero and read zero payload files.

| Construction plus capture seal | Plain mapping | Resident references |
|---|---:|---:|
| Profiled pair | 39.67255 s | 0.46484 s |
| Unprofiled repeat 1 | 39.34823 s | 0.67525 s |
| Unprofiled repeat 2 | 39.02842 s | 0.72452 s |
| Unprofiled mean | 39.18833 s | 0.69989 s |

The measured saving is **38.49 seconds of startup sealing**. Both profile
arms consume the same actual CUDA Hessian, retain its consumption hash and
produce exactly matching LDL/metric values. Following the comparison, the
ordinary R832/B8 campaign prefix produces 16 actual wires and scores exactly
matching the frozen b1 baseline. Its first `_prepare_anchor` takes 0.01538 s.
This is bounded native qualification, not a complete pricing row, full-model
throughput or served-quality measurement.

Both CPU/CUDA traces are complete. The plain trace has 870 GPU memcpy events,
the resident trace nine; each includes one actual unit's consumption. The
observer records 82 Netdata samples per host, and recovered power has 841/842
samples on Sparky/Sparklina. Subsecond resident sealing is too short for a
precise energy comparison from 2 Hz samples, so no work/J ratio is claimed.

Native exit 0, cleanup, actual stdout, CAS payload/receipt hashes and executed
source were checked independently. CPU artifact audit
`078e56c52471742be144b6dddd932b870e24048529b1fb9e1c7d66ce40b15234`
checks both trace files, annotation categories, actual wire hashes and
checkpoint scores, reference identity, memory guard and telemetry coverage.
Its first attempt `78cafd794245` incorrectly counted CPU and GPU annotations
as one category and failed; the audit was corrected without rerunning native
work. No product change was needed.

Evidence root:
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02/performance-resident-h-seal-ab-01/`.
Root verification records under `/mnt/shared/tessera-measurements/`:
`glm-resident-hessian-native-cas-source-20260909.json` and
`glm-resident-hessian-native-audit-cas-source-20260909.json`.
The exact native invocation is the evidence root with `-invocation.json`
appended. Harness and audit are retained in the separate measurement branch,
not imported by production.

## CPU and regression evidence

Before the integration fix, the new actual-CLI regression failed specifically
at the redundant `tensor_identity` call, PB
`eeedf6f826a8d8960199d40e673b46db23bd9bd86acdda1b150bb4805e944cb2`.
The corrected integration has 136 passes and two CUDA skips across nine
focused files: handoff, priced export, resume, bound unit identity, batch path,
campaign pins, serving pins, architecture and staleness. This total combines
the unaffected shards with corrected affected shards; repeated passes are
not counted twice. Early integration mistakes (binding before publication
and an unsynchronized pin constant) failed seven tests and were fixed before
the native run. Native device behavior is evidenced above, not by CPU skips.

Tessera final API head `43ea9248ae` has 62 focused CPU passes and one CUDA skip,
plus pure CI with 1,594 passes and 98 skips and successful wheel/sdist checks.
Root verified all six focused CAS source snapshots against the exact head.
Records: `glm-resident-hessian-{root-regression-before,fixed-integration-cas-source,unaffected-cas-source,producer-final-cas-source,producer-ci-audit}-20260909.json`
under the shared measurement directory. Source tests cover unchanged capture
and unit identities, mutation/roster/provenance refusals and successful/error
owner cleanup. The producer digest changes conservatively; old priced rows
are not relabeled under the updated producer.
