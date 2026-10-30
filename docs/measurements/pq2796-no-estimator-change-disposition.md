# #2796: no-estimator-change disposition for TrustLadder

## Decision and status

Take #2796's no-estimator-change branch: the supplied material does not evidence
W2's causal-verdict prerequisite. This records an unresolved mechanism, not a
negative W2 verdict, a demonstrated estimator defect, or resolution of #2796 or
parent #1962. W4 remains held until an actual demonstrated, cause-linked fix has
merged. Dispositioning this documentation does not release that hold.

This change is documentation-only. Production estimator/statistics/replay logic,
pricing, probe SE, sidecar reconstruction/component gates, and existing tests
remain unchanged. No speculative regression is added: without a named defect it
would encode a speculative causal verdict. No A shift, W stability, or dense/shared
A delta versus probe SE has been measured or is asserted here. No scale factor,
format ban, probe-SE change, imported Tessera runtime, environment-specific
behavior, GPU job, download, or checkpoint is introduced.

## Evidence and its limits

The inspected working checkout is at
`6d8288adc94b3c4b624c143ed372fb6513f32784`. The supplied research handoff reports
that it and the served PrismaQuant checkout at the same revision supplied no
authenticated W2 verdict. This is bounded absence of usable evidence, not an
explicit negative verdict and not proof that no verdict exists elsewhere.

* The expected mechanism owner is `prismaquant/joint_aura.py`.
  `SignedJointProjectionLease._observe` contracts W, A and mixed terms from
  observed inputs and output cotangents, and `finish_probe` sums the signed
  components (`joint_aura.py:513–590`). `JointOperatorStatisticsLease` shares
  observation arithmetic with invocation spill replay through
  `_observe_invocation`; row-chunk replay shares `_observe_rows`
  (`joint_aura.py:591–606,713–802`; `joint_replay_spill.py:1925–1930`). Deferred
  statistics explicitly have a distinct rounding identity. These are ownership
  boundaries, not evidence of a defect in either path.
* [The attribution contract](../design/joint_aura_runtime_allocation.md),
  lines 144–207, keeps authoritative whole-draw pricing separate from block
  attribution. Total and W/A/mixed reconstruction must pass independently;
  block geometry must be genuine. The existing uncertainty is probe-only,
  conditional on calibration, not calibration uncertainty. The document
  expressly leaves #1962's estimator question open. No gate is relaxed here.
* Targeted measurement inspection confirms that
  `/mnt/shared/tessera-measurements/pq1962-sol-20261002/glm-comparison-audit.json:244–255`
  sets `exact_subset_price_derivable=false`. The retained whole-draw components
  sum over 512 complete sequences; they cannot recover the selected 25 sequences
  and output positions 0–510. Dropping activation position 511 would still retain
  the output-position-511 Fisher contribution in earlier activation cotangents.
  The audit explicitly does not establish the factor-seven cause, actual GLM row
  correctness, a subset variance bound, or matched standalone-A/joint-network
  Fisher versus actual KL. These are diagnostic precursors, not a W2 verdict.
* The inspected local replay reference file
  `pq1962-sol-20261002/tmp/pq2572-replay-t3/l05-s000-003/reference-comparison.json:1–5`
  under that measurement root says `no matched historical four-row resident price
  exists` and `used_for_acceptance=false`. A local replay pass cannot substitute
  for that missing comparison or establish estimator correctness.
* [#2571's Qwen3 study](pq2571-qwen3-estimator-study.md), lines 7–15 and 101–106,
  is a **dense Qwen3-0.6B** numerical study with two eight-probe draws, not a
  routed-MoE mechanism diagnosis. Its final section explicitly prohibits reading
  unstable correlations at eight probes as estimator bias; bias-versus-MoE
  separation remains with the parent. Preserve that limitation.

Rob's supplied #2705 comments concern an already-shipped API migration; they do
not authorize an estimator correction.

## Required next diagnosis: existing #2795 / #2853 owner

TrustLadder should request the missing diagnosis through the **existing #2795
replay lane and its open follow-up #2853**, not create a parallel replay
implementation or duplicate issue. Continue #2574's mechanism-fix ownership only
once that lane supplies an authenticated causal verdict containing:

1. Source, calibration, probe and arithmetic identities, candidate identities,
   and retained provenance binding the compared executions.
2. Matched sequence and output-token scopes for Fisher and actual KL, including
   standalone-A and joint-network comparisons. Unrecoverable subsets of
   whole-draw sums are not matched-scope evidence.
3. Genuine signed block W/A/mixed attribution with total **and** component
   reconstruction gates satisfied, without silently replacing authoritative
   whole-draw prices or inventing block geometry.
4. A controlled causal intervention identifying a **named defect**, isolating
   it from scope/arithmetic differences, and predicting the direction of A's
   change. A discrepancy or replay pass alone is not that intervention.

Only such a verdict can justify a subsequent cause-linked production diff and a
seeded, checkpoint-free tiny CPU regression reproducing the named defect. That
future work must demonstrate the predicted A direction, measure W and dense/shared
A controls against probe SE, and preserve the attribution gates. None of those
future measurements is claimed by this disposition. The missing diagnosis is
already owned by #2795/#2853; this handoff does not open another replay lane.

## CPU baseline evidence (not fix or G3 acceptance)

Required command, using tiny CPU fixtures and no downloads or checkpoints:

```sh
python -m pytest tests/test_joint_sequence_attribution.py tests/test_joint_aura_projection.py tests/test_joint_aura_streamed.py -q --basetemp="$TMPDIR/pytest"
```

The supplied research baseline observed exit 0: **118 passed, 14 torch deprecation
warnings in 4.93s**, with retained stdout read back. Its evidence line is retained
verbatim:

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"0aec82d81edb52a1fa4c41e0bd3edd822bd6c906452a9c855df3d2cc000c4a21","size":902}},"operation":"26b09de2c433f76d054643ec7082b5cfb26078c828f2c283e051c56894ffcb17","receipt":"572a54b72783863962836e03256d8968bad29697c9b7d5ccd11698a80953914e"}]

The implementation-session rerun of the same command observed exit 0:
**118 passed, 14 torch deprecation warnings in 4.74s**. Retained stdout was read
back with `read_test_artifact`.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"d7ef3e410c6212f4b4212df604a3b5cd83ea957325a492c470ce63beeab157d7","size":902}},"operation":"e72156d75af750c4f9784ab5678d869fbd79b4b02d91b77db38a302a24b3e765","receipt":"6159922ea9477a48ab26e18c5acdb018ea799175d559ca1ccb7ef9ecf44cbc91"}]

Both runs are baseline evidence only: the suite already passes without an
estimator change. Neither proves a fix, the evidence-qualified disposition, or
G3 acceptance. No new executable estimator acceptance claim is made. W2's causal
verdict remains unavailable in the supplied evidence; parent #1962 remains
unresolved and W4 remains held.
