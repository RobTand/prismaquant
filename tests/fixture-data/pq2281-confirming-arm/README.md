# #2776 retained failed-arm acceptance fixtures

These fixtures support CPU-only publication checks, not a new measurement.
Rob's binding decision `dec-1009-211419-a139` prohibits another GPU run.

## Provenance

The three result JSON files are byte-for-byte copies from the now-readable
retained measurement bundle. `evidence-index.json` records their original
paths, SHA-256 digests and lengths:

- `paired-result.json`: the historical paired report.
- `candidate-result.json`: `gpu-score/latecut41_44_wa-latecut41-44/result.json`.
- `a8s-result.json`: the retained `a8s_wa-gate2/result.json` control.

The index is a **research projection**, not a producer receipt. It selects
bindings from `metadata/handoff-proof-ready-exact.json`,
`metadata/cpu-qualification.json`, `ready-handoff.json`, the GPU result,
and the PrismaBuild request/outcome/receipt for action
`1cb0d64a4d0a745288b23cae33b351aeb2c25d89192f105e2e80f578801cf56f`.
Its original absolute paths and hostnames are historical citations only.
Tests never open them and need no fleet environment variables or model data.
Source identifiers and composite identities must not be described as raw-file
hashes. The index preserves those distinctions.

Input/profile file SHA-256 digests were observed from the readable files in
research; telemetry digests also match the producer result's declared hashes.
For the two profiler traces, hashes are research observations, not a previously
published immutable binding. The complete large manifests, traces and Netdata
series remain in the retained bundle; these tiny fixtures do not replace them.
The GPU receipt is projected verbatim as a JSON object. Its canonical receipt,
manifest, attestation and result digests are distinct from raw-file SHA-256s.

## Report's machine-readable numerical summary

Publish readable prose/tables in `docs/results/pq2281-confirming-arm.md`, plus
exactly one fenced `json` block with the following fields. This makes numerical
accuracy executable without constraining the prose's organization:

- `schema`: `pq2281.confirming_arm.summary.v1`.
- `candidate_action`: full retained action key.
- `primary_teacher`: `teacher2`; `primary_population`: `all25`.
- `delta_definition`: `arm-minus-A8S`.
- `pre_registered_delta_threshold`: the frozen threshold.
- `criterion_met`: the literal outcome of the preregistered comparison.
- `results`: object keyed by `teacher2` and `teacher04`, each containing an
  `all25` object. An optional `clean17` object is checked against the frozen
  subset if provided, but must not replace the primary population.
- Each population object contains `n_windows`, `candidate_mean`, `a8s_mean`,
  `paired_delta_mean`, `paired_delta_se`. Keep at least twelve decimal places
  of numerical accuracy in this block; rounded prose is fine.

For each matched window, d = arm KL - A8S KL. The reported mean is mean(d);
SE is sample SD(d) / sqrt(n), with n windows, not 51,175 independent positions.
This is descriptive across-window uncertainty, not repeat-run uncertainty.
The complete result files retain window IDs, teacher-logit hashes, source and
producer identities, scored-position counts and per-window values, so the test
can verify matching before recomputing. The historical paired report is not
used as a substitute for recomputing both teachers from their retained samples.
The summary also records `original_regression`, the historical
+0.001982807772259931, and `fraction_of_original_regression`, the primary
teacher2/all25 signed delta divided by that value. This descriptive fraction
does not establish additivity or shipping prices.

The other two acceptance tests require artifact filenames/digests, identity
bindings, CPU/GPU action and receipt identifiers, failed preparation attempts,
the per-attempt outcome path, and profiler/telemetry citations in the report.
Use full paths or a clearly declared root plus relative paths. Human review
still verifies the complete assignment comparison, authorization chronology,
preflight duration/metadata-only evidence, resource admission/return codes,
review counters, telemetry alignment and honestly stated evidence gaps.

## Recovered per-position arrays

The four NPY files are adopted byte-for-byte from PR #3023 head
`897fb7b521391ea2af1ebcbdb1b9b2ab74e610ee`. They are little-endian float64,
C-order arrays with shape `(25, 2047)`, 409,528 bytes each. Row order follows
the corresponding result JSON's ordered `windows` list.

| Fixture | SHA-256 |
| --- | --- |
| candidate-teacher2.npy | `43c0dfa831c538d8efa574ab961fe34d9f60e677dbb3a912a7d98346b5b70f69` |
| candidate-teacher04.npy | `25083251ab141744dbee33961bd7a3b19da5c851a42cfebd69de11d643b9ea4e` |
| a8s-teacher2.npy | `5e13b59a9ce27df01357c137ace36c725dd1f72dccaab65ab89e3cae94507a62` |
| a8s-teacher04.npy | `5debf6cff9bf04f0bd2dc06f81c840b087eced3f82ab6f6ce3f4616937235973` |

Candidate historical originals are
`/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/gpu-score/latecut41_44_wa-latecut41-44/per_position_kl.{teacher2,teacher04}.npy`.
Control historical originals are
`/mnt/shared/tessera-measurements/surrogate-diag-20260929/g3/runs/a8s_wa-gate2/per_position_kl.{teacher2,teacher04}.npy`.

The existing numerical test now checks these digests, reconstructs each window
mean from its 2,047 positions, compares it with the result JSON, and computes
the paired delta and sample-SD/sqrt(25) uncertainty from the reconstructed means.
The arrays add a check of per-position reduction that the existing JSON means
cannot provide. They do not authenticate an uninterrupted custody chain or a
teacher producer. The index and three result JSON files remain unchanged.

## Known limits to preserve in the report

The historical run predates `dec-1009-165401-0740`; do not manufacture October 9
Window4/kernels evidence to authorize an October 5 action. The exact handoff
contains an explicit CEO-release declaration, not the primary authorization
transcript. The campaign record outside readable mounts was not independently
read. Sidecar attempt stdout/stderr files named in the outcome were absent in
research, although CAS text output and terminal summaries were readable.
There is no separately established independent result-review count or timed
GPU-preflight interval in this fixture. Missing evidence must remain missing,
not become a fabricated zero, success, or new GPU job. The handoff's "No quality
result yet" sentence is stale relative to the later paired result. Preserve
prior CPU failures and the nanosecond-integer precision repair; use the exact
handoff proof actually bound into the executed request.

## Historical research acceptance run

`python -m pytest tests/test_pq2281_confirming_arm_report.py -q --basetemp="$TMPDIR/pytest"`
ran as a pre-implementation test invocation: three assertions failed because
the requested report was absent from main at that historical revision. There were no collection/import errors or
skips. This is expected pre-implementation failure, not a measurement run.
The retained stdout was read back with `read_test_artifact`.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3071cab6089219244ad9539e66576d9420cb1342577c84883ddc06c1d668a741","size":3676}},"operation":"dfc8f294df1e05f6ca709e666d01514a55019e06288938747729efa09abee846","receipt":"6614e70e4f753604df01bd5f6b59c43fa05fe9f5fa5422a27a0aa001dd61baba"}]
