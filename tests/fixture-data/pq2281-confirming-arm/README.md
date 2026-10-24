# #2776 standalone CPU-note acceptance fixtures

These are retained October 5 measurements, not new measurements. Implementation
is docs/changelog only. Tests never access fleet paths, Git history, a model,
or a GPU; all data needed for the CPU checks are committed here.

## Adoption and provenance

Existing work was found on branch `tl/prismaquant-ca44e1548449-wp1`:

- `a7d20474c0bf21a543514294474a1ccca9bdcc9b` supplied the three result JSONs
  and `evidence-index.json`, copied here byte-for-byte.
- `4accbe7cbde18c15e0d6b26c08e87e07c32d3f8a` supplied the matching report
  `docs/results/pq2281-confirming-arm.md`. Adopt that report's useful evidence
  rather than create another report or run. It is absent from this base's HEAD.
  Its #2533 ownership wording is superseded by #2776: #2533 is stopped.

The current issue/child-ledger mapping was not independently recovered. The
branch proves existing report content, not live scheduler ownership or parent
acceptance. No bookkeeping transition is required from the implementer.

`evidence-index.json` is the earlier research projection, NOT a new producer
receipt or a fresh verification of every field. Raw JSON hashes:

- paired-result.json: `1133209b27b114c739fff1d803280d6a33dae1cf949daa7cca0f693eaf11d675`
- candidate-result.json: `168707a152c5f71202da55c8c13823a08a24b811829ca5576d6def3647595d13`
- a8s-result.json: `7b18d2a579df67324c5252c3d1078b7eb3a82de7920217c47bab44c354dd6df5`

Four original per-position arrays were newly recovered and copied byte-for-byte
from the readable retained bundle. All are NPY v1 little-endian float64,
C-order, shape (25, 2047), 409,528 bytes. Array row order is the ordered
`windows` list in each producer result. Raw file hashes:

| Fixture | SHA-256 |
|---|---|
| candidate-teacher2.npy | 43c0dfa831c538d8efa574ab961fe34d9f60e677dbb3a912a7d98346b5b70f69 |
| candidate-teacher04.npy | 25083251ab141744dbee33961bd7a3b19da5c851a42cfebd69de11d643b9ea4e |
| a8s-teacher2.npy | 5e13b59a9ce27df01357c137ace36c725dd1f72dccaab65ab89e3cae94507a62 |
| a8s-teacher04.npy | 5debf6cff9bf04f0bd2dc06f81c840b087eced3f82ab6f6ce3f4616937235973 |

Candidate originals:
`/mnt/shared/tessera-measurements/t8-latecut-L41-L44-20261005/gpu-score/latecut41_44_wa-latecut41-44/per_position_kl.{teacher2,teacher04}.npy`.
A8S originals:
`/mnt/shared/tessera-measurements/surrogate-diag-20260929/g3/runs/a8s_wa-gate2/per_position_kl.{teacher2,teacher04}.npy`.

These absolute paths are historical citations, not application configuration
or test runtime dependencies. The index also distinguishes logical input
identities, raw artifact hashes and canonical receipt digests. The original
GPU CAS text receipt binds captured text, not directly these result JSONs or
arrays. Recovery of bytes does not fabricate an uninterrupted custody chain.

## Numerical publication contract

Retain readable prose and exactly one fenced JSON summary at the adopted report
path, schema `pq2281.confirming_arm.summary.v1`. Fields:

- candidate_action, primary_teacher (`teacher2`), primary_population (`all25`)
- delta_definition (`arm-minus-A8S`), pre_registered_delta_threshold (`0.0015`),
  criterion_met (`false`)
- results: teacher2 and teacher04 objects, each containing all25 with n_windows,
  candidate_mean, a8s_mean, paired_delta_mean and paired_delta_se.

The test recomputes array row means, checks them against the retained JSON,
verifies ordered window and teacher-logit hashes and matching producer/source
identities, then computes mean(d) and sample SD(d)/sqrt(25), d=arm minus A8S.
Keep twelve-decimal accuracy in JSON; rounded prose is fine. The uncertainty is
across matched windows, NOT 51,175 independent positions or repeat-run error.
The approximately 42% ratio is arm-minus-A8S divided by the original positive
0.001982807772259931 regression, not a proven additive causal contribution or
an improvement of the arm over A8S. Do not invent uncertainty for a difference
between the original regression and this delta without matched covariance.

## Review boundaries

Executable tests check publication, arithmetic, citations and selected required
topics. A reviewer must independently judge attribution, chronology, custody,
assignment scope and scientific limits; keyword presence does not prove them.

- Rob's 2026-10-09T21:16:20Z correction is binding CEO-attributed evidence. In
  this research original arrays and a terminal receipt became accessible;
  distinguish their CPU reconstruction from the earlier attributed numbers.
- Both October 9 decision IDs postdate the October 5 run. The exact handoff
  records an explicit CEO-release declaration but not its primary transcript.
  Actual prior authorization time remains unverified. Do not infer it from
  the October 9 decision IDs or their digits. Audit Window4/republish only as
  historical compliance, never as a new dispatch order.
- Independent-review counters, primary decision transcripts, missing attempt
  log sidecars and a separately measured under-ten-minute metadata-only GPU
  preflight remain unestablished. Unknown is not zero/success. The retained
  invocation preflight includes decode checks; do not relabel it metadata-only.
- Preserve failed CPU actions and the exact-handoff nanosecond precision repair.
  No new experiments, weight rewrite, instrument changes or shipping-price claim.

## Research check

Run from the repository root:
`python -m pytest tests/test_issue2776_cpu_note.py -q -s --basetemp="$TMPDIR/pytest"`

Observed CPU reconstruction: teacher2 arm-minus-A8S +0.000832356215593703,
SE 0.00016023683913712466, ratio 0.419786641568897 (41.978664157%).
Teacher04: +0.0008566628756789864, SE 0.00015374870411476006.
This verifies arithmetic on recovered retained bytes, not a new measurement
or complete chain-of-custody attestation.

The invocation completed with exit code 1: four expected assertion failures
because the adopted report is absent on main. Reconstruction, fixture hashes
and matched-window checks completed before the first publication assertion.
There were no setup/import/discovery errors or skips. Full retained stdout
was read back using read_test_artifact.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"75ed65db48942e20368a1313eddf3b1d39ee21a134913a628d497f49d5ed7395","size":7353}},"operation":"5e89b3cf1c84b582559c83ea26685baff6d07296e6275dcf8d55fac7b9d76cfb","receipt":"8aa92e2cf6df4b74be78cf538eb06427334931d4604eee6f2c701877fb61861c"}]
