# Selected-anchor observer test readiness — 2026-09-08

PR #429's original success test ran five instantaneous fake anchors immediately
after asynchronous monitor startup. Shutdown could occur before either monitor
sampled, correctly causing the runtime to refuse incomplete telemetry despite
valid fake CUDA windows.

PB negative action `a435939746932b8c578e5f3a884e5b0011a15bb172b24ccb822cc23fbea183af`
forced both monitors to begin only after shutdown. The original success test
failed. Its actual `result.json` records five returned anchor calls,
`native_anchor_profiled: true`, and exactly these errors:

- `no netdata sample was recorded`
- `no python_sampler sample was recorded`

This deterministically reproduces the CI failure mechanism. The CI log itself
contains only the generic evidence-refusal exception, so it cannot independently
identify which instrument missed its sample in that run.

The test now uses bounded events signalled only after the real monitor has
incremented its sample count. It waits for both first samples before executing
the fake anchors, with no arbitrary sleep or fabricated sample. Both ordinary
startup and startup explicitly delayed until after observer entry are tested.
Separate deterministic cases hold Netdata, Python sampling or both until
shutdown and assert the exact missing-instrument refusal while preserving the
successful anchor return for journaling.

Production `experiments/glm_full_capture_profile.py` remains byte-identical to
base commit `571d140bc34b68c62698ac3b73894677de9b31e2`. Runtime evidence,
startup, shutdown and profiling requirements are unchanged.

PB CPU validation on `dl380g10`, Python 3.12, four pytest workers with 8 GiB
reserved and OMP/MKL/OpenBLAS/NumExpr threads bounded to one:

- Selected-anchor and full-capture observer tests: **20 passed, 1 skipped**
  (native CUDA qualification), action
  `8a43e4fed8a8e0ccd95c2f89a9e9db4af253b3b071247d4567a7df9c49440853`.
- Compile the edited test module: **exit 0**, action
  `f9e96dcfcc0418ef37a5355a9f1e7e863d361056c92efbd5d6c8347add19af80`.

Commands and evidence are under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/observer-test-readiness-01/`:
`negative-cpu-campaign.json`, `negative-observer-result.json`,
`positive-cpu-campaign.json`, action-prefix stdout/terminal/CAS JSON and
`verified-actions.json`. Terminal exit, stdout SHA, CAS payload checks and exact
executed test/runtime snapshot bytes were independently verified. Initial CAS
reads reported receipts absent; later verification succeeded, and both reads
are retained. These checks do not assert signer-attestation verification. No
GPU run or native qualification was performed.
