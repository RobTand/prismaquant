# Tessera calibration forward ownership qualification

Measured 2026-10-15 for source unit `408fce199d`. The public
`tessera_calibration_forward.capture_activation_statistics` keeps the original
capture and cleanup body. `CalibrationStatistics` names its four tuple fields.
Caller-owned model, calibration, callbacks and existing stores retain their
lifetimes. Current package source seals include the new owner; historical
operation spellings, schemas and explicit source-migration proofs remain.

## Fixed inputs and runtime

The real Qwen checkpoint and WikiText draw were already retained under
`/mnt/shared/pq-refactor-20261014-native`. Both arms use the same model instance
in evaluation mode, seven first-layer Linear projections and four `(1, 256)`
calibration windows. The scoring prefix cap is 32; each Hessian, count and
maximum covers all 1,024 observed rows.

| Input | SHA-256 |
| --- | --- |
| `model/model.safetensors` | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| `model/config.json` | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| `calibration.pt` | `80213b07bf7e18a231e7e214daa66e9722ccc1d69e034ec55088b12bc4480b05` |
| Calibration int64 tensor, shape `[4, 256]` | `070fd77023a026933122b17e1142ed4027bb0d57da887f9f7c82948498519ce5` |
| PB `baseline.inputs.json` | `b98497c60eaebdfd396e73ad8e5a4d04654d456743d073348dcc2c38b3745bbc` |

The baseline is the actual campaign source at
`ff8d6a14dba8d38e7914acf5d3ecb75eef49040e`, file SHA
`c7fbad6feb01875e55ade75f9c10f34922bd3826bf9aaacef0a28bae3f38a1d2`.
The candidate uses the new public owner. The sealed verification dependency is
producer commit `d2e98b3cb2fcb6fdc27a552442ea021ee02da8ff`, wheel SHA
`30748bc3df5380475fe6dfbf3b49aa95d4b73467fe7ed8680938fe940382aa39`.
The producer package and serving gates are unchanged by this source move.

The existing PB container launcher used image ID
`sha256:44426342348a2fb1ae1191e09e5d4e19ab1231ab6dbdf0e17e8192bfdbeff025`,
content `ac918444aa2d16d1d19771510963d138981fdd468db44a2bf10f9d627b3af12a`.
The eligible worker ran Torch `2.13.0+cu130`, CUDA `13.0`, on NVIDIA GB10.
PB reserved two CPUs, one GPU and its 16 GiB admitted host-memory floor.
Native threads were one; assigned CPU affinity `[5, 6]` was retained.

## Actual results

GPU action
`e9abb06c564835197f451100d1eef791b9673e69ad79bf9bb8466f556a6fd322`
exited zero. Both arms have bit-identical scoring rows, full Hessians, counts
and maxima for all seven projections. Both restore the original hook counts.
Before and after profiles each contain 7,576 actual CUDA kernel events.
Netdata records five actual samples per host on both Sparky and Sparklina,
including power charts. No speed or energy improvement is claimed.

Evidence directory:
`/mnt/shared/pq-refactor-20261014-native/calibration-owner-comparison-batched-20261015`.
`comparison.json` names every trace, table and telemetry artifact with its size
and hash. Every named artifact was independently read and checked.

| Successful CAS artifact | SHA-256 |
| --- | --- |
| Result payload | `827b862997358dcfd10d4006b16fc460afaf41160fb9c796a9b514bb4e83fc7b` |
| Published receipt JSON file | `69cadd1facb4428b856da075b2c75c26c282a2b85d86c28e2db23477165f14cb` |
| Receipt's `receipt_sha256` field | `1e9f75641822936d72e18fce4cb71bb151ef88dcf548761d7c5ad2ea80d3fd6d` |

The rerunnable tool is `tools/verify_tessera_calibration_forward.py`. It takes
explicit baseline source and digest, input root and new output directory.
Run it only through PB and the known-image container launcher, with the sealed
baseline and exact producer dependency in the checkout and the declared data
manifest. It refuses missing CUDA, changed baseline bytes, a different
calibration shape, numerical disagreement, hook leaks or incomplete telemetry.

## Failures, CPU checks and limits

First GPU action
`8da25a10f7410e19381529b34f1e96dd7e4169699fa84d33b05145fed5ef3156`
failed inside the original baseline model forward because the harness iterated
the cached matrix into one-dimensional token IDs. The retry uses the same bytes
split into the campaign's `(1, 256)` windows. The failed action and logs remain;
it is not counted as parity evidence.

CPU baseline passed 83 tests and two subtests. Candidate action
`fea3747df87b38b9724a38b688870a2599b94b763983e25b167675e0b4d8ddc5`
passed 187 tests and two subtests, with one explicit CUDA-profiler skip.
The isolated capture-only action
`0270b0e2f1c03d2a03c9764eb86b71067230284c63868867ea67e635e010bae6`
passed 153 tests with zero skips, compiled 43 paths and passed the full
duplication and architecture/doc guards. Its result payload SHA is
`618195b92827450fc731bca77027725643b2a587aa815b1d2d90e3d98ff0020a`;
published receipt JSON SHA is
`0b21ab73e5126838ff91d4b1543cf6cd525c8a89a783ad001d76964e78422777`.

CPU cases retain packed-input grouping, full statistics versus bounded rows,
resource refusals, source release and exception cleanup. Live CUDA coverage is
limited to the dense Qwen capture above. This measurement does not qualify
encoded wire output, served KL or additional engine cells.
