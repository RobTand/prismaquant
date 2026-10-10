# pq2571: Qwen3 estimator study (dense, small scale)

Part of prismaquant#1962. No source changes. All runs reuse
`experiments/alloc_lead_asd/a_side_diag.py` (`a_side_diag.v2.seqmajor`).

## Identity (both draws)

- Model: Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Inputs: `inputs_qwen3.safetensors`, text `fit_s42`, 4 rows x 512.
- Token prefix SHA256: `3d2fdbd12bd2e3fb...` (identical in all four legs).
- Global tokens N = 2048, scope all, temperature 1, dz storage FP32.
- Units: 196 decoder Linears, identical roster in all legs.
- Arms (8): A_all, A_attn, A_mlp, A_all_nopos0, W4_all, W4_attn, W4_mlp, W4A8_all.
- Draw 1 FP32 comes from wrapper `a2ad6761` (FP32 leg complete; wrapper exit 1 came only from the BF16-leg refusal). Draw 1 BF16 comes from re-run `cec83e22`.
- Draw 2 probes: seeds 7100..7107 (new). FP32 legs use plain backward. BF16 legs use strict deterministic backward.

## Actions and receipts

| Leg | Action | Host | Elapsed | Exit | Receipt |
| --- | --- | --- | --- | --- | --- |
| Draw 1 FP32 | `a2ad67617ebbe6679157a20ba8c2a9311188f5b179cbc4251a9afde472d9e6de` (wrapper; FP32 leg complete, BF16 leg refused) | sparky | - | 1 (FP32 output complete) | `/mnt/shared/prismabuild-fleet/pb-queue/failed/a2ad67617ebbe6679157a20ba8c2a9311188f5b179cbc4251a9afde472d9e6de.json` |
| Draw 1 BF16 | `cec83e225733784fd80235019c01fd3d6b1ec4ce2c6c7662b13c82918aeb1400` | sparklina | 61 s | 0 | `/mnt/shared/prismabuild-fleet/pb-queue/done/cec83e225733784fd80235019c01fd3d6b1ec4ce2c6c7662b13c82918aeb1400.json` |
| Smoke (this issue) | `1878f26e4603e00df936ee0a987dfabf7cadcd9e2e9417d9d55e80f46199c639` (`cpu_check_v2`, exit 0) | sparklina | 11 s | 0 | `/mnt/shared/prismabuild-fleet/pb-queue/done/1878f26e4603e00df936ee0a987dfabf7cadcd9e2e9417d9d55e80f46199c639.json` |
| Draw 2 FP32 | `f730b73f2b4a514cb26a675af2a73c2c3c7877f2801929b2c50e555ab9f603e1` | sparklina | 63 s | 0 | `/mnt/shared/prismabuild-fleet/pb-queue/done/f730b73f2b4a514cb26a675af2a73c2c3c7877f2801929b2c50e555ab9f603e1.json` |
| Draw 2 BF16 | `9e3fba132e90336518c3840ee18df6b46f05f1d7dfa6707e7518005d5402df3b` | sparky | 59 s | 0 | `/mnt/shared/prismabuild-fleet/pb-queue/done/9e3fba132e90336518c3840ee18df6b46f05f1d7dfa6707e7518005d5402df3b.json` |

Run stdout logs (under `/mnt/shared/prismabuild-fleet/pb-queue/`):

- Draw 1 FP32 (wrapper): `attempts/a2ad67617ebbe6679157a20ba8c2a9311188f5b179cbc4251a9afde472d9e6de/640621421a0f5b724b76dc24041be09c2538b0521808c2fbf37f9338c8c5546e/00000001.stdout.bb99698cd49450145652cdaf5005a8f5f9d2b3c2b22f84d10c1e144c0b8bbe8a.log` (sha `bb99698cd4945014`; records `wrote /output/pair2/float32.json (complete, 51s)` at 2026-10-02 04:38:57 UTC, then the BF16-leg v1-vs-v2 refusal with `kl_rel 9.61e-16`).
- Draw 1 BF16: `attempts/cec83e225733784fd80235019c01fd3d6b1ec4ce2c6c7662b13c82918aeb1400/6a252b2ebfe489424a7f44d06a528c74c082b38f15aeaaca7e7e5142710c13af/00000001.stdout.a1a9a782c1f160457fa006c885f39c1ef509c8ad0c392f0db17777b55efd9753.log` (sha `a1a9a782c1f16045`).
- Smoke: `attempts/1878f26e4603e00df936ee0a987dfabf7cadcd9e2e9417d9d55e80f46199c639/179e98d53b9e3be91f1e66a6cf29576473229dd992538eb3fc7146ae0f77c107/00000001.stdout.b55dad9fcec6f8e2d8127a8d48aa66fb9d64a9b431f5cb9365ddeb18a9b4e1d0.log` (sha `b55dad9fcec6f8e2`).
- Draw 2 FP32: `attempts/f730b73f2b4a514cb26a675af2a73c2c3c7877f2801929b2c50e555ab9f603e1/2fd352b0454948e2f0ced63ed224d36a9e6c868cab0ec4058ae1e03dd7c73fd0/00000001.stdout.c1ca890b02409ebf885a8974a344048700b9f2b84ad4302a375f684941ff7d63.log` (sha `c1ca890b02409ebf`).
- Draw 2 BF16: `attempts/9e3fba132e90336518c3840ee18df6b46f05f1d7dfa6707e7518005d5402df3b/84f7b2ed302491ebed92edda728fdbf88d13d6d4fcf041bf512f4cd6c55c57dd/00000001.stdout.a60f1e0cfac06a6ba0bbb3dd6af9710e146b1d1b47ca3776aecb00dfe642c490.log` (sha `a60f1e0cfac06a6b`).

Result tables:

- Draw 1: `/mnt/shared/tessera-measurements/pq1962-sol-20261002/pair2/float32.json`,
  `/mnt/shared/tessera-measurements/pq1962-sol-20261002/bf16-complete3/bfloat16.json`.
- Draw 2: `/mnt/shared/tessera-measurements/pq1962-sol-20261002/pq2571-draw7100-float32.json`,
  `/mnt/shared/tessera-measurements/pq1962-sol-20261002/pq2571-draw7100-bfloat16.json`.
- Per-action stdout logs live under
  `/mnt/shared/prismabuild-fleet/pb-queue/attempts/<key>/.../00000001.stdout.*.log`.

Draw 2 crosschecks (v1 lease vs v2, profile on):

- FP32: component max/RMS 2.1e-6 (A8), 2.1e-6 (A8N), 2.7e-6 (W4); KL rel 1.8e-15.
- BF16: component max/RMS 6.8e-7 (A8), 6.6e-7 (A8N), 4.0e-6 (W4); KL rel 9.6e-16.

## Draw 1 FP32 (seeds 7000..7007)

| Arm | P_add | P_joint | S_real | Q_real | KL_true | corr |
| --- | --- | --- | --- | --- | --- | --- |
| A_all | 0.0112951 | 0.00937987 | 0.00929878 | 0.0104181 | 0.0104801 | 0.4343 |
| A_attn | 0.00446079 | 0.000594855 | 0.00284281 | 0.00428958 | 0.00430094 | 0.4549 |
| A_mlp | 0.00683429 | 0.00793902 | 0.00307546 | 0.00660935 | 0.00661073 | 0.1968 |
| A_all_nopos0 | 0.0110306 | 0.0107717 | 0.00471721 | 0.0110625 | 0.0109584 | 0.2506 |
| W4_all | 0.21011 | 0.199233 | 0.155079 | 0.207687 | 0.207851 | 0.6397 |
| W4_attn | 0.0813069 | 0.0643122 | 0.0891009 | 0.0862923 | 0.0875216 | 0.9407 |
| W4_mlp | 0.128803 | 0.0978418 | 0.068323 | 0.117991 | 0.115954 | 0.7820 |
| W4A8_all | 0.221405 | 0.26616 | 0.205663 | 0.217351 | 0.219123 | 0.7000 |

## Draw 2 FP32 (seeds 7100..7107)

| Arm | P_add | P_joint | S_real | Q_real | KL_true | corr |
| --- | --- | --- | --- | --- | --- | --- |
| A_all | 0.0126102 | 0.0148404 | 0.00904271 | 0.0104181 | 0.0104801 | 0.6199 |
| A_attn | 0.00558466 | 0.0101575 | 0.0105891 | 0.00428958 | 0.00430094 | 0.6755 |
| A_mlp | 0.00702556 | 0.00491202 | 0.00533987 | 0.00660935 | 0.00661073 | 0.1742 |
| A_all_nopos0 | 0.012221 | 0.0142454 | 0.010432 | 0.0110625 | 0.0109584 | 0.4742 |
| W4_all | 0.246787 | 0.130486 | 0.126698 | 0.207687 | 0.207851 | 0.2557 |
| W4_attn | 0.103081 | 0.0528858 | 0.0587356 | 0.0862923 | 0.0875216 | 0.8219 |
| W4_mlp | 0.143706 | 0.0674754 | 0.0537528 | 0.117991 | 0.115954 | 0.2865 |
| W4A8_all | 0.259398 | 0.0707191 | 0.117244 | 0.217351 | 0.219123 | -0.0791 |

## Draw 1 BF16 (seeds 7000..7007)

| Arm | P_add | P_joint | S_real | Q_real | KL_true | corr |
| --- | --- | --- | --- | --- | --- | --- |
| A_all | 0.0107347 | 0.00522116 | 0.0107088 | 0.0118927 | 0.0116637 | 0.5784 |
| A_attn | 0.00402396 | 0.00239704 | 0.00308781 | 0.00600774 | 0.00594784 | 0.3142 |
| A_mlp | 0.00671078 | 0.00551209 | 0.00559511 | 0.00799571 | 0.00799744 | 0.5170 |
| A_all_nopos0 | 0.0104136 | 0.00595433 | 0.0166237 | 0.0117904 | 0.0118199 | 0.8568 |
| W4_all | 0.209405 | 0.193686 | 0.157177 | 0.210437 | 0.209612 | 0.6083 |
| W4_attn | 0.0806637 | 0.0671909 | 0.0932459 | 0.0878294 | 0.0889777 | 0.9456 |
| W4_mlp | 0.128741 | 0.0990631 | 0.0759695 | 0.120423 | 0.117934 | 0.8077 |
| W4A8_all | 0.22014 | 0.235651 | 0.186329 | 0.219403 | 0.220539 | 0.7762 |

## Draw 2 BF16 (seeds 7100..7107)

| Arm | P_add | P_joint | S_real | Q_real | KL_true | corr |
| --- | --- | --- | --- | --- | --- | --- |
| A_all | 0.011377 | 0.0108905 | 0.00305767 | 0.0118927 | 0.0116637 | -0.0678 |
| A_attn | 0.00509652 | 0.00758408 | 0.00675896 | 0.00600774 | 0.00594784 | 0.7916 |
| A_mlp | 0.00628048 | 0.00725707 | 0.0158874 | 0.00799571 | 0.00799744 | -0.3179 |
| A_all_nopos0 | 0.0111489 | 0.0107 | 0.0096086 | 0.0117904 | 0.0118199 | -0.1522 |
| W4_all | 0.243918 | 0.132715 | 0.134411 | 0.210437 | 0.209612 | 0.3407 |
| W4_attn | 0.101886 | 0.0532323 | 0.0651629 | 0.0878294 | 0.0889777 | 0.8434 |
| W4_mlp | 0.142031 | 0.0673033 | 0.0609902 | 0.120423 | 0.117934 | 0.3649 |
| W4A8_all | 0.255295 | 0.09598 | 0.131272 | 0.219403 | 0.220539 | 0.5163 |

## Read

- Q_real and KL_true are probe independent, so they match across draws by design. Both draws share the token scope above, so the match is exact.
- P_add, P_joint, S_real and corr move with the probe draw. All four result files record `complete: true`.
- Corr is unstable across draws at 8 probes. Sign flips: A_all BF16 0.58 then -0.07, A_mlp BF16 0.52 then -0.32, W4A8_all FP32 0.70 then -0.08. Stable only at W4_attn (FP32 0.94 then 0.82; BF16 0.95 then 0.84).
- Do not read corr at 8 probes as estimator bias. The parent item owns the bias-vs-MoE separation. This study only supplies the numbers.
