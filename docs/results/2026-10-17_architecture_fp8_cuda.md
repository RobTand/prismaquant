# FP8 batched reconstruction CUDA qualification

Recorded 2026-10-17 for architecture finding RE-08. Four CUDA cases passed,
with four CPU cases deselected and no skips. The cases compare batched and
individual reconstruction at FP8 midpoint boundaries for FP8_E4M3 and
FP8_E5M2 with FP32 and BF16 source weights. This establishes codec parity on
these inputs; it is not a served-artifact, throughput or memory measurement.

The frozen source parent was `263c7e41ad3be8f17180734dad5a26cda647e735`.
The source snapshot also contains the recorded GPU qualification driver.
PrismaBuild admitted action
`28b749a2b9fc6639b1a1bfb0f5f9b370b1d893e06c9ad19338c4bf486441280e`
with one GPU, two CPUs, 16 GiB host memory and a 12 GiB GPU reservation.
Portable placement selected sparklina, an NVIDIA GB10. The driver preserved
the assigned CPU affinity and bounded native threads to one.

The container was
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`,
with PyTorch 2.13.0+cu130, CUDA 13.0 and Python 3.12.3. It used the existing
Tessera API wheel for commit `aaa417003f0ee690e8ace138acd18df7f9531369`,
whose wheel SHA-256 is
`5823979dbba0404b5e926dccacab9a2acd4ab15f32ace7db79022f541c32fae5`.
The packaged contract SHA-256 remained
`ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e`.

The test selection inside the admitted container was:

```text
pytest tests/test_fp8_batched_midpoint_parity_audit.py::test_plain_fp8_batch_preserves_each_slice_and_row_scale -k cuda
```

The first GPU action stopped before tests because the image lacked pytest.
A separate admitted download fetched pytest 8.4.2 and its pure-Python
dependencies. The retry verified each wheel's SHA-256 and extracted them to
a temporary overlay. It changed no active installation, image or runtime pin.

Commands, source manifests, logs, dependency hashes and verified receipts are
under `/mnt/shared/prismaquant-architecture-audit/20261017/e2effc2f3c/celestia-tmux/`:

- `fp8-cuda-green-v2-command.json`, `fp8-cuda-green-v2-source.json` and
  `fp8-cuda-green-v2.log` record the successful action.
- `fp8-cuda-green-v2-evidence.json` verifies the terminal result, log hashes,
  all 3,749 source files and CAS receipt
  `316e21d9fe2ab062f5ca1a40bd4c0aa1c76b19bf57feee4fe76847f855dadab3`.
- `gpu-test-tools-manifest.json` records the exact dependency wheels.
- `fp8-cuda-green-v1-evidence.json` preserves the unsuccessful first action;
  `gpu-test-tools-download-v1-evidence.json` records the admitted download.
