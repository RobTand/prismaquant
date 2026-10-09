# Exact-candidate GPU qualification commands (prepared, not executed)

Candidate: Tessera `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`, contract v60.
Image X: `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`.
Target: sm_121. Priority: 0.

## MoE suite (Tessera #610)

Run from the Tessera candidate checkout on a GB10 worker:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --gpu --tag gb10 --cpus 2 --demand mem_gb=8 \
  --container-image 'localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5' \
  --priority 0 --timeout-s 5400 -- \
  python3 -m pytest -n 2 --dist worksteal --strict-cuda \
  tests/test_native_window_moe.py \
  tests/test_native_window_moe_method.py \
  tests/test_serving_moe_tp2.py
```

## Dense suite (Tessera #611)

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --gpu --tag gb10 --cpus 2 --demand mem_gb=12 \
  --container-image 'localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5' \
  --priority 0 --timeout-s 5400 -- \
  python3 -m pytest -n 2 --dist worksteal --strict-cuda \
  tests/test_serving_bf16_gemv.py \
  tests/test_serving_fp8_gemv.py
```

## Rules

- Submit only after CEO decision `dec-1009-211441-d65c` authorizes GPU work.
- Run D38 CPU preflight first for each entry point.
- Keep every node outcome, oracle value, device fact, and receipt.
- Mark skips and unexecuted nodes as incomplete. Do not count CPU skips.
