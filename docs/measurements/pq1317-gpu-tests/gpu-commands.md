# Exact-candidate GPU qualification commands (executed)

Candidate: Tessera `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`, contract v60.
Image X: `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`.
Target: sm_121 on NVIDIA GB10. Priority: 0.

## Preflight collection on CPU (D38)

Action `57575b6ee1afe4f3c41496acdfaf46501186e2c25126da36760e901734467a22` ran on sparky.
It used `--tag gb10` with the target image and no GPU.
It installed pinned pytest and collected 136 nodes.
It proves collection only. It gives no GPU qualification.

## MoE suite (Tessera #610)

Action `fe1ae1f2402b865d7dc84a6f86e56745f4767db4b8aee0bddeec73d964d03462` ran on sparky.
It used `--cpus 2 --demand mem_gb=8 --gpu --tag gb10` with the target image.
It used `--priority 0 --timeout-s 5400 --detach`.
Inside, Docker ran with `--gpus all` and the candidate mount.
It installed `pytest==8.4.2` and `pytest-xdist==3.8.0` plus deps.
It ran `pytest -n 2 --dist worksteal --strict-cuda` on three files.
Files: `test_native_window_moe.py`, `test_native_window_moe_method.py`, `test_serving_moe_tp2.py`.
Result: 71 passed, 0 failed, 0 skipped in 21 s.

## Dense suite (Tessera #611)

Action `370fead21f36ad81db81b7f6b536e85c09ccc97baae95f297105510dd2931fac` ran on sparklina.
It used `--cpus 2 --demand mem_gb=12 --gpu --tag gb10` with the target image.
It used `--priority 0 --timeout-s 5400 --detach`.
Inside, Docker ran with `--gpus all` and the candidate mount.
It installed the same pinned pytest set.
It ran `pytest -n 2 --dist worksteal --strict-cuda` on two files.
Files: `test_serving_bf16_gemv.py`, `test_serving_fp8_gemv.py`.
Result: 65 passed, 0 failed, 0 skipped in 120 s.

## Rules

Use the target image for all GPU runs.
Run D38 CPU preflight before each new GPU entry point.
Keep every node outcome, device fact, and receipt.
Mark skips and unexecuted nodes as incomplete. Do not count CPU skips.
