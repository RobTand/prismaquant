# Real served route sweep — Mixtral-tiny FP8 MoE, compressed-tensors lane

`sweep_rank0.json` is not synthetic. It is the
`prismaquant.compressed_route_sweep/1` file that
`prismaquant.validate_native_export --route-sweep-out` wrote from its own
eager load-and-generate smoke, on sparky (NVIDIA GB10, sm_121), inside the
pinned image `vllm/vllm-openai@sha256:61fc8a896b0a...` (vLLM 0.28.0,
torch 2.13.0+cu130), on 2026-10-10 (PQ #706).

- Artifact: a hand-built FP8-dynamic compressed-tensors packing of
  `hf-internal-testing/Mixtral-tiny` (2 layers, 8 experts, hidden 1024):
  every 2D weight except `model.embed_tokens` and `lm_head` quantized
  per-channel to FP8 E4M3 with an FP32 per-row scale, priced by one
  `float-quantized` config group (channel weights, token-dynamic
  activations) with the MoE router on the ignore list. `config.json` here
  is that artifact's, so the test prices against exactly what the runtime
  resolved against.
- What the serve did: 4 dense linears resolved
  `CompressedTensorsLinearMethod` → `CompressedTensorsW8A8Fp8`
  (`is_static_input_scheme=False`), and the 2 `RoutedExperts` modules
  resolved `CompressedTensorsW8A8Fp8MoEMethod` with no `scheme`, reading
  its contract off the method object's own `weight_quant`/`input_quant`
  plus `static_input_scales`. Each dispatched 4 forwards, except the
  experts rows themselves: vLLM 0.28's modular MoE runner invokes its
  experts without passing through their `__call__`, so the experts rows
  read `dispatches: 0` with `parent_dispatches: 4` from the runner.
- The 2 router gates resolve `UnquantizedLinearMethod` and sit on the
  ignore list; the 2 `Attention` modules carry
  `CompressedTensorsKVCacheMethod` and no `config_groups` entry, so they
  are unpriced rather than a refusal.

No vLLM import is needed to read it. That is the point: the attestation
travels as data (`AGENTS.md` forbids vendoring the serving runtime in a test).
