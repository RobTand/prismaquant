# Real served route sweep — Qwen3-0.6B NVFP4, compressed-tensors lane

`sweep_rank0.json` is not synthetic. It is the
`prismaquant.compressed_route_sweep/1` file that
`prismaquant.validate_native_export --route-sweep-out` wrote from its own
eager load-and-generate smoke, on sparklina (NVIDIA GB10, sm_121), inside the
pinned dense image `vllm/vllm-openai@sha256:61fc8a896b0a...` (vLLM 0.28.0), on
2026-09-17.

- Artifact: `/home/rob/dq-runs/fc45-0p6b-nvfp4/exported` (Qwen3-0.6B, one
  NVFP4 config group, 252 targets, `lm_head` and `model.embed_tokens`
  ignored). `config.json` here is that artifact's, byte for byte, so the test
  prices against exactly what the runtime resolved against.
- What the serve did: 112 modules (28 × `qkv_proj`, `gate_up_proj`, `o_proj`,
  `down_proj`) resolved `CompressedTensorsLinearMethod` →
  `CompressedTensorsW4A4Fp4(use_a16=False, group_size=16)` on
  `FlashInferCutlassNvFp4LinearKernel`, and each dispatched 16 forwards. The
  28 `Attention` modules carry `CompressedTensorsKVCacheMethod` and no
  `config_groups` entry, so they are unpriced rather than a refusal.
- The 140 unfused leaf targets (`q_proj`, `k_proj`, `v_proj`, `gate_proj`,
  `up_proj`) are covered through the `packed_modules_mapping` the sweep read
  off the live `Qwen3ForCausalLM` class, not through a roster in this repo.

No vLLM import is needed to read it. That is the point: the attestation
travels as data (`AGENTS.md` forbids vendoring the serving runtime in a test).
