# Render and packed export CUDA qualification

Two admitted CUDA checks passed on frozen source parent
`408c15b4f7af83953b3e6df6231429d651c338e3`, after the stored-candidate scoring
repair. Both exited zero and emitted their required success marker. These are
toy numerical equivalence checks, not served-artifact, throughput or memory
measurements.

| Check | PrismaBuild action | Worker | Marker |
| --- | --- | --- | --- |
| Resident versus streamed production render | `e33c2fa8b647f904ded97addccac4f80da44544e1fa168b943b9a751b4233eba` | sparklina | `STREAMING_EQUIVALENCE_OK` |
| Cached versus inline packed export | `67e58964e1967fdb8990e75e99a018c44eea1bc95c064bdf9716d7aaf39b97c0` | sparky | `INLINE_EXPERT_EXPORT_OK` |

The first calls `tests/test_streaming_production_cache.py::_equivalence_body`
on a tiny model whose layers are already resident. Its no-op layer installer
lets it compare the actual resident and streamed rendering paths, including
NVFP4, GPTQ and FP8 dense candidates. It does not qualify disk streaming or
large-model residency. The second calls
`tests/test_export_inline_expert_gptq.py::_inline_export_body` and compares
materialized packed tensor bytes for cached and inline NVFP4 GPTQ. It does not
write and serve a complete checkpoint.

Each action requested one GPU, two CPUs, 16 GiB host memory and 12 GiB GPU
memory. PrismaBuild selected eligible GB10 workers. The driver preserved each
assigned CPU set, `[5, 6]`, and bounded native threads to one. The fixed image was
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`,
with PyTorch 2.13.0+cu130 and CUDA 13.0. The temporary dependency overlay used
the unchanged Tessera API wheel and pytest wheels recorded in
[the earlier FP8 qualification](2026-10-17_architecture_fp8_cuda.md).
No active installation or runtime pin changed.

The audit directory is
`/mnt/shared/prismaquant-architecture-audit/20261017/e2effc2f3c/celestia-tmux/`.
`score-gpu-green-v1-source.json` records the frozen checkout and injected
qualification driver. The two `score-*-cuda-green-v1-evidence.json` files
verify terminal outcomes, log hashes and all 3,759 source files. Their CAS
receipts are, respectively,
`443dd7061b5d71718bcbaa64c893afdf9976c7b95b460126c8ce7fa489cc2342`
and `e9dc3fd6e319d20622ce03d5646ac643e63251bbdb79215c9d4dd4dfcbd87549`.
The matching command JSON and logs preserve the admitted executions.
