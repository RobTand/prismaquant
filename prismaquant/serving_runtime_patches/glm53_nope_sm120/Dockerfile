# GLM-5.3-Flash NoPE sparse MLA on the pinned PrismaQuant serve image (GB10 / sm121).
#
# FROM the pinned digest so the derivation stays attributable:
#   eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c
#   (vLLM 0.28.1rc1.dev397+gfd4a15126, flashinfer 0.6.18)
#
# Python source patches only -- no compile, no new wheels, no vendored runtime.
ARG BASE=eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c
FROM ${BASE}

COPY patch_glm53_nope_sm120.py /opt/glm53/patch_glm53_nope_sm120.py
COPY patch_gb10_hardening.py /opt/glm53/patch_gb10_hardening.py
RUN python3 /opt/glm53/patch_glm53_nope_sm120.py \
 && python3 /opt/glm53/patch_gb10_hardening.py

RUN python3 - <<'PY'
import inspect
from pathlib import Path

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl as impl,
)

assert "do_kv_cache_update" in impl.__dict__
init_src = inspect.getsource(impl.__init__)
assert "self.is_nope_mla = self.qk_rope_head_dim == 0" in init_src
fwd = inspect.getsource(impl.forward_mqa)
for t in (
    "return_valid_counts=True",
    "sparse_mla_top_k_lens",
    "sparse_mla_top_k=sparse_topk_capacity",
    "max_seq_len=sparse_topk_capacity",
    "seq_lens=seq_lens_arg",
    "out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)",
):
    assert t in fwd, t
kv = inspect.getsource(impl.do_kv_cache_update)
assert "kv_c_normed.new_zeros((kv_c_normed.shape[0], 1, 64))" in kv

site = Path("/usr/local/lib/python3.12/dist-packages/vllm")
warm = (site / "model_executor/warmup/kernel_warmup.py").read_text()
compile(warm, "kernel_warmup.py", "exec")
assert "flashinfer_sparse_mla_decode_autotune_warmup(worker)" not in warm
assert "Skipping FlashInfer autotune on SM121" in warm
cuda = (site / "platforms/cuda.py").read_text()
compile(cuda, "cuda.py", "exec")
assert "return major in (9, 10)" in cuda
print("glm53 NoPE sparse-MLA patch verify OK")
PY
