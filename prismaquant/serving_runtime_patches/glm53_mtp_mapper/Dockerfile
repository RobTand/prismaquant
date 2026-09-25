# GLM-5.3-Flash MTP drafter quantization-target mapping on the GLM serving image (GB10 / sm121).
#
# FROM the registry digest of the image the routed Tessera cells already name:
#   192.168.1.107/prismaquant/spark-vllm-nccl230@sha256:a5424378...
#   (= localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378..., which is
#    eugr/spark-vllm@sha256:0afec8d4... with libnccl 2.31.2 replaced by 2.30.7;
#    vLLM 0.28.1rc1.dev397+gfd4a15126.d20260904, flashinfer 0.6.18)
#
# One Python source edit -- no compile, no new wheels, no vendored runtime. The
# MLA sparse-attention sources stay stock, so a runtime that requires them
# unchanged (Tessera's GLM53 NoPE backend) still loads.
ARG BASE=192.168.1.107/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb
FROM ${BASE}

COPY patch_glm5next_mtp_mapper.py /opt/glm53/patch_glm5next_mtp_mapper.py
RUN python3 /opt/glm53/patch_glm5next_mtp_mapper.py

RUN python3 - <<'PY'
import hashlib
from pathlib import Path

from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP

mapper = Glm5NextMTP.hf_to_vllm_mapper.get_rename_mapper()
assert mapper.apply_list(["model.language_model.layers.45.mlp.experts"]) == [
    "model.layers.45.mlp.experts"], mapper.apply_list(["model.language_model.layers.45.mlp.experts"])
assert mapper.apply_list(["model.layers.45.mlp.experts"]) == ["model.layers.45.mlp.experts"]

site = Path("/usr/local/lib/python3.12/dist-packages/vllm")
stock = {
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py": "a0023f72125cb0d5599b5bf940c86be1f0c9985bd62b0919f243a8fda76f4449",
    "v1/attention/backends/mla/flashinfer_mla_sparse.py": "093181e4e0198b34075a3713fa62267c3f1fe74b1c96bb08298d553727d44478",
    "v1/attention/backends/mla/sparse_utils.py": "20372237899fb0a0c9152e12eab40396119b76c1ac5f1c9bdfdbed53f8146559",
    "v1/attention/backend.py": "8ddf8dd73c2b953a79f99e8de74db616b135590ac1f74e21e20b9ddc366b9994",
}
for relative, expected in stock.items():
    got = hashlib.sha256((site / relative).read_bytes()).hexdigest()
    assert got == expected, (relative, got)
print("glm53 MTP mapper patch verify OK; MLA sources stock")
PY
