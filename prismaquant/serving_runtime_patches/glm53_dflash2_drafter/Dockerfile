# GLM-5.3 DFlash2 drafter on the GLM serving image (GB10 / sm121).
#
# FROM the registry digest of the glm53_kpool_tail_slot_mapping derived image,
# the image the GLM Tessera cells name:
#   192.168.1.107/prismaquant/spark-vllm-nccl230@sha256:c2e75e03...
#   (vLLM 0.28.1rc1.dev397+gfd4a15126.d20260904, flashinfer 0.6.18)
#
# Two Python source files edited -- no compile, no new wheels, no vendored
# runtime:
# - models/glm5next/nvidia/model.py: the target implements SupportsEagle3
#   and returns aux hidden states the DFlash2 drafter names.
# - v1/core/kv_cache_utils.py: the drafter's sliding-window layers leave the
#   GLM5-next fast-path guard and form one standalone group, so the model no
#   longer falls back to the generic page path that rescales the kpool tail.
# The MLA sparse-attention sources stay stock, so Tessera's GLM53 NoPE
# backend, which requires them unchanged, still loads.
ARG BASE=192.168.1.107/prismaquant/spark-vllm-nccl230@sha256:c2e75e03cfc52c15489b40fe58e65acb7347f6fa3ddf2e81afda86760698147b
FROM ${BASE}

# BuildKit resolves a digest FROM against the registry over HTTPS, and the LAN
# registry is plain HTTP, so this image is built with the classic builder
# (DOCKER_BUILDKIT=0) from the local copy of the base digest. The classic
# builder drops a RUN heredoc's body, so the self-check is a copied file.
COPY patch_glm5next_eagle3.py /opt/glm53/patch_glm5next_eagle3.py
COPY patch_glm5_drafter_kv_group.py /opt/glm53/patch_glm5_drafter_kv_group.py
COPY selfcheck_glm53_dflash2_drafter.py /opt/glm53/selfcheck_glm53_dflash2_drafter.py
RUN python3 /opt/glm53/patch_glm5next_eagle3.py \
 && python3 /opt/glm53/patch_glm5_drafter_kv_group.py \
 && python3 /opt/glm53/selfcheck_glm53_dflash2_drafter.py
