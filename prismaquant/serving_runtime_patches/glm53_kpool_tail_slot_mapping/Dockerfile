# GLM-5.3 kpool-tail slot mapping on the GLM serving image (GB10 / sm121).
#
# FROM the registry digest of the glm53_mtp_mapper derived image, the image the
# GLM Tessera cells name:
#   192.168.1.107/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0...
#   (= localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0...;
#    vLLM 0.28.1rc1.dev397+gfd4a15126.d20260904, flashinfer 0.6.18)
#
# One Python source file edited -- no compile, no new wheels, no vendored runtime.
# Backports vLLM #57317 (70df48dc3d01): the kpool-tail KV group is kept out of
# the V2 runner's generic slot mapping, whose kernel otherwise reads the tail's
# 32-entry block-table row by absolute position (tessera#508). The MLA
# sparse-attention sources stay stock, so Tessera's GLM53 NoPE backend, which
# requires them unchanged, still loads.
ARG BASE=192.168.1.107/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5
FROM ${BASE}

# BuildKit resolves a digest FROM against the registry over HTTPS, and the LAN
# registry is plain HTTP, so this image is built with the classic builder
# (DOCKER_BUILDKIT=0) from the local copy of the base digest. The classic
# builder drops a RUN heredoc's body, so the self-check is a copied file.
COPY patch_kpool_tail_slot_mapping.py /opt/glm53/patch_kpool_tail_slot_mapping.py
COPY selfcheck_kpool_tail_slot_mapping.py /opt/glm53/selfcheck_kpool_tail_slot_mapping.py
RUN python3 /opt/glm53/patch_kpool_tail_slot_mapping.py \
 && python3 /opt/glm53/selfcheck_kpool_tail_slot_mapping.py
