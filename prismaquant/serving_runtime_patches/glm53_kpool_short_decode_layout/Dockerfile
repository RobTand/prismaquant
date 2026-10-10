# GLM-5.3 kpool short-decode layout on the kpool-tail image (GB10 / sm121).
#
# FROM the registry digest of the glm53_kpool_tail_slot_mapping derived image,
# the image the GLM Tessera cells name:
#   192.168.1.107/prismaquant/spark-vllm-nccl230@sha256:c2e75e03...
#   (= localhost/prismaquant/spark-vllm-nccl230@sha256:c2e75e03...;
#    vLLM 0.28.1rc1.dev397+gfd4a15126.d20260904, flashinfer 0.6.18)
#
# One Python source file edited -- no compile, no new wheels, no vendored runtime.
# The short-decode fast path writes the general-path sparse-index row, and the
# general path sorts each short row's pool ids before expansion (PQ #1631).
# The expand kernel itself stays stock.
ARG BASE=192.168.1.107/prismaquant/spark-vllm-nccl230@sha256:c2e75e03cfc52c15489b40fe58e65acb7347f6fa3ddf2e81afda86760698147b
FROM ${BASE}

# BuildKit resolves a digest FROM against the registry over HTTPS, and the LAN
# registry is plain HTTP, so this image is built with the classic builder
# (DOCKER_BUILDKIT=0) from the local copy of the base digest. The classic
# builder drops a RUN heredoc's body, so the self-check is a copied file.
COPY patch_kpool_short_decode_layout.py /opt/glm53/patch_kpool_short_decode_layout.py
COPY selfcheck_kpool_short_decode_layout.py /opt/glm53/selfcheck_kpool_short_decode_layout.py
RUN python3 /opt/glm53/patch_kpool_short_decode_layout.py \
 && python3 /opt/glm53/selfcheck_kpool_short_decode_layout.py
