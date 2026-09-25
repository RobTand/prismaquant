# Serving-runtime patch sets

A patch set is what the repository records when the **pinned** serving image does
not serve a model and a **derived** image does. It exists because an image tag is
not a pin: a tag on one box is unreadable by a gate, a reviewer, or a future
session, and the edits it carries have no home.

Each subdirectory holds:

* `MANIFEST.json` — schema `prismaquant.serving_runtime_patch_set.v1`: the base
  image by digest, the derived image (by registry digest when one exists, and
  otherwise by local docker image id, in a separate field, because a config
  digest cannot be pulled), every edit by tag, and the qualification scope.
* The patch scripts themselves. Each edit asserts its target occurs exactly once,
  so the build fails loudly when the base image moves instead of quietly serving
  an unpatched backend.
* A `Dockerfile` that builds `FROM` the base digest and re-imports the patched
  module to assert every edit landed.

Read one with `prismaquant.serving_runtime_patch_set.load_serving_runtime_patch_set`.

Two rules the reader enforces:

* **`attested` must be `false`.** Principle 14 says a claim about what a serving
  runtime *does* is derived from a machine-readable table that runtime publishes,
  or refused. A patch set is the opposite shape — a producer-side statement about
  a runtime this repository modified — so nothing here hands a caller a route, an
  activation contract, or an eligibility answer.
* **Qualification carries its own scope.** `require_qualified_for` refuses a body
  wider than the one the patch set was measured on, so "READY" cannot quietly
  become "serves".

## Sets

| Set | Base | Qualified on | Status |
|---|---|---|---|
| `glm53_nope_sm120` | `eugr/spark-vllm@sha256:0afec8d4…` (vLLM 0.28.1rc1.dev397, flashinfer 0.6.18) | GLM-5.3-Flash **4-layer** stub, BF16, TP1, eager + CUDA graph, sparklina 2026-09-13 | RECORDED. The 45-layer body is untested; eager and graph disagree by up to 0.67 nats. |
| `glm53_mtp_mapper` | `192.168.1.107/prismaquant/spark-vllm-nccl230@sha256:a5424378…` (the image the routed Tessera cells name; stock MLA sources) | Nothing yet: build self-check only (the mapper sends `model.language_model.layers.45.*` to `model.layers.45.*`; the four MLA sources hash stock), sparky + sparklina 2026-09-25 | RECORDED. Derived image `…@sha256:f8dbe1a0…` is in the LAN registry. No model served on it yet; the drafter claiming a quantized layer 45 is unexercised (#1271). |
