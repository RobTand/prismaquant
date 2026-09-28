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
| `glm53_mtp_mapper` | `192.168.1.107/prismaquant/spark-vllm-nccl230@sha256:a5424378…` (the image the routed Tessera cells name; stock MLA sources) | The GLM-5.3-Flash BAL Tessera export, 45 layers + the MTP layer, TP2, eager: run `u4-BAL-20260928T0540Z-2c-r6-2c` (Tessera `f18f08b5`, 2026-09-28), census verdict `served` with a VALID receipt (98 checks, 0 failed); the drafter's quantized layer-45 experts trace as `model.layers.45.mlp.experts` (PQ #1490). Build self-check also passed, sparky + sparklina 2026-09-25 | RECORDED. Derived image `…@sha256:f8dbe1a0…` is in the LAN registry. Eager only; CUDA-graph mode and drafter fidelity against BF16 are not measured. |
| `glm53_kpool_tail_slot_mapping` | `192.168.1.107/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0…` (the `glm53_mtp_mapper` derived image) | GLM-5.3-Flash **4-layer** stub, Tessera A4, TP1, sparky 2026-09-28: eager, and CUDA graphs FULL_DECODE_ONLY, PIECEWISE and FULL_AND_PIECEWISE with every capture size 1 to 8. Equality-suite completions are eager outcomes; 19 two-chunk 3649-token prefills ran with no illegal memory access; memcheck on one eager serve each: the base image 128 invalid reads and a dead engine, this image 0 (tessera#508, tessera PR #699) | RECORDED. Derived image `…@sha256:c2e75e03…` is in the LAN registry. Backports vLLM #57317: the kpool-tail KV group leaves the V2 runner's generic slot mapping, whose kernel read that group's 32-entry block-table row by absolute position and faulted in CUDA-graph serves (tessera#508). The 45-layer body, TP2 and the MTP drafter are not measured on it, and a padded capture list runs another reduction order. |
