# DFlash2 drafter candidate — PQ #1617

This source candidate targets `glm53_kpool_tail_slot_mapping`'s digest
`c2e75e03cfc52c15489b40fe58e65acb7347f6fa3ddf2e81afda86760698147b`.
It carries two edits:

- `patch_glm5next_eagle3.py` (7 anchored edits on
  `models/glm5next/nvidia/model.py`): the GLM5-next target implements
  `SupportsEagle3` on both wrappers and returns auxiliary hidden states
  for the layers the DFlash2 draft config names. Without it the V2
  runner raises `Model does not support EAGLE3 interface` at load, and
  then asserts a tuple return the model never produces.
- `patch_glm5_drafter_kv_group.py` (11 anchored edits on
  `v1/core/kv_cache_utils.py`): exact-type `SlidingWindowSpec` drafter
  layers leave the GLM5-next fast-path guard and form one standalone
  native-page group appended last. Without it the model falls back to
  the generic page path, whose page unification rescales the kpool
  tail's block away from its pool size.

Base bytes are verified, not assumed. Both target files in the base
image equal upstream vLLM commit
`fd4a1512628ad17944095263c1fe89598710a3ce` byte for byte
(`model_runner.py` reproduces the recorded kpool-tail sha `1c30b8c0…`
after the recorded kpool edits; `mtp.py` reproduces the recorded mapper
base sha `715768cd…`). Each patch script refuses any other base bytes,
asserts each edit lands exactly its expected count, and exposes a pure
`patched_source` that CPU tests exercise without a serving runtime.
Patched file hashes (reproduced locally from the verified base bytes):

- `models/glm5next/nvidia/model.py`:
  `f6a27dfd2306056f51335eed22cc4ffc666e35bad0aa9a457d27c33c6490e444`
- `v1/core/kv_cache_utils.py`:
  `792d068189e3fd4703ae823d1bd223fba1f6fb220615f7ffb2b89e053a6f4e48`

The base image already carries the DFlash2 model
(`model_executor/models/qwen3_dflash2.py`) and the DFlash2 speculator
dispatch (`spec_decode/__init__.py`); this set changes neither and the
self-check pins their stock hashes.

There is no built image or serving qualification yet. This directory is
not a recorded `serving_runtime_patch_set.v1`; it intentionally has no
qualifying manifest or image reference, like `glm53_draft_cache_copy`.
The EXL3 reference overlays informed the design but are unrun here and
target a different image. Open before a build:

- Build the Dockerfile on a Spark against the base digest and record
  the derived digest (needs docker + the LAN registry).
- Serve DFlash2 at k = 7 eager on the 45-layer body at TP2 and record
  acceptance (loads, generates, acceptance rate).
- Tessera admits a CUDA-graph mode for dflash only by receipt; until
  then the Tessera gate refuses dflash by name (Tessera repo work).
