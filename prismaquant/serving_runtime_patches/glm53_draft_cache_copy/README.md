# Draft cache dtype copy candidate — PQ #1514

This source candidate targets `glm53_kpool_tail_slot_mapping`'s digest
`c2e75e03cfc52c15489b40fe58e65acb7347f6fa3ddf2e81afda86760698147b`.
It changes only the MTP/EAGLE draft dtype copy in `eagle/utils.py`, restoring
`user_specified_block_size` from the original config after the validated copy
and before `VllmConfig` revalidation. Automatic backend selection keeps its
resolved size and flag; explicit user constraints remain explicit. Backend
support tables and `CacheConfig`'s general replacement behavior are unchanged.

The script refuses any source bytes except the pinned file. CPU tests exercise
the pinned Pydantic validator field shape, derived sizes 16/64/256, explicit
constraints, unchanged target state and nested revalidation. They do not import
or vendor a serving runtime and cannot qualify a model serve.

There is no built image or serving qualification yet. This directory is not a
recorded `serving_runtime_patch_set.v1`; it intentionally has no qualifying
manifest or image reference. Root must review any image build and subsequent
runtime contract or pin change. The upstream `CacheConfig` defect remains open
until that delivery is qualified or upstream fixes it.
