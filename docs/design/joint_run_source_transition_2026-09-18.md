# Run source transition for the sealed GLM-5.3-Flash prepare — 2026-09-18

## What failed

PrismaBuild action `4bafaaeb6a0e3d50873506885be9fdc3809ebb996f0d1655b8c898a1171cccfd`
(the joint `run` stage of the GLM-5.3-Flash campaign, sparky, stage-fed) walked
its intake for about 50 minutes and then raised

    RuntimeError: prepared render tensor proof differs for
    model.language_model.layers.0.mlp.down_proj@TESSERA_BF16_K1_R1024

from the `prepared_render_identities` block of `compute_aura_cost_streamed`
(`prismaquant/aura_cost.py`). No checkpoint was written. The prepare stage had
verified all 197,990 cells against the installed bf16 tensors
(`verify_anchor_render`), and the prepared record binds each rendered weight as
`{shape, dtype, logical_bytes, content_sha256}`.

## Why

The block compared every pair, up front, against `linears[name].weight` of the
live model. On the streamed cost path every decoder-layer parameter is a meta
tensor until `_fast_install` replaces it, and the meta skeleton comes from
`build_streaming_skeleton`, which calls `model_cls._from_config(config)` with no
dtype. transformers `_from_config` (5.16.1 in the campaign image, 5.6.0 locally)
takes `config.dtype`, overwrites the sub-config dtypes with it, and enters a
dtype context only when it is not None. GLM-5.3-Flash's wrapper config carries
`dtype: None` (its `text_config.dtype` is bfloat16, and is overwritten), so the
skeleton is float32. The prepared proof says bfloat16 and 100,663,296 logical
bytes; the meta parameter says float32 and twice that.

Measured on the x86 lane (PrismaBuild key `cfa6adf0132c`, transformers 5.16.1):
the GLM skeleton's `layers[0].mlp.down_proj.weight` is meta float32
`[4096, 12288]`; a `Qwen3Config(dtype="bfloat16")` skeleton is meta bfloat16,
which is why the block had never fired on a dense Qwen model; the same Qwen
config with `dtype=None` is float32.

## The fix

The dtype and byte identity of a source tensor is a fact about the installed
checkpoint tensor, not about the skeleton. The installed dtype is decided by the
loader (`_read_layer_to_device` casts to the recorded model dtype, and fp8
dequantization hard-codes bf16), so it cannot be read off a safetensors header
either; the faithful comparison is the one prepare made, against the installed
tensor.

`compute_aura_cost_streamed` keeps the up-front coverage, key, digest-format,
shape and verified-cell checks (shape is a skeleton fact), and checks that the
dtype and byte fields are well-typed. The dtype and byte comparison moves to
`_require_installed_render_sources(layer)`, called in the reverse loop right
after `_refresh_packed_layer_views(layer)`, when `linears[name].weight` is the
installed tensor. It refuses a parameter that is still meta and a prepared
proof whose dtype or byte count differs from the installed source, before any
render of that layer is consumed. The skeleton dtype is unchanged (it is part
of the sealed source-execution identity), and the prepare path is untouched.

The forward capture of a layer runs before its reverse-loop install check, so
a real mismatch now surfaces after that capture rather than at intake. No
render is consumed before the check.

Tests: `tests/test_joint_retained_streamed.py` gained a `skeleton` hook that
replaces each Linear's weight with a meta parameter of a chosen dtype until
`install`. The first new test was red on the campaign commit `208d340d27`
(PrismaBuild key `e4c59e50f0fb`: "prepared render tensor proof differs for
model.layers.0.proj@FP8_E4M3") and is green with the fix; the second refuses a
proof that claims bf16 when the installed source is fp32.

## Carrying the fix to the sealed campaign

`_preflight_run_prepared` binds `implementation_sha256` to the digest of the
whole `prismaquant/` package, so any fix would make the run refuse the prepared
record, and a re-prepare costs about 7.4 hours. The sanctioned recovery is a
closed source transition.

`prismaquant/joint_aura_run_transition.py` (version
`meta_skeleton_render_proof_v1`, schema
`prismaquant.joint_aura.run_source_transition.v1`) is a sibling of the
2026-09-07 module rather than an edit of it; `joint_aura_source_transition.py`
is byte-for-byte unchanged, and `prismaquant/joint_aura_transitions.py`
dispatches a receipt to the module that wrote it, so both contracts live side by
side.

The contract binds the sealed prepare:

| field | value |
|---|---|
| source_sha256 | `192e73f9d2388a80caa3a4b9da3a59fda74530bb1949fcf7b9aeadad86c52a8f` |
| git_commit | `208d340d27caa3f0e85b399893fca78a8cfebfa0` |
| plan_sha256 | `0b2cc0066bb612e32af6d0c8c809912d325b2975583297eedeee97851ee545da` |
| prepared_sha256 | `962207a3385e9531adaf951b823871a2fb7ff4684320e7a8e19a1d0aa85d8f16` |
| production_cache_sha256 | `5bdd0f97849d1e9c6cd2ce126f582e551191b3684a48b7215471407ba13f0021` |
| campaign_identity_sha256 | `beaa5ed9a00cfa1a8cf2cb8c0b4b1f838484b648581d064d6ea0ed8a1da7fd13` |
| measured_cells | 197990 |

`source_proof` reverses the exact rewrites (four `aura_cost.py` hunks and the
import glue in `tessera_joint_aura.py`), omits only the two new modules, hashes
every remaining package byte with the complete-package algorithm and requires
the sealed digest. A one-byte change anywhere else, a missing module or an
extra file is refused (`tests/test_joint_aura_run_transition.py`).

Differences from the 2026-09-07 contract, and why:

- **It binds bytes, not a commit.** Every PrismaBuild submission snapshots the
  checkout into a fresh commit (the closure file is in the tree), so the commit
  the receipt was created under is never the commit the run executes under. The
  receipt records the producer package digest, the reconstructed digest and the
  verifier module digest; the loader records the executing checkout's HEAD
  (read from `.git` without a git binary) and requires it to equal the commit
  the checkpoint stamps. `execution.git_commit` and `git_parent_commit` are
  recorded for the reader, not compared.
- **There is no predecessor chain and no preserved-unit roster.** The failed run
  wrote no checkpoint. The loader accepts a missing manifest, refuses a manifest
  that names another measurement source, and refuses unit checkpoints without a
  manifest.
- **The git identity override is accepted, and verified.** The campaign image
  has no git binary, and `aura_cost._checkpoint_git_commit()` needs one unless
  `PRISMAQUANT_IDENTITY_GIT_COMMIT` names the commit. The launcher
  (`tools/tessera_campaign_container.py`) reads HEAD on the host, refuses a
  checkout whose tracked `prismaquant/` bytes differ from it (the check the
  variable disables inside the container, performed over the whole package
  instead of `aura_cost.py` alone), reports it on its JSON line and hands it to
  the container. A spec may not supply the value. The transition loader then
  requires the stamped commit to equal the checkout's own HEAD.

The receipt is still `run --resume` only, created once with exclusive creation,
and the checkpoint manifest and every unit checkpoint carry the receipt and the
actual execution identity.

## Receipt

`/mnt/shared/tessera-runs/receipts/glm53-flash-run-transition-20260918/transition.json`,
sha256 `ef7197fee1e0e2c461731246c535cc2466bd20931904b26094f190a64c0f50e3`, created
on the x86 lane from the branch checkout (PrismaBuild key `4cc44a59fc1c`;
`git_parent_commit` `854c89e811`, producer package
`5f708659593bcafe17914137cc0c418579b1c5e50b2615aa2c6f74a4c4d4c1bc`). Any later
change under `prismaquant/` on the branch changes the producer digest and needs
a new receipt.

## Submission

`tools/dispatch_tessera_campaign.py submit-joint run --resume` accepts
`--source-transition PATH` (digest computed, or checked when given), forwards
both to the pass, and declares the receipt as a head read of the joint pass
manifest (`experiments/glm_data_manifests.py`).
