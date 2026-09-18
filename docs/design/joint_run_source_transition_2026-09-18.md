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

---

# Second version: a retained-budget-only plan correction (PQ #747)

## What failed, again

The corrected COST run cannot use the sealed plan. PrismaBuild action
`ad8803aa32333dffc52b5d35d7203e0a544a62c88babb013fe763312c35fefe4` captured
boundaries for 137.4 minutes and refused
`model.language_model.layers.44.mlp.experts.0.down_proj` on a
`candidate_delta_bytes` of 4,194,304, against a roster whose smallest target
demands 33,554,432 and whose largest demands 201,326,592. `max_windows_per_layer:
2` is a second refusal against a packing that needs 78. #743 has the measurement
and #745 derives every demand-driven cap from the roster it must admit, so a
corrected plan is available on demand.

Nothing could run it. `tessera_joint_aura.execute` compares
`prepared.plan_sha256` with the running plan's digest in two places, and
`joint_aura_run_transition._CONTRACT` pins the sealed plan digest as a literal.
A re-prepare costs 7.41 hours at an aggregate read rate of 233 MB/s (#725), and
every fix in #725 moves `implementation_sha256`, which the same contract pins,
so a prepare-code change forces a re-prepare of its own.

## Why reusing the prepare is sound

The retained-window budget governs how the run packs its windows and how many
times it replays each probe's reverse pass. It governs nothing the prepare
measured.

The prepared record has fifteen top-level fields. Fourteen of them are
functions of inputs the budget does not reach:

| field | what produces it |
|---|---|
| `calibration_input` | the calibration draw, `config["calibration_input"]` |
| `encoder_source_reuse` | `config["historical_encoder_reuse"]` and the measured intake |
| `formats_by_qname` | the candidate roster in `config["inputs"]` |
| `implementation_sha256` | the `prismaquant/` package |
| `measured_cells` | the same roster |
| `plan_sha256` | **the plan** |
| `production_cache` | the renders `prepare_cache` wrote |
| `projection_backend` | `execution["projection_backend"]` |
| `reader_identity` | `config["reader"]` |
| `render_comparisons`, `render_origins` | the intake census |
| `schema`, `status` | constants |
| `source_execution` | the built model's modules |
| `source_model_identity` | `config["model"]` and the source prefetch |

`plan_sha256` is the one field that is a function of the plan at all, and the
substitution is what this version admits.

The prepare command never reads the budget. `compute_aura_cost_streamed` is
given `retained_operator_windows` only on the cost leg
(`tessera_joint_aura.py`, the `command == "run"` call); `prepare_cache` is not
given it at all. Two places touch the block for both commands and neither
produces a measured byte: `_operator_window_policy` calls
`normalize_retained_execution`, which validates and returns a normalized copy,
and `execute` calls `require_bounded_capture_environment` when a plan declares
either a qualification window or a retained block, which checks the process
environment. `_admit_candidate_phase` bounds a prepare by
`config["max_render_bytes"]`, not by the retained budget.

That is the field-by-field half. The structural half is stronger: the
transition's own proof holds every plan byte outside the enumerated keys
identical, and the run re-derives `source_model_identity`, `source_execution`,
`calibration_input`, `measured_cells`, `reader_identity`,
`encoder_source_reuse`, `render_origins`, `render_comparisons`,
`projection_backend` and `formats_by_qname` from the running plan and compares
each with the prepared record. Those comparisons are not weakened by the
substitution -- they still run, against bytes the proof holds identical. That
is why `plan_sha256` can be substituted rather than checked twice.

## The mechanism

`prismaquant/joint_aura_retained_budget_transition.py`, version
`meta_skeleton_render_proof_retained_budget_v1`, schema
`prismaquant.joint_aura.run_source_transition.v2`. It is a sibling of the two
transitions beside it, not an edit of either; `joint_aura_run_transition.py` is
byte-for-byte unchanged and `joint_aura_transitions.py` dispatches a receipt to
the version that wrote it.

It binds two plans. `prepared_plan` is pinned by the contract as a literal, the
same digest `meta_skeleton_render_proof_v1` pins. `run_plan` is pinned by the
receipt: its digest cannot be a literal, because the plan that corrects the
caps is composed from the roster after this module is written.

Admission is `plan_difference`. Each enumerated key is removed from both parsed
plans with `_take`, one whole key at a time, and the two residues must be equal
as canonical JSON. A path names one key, so a key that carries a subtree needs
no prefix rule to describe, and there is no walk to get wrong. The residue
comparison is on the parsed plans because that is the object the pass consumes
(`execute` is handed the parsed `config`); both files are pinned by their own
SHA256 besides, so a reformatted file is still a bound file.

The enumerated set is two literals in the contract:

- `admitted_budget_keys`: the thirteen `RetainedWindowBudget` fields under
  `execution.retained_operator_windows.budget`. Each must be present in both
  plans and hold an exact integer. The budget's own `schema` is deliberately
  not among them, so a plan that reversions the budget refuses.
- `admitted_record_keys`: `retained_window_budget_derivation`, the record
  `tools/derive_retained_window_budget.py` stamps beside the caps it composed.
  It may be absent from either plan, it is recorded verbatim in the receipt,
  and nothing reads it. It explains the caps; it never stands for them.

The receipt carries `plan_difference`: one entry per enumerated key whose value
differs, with the path and both sides. A budget cap is recorded as the integer
it is; a record key is recorded by the SHA256 of its canonical form, because
the difference is stamped into every unit checkpoint through
`execution_provenance` -- 36,423 of them on this campaign -- and the plan that
holds the record is bound by its own digest besides. The loader re-derives the difference
from the bound plans and requires the recorded one to equal it, so the record a
reader sees is the record the gate read.

`tessera_joint_aura.execute` asks which plan digest the prepared record must
carry, at both places that compared it, through
`joint_aura_transitions.transition_prepared_plan_sha256`. That is a literal
table keyed by verified capability type: the resume transition and
`meta_skeleton_render_proof_v1` answer with the running plan's digest, this
version answers with the plan the prepare was made against, and a type the
table does not name refuses. The wiring travels in this version's rewrite table,
so `source_proof` requires each of the three hunks present exactly once in the
executing package.

## The rewrite table, and the branch it describes

`_SOURCE_REWRITES` is this module's own literal table, not a reference to the
sibling's: an edit there must never move this proof. It carries nine hunks over
two files -- five in `aura_cost.py`, the install-time render proof and its
import glue, and four in `tessera_joint_aura.py`, that file's import glue plus
the three that wire `prepared_plan_sha256` -- and `_NEW_FILES` names the three
transition modules. `tools/generate_transition_rewrites.py` chose that split;
`meta_skeleton_render_proof_v1`'s hand-written table expresses the same change
in five.

That table describes the **campaign branch**: the sealed commit `208d340d27`
plus the install-time render proof plus this version. It does not describe
`main`, which has moved by 21 files since the prepare was sealed. This is the
same arrangement `meta_skeleton_render_proof_v1` has: its table has never
reconstructed `main` either, and both real-package tests are environment-gated
for exactly that reason. The campaign preflight for a run under this version runs
`tests/test_joint_aura_retained_budget_transition.py` with
`PRISMAQUANT_BUDGET_TRANSITION_REAL_PACKAGE=1`, not the sibling's gate.

On that branch `meta_skeleton_render_proof_v1` refuses, and refuses harder than
before: the version-two module is a file its `_NEW_FILES` does not name, so it
is hashed into the reconstruction and the package no longer reaches the sealed
digest. Measured on PrismaBuild `b21a1ba13b8c`: `unapproved producer package
change`, `joint_aura_run_transition.py:171`. Version one got stricter, never
looser -- its own contract, rewrites, tests and behaviour are unchanged, and a
run that wants it must run a checkout that carries only what its table
describes.

`tools/generate_transition_rewrites.py` writes the block. Give it the sealed
package and the package that will execute and it emits each hunk with the
smallest line context that makes both sides unique in their own file, then
applies the table in reverse and requires the sealed digest. It decides
nothing: whether a change belongs in a transition at all stays the author's
judgement, recorded here.

    python3 tools/generate_transition_rewrites.py \
        --sealed-rev 208d340d27caa3f0e85b399893fca78a8cfebfa0 \
        --package prismaquant --repo <campaign checkout> \
        --new-file joint_aura_run_transition.py \
        --new-file joint_aura_transitions.py \
        --new-file joint_aura_retained_budget_transition.py \
        --expect-sealed-sha256 192e73f9d2388a80caa3a4b9da3a59fda74530bb1949fcf7b9aeadad86c52a8f

## What this version does not admit

`max_gpu_bytes` and `aggregate_memory_bytes` are top-level plan keys and are not
in the enumerated set. This matters for the host-versus-device split, which is
what sets the replay multiplier: #745 measures 78 windows and 44.8x replay at
the sealed 24 GiB host cap, 13 windows and 8.0x at 32 GiB, and 7 windows and
4.98x at 48 GiB. The host cap is `physical_limit_bytes - max_gpu_bytes`.
`physical_limit_bytes` is an enumerated budget key, so raising it alone is
admitted; lowering `max_gpu_bytes` to buy host capacity at a fixed
`physical_limit_bytes` is not, and refuses with the message that names the
residue.

That is a deliberate boundary, not an oversight. Neither key reaches a prepared
field, so extending the set is a one-line change to a literal in a module the
reconstruction omits, plus a new receipt. It is a decision about what a run
transition is allowed to move, and it belongs to whoever is composing the
corrected plan.

## Evidence

`tests/test_joint_aura_retained_budget_transition.py` holds the refusal and
admission proofs: every enumerated budget key admitted alone, the derivation
record admitted and recorded, ten mutations outside the set refused at every
depth (top-level value, added key, removed key, a value inside `execution`, a
sibling of the budget, the budget's own schema, a removed budget key, the
retained block's schema, a deep value inside `inputs`, a scalar turned into a
list), a non-integer cap refused, the sibling transition refusing the corrected
plan, and the receipt carrying both digests and the difference.
`tests/test_generate_transition_rewrites.py` holds the generator.
`tests/test_joint_aura_run_transition.py` is unchanged and still passes.
