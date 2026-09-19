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

# The ram half of the residency map, carried to this branch (PQ #751)

## What this branch now carries, and why

PrismaBuild's ram tier (RobTand/prismabuild#640) promotes staged ranges onto a
`noswap` tmpfs and overlays the composed residency map with the tmpfs copies:
an entry keeps the `stage_path` it already had and gains `ram_path`, and the
map's header gains `ram_tier_id`, `ram_root` and `ram_epoch`. The sealed-era
reader this branch carried until now refuses such a map whole — the header
fields are `unknown residency map fields` and the per-entry `ram_path` is an
unknown entry field — which is the ordinary fail-open: every read falls to the
declared pool path at full pool cost, behind an ARC the same cutover shrank to
22 GiB. A campaign resubmitted under the #640 generation would have paid the
whole cost of a tier it was holding a map for. PQ #751 (merged to main as
`b568539262..55fc60c8ad`) teaches the reader the ram half; this branch now
carries it, re-tabled, so the resubmitted campaign can actually reach the tmpfs.

The semantics are exactly the merged reader's: the header trio and the
per-entry `ram_path` are validated the way PrismaBuild's own `validate_map`
validates them (a map naming ram paths must announce tier, root and epoch
together, and a `ram_path` outside `ram_root` refuses the map whole); a ram
copy is offered only while the map's `ram_epoch` equals the epoch the pool's
tier record (`<queue>/tiers/<ram tier id>.json`, `prismabuild.storage_tier.v1`,
re-read when its stat identity changes) currently announces, because a tmpfs
empties on reboot while the map survives on the shared mount; and every ram
failure — no record, an unreadable record, a missing or stale epoch, a ram
file that fails its fence — fails closed **on the ram half only**, falling back
ram → stage → declared. `residency_report()` records `ram_hits`,
`bytes_from_ram`, `ram_fallbacks` and `ram_fallback_count` beside the stage's
own counters, plus the ram header and any `ram_refused` reason, so the run's
`results.json` says what each tier served.

## What was adapted, and what did not travel

This branch's `prismaquant/residency_map.py` is the sealed-era reader: it has
the whole-file `staged_read` and predates the ranged shard reader
(`staged_range`, `_interval_index`) that main gained later, and nothing on this
branch calls the ranged half. The ram half therefore landed on the structures
that exist here — the header and entry validation in `_adopt`/`_entry`, the
epoch check (`_announced_ram_epoch`/`_ram_live`/`_ram_offer`), the ram-aware
fence inside `staged_read`, `record_ram_read`/`record_ram_fallback`, and the
per-tier `report()` — byte-identical to the merged reader where the two
structures agree. `tessera_joint_aura._read_verified_wire_blob` reads ram, then
stage, then the declared path, counting each half separately; the transition
wiring (`prepared_plan_sha256` at both its call sites) is untouched.

What did not travel: the merged reader's `staged_range` fence and its test
(`test_a_staged_range_offers_the_ram_copy_and_retires_it_with_the_epoch`) —
there is no `staged_range` on this branch to fence. The epoch retirement that
test proves is covered on the whole-file path by
`test_an_epoch_rollover_between_lookups_switches_the_ram_half_off`, which is
carried. `tests/test_prismabuild_ram_tier_residency.py` otherwise travels
verbatim from the merge.

`residency_map.py` is not a `_NEW_FILES` entry — the sealed package has the
file — so the ram half travels through the rewrite table like every other
sealed-file change: the regenerated `_SOURCE_REWRITES` below carries its
hunks, `source_proof` still reconstructs the sealed package byte-for-byte, and
the producer digest changes, which is why carrying this needs a new receipt
before the resubmission.

## The measurement D44 still owes

Main's `docs/ARCHITECTURE.md` §12 D44 records it and this carry does not
discharge it: no consumer-side ram read has ever been measured, and no
`bytes_from_ram` number has been written by a run. The fences and the epoch
gate are proven by the carried tests, but the ram tier's payoff is a design
claim until one joint pass on the #640 generation reports its per-tier
counters. The campaign run this branch is composed for is exactly that
instrument; read its `residency` block — `ram_hits`, `bytes_from_ram`,
`bytes_from_stage`, `bytes_from_pool` — against the stage-only arm before
extending the preference to any other read site. D44(b) applies here too: only
the wire reader prefers the ram copy; the production weight cache's shard load
still reads `stage_path` unchanged, so a fully ram-promoted window serves that
site from the SSD.

# The resumable parallel head walk, carried to this branch (PQ #754/#763)

## What this branch now carries, and why

The withdrawn campaign night paid the joint command's head walk —
`tessera_joint_aura.load_measured_anchor_input` reading and verifying the
whole measured-anchor roster — and then paid it again on every restart:
`c5c88680bd86` walked 54m37s, committed its progress, and lost all of it
when the action was withdrawn; the resubmission re-paid the walk before
its first unit of new work. A resubmitted campaign must not open that
way, so this branch now carries #754's walk machinery (main `f1b4ab1d7a`):
every verified unit is banked under the campaign's own checkpoint
machinery (`prepare_journal`/`write_unit`) into `<output
root>/head-walk`, the journal's identity binding the exact input set
(plan, census, receipts, merged cost and checkpoint digests), the
checkpoint seal, `roster_sha256`, the render mirror root and the admitted
encoder reuse; a resume re-verifies the banked prefix against the very
bytes it was banked from — the journal shard's file digest plus each
render/marker/wire stat fence — truncating at the first drift, and a
journal that fails its own checks is moved aside (`.stale`) and the walk
restarts fresh: ignored, never reused, never a refusal that blocks the
run. The verification fans out over threads — the count is the CPU set
PrismaBuild assigned this container (`os.sched_getaffinity`, capped at
16, `PRISMAQUANT_HEAD_WALK_WORKERS` to lower it or force the serial path)
— while commitment stays in the roster's one deterministic order, so a
parallel walk's durable state, progress sequence and final input are
identical to the serial one's, and the one thread-unsafe step (a missing
render's synthesis through the bound single-threaded reader) holds a
lock. The two gate test fixes ride along: the gate's roster order is
`sorted(names)`, and the interrupted run's own report is held.

## What was adapted, and what did not travel

Nothing was adapted: the carry is byte-identical to the merge. The
ram-reader carry had already made this branch's `tessera_joint_aura.py`
equal to main's pre-walk file, so the walk hunks applied with zero
conflicts, and the walk banks through `cost_stage_checkpoint`, which the
sealed-era package has carried all along. It touches no reader structure
— `residency_map.py` is untouched by it — so nothing here depends on
main's interval machinery, and the sealed-era whole-file reader with the
ram half keeps serving the wire reads exactly as the ram carry left
them. `tests/test_joint_head_walk_754.py` (12 tests) and the three small
gate fixes travel verbatim. What did not travel is main's
`docs/ARCHITECTURE.md` D45 stamp: it belongs to the merge this carries
from, and this note is the branch's record of the same debt.

`tessera_joint_aura.py` is a sealed file, so the walk travels through the
rewrite table like every other sealed-file change: the regenerated
`_SOURCE_REWRITES` below carries its hunks (6 rewritten files, 73 hunks,
4 new files), `source_proof` on the live tree still reconstructs
`192e73f9d2388a80caa3a4b9da3a59fda74530bb1949fcf7b9aeadad86c52a8f`
byte-for-byte, and the producer package digest moves, so the
resubmission needs a new receipt before it runs — the coordinator's, not
this carry's.

## The measurements D45's campaign run still owes

D45's open half is discharged by no test and not by this carry: the
mechanism is gated (`test_joint_head_walk_754.py`: resumed equals fresh,
parallel equals serial, workers never exceed the assignment, unverifiable
journal state discarded fail-closed) but its payoff on the real campaign
is unmeasured, because no flagship joint pass has run since it landed.
The campaign this branch is composed for is the instrument, and it owes
three readings: (a) the parallel/serial A/B of the walk itself —
`PRISMAQUANT_HEAD_WALK_WORKERS=1` against the affinity default under the
same reservation, head-phase wall time and `/proc/PID/io` either side
(threads bet on I/O latency; the unpickle half of each unit still holds
the GIL, so expect a partial, not linear, speedup); (b) a restart
mid-walk resuming from the journal rather than zero —
`head_walk_resumed_units` in `results.json` beside the resumed wall
time; (c) whether the 4-CPU reservation the withdrawn submission carried
should be raised — PB's placement already prefers physical cores and
demotes SMT siblings and E-cores on both boxes, applied with taskset
before exec, so a larger reservation lands on P cores by construction
and the only open question is its size, to be repriced from the measured
bottleneck.
