# Cross-box render identity: sparky and sparklina agree bit-for-bit

**Date:** 2026-08-31 · **Status:** measured, small scope — read the Limits.
**Reproduce:** `python3 tools/render_identity.py --out <file>`, run on both
boxes. Moved here with the script from PrismaBuild's `docs/` and
`tools/fleet/` on 2026-09-28 (RobTand/prismabuild#1076).

## Why this had to be measured before anything was built on it

PrismaBuild keys results by an action key and serves them from a CAS. That is a
promise: *the same action produces the same bytes anywhere in the fleet.* If a
production render differs between sparky and sparklina, the promise is false —
the cache would hand one box's result to the other and call it a hit, and the
action key would be a lie rather than an identity. Distributing render work
across boxes is unsafe until this is checked, and it had never been checked.

## Result

Real production path — `render_production_weight` with the shipping levers
(`gptq`, `static_act_order`, `joint_scale_opt`) — on real GLM-5.3-Flash weights,
512x1024 slices of layer 0, hashed as raw bits (`view(torch.uint8)`, not cast to
float32, which would hide a low-bit difference):

| format | tensor | agreement |
|---|---|---|
| NVFP4 | `self_attn.q_proj` | identical |
| FP8_E4M3 | `self_attn.q_proj` | identical |
| NVFP4 | `mlp.gate_proj` | identical |
| FP8_E4M3 | `mlp.gate_proj` | identical |
| NVFP4 | `mlp.down_proj` | identical |
| FP8_E4M3 | `mlp.down_proj` | identical |

**6 identical, 0 differing.** Both boxes: NVIDIA GB10, driver 595.84,
torch 2.11.0+cu130. The synthetic activations hash identically on both boxes
first, so any render difference would have been the render, not the input.

## The first run of this measured the wrong code path

Worth recording, because the failure was silent and the class of it recurs.
`render_production_weight` looks its activations up by **qname**
(`activations.get(qname)`), and the first version of this script passed
`{"input": X}`. The lookup returned `None`, GPTQ had nothing to run on, and all
four lever settings — full, no-JSO, no-GPTQ, bare RTN — produced **the same
digest**. Nothing raised. The 6/6 identity result was real but it was identity
of the *RTN* path, while this document claimed the shipping levers.

A wrong activation key is therefore not a crash, it is a quiet demotion to a
weaker render. The script now proves the levers engaged before it will report
anything: it renders one tensor with `levers={}` and exits non-zero unless that
digest **differs** from the levered one. Both boxes report
`levers engaged: True`, with matching levered (`7498639d…`) and RTN
(`dfc49c9c…`) digests, and the proof is asserted to describe a row in the table
above rather than a separate render. The lesson is the repo's own: *check the
code executes before believing the number it produced.*

## What this does and does not clear

**Cleared:** distributing the **production-cache render** across these two
boxes. The stripe/union toolchain can now be driven by PrismaBuild actions
without the CAS silently conflating two different results.

**NOT cleared:**

- **The KL stage stays blocked.** Different code path — a full forward, not a
  weight render — and prismaquant's own record is that KL is bit-identical
  within a docker session but drifts 4-8x *across* sessions. Nothing here speaks
  to that.
- **Scope is three tensors of one layer of one model**, 512x1024 slices.
  Determinism that holds on these shapes can still fail on a shape that selects
  a different kernel or a split-k reduction.
- **Activations are synthetic and deterministic**, not real calibration
  activations from a streaming cache. The real path's activation provenance is
  an additional variable this does not test.
- **Matched hardware only.** Same GPU model, same driver, same torch build. This
  is identity across two boxes that happen to be twins; it is not a claim about
  heterogeneous hardware, and the fleet doc's `rocm-16g` class is untested.
- This is a **weight-space digest**, not a served metric. It says the bytes
  agree, not that the artifact is good.
- **One lever setting.** The proof above shows the levers engaged; it does not
  survey them. Identity is measured for `{gptq, static_act_order,
  joint_scale_opt}` only, not for every combination the pipeline can select.
