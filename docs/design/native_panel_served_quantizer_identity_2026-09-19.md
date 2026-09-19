# Frozen native panels record no served-quantizer backend — decision record and schema proposal

Date: 2026-09-19 · Branch: `flash/773-775-docs-20260919` · Issue: RobTand/prismaquant#775 (open, informational).

**Status: proposal only. No code changed on this branch, no sealed surface
moved. GLM decides.** This record exists so #775 can be closed by a later
implementing PR with its own tests, not by this docs PR.

## What fails

A frozen native panel's `reference_qdq` cannot be re-derived from the panel
alone. `inputs.json` (schema `prismaquant.native_dense_inputs.v1`) names
only the method —
`quantizer: prismaquant.nvfp4_activation_contract.StaticActivationContract.quantize_dequantize`
(`prismaquant/joint_aura.py:127`, via `activation_identity`) — and that
method runs one of three arithmetics depending on the process binding
(`prismaquant/nvfp4_activation_contract.py:2376-2418`):

- no binding + `measured_as_served=False`: this tree's Torch model
  (`nvfp4_activation_qdq_served`);
- `SERVED_QUANTIZER_BACKEND_REGISTERED_OP` (`"registered_scaled_fp4_quant"`):
  `torch.ops._C.scaled_fp4_quant`, the operator the serve executes;
- `SERVED_QUANTIZER_BACKEND_MODEL` (`"prismaquant_model"`): the Torch model,
  bound explicitly.

Nothing in `inputs.json` or the frozen panel (schema
`tessera.native_dense_panel.v1`) says which one produced the reference.

Measured on the #717 probe (2026-09-19, panel image family,
`eugr/spark-vllm@sha256:0afec8d4`, sparky GB10): recomputing the refused
`q_proj` K2 cell's captured input with the model oracle misses the frozen
`reference_qdq` on 58 of 524,288 elements (max_abs 0.1435546875, all at
normalized exactly ±0.75 — the E2M1 midpoint where the model ties even and
the kernel takes index-1 under a non-dyadic used scale, the documented
`rcp.approx` behavior the attestation does-not-attest); the registered-op
leg reproduces it bit-exact (0 differ). The probe had to try all three
backends to learn that. Today the guess is cheap and the bit-exact gate
picks the winner; a future backend, or a variant that does not reproduce,
turns the guess into an investigation. Severity P2 (provenance), per #775.

## The established pattern this should follow

The render-score path already solved the same problem and is the template.
`production_weight_cache.py:2287` stamps `served_quantizer` beside
`input_global_scale_policy` on every retained render-score record, read
through the one effective-identity accessor
(`_scored_served_quantizer_record`, `:2301`: the arithmetic the row ran and
the arithmetic it is stamped with cannot be two answers), with `None`
meaning "no static G was priced, so no quantiser was touched" — never
"unknown". Reuse refuses on mismatch
(`require_matching_served_quantizer`,
`nvfp4_activation_contract.py:2213`: a model-priced row and a
registered-operator row are different objects — the retained 84-group
differential priced 24 of 172,032 probed elements one E2M1 code apart — so
a cache filled under one is refused for the other).

The serial form already exists: `ServedQuantizerIdentity.as_record()`
(`nvfp4_activation_contract.py:1969`) carries `schema`
(`prismaquant.served_quantizer_identity.v1`), `backend`, `op`, `platform`,
`torch`, `torch_git`, `vllm`, `image_content_sha256`.

The panel path is the one place that prices through
`contract.quantize_dequantize` — `prepare_native_inputs`
(`native_operator_panel.py:260`) reaches it via `_activation_qdq`
(`perturbed_x_cache.py:123`) — but stamps only the method name. The
`activation` dict it freezes carries `static_contract.execution`,
`group_size`, `measured_as_served`, `input_global_scale` and
`served_scales_enabled` (an env bool, not the backend); the in-process
binding (`bind_served_quantizer_identity` /
`effective_served_quantizer_identity`) is ambient state that the freeze
never records.

## Schema proposal (additive; needs GLM sign-off, not implemented here)

1. `prepare_native_inputs` stamps `served_quantizer:
   effective_served_quantizer_identity(contract).as_record()` (or `None`
   where the panel prices no static scalar — the same none-meaning the
   render-score record uses) as a new **top-level** inputs member, beside
   `activation_quantizer_attestation` — not inside `activation`, which
   `freeze_native_panel` binds by digest against the cost row's joint
   activation (`:336-340`).
2. `freeze_native_panel` carries it into the panel unchanged.
3. `consume_native_receipt` requires it on panels that carry it (same
   refuse-on-mismatch shape as `require_matching_served_quantizer`) and
   treats an absent key as **unknown-by-construction** for older panels —
   never inferred, never backfilled. Refusing pre-record panels outright
   would strand every frozen panel in the corpus, including the 49-cell
   study set D37 already owes a re-freeze for; the re-derivation fallback
   (try the bound backends, bit-exact gate picks one) keeps working and is
   recorded as the legacy path.
4. No schema-version bump is proposed yet: whether the member rides inside
   the v1 schemas as optional or opens v2 inputs/panel schemas is the
   decision GLM owns, with the Tessera side in the room.

## Why this is a proposal and not a commit

Stamping the member changes frozen bytes, and frozen bytes are sealed in
three places, two of them cross-repo:

- `consume_native_receipt:374-375` binds `receipt["panel_sha256"]` to the
  panel object and `:372` requires the receipt's carried panel to equal
  the expected one — a new member re-digests every panel and strands
  every receipt that cites the old digest.
- Tessera's producer harness checks the frozen panel by exact field set
  (the D37/tessera#546 lesson: `experiments/bench_native_operator.py`
  `validate_panel` refused every panel carrying the two newer members
  until #546 widened the tuple). A new member here needs the same
  widening there first, or the re-freeze cannot run at all.
- `freeze_native_panel`'s `_equal` chain binds inputs against the cost
  row, the preflight and the wire; the member's placement (top-level vs
  inside `activation`) decides which old/new combinations refuse, and
  that compat table is a contract decision, not a worker fix.

Per AGENTS.md principle 14 (needs a decision only Rob prices: a sealed
surface moves), this branch implements **nothing** — not even the
additive half. There is no additive-only code step: the smallest honest
step (record the identity at freeze time) IS the sealed-surface change.
What lands here is this record plus the ARCHITECTURE.md re-stamp that
names it.

## Owed (whoever implements, with GLM sign-off)

- The three code steps above, with the compat table for old panels, old
  cost rows and old receipts spelled in the PR.
- Tessera-side `validate_panel` widening (same shape as #546), or panels
  the producer cannot freeze.
- Tests: freeze-then-consume round trip carrying the stamp; mismatch
  refusal naming the backend; absent-key panels consuming as
  unknown-by-construction; a three-backend re-derivation test pinning the
  legacy fallback the #717 probe used.
- Re-freeze policy for the existing corpus (fold into D37's owed
  re-freeze or justify separately).
