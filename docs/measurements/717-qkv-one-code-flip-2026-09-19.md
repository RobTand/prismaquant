# #717: the q/k/v one-code flip (2026-09-19 note, no new run)

No new measurement here. This records what the 2026-09-18 re-measurement
decided and what is still unrun, so the next probe starts from the receipts
instead of re-deriving them.

## What the receipts say

`consume-native-receipts.json` in
`/mnt/shared/tessera-runs/receipts/frontier-qwen3-0.6b-20260918-panels/`:
53 admitted, 3 refused under the attested image
`vllm/vllm-openai@sha256:61fc8a89` (vLLM 0.28.0, GPU-b1eceeea, producer tree
`0f98fc010`, contract `db9ca4c0…` v29, PrismaQuant `f40f4c0364`):

- `model.layers.0.self_attn.{q,k,v}_proj__TESSERA_E2M1_K2_R896`: prefill
  `qdq_numerics` differs by exactly 0.015625 -- one E2M1 code at the block
  scale, with a bit-identical stored scale.
- `o_proj`, `gate_proj`, `up_proj`, `down_proj` K2 cells: bit-exact, admitted.

The same three units refused with the same 0.015625 under
`eugr/spark-vllm@sha256:0afec8d4` (PB `31b9643ed855`). Two images, two vLLM
builds, one refusal set: the executing image is refuted as the variable. The
three refused units are precisely the three that read the post-layernorm
hidden state.

## What this branch does about it

Nothing widens `atol`/`rtol`: the refusal text is right that a flipped code
is a different activation residual (principle 8, #567). The gate behavior is
locked by `tests/test_native_qdq_exact_gate.py::
test_the_september_18_qkv_refusals_are_one_code_flips_that_stay_refused`
(0.015625 refused on q/k/v, 0.0 admitted on the other four).

## Still unrun (the second cheap test from the issue)

Isolate the block-scale swizzle and the static
`trellis_input_global_scale` against that tensor's tails on the
post-layernorm input: localise WHICH positions flip (tail-driven?) by
comparing PrismaQuant's K2 quantisation of the captured post-layernorm input
against the runtime's, block by block. Inputs needed: the captured layer-0
post-layernorm tensor from the frozen panel's `reference_qdq` identity, the
pinned producer's block-scale assignment, and one GB10 action. Acceptance:
name the block(s) and the rounding-rule difference, or refute the swizzle
hypothesis with the per-block comparison attached.
