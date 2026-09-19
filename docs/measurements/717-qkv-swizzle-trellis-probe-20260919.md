# #717 probe: the q/k/v one-code flip is not in the quantizer (2026-09-19)

Ran the issue's second cheap test — isolate the block-scale swizzle and the
static `trellis_input_global_scale` against the post-layernorm input — on the
refused `q_proj` K2 cell. The flip does not reproduce in the quantizer. What
does reproduce is a model-oracle-vs-kernel gap at one E2M1 midpoint, which is
not the receipt refusal (the panel priced the operator leg, not the model).

## What ran

- Input: `prefill.input` [512, 1024] BF16 from the refused cell's
  `tensors.safetensors`
  (sha256 `6896f34e1cb10187693db8fe404c9c734a0382c4b3bacc53559808092b30c9c6`,
  copied byte-identical host-local because the container cannot see the
  shared NFS mount), `G = 1.7454545497894287` from the cell's `inputs.json`.
- Probe: re-ran vLLM's compiled `torch.ops._C.scaled_fp4_quant` on the
  captured input and compared against PrismaQuant's served E2M1 oracle,
  block by block (32,768 groups of 16).
- Environment: `eugr/spark-vllm@sha256:0afec8d4…` (torch 2.13.0+cu130),
  sparky GB10, seconds of GPU at ~15 W on an otherwise idle box. The panel
  froze under `vllm/vllm-openai@sha256:61fc8a89…` on sparklina — a different
  image, different box, which is the point: a quantizer-side mechanism must
  reproduce across both (the issue already refuted the image as the variable
  for the refusal itself).
- PQ side: the real `prismaquant.nvfp4_activation_contract` from
  `origin/main` @ `54a2dc49b7` (no oracle changes since the freeze commit
  `f40f4c0364`; verified by empty log range). vLLM-exempt direct run (Rob
  2026-09-06/07): the compiled op exists only inside a vLLM image, which no
  PrismaBuild worker provides.
- Harness (one-off, outside the repo): `/tmp/opencode/pq717_probe.py`;
  canonical log `/tmp/opencode/pq717_probe_output.txt` (exit 0).

## Measured

1. **The frozen `reference_qdq` is the operator leg, bit-exact.**
   `model@g64` (pure-torch oracle): max_abs `0.1435546875`, 58 elements.
   `model@g32`: identical (G is fp32-exact, so G precision is not a
   variable). `registered-op@g64` (op codes + the contract's scale rule):
   max_abs `0.0`, 0 of 524,288 differ. The panel priced the kernel's codes.
2. **Scale bytes: 0 of 32,768 groups differ** (op plane unswizzled through
   the exact inverse of the pinned `blocked_scales`; the pair round-trips
   byte-exact both ways). The amax→UE4M3 mapping is identical.
3. **Codes: identical.** Unpacked low_first (the order PQ's own leg uses;
   high_first disagrees everywhere), dequantized through the real
   registered-leg recipe, the probe reproduces the frozen reference
   bit-exact (`max_abs=0.0`, 0 mismatched) — across image, box and day.
4. **Swizzle: refuted as a code-assignment difference.** Block partition is
   contiguous-16 along K on both sides; the 128x4 permutation is layout-only,
   exactly as the pinned route documents.
5. **Model oracle vs kernel: 58 elements, one mechanism.** Every mismatch
   sits at normalized exactly `±0.75` — the E2M1 0.5/1.0 midpoint: the model
   ties-even to index 2, the kernel takes index 1. Stored bytes `0x18` /
   `0x20` / `0x30`, used scales non-dyadic (`stored/G`, `G=1.745…`). This is
   the documented `rcp.approx.ftz` behavior the oracle docstring warns about
   and the attestation's `does_not_attest: ["non_dyadic_used_scale"]` names:
   under a non-dyadic used scale the midpoint never presents as a tie to the
   kernel. It is a priced-oracle gap wherever the model leg is priced, not
   the receipt refusal (the receipt used the op leg, which reproduces).

## What this decides and what it does not

- The 0.015625 receipt flip is **not** in `scaled_fp4_quant` on the captured
  input: same op, same input, same G reproduces the freeze bit-exact. The
  flip entered downstream — the serve's actual quantized input bytes or its
  dequant path — on q/k/v only. The gate stays locked (0.015625 refused,
  0.0 admitted); no tolerance moves.
- Not established: which serve-side bytes differ. The cell dir keeps no
  serve-side packed/scales tensor (`post-native-core.json` carries only the
  return code), so that comparison needs a serve that persists them. That is
  the concrete next probe, and it is a producer-harness change first.
- Incidental (tripped over, filed not fixed): the frozen panel records no
  served-quantizer backend — nothing in `inputs.json`/`panel.json` says the
  reference came from the operator leg, which cost this probe its first hour
  (filed PQ-side; the panel schema is not mine to widen in a remainder).
