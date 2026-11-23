# Native panels record the served quantizer that produced the reference

**Status: shipped.**

## Preparation

`prepare_native_inputs` computes `reference_qdq` with the selected format.
For a measured-as-served static activation contract, it writes the top-level
`reference_served_quantizer` stamp next to `activation_quantizer_attestation`.
The stamp records the effective served quantizer identity for the contract
that computes the reference. An explicit contract binding takes priority
over the process binding. Preparation does not write the stamp for other
activation paths.

## Freeze

Both freeze paths require a dictionary stamp with backend
`registered_scaled_fp4_quant` for a measured-as-served static reference.
Both paths carry the stamp into the frozen panel unchanged.

- Ordinary freeze checks the stamp against the validated joint AURA probe
  arithmetic. The comparison requires a quantizer identity in the probe.
  It checks `schema` and the reuse axes: `backend`, `op`, `platform`,
  `torch`, `torch_git`, `vllm`, and `image_content_sha256`.
  It does not check `dequant_kernel`.
- Execution-first freeze has no joint quality row. It checks the stamp
  without comparing joint arithmetic.

## Receipt admission

`consume_native_receipt` checks the expected panel's stamp against its final
joint arithmetic. For a late-bound receipt, it resolves the binding first.
The binding must preserve the measured execution panel, including the stamp.
Admission then applies the same arithmetic comparison as ordinary freeze.
It refuses a measured-as-served static reference without a registered stamp.
Other activation paths do not require a stamp.

The exact QDQ gate still requires zero absolute error.
