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

Both freeze paths and receipt admission validate the stamp for a
measured-as-served static reference. Other paths do not require a stamp.
The check requires schema `prismaquant.served_quantizer_identity.v1`,
backend `registered_scaled_fp4_quant`, and `op` set to `scaled_fp4_quant`.
It requires nonempty strings for `platform`, `torch`, `vllm`, and
`image_content_sha256`. It requires `torch_git` but permits `null`.
The stamp can omit `dequant_kernel`.

Both freeze paths carry the stamp unchanged. Ordinary freeze compares its
schema and reuse axes against the validated joint AURA probe arithmetic.
The comparison ignores `dequant_kernel`. Execution-first freeze does not
compare arithmetic because it has no joint quality row.

## Receipt admission

Late-bound admission resolves the binding before it checks the stamp.
The binding must preserve the measured panel and its stamp.
Admission compares the schema and reuse axes against the final joint arithmetic.

The exact QDQ gate still requires zero absolute error.
