# One admitted fp4 native cell, 2026-09-18

`panel.json` and `receipt.json` of
`model.layers.0.self_attn.o_proj__TESSERA_E2M1_K2_R896` from
`/mnt/shared/tessera-runs/receipts/frontier-qwen3-0.6b-20260918-fp4/native-fp4/cells/`,
the first `TESSERA_E2M1_K2` cells ever frozen. This cell is one of the four
whose activation representation is bit-exact with the served quantiser
(`qdq max_abs = 0.0` at both phases), so the only refusal a test built on it
can see is the one the test is about.

Verbatim except for the derivations recorded here:

* `resources.status` is `incomplete` rather than `complete_operator_bound`,
  and `resources.trace_sha256` is dropped with it. The real receipt's memory
  trace is 3.7 MB and `consume_native_receipt` demands it only for a complete
  bound; nothing this fixture is used for reads a resource bound.
* Re-serialised with sorted keys and two-space indent, so the bytes differ
  from the originals while the objects do not. Every test that consumes it
  writes its own bytes and digests what it wrote.

* The other `reference_served_quantizer` fields were NOT RECORDED in the 2026-09-18 receipt. `None` means "not recorded", not "absent by design".
* Today's code derives `prismaquant_model` from the recorded quantizer name and execution. It is not a recorded field. See `prismaquant/nvfp4_activation_contract.py`, `SERVED_QUANTIZER_BACKEND_MODEL` and `resolve_served_quantizer_identity`.
* Source: `/mnt/shared/tessera-runs/receipts/frontier-qwen3-0.6b-20260918-fp4/native-fp4/cells/model.layers.0.self_attn.o_proj__TESSERA_E2M1_K2_R896/receipt.json`.

Measured in `eugr/spark-vllm@sha256:0afec8d4…`, which is NOT the image the
pinned contract's quantiser table was generated in
(`vllm/vllm-openai@sha256:61fc8a89…`). That difference is
RobTand/prismaquant#715.
