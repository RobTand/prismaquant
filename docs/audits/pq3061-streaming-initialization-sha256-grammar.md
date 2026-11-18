# Streaming initialization SHA-256 grammar: #3318

## Changed sites

Eight checks in `prismaquant/streaming_initialization.py` call
`is_sha256hex` instead of a bare regex:
- `validate_streaming_initialization_contract`: `state_sha256` and
  `source_map_sha256` (`:55`).
- `validate_streaming_prefix_initialization_contract`: each
  derived-buffer `sha256` and `source_map_sha256` (`:109`, `:116`).
- `validate_streaming_selected_initialization_witness`: the same two
  fields (`:164`, `:180`).
- `validate_mtp_layer_initialization_contract`: `source_map_sha256`, each
  tensor witness `sha256` and the input `sha256` (`:330`, `:341`, `:348`).
The function-local `hex64` pattern is gone. The layer-name regex (`:38`)
stays.
Caller-side `str(...)`, type checks, key sets, digest computation, count
rules and error order stay.
`prismaquant/digests.py:120` owns `is_sha256hex`.
It accepts string subclasses.
Callers: `prismaquant/streaming_model.py`, `prismaquant/native_moe_panel.py`
and `prismaquant/glm_mtp.py`.
Line numbers are for main at `6ca8ada35`.

## Evidence

Merge `8f90fad44` (PR #4651, #4609) made the change.
Flux job `f3B7r1YvEWb` recorded 103 compatibility cases through the
unedited consumer.
`test_initialization_contracts_use_shared_sha256_grammar` failed by
assertion on base and passed on head.
Main record run 44303024917 on `dbb7f578d` passed
`//tests:test_streaming_initialization` on amd64 and arm64.

## Residual scope

The module has no other inline SHA-256 grammar.
The layer-name regex is not digest grammar.
The duplication inventory and its shrink-only baseline stay unchanged.
#3061 retains inventory integration and scanner gaps.
#1301 recipes stay unchanged.
This note makes no speed or serving-quality claim.
