# PQ #3318: streaming initialization SHA-256 grammar (part of #3061)

Owner: `prismaquant.digests.is_sha256hex` (`SHA256_HEX`, 64 lowercase hex,
full match, `str` and subclasses). `digests.py` is unchanged.

## Changed sites (`prismaquant/streaming_initialization.py`)

| Validator | Fields | Callers |
| --- | --- | --- |
| `validate_streaming_initialization_contract` | `state_sha256`, `source_map_sha256` | merge, `validate_source_initialization_contract`, `streaming_model`, `native_moe_panel` |
| `validate_streaming_prefix_initialization_contract` | derived-buffer `sha256`, `source_map_sha256` | `streaming_model`, `native_moe_panel` |
| `validate_streaming_selected_initialization_witness` | derived-buffer `sha256`, `source_map_sha256` | merge, `streaming_model` |
| `validate_mtp_layer_initialization_contract` | `source_map_sha256`, state `sha256`, input `sha256` | `validate_source_initialization_contract`, `glm_mtp` |

The local `hex64` pattern is gone. Each `str(...)` coercion, `isinstance`
check, key set, count rule, digest computation and error order is kept.

## Compatibility evidence

`tests/test_streaming_initialization.py` records owner calls per site
(`test_initialization_contracts_use_shared_sha256_grammar`). Controls cover
valid copies, `str` subclasses, `str` coercion of an int, uppercase, nonhex,
newline suffix, wrong lengths and wrong types at each field, plus state
digest equality, counts, MTP input binding and prefix/selected/complete
distinctions. Run receipts are in the change report, not here.

## Untouched residual sites

- `_prefix_layer_index` keeps its layer-name regex and the `re` import.
- No hash constructor or JSON encoding moved (#1301).
- The parent #3061 keeps the repository-wide inventory integration and the
  scanner gaps outside this module.
