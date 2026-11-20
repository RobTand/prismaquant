# PQ #2582 allocator offers and export refusal

The issue body gives these acceptance criteria. Its only comment records a prerequisite.

> - Allocator bids KDA, MLA and kv_b_proj under --allow-pinned with class menus. (serves C2). Verification: Allocator unit test asserts bid set and rung menus per class.
> - Export refuses attention routes while route lacks support. (serves C2). Verification: Export unit test asserts refusal for each unsupported route.
> - DSA indexer stays pinned and takes no rung. (serves C2). Verification: Roster test asserts indexer stays in pinned set.
> - Run without flag keeps prior selection bytes. (serves C2). Verification: Current allocator tests pass with no change.

| Criterion | Main verdict and nodes | Added check |
| --- | --- | --- |
| Attention bids and class menus | PARTLY. `tests/test_glm_attention_census.py::test_scoped_roster_lists_every_kda_mla_and_kv_b_unit:55` checks the campaign roster. `tests/test_glm_attention_unit_menus.py::test_absorbed_projection_keeps_t16_while_equal_shape_q_retains_full_menu:35` checks the campaign menu. | `tests/test_allocator_measured_runtime_cli.py::test_glm_attention_lift_offers_the_priced_class_menus:238` checks the bids the allocator admits from synthetic joint AURA prices. |
| Unsupported export routes | PARTLY. `tests/test_tessera_export_scope.py::test_research_route_is_refused_without_the_override:532` checks generic route refusal. `tests/test_prismaquant_export_native_compressed.py::TestRuntimeLegalAssignment::test_unexportable_format_hard_fails_instead_of_bf16_coercion:3180` checks generic container refusal. | `tests/test_tessera_export_scope.py::test_glm_attention_export_refuses_a_rung_without_a_cell:549` checks one missing route for each GLM attention class. |
| DSA indexer stays pinned and takes no rung | COVERED. `tests/test_tessera_campaign_pinned_roster.py::test_pinned_roster_only_prices_attention_and_keeps_the_indexer_pinned:62` checks the pin and excludes all indexer Linears from the priced roster. | The allocator bid check also supplies indexer prices and requires no indexer bid. |
| Selection bytes without the flag | PARTLY. `tests/test_allocator_output_pin_1304.py::test_stock_menu_preserves_decisions_accounting_and_companion_bytes:139` checks historical stock output bytes. `tests/test_glm_attention_census.py::test_body_census_without_flags_stays_byte_identical:130` checks GLM census bytes. | `tests/test_allocator_serve_constraints.py::test_glm_selection_bytes_match_when_attention_costs_are_present_without_a_lift:250` compares raw `selection.json` bytes at a binding budget. |

Commit `d19d3ef8b` removed render-score allocation. The current bid check uses joint AURA prices and checks the handoff from the priced menu.
The campaign menu test checks the BF16 activation rule for `kv_b_proj`. Its synthetic allocator menu contains T-16 and the BF16 source choice only.

Use one pytest worker and one native thread. The byte fixture writes four 256 by 256 BF16 weights, 512 KiB of tensor data.
The scoped export test uses only a synthetic contract and source header. Positive route qualification requires a matching pinned runtime cell and device evidence.
Flux CPU job `f3czuJST9U3` at `9273a0d3f` (gate venv): the six touched test files, 111 passed, exit 0; `tools/ci/static_checks.sh --merge-main` exit 0. Each of the three one-line mutations above made its test fail (1/1, 3/3, 1/1).
