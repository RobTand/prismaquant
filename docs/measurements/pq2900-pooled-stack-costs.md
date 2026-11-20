# PQ #2900 pooled stack costs

The open issue snapshot has no comments. Its current acceptance criteria are:

> 1. Make the existing transfer-law table construction support initial distributed evidence and subsequent measured replacement. This extends the current implementation rather than rebuilding the law or its provenance.
> 2. Two static integration hazards need coverage. _fit_stack_transfer_laws pools only the anchors supplied to the payload builder (prismaquant/tessera_campaign_table.py:892-937); merging already-priced shards does not perform a global refit. A one-stack shard can therefore lack the other-stack evidence necessary for held-out prediction. Also, _stack_rate_evidence requires exactly one full-coverage reference rung (:842-890); selective completion can create a second full rung and invalidate that inference. These are inspected source findings, not reproduced failures.
> 3. Use W1's explicit reference identity and assemble compatible authenticated anchor evidence across campaign rows before pooled fitting. Continue leaving the predicted stack out of its slope fit. Preserve the pinned reference when additional target rungs become fully measured. Rebuild costs from accumulated measurements: a fully completed target replaces its predicted row at the same packed decision/format key with measured provenance, while incomplete targets remain honestly predicted or refused. Remove stale transfer-law/interpolation metadata from measured replacements; model-predicted rows must never acquire HT estimator or sampling-standard-error fields. Preserve untouched predictions and dense-unit behavior.
> 4. Keep packed bytes, parameter counts and quality normalization on the existing full-stack accounting path, independent of draw size. Atomic measurements must not become additional scalar decisions alongside their packed owner. Preserve member anchor and wire evidence for later joint intake rather than treating a packed cost as a source wire.
> 5. CPU acceptance: equivalent monolithic and cross-row pooled evidence, honest refusal without a held-out peer, second-full-rung refresh with the original reference retained, partial-to-complete measured replacement, consistent currency/shape accounting, deterministic replay and no atomic/packed double counting. Extend existing transfer-row, sample-cost and merge-integrity suites.

| Criterion | Main at `2e62978d1` | Row 72 coverage |
| --- | --- | --- |
| 1. Distributed construction and replacement | PARTLY. `tests/test_tessera_campaign_merge_integrity.py::test_valid_merged_journal_resumes_through_the_owned_reader:144`. | Journal replay now checks pooled prices against a monolith. The refresh test checks replacement at the same key. |
| 2. Refit and second full rung | PARTLY. `tests/test_tessera_campaign_fanout.py::test_merge_reproduces_a_whole_scope_table:366` checks dense rows. `tests/test_tessera_stack_transfer_law_rows.py::test_incomplete_rung_keeps_transfer_law_after_extra_expert:217` checks partial coverage, added in `308b76d1e`. | The journal union test checks pooling before pricing. The refresh test adds a second full rung on either side of the reference. |
| 3. Reference, pooling, and provenance | PARTLY. `tests/test_tessera_stack_transfer_law_rows.py::test_a_predicted_rung_declares_the_law_it_was_predicted_by:162` and `tests/test_tessera_stack_transfer_law_rows.py::test_a_predicted_rung_carries_no_sampling_estimator:192`. | The refresh test checks clean measured provenance and preservation of other predictions. The sample-cost test checks the dense price and wire size. |
| 4. Full-stack accounting and source evidence | PARTLY. `tests/test_tessera_stack_sample_cost.py::test_stack_row_prices_exactly_what_the_expanded_experts_would:193`, `tests/test_tessera_stack_sample_cost.py::test_measured_members_are_not_also_cost_keys:239`, `tests/test_tessera_shape_dependent_recipe.py::test_a_packed_expert_stack_is_priced_as_one_unit_per_expert:435`, and `tests/test_tessera_stack_population_bridge.py::test_census_packed_assignment_carries_all_selected_source_receipts:84`. | The census and draw cases check the same shape, parameter count, and MSE. Journal replay checks retention of anchor and wire records. |
| 5. CPU acceptance | PARTLY. `tests/test_tessera_campaign_merge_integrity.py::test_refusal_order_does_not_depend_on_anchor_insertion_order:156` and the transfer-row and sample-cost nodes above. | The new cases check journal replay in both orders, one-stack refusal, and complete-target refresh. |

## Tests for the lead

- `tests/test_tessera_campaign_merge_integrity.py::test_authenticated_row_journal_union_prices_like_one_campaign:173`. Drop `wire_records` when `merge_checkpoint` assigns `states[qname]`.
- `tests/test_tessera_stack_transfer_law_rows.py::test_a_stack_without_a_peer_only_prices_its_reference:241`. Change `stack_transfer_law_absent` to `stack_rung_incomplete_over_sample`.
- `tests/test_tessera_stack_transfer_law_rows.py::test_completed_target_replaces_prediction_and_keeps_reference:256`, cases `[832]` and `[1088]`. Require `len(census) == 1` after a target becomes complete. Main has this condition at `prismaquant/tessera_campaign_table.py:839`.
- `tests/test_tessera_stack_sample_cost.py::test_measured_members_are_not_also_cost_keys:240`, cases `[census]` and `[draw]`. Put `a.memory_bytes` in the member's `wire_bytes` field.

The lead runs the CPU tests in Flux. Source inspection predicts that both refresh cases lose the remaining price on main.
The retained `merge_payloads` helper still unions priced rows at `prismaquant/tessera_campaign_table.py:2357`.
The former dispatcher route is MOOT. `bc0f1420c` removed its only live caller of `merge_payloads`.
The journal test supplies the sample plan and synthetic member wire records. It checks journal integrity and cost construction.
Live selective refresh still needs an explicit reference in the producer plan. Main supplies no W1 reference field in `StackExpertSample`.
Production wire validation and joint intake need an authenticated campaign artifact. See architecture debt D35 and D36.

## Explicit reference follow-up

Criteria 1–3 have CPU coverage for pooled pricing and refresh. The stack plan can bind `reference_q256` in the journal identity.
The refit keeps completed targets in peer fits. Older records still require exactly one census rung.
Criterion 4 has CPU accounting and journal coverage. Live cross-row refresh and authenticated joint intake remain open.
Criterion 5 has CPU cases for resume, old records, and byte equality. Flux `f3dNJt2YQw1` at `e8d02713a`: 90 passed; the eight byte cases (dense, solo, peers, five LFM subsets) have the same digests on main and on this head (Flux `f3dCiUPoqEX`); the order mutant, the absent-as-present mutant and main's source each fail the cases built against them.
The inventory found the defect in 0 of 28 readable tables. Four tables were dense. One named table was absent.
The 14 checkpoints without `cost.pkl` list zero packed units. The live campaign has no two-tier plan generator.
No producer writes `reference_q256` today: only the tests set it. The pinned path is dormant until a two-tier plan generator emits the field; without it the two-census state refuses as before.
