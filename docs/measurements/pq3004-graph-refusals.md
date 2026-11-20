# PQ #3004 graph receipt population checks

The issue has one acceptance item that covers the O1 and O4 refusal cases.

> Remaining O1/O4 completeness requires producer-owned contract resolution. Installed Tessera aaa41700 has no completeness-validation API and its verifier accepts incomplete integer populations. Served Tessera main already contains the authoritative shared population gate, but using or activating that newer implementation is not authorized here. The existing producer owner must resolve compatible delivery through the verifier contract; this consumer must not copy its arithmetic. Current-consumer failure is retained in operation abadc449dd45e96e235d53e7cbc4eb18b2792c3a8750d9150b6aa4d9e2656d68.

| Check | `origin/main` at `7a58d7ef4` | Evidence in this change |
| --- | --- | --- |
| O1. Refuse a graph pass with short integer counts. | PARTLY. `tests/test_graph_shipcard_numerical.py::test_shared_rule_rederives_numerical_and_replay_refusals[incomplete_population]:64` removes a pass. `[forged_equal]` reduces membership only. | `tests/test_graph_shipcard_numerical.py::test_shared_rule_refuses_truncated_population[graph]:25` reduces both counts and checks the population refusal. |
| O4. Refuse a truncated eager population. | NOT COVERED. | `tests/test_graph_shipcard_numerical.py::test_shared_rule_refuses_truncated_population[eager]:25` reduces both eager repeat counts and checks the population refusal. |
| Deliver the producer check without copying its arithmetic. | COVERED. `tests/test_graph_shipcard_numerical.py::test_real_shared_rule_accepts_complete_synthetic_record:19` uses Tessera. `tests/test_graph_shipcard_receipt.py::test_missing_compatible_verifier_refuses:145` checks refusal without the producer API. | The tests use the same producer API. |
| Resolve the missing API in the old `aaa41700` install. | MOOT for the current pin. `c503b16e1` selects Tessera `e89757b9`, which supplies `tessera.graph_receipt.population_refusal`. | The old operation remains historical evidence. |

Each test preserves the original `equal` verdict and updates both receipt digests.
In Tessera `graph_receipt.py`, replace `for arm in eager + graph:` with
`for arm in eager:` for `[graph]`, or `for arm in graph:` for `[eager]`.
The graph mutation returns a verdict refusal instead of a population refusal.
The eager mutation accepts the receipt.

The lead must run the baseline and mutation checks in Flux with pinned Tessera.
Require zero skipped nodes. Use one CPU worker and a 1 GiB memory budget.
The test and mutation results are pending. These checks require no GPU or model checkpoint.
