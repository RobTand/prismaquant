# Conditional interim depth admission (#2535 / #2774)

Adopted application and policy documentation changes from retained c46b833ea161cb3f6321baac42c0ffad4a4e6404, not the protected acceptance files or the historical review report. Added independent reviewed decision intake through `authorized_interim_record` and `--depth-price-authorization`. Interim admission checks attribution and exact numeric/source/scope/roster/contract bindings, retains owned decision provenance and remains `validated=false`, including production-purpose invocation. Research production admission still requires qualification. No synthetic value is a default or real authorization.

The actual attributable numeric decision remains missing under existing #2535/#2972 ownership. Population enforcement beyond the explicit source marker belongs to #3013/#3019. This change does not claim actual rollout, scientific success, GPU qualification, parent closure, or independent release-gate completion. W1 paired-trade tests and owned documentation blocks are unchanged. Current main resolves to the checkout's base 6d8288adc94b3c4b624c143ed372fb6513f32784; no later main changes were available to incorporate.

## Verification

- Compilation: `PYTHONPYCACHEPREFIX="$TMPDIR/pycache" python -m compileall -q prismaquant tests/test_interim_depth_record_admission.py tests/test_routed_depth_pricing.py tests/test_routed_depth_price_consumption.py` passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_interim_depth_record_admission.py tests/test_routed_depth_pricing.py tests/test_routed_depth_price_consumption.py tests/test_paired_rate_trade.py tests/test_priced_cell_basis.py tests/test_priced_cell_policy.py tests/test_canonical_quality_quantity.py tests/test_allocator_packed_group_units.py tests/test_allocator_measured_runtime_cli.py tests/test_tessera_expert_projection.py tests/test_tessera_per_unit_rates.py tests/test_architecture_doc.py tests/test_docs_staleness.py tests/test_docs_entrypoints.py tests/test_duplication_baseline.py -q --basetemp="$TMPDIR/pytest"`: **363 passed**, 14 Torch deprecation warnings, exit 0. Includes all O1–O5 specified cases. Initial acceptance run found three failures (missing documentation and two refusal-message mismatches); corrected before this final successful run.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"942166c11fd20e6153b06e00b904f4786c620fced94a012969ed5aef7fc1b865","size":1234}},"operation":"d7a21ad38d5b437fb916db5379ac97506e9a71e2a4bcb0447e0b7adf7a3ced8b","receipt":"7fba056ece20619fcecb116b8cdefb809dae5538e4164fcf766b8300636db063"}]

- `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_tessera_serving_pin.py -q --basetemp="$TMPDIR/pytest"`: **16 passed**, exit 0.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"fe87f63da05fde8015e34b9303c6fc8aad745f5c619ab32b88d0a00952bbc5a8","size":269}},"operation":"04e8e11c6317f92ad15ca67de171913db4a46f58f3766c4af4baa33b9cb176bc","receipt":"d0c79b31495e835f2a4391ac14d5551beb497d722cdcb9d9cd943b40d1a65fa4"}]

Final diff inspected; `git diff --check` passed. No protected tests changed, no timing telemetry altered, and no downloads, fitting, GPU measurements or campaigns performed. CPU checks establish software behavior only.
