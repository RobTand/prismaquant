# Conditional interim depth admission (#2535 / #2774)

Adopted application and policy documentation changes from retained c46b833ea161cb3f6321baac42c0ffad4a4e6404, not the protected acceptance files or the historical review report. Added independent reviewed decision intake through `authorized_interim_record` and `--depth-price-authorization`. Interim admission checks attribution and exact numeric/source/scope/roster/contract bindings, retains owned decision provenance and remains `validated=false`, including production-purpose invocation. Research production admission still requires qualification. No synthetic value is a default or real authorization.

The actual attributable numeric decision remains missing under existing #2535/#2972 ownership. Population enforcement beyond the explicit source marker belongs to #3013/#3019. This change does not claim actual rollout, scientific success, GPU qualification, parent closure, or independent release-gate completion. W1 paired-trade tests and owned documentation blocks are unchanged. Current main resolves to the checkout's base 6d8288adc94b3c4b624c143ed372fb6513f32784; no later main changes were available to incorporate.

## Initial implementation verification (before review repair)

- Compilation: `PYTHONPYCACHEPREFIX="$TMPDIR/pycache" python -m compileall -q prismaquant tests/test_interim_depth_record_admission.py tests/test_routed_depth_pricing.py tests/test_routed_depth_price_consumption.py` passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_interim_depth_record_admission.py tests/test_routed_depth_pricing.py tests/test_routed_depth_price_consumption.py tests/test_paired_rate_trade.py tests/test_priced_cell_basis.py tests/test_priced_cell_policy.py tests/test_canonical_quality_quantity.py tests/test_allocator_packed_group_units.py tests/test_allocator_measured_runtime_cli.py tests/test_tessera_expert_projection.py tests/test_tessera_per_unit_rates.py tests/test_architecture_doc.py tests/test_docs_staleness.py tests/test_docs_entrypoints.py tests/test_duplication_baseline.py -q --basetemp="$TMPDIR/pytest"`: **363 passed**, 14 Torch deprecation warnings, exit 0. Includes all O1–O5 specified cases. Initial acceptance run found three failures (missing documentation and two refusal-message mismatches); corrected before this final successful run.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"942166c11fd20e6153b06e00b904f4786c620fced94a012969ed5aef7fc1b865","size":1234}},"operation":"d7a21ad38d5b437fb916db5379ac97506e9a71e2a4bcb0447e0b7adf7a3ced8b","receipt":"7fba056ece20619fcecb116b8cdefb809dae5538e4164fcf766b8300636db063"}]

- `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_tessera_serving_pin.py -q --basetemp="$TMPDIR/pytest"`: **16 passed**, exit 0.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"fe87f63da05fde8015e34b9303c6fc8aad745f5c619ab32b88d0a00952bbc5a8","size":269}},"operation":"04e8e11c6317f92ad15ca67de171913db4a46f58f3766c4af4baa33b9cb176bc","receipt":"d0c79b31495e835f2a4391ac14d5551beb497d722cdcb9d9cd943b40d1a65fa4"}]

## Review repair: exact decision values and enclosing coordinates

Rechecked the two reported bypasses on the retained implementation. Added small regressions in `tests/test_depth_admission_binding_regressions.py`, reusing the existing synthetic fixtures without changing protected acceptance files. Before repair, `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_depth_admission_binding_regressions.py -q --tb=short --basetemp="$TMPDIR/pytest"` returned **26 failed, 2 passed**, exit 1. Failures reproduced boolean/int decision equality and non-joint/relocated per-unit rows. This corrects the initial report's assumption that the passing supplied checks established full consumption binding.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"848e88bb3976c9d8eefb01626eb3103a712d532cb3bf31a480885956b01e9ded","size":15377}},"operation":"b0d2ab729b794667d3724774b3fab1ff33acc11cde14569d1b0bfdb5874a4b5c","receipt":"c6d073ff207efc1e4c4f2c93e18d98ec1159603373d17e245fb97f89d9393818"}]

The repair independently validates decision parameters with the same numerical validator as profiles, recomputes the decision's own roster/parameter digests, and compares scope/bands/parameters through type-sensitive canonical JSON digests. Per-unit selection passes its actual unit and format into `loss_multiplier`, which rejects missing joint identity at admitted units and mismatched coordinates before passthrough or scaling. Existing frozen scientific-field checks remain in force. Tests cover baseline/candidate replacements, unit/format relocation, both scaled and multiplier-1.0 bands, all three admission states, and unchanged unrelated controls. The paired consumer already checks the enclosing coordinate through `joint_row_binds_cell`; no new allocator or population guard was added.

Final command:

```sh
PYTHONPYCACHEPREFIX="$TMPDIR/pycache" python -m compileall -q prismaquant tests/test_interim_depth_record_admission.py tests/test_routed_depth_pricing.py tests/test_routed_depth_price_consumption.py tests/test_depth_admission_binding_regressions.py && PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_depth_admission_binding_regressions.py tests/test_interim_depth_record_admission.py tests/test_routed_depth_pricing.py tests/test_routed_depth_price_consumption.py tests/test_paired_rate_trade.py tests/test_priced_cell_basis.py tests/test_priced_cell_policy.py tests/test_canonical_quality_quantity.py tests/test_allocator_packed_group_units.py tests/test_allocator_measured_runtime_cli.py tests/test_tessera_expert_projection.py tests/test_tessera_per_unit_rates.py tests/test_architecture_doc.py tests/test_docs_staleness.py tests/test_docs_entrypoints.py tests/test_duplication_baseline.py tests/test_tessera_serving_pin.py -q --basetemp="$TMPDIR/pytest"
```

Compilation passed; **407 tests passed**, 14 Torch deprecation warnings, exit 0. This includes all specified O1–O5 cases and 28 added binding regressions. Retained stdout was read back and verified.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"40c3d174ac0cf90b63ac7bd79a1a98be626335142ae8f687432d61b0ee4673f8","size":1234}},"operation":"f08bcce85bf8dbad7cbc2bf7c011a991c963e00eeb360ad3fd864dac4f76089c","receipt":"ca5f39d7b2c3a21c5f0c371db5907e31be6e94172126911b04ebfedb0097f518"}]

Final diff inspected; `git diff --check` passed. No protected tests changed, no timing telemetry altered, and no downloads, fitting, GPU measurements or campaigns performed. CPU checks establish software behavior only. Actual decision recovery and population enforcement retain the existing owners described above.
