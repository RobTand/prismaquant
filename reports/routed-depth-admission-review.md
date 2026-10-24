# Routed-depth admission implementation (#2535 / #2774)

Adopted the implementation/docs from 5a87ec83f875aa894edc2b234f672359b7f19a8d, not its historical acceptance tests. The supplied corrected acceptance files remain unchanged. Added immutable per-coordinate scientific field digests, consumption checks for replacement measurements, complete scientific joint-price transformation preserving existing hedge semantics, and finite effective contribution/price/paired-statistic checks. Renamed the admission mapping helper instead of relaxing the duplication baseline.

## Decision and scope limits

O1 remains unmet: no authentic attributable numeric flat decision with coefficients, model/lane/all-rates population, provenance and authorization digest is available in the supplied record or adopted implementation. Existing #2535/#2972 ownership must recover or obtain it; this delivery does not create another blocker or invent values. `flat` and `interim_flat` admission remain refused, including research use. CPU synthetic factors are not authorization.

The portable explicit `provenance.depth_price_scope` declares software scope. Unscoped stock/uniform/unrelated inputs retain legacy behavior. The actual authorized population is still unknown: this declaration cannot establish coverage of that population, and a missing marker must not be treated as a rollout exemption once that decision is recovered. The source roster is checked against source-table expert keys, not an independently recovered authoritative model manifest. Full interim rollout and exact authorized-value verification cannot be claimed.

Receipt checks establish recorded software consistency, not authenticity, actual independent measurement, scientific success, export or serving authority. No fitting, GPU qualification, campaign execution, serving pin or allocation-policy rewrite was performed.

## Final verification

Executed the full O7 command: compileall for `prismaquant` and both protected acceptance files, followed by pytest covering both acceptance files, paired-rate trade, priced-cell basis/policy, canonical quantities, packed groups, allocator CLI, architecture, documentation staleness/entrypoints, duplication, expert projection, per-unit rates and serving-pin guards, with `--basetemp="$TMPDIR/pytest"` and bytecode outside the checkout.

Final observed result: **335 passed**, 14 existing Torch/Python deprecation warnings, exit 0. Compilation passed. An earlier run of the same command passed 331 with four skips; the final run reported no skips. These are CPU software checks, not GPU evidence. `git diff --check` also passed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"7bd90875b800f400872ffdc19d331c729dbc2a478bdc9d8d69065ebf4fc5bb8a","size":1154}},"operation":"b374a013214590b15d09089212fc79a8ba1e79562fca0992adba15be5f6579fc","receipt":"7750613cde977c35072894ad8ce9f1907e04197330c07088ed12ad13d0fce49a"}]
