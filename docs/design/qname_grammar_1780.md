# Roster qname grammar ownership (#1780, #1303 slice)

`prismaquant.qnames.LAYER_QNAME` owns the exact legacy pattern
`^.*\.layers\.(\d+)(?:\.|$)`. Both `joint_layer_quanta` and
`joint_cost_read_schedule` import it under their existing private aliases.
Only ownership moved: each `.match`, group conversion, exact-string check,
roster grouping, comparison and caller-specific refusal remains unchanged.

The grammar is deliberately greedy, accepts Unicode decimal digits and
requires a dot before `layers`. It is not a generic model-name parser;
other layer-name regexes must not be normalized into it without equivalence
proof. The owner imports only the standard library and adds no torch load.

`tests/test_qname_grammar_1780.py` proves source-level owner imports as well
as literal old-pattern flags/group/span identities and caller type policies.
Identity alone would not prove delegation because `re.compile` caches equal
patterns. Routing RED must therefore fail on the missing owner imports.
Existing COST read-schedule and quantum/partition suites verify the consumers.

On fresh main the two remaining exact literals are cut over too (#2330):
`tools/compare_joint_layer_gate.py` selects layers through
`DOTTED_LAYER_QNAME` (first complete dotted component, both dots required)
and `tools/derive_retained_window_budget.py` groups its roster through
`LAYER_QNAME` (greedy final component, Unicode digits) at its existing
post-argparse import boundary, which keeps `--help` stdlib-only. Only
ownership moved: selection, filters, refusals, grouping and outputs are
unchanged, and the original and candidate CLI bodies were run
parity-identical on shared tiny journals and a real retained-budget
fixture through PrismaBuild.

This slice changes no pipeline defaults, serving metrics, bytes, or ship gate.
It does not close the broader #1303 domain-numerics census. No performance,
GPU, or served A/B measurement is claimed.
