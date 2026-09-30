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

This slice changes no pipeline defaults, serving metrics, bytes, or ship gate.
It does not close the broader #1303 domain-numerics census. No performance,
GPU, or served A/B measurement is claimed.
