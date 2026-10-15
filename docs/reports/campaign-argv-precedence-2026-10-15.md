# Campaign argument agreement

The actual campaign parser and planning metadata disagree on repeated and
equals-form options at source `60e749c5dc`. Execution takes the last value;
`campaign_argv_argument` takes the first space-form value and misses equals-form
values. The readset prewarm helper uses the last space/equal value.

| Actual arguments | Planner value | Execution value |
| --- | --- | --- |
| `--row-head load-all --row-head stream` | `load-all` | `stream` |
| `--row-head=load-all` | default `stream` | `load-all` |
| `--publication-overlap-bytes 0 --publication-overlap-bytes 4096` | `0` | `4096` |

The dispatcher uses these helpers for actual admission, head dependencies and
expert partitioning. Prewarm's actual consumers in the inspected readset owner
read seed/acquisition options. Other generic helper comparisons characterize
syntax only; they do not establish a source-digest planning defect without a
corresponding caller.

The witness invokes actual `campaign.main`, calls the real argparse parser,
then stops with a sentinel immediately after parsing. It loads no model/data,
and does not qualify post-parse validation, complete campaigns, GPU execution,
export, serving or performance. Eleven actual parser cases expose ten
planner/execution disagreements. The original sealed witness is unchanged.

PB `35f2bf9e138305b334d78c33916686d56bea2cdb8ddd4ddc8d4db98dba0aa38d`
exits zero because it verifies the observed baseline discrepancy. Its result
is `cfdae12abf075e07c68f85e81dce9432f7ba4b423db40e58182b475ba81f13d0`,
receipt `9cbf102d1f99a0b3a4958aa466459da8f3c0d0bfa08a5e3c209095fec96b8660`.
The separate agreement requirement correctly fails in PB
`0917654ec44277d4d2784d24a9d9e02ad19b216a176645ade481ab8ad06f83a9`,
exit one with all eleven cases and ten disagreements. It has no passing CAS
receipt. Terminal status and stored logs were verified.

Rerunnable witness and complete evidence are under
`/home/rob/tmp/pq-refactor-20261014/reports/argv_precedence_contract.py`,
`argv-precedence-baseline-evidence.json` and
`argv-precedence-agreement-red-evidence.json`. The agreement mode must pass
following the separate behavior correction. Structural request extraction first
preserves original CLI syntax/defaults and the existing settings wire codec.
The correction must share the actual parser's effective values and explicit
option presence rather than introducing another independent tokenizer.
