# Digest census reconciliation for #1301

The root remains open. The retained `refresh_2540` census describes source
`30b6231df000726beda55fb6706e641a5cb2ccb0`. It records 420 primitive calls,
275 supplemental JSON calls and 203 mutable modules. Those counts describe
that refresh, rather than the current source or completed migrations.

PRI-225 resumed this root from main `05b62b47427` on 2026-10-27 UTC.
The separate #3061 program owns logical duplication outside this digest root.

## Existing delivery owners

The native issue tree and parked handoffs were checked on 2026-10-27 UTC.
The following routes remain canonical. Their open state supplies no completion
evidence. Follow their exact call manifests before editing a consumer.

| Contract | Native owner | Retained delivery or handoff |
| --- | --- | --- |
| Value and JSON identities | #2541 | Parent of #2646, #2648, #2649, #2650 and #2651 |
| Byte digests | #2646 | PR #2665, repair handoff PRI-83 |
| Package framing | #2648 | PR #3048, repair handoff PRI-108, supersedes #2664 |
| Seven recipe gaps | #2649 | Freeze #2660, recipes #2662, routing #2663 |
| Bounded reads and tensor identities | #2542 | Leaf contracts remain separate |
| Shipcard file hashing | #2635 | Preserve the error wrapper and read contract |
| Tensor identities | #2638 | PR #3022 / PRI-105 and PR #3047 / PRI-107 |
| Standalone deployment contracts | #2543 | Existing leaves #2626 through #2630 |
| Shared integration residuals | #3032, #3033, #3035 through #3038, #3040 through #3042 | Reconcile retained ledgers before a new implementation |
| Independent closure audit | #2544 | Required after integration, before closing #1301 |

PRI-225 serializes shared digest-owner, census and baseline changes. A parked
handoff is reused through its existing issue. It is not a second migration.
GGUF remains deprecated and Tessera implementation tasks remain parked under
the current authorization.

## Reproduce current observations

Run these commands through PrismaBuild from the source snapshot being audited:

```sh
python -m tools.reconcile_digest_census --output /path/to/report.json
python -m tools.reconcile_digest_census --check /path/to/report.json
```

The command reads `refresh_2540` and uses the existing inventory traversal.
Its primitive scope projection must equal `duplication_inventory.digest_sites()`.
Supplemental `json.dump` and JSON calls outside that population stay separate.
Excluded directories and static alias-resolution limits remain those of the
existing scanner. This command does not expand the duplication gate.

Each observation records its current location, source SHA-256, expression,
historical candidates and existing module-level epic. A new module routes to
the root for review. Matching uses the expression AST within the same family,
file and scope. It ignores formatting and line movement. Repeated expressions
remain ambiguous. A matching expression in a changed module requires review.

The report is an observation artifact, not a completion ledger. A historical
call absent from the current scan remains `not_observed_historical_ids`.
Absence can mean archival, deletion, rewriting, scanner limits or migration.
It does not establish any of those explanations. Historical classifications
and retained byte fixtures still need current caller and consumer evidence.

`--check` regenerates the observations and refuses drift without rewriting the
saved report. The check binds observed source and the historical refresh; it
does not establish byte equivalence, read boundaries, GPU qualification or
served qualification. Keep reports and fleet receipts with the task evidence.

## Remaining acceptance

Reconcile every current call with its native owner or an evidenced retention.
For each migrated call, integrate the independent original fixture and actual
consumer result, then update shared census and baseline records serially.
Do not treat an expression match or an absent primitive as that evidence.
The independent #2544 audit must verify complete coverage before #1301 closes.
