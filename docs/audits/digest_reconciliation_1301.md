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

## Retained source observations and validation

The 2026-10-27 UTC fleet run observed 378 primitive calls in 294 scopes,
including 277 scopes outside the two digest owners, plus 300 supplemental
JSON calls. It found 201 source-identical observations, 372 expression-only
matches, 42 ambiguous observations and 63 new or changed expressions.
Eighty historical call IDs were not observed. These are source observations;
none is a migration or retention verdict.

The implementation and tests are those committed at `90947646f9d79e37805b9bd95481ecf4deb880d7`.
This validation appendix was added afterwards. The source bundles were read
back and compared with the delivered files. Terminal log hashes, receipt
bindings and result payload hashes were checked. Full worker attestation was
not verified.

| Coverage | PrismaBuild action | Outcome |
| --- | --- | --- |
| Before implementation | `ebfc53c67fd20f43e5e6f30268623f09f5d7d791553b5320672f74f8898d46f4` | Eight expected missing-command failures |
| Reconciliation mutations and CLI | `9037022363f9cbf93cad947d09f2cc49b65616f4889defb6a9d4ef9b82af1df9` | 12 passed |
| Existing value profiles | `b22572ac84d0cb0b41950bbb628aa3a2031f0a80c08ecf9dd1e4144996d41521` | 89 passed |
| Existing file profiles | `85d04954b7854d365eb46093e11ac812550beef2340fa8ba822bd019a64d82c5` | 45 passed |
| Duplication gate | `5903718e360c6168702908e95a08506366659ea22f8f2fa92f437e8d87cdf7e9` | Seven passed |
| Actual source report, regeneration check and compile checks | `fdbb7b1fea95bbd527f15906bb2299c92dfcc499899b0fe585b035b0b555bb1c` | Passed on dl380g10 |

The four passing test shards ran under transient unit `pb-pri225-green-v2`:

```sh
python3 /mnt/shared/prismabuild-fleet/repo/tools/fleet/pbtest.py \
  --checkout /home/rob/paperclip-worktrees/prismaquant/PRI-225 \
  --python /home/rob/venvs/pq-pin-aaa41700-pb20261016-git/bin/python \
  --cpus-per-shard 1 --threads-per-shard 1 --mem-gb 4 --timeout-s 300 \
  --json /mnt/shared/paperclip-runs/PRI-225/green-v2.json \
  tests/test_digest_census_reconciliation.py tests/test_digest_profiles_1301.py \
  tests/test_digest_profiles_1301_files.py tests/test_duplication_baseline.py
```

The report and evidence remain under `/mnt/shared/paperclip-runs/PRI-225/`:
`current-source-v1.json`, `green-v2.json`, `green-endings.json`,
`verified-evidence.json` and `claim-verification.jsonl`. The verification record
contains exact terminal log paths and source-bundle identities. Earlier
`red.json` and `red-pq.json` runs refused before pytest because those
interpreters lacked the pinned PrismaBuild distribution. They do not count
as regression evidence. No GPU, serving or performance result is claimed.
