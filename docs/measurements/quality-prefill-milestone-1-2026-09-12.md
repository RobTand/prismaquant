# Quality–prefill experiment, milestone 1: inventory, schemas and dependency report

Date: 2026-09-12. Spec: `docs/design/tessera_quality_prefill_experiment.md` (merged
to main in PR #506 at `395770ddeb`). Issue #505 closed; #237 remains the
experiment/generalization issue.

Everything below is a **source derivation or a code-derived count**. No encode,
decode, serve or GPU measurement was performed for this milestone, and no
quality, speed or size claim is made.

## 1. What the source says

Derived through the reader pin `387eda36fd410d6b2a4fb86b22285eab2a5e072c`
(contract sha256 `a688f8de244f936e…`), read as bytes from a read-only
`git archive` extraction, never through the editable working checkout.

| family | domain | legal rates | holes | table width | resolver transitions |
|---|---|---|---|---|---|
| `TESSERA_E4M3_K1` | R256–R2048 | **1,793** | 0 | L14 throughout | 14 |
| `TESSERA_BF16_K1` | R256–R4096 | **3,841** | 0 | L14, L15 at R3585, L16 at R3841 | 30 |

Both counts agree with the independent source audit
(`tessera-legal-domain-source-audit-01.md`, sha256 `b2ec2c6be76c6c0c…`).

**The two pins are one source for these families.** Closed from bytes, not from
prose: `grammar.py` — which sets the endpoints through its rate-range and
whole-unit-quota refusals — is byte-identical at the reader pin, the producer
pin `d403cc5a3199a348…` and the working checkout's HEAD. `_window_bits_for`,
`wire_recipe`, the WINDOW raw-cap expression, the `*_WINDOW_BITS` constants,
`runtime_contract.json` and `calculator.py` are byte-identical between the two
pins. The pins' `export.py` differ only by the additive `ScalePlaneKind.MX`
plane (tessera#443), which no WINDOW-over-CHANNEL rung reaches. The
producer/reader divergence recorded elsewhere does not bite this domain.

**256-multiples are resolver transitions, and so are 256k+1.** The Bresenham
column schedule is uniform at per-column rate `k` when `R = 256k` over a column
count divisible by 256, and mixed `{k, k+1}` at `R = 256k+1` — identically at
2048, 4096 and 12288 columns. The table-width transitions (R3585→L15,
R3841→L16) are a subset of the 256k+1 group.

**R4096 is not source-BF16 passthrough.** It is a WINDOW/CHANNEL representation,
flagged as such, and costs 16.13 bpw on (2048, 4096) rather than 16.0 because of
its 128 KiB alphabet table. No BF16 candidate at any rate claims passthrough.

## 2. Support is five facts, and four of them are nearly empty

Per candidate the ledger records `producer_legal`, `reader_supported`,
`implemented_route`, `native_qualification` and `export_and_served_validation`
as five independent fields. They disagree, which is the point.

- **Native qualification is 4 triples** against 5,634 producer-legal candidates:
  E4M3 R1024 dense, E4M3 R1024 routed-MoE, BF16 R1792 dense, E2M1 R896 dense.
- **Routed-MoE attestation is runtime-image-bound.** E4M3 R1024 routed-MoE reads
  *unattested* when asked with the contract's `default_serve_image`; the routed
  cells declare their own image and `eager`-only execution. Native membership is
  not image-agnostic.
- **`export_and_served_validation` is false for every E4M3 and BF16 candidate.**
  The only artifact evidence at the pin is a Qwen3-0.6B artifact, out of
  GLM-5.3-Flash scope. Recorded in `detail`, never in the value, per principle
  14's scope corollary.

Singleton attestation does not shrink the research domain: the legal count is
unaffected by the native set, and a test asserts it.

## 3. Dry-run acquisition size

Real counts from `build_mandatory_rate_set` over the derived domains, screen
policy `coverage_first_v1`, `selection_seed = 0`:

| family | legal | mandatory | roster | strata |
|---|---|---|---|---|
| `TESSERA_E4M3_K1` | 1,793 | 29 | 141 | 14 |
| `TESSERA_BF16_K1` | 3,841 | 61 | 301 | 30 |
| **total** | **5,634** | **90** | **442** | 44 |

**442 rates, 7.8% of the legal domain.** Population side: six strata over 42 MoE
layers, 126 shared units and 36,288 routed units, with deterministic
pilot/confirmation thirds and an exposure ledger.

No GPU-hour or wall-clock estimate is given: the per-rate encode cost at these
shapes has not been measured on this hardware, and an unmeasured cost estimate
is not evidence.

## 4. Blocked, and on whom

| blocker | state | what it stops |
|---|---|---|
| prismaquant #420 | open | `runtime_provenance.admit_fixed_resources` refuses **every** full-engine v2 partition — correctly; no recomputable partition schema is implemented. Stops all runtime pricing, allocation and frontier work. |
| prismaquant PR #503 | open, unmerged, draft | The scalar-transfer dependency the spec names at `a115424b60`. Its `experiments/sparse_rate_family_transfer.py` is the extension point for the prospective-transfer audit. Not merged as a side effect of this work. |
| prismabuild PR #518 | open, draft, **not deployed** | The deployed generation carries no `decomposition.py` and no `decompose` entry in `pbcampaign.py`. Publishing a runtime generation needs Rob's explicit word and an idle queue. The PQ→PB adapter therefore refuses by name rather than falling back, and is exercised in-process only. |
| prismabuild #499 | open; prewarm loop **paused** | `kill -STOP 1107503` on dl380g10 must be `kill -CONT`'d before any runtime republish. |

**Three inputs are owed by Rob and are not guessable.** `freeze` refuses and
names all three rather than defaulting: `allocation.byte_budgets`,
`allocation.prefill_budget_sweep`, `runtime.workload_manifests`.

## 5. Known debt in what shipped

- **Two `RateDomain` dataclasses.** `tessera_legal_domain.RateDomain` and
  `quality_prefill_population.RateDomain` are structurally identical and
  distinct; the join currently needs a field-for-field conversion. Converge them.
- **Two strict readers.** `quality_prefill_contract` and
  `quality_prefill_pb_adapter` each carry one, because `prismaquant/schemas.py`
  is deliberately an *open* mapping ("older artifacts with extra fields still
  load") while a document hashed into a sealed action key needs closed key sets
  and one canonical byte spelling. They meet at `SchemaValidationError`.
  Converging them is a follow-up, not a completed item.
- **The L20 shared-down confirmation exclusion is vacuous at the real N.** With
  42 eligible layers, shared-down draws confirmation from a third that does not
  contain L20. The guard is implemented and tested; only the L10 half was
  observed to fire.
- **PrismaBuild publishes no capability token for #517.** Support is probed as
  the two artifacts it is made of. A token would be the honest mechanism.
- The adapter harness **skips loudly** when the #518 worktree is absent, so its
  cover criteria are checkout-dependent.

## 6. Evidence

All test runs through PrismaBuild at `--priority -10` on sparky, CPU-only.

| package | action key | result |
|---|---|---|
| legal inventory + support ledger | `3a73224ff7c9` | 79 passed, 1 skipped |
| manifest schema + driver | `b2b6bbd9e34b` | 78 passed |
| population selection | `f61cb45a99d9` | 52 passed |
| PQ→PB adapter | `a1ab8f15a7dd` | 62 passed |
| canonicalizer regression (20 callers in main) | `70c17d0a9cab` | 490 passed, 1 pre-existing failure, 3 skipped |
| that failure on clean main, unmodified | `609e77a2808d` | 1 failed, 10 passed — pre-existing |
| spec merge, docs currency | `391c99ea4b4c` | 19 passed |
