# PQ2281 source qualification: paired rate trades (PQ #2531, Refs #2281)

Status: CPU qualification of the paired-trade source policy.
This record does not close the measured P0 #2281 defect.
It runs no new GPU arm, reprices no layer, fits no calibration,
and changes no source, teacher, format menu, or serving pin.

## Bound inputs

Base commit: `30b6231df000726beda55fb6706e641a5cb2ccb0`.
SHA-256 is over file bytes at that commit.

| File | SHA-256 |
| --- | --- |
| `prismaquant/joint_aura.py` | `a319df4e992ba8f826d7ba680b8ab515c5eb4bc364a7fde7bceb6cc0a9c15c31` |
| `prismaquant/allocator_candidates.py` | `0ab149c65acbe43fe7073a8102a1af68007b46ad387d187b3e3d85d0beda891c` |
| `prismaquant/cost_currency.py` | `c3fbf065d4a71393140067ea5bd5e428d927ec1832da32254bd6d1ce9aa05469` |
| `prismaquant/aura_additivity_gate.py` | `59337078c369a0d815bd01452b2a5baa1785299765a608f574567fd390eaa8cc` |
| `tests/test_paired_rate_trade.py` | `443a87cb3ec9e9f8357f481b2003dc7b03aed79f63d8980900ba1cf371328401` |
| `tests/test_joint_aura_assignment_diagnostics.py` | `42d9ae0b7a8972152e0cfb1fc4fcb8a614c41a0b7bd72c0077976284934b93c4` |
| `tests/test_joint_aura_allocator_currency.py` | `e3fc6e31d8df497524a4a67b4d28f6a0fbbbc8d95c7bd2fb3b76b9682e5c3a74` |
| `tests/test_aura_additivity_identity.py` | `5eed87ea707c988fbe07bbcdfbb7f5c97d2f6b9873531c3dfd8f6e89dcdb6036` |
| `tests/test_joint_sequence_attribution.py` | `c23097c0c233ef2228a6e8759349f80171ed7253609fee5456560152d5141f41` |

## Evidence table

Rank runs from strongest support to weakest. Every row states
its support and its limits.

| # | Evidence | Support | Limits | Receipt |
| --- | --- | --- | --- | --- |
| 1 | Noise: signed differences keep common-probe covariance; the paired standard error hedges the trade; zero z keeps the candidate sum bitwise. | 91 paired-trade tests plus 45 assignment-diagnostic tests assert exact arithmetic: common-noise cancellation, retained member correlation, unclipped negative differences, and bitwise zero-z identity. | Conditional on fixed calibration. The hedge is sampling uncertainty only. It estimates no generalization. | PB `bbb18047`, `6e29893a`: verified. |
| 2 | Instrument: streamed joint rows admit in one AURA currency with bound coordinates. | 21 allocator-currency tests use real streamed rows: one-currency admission, row-to-cell binding, probe-population alignment refusal, and alias-canonical price identity. | Research status. Rows need at least two probes. The fixture proves the consumer contract, not a new measurement. | PB `d81b1fa8`: verified. |
| 3 | Normalization: the global KL Fisher normalization lives inside the signed projections; paired pricing adds no gain, divisor, or rescale. | The paired suites assert the preserved `normalization` field and common-noise cancellation; the allocator-currency tests refuse a second gain and any activation-penalty reapplication. | Applies only to authenticated joint rows. Weight-only and output-MSE rows never enter this path. | PB `bbb18047`, `6e29893a`, `d81b1fa8`: verified. |
| 4 | Additivity: the gate reports the paired additive price; aligned rows use empirical covariance; bare arrays stay unverified; missing arrays assume independence. | 19 additivity-identity tests plus the paired objective tests assert finite-negative KL handling, render binding before sampling claims, and refusal of foreign payload coordinates. | The gate describes one assignment. It certifies no serving quality and no held-out dataset. | PB `dbed584b`: verified. |
| 5 | Reconstruction: the sidecar residual reconciles against authoritative totals and never moves the price; G3 hash agreement covers only export-versus-priced reconstruction. | 41 sequence-attribution tests assert residual gates, zero-mass exactness, and coverage without gaps or overlap. | The G3/export half was not exercised here. No G3 arm ran. No served KL is claimed. | PB `6800e717`: verified for the sidecar; the G3 scope limit is a stated boundary, not a new receipt. |

## Receipt ledger

All runs used `--tag x86` on `dl380g10` through `pbtest`.

| Action | Suite | Outcome |
| --- | --- | --- |
| `bbb1804709edd70dcaec2f580cf753385ccdc2caca2a0198a7a567e3906a5194` | `tests/test_paired_rate_trade.py` | 91 passed |
| `6e29893a1cdbcb42b6ace382417f8359898f86e612f2d9e22d5546a9a9d61e07` | `tests/test_joint_aura_assignment_diagnostics.py` | 45 passed |
| `d81b1fa8acb625ed40aef000027b8dc70acd0eee30a54425a097dd658a25d4e5` | `tests/test_joint_aura_allocator_currency.py` | 21 passed |
| `dbed584b4d0cb74a30aba9bb9f8d915c62b1feb97471ab83ffba5acac3c02b52` | `tests/test_aura_additivity_identity.py` | 19 passed |
| `6800e717f62c9952f7c679c41f18102f00c2afdb681272c816c34e1723c11dc4` | `tests/test_joint_sequence_attribution.py` | 41 passed |

Verified receipts are the five rows above. Unverified statements are
the G3 scope boundary and every parent-campaign claim in #2281:
this record neither verifies nor repeats them.

## Sample coordinates and assignments

Scalar oracles use synthetic signed samples under canonical probe
identities. They prove arithmetic, not measurement. The
allocator-currency suite uses one streamed measurement fixture and
binds each row to its own cell. Complete routed-layer rosters derive
from the model profile through `routed_expert_identity`, never from
an asserted list. The suites exercise subgroup retention,
incomplete rosters, the exact-half boundary with no tolerance,
cancellation, all-zero experts, unattributable baseline members,
and the final emission guard.

## Failed attempts

Attempt 1 of this child ended failed. It could not reach the parent
campaign ledgers and refused to invent counters. This record
preserves that receipt. This attempt ran five suites with zero
failures, so it adds no new failure receipt and repairs no code.
The landed #2282 repair and the #2288 subgroup correction stand
unchanged.

## Independent review counters

The parent records name independent review counters. This child
received none through a reachable path. It resets none, claims
none, and invents none.

## Qualification verdict

- Mismatched sample coordinates and currency refuse by name in both
  modes. Mathematical coordinates stay mandatory under dev mode.
- Signed differences, common-probe covariance, and the global KL
  Fisher normalization are preserved through pricing and hedging.
- The complete-layer dominance rule holds: subgroups reprice
  without pruning, exactly half is allowed, dominance and
  indeterminate cancellation refuse at the complete-assignment guard.
- Legacy pricing without a paired baseline and zero-z prices are
  bitwise unchanged outside the paired-trade scope.
- No qualification failure occurred, so no repair was made and no
  regression test was added beyond the existing suites.
