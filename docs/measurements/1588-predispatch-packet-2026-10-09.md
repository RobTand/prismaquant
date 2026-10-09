# #1588 pre-dispatch packet — 2026-10-09

This packet publishes what PrismaBuild dispatch needs before it starts.
It mints no GPU row and no price. Its machine-readable twin is
`1588-predispatch-packet-2026-10-09.json`. The test
`tests/test_1588_predispatch_packet.py` re-derives each claim.

## Source and pins

- PrismaQuant source: `30b6231df000726beda55fb6706e641a5cb2ccb0` (reference only).
- Tessera pin: commit `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`, contract `ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e`.
- Derivation ran through Tessera state `reader-pin-fca4c6ce0` (a pin).
- PB probe action: `1237dd52276dc7a1b354c653969f8ff16ee0006a5359412b48b088d9e12ece4e`.

## Requested rows and current state

All 11 unique rates are producer-legal with no hole refusals.
No requested row is priced in a joint pickle. Cell state follows.

| Row | Legal | Cell for the class |
|---|---|---|
| Routed BF16 R960, R1088, R1152 | Yes | No (dense only) |
| Routed E4M3 R768 | Yes | No (dense only) |
| Routed E4M3 R1152 | Yes | Yes (dense, routed_moe) |
| Routed E2M1 R640, R768 | Yes | No (no cell) |
| Dense BF16 R1152, R1408, R1792 | Yes | Yes (dense) |
| Dense E4M3 R1152, R1536, R2048 | Yes | Yes (dense; R1152 also qualifies routed_moe) |
| Dense E2M1 R768 | Yes | No (no cell) |

Cells carry route-only evidence with smoke not recorded.
They name eager execution under resident serve flags on sm_121.
Cells belong to Tessera #689. Absent cells never block honest prices.

## Revised estimate

- Cost rows: 145–190 GPU-h for encode plus joint AURA rows.
- The 30–41 GPU-h receipt half is gone under per-shape time.
- Shape-time rows: 140 bench cells, each seconds long.
- The dispatcher re-estimates before dispatch.
- Basis: `PACT-APPLICATION-DESIGN-2026-09-28` sections 3.1 and 3.2.

## Namespace

- Proposed root: `/mnt/shared/tessera-measurements/pq-stageb-1588-v2-20261001-bff19452/`.
- The root does not exist yet. This packet creates no directory.
- Keep the fresh-namespace decision. Never migrate the old namespace in place.
- Bind the reviewed head at dispatch, not the 2026-10-01 reference commit.

## Reconciliation

- The 102 legacy rows were not located from this box.
- The campaign lead reconciles at dispatch through the census.
- Never reuse a published path. Never rewrite a sealed request.
- Adopt seeds only through the existing content gates.

## Scope note

The 42 historical gamut rungs do not define the full legal domain.
See `docs/measurements/glm-full-domain-acquisition-2026-09-13.md:123-127`.
This packet covers the 14 requested rows only.
