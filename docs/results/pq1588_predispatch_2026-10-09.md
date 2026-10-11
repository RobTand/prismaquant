# #1588 pre-dispatch packet — 2026-10-09

This packet fixes dispatch inputs for the 14 open rungs of
RobTand/prismaquant#1588. It mints no cost row and starts no GPU work.
Its machine-readable twin is `pq1588_predispatch_2026-10-09.json`. The test
`tests/test_pq1588_predispatch_packet.py` re-derives each claim through the
repo APIs. This packet serves RobTand/prismaquant#2558.

## Source and pins

- PrismaQuant source: `30b6231df000726beda55fb6706e641a5cb2ccb0` (reference only).
- Short id `30b6231d` derives from the full source SHA above.
- Tessera pin: commit `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`, contract `ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e`.
- Derivation ran through Tessera state `reader-pin-fca4c6ce0` (a pin).

## Requested rows and current state

All 11 unique producer-legal rates have no price row in the joint pickle.
No requested row is priced. Cell state follows per row.

| Family | Rate | Class | Producer legal | Priced | Cell for class |
|---|---|---|---|---|---|
| TESSERA_BF16_K1 | 960 | routed | yes | no | no (dense only) |
| TESSERA_BF16_K1 | 1088 | routed | yes | no | no (dense only) |
| TESSERA_BF16_K1 | 1152 | routed | yes | no | no (dense only) |
| TESSERA_BF16_K1 | 1152 | dense | yes | no | yes (dense) |
| TESSERA_BF16_K1 | 1408 | dense | yes | no | yes (dense) |
| TESSERA_BF16_K1 | 1792 | dense | yes | no | yes (dense) |
| TESSERA_E4M3_K1 | 768 | routed | yes | no | no (dense only) |
| TESSERA_E4M3_K1 | 1152 | routed | yes | no | yes (dense, routed_moe) |
| TESSERA_E4M3_K1 | 1152 | dense | yes | no | yes (dense, routed_moe) |
| TESSERA_E4M3_K1 | 1536 | dense | yes | no | yes (dense) |
| TESSERA_E4M3_K1 | 2048 | dense | yes | no | yes (dense) |
| TESSERA_E2M1_K2 | 640 | routed | yes | no | no (no cell) |
| TESSERA_E2M1_K2 | 768 | routed | yes | no | no (no cell) |
| TESSERA_E2M1_K2 | 768 | dense | yes | no | no (no cell) |

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

- Proposed root: `/mnt/shared/tessera-measurements/pq-stageb-1588-v2-20261009-30b6231d/`.
- The root does not exist yet. This packet creates no directory.
- The short id in the root derives from the packet full source SHA.
- Keep the fresh-namespace decision. Never migrate the old namespace in place.
- Bind the reviewed head at dispatch, not the reference commit.

## Reconciliation

- The 102 legacy rows were not located from this box.
- The campaign lead reconciles at dispatch through the census.
- Never reuse a published path. Never rewrite a sealed request.
- Adopt seeds only through the existing content gates.

## Scope note

The 42 historical gamut rungs do not define the full legal domain.
See `docs/measurements/glm-full-domain-acquisition-2026-09-13.md:123-127`.
This packet covers the 14 requested rows only.
