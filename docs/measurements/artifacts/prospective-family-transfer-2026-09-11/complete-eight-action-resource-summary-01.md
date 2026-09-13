# L20 shared-down: terminal resource evidence

The complete set contains 8 curves and 579 durable rate-point receipts. Each of its 16 actions has a successful PB terminal, a CAS receipt, zero live terminal processes, and completed scope cleanup.

Power fields in the JSON are sampled box-window telemetry. The initial endpoint measurements overlapped, so they must not be summed into a study energy total. GPU utilization is intentionally omitted as a saturation claim. dl380g10 collector profiles record the absence of pqteld CSV coverage.

| role | measurement action | collector action | points |
| --- | --- | --- | ---: |
| bf16_full | `2b88c0760773` | `ad7cff841c16` | 257 |
| e4_endpoint_left | `2ba110537d2e` | `fbeebdc66606` | 1 |
| e4_interior | `54af7a7409cf` | `c0985ad92697` | 255 |
| e4_endpoint_right | `d9b371d1ae83` | `d15421cab3ba` | 1 |
| e2_endpoint_left | `4bd3e07c44f7` | `c16395c6276f` | 1 |
| e2_interior | `0eb5ff1625a9` | `21ceeb26e911` | 62 |
| e2_endpoint_right | `8e5c87ddba0e` | `e2529e98b766` | 1 |
| e2_terminal | `505a95f6a747` | `7d84d6c6e451` | 1 |

Canonical machine-readable ledger: `complete-eight-action-curve-manifest-01.json`.
Terminal resource fields: `complete-eight-action-resource-summary-01.json`.
