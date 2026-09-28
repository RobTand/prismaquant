# Tessera route-trace fixtures, identity schema v1 (#509)

These files are what **Tessera's own serving telemetry wrote**. None of the
fields here were authored by PrismaQuant: the consumer under test
(`prismaquant/tessera_route_trace_gate.py`) is checked against the producer's
output, so a change to the producer's schema must fail this repo's gate rather
than be absorbed by a fixture that drifted with it.

| | |
|---|---|
| producer tree | `tessera-509-route-trace-identity` |
| producer commit | `8104dc6` — "route trace: the header reads the latched platform, never probes for one" |
| identity version | `1` (`tessera.serving.telemetry.IDENTITY_VERSION`) |
| platform stamped | `sm_121` |
| topology | `world_size = 2`, one file per rank, two token counts (M1, M178) |
| artifact priced against | `../tessera_route_trace_m44e1/config.json` (same five modules) |

Written by `generate.py`, which imports
`tessera.serving.telemetry` from the producer tree, says what that process
would have observed (rank `n` of 2, platform `sm_121`) and flushes through the
producer's own `_RouteTrace`:

```
python3 \
  tests/fixtures/tessera_route_trace_509/generate.py \
  --producer-src /home/rob/tmp/tessera-509-route-trace-identity/src \
  --producer-commit 8104dc6 \
  --out-dir tests/fixtures/tessera_route_trace_509
```

| file | serve |
|---|---|
| `routes-rank0.json`, `routes-rank1.json` | every module rides the contract its price names |
| `routes-swapped-rank0.json`, `routes-swapped-rank1.json` | layer 0's BF16 MLP and layer 1's NVFP4 shared-expert down projection ride each other's contract, leaving every count in the histogram identical |

Regenerating rewrites `pid`, `started_utc`, `flushed_utc` and `flushes`; the
`entries` and the identity header are deterministic.

## Module names are the serve's, not the checkpoint's (PQ #1490)

Regenerated 2026-09-28 at the same producer commit (`8104dc6`) with the
system `python3` (3.12; the producer's telemetry module is stdlib-only, and
the venv named above no longer exists). The only change to `entries` is the
module names. Each stand-in layer's `prefix` was the checkpoint target
(`model.language_model.layers.N.…`), which no GLM serve records: vLLM builds
the body under `language_model.model.layers.N.…`, and Tessera records the
layer's `prefix`. The measured U4 BAL TP2 serves (2026-09-28) confirm it.
`generate.py` now applies that rule (`served_prefix`) before it emits, and
the gate maps the m44e1 config's targets through the `glm5_next` profile to
compare them. Before this change the fixture agreed only because both sides
used the same wrong namespace.
