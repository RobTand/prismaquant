# GLM full-domain endpoint panel, 2026-09-13

Six scalar observations were acquired through PrismaBuild; the two BF16 R4096 bookends were withdrawn while unstarted and remain unknown. The complete legal domains are unchanged. Those expensive observations are deferred until a valid probe/allocation multiplier or other declared decision bound establishes their value. No zero price or runtime-dominance claim was substituted.

The paired units are dense `model.language_model.layers.0.mlp.down_proj` and routed `model.language_model.layers.3.mlp.experts.0.down_proj`. These are integration examples, not a representative or held-out population sample. Every row uses the existing canonical 512×512 calibration/capture/H and frozen Tessera d403cc5a producer. The currency is `output_mse_under_route_activation_contract`, not joint AURA or observed model KL.

| Unit | Family | Rate q256 | Output MSE | Profiled PB action seconds |
|---|---|---:|---:|---:|
| Dense L0 | TESSERA_E4M3_K1 | 256 | 2.34068829741e-06 | 127.37 |
| Dense L0 | TESSERA_E4M3_K1 | 2048 | 2.10617585594e-08 | 352.75 |
| Dense L0 | TESSERA_BF16_K1 | 256 | 2.33555147133e-06 | 168.03 |
| Routed L3 expert 0 | TESSERA_E4M3_K1 | 256 | 0.0128273665905 | 65.30 |
| Routed L3 expert 0 | TESSERA_E4M3_K1 | 2048 | 7.82225324656e-05 | 105.99 |
| Routed L3 expert 0 | TESSERA_BF16_K1 | 256 | 0.0128078358248 | 75.77 |

The routed measurements required the explicit exact-member research scope in PR #585 / issue #583. It validates the original 864-member group before acquiring the one named member, stamps that limited scope, and prohibits treating its output as a stack estimate or allocator table. No Fisher vector was fabricated.

## Reproducible evidence

Ledger: `/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/endpoints/endpoint-results-06.json`.

Ledger SHA256: `b6185ee9517ea75b16004fdfc37ae59467dc2eb955351b44aee820b396bd1336`.

The ledger records all six GPU action keys and CAS receipts, six CPU collector keys and CAS receipts, frozen plans, journal/wire/point-receipt hashes, values, cProfile files, full-action power summaries, and the two deferred withdrawal records. Coordinator verification independently checked terminal return codes, CAS payloads and every point/wire/profile hash without repeating any numerical encoding:

`/home/rob/dq-runs/glm-campaign-takeover-20260913/allocation/endpoints/coordinator-verified.json`.

Both boxes’ raw Netdata series are retained as `netdata-both-*.json` in that same local directory, collected from 23:43 UTC through the completed panel. Actual PB box profiles use pqteld power and host telemetry. The conservative native-backend overlap window 23:44:49.364–23:45:08.629 UTC belongs to an initial failed pre-encode attempt; no uncontaminated full-action performance claim is made for that attempt.

## Resource and performance scope

Initial 12 GiB reservations were refused by the checked selected-source phase plan before any anchor was encoded. The actual shared planner returned 38.204 GiB dense / 39.045 GiB routed deltas; adding the observed process floor supported the subsequent 48 GiB reservations. The failed profiles and sealed requests were retained. All successful actions were admitted as GB10 measurements with four CPUs and in-container cProfile; no completed quality observation was repeated.

The timings above include acquisition setup, encoding, scoring and publication under cProfile. They are not native serving latency or estimates of unprofiled runtime. cProfile can substantially amplify Python-call-heavy setup; no profiling-overhead correction was measured. Its accumulated function times can also overlap, so they are not converted to wall percentages.

For the dense E4 endpoints, the recorded encoder spans were 33.159 s at R256 and 269.430 s at R2048. cProfile recorded about 43.8 s in full legal-menu expansion in each action, including about 39 s in exact footprint enumeration; this identifies a setup hypothesis for a separate metadata-only comparison, not a demonstrated unprofiled bottleneck. The ledger’s approximate SoC joules use full-action mean power times wall duration and include startup/idle; they do not isolate encoder energy. GPU utilization is not used as a saturation diagnostic.

No full-model assignment, interpolation acceptance, prefill result, serving qualification or completed full-domain curve follows from this panel. The generic allocator remains dependent on its valid measured probe. The BF16 R4096 cells are still legal, unmeasured and deferred.
