# Stage B resource descriptor reuse: CPU profile

## Scope

PQ #1883 is a CPU preparation follow-up to #1367. `derive_policy` previously synthesized the same Tessera descriptor once per unit. The change resolves each distinct name once per invocation across layers. Per-unit shapes, activation receipts, grouping, ordering, policy bytes and refusals remain unchanged. A later invocation resolves again; no process-global registry or admission memo is introduced.

This is one instrumented pair over the complete bound production resource policy, not a GPU measurement or full metadata-producer benchmark. It does not establish the original #1367 checkpoint-write, 22.6-second, GPU-overlap, residency or whole-row acceptance.

## Reproduction

Both actions ran on admitted dl380g10 CPU workers through published `pbrun.py`, priority -10, CPU 8, memory 32 GiB, native threads 1, 450-second outer deadline. The prior complete preparation run peaked at about 20 GB; the conservative reservation was retained. These narrower actions peaked at 3.81/3.80 GB, so future repetitions can reserve less.

The standalone operation was `verify_policy(binding)` over this unchanged input:

- Path: `/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/ws-sb4-1151/policy/stageb-resource-policy.r13-capture-b4-gpu68-chain.json`
- SHA-256: `53be97d071be23c20a571bdf51a99fba8e08993c7bfdb0aaea5850c6fc308a62`

Both arms used the same checkout and dependencies. The before arm loaded the exact module from refactor-only commit `40ea1c05b382fbd22f3ab0433f3831461e7d5cc2` as a sealed benchmark input, SHA-256 `48b913f5d969ecc16eb1ce1336aea53c55e039b06d25284dc459dc4623ccea87`. The after arm used `5aa69f3f1a73724e2d207f01e5b87b55c8a270fc`. This isolates the resource module without changing production runtime source gates or input bindings. It also avoids a baseline-worktree snapshot that exceeded the PB client's Git time limits before publication.

The instrument was cProfile in the operation's child process, with both Spark Netdata series collected by the existing `workspace_netdata` sampler. The benchmark driver and before-module fixture were sealed, untracked experiment inputs, not shipped source. The input policy was independently rederived, not read from the process-global verified-policy memo. Exact Tessera b40 package/RECORD preflight preceded each operation.

## Observations

| Instrumented observation | Before | After |
| --- | ---: | ---: |
| Operation wall seconds | 139.588 | 47.769 |
| `get_format` calls | 1,097,726 | 79 |
| Tessera synthesis calls | 234,278 | 19 |
| `get_format` inclusive seconds | 93.663 | 1.193 |
| Tessera synthesis inclusive seconds | 85.710 | 0.627 |
| `_resource_specs_for_layer` inclusive seconds | 94.126 | 1.239 |
| PB scope CPU seconds | 148.379 | 56.274 |
| PB scope peak bytes | 3,813,847,040 | 3,802,267,648 |
| Worker Netdata CPU mean percent | 49.921 | 39.673 |

Inclusive times overlap and must not be summed. The single cProfile-instrumented wall-time ratio is 2.922. Before ran first; input/cache warmth and differing box load are confounders. cProfile overhead scales with the number of calls, so this is not an uninstrumented throughput estimate. The changed call path and exact result equivalence are stronger evidence than extrapolating that ratio to full preparation or GPU work.

Both arms returned byte-identical canonical policy JSON: 96,380,514 bytes, SHA-256 `85240d905a02f344e6e82ba6fb7b78f445d37abaa3cc414c958bdc2c015b429a`.

Both-box environmental series were retained:

| Series | Samples before/after | CPU mean percent before/after | GPU power W mean before/after |
| --- | ---: | ---: | ---: |
| sparky | 73 / 28 | 22.510 / 22.849 | 23.110 / 19.143 |
| sparklina | 73 / 28 | 14.537 / 13.271 | 32.370 / 46.786 |

These are environmental observations while the measured operation ran on x86, not GPU performance or work-per-joule results. GPU utilization was not used to infer work. Before power ranges were 13–51 W / 12–50 W; after ranges were 14–37 W / 15–53 W. Netdata intervals were 1790847333.246–1790847478.280 and 1790847531.258–1790847585.663 Unix seconds.

## Evidence

- Before PB action: `6aaa4678e67376208d1260c22a709b3920cb2bfa49dd7727401c52e4f52f882f`; done/returncode 0, complete and unambiguous. CAS receipt `f912d6b17f32316e264238f5097db831e800b7a1bbd2b56ff800dbd575c75985`; result `5c48fdaa8e49cb6bb594b7ca7b9ec0f3201eb9e5add32d368db28aa9237a5dab`.
- After PB action: `c6f57f0b1d92819acc8576b913958c34f1d12101790a4a72505bb93457b76154`; done/returncode 0, complete and unambiguous. CAS receipt `d676589862e7bb72d2ed138016e79a4dc18599ae8994c8605d7e2add07561399`; result `7ec66b7c06f62ca648c7e781b8195af46df527f7f6981c29c71b0da6a8841ef5`.
- Owned artifacts: `/mnt/shared/tessera-measurements/pq-stageb-1367-20261001/resource1883-before-03/` and `/mnt/shared/tessera-measurements/pq-stageb-1367-20261001/resource1883-after-03/`; each contains `operation/profile.pstats`, `operation/policy.json`, `operation/result.json`, `netdata.jsonl`, and `ending.json`.
- Before pstats SHA-256: `633cbdabe149f0031842aed9168cda96b7ebc050dbc84875c1177d1aa3ecbbb9`.
- Before Netdata SHA-256: `a7b3bab2add4d40507e172f25e129e4ddd101fb7883384db5957ffbfbb8c0a93`.
- After Netdata SHA-256: `2fa3331b8453181c3ef2153873f05d3280f57af2e9fba77d46eb9a4c08b1068d`.

The first instrument attempt failed before operation completion because the worker could not resolve SSH aliases as DNS names. The retained failure is action `6accbf902a3ac0713b9c77361b1f59839c9014de2823cec30e948463ff7a0127`, returncode 1. The successful collector used the addresses from `ssh -G` while retaining stable host names in the series. No missing-chart gate was weakened.

## Correctness gate

Behavioral RED `c310433c6a1a5e450ae58f7ad5f6a7333411b17504d3c51aa061532a80a0b333`: 1 failed, 0 passed, 32 deselected. Each of two actual Tessera names was resolved 12 times. GREEN `a466f3615c58209ed5ee85d2ae3580fa02e51c7148cd171751550f2fbbfd3da1`: 131 passed, 0 failed, 0 skipped; compilation and exact runtime preflight passed. The regression compares the complete original policy before checking counts, repeats a fresh invocation, and verifies that a later resolver refusal still propagates. Fresh read-only review accepted the frozen two-file change; it did not independently authenticate receipts or admit original #1367 performance.
