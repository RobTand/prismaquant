# Sealed checkpoint streaming

The optional executable-readset selection `checkpoint_incoming_mode="stream_once_staged"`
connects the existing `CheckpointIncoming` adapter to immutable PB input phases.
It is selected by post-capture regeneration with `--checkpoint-incoming-mode
stream_once_staged --executable-readsets`. Omission preserves the previous
manifest, record, phase names and runtime behavior. An explicit execution setting
must agree with the sealed selection; it cannot enable an unsealed row or disable
a sealed one.

`checkpoint-load` retains the verified checkpoint manifest and shared owner/pass
states. A referenced first-chain checkpoint places its original cotangents in the
first `chain-NNN-bound` phase, boundary window followed by cotangent window,
probe-major, with repeated boundaries retaining their first phase position. Later
rolls use the destination plane as before. A chain-empty row requires spill and
places each probe's original rows after its boundary entries in `spill-pP`. The
phase's extra read entries do not increase derived chain batches or capture groups.
The dispatcher validates these counts against its hash-bound Stage A slice.

The runtime requires strict allowed tiers and the protected PB residency context.
`staged_lease.load_sealed_manifest` exposes metadata through the existing bounded,
digest-bound claim/CAS/public-SDK reader. The selected mode, quantum, slice, probe
count and exact consuming membership must match that authenticated manifest.
This is input authority, not a payload lease: existing exact readers retain range
staging, lifetime, residency and digest checks. No alternate cache or reader is
introduced. Destination reservation and cotangent resource gates remain unchanged.

The research selection `stream_once_research` retains all its CPU-only refusals.
The staged first-chain path currently requires the default unfused, single-batch
chain and replay regime. The staged chain-empty path uses the existing final-pass
incoming seam with the sealed replay regime, including grouped capture and operator
GEMM accumulation. Capture-workspace and capture/shadow pass profiles refuse;
render-only profiling remains available because it does not reread incoming rows.
Band-serial handoffs cannot be combined with this selection. Complete or partial
pricing resume still runs the final passes and consumes every original operand.

## Finite original GLM protocol

This prerequisite does not establish GPU correctness, production lease lifetimes,
or a speedup. The root coordinator authorizes and reviews the following two
serial, measurement-isolated actions; neither is submitted by this change.

The original reference is #1366/#1348 `prof-1`, PB action
`f3f4489d2106b73f876fb3212ab0ef9ef03cbd3afa64a7678178c3a7cc3c2701`,
sparky, parent source `090f3463c6824ea4888f4898ba9f86a27aae6e4c` and sealed
snapshot commit `b016190cddf415f216738cdd98003b13d0fcf7ce`.
The actual layer-7 row has **no chain layers**, checkpoint 8 (v3), 2,048 incoming
rows (4 probes, 512 batches), and 15 retained windows. Its actual replay regime is
`capture_batch=4,accumulation=operator_gemm,chunk_rows=65536`; changing it to the
research default would invalidate this comparison.

| Existing reference | SHA-256 |
|---|---|
| `/mnt/shared/tessera-measurements/ws-ra-1291/prof-1/plan.json` | `ca6fa52e03e619e947ab60daf313080dd47e3b89ee8d4fa5ec10f4976b4fc60f` |
| `.../prof-1/records/layer-007.json` | `70b9e8d554df4b83d646fba881b0c521ddd75c27ba37f5c0dd5c406713efb7b8` |
| `.../prof-1/stage-b-spec.v9-render-profile-1348.json` | `eaed705cb666bdaf12b558455caea77e923699a30ac73d42377ccba6464946f2` |
| Original bound Stage A slice | `99301ad3655f3dbfe80a05dc1ab1dc9f5d547715b9f928fd11fadff69f5d27e1` |
| Prepared inputs | `2fb271d2febe9e206c081c57f46a602d41c502ec39326f588aad72b0eda8e83c` |
| Exact calibration safetensors (512 samples, sequence length 512) | `9cd1fa129f249abd80d22efaeb8bc7e8b2d3b4252f173a8c6f2b2e496a4f8329` |

1. Recover the original sealed invocation, accepted inputs and output namespace
   through the PB public result/source owners. Verify each reference above and
   actual current package/container/reader qualification. The old plan's chain
   regime is batch 4 with fusion; no first-chain roll runs on layer 7. Keep that
   identity, calibration, projection backend, render membership and numerical
   regime. Reuse valid immutable artifacts and their hash evidence.
2. Regenerate two fresh, separately sealed row generations on the same reviewed
   source: control omits the new selection; candidate seals it. Both preserve the
   original calibration, Stage A checkpoint/slice, prepared weights, regime and
   window membership. Use disjoint owned output, journal, spill, scratch and
   observation directories so neither resumes the other's prices. New identities
   are recorded; these are comparable operands, not identical plan hashes.
3. Freeze both action descriptors before submission. Use the existing
   `tools.pq_admitted_profile` / `tools.pq_row_profile_observer` owners with
   same-UID py-spy and render-only torch profiling at window 5,
   `capture=,windowed=none,render=5,render_wait=1,render_warmup=1,render_active=1`.
   Preserve PB affinity, bound native threads to 1, and declare actual aggregate
   CPU, memory, GPU, scratch, spill and cache lifetimes through PB. Do not copy the
   historical CPU9/memory96/spool218 reservation blindly: that run predates
   corrected spool accounting, and current admission must include every owner.
4. Submit control then candidate through PB's supported campaign transport,
   `max_attempts=1`, `retry_safe=false`, at most two GPU actions in total.
   Use the same qualified GB10 host for measurement isolation, report its UUID,
   and collect Netdata on **both** sparky and sparklina. Root GO must name the
   quiet slot, reviewed finite scope, honest resource envelope and published
   runtime capabilities. A host dependency here is measurement isolation.
   Retain a 1,800-second hard bound per row (3,600 seconds total workload cap),
   and the existing derived, ordered semantic progress/stall policy. Do not
   alter a sealed request to extend the bound. Failure ends this two-arm attempt;
   recovery is separately scoped and authorized.
5. Require actual terminal exit 0, complete semantic units, unambiguous attempts,
   CAS receipts, authentic source snapshots and verified local output claims.
   Compare every cost/journal record and final-plane digest bit for bit. Compare
   incoming identities/coordinates, final-pass coverage on resume, and per-phase
   IO deltas. Metadata moves phases; it must not disappear from accounting.
6. Report wall time, checkpoint-load wall/read/write bytes, moved-phase reads and
   waits, scratch writes, peak resident/cotangent memory, CPU/IO pressure,
   py-spy stacks, render kernel/host gaps, and GPU energy from finite Netdata
   power series against the ~140 W envelope. Rank the same completed workload
   by work per joule. Include ready/admission evidence and external GPU load.
   Compare the fresh arms for the claim; the issue's historical 161.2-second
   checkpoint load, 34.4 GB reads, 34.36 GB scratch writes and 28.9 kJ whole row
   remain historical evidence, never a fresh baseline or promised saving.

The numerical CPU tests explicitly double the protected input context and use
fixture source installs. They qualify numerical and phase-membership behavior;
they cannot qualify actual protected opens, stage residency, device shapes or GPU
performance. #1366 remains open until the real original row's acceptance is met.
