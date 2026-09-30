# Timed anchor profiler lifecycle: CPU checkpoint

Refs #1245. The bounded CPU lifecycle issue is #1823. This checkpoint does not establish a native CUDA fix.

## Scope

Timed CUDA-only observation starts and finalizes the profiler on one observer-owned thread. Readiness precedes the original anchor; a deadline or early return ends collection. The caller joins before export or a subsequent anchor. Original arguments, exactly-once execution on the caller thread, return identity and exception identity remain unchanged. Untimed observation stays on the caller thread.

The requested interval is not a hard wall-clock bound on initialization, scheduling or teardown. Trace caps, native-event checks and complete telemetry remain mandatory. There are no extra forwards, retained tensors, new cache mechanism, model-math changes, serving changes, wire changes, release-input edits or pin changes.

## Source and environment

Baseline: `bdb859836d6dbde3f0ce9c55f995e750ebed3726`.

| File | Final CPU-tested SHA-256 |
|---|---|
| `experiments/glm_full_capture_profile.py` | `14d6f7132e5b73c478d9c7e1b1e4b25075de56ff8a0a812852bc57162fd93afc` |
| `tests/test_glm_selected_anchor_profile.py` | `c064d94d5e2b1fd748d139eacae9f00cb715b1bf3bb5bdabfbaae51827be10e8` |

The recorded diff has 93 insertions and 57 deletions; SHA-256 `1672c9af5cc705d75d1e3a6e03588e2a35682043cfbcecfed7652614554c6447`. The subsequent measurement document does not change these executable bytes.

Final CPU environment: x86_64, Python 3.14.4, Torch 2.11.0+cpu, pytest 9.1.1, Transformers 5.16.1. Tessera pin remains `b40c93cb73745097e57a1ba4cf5b9eee166c759a`. These CPU tests and compile checks used admitted PrismaBuild, priority -10, explicit 600-second timeout/observation, aggregate two CPU and 4 GiB, native threads one and preserved affinity. The release-window sentinel was checked immediately before each client.

## RED and GREEN

| Action | Source and result |
|---|---|
| `cc55faf97560b2806334fe2d197dd0cf9ee30956e3b51b357ad203b238bef3d4` | Correct-x86 RED: baseline production with current regression tests in an admission-owned temporary tree; two assertion failures, zero skips, exit1 |
| `9cc97782e5bfcde3aae608dda176c1978ff4ac30853a5125dabe79d1ad2dc89a` | Final CPU source: 39 passed, three skipped, 14 warnings; two modules compiled; exit0 |

Both RED cases are `test_timed_profiler_lifetime_has_one_owner_and_preserves_call_thread`, parameterized by deadline versus early return. They cover real context-lifecycle observations using a controlled CPU profiler stand-in, not synthesized CUDA events. Current/read-only snapshots and shared checkouts were not modified to produce RED.

Literal GREEN skips:

- `tests/test_glm_selected_anchor_profile.py:290`: `native CUDA-only observer qualification`.
- `tests/test_glm_selected_anchor_profile.py:398`: `native timed CUDA observer qualification`.
- `tests/test_glm_full_capture_profile.py:12`: `native CUDA profiler qualification`.

Parent verified terminal status/exit, full stdout/stderr descriptor hashes and lengths, JUnit counts and the successful CAS result payload. GREEN terminal SHA-256: `51f5e93d158fe74fe60eac24c98059f31b504ffed704c08cec8e8f0aa7334c81`; successful receipt `8b5f67245419318f38d7ba917ad33ae4a965b8758f516c4f333d7a09408ab2b1`; CAS result `47566e30b47c73fa3c185d33bf4573ad758b4e60b9160847003ac4acba674aaf`, 12916 bytes. RED terminal SHA-256: `c7d69e767e32e50bf9bee6babaee3ba774f272ba0ec4a6e69eee59aa7b4a34f8`. Failed actions have no success receipt.

Exact commands, source snapshots, runtime inventories, JUnit, result manifests and copied canonical logs are under `/mnt/shared/tessera-measurements/pq1245-cuda-window-20260930/audit/`, `/mnt/shared/tessera-measurements/pq1245-cuda-window-20260930/cpu-red-x86/` and `/mnt/shared/tessera-measurements/pq1245-cuda-window-20260930/cpu-green-final/`. The exact CPU driver is `/mnt/shared/tessera-measurements/pq1245-cuda-window-20260930/audit/pq1245-cpu-final-driver.py`, SHA-256 `b0b2cc1fafefee13310fa953ccc5bb0770c6ee69f5269e95d4f88f463ea1489b`.

Independent fresh review found no issues and accepted only a CPU lifecycle delivery boundary. No global type-clean or native CUDA claim is made; unchanged baseline heterogeneous-result advisories remain documented in the audit.

## Native qualification remains open

Before the implementation change, genuine native action `69d18114436c2e6cfd554437b8897a263301059fbb6e93ea3172c0e4994301b6` on GB10/sparklina failed one test with zero skips. Torch 2.11.0+cu130 recorded zero CUDA events and a warning about activity toggling/correlation. The generated non-release fixture uses `[8,64,64]` BF16 tensors and two original bmms with library warmup. Its requested interval was 0.25 seconds; the old result recorded 0.25301401800243184 seconds and `stopped_by=deadline`. These facts do not establish the cause of missing events or a speed claim.

Separately, real telemetry on both boxes failed with:

```text
RuntimeError("Netdata required charts missing: ['prismabuild.mount_latency', 'prismabuild.mount_probe_state']")
```

The required Netdata ledger was empty. `netdata.samples=1` counted an attempted round, not accepted telemetry. PrismaBuild issue 1392 tracks the missing charts; no required chart, failure gate, service or runtime was changed here. No after-native profile exists.

The exactly-one-real-first-kernel/no-later-kernel native regression remains mandatory on #1245, with complete both-box telemetry once that prerequisite holds. CPU passes and native skips certify neither CUDA drain/exclusion, model capture, memory/residency, performance, serving numerics nor a runtime image.

## Disclosed unsuccessful attempts

- `64427d9312f2d33d3ffe80a8544a0d380f234efad8fc83a4df6a8408223b67a1`: wrapper failed before pytest because distribution metadata was requested as `tessera` instead of `tessera-quant`. This is not behavioral RED.
- `c8dc9cf16dc4c86cdd8caa1b3b421e8bcc063e003984b8b7c40d16f1180adb40`: two CPU assertion failures on sparky, incorrectly routed instead of required x86. Disclosed and superseded by the correct-x86 RED above; no GPU work occurred.
- `31168ee3602830520636f60d5c28e30b2d0f6cace537790269cbb8bb857a0916`: 39 passed/three skipped before final typing/test edits, not final-source acceptance.
- The initial native worker timed out after 1800000 ms. Its partial diff was retained and the same native/model/tool contract resumed. No alternate CLI, local test/compile, Docker or admission bypass was used.
