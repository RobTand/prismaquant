# Timed CUDA anchor window: GB10 qualification

Closes #1245 at the native-window regression's scope. The CPU lifecycle fix
already landed in PR #1825; this is its missing hardware qualification. No
encoder, observer or test implementation changed for this run.

## Before and after

The retained native RED is PB
`69d18114436c2e6cfd554437b8897a263301059fbb6e93ea3172c0e4994301b6`,
recorded in `timed-anchor-profiler-lifecycle-2026-09-30.md`: Torch
2.11.0+cu130 on sparklina, requested window 0.25 s, observed 0.253014018 s,
`stopped_by=deadline`, zero CUDA events, exit 1. Its missing Netdata mount
charts were an independent incomplete-evidence failure. This historical
record is retained; no performance or energy delta is inferred from it.

The after run is PB
`886e632afd7182f6ab47de2140b0098f26043fd840060ae809a99068bf256d55`:
two native tests passed, zero skipped, exit 0. The timed test executes a first
BF16 bmm, synchronizes it, waits for the real profiler teardown, then executes
and checks the later bmm. Its trace contains exactly one CUDA kernel, so the
first is present and the later operation is excluded. The untimed control
also records one real CUDA kernel. Both attempt records are `complete` with
`errors=[]`; each raw Netdata ledger contains successful samples for sparky
and sparklina, with neither host marked failed.

Timed `result.json` excerpt:

```json
{
  "status": "complete",
  "cuda_events": 1,
  "collection_window": {
    "requested_seconds": 0.25,
    "stopped_by": "deadline",
    "elapsed_seconds": 0.2519180279923603
  },
  "trace": {
    "path": "anchor-000000.trace.json",
    "bytes": 11313,
    "sha256": "271831478eba9ef032af707f4da45ffe1fb055caa691c99f9659b0275e189b47"
  }
}
```

## Executing identity and evidence

The admitted action ran on sparky, GB10 SM121, driver 595.91.07, Python
3.12.3, Torch 2.11.0+cu130 (Git
`70d99e998b4955e0049d13a98d77ae1b14db1f45`), CUDA 13.0 and CUPTI package
13.0.85. Loaded `libcupti.so.13` mappings are recorded in `summary.json`.
The existing native interpreter was
`/home/rob/venvs/pq-fixtures1947-1939-native-tf516/bin/python`. The known
producer/stock containers were inspected first and carried Torch 2.13.0,
so root explicitly approved the existing 2.11 native environment for this
control. No install, rebuild or host-service change occurred.

Before CUDA, published `pbtest_pins.check_pins` verified the installed
PrismaBuild Git commit `95a59051d48cda82eea7927f31870c6c862d7174` and Tessera
`b40c93cb73745097e57a1ba4cf5b9eee166c759a`, their RECORD bytes and import
ownership. The driver refused a different Torch/CUDA or observer/test digest.

| Source | SHA-256 |
| --- | --- |
| `experiments/glm_full_capture_profile.py` | `14d6f7132e5b73c478d9c7e1b1e4b25075de56ff8a0a812852bc57162fd93afc` |
| `tests/test_glm_selected_anchor_profile.py` | `c064d94d5e2b1fd748d139eacae9f00cb715b1bf3bb5bdabfbaae51827be10e8` |
| Retained native driver | `23e43d55309772646ec359998b1f120e1f295a2162b33c5be076319004263e19` |

The CAS source bundle is
`1d525ba0748ec40f86779fb9d45ee16acc6ab534fe0a93bce55c661e4c92b105`,
40,662,016 bytes, snapshot `cf80abbf6fc82152f1253691601033d47947d01e`
over main `8811ad5f1e9586ce0de9f5db7f64d1c8589f70d2`. Independent bundle
readback reproduced both executable file digests above. The successful CAS
result is `294172c29529a2b2e958b219d5d20df69c3eee8d57f38169c0c83e24bed1f72e`,
6099 bytes; receipt
`d77f0e356784403ac6faeca96a930ae26a92ed4953afeb1ca09bc7ab44d2606b`.

Raw trace/result/progress/profile text, both-host Netdata JSONL, Python
stack/IO samples, runtime identity, JUnit and summary are under
`/mnt/shared/tessera-measurements/pq-serving-instrumentation-20261002/native-window/`.
The driver is retained beside that directory as `native_window_1245.py`.
It invoked only
`test_native_timed_cuda_window_excludes_later_work` and
`test_native_cuda_only_batch_trace_has_actual_events`, using generated
`[8,64,64]` BF16 tensors and the tests' library warmup. No model inputs,
serving process or additional profiling workload were launched.

The request reserved two CPUs and 6 GiB aggregate memory, with a 1 GiB GPU
subset; OMP/MKL/OpenBLAS and Torch threads were bounded to one. PB preserved
affinity `[5,6]`, enforced priority -10, a 180-second hard bound and one
attempt. The terminal's scope memory peak is 1,230,675,968 bytes. The child
completed; no container or persistent CUDA process was created.

This qualifies native event drain and finite-window exclusion for this
generated control. It establishes no original-model capture, serving
numerics, workload throughput, full-model profiler memory bound or energy
claim. The clocks/power agreement remains outside this result.
