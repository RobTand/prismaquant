# Timed selected-anchor CUDA observation — 2026-09-08

The dense qualification completed fourteen encoding calls but its two full-call
CUDA traces exported 2,318,078,289 and 2,312,576,791 bytes, exceeding the declared
512 MiB cap. PB action `7c417f6473182bed07ba2d5e65db61fe21c31ad7ac8929b492b29b5f9a45d921`
failed observation (exit 1); retained costs do not establish qualification.
Export-size rejection cannot bound the collection interval.

The existing observer accepts explicit `--anchor-profile-seconds` with
`--anchor-cuda-only`. A timer uses PyTorch's public dynamic collection API to
stop initial CUDA collection while the original encoding call continues.
The timer joins before profiler teardown. Invalid/nonfinite windows, CPU
collection and toggle failures refuse; original encoding results/exceptions
retain their exactly-once behavior. Omitting the option retains the original
full-call observation. This changes no pricing, capture, producer or serving
contract. CUDA toggling is Kineto-wide whereas CPU state is thread-local; see
[PyTorch's implementation](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/profiler_kineto.cpp).

The deadline is cooperative. The record includes requested and observed seconds;
scheduler/toggle delays can extend it. Exported-byte limits remain separate,
and neither limit claims a hard bound on live profiler memory.

PB validation, with actual CAS payloads and source snapshots independently read:

- Regression `71a6cc64040533c781b1b5a02d8673e79efa68315fe98b16919a5be7a518aa23`:
  eight unsupported-option failures, 24 deselected before the implementation.
  These establish the new option boundary; the failed dense action above is
  the observed full-call defect.
- CPU `3812cc133ecebc02d023b2bb22374a68fddcbf43b06844bbc3a14f6939248b88`:
  31 passed, two native CUDA skips.
- Native GB10 `def003ee7dbd68035dd400326bc31f794ed94a7bec04092e8149e2c9d3456465`:
  45 passed, zero skips, Torch 2.13.0+cu130 in the qualified producer container.
  The real timed fixture returns both matrix-multiplication results unchanged;
  the trace contains exactly the first CUDA kernel and excludes the later one.
  Requested 0.25 s, recorded 0.250759305 s; trace 12,584 bytes, SHA256
  `dd5be3ce6931f03d05af4f6b7c43608adefb351441e786d9e56014f0e89217b3`.

The adjacent audit records source diffs, receipts and the independently parsed
trace. Both-host telemetry remains in the original fixture evidence under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/timed-observer-01`.
A fresh dense qualification is running at source `8e8506e584`, action
`9f46d50b9abd700c67f2b52539fe153ab574e2aee70cfc30d08dab3347e65e28`;
its result is pending. This fixture is instrument correctness, not a throughput
or quantization-quality comparison.


## Completed dense qualification

The pending action above completed with exit 0 and complete PB resource cleanup.
The actual CAS payload records fourteen anchor calls (ten first-round, four
refinement), two units and 1,030 priced rungs across 515 formats. Its snapshot
differs from `8e8506e584` only by the declared PB closure. The complete
PrismaQuant source package and producer remain byte-identical to the failed
dense baseline. The original canonical capture `f4bcbf40…` is unchanged.

Both native traces are accepted, with no observer errors. Requested 0.25 s;
observed collection intervals are 0.251329934 and 0.255068171 s. Trace sizes
are 1,683,273 and 72,418,678 bytes. Independently parsing the actual JSON finds
781 and 66,262 kernels, spanning 0.053970752 and 0.249615059 s respectively.
The second window contains 65,536 `_step` kernels totaling 169,734.857 us;
that finding covers this dense window only, not the expert workload or a full
anchor. The first window largely covers factorization startup.

The observer recorded 228 main-thread samples and 46 Netdata samples for each
host over 230.634 s. The earlier failed dense action ran on Sparky and this
one ran on Sparklina: this is an instrument validation, not a controlled
throughput A/B. The adjacent final audit seals actual traces and artifacts.

Independent cost-row comparison found all 1,030 quality and byte-cost rows
identical to the failed dense attempt. Only the fourteen measured
`encode_seconds` fields differ; paths and run provenance also differ normally.
The comparison is sealed in `priced-row-parity.json`.
