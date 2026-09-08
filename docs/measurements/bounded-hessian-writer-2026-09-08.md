# Bounded Hessian sidecar writer — 2026-09-08

Issue #377 addresses the remaining whole-file page owner in selected-source
anchor preparation. The existing writer held all selected source weights,
Hessians and X prefixes while `torch.save` produced one complete Hessian archive.
Advice after publication could not bound the within-file peak. The v1 phase
plan correctly refused a full GLM routed stack at 124.783 GiB.

The selected writer now uses Torch's same path writer and serializer, pausing
after synchronous tensor-record writes. The existing page helper verifies the
stable visible file extent, fences durability and advises its pages. Tensor
storage, pickle protocol, archive record names, CRCs, content seals and the
atomic publication sequence are unchanged. A file-like substitute was avoided
because it changes archive names. Resident-source serialization keeps its prior
policy. The two Torch internal entry points used here are qualified against the
CPU and native producer runtimes; future runtime changes still require parity.

## Measured before and after

All tests and measurements used PrismaBuild. The native workload contains
16 resident CUDA FP32 Hessians of shape 4096×4096, totaling 1 GiB, and writes the
ordinary `hessian_capture.pt` artifact. It is a writer qualification with fixed
values and provenance, not a canonical model capture or full GLM fit test.

Both actions ran on Sparklina in the same immutable producer image
`sha256:9f9b9f05b17531399ba66dc6415b054cf5d68c82270626d0e9150e75c808435f`,
Torch 2.13/CUDA 13.0 and Transformers 5.16.1, with four CPUs, 24 GiB physical
memory and a 16 GiB GPU subset. Native threads were bounded to one per process;
a separate thread sampled cgroup memory, anonymous/file/dirty pages and CUDA
reservation every 20 ms. Torch recorded CPU/CUDA operations and memory, and
`/proc/self/io` supplied per-process I/O counters. Netdata CPU, RAM, available
memory and power series cover both Sparks and each measurement window, with
three seconds of surrounding context. Sparky's coordinating canonical capture
was external load; Sparklina's measurement window was isolated.

| Observed quantity | Original writer | Prefix-advised writer |
| --- | ---: | ---: |
| Cgroup before writer (bytes) | 651,599,872 | 652,439,552 |
| Sampled peak cgroup (bytes) | 1,746,935,808 | 809,885,696 |
| Sampled peak file cache (bytes) | 1,120,169,984 | 113,537,024 |
| Sampled peak dirty file pages (bytes) | 1,006,657,536 | 23,089,152 |
| CUDA reservation (bytes) | 1,090,519,040 | 1,090,519,040 |
| Profiled writer wall time (seconds) | 2.707 | 3.036 |
| File bytes | 1,073,748,005 | 1,073,748,005 |

The identical CUDA reservation is reported separately because GB10 cgroup
charges can omit GPU allocations. The implementation reduces file ownership;
this single pair records a small time increase and supports no speedup or
work-per-joule claim. The original trace has 94 memory samples and the bounded
trace 130. Netdata has 10/11 one-second samples per chart around these short
windows; it cannot resolve the 20 ms transient alone.

Both archives have exact SHA-256
`150346bf06aed6c12d6699252bef4e4990dd91756ce87924e491c990ff90755e`.
Both return content seal
`02c6fcec3deee7c75a1aaaf83831070f0acfcb8d2b1cf14908422720325184b8`.
No bytes were renamed, patched or normalized after writing.

## Validation and disposition

The regression first failed under the unchanged writer (`783031f412d7`): all
bytes matched, but no tensor prefix was advised before publication. An initial
implementation attempt omitted Torch's path normalization, so three tests
failed on a Path object's missing `encode` method (`e259cf8a648a`). Passing
`os.fspath(path)`, as the public save wrapper does, fixed the actual cause.

The corrected focused CPU suite passed 33 tests with one explicit native
measurement skip (`51f662ae5e44`). Additional cases passed for FP32, FP64 and
BF16, shared/noncontiguous storage, ASCII and Unicode paths, and preservation of
the previous published capture/sidecar when page advice or the memory guard
fails (`c65629a5d7fe`, eight passed, one measurement skip). After v2 phase
accounting, four independent CPU shards passed 63 tests with two skips: the
explicit writer measurement and the CUDA-only memo comparison. CPU execution
used DL380's pq-cpu312 interpreter, Torch 2.10 and Transformers 5.16.1.

Native baseline `3e6131ae06c5` passed its measurement; native after action
`ac14c7cf0446` passed 34 cases with no skips including the measurement and capture
suite. Final native integration `ab7b3c10a237` passed 20 cases with one explicit
measurement skip, including real tiny GLM original-layout capture, selected
source preparation, native anchor and resume under the v2 resource plan. The
measurement was already completed, so it was not repeated in that integration
run. Each native action exited zero, recorded completed resource cleanup, and
was followed by an independently empty Docker listing.

Evidence root:
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/bounded-hessian-writer-01/`.
`baseline-invocation.json`, `bounded-invocation.json` and
`native-final-invocation.json` seal the exact commands and source snapshots.
`audit.json` verifies terminal hashes, exit statuses, CAS receipts and result
bytes, including the failed attempts. `baseline-01/` and `bounded-01/` retain the
original archives, memory/I/O samples, Torch traces, hash-bearing profile
summaries and both Sparks' Netdata. `before-after-summary.json` compares the
unchanged artifacts and measured owners. These artifacts are retained as
bounded evidence; the detached baseline checkout is disposable because its
exact test-only source is committed as `c635790f8` and snapshotted by PB.

## Admission implication and remaining gate

The v2 selected-anchor resource plan charges one largest tensor record plus
metadata as the file-page window, while retaining all H/X/source weights and
serialization scratch. The physical guard remains decisive: advice alone does
not guarantee page reclamation. No partial capture can become canonical and no
exact routed-stack group is split or replaced with a sampled estimator.

Resource action `ff10b7324a2a` recomputed the representative GLM stack from the
same canonical census and source headers: source preparation 68.221 GiB,
export inputs 84.345 GiB and resident anchors 98.019 GiB. Its maximum is now
98.019 GiB. Dense fused/singleton maxima remain 28.140/30.141 GiB.
`resource-inspection-invocation.json` and `full-glm-derived-resources.json`
record the derivation. This supersedes the writer's old 124.783 GiB phase for
this implementation; the earlier dated report remains unchanged history.

These are derived admission bounds. A full 864-unit GLM routed stack still
requires execution from the completed common canonical capture, verified
physical residency and the unchanged quality/cost/wire gates before any full
fit claim is made. This change adds no cache, dispatcher, format, serving lane,
calibration policy or release pin.


## Later same-day complete-roster metadata check

The 28.140/30.141 GiB dense values above are representative layer-zero groups,
not maxima over all original GLM layers. Replaying current v2 admission for all
132 groups at main `9ff3b97f7` derives a 98.041521 GiB maximum for routed stacks,
54.752041 GiB for dense fused groups and 54.736416 GiB for dense singletons;
shared-expert selections still prepare their containing routed layer. All 132
rows fit the declared 104 GiB planning budget without sampling. This extends
metadata coverage only; full-stack native residency remains unqualified.
Evidence: `experiments/measurements/glm-full-stack-admission-20260908/`.
