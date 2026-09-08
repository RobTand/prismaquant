# Closed GLM KDA derivative: image and CPU gates

2026-09-08. This record covers implementation, a separate derived image, CPU
source equivalence and actual-image CPU/meta authentication. It does not
qualify native corrected backward execution, the 72-call graph, original-capture
compatibility, quality, bpp, serving or performance.

The original modeling file `2092bbb4…` is transformed at exactly one expression:
strictly upper-triangular cumulative-gate differences become zero **before**
`exp`; the diagonal and lower triangle retain their original operations. The
corrected full modeling file is `416bd616…`. The derived Docker image content is
`d0256efb…`, with the original `eb8592ab…` config and ordered rootfs prefix plus
one new layer containing only that modeling file. The original image is retained.
Full hashes and image inspections are in `image-build-result.json`.

| Gate | Action key | Verified result |
| --- | --- | --- |
| Separate derived image | `d33e5d2b8969084fc8f34fb66a715ae38ecf3363c7c2d645d09036dc78531754` | Exit 0; original config/layer prefix unchanged; exactly one new modeling payload |
| Independent archive inspection | `a6b3c3f460b6cfab52738cc17d76c72a3d1934128a0d5e894d6a68aaa7232c62` | Exit 0; all 20,891,473,920 archive bytes hashed; config, added layer, file path and modeling hash checked |
| Final causal/core/final-state CPU proof | `1c91ea18a1c8da392318587059806974770521ab18aebfdae520ca70b616b8f7` | Six cases; original/corrected core and applicable final-state bytes equal; corrected gradients finite and FP32 oracle tolerances passed |
| Actual-image CPU/meta binding | `1fb94e6b11a125147cb2bf8053d36bbeffd4e98443c24c6b911de82bf0f813c4` | Exit 0; all 45 layers built on meta, 34 real KDA modules authenticated, 11 tampering/omission cases refused; CUDA unavailable and uninitialized |
| Code-field authentication regressions | `4f3bab55b3b912d35bae9dc9ed07c9b3278053f086e374efffb15a067635917d` | 42 passed |
| Final related CPU suite | `ad3591a03e80a28588ca07873f4d7397625a8bdcb9cb6373d897dabe62c3d4f5` | 457 passed, 72 skipped, 1 xfailed in 17.49 s |

The final suite covered derivative identity/receipts, original graph helper,
qualification windows, projection backend, container launch, streamed handoff,
model-profile conformance and architecture/doc staleness. Skips are 36 checks
requiring vLLM, 24 checkpoint conformance cases without `PQ_CONFORMANCE_MODELS`,
and 12 explicitly documented default/profile-structure cases. CPU suite logs
and the CAS outputs remain referenced by `cpu-gates-audit.json`.

The final numerical proof is separately retained in
`cpu-causal-final-state-result.json` and `cpu-causal-final-state-audit.json`.
It supersedes the earlier proof for this contract: it uses the exact reviewed
strictly-upper expression and explicitly checks final-state byte equality.
The earlier dated source-reproduction evidence remains history.

Actual-image binding ran in PyTorch `2.13.0+cu130` on Sparklina, using CPU/meta
only, two admitted CPUs and an 8 GiB bound. The recorded scope used 8.38 CPU
seconds and peaked at 612,270,080 bytes. Host telemetry averaged 4.37 W GPU power;
this is evidence of the CPU-only run's environment, not a GPU speed or energy
claim. The archive verifier ran on dl380g10 with one CPU and 2 GiB, used 63.65 CPU
seconds and peaked at 50,298,880 bytes. Netdata CPU records are in each action
profile; dl380g10 has no pqteld GPU series, explicitly recorded by PB.

## Refusals and implementation corrections

All attempts are retained in `cpu-gates-audit.json`; image build failures have
separate `image-build-negative-*.json` records. Canonical terminal records are
referenced by path and SHA256. Derived audit cleanup records retain safe
completion/release fields, excluding broker nonce and capability data.

- `26ceef57…`: the Docker build backend could escape the admitted scope; the
  contained archive transformation replaced it. No image was changed.
- `a48fcb5b…`: the first archive bound used Docker's reported size, which was
  compressed on one engine. The explicit uncompressed archive bound became
  32 GiB per file with an aggregate disk-space check; the complete exported base
  archive was reused without repeating export.
- `aa346c70…`: the initial CPU suite found a message expectation typo, minimal
  plan fixtures missing `canonical_capture`, and a two-worker PWC test submitted
  with only one admitted CPU. Fixtures and the reservation were corrected;
  `aff8f101…` passed 438 tests.
- `42bd184b…` and `540e43a5…`: recompiling hub source inherited this verifier's
  future-annotations flag. The diagnostic proved `co_flags` was the only differing
  field. `dont_inherit=True` authenticates the target's own compiler flags.
- `bc77f197…`: the real attention method is wrapped by the original
  `force_accelerate_hooks('conv1d')`. CPU inspection `beac04df…` bound the unchanged
  accelerate integration source `4469496d…`; authentication now checks its exact
  wrapper code, globals, child-list closure and original modeling forward body.
  Execution still uses the original wrapper.
- `a9c02ad0…`: every immutable code field compared equal while `marshal` byte
  encodings differed due to reference/interning state. The verifier now compares
  all public immutable code fields recursively, including nested source paths,
  flags and constants. Tampering tests and the actual-image gate passed.
- `dd4e53a0…`: two regression cases demonstrated an unretired streaming context
  when lookahead parsing or the runner constructor failed. Commit `bfffea4309`
  extends existing context shutdown to these failures; the final suite passed.
  This is the additional lifecycle fix discovered during this work.

The closed consumer receipt rechecks exact CPU case/expression identity, native
schemas, actual derivative execution, the exact backward and diagnostic roster,
primary output finiteness, finite nonzero gradients, output equality, per-arm
cotangent/stimulus identity and fork/final route activity. It also requires the
completed original capture action and canonical producer/CAS evidence; it never
changes the original capture identity or overrides its runtime validation.

`native-corrected-diagnostic-invocation.json` freezes one native corrected
row0/layer0/seed7000 baseline backward, bound to the original native layer0 forward
bytes and the separate image. It is **not submitted** and awaits root review.
The 72-call corrected graph and any compatibility receipt remain subsequent gates.

## Subsequent native layer0 diagnostic — 16:41 UTC

Root reviewed and authorized exactly the frozen diagnostic above. PB action
`37056154d67ef1e47ca0a5079e4d6f01e226c0e313fd1925387e22fd1a0c092a`
executed once on Sparklina, with six CPUs, 104 GiB total memory and a 92 GiB GPU
subset under measurement admission. It exited 0 in 31.32 s. Snapshot
`416410f93925525d4a8ecd8b332e622815f0daa7` is closure-only above
`0475656fd3`; all 24 frozen source-file hashes and the exact invoked command
were checked. `native-corrected-diagnostic-audit.json` binds the terminal,
canonical CAS payload/producer receipt, source snapshot, actual result, original
reference, profiler traces and both-box telemetry. Root independently reviewed
the same action and artifacts.

Exactly one `(layer0, original-row0, seed7000, baseline)` backward completed.
The entire primary output record, including BF16 output SHA256 `f0215afb…`,
equals the original native diagnostic. All 8,388,608 leaf-gradient elements are
finite; 8,388,322 are nonzero and 286 are zero. The original leaf gradient had
8,388,608 NaNs. All 60 observed branch forward/backward tensors in the corrected
run are finite. The native derivative identity equals the actual-image meta
identity, including original decorated fallback/accelerate dispatch and all 34
live gate configurations.

All four consumed original source-file hashes match, held descriptors close,
and source owners expire. Peak CUDA allocated/reserved bytes were
11,080,388,096 / 11,465,129,984; after cleanup, 67,108,864 / 104,857,600.
The final PB scope cleanup is complete, with no reported source violations or
telemetry/cleanup errors.

The before/after Torch traces contain 4,521 / 4,558 CUDA kernel events and full
autograd events. The corrected trace is 15,761,804 bytes, SHA256 `37ae5676…`.
Both-box Netdata has 17 original samples per host and 21 corrected samples per
host. Corrected whole-action host power averaged 6.67 W and peaked at 21.03 W,
with mean CPU busy 7.08%; this short diagnostic is not a GPU saturation or
throughput comparison. Numerical finiteness with equal forward bytes is the
measured result; no performance improvement is claimed.

This passes only the native layer0 diagnostic. The corrected 72-call bounded
prefix graph is the next independent gate. Its invocation is frozen separately
for review in `native-corrected-graph-invocation.json`; it is not authorized by
this record. The original capture is still running, and no original-capture
compatibility receipt has been issued.
