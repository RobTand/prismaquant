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

## Subsequent corrected 72-call bounded graph — 16:50 UTC

Root reviewed and authorized exactly `native-corrected-graph-invocation.json`.
PB action `abad03ae5f0d53afc801b6a206501a41b9a52bebb8bc850a130a6da1ed6ff91d`
ran once on Sparklina and exited 0 in 190.60 s. The unchanged six-CPU,
104/92-GiB measurement reservation was used. Snapshot
`7376380fbbc40fae586e9ef001b83c1fc6dd1051` is closure-only above
`591772439`; all 24 frozen source-file hashes and the exact requested command
match. `native-corrected-graph-audit.json` binds the actual result and complete
source/CAS/producer/claim, numerical, route, ownership, profiler and telemetry
checks.

The exact 72 scheduled backwards completed across layers 0/3/4, original rows
0/511, seeds 7000–7003 and isolated baseline/nonfinal fork/final original-owner
arms. All six primary outputs are finite. The original row0/layer0 primary
record still equals the unmodified native reference. All 72 leaf cotangents
have 8,388,608 finite elements and a positive nonzero-element count; the three
arms have identical cotangent and stimulus bytes in every one of the 24 groups.
All replay output bytes equal their primary output.

All 16 routed groups have identical fork/final routing and activity, with 4,096
expert assignments covering all 512×8 token/choice slots. Layer3 rows0/511 touch
286/287 of 288 experts; layer4 touches 287 in each row. The original 45-layer
configuration, dense layer0 MLP, DSA layer3, KDA layer4 and shared expert paths
remain the actual source paths. No shared adjoints were produced or retained.

The 512-row metadata check had zero graph calls and released every metadata
owner. Six measured source-residency snapshots cover current plus prefetched
lookahead at each tested layer/row. Sixteen original source files were hashed
(73,884,166,070 bytes), and all expected content hashes matched; all held
file descriptors closed, source owners expired and source violations stayed
empty. CUDA allocated/reserved peaks were 43,081,193,984 / 48,043,655,168 bytes;
after cleanup they were 67,108,864 / 104,857,600 bytes. The existing boundary
store recorded zero hot-read misses and zero remaining resident tensor bytes.
Its 201,352,428 bytes of bounded prefix artifacts are retained as experiment
evidence, alongside the result, traces and telemetry; they are not a new cache.

All nine layer/arm Torch traces contain CUDA kernels and autograd events and
passed size/SHA256 verification. Both-box Netdata has 177 samples per host,
maximum gap 1.214 s. Whole-action host telemetry records 9.63 W mean / 28.99 W
peak GPU power and 8.52% mean CPU busy. These totals include source content
hashing and metadata preparation; they are not evidence of GPU saturation or
production throughput. This experiment qualifies numerical/replay behavior,
not performance. The native result is 6,362,692 bytes with SHA256
`4126fe86231a1c4ae8dbaee6026bc2781de7ee0a458b2e9a84dd22daadb47e2f`.

The bounded graph gate passes. This does not qualify all 45 layer derivatives,
full-model cost or quality, or serving. Original-capture compatibility remains
a separate gate requiring the original capture action to complete; no
compatibility receipt or full-model consumer run was created by this work.

## Mainline integration — 17:04 UTC

Integration commit `5806aae588596298657034565261cb31cf6a9a34` combines the
reviewed derivative implementation with the selected source authentication and
verified capture loader. Both builder policies and both preparation policies
remain explicit. The builder regression checks authenticated context creation
before derivative binding and runner cleanup on refusal. The original-capture
compatibility receipt remains unavailable until that producer completes.

PB split 15 related test files into six independent CPU shards: **632 passed,
73 skipped, one expected failure**. Each shard reserved five CPUs to cover the
reader-thread fixtures, four GiB memory and one native thread. All ran on
dl380g10 under Python 3.12/Torch 2.10, with CUDA disabled. The skips comprise
36 missing-vLLM registry cases, 24 unconfigured checkpoint cases, 12 profile
documentation/default cases, and one real-encoder CUDA case. The existing
serving-profile field-reader expected failure is retained. Scope peaks were 369–457 MiB, all
exit statuses were zero, and all cleanup records were complete.

`root-integrated-cpu-cas-audit.json` binds all six actual outputs, receipts and
source bundles; each snapshot differs from the integration commit only by PB's
closure file. The coordinator's native CAS and artifact audits are also copied
here; they independently check the exact 72-call schedule, per-arm equality,
finite gradients, source hashes, nine trace files and both-host telemetry.
The initial CPU submission was rejected for unsupported pytest `-q` forwarding;
no action ran from that rejected submission. The corrected invocation used
pbtest's supported defaults and completed all six shards.

The 13 changed implementation modules also passed PB compileall action
`d9b6fa26cbd6a33d8570c15ad406825db326fd171e7592469d62d60e4182389e`
under one CPU/one GiB, with exit 0 and verified source/CAS/cleanup evidence in
`root-integrated-compile-cas-audit.json`.
