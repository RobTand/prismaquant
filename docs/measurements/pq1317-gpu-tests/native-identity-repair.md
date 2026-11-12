# Native backing-file identity repair

## Current integration decision: CPU only

No new GPU run is decided for the terminal-refusal integration; estimated GPU
minutes: **0**. Backing-file refusal regressions and replay of retained evidence
are CPU-decidable. The native-v2 GPU records below predate the terminal-refusal
repair. They did not execute the changed observer, merge or inventory validator.
The current integration is CPU replay, not a new measured qualification, and
qualifies neither the changed code on GPU nor a newer Tessera pin.

The retained EPERM records show reader unavailability and verified pathname
fallback, not mutation during those runs. Detected integrity failures must be
terminal rather than trigger another reader; that distinction is checked on CPU.
Image X is not the supplied Torch 2.11 arm. A future claim of GPU execution of
changed measured code requires a separate exact-image decision and execution.
See README.md, **How to check**, for the complete five-module CPU command.

## Historical native-v2 GPU admission decision (written before that run)

The following admission narrative, budgets, observations and receipts describe
the earlier native-v2 campaign only, not new work for terminal-refusal integration.

The requested requalification must use image X (digest in `candidate.json`),
Tessera `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`, and sm_121. The retained
image-X reports name Python 3.12.3, Torch 2.13.0+cu130, Triton 3.7.1 and vLLM
0.28.1rc1.dev397+gfd4a15126.d20260904. The supplied test environment instead
names Python 3.14 and Torch 2.11.0+cu130 on its GPU arm.

One targeted GPU-arm **availability preflight**, not a qualification run, will
record Python/Torch/Triton/vLLM versions, GB10 capability and container-engine
availability. It decides whether the admitted arm can execute the specified
image-X requalification without an install, download or nested fleet action.
CPU cannot establish the actual GPU arm's runtime/device. The smallest case is
one device-property query and package-metadata reads: no kernels, builds, model
inputs or test sweep. Expected allocation: less than 0.25 GPU minutes. Stop on a
runtime mismatch; do not spend GPU time running the 136-node roster under the
wrong runtime or relabel that as image X.

The availability query observed Python 3.14.4, Torch 2.11.0+cu130,
Triton 3.6.0, no vLLM, and NVIDIA GB10 capability 12.1. `docker` resolved
to the fleet CLI shim; this is not proof of an available container engine or
image. One follow-up read-only `docker image inspect` of the exact digest will
resolve that uncertainty. It cannot pull an image or launch a container. CPU
placement cannot establish the GPU host's image inventory. Expected allocation:
less than 0.25 GPU minutes, no kernels or builds. If unavailable, stop rather
than replace image X with the admitted Torch 2.11 runtime.

Image inspection succeeded: the exact requested repository digest is cached.
Docker's config `.Id` is `sha256:9e6882df27dd5274bcb8fe048a0c435c531bbd794247cfc964556cf7d60fbc7d`;
it is not the repository/manifest digest. Requalification can therefore use the
container, not the incompatible default Python runtime.

### Image-X preflight and targeted execution budget

Run the two full-roster collection preflights without CUDA inside the cached
image, using the retained exact candidate as a local Git origin and the retained
pinned pytest-tool directory read-only. There is no network, pull or install.
The preflights decide source/roster/probe-tool compatibility; no CUDA qualification
is inferred. GPU-host placement is needed for the cached image, not its device.
Expected combined allocation: under 1 GPU minute.

After a successful preflight, run one BF16 decode M=1 fused node inside image X
with the probe. This is the smallest current dense native launch that can prove
the observer can identify a genuinely loaded Tessera extension and retain the
live BF16 numerical oracle. CPU cannot execute that launch. Expected budget:
under 2 GPU minutes including a fresh build. Only then run the full rosters.

The first dense preflight refused before collection: image X provides Pygments
2.21.0, not the historical install-if-missing fallback pin 2.19.2. No test ran.
The offline loader now pins the supplied pytest/xdist execution toolchain,
requires all dependencies to exist, and records the image's Pygments/packaging
versions rather than imposing a new renderer-version requirement. This keeps
image X unchanged. Repeat the same two preflights (same budget).

The first targeted command stopped before launching Docker due to an inline
script using the wrong runner helper signature (`AttributeError: CANDIDATE`).
No CUDA test ran. Correct that command and repeat the one-node smoke (unchanged
under-2-minute build budget; no harness change).

The corrected smoke command reached Docker but its manually created output
root was mode 0700, unlike the normal runner's mode 0755; the daemon could not
traverse its source bind mount (exit 126). No CUDA test ran. Make that output
root traversable and reuse the same fresh source/build directory for the smoke.
This changes no harness or measured source and keeps the original build budget.

Reusing the saved command on another admitted action exposed its stale CPU
set: the Docker shim returned 125 before container launch. Recompute affinity
from the current action, as the normal runner does; this is an unavailable
attempt, not a test failure. The one-node decision and build budget are unchanged.

The container then refused the smoke CLI's misspelled `--surface-out` option
before collection (exit 4). Use the candidate's `--surface-json` option, exactly
as the normal runner does. No measured GPU node has run in these setup attempts;
the original one-node build budget still applies.

### Full exact-candidate requalification budget

The corrected M=1 fused smoke passed in 52.18 s. Its device-allocating test
record contains a CUDA numerical measurement (maximum absolute error
0.02324962615966797; zero bound violations) and four native mapping segments
with verified device/inode identity. `map_files` returned EPERM; the recorded
reader was `pathname`, with matching opened-descriptor identity and stable file
metadata. This exercised the earlier native-v2 provenance fallback on a real
Tessera native launch, not the later terminal-refusal repair.

Next run both whole rosters twice, each in a fresh source checkout, container,
worker-specific extension directory and Triton cache. These four suite launches
are the smallest complete independent-repeat proof of all affected evidence:
71 MoE + 65 dense nodes per round, including every correction node and all
current BF16 numerical comparisons. CPU cannot execute the CUDA population or
establish its native-library/device identities. Historical durations were 48 s
and 130 s per pair; allow up to 8 GPU minutes for two pairs including new builds,
file-identity checks and live numerical reductions. No model, TP2 or serving
campaign is part of this run. Stop at the first suite failure.

## Historical native-v2 status

The earlier native-v2 requalification completed: two independent rounds each passed all 136 required
nodes (84 device-allocating, including all 54 correction nodes), with no skip,
failure or omission. Each dense round retained 23 measured CUDA BF16 comparisons
on 19 nodes. All native observations have verified provenance and match the
built inventory; every merge and outer inventory gate passed. Four launches
took 374.59 s in total (6.24 minutes), within the 8-minute estimate. The
successful smoke took 52.18 s. Historical receipts remain unchanged; they cannot
supply missing device/inode or reader provenance retrospectively.

The authoritative portable evidence is `results/requalification.json`. Its
content-addressed native inventories expand losslessly to the original probe
JSON bytes, checked against their original digests. Large source trees, binaries,
raw JSONL duplicates and build caches stay in the external measurement outputs.
`results/requalification-evidence.md` records the full-roster operation and CPU
publication checks. The legacy gate's historical-harness/current-harness
comparison must not be "fixed" by falsifying old manifests.

## Availability, setup and smoke receipts

These preserve unsuccessful setup attempts as well as the successful smoke.
The default-runtime query and image inspection are availability observations,
not GPU qualification. The affinity refusal (125) could not launch a container
and is not a failing CUDA test. No required node failed in a launched test run.

```text
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ec59e8117d056438362a734e8efe4db855ec0f916a0a1b6b0ccbc9b2a8a7166a","size":401}},"operation":"c41cb9471fe90a560281300dd69f6a6d1f41705e098579842e73207f90bac9a8","receipt":"a085696f3dee153c0e8dd66c0852f73e380675869a0136f61b34447b86b8d635"}]
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"938cd640cede7d7dda3858011dcc0a7ac0f7ef54d9ac5da494d7708d5d37bb9d","size":191}},"operation":"d3e86fc279d6b1d99d008a13a67fc3e478f5b5401aa1e7135cc4efd100a36153","receipt":"036f4708e8ab6a32d4f817cbacdda7af06d09a92f0cba4d156167ee2cbed74b3"}]
[Test evidence {"artifacts":{"stderr":{"digest":"be4445e13edb5910714c2c91686e3f26a0aa7a8407354a5a86d7e17142562cd1","size":801},"stdout":{"digest":"5a3bb6a4649adf6aa6e0b07f0bf8f58ec59f615806ce47980fcbcfe923c1d568","size":4226}},"operation":"b783d061664048fb354a9904b4abd7f8ff5ca5d4a6784a5643c675902c5d3639","receipt":"8efe80e30234093dd73e0b5b6fa332fd790f43958947366482302106a21e980d"}]
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"4f07cdadce55ed3c47890b2976deef9f5e3b63cda1729cc1f0f880be7cbbcbb1","size":11516}},"operation":"5267a884d103f822f34a4ff5022bdd80bc6006db6f937d89c417547e02ef01cf","receipt":"34211f735666aeb6d6488bfa2bf2ed87cd7e319fb8367e9d9e2ef40f5ebd35f3"}]
[Test evidence {"artifacts":{"stderr":{"digest":"8a63b563fa8e2e35464830e3bab9f960f1351f43d9fa3659ad862efaa606c16a","size":2379},"stdout":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0}},"operation":"df9a403935fce3dd967fc9a3fba8e1c60118ceeb04af150e8715057a07fcfc91","receipt":"aa0359de16e608ecdc0be3b7d2eb446aa02d1067cee3342e6f2bbc664fa3314b"}]
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f2f0a0eb86ffa84ebeaf37ff4c8f6293c79f1d9206ac87dfecd59a240af58579","size":1176}},"operation":"f010bbb2beb3cff791e786c8f0a0bcab097e417dc97771f56f8ce29c1e23af2c","receipt":"f21e8e68e0fad22be92680181d3a2e21d1908921a7c468c6cbbe8864ccfa7983"}]
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"06eb87eae459be25390284f8f65946e8eebf6f83789b72c33dda158e1bc0b332","size":177}},"operation":"4caeb85c6f4a770d43b9d9c660216912b9b42601943f0b6e8cdb566a0d597423","receipt":"02176f5406fc54340a253d15e0ab2e41535ae70cc468d684b37a3e7a58094b84"}]
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"069e8f8a844c00e322389e1fc503ac49b59b7bd83a44f5593409012ea65c5ba3","size":1086}},"operation":"9bdc7e7c2722c3b6a0976fbfd10bd5e092ae1d3ae8dfed896a4755c9ae250bfd","receipt":"baefd3d4f4818341f01e79ec202257e08f4087f1673b35338b0050d145f243a1"}]
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"4830ea485a08a07c329f56b9ab70de9f161a13903f7f9e70264bbd60e2760322","size":2628}},"operation":"3dd94d2b4c36ee3e660483a27d97135d49b0f9bbfe4d2d3828d940b0b470a29b","receipt":"fa55c88294cc14929393bf44f1e8cff06ac4176d1f1fd72fa583d6145a6a180b"}]
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3cf9cfa2855e121f8589ba9ba8d5d28b9f10a502a8e2758ffa9b0090c6d95c6b","size":120372}},"operation":"a577b8d9db2049265c77f9c211f8a7994b45374c25575163ddc6310664be1cb7","receipt":"a6f369021f33b03630516fe352c784c8a1a0dab954c0d49ac14e9efcea0b547f"}]
```
