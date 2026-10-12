# Exact-candidate GPU qualification for Tessera #610 and #611

Status: **complete**. All 136 required nodes passed on sm_121 in image X at one candidate commit.
No node failed, skipped or went uncollected. A second, independent run agrees node by node.
All 54 nodes that carry a correction ran on the device. 52 further nodes are host logic or refuse
before they allocate. They passed, but this package does not count them as GPU proof.

This package answers prismaquant#2547, a part of prismaquant#1317 (criterion C5).
It qualifies one candidate. It does not qualify whole-model serving. See "What this does not show".

## Candidate, image and device

| Item | Value |
|---|---|
| Tessera commit | `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb` (merge of tessera#1033), contract v60 |
| Why this commit | `prismaquant/tessera_runtime/tessera_serving_runtime_pin.json` on `main` pins it |
| Contract digest | `ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e`, checked inside every run |
| Image X | `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5` |
| Software in the image | Torch 2.13.0+cu130, vLLM 0.28.1rc1.dev397+gfd4a15126.d20260904, Triton 3.7.1, Python 3.12.3 |
| Test tools | pytest 8.4.2, pytest-xdist 3.8.0 (installed into the output directory) |
| Device | NVIDIA GB10, compute capability 12.1 (sm_121), driver 595.99.02, 48 SMs |

`candidate.json` holds the full identity record.

## Result

| Suite | Issue | Files | Nodes | Passed | Device-allocating | Accepted run (PrismaBuild action) | Host |
|---|---|---|---|---|---|---|---|
| MoE | tessera#610 | 3 | 71 | 71 | 38 | `7a3dac4d75437bb570b7c27379112cf65bff1fcc8bc6728d72a6b6c07e19db4d` | sparky |
| Dense | tessera#611 | 2 | 65 | 65 | 46 | `ec7d9d959d863db83a37063203f0d34d32dc4a588230c7f4856ad25921c8cbd6` | sparky |
| Total | | 5 | 136 | 136 | 84 | | |

Every GPU run used `--strict-cuda`. The instrument reported 0 skipped tests and 0 modules not collected.

**What counts as GPU proof.** Tessera's instrument marks a node "device-allocating" when
the CUDA allocator saw a new allocation during the test call. That count is a floor.

* 84 nodes are device-allocating. All 54 nodes that carry a correction are among them.
* 18 MoE nodes are CUDA-gated but refuse or check before they allocate. By design, they allocate nothing.
* 34 nodes need no CUDA. They are host logic: eligibility rules, constants, config checks.

The 52 nodes in the last two groups passed on the GPU box. They are **not** counted as GPU proof.
`results/moe.json` and `results/dense.json` give the class of every node.

## Resolution records

Facts were read from GitHub on 2026-10-09. `corrections.json` has the full records.

| Issue | State | Fix | Merge commit | Ancestor of candidate |
|---|---|---|---|---|
| tessera#610 | closed 2026-09-30 | PR #642, merged 2026-09-30 (`Closes #610`) | `86cda15daa16c4aba6b064cbd9bab72013d9a115` | yes: the candidate is 1317 commits ahead, 0 behind |
| tessera#611 | closed 2026-09-27 | PR #643, merged 2026-09-27 (`Closes #611`) | `66f3626f22c71396d7cda47c8ce93906290918cf` | yes: the candidate is 1593 commits ahead, 0 behind |

Both fixes changed tests only. #642 changed three MoE test files. #643 changed two dense test files.

The old green receipts do not qualify the candidate:

* #610: action `d1080d3b…` passed 71 nodes on image X, but at an earlier Tessera commit.
* #611: action `3514070f…` passed 59 nodes with Torch 2.11.0+cu130. Image X carries Torch 2.13.0+cu130.
  So the #611 fix had no green result on image X before this package.
* The three MoE files did not change after #642. The two dense files changed in six later commits after #643.
  Those changes add the fused dense launch and its parametrization (59 nodes became 65).
  The history is in `corrections.json`, section `test_file_history_after_fix`.

## Which nodes carry the corrections

The two issues name 4 and 39 failing nodes at master f587d5b62 (43 in total).
All 43 map to 54 nodes at the candidate: 9 MoE nodes and 45 dense nodes.
Parametrization makes the count grow. The oracle test split into 6 arms. Six BF16 decode nodes doubled (fused and Triton lanes).
`corrections.json` lists each original node and the candidate nodes that replace it.
`roster.json` tags each candidate node with the correction elements it carries.

| Element | What it requires | Nodes |
|---|---|---|
| 610.A | Input-weight placement at topk=1 matches the stock modular prepare | 3 oracle arms, `router_weight_on_input…`, 2 TP2 loader nodes |
| 610.B | Input weight at topk=2 is refused by name | 3 oracle arms, 2 TP2 loader nodes |
| 610.C | Output weight at topk=2 stays covered | 3 oracle arms, 2 TP2 loader nodes |
| 610.D | Family and arithmetic arms report as separate nodes | 6 oracle arms |
| 611.A | Decode serves the packed native window GEMM | 12 BF16 nodes (both lanes), 6 FP8 nodes |
| 611.B | Prefill keeps the same native path | 2 BF16 nodes, 3 FP8 nodes |
| 611.C | The module holds only the packed native bundle | streamed and resident nodes |
| 611.D | Dispatch does not need the retired GEMV module | rung R=7, rate-1 columns, compile identity, optional extension |
| 611.E | A compiled forward with a dynamic token dimension does not recompile | 1 BF16 node, 1 FP8 node |
| 611.F | Retained GEMV reference coverage runs on a test-owned holder | precision, scale, host descriptor, graph capture |

**Roster provenance.** No Tessera issue or PR publishes a node list for the candidate.
The roster is every node that pytest collects from the five files the two fix PRs changed.
The harness refuses any run whose collected nodes differ from the roster. A Tessera owner can supersede it.

## Current dense launches: numerical and boundary proof

At the candidate, the dense routes launch one of two native paths per module. The route stamps which one.
The tests read the stamp back and assert it. The accepted dense run stamped:

| File | Launch (symbol / decoder) | Nodes |
|---|---|---|
| BF16 | `tessera::fused_window_dense` / `native_fused_window_dense_folded` | 12 |
| BF16 | `tessera::window_gemm_dense` / `native_window_gemm_folded` (Triton lane, `TESSERA_DENSE_FUSED=0`) | 6 |
| FP8 | `tessera::fused_window_dense` / `native_fused_window_dense_e4m3mma` | 11 |

The other dense nodes check state, precision or graph behaviour and assert no stamp.
The FP8 file does not parametrize the lane. It tests the default fused lane only.
The route does not switch on M. Every M runs the same packed launch.
The boundaries are the places where the retired GEMV lane had a limit.

| Boundary | BF16 nodes | FP8 nodes |
|---|---|---|
| Decode, M = 1, 2, 3, 4, 5, 8 (8 is the retired lane's `GEMV_MAX_M`) | 12 (both lanes) | 6 |
| Prefill, M above 8 | M = 16, 64 | M = 9, 32, 64 |
| Rung outside the retired lane's range (R=7, q256=1792) | 1 | |
| Rate-1 columns, M = 2 and 4 (BF16: q256=256 on the current route; FP8: the retained reference path) | 1 | 1 |
| Compiled forward, dynamic M = 2, 1, 4, 8, 64, no recompile | 1 | 1 |
| CUDA graph capture of the retained reference path | 2 | |
| Residency, streamed and resident | 2 | 2 |

BF16 has no GPU node at M = 9. The retired rule at M = 9 runs on CPU only (`test_decode_is_gemv_is_the_m_rule_in_one_place`).
FP8 covers M = 8 and M = 9 on the GPU.

**Numerical oracles.** The harness records the evaluated text of each passing float assertion.
`results/*.json` keep them per node (`float_comparisons`). Examples from the accepted runs:

| Node | Measured | Bound |
|---|---|---|
| MoE oracle `[value-epilogue-False]`, fused / split | 0.0 / 9.8e-4 | 0.0125 / 0.0095 |
| MoE oracle `[value-folded-True]`, fused / split | 4.9e-4 / 0.0 | 0.0078 / 0.0078 |
| MoE oracle `[e4m3-epilogue-True]`, fused / split | 2.1e6 / 0.0 | 2.5e8 / 1.9e8 |
| `router_weight_on_input…` vs the stock modular kernel | 1.7e-5 | 5e-3 + 1e-2 × max abs of the stock output (0.00099) |
| TP2 loader tiles `[0]`, plain / input-weight | 1.1e-5 / 1.2e-5 | 0.05 + 0.02 × 6.4e-4 / 0.005 + 0.01 × 2.3e-4 |
| FP8 decode M = 1 to 8, relative error | 0.0 | 0.008 |
| FP8 prefill M = 9 / 32 / 64, relative error | 0.0 / 5.6e-6 / 5.6e-6 | 0.008 |
| FP8 GEMV vs materialized, bit-identical share | 1.0 | 0.9 |

The BF16 dense nodes assert an elementwise bound and print only `True`. They show no margin.
`test_gemv_and_torch_agree_with_measured_differences` prints its own measurement:
`max_abs=0.000e+00 max_rel=0.000e+00` (`results/logs/run2-dense/pytest.log`).

**Obsolete launch assertions.** None remain. The old expectations were the symbols
`tessera_window_gemv::gemv`, `torch._scaled_mm` and `torch.mm`, and the `tessera_gemv` holder.
Tessera#643 ported all 38 test functions and deleted none. `corrections.json` lists what remains of those
names in the candidate tests: constant checks, negative assertions, and one reference computation.
**CPU skips.** None. The harness runs under `--strict-cuda`, which refuses a device-less run.

## Native path evidence

**Dense suites.** Each worker builds its own extensions in its own directory. The accepted run mapped these libraries:

| Worker, file | Library | sha256 of the mapped bytes |
|---|---|---|
| gw0, BF16 | `tessera_routed_fused_value.so` | `3bb266f43d00…` |
| gw0, BF16 | `tessera_window_gemv.so` (reference tests) | `13177f1c76eb…` |
| gw1, FP8 | `tessera_routed_fused_mma_e4m3.so` | `ff76725e9641…` |
| gw1, FP8 | `tessera_window_gemv.so` (reference tests) | `f9f55e5f51a4…` |

Each digest equals the digest of the built file in `native-so.sha256`.
The two `window_gemv` builds differ in bytes, so build identity is per run. Full digests are in `results/dense.json`.
The builds target sm_121 (`torch-ext/<worker>/tessera_*_sm_121_tessera_guarded_v1`).
Triton compiled `_window_gemm_kernel`, `_window_repack_kernel` and the trellis kernels for arch 121.

**Substitution.** The fused lane falls back to the Triton lane when its library cannot build.
That path logs a WARNING. The runs log no such warning. The only lane decisions logged are 6 INFO lines,
`disabled by TESSERA_DENSE_FUSED=0`, from the six BF16 Triton-lane nodes. A node that asserts the fused pair
passes only when the fused library served it.

**MoE suites.** These three files map no Tessera shared object. They run the compact Triton adapter
and the Triton grouped window GEMM (`_grouped_window_gemm_kernel`, `_unpack_body_kernel`, `_window_repack_kernel`),
and vLLM's stock `fused_moe_kernel` as the oracle. All compiled for arch 121.
The log says why the compact adapter serves: the routed fused lane needs at least 128 columns, and the test stacks have 64
(`compact window MoE adapter kept for a e4m3 stack of 3 experts: gate has 64 columns`).
That is a shape refusal by the lane's own predicate. It is not a failed build. The routed fused extension lane
(`tessera_routed_fused_*`) is therefore not exercised by these files. It belongs to the native-cell work in prismaquant#1317.
The pin's four `serving_native_extensions` are listed in `candidate.json`.

## Identity chain

1. Candidate commit `fca4c6ce0…` is the merge of tessera#1033 on Tessera `master`. The pin file on `main` names it.
2. The harness fetches exactly that commit into a fresh git checkout and refuses any other HEAD.
3. Tessera's own instrument (`tessera.suite_source.v1`) reports `verified`: 2005 files checked against the commit's
   blobs, source digest `980a9718a3ff28714794ce2d3c04cea79e045318f9f16f566db361e9db42d627`.
   Both xdist workers agree, and the identity did not move between suite entry and exit.
4. The shared pin directory `/mnt/shared/tessera-pins/fca4c6ce0…` digests to `57e16866…` over 2005 files.
   A tarball of the commit from GitHub digests to the same value, with no extra or different file.
5. The contract file in the checkout hashes to the pinned digest.
6. PrismaBuild ran the harness from PrismaQuant commit `dffc6a6b72949a5d149406c66bcb16ccfd74e4e7` with a clean snapshot
   (`dirty_sha256` is the hash of empty input). `results/log-digests.json` binds the harness files to that run.
7. Docker on the host reports image ID `sha256:f8dbe1a0…` for the reference. Architecture arm64.

## What this does not show

* **Whole-model serving.** The nodes are component tests. Dense tests stub vLLM's linear base and parameters.
  FP8 dense tests also replace the activation quantizer with a reference quantizer.
  Native routed E4M3 cells, compiled cells, full-model TP2 and smoke receipts are other children of prismaquant#1317.
* **Real TP2.** The TP2 nodes simulate rank geometry on one GPU. They are not a distributed serve.
* **A routed fused-lane result.** See "MoE suites" above.
* **Margins for BF16 dense nodes.** They assert a bound and report `True`.
* **Exhaustive GPU use.** `cuda_allocated` is a floor. Some nodes may use the device without a new allocation.
* **Performance.** No speed claim is made.
* **A fixed owner roster.** See "Roster provenance".
* **Host variety.** The accepted runs both used sparky. Corroborating GPU runs of the same candidate passed on sparklina
  (see "Runs"). The first run's dense suite used sparklina.

## Requalification

Bind every result to this one candidate. After any change to the candidate commit, the image or the harness, requalify.
Treat a change under `src/tessera/`, `csrc/`, the five test files or the test modules they import as a change to measured code.
Rerun both suites. For any other change, write a dependency analysis that names each retained node and why the change cannot reach it.
Tessera `master` is 439 commits past the candidate (2026-10-09). If the pin moves, rerun:

```
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py --cwd <checkout> --cpus 2 --demand mem_gb=24 --gpu --tag gb10 \
  --container-image '<image from candidate.json>' --priority 0 --timeout-s 5400 -- \
  python3 docs/measurements/pq1317-gpu-tests/harness/run_suite.py --suite dense --mode run
```

Run `--mode collect` first without `--gpu` (the D38 preflight). `gpu-commands.md` lists the exact commands that were run.

## Runs

| Run | What | Where | Result |
|---|---|---|---|
| run2 (accepted) | Harness `dffc6a6b72`, git checkout, instrument `verified`, per-worker extension directories | `results/logs/run2-*` | 71 + 65 passed |
| run1 (corroborating) | The first run of the five files, before the harness existed. Source read from the shared pin directory, instrument `unknown` | `results/logs/run1-*` | 71 + 65 passed |
| development | 12 other PrismaBuild actions of this attempt | `results/pb-receipts.json` | 10 passed, 2 refused by the preflight (see below) |

Node outcomes and the per-node CUDA flag agree between run1 and run2 for all 136 nodes.
Development actions include a run of the same dense suite on sparklina under another file-to-worker rule.
It passed 65 nodes. It and the MoE run on sparklina corroborate across hosts but do not count as qualification.
Two preflights failed on purpose: they caught a defect in the probe's own self-test (pytest elides long values in an assertion text). The fix is in the harness.
One diagnostic run showed why an earlier probe missed libraries. The extension directory sits on NFS.
NFS renames a busy library to `.nfsXXXX` when another worker rebuilds it. The final harness gives each worker its own directory.

## How to check

```
python3 -m pytest tests/test_pq1317_gpu_qualification_package.py -q
```

The test recomputes every node outcome from the retained `junit.xml` files.
It also checks the digest of every retained log, the roster, the correction map, the identity chain and the native-library records.
External facts to recheck by hand:

```
gh issue view 610 -R RobTand/tessera --json state,closedAt
gh api repos/RobTand/tessera/compare/86cda15daa16c4aba6b064cbd9bab72013d9a115...fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb --jq '{status,behind_by}'
```

## Files

| File | Content |
|---|---|
| `candidate.json` | Commit, pin, contract, tree digest, image, extensions |
| `corrections.json` | Resolution records, original failures mapped to candidate nodes, obsolete assertions, exclusions, test-file history |
| `roster.json` | The 136 required nodes with suite, line, CUDA gate, tier and elements |
| `gpu-commands.md` | The commands that ran, with action keys |
| `harness/` | `run_suite.py` (host), `container_entry.py` (container), `native_probe.py` (pytest plugin), `probe_selftest.py` |
| `results/summary.json` | Verdict, counts, identities, native evidence, run table |
| `results/candidate-manifest.json` | Digests that bind all result files to the candidate |
| `results/moe.json`, `results/dense.json` | Per-node results of the accepted and corroborating runs |
| `results/pb-receipts.json` | PrismaBuild facts for every action: state, host, memory peak, CAS receipt digests |
| `results/log-digests.json` | sha256 of every retained log and of the harness files |
| `results/logs/<run>/` | `junit.xml`, `pytest.log`, `surface.json`, probe records, build digests, run manifest |
