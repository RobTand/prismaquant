# D44 retained-scope repair (#2446)

This is a repair of the materialized D44 controller, not a replacement controller
or a change to the active campaign, scientific inputs, source counter, accepted
lifetime seed, or native queue ownership.

## Refusals and owners

- Fit, held-out and selection identities come from the current task's frozen
  reads, not a new roster producer. Each role has one unambiguous frozen read.
  Both authoritative and encode/HELD values must be lowercase hexadecimal
  SHA-256, not merely strings of length 64. Scope is checked before replacement
  and shape validation and before any task can be removed.
- Both encode and HELD selection values must be present and match. A supplied
  resume manifest must match the stored adopted population before native-tool
  loading or controller construction. Matching recovery leaves native admission
  custody intact; it does not generate replacement receipts for retained work.
- D32: scope and resume-population refusals are **correctness comparisons against
  admitted frozen input bytes**, not run-identity seals. They apply in dev and
  certified modes. The non-hex regression cases exercise both modes.
- Native occupancy owns ready/claimed/finish-cleanup custody; `current_ending`
  owns terminal-generation selection. Unknown occupancy or ambiguous/unreadable
  endings cannot establish completion. A historical `done` cannot hide a newer
  `failed`. GPU cost recovery uses `queue.records.item_path`, retains reservations
  on unknown attempt history, and does not reset the lifetime balance.
- File bytes use `prismaquant.digests` byte/file owners. Proof value identities
  retain the compact `DIRECT_ASCII_LAX` contract; CLI/report text retains
  `DIRECT_ASCII_SPACED_LAX`. No digest/boundary baseline was expanded.

## Configuration

The controller no longer selects a mutable deployment checkout or guesses a
fleet from an architecture tag.

- `PRISMABUILD_READER_HELPER_ROOT` (or the existing PQ native-generation discovery
  mechanism) names the native generation. Public `prismabuild.client` is loaded
  from its `src` tree, and the public `pbcampaign` tool from its `tools/fleet` or
  legacy `tools` layout. Divergent preimports refuse; temporary import paths are
  restored. PQ's separate staged-residency SDK pin is unchanged.
- `D44_WAVE_QUEUE_DIR`, or the launcher-published queue root, names the queue.
  `D44_WAVE_CAS_DIR` names the CAS. Native preparation must agree with these
  stores; the controller does not silently publish into another native default.
- `--disk-targets FILE` or `D44_DISK_TARGETS` declares the D1 inventory hosts and
  actual scratch/output mounts for each exact native placement. For example
  (illustrative names, not deployment defaults):

  ```json
  {"targets": [{
    "placement": {"required_host": "worker-17", "required_tags": ["accelerated"]},
    "hosts": ["inventory-17"],
    "paths": ["/scratch/jobs", "/results"]
  }]}
  ```

  Declare all eligible hosts and scratch/output filesystems for the placement.
  Missing, duplicate or unknown mappings refuse before D1 can authorize admission.
  `D44_DISKCHECK` selects the check command (default: `fleet-diskcheck` on PATH).
  Freshness, complete host/filesystem coverage, projected space, inodes and pass
  checks remain mandatory. This is configured declaration, not filesystem discovery.
- Every CLI invocation is parsed first. `--help` exits without submission;
  unknown/incomplete arguments refuse. Legacy submission requires exactly
  `--legacy` and configured stores; it cannot be combined with routed flags.
  The existing legacy campaign/exec environment overrides remain available.
  Routed state defaults to `d44-wave-state` in the caller's working directory,
  with `wait.json` beside it, or explicit `--state-dir` / `--wait-file` paths.
- The optional standalone CPU proof now takes `--disk-hosts` and `--disk-paths`
  matching its read-only service's actual declaration. It is not part of the
  checks below and was not executed in this repair.

## CPU verification

Only bounded CPU pytest checks were run, through the admitted `run_tests` path.
No private executor, standalone autonomous proof, read-only disk service, GPU,
served model, science replay or quality measurement was run. Exact-head external
review and publication remain pipeline actions, not claims made by this document.

During repair, a recheck after initial fixture migration still produced 47 passed
and 4 failed: production live recovery still called missing `PoolQueue.item_path`,
and digest-owner / PrismaBuild-boundary guards identified the added sites. These
were repaired rather than waived:

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b07bde2e7c963c35189ea984c2a09dc1ffe1cd43dd5a14903375d0eb84110596","size":10002}},"operation":"5f33098a7c60c98f56629ef4012d7104a8e29420725150fd478097f387fe352c","receipt":"d59b9c4299396c71b07f91f615737a8d67a7511f880f17ffb9f7b9e70f39be2a"}]

The issue's historical `645e9c2b...` failing-before proof predates this checkout;
it was not replayed. The scope tests now directly assert the previously reported
non-hex refusals. No new exact historical-head proof is claimed.

Final corrected CPU command (after Python AST syntax checks of the controller
and proof scripts, without executing the standalone proof):

```sh
python -m pytest \
  tests/test_d44_retained_scope_2446.py tests/test_d44_runtime_safety_2446.py \
  tests/test_d44_native_catalogue_2449.py tests/test_d44_native_controller_2449.py \
  tests/test_d44_native_phase_2449.py tests/test_architecture_doc.py \
  tests/test_duplication_baseline.py tests/test_prismabuild_boundary.py \
  tests/test_tool_imports_1304.py tests/test_file_hash_routing_1672.py \
  tests/test_tools_hash_routing_1680.py tests/test_digest_hex_1457.py \
  tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py \
  -q --basetemp="$TMPDIR/pytest"
```

Observed: **605 passed**, 14 torch/Python 3.14 deprecation warnings, exit 0.
The D44 suites cover non-hex scope values on both sides and in both modes,
resume refusal before construction, newer-failure/older-success generations,
ambiguous/unreadable endings, cleanup occupancy, live partial-cost recovery,
matching restart without duplicate admission, ship/memory/disk/lifetime gates,
help/invalid CLI invocations, and configured host/mount selection. These are
bounded CPU proofs, not a whole-repository or native-worker execution proof.
No source counter, lifetime seed, boundary allowlist or duplication baseline was reset.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f79a2dca3683c2eb72dcaaa59b6b7122a779a32e3a7bef06075f9fbbdf141794","size":1535}},"operation":"607b48a51f9d93d6e0678d6a9970f850d2287fe6996c5e43450da752a719d9fd","receipt":"8bdbf339aef2826854aa4c07c5465d812716c3dff6bd97860f00ffb475b5dfe2"}]
