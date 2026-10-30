# D44 retained-scope repair (#2446)

This is a repair of the materialized D44 controller, not a replacement controller
or a change to the active campaign, scientific inputs, source counter, accepted
lifetime seed, or native queue ownership.

## Refusals and owners

- Fit, held-out and selection identities come from the current task's frozen
  reads, not a new roster producer. Tensor filenames use the full qname. Each
  role has one unambiguous frozen read.
  Both authoritative and encode/HELD values must be lowercase hexadecimal
  SHA-256, not merely strings of length 64. Scope is checked before replacement
  and shape validation and before any task can be removed.
- Encode selection must be present and match. Historical HELD files bind that
  selection through the verified replacement receipt path and SHA-256. HELD
  conditioning, when supplied, must also contain a matching selection. A supplied
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
  Legacy wait prompts use the current controller's resolved script path with
  `--legacy`, not a fixed deployment script. No-argument invocation still refuses.
  Routed state defaults to `d44-wave-state` in the caller's launch working directory,
  with `wait.json` beneath it, or explicit `--state-dir` / `--wait-file` paths.
  State, wait, plan, ledger seed, retained manifest and disk-target paths are resolved
  there before opening custody or persisting a shell-quoted resume command.
  A configured `D44_DISK_TARGETS` fallback is captured as an absolute `--disk-targets`
  argument; explicit CLI targets still win. Replaying a newly generated handoff from
  another working directory reopens the same lock, catalogue and custody history.
  No stored state is migrated or reset. Previously generated relative handoffs need
  regeneration from the original launch directory (or known absolute paths).
  The configured native runtime and stores must remain available when resuming.
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

Previous repair's corrected CPU command (after Python AST syntax checks of the
controller and proof scripts, without executing the standalone proof):

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

### Recovery-handoff follow-up

The handoff regressions first ran against the pre-fix `9afbb89c` controller:

```sh
python -m pytest tests/test_d44_runtime_safety_2446.py -q -k handoff --basetemp="$TMPDIR/pytest"
```

Observed: **4 failed**, 19 deselected, exit 1. Default-state replay used a different
state directory; relative explicit/environment-path replays could not find the
plan; the legacy prompt lacked the required mode and exited 2. These failures
were reproduced before changing the handoff implementation, not inferred from
source inspection alone.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e9e79aa47acaca5b81522e3035113669ab0caf2fd6b897b1c1523663b830000b","size":39495}},"operation":"5ba23e0e5a9c163aac4b61cc104cba5d1c2d0bd2e6b0251282d58b13a97ca88e","receipt":"6c14c1361ae7aae444afb5b11b60e49083a7ce76969f0f576ee216f92e8844ca"}]

After the fix, all five D44 suites pass, including the four new handoff cases:

```sh
python -m pytest \
  tests/test_d44_retained_scope_2446.py tests/test_d44_native_catalogue_2449.py \
  tests/test_d44_native_controller_2449.py tests/test_d44_native_phase_2449.py \
  tests/test_d44_runtime_safety_2446.py -q --basetemp="$TMPDIR/pytest"
```

Observed: **88 passed**, 14 torch/Python 3.14 deprecation warnings, exit 0.
The new tests replay the persisted prompt from another directory with spaces in
paths, covering default and explicit state/wait paths, relative optional inputs,
CLI precedence over environment disk targets, and environment-only disk targets.
They exercise the real CLI, custody restoration and ledger path reuse, with native
catalogue creation/controller execution mocked. Catalogue creation occurs once,
existing custody and lifetime ledger bytes remain, and no state is created in the
second directory. The legacy test follows the saved prompt through the parser to
a mocked legacy controller, using a configured script path containing spaces.
The existing no-argument refusal remains covered. No native worker or GPU proof
is claimed by these tests, and the source counter and accepted controls are unchanged.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f95f2ed9f6b98a4ed2d71e9c43fc7ee6b4122fc25172e60c57dfd2728ab4a4fe","size":902}},"operation":"c6b85b5403bdeff429ba103f6a696189ab60b0cdfa02b6e13694f97dfd4a49a2","receipt":"83f3fb289404af33169e21f8f7359e32a94bd401baef46f097b283d9a8ae26ca"}]

Repository guard command for this follow-up:

```sh
python -m pytest \
  tests/test_architecture_doc.py tests/test_doc_claims_driver_truth.py \
  tests/test_docs_entrypoints.py tests/test_docs_staleness.py \
  tests/test_duplication_baseline.py tests/test_prismabuild_boundary.py \
  tests/test_tool_imports_1304.py tests/test_file_hash_routing_1672.py \
  tests/test_tools_hash_routing_1680.py tests/test_digest_hex_1457.py \
  tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py \
  -q --basetemp="$TMPDIR/pytest"
```

Observed: **533 passed**, 14 torch/Python 3.14 deprecation warnings, exit 0.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6483945c72e3948a98b81bb688ca7ea48e939b6745e17d1d6a4c9a86d7a3b10b","size":1383}},"operation":"337dfe7ae44e57a720a48e9a2ab342f4e4afa7912ecc4429b623f6b00a434120","receipt":"f980e2f74fa5e9c3a3d31fd6c6e010804edf2116146ddf78ab8e65473ae34ed2"}]

## PRI-101 integration, 2026-10-27

PRI-218 authorizes this source repair without restarting the active campaign.
The delivery reuses #2976 at `1de5bf4b7246835ee331ead7b47f88d52f9db042`.
Its merge-base with precursor #2599 is #2599's exact head,
`fa59b9705fcd8c6f65d0ccc9bd50229bc7d09ac9`, so the precursor is fully inherited.
Integration onto main `05b62b4742793a2188abd4ed041fc4d46fd6c2c0` changes only
conflicting changelog and architecture additions. All five D44 test files,
controller and proof scripts, lifetime seed, and source-evidence bytes equal
the preserved source. The task's `verify_preserved.py` repeats that comparison.
Historical failing-before actions above remain historical evidence.
Fresh integrated-head fleet checks and independent review are separate gates;
this note does not claim either has passed.

## Retained-input compatibility repair, 2026-10-27

Independent review of PR #3063 at `9923abf268c7d6d1935c616feaf21733d29c78ac`
found that the fixtures shortened frozen tensor filenames and invented HELD
conditioning. All three retained task records use full-qname tensor filenames.
Their historical HELD files omit conditioning and bind the encode receipt by
path and digest. The existing adoption gate verifies that relationship against
the manifest's content-verified encode reference before any task is removed.
The corrected selection comparison preserves that owner and validates optional
HELD conditioning when present. Historical files, lifetime seed and source
evidence are unchanged.

Regressions at `b124c38006` precede the correction. PB action
`8e8876fb25e616cdca87d4a6ac7b755653db60be4c1712e9f82da9380c4e5bd0`
ran the two isolated regression methods on dl380g10 and ended failed/1:
`2 failed in 1.57s`. The failures reproduce unresolved full-qname scope and
missing HELD selection. The original review probe remains available as action
`6b451f9398c2c3d072f89db56fe7b8a5bdafc9b95895214da6fbe39bd4aa9261`.
These are CPU metadata checks, with no scientific or performance qualification.
