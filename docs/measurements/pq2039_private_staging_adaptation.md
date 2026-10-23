# PQ #2039 private staging: historical evidence and bounded verification

## Scope and reuse decision (O2)

Reuse the historical report as evidence for the removed private staging copy,
not as a measurement of the current adaptation. Current behavioral validation
is recorded below separately. No current timings or speedup are claimed.

Reference inspected: `020c6faab70016a53382d0cce0fd0d9bed01e842`,
`docs/measurements/pq2039_staging_copy_2026-10-10.md` (30,967 bytes;
SHA-256 `713c09fa69f75a1195c8bffc4e138515283f2dae4b4e5b05e1ccd0c47a05b2f9`).
The reference includes later behavioral repairs; its paired performance result
remains bound to the older measured tree, not those repairs or today's code.

## Available artifacts verified

Files were read and SHA-256 recomputed under
`/mnt/shared/tessera-measurements/pq2039-staged-pin-20261010/`.
All three result digests match the reference report:

| Result | Bytes | SHA-256 |
| --- | ---: | --- |
| `final.json` | 107900 | `6a3dd79be9de4cd8ae2435abe20970fbe507c1a9cc9149539c09683a6a4fc40e` |
| `final-v2.json` | 124790 | `1717594007623a914429ec59ff7d19cac43719f8a3966b81bc982f00c6f3d730` |
| `final-v4.json` | 118899 | `792f3abf2af14f767b312adcfc3809dd9c04d689fc39db52ccbc8e5e03a99e91` |

All eight `final-v4-traces/` files match both byte size and SHA-256 recorded
in `final-v4.json` (first four staged ABBA, last four unmapped control):

| Trace | Bytes | SHA-256 |
| --- | ---: | --- |
| `before-1791601595.json` | 16341650 | `ed9f08863360d503e56b0f6ede902e45a4210f30fbe1a59b3d7f5d16de0c3bbf` |
| `after-1791601626.json` | 16515317 | `f2d7460773f395138ac856870ca86e34974bec2dbd797964cd55e9e550c4471b` |
| `after-1791601648.json` | 16518334 | `5b6787d56d111153b49e2da03043a3db114e641c349576913badae5ab0af47ea` |
| `before-1791601670.json` | 16352043 | `ae4478b0f47bf9555e54e9ec0dc1345280e3937f9a7e48b3c7b959467a2f3433` |
| `before-1791601702.json` | 4218375 | `739bceae86cf80f83aaa6a68f429507b50a5605d6429b9882f8a09b718579a14` |
| `after-1791601704.json` | 4738380 | `ed05d7d028e884a3a3e0fafc25d88ac4b6e7776012107854582156109d45db32` |
| `after-1791601706.json` | 4739170 | `6551017b7b383fa42f41f04dec6fdd5231c291943b465a2dccbe4f4ce5a1629d` |
| `before-1791601708.json` | 4302146 | `5d4cf0e8a90c07b02573c911572fd322e807c6857e413a0d43ca5d0421db9b2f` |

### Narrow binding and result checks

- The v4 record names action
  `b54f384bd46ae0bf31651b0f34825fa8a00cc5c0262db2af82570c6dba7b347c`,
  historical GB10 host `sparky`, Python 3.12.3, torch 2.11.0+cu130.
- `binding.parent` and the embedded `pbrun_closure` head both name measured
  commit `27a6385dbd237427522fae2d86bb2e3ebe02832b`. The closure's
  `dirty_sha256` is the empty-input digest
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
  Distinguish this from snapshot `binding.head`
  `290cb9f7cdd00169a6880613a7a6361ac9f5629b` and top-level
  `binding.dirty_sha256`
  `da2b67275a8f2643aeedbc8a2ae27b1cf2af48e4e7e46dfc6ff6ed429e98fa9b`;
  do not describe every dirty field as empty.
- Record fields confirm ABBA, 426 passes per arm, 32 BF16 units of
  `[2048, 4096]`, 16 MiB each. Each staged arm records 230,854,492,160
  total staged bytes, zero pool bytes/fallbacks, matching its expected bytes;
  `stage_gate_problems` is empty. `netdata.complete` is true in the record.
- Historically recorded pooled medians are 73.16 to 49.90 ms/pass (31.8%
  reduction), with private staging copies 27,264 to zero. Recorded profile
  `aten::copy_` counts are 195/99/99/195 for staged ABBA, versus 195 in
  every unmapped control arm. The before profile attributes 96 copies to
  `_prepare_device_projected_check`; after profiles omit that site. These
  are historical observations, not a fresh profile analysis or current result.

Digest agreement establishes artifact identity and internal binding consistency,
not independent attestation of execution. No scheduler receipt, action exit/time,
source closure contents, SDK bundle digest, or raw Netdata validity was independently
revalidated here. Earlier result files were digest-checked only; earlier traces
were not checked. Historical test receipts in the reference were not replayed.

## Avoid obsolete SDK assumptions

The v4 record explicitly bound SDK 5 source at `027103d9...`; the report's
SDK-6 refusal/live-reachability discussion describes that historical environment.
Neither that SDK requirement nor that refusal is a claim about today's runtime.
Do not copy its interpreter paths, sealed bundle workaround, host, or SDK pin
into current application logic or a new test recipe. Use the current project's
configured dependencies, validator and test fixtures; report unavailable
compatibility as a limitation rather than bypassing admission. Reusing the
historical report requires no historical SDK installation or timing rerun.

Keep energy/work-per-joule and cross-host clock alignment on HOLD. The old
local-disk warm-cache fixture does not establish NFS, qualified-original-owner,
full-campaign, export, serving, KL, bpp, or current performance behavior.

## Pre-GPU decision (not executed)

Use targeted staging/preparation and projected-unit checks with the smallest
existing safetensors fixtures (for example the private-staging acceptance
fixture's 2-by-3 BF16/FP32 payloads), not the historical 512 MiB pass, a model
download, or a timing campaign. Relevant files are:

- `tests/test_projected_private_staging_acceptance_2039.py`
- `tests/test_projected_staging_copy_2039.py`
- `tests/test_projected_preparation_2039.py`
- `tests/test_projected_unit_check_1935.py`

CPU checks can establish allocation/copy ordering, metadata refusal before
allocation, destination ownership, fallback behavior and diagnostic ordering.
The acceptance fixture explicitly emulates pinning; CPU success cannot establish
real page locking, asynchronous H2D completion/lifetime, or CUDA cancellation
and draining. A later admitted targeted GPU run should check those boundaries,
adoption without a second host copy, and unchanged equality/first-mismatch
semantics. Expected budget: **at most two GPU minutes**, an estimate rather than
an observed duration. No GPU was requested or run in this research task.

## Current execution evidence

Implementation call sites: `source_unit_weight(pin_memory=True,
expected_metadata=...)` checks same-handle metadata before fetching payload.
`StagedShardReader._read_payload` allocates fresh private pinned uint8 storage;
`_read_span_into` sends preadv directly into that storage (kernel file reads
remain). `_prepare_device_projected_check` adopts its contiguous typed view,
removing the second user-space payload copy. The ordered CUDA launch still
performs H2D and device comparison copies; this is not attribution of every
microsecond in the original 1.043855-second aggregate. Unmapped, pool and
qualified-original reads still pin/copy privately. Pin failures clear source
aliases before retirement, including when exceptions are retained by futures.
Host consumers retain the pageable default. No cache or second executor is added.

The adapted harness rejects units below three during argument validation,
before CUDA probing, fixture creation, timing or profiling. CPU tests cover
both rejected sizes and defensive duplicate-pick refusal/restoration. It uses
the current guarded client SDK, not the historical SDK5 binding. No new timing
comparison was needed or attempted; matched ABBA timing/profiles and both-host
Netdata remain available for a future separately justified paired action.

Current commands and observed outcomes:

- `python -m pytest tests/test_projected_private_staging_acceptance_2039.py tests/test_projected_preparation_2039.py tests/test_projected_unit_check_1935.py tests/test_residency_shard_reader.py tests/test_tessera_expert_projection.py tests/test_projected_staging_copy_2039.py -q --basetemp="$TMPDIR/pytest"`: **144 passed, 14 skipped** on CPU. All four formerly failing acceptance cases passed; emulated pinning is not GPU evidence.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"8aeee91563a9d94da2dc488ce93632e43a0f1b4c8ad3a6ab4a4174224e45e751","size":442}},"operation":"1a4e328fb022010ac21b246524de634115e1d1833a65f4aae79d4ebe898120c0","receipt":"cfb3a6d81bf730f5ea8aee494756fcf619a74823251bbfcbb85a924258b16b05"}]

- Targeted admitted GPU action, following the pre-run decision above:
  `python -m pytest tests/test_projected_staging_copy_2039.py tests/test_projected_preparation_2039.py tests/test_projected_unit_check_1935.py -q --basetemp="$TMPDIR/pytest"`:
  **111 passed, no skips, 8.72 seconds** on CUDA torch 2.11.0+cu130.
  This exercises real pinned allocation/adoption, completion, cancellation and
  ordered mismatch checks, not empirical performance or full-campaign safety.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"7c81694274564dc5de78a73955fa6cedd10d705b2d61f6f4cd9b463b3d118920","size":354}},"operation":"a65cb2f7037b98fed9644816f32abd5bcfa8832daa490dff38c557d13fecd373","receipt":"5882f8ba50571830d076eb19d61db638448033420eea671246176b634406cff1"}]

- Final broad CPU staging/tool, original-material, authentication, I/O-site,
  dependency/import, duplication and documentation guard selection:
  **700 passed, 12 skipped, one explicitly deselected**. Selection and full
  command are retained in the receipt. The initial broad run failed the
  untouched `test_source_consumers_restore_the_public_plugin_import_graph`:
  its nested pilot-result fixtures require `published_unix`, absent from the
  installed SDK result (`KeyError` in `test_pilot_verified_result_1293.py:135`).
  That unrelated outer test was deselected for the final rerun, not fixed or
  presented as passing. The initial failed run remains evidence.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6ca46ca5d4e681a2d5c01c0d3c90a74c930bfe5686f964202dc576e94a079e1b","size":140501}},"operation":"22528d26af3aa30eda72d67fb6f9843dca1866e8a9e84067405b4cfe0193817d","receipt":"b589a6d1de357fd2b6b821ef554bc0a9cdaead9da451204c14fa6d9d26044e50"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e338b1e58c32207cfc4e3d758f3d4786f2b17549756429fb125539ce72e9706e","size":2275}},"operation":"448faca2606e1e8e613e0d689d3c1e21803437cb50f25aee64553a85f8963a2a","receipt":"6d55ab238ecb0ffba8fb23c277f909b07977ab64a49b323856b503abcc06efd4"}]

- Final exact O1 acceptance command (the first CPU command above without
  `tests/test_projected_staging_copy_2039.py`): **112 passed, 12 skipped**.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"2770215a1430a0732e518b3816d76bd8a9cbb1feb1f61e10cab7c2c554ef23a0","size":362}},"operation":"c1143714ca47f44fc3d74baa5a9f16db9522c31a9efc8a28e900db9f21404358","receipt":"f137d4a566bf2882ea4420dab3cbd80982e19c66b5d02c4d4fecd7c475243e8f"}]

`git diff --check` passed. Clock alignment, energy/work-per-joule and
full-campaign qualification remain **HOLD**.
