# Empirical expert calibration byte hashes (#2102)

Refs #1301. Inspected base `693a38f3ae3467ac3234cd9a7d756e41b69c3242`;
branch `sol/pq-expert-calibration-digests-20261002`.

| Consumer | Existing input-byte recipe | Identity use | New owner |
| --- | --- | --- | --- |
| `expert_empirical_cost._expert_checkpoint_identity`, calibration `sha256` | detach, CPU transfer, contiguous, uint8 view, NumPy C-order bytes | load-bearing checkpoint/resume identity | `digests.bytes_sha256hex` |
| `expert_empirical_cost.main`, provenance `calib_sha256` | CPU transfer, NumPy C-order bytes without detach or uint8 normalization | persisted calibration provenance | same exact-byte owner |

Only the final SHA constructor is shared. Shape/dtype metadata, calibration
hash, identity field order, full64 lowercase hex, source identity validation,
canonical normalization, checkpoint policy, CLI pickle emission and the native
extraction refusals remain local. The CLI still refuses grad tensors and
unsupported NumPy bfloat16; the checkpoint still refuses a wide scalar uint8
view. Their different input contracts have not been collapsed into a tensor
normalizer. No numeric metric, JSON profile, cache, format, input, pin, serving
gate or default changes. No pre-change numeric/digest disagreement was found.

The primitive baseline shrinks exactly two `hashlib` scopes, 497 to 495 at
this base; the same-name, near-duplicate and must-differ fields are unchanged.
Parent #1301 remains open for the full load-bearing/ephemeral/profile census.
The wider domain inventory is in the separate #1303 slice's
`domain_numerics_census_1303_20261002.md`; this document completes only these two
byte-hash sites.

## Validation and retained negative evidence

OLD PB `09d0f29b8873e5859c73370d079e517e8bab84b7c604e47cb4965408da39c80e`:
12 byte/refusal passes plus two expected missing-route failures, zero skips;
all 14 outcomes reconciled. Fixtures use independently packed literal integer
bytes and exercise contiguous, transposed, sliced, int32 and empty inputs
through both existing consumers. The CLI test isolates model/GPU loading and
measurement; provenance construction and file emission are real CPU calls.

Accepted evidence: 101 tests across three successful PB shards
`7a8084b40e5c`, `78c02801cd13`, `f9552fb6ea3c`, plus all seven duplication
ratchet tests at corrected-baseline action `85907ddf87c2`: 108 unique tests
passed and reconciled, zero failures/skips/missing collection in the accepted
set. The initial ratchet shard `6a0f90f127e2` had six passes and one failure
because an indentation guard refused the baseline edit. Its failure remains
retained; only that shard was rerun after the exact shrink. The other three
results are reused with tested/current source/test/owner/pin blob equality.
Two modules compiled through PB action `a06067bdd4ec`.

Commands: published `pbtest.py --checkout <owned-worktree> --python
/home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python --shards 4
--workers-per-shard 2 --threads-per-shard 1 --cpus-per-shard 2 --mem-gb 6
--priority -10 --timeout-s 300 --wait-s 900` on
`test_expert_calibration_digest_owner_1301`, `test_expert_empirical_cost`,
`test_streamed_cost_checkpoints`, `test_digest_profiles_1301`,
`test_cost_stage_checkpoint_manifest`, `test_duplication_baseline`.
The corrected ratchet used one shard/worker/thread/core, mem_gb 4 and a
180-second action bound. `pbrun.py --tag x86 --cpus 1 --demand mem_gb=2
--priority -10 --timeout-s 60` ran the same interpreter's `compileall -q`
over the touched module and new tests.

Environment: dl380g10, CPU, Python 3.14.4, torch 2.11.0+cpu, Transformers
5.16.1. PB verified installed Git provenance for PB `95a59051...` and Tessera
`b40c93cb...`. Terminal/log/CAS evidence and five successful hashed claim/
payload checks were consumed; claim verification excludes worker attestation.
Thirty-two tested/current source/test/owner/pin/final-baseline blob comparisons
are equal. Cgroup CPU/memory and Netdata host windows are retained in PB
endings; absent dl380g10 pqteld CSV is recorded. There is no performance,
GPU, served-artifact or full-suite qualification claim.

Artifacts: `/home/rob/tmp/astra-resume-20261002/pq_dedup/` contains
`digest-red.json`, `digest-green.json` (including the initial ratchet failure),
`digest-ratchet.json`, `digest-compile.stdout`, corresponding stdout/stderr,
typed `digest-*-action.json`, `digest-claims.json`, `digest-source-binding.json`
and `digest-proof.git`. The failed ratchet is superseded by the corrected
ratchet only; no unrelated test or source result is discarded.
