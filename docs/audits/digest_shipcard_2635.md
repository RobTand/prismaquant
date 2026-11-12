# Shipcard digest reconciliation for #2635

Inspected main: `05b62b4742793a2188abd4ed041fc4d46fd6c2c0`.
This slice reconciles the three primitive shipcard calls in `refresh_2540`.
Root #1301 and shipcard's supplemental JSON contracts remain open.

## Existing implementation

The original private `_file_content_sha256` streamed SHA-256 in 1 MiB blocks.
Its source is blob `acf1b46ab0802eada0512a22b78828bd9420bd8a`, recorded by
`refresh_2540` at commit `30b6231df000726beda55fb6706e641a5cb2ccb0`.

Commit `60060a69c04e5d8e8d1970eca9746093b25c0d4e` removed that helper and
routed its callers through `digests.file_sha256hex`. That commit is an ancestor
of inspected main. The current callers are `compute_model_sha`,
`build_weight_content_manifest`, and `file_sha256`. The last retains its
error-to-None wrapper. The shared owner opens once and uses its 8 MiB default.
This reconciliation preserves the existing read size and changes no runtime code.

Historical PRs #2643, #2784 and #2824 closed without merge. Their bodies are
not merged proof. The source commit above establishes the current routes.
The reconciliation does not endorse that commit's unrelated ship-gate changes
or claim that its complete model identity was unchanged.

## Reused compatibility evidence

The 40 additional golden rows come unchanged from commit
`a9d2c77ce8bcf7cccec5d246dbd655e80db9d44d`, the original #2643 delivery.
They append to `digest_profiles_1301_files.json` without changing existing rows.
Their Path and string inputs cover empty files, one byte, boundaries around
256 KiB and 1, 4, 8 and 16 MiB, missing files, directories and a symlink.

The historical recording receipt has not been re-attested. Replay therefore
also runs the original seven-line algorithm copied from the recorded source
blob. It compares returned digests, exception types, messages and causes
against the current binding. Runtime tests need no historical Git object.
The wrapper comparisons require None on those same read errors.

`test_shipcard_digest_routing_2635.py` checks actual model and manifest results,
owner calls, one-open read boundaries, and the wrapper's exception handling.
Its model fixture exercises the supported auxiliary-file path. The manifest
fixture contains small file bytes and does not qualify a serving artifact.

## Retained descriptor contract

The two raw constructors in `_verify_open_safetensors_fd` remain intentional.
The function's AST is identical to its `refresh_2540` source. It borrows a
descriptor, consumes the prefix, header and contiguous tensor spans once,
feeds container and tensor hashes from the same reads, then checks one final
stat fence. A path-opening file digest cannot replace that operation.

The real-file test forces partial reads, checks offsets, full and tensor
digests, byte and call counts, and the still-open descriptor. A deterministic
mtime change checks refusal at the final fence. No timing assertion is used.

## Shared records and limits

The census adds a separate `reconciliation_2635` section. Historical refresh
rows remain unchanged. It binds the three historical IDs, current source,
successor consumers, retained contracts and fixture provenance.
The committed baseline already lacks the removed helper and retains the
descriptor verifier. This slice changes no baseline row.

Independent review must accept the current-source reconciliation before #2635
closes. Root #1301 still needs its other existing owners and independent #2544
audit. No performance, GPU, numerical or serving qualification follows.

## Fleet validation on 2026-10-27 UTC

The implementation/test/census tree is commit
`87e93461385cd087fc3f5b69f8867e7c7a8ef2dd`. This validation appendix was
added afterwards. Selected source bundles match every executed test file,
fixture, census, owner and production module. Five earlier shards precede the
extra owner-binding assertion in the new test file; its six tests were rerun
at the final implementation bytes. Full worker attestation remains unverified.

The selected actions pass 166 tests. Terminal status, complete cleanup,
source bundles, log hashes, receipt bindings and payload hashes were checked.
All seven read-only claim checks returned `checks_passed=true`.
Evidence and copied terminal logs are under
`/mnt/shared/paperclip-runs/PRI-225/shipcard-2635/`.
`verified-evidence.json` records original log paths and exact source identities.

| Check | Action | Outcome | Copied stdout log |
| --- | --- | --- | --- |
| `tests/test_digest_profiles_1301_files.py` | `382d8d8b975db57571d599fb6c4b8476d3ee954e909a79920bc9b56cd1bb79f7` | 65 passed | `382d8d8b975d.stdout.log` |
| `tests/test_shipcard.py` | `cfb3ff01ef9f5bb08fa6b862cce9b77b92260b48c09927b862bc38f5eeae713b` | 57 passed | `cfb3ff01ef9f.stdout.log` |
| `tests/test_shipcard_integrity.py` | `da223e7aca7ed1c1a6390985391b105f0bb1ee1396e0525d57b5f5e93594a77b` | 9 passed in 0.75s | `da223e7aca7e.stdout.log` |
| `tests/test_exact_activation_entry_read_amplification.py` | `77734e7956a106bfc5812a845319025e5ab04a87e34ecc586b80763947592fb8` | 22 passed | `77734e7956a1.stdout.log` |
| `tests/test_duplication_baseline.py` | `86ed2d47a583ba455896d6346e6d44c7fc7389b6b8f04eb3b61154780e5c09bf` | 7 passed in 16.16s | `86ed2d47a583.stdout.log` |
| `tests/test_shipcard_digest_routing_2635.py` | `2d3d1a6df2e92d8c4082e3a71d9dbba7027503812b26d9e3ac70a2c41c66398f` | 6 passed in 1.31s | `2d3d1a6df2e9.stdout.log` |
| Compile, recipe/source checks and baseline regeneration | `13e35110b794cef95924ed3502477f016c6392dc5717307e8caad61c20b49256` | Passed | `13e35110b794.stdout.log` |

The initial file-matrix action
`db81e1b91e36c0f9b853e52835096353d85fc312e64efa336356faa690659fb1`
reported 20 missing-fixture failures and 25 passes. This demonstrates the
missing evidence, not a production hashing defect. No golden recording
variable was enabled. The 40 added outcomes were copied, not regenerated.

Two earlier source-check commands, `9989292e7a69` and `3767587da4b1`, failed
my byte-order and list-order baseline assertions. The final command confirms
identical scope/helper membership after regeneration; existing lists reorder.
The committed baseline remains byte-unchanged. These command failures do not
represent repository test failures and are retained in `diagnostic-endings.json`.

The six-file test run used transient unit `pb-pri225-shipcard-green-v1`:

```sh
python3 /mnt/shared/prismabuild-fleet/repo/tools/fleet/pbtest.py \
  --checkout /home/rob/paperclip-worktrees/prismaquant/PRI-225-shipcard \
  --python /home/rob/venvs/pq-pin-aaa41700-pb20261016-git/bin/python \
  --cpus-per-shard 1 --threads-per-shard 1 --mem-gb 4 --timeout-s 300 \
  --json /mnt/shared/paperclip-runs/PRI-225/shipcard-2635/green-v1.json \
  tests/test_digest_profiles_1301_files.py \
  tests/test_shipcard_digest_routing_2635.py tests/test_shipcard.py \
  tests/test_shipcard_integrity.py \
  tests/test_exact_activation_entry_read_amplification.py \
  tests/test_duplication_baseline.py
```

Unit `pb-pri225-shipcard-green-v2` reran only
`tests/test_shipcard_digest_routing_2635.py` with the same options and
`green-v2.json`. The source check is retained as `source_check_v3.py`; the
exact admitted `python -c` argument is in its CAS request. All actions have
ended. No fleet work remains pending for this slice.
