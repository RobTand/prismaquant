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
