# Stage-A custody digest slice

Research base: `6d8288adc94b3c4b624c143ed372fb6513f32784`.

## Changes and boundaries

Three frozen primary calls migrated: replay `_framed` and `roster_sha256` use
`prismaquant.digests.length_framed_bytes_sha256`; resume `_read` uses
`prismaquant.digests.file_sha256hex` with the original 1 MiB block size.
The replay binary/hex split and both domain prefixes remain unchanged.
The resume stat fences, read ordering, second hash, refusal ordering and
bounded JSON loading remain unchanged. Its original descriptor lifetimes
are preserved; this is not a TOCTOU redesign.

`stage_a_chain_seed.py` is intentionally unchanged: the tensor owner copies a
whole tensor to CPU, not bounded host chunks. Two supplemental JSON calls
remain: all available compact UTF-8 JSON profiles sort keys, unlike the
original calls. No exact replacement route was found. See
`ledger_replay_seed.json` for all six original IDs, completed/retained status,
source blobs, exact fixture paths and literal executable test node IDs.

Selected original functions and necessary dependencies were extracted from
the research base with `git show` during development. Tests use checked-in
source only, not git and not migrated owners as their oracle. Stored replay
outcomes were generated from those original functions. Refusal strings and
empty tensor output are also checked against both original and current
consumers. Tests cover ordered Unicode frames, empty streams, permissive
unsorted JSON inputs, replay seal acceptance/refusal, pinned/unpinned reads,
pre-materialization pin failure, stat mutation fences, regular-file and
declared-group bounds, and scalar/empty/noncontiguous/BF16/uint8 tensor bytes
with bounded per-chunk host-copy assertions.

## Observed checks

- First custody run: 6 passed, 1 failed due to the new test calling a
  nonexistent `replay_phase`; fixed to existing `replay_phase_name`.
- First expanded run: collection failed because legacy fixtures load replay
  by standalone file spec. Changed its owner import to absolute
  `prismaquant.digests`, retaining standalone consumer compatibility.
- Final focused run: **79 passed**, 14 existing Torch JIT deprecation warnings.
  Python 3.14.4, Torch 2.11.0+cpu. GPU behavior was not tested.
- `git diff --check`: passed after the production and executable test changes.
- No parent acceptance/scanner/baseline tests were claimed. Shared inventory
  reconciliation is outside this slice's edit permissions.

Final test command:

```sh
python -m pytest -q tests/test_digest_1301/stagea-custody/ tests/test_joint_replay_frontier_607.py tests/test_joint_forward_resume.py tests/test_stage_a_chain_seed.py --basetemp="$TMPDIR/pytest"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"be3e8c67d10856b048220f7a380f740be97e9724a497aaf28b832f1a740ddfce","size":902}},"operation":"dd45b463ca5ef4fdf413741406ea80e1528ca39ef1dc5c585f990ab5d691f77f","receipt":"25fd77cfda52baf584b0adf9ffb9534d6deec23240eec97b78dd0f287ae0075e"}]
