# Stage-A custody integration handoff

Partial completion, not parent closure: the local ledger accounts for all 20 frozen calls in the eleven assigned files: nine completed owner routes and eleven explicitly blocked calls. No blocked call is justified as a permanent retention. The aggregate ledger.json is the handoff; the three shards retain detailed independent fixture provenance and contract boundaries.

An independent read-only source/test audit reconciled all twenty original IDs, inspected actual tests and original source slices, and caught a falsely blocked resume serializer, a stale standalone blocker and an incorrect tensor API reference. These were corrected. Full module fixtures were reduced to selected original source functions/methods and necessary dependencies. Fixture expectations come from original consumers, not the new owner.

## Owner integration needed

No owner/shared census/baseline/scanner/acceptance/pin files were edited. Exact routes remain absent for insertion-order ASCII indent-1 lax publication, insertion-order compact ASCII lax serialization, unsorted compact UTF-8 lax serialization, strict sorted default-repr round trips, strict ASCII indent-1 retirement publication, unsorted CLI JSON, and bounded payload-only tensor hashing. See blocked_calls for exact source IDs, recipes, independent fixtures and executable tests. Adding sorting, changing fallback or whole-tensor host materialization is not an acceptable substitute. The owner integrator must supply exact interfaces before those calls can move.

The baseline integrator must shrink eight genuinely removed scope entries (nine completed calls, two share chain_split.main). The ratchet run reports these stale rows; this leaf is forbidden to edit the shared baseline.

## Verification

Final focused CPU command:

```sh
python -m pytest tests/test_digest_1301/stagea-custody tests/test_joint_replay_frontier_607.py tests/test_joint_forward_resume.py tests/test_glm_joint_data_manifest_at_submit.py tests/test_stage_a_chain_seed.py tests/test_stage_a_chain_resume.py tests/test_stage_a_chain_split.py tests/test_stage_a_retirement_1073.py tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py -q --basetemp="$TMPDIR/pytest"
```

437 passed; 14 existing torch deprecation warnings. Three preceding command attempts used nonexistent test filenames and collected nothing; they are not passes. No GPU, full-suite or parent acceptance completion is claimed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"caa5633f60b0e42af133c9abc1e42093b2f0855e55040cd5d87f6fac6f5aef57","size":1313}},"operation":"3f147451941ff25e9407db21a80c3ee1d1e2e03807f8f31a7af2495a8e4c9332","receipt":"5692d0046e96f30277eb1d1d334d00d2f9baf32ba46f9d7bcb923e74be8778c1"}]

Preservation/ratchet command (before final fixture minimization):

```sh
python -m pytest tests/test_digest_1301/stagea-custody tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py tests/test_duplication_baseline.py -q --basetemp="$TMPDIR/pytest"
```

231 passed; one failure: test_primitive_digest_sites_only_shrink demands removal of eight obsolete scope rows. No additions or baseline relaxation was attempted. This remains an integration dependency, not a passing ratchet claim.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f3817cd0cdb5246e1506cb9a52ceaf338ad8f88f45352eba166bede5c6443fd4","size":3542}},"operation":"b1e440c2d5ecbf98f2f641b01bcdfbfb141f690d8127929bf1434a009e9009a3","receipt":"d16996fc1b8a37d0047f5e79695d9c6eaea355a8723f9575a88c4374928f665d"}]

Final diff inspection and git diff --check passed. Literal executable test node IDs and fixture paths in the aggregate ledger were checked locally. Supplemental/embedded boundaries are explicit in the shards; this is not an exhaustive shell/generated/dynamic-alias census or deployment qualification.
