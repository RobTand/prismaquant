# Stage-A manifest custody slice

## Scope and policy

Inspected repository AGENTS.md, docs/ARCHITECTURE.md, docs/design/design_guidelines.md, the selected consumers, the frozen completion registry, the digest owner implementation, and the existing streamed atomic writer. No owner, shared census, baseline, scanner, acceptance, pin, replay/seed, or chain-receipt file was edited by this slice.

Production change: only `prismaquant/streamed_boundary_artifacts.py`, routing `_publish_status` to the exact existing named owner `prismaquant.digests.indent2_json_file_bytes`. This owner emits sorted ASCII indent-2 strict JSON with default separators and one LF. The consumer still owns status construction, embedded identity/policy/debt, readonly/no-directory behavior, and atomic publication via `cost_stage_checkpoint.atomic_write_bytes`.

`campaign_data_manifests.py` and `joint_layer_quanta.py` are deliberately unchanged. Their three insertion-order calls have no exact existing digest owner. Routing to any sorted profile would change bytes; adding an owner is outside this assignment. These are recorded as explicit route gaps, not completions.

## Independent originals and observations

All three `manifest_original_*.py` sources were retrieved directly with `git show` from research base `6d8288adc94b3c4b624c143ed372fb6513f32784`. Full source blobs are retained; the ledger records Git blob IDs, original SHA-256, current SHA-256 witnesses, and literal registry AST node IDs.

`manifest_original_outcomes.json` contains ten observations captured only by executing those original consumer function ASTs, without rewriting bodies or using a digest-owner encoder as the expected-value oracle. Original compilation controls external campaign discovery inputs and logging seams. Current-side tests import and invoke the actual production functions/method directly with the same controlled inputs. The fixture digest is SHA-256 `179dba54551aa78a191b7a0d326709b13dd00df5a7b99120517bbed5135f1d9f` (17,960 bytes). An independently repeated capture produced the identical digest.

Coverage compares entire serialized and compressed byte strings, not merely object equality or routing: compact insertion-order JSON, pretty indent-1 LF manifests and output rows, deterministic gzip member, ASCII escaping, NaN/negative-zero behavior, invalid manifest/key/value refusals and chaining, dryrun, second-row refusal, partial publication with failure at final output replacement, actual replacement order, preservation of prior status on strict refusal, atomic fsync/replace order, owner/generation envelopes, embedded boundaries, and readonly/no-directory no-ops.

The initial capture attempt failed because changing directory removed the repository from Python's import path; adding the capture root to sys.path resolved that harness issue. No expected outcomes were accepted from that failed attempt. Tests were run exclusively through run_tests. No full suite, GPU, or cross-platform validation is claimed. The existing streamed suite passes with 15 upstream torch deprecation/profiler warnings.

## Capture evidence

Successful original-only capture:

[Test evidence {"artifacts":{"outputs/tests/fixtures/digest_1301/stagea-custody/manifest_original_outcomes.json":{"digest":"179dba54551aa78a191b7a0d326709b13dd00df5a7b99120517bbed5135f1d9f","size":17960},"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"8f41ff641061e949d0ca39e365cebf77bd790c67291dad54bdd7ae165869dd96","size":90}},"operation":"8ab4784b8b4febe6b226b43efc00e22e1959f511481ed79e1e332d565355df36","receipt":"fc63cd4f72cbe945e5bc0fb038003172fa6a999a1dc0c207644568e3c857406c"}]

## Final focused validation

Command: `PYTHONPATH=. python -m pytest -q tests/test_digest_1301/stagea-custody/test_manifest_consumers.py tests/test_streamed_boundary_artifacts.py --basetemp="$TMPDIR/pytest"`

Result: **32 passed, 15 warnings**. This includes 11 new custody tests and 21 existing streamed artifact tests.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"06ba96921ea66f84261fc49f00c4ea4e0592a6852a7b25522faeb286fb7e44b2","size":1228}},"operation":"9d1653b61e8835646d86ad51679a89eb8399d0a59601e4975d6839186efcc2f5","receipt":"ef17c56621f2d78cedaa1e0ed39527c751cb2656e90804254cb442b71fe80682"}]
