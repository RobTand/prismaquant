# Stage-A chain custody evidence

## Scope and result

Inspected repository AGENTS.md, docs/design/design_guidelines.md and docs/ARCHITECTURE.md; frozen registry: tests/fixtures/digest_parent_completion_1301.json. Original consumer sources are complete git-show blobs from research base `6d8288adc94b3c4b624c143ed372fb6513f32784`. Tests verify their Git blob identities against that registry. Original outcomes were captured by executing those sources, not by calling a replacement owner to construct expectations. The retained outcome artifact has SHA-256 `f6ebf9faeec7cad60ff27919250222b840700b4ccbdf19755491bcafc5111c27` (3333 bytes).

Ten frozen calls selected: five migrated, five explicitly blocked. See ledger_chains.json for literal original/current executable AST node IDs, owner routes, fixtures, tests, current source blobs and exact retained expressions.

Production changes are restricted to stage_a_chain_split.py, stage_a_retirement.py, joint_prewarm_phases.py and stage_a_chain_resume.py. joint_cost_stage_a.py remains unchanged because its selected calls need unavailable exact owner APIs. No owner, shared census, baseline, scanner, acceptance, pin, replay/seed or manifest-agent files were edited by this slice.

Exact migrations:
- Resume publication: existing indent2_json_file_bytes(record), preserving sorted ASCII strict indent=2 JSON, UTF-8 and exactly one trailing LF. Actual consumer routing is spied in test_chain_consumer_outcomes, alongside frozen original outcomes.
- Split receipt: DIRECT_ASCII_INDENT2_LAX.encoded plus existing trailing newline; stdout: DIRECT_ASCII_SPACED_LAX.text.
- Retirement refusal report: DIRECT_ASCII_SPACED_LAX.text.
- Prewarm digest of raw manifest bytes before gzip decode: bytes_sha256hex, including a checkout-local stdlib-only owner binding for path imports.

Missing APIs (escalated, not eligible retention): sorted ASCII strict indent 1; sorted ASCII strict default=repr (including the JSON loads roundtrip and exception containment); unsorted ASCII lax stdout. Five calls are retained unchanged, with exact AST witnesses. The two supplemental main stdout calls have AST witnesses only, not full main/GPU execution.

Mock boundaries: actual consumer functions run, but PB SDK/publication and split geometry are mocked; no GPU or real reclaim is attempted. Original prewarm capture isolates the separately owned replay parser with a no-replay result. Current package consumer uses the current replay parser. Python -I standalone integration now loads plain and gzip manifests through the actual prewarm consumer and real repaired replay sibling without loading prismaquant or torch. No replay stub is used in this regression. All ledger test references are function-level nodes (no parameter suffixes).

## Historical checks (before audit repair)

- Initial capture failed because frozen prewarm source resolved its replay sibling relative to the fixture directory. The corrected capture explicitly isolates that separately owned parser; no original selected hash/serialization expression was changed.
- Successful independent original capture: operation 67397cc1a0824cc8d29f5905a0314cb714a6e7d012b71bd5fd64cb877937c5b5.
- Initial focused test run: 1 failed, 8 passed. The NaN compatibility case caught an incorrect strict file helper choice for the split's lax receipt. Corrected to existing DIRECT_ASCII_INDENT2_LAX.encoded, preserving NaN and newline.
- Final chain-specific suite: 11 passed.
- Final focused regression suite (chain consumers, existing resume, split and retirement tests): **109 passed**, 14 torch Python-3.14 deprecation warnings, exit 0.
- git diff --check: exit 0.
- Separate full standalone prewarm probe under Python -I: **failed**, exit 1. The other agent's current joint_replay_frontier.py imports prismaquant.digests by package name and raises ModuleNotFoundError when the package is absent. This was a historical cross-slice blocker (now resolved by the sibling loader repair and regression below); it was not hidden with an import stub or repaired outside scope. Standalone prewarm module import itself passes. The replay owner subsequently restored its package-free binding; the current integration regression passes.

No full suite, scanner, acceptance or census checks were run.

## Historical retained test evidence

Initial capture failure:
[Test evidence {"artifacts":{"stderr":{"digest":"401247a74394dd85a6e30d1dbe86c562e0d84a24e328c2e233ed42abb08e8fc1","size":1305},"stdout":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0}},"operation":"f5805937f719745d46d9e47ded82494227bddbc091121eca58fee9545b674651","receipt":"8ae851bdb58ef036f7dc7f761d7ac47accdd42d1778f7a6b787515bafa746365"}]

Independent corrected capture:
[Test evidence {"artifacts":{"outputs/tests/fixtures/digest_1301/stagea-custody/chain_outcomes.json":{"digest":"f6ebf9faeec7cad60ff27919250222b840700b4ccbdf19755491bcafc5111c27","size":3333},"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"fcbab8595279c0595edc3d23545524ef1c667be336dab3a98745220d92ee91cb","size":93}},"operation":"67397cc1a0824cc8d29f5905a0314cb714a6e7d012b71bd5fd64cb877937c5b5","receipt":"2333df5dd24597456195131f7bc71a8f2cd066f5c73f2a92c2ac73f1fc8d5737"}]

Initial compatibility test failure (subsequently fixed):
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"15f930d2ae2561c324ccd84c764aeb956ddda14a931b45b6feae061c77739e9f","size":4938}},"operation":"20be6b63e8dc9b7bec43b88a0103b81c959bca006916fc43f09de5020e23e680","receipt":"a0a831e49dd347480c89522481b3080a42fb30fdf3f058980ab7599b5c7774d6"}]

Final chain suite:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"2a4468ea9177132380692d8e0dc639abd3abe997759b0597c9329b1087b5d150","size":269}},"operation":"4d089629c0df3289bfcaaa9f67b120eda0343aa0dcf8173ceee388fcd203d945","receipt":"e2a3372bbf5a672d8abb053748f1c568cfd99c5eed7c6fe158a8bffa481deed6"}]

Full standalone dependency probe failure:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"76db155fa304aa958d84eed2222841d3fc608f4b4ca3e6448a0667c4a2a02798","size":1139}},"operation":"ec51dad09b1f8f1244028fb469c215d803e458d1482c709200678b6b41a558ab","receipt":"309fb29d100ca8d0e7fabdf32ec2588b610d3b4f0a26571d2e094559cee0ed81"}]

Final focused regression command: `python -m pytest -q tests/test_digest_1301/stagea-custody/test_chain_consumers.py tests/test_stage_a_chain_resume.py tests/test_stage_a_chain_split.py tests/test_stage_a_retirement_1073.py --basetemp="$TMPDIR/pytest"`
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f8bdaaef8068e28a458c7e3c1b0fd41bc45fb2ad2661df7a316fef1739bed632","size":903}},"operation":"56d0f65cfab076e972b04a41af299fef6f5bab21ff627ebd3511ebf6e96cbe9a","receipt":"95c25e54201431a1f5020e03c95f495517a8789e5899bb9f1a6769309fca58c2"}]

## Audit repair verification (current)

The independent audit identified an existing exact owner for resume publication: digests.indent2_json_file_bytes. The call is migrated and removed from retained calls and missing-owner blockers. Counts are now five completed and five blocked; retained AST witnesses cover only the five genuinely unmatched calls. Frozen original fixtures/outcomes were not changed. The stale active cross_slice_standalone_dependency blocker is removed: the repaired sibling loader is exercised by the real Python -I package-free integration regression. No owner or other shard was modified in this repair.

Focused command: `python -m pytest -q tests/test_digest_1301/stagea-custody/test_chain_consumers.py tests/test_stage_a_chain_resume.py tests/test_stage_a_chain_split.py tests/test_stage_a_retirement_1073.py --basetemp="$TMPDIR/pytest"`

**109 passed**, 14 torch Python-3.14 deprecation warnings, exit 0. Includes actual resume owner routing, frozen original bytes/outcomes, strict publication/conflicts, completed-node provenance, retained AST witnesses and package-free plain/gzip prewarm integration with the real replay sibling. No full suite, GPU capture, scanner or acceptance checks run.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6c948001c28b5b72bfe6a7eac371bffafb95277823ce9768617c942ebec6be11","size":914}},"operation":"e71a17ec53c21945b9a9a8ad585c76a054dac8a3689a2173065db49214de5ba0","receipt":"cfc73918eda7428fb6337623c1c45d9ac36b0cfa03df987c754c7e1b68a48ec2"}]
