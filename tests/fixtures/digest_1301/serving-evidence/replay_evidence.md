# GLM routing replay raw-hash migration

Scope: only `prismaquant/glm_routing_replay.py` production changes. No owner or shared production changes.

## Provenance and accounting

Per-call completion provenance in `replay_registry.json` and `ledger_replay.json` is pinned to the frozen research registry `tests/fixtures/digest_parent_completion_1301.json`: source_commit `6d8288adc94b3c4b624c143ed372fb6513f32784`, source_blob `91e76ed34850f4ef5090b35870b76d8501856654`. The regression test compares both records directly with that registry. Separately, the two executable snippets retain original hash expressions extracted from historical commit `30b6231df000726beda55fb6706e641a5cb2ccb0`, blob `871e917e34189843451dadcb8fdaa2368ca87c77`, independently of the candidate owners. That snippet origin is not per-call completion provenance.

Whole-file accounting: 2 original raw hash calls, 2 replacements, 0 remaining raw hash calls. One text owner call preserves default strict UTF-8 encoding of `inspect.getsource(type(router))`; one bytes owner call preserves `Path(__file__).read_bytes()`. No other digest calls in this file. File acquisition remains at the caller (not moved into an owner).

## Provenance correction test receipt

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ea13031fd46b5cc086ada25a5b5f13fd35fd3a99597f03e376cb82b75c50f0c6","size":269}},"operation":"e7c2d1388d12acc86ef92795e05c16b2729f8ac19b87076c58cd595346912341","receipt":"635aaaf6855365ba6bdcfaa609d7475c750c50057782053bcca78e4ed1a93002"}]

## Validation

Focused tests exercise real `glm_route_record` metadata construction with mocked source acquisition and unrelated metadata dependencies. Fixtures supply the independent original-expression oracle. Cases cover empty input, Unicode, BOM, CRLF, missing final newline, arbitrary bytes, exact acquisition arguments, and unchanged propagation of inspect/read failures and strict UTF-8 errors. AST accounting verifies all raw hash calls were replaced.

Command: `python -m pytest -q tests/test_digest_1301/serving-evidence/test_serving_1301_replay.py --basetemp="$TMPDIR/pytest"`

Provenance correction rerun: 27 passed (including frozen-registry provenance regression). Previous run: 26 passed. `git diff --check` also passed. No full-suite or live model run claimed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"c5b31bb5e61847d94160b17afa7ee88581484bdf919de7786a0173371a41b3e1","size":269}},"operation":"cc2cd82d07bce283eb3ee0f31b9c27f6a4e1ffc9a1cb97db85407a068dfbc3d7","receipt":"ffa74938f7d5da660600b91833c8a93eea5a6a15160877d736beabd0fa349f8e"}]
