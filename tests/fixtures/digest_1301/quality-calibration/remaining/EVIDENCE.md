# Remaining quality/calibration audit

No application source, shared owner, existing qualification witness, seal or roster was edited. New artifacts are this evidence, `ledger.json`, and `tests/test_digest_1301/quality-calibration/test_remaining.py`.

## Native inventory

Original commit: `6d8288adc94b3c4b624c143ed372fb6513f32784`. Six audited source files still equal their original blobs. vLLM's two sorted calls were migrated independently; its original main fixture, with a recorded +8 line offset, now supplies the original-population witness without freezing the migrated source hash. The ledger records every native call's lexical scope, original line/UTF-8 column, Git blob, expression, explicit/effective options, caller framing, significance, and independent literal witness.

- 15 JSON calls: 13 `dumps`, 2 `dump`.
- 13 use insertion order and lack an equivalent existing JsonProfile, whose `_encoder` always sets `sort_keys=True`.
- **Exception to the initial hypothesis:** `vllm_prompt_smoke.py:74:10` and `:78:24` already explicitly sort and match `DIRECT_ASCII_INDENT2_LAX`; retain print/file newline framing if migrated.
- `quality_cpu_smoke.py:66:27` is separately recorded as `pickle.dumps`, outside JSON migration, not disguised as scientific retention.
- The final smoke stdout is a single-key object: sorting has no ordering effect on that current shape, although its native recipe is unsorted. General counterexamples show the missing owner capability, not an observed CPU/model output regression.

These are migration eligibility/owner capability findings, **not scientific-retention justifications**. Tiny tests compare original call options against hand-written literal expected strings and separately against the owner. They cover every JSON site, mixed-key acceptance, default=str/ASCII behavior, 400-character diagnostic prefixes, and source-bound complete population. They do not rerun producers or qualify GPU/native behavior.

## Why dump needs a stream operation

The two validators call `json.dump(..., fh, indent=2)` after opening/truncating their files. Dump feeds encoder chunks to `fh.write`. With an unsupported later value, a prefix is already written; a sink failure can occur before that later serialization error. `fh.write(profile.text(value))` first materializes the entire text, writes nothing on encoding failure, and changes error order. A future owner route must preserve incremental writes, partial files, exception timing, unsorted keys and the existing open/text-mode/no-terminal-newline behavior. Passing final text comparisons alone is insufficient.

## Read-only check of the 12 already-changed files

The ledger retains all 12 paths and original HEAD identities, selected historical identity references, and precise current migration-ledger call IDs and named consumer tests. Repetitive diagnostic path/basename search hits and stale working hashes were removed. Historical qualification concerns below remain distinct from current CPU witnesses. No seals were refreshed.

**Potential protected-expression conflicts:** `docs/audits/pq2317-consumer-boundary-evidence-20261005-v3.json` explicitly records reversion of unproved residuals to exact original expressions. Five such expressions are absent from the changed worktree:

- `capture_glm_routing_replay.py`: receipt and stdout (2).
- `check_stageb_a4_group_quantizer.py`: result stdout (1).
- `check_stageb_a4_quantizer.py`: result stdout (1).
- `pq282_equality/recheck_refusal_order.py`: refusal multiset key (1).

The old proof says no producer boundary was executed for those sites; the quantizer row also records registered-op/container-access blockers. New CPU encoding witnesses do not automatically discharge these historical qualification requirements. This is a potential protection/qualification issue for review, not a claim that the changed serializers produce different bytes.

All 12 have HEAD-blob references in `tests/fixtures/digest_parent_completion_1301.json`. The historical digest census additionally matches exact HEAD blobs/SHA256s for both quantizers, EXL3 replay, WikiText materialization and the waiter. `tests/test_digest_parent_completion_1301.py:109-205` checks current census identity and independent per-call completion evidence; an output-equivalent refactor alone does not renew it. Launch contracts also distinguish package/help checks from native GPU qualification. Exact file seals covering any changed file necessarily change even if output bytes do not.

Search boundary: tracked repository text, including hidden tracked files, literal full paths/basenames and original/current digests. Dynamic closures and external/ignored deployment receipts were not exhaustively searched or requalified. The local suite passing is not parent closure or native qualification.

## Observed tests

Initial focused run: 20 passed, 1 failed because the test sandbox's Git snapshot lacks the historical path at the recorded original commit. Fixed the witness to validate the independently recorded original blob directly against unchanged source bytes, without requiring a historical Git DAG. No source fixture copies added.

Focused rerun: **21 passed**.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e83ac29c0c875f05ec9e784037644fd10771f87a184714e3f051de32509eefc1","size":269}},"operation":"638e8720fb873309695cda362f3a5b7f6b06dd4373881b388f17141066e95e99","receipt":"2e2b4607e18e88016fd96beb63f055b804d090a2eafcd64cdebb2b4d15e1f6da"}]

Whole assigned local directory: `python -m pytest tests/test_digest_1301/quality-calibration -q --basetemp="$TMPDIR/pytest"`: **56 passed** (CPU).

Final rerun after ledger clarification: **56 passed**.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"48f3ac6307e0ed39e15192cce83d25baf7ff5c79da61faa13623b6592afa5ceb","size":269}},"operation":"6072ee2802509a47f03712d2ecd7c6748892e5af7765f0b5d183ce496ce630b1","receipt":"3d6a32215d8eedc7a27d264fe17b94873b857ff84fbd1fd7a7008cce40ad8a96"}]

## Integration repair

The two sorted vLLM call dispositions are now `migrated`, linked to the vLLM ledger. Original expressions and coordinates remain intact. The population test reads the original main fixture (verified against that ledger's fixture digest), reconstructs original lines with the recorded offset, and checks migrated statuses. Other sources remain checked against original blobs. No current producer hash is frozen. Current witness references replace wholesale search diagnostics in the changed-file audit.

Focused integration rerun: **21 passed**, CPU only.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"0d0db27590917ab311b3defdf5ca3691c23bf5a8bea7021abb18f74e89ead66c","size":269}},"operation":"6baf614bf33409d6a7940f6dea32c36de5f6e183310a33285e82184dec9cb7e8","receipt":"ee36268e9e0f849622349e3ea754b831e5ef03eae8d8d25ff10ace46a8ad71cf"}]
