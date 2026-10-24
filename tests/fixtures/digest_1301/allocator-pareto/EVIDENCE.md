# Allocator Pareto digest slice

## Scope and compatibility

Nine research-base calls are replaced: the compact Pareto encoder and SHA-256 constructor, plus seven sorted indent-two publications. `ledger.json` records each frozen ID separately. The actual main publication block still controls duplicate suppression, twelve-hex labels, paths, assignment and manifest writes, budget stamps and stdout. Its compact identity uses `DIRECT_ASCII_LAX.sha256`; sorted pretty publications use `DIRECT_ASCII_INDENT2_LAX.text` with the existing caller LF. No allocation policy, numerical calculation, runtime/qualification pin, deployment roster or default changed.

Independent expected outcomes execute the existing verbatim original Pareto fixture and seven original publication statements, not the replacement owner. A read-only provenance audit compared all seven statements verbatim with the served research-base checkout `6d8288adc94b3c4b624c143ed372fb6513f32784`, verified allocator blob `edb5a2ae0cf799f33d570aef8deec513ec30ff65`, and compared the original Pareto fixture AST with that checkout's main block. Test execution does not need historical Git objects.

The local tests exercise complete publication bytes, hash feeds, duplicate suppression, twelve-character labels, stdout and ordered hash/write events, with and without budget stamps; mixed-key, unsupported-object and cycle refusals; Unicode, escaped surrogates and lax nonfinite values; and every sorted publication statement. These CPU/isolated-AST checks do not establish model setup, serving-enabled feasibility calculations, GPU or whole-suite qualification. A separate read-only subagent review found no defects in recipe equivalence, fixture independence or ledger coverage, with those integration limits.

## Remaining owner dependency (not retention)

Three supplemental encodings remain eligible and unresolved:

- `_write_bit_attribution_reports`: insertion-ordered ASCII/lax indent-two `json.dumps` plus LF.
- `main._write_layer_config`: insertion-ordered ASCII/lax indent-two `json.dumps`, UTF-8, no LF, create-only byte publication on replay.
- `main._write_layer_config`: insertion-ordered `json.dump` to the caller-owned text stream, no LF; chunked writes and partial-output/refusal behavior must remain.

`JsonProfile._encoder` unconditionally sorts keys. No current owner supplies these insertion-ordered text and stream-write recipes. Owners are read-only for this piece; this dependency must return to the shared-owner implementer, not be hidden with retention or a local wrapper. Load/loads parsing calls and existing owner routes are unchanged. The ledger records inspection boundaries and the three unresolved frozen IDs.

Shared census and baseline are also read-only. The integration owner must merge the nine completion rows and reconcile current source identities. The exact removed baseline scopes are `hashlib:prismaquant/allocator.py::main` and `sorted-json:prismaquant/allocator.py::main`. Do not remove or mark complete the three supplemental calls until their exact owner routes exist.

## Verification

Final focused command:

`python -m pytest tests/test_digest_1301/allocator-pareto/test_publication.py tests/test_digest_parent_completion_1301.py::test_allocator_pareto_owner_route_preserves_original_publication -q --basetemp="$TMPDIR/pytest"`

Result: **6 passed**, 14 torch deprecation warnings.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"c6cab29ffa6732cba79b14cd2dc056490475580781032afac38a50ab297a2e95","size":820}},"operation":"b5918ee7cb1cd3ee430db7c317843a38ee02891211d16577043eece3ac7d5d48","receipt":"e944bc1dcbddfab2de9b54adeed01413f8511129d308a02294009aa5f0985cd4"}]

Broader command added `tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py tests/test_duplication_baseline.py` to the focused selection.

Result: **207 passed, 1 failed**, 14 warnings. The sole failure is `test_primitive_digest_sites_only_shrink`, reporting exactly the two removed allocator scope rows above. No baseline growth occurred. Shared baseline edits are deliberately not made by this piece; this run is not a ratchet pass.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"45777b18514fe8f078fe17bf195d837d7c7a55efc7c27ffea0d3d786489d2c48","size":3078}},"operation":"a0e98ef05ef0e9941ea08de4594fa969148e017f12b3e7261dc515fa69904eb8","receipt":"209c55034b8fe1ffca8fdcf8ca4a85394eb56feefd8133946c5489f3933f411e"}]

Initial focused run had two failures: the new budget-path test incorrectly assumed its first two hashes were Pareto feeds (budget stamps interleave their own hashes), and removing the unused hashlib import broke the immutable acceptance fixture's inherited allocator namespace. The test now counts the literal Pareto feeds, and the existing import remains. Both failures are resolved in the final focused run.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"c32d824668f710ece4902a971cc6b94455f001f7306bcefaef41e8c7aad3f1c3","size":7056}},"operation":"4f436f1ee617a235da22ff21751c096269be1a0e25187095dc654cc74c2278ce","receipt":"1253517dd862dc0ced82385e715e5358efeb297c46fb719699ec56a70a4ae5dc"}]

`git diff --check` passed. All three immutable acceptance files, digest owners, scanner, historical census and baseline remain unchanged. Parent-wide census and closure tests are not claimed complete by this slice.
