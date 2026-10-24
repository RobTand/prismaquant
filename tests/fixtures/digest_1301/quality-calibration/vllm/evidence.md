# vLLM smoke JSON migration evidence

Scope: only the two sorted indent=2 result serializations in `tools/vllm_prompt_smoke.py` now use the existing `DIRECT_ASCII_INDENT2_LAX.text`. No shared owner changes. The unsorted CPU preflight serialization remains a GAP. Existing output path construction, parent creation, overwrite, stdout ordering, and trailing newline remain at the consumer.

`main.py` preserves the original consumer's main function from commit `8bc0b5af8c69af352b559dccda99bd4e6fa8b418`; only a fixture header was added and imports are supplied by the test harness. `ledger.json` records the source blob, call coordinates, fixture SHA-256, owner route, and test references. No fixture imports the migrated consumer or the digest owner.

CPU tests invoke both the original main block and actual migrated main, with deterministic time and vLLM doubles. They compare complete stdout/stderr, return values, exception type/messages, engine/generation calls, and output bytes. Cases include Unicode, non-finite values, signed zero, graph/eager/configuration arguments, overwrite/new parents, omitted publication, invalid arguments, initialization/generation/serialization refusal, and publication failure after stdout. The unchanged unsorted preflight is compared directly. Standalone launch runs the real script in an unrelated working directory under `python -I`, ignoring PYTHONPATH, and verifies successful preflight. The source loader resolves relative to `__file__`; it does not add sys.path entries or require package initialization.

Final validation:

`python -m pytest -q tests/test_digest_1301/quality-calibration/test_vllm.py tests/test_digest_profiles_1301.py tests/test_profile_digest_bootstrap_1899.py --basetemp="$TMPDIR/pytest"`

113 passed, 14 existing torch Python 3.14 deprecation warnings; exit 0. Includes 23 consumer cases. No GPU or real vLLM/tokenizer execution was performed; this establishes JSON/consumer equivalence, not served model qualification. All test execution used run_tests.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e7bae3772c431e95f806d912ada8868eb0fae7f935808bd3666f65c0931817df","size":903}},"operation":"20c818487f616c3474a23c0079c10357c763d8330767d3d1b0418b948802932a","receipt":"0923a275046566e4446af8975a8dddd8fee1d11130bd1912a64155ef9ed3db0d"}]
