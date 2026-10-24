# Standalone retained-operation evidence

Scope: only `test_standalone_retention.py` and this fixture directory. No application or deployment edits.

`retention_ledger.json` records each of the ten census retained calls individually, including original exact ID, original blob, syntax digest, current ID/blob, original source fixture checksum, supported-population witness, operation test, original outcome and current comparison. One supplemental `json.loads` call records the JSON reader outside the sorted-JSON census. Its array/malformed/duplicate rejection and existing default NaN acceptance are tested explicitly; this evidence does not claim the reader is strict JSON.

Frozen executable excerpts are derived from the recorded original Git blobs, not generated from current source during tests. AST rendering drops comments/docstrings; the docstring-only error class uses `pass`. The identity excerpt omits mounted-runtime verification, which owns none of the retained calls. The progress excerpt includes only its four retained writers, constants and imports. Original and current run independently in different temporary directories and different `-I -S` interpreters. A meta-path guard forbids PrismaQuant and optional integration package imports throughout every operation. The parent then compares full outcomes and explicit digest/preimage, serialization-byte, return-value and error oracles. The ledger test binds every exact current call coordinate/callee to the corresponding frozen function's AST and checks source/fixture blob digests.

Deployment witness distinction: identity has current `runpy.run_path` loaders in `tools/tessera_campaign_container.py` and `tools/build_glm_derivative_image.py`. Progress has an explicit standalone path-loading contract in `tests/test_tessera_campaign_container.py` and a source comment supporting path loading in containers with stamp integration disabled. Repository loader searches did **not** establish an additional production file-spec progress loader; ordinary package-import callers are not presented as proof of one. These tests establish the supported file-spec operation population, not execution in a live container deployment. No GPU, live container or full-suite check was performed.

## Focused validation

Command:

```sh
python -m pytest -q tests/test_digest_1301/orchestration-runtime/test_standalone_retention.py --basetemp="$TMPDIR/pytest"
```

Final observed result: **3 passed in 0.80s**, Python 3.14.4, exit 0. This includes four independent child interpreters for original/current identity and progress operations. Source files are compiled by their file-spec loader during those executions.

Earlier fixture/test-development attempts failed because removing a class's only docstring left an empty body, and because two nested AST calls shared a source coordinate. Those test-harness defects were fixed (`pass` and callee-qualified lookup respectively); final validation passed. No source behavior was altered.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f2092aa9c8f6abc52c51fc0b9bf07d76672a4ac21607cbc7da8d900b0b3b3db6","size":268}},"operation":"c433d5865b8fcfb6e1924e857a1ee9315508082ecc619c1f920e0a27d639f3dd","receipt":"9447a070c778c0fc66326ed99081eb606b1fe5f393fd4a7f426e2a6726242066"}]
