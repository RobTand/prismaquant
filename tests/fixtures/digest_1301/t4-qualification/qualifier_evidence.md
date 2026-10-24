# Qualifier routing evidence

Scope: qualifier only; no shared owner/census/baseline/acceptance changes. Existing independent tools/test_qualify_t4_overlay.py is unchanged. No numerical GPU, performance, serving or immutable-current-pin claim.

Final CPU command:

```sh
python -m pytest tests/test_digest_1301/t4-qualification/test_qualifier.py tools/test_qualify_t4_overlay.py --basetemp="$TMPDIR/pytest" -q
```

32 passed, 14 upstream Torch deprecation warnings. The same retained operation also compiled all three scoped Python files using compile() without writing bytecode. Tiny CPU read/finish and ordered-walk adapters exercise actual qualifier main/nested read_task/commit_task, both manifest modes, new publication, pinned and unpinned adoption, exact bytes/newline/ASCII/nonfinite values/hash inputs, mixed-key serialization refusal, malformed checkpoints, identity mismatches, missing pinned outputs and changed stat fences. Adoption and refusal forbid read/decode calls. Decoder direct_url hashing checks raw text independently of parsed JSON, absent metadata and malformed JSON refusal.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"a8c9241b4ee24fee14ba80fe5e85672f2a8b49da7112ff746397f5483c781c6d","size":855}},"operation":"45f191f9573dba696e3ac9ba4c86f59e18b6d1b52d9a959953ae52d4e6fb0195","receipt":"18829c860428e516d77ff1e2d2582f7cbcbafa254a0b89518905da7836722915"}]

Earlier combined check: tests passed (32) and compilation succeeded, but a supplemental fixture comparison against test-sandbox HEAD failed. HEAD in the test snapshot is not the original pre-edit source. Host-side comparison against explicit pre-edit commit 8bc0b5af8c69af352b559dccda99bd4e6fa8b418 returned True for exact AST function-line extraction including decorator. Runtime tests do not depend on Git. The failed supplemental check is retained, not counted as a passing check.

[Test evidence {"artifacts":{"stderr":{"digest":"c52aff3189b5d5c84f3951780f2322da36d5835de6446865d5a1db3c3d98a861","size":1721},"stdout":{"digest":"13006f923506f3fd68fe3ddbd55281e9c358d39469854f1b4a36ea4789b3c1d9","size":821}},"operation":"b6898a2f8a83fdd7a7c9cadb39c0975ca435457dd7302634c524abbab1c34219","receipt":"f986474722500a11c7a638182ccdf8b11b1346b5cc81420f36f376bd3b4a581f"}]

Residual exact gap: supplemental-json:tools/qualify_t4_overlay.py::main@67:8 and supplemental-json:tools/qualify_t4_overlay.py::main.commit_task@84:8 remain insertion-order, unsorted stdout json.dumps. No settled insertion-order owner is assigned; these calls are intentionally unchanged. The ledger lists all seven replaced registry calls and both deferred calls with provenance, routes, fixtures and executable tests. Registry source bindings describe provenance rather than immutable retention; historical measured source is not a current immutable pin.
