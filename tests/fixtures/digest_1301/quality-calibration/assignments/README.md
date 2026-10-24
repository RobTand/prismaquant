# Assignment publication evidence

The three Python fixtures are verbatim contiguous original consumer functions or
blocks extracted with `git show` from research base
`6d8288adc94b3c4b624c143ed372fb6513f32784`. They are intentionally not whole-source
copies. `ledger.json` records each replaced scanner call ID, exact owner route,
base blob (`git rev-parse <base>:<path>`), fixture and executable test nodes.
The uniform fixture covers both assignment and optional report publication.

CPU tests execute the original fixture and the corresponding current consumer
AST block with the same inputs. They compare published bytes and stdout, assert
owner routing, final LF, Unicode escaping and lax nonfinite behavior, and cover
missing overlay parents, serialization failures without clobbering existing
files, report failures after successful assignment publication, and no-report
behavior. Model loading and assignment computation outside these blocks are not
exercised. No owner, shared census, baseline, or unrelated tool was modified.

Run without pytest or accelerator dependencies:

```sh
PYTHONPATH=. python tests/test_digest_1301/quality-calibration/test_assignments.py
```

Final result: **9 passed**, exit 0. `git diff --check` also passed.
An earlier provenance test attempted to read research-base Git history inside
the test snapshot and failed because that object was unavailable there. The
final tests do not depend on repository history; base blobs were read from the
worktree Git object database when the fixtures were extracted. The final suite
checks ledger fixture/node integrity without requiring Git history.

[Test evidence {"artifacts":{"stderr":{"digest":"d6403ba4a2b9a41908ac1d20811b381803b21faf5853ca9d314c6417d8b15b07","size":668},"stdout":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0}},"operation":"34f375eae0635852b4b8dfef48dd9b698806cce1a80c732a9418ef36019bcad0","receipt":"52fe13e708cf4d54dea4dc9f3119b61ecc3381aa299413874d5a4b949d1c6d73"}]
