# shipcard inspection and behavioral evidence (no production migration)

## Accounting and provenance

`tests/fixtures/digest_1301/serving-evidence/ledger_shipcard.json` accounts for exactly four remaining executable raw calls in `prismaquant/shipcard.py`: two SHA-256 constructors in the same `_verify_open_safetensors_fd` scope, and both JSON publication encoders in `_publish_shipcard`. Accounting is per AST Call, not per scope or parameter name. All four are **blocked eligible**, with **zero justified retentions** and zero replacements in this change.

Original registry IDs, line/column, source blob, source commit, exact recipes, owner gaps and null retention evidence are recorded per call. Registry source commit is `30b6231df000726beda55fb6706e641a5cb2ccb0`, shipcard blob `acf1b46ab0802eada0512a22b78828bd9420bd8a`. The original publication IDs name `write_shipcard`; today's encoders reside in `_publish_shipcard`. Inspection HEAD is `8bc0b5af8c69af352b559dccda99bd4e6fa8b418`, shipcard blob `bb7bc02b915b0906470f5cc9fdd881d218f361eb`.

## Missing shared owner required

Inspection of `prismaquant/digests.py` finds no public incremental raw SHA-256 state owner exposing the needed update/hexdigest lifecycle for caller-owned reads. The verifier must retain one forward descriptor traversal, feed each read to the full-container and active per-tensor state, and preserve header/span validation and final stat fencing. A pathname/file digest route would change acquisition ownership or add a second traversal.

The existing JSON profiles sort keys. Neither provides these insertion-order publication recipes: ASCII escaping, lax nonfinite values, `default=str`, and either indent=2 or compact separators. The consumer must still append LF, encode UTF-8, attempt pretty before compact, pad to the reservation and refuse invalid/insufficient reservations. A nearby sorted-key or strict profile is not equivalent. Shared owner work is required before migration; these are not justified permanent retentions.

## Independent fixtures and checks

`shipcard_original.py` contains only selected original full consumers plus their transitive stdlib-only validation helpers, extracted from the registry source blob (prefix helper from the same source commit's `source_read_plan.py`). It does not copy the full production module, import current production helpers, or derive its functions from current production at test runtime.

`test_shipcard.py` exercises actual `_publish_shipcard` against the independent original full `write_shipcard`: insertion order, Unicode escaping, default=str, nonfinite values, pretty output, pretty padding, compact fallback, exact and padded reservations, invalid/insufficient reservations, cycles, invalid keys and fallback errors. Refusals leave existing publication bytes untouched.

Actual `_verify_open_safetensors_fd` is compared with the independent original full function and independent original validation helpers. Tests observe descriptor read offsets, requested sizes and bytes, prohibit implementation seeking, compare complete result/exception contracts, check full and per-tensor hashes independently, include zero-length tensors and out-of-name-order spans, short reads, the 1 MiB read-request boundary, malformed headers/spans, truncated reads and final stat drift. These are local CPU fixtures, not live serving, GPU, throughput or full-model qualification.

## Observed results

Initial targeted run: 34 passed (before adding the megabyte-boundary case).

Final command:

```sh
python -m pytest -q tests/test_digest_1301/serving-evidence --basetemp="$TMPDIR/pytest" && python -c "import ast; from pathlib import Path; [compile(p.read_text(), str(p), 'exec') for p in [Path('tests/test_digest_1301/serving-evidence/test_serving_1301_shipcard.py'), Path('tests/fixtures/digest_1301/serving-evidence/shipcard_original.py')]]"
```

Observed: **86 passed**, exit 0; both new Python files compiled successfully. Of these, 35 cases are shipcard-specific.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b29b13c1bcd2949900106aca74cf5b00bdd3027551af3ae0dedf7e74dbfe7a13","size":349}},"operation":"07686809a37de702670ee925c00cc4bf1c2a4e82aeb9a25dfbecf5bbfaca19dd","receipt":"d13cec1393b2c834cfe01cceece38f889d40abb1784751dba7444f10c70176bf"}]

`git diff --exit-code HEAD -- prismaquant/shipcard.py` succeeded with no output. No production file was edited by this work. The pre-existing modifications to `glm_routing_replay.py`, `tessera_export_lane.py`, and `tessera_materialization.py` were left untouched; the assigned directories already contained other serving-evidence work.
