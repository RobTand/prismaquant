# Runtime I/O independent subset — repaired ledger and witnesses

This repair changes only the runtime_io ledger, evidence, and assigned test file. It preserves main's io_engine edit (no needless empty hash for nonempty buffers). No shared owner, registry, census, baseline, publication, or workflow changes.

## Provenance and accounting

The four original identities are taken exactly from `tests/fixtures/digest_parent_completion_1301.json`, at requested research commit `6d8288adc94b3c4b624c143ed372fb6513f32784`:

- `hashlib:prismaquant/io_engine.py::SealedBuffer.seal@261:17`
- `sorted-json:prismaquant/dev_mode.py::_shown@130:18`
- `sorted-json:prismaquant/io_spans.py::IoSpanLog.close@252:24`
- `sorted-json:prismaquant/io_spans.py::ReadRateReporter._emit@415:24`

All raw original call membership in these three files is accounted for exactly, without substituting scope labels for IDs. Already-owned calls are separate context, not extra frozen originals. Source blobs were verified against the requested commit with `git rev-parse commit:path`: dev_mode `1de4f205ed5fdd69ceadcc8716cdf801f9046ca2`, io_engine `393ae5755633992c900cbf55c24b3f50a0ca257d`, io_spans `b2c028c05902347da251336d11009a846c5b171a`. These match the earlier excerpt source blobs, so excerpts need no content change.

The ledger uses separate `file`, repo-relative `fixtures`, and full unparameterized `tests` nodes. `test_ledger_witnesses` checks exact frozen identities, membership, source commit/blob/syntax identity, fixture digests, executable nodes, and current blobs. Current blob calculation uses Python SHA-1 over Git's `blob <length>\0<bytes>` representation, not a Git subprocess. The io_engine current blob is refreshed to `04e9b84c5f1d2ac163828f43f0178102d86d0397`.

## Decisions and witnesses

- `SealedBuffer.seal`: replaced by existing `bytes_sha256hex` owner. Real memfd tests compare original/current digest bytes, kernel seals, immutable lengths, mapping closure and descriptor lifetime. An independent proxy mapping trace compares original/current ordering: hash, view release (verified by successful resize), mapping close, kernel seal request, return. The owner spy asserts exactly one feed for empty, binary, and UTF-8 payloads, preserving payload bytes. Injected owner failure proves view release while retaining the mapping.
- `_shown`: **blocked**, not justified retained. Missing exact spaced ASCII/lax `default=repr` recipe is a migration blocker. Existing bytes owner remains unchanged. Tests cover repr versus str, mixed-key/circular fallback, Unicode, surrogate escaping, warning and certified refusal.
- Both `io_spans` encoders: **retained for supported deployment**, separately from recipe availability. `tools/pq_profile_source.py` loads this stdlib-only module by file spec without package initialization. The isolated bootstrap test prohibits all prismaquant imports and exercises both consumers. Reporter bytes match `DIRECT_ASCII_SPACED_STRICT`; the span's missing strict `default=str` recipe is recorded separately, not used as the deployment justification. Tests cover marker bytes, string fallback, NaN refusal warnings, logger failure, duplicate close, error outcomes and state updates.

## Focused checks

```sh
python -m pytest -q tests/test_digest_1301/orchestration-runtime/test_runtime_io.py tests/test_profile_observer_bootstrap_1942.py tests/test_io_engine.py tests/test_io_engine_sealed_memory.py tests/test_io_spans.py tests/test_digest_prepare_io_1301.py --basetemp="$TMPDIR/pytest"
```

Final observed result: **100 passed**, 14 upstream Torch Python 3.14 deprecation warnings, exit 0; no skips. `git diff --check` also passed. No full-suite or GPU claim.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"50e8e1f6816d94f835b18bdaa653a79f6197bbd77daca9ec5403fea333e593c0","size":902}},"operation":"d1a541748f394f24f95fc60c3672f70fde6ff9c8408238f5c6cfca8bc34e9993","receipt":"866275b179201585c064844688816a65db1479a0d6a9ed7c626d008fc086cd34"}]
