# Orchestration-runtime publication slice (review revision)

## Scope and counts

Three frozen calls remain migrated: campaign run stdout via DIRECT_ASCII_SPACED_LAX.text, campaign status stdout and pipeline spec stdout via DIRECT_ASCII_INDENT2_LAX.text. Five calls are recipe gaps: three indent-1 pipeline publications, write_pipeline_spec, and ArtifactRegistry._write. No shared owners, shared census, frozen baseline, or other slices were changed in this revision.

The two file writers retain their original json.dump calls. Buffered text serialization is not equivalent to streaming dump: it changes sink write partitions and partial bytes before TypeError. For the registry this affects temporary-file operation traces even when successful final bytes agree. A missing owner streaming sink API (not a new local encoder or an owner change) blocks both routes. LF, flush, fileno, fsync, close, replace, and finally cleanup are retained in their original order.

The three indent-1 gaps require direct ASCII, sort_keys=True, indent=1, allow_nan=True and no default coercion. The existing indent-1 owner adds default=str. These are _record_unverified_reuse, _record_stage_settings, and main's stage-settings file publication (not stdout).

## Provenance and coverage

ledger.json now records all eight calls individually, including source commit/blob, frozen call ID, fixture path/hash, and executable test references for every gap as well as completed migration. Six original consumer snippets come from commit 6d8288adc94b3c4b624c143ed372fb6513f32784. The two additional indent-1 function excerpts were extracted directly from that commit with AST source spans. Tests validate all ledger source IDs/blobs against the frozen census and all fixture hashes without needing git history at runtime.

Tests execute original and current consumers. Pipeline file comparisons now include bytes on every failure, without skipping unsupported objects or mixed-key sorting rejection. Registry sink trace comparisons cover every write partition, injected fourth-write failure, partial temporary bytes at close, success LF, flush/fileno/fsync/replace order, destination preservation on failure, and temporary-file cleanup. The original streamed writer has multiple writes; buffered text would fail the trace equality. All three indent-1 gap routes compare original/current success bytes and rejected-payload outcomes. Other coverage retains stdout LF, ordering, nested values, non-ASCII escaping, empty objects, nonfinite floats, negative zero, real pipeline-spec reload and registry destination bytes.

## Checks

Revision's first run: 103 passed, 1 failed. The new indentation assertion incorrectly expected an indented member in an empty JSON object; fixed the test assertion to accept the original `{}` representation. No production behavior was changed to satisfy it.

Final command:

`python -m pytest -q tests/test_digest_1301/orchestration-runtime/test_publications.py tests/test_artifact_registry.py tests/test_artifact_registry_transactions_audit.py tests/test_pipeline_contracts.py tests/test_campaign_digest_profiles_2174.py tests/test_campaign_write_destinations_2231.py --basetemp="$TMPDIR/pytest"`

Result: 104 passed, 14 dependency deprecation warnings, exit 0. No GPU/performance or full-suite claim.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ca65e6f892063ea21820f25ebe912331e3ae597621db30e4fa151abd50406dc6","size":902}},"operation":"038bee54c64f823f4fbfe4301aca750199b9950be2a34664e4237216d57b64de","receipt":"6a45b512ad22b0851ccedae3d8038020d49c843b3e5cb709786a320b13de29e7"}]
