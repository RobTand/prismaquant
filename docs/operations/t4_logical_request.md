# Model-independent retained-render qualification requests (PQ #1503)

`tools/build_t4_logical_request.py` builds a new, exclusive-write PrismaBuild
logical request from a digest-bound v1 or v2 adopted catalog. PrismaBuild still
owns partitioning and placement. The qualifier, PB schemas, CPU/GPU/memory
reservations, tier policy, mover declarations and batch limits are unchanged.
No export bytes, serving numerics, release inputs or retained results are changed.

```bash
PYTHONPATH=. python tools/build_t4_logical_request.py \
  --root /path/to/new-campaign \
  --catalog /path/to/catalog.json --catalog-sha256 CATALOG_SHA256 \
  --python /path/to/pinned/worker/python \
  --qualifier-checkout /path/to/qualifier-checkout \
  --prior-result /path/to/qualified-cell.json RESULT_SHA256 \
  --prior-child CHILD_MANIFEST_SHA256 FULL_PARENT_ACTION_KEY \
  --out /path/to/new-request.json
```

Both prior-input flags are optional and repeatable; omit them for a fresh run.
`--prior-result PATH SHA256` names one per-cell qualification JSON. Its file
digest, `(qname, format)`, `cell_sha256` against the exact catalog cell and
`verified_cell_sha256` against its qualification content are checked. The request
resolves that path to an absolute path in the builder's cwd before reading or
storing it, then reuses it in place: no copy, rename or rewrite. Legacy support
here is **filename-only**: a legacy qname-named file must still have the modern
`cell_sha256`, `verified_cell` and `verified_cell_sha256` fields. The original
37 pilot-era results without a qualification-content digest are refused;
there is no automatic migration or weakened digest admission.

`--prior-child SHA256 PARENT_KEY` reads a
`prismabuild.child_result_manifest.v1` at
`CAS_ROOT/blobs/<first-two-digest-characters>/<digest>`; `--cas-root` defaults to
`/mnt/shared/prismabuild-fleet/cas`. The manifest digest, schema, caller-declared
full parent key and every row's task/output identity are checked. Each row's
qualified result must exist at `ROOT/qualified/<pair-id>.json`, match its
`value_sha256`, and pass the same content checks as an explicit result.
Legacy qname-only child manifests are refused, not inferred, even for a
single-format catalog. No fixed parent, child digest, pilot path or result
count is assumed. Unrequested pilot files are ignored. Conflicting digests for
one pair, unknown pairs and duplicate catalog pairs are refused before output.

Task IDs and output IDs are the SHA-256 of compact, sorted-key, ASCII JSON
`[qname, format]` (the shared `DIRECT_ASCII_LAX` encoding). New output paths are
`ROOT/qualified/<pair-id>.json`, so formats of one unit cannot collide. Each
task's residency key is its catalog format; the batch policy names every
catalog format in sorted order. Reused tasks carry the verified digest and no
payload reads; fresh tasks declare both wire and render spans through the
existing task-data-manifest mechanism. All prior inputs are read-only. Output
paths are normalized to absolute paths (including symlinks and `..`); distinct
pairs may not share a normalized output path. A reused result at another pair's
fresh/default output path is refused before the request is published.

Fresh task destinations also reserve the qualifier's exact temporary filename:
`Path(output).with_suffix('.tmp')` (suffix replacement, not `.json.tmp`). Before
publication, its normalized path may not alias any declared prior qualification
file or CAS child manifest, any task output, or another fresh temporary. Every
verified prior path is retained for this check, including equal-digest duplicate
inputs not selected for reuse. Reused tasks do not reserve a temporary write.

**Occupied fresh temporary paths now refuse**, including ordinary stale files,
directories, hardlinks and broken symlinks. This conservative preflight prevents
`wb` from following a hardlink into retained bytes; path normalization alone
does not identify hardlinks. There is no automatic cleanup, overwrite, rename
or migration. These are planning-time checks, not filesystem locks against
external mutations after publication; the qualifier's bytes/behavior are unchanged.

## Consumer input lookup

`assemble_t4_overlay.py` and `rebind_t4_qualified_results.py` share the existing
`result_path` lookup, now extended to the builder's pair filenames. The default
`--qualified-key auto` accepts exactly one existing pair-id or legacy qname-id
file for each pair in the selected directory. If neither exists, the missing
pair-id path is reported. If both exist, intake refuses as ambiguous, even when
their contents agree. `--qualified-key pair` or `--qualified-key qname` explicitly
selects one layout; neither falls back to the other. The assembler retains
`--format-qualified-dir FORMAT=DIR` for per-format directory overrides. The
original `result_path(dir, qname)` helper API remains qname-only.

Fresh builder outputs therefore feed direct assembly and anchor-only rebinding
without renaming. Existing canonical legacy directories still work without
new flags. Lookup changes no result bytes and preserves each consumer's existing
content/rebinding checks; it does not broaden the builder's digest admission.

An arbitrary custom filename supplied via `--prior-result` is usable by the
request/qualifier only. Directory-discovery consumers require existing canonical
pair-id or qname-id filenames in their selected directories. They do not discover
arbitrary per-task paths or perform copies, renames, or retained-data migrations.

The historical pilot-derived planning estimates remain allowances, not
measurements for a new catalog or any performance claim. This change does not
qualify a model, execute the GPU decoder, or establish numerical equality;
those remain the existing qualifier's responsibility.

Regression coverage: `tests/test_build_t4_logical_request.py` uses synthetic
catalogs, qualification files and CAS children only;
`tests/test_assemble_t4_overlay_original_pair.py` exercises fresh pair outputs
through direct assembly and rebinding, legacy directory lookup and ambiguity
refusals. Its builder-to-consumer fixture simulates qualification metadata and
replaces the source-artifact fence; the separate fence tests keep testing that
unchanged boundary. No GPU decoder runs in these tests. Tests and compile checks
must be run through PrismaBuild, not by executing the example as a test locally.
