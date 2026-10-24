# Assigned package slice evidence

## Scope and provenance

Only the three assigned package consumers were modified. No digest owner, census,
baseline, acceptance file, or frozen parent registry was changed.

`ledger_packages.json` maps all 18 replaced calls to the frozen registry's
original IDs, source commit and source blobs, existing digest owners, independent
original function excerpts and AST-resolvable behavioral tests. Corrected counts:
**21 eligible calls, 18 replaced, 3 unresolved owner API gaps, 0 justified
retentions**. The three unchanged unsorted JSON calls are eligible, not exempt;
they have genuine original source witnesses and exercised tests.
The excerpts were extracted directly with `git show` at the registry's source
commit, not reconstructed from the migrated implementation. Their full function
bodies are needed to exercise refusal ordering and package publication.

Existing digest APIs own compact sorted ASCII JSON, spaced sorted ASCII output,
indent-2 sorted lax JSON text, and raw-byte SHA-256. Newlines and encoding remain
at the original caller boundaries. Gzip, input
reading, decoding, copying, text/byte write methods, validation, directory creation,
and refusals remain consumer-owned. Unsorted JSON is blocked on an exact owner
API, not justified retention. Existing finished JSON profiles sort keys and cannot
replace these calls without changing behavior:

- `recovery_manifest@85:26`: `json.loads(json.dumps(original))`, preserving
  insertion order, default ASCII escaping and lax non-finite handling, integer-key
  stringification and non-mutation.
- `recovery_manifest@133:26`: the same default unsorted round trip for the tail
  phase before modifying its entry indices.
- `main@194:45`: unsorted `json.dumps(fields, indent=2) + '\\n'` text publication
  of launch fields, preserving insertion order, ASCII escaping and lax numbers.

Sorting also changes mixed-key acceptance/refusal. No owner edits are authorized
in this slice; these three calls remain unchanged pending exact insertion-order
owner APIs. The slice is therefore not complete migration closure.

Independent-excerpt verification against the pinned commit confirmed verbatim
function bodies, not whole-file copies: recovery 144/212 lines (3 functions),
seed 69/292 lines (3 functions), split 93/475 lines (3 functions). All 18 replaced
calls map to behavioral tests comparing original and migrated outputs: source
pins/refusals (2), recovery publication and stdout (7), seed publication (5),
manifest wire/refusals (1), split description/publication (2), checkpoint digest
(1). These counts include both digest and serialization calls.

## Lax indent-2 correction

Five migrated lax indent-2 calls now use `DIRECT_ASCII_INDENT2_LAX.text`
instead of the strict `indent2_json_file_bytes` helper: recovery template and
pins, seed spec and package, and chain-split package. Original LF concatenation,
text writes, and seed-spec `.encode()` are preserved exactly, with no added
encode/decode round trip. The two pre-existing strict helper calls in the split
consumer remain unchanged. All five ledger owner routes are corrected; no owner
or original excerpt was edited.

Publication tests now run both finite and nonfinite cases against the original
excerpts. NaN, positive infinity, and negative infinity reach the actual seed spec,
seed package, split package, and recovery template publications; tests compare
complete original/migrated file sets and bytes and explicitly assert nonfinite
tokens and trailing LF. Recovery pins are compared byte-for-byte as well (their
values are digest strings, not floating-point numbers). Dependency stubs remain
as described below; these checks concern serialization, not upstream validation.

Focused rerun using the command below: **75 passed**, 14 torch deprecation
warnings. No full-suite or remote checks were run for this correction.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"996e6479c4a552b74582676b8565c8520bc929548b47e34b17a8b5ca43fdf625","size":902}},"operation":"60e84df27409286534868b86817b789828b75ecd5a0a4434cab94cdb9a8ad934","receipt":"f955201f612595be7cf25f3960b63cf58941fbc4d8a18e1922611e4466a7db2b"}]

## Behavioral checks

The new tests execute original excerpts against migrated consumers and compare
compressed bytes, package file sets and exact file contents, hashes, returned
metadata and stdout. They also cover validation before publication, existing
output refusal, pin mismatch, absent input, plan mismatch, invalid labels,
non-ASCII data, non-finite JSON numbers, lone surrogates, mixed-key refusal,
retained insertion-order publication and JSON-roundtrip non-mutation. Dependency
stubs isolate package construction from checkpoint I/O and the remote SDK; the
existing targeted tests also passed.

Initial new-test run: 7 passed, 1 failed because the recovery test fixture lacked
boundary inputs required by the real consumer. Added those inputs; no application
behavior was changed to accommodate the test.

Final command:

```sh
python -m pytest -q tests/test_digest_1301/joint-handoff/test_packages.py tests/test_stage_a_seed_package.py tests/test_stage_a_split_package.py tests/test_forward_recovery_chain_tools.py tests/test_joint_forward_resume.py tests/test_split_package_cli_docstring.py --basetemp="$TMPDIR/pytest"
```

Result: **72 passed**, 14 existing torch Python-3.14 deprecation warnings.
Re-run after the ledger correction: **72 passed**, 14 warnings; the provenance
test now enforces corrected counts and unresolved (not retained) classification.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"501cc5de10184d0e57d300a5c9dba2fc7527d29058029b54fb65f9949048d357","size":832}},"operation":"4048e81e30b6069f3878d9a98ed5e97fc7c98a62577d58956d835b2a99f0d111","receipt":"5efb73dcd8c50e4f8c9f7840e50ddc5098781ba23d53f3719a2e4bc146e53243"}]
No full-suite or remote-service validation was performed. Blocker: three eligible
unsorted JSON calls lack exact owner APIs, as detailed above; owner changes are
out of scope.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f078f3fdeb8d61ec6d73273271994fc7fbb4cebec98949f13d73a7256c2c9554","size":822}},"operation":"50112fc3f9cfb6551faed588320563936ac49d57ae58e44f49f822cb89a1b337","receipt":"53eedfca889e999d621bf41c51a95a6f5912d63be2d28b07c103e458366f861f"}]
