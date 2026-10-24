# Read-measurement windows subset evidence repair

## Scope and ledger

This repair changes only windows tests, windows fixtures, this evidence document and `windows_ledger.json`. It does not revise application sources or other subsets' tests.

The ledger inventories 54 original calls: **17 migrated, 37 blocked**. Primary `hashlib:` and `sorted-json:` IDs now include `@line:col`, consistent with individual scanner call identities; supplemental JSON IDs retain that suffix. All migrated `owner_route` values are fully qualified `prismaquant.digests.*` names. Test mappings identify actual consumer tests rather than the supplementary replaced-expression check. A ledger contract test checks uniqueness, coordinates, owner prefixes and resolvable consumer test names.

The 37 blocked rows are not claimed as retained migrations. Their recipe-specific blockers remain unchanged: four incremental digest recipes and 33 JSON option recipes without exact settled owners. This includes unsorted `indent=1, default=str`; no substitute sorting profile is asserted equivalent.

## Independent originals and actual execution

`tests/fixtures/digest_1301/read-measurement/windows_source.json` remains the independent original-expression/helper fixture. The added `windows_consumers.json` contains only the 12 original migrated consumer functions, extracted from the git blobs at `6d8288adc94b3c4b624c143ed372fb6513f32784`. It contains no whole source files and is not generated from transformed expressions. The extraction succeeded in the working environment. Earlier evidence reported history unavailable inside an earlier test snapshot; that was not a successful snapshot provenance check, and the new pytest run does not claim to reverify git history.

`test_windows_consumers.py` compiles the actual current function ASTs and independent original SOURCE function snippets. Narrow dependencies stand in for campaigns, cache, checkpoint services and clocks. Hash adapters use stdlib SHA-256 and record bytes/failures; expected original values do not use the destination owner. Comparisons cover return/exception outcomes, event traces and output bytes:

- `_sha256_file`: old/new bounded read and close traces, digest results and read exceptions. The existing explicit read-size witness remains.
- `run_setup`: real staged file writes precede hashing; hashing precedes publication and map creation. A failed hash prevents publication and fixture output.
- `run_window`: receipt metadata JSON enters the incremental digest before tensor bytes; exact old/new traces and output documents match. Invalid metadata fails before tensor access and exits the retained window. Cache/campaign behavior itself is adapted, not certified.
- `run_quantum`: actual setup, pinned producer patch and restoration, campaign/quantum invocation, JSON/pickle/raw file receipts, leaf dispatch, payload hashing and summary-file output. Hash exceptions are injected at all six hash positions, including final payload hashing; no summary is emitted on these failures.
- `_renders`: executes the actual Torch function twice against real source/copy files; compares every file byte, manifest and hash trace.
- `_commit_unit`: executes both timed and publication branches. Tests exercise shared pickle memo references and large framed payloads, compare exact nested pickle bytes, validate the envelope digest, and check preparation/state/hash/write/publication ordering and hash-failure behavior.
- `cmd_child`: executes the actual child function with narrow input/commit adapters, real checkpoint receipt files and result JSON. Tests compare setup identity, commit/cache-drop/hash events, output and cleanup, including both hash failure points. The idle-thread/cgroup and concurrent arms are not exercised.
- `child_main`: executes actual staged child dispatch and stdout generation; compares escaped JSON and marker output, plus serialization exceptions with no stdout.

Existing full-function comparisons cover verified JSON success/failure, leaf walking (including both migrated leaf calls), real CPU tensor views and empty tensors. The expression-level check remains supplementary, not the sole consumer evidence. Lexical timing/retention witnesses remain supplementary too; no memory-complexity or performance claim is made.

## Imports and limitations

Inspected top-level and nested digest imports in all seven scoped sources. The six migrated files use the existing standalone `digests` bootstrap for the owner exports; the chain file also has lazy profile imports, and `render_identity.py` uses its existing tensor-digest owner. Imported all migrated modules through the test loader during actual consumer tests. No import/source revisions were made. This is not a fresh CLI subprocess or deployment-path verification.

No GPU, full staged-reader/campaign, performance or numerical-equivalence campaign was run. Narrow adapters deliberately isolate consumer control flow and serialization; they do not certify the external services they replace.

## Observed checks

Both repair runs completed successfully through `run_tests`: first 80 passed; final 84 passed after adding the ledger contract, actual original/new file consumer comparisons and the final quantum payload failure point. The final focused command was:

```sh
python -m pytest tests/test_digest_1301/read-measurement/test_windows.py tests/test_digest_1301/read-measurement/test_windows_consumers.py -q --basetemp="$TMPDIR/pytest"
```

Final result: **84 passed**, Python 3.14.4, Torch 2.11.0+cpu.

Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"af074810ae1ee8a61c6dba614896a5de015c633a6c84a0ef93e996b2327781a3","size":349}},"operation":"e60c5c1386cea34a1978da8e3f4e0cfd140a3edaff3feb69bc1ede4f6c8b356b","receipt":"f5de10623ef97bf16b1dc9fcd00dd12037360d4fc94c72488b44969e6bab3ad4"}
