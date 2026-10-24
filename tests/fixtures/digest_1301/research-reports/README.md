# Research/report serializer prerequisite handoff

This is **not a completed migration**. `ledger.json` records 26 blocked
supplemental JSON calls in the fifteen assigned files, including
`freeze_refactor_checkout`. No raw call was replaced, no retention is asserted,
and `completed_calls` is deliberately empty. Source identities match the frozen
research-base registry. Consumer source, owner, scanner, baseline, census,
qualification files and deployment rosters are unchanged.

## Missing exact owner contracts

The inspected `prismaquant/digests.py::JsonProfile._encoder` unconditionally uses
`sort_keys=True`. All 26 assigned calls omit sorting. Passing an insertion-ordered
object to a sorted owner does not preserve its bytes, and bypassing the owner
with a new local encoder would not complete the route. Agreement S restricts
this piece to finished exact APIs; return this prerequisite to owner-recipes:

* insertion-order ASCII / default spacing / lax nonfinite values / no fallback;
* insertion-order ASCII / indent two / lax nonfinite values / no fallback;
* insertion-order ASCII / indent two / strict nonfinite refusal / no fallback;
* streaming `dump` for the first recipe, retaining partial writes on errors.

Keep newlines at the existing consumer boundaries. The consumer tests distinguish
strict JSON+LF from lax status files without LF, event JSONL, atomic status
replacement, exclusive-create refusal, and original workload/observer error and
cleanup order. `json.dump` cannot generally become `handle.write(profile.text())`:
that buffers the document and changes partial-file outcomes on encoding failure.
No owner name is invented here; `owner_route` remains null pending the prerequisite.

## Independence and bounded audit

`original_consumers.json` contains only verbatim original helper fragments,
pinned by full source blob and commit, not copies of the fifteen source files.
The tests execute those original bodies and the corresponding current bodies.
Literal bytes additionally establish insertion order, ASCII escaping, lax NaN,
LF framing and digest-independent error outcomes. Process APIs use explicit
adapters, never production workloads; the observer test extracts only its event
helper and does not import its host-contacting module body. A deliberately
injected status-loader result tests streaming failure at the writer seam; it is
not a claim that unsupported Python objects can pass the production JSON loader.

Coverage is preparatory and bounded to five consumer helpers (nine executed
cases), not compatibility proof for all 26 calls. Other publication paths,
readiness failures, malformed status records, complete CLI output, and numerical
or workload behavior still need consumer-specific evidence after exact owners
are available. No GPU, native-reader, model, profiling campaign, Netdata host,
freeze/worktree-creation, or fleet publication was launched or qualified.

In `paired_trade_report_proof`, independent synthetic fixture construction and
independent assignment/arithmetic checks do not imply a requirement for an
independent serializer. Its baseline and archived proof JSON have no discovered
serialization-retention witness. Preserve the construction and checks when the
exact insertion-order owners are supplied. Research, preflight and provisional
labels do not justify retention. The ledger's supplemental boundary is literal
`json.dump`/`json.dumps` AST calls in the assigned files, not proof that aliases,
generated code, shell snippets or deployment populations are exhaustive. The
parent's independent final audit remains necessary.

## Observed checks

Final focused command:

```sh
python -m pytest tests/test_digest_1301/research-reports/test_original_publication.py tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py tests/test_duplication_baseline.py -q --basetemp="$TMPDIR/pytest"
```

211 passed, 14 existing torch deprecation warnings, no failures/skips.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"5a0c9c11673830acfe4f991b31ba332c0eeb72e5edd3ca167163cdc649c536e9","size":983}},"operation":"ff1f60d347b3c479a6fd33cd7cff27cf4bd13ace1534da92da0d4e7ed6771ed9","receipt":"7b1f20558591cb1a9ad357c8ccb4d283769158a757f6bb734e613baa448e1b0b"}]

Parent closure check (unchanged acceptance files):

```sh
python -m pytest tests/test_digest_parent_completion_1301.py -q --basetemp="$TMPDIR/pytest"
```

Four failures remain: stale current census, unresolved eligible calls, missing
completed-call evidence, and allocator's raw Pareto constructor. These are not
passes or acceptance-test defects, and this assigned slice does not own the
shared census/baseline or allocator. Closure is not claimed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"a030e9b958bc48fe09345f0e2c1ff8715b8a87e137eb16f7209dbad811fa2689","size":11313}},"operation":"b5622209f6a35fa1fc0a2902139fdda420e8bc1bd233e04281c95ee3aa9f8a70","receipt":"ac4f721f9f3b727adf2a4e3ce762ae71c79bfd8e6264ea279248ed7045280576"}]
