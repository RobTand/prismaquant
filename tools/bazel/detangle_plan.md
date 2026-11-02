# Detangle planner inputs and estimates

`detangle_plan.py` replays a fixed merge history against static reach and
observed execution, weighted by test-file runtime. It performs offline
analysis and does not run tests. `detangle_inputs.py` maps gate inputs and
locked distributions into that graph.

The CLI defaults to the narrowed fallback. `--fallback-rule old` selects
the earlier rule. JSON and Markdown reports compare both policies.

The narrowed rule selects everything for the gate's own workflow,
Bazel configuration and build files, `tools/bazel/**`, and the gate tooling
listed in `Config.narrowed()`. It does not classify other workflows,
`.github/scripts`, `.mergify.yml`, or documentation as fallback. Existing
file edges select their readers. An input with no reader selects no tests.

For each historical `uv.lock` edit, the planner compares the full package
records before and after the merge. Changes include removed distributions,
wheel hashes, sources, versions, and dependency metadata. The resulting
`wheel:<normalized-name>` nodes select their graph dependents. Explicit
readers of the raw lockfile remain selected through file edges. Lockfile
metadata changes that alter no package record select only explicit lockfile readers. Missing lock
diff metadata fails closed. The wheel graph unions dependency markers and
versions from the analyzed source tree's lockfile. Distribution import aliases
are explicit in `IMPORT_NAMES`. The repository's root editable distribution
is excluded from wheel imports; its source-module edges remain authoritative.

`pytest.ini` configures every pytest invocation. The gate installs the
editable project with `pyproject.toml` and `.github/ci-constraints.txt` in
every shard. These inputs therefore select every test through explicit
input mappings. They are reported as selected graph cost, outside fallback.
This keeps fallback share separate from other causes of full selection.
Conftests retain their existing graph edges and scope.

The original coverage sweep retained repository filenames only. Dynamic
wheel selection uses imports in observed repository source files, plus the
lock dependency graph. This is an estimate: executing a module does not prove
that every conditional import ran, and untraced dynamic imports can be missed.
The report disables stopping claims when the replay contains lock edits.
Untraced data reads, skipped behavior, SDK availability, and execution pairs
outside static reach also limit the interpretation of the dynamic floor.

`--warm-timings warm.json` accepts a mapping from test path to seconds, or
`{"files": {"tests/test_x.py": 4.0}}`. A runtime candidate saves
`(current_seconds - target_seconds) * selection_frequency` seconds per merge.
Fallback selections count toward frequency. Greedy ranking recomputes savings
after each graph cut or runtime change, so overlapping gains are not counted
twice and runtime fixes can reduce fallback cost.

`--runtime-fixes fixes.json` accepts `effort`, `estimates`, `measured`, and
`default_effort` arguments to `cheaper_tests`. Measured runtimes take precedence
over estimates, which take precedence over warm timings. For example:

```json
{
  "effort": {"tests/test_x.py": 25},
  "measured": {"tests/test_x.py": 3.5}
}
```

Effort is an estimated changed-line count, shared with graph cuts. The default
is 100 lines and callers can override it. Runtime targets must be finite,
nonnegative, and no larger than the current runtime. Cold versus warm
measurements suggest possible savings; they do not establish that an actual
fix achieves the warm runtime. Use `measured` to record a completed fix.

Each report includes every test's runtime, static and dynamic selection
frequency, and static and dynamic seconds per merge, sorted by static cost.
Full-suite hosted durations can be supplied as pluggable estimates when paired
warm measurements are unavailable. Such rankings are planning estimates,
not measured performance claims.
