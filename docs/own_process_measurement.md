# Isolated pytest measurement

Set `PQ_OWN_PROCESS_MEASURE=1` on an admitted PrismaBuild test action to
instrument the existing `own_process` child runner. Use pytest's `--basetemp`
to choose the run's temporary root. Each xdist worker uses the shared parent
of its own temporary root, as the existing runner does.

The ordinary path keeps the same child command, terminal state, and outcome
reports. Measurement adds cProfile to the child launch and records durations
from pytest's actual child reports. It preserves the chosen node IDs, group
lock, per-test bounds, and terminal-exit handling. It does not submit nested
PrismaBuild actions or drive pytest hooks itself.

The private child launcher uses `cProfile.Profile.runcall` and writes its
profile in `finally`. Pytest's `SystemExit` propagates unchanged. The standard
`python -m cProfile` command catches `SystemExit` and would turn pytest failure,
usage-error, and no-tests exit codes into success.

Files live under `<basetemp>/own-process/`. The key is the existing hash of
the selected child node IDs.

| File | Contents |
| --- | --- |
| `<key>.jsonl` | Original node IDs, outcomes, and child phase durations |
| `<key>.pstats` | cProfile output from the isolated pytest process |
| `<key>.child-measurement.json` | Shared child elapsed time and artifact paths |
| `<key>.<parent-pid>.measurement.json` | One caller's elapsed time and lock wait, plus shared child costs |
| `<key>.coverage` | Separate child coverage data when coverage is requested |

The existing `.done.json` terminal state retains its exact shape. Separate
sessions must use separate temporary roots. Workers share child artifacts
through the existing group lock and write their own caller measurements.

The caller measurement schema is `prismaquant.own_process_measurement.v1`.
`parent_elapsed_s` ends after reading the child reports. `lock_wait_s` covers
the blocking lock acquisition. `executed_child` identifies the one caller
that launches the child. Other callers reference that same child evidence.
`child_phase_s` sums setup, call, and teardown durations. Subtracting those
durations from `child_elapsed_s` gives `child_unattributed_s`. That residual
includes startup, collection, reporting, and exit. It does not isolate any
one of them.

Collection reports retain their outcomes and reasons without a phase
duration. Pytest's `CollectReport` has no `duration` field.

Count each shared child once. Do not add parent elapsed time to child phase
time or sum worker elapsed times as suite wall time. Child profile and
coverage paths name expected artifacts. A crash can leave them absent or
partial. Check the files and terminal outcome before treating them as
complete evidence.

## Collect coverage

Request coverage with the parent's usual `--cov`, `--cov-config`,
`--cov-branch`, and `--cov-context=test` options. Measurement forwards their
selection to a child pytest-cov invocation and gives it a separate
`COVERAGE_FILE`. The child loads the `pytest_cov` entry point explicitly so
disabled plugin autoload remains supported. Child reporting and its local
coverage threshold are disabled. Evaluate coverage acceptance after
combining the complete population.

Combine the parent's data and the child data explicitly with the supported
coverage API. Preserve the original files for audit:

```python
import coverage

combined = coverage.Coverage(data_file=combined_path)
combined.combine(data_paths=[parent_path, *child_paths], keep=True, strict=True)
combined.save()
```

The regression checks require an isolated sample's call line to carry its
original `nodeid|run` context, both before and after combination. They cover
serial and xdist sessions. Custom coverage configurations and the supplied
coverage version still need qualification in the measurement environment.
See the upstream [pytest-cov plugin options](https://github.com/pytest-dev/pytest-cov/blob/master/src/pytest_cov/plugin.py)
and [coverage combination API](https://coverage.readthedocs.io/en/latest/api_coverage.html#coverage.Coverage.combine).

## Verify the runner

Run these files through PrismaBuild with the supplied interpreter:

- `tests/test_own_process_measurement.py`
- `tests/test_own_process_isolation.py`
- `tests/test_own_process_child_exit_audit.py`

Also run the terminal-exit controls with `PQ_OWN_PROCESS_MEASURE=1`. Existing
controls check the exact terminal state and detect an exit after passing
reports. The measurement checks inspect real child profile frames, reconcile
phase totals, preserve disabled outcomes, and separate concurrent sessions.

The failing-before action
`de7a185d06ba79353cdbcb8f26ce93126db1631ae81b07b1389031d756527035`
at source `05b62b47427` plus the original regression overlay reports two
missing-measurement failures, one disabled-path pass, and one coverage skip.
Its supplied interpreter has neither coverage nor pytest-cov. Coverage
integration remains unavailable there and is required by #2804 before a
full-suite coverage baseline. No profiling or coverage packages are installed
by these tests.

Keep diagnostic instrumentation separate from the uninstrumented wall-time
baseline. This instrumentation supplies no suite speedup or root #1929
acceptance result.

## Retained verification failures

The enabled-measurement run at `b7f49eb73a4aee9eb794076fa48b77982ec8833e`,
action `9d88321c298eab9c04498df295ba11c83d989e23a0efd69f8d8b843ba62acdac`,
reported `3 failed, 29 passed, 2 skipped`. Its terminal controls caught the
standard cProfile command suppressing pytest's exit status. The focused
serial and xdist regression at `6f9d8c9ffb`, action
`aa4ce430ecf89c0a77f289d3db8dca9f7410b774acbb767585f85d659f263276`,
reported `2 failed, 8 deselected` on the recorded child exit assertion.

After the private launcher repair, the enabled controls at
`a75e90b3f915929f0c5b8e95a1fe08c7ad830402`, action
`aa840fadd284ff7229e66882c40ea46235bf6f62cda45ea6ac725c65244d9e9f`,
reported `1 failed, 31 passed, 2 skipped`. The remaining failure was an
attempt to read `CollectReport.duration`. The focused collection regression
at `3f529e178f`, action
`b0cf41454e5d41e8f652b206d6ed7f037bb9228bb327c50f70c2684e32b15528`,
reported `1 failed, 10 deselected` with that `AttributeError`.

These are correctness runs. Their times do not establish a performance
comparison. Final verification of both repairs remains required.
