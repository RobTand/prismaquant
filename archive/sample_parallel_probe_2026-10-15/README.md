# Retired sample-parallel probe execution

Archived 2026-10-15 from integration 9a80849241711e78e7968e992cd06ab081b854c8.
Provenance: 2026-10-14 PrismaQuant architecture refactor, base 138069b7993.

The incremental worker has refused this mode since 2026-09-02. Its source census
producer retired with the Gridbook lane. This archive records the complete
producer, reducer, priority producer contract, pre-edit incremental implementation,
uncalled whole-model pass, tests and operator runbook. These files are history,
not an executable package. Their old imports are intentionally recorded verbatim.

The five manual offline commands prepare-calibration, prepare-worker-source-cache,
merge-importance, build-cover and merge are now removed from the live package.
No live tool or shell entry point invokes them. This is an explicit manual API
retirement, not an inference that nobody ever invoked a CLI.

Live readers use prismaquant.sample_parallel_artifacts for the unchanged recorded
schema/digest, source-census, dense-row and activation-priority contracts. The
incremental CLI retains its early refusal and cache admission validators. Existing
merged ActivationIndex reads retain their independent same-byte and row-stamp gates.
Restoration needs a lane-independent source census and a separately reviewed
execution contract; importing this archive is not a supported restoration.

## Frozen golden rows (added 2026-10-16)

`735f683ed5` removed two live golden-test parametrizations that called this
probe, but left their frozen rows in the live fixtures, where no test read
them. Claude review finding R3B-10 moved those rows here:

- `tests/fixtures/digest_profiles_1301.sample_parallel_probe.json`: 33 rows
  of `test_site_keeps_its_bytes_and_refusals[prismaquant.sample_parallel_probe._runtime_snapshot_closure_sha256]`
  from `tests/fixtures/digest_profiles_1301.json`. They are the outcomes of
  `sample_parallel_probe._runtime_snapshot_closure_sha256` on the generic
  inputs plus the site inputs
  `[{"path": "a", "sha256": "0" * 64}, {"path": "b", "sha256": "1" * 64}]`,
  `({"b": 1},)`, `[{"café": "ü"}]` and `[{"x": float("nan")}]`.
- `tests/fixtures/contract_checks_1300.sample_parallel_probe.json`: 13 rows
  of `test_text_loader_refuses_as_before[...-sample_parallel_probe]` from
  `tests/fixtures/contract_checks_1300.json`, the outcomes of
  `sample_parallel_probe._strict_json_loads(text, where="probe")`.

Each row line is the live fixture's line, byte for byte, except that the last
row in each file drops its trailing comma. The live fixtures before the move
had SHA-256 `cb693051eca8261e7c41a9ef597f503287a618b7f0a7b8f127dc7b30f5a26286`
and `5b4b778cb079ca99bbb16713647de100b33b8659fc3d6418c5fc4c2f4fe4605d` at
`b20d0d863989dea53e5353a1433bc859040f1fc4`. The test code that produced the
rows is in `tests/test_digest_profiles_1301.py` and
`tests/test_contract_checks_1300.py` at `735f683ed5^`. These rows are records;
no live test reads them.
