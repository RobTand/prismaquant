# Inspect release evidence without changing an export

Use the existing lane-aware shipcard CLI to see every current verifier refusal
before opening a canonical card or invoking the publisher. This is a CPU-only,
read-only diagnostic. It does not run serving gates, fill evidence, freeze
payloads, invoke Hub APIs, or upload anything.

## Run a preflight

Use a qualified PrismaQuant Python environment on the admitted worker. Set
`ARTIFACT` to its exported directory and `PYTHON` to that environment's executable.
Submit the check through PrismaBuild:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd "$PWD" --anywhere --cpus 1 --demand mem_gb=4 --priority -10 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  -- "$PYTHON" -m prismaquant.lane_shipcard preflight \
  --lane tessera --artifact "$ARTIFACT"
```

The JSON report contains:

- The required slots derived by the existing shipcard policy, including
  conditional obligations such as `uniform_control`.
- All existing verifier problems and the unfilled slots.
- Declared lane gate runners. `filled` means a record is present, **not** that
  it passed verification. Slots without a declared runner are listed separately;
  the diagnostic does not invent a runner or remove those obligations.
- The inspected card path, model identity and whether the card is provisional.

Exit codes are **0** when the shared verifier reports no problems, **1** for
reported refusals, and **2** for invalid inputs. A successful result is scoped
only to shipcard evidence (`publication_checked: false`), not release approval.

## Missing or external cards

If `<artifact>/shipcard.json` does not exist, preflight builds a provisional
record **in memory**, lists the missing-card refusal and all other refusals,
and exits1. It never writes that record. Use `--build-json` to supply actual
build-anchor facts for this provisional record; it does not manufacture a
route histogram or measurements.

Use `--shipcard PATH` to inspect an existing external record. An explicit missing,
malformed or wrong-lane card is an error, not permission to replace it. Build
facts cannot override an existing record during preflight. The publisher still
requires its own canonical in-tree card regardless of what an external card says.

## Finish the release separately

Stage final identity-bearing files, including the applicable LICENSE/NOTICE,
before opening the authoritative card and collecting bound evidence. README and
runtime-manifest handling follow the existing shipcard identity contract; this
command does not change it. In particular, the model digest is not a complete
native-weight payload hash.

Use `lane_shipcard open` explicitly to create the real card. Run and record the
existing qualification, quality and route checks, then repeat this preflight.
Finally run `tools/publish_artifact.py --dry-run` with the confirmed repository
ID. Its canonical-card and frozen-content checks remain mandatory. A dry run
is not permission to upload, and `--force-unverified` is not a substitute for
missing evidence (it can mutate a card even with `--dry-run`).
