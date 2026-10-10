# Integrator suite scripts (Refs #1929)

These are the three scripts the prismaquant integrator used from 2026-10-05 to 10-09 to run the full test suite on PrismaBuild before a merge. They were files in `/home/rob/fleet/inventory`, which is not a git repository. This branch exists so that the files are not lost (CEO order D65). It is not a pull request, and the placement under `docs/measurements` is only a holding place. The issuegraph merge trains now run the suite and merge (D57 rule 10), so these scripts are a record of the old procedure and a starting point, not a supported tool.

## Files
- `gen-argv.py`: builds three `pbtest` argument files for one checkout: an untagged group (12 shards, tag `x86`), a pinned group and a spill group (1 shard each, tag `dl380g10`). The interpreter comes from `PQ_PYTHON`.
- `run-suite.sh <name>`: reads the three argument files for `<name>` and runs the three `pbtest` commands in parallel. I ran it as a systemd user unit (`systemd-run --user`) so that a restart of the lead process did not kill it.
- `verify-suite-receipts.py --checkout DIR --head SHA RESULT.json ...`: checks each shard result against the head it claims. It checks the return code and the CAS receipt, the bundle digest against the CAS blob, and that the shard's snapshot has the head as its only parent and adds exactly one `.pbrun-closure.*.json` file. It sums the passed counts and exits 1 on any bad shard. I tested it on a known good run (14 receipts, 14 snapshots, 19397 passed) and on a wrong head (exit 1).

## What does not work as committed
- `gen-argv.py` cannot run. It does `os.chdir('/home/rob/tmp/pq-next-batch25')`, a tree that no longer exists, and it reads `pq-integrator-batch25-*-argv-20261005.json` and the batch25 history file from `/home/rob/fleet/inventory`. Those three JSON files still exist; the batch25 tree does not. It also names an interpreter default, `/home/rob/venvs/pq-d13-candidate-2dbac191`, that was lost in the 10-10 reboot.
- `run-suite.sh` writes a scratch file under `/tmp` and names `/home/rob/fleet/inventory`.
- All three call a `pbtest` client that lived under `/home/rob/tmp/pb-submit-celestia-20261003` (lost in the reboot). The scripts themselves do not name it; the commands in the argument files do.
- `verify-suite-receipts.py` works with any checkout that holds the head. It reads the CAS under `/mnt/shared/prismabuild-fleet/cas/blobs`.

## Lessons the scripts encode
- Report the number of tests that ran and the number that passed separately. The sum of the shard summaries counts passes only; skips and xfails are in neither.
- A suite result binds to one head. When main moves, a vehicle needs a new head, a new review and a new run.
