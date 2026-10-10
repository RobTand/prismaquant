# Head walk worker-count measurement, 2026-10-08 (Refs #1492, #1247)

This directory holds the files of one bounded, read-only measurement. It is not a pull request. The branch exists so that the files are not lost (CEO order D65). Where the files finally live is the owner's choice.

## What ran
- One PrismaBuild x86 action on dl380g10, tree `cad7513f0` (main with PR #2377), interpreter `/home/rob/venvs/pq-pin-fca4c6ce0/bin/python`.
- Command: `python -m tools.profile_stage_b_head --mode scoped-walk` with the sweep options `--sweep-start 10000 --slice-units 249 --sweep-workers 4,1,8,2,2,8,1,4`.
- Inputs: record `r13-stageb-20260923/meta-045-fae344d/records/layer-040.json` (sha256 `f179640dea63c585762243507a6c656b34d50f366cf4eecd6c03e09725b18348`) and plan `r13-stageb-20260923/a4/overlay/plan.json` (sha256 `c9b879b48d96e7d6fa6a7d6bb57bc4f014cf1e8a5fe7ca1865ec36739f2aa3dc`). Both are under `/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/`.
- Window: about 14:26Z to 14:30Z on 10-08. The sweep walked 1993 units: one baseline unit and eight slices of 249 units. The metadata loaded once.

## Files
- `run.sh`: the launcher. It refuses unless no suite unit runs and the pool is quiet for 120 s. It starts the watcher, then the action. The paths in it point at `/home/rob/fleet/inventory` and `/home/rob/wt/lead-pq-integrator`.
- `pool-watch.py`: the watcher. The first commit on this branch is the Spark version written on 10-07. The second commit adds the box mode used here (`--stop-host`, `--no-spark-busy`) and the disk read wait limit, because dl380g10 reads the pool as local ZFS and has no NFS read round trip.
- `result.json`: the tool's own report line. `watch.jsonl` and `quiet.jsonl`: the watcher samples. `run.log`: the launcher log. `done.txt`: the end marker.

## Result (as posted on #1492, comment 6062139535)
Mean wall seconds per 249-unit slice: 1 worker 17.11, 2 workers 9.02, 4 workers 6.83, 8 workers 4.27. Baseline 73.8 s, whole run 176.5 s. The pool peaked at 27 percent disk use, 8.35 ms read wait and 18.8 percent 10-second I/O pressure. No stop limit fired.

## Limits of this result
- The scoped walk ran with `verify_payloads=False`. It read the metadata and only stat the per-unit files. It did not read render contents, so it cannot set the `head_walk_workers` default.
- dl380g10 reads the pool as local ZFS (`nfs_ops_total` 0). The disk read wait stood in for the NFS read round trip.
- The metadata was mostly cached (`rchar` 15.5 GiB, `read_bytes` 1.9 GiB).
- One run, one host, two slices per worker count. The repeat slices at 1 and 4 workers ran 40 and 24 percent faster than the first ones.
- A content-hashing follow-up on a fixed render sample was never run. It needs a CEO decision on the sample size, and issuegraph refuses measurement work (gap posted on fleetgraph#16, comment 6084230207).
