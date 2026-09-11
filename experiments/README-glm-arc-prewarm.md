# GLM census ARC prewarm

Two programs, both aimed at one number: the ~1.83 Gbit/s a campaign row gets
while it reads its 864 capture files and its layer's weights off dl380g10.

## The problem

The census reads its whole 1.894 TiB capture working set exactly once, spread
over 132 rows of ~49.4 GB each, through a server whose ARC is ~200 GB. A byte
is read once and never asked for again, so no caching policy applied *after* a
read can help. The only fast read is one that was already resident, and the only
time available to make it resident is the ~1.8 hours the current row spends on
the GPU after its own load finishes.

## `glm_pool_read_ceiling.py` — is 204 MB/s the pool or the client?

Reads disjoint prefixes of already-priced expert rows straight off the local
pool mount at 1, 4, 8 and 16 concurrent readers, with a repeated single-reader
arm last as a drift control. Each arm gets its own cold set, so no arm warms
another. Coldness is established after the fact from the raidz1 members'
`/proc/diskstats` read bytes against the arm's logical bytes, and arcstats plus
every relevant `/sys/module/zfs/parameters` value are recorded on both sides.

Run it through PrismaBuild, pinned to the server:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /home/rob/tmp/pq-glm-arc-prewarm \
  --tag dl380g10 --cpus 4 --demand mem_gb=4 \
  --priority -10 --timeout-s 3000 --detach \
  -- /home/rob/venvs/pq-cpu312/bin/python experiments/glm_pool_read_ceiling.py \
     --out <BASE>/arc-prewarm-20260910/pool-read-ceiling-01.json
```

`glm_pool_read_ceiling_arms.json` holds the arm definitions, so the file sets
travel in the action's snapshot and the measurement is reproducible.

## `glm_arc_prewarm.py` — the prewarm daemon

Predicts which rows run next, derives exactly what each will read, and pulls
those bytes into ARC ahead of the row.

**Prediction.** It reads PrismaBuild's ready queue and reproduces PB's own claim
order — `(-priority, -passes, published_unix)`, from `prismabuild/pool.py` — then
keeps the items that resolve to a row of this campaign. An action key becomes a
row id through the sealed CAS request's argv (`--units .../units/row-XXXX.json`);
a roster file is only a fallback. It predicts *which rows*, never which box:
placement is PrismaBuild's and happens at claim time, so warming the top
`window x (GPU hosts)` ready rows is correct whichever Spark claims first.

**Read set.** Per row: the capture files named by the campaign's capture
manifest, in the order `prefetch_capture` consumes them (sorted member name);
the byte ranges of the layer's selected weight tensors inside the safetensors
shards, parsed from each shard's header and coalesced to 1 MiB record
boundaries, because `--source-snapshot-policy selected-tensors-v1` reads ranges
rather than whole shards; and the seed checkpoint's manifest and unit shards
when the row declares one. The 16 remaining rows declare no seed checkpoint, so
that term is zero for them.

**Safety.** It holds off while any claimed row is still inside its own load
phase, detected by the immutable `capture-load-execution-<sha>.json` the row
writes beside its cache (with a `--load-grace-s` fallback), so the current row's
still-needed files are never evicted for the next row's. It reads ARC `size`
against `c_max` and spends at most `--arc-reserve-fraction` of the headroom. It
opens PrismaBuild queue files read-only and never writes one, and logs only
`action_key -> row_id -> bytes`, never raw queue or CAS records.

### Modes

```bash
# what would be warmed, given the live queue (no reads)
python experiments/glm_arc_prewarm.py --once --dry-run

# the same against a hypothetical ready set
python experiments/glm_arc_prewarm.py --once --dry-run \
    --simulate-ready row-0065,row-0084,row-0064

# one row, for real (the acceptance warm)
python experiments/glm_arc_prewarm.py --warm-row row-0079 --readers 8

# service
systemd-run --user --unit glm-arc-prewarm \
  /home/rob/venvs/pq-cpu312/bin/python experiments/glm_arc_prewarm.py --daemon --dry-run
```

Switch a dry-run service to live warming by stopping the unit and starting it
without `--dry-run`:

```bash
systemctl --user stop glm-arc-prewarm
systemd-run --user --unit glm-arc-prewarm \
  /home/rob/venvs/pq-cpu312/bin/python experiments/glm_arc_prewarm.py \
    --daemon --readers 8 --window 1
```

### Cost to watch

ZFS ARC is not reclaimable through `MemAvailable` on Linux, so ARC growth
lowers dl380g10's advertised `observed_capacity.mem_gb` in
`pb-queue/workers/dl380g10.json` roughly one for one. Warming a 64 GB row
therefore tightens CPU admission on that box for everything else. Check that
field before and after a live warm.
