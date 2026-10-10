#!/bin/bash
# Head walk measurement (#1492, #1247), CEO dec-1007-031239-b815 and dec-1008-140953-047f.
# One PrismaBuild x86 action on dl380g10. Starts only on a quiet pool and with no suite.
set -u
D=/home/rob/fleet/inventory
W=/home/rob/wt/lead-pq-integrator
R=/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/r13-stageb-20260923
LOG=$D/pq1492-headwalk-run-20261008.log
WLOG=$D/pq1492-headwalk-watch-20261008.jsonl
DONE=$D/pq1492-headwalk-done-20261008.txt
rm -f "$DONE"
echo "start $(date -u +%FT%TZ)" > "$LOG"
# 1. No suite of mine may run.
if systemctl --user list-units --state=active 'pq-batch*-suite.service' --no-legend | grep -q .; then echo "REFUSED: a suite runs" >> "$LOG"; exit 3; fi
# 2. The pool must be quiet for 120 s (HDD util below 20 percent, I/O full pressure below 5 percent).
python3 $D/pq1492-pool-watch.py quiet --log $D/pq1492-headwalk-quiet-20261008.jsonl --seconds 120 >> "$LOG" 2>&1 || { echo "REFUSED: pool not quiet" >> "$LOG"; exit 4; }
# 3. The watcher stops the action with one SIGTERM on dl380g10 at the three limits.
python3 $D/pq1492-pool-watch.py watch --stop-host dl380g10 --no-spark-busy --log "$WLOG" --done-file "$DONE" --max-seconds 5400 >> "$LOG" 2>&1 &
WPID=$!
sleep 20   # the first watcher samples set the pre-run read-wait baseline
cd $W
timeout 4800 /home/rob/tmp/pb-submit-celestia-20261003/bin/python /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py --tag x86 --cpus 4 --demand mem_gb=96 --timeout-s 4500 --wait-s 600 -- /home/rob/venvs/pq-pin-fca4c6ce0/bin/python -m tools.profile_stage_b_head --mode scoped-walk --quantum $R/meta-045-fae344d/records/layer-040.json --quantum-sha256 f179640dea63c585762243507a6c656b34d50f366cf4eecd6c03e09725b18348 --plan $R/a4/overlay/plan.json --plan-sha256 c9b879b48d96e7d6fa6a7d6bb57bc4f014cf1e8a5fe7ca1865ec36739f2aa3dc --scratch /tmp/pq1492-run --sweep-start 10000 --slice-units 249 --sweep-workers 4,1,8,2,2,8,1,4 >> "$LOG" 2>&1
echo "pbrun rc=$?" >> "$LOG"
echo "done $(date -u +%FT%TZ)" > "$DONE"
wait $WPID
echo "end $(date -u +%FT%TZ)" >> "$LOG"
