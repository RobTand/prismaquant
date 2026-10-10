#!/bin/bash
n=$1; d=/home/rob/fleet/inventory
python3 - "$n" <<'P' > /tmp/pq-$n-cmds.sh
import json,sys,shlex
n=sys.argv[1]
for k in ('untagged','pinned','spill'):
    a=json.load(open(f'/home/rob/fleet/inventory/pq-integrator-{n}-{k}-argv-20261005.json'))
    print(shlex.join(a)+f' > /home/rob/fleet/inventory/pq-integrator-{n}-{k}-20261005.log 2>&1 &')
print('wait')
P
bash /tmp/pq-$n-cmds.sh
echo "$n done $(date -u +%FT%TZ)" > $d/pq-integrator-$n-done-20261005.txt
