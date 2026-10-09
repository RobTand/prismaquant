#!/usr/bin/env python3
"""Same encode entry on CPU; existing producer container on GPU.

D30 floors are imported from the current G3 launcher, not retuned. The producer
is not an immutable-source-provider claim. PB owns placement and batch cuts.
The host launcher publishes real child results only after the numerical child succeeds.
Dry runs write preflight files. They never publish scientific child results.
The source workspace remains read-only.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import stage1

OWNER = stage1.OWNERS
sys.path.insert(0, str(OWNER))
spec = importlib.util.spec_from_file_location('stage1_d30_owner', OWNER/'v2_launch.py')
D30 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D30)


def _publish_batch_result(batch_path, root, stage='encode', *, dry_run=False):
    if dry_run:
        return
    batch = stage1.load(batch_path)
    directory = 'grid' if stage == 'select-unit' else 'receipts'
    results = [{'task_id': task['id'], 'output_id': task['output_id'],
                'value_sha256': stage1.sha(root/directory/(task['payload']['qname'].replace('.', '__')+'.json'))}
               for task in batch['tasks']]
    stage1.save(batch['result_manifest_path'], {'schema':'prismabuild.child_result_manifest.v1',
                **{key: batch[key] for key in ('parent_key','plan_key','child_ordinal')}, 'results':results})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu','cuda'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage',choices=('rank','encode','closure-unit','select-unit'),default='encode')
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args, rest = parser.parse_known_args()
    entry = 'd44_training.py' if args.stage == 'select-unit' else ('d44.py' if args.stage == 'closure-unit' else 'stage1.py')
    payload = [entry,args.stage,'--device',args.device,'--root',str(args.root),*rest]
    if args.dry_run:
        payload.append('--dry-run')
    if args.batch is not None:
        payload.extend(['--batch', str(args.batch)])
    if args.device == 'cpu':
        # D38 invokes exactly this entry, without Docker/GPU and with actual H.
        rc = subprocess.call([sys.executable,*payload])
        if rc == 0 and args.batch is not None:
            _publish_batch_result(args.batch, args.root, args.stage, dry_run=args.dry_run)
        return rc
    args.root.mkdir(parents=True,exist_ok=True)
    stage1.require(D30.available() >= 8*2**30, 'D30 available below 8GiB start floor')
    launch = stage1.load(D30.TEMPLATE)
    container = launch['spec']
    for mount in container['container']['mounts']:
        if mount['target'] == '/out':
            mount['source'] = str(args.root)
    container['container']['mounts'].append({'source':'/mnt/shared','target':'/mnt/shared','readonly':True})
    # Only declared output is writable. The PB task-batch CAS remains readable
    # through /mnt/shared, never an application-managed host partition.
    from g3_residency import container_contract
    mounts, env = container_contract()
    container['container']['mounts'].extend(mounts)
    container['env'].update(env)
    container['env'].update(PRISMAQUANT_DEV_MODE='1',G3_PQ_ROOT='/pq',TESSERA_SRC='/tessera/src')
    container['env']['G3_HOST_MOUNTS'] = json.dumps({m['target']:m['source'] for m in container['container']['mounts']})
    container['container']['mounts'].append({'source':str(args.root),'target':str(args.root),'readonly':False})
    command = ['python3','/workspace/'+payload[0],*payload[1:]]
    token = hashlib.sha256(json.dumps(command).encode()).hexdigest()[:20]
    stage1.save(args.root/('encode-launch-'+token+'.json'),{'spec':container,'command':command,'resident_tier':'PB stage+auto RAM','guard':'unchanged G3 D30 8GiB start / 2GiB abort'})
    child = subprocess.Popen([sys.executable,'v2_launch.py','--container',json.dumps(container),json.dumps(command)])
    stop = threading.Event()
    guard = {'min_available_bytes':D30.available(),'abort_below_bytes':2*2**30}
    def monitor():
        while not stop.wait(1):
            available = D30.available()
            guard['min_available_bytes'] = min(guard['min_available_bytes'],available)
            if available < 2*2**30:
                guard['abort'] = 'D30 MemAvailable below 2GiB'
                child.terminate()
                return
    thread = threading.Thread(target=monitor,daemon=True)
    thread.start()
    try:
        rc = child.wait(timeout=1800)
        stage1.require('abort' not in guard,guard.get('abort','D30 abort'))
        if rc == 0 and args.batch is not None:
            _publish_batch_result(args.batch, args.root, args.stage, dry_run=args.dry_run)
        return rc
    finally:
        stop.set(); thread.join()
        if child.poll() is None:
            child.terminate(); child.wait()
        stage1.save(args.root/('encode-guard-'+token+'.json'),guard)


if __name__ == '__main__':
    raise SystemExit(main())
