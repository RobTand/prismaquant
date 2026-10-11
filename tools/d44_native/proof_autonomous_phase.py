"""Exercise the actual autonomous CLI and native CPU children inside PrismaBuild.

Use an isolated native queue and CAS, as the accepted native controller proof
uses. The CPU executor runs real child commands and publishes real receipts.
The read-only disk service runs the actual fleet-diskcheck for each admission.
This proof executes no science and has no production GPU descendants.
"""
import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import next_wave as wave
sys.path.insert(0, '/mnt/shared/prismabuild-fleet/repo/tools')
import pbcampaign as pc

PRODUCER = r'''
import hashlib,json,sys
from pathlib import Path
batch=json.loads(Path(sys.argv[1]).read_text())
rows=[]
for task in batch['tasks']:
    value={'sum_of_squares':sum(v*v for v in task['payload']['values'])}
    raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    rows.append({'task_id':task['id'],'output_id':task['output_id'],
                 'value_sha256':hashlib.sha256(raw).hexdigest()})
Path(batch['result_manifest_path']).write_text(json.dumps({
    'schema':'prismabuild.child_result_manifest.v1',
    'parent_key':batch['parent_key'],'plan_key':batch['plan_key'],
    'child_ordinal':batch['child_ordinal'],'results':rows}))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disk-address', required=True)
    parser.add_argument('--disk-port', type=int, required=True)
    options = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    with tempfile.TemporaryDirectory(prefix='d44-autonomous-native-', dir='/tmp') as scratch:
        root = Path(scratch)
        work = root / 'work'
        work.mkdir()
        (work / 'seed.txt').write_text('Native CPU arithmetic proof only.\n')
        for args in [('init', '-q'), ('config', 'user.email', 'test@example.invalid'),
                     ('config', 'user.name', 'D44 CPU proof'), ('add', 'seed.txt'),
                     ('commit', '-qm', 'Native CPU fixture')]:
            subprocess.run(['git', '-C', str(work), *args], check=True, capture_output=True)
        wave.QUEUE, wave.CAS = root / 'pb-queue', root / 'cas'
        pc.pbrun.SH = root
        queue = pc.pool.PoolQueue(wave.QUEUE)
        queue.announce(host=socket.gethostname(), tags=['x86'], has_gpu=False,
                       capacity={'cpu': 2, 'mem_gb': 2, 'gpu': 0})
        wave.routed_api = lambda: pc
        evidence = 'cas:sha256:' + '0' * 64
        request = {'schema': pc.dc.LOGICAL_REQUEST_SCHEMA_V1,
            'common': {'argv': ['python3', '-c', PRODUCER, pc.dc.TASK_BATCH_PLACEHOLDER],
                       'cwd': str(work), 'demand': {'cpu': 1, 'mem_gb': 1},
                       'gpu_memory_gb': None, 'data_manifest': None,
                       'env': {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                               'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                       'tags': ['x86'], 'timeout_s': 30},
            'roster': {'schema': pc.dc.LOGICAL_TASK_ROSTER_SCHEMA_V1, 'tasks': [
                {'id': f't{i}', 'output_id': f'out{i}', 'payload': {'values': [i, i + 1, i + 2]},
                 'residency_key': 'r', 'estimated_seconds': 10.0,
                 'estimate_evidence': evidence} for i in range(8)]},
            'batch_policy': {'schema': pc.dc.ROSTER_BATCH_POLICY_SCHEMA_V1, 'residencies': [
                {'key': 'r', 'setup_seconds': 1.0, 'setup_evidence': evidence}],
                'max_setup_fraction': 0.2, 'max_estimated_wall_seconds': 11.0}}
        plan_file = root / 'logical.json'
        plan_file.write_text(json.dumps(request))
        state_dir = root / 'state'
        state_dir.mkdir()
        wait_file = root / 'wait.json'
        _, prepared_group = pc.decompose(pc.dc.validate_logical_request(request), transport='pool', priority=0)
        catalog = {'parent_key': prepared_group['plan']['parent_key'],
            'roster_input': next(item for item in prepared_group['children'][0]['inputs']
                                 if item['id'] == pc.dc.TASK_ROSTER_INPUT_ID),
            'request': {name: value for name, value in prepared_group['request'].items() if name != 'roster'}}
        wave.durable_json(state_dir / 'catalog.json', catalog)
        cas = pc.pb.PrismaBuildCAS(wave.CAS)
        native = pc.decomposition_dir(cas, catalog['parent_key'])
        keys = pc._stored_document(native / 'publication.json')['child_action_keys']
        assert len(keys) == 8
        for key in keys:
            queue.item_path('ready', key).unlink()
        seed = root / 'closed-balance.json'
        seed.write_text(json.dumps({'schema': 'fleet.d44.native_phase_ledger_seed.v1',
            'authoritative_seed_GPU_hours': 21.187425203522047,
            'global_ceiling_GPU_hours': 110, 'GPU_hour_multiplier': 1}))
        disk_command = shlex.join(['python3', str(Path('proof_disk_client.py').resolve()),
                                  '--address', options.disk_address, '--port', str(options.disk_port)])
        env = {**os.environ, 'D44_WAVE_QUEUE_DIR': str(wave.QUEUE),
               'D44_WAVE_CAS_DIR': str(wave.CAS), 'D44_DISKCHECK': disk_command}
        command = ['python3', str(Path('next_wave.py').resolve()), '--routed-plan', str(plan_file),
                   '--state-dir', str(state_dir), '--wait-file', str(wait_file),
                   '--ledger-seed', str(seed), '--disk-need-gb', '0.1']
        initial = subprocess.run(command, env=env, capture_output=True, text=True, timeout=150)
        print(initial.stdout, end='', flush=True)
        assert initial.returncode == 3, initial.stderr
        saved = json.loads((state_dir / 'wave-state.json').read_text())
        assert [m['key'] for w in saved['waves'] for m in w['members']] == keys[:1]
        executed, errors, receipts = [], [], {}
        stop = threading.Event()
        max_live = [0]

        def execute_cpu_children():
            try:
                while not stop.is_set():
                    ready = list(queue.dir('ready').glob('*.json'))
                    claimed = list(queue.dir('claimed').glob('*.json'))
                    max_live[0] = max(max_live[0], len(ready) + len(claimed))
                    assert len(ready) + len(claimed) <= 1
                    if not ready:
                        stop.wait(0.01)
                        continue
                    row = json.loads(ready[0].read_text())
                    key = row['action_key']
                    assert key not in executed
                    row.update(claimed_unix=time.time(), claimed_host=socket.gethostname(),
                               claimed_by='isolated-native-cpu-executor')
                    wave.durable_json(queue.item_path('claimed', key), row)
                    ready[0].unlink()
                    child = cas.read_action_request(key)
                    assert child['params']['demand'].get('gpu', 0) == 0
                    started = time.monotonic()
                    with pc.pbrun.materialize._execution_checkout({
                            'action_key': key, 'cas_root': str(wave.CAS),
                            'checkout_snapshot': child['params']['checkout_snapshot']},
                            local_checkout_root=root / 'checkouts') as checkout:
                        attestation = pc.pb.preflight_action(child, cas_root=wave.CAS, checkout_root=checkout)
                        cwd = checkout / child['task']['working_directory']
                        result = subprocess.run(child['task']['argv'], cwd=cwd,
                            env={**os.environ, **child['environment']['variables']},
                            capture_output=True, text=True, check=True, timeout=30)
                        receipt, won = cas.publish_result(child, cwd / child['task']['result_path'],
                            attestation=attestation,
                            precommit_verify=lambda: pc.pb.verify_code_closure(child['code_closure'], checkout))
                        assert won and result.returncode == 0
                    row.update(finished_unix=time.time(), finished_host=socket.gethostname())
                    detail = {'elapsed_s': time.monotonic() - started, 'returncode': 0,
                              'stdout': result.stdout, 'stderr': result.stderr}
                    row['attempt_history'] = queue.archive_attempt(row, attempt=1, status='executed',
                                                                   disposition='done', detail=detail)
                    row.update(schema=pc.pool.POOL_OUTCOME_SCHEMA_V1, attempts=1)
                    adopted = queue.adopted_attempt_summary(row)
                    row.update({field: adopted[field] for field in
                                ('status', 'finished_unix', 'finished_host', 'detail')})
                    wave.durable_json(queue.item_path('done', key), row)
                    queue.item_path('claimed', key).unlink()
                    executed.append(key)
                    receipts[key] = cas.lookup(child)
            except BaseException as exc:
                errors.append(repr(exc))

        executor = threading.Thread(target=execute_cpu_children, daemon=True)
        executor.start()
        try:
            completed = subprocess.run([*command, '--autonomous'], env=env,
                                       capture_output=True, text=True, timeout=300)
        finally:
            stop.set()
            executor.join(timeout=30)
        print(completed.stdout, end='', flush=True)
        print(completed.stderr, end='', file=sys.stderr, flush=True)
        assert not errors, errors
        assert not executor.is_alive()
        if completed.returncode != 0:
            print('D44_AUTONOMOUS_FAILURE=' + json.dumps({
                'returncode': completed.returncode, 'executed': executed,
                'state': json.loads((state_dir / 'wave-state.json').read_text())}), flush=True)
        assert completed.returncode == 0, completed.stderr
        assert executed == keys
        state = json.loads((state_dir / 'wave-state.json').read_text())
        assert len(state['disk_checks']) == 8
        assert [row['action_key'] for row in state['disk_checks']] == keys
        assert all(row['evidence']['pass'] is True for row in state['disk_checks'])
        assert state.get('pending_submission') is None
        assert state['last_completion']['succeeded'] is True
        ledger = json.loads((state_dir / 'lifetime-cost.json').read_text())
        assert ledger['seed']['authoritative_seed_GPU_hours'] == 21.187425203522047
        assert ledger['charges'] == {} and ledger['reservations'] == {}
        group = json.loads((native / 'group.json').read_text())
        expected = [[f't{i}', f'out{i}', hashlib.sha256(json.dumps({
            'sum_of_squares': sum(v * v for v in [i, i + 1, i + 2])},
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()] for i in range(8)]
        assert group['task_count'] == group['child_count'] == 8
        assert group['merged_result_sha256'] == pc.pb.canonical_sha256(expected)
        duplicate = subprocess.run([*command, '--autonomous'], env=env,
                                   capture_output=True, text=True, timeout=150)
        assert duplicate.returncode == 0, duplicate.stderr
        assert len(json.loads((state_dir / 'wave-state.json').read_text())['disk_checks']) == 8
        assert max_live[0] == 1
        print('D44_AUTONOMOUS_CPU_PROOF=' + json.dumps({
            'actual_entry': 'next_wave.py --routed-plan --autonomous',
            'restarted_after_first_native_admission': True, 'autonomous_exit': completed.returncode,
            'duplicate_exit': duplicate.returncode, 'native_CPU_children': len(executed),
            'maximum_live_children': max_live[0], 'fresh_actual_D1_checks': state['disk_checks'],
            'closed_balance_GPU_hours': ledger['seed']['authoritative_seed_GPU_hours'],
            'GPU_admissions': 0, 'science_replayed': False,
            'group': group, 'native_receipts': receipts,
            'next_wave_sha256': hashlib.sha256(Path('next_wave.py').read_bytes()).hexdigest()},
            sort_keys=True), flush=True)
    print('D44_AUTONOMOUS_CLEANUP=isolated CPU roots removed', flush=True)


if __name__ == '__main__':
    main()
