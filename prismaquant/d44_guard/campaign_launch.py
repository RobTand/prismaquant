#!/usr/bin/env python3
"""D44 shared guard for grid, final encode, and encode-cost entries.

The frozen numerical source stays unchanged. CPU entries do not use Docker.
Use route-plan PATH --out OUT after each frozen plan builder. The input plan stays unchanged.
The adapter maps residency identifiers together and validates the native request before output.
The shared guard also supports final encode result publication.
Cancellation before decide_commit rejects the result. Cancellation after that decision cannot undo its publication.

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
import signal
from pathlib import Path
import subprocess
import sys
import threading
import time
import stage1
import stage1 as S

OWNER = stage1.OWNERS
sys.path.insert(0, str(OWNER))
spec = importlib.util.spec_from_file_location('stage1_d30_owner', OWNER/'v2_launch.py')
D30 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D30)


def _publish_batch_result(batch_path, root, stage='encode', *, dry_run=False):
    """Write the owned scientific result manifest. Report if the write ran."""
    if dry_run:
        return False
    batch = stage1.load(batch_path)
    directory = 'grid' if stage == 'select-unit' else 'receipts'
    results = [{'task_id': task['id'], 'output_id': task['output_id'],
                'value_sha256': stage1.sha(root/directory/(task['payload']['qname'].replace('.', '__')+'.json'))}
               for task in batch['tasks']]
    stage1.save(batch['result_manifest_path'], {'schema':'prismabuild.child_result_manifest.v1',
                **{key: batch[key] for key in ('parent_key','plan_key','child_ordinal')}, 'results':results})
    return True

def decide_commit(guard, returncode):
    """Cancel before the decision rejects the result. Cancel after it cannot undo commit."""
    # This single read is the commit decision point. The monitor has already stopped.
    abort = guard.get('abort')
    decision = {'allowed': returncode == 0 and abort is None,
                'abort_before_decision': abort, 'child_returncode': returncode}
    guard['commit_decision'] = decision
    return decision


OWNER_ENV = 'PRISMABUILD_CONTAINER_OWNER'
OWNER_LABEL = 'prismabuild.action'
ENTRY = {'select-unit': 'd44_training.py', 'encode-cost': 'd44_subsample.py', 'encode': 'stage1.py'}


class LauncherSignal(Exception):
    pass


def valid_owner(owner):
    return isinstance(owner, str) and len(owner) == 64 and all(c in '0123456789abcdef' for c in owner)


def group_alive(pgid):
    """True while ANY process of the group exists, also after its leader has exited (review fix 1)."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _docker(args, *, deadline=None):
    """A failed command or an expired deadline is not an empty result."""
    timeout = 5 if deadline is None else min(5, deadline - time.monotonic())
    if timeout <= 0:
        return None, 'D30 cleanup deadline expired'
    try:
        r = subprocess.run(['docker', *args], capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return None, repr(exc)
    if r.returncode != 0:
        return None, f'docker {args[0]} rc {r.returncode}: {r.stderr.strip()[-300:]}'
    return r.stdout.split(), None


def owned_containers(owner, *, deadline=None):
    return _docker(['ps', '-q', '--filter', f'label={OWNER_LABEL}={owner}'], deadline=deadline)


def halt_child(child, owner, reason, term_wait_s=10, kill_wait_s=5, docker_wait_s=30):
    """Stop the process group and owned containers with TERM, grace, then KILL.

    Every Docker query must succeed. No live process or container can remain.
    """
    S.require(valid_owner(owner), 'D30 halt needs the PB container owner')
    started = time.monotonic()
    pgid = child.pid
    steps, errors = [], []
    cleanup_deadline = started + term_wait_s + kill_wait_s + docker_wait_s
    before, err = owned_containers(owner, deadline=cleanup_deadline)
    if err:
        errors.append(err)
    remaining = before
    for sig, wait_s in ((signal.SIGTERM, term_wait_s), (signal.SIGKILL, kill_wait_s)):
        if remaining:
            command = ['kill', '--signal', sig.name, *remaining] if sig == signal.SIGTERM else ['kill', *remaining]
            _, err = _docker(command, deadline=cleanup_deadline)
            step = 'docker SIGTERM ' if sig == signal.SIGTERM else 'docker kill '
            steps.append(step + ' '.join(remaining))
            if err:
                errors.append(err)
        child.poll()
        if group_alive(pgid):
            try:
                os.killpg(pgid, sig)
                steps.append(sig.name)
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors.append(repr(exc))
        deadline = min(cleanup_deadline, time.monotonic() + wait_s)
        while True:
            child.poll()
            remaining, err = owned_containers(owner, deadline=cleanup_deadline)
            if err:
                errors.append(err)
            if not group_alive(pgid) and remaining == []:
                break
            if err or time.monotonic() >= deadline:
                break
            time.sleep(0.2)
        if not group_alive(pgid) and remaining == []:
            break
    deadline = cleanup_deadline
    while True:
        if time.monotonic() >= deadline:
            errors.append('D30 cleanup deadline expired')
            break
        child.poll()
        remaining, err = owned_containers(owner, deadline=cleanup_deadline)
        if err:
            errors.append(err)
            remaining = None
            break
        if not group_alive(pgid) and remaining == []:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.2)
    group_after = group_alive(pgid)
    ok = not group_after and not errors and remaining == []
    return {'reason': reason, 'ok': ok, 'containers_before': before, 'steps': steps, 'child_returncode': child.poll(),
            'group_alive_after': group_after, 'containers_remaining': remaining, 'docker_errors': errors,
            'seconds': time.monotonic() - started, 'cleanup_limit_seconds': term_wait_s + kill_wait_s + docker_wait_s}

def route_plan(path, out):
    """Route the launcher and native residency identifiers without changes to frozen inputs."""
    plan = S.load(path)
    argv = plan['common']['argv']
    index = next((i for i, token in enumerate(argv) if Path(token).name == 'encode_launch.py'), None)
    S.require(index is not None, 'Plan has no encode_launch.py entry')
    argv[index] = str(Path(argv[index]).with_name('campaign_launch.py'))
    if '--stage' not in argv:
        argv[index + 1:index + 1] = ['--stage', 'encode']
    S.require(argv[argv.index('--stage') + 1] in ENTRY, 'Plan has an unsupported guard stage')
    tasks = plan['roster']['tasks']
    residencies = plan['batch_policy']['residencies']
    keys = [task['residency_key'] for task in tasks] + [row['key'] for row in residencies]
    mapping, owners = {}, {}
    for key in keys:
        routed_key = key.lower()
        S.require(routed_key not in owners or owners[routed_key] == key,
                  f'Residency key collision after lowercase mapping: {key!r} and {owners.get(routed_key)!r}')
        mapping[key] = routed_key
        owners[routed_key] = key
    for task in tasks:
        task['residency_key'] = mapping[task['residency_key']]
    for row in residencies:
        row['key'] = mapping[row['key']]
    sys.path.insert(0, '/mnt/shared/prismabuild-fleet/repo/src')
    from prismabuild.decomposition import validate_logical_request
    validate_logical_request(plan)
    S.save(out, plan)
    print(json.dumps({'plan': str(path), 'argv': argv}), flush=True)


def start_container_client(argv):
    """The one place that starts the container client, in its own process group (tests replace only this)."""
    return subprocess.Popen(argv, start_new_session=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'route-plan':
        parser = argparse.ArgumentParser()
        parser.add_argument('plan', type=Path)
        parser.add_argument('--out', type=Path, required=True)
        route = parser.parse_args(sys.argv[2:])
        route_plan(route.plan, route.out)
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu','cuda'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=tuple(ENTRY), required=True)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args, rest = parser.parse_known_args()
    payload = [ENTRY[args.stage], args.stage, '--device', args.device, '--root', str(args.root), *rest]
    if args.dry_run:
        payload.append('--dry-run')
    if args.batch is not None:
        payload.extend(['--batch', str(args.batch)])
    publish = args.stage in ('select-unit', 'encode') and args.batch is not None
    if args.device == 'cpu':
        rc = subprocess.call([sys.executable, *payload])
        if rc == 0 and publish:
            _publish_batch_result(args.batch, args.root, args.stage, dry_run=args.dry_run)
        return rc
    owner = os.environ.get(OWNER_ENV, '')
    S.require(valid_owner(owner), f'{OWNER_ENV} must be the 64-hex PB owner; refusing to start a container')
    args.root.mkdir(parents=True, exist_ok=True)
    S.require(D30.available() >= 8*2**30, 'D30 available below 8GiB start floor')
    launch = S.load(D30.TEMPLATE)
    container = launch['spec']
    for mount in container['container']['mounts']:
        if mount['target'] == '/out':
            mount['source'] = str(args.root)
    container['container']['mounts'].append({'source':'/mnt/shared','target':'/mnt/shared','readonly':True})
    from g3_residency import container_contract
    mounts, env = container_contract()
    container['container']['mounts'].extend(mounts)
    container['env'].update(env)
    container['env'].update(PRISMAQUANT_DEV_MODE='1',G3_PQ_ROOT='/pq',TESSERA_SRC='/tessera/src')
    container['env']['G3_HOST_MOUNTS'] = json.dumps({m['target']:m['source'] for m in container['container']['mounts']})
    container['container']['mounts'].append({'source':str(args.root),'target':str(args.root),'readonly':False})
    command = ['python3','/workspace/'+payload[0],*payload[1:]]
    token = hashlib.sha256(json.dumps(command).encode()).hexdigest()[:20]
    S.save(args.root/('encode-launch-'+token+'.json'),{'spec':container,'command':command,'resident_tier':'PB stage+auto RAM','guard':'G3 D30 8GiB start / 2GiB abort; repaired group+container escalation'})
    test_abort = os.environ.get('D44_GUARD_TEST_ABORT_AFTER_S')
    guard = {'min_available_bytes': D30.available(), 'abort_below_bytes': 2*2**30, 'owner': owner,
             'test_abort_after_s': float(test_abort) if test_abort else None}
    lock, stop = threading.Lock(), threading.Event()
    child = thread = None

    def halt(reason):
        with lock:
            if 'halt' not in guard:
                try:
                    guard['halt'] = halt_child(child, owner, reason)
                except Exception as exc:
                    guard['halt'] = {'reason': reason, 'ok': False, 'error': repr(exc)}

    def monitor():
        t0 = time.monotonic()
        try:
            while not stop.wait(1):
                available = D30.available()
                guard['min_available_bytes'] = min(guard['min_available_bytes'], available)
                if available < 2*2**30:
                    guard['abort'] = 'D30 MemAvailable below 2GiB'
                elif guard['test_abort_after_s'] is not None and time.monotonic() - t0 >= guard['test_abort_after_s']:
                    guard['abort'] = 'TEST abort (D44_GUARD_TEST_ABORT_AFTER_S)'
                if 'abort' in guard:
                    halt(guard['abort'])
                    return
        except Exception as exc:
            guard['abort'] = 'D30 monitor failed: ' + repr(exc)
            halt(guard['abort'])

    def on_signal(signum, _frame):
        # The handler records cancellation. It never interrupts a cleanup sweep.
        guard.setdefault('cancel_signal', signal.Signals(signum).name)
        guard.setdefault('abort', 'signal: ' + guard['cancel_signal'])

    # Review fix 3: from here on, every exit path (exception, signal, timeout, normal exit) ends in one checked sweep.
    previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    rc, published, caught = None, False, None
    try:
        child = start_container_client([sys.executable,'v2_launch.py','--container',json.dumps(container),json.dumps(command)])
        guard['child_pid'] = child.pid
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        rc = child.wait(timeout=1500)  # The sweep has 300 seconds before the PB deadline.
        stop.set(); thread.join()
        halt(guard.get('abort', 'normal exit sweep'))
        S.require(guard['halt']['ok'], f"D30 sweep not verified: {guard['halt']}")
        decision = decide_commit(guard, rc)
        if decision['abort_before_decision'] is not None:
            if 'cancel_signal' in guard:
                raise LauncherSignal(guard['cancel_signal'])
            S.require(False, decision['abort_before_decision'])
        if decision['allowed'] and publish:
            published = _publish_batch_result(args.batch, args.root, args.stage, dry_run=args.dry_run)
        return rc
    except BaseException as exc:
        caught = exc
        guard['exception'] = repr(exc)[:300]
        raise
    finally:
        try:
            stop.set()
            if thread is not None and thread.ident is not None:
                thread.join()
            if child is not None and 'halt' not in guard:
                halt(guard.get('abort') or 'exception or signal: ' + guard.get('exception', 'unknown'))
            guard.update(returncode=rc, published=published)
            S.save(args.root/('encode-guard-'+token+'.json'), guard)
            committed = guard.get('commit_decision', {}).get('allowed', False)
            if 'cancel_signal' in guard and not isinstance(caught, LauncherSignal) and not committed:
                raise LauncherSignal(guard['cancel_signal'])
            if child is not None and not guard['halt']['ok'] and 'exception' not in guard:
                raise RuntimeError(f"D30 termination not verified: {guard['halt']}")
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
