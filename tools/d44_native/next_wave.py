#!/usr/bin/env python3
"""D44 controller for durable waves with at most four live batches.

The state and wait keys are saved before index reads and before each submission.
PB queue records and CAS results recover admission when the client response is lost.
The request command binds each recovered key to its batch and grid output.
A partial wave retains its members and fills only its remaining slots.
A new wave opens only after all recorded live keys become terminal.
Native claim tombstones retain live custody through their addressed immutable requests.
An unresolved admission remains pending and blocks new submission.
Only a proven client-start failure permits retry without admission reconciliation.
The controller saves state and wait keys before the journal.
Unknown keys count as live. The live-key limit is four.
The controller restores the journal from state before new admission.
Missing wave state is rebuilt from the journal.
Exit codes: 0 success, 1 submission failure, 3 live work or unresolved admission.
Test paths: D44_WAVE_EXEC_DIR, D44_WAVE_CAMPAIGN_DIR, D44_WAVE_SUBMIT,
D44_WAVE_QUEUE_DIR and D44_WAVE_CAS_DIR.
Routed usage: --routed-plan PLAN [--state-dir DIR] [--wait-file FILE].
Use --l40 --retained-units FILE --catalog-only for the released native catalogue.
The retained unit manifest names each qname and its encode and HELD document.
Each document reference has a path and its own SHA-256.
The catalogue keeps completed pairs as references and excludes them from native work.
Native group closure covers the remaining tasks, not the adopted science.
The catalogue-only entry validates stored native objects without child admission.
The routed mode resumes native PB children from their stored publication index.
It uses separate state, and leaves the grid mode unchanged.
Each portable key with no claim host counts against every possible Spark.
Each Spark has one slot; each wave has four members.
The complete ship queue must be empty before each native admission.
The routed state and wait keys precede the journal after every admission.
The caller retains scientific canary and dry-run authority. The controller owns lifetime cost.
Use --autonomous to resume on native completion, --ledger-seed for the closed balance,
and --disk-need-gb for each child's declared output and scratch demand.
Fresh D1 evidence and the 110 GPU-hour reservation precede every child_record call.
Failed or unknown costs retain custody. The controller never retries failed science.
"""
import json
import os
import subprocess
from pathlib import Path

O = Path(os.environ.get('D44_WAVE_EXEC_DIR', '/home/rob/fleet/ceo/exec/eng-d44-campaign-opus'))
C = Path(os.environ.get('D44_WAVE_CAMPAIGN_DIR',
                        '/mnt/shared/tessera-measurements/eng-ldlq-indomain-20261006/frozen-method-01/campaign-01'))
SUBMIT = os.environ.get('D44_WAVE_SUBMIT', str(O / 'submit_gpu_batch.sh'))
QUEUE = Path(os.environ.get('D44_WAVE_QUEUE_DIR', '/mnt/shared/prismabuild-fleet/pb-queue'))
CAS = Path(os.environ.get('D44_WAVE_CAS_DIR', str(QUEUE.parent / 'cas')))
WAVE = 4
LAYERS = (40, 3, 41, 42, 43, 44)
TERMINAL = ('done', 'failed', 'withdrawn')
RESUME = ('Subsample grid wave {wave} terminal ({n} batches{partial}). Check each status; for a failed batch rebuild only '
          'its missing units (campaign_make_batches.py --only-missing --sample, new tag) and resubmit. Then run '
          'python3 /home/rob/fleet/ceo/exec/eng-d44-campaign-opus/next_wave.py again. {left} batches are not yet submitted. '
          'When none remain and all 576 sampled grids exist, run the subsample reduce on x86 and report to the CEO. '
          'CEO approved the whole campaign up to 45 GPU-hours: grids, reduce, final encode (D38 dry run first), HELD '
          'scoring, G3. Report measured cost after each phase. The execution state is in record eng-d44-execution.')


def durable_append(path, line):
    with path.open('a') as f:
        f.write(line + '\n')
        f.flush()
        os.fsync(f.fileno())
    sync_directory(path.parent)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def durable_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)

def claim_transition_keys():
    return {path.name.split('.', 1)[0] for path in (QUEUE / 'claimed').iterdir()
            if path.name.endswith(('.tombstone', '.late-finish'))}


def key_state(key):
    """'live' unless the queue holds the key in a terminal state. Unknown counts as live."""
    if len(key) != 64:
        return 'live'  # never treat an unreadable key as finished
    if key in claim_transition_keys():
        return 'live'
    if any((QUEUE / state / f'{key}.json').exists() for state in ('ready', 'claimed')):
        return 'live'
    for state in TERMINAL:
        if (QUEUE / state / f'{key}.json').exists():
            return state
    if (CAS / 'actions' / 'v3' / key[:2] / f'{key}.json').exists():
        return 'done'
    return 'live'


def read_key_lines():
    path = O / 'sub-keys.txt'
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                out.append((parts[0], parts[1]))
    return out


def load_state(key_lines):
    path = O / 'wave-state.json'
    if path.exists():
        state = json.loads(path.read_text())
    else:
        state = {'waves': []}
        for i in range(0, len(key_lines), WAVE):
            chunk = key_lines[i:i + WAVE]
            state['waves'].append({'wave': len(state['waves']) + 1, 'closed': len(chunk) == WAVE, 'migrated': True,
                                   'members': [{'batch': b, 'key': k} for b, k in chunk]})
        return state, path
    recorded = {m['key'] for w in state['waves'] for m in w['members']}
    orphans = [(b, k) for b, k in key_lines if k not in recorded]
    if orphans:  # accepted and saved, but the state write did not happen
        current = state['waves'][-1] if state['waves'] and not state['waves'][-1]['closed'] else None
        if current is None:
            current = {'wave': len(state['waves']) + 1, 'closed': False, 'members': []}
            state['waves'].append(current)
        for b, k in orphans:
            current['members'].append({'batch': b, 'key': k, 'reconciled': True})
        if len(current['members']) >= WAVE:
            current['closed'] = True
    return state, path


def write_wait(state, live_other, left, failure=None):
    wave = state['waves'][-1] if state['waves'] else {'wave': 0, 'members': []}
    keys = [m['key'][:12] for m in wave['members']] + [k[:12] for k in live_other]
    partial = '' if failure is None else f'; PARTIAL: {failure}. Read the durable PB admission before any retry'
    keys += [key[:12] for key in state.get('pending_submission', {}).get('request_keys', [])]
    handed = state.get('pending_submission', {}).get('key')
    if handed:
        keys.append(handed[:12])
    keys = list(dict.fromkeys(keys))
    durable_json(O / 'wait.json', {'pb_action_keys': keys, 'max_wait_hours': 2,
                                   'resume_prompt': RESUME.format(wave=wave['wave'], n=len(wave['members']),
                                                                  partial=partial, left=left)})


def persist_custody(state, state_path, left=None, failure=None):
    durable_json(state_path, state)
    current = state['waves'][-1] if state['waves'] else {'wave': 0, 'members': []}
    members = {m['key'] for m in current['members']}
    other = [m['key'] for wave in state['waves'] for m in wave['members']
             if m['key'] not in members and key_state(m['key']) == 'live']
    write_wait({'waves': [current], 'pending_submission': state.get('pending_submission', {})}, other,
               'unknown' if left is None else left, failure=failure)


def accept_key(state, state_path, key_lines, batch, key):
    if not any(m['key'] == key for wave in state['waves'] for m in wave['members']):
        current = state['waves'][-1] if state['waves'] and not state['waves'][-1]['closed'] else None
        if current is None or len(current['members']) >= WAVE:
            current = {'wave': len(state['waves']) + 1, 'closed': False, 'members': []}
            state['waves'].append(current)
        current['members'].append({'batch': batch, 'key': key})
    persist_custody(state, state_path)
    if key not in {k for _, k in key_lines}:
        durable_append(O / 'sub-keys.txt', f'{batch} {key}')
        key_lines.append((batch, key))


def admitted_batches(known):
    """Recover keys from PB publications and CAS results, not client stdout."""
    keys = set()
    for name in ('ready', 'claimed', *TERMINAL):
        keys.update(path.stem for path in (QUEUE / name).iterdir() if path.suffix == '.json')
    results = CAS / 'actions' / 'v3'
    for prefix in results.iterdir():
        if prefix.is_dir():
            keys.update(path.stem for path in prefix.iterdir() if path.suffix == '.json')
    keys.update(claim_transition_keys())
    for key in sorted(keys - known):
        batch = request_batch(key)
        if batch is not None:
            yield batch, key


def request_batch(key):
    if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        return None
    request_path = CAS / 'requests' / key[:2] / (key + '.json')
    try:
        raw = request_path.read_text()
    except FileNotFoundError:
        return None
    if str(C / 'batches') not in raw:
        return None
    request = json.loads(raw)
    if request.get('action_key') != key:
        raise ValueError('PB request key differs from its address')
    command = request.get('params', {}).get('command', [])
    if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
        return None
    if '--stage' not in command or command[command.index('--stage') + 1] != 'select-unit':
        return None
    if '--device' not in command or command[command.index('--device') + 1] != 'cuda' or '--dry-run' in command:
        return None
    if '--root' not in command or Path(command[command.index('--root') + 1]) != C / 'select-01':
        return None
    if '--batch' not in command:
        return None
    path = Path(command[command.index('--batch') + 1])
    if path.parent != C / 'batches' or not path.name.startswith('sub-L') or not path.name.endswith('.batch.json'):
        return None
    return path.name[:-len('.batch.json')]


def reconcile_admissions(state, state_path, key_lines):
    known = {m['key'] for wave in state['waves'] for m in wave['members']}
    for batch, key in admitted_batches(known):
        accept_key(state, state_path, key_lines, batch, key)
        known.add(key)


def reconcile_pending(state, state_path, key_lines):
    pending = state.get('pending_submission')
    if pending is None:
        return True
    batch = pending['batch']
    handed_key = pending.get('key')
    if handed_key and request_batch(handed_key) == batch:
        accept_key(state, state_path, key_lines, batch, handed_key)
    if any(b == batch and request_batch(k) == batch for b, k in key_lines):
        del state['pending_submission']
        persist_custody(state, state_path)
        return True
    request_keys = []
    for prefix in (CAS / 'requests').iterdir():
        if prefix.is_dir():
            for path in prefix.iterdir():
                if path.suffix == '.json' and request_batch(path.stem) == batch:
                    request_keys.append(path.stem)
    pending['request_keys'] = sorted(set(request_keys))
    persist_custody(state, state_path, failure=batch)
    return False



def main():
    key_lines = read_key_lines()
    state, state_path = load_state(key_lines)
    # Recovery owns the keys before any external read can fail.
    persist_custody(state, state_path)
    for wave in state['waves']:
        for member in wave['members']:
            if member['key'] not in {k for _, k in key_lines}:
                durable_append(O / 'sub-keys.txt', f"{member['batch']} {member['key']}")
                key_lines.append((member['batch'], member['key']))
    reconcile_admissions(state, state_path, key_lines)
    if not reconcile_pending(state, state_path, key_lines):
        print('REFUSED: unresolved PB admission for', state['pending_submission']['batch'])
        return 3
    submitted = {b for b, _ in key_lines}
    order = []
    for layer in LAYERS:
        with (C / 'batches' / f'sub-L{layer:03d}-index.json').open() as stream:
            order += [b['batch'] for b in json.load(stream)['batches']]
    pending = [b for b in order if b not in submitted]
    live = [m['key'] for wave in state['waves'] for m in wave['members'] if key_state(m['key']) == 'live']
    current = state['waves'][-1] if state['waves'] and not state['waves'][-1]['closed'] else None
    if current is None:
        if live:
            persist_custody(state, state_path, len(pending))
            print('REFUSED new wave:', len(live), 'recorded keys are still live')
            return 3
        if not pending:
            print('nothing to submit; all batches have keys')
            return 0
        current = {'wave': len(state['waves']) + 1, 'closed': False, 'members': []}
        state['waves'].append(current)
    capacity = min(WAVE - len(current['members']), WAVE - len(live))
    persist_custody(state, state_path, len(pending))
    if len(live) > WAVE:
        print('REFUSED:', len(live), 'keys remain live; no batch was submitted')
        return 3
    for batch in pending[:max(capacity, 0)]:
        reconcile_admissions(state, state_path, key_lines)
        if batch in {b for b, _ in key_lines}:
            continue
        live_now = sum(key_state(m['key']) == 'live' for wave in state['waves'] for m in wave['members'])
        if len(current['members']) >= WAVE or live_now >= WAVE:
            break
        state['pending_submission'] = {'batch': batch}
        persist_custody(state, state_path)
        try:
            result = subprocess.run([SUBMIT, batch, '0'], capture_output=True, text=True)
        except OSError as exc:
            reconcile_admissions(state, state_path, key_lines)
            if isinstance(exc, (FileNotFoundError, PermissionError)):
                state.pop('pending_submission', None)
            else:
                reconcile_pending(state, state_path, key_lines)
            persist_custody(state, state_path, failure=batch)
            print('FAIL', batch, repr(exc))
            return 1
        lines = result.stdout.strip().splitlines()
        parts = lines[-1].split() if lines else []
        has_key = len(parts) == 2 and parts[0] == batch and len(parts[1]) == 64 and all(c in '0123456789abcdef' for c in parts[1])
        valid = result.returncode == 0 and has_key
        if has_key:
            state['pending_submission']['key'] = parts[1]
            persist_custody(state, state_path)
        if valid:
            accept_key(state, state_path, key_lines, batch, parts[1])
            state.pop('pending_submission', None)
        reconcile_admissions(state, state_path, key_lines)
        if not valid:
            reconcile_pending(state, state_path, key_lines)
        accepted = any(b == batch for b, _ in key_lines)
        left = len([b for b in pending if b not in {name for name, _ in key_lines}])
        persist_custody(state, state_path, left, failure=None if valid else batch)
        if not valid:
            print('FAIL', batch, 'accepted', accepted, (result.stderr or result.stdout)[-400:])
            return 1
        print(f'{batch} {parts[1]}', flush=True)
    left = len([b for b in pending if b not in {name for name, _ in key_lines}])
    if len(current['members']) >= WAVE or left == 0:
        current['closed'] = True
    persist_custody(state, state_path, left)
    print('wave', current['wave'], 'members', len(current['members']), 'closed', current['closed'], 'not submitted', left)
    return 0


def routed_status(key):
    """Treat transitions and unknown ownership as live, without a host guess."""
    claimed = QUEUE / 'claimed' / (key + '.json')
    transitions = list((QUEUE / 'claimed').glob(key + '.*.tombstone'))
    transitions += list((QUEUE / 'claimed').glob(key + '.*.late-finish'))
    if transitions or (QUEUE / 'ready' / (key + '.json')).exists():
        return 'live', None
    if claimed.exists():
        try:
            host = json.loads(claimed.read_text()).get('claimed_host')
        except (OSError, ValueError, AttributeError):
            host = None
        return 'live', host if isinstance(host, str) and host else None
    for ending in TERMINAL:
        if (QUEUE / ending / (key + '.json')).exists():
            return ending, None
    cas = routed_api().pb.PrismaBuildCAS(CAS)
    child = cas.read_action_request(key)
    if child is not None and cas.lookup(child) is not None:
        return 'done', None
    return 'unknown', None


def routed_ship_queue(own_keys):
    """Read every live queue row before each native admission.

    Non-D44 GPU work at priority zero or above takes precedence. This includes
    PACT at priority ten and ship checks at priority zero. No truncated view
    or unreadable record can establish an empty ship queue.
    """
    blockers, errors = set(), []
    for state in ('ready', 'claimed'):
        try:
            paths = list((QUEUE / state).iterdir())
        except OSError as exc:
            errors.append(f'{state}: {exc}')
            continue
        for path in paths:
            if not path.name.endswith(('.json', '.tombstone', '.late-finish')):
                continue
            key = path.name.split('.', 1)[0]
            if key in own_keys:
                continue
            try:
                row = json.loads(path.read_text())
                resources = row['resources']
                gpu = resources.get('gpu', 0)
                priority = row['priority']
                if (row['action_key'] != key or type(gpu) is not int or gpu < 0
                        or type(priority) is not int):
                    raise ValueError('Invalid queue resource or priority fields.')
                if gpu > 0 and priority >= 0:
                    blockers.add(key)
            except FileNotFoundError:
                continue  # The queue retired this row during the census.
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                errors.append(f'{path}: {exc}')
    return {'clear': not blockers and not errors, 'blockers': sorted(blockers), 'errors': errors}

def routed_api():
    import sys
    sys.path.insert(0, '/mnt/shared/prismabuild-fleet/repo/tools')
    import pbcampaign
    return pbcampaign


RETAINED_SCOPE_ROLES = ('fit', 'heldout')


def retained_task_scope(task):
    """Return the authoritative frozen scope for one roster task."""
    payload = task.get('payload', {})
    scope = payload.get('retained_scope')
    if not isinstance(scope, dict):
        return None
    fit = scope.get('fit')
    heldout = scope.get('heldout')
    selection = scope.get('selection_sha256')
    if not isinstance(fit, dict) or not isinstance(heldout, dict):
        return None
    if not isinstance(selection, str) or len(selection) != 64:
        return None
    return {'fit': dict(fit), 'heldout': dict(heldout), 'selection_sha256': selection}


def check_retained_scope(name, encode, held, task):
    """Refuse retained rows or a selection that differ from the frozen task."""
    expected = retained_task_scope(task)
    if expected is None:
        return
    for role in RETAINED_SCOPE_ROLES:
        if encode[role] != expected[role]:
            raise ValueError(f'Retained {role} rows differ from the frozen task: {name}')
        if held[role] != expected[role]:
            raise ValueError(f'Retained {role} rows differ from the frozen task: {name}')
    actual = (encode.get('conditioning') or {}).get('selection_sha256')
    if actual is None:
        actual = (held.get('conditioning') or {}).get('selection_sha256')
    if actual is None:
        return
    if actual != expected['selection_sha256']:
        raise ValueError(f'Retained selection differs from the frozen task: {name}')


def verified_retained_document(reference):
    import hashlib
    path = Path(reference['path'])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError(f'Retained document digest differs: {path}')
    return json.loads(raw)


def adopt_retained_units(request, manifest):
    """Exclude verified completed pairs from native work without changing the input roster.

    The catalogue retains the original output references. It does not copy science
    outputs, create native receipts for old work, or claim a new measurement.
    """
    import hashlib
    if manifest.get('schema') != 'd44.retained_units.v1':
        raise ValueError('Unsupported retained unit manifest.')
    tasks = request['roster']['tasks']
    by_name = {}
    for task in tasks:
        name = task['payload']['qname']
        if name in by_name:
            raise ValueError(f'The roster has a duplicate qname: {name}')
        by_name[name] = task
    adopted, seen = [], set()
    for unit in manifest['units']:
        name = unit['qname']
        if name in seen:
            raise ValueError(f'Retained units have a duplicate qname: {name}')
        if name not in by_name:
            raise ValueError(f'Retained unit is outside the roster: {name}')
        encode = verified_retained_document(unit['encode'])
        held = verified_retained_document(unit['held'])
        if (encode['qname'] != name or held['qname'] != name
                or encode['dry_run'] is not False or held['dry_run'] is not False):
            raise ValueError(f'Retained unit has no actual paired result: {name}')
        blob = Path(encode['blob_path'])
        digest = hashlib.sha256()
        with blob.open('rb') as stream:
            while chunk := stream.read(8 << 20):
                digest.update(chunk)
        if digest.hexdigest() != encode['blob_sha256'] or blob.stat().st_size != encode['blob_bytes']:
            raise ValueError(f'Retained blob bytes or digest differ: {name}')
        replacement = held['replacement']
        expected = {'blob': str(blob), 'blob_sha256': encode['blob_sha256'],
                    'bytes': encode['blob_bytes'], 'receipt': unit['encode']['path'],
                    'receipt_sha256': unit['encode']['sha256']}
        if any(replacement.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Retained HELD replacement differs: {name}')
        if encode['rendered_shape'] != held['actual_source_shape']:
            raise ValueError(f'Retained paired shapes differ: {name}')
        for role in ('fit', 'heldout'):
            if encode[role] != held[role]:
                raise ValueError(f'Retained paired {role} rows differ: {name}')
        task = by_name[name]
        check_retained_scope(name, encode, held, task)
        adopted.append({**unit, 'task_id': task['id'], 'output_id': task['output_id']})
        seen.add(name)
    pending = [task for task in tasks if task['payload']['qname'] not in seen]
    return {**request, 'roster': {**request['roster'], 'tasks': pending}}, adopted


def retained_population_key(units):
    """Return the adopted population identity for one retained unit list."""
    keyed = []
    for unit in units:
        keyed.append({'qname': unit['qname'], 'encode_sha256': unit['encode']['sha256'],
                      'held_sha256': unit['held']['sha256']})
    return sorted(keyed, key=lambda row: row['qname'])


def check_retained_resume(catalog, retained_path):
    """Refuse a supplied manifest that differs from the stored population."""
    import json
    stored = retained_population_key(catalog.get('retained_units', []))
    if retained_path is None:
        return
    supplied = json.loads(retained_path.read_text())
    if supplied.get('schema') != 'd44.retained_units.v1':
        raise ValueError('Unsupported retained unit manifest.')
    offered = retained_population_key(supplied.get('units', []))
    if offered != stored:
        raise ValueError('The supplied retained manifest differs from the stored catalogue population.')


def prepare_routed_catalog(pc, plan_path, catalog_path, *, l40=False, retained_path=None):
    """Use native decomposition primitives, but do not admit the children."""
    request = pc.load_manifest(plan_path, transport='pool')
    if l40:
        tasks = [task for task in request['roster']['tasks']
                 if '.layers.40.' in task['payload']['qname']]
        request = {**request, 'roster': {**request['roster'], 'tasks': tasks}}
    adopted, retained_tasks = [], []
    if retained_path is not None:
        original_tasks = request['roster']['tasks']
        request, adopted = adopt_retained_units(request, json.loads(retained_path.read_text()))
        retained_ids = {row['task_id'] for row in adopted}
        retained_tasks = [task for task in original_tasks if task['id'] in retained_ids]
    request = pc.dc.validate_logical_request(request)
    request['common'] = {**request['common'], 'exclusive': True}
    args = pc.pbrun.parse_args(['--detach', '--transport', 'pool', '--priority', '0',
                               *pc.pbrun_argv(request['common'])])
    prepared = pc.pbrun.prepare_submission(args)
    template = prepared['template']
    cas = template['cas']
    if 'task_data_manifest' in request:
        template = {**template, 'params': {**template['params'],
                    pc.dc.TASK_DATA_POLICY_PARAM: request['task_data_manifest']}}
    frozen = pc.dc.freeze_common(request['common'],
                                action_common=pc.pbrun.template_action_common(template))
    plan = pc.frozen_plan(request, frozen, cas=cas)
    roster_input, _ = cas.ingest_bytes(pc.dc.document_bytes(request['roster']),
                                     input_id=pc.dc.TASK_ROSTER_INPUT_ID)
    batches = pc.dc.PreparedBatches(request, plan)
    children, digests = [], []
    for ordinal in range(len(plan['partitions'])):
        envelope = batches.envelope(ordinal)
        batch_input, _ = cas.ingest_bytes(pc.dc.document_bytes(envelope),
                                        input_id=pc.dc.TASK_BATCH_INPUT_ID)
        children.append(pc.pbrun.seal_decomposed_child(
            template, request=request, plan=plan, child_ordinal=ordinal,
            roster_input=roster_input, batch_input=batch_input, cas=cas,
            prepared_batches=batches,
            data_manifest=pc.dc.task_data_manifest(request['task_data_manifest'], envelope)
            if 'task_data_manifest' in request else None))
        digests.append(batch_input['sha256'])
    queue = pc.pool.PoolQueue(QUEUE)
    if 'task_data_manifest' in request:
        pc.frozen_child_data_plans(request, plan, children, template=template,
                                   args=args, queue=queue, cas=cas)
    pc.pbrun.announce_placement(prepared['offer_queue'](), children[0], args=args,
                               cwd=prepared['cwd'],
                               portable_checkout=prepared['portable_checkout'])
    pc.publish_index(pc.dc.publication_index(
        plan, batch_input_digests=digests,
        child_action_keys=[child['action_key'] for child in children]),
        cas=cas, parent_key=plan['parent_key'])
    for child in children:
        cas.publish_action_request(child)
    # The native index and requests precede this pointer and every admission.
    catalog = {'parent_key': plan['parent_key'], 'roster_input': roster_input,
               'request': {name: value for name, value in request.items() if name != 'roster'},
               'retained_units': adopted, 'retained_tasks': retained_tasks}
    durable_json(catalog_path, catalog)
    return catalog


def restore_routed_custody(state_dir, wait_path, resume_command):
    """Restore known keys before a native metadata read can fail."""
    state_path, journal_path = state_dir / 'wave-state.json', state_dir / 'sub-keys.txt'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'waves': []}
    journal = ([tuple(line.split()) for line in journal_path.read_text().splitlines() if line.strip()]
               if journal_path.exists() else [])
    known = {member['key'] for wave in state['waves'] for member in wave['members']}
    for batch, key in journal:
        if key in known:
            continue
        current = state['waves'][-1] if state['waves'] else None
        if current is None or current['closed'] or len(current['members']) == WAVE:
            current = {'wave': len(state['waves']) + 1, 'members': [], 'closed': False}
            state['waves'].append(current)
        current['members'].append({'batch': batch, 'key': key})
        current['closed'] = len(current['members']) == WAVE
        known.add(key)
    pending = state.get('pending_submission')
    keys = list(known) + ([pending['key']] if pending else [])
    durable_json(state_path, state)
    if keys:
        durable_json(wait_path, {'pb_action_keys': list(dict.fromkeys(key[:12] for key in keys)),
                                'max_wait_hours': 2,
                                'resume_prompt': 'Routed custody is saved. Check each status. '
                                'Recover the native metadata. Resume with: ' + resume_command})
    return state, journal


def finite_nonnegative(value, name):
    import math
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative number.')
    return value


class RoutedLedger:
    """Keep the closed balance, immutable attempt charges and live reservations.

    One controller lock protects this file. Each atomic checkpoint contains both
    charges and reservation release, so restart cannot charge an attempt twice.
    The seed is a closed balance, not a list of components to add again.
    """

    def __init__(self, path, *, seed_path=None, cpu=False):
        self.path = path
        if path.exists():
            self.value = json.loads(path.read_text())
            if seed_path is not None:
                supplied = json.loads(seed_path.read_text())
                from prismaquant.dev_mode import seal_check
                seal_check('d44 ledger seed', self.value['seed'], supplied,
                           where=str(path), refusal=ValueError)
        else:
            if seed_path is None and not cpu:
                raise ValueError('The lifetime cost ledger requires the closed balance seed.')
            seed = (json.loads(seed_path.read_text()) if seed_path is not None else {
                'schema': 'fleet.d44.native_phase_ledger_seed.v1',
                'authoritative_seed_GPU_hours': 0.0,
                'global_ceiling_GPU_hours': 110, 'GPU_hour_multiplier': 1})
            self.value = {'schema': 'd44.native_lifetime_cost.v1', 'seed': seed,
                          'cpu_only': cpu, 'charges': {}, 'reservations': {}}
        if self.value.get('schema') != 'd44.native_lifetime_cost.v1':
            raise ValueError('Unsupported lifetime cost ledger.')
        if self.value.get('cpu_only') is True and not cpu:
            raise ValueError('A CPU-only ledger cannot supply the GPU lifetime balance.')
        seed = self.value['seed']
        if (seed.get('schema') != 'fleet.d44.native_phase_ledger_seed.v1'
                or seed.get('global_ceiling_GPU_hours') != 110
                or seed.get('GPU_hour_multiplier') != 1):
            raise ValueError('The lifetime cost contract requires 110 GPU-hours and multiplier one.')
        finite_nonnegative(seed['authoritative_seed_GPU_hours'], 'Closed balance')
        self.seeded = set()
        for name in ('already_charged_native_action_keys', 'already_charged_preflight_action_keys',
                     'already_charged_known_historical_action_keys'):
            self.seeded.update(seed.get(name, []))
        for amount in self.value['reservations'].values():
            finite_nonnegative(amount, 'Reservation')
        for row in self.value['charges'].values():
            finite_nonnegative(row['GPU_hours'], 'Attempt cost')
        self.save()

    def save(self):
        durable_json(self.path, self.value)

    def charged(self, key):
        return sum(row['GPU_hours'] for row in self.value['charges'].values()
                   if row['action_key'] == key)

    def total(self):
        return (self.value['seed']['authoritative_seed_GPU_hours']
                + sum(row['GPU_hours'] for row in self.value['charges'].values())
                + sum(max(0, amount - self.charged(key))
                      for key, amount in self.value['reservations'].items()))

    def reserve(self, key, seconds, gpu):
        amount = finite_nonnegative(seconds, 'Maximum seconds') * gpu / 3600
        existing = self.value['reservations'].get(key)
        if existing is not None:
            if existing != amount:
                raise ValueError('The live lifetime cost reservation differs from the child maximum.')
            if self.total() > 110:
                raise ValueError('The lifetime cost plus reservation exceeds 110 GPU-hours.')
            return
        if self.total() + max(0, amount - self.charged(key)) > 110:
            raise ValueError('The lifetime cost plus reservation exceeds 110 GPU-hours.')
        self.value['reservations'][key] = amount
        self.save()

    def checkpoint(self, key, attempts, *, terminal, gpu):
        if key in self.seeded:
            return  # The closed balance already owns these historical attempts.
        if terminal and gpu and not attempts:
            raise ValueError('Unknown GPU attempt cost; retain the lifetime reservation.')
        for attempt in attempts:
            if attempt.get('action_key') != key:
                raise ValueError('The GPU attempt cost belongs to another child.')
            number = attempt['attempt']
            generation = finite_nonnegative(attempt['published_unix'], 'Attempt generation')
            if type(number) is not int or number < 1:
                raise ValueError('The GPU attempt cost has no valid attempt number.')
            seconds = finite_nonnegative(attempt.get('detail', {}).get('elapsed_s'), 'GPU attempt cost')
            identity = f'{key}:{generation}:{number}'
            charge = {'action_key': key, 'published_unix': generation, 'attempt': number,
                      'status': attempt['status'], 'elapsed_s': seconds, 'GPU_hours': seconds * gpu / 3600}
            previous = self.value['charges'].get(identity)
            if previous is not None and previous != charge:
                raise ValueError('Conflicting GPU attempt cost evidence; retain the reservation.')
            self.value['charges'][identity] = charge
            self.save()  # Retain a valid partial prefix even if a later cost is missing.
        if terminal:
            self.value['reservations'].pop(key, None)
            self.save()


def routed_diskcheck(child, *, need_gb, command=None):
    """Run a fresh D1 check on all possible hosts and on shared output storage."""
    import datetime
    import shlex
    import time
    finite_nonnegative(need_gb, 'Declared disk output')
    tags = child.get('params', {}).get('placement', {}).get('required_tags', [])
    hosts = [host for host in ('sparky', 'sparklina', 'dl380g10') if host in tags]
    if not hosts:
        hosts = ['dl380g10'] if 'x86' in tags else ['sparky', 'sparklina']
    paths = ['/tmp', '/mnt/shared'] if hosts == ['dl380g10'] else ['/', '/mnt/shared']
    argv = shlex.split(command or os.environ.get('D44_DISKCHECK', '/home/rob/fleet/ceo/bin/fleet-diskcheck'))
    started = time.time()
    result = subprocess.run([*argv, '--need-gb', str(need_gb), '--hosts', ','.join(hosts),
                             '--paths', ','.join(paths)], capture_output=True, text=True, timeout=120)
    body = json.loads(result.stdout)
    checked = datetime.datetime.fromisoformat(body['at']).timestamp()
    if (body.get('tool') != 'fleet-diskcheck' or body.get('need_gb') != need_gb
            or checked < started - 1 or checked > time.time() + 1
            or set(body.get('hosts', {})) != set(hosts)):
        raise ValueError('Incomplete or stale D1 evidence.')
    for host in hosts:
        row = body['hosts'][host]
        if set(row.get('filesystems', {})) != set(paths):
            raise ValueError('Incomplete D1 filesystem evidence.')
        for path in paths:
            fs = row['filesystems'][path]
            if not all(name in fs for name in ('avail_gib', 'free_pct', 'required_gib', 'inodes_free_pct', 'pass')):
                raise ValueError('Incomplete D1 space evidence.')
        if row.get('pass') is not True:
            body['pass'] = False
    body['pass'] = (result.returncode == 0 and body.get('pass') is True
                    and all(fs.get('pass') is True for host in hosts
                            for fs in body['hosts'][host]['filesystems'].values()))
    return body


class RoutedWave:
    """Keep native child custody separate from the existing grid state."""

    def __init__(self, pc, catalog, state_dir, wait_path, resume_command, *, custody=None,
                 seed_path=None, autonomous=False, disk_need_gb=None):
        self.pc, self.catalog = pc, catalog
        self.state_dir, self.wait_path = state_dir, wait_path
        self.resume_command = resume_command
        self.state_path = state_dir / 'wave-state.json'
        self.journal_path = state_dir / 'sub-keys.txt'
        self.state, self.journal = (custody if custody is not None
                                    else restore_routed_custody(state_dir, wait_path, resume_command))
        self.cas = pc.pb.PrismaBuildCAS(CAS)
        self.queue = pc.pool.PoolQueue(QUEUE)
        native_dir = pc.decomposition_dir(self.cas, catalog['parent_key'])
        self.plan = pc.dc.validate_plan(pc._stored_document(native_dir / 'plan.json'))
        if self.plan['parent_key'] != catalog['parent_key']:
            raise pc.ManifestError('The native plan parent differs from its catalog address.')
        self.request = pc.dc.validate_logical_request({**catalog['request'],
            'roster': json.loads(self.cas.input_path(catalog['roster_input']).read_text())})
        if catalog.get('retained_units'):
            adopt_retained_units({'roster': {'tasks': catalog['retained_tasks']}},
                {'schema': 'd44.retained_units.v1', 'units': catalog['retained_units']})
        publication = pc._stored_document(native_dir / 'publication.json')
        if publication is None:
            raise pc.ManifestError('The native publication index is missing.')
        self.keys = publication['child_action_keys']
        batches = pc.dc.PreparedBatches(self.request, self.plan)
        expected_index = pc.dc.publication_index(
            self.plan, child_action_keys=self.keys,
            batch_input_digests=[pc.dc.document_sha256(batches.envelope(ordinal))
                                for ordinal in range(len(self.plan['partitions']))])
        pc.publish_index(expected_index, cas=self.cas, parent_key=catalog['parent_key'])
        children = [self.cas.read_action_request(key) for key in self.keys]
        for ordinal, child in enumerate(children):
            if child is None or child.get('params', {}).get(pc.dc.LOGICAL_BATCH_PARAM) != batches.membership(ordinal):
                raise pc.ManifestError(f'Native child {ordinal} does not match the stored batch membership.')
        self.args = pc.pbrun.parse_args(['--detach', '--transport', 'pool', '--priority', '0',
                                        *pc.pbrun_argv(catalog['request']['common'])])
        if 'task_data_manifest' in catalog['request']:
            if pc._stored_document(native_dir / 'data-plans.json') is None:
                raise ValueError('The required native data plans are missing; no child was submitted.')
            self.data_plans = pc.frozen_child_data_plans(
                catalog['request'], self.plan, children,
                template={**children[0], 'cas': self.cas}, args=self.args,
                queue=self.queue, cas=self.cas)
        else:
            self.data_plans = [None] * len(self.keys)
        self.ordinal = {key: ordinal for ordinal, key in enumerate(self.keys)}
        self.autonomous, self.disk_need_gb = autonomous, disk_need_gb
        cpu = not any(child['params']['demand'].get('gpu', 0) for child in children)
        ledger_path = state_dir / 'lifetime-cost.json'
        self.ledger = (RoutedLedger(ledger_path, seed_path=seed_path, cpu=cpu)
                       if cpu or seed_path is not None or ledger_path.exists() else None)

    def members(self):
        return [member for wave in self.state['waves'] for member in wave['members']]

    def persist(self, failure=None):
        members = self.members()
        accepted = {member['key'] for member in members}
        current = self.state['waves'][-1] if self.state['waves'] else {'wave': 0, 'members': []}
        keys = [member['key'] for member in current['members']]
        keys += [key for key in accepted if routed_status(key)[0] in ('live', 'unknown')]
        pending = self.state.get('pending_submission')
        if pending:
            keys.append(pending['key'])
        left = len(self.keys) - len(accepted)
        durable_json(self.state_path, self.state)
        if keys:
            wait = {'pb_action_keys': list(dict.fromkeys(key[:12] for key in keys)),
                    'resume_command': self.resume_command, 'max_wait_hours': 2,
                    'native_children_left': left, 'failure': failure}
            if self.autonomous:
                wait['completion_receiver'] = 'RoutedWave.await_completion'
            else:
                wait['resume_prompt'] = 'Native custody is saved. Resume with: ' + self.resume_command
            durable_json(self.wait_path, wait)

    def accept(self, key):
        if key not in {member['key'] for member in self.members()}:
            current = self.state['waves'][-1] if self.state['waves'] else None
            if current is None or current['closed'] or len(current['members']) == WAVE:
                current = {'wave': len(self.state['waves']) + 1, 'members': [], 'closed': False}
                self.state['waves'].append(current)
            current['members'].append({'batch': f'child-{self.ordinal[key]:05d}', 'key': key})
            current['closed'] = len(current['members']) == WAVE
        self.persist()
        for member in self.members():
            line = (member['batch'], member['key'])
            if line not in self.journal:
                durable_append(self.journal_path, ' '.join(line))
                self.journal.append(line)

    def reconcile(self):
        recorded = {member['key'] for member in self.members()}
        for _, key in self.journal:
            if key not in self.ordinal:
                raise ValueError('The routed journal contains a key outside the native publication.')
            if key not in recorded:
                self.accept(key)
                recorded.add(key)
        for key in self.keys:
            if key not in recorded and routed_status(key)[0] != 'unknown':
                self.accept(key)
                recorded.add(key)
        self.persist()
        # Repair a crash after state and wait publication but before the journal.
        for member in self.members():
            if (member['batch'], member['key']) not in self.journal:
                self.accept(member['key'])
        pending = self.state.get('pending_submission')
        if pending and pending['key'] in {member['key'] for member in self.members()}:
            self.state.pop('pending_submission')
            self.persist()

    def capacity(self):
        if self.state.get('pending_submission'):
            return 0
        current = self.state['waves'][-1] if self.state['waves'] else None
        statuses = [routed_status(member['key']) for member in self.members()]
        live = [status for status in statuses if status[0] in ('live', 'unknown')]
        if current and current['closed'] and live:
            return 0
        host_counts, unassigned = {}, 0
        for _, host in live:
            if host is None:
                unassigned += 1
            else:
                host_counts[host] = host_counts.get(host, 0) + 1
        # A portable admission can land on any Spark. Do not assume balance.
        host_room = 1 - unassigned - max(host_counts.values(), default=0)
        wave_room = WAVE - len(current['members']) if current and not current['closed'] else WAVE
        return max(0, min(wave_room, WAVE - len(live), host_room))

    def checkpoint_costs(self):
        for member in self.members():
            key = member['key']
            child = self.cas.read_action_request(key)
            if child is None:
                raise ValueError('The native child request is missing during cost recovery.')
            params = child['params']
            gpu = params['demand'].get('gpu', 0)
            if key in self.ledger.seeded:
                continue
            status = routed_status(key)[0]
            terminal = status in TERMINAL
            if not gpu:
                self.ledger.checkpoint(key, [], terminal=terminal, gpu=0)
                continue
            if key not in self.ledger.value['reservations']:
                # The child already owns admission. Recover its reservation even over the limit.
                self.ledger.value['reservations'][key] = params['execution_timeout_s'] * gpu / 3600
                self.ledger.save()
            ending = self.queue.current_ending(key)
            if ending['ambiguous']:
                raise ValueError('Ambiguous native GPU cost evidence; retain custody and reservation.')
            row = ending['record']
            if row is None:
                for name in ('ready', 'claimed'):
                    path = self.queue.item_path(name, key)
                    if path.exists():
                        row = json.loads(path.read_text())
                        break
            attempts = []
            if row is not None:
                if row.get('attempt_history_missing_before', 0):
                    raise ValueError('Unknown partial GPU attempt cost; retain the reservation.')
                attempts = self.queue.attempt_outcomes(row)
                member['published_unix'] = row['published_unix']
            self.ledger.checkpoint(key, attempts, terminal=terminal, gpu=gpu)
        self.persist()

    def await_completion(self):
        import time
        live = [member for member in self.members()
                if routed_status(member['key'])[0] in ('live', 'unknown')]
        if not live:
            if self.state.get('wait_reason') == 'ship':
                time.sleep(self.pc.pbrun.POLL_S)
                return True
            return False
        for member in live:
            key = member['key']
            if routed_status(key)[0] == 'unknown' and not self.pc.pbrun.bounded_attachment(self.queue, key):
                print('REFUSED: native completion ownership is unknown; retain custody', key)
                return False
            result = self.pc.pbwait.wait_one(self.queue, key, cas=self.cas, deadline=float('inf'),
                                            generation=member.get('published_unix'), queue_root=QUEUE)
            self.state['last_completion'] = result
            self.persist()
            if result['status'] in ('record_error', 'unreadable', 'waiting', 'not_submitted'):
                return False
            # A receipt or immutable ending does not release a cleanup tombstone.
            while self.pc._pool_slot_occupied(self.queue, key):
                self.checkpoint_costs()
                time.sleep(self.pc.pbrun.POLL_S)
        return True

    def run(self):
        while True:
            result = self.run_once()
            if result != 3 or not self.autonomous or self.state.get('pending_submission'):
                return result
            if not self.await_completion():
                return result

    def run_once(self):
        self.reconcile()
        try:
            if self.ledger is None:
                raise ValueError('The lifetime cost ledger requires the closed balance seed.')
            self.checkpoint_costs()
        except (Exception, SystemExit) as exc:
            self.persist(failure=str(exc))
            print('REFUSED: lifetime cost recovery:', str(exc))
            return 1
        if self.state.get('pending_submission'):
            print('REFUSED: routed admission remains unresolved')
            return 3
        if any(routed_status(member['key'])[0] in ('failed', 'withdrawn') for member in self.members()):
            print('REFUSED: a routed child failed; retain custody and rebuild only missing units')
            return 1
        accepted = {member['key'] for member in self.members()}
        for ordinal, key in enumerate(self.keys):
            if key in accepted:
                continue
            self.reconcile()
            try:
                self.checkpoint_costs()
            except (Exception, SystemExit) as exc:
                self.persist(failure=str(exc))
                print('REFUSED: lifetime cost recovery:', str(exc))
                return 1
            accepted = {member['key'] for member in self.members()}
            if any(routed_status(member['key'])[0] in ('failed', 'withdrawn') for member in self.members()):
                print('REFUSED: a routed child failed; no further child was submitted')
                return 1
            if key in accepted:
                continue
            if self.capacity() == 0:
                self.state['wait_reason'] = 'completion'
                self.persist()
                print('WAIT: the routed wave or Spark limit has no capacity')
                return 3
            child = self.cas.read_action_request(key)
            if child is None:
                raise ValueError('The native child request is missing.')
            ship_queue = routed_ship_queue(set(self.keys))
            if not ship_queue['clear']:
                self.state['wait_reason'] = 'ship'
                self.persist()
                print('WAIT: the ship queue blocks admission', json.dumps(ship_queue, sort_keys=True))
                return 3
            had_reservation = key in self.ledger.value['reservations']
            try:
                gpu = child['params']['demand'].get('gpu', 0)
                if gpu and (gpu != 1 or child['params']['execution_timeout_s'] > 1800
                            or child['params']['retry_policy']['max_attempts'] != 1):
                    raise ValueError('The native phase requires one GPU, 1800 seconds or less and no automatic retry.')
                if gpu and self.disk_need_gb is None:
                    raise ValueError('Declare per-child output and scratch GiB for D1.')
                self.ledger.reserve(key, child['params']['execution_timeout_s'], gpu)
                disk = routed_diskcheck(child, need_gb=self.disk_need_gb if self.disk_need_gb is not None else 0.1)
                self.state['last_disk_check'] = {'action_key': key, 'evidence': disk}
                self.state.setdefault('disk_checks', []).append(self.state['last_disk_check'])
                self.persist()
                if disk.get('pass') is not True:
                    raise ValueError('D1 refuses the native child before admission.')
            except (Exception, SystemExit) as exc:
                if not had_reservation:
                    self.ledger.value['reservations'].pop(key, None)  # No child_record call happened.
                self.ledger.save()
                self.persist(failure=str(exc))
                print('REFUSED:', str(exc))
                return 1
            ship_queue = routed_ship_queue(set(self.keys))
            if not ship_queue['clear']:
                if not had_reservation:
                    self.ledger.value['reservations'].pop(key, None)
                    self.ledger.save()
                self.state['wait_reason'] = 'ship'
                self.persist()
                print('WAIT: the ship queue changed during D1', json.dumps(ship_queue, sort_keys=True))
                return 3
            self.ledger.reserve(key, child['params']['execution_timeout_s'], gpu)
            self.state.pop('wait_reason', None)
            self.state['pending_submission'] = {'batch': f'child-{ordinal:05d}', 'key': key}
            self.persist()
            try:
                result = self.pc.child_record(child, args=self.args, queue=self.queue,
                                              cas=self.cas, staged_plan=self.data_plans[ordinal])
            except (Exception, SystemExit) as exc:
                self.reconcile()
                self.persist(failure=key)
                print('FAIL', key, str(exc))
                return 1
            if result.get('action_key') != key or result.get('status') not in ('submitted', 'attached', 'cache_hit'):
                self.reconcile()
                self.persist(failure=key)
                print('FAIL: the native child response does not confirm admission', key)
                return 1
            self.accept(key)
            for member in self.members():
                if member['key'] == key and result.get('published_unix') is not None:
                    member['published_unix'] = result['published_unix']
            self.state.pop('pending_submission', None)
            self.persist()
            accepted.add(key)
            print(f'child-{ordinal:05d} {key}', flush=True)
            if self.state['waves'][-1]['closed']:
                break
        left = len(self.keys) - len({member['key'] for member in self.members()})
        if self.state['waves'] and left == 0:
            self.state['waves'][-1]['closed'] = True
        self.persist()
        if left or any(routed_status(key)[0] in ('live', 'unknown') for key in self.keys):
            self.state['wait_reason'] = 'completion'
            self.persist()
            return 3
        self.checkpoint_costs()
        children = [self.cas.read_action_request(key) for key in self.keys]
        return self.pc.close_group({'request': self.request, 'plan': self.plan, 'children': children}, cas=self.cas)


def routed_main(argv):
    import argparse
    import fcntl
    import shlex
    parser = argparse.ArgumentParser(description='Submit and resume durable native PB final-encode children.')
    parser.add_argument('--routed-plan', required=True, type=Path)
    parser.add_argument('--state-dir', type=Path, default=O / 'routed-wave-state')
    parser.add_argument('--wait-file', type=Path, default=O / 'wait.json')
    parser.add_argument('--l40', action='store_true', help='Prepare only the released layer forty roster.')
    parser.add_argument('--retained-units', type=Path, help='Verified encode and HELD pairs to adopt without replay.')
    parser.add_argument('--catalog-only', action='store_true', help='Validate native objects without child admission.')
    parser.add_argument('--autonomous', action='store_true', help='Resume after native completion without another invocation.')
    parser.add_argument('--ledger-seed', type=Path, help='The accepted closed lifetime balance, required for new GPU ledgers.')
    parser.add_argument('--disk-need-gb', type=float, help='Output and scratch GiB for each child, required for GPU admission.')
    options = parser.parse_args(argv)
    options.state_dir.mkdir(parents=True, exist_ok=True)
    with (options.state_dir / 'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('WAIT: another routed controller owns the state')
            return 3
        command = shlex.join(['python3', str(Path(__file__).resolve()), '--routed-plan',
                              str(options.routed_plan), '--state-dir', str(options.state_dir),
                              '--wait-file', str(options.wait_file)])
        if options.l40:
            command += ' --l40'
        if options.retained_units:
            command += ' --retained-units ' + shlex.quote(str(options.retained_units))
        if options.autonomous:
            command += ' --autonomous'
        if options.ledger_seed:
            command += ' --ledger-seed ' + shlex.quote(str(options.ledger_seed))
        if options.disk_need_gb is not None:
            command += ' --disk-need-gb ' + str(options.disk_need_gb)
        custody = restore_routed_custody(options.state_dir, options.wait_file, command)
        pc = routed_api()
        catalog_path = options.state_dir / 'catalog.json'
        if catalog_path.exists():
            catalog = json.loads(catalog_path.read_text())
            check_retained_resume(catalog, options.retained_units)
        else:
            catalog = prepare_routed_catalog(pc, options.routed_plan, catalog_path,
                        l40=options.l40, retained_path=options.retained_units)
        controller = RoutedWave(pc, catalog, options.state_dir, options.wait_file, command, custody=custody,
                                seed_path=options.ledger_seed, autonomous=options.autonomous,
                                disk_need_gb=options.disk_need_gb)
        if options.l40 and any('.layers.40.' not in task['payload']['qname']
                              for task in controller.request['roster']['tasks']):
            raise ValueError('The stored catalogue includes a layer that is not released.')
        if options.catalog_only:
            print(json.dumps({'catalog': str(catalog_path), 'parent_key': catalog['parent_key'],
                'native_children': len(controller.keys), 'pending_tasks': len(controller.request['roster']['tasks']),
                'retained_units': catalog.get('retained_units', []), 'admitted': False}, sort_keys=True))
            return 0
        return controller.run()


if __name__ == '__main__':
    import sys
    routed = any(arg == '--routed-plan' or arg.startswith('--routed-plan=') for arg in sys.argv[1:])
    raise SystemExit(routed_main(sys.argv[1:]) if routed else main())

