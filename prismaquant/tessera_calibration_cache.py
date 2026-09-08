"""Campaign capture receipts over the existing per-unit activation cache storage.

Scoring inputs retain the campaign's first rows in float32, while Hessians,
counts and maxima cover the entire draw. No row sampling or runtime scheduling
lives here. Readers verify inputs and prefetch their selected scope before use.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .cost_stage_checkpoint import atomic_write_bytes, prepare_journal, write_unit

SCHEMA = 'prismaquant.tessera_calibration_cache.v2'
STAGE = 'tessera_calibration_capture'
SOURCE = 'tessera_campaign_prefix_f32_v1'


def sha256(path, *, resource_check=None, release_read_pages=False):
    with Path(path).open('rb') as handle:
        if resource_check is not None or release_read_pages:
            import os
            import stat
            original = os.fstat(handle.fileno())
            if release_read_pages and not stat.S_ISREG(original.st_mode):
                raise RuntimeError('source hash page release requires a regular file')
            digest = hashlib.sha256()
            consumed = advised = 0
            while True:
                if resource_check is not None:
                    resource_check(f'before_capture_hash:{Path(path).name}')
                block = handle.read(16*1024**2)
                if not block:
                    after = os.fstat(handle.fileno())
                    named = Path(path).stat()
                    if any(getattr(original, key) != getattr(actual, key)
                           for actual in (after, named) for key in
                           ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
                        raise RuntimeError('source changed during guarded capture hashing')
                    return digest.hexdigest()
                digest.update(block)
                consumed += len(block)
                del block
                if release_read_pages:
                    page = os.sysconf('SC_PAGE_SIZE')
                    end = consumed//page*page
                    if end > advised:
                        os.posix_fadvise(handle.fileno(), advised, end-advised, os.POSIX_FADV_DONTNEED)
                        advised = end
                if resource_check is not None:
                    resource_check(f'after_capture_hash:{Path(path).name}')
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _json(path, value):
    atomic_write_bytes(Path(path), (json.dumps(value, sort_keys=True, indent=2,
                                              allow_nan=False) + '\n').encode())


def capture_identity(census_path, *, calibration, max_act_rows,
                     model_load_contract, attention_implementation,
                     resource_check=None, release_read_pages=False):
    """Hash source bytes on every invocation; mtimes never authorize reuse."""
    import importlib.metadata
    import torch
    census_path = Path(census_path)
    census = json.loads(census_path.read_text())
    if type(max_act_rows) is not int or max_act_rows < 1:
        raise ValueError('capture scoring prefix must have positive max_act_rows')
    from prismaquant import validate_source_initialization_contract
    contract = validate_source_initialization_contract(model_load_contract)
    recorded = validate_source_initialization_contract(census.get('model_load_contract'))
    runtime = dict(torch=torch.__version__,cuda=torch.version.cuda,
                   transformers=importlib.metadata.version('transformers'))
    if (contract != recorded or census.get('capture_runtime') != runtime or
            attention_implementation not in ('eager','sdpa') or
            census.get('attention_implementation') != attention_implementation):
        raise RuntimeError('canonical model initialization, runtime or attention differs from census')
    root = Path(census['model'])
    files = sorted({*root.glob('*.safetensors'), *root.glob('*.json'),
                    *root.glob('*.model'), *root.glob('*.txt')})
    if not files or not (root / 'config.json').is_file():
        raise RuntimeError('calibration capture needs a complete local source checkpoint')
    def source_digest(path):
        return sha256(path, resource_check=resource_check, release_read_pages=release_read_pages)
    source = {p.name:source_digest(p) for p in files if p.is_file()}
    # The census already seals the producer's complete source/auxiliary
    # roster (including non-JSON tokenizer assets such as chat_template.jinja).
    # Check those bytes too, without inventing another producer identity.
    producer = (census.get('expert_projection') or {}).get('producer') or {}
    declared = producer.get('source') or {}
    expected = {**declared.get('files',{}),**declared.get('auxiliary_sha256',{})}
    if declared.get('config_sha256'):
        expected['config.json'] = declared['config_sha256']
    for name,digest in expected.items():
        actual = source[name] if name in source else source_digest(root/name)
        if actual != digest:
            raise RuntimeError(f'calibration source differs from census producer: {name}')
    if not any(name.endswith('.safetensors') for name in source):
        raise RuntimeError('calibration capture source has no safetensors weights')
    return dict(schema=SCHEMA, model_load_contract=contract,
                attention_implementation=attention_implementation,
                census_sha256=sha256(census_path),capture_runtime=runtime,
                source_files=source, calibration=dict(calibration),
                max_act_rows=int(max_act_rows), storage_source=SOURCE,
                units={name:list(shape) for name,shape in sorted(census['unit_shapes'].items())})


def _validate_tensors(name, payload, census, max_rows, *, check_finite=True):
    import torch
    columns = int(census['unit_shapes'][name][1])
    count = int(census['counts'][name])
    x, h = payload.get('inputs'), payload.get('hessian')
    if (payload.get('name') != name or payload.get('source') != SOURCE or
            payload.get('count') != count or count <= 0 or
            payload.get('max_abs') != census['max_abs'][name] or
            not math.isfinite(float(payload['max_abs']))):
        raise RuntimeError(f'{name}: calibration capture metadata disagrees with census')
    if (not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or
            list(x.shape) != [min(count, max_rows), columns] or
            not isinstance(h, torch.Tensor) or h.dtype != torch.float32 or
            list(h.shape) != [columns, columns]):
        raise RuntimeError(f'{name}: calibration capture tensor geometry or precision changed')
    if check_finite and (not torch.isfinite(x).all() or not torch.isfinite(h).all()):
        raise RuntimeError(f'{name}: calibration capture contains nonfinite tensors')
    return x, h


def _capture_storage_bytes(name, census, max_rows):
    columns, count = census['unit_shapes'][name][1], census['counts'][name]
    if any(type(value) is not int or value <= 0 for value in (columns, count, max_rows)):
        raise ValueError('verified capture needs positive exact census geometry')
    return 4 * (columns**2 + min(count, max_rows)*columns)


def _load_execution(policy, identity, output=None):
    from .perturbed_x_cache import normalize_verified_activation_load
    policy = normalize_verified_activation_load(policy)
    if policy is None:
        return None
    descriptor = dict(schema='prismaquant.capture_load_execution.v1', policy=policy,
                      capture_identity=identity)
    hasher = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True, separators=(',', ':')).iterencode(descriptor):
        hasher.update(chunk.encode())
    digest = hasher.hexdigest()
    value = dict(schema=descriptor['schema'], policy=policy, identity_sha256=digest,
                 loaded_entries=0, source_read_bytes=0, peak_buffer_bytes=0,
                 peak_archive_storage_bytes=0, live_buffer_bytes=0,
                 ordered_load_identities_sha256=hashlib.sha256(b'').hexdigest())
    if output is not None:
        if not isinstance(output, dict) or output:
            raise ValueError('capture load execution receipt must be an empty dictionary')
        output.update(value)
        return output
    return value


def merge_load_execution(total, partial):
    if total['identity_sha256'] != partial['identity_sha256'] or total['policy'] != partial['policy']:
        raise RuntimeError('capture load execution identity changed between units')
    for key in ('loaded_entries', 'source_read_bytes'):
        total[key] += partial[key]
    for key in ('peak_buffer_bytes', 'peak_archive_storage_bytes'):
        total[key] = max(total[key], partial[key])
    total['ordered_load_identities_sha256'] = hashlib.sha256((
        total['ordered_load_identities_sha256'] + partial['ordered_load_identities_sha256']).encode()).hexdigest()


def preflight_verified_capture_entries(root, entries, *, names, policy, census, max_rows):
    """Check the complete selected roster's file/geometry bounds before loading."""
    import stat
    from .perturbed_x_cache import activation_cache_filename, normalize_verified_activation_load
    policy = normalize_verified_activation_load(policy)
    if policy is None:
        raise ValueError('verified capture preflight requires an explicit load policy')
    largest_file = largest_storage = 0
    for name in names:
        expected = str(Path('inputs') / activation_cache_filename(name))
        record = entries[name]
        if record.get('path') != expected:
            raise RuntimeError(f'{name}: noncanonical capture artifact path')
        observed = (Path(root)/expected).lstat()
        if not stat.S_ISREG(observed.st_mode):
            raise RuntimeError(f'{name}: verified capture requires a regular nonsymlink file')
        if observed.st_size <= 0 or observed.st_size > policy['max_buffer_bytes']:
            raise RuntimeError(f'{name}: capture file exceeds verified serialized buffer budget')
        largest_file = max(largest_file, observed.st_size)
        largest_storage = max(largest_storage, _capture_storage_bytes(name, census, max_rows))
    return dict(max_file_bytes=largest_file, max_storage_bytes=largest_storage)


def _verified_capture_entry(path, name, *, expected_sha256, census, max_rows,
                            policy, execution, resource_check=None,
                            release_file_pages=False, expected_stat=None):
    from .perturbed_x_cache import load_verified_activation_cache_entry
    def validate(payload, *, check_finite):
        if (not isinstance(payload, dict) or set(payload) !=
                {'inputs', 'hessian', 'name', 'source', 'count', 'max_abs'}):
            raise RuntimeError(f'{name}: verified capture payload has unexpected owners')
        x, h = _validate_tensors(name, payload, census, max_rows, check_finite=False)
        if not x.is_contiguous() or not h.is_contiguous():
            raise RuntimeError(f'{name}: verified capture requires contiguous canonical tensors')
        if check_finite:
            from .perturbed_x_cache import bounded_cpu_float32_isfinite
            for tensor in (x, h):
                if not bounded_cpu_float32_isfinite(tensor,
                        max_scratch_bytes=policy['max_scratch_bytes']):
                    raise RuntimeError(f'{name}: calibration capture contains nonfinite tensors')
    payload, receipt = load_verified_activation_cache_entry(path,
        expected_sha256=expected_sha256, policy=policy,
        max_storage_bytes=_capture_storage_bytes(name, census, max_rows),
        validate=validate, expected_stat=expected_stat, resource_check=resource_check,
        release_file_pages=release_file_pages)
    execution['loaded_entries'] += 1
    execution['source_read_bytes'] += receipt['source_read_bytes']
    execution['peak_buffer_bytes'] = max(execution['peak_buffer_bytes'], receipt['file_bytes'])
    execution['peak_archive_storage_bytes'] = max(execution['peak_archive_storage_bytes'],
                                                  receipt['archive_storage_bytes'])
    execution['ordered_load_identities_sha256'] = hashlib.sha256((
        execution['ordered_load_identities_sha256'] + receipt['identity_sha256']).encode()).hexdigest()
    return payload


def publish_capture(root, *, census_path, identity, acts=None, hessians=None,
                    counts=None, maxima=None, existing_entries=None,
                    release_file_pages=False, resource_check=None,
                    verified_load_policy=None, load_execution=None):
    """Seal a complete capture, journalling per-unit file receipts atomically.

    ``existing_entries`` seals a previously measured raw capture without another
    model forward. Its bytes receive exactly the ordinary writer's validation.
    """
    import torch
    from .perturbed_x_cache import activation_cache_filename, write_activation_cache_entry
    root = Path(root).resolve()
    census = json.loads(Path(census_path).read_text())
    execution = _load_execution(verified_load_policy, identity, load_execution)
    names = sorted(identity['units'])
    if len({activation_cache_filename(n) for n in names}) != len(names):
        raise RuntimeError('calibration unit filenames collide')
    if set(names) != set(census['counts']):
        raise RuntimeError('calibration capture must cover the full census scope')
    if existing_entries is None and any(set(values or {}) != set(names)
                                       for values in (acts,hessians,counts,maxima)):
        raise RuntimeError('calibration capture arrays must cover the complete census')
    if existing_entries is not None and set(existing_entries) != set(names):
        raise RuntimeError('raw capture does not cover the full census')
    journal, digest, completed = prepare_journal(root/'journal', stage=STAGE,
        resume=True, identity=identity, qnames=names)
    if execution is not None:
        available = {**(existing_entries or {}), **completed}
        preflight_verified_capture_entries(root, available, names=sorted(available),
            policy=execution['policy'], census=census, max_rows=identity['max_act_rows'])
    records = {}
    for name in names:
        loaded_verified = False
        if resource_check is not None:
            resource_check(f'before_capture_seal:{name}')
        expected_path = Path('inputs') / activation_cache_filename(name)
        record = completed.get(name) or (existing_entries or {}).get(name)
        if record is None:
            payload = dict(inputs=acts[name],hessian=hessians[name],count=counts[name],
                           max_abs=maxima[name],name=name,source=SOURCE)
            _validate_tensors(name,payload,census,identity['max_act_rows'])
            path = write_activation_cache_entry(root/'inputs',name,acts[name],
                source=SOURCE,durable=True,hessian=hessians[name],count=counts[name],max_abs=maxima[name])
            file_stat = path.stat() if release_file_pages else None
            record = dict(path=str(expected_path),sha256=sha256(path))
        else:
            if record.get('path') != str(expected_path):
                raise RuntimeError(f'{name}: capture file is outside its canonical location')
            path = root/expected_path
            file_stat = path.stat() if release_file_pages else None
            if execution is None:
                if sha256(path) != record['sha256']:
                    raise RuntimeError(f'{name}: capture artifact checksum mismatch')
                _validate_tensors(name,torch.load(path,map_location='cpu',weights_only=True),
                                  census,identity['max_act_rows'])
            else:
                payload = _verified_capture_entry(path, name, expected_sha256=record['sha256'],
                    census=census, max_rows=identity['max_act_rows'], policy=execution['policy'],
                    execution=execution, resource_check=resource_check,
                    release_file_pages=release_file_pages, expected_stat=file_stat)
                del payload
                loaded_verified = True
        if release_file_pages and not loaded_verified:
            from .perturbed_x_cache import release_activation_cache_file_pages
            release_activation_cache_file_pages(path, expected_stat=file_stat)
        if name not in completed:
            write_unit(journal,stage=STAGE,qname=name,identity_sha256=digest,state=record)
        records[name] = record
        if resource_check is not None:
            resource_check(f'after_capture_seal:{name}')
    manifest = dict(schema=SCHEMA,status='complete',identity=identity,entries=records)
    path = root/'capture_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('existing complete calibration capture changed')
    _json(path,manifest)
    return dict(path=str(path),sha256=sha256(path))


class CaptureWriter:
    """Drain completed layers through the existing per-unit writer and journal.

    An interrupted traversal may leave valid unit entries, but no complete
    manifest. A retry recomputes the source forward and must match every entry
    it reuses. Completion additionally requires the actual initialization
    witness from that traversal, not only the census's expected descriptor.
    """

    def __init__(self, root, *, census_path, identity,
                 release_file_pages=False, resource_check=None,
                 verified_load_policy=None):
        self.root = Path(root).resolve()
        self.census_path = census_path
        self.census = json.loads(Path(census_path).read_text())
        self.identity = identity
        self.release_file_pages = release_file_pages
        self.resource_check = resource_check
        self.load_execution = _load_execution(verified_load_policy, identity)
        self.seal_load_execution = None
        self.names = sorted(identity['units'])
        if set(self.names) != set(self.census['counts']):
            raise RuntimeError('calibration writer scope differs from census')
        import shutil
        from .perturbed_x_cache import activation_cache_filename
        self.root.mkdir(parents=True, exist_ok=True)
        entries = {name: 4*(int(self.census['unit_shapes'][name][1])**2 +
            min(int(self.census['counts'][name]), identity['max_act_rows'])*
            int(self.census['unit_shapes'][name][1]))+16384 for name in self.names}
        existing = 0
        for name, bound in entries.items():
            path = self.root/'inputs'/activation_cache_filename(name)
            if path.is_file():
                existing += min(path.stat().st_size, bound)
        required = sum(entries.values())+max(entries.values(), default=0)-existing
        available = shutil.disk_usage(self.root).free
        if available < required:
            raise RuntimeError(f'canonical capture needs {required} additional disk bytes; '
                               f'only {available} are available')
        self.journal, self.digest, self.completed = prepare_journal(
            self.root/'journal', stage=STAGE, resume=True, identity=identity, qnames=self.names)
        if self.load_execution is not None:
            preflight_verified_capture_entries(self.root, self.completed, names=sorted(self.completed),
                policy=self.load_execution['policy'], census=self.census,
                max_rows=self.identity['max_act_rows'])
        self.records = {}

    def write(self, *, acts, hessians, counts, maxima):
        from .perturbed_x_cache import activation_cache_filename, write_activation_cache_entry
        names = set(acts)
        if (not names <= set(self.names) or names.intersection(self.records) or
                any(set(values) != names for values in (hessians, counts, maxima))):
            raise RuntimeError('calibration writer has repeated or inconsistent layer scope')
        for name in sorted(names):
            if self.resource_check is not None:
                self.resource_check(f'before_capture_write:{name}')
            payload = dict(inputs=acts[name], hessian=hessians[name], count=counts[name],
                           max_abs=maxima[name], name=name, source=SOURCE)
            _validate_tensors(name, payload, self.census, self.identity['max_act_rows'])
            previous = self.completed.get(name)
            if previous is not None:
                import torch
                expected = str(Path('inputs')/activation_cache_filename(name))
                path = self.root/expected
                file_stat = path.stat() if self.release_file_pages else None
                if previous.get('path') != expected:
                    raise RuntimeError(f'{name}: interrupted capture entry changed')
                if self.load_execution is None:
                    if sha256(path) != previous.get('sha256'):
                        raise RuntimeError(f'{name}: interrupted capture entry changed')
                    old = torch.load(path, map_location='cpu', weights_only=True)
                else:
                    old = _verified_capture_entry(path, name, expected_sha256=previous.get('sha256'),
                        census=self.census, max_rows=self.identity['max_act_rows'],
                        policy=self.load_execution['policy'], execution=self.load_execution,
                        resource_check=self.resource_check, release_file_pages=self.release_file_pages,
                        expected_stat=file_stat)
                old_x = old_h = None
                try:
                    old_x, old_h = _validate_tensors(name, old, self.census, self.identity['max_act_rows'],
                                                    check_finite=self.load_execution is None)
                    if not torch.equal(old_x, acts[name]) or not torch.equal(old_h, hessians[name]):
                        raise RuntimeError(f'{name}: replayed capture differs from interrupted entry')
                    record = previous
                finally:
                    # One loaded validation entry expires before its successor,
                    # including when replay equality or geometry refuses.
                    del old, old_x, old_h
            else:
                path = write_activation_cache_entry(self.root/'inputs', name, acts[name],
                    source=SOURCE, durable=True, hessian=hessians[name],
                    count=counts[name], max_abs=maxima[name])
                file_stat = path.stat() if self.release_file_pages else None
                record = dict(path=str(Path('inputs')/activation_cache_filename(name)), sha256=sha256(path))
            if self.release_file_pages and (self.load_execution is None or previous is None):
                from .perturbed_x_cache import release_activation_cache_file_pages
                release_activation_cache_file_pages(path, expected_stat=file_stat)
            if previous is None:
                write_unit(self.journal, stage=STAGE, qname=name,
                           identity_sha256=self.digest, state=record)
            self.records[name] = record
            if self.resource_check is not None:
                self.resource_check(f'after_capture_write:{name}')

    def finish(self, *, model_load_contract):
        from prismaquant import validate_source_initialization_contract
        actual = validate_source_initialization_contract(model_load_contract)
        if actual != self.identity['model_load_contract']:
            raise RuntimeError('actual capture initialization differs from the census')
        extra = {}
        if self.load_execution is not None:
            self.seal_load_execution = {}
            extra = dict(verified_load_policy=self.load_execution['policy'],
                         load_execution=self.seal_load_execution)
        return publish_capture(self.root, census_path=self.census_path,
                               identity=self.identity, existing_entries=self.records,
                               release_file_pages=self.release_file_pages,
                               resource_check=self.resource_check, **extra)


def require_capture_contract(path, expected_sha256=None):
    """Validate a complete canonical capture before downstream preparation."""
    from prismaquant import validate_source_initialization_contract
    path = Path(path)
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise RuntimeError('priced calibration capture manifest changed')
    manifest = json.loads(path.read_text())
    identity = manifest.get('identity') or {}
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'complete':
        raise RuntimeError('not a complete canonical calibration capture v2')
    contract = validate_source_initialization_contract(identity.get('model_load_contract'))
    runtime = identity.get('capture_runtime')
    if (not isinstance(runtime,dict) or set(runtime) != {'torch','cuda','transformers'} or
            not isinstance(runtime.get('torch'),str) or not runtime['torch'] or
            (runtime.get('cuda') is not None and not isinstance(runtime['cuda'],str))):
        raise RuntimeError('canonical capture runtime identity is incomplete')
    if (identity.get('schema') != SCHEMA or
            identity.get('attention_implementation') not in ('eager','sdpa') or
            (identity.get('capture_runtime') or {}).get('transformers') != contract['transformers_version'] or
            not identity.get('source_files') or not identity.get('units') or
            set(manifest.get('entries',{})) != set(identity['units'])):
        raise RuntimeError('canonical capture runtime, source or completeness is invalid')
    return manifest


def prefetch_capture(path, *, expected_identity, census, names, device,
                     expected_sha256=None, resource_check=None,
                     release_file_pages=False, verified_load_policy=None,
                     load_execution=None):
    """Verify selected files and make all selected X/H resident before encoding."""
    import torch
    from .perturbed_x_cache import activation_cache_filename
    path = Path(path)
    execution = _load_execution(verified_load_policy, expected_identity, load_execution)
    digest = sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError('priced calibration capture manifest changed')
    manifest = require_capture_contract(path, expected_sha256=expected_sha256)
    names = sorted(names)
    if (manifest.get('schema') != SCHEMA or manifest.get('status') != 'complete' or
            manifest.get('identity') != expected_identity or
            set(manifest.get('entries',{})) != set(expected_identity['units']) or
            not set(names) <= set(expected_identity['units'])):
        raise RuntimeError('calibration capture identity, completeness or scope mismatch')
    if execution is not None:
        preflight_verified_capture_entries(path.parent, manifest['entries'], names=names,
            policy=execution['policy'], census=census, max_rows=expected_identity['max_act_rows'])
    acts, hessians, counts, maxima = {}, {}, {}, {}
    payload = x = h = None
    try:
        for name in names:
            record = manifest['entries'][name]
            relative = str(Path('inputs') / activation_cache_filename(name))
            if record.get('path') != relative:
                raise RuntimeError(f'{name}: noncanonical capture artifact path')
            artifact = path.parent/relative
            file_stat = artifact.stat() if release_file_pages else None
            if execution is None:
                if sha256(artifact, resource_check=resource_check,
                          release_read_pages=release_file_pages) != record.get('sha256'):
                    raise RuntimeError(f'{name}: capture artifact checksum mismatch')
                if resource_check is not None:
                    columns = int(census['unit_shapes'][name][1])
                    resource_check(f'before_capture_prefetch:{name}', reserve_bytes=8*(
                        columns**2+min(census['counts'][name], expected_identity['max_act_rows'])*columns))
                payload = torch.load(artifact,map_location='cpu',weights_only=True)
            else:
                payload = _verified_capture_entry(artifact, name, expected_sha256=record.get('sha256'),
                    census=census, max_rows=expected_identity['max_act_rows'], policy=execution['policy'],
                    execution=execution, resource_check=resource_check,
                    release_file_pages=release_file_pages, expected_stat=file_stat)
                if resource_check is not None:
                    resource_check(f'before_capture_prefetch:{name}', reserve_bytes=
                        2*_capture_storage_bytes(name, census, expected_identity['max_act_rows']))
            x,h = _validate_tensors(name,payload,census,expected_identity['max_act_rows'],
                                    check_finite=execution is None)
            acts[name],hessians[name] = x.to(device),h.to(device)
            counts[name],maxima[name] = payload['count'],payload['max_abs']
            if release_file_pages:
                from .perturbed_x_cache import release_activation_cache_file_pages
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize(device)
            del payload, x, h
            payload = x = h = None
            if release_file_pages and execution is None:
                release_activation_cache_file_pages(artifact, expected_stat=file_stat)
            if resource_check is not None:
                resource_check(f'after_capture_prefetch:{name}')
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
    except BaseException:
        if execution is not None:
            acts.clear()
            hessians.clear()
            payload = x = h = None
        raise
    resident = sum(t.numel()*t.element_size() for t in (*acts.values(),*hessians.values()))
    print(f'[campaign] calibration prefetched: {len(names)} units, {resident} resident bytes, 0 misses',flush=True)
    return (acts,hessians,counts,maxima),dict(path=str(path.resolve()),sha256=digest)
