"""Campaign capture receipts over the existing per-unit activation cache storage.

Scoring inputs retain the campaign's first rows in float32, while Hessians,
counts and maxima cover the entire draw. No row sampling or runtime scheduling
lives here. Readers verify inputs and prefetch their selected scope before use.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import threading

from .cost_stage_checkpoint import atomic_write_bytes, prepare_journal, write_unit

SCHEMA = 'prismaquant.tessera_calibration_cache.v2'
STAGE = 'tessera_calibration_capture'
SOURCE = 'tessera_campaign_prefix_f32_v1'


def sha256(path, *, resource_check=None, release_read_pages=False, file_descriptor=None):
    # A descriptor alias opens the already owned object, never its possibly
    # replaced source pathname. The ordinary full-capture path is unchanged.
    if file_descriptor is not None and (type(file_descriptor) is not int or file_descriptor < 0):
        raise ValueError('source hash descriptor must be an open file descriptor')
    read_path = Path(path) if file_descriptor is None else Path(f'/proc/self/fd/{file_descriptor}')
    with read_path.open('rb') as handle:
        if resource_check is not None or release_read_pages or file_descriptor is not None:
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
                     resource_check=None, release_read_pages=False,
                     source_authentication=None):
    """Preserve full identity; selected readers authenticate consumed objects.

    Canonical capture hashes every source file. A hash-bound complete capture
    can supply its original roster only through the descriptor owner below;
    no pathname/mtime digest cache authorizes a selected source read.
    """
    import importlib.metadata
    import torch
    census_path = Path(census_path)
    census_raw = census_path.read_bytes()
    census = json.loads(census_raw)
    census_digest = hashlib.sha256(census_raw).hexdigest()
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
    # The census already seals the producer's complete source/auxiliary
    # roster (including non-JSON tokenizer assets such as chat_template.jinja).
    # Check those bytes too, without inventing another producer identity.
    producer = (census.get('expert_projection') or {}).get('producer') or {}
    declared = producer.get('source') or {}
    expected = {**declared.get('files',{}),**declared.get('auxiliary_sha256',{})}
    if declared.get('config_sha256'):
        expected['config.json'] = declared['config_sha256']
    if source_authentication is None:
        source = {p.name:source_digest(p) for p in files if p.is_file()}
        for name,digest in expected.items():
            actual = source[name] if name in source else source_digest(root/name)
            if actual != digest:
                raise RuntimeError(f'calibration source differs from census producer: {name}')
    else:
        if not isinstance(source_authentication, CaptureSourceAuthentication):
            raise TypeError('selected source needs the complete-capture descriptor owner')
        source = source_authentication.source_files(root, census_digest,
            {p.name for p in files if p.is_file()}, expected)
    if not any(name.endswith('.safetensors') for name in source):
        raise RuntimeError('calibration capture source has no safetensors weights')
    return dict(schema=SCHEMA, model_load_contract=contract,
                attention_implementation=attention_implementation,
                census_sha256=census_digest,capture_runtime=runtime,
                source_files=source, calibration=dict(calibration),
                max_act_rows=int(max_act_rows), storage_source=SOURCE,
                units={name:list(shape) for name,shape in sorted(census['unit_shapes'].items())})


def _source_stat(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class CaptureSourceAuthentication:
    """One complete capture's source descriptors, not a weight/digest cache.

    Header inspection may open an unconsumed shard without hashing its payload.
    Tensor reads authenticate the same held object first, once in this owner's
    lifetime. Stat fences reject mutation/replacement; only SHA256 authenticates
    content. Ordinary source files must remain stable through their read leases.
    Construct through ``authenticate_selected_capture_source``.
    """

    def __init__(self, root, identity, producer_source, *, manifest_sha256,
                 resource_check=None, release_read_pages=False):
        self.root = Path(os.path.abspath(root))
        self._identity_json = json.dumps(identity, sort_keys=True, allow_nan=False)
        self._source_files = dict(identity['source_files'])
        self._expected = dict(self._source_files)
        declared = {**producer_source.get('files', {}),
                    **producer_source.get('auxiliary_sha256', {})}
        if producer_source.get('config_sha256'):
            declared['config.json'] = producer_source['config_sha256']
        for name, digest in declared.items():
            if name in self._expected and self._expected[name] != digest:
                raise RuntimeError(f'capture source roster differs from census producer: {name}')
            self._expected[name] = digest
        for name, digest in self._expected.items():
            if (not isinstance(name, str) or not name or Path(name).name != name or
                    name in ('.', '..') or not isinstance(digest, str) or
                    len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
                raise RuntimeError('capture source roster has an invalid path or SHA256')
        self.manifest_sha256 = manifest_sha256
        self.resource_check = resource_check
        self.release_read_pages = release_read_pages
        self._files = {}
        self._lock = threading.RLock()
        self._readers = 0
        self._closed = False

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_args):
        self.close()

    def _require_open(self):
        if self._closed:
            raise RuntimeError('capture source descriptor owner is closed')

    def _name(self, path):
        value = Path(os.path.abspath(path))
        if value.parent != self.root or value.name not in self._expected:
            raise RuntimeError(f'consumed source is absent from the sealed roster: {path}')
        return value.name

    def _file(self, path):
        name = self._name(path)
        with self._lock:
            self._require_open()
            if name not in self._files:
                # A replaced FIFO must be refused by fstat, never block in
                # open waiting for a writer. Regular files and HF symlinks
                # retain the same read semantics with O_NONBLOCK.
                fd = os.open(self.root/name, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
                try:
                    before = os.fstat(fd)
                    if not stat.S_ISREG(before.st_mode):
                        raise RuntimeError('authenticated source must be a regular file')
                    state = dict(fd=fd, before=before, sha256=None, payload_reads=0,
                                 lock=threading.Lock())
                    self._check_file(name, state)
                    self._files[name] = state
                except BaseException:
                    os.close(fd)
                    raise
            return name, self._files[name]

    def _check_file(self, name, state):
        try:
            current = (os.fstat(state['fd']), os.stat(self.root/name))
            if any(_source_stat(value) != _source_stat(state['before']) for value in current):
                raise RuntimeError(f'authenticated source changed during consumption: {name}')
        except OSError as exc:
            raise RuntimeError(f'authenticated source changed during consumption: {name}') from exc

    def require_unchanged(self):
        with self._lock:
            self._require_open()
            for name, state in self._files.items():
                self._check_file(name, state)

    def _authenticate(self, name, state):
        with state['lock']:
            self._check_file(name, state)
            if state['sha256'] is None:
                digest = sha256(self.root/name, file_descriptor=state['fd'],
                    resource_check=self.resource_check,
                    release_read_pages=self.release_read_pages)
                self._check_file(name, state)
                if digest != self._expected[name]:
                    raise RuntimeError(f'calibration source content differs from sealed capture: {name}')
                state['sha256'] = digest

    def source_files(self, root, census_digest, names, producer_digests):
        canonical = json.loads(self._identity_json)
        if (Path(os.path.abspath(root)) != self.root or
                census_digest != canonical['census_sha256'] or names != set(self._source_files) or
                any(self._expected.get(name) != digest for name, digest in producer_digests.items())):
            raise RuntimeError('selected source census or complete source roster changed')
        # Metadata includes producer auxiliaries outside capture's historical
        # glob. Keep that glob/identity unchanged, while still authenticating it.
        for name in sorted(self._expected):
            if not name.endswith('.safetensors'):
                _, state = self._file(self.root/name)
                self._authenticate(name, state)
        self.require_unchanged()
        return dict(self._source_files)

    def read_json(self, path):
        with self._lock:
            name, state = self._file(path)
            if name.endswith('.safetensors'):
                raise RuntimeError('source JSON reader cannot read a payload shard')
            self._readers += 1
        try:
            self._authenticate(name, state)
            with open(f"/proc/self/fd/{state['fd']}", 'rb') as handle:
                result = json.load(handle)
            self._check_file(name, state)
            return result
        finally:
            with self._lock:
                self._readers -= 1

    def safe_open(self, factory, path, *args, **kwargs):
        return _CaptureSourceSafeOpen(self, factory, path, args, kwargs)

    def file_stat(self, path):
        name, state = self._file(path)
        self._check_file(name, state)
        return state['before']

    def descriptor_path(self, path):
        name, state = self._file(path)
        self._check_file(name, state)
        if state['sha256'] is None:
            raise RuntimeError('source payload descriptor has not been authenticated')
        return f"/proc/self/fd/{state['fd']}"

    def receipt(self):
        self.require_unchanged()
        verified = [{"name": name, "sha256": state['sha256'],
            "bytes_hashed": state['before'].st_size, "payload_reads": state['payload_reads']}
            for name, state in sorted(self._files.items()) if state['sha256'] is not None]
        return dict(schema='prismaquant.selected_source_authentication.v1',
            capture_manifest_sha256=self.manifest_sha256,
            census_sha256=json.loads(self._identity_json)['census_sha256'],
            authentication='fresh SHA256 through held read-only source descriptors',
            verified_files=verified,
            payload_bytes_hashed=sum(row['bytes_hashed'] for row in verified
                                     if row['name'].endswith('.safetensors')),
            metadata_only_shards=sorted(name for name, state in self._files.items()
                if name.endswith('.safetensors') and state['sha256'] is None))

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._readers:
                raise RuntimeError('cannot close capture source with active readers')
            try:
                self.require_unchanged()
            finally:
                self._closed = True
                for state in self._files.values():
                    os.close(state['fd'])


class _CaptureSourceSafeOpen:
    def __init__(self, owner, factory, path, args, kwargs):
        self.owner, self.name = owner, owner._name(path)
        self.entered = self.closed = self.authenticated = False
        with owner._lock:
            _, self.state = owner._file(path)
            owner._check_file(self.name, self.state)
            owner._readers += 1
        self.context = None
        try:
            self.context = factory(f"/proc/self/fd/{self.state['fd']}", *args, **kwargs)
            owner._check_file(self.name, self.state)
        except BaseException as exc:
            try:
                if self.context is not None:
                    self.context.__exit__(type(exc), exc, exc.__traceback__)
            finally:
                with owner._lock:
                    owner._readers -= 1
            raise

    def __enter__(self):
        try:
            self.handle = self.context.__enter__()
            self.owner._check_file(self.name, self.state)
            self.entered = True
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        if self.closed:
            return
        try:
            self.context.__exit__(*args)
            self.owner._check_file(self.name, self.state)
        finally:
            self.closed = True
            with self.owner._lock:
                self.owner._readers -= 1

    def _require_entered(self):
        if not self.entered or self.closed:
            raise RuntimeError('authenticated source reader is outside its read lease')

    def _payload(self):
        self._require_entered()
        self.owner._check_file(self.name, self.state)
        if not self.authenticated:
            self.owner._authenticate(self.name, self.state)
            self.authenticated = True
        with self.state['lock']:
            self.state['payload_reads'] += 1

    def keys(self):
        self._require_entered()
        return self.handle.keys()

    def metadata(self):
        self._require_entered()
        return self.handle.metadata()

    def get_tensor(self, name):
        self._payload()
        return self.handle.get_tensor(name)

    def get_slice(self, name):
        self._require_entered()
        return _CaptureSourceSlice(self, self.handle.get_slice(name))


class _CaptureSourceSlice:
    def __init__(self, reader, value):
        self.reader, self.value = reader, value

    def get_shape(self):
        self.reader._require_entered()
        return self.value.get_shape()

    def get_dtype(self):
        self.reader._require_entered()
        return self.value.get_dtype()

    def __getitem__(self, index):
        self.reader._payload()
        return self.value[index]


def _validate_tensors(name, payload, census, max_rows):
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
    if not torch.isfinite(x).all() or not torch.isfinite(h).all():
        raise RuntimeError(f'{name}: calibration capture contains nonfinite tensors')
    return x, h


def publish_capture(root, *, census_path, identity, acts=None, hessians=None,
                    counts=None, maxima=None, existing_entries=None,
                    release_file_pages=False, resource_check=None):
    """Seal a complete capture, journalling per-unit file receipts atomically.

    ``existing_entries`` seals a previously measured raw capture without another
    model forward. Its bytes receive exactly the ordinary writer's validation.
    """
    import torch
    from .perturbed_x_cache import activation_cache_filename, write_activation_cache_entry
    root = Path(root).resolve()
    census = json.loads(Path(census_path).read_text())
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
    records = {}
    for name in names:
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
            if sha256(path) != record['sha256']:
                raise RuntimeError(f'{name}: capture artifact checksum mismatch')
            _validate_tensors(name,torch.load(path,map_location='cpu',weights_only=True),
                              census,identity['max_act_rows'])
        if release_file_pages:
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
                 release_file_pages=False, resource_check=None):
        self.root = Path(root).resolve()
        self.census_path = census_path
        self.census = json.loads(Path(census_path).read_text())
        self.identity = identity
        self.release_file_pages = release_file_pages
        self.resource_check = resource_check
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
                if previous.get('path') != expected or sha256(path) != previous.get('sha256'):
                    raise RuntimeError(f'{name}: interrupted capture entry changed')
                old = torch.load(path, map_location='cpu', weights_only=True)
                old_x, old_h = _validate_tensors(name, old, self.census, self.identity['max_act_rows'])
                if not torch.equal(old_x, acts[name]) or not torch.equal(old_h, hessians[name]):
                    raise RuntimeError(f'{name}: replayed capture differs from interrupted entry')
                record = previous
                # Admission allows one loaded validation entry alongside the
                # current capture. Drop all views before loading its successor.
                del old, old_x, old_h
            else:
                path = write_activation_cache_entry(self.root/'inputs', name, acts[name],
                    source=SOURCE, durable=True, hessian=hessians[name],
                    count=counts[name], max_abs=maxima[name])
                file_stat = path.stat() if self.release_file_pages else None
                record = dict(path=str(Path('inputs')/activation_cache_filename(name)), sha256=sha256(path))
            if self.release_file_pages:
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
        return publish_capture(self.root, census_path=self.census_path,
                               identity=self.identity, existing_entries=self.records,
                               release_file_pages=self.release_file_pages,
                               resource_check=self.resource_check)


def require_capture_contract(path, expected_sha256=None):
    """Validate a complete canonical capture before downstream preparation."""
    from prismaquant import validate_source_initialization_contract
    path = Path(path)
    raw = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError('priced calibration capture manifest changed')
    manifest = json.loads(raw)
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


def authenticate_selected_capture_source(census_path, capture_path, *, expected_sha256,
        model, max_act_rows, attention_implementation, calibration_parameters=None,
        resource_check=None, release_read_pages=False):
    """Bind the public complete-capture contract before selected source loading.

    The original identity stays equivalent. Payload validation is deferred only
    to the authenticated reader's first tensor consumption; small metadata and
    all identity fields are checked before model construction.
    """
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or
            any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise RuntimeError('selected source requires a hash-bound complete capture')
    manifest = require_capture_contract(capture_path, expected_sha256=expected_sha256)
    canonical = manifest['identity']
    census = json.loads(Path(census_path).read_text())
    if (census.get('model') != str(model) or
            canonical['model_load_contract']['schema'] != 'prismaquant.streaming_initialization.v1' or
            any(census.get(key) != value for key, value in (calibration_parameters or {}).items())):
        raise RuntimeError('selected source model, draw or streaming witness differs from census')
    producer = ((census.get('expert_projection') or {}).get('producer') or {}).get('source') or {}
    owner = CaptureSourceAuthentication(model, canonical, producer,
        manifest_sha256=expected_sha256, resource_check=resource_check,
        release_read_pages=release_read_pages)
    try:
        actual = capture_identity(census_path, calibration=canonical['calibration'],
            max_act_rows=max_act_rows, model_load_contract=census.get('model_load_contract'),
            attention_implementation=attention_implementation, source_authentication=owner)
        if actual != canonical:
            raise RuntimeError('selected source capture identity differs from the canonical census')
        return owner
    except BaseException:
        owner.close()
        raise


def prefetch_capture(path, *, expected_identity, census, names, device,
                     expected_sha256=None, resource_check=None,
                     release_file_pages=False):
    """Verify selected files and make all selected X/H resident before encoding."""
    import torch
    from .perturbed_x_cache import activation_cache_filename
    path = Path(path)
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
    acts, hessians, counts, maxima = {}, {}, {}, {}
    for name in names:
        record = manifest['entries'][name]
        relative = str(Path('inputs') / activation_cache_filename(name))
        if record.get('path') != relative:
            raise RuntimeError(f'{name}: noncanonical capture artifact path')
        artifact = path.parent/relative
        file_stat = artifact.stat() if release_file_pages else None
        if sha256(artifact, resource_check=resource_check,
                  release_read_pages=release_file_pages) != record.get('sha256'):
            raise RuntimeError(f'{name}: capture artifact checksum mismatch')
        if resource_check is not None:
            columns = int(census['unit_shapes'][name][1])
            resource_check(f'before_capture_prefetch:{name}', reserve_bytes=8*(
                columns**2+min(census['counts'][name], expected_identity['max_act_rows'])*columns))
        payload = torch.load(artifact,map_location='cpu',weights_only=True)
        x,h = _validate_tensors(name,payload,census,expected_identity['max_act_rows'])
        acts[name],hessians[name] = x.to(device),h.to(device)
        counts[name],maxima[name] = payload['count'],payload['max_abs']
        if release_file_pages:
            from .perturbed_x_cache import release_activation_cache_file_pages
            if str(device).startswith('cuda'):
                torch.cuda.synchronize(device)
            del payload, x, h
            release_activation_cache_file_pages(artifact, expected_stat=file_stat)
        if resource_check is not None:
            resource_check(f'after_capture_prefetch:{name}')
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    resident = sum(t.numel()*t.element_size() for t in (*acts.values(),*hessians.values()))
    print(f'[campaign] calibration prefetched: {len(names)} units, {resident} resident bytes, 0 misses',flush=True)
    return (acts,hessians,counts,maxima),dict(path=str(path.resolve()),sha256=digest)
