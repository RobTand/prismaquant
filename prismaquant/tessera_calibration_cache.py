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
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import stat
import threading
from types import MappingProxyType

from .cost_stage_checkpoint import atomic_write_bytes, prepare_journal, write_unit
from .digests import (
    DIRECT_ASCII_LAX,
    DIRECT_ASCII_STRICT,
    SOURCE_HASH_BLOCK_BYTES,
    bytes_sha256hex,
    hex_chain_sha256hex,
    indent2_json_file_bytes,
)
from .file_identity import file_stat_signature
from .memory_management import reserve_allocation

SCHEMA = 'prismaquant.tessera_calibration_cache.v2'
STAGE = 'tessera_calibration_capture'
SOURCE = 'tessera_campaign_prefix_f32_v1'
# A capture manifest is retained only by an explicitly scoped owner.  The
# current GLM manifest is 12.67 MiB; keep the reuse path bounded rather than
# making a successful large campaign an unbounded metadata cache.
MAX_CAPTURE_METADATA_BYTES = 16 * 1024**2
MAX_CAPTURE_EXECUTION_POLICIES = 8
#: The receipt of a streamed capture's recording source owner (PQ #1896).
RECORDING_RECEIPT_SCHEMA = 'prismaquant.capture_source_recording.v1'


#: The guarded source hash's read block is ``digests.SOURCE_HASH_BLOCK_BYTES``,
#: imported above: admission (``autoscale.selected_anchor_resources``) reads the
#: same number without importing this lane module.


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
                block = handle.read(SOURCE_HASH_BLOCK_BYTES)
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


@contextmanager
def _on_io_engine(calls):
    """Run ``{key: callable}`` together on the process's IO engine; yield ``result_of(key)``.

    The pool is ``io_engine.ENGINE`` (#1294 forbids a new one). Read the
    results in the serial loop's order, so the first error raised is the one
    that loop raised. On exit the calls not yet started are cancelled and the
    running ones awaited, so no read is still in flight when an error
    propagates.
    """
    from concurrent.futures import wait
    from .io_engine import ENGINE
    futures = {key: ENGINE.submit(call) for key, call in calls.items()}
    try:
        yield lambda key: futures[key].result()
    finally:
        for future in futures.values():
            future.cancel()
        wait(futures.values())


@contextmanager
def _engine_source_hashes(paths, *, resource_check=None, release_read_pages=False):
    """Hash ``paths`` together on the process's IO engine (PQ #1887).

    Each file gets the same guarded :func:`sha256` the serial loop ran: every
    byte, the identity fence, ``resource_check`` and page release. Only the
    files overlap. One reader at a time took 71 min over the 599 GB
    GLM-5.3-Flash source on NFS-RDMA while the row's GPU idled. At most
    ``min(ENGINE.width, len(paths))`` hashes are in flight, each holding two
    ``SOURCE_HASH_BLOCK_BYTES`` blocks; this runs before the model is
    resident, and ``resource_check`` still refuses on absolute bytes.

    Yields ``digest_of(path)``, with :func:`_on_io_engine`'s ordering and
    cancellation. A streamed capture does not take this path: it records
    each digest from the read that streams the file (PQ #1896).
    """
    with _on_io_engine({path: partial(sha256, path, resource_check=resource_check,
                                      release_read_pages=release_read_pages)
                        for path in paths}) as digest_of:
        yield digest_of


def _json(path, value):
    atomic_write_bytes(Path(path), indent2_json_file_bytes(value))


def capture_source_files(root):
    """The source files a capture identity seals: every weight and auxiliary file."""
    root = Path(root)
    return sorted(path for path in {*root.glob('*.safetensors'), *root.glob('*.json'),
                                    *root.glob('*.model'), *root.glob('*.txt')}
                  if path.is_file())


def capture_identity(census_path, *, calibration, max_act_rows,
                     model_load_contract, attention_implementation,
                     resource_check=None, release_read_pages=False,
                     source_authentication=None):
    """Preserve full identity; selected readers authenticate consumed objects.

    Without an owner, canonical capture hashes every source file. A
    hash-bound complete capture can supply its original roster only through
    the descriptor owner below; no pathname/mtime digest cache authorizes a
    selected source read.

    A recording owner (:meth:`CaptureSourceAuthentication.recording`) is a
    streamed capture's (PQ #1896): its source digests are the digests of the
    bytes the capture reads, recorded as it reads them, so they do not exist
    yet. The identity it returns carries no ``source_files``; the capture
    binds them at its seal (:func:`bind_capture_source`).
    """
    import importlib.metadata
    import torch
    census_path = Path(census_path)
    census_raw = census_path.read_bytes()
    census = json.loads(census_raw)
    census_digest = bytes_sha256hex(census_raw)
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
    files = capture_source_files(root)
    if not files or not (root / 'config.json').is_file():
        raise RuntimeError('calibration capture needs a complete local source checkpoint')
    hashing = dict(resource_check=resource_check, release_read_pages=release_read_pages)
    # The census already seals the producer's complete source/auxiliary
    # roster (including non-JSON tokenizer assets such as chat_template.jinja).
    # Check those bytes too, without inventing another producer identity.
    producer = (census.get('expert_projection') or {}).get('producer') or {}
    declared = producer.get('source') or {}
    expected = _producer_digests(declared)
    fields = dict(schema=SCHEMA, model_load_contract=contract,
                  attention_implementation=attention_implementation,
                  census_sha256=census_digest, capture_runtime=runtime,
                  calibration=dict(calibration), max_act_rows=int(max_act_rows),
                  storage_source=SOURCE,
                  units={name: list(shape) for name, shape in sorted(census['unit_shapes'].items())})
    if getattr(source_authentication, 'is_recording', False):
        if not isinstance(source_authentication, CaptureSourceAuthentication):
            raise TypeError('a recording source owner must be the capture descriptor owner')
        # Nothing is hashed here: each file is hashed from the read that
        # consumes it, and the rest at the seal (PQ #1896).
        source_authentication.require_recording_roster(root, {p.name for p in files}, expected)
        if not any(p.name.endswith('.safetensors') for p in files):
            raise RuntimeError('calibration capture source has no safetensors weights')
        return fields
    if source_authentication is None:
        present = [p for p in files if p.is_file()]
        with _engine_source_hashes(present, **hashing) as digest_of:
            source = {p.name:digest_of(p) for p in present}
        with _engine_source_hashes([root/name for name in expected if name not in source],
                                   **hashing) as digest_of:
            for name,digest in expected.items():
                actual = source[name] if name in source else digest_of(root/name)
                if actual != digest:
                    raise RuntimeError(f'calibration source differs from census producer: {name}')
    else:
        if not isinstance(source_authentication, CaptureSourceAuthentication):
            raise TypeError('selected source needs the complete-capture descriptor owner')
        source = source_authentication.source_files(root, census_digest,
            {p.name for p in files if p.is_file()}, expected)
    if not any(name.endswith('.safetensors') for name in source):
        raise RuntimeError('calibration capture source has no safetensors weights')
    return dict(fields, source_files=source)


def _require_sha256_roster(source_files, *, what):
    for name, digest in source_files.items():
        if (not isinstance(name, str) or not name or Path(name).name != name or
                name in ('.', '..') or not isinstance(digest, str) or
                len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
            raise RuntimeError(f'{what} has an invalid path or SHA256')


def bind_capture_source(identity, source_files):
    """A streamed capture's identity with the source digests it recorded (PQ #1896).

    The traversal identity (``capture_identity`` through a recording owner)
    names everything a capture is computed from except the source digests,
    which exist only once every file has been read. The seal binds them
    here, and the sealed manifest's identity has the shape every downstream
    reader already verifies against.
    """
    if 'source_files' in identity:
        raise RuntimeError('this capture identity already binds its source files')
    source_files = dict(sorted(source_files.items()))
    _require_sha256_roster(source_files, what='recorded capture source')
    if not any(name.endswith('.safetensors') for name in source_files):
        raise RuntimeError('calibration capture source has no safetensors weights')
    return dict(identity, source_files=source_files)


def _producer_digests(producer_source):
    """``{file name: sha256}`` a census producer declares for its source."""
    declared = {**producer_source.get('files', {}),
                **producer_source.get('auxiliary_sha256', {})}
    if producer_source.get('config_sha256'):
        declared['config.json'] = producer_source['config_sha256']
    return declared


#: Initialization contracts a selected-source capture may carry: the streamed
#: text forward, and the checkpoint load of an MTP layer outside it.
SELECTED_SOURCE_LOAD_SCHEMAS = frozenset({
    'prismaquant.streaming_initialization.v1',
    'prismaquant.mtp_layer_initialization.v1',
})


def streamed_identity_proof_digests(root, cache_path, source_files=None, *, live_stat,
                                    expected_sha256=None):
    """Check a streamed identity proof as adoption checks it; return its digests.

    The one test of whether a ``prismaquant.streamed_model.identity_cache.v1``
    proof may stand in for hashing ``root``'s shards: the proof's content seal,
    its declared digest when one is bound, the complete checkpoint index and
    shard coverage, every shard's SHA against ``source_files`` (a capture's
    roster), and the six-field stat predicate against ``live_stat(name)``, the
    stat of the object the caller will read. A capture owner adopts through
    it (``CaptureSourceAuthentication.adopt_streamed_identity_cache``, with
    its held descriptors' stats) and the campaign planner refuses through it
    (``dispatch_tessera_campaign``, with a fresh ``os.stat``, PQ #1654), so a
    proof the planner binds is one the row adopts unless the source changes
    in between. Without ``source_files`` the shards are checked against the
    checkpoint index alone. Returns ``({shard name: sha256}, proof sha256)``.
    """
    from .cost_streaming import (_local_checkpoint_shards,
                                 _read_streamed_model_identity_cache,
                                 stat_fingerprint, stat_fingerprint_reusable)

    root = Path(os.path.abspath(root))
    path = Path(cache_path)
    before = path.stat()
    raw = path.read_bytes()
    cached = json.loads(raw)
    after = path.stat()
    if file_stat_signature(before) != file_stat_signature(after):
        raise RuntimeError('streamed source identity cache changed while reading')
    if expected_sha256 is not None and bytes_sha256hex(raw) != expected_sha256:
        raise RuntimeError('streamed source identity cache differs from its declared SHA256')
    checked_cache, identity = _read_streamed_model_identity_cache(
        path, source_model=str(root))
    if cached != checked_cache:
        raise RuntimeError('streamed source identity cache changed while validating')
    checkpoint_map, indexed_shards = _local_checkpoint_shards(root)
    if (indexed_shards is None or checkpoint_map is None or
            identity.get('checkpoint_weight_map') != checkpoint_map):
        raise RuntimeError('streamed source proof differs from complete checkpoint index')
    fingerprints = cached.get('fingerprints')
    if not isinstance(fingerprints, list):
        raise RuntimeError('streamed source identity cache has no shard fingerprints')
    fp_by_path = {str(row.get('path')): row for row in fingerprints
                  if isinstance(row, dict)}
    shards = identity.get('shards')
    if (not isinstance(shards, list) or set(fp_by_path) != {
            str(row.get('path')) for row in shards} or
            set(fp_by_path) != {str(item.resolve()) for item in indexed_shards}):
        raise RuntimeError('streamed source identity cache shard coverage differs')
    expected_shards = ({name for name in source_files if name.endswith('.safetensors')}
                       if source_files is not None else
                       {Path(str(item)).name for item in indexed_shards})
    digests = {}
    for row in shards:
        source_path = Path(str(row['path']))
        name = source_path.name
        if (source_path.resolve() != (root / name).resolve() or name not in expected_shards
                or (source_files is not None and row.get('sha256') != source_files[name])):
            raise RuntimeError(f'{name}: streamed source SHA differs from canonical capture')
        fingerprint = fp_by_path[str(source_path)]
        # The same predicate the identity cache is built and validated
        # with: device may differ only where it is client-local (NFS) or
        # in dev mode (PQ #1363); anything else names another object.
        live = stat_fingerprint(str(source_path), live_stat(name))
        if not stat_fingerprint_reusable(live, fingerprint):
            raise RuntimeError(f'{name}: streamed source proof names another object')
        digests[name] = row['sha256']
    if set(digests) != expected_shards:
        raise RuntimeError('streamed source proof omits canonical capture shards')
    return digests, bytes_sha256hex(raw)


class CaptureSourceAuthentication:
    """One complete capture's source descriptors, not a weight/digest cache.

    Header inspection may open an unconsumed shard without hashing its payload.
    Tensor reads authenticate the same held object first, once in this owner's
    lifetime. Stat fences reject mutation/replacement; only SHA256 authenticates
    content. Ordinary source files must remain stable through their read leases.
    Construct through ``authenticate_selected_capture_source``.

    **Recording mode** (:meth:`recording`, PQ #1896) is a streamed canonical
    capture's owner. No sealed roster exists to compare against, so the first
    payload read of a file hashes all of it through the held descriptor and
    records the digest; a census that declares producer digests is compared
    there, and a mismatch refuses before the first tensor. The hash leaves the
    file's clean pages cached for subsequent tensor reads; kernel reclamation
    can still cause physical rereads. :meth:`release_retained_pages` drops a file's pages
    after its last consumer, and :meth:`authenticate_complete_source` hashes
    the files the capture never read.
    """

    def __init__(self, root, identity, producer_source, *, manifest_sha256,
                 resource_check=None, release_read_pages=False):
        self._setup(root, producer_source, manifest_sha256=manifest_sha256,
                    resource_check=resource_check, release_read_pages=release_read_pages,
                    source_files=dict(identity['source_files']))
        self._identity_json = json.dumps(identity, sort_keys=True, allow_nan=False)

    @classmethod
    def recording(cls, root, producer_source, *, binding_sha256=None, fingerprints=None,
                  resource_check=None, release_read_pages=False):
        """A streamed capture's owner: digests are recorded from its own reads.

        The roster is every file a capture identity names
        (:func:`capture_source_files`), plus any auxiliary file the census
        producer declares, which is authenticated but stays outside the
        identity as it always has. ``fingerprints`` (a capture chain prep's
        ``source_fingerprints``) holds every file this owner opens to the
        object the prep stat. ``binding_sha256`` names what the owner is bound
        to (the chain prep's seal), carried on its receipt.

        ``release_read_pages`` is the bounded capture's page policy: retained
        pages are dropped after a file's last consumer and at close. A payload
        hash never drops them block by block, because the tensor reads that
        follow it are the ones that need them.
        """
        self = cls.__new__(cls)
        self._setup(root, producer_source, manifest_sha256=binding_sha256,
                    resource_check=resource_check, release_read_pages=release_read_pages,
                    source_files=None)
        self._identity_json = None
        roster = frozenset(path.name for path in capture_source_files(self.root))
        if not roster:
            raise RuntimeError('calibration capture needs a complete local source checkpoint')
        if fingerprints is not None:
            if set(fingerprints) != roster:
                raise RuntimeError('the capture source roster changed since its prep')
            fingerprints = dict(fingerprints)
        self._roster = roster
        self._authorized = roster | frozenset(self._expected)
        self._fingerprints = fingerprints
        return self

    def _setup(self, root, producer_source, *, manifest_sha256, resource_check,
               release_read_pages, source_files):
        self.root = Path(os.path.abspath(root))
        self.is_recording = source_files is None
        self._source_files = source_files
        self._expected = {} if source_files is None else dict(source_files)
        declared = _producer_digests(producer_source)
        self._producer = dict(declared)
        self._derived_censuses = {}
        for name, digest in declared.items():
            if name in self._expected and self._expected[name] != digest:
                raise RuntimeError(f'capture source roster differs from census producer: {name}')
            self._expected[name] = digest
        _require_sha256_roster(self._expected, what='capture source roster')
        self._roster = None
        self._authorized = frozenset(self._expected)
        self._fingerprints = None
        self._released = set()
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
        if value.parent != self.root or value.name not in self._authorized:
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
                    state = dict(fd=fd, before=before, sha256=None, sha256_source=None,
                                 payload_reads=0,
                                 lock=threading.Lock())
                    self._check_file(name, state)
                    self._check_fingerprint(name, before)
                    self._files[name] = state
                except BaseException:
                    os.close(fd)
                    raise
            return name, self._files[name]

    def _check_fingerprint(self, name, observed):
        """Hold a recording owner's held object to the object its prep stat."""
        if self._fingerprints is None or name not in self._fingerprints:
            return
        from .cost_streaming import stat_fingerprint, stat_fingerprint_reusable
        live = stat_fingerprint(str((self.root/name).resolve()), observed)
        if not stat_fingerprint_reusable(live, self._fingerprints[name]):
            raise RuntimeError(f'source file changed since the capture prep stat it: {name}')

    def _check_file(self, name, state):
        try:
            current = (os.fstat(state['fd']), os.stat(self.root/name))
            if any(file_stat_signature(value) != file_stat_signature(state['before']) for value in current):
                raise RuntimeError(f'authenticated source changed during consumption: {name}')
        except OSError as exc:
            raise RuntimeError(f'authenticated source changed during consumption: {name}') from exc

    def require_unchanged(self):
        with self._lock:
            self._require_open()
            for name, state in self._files.items():
                self._check_file(name, state)

    def _authenticate(self, name, state, *, unconsumed=False):
        """Hash ``name`` once through its held descriptor; compare or record.

        A sealed owner compares with the sealed digest. A recording owner
        compares with a census producer's digest where one is declared and
        records the digest otherwise. Its payload hash keeps the pages the
        tensor reads need; ``unconsumed`` (a file no reader will consume)
        drops them block by block under the bounded page policy.
        """
        with state['lock']:
            self._check_file(name, state)
            if state['sha256'] is None:
                release = (self.release_read_pages if not self.is_recording
                           else unconsumed and self.release_read_pages)
                digest = sha256(self.root/name, file_descriptor=state['fd'],
                    resource_check=self.resource_check, release_read_pages=release)
                self._check_file(name, state)
                expected = self._expected.get(name)
                if self.is_recording:
                    if expected is not None and digest != expected:
                        raise RuntimeError(f'calibration source differs from census producer: {name}')
                elif digest != expected:
                    raise RuntimeError(f'calibration source content differs from sealed capture: {name}')
                state['sha256'] = digest
                state['sha256_source'] = 'fresh_descriptor_sha256'

    @property
    def adopted_identity_cache_sha256(self):
        """The SHA-256 of the identity cache this owner adopted, or ``None``."""
        return getattr(self, '_adopted_cache_sha256', None)

    def adopt_streamed_identity_cache(self, cache_path, *, expected_sha256=None):
        """Reuse the existing full-checkpoint SHA proof for held source objects.

        The existing cache validator checks the identity's content seal. We
        also check its complete index map, every six-field fingerprint, and
        each full-file SHA against the hash-bound canonical capture, then
        open and hold those exact objects through all future reads. The live
        runner's ``build_streamed_model_identity`` still checks its semantic
        config and executable weight map before qualification. A changed file
        (including a same-size edit with restored mtime) refuses; it is never
        silently rehashed under a manifest that omitted the source read.

        ``expected_sha256`` binds the proof file itself: the bytes read here
        must hash to it, checked before any held state changes. A caller that
        planned a proof by digest (a Stage A row, PQ #1497) passes it.
        """
        if self.is_recording:
            raise RuntimeError('a recording capture owner records digests from its own reads; '
                               'it adopts no identity proof')
        held = {}

        def live_stat(name):
            _, held[name] = self._file(self.root / name)
            return held[name]['before']

        # ``_expected`` agrees with ``_source_files`` on every roster name (the
        # constructor refuses otherwise), so the roster is the capture's own.
        digests, proof_sha256 = streamed_identity_proof_digests(
            self.root, cache_path, self._source_files, live_stat=live_stat,
            expected_sha256=expected_sha256)
        self.require_unchanged()
        for name, digest in digests.items():
            held[name]['sha256'] = digest
            held[name]['sha256_source'] = 'verified_streamed_identity_cache'
        self._adopted_cache_sha256 = proof_sha256
        return len(digests)

    def adopt_identity_proof_or_hash(self, cache_path, expected_sha256):
        """Adopt a planned identity proof, or leave every read to hash fresh.

        A Stage A row reads a few shards of a complete source. Hashing each
        one on first read idled the GPU reservation for 41% of a row
        (PQ #1497). The campaign already holds a full-file proof of every
        shard, so the row adopts it through
        :meth:`adopt_streamed_identity_cache`, with that method's own checks
        and nothing weaker. A proof that refuses -- a changed stat
        fingerprint, a proof file that differs from its declared digest, a
        missing file -- changes no held state: each payload read then hashes
        its shard through the held descriptor, as it did before this method
        existed, and the receipt records why. Integrity is the same either
        way; only where the digest came from differs. Returns the count of
        shards adopted, 0 on refusal.
        """
        try:
            adopted = self.adopt_streamed_identity_cache(
                cache_path, expected_sha256=expected_sha256)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            # KeyError/TypeError: a proof whose shard rows lack the fields
            # adoption reads is malformed, and refuses like any other.
            self._identity_proof_refusal = dict(
                path=str(cache_path), declared_sha256=expected_sha256,
                reason=f'{type(exc).__name__}: {exc}')
            print(f'[source] identity proof {cache_path} refused ({exc}); '
                  'every payload read hashes its shard fresh', flush=True)
            return 0
        print(f'[source] adopted {adopted} full-file SHA proofs from {cache_path}',
              flush=True)
        return adopted

    def admit_derived_census(self, census_path):
        """Let a census derived from this capture's source bind its roster.

        A derived capture covers units the canonical capture never ran (an
        MTP layer outside the streamed text forward) over the same source
        checkpoint. It inherits this owner's hash-bound source roster instead
        of re-hashing every shard, and its payload reads are still
        authenticated one shard at a time. The derived census must name the
        same model and declare the same producer source digests; its own
        digest is then the one other census :meth:`source_files` accepts.
        Returns that digest.
        """
        if self.is_recording:
            raise RuntimeError('a derived census binds a sealed capture roster, not a recording one')
        raw = Path(census_path).read_bytes()
        census = json.loads(raw)
        digest = bytes_sha256hex(raw)
        producer = ((census.get('expert_projection') or {}).get('producer') or {}).get('source') or {}
        if (not isinstance(census.get('model'), str) or
                Path(os.path.abspath(census['model'])) != self.root or
                _producer_digests(producer) != self._producer):
            raise RuntimeError('derived census names another source model or producer roster')
        with self._lock:
            self._require_open()
            self._derived_censuses[digest] = str(Path(os.path.abspath(census_path)))
        return digest

    def require_recording_roster(self, root, names, producer_digests):
        """A streamed capture's identity names this owner's source and producer."""
        if (not self.is_recording or Path(os.path.abspath(root)) != self.root or
                set(names) != self._roster or producer_digests != self._producer):
            raise RuntimeError('streamed capture source, roster or census producer differs '
                               'from its recording owner')

    def adopt_recorded_digests(self, digests):
        """Bind digests other readers of these objects recorded; nothing is read.

        A capture chain's join (PQ #1896) holds the digests its quanta
        recorded, each from the held descriptor it streamed a file through.
        Each file is opened and fenced here as any read would be, held to the
        prep's stat fingerprint, and compared with a declared producer digest.
        Returns the count bound.
        """
        if not self.is_recording:
            raise RuntimeError('only a recording owner binds recorded digests')
        _require_sha256_roster(digests, what='recorded capture source digests')
        for name, digest in sorted(digests.items()):
            _, state = self._file(self.root/name)
            with state['lock']:
                self._check_file(name, state)
                expected = self._expected.get(name)
                if expected is not None and digest != expected:
                    raise RuntimeError(f'calibration source differs from census producer: {name}')
                if state['sha256'] is not None and state['sha256'] != digest:
                    raise RuntimeError(f'recorded source digests disagree: {name}')
                if state['sha256'] is None:
                    state['sha256'] = digest
                    state['sha256_source'] = 'recorded_by_capture_reader'
        return len(digests)

    def release_retained_pages(self, keep=()):
        """Drop the retained pages of every recorded file not named in ``keep``.

        The bounded page policy of a recording owner (PQ #1896): a payload
        hash keeps a file's pages for the tensor reads that follow it, and the
        traversal calls this once a file has no later consumer. ``keep`` holds
        the files a later layer still reads. Advice is not proof of release;
        the capture guard stays final. Returns the names released.
        """
        if not self.is_recording or not self.release_read_pages:
            return ()
        keep = {self._name(path) for path in keep}
        with self._lock:
            self._require_open()
            ready = [(name, state) for name, state in sorted(self._files.items())
                     if state['sha256'] is not None
                     and name not in keep and name not in self._released]
        released = []
        for name, state in ready:
            with state['lock']:
                self._check_file(name, state)
                os.posix_fadvise(state['fd'], 0, 0, os.POSIX_FADV_DONTNEED)
            with self._lock:
                self._released.add(name)
            released.append(name)
        return tuple(released)

    def recorded_source_files(self):
        """The identity roster's recorded digests, for :func:`bind_capture_source`.

        Requires every roster file authenticated (``authenticate_complete_source``)
        and the source directory to still list exactly the roster this owner
        was built over: a file added or removed during the capture refuses.
        """
        if not self.is_recording:
            raise RuntimeError('only a recording owner has recorded source files')
        if frozenset(path.name for path in capture_source_files(self.root)) != self._roster:
            raise RuntimeError('the capture source roster changed during the capture')
        with self._lock:
            self.require_unchanged()
            missing = sorted(name for name in self._roster
                             if (self._files.get(name) or {}).get('sha256') is None)
            if missing:
                raise RuntimeError(f'capture source digests are incomplete: {missing[:8]}')
            return {name: self._files[name]['sha256'] for name in sorted(self._roster)}

    def source_files(self, root, census_digest, names, producer_digests):
        if self.is_recording:
            raise RuntimeError('a recording owner has no sealed source roster')
        canonical = json.loads(self._identity_json)
        if (Path(os.path.abspath(root)) != self.root or
                census_digest not in {canonical['census_sha256'], *self._derived_censuses} or
                names != set(self._source_files) or
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
        if self.is_recording:
            return self._recording_receipt()
        adopted = self.adopted_identity_cache_sha256 is not None
        verified = [{"name": name, "sha256": state['sha256'],
            "bytes_hashed": (state['before'].st_size if state['sha256_source'] ==
                             'fresh_descriptor_sha256' else 0),
            **({'proof_source': state['sha256_source']} if adopted else {}),
            "payload_reads": state['payload_reads']}
            for name, state in sorted(self._files.items()) if state['sha256'] is not None]
        return dict(schema='prismaquant.selected_source_authentication.v1',
            capture_manifest_sha256=self.manifest_sha256,
            census_sha256=json.loads(self._identity_json)['census_sha256'],
            authentication=('cached full-file SHA256 bound to held source descriptors'
                            if adopted else
                            'fresh SHA256 through held read-only source descriptors'),
            **({'streamed_identity_cache_sha256': self._adopted_cache_sha256}
               if adopted else {}),
            **({'streamed_identity_cache_refused': self._identity_proof_refusal}
               if getattr(self, '_identity_proof_refusal', None) else {}),
            **({'derived_census_sha256': sorted(self._derived_censuses)}
               if self._derived_censuses else {}),
            verified_files=verified,
            payload_bytes_hashed=sum(row['bytes_hashed'] for row in verified
                                     if row['name'].endswith('.safetensors')),
            metadata_only_shards=sorted(name for name, state in self._files.items()
                if name.endswith('.safetensors') and state['sha256'] is None))

    def _recording_receipt(self):
        """What a recording owner read and recorded, the data a chain join consumes."""
        verified = [{"name": name, "sha256": state['sha256'],
            "sha256_source": state['sha256_source'],
            "bytes_hashed": (state['before'].st_size if state['sha256_source'] ==
                             'fresh_descriptor_sha256' else 0),
            "payload_reads": state['payload_reads']}
            for name, state in sorted(self._files.items()) if state['sha256'] is not None]
        return dict(schema=RECORDING_RECEIPT_SCHEMA, binding_sha256=self.manifest_sha256,
            authentication='SHA256 recorded through the held read-only descriptors the '
                           'capture read its source through',
            producer_verified=sorted(name for name in self._producer
                                     if (self._files.get(name) or {}).get('sha256') is not None),
            verified_files=verified,
            payload_bytes_hashed=sum(row['bytes_hashed'] for row in verified
                                     if row['name'].endswith('.safetensors')),
            released_files=sorted(self._released),
            metadata_only_shards=sorted(name for name, state in self._files.items()
                if name.endswith('.safetensors') and state['sha256'] is None))

    def authenticate_complete_source(self):
        """Finish a complete-capture consumer's full source-byte proof.

        Selected consumers need only authenticate payload shards they read.
        A joint preparation promises the complete canonical capture identity,
        including auxiliary or MTP shards its streamed model never installs.
        Its first-use readers have already authenticated their shards through
        this owner; finish only the untouched files before publishing a
        prepared artifact. Every hash uses the same held descriptor and stat
        fences as a selected read, so a replaced or changed source refuses.

        A recording owner (PQ #1896) finishes the files its capture never
        read -- MTP sidecar shards, tokenizer and config files -- together on
        the IO engine, since no other reader wants those bytes, and checks
        them in name order so the first error is the serial loop's.
        """
        names = sorted(self._authorized)
        if not self.is_recording:
            for name in names:
                _, state = self._file(self.root / name)
                self._authenticate(name, state)
        else:
            states = {name: self._file(self.root / name)[1] for name in names}
            pending = [name for name in names if states[name]['sha256'] is None]
            with _on_io_engine({name: partial(self._authenticate, name, states[name],
                                              unconsumed=True)
                                for name in pending}) as finished:
                for name in pending:
                    finished(name)
        receipt = self.receipt()
        if {row['name'] for row in receipt['verified_files']} != set(self._authorized):
            raise RuntimeError('complete capture source authentication omitted a file')
        return receipt

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
                for name, state in self._files.items():
                    if (self.is_recording and self.release_read_pages
                            and name not in self._released):
                        # The bounded page policy's last word: nothing this
                        # owner retained outlives it. Advice only, so a
                        # failure to advise never leaks a descriptor.
                        try:
                            os.posix_fadvise(state['fd'], 0, 0, os.POSIX_FADV_DONTNEED)
                        except OSError:
                            pass
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


def _load_execution(policy, identity, output=None, *, identity_sha256=None):
    from .perturbed_x_cache import normalize_verified_activation_load
    policy = normalize_verified_activation_load(policy)
    if policy is None:
        return None
    if identity_sha256 is None:
        descriptor = dict(schema='prismaquant.capture_load_execution.v1', policy=policy,
                          capture_identity=identity)
        identity_sha256 = DIRECT_ASCII_LAX.sha256_streamed(descriptor)
    elif (not isinstance(identity_sha256, str) or len(identity_sha256) != 64 or
          any(c not in '0123456789abcdef' for c in identity_sha256)):
        raise ValueError('capture load execution needs a SHA256 identity')
    value = dict(schema='prismaquant.capture_load_execution.v1', policy=policy,
                 identity_sha256=identity_sha256,
                 loaded_entries=0, source_read_bytes=0, peak_buffer_bytes=0,
                 peak_archive_storage_bytes=0, live_buffer_bytes=0,
                 ordered_load_identities_sha256=bytes_sha256hex(b''))
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
    total['ordered_load_identities_sha256'] = hex_chain_sha256hex(
        total['ordered_load_identities_sha256'], partial['ordered_load_identities_sha256'])


def preflight_verified_capture_entries(root, entries, *, names, policy, census, max_rows):
    """Check the complete selected roster's file/geometry bounds before loading.

    The ``lstat`` calls go out together on the process's IO engine
    (``io_engine.ENGINE``): on a network mount one round trip per entry was a
    serial 7.3 s for a 864-unit GLM-5.3 row before its first encode
    (PQ #1654). The checks still run in name order on this thread, so the
    first failure raised is the one the serial loop raised.
    """
    from .io_engine import ENGINE
    from .perturbed_x_cache import activation_cache_filename, normalize_verified_activation_load
    policy = normalize_verified_activation_load(policy)
    if policy is None:
        raise ValueError('verified capture preflight requires an explicit load policy')
    names = list(names)
    expected = {name: str(Path('inputs') / activation_cache_filename(name)) for name in names}
    stats = {name: ENGINE.submit((Path(root)/expected[name]).lstat)
             for name in names if entries[name].get('path') == expected[name]}
    return _checked_preflight(names, stats, policy=policy, census=census, max_rows=max_rows)


def _checked_preflight(names, stats, *, policy, census, max_rows):
    """The serial loop's checks, in name order, over already-issued ``lstat`` calls."""
    import stat
    largest_file = largest_storage = 0
    for name in names:
        if name not in stats:
            raise RuntimeError(f'{name}: noncanonical capture artifact path')
        observed = stats[name].result()
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
    """Load one admitted entry and return it beside its own load receipt.

    ``execution`` folds the receipt here when it is given; a reader that
    loads entries out of order passes ``None`` and folds in name order."""
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
    if execution is not None:
        fold_load_receipt(execution, receipt)
    return payload, receipt


def fold_load_receipt(execution, receipt):
    """Accumulate one entry receipt into the run's load execution record.

    The ordered identity is a chain, so the fold order IS the receipt: a
    reader that loads out of order must still fold in the loaded-name order
    the serial path would have used. ``merge_load_execution`` cannot stand in
    for this, because it chains an already-chained partial.
    """
    execution['loaded_entries'] += 1
    execution['source_read_bytes'] += receipt['source_read_bytes']
    execution['peak_buffer_bytes'] = max(execution['peak_buffer_bytes'], receipt['file_bytes'])
    execution['peak_archive_storage_bytes'] = max(execution['peak_archive_storage_bytes'],
                                                  receipt['archive_storage_bytes'])
    execution['ordered_load_identities_sha256'] = hex_chain_sha256hex(
        execution['ordered_load_identities_sha256'], receipt['identity_sha256'])


def capture_entry_fingerprint(path):
    """The stat fingerprint of one regular, nonsymlink capture entry file."""
    from .cost_streaming import stat_fingerprint
    observed = Path(path).lstat()
    if not stat.S_ISREG(observed.st_mode):
        raise RuntimeError(f'{path}: capture entry must be a regular nonsymlink file')
    return stat_fingerprint(str(path), observed)


def _reverify_capture_entry(path, name, record, *, census, max_rows, execution,
                            resource_check=None, release_file_pages=False, file_stat=None):
    """Hash and validate one written entry; True when the read released its pages."""
    import torch
    if execution is None:
        if sha256(path) != record['sha256']:
            raise RuntimeError(f'{name}: capture artifact checksum mismatch')
        _validate_tensors(name, torch.load(path, map_location='cpu', weights_only=True),
                          census, max_rows)
        return False
    payload, _receipt = _verified_capture_entry(path, name, expected_sha256=record['sha256'],
        census=census, max_rows=max_rows, policy=execution['policy'],
        execution=execution, resource_check=resource_check,
        release_file_pages=release_file_pages, expected_stat=file_stat)
    del payload, _receipt
    return True


def _write_capture_entry(root, name, *, inputs, hessian, count, max_abs):
    """Write one unit's entry and hash it as it is written (PQ #1896).

    The digest is of the bytes the serializer handed the file
    (``SerializedEntryDigest``), the file is fsynced before it is published,
    and the stat fingerprint taken here is what the seal holds the entry to.
    Nothing reads the entry back: re-reading the 37 GB GLM-5.3 cache it had
    just written was the capture seal's whole final phase on s27. Returns
    ``(path, record, fingerprint)``.
    """
    from .perturbed_x_cache import (SerializedEntryDigest, activation_cache_filename,
                                    write_activation_cache_entry)
    digest = SerializedEntryDigest()
    path = write_activation_cache_entry(Path(root)/'inputs', name, inputs, source=SOURCE,
        durable=True, serialized_digest=digest, hessian=hessian, count=count, max_abs=max_abs)
    fingerprint = capture_entry_fingerprint(path)
    if fingerprint['size'] != digest.bytes:
        raise RuntimeError(f'{name}: capture entry differs from its serialized bytes')
    record = dict(path=str(Path('inputs')/activation_cache_filename(name)),
                  sha256=digest.hexdigest())
    return path, record, fingerprint


def _require_verified_entry(path, name, record, verified):
    """An entry this capture wrote or verified, still the object it was then."""
    from .cost_streaming import stat_fingerprint_reusable
    if (not isinstance(verified, dict) or verified.get('sha256') != record['sha256']
            or verified.get('path') != record['path']):
        raise RuntimeError(f'{name}: verified capture record differs from the journal')
    if not stat_fingerprint_reusable(capture_entry_fingerprint(path), verified.get('fingerprint')):
        raise RuntimeError(f'{name}: capture entry changed since its writer or quantum verified it')


def publish_capture(root, *, census_path, identity, acts=None, hessians=None,
                    counts=None, maxima=None, existing_entries=None,
                    release_file_pages=False, resource_check=None,
                    verified_load_policy=None, load_execution=None, verified=None,
                    source_files=None):
    """Seal a complete capture, journalling per-unit file receipts atomically.

    ``existing_entries`` seals a previously measured raw capture without another
    model forward. Its bytes receive exactly the ordinary writer's validation.

    ``verified`` maps a unit to the record its writer hashed as it wrote it,
    or a capture chain quantum or a resumed writer verified, and the stat
    fingerprint taken then (PQ #1885, #1896). Such an entry is held to that
    fingerprint instead of being read again. An entry this call writes is
    hashed as it is written and never read back.

    ``source_files`` is a streamed capture's recorded source roster (PQ
    #1896): ``identity`` is then the traversal identity, which keys the
    journal, and the manifest seals ``bind_capture_source(identity,
    source_files)``. Without it ``identity`` must already bind its source.
    """
    from .perturbed_x_cache import activation_cache_filename
    root = Path(root).resolve()
    census = json.loads(Path(census_path).read_text())
    if source_files is None:
        if 'source_files' not in identity:
            raise RuntimeError('a capture seals the source it read: its identity binds no source files')
        sealed_identity = identity
    else:
        sealed_identity = bind_capture_source(identity, source_files)
    execution = _load_execution(verified_load_policy, sealed_identity, load_execution)
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
            path, record, _fingerprint = _write_capture_entry(root, name, inputs=acts[name],
                hessian=hessians[name], count=counts[name], max_abs=maxima[name])
            file_stat = path.stat() if release_file_pages else None
        else:
            if record.get('path') != str(expected_path):
                raise RuntimeError(f'{name}: capture file is outside its canonical location')
            path = root/expected_path
            if verified is not None and name in verified:
                # Nothing is read, so there are no pages to release.
                _require_verified_entry(path, name, record, verified[name])
                loaded_verified = True
            else:
                file_stat = path.stat() if release_file_pages else None
                loaded_verified = _reverify_capture_entry(path, name, record, census=census,
                    max_rows=identity['max_act_rows'], execution=execution,
                    resource_check=resource_check, release_file_pages=release_file_pages,
                    file_stat=file_stat)
        if release_file_pages and not loaded_verified:
            from .perturbed_x_cache import release_activation_cache_file_pages
            release_activation_cache_file_pages(path, expected_stat=file_stat)
        if name not in completed:
            write_unit(journal,stage=STAGE,qname=name,identity_sha256=digest,state=record)
        records[name] = record
        if resource_check is not None:
            resource_check(f'after_capture_seal:{name}')
    manifest = dict(schema=SCHEMA,status='complete',identity=sealed_identity,entries=records)
    path = root/'capture_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('existing complete calibration capture changed')
    # The manifest's digest is of the bytes written, as every entry's is.
    raw = indent2_json_file_bytes(manifest)
    atomic_write_bytes(path, raw)
    return dict(path=str(path),sha256=bytes_sha256hex(raw))


class CaptureWriter:
    """Drain completed layers through the existing per-unit writer and journal.

    An interrupted traversal may leave valid unit entries, but no complete
    manifest. A retry recomputes the source forward and must match every entry
    it reuses. Completion additionally requires the actual initialization
    witness from that traversal, not only the census's expected descriptor.

    Each entry is hashed as it is written and the seal holds it to the stat
    fingerprint taken then; nothing this writer wrote, or verified on resume,
    is read again (PQ #1896). ``identity`` may be a streamed capture's
    traversal identity, which keys the journal; :meth:`finish` then binds the
    source digests the capture recorded.
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
        # Each entry this process wrote or replay-verified, with its stat
        # fingerprint: the seal and a chain join hold it to that (PQ #1896).
        self.verified = {}

    def write(self, *, acts, hessians, counts, maxima):
        from .perturbed_x_cache import activation_cache_filename
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
                fingerprint = capture_entry_fingerprint(path)
                _old_receipt = None
                if self.load_execution is None:
                    if sha256(path) != previous.get('sha256'):
                        raise RuntimeError(f'{name}: interrupted capture entry changed')
                    old = torch.load(path, map_location='cpu', weights_only=True)
                else:
                    old, _old_receipt = _verified_capture_entry(path, name, expected_sha256=previous.get('sha256'),
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
                    del old, old_x, old_h, _old_receipt
                if capture_entry_fingerprint(path) != fingerprint:
                    raise RuntimeError(f'{name}: interrupted capture entry changed while it was verified')
            else:
                path, record, fingerprint = _write_capture_entry(self.root, name,
                    inputs=acts[name], hessian=hessians[name], count=counts[name],
                    max_abs=maxima[name])
                file_stat = path.stat() if self.release_file_pages else None
            if self.release_file_pages and (self.load_execution is None or previous is None):
                from .perturbed_x_cache import release_activation_cache_file_pages
                release_activation_cache_file_pages(path, expected_stat=file_stat)
            if previous is None:
                write_unit(self.journal, stage=STAGE, qname=name,
                           identity_sha256=self.digest, state=record)
            self.records[name] = record
            self.verified[name] = dict(record, fingerprint=fingerprint)
            if self.resource_check is not None:
                self.resource_check(f'after_capture_write:{name}')

    def verify_entries(self, names):
        """This process's entry records, each with the fingerprint taken when it was written.

        A capture chain quantum (PQ #1885) hands these to the join, which
        holds each entry to its fingerprint instead of reading it. Each digest
        is of the bytes the serializer wrote (PQ #1896), so nothing is read
        back here either; a resumed entry's is the one its replay verified.
        """
        verified = {}
        for name in sorted(names):
            record = self.verified.get(name)
            if record is None:
                raise RuntimeError(f'{name}: this capture process wrote no entry to verify')
            path = self.root/record['path']
            _require_verified_entry(path, name, self.records[name], record)
            verified[name] = dict(record)
        return verified

    def finish(self, *, model_load_contract, verified=None, source_files=None):
        """Seal the capture; a traversal identity binds ``source_files`` here (PQ #1896)."""
        from prismaquant import validate_source_initialization_contract
        actual = validate_source_initialization_contract(model_load_contract)
        if actual != self.identity['model_load_contract']:
            raise RuntimeError('actual capture initialization differs from the census')
        held = dict(verified or {})
        for name, record in self.verified.items():
            if held.setdefault(name, record) != record:
                raise RuntimeError(f'{name}: two verified records name one capture entry')
        extra = {}
        if self.load_execution is not None:
            self.seal_load_execution = {}
            extra = dict(verified_load_policy=self.load_execution['policy'],
                         load_execution=self.seal_load_execution)
        # The journal's completed records too: a capture chain's units were
        # written by earlier processes (PQ #1885). After a whole traversal
        # this process's records already cover them.
        return publish_capture(self.root, census_path=self.census_path,
                               identity=self.identity,
                               existing_entries={**self.completed, **self.records},
                               release_file_pages=self.release_file_pages,
                               resource_check=self.resource_check, verified=held or None,
                               source_files=source_files, **extra)


def require_capture_contract(path, expected_sha256=None):
    """Validate a complete canonical capture before downstream preparation."""
    path = Path(path)
    raw = path.read_bytes()
    if expected_sha256 is not None and bytes_sha256hex(raw) != expected_sha256:
        raise RuntimeError('priced calibration capture manifest changed')
    manifest = json.loads(raw)
    return validate_capture_contract(manifest)


def _capture_manifest_stat(path):
    """Return the mutation fence for a regular, non-symlink manifest."""
    observed = Path(path).lstat()
    if not stat.S_ISREG(observed.st_mode):
        raise RuntimeError('canonical capture manifest must be a regular nonsymlink file')
    return file_stat_signature(observed)


def _freeze_capture_metadata(value):
    """Make the owner snapshot non-mutable without copying it per consumer."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_capture_metadata(item)
                                 for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_capture_metadata(item) for item in value)
    return value


class CaptureMetadataOwner:
    """One hash-bound, bounded capture-manifest snapshot for selected rows.

    The owner is deliberately separate from resident X/H ownership.  It keeps
    the already validated metadata for a run that consumes many selected
    units, while a strict same-path stat fence detects any replacement or
    mutation before each reuse.  On a fence change it rehashes the path for
    evidence and refuses; a changed pathname is never silently rebound.
    """

    def __init__(self, path, *, expected_identity, expected_sha256):
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or
                any(c not in '0123456789abcdef' for c in expected_sha256)):
            raise RuntimeError('capture metadata owner requires a hash-bound complete capture')
        self.path = Path(path).resolve()
        before = _capture_manifest_stat(self.path)
        raw = self.path.read_bytes()
        after = _capture_manifest_stat(self.path)
        if before != after:
            raise RuntimeError('canonical capture manifest changed while its metadata was read')
        if len(raw) > MAX_CAPTURE_METADATA_BYTES:
            raise RuntimeError('canonical capture manifest exceeds bounded metadata budget')
        self.sha256 = bytes_sha256hex(raw)
        if self.sha256 != expected_sha256:
            raise RuntimeError('priced calibration capture manifest changed')
        try:
            manifest = validate_capture_contract(json.loads(raw))
        except (TypeError, ValueError) as error:
            raise RuntimeError('canonical capture manifest is not valid JSON') from error
        expected = DIRECT_ASCII_STRICT.text(expected_identity)
        identity = DIRECT_ASCII_STRICT.text(manifest['identity'])
        if identity != expected:
            raise RuntimeError('calibration capture identity, completeness or scope mismatch')
        self._stat = after
        # No warm reader may alter an entry path, checksum, identity or unit
        # geometry in the retained object.  The conversion happens once at
        # ownership, rather than copying/serializing the large manifest per
        # selected singleton.
        self._manifest = _freeze_capture_metadata(manifest)
        self._identity_json = identity
        self._execution_digests = {}

    def _assert_unchanged(self, path):
        candidate = Path(path).resolve()
        if candidate != self.path:
            raise RuntimeError('capture metadata owner path differs from requested capture')
        observed = _capture_manifest_stat(candidate)
        if observed != self._stat:
            # The rehash is intentionally not an admission mechanism: a
            # replacement with identical bytes still violates the held-path
            # mutation fence.  It tells a caller whether content changed while
            # preserving that fail-closed rule.
            changed = bytes_sha256hex(candidate.read_bytes()) != self.sha256
            raise RuntimeError('canonical capture manifest metadata changed'
                               + (' and content differs' if changed else ''))

    def open(self, path):
        self._assert_unchanged(path)
        return self._manifest

    def load_execution(self, policy, output=None):
        from .perturbed_x_cache import normalize_verified_activation_load
        policy = normalize_verified_activation_load(policy)
        if policy is None:
            return None
        policy_json = DIRECT_ASCII_STRICT.text(policy)
        digest = self._execution_digests.get(policy_json)
        if digest is None:
            # This is byte-for-byte the old sorted JSON descriptor, assembled
            # from the sealed identity snapshot instead of reserializing it
            # for every singleton prefetch.
            raw = ('{"capture_identity":' + self._identity_json + ',"policy":' +
                   policy_json + ',"schema":"prismaquant.capture_load_execution.v1"}').encode()
            digest = bytes_sha256hex(raw)
            if len(self._execution_digests) >= MAX_CAPTURE_EXECUTION_POLICIES:
                self._execution_digests.pop(next(iter(self._execution_digests)))
            self._execution_digests[policy_json] = digest
        return _load_execution(policy, None, output, identity_sha256=digest)


def open_capture_metadata(path, *, expected_identity, expected_sha256):
    """Create the explicit metadata owner required for warm selected reuse."""
    return CaptureMetadataOwner(path, expected_identity=expected_identity,
                                expected_sha256=expected_sha256)


def validate_capture_contract(manifest):
    """Validate the canonical contract on an already owned metadata snapshot."""
    from prismaquant import validate_source_initialization_contract
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


def open_hessian_reference(path):
    """Reuse the producer's bounded reader under the full canonical contract.

    No new H storage or residency cache is created. The returned owner retains
    only metadata; every mapping value access authenticates one existing input
    file and its committed H. The caller must close this owner.
    """
    try:
        from tessera.hessian_capture import ReferenceHessians
    except ImportError as error:
        raise RuntimeError('canonical Hessian references require the reviewed Tessera reference reader') from error
    from .tessera_reuse_authority import CANONICAL_CAPTURE
    collection = str(path).endswith('.collection.references.json')
    if collection:
        try:
            from tessera.hessian_capture import ReferenceHessianCollection
        except ImportError as error:
            raise RuntimeError('Hessian reference collections require the reviewed Tessera collection reader') from error
        owner = ReferenceHessianCollection(path, canonical_capture=CANONICAL_CAPTURE)
    else:
        owner = ReferenceHessians(path, canonical_capture=CANONICAL_CAPTURE)
    try:
        if collection:
            # The collection reader proves each disjoint v1 child and holds
            # their descriptors. Retain PrismaQuant's stronger source/runtime
            # capture gate on every child's canonical metadata as well.
            for child in owner.binding()['references']:
                with ReferenceHessians(child['path'], canonical_capture=CANONICAL_CAPTURE) as reference:
                    if (reference.document_sha256 != child['sha256'] or
                            reference.binding() != child['binding']):
                        raise RuntimeError('Hessian collection child changed during capture validation')
                    validate_capture_contract(reference.canonical_manifest())
            owner.require_current()
        else:
            validate_capture_contract(owner.canonical_manifest())
        return owner
    except BaseException:
        owner.close()
        raise


def write_hessian_reference(path, descriptor):
    """Publish a metadata-only handoff after both owners accept its commitments."""
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    try:
        _json(temporary, descriptor)
        with open_hessian_reference(temporary) as owner:
            digest = owner.descriptor['capture_sha256']
        os.replace(temporary, path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def canonical_hessian_reference_descriptor(*, hessians, counts, provenance,
        canonical_capture, census_path, load_policy, identities=None):
    """Commit resident row H to the original capture without copying its bytes.

    ``identities`` (``{unit: tensor_identity}``) are receipts a caller already
    sealed from these same resident tensors (the campaign identity hold);
    they are taken as they are, with their dtype/shape checked against the
    tensor, and only the units without one are digested here.  Every unit
    named must be a resident H of this descriptor.
    """
    try:
        from tessera.cached_unit import tensor_identity
        from tessera.hessian_capture import REFERENCE_SCHEMA, capture_sha256_from_units
    except ImportError as error:
        raise RuntimeError('canonical Hessian references require the reviewed Tessera reference reader') from error
    if not isinstance(canonical_capture, dict) or set(canonical_capture) != {'path','sha256'}:
        raise RuntimeError('Hessian reference needs a hash-bound canonical capture')
    manifest = require_capture_contract(canonical_capture['path'], canonical_capture['sha256'])
    census_path = Path(census_path).resolve()
    census_digest = sha256(census_path)
    if census_digest != manifest['identity']['census_sha256']:
        raise RuntimeError('Hessian reference census differs from the complete capture')
    sealed = {} if identities is None else dict(identities)
    resident = {name for name, value in hessians.items() if value is not None}
    if set(sealed) - resident:
        raise RuntimeError('Hessian reference identities name units without resident H: '
                           + ', '.join(sorted(set(sealed) - resident)))
    identities = {}
    for name, value in hessians.items():
        if value is None:
            continue
        known = sealed.get(name)
        if known is None:
            if value.device.type == 'meta':
                # A stand-in carries geometry only. Digesting it would commit
                # to bytes nobody read, so a stand-in needs a sealed receipt.
                raise RuntimeError(f'Hessian reference for {name} has neither a sealed '
                                   'receipt nor resident bytes')
            identities[name] = tensor_identity(value)
            continue
        if (not isinstance(known, dict) or set(known) != {'algorithm','dtype','shape','sha256'}
                or known['dtype'] != str(value.dtype) or list(known['shape']) != list(value.shape)
                or not isinstance(known['sha256'], str) or len(known['sha256']) != 64):
            raise RuntimeError(f'Hessian reference identity for {name} does not describe its resident H')
        identities[name] = dict(known, shape=list(known['shape']))
    digest = capture_sha256_from_units(provenance, {n:v['sha256'] for n,v in identities.items()})
    return dict(schema=REFERENCE_SCHEMA,
        canonical_capture=dict(path=str(Path(canonical_capture['path']).resolve()),
                               sha256=canonical_capture['sha256']),
        census=dict(path=str(census_path),sha256=census_digest),
        provenance=dict(provenance),counts=dict(counts),hessians=identities,
        capture_sha256=digest,rows=[dict(units=sorted(identities),capture_sha256=digest)],
        load_policy=dict(load_policy))


def hessian_reference_binding(canonical_capture_sha256, census_sha256):
    """The optional source binding carried unchanged from prices to export."""
    from tessera.hessian_capture import BINDING_SCHEMA, normalize_reference_binding
    return normalize_reference_binding(dict(schema=BINDING_SCHEMA,
        canonical_capture_sha256=canonical_capture_sha256,census_sha256=census_sha256))


def merge_hessian_reference_descriptors(descriptors):
    """Union accepted metadata snapshots without reading or retaining any H."""
    import copy
    from tessera.hessian_capture import capture_sha256_from_units
    result = None
    for descriptor in descriptors:
        if result is None:
            result = copy.deepcopy(descriptor)
            continue
        for key in ('schema','canonical_capture','census','provenance','counts','load_policy'):
            if result[key] != descriptor[key]:
                raise RuntimeError(f'Hessian reference union differs at {key}')
        overlap = result['hessians'].keys() & descriptor['hessians'].keys()
        if overlap:
            raise RuntimeError('Hessian reference units occur in multiple rows: '+', '.join(sorted(overlap)[:4]))
        result['hessians'].update(copy.deepcopy(descriptor['hessians']))
        result['rows'].extend(copy.deepcopy(descriptor['rows']))
    if result is None:
        raise RuntimeError('Hessian reference union needs at least one accepted row')
    result['hessians'] = dict(sorted(result['hessians'].items()))
    result['capture_sha256'] = capture_sha256_from_units(result['provenance'],
        {n:v['sha256'] for n,v in result['hessians'].items()})
    return result


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
            canonical['model_load_contract']['schema'] not in SELECTED_SOURCE_LOAD_SCHEMAS or
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


def require_automatic_capture_source_recording():
    """The existing source owner owns automatic capture recording admission.

    Separated from explicit descriptor-owner construction so caller policy
    cannot turn an observed digest or lifetime/stat fence into qualification.
    """
    return None


def record_capture_source(census_path, *, model, binding_sha256=None, fingerprints=None,
                          resource_check=None, release_read_pages=False):
    """The recording source owner of a streamed capture of ``census_path`` (PQ #1896).

    The streamed runner reads ``model`` through this owner, so every file the
    capture consumes is hashed once, through the held descriptor its tensors
    are read through, before its first tensor reaches the capture. The census
    must name the same source: a capture that streams one checkpoint and
    seals another's digests would be a lie the manifest tells downstream.
    """
    census = json.loads(Path(census_path).read_text())
    if (not isinstance(census.get('model'), str) or
            Path(os.path.abspath(census['model'])) != Path(os.path.abspath(model))):
        raise RuntimeError('a streamed capture reads the source its census names')
    producer = ((census.get('expert_projection') or {}).get('producer') or {}).get('source') or {}
    return CaptureSourceAuthentication.recording(model, producer,
        binding_sha256=binding_sha256, fingerprints=fingerprints,
        resource_check=resource_check, release_read_pages=release_read_pages)


def capture_read_threads() -> int:
    """Reader count for the verified capture prefetch.

    ``PRISMAQUANT_CAPTURE_READ_THREADS`` overrides; 1 (the default) restores
    the byte-identical serial read. The layer-weight gather spells its own
    count the same way in ``layer_streaming.layer_read_threads``; this is a
    separate stage with a separate working set, so it keeps a separate name.
    """
    raw = str(os.environ.get('PRISMAQUANT_CAPTURE_READ_THREADS', '')).strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 1


class _ConcurrentReservation:
    """Charge every concurrent reader's live future allocation to each check.

    A memory guard reads absolute residency and adds ONE caller's future
    allocation. Under N readers the peak is the sum of the live reservations,
    so each call presents that sum; a reader that has not yet reserved
    contributes nothing, and a refusal leaves the caller's previous
    reservation in place. Serialising the calls also gives the guard's own
    baseline/peak bookkeeping a single writer.
    """

    def __init__(self, check):
        self._check = check
        self._lock = threading.Lock()
        self._live = {}

    def check(self, label, *, reserve_bytes=0, reserve_device_bytes=0):
        if self._check is None:
            return None
        key = threading.get_ident()
        with self._lock:
            previous = self._live.get(key, (0, 0))
            self._live[key] = (reserve_bytes, reserve_device_bytes)
            try:
                return reserve_allocation(
                    self._check, label,
                    cpu_bytes=sum(pair[0] for pair in self._live.values()),
                    device_bytes=sum(pair[1] for pair in self._live.values()))
            except BaseException:
                self._live[key] = previous
                raise

    def release(self):
        with self._lock:
            self._live.pop(threading.get_ident(), None)


def _capture_entry_artifact(path, manifest, name):
    from .perturbed_x_cache import activation_cache_filename
    relative = str(Path('inputs') / activation_cache_filename(name))
    if manifest['entries'][name].get('path') != relative:
        raise RuntimeError(f'{name}: noncanonical capture artifact path')
    return path.parent/relative


def prefetch_capture(path, *, expected_identity=None, census, names, device,
                     expected_sha256=None, resource_check=None,
                     release_file_pages=False, verified_load_policy=None,
                     load_execution=None, metadata_owner=None):
    """Verify selected files and make all selected X/H resident before encoding."""
    import torch
    from .perturbed_x_cache import activation_cache_filename
    path = Path(path)
    if metadata_owner is None:
        if expected_identity is None:
            raise TypeError('prefetch needs an expected identity without a capture metadata owner')
        execution = _load_execution(verified_load_policy, expected_identity, load_execution)
        digest = sha256(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError('priced calibration capture manifest changed')
        manifest = require_capture_contract(path, expected_sha256=expected_sha256)
    else:
        if not isinstance(metadata_owner, CaptureMetadataOwner):
            raise TypeError('prefetch metadata owner has the wrong type')
        if expected_identity is not None:
            raise TypeError('prefetch metadata owner supplies its sealed identity')
        if expected_sha256 is not None and expected_sha256 != metadata_owner.sha256:
            raise RuntimeError('prefetch metadata owner SHA256 differs from requested capture')
        manifest = metadata_owner.open(path)
        expected_identity = manifest['identity']
        digest = metadata_owner.sha256
        execution = metadata_owner.load_execution(verified_load_policy, load_execution)
    names = sorted(names)
    if (manifest.get('schema') != SCHEMA or manifest.get('status') != 'complete' or
            manifest.get('identity') != expected_identity or
            set(manifest.get('entries',{})) != set(expected_identity['units']) or
            not set(names) <= set(expected_identity['units'])):
        raise RuntimeError('calibration capture identity, completeness or scope mismatch')
    if execution is not None:
        preflight_verified_capture_entries(path.parent, manifest['entries'], names=names,
            policy=execution['policy'], census=census, max_rows=expected_identity['max_act_rows'])
        threads = capture_read_threads()
        if threads > 1:
            return _parallel_prefetch_capture(path, manifest=manifest,
                expected_identity=expected_identity, census=census, names=names, device=device,
                digest=digest, execution=execution, resource_check=resource_check,
                release_file_pages=release_file_pages, threads=threads)
    acts, hessians, counts, maxima = {}, {}, {}, {}
    payload = x = h = _entry_receipt = None
    try:
        for name in names:
            record = manifest['entries'][name]
            relative = str(Path('inputs') / activation_cache_filename(name))
            if record.get('path') != relative:
                raise RuntimeError(f'{name}: noncanonical capture artifact path')
            artifact = path.parent/relative
            file_stat = artifact.stat() if release_file_pages else None
            # ONE LOADED UNIT, TWO BUDGETS, and the refusal comes before either
            # allocation. The serialized payload this process is about to hold
            # is cgroup-accounted; the X and H the call moves to ``device`` are
            # charged to the device envelope when that device is the GPU (a CPU
            # run moves nothing). The guard used to be handed their sum through
            # ``reserve_bytes``, which charged device residency to the CPU cap
            # the kernel enforces -- 80 GiB of it against a 21 GiB cap.
            storage_bytes = _capture_storage_bytes(
                name, census, expected_identity['max_act_rows'])
            cuda = str(device).startswith('cuda')
            reserve_allocation(
                resource_check, f'before_capture_prefetch:{name}',
                # ONE loaded unit is TWO allocations when the device is the GPU
                # -- the serialized payload in this process, and the X/H moved
                # onto the device -- and ONE when it is not: `_validate_tensors`
                # hands back the payload's own tensors and ``to`` on the same
                # device and dtype returns the same tensor, so the CPU arm's
                # live footprint is the payload alone. The old single number
                # charged 2x on both arms; the second copy only exists on one.
                cpu_bytes=storage_bytes,
                device_bytes=storage_bytes if cuda else 0)
            if execution is None:
                if sha256(artifact, resource_check=resource_check,
                          release_read_pages=release_file_pages) != record.get('sha256'):
                    raise RuntimeError(f'{name}: capture artifact checksum mismatch')
                payload = torch.load(artifact,map_location='cpu',weights_only=True)
            else:
                payload, _entry_receipt = _verified_capture_entry(artifact, name, expected_sha256=record.get('sha256'),
                    census=census, max_rows=expected_identity['max_act_rows'], policy=execution['policy'],
                    execution=execution, resource_check=resource_check,
                    release_file_pages=release_file_pages, expected_stat=file_stat)
            x,h = _validate_tensors(name,payload,census,expected_identity['max_act_rows'],
                                    check_finite=execution is None)
            acts[name],hessians[name] = x.to(device),h.to(device)
            counts[name],maxima[name] = payload['count'],payload['max_abs']
            if release_file_pages:
                from .perturbed_x_cache import release_activation_cache_file_pages
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize(device)
            del payload, x, h, _entry_receipt
            payload = x = h = _entry_receipt = None
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
            payload = x = h = _entry_receipt = None
        raise
    resident = sum(t.numel()*t.element_size() for t in (*acts.values(),*hessians.values()))
    print(f'[campaign] calibration prefetched: {len(names)} units, {resident} resident bytes, 0 misses',flush=True)
    return (acts,hessians,counts,maxima),dict(path=str(path.resolve()),sha256=digest)


def _parallel_prefetch_capture(path, *, manifest, expected_identity, census, names, device,
                               digest, execution, resource_check, release_file_pages, threads):
    """Read and verify entries on N readers; consume them in name order.

    Every entry passes through the same verified owner the serial path uses,
    so the per-entry receipt, the payload checks and every failure mode are
    unchanged. What differs is only that N entries are in flight at once:

    * the ordered identity chain is folded by the single consumer in sorted
      name order, which is the order the serial reader folded it in;
    * the per-unit ``before_capture_prefetch`` / ``after_capture_prefetch``
      calls stay in that same order because the consumer makes them;
    * every guard call, inner and outer, presents the SUM of the live
      concurrent reservations, because N buffers can be admitted at once.

    The window lives on the CONSUMER, not on the readers: at most ``threads``
    entries are ever submitted, and the next one is submitted only once an
    entry has been consumed. A reader therefore never waits on anything, and
    the entry the consumer is about to want is always already running. The
    earlier shape -- readers taking a semaphore permit the consumer returned --
    deadlocks, because permits are granted in wakeup order rather than name
    order, so workers can run ahead while the consumer's own next entry is
    still waiting for a permit that only the consumer can release.

    The transfer stays on the consumer thread and the default stream. The
    source tensors are pageable, so ``Tensor.to`` is host-synchronous and a
    side stream would not overlap anything without pinned staging.
    """
    import torch
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    max_rows = expected_identity['max_act_rows']
    guard = _ConcurrentReservation(resource_check)

    def read(name):
        artifact = _capture_entry_artifact(path, manifest, name)
        file_stat = artifact.stat() if release_file_pages else None
        try:
            return _verified_capture_entry(artifact, name,
                expected_sha256=manifest['entries'][name].get('sha256'), census=census,
                max_rows=max_rows, policy=execution['policy'], execution=None,
                resource_check=None if resource_check is None else guard.check,
                release_file_pages=release_file_pages, expected_stat=file_stat)
        finally:
            guard.release()

    acts, hessians, counts, maxima = {}, {}, {}, {}
    payload = x = h = None
    window = deque()
    submitted = 0
    pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix='capture-read')
    try:
        while submitted < len(names) and len(window) < threads:
            window.append(pool.submit(read, names[submitted]))
            submitted += 1
        for name in names:
            payload, receipt = window.popleft().result()
            fold_load_receipt(execution, receipt)
            storage_bytes = _capture_storage_bytes(name, census, max_rows)
            cuda = str(device).startswith('cuda')
            guard.check(
                f'before_capture_prefetch:{name}',
                reserve_bytes=storage_bytes,
                reserve_device_bytes=storage_bytes if cuda else 0)
            x, h = _validate_tensors(name, payload, census, max_rows, check_finite=False)
            acts[name], hessians[name] = x.to(device), h.to(device)
            counts[name], maxima[name] = payload['count'], payload['max_abs']
            if release_file_pages and str(device).startswith('cuda'):
                torch.cuda.synchronize(device)
            del payload, x, h
            payload = x = h = None
            if resource_check is not None:
                guard.check(f'after_capture_prefetch:{name}')
            if submitted < len(names):
                window.append(pool.submit(read, names[submitted]))
                submitted += 1
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
    except BaseException:
        acts.clear()
        hessians.clear()
        payload = x = h = None
        raise
    finally:
        # An abandoned reader owns an admitted buffer; none outlives this call.
        pool.shutdown(wait=True, cancel_futures=True)
        for pending in window:
            if pending.cancelled() or pending.exception() is not None:
                continue
            pending.result()[0].clear()
        window.clear()
        guard.release()
    resident = sum(t.numel()*t.element_size() for t in (*acts.values(),*hessians.values()))
    print(f'[campaign] calibration prefetched: {len(names)} units, {resident} resident bytes, '
          f'0 misses, {min(threads, len(names))} active readers '
          f'({threads} configured maximum)',flush=True)
    return (acts,hessians,counts,maxima),dict(path=str(path.resolve()),sha256=digest)
