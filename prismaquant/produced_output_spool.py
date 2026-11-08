"""Bounded same-box writer copies, published through the existing IO engine.

Application ceilings are process-owned bounds, not persistent capacity or fleet
reservations. Completion means verified durable destination and batch manifest.
Reader holds and owned-copy completion precede release and deferred unlinks.
"""
from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import os
from pathlib import Path
import shutil
import threading

from . import io_engine
from .staged_manifest import open_directory
from .stage_a_produced_output import (MAX_GROUPS, MAX_GROUP_ENTRIES, MAX_PAYLOAD_BYTES,
    produced_file_identity, _limit, produced_identifier)

ARTIFACT_CLASSES = ('payload', 'checkpoint')
ENTRY_HEADER_ENVELOPE_BYTES = 65536


def plane_partitions(*, n_rows, probe_microbatch):
    if type(n_rows) is not int or n_rows <= 0:
        raise ValueError('plane partitions need a positive int row count')
    if type(probe_microbatch) is not int or probe_microbatch < 0:
        raise ValueError('plane partitions need a nonnegative int probe_microbatch')
    batch_rows = min(probe_microbatch or n_rows, n_rows)
    return batch_rows, list(range(0, n_rows, batch_rows))


def two_plane_window_bytes(*, n_probes, n_batches, group_size, group_ceiling=None,
                           max_entry_tensor_bytes=None):
    for value in (n_probes, n_batches, group_size):
        if type(value) is not int or value <= 0:
            raise ValueError('two-plane geometry needs positive integers')
    if group_ceiling is None:
        if type(max_entry_tensor_bytes) is not int or max_entry_tensor_bytes <= 0:
            raise ValueError('two-plane window needs the per-entry tensor bound')
        def group_ceiling(entries):
            return entries * (max_entry_tensor_bytes + ENTRY_HEADER_ENVELOPE_BYTES)
    full, remainder = divmod(n_batches, group_size)
    return 2 * n_probes * (full * group_ceiling(group_size) +
                           (group_ceiling(remainder) if remainder else 0))


class ProducedOutputSpoolRefused(RuntimeError):
    pass


class ProducedExportRefused(ProducedOutputSpoolRefused):
    pass


class ProducedWindowRefused(ProducedOutputSpoolRefused):
    pass


class ProducedOutputWaitCancelled(RuntimeError):
    pass


class ProducedOutputSpool:
    def __init__(self, publication, *, root, max_bytes):
        self.publication = publication
        self.max_bytes = _limit(max_bytes, MAX_PAYLOAD_BYTES)
        base = Path(root)
        # The operator provisions capacity and this host-local root. A job may
        # create only its own launch namespace, never a new shared cache.
        if (not base.is_absolute() or not base.is_dir() or base.resolve() != base
                or str(base).startswith('/mnt/shared')):
            raise ProducedOutputSpoolRefused('explicit provisioned host-local spool root required')
        job_component = produced_identifier(publication.producer['job_id'].replace('ƒ', 'f', 1))
        namespace = base / job_component
        namespace.mkdir(mode=0o700, exist_ok=True)
        if namespace.is_symlink() or namespace.resolve() != namespace:
            raise ProducedOutputSpoolRefused('job scratch namespace changed')
        self.root = namespace / ('produced-' + publication.producer['launch_sha256'])
        self.root.mkdir(mode=0o700, exist_ok=False)
        self._root_identity = (self.root.stat().st_dev, self.root.stat().st_ino)
        self._cond = threading.Condition(threading.RLock())
        self._groups, self._local, self._durable_progress = {}, {}, []
        self.cancelled = None
        self._cancelled = False
        self.refusals, self.evictions = [], []

    @classmethod
    def from_publication(cls, publication):
        config = publication.declaration['spool']
        return None if config is None else cls(publication, **config)

    def reserve(self, batch_id, ceiling_bytes, *, wait=True):
        produced_identifier(batch_id); _limit(ceiling_bytes, self.max_bytes)
        with self._cond:
            self._check_root()
            if batch_id in self._groups:
                prior = self._groups[batch_id]
                if prior['ceiling_bytes'] != ceiling_bytes or prior['released']:
                    raise ProducedOutputSpoolRefused('changed/released reservation')
                return prior['directory']
            if len(self._groups) >= MAX_GROUPS:
                raise ProducedWindowRefused('group roster exceeds bound')
            while self._held_bytes() + ceiling_bytes > self.max_bytes:
                self._check_failures()
                if self._make_room():
                    continue
                live = [g for g in self._groups.values() if g['future'] is not None and not g['durable']]
                if not wait or not live:
                    raise ProducedWindowRefused('bounded local window has no releasable completed group')
                self._raise_if_cancelled()
                self._cond.wait()
            free = shutil.disk_usage(self.root).free
            if free < ceiling_bytes:
                raise ProducedWindowRefused('local filesystem lacks declared headroom')
            directory = self.root / batch_id
            directory.mkdir(mode=0o700)
            self._groups[batch_id] = {'directory': directory, 'ceiling_bytes': ceiling_bytes,
                'identity': (directory.stat().st_dev, directory.stat().st_ino),
                'references': [], 'future': None, 'durable': False, 'released': False,
                'reading': 0, 'retired': set(), 'read_back': False, 'failure': None,
                'binding': None, 'destinations': {}}
            return directory

    def _held_bytes(self):
        return sum(g['ceiling_bytes'] for g in self._groups.values() if not g['released'])

    def _check_root(self):
        directory = open_directory(str(self.root))
        try:
            info = os.fstat(directory)
            if (info.st_dev, info.st_ino) != self._root_identity:
                raise ProducedOutputSpoolRefused('owned spool root changed')
        finally:
            os.close(directory)

    def directory(self, batch_id):
        return self.writer_directory(batch_id)[0]

    def writer_directory(self, batch_id):
        with self._cond:
            self._check_root()
            group = self._groups[batch_id]
            descriptor = open_directory(str(group['directory']))
            try:
                info = os.fstat(descriptor)
                if (info.st_dev, info.st_ino) != group['identity']:
                    raise ProducedOutputSpoolRefused('owned group writer directory changed')
                return group['directory'], group['identity']
            finally:
                os.close(descriptor)

    def record(self, batch_id, local_reference, canonical_directory, *, canonical_directory_identity,
               artifact_class='payload', read_back=False):
        if artifact_class not in ARTIFACT_CLASSES or type(read_back) is not bool:
            raise ProducedOutputSpoolRefused('invalid artifact class/readback policy')
        canonical = dataclasses.replace(local_reference,
            path=str(Path(canonical_directory) / Path(local_reference.path).name))
        with self._cond:
            self._check_root()
            group = self._groups[batch_id]
            parent = str(Path(canonical.path).parent)
            destination_identity = tuple(canonical_directory_identity)
            if group['destinations'].get(parent, destination_identity) != destination_identity:
                raise ProducedOutputSpoolRefused('canonical writer directory identity changed')
            if (group['future'] is not None or len(group['references']) >= MAX_GROUP_ENTRIES
                    or Path(local_reference.path).parent != group['directory']
                    or not self.publication.contains(canonical.path)
                    or canonical.path in self._local):
                raise ProducedOutputSpoolRefused('foreign, duplicate or completed local entry')
            identity = produced_file_identity(local_reference.path)
            writer = getattr(local_reference, 'writer_file_identity', None)
            if (writer is None or list(writer) != identity or identity[2] != local_reference.file_bytes or
                    sum(row[1].file_bytes for row in group['references']) + canonical.file_bytes > group['ceiling_bytes']):
                raise ProducedOutputSpoolRefused('writer exceeded local prewrite ceiling')
            group['references'].append((local_reference, canonical, artifact_class, identity))
            group['destinations'][parent] = destination_identity
            group['read_back'] |= read_back
            self._local[canonical.path] = (batch_id, local_reference)
        return canonical

    def submit(self, batch_id, *, lifetime='retain'):
        with self._cond:
            group = self._groups[batch_id]
            if group['future'] is not None:
                return
            if not group['references']:
                raise ProducedOutputSpoolRefused('cannot submit an empty group')
            group['future'] = io_engine.ENGINE.submit(self._publish, batch_id, lifetime)
            group['future'].add_done_callback(lambda future: self._complete(batch_id, future))

    def _publish(self, batch_id, lifetime):
        group = self._groups[batch_id]
        descriptors = []
        for local, canonical, artifact_class, identity in group['references']:
            identity = io_engine.publish_copy(local.path, canonical.path, length=canonical.file_bytes,
                sha256=canonical.sha256, source_identity=identity,
                destination_directory_identity=group['destinations'][str(Path(canonical.path).parent)])
            descriptors.append(self.publication.descriptor_for(canonical,
                producer_generation=batch_id, artifact_class=artifact_class, file_identity=identity))
        return self.publication.publish(batch_id=batch_id, descriptors=descriptors, lifetime=lifetime)['ref']

    def _complete(self, batch_id, future):
        with self._cond:
            group = self._groups[batch_id]
            try:
                group['binding'] = future.result()
                group['durable'] = True
                self._durable_progress.extend(row[1] for row in group['references'])
            except BaseException as exc:
                group['failure'] = exc
                self.refusals.append({'batch_id': batch_id, 'publication_error': repr(exc)})
            self._cond.notify_all()

    def _raise_if_cancelled(self):
        if self._cancelled or (self.cancelled is not None and self.cancelled()):
            raise ProducedOutputWaitCancelled('producer is failing; publication remains separate')

    def cancel_waits(self):
        with self._cond:
            self._cancelled = True
            self._cond.notify_all()

    def _check_failures(self):
        failed = [(key, g['failure']) for key, g in self._groups.items() if g['failure'] is not None]
        if failed:
            key, cause = failed[0]
            raise ProducedExportRefused(f'group {key}: durable publication failed: {cause}') from cause

    def await_group(self, batch_id, *, where='barrier'):
        with self._cond:
            group = self._groups[batch_id]
            if group['future'] is None:
                raise ProducedOutputSpoolRefused('incomplete group has no copy to await')
            while not group['durable']:
                self._check_failures()
                self._raise_if_cancelled()
                self._cond.wait()
            if self._done(group):
                self._release(batch_id, group)

    def _done(self, group):
        return not group['reading'] and (not group['read_back'] or
            {row[1].name for row in group['references']} <= group['retired'])

    def _release(self, batch_id, group):
        if group['released']:
            return
        if not group['durable'] or group['reading']:
            raise ProducedOutputSpoolRefused('durable publication and reader completion precede release')
        self._check_root()
        # Both sides are still exact owned files. A changed destination cannot
        # make the local last good copy disappear.
        from .native_results import verify_result_artifact
        from .stage_a_produced_output import read_bound_document
        manifest = read_bound_document(group['binding'])
        for row in manifest['entries']:
            verify_result_artifact(row, max_bytes=MAX_PAYLOAD_BYTES)
        self._discard_local(group)

    def discard_unsubmitted(self, batch_id):
        with self._cond:
            group = self._groups[batch_id]
            if group['future'] is not None or group['reading']:
                raise ProducedOutputSpoolRefused('copy or reader still owns local files')
            self._discard_local(group)

    def _discard_local(self, group):
        self._check_root()
        for local, _canonical, _artifact_class, identity in group['references']:
            if produced_file_identity(local.path) != identity:
                raise ProducedOutputSpoolRefused('local file changed before release')
        directory = group['directory']
        descriptor = open_directory(str(directory))
        try:
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != group['identity']:
                raise ProducedOutputSpoolRefused('owned group directory changed')
            expected = {Path(row[0].path).name for row in group['references']}
            if set(os.listdir(descriptor)) != expected:
                raise ProducedOutputSpoolRefused('unknown local aliases retain group')
            for local, canonical, _artifact_class, identity in group['references']:
                from .file_identity import file_stat_signature
                if list(file_stat_signature(os.stat(Path(local.path).name,
                        dir_fd=descriptor, follow_symlinks=False))) != identity:
                    raise ProducedOutputSpoolRefused('local file changed at release boundary')
                os.unlink(Path(local.path).name, dir_fd=descriptor)
                self._local.pop(canonical.path, None)
            self._check_root()
            directory.rmdir()
        finally:
            os.close(descriptor)
        group['released'] = True
        self._cond.notify_all()

    def _make_room(self):
        for key, group in self._groups.items():
            if group['durable'] and not group['released'] and not group['reading']:
                if self._done(group) or not group['retired']:
                    self._release(key, group)
                    self.evictions.append({'batch_id': key, 'ceiling_bytes': group['ceiling_bytes']})
                    return True
        return False

    def prepare_reads(self, references):
        """Restore evicted own groups without publishing or submitting again."""
        keys = set()
        with self._cond:
            self._check_root()
            self._check_failures()
            for ref in references:
                for key, group in self._groups.items():
                    if any(row[1] == ref for row in group['references']):
                        keys.add(key)
                        break
                else:
                    raise ProducedOutputSpoolRefused('read names an unknown spool entry')
            if sum(self._groups[key]['ceiling_bytes'] for key in keys) > self.max_bytes:
                raise ProducedWindowRefused('read set exceeds bounded local window')
            # Hold every wanted existing group during room-making so one
            # restoration cannot evict another member of the same read set.
            for key in keys:
                self._groups[key]['reading'] += 1
            try:
                for key in keys:
                    group = self._groups[key]
                    if not group['released']:
                        continue
                    while self._held_bytes() + group['ceiling_bytes'] > self.max_bytes:
                        if not self._make_room():
                            raise ProducedWindowRefused('live readers retain rematerialization room')
                    directory = group['directory']
                    directory.mkdir(mode=0o700)
                    group['identity'] = (directory.stat().st_dev, directory.stat().st_ino)
                    restored = []
                    from .stage_a_produced_output import read_bound_document
                    manifest = read_bound_document(group['binding'])
                    by_path = {row['path']: row for row in manifest['entries']}
                    try:
                        for local, canonical, kind, _identity in group['references']:
                            descriptor = by_path[canonical.path]
                            identity = io_engine.publish_copy(canonical.path, local.path,
                                length=canonical.file_bytes, sha256=canonical.sha256,
                                source_identity=descriptor['file_identity'],
                                destination_directory_identity=group['identity'])
                            restored.append((local, canonical, kind, identity))
                    except BaseException:
                        # A partial restore remains owned and charged, and
                        # cannot be mistaken for a complete local read set.
                        group['released'] = False
                        group['failure'] = RuntimeError('local rematerialization incomplete')
                        raise
                    group['references'] = restored
                    group['released'] = False
                    for local, canonical, _kind, _identity in restored:
                        self._local[canonical.path] = (key, local)
            finally:
                for key in keys:
                    self._groups[key]['reading'] -= 1
                self._cond.notify_all()

    @contextmanager
    def local_reads(self, references):
        held, keys = {}, set()
        with self._cond:
            for ref in references:
                row = self._local.get(ref.path)
                if row is not None:
                    batch, local = row
                    group = self._groups[batch]
                    if not group['released']:
                        held[ref] = Path(local.path)
                        keys.add(batch)
            for batch in keys:
                self._groups[batch]['reading'] += 1
        try:
            yield held
        finally:
            with self._cond:
                for batch in keys:
                    group = self._groups[batch]
                    group['reading'] -= 1
                    if group['durable'] and self._done(group):
                        self._release(batch, group)
                self._cond.notify_all()

    def retire_entry(self, batch_id, canonical_reference):
        with self._cond:
            group = self._groups[batch_id]
            if canonical_reference not in [row[1] for row in group['references']]:
                raise ProducedOutputSpoolRefused('retirement names a foreign reference')
            group['retired'].add(canonical_reference.name)
            if group['durable'] and not group['released'] and self._done(group):
                self._release(batch_id, group)
            return group['released']

    def commit_origin(self, batch_id, descriptors=None, *, lifetime):
        self.await_group(batch_id)
        if descriptors is None:
            from .stage_a_produced_output import read_bound_document
            binding = self._groups[batch_id]['binding']
            if read_bound_document(binding)['lifetime'] != lifetime:
                raise ProducedOutputSpoolRefused('published group lifetime differs')
            return {'ok': True, 'ref': dict(binding)}
        return self.publication.commit_origin(batch_id=batch_id, descriptors=descriptors, lifetime=lifetime)

    def landed(self, batch_id):
        with self._cond:
            self._check_failures()
            return self._groups[batch_id]['durable']

    def holds(self, batch_id):
        with self._cond:
            return batch_id in self._groups and not self._groups[batch_id]['released']

    pending = holds

    def released(self, batch_id):
        with self._cond:
            return batch_id in self._groups and self._groups[batch_id]['released']

    def export_live(self, batch_id):
        with self._cond:
            group = self._groups[batch_id]
            return group['future'] is not None and not group['durable'] and group['failure'] is None

    def exporting(self):
        with self._cond:
            return any(self.export_live(key) for key in self._groups)

    def durable_entries(self):
        with self._cond:
            result, self._durable_progress = self._durable_progress, []
            return result

    def drain(self, *, release=False):
        for key, group in list(self._groups.items()):
            self.await_group(key, where='drain')
            if release:
                with self._cond:
                    self._release(key, group)

    def release_landed(self):
        # Join every owned copy even on failure. No async writer outlives the
        # source retention barrier. A publication error remains independently
        # visible and keeps its local bytes.
        self.join_copies()
        with self._cond:
            for key, group in self._groups.items():
                if group['durable'] and not group['reading']:
                    self._release(key, group)
            return [key for key in self._groups if self.holds(key)]

    def join_copies(self):
        for group in self._groups.values():
            future = group['future']
            if future is not None:
                try:
                    future.result()
                except BaseException:
                    pass
                with self._cond:
                    while not group['durable'] and group['failure'] is None:
                        self._cond.wait()

    def report(self):
        with self._cond:
            return {'schema': 'prismaquant.produced_output_spool.v4',
                'groups': len(self._groups), 'durable_groups': sum(g['durable'] for g in self._groups.values()),
                'pending_groups': sum(not g['released'] for g in self._groups.values()),
                'retained_ceiling_bytes': self._held_bytes(), 'window_bytes': self.max_bytes,
                'refusals': list(self.refusals), 'evictions': list(self.evictions),
                'exports': [{'batch_id': key, 'publication': g['binding'],
                    'entries': len(g['references']), 'durable': g['durable'], 'released': g['released'],
                    'reading': g['reading'], 'publication_error': repr(g['failure']) if g['failure'] else None}
                    for key, g in self._groups.items()]}
