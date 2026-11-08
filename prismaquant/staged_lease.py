"""Owned descriptor windows under immutable staged-input lifetimes.

Host-local eviction pins belong to canonical fluxcache run-set. Shared phases
remain retained until every reader scope is stopped or fenced. Windows close
only after their caller joins copies and reader threads; no PB SDK is loaded.
"""
from __future__ import annotations

import os
import threading

from .staged_tier_policy import TierPolicyRefused, tier_is_allowed
from .residency_map import residency_map_key

class LeaseRefused(TierPolicyRefused):
    def __init__(self, reason, *, kind):
        super().__init__(reason)
        self.reason, self.kind = reason, kind


def classify_refusal(reason):
    return 'availability' if str(reason).split(':', 1)[0] in {
        'unpublished', 'stale-epoch', 'retiring', 'file-missing',
        'ram-not-offered', 'ram-not-allowed', 'ssd-not-allowed'} else 'integrity'


_classify = classify_refusal


class ReadsetUnbound(RuntimeError):
    pass


def _authority_hold(*args, **kwargs):
    raise LeaseRefused('unit4 native source/result/publication authority is unqualified', kind='integrity')


# Explicit owning-boundary holds for inventoried unit4 consumers. No SDK,
# injected claim, queue discovery or PB-compatible client is available here.
client_sdk = sdk_submodule = resolve_context = _load_sealed_payload = _authority_hold
load_claimed_manifest = _authority_hold
reviewed_installed_distribution = inject_installed_sdk_for_tests = _authority_hold
set_lease_helper_root = _authority_hold


def lease_helper_root():
    return None


def discover_launch_queue_root(*args, **kwargs):
    return _authority_hold()


def clear_injected_sdk_for_tests():
    return None


def resolve_sealed_readset(*, env=None):
    raise ReadsetUnbound('unit4 legacy claim/CAS launch authority is unqualified; use explicit staged manifest')


def load_sealed_manifest(bound_manifest_sha256):
    from .staged_launch import read_contract_document
    from .staged_manifest import read_document, MAX_BYTES
    from .residency_map import residency_resolver
    from .digests import bytes_sha256hex
    resolver = residency_resolver()
    filename = os.environ.get('PRISMAQUANT_READ_CONTRACT_PATH')
    if resolver is None or not filename:
        raise ReadsetUnbound('explicit scientific read-contract path required; no PB claim discovery')
    resolver.bind_manifest_sha256(bound_manifest_sha256)
    raw, _ = read_document(filename, MAX_BYTES)
    if bytes_sha256hex(raw) != bound_manifest_sha256:
        raise ReadsetUnbound('scientific read-contract digest mismatch')
    return read_contract_document(raw)


def load_sealed_readset(bound_manifest_sha256):
    from .residency_map import residency_resolver
    resolver = residency_resolver()
    if resolver is None:
        raise ReadsetUnbound('explicit immutable staged manifest required')
    resolver.bind_manifest_sha256(bound_manifest_sha256)
    return resolver.declared_readset()


def load_sealed_read_order(bound_manifest_sha256):
    return [(path, start, end) for path, spans in load_sealed_readset(bound_manifest_sha256).items()
            for start, end in spans]


class LeaseWindow:
    def __init__(self, resolver, entries):
        self.resolver, self.entries = resolver, dict(entries)
        self._owner_pid = os.getpid()
        self._entered = self._exited = False
        self._fds = {}
        self._lock = threading.RLock()
        self.serving_tier = 'stage'

    def _require_owner(self):
        if os.getpid() != self._owner_pid:
            raise RuntimeError('LeaseWindow operation from a forked child is refused')

    def __enter__(self):
        self._require_owner()
        if self._entered or self._exited:
            raise RuntimeError('LeaseWindow cannot re-enter or nest')
        self.resolver._require_owner()
        self.resolver.retain_window()
        self._entered = True
        return self

    def open(self, key):
        self._require_owner()
        with self._lock:
            if not self._entered or self._exited or key not in self.entries:
                raise LeaseRefused('window is not live for key', kind='integrity')
            fd = self.resolver.open_entry(self.entries[key])
            self._fds[fd] = key
            return fd, {'pin_id': self.resolver.manifest.phase_id, 'range_ref': key}

    def check_fd(self, fd):
        self._require_owner()
        with self._lock:
            self.resolver.check_entry(self.entries[self._fds[fd]], fd)

    def close_fd(self, fd):
        self._require_owner()
        with self._lock:
            if fd in self._fds:
                try:
                    os.close(fd)
                finally:
                    del self._fds[fd]

    def stage_path(self, key):
        return self.entries[key]['stage_path']

    def read_receipt(self, fd, key, serving):
        from .file_identity import file_stat_signature
        self.check_fd(fd)
        return {'source_fd_stat': list(file_stat_signature(os.fstat(fd))),
                'staged_manifest_sha256': self.resolver.manifest.sha256,
                'read_contract_sha256': self.resolver.manifest.read_contract_sha256,
                **serving}

    def __exit__(self, *args):
        self._require_owner()
        if self._exited:
            raise RuntimeError('LeaseWindow cannot re-exit')
        failure = None
        for fd in list(self._fds):
            try:
                self.check_fd(fd)
            except BaseException as exc:
                failure = failure or exc
            finally:
                self.close_fd(fd)
        self._exited = True
        self.resolver.release_window()
        if failure:
            raise failure
        return False


def acquire_entries_window(resolver, items, *, tier):
    if tier != 'ssd' or not tier_is_allowed('ssd'):
        raise LeaseRefused(f'{tier}-not-allowed', kind='availability')
    items = list(items)
    if not items:
        raise ValueError('nonempty staged window required')
    entries = {residency_map_key(path, entry['offset']): entry for path, entry in items}
    return LeaseWindow(resolver, entries), list(entries)


def acquire_entry_window(resolver, declared, entry):
    window, keys = acquire_entries_window(resolver, [(declared, entry)], tier='ssd')
    return window, keys[0]


def stage_cover_is_published(resolver, declared, entry):
    return entry.get('_resolver') is resolver


def stage_covers_are_published(resolver, items):
    return all(stage_cover_is_published(resolver, path, e) for path, e in items)


EXACT_LEASE_COUNTERS = dict(windows_batched=0, entries_batched=0, entries_single=0, batch_fallbacks=0)


def exact_lease_counters():
    return dict(EXACT_LEASE_COUNTERS)


def enter_group_lease(entry_resolver, members, live_windows, counters=None):
    counters = EXACT_LEASE_COUNTERS if counters is None else counters
    window, keys = acquire_entries_window(entry_resolver,
        [(ref.path, staged) for ref, staged in members], tier='ssd')
    window.__enter__()
    live_windows.append(window)
    counters['windows_batched'] += 1
    counters['entries_batched'] += len(members)
    return {ref: (window, key) for (ref, _), key in zip(members, keys)}
