"""Load profiler dependencies from their stdlib source owners, not package init."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_PROFILE_OWNERS = {
    'digests': ('pq_profile_shared_digest_owner', 'prismaquant/digests.py'),
    'io_spans': ('pq_profile_shared_sampler_owner', 'prismaquant/io_spans.py'),
    'file_identity': ('pq_profile_shared_file_identity_owner', 'prismaquant/file_identity.py'),
}


def profile_source_owner(source: str) -> ModuleType:
    """Reuse only the profiler's three stdlib owners from this source snapshot."""
    owner_name, filename = _PROFILE_OWNERS[source]
    if owner_name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / filename
        spec = importlib.util.spec_from_file_location(owner_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError('profile source owner unavailable: ' + source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[owner_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(owner_name, None)
            raise
    return sys.modules[owner_name]


def profile_digest_owner():
    return profile_source_owner('digests')


bytes_sha256hex = profile_digest_owner().bytes_sha256hex
file_sha256hex = profile_digest_owner().file_sha256hex
canonical_json_sha256 = profile_digest_owner().canonical_json_sha256
canonical_json_bytes = profile_digest_owner().canonical_json_bytes
