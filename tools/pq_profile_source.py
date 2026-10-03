"""Load profiler dependencies from their stdlib source owners, not package init."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_PROFILE_OWNERS = {
    'digests': 'pq_profile_shared_digest_owner',
    'io_spans': 'pq_profile_shared_sampler_owner',
    'file_identity': 'pq_profile_shared_file_identity_owner',
}


def profile_source_owner(source: str) -> ModuleType:
    """Reuse only the profiler's three stdlib owners from this source snapshot."""
    owner_name = _PROFILE_OWNERS[source]
    if owner_name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / 'prismaquant' / (source + '.py')
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
