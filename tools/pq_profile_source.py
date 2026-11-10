"""Load profiler dependencies from their stdlib source owners, not package init."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_profile_source(owner_name: str, path: Path) -> ModuleType:
    """Load the caller's explicit stdlib owner once without package initialization."""
    if owner_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(owner_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError('profile source owner unavailable: ' + str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[owner_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(owner_name, None)
            raise
    return sys.modules[owner_name]


def profile_digest_owner():
    path = Path(__file__).resolve().parents[1] / 'prismaquant/digests.py'
    return load_profile_source('pq_profile_shared_digest_owner', path)


bytes_sha256hex = profile_digest_owner().bytes_sha256hex
file_sha256hex = profile_digest_owner().file_sha256hex
canonical_json_sha256 = profile_digest_owner().canonical_json_sha256
canonical_json_bytes = profile_digest_owner().canonical_json_bytes
