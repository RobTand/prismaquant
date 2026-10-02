"""Load the shared stdlib-only digest owner without importing the heavy package."""
import importlib.util
from pathlib import Path
import sys

OWNER_NAME = 'pq_profile_shared_digest_owner'


def profile_digest_owner():
    if OWNER_NAME not in sys.modules:
        path = Path(__file__).resolve().parents[1] / 'prismaquant' / 'digests.py'
        spec = importlib.util.spec_from_file_location(OWNER_NAME, path)
        if spec is None or spec.loader is None:
            raise RuntimeError('profile digest owner unavailable')
        module = importlib.util.module_from_spec(spec)
        sys.modules[OWNER_NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(OWNER_NAME, None)
            raise
    return sys.modules[OWNER_NAME]


bytes_sha256hex = profile_digest_owner().bytes_sha256hex
file_sha256hex = profile_digest_owner().file_sha256hex
canonical_json_sha256 = profile_digest_owner().canonical_json_sha256
canonical_json_bytes = profile_digest_owner().canonical_json_bytes
