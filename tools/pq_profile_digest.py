"""Load the shared stdlib-only digest owner without importing the heavy package."""
from tools.pq_profile_source import profile_source_owner


def profile_digest_owner():
    return profile_source_owner('digests')


bytes_sha256hex = profile_digest_owner().bytes_sha256hex
file_sha256hex = profile_digest_owner().file_sha256hex
