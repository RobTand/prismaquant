import gzip

import json

def seal_manifest_bytes(manifest: Mapping) -> bytes:
    """The exact bytes a slice/adjoint manifest file carries.

    Mirrors the run-manifest sealing (``dispatch_tessera_campaign``): compact
    JSON with insertion order, one trailing newline, one gzip member with
    ``mtime=0``. Insertion order is deterministic because every manifest dict
    is constructed key by key in this module.
    """
    if not isinstance(manifest, dict):
        raise ValueError("a data manifest must be a JSON object")
    try:
        decoded = json.dumps(manifest, separators=(",", ":"),
                             sort_keys=False).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise ValueError("a data manifest must be canonical JSON data") from exc
    return gzip.compress(decoded, mtime=0)

