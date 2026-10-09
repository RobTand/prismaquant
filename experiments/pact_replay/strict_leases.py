"""Require published PB leases for every declared numerical input range."""
from contextlib import contextmanager
import os


@contextmanager
def strict_input_leases(required):
    import g3_residency
    previous = g3_residency.staged_range
    audit = {"required": bool(required), "reads": 0, "bytes": 0,
             "missing_ranges": [], "origin_fallbacks": 0}
    if required and not os.environ.get("PRISMABUILD_RESIDENCY_MAP"):
        raise RuntimeError("A GPU numerical job needs the admitted PB residency map")
    def leased(path, offset, size):
        raw = previous(path, offset, size)
        if raw is None:
            if required:
                audit["missing_ranges"].append({"path": str(path), "offset": offset, "bytes": size})
                raise RuntimeError("The numerical input has no published PB covering lease: " + str(path))
            audit["origin_fallbacks"] += 1
            return None
        audit["reads"] += 1
        audit["bytes"] += size
        return raw
    g3_residency.staged_range = leased
    try:
        yield audit
    finally:
        g3_residency.staged_range = previous
