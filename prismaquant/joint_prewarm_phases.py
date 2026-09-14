"""Pure, shared naming and validation for joint prepare ARC read frontiers."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


# Smaller than the storage role's 192 GiB nominal ARC budget even when a
# claimed window and another ready row share that budget. A boundary is placed
# only after a complete qualification unit, never in the middle of a read.
MAX_PHASE_BYTES = 32 * 1024**3


def phase_name(layer: int, part: int) -> str:
    return f"layer-{layer}-part-{part}"


def load_prepare_frontier(path: str, sha256: str, plan_sha256: str) -> dict[str, str]:
    """Bind runtime progress to the exact PB manifest used by the storage role.

    The manifest is an immutable CAS input at submission. The action receives
    its digest in the sealed command; replacing the live pathname later must
    stop the action before it can announce a false consumed prefix.
    """
    blob = Path(path).read_bytes()
    if hashlib.sha256(blob).hexdigest() != sha256:
        raise RuntimeError("joint prepare data manifest changed after submission")
    decoded = gzip.decompress(blob) if path.endswith(".gz") else blob
    manifest = json.loads(decoded)
    annotations = manifest.get("annotations")
    if (not isinstance(annotations, dict)
            or annotations.get("entry_point") != "prismaquant.tessera_joint_aura:prepare"
            or annotations.get("plan_sha256") != plan_sha256):
        raise RuntimeError("joint prepare data manifest has a foreign plan or command")
    starts = annotations.get("phase_start_units")
    phases = annotations.get("phases")
    if not isinstance(starts, dict) or not isinstance(phases, list):
        raise RuntimeError("joint prepare data manifest has no read-frontier plan")
    declared = [row.get("name") for row in phases if isinstance(row, dict)]
    if len(declared) != len(phases) or len(declared) != len(set(declared)):
        raise RuntimeError("joint prepare data manifest has ambiguous phases")
    if any(not isinstance(unit, str) or not isinstance(phase, str)
           or phase not in declared for unit, phase in starts.items()):
        raise RuntimeError("joint prepare data manifest has invalid unit frontiers")
    if "head" not in declared or not starts:
        raise RuntimeError("joint prepare data manifest has incomplete frontiers")
    return starts
