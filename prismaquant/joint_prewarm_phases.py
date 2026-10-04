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

#: The first phase a joint prepare declares, and the only name every manifest
#: is required to carry (``load_prepare_read_set`` refuses one without it). It
#: covers everything the action does before the first layer part: the measured
#: anchor intake, source authentication and model construction. Named here,
#: beside the layer/part names, so the reporting side spells no phase of its
#: own (RobTand/prismaquant#678).
HEAD_PHASE = "head"


def phase_name(layer: int, part: int) -> str:
    return f"layer-{layer}-part-{part}"


def load_prepare_frontier(path: str, sha256: str, plan_sha256: str) -> dict[str, str]:
    """Bind runtime progress to the exact PB manifest used by the storage role.

    The manifest is an immutable CAS input at submission. The action receives
    its digest in the sealed command; replacing the live pathname later must
    stop the action before it can announce a false consumed prefix.

    Returns the unit-to-phase table. A resumed manifest also seals a replay
    frontier beside it; ``load_prepare_read_set`` returns both from one read.
    """
    return load_prepare_read_set(path, sha256, plan_sha256)[0]


def load_prepare_read_set(
    path: str, sha256: str, plan_sha256: str
) -> "tuple[dict[str, str], dict | None, tuple[str, ...]]":
    """The sealed phase table, the sealed replay frontier and the phase names,
    from one read.

    A fresh manifest carries no replay block, so the second element is
    ``None``. A resumed one names the journal units ``_qualification_replay``
    will re-read, their exact order and the journal identity they were
    committed under (RobTand/prismaquant#607); ``sealed_from_annotations``
    refuses a half-sealed block rather than reporting against a guess.

    The third element is why a resumed walk can still announce a phase that
    carries no unit of its own: a layer whose units are all replayed has a
    phase when it declares source extents, and the action has to know whether
    ``phase_name(layer, 0)`` is one the storage role is waiting on before it
    announces it.
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
    if HEAD_PHASE not in declared or not starts:
        raise RuntimeError("joint prepare data manifest has incomplete frontiers")
    return starts, _sealed_replay(annotations), tuple(declared)


#: The sealed-frontier definitions, loaded by path like this module itself.
#: The CPU environment that builds a data manifest has no GPU image, so
#: neither file may import the ``prismaquant`` package.
_REPLAY_MODULE = None


def _sealed_replay(annotations: dict):
    global _REPLAY_MODULE
    if _REPLAY_MODULE is None:
        import importlib.util
        import os

        spec = importlib.util.spec_from_file_location(
            "joint_replay_frontier",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "joint_replay_frontier.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _REPLAY_MODULE = module
    return _REPLAY_MODULE.sealed_from_annotations(annotations)
