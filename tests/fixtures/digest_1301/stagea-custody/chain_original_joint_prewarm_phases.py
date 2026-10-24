# Exact original AST-selected source; boundaries in ledger_chains.json.
# Only exercised consumer functions and required dependencies are retained.

# Original joint_prewarm_phases.py:3-3
from __future__ import annotations

# Original joint_prewarm_phases.py:5-5
import gzip

# Original joint_prewarm_phases.py:6-6
import hashlib

# Original joint_prewarm_phases.py:7-7
import json

# Original joint_prewarm_phases.py:8-8
from pathlib import Path

# Original joint_prewarm_phases.py:22-22
HEAD_PHASE = "head"

# Original joint_prewarm_phases.py:42-82
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

# Original joint_prewarm_phases.py:88-88
_REPLAY_MODULE = None

# Original joint_prewarm_phases.py:91-104
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
