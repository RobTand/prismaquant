"""Route a bound transition receipt to the closed module that owns its version.

Each closed transition is its own module with its own contract, rewrites and
verified capability type; nothing here interprets a receipt beyond reading the
version field that names its owner. A receipt whose version no module owns is
refused, and a capability that no module issued is refused the same way.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import joint_aura_run_transition as _run
from . import joint_aura_source_transition as _resume

_LOADERS = {_resume.VERSION: _resume, _run.VERSION: _run}
_VERIFIED = ((_resume.VerifiedTransition, _resume), (_run.VerifiedRunTransition, _run))


def _require(ok, message):
    if not ok:
        raise ValueError(f"joint source transition: {message}")


def receipt_version(bound_receipt):
    """The version a bound receipt names, read only after its bytes match the binding."""
    _require(isinstance(bound_receipt, dict) and set(bound_receipt) == {"path", "sha256"},
             "transition receipt requires independently bound path/SHA256")
    path = Path(bound_receipt["path"])
    _require(path.is_file(), "transition receipt is missing")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == bound_receipt["sha256"], "transition receipt bytes changed")
    try:
        receipt = json.loads(raw)
    except ValueError:
        receipt = None
    version = receipt.get("version") if isinstance(receipt, dict) else None
    _require(version in _LOADERS, f"unknown transition version {version!r}")
    return version


def load_transition(bound_receipt, **kwargs):
    return _LOADERS[receipt_version(bound_receipt)].load_transition(bound_receipt, **kwargs)


def require_verified_transition(value, **kwargs):
    for verified_type, module in _VERIFIED:
        if type(value) is verified_type:
            return module.require_verified_transition(value, **kwargs)
    _require(False, "transition must be issued by the verified receipt loader")
