"""Route a bound transition receipt to the closed module that owns its version.

Each closed transition is its own module with its own contract, rewrites and
verified capability type; nothing here interprets a receipt beyond reading the
version field that names its owner. A receipt whose version no module owns is
refused, and a capability that no module issued is refused the same way.

One thing about a transition does have to be answered here rather than in the
module that owns it, because the pass asks it of every transition: which plan
digest the PREPARED record must carry. Almost every transition binds the one
plan the run is bound to; the retained-budget transition admits a prepared
record made against another plan, and only because its own proof holds the two
plans identical outside the budget keys its contract enumerates.
``transition_prepared_plan_sha256`` states that per capability type, from a
literal table, so a type nobody taught it refuses instead of defaulting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import joint_aura_retained_budget_transition as _budget
from . import joint_aura_run_transition as _run
from . import joint_aura_source_transition as _resume

_LOADERS = {_resume.VERSION: _resume, _run.VERSION: _run, _budget.VERSION: _budget}
_VERIFIED = ((_resume.VerifiedTransition, _resume), (_run.VerifiedRunTransition, _run),
             (_budget.VerifiedRetainedBudgetTransition, _budget))
#: The plan digest each admitted transition requires of the prepared record.
#: One entry per verified capability type, stated rather than derived.
_PREPARED_PLAN = {
    _resume.VerifiedTransition: lambda value, plan_sha256: plan_sha256,
    _run.VerifiedRunTransition: lambda value, plan_sha256: plan_sha256,
    _budget.VerifiedRetainedBudgetTransition: lambda value, plan_sha256: value.prepared_plan_sha256,
}


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


def transition_prepared_plan_sha256(value, *, plan_sha256):
    """The plan digest the prepared record must carry under this transition."""
    reader = _PREPARED_PLAN.get(type(value))
    _require(reader is not None, "transition must be issued by the verified receipt loader")
    return reader(value, plan_sha256)
