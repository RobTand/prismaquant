"""Torch-free bound acquisition metadata shared by runtime and readset builders.

These are input bindings, not scientific or allocator admission. The runtime
must still validate the entire original request/cost before projecting units.
"""
from __future__ import annotations

from pathlib import Path

if __package__:
    from . import digests as _digests, file_identity as _file_identity, schemas as _schemas
else:
    # CPU metadata builders load exact adjacent stdlib owners by source file,
    # like their existing selection/phase owners; never fabricate a package.
    import importlib.util
    import sys

    _owners = {}
    for _name in ("digests", "file_identity", "schemas"):
        _spec = importlib.util.spec_from_file_location(
            "tessera_acquisition_" + _name, Path(__file__).with_name(_name + ".py"))
        if _spec is None or _spec.loader is None:
            raise RuntimeError("acquisition input source owner unavailable: " + _name)
        _owner = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _owner
        try:
            _spec.loader.exec_module(_owner)
        except BaseException:
            sys.modules.pop(_spec.name, None)
            raise
        _owners[_name] = _owner
    _digests, _file_identity, _schemas = (_owners[name] for name in (
        "digests", "file_identity", "schemas"))

_require = _schemas.Contract(ValueError).require
# One metadata pair only: no request/cost payload bytes or scientific result.
# Every reuse checks both stat fences; runtime readers retain their own cache.
_VERIFIED_CONTROL_INPUTS = None


def _control_input_fence(inputs) -> tuple:
    """The existing file-stat owner fences metadata reuse, not admission."""
    result = []
    for entry in inputs:
        value = Path(entry["path"]).stat()
        result.append((value.st_mode, *_file_identity.file_stat_signature(value)))
    return tuple(result)


def _metadata_input(binding: dict, label: str, *, retain_bytes: bool = False):
    """Hash a fenced metadata input; only the small request needs raw bytes."""
    _require(isinstance(binding, dict) and set(binding) == {"path", "sha256"},
             f"{label}: bound path and SHA256 required")
    path = Path(binding["path"])
    before = path.stat()
    before_signature = (before.st_mode, *_file_identity.file_stat_signature(before))
    raw = path.read_bytes() if retain_bytes else None
    digest = (_digests.bytes_sha256hex(raw) if retain_bytes
              else _digests.file_sha256hex(path))
    after = path.stat()
    _require(before_signature == (after.st_mode, *_file_identity.file_stat_signature(after)),
             f"{label}: input changed while hashing")
    _require(digest == binding["sha256"], f"{label}: owned bytes: identity mismatch")
    return {**binding, "bytes": after.st_size}, raw


def read_joint_campaign_acquisition_document(binding: dict, *, reader=None) -> tuple[dict, bytes]:
    """One strict request parser; an admitted runtime supplies its bound reader."""
    if reader is None:
        _, raw = _metadata_input(binding, "joint acquisition request", retain_bytes=True)
    else:
        raw = reader(binding, "joint acquisition request")
    document = _schemas.strict_json_loads(
        raw, duplicate=lambda key: ValueError(f"joint acquisition duplicate JSON key: {key}"),
        constant=lambda value: ValueError(f"joint acquisition nonfinite JSON value: {value}"))
    _require(isinstance(document, dict) and document.get("schema") ==
             "prismaquant.tessera_full_domain_campaign_acquisition.v1",
             "joint acquisition requires the campaign request schema")
    return document, raw


def joint_campaign_acquisition_control_inputs(binding: dict, *, reader=None) -> list[dict]:
    """Actual request then raw-cost readset entries; neither is a price claim.

    Standalone builders stream-hash the cost rather than loading its pickle.
    One fenced entry-only memo avoids repeating that whole read for every row.
    An admitted runtime can supply its existing fenced staged reader.
    """
    global _VERIFIED_CONTROL_INPUTS
    _require(isinstance(binding, dict) and set(binding) == {"path", "sha256"},
             "joint acquisition request: bound path and SHA256 required")
    previous = _VERIFIED_CONTROL_INPUTS
    if reader is None and previous is not None:
        held, fences = previous
        if ({key: held[0][key] for key in ("path", "sha256")} == binding
                and _control_input_fence(held) == fences):
            return [dict(entry) for entry in held]
    before_request = _control_input_fence([binding]) if reader is None else None
    document, request_raw = read_joint_campaign_acquisition_document(binding, reader=reader)
    cost_binding = {"path": document.get("cost_path"), "sha256": document.get("cost_sha256")}
    _require(isinstance(cost_binding["path"], str) and bool(cost_binding["path"]),
             "joint acquisition requires bound cost_path")
    before_cost = _control_input_fence([cost_binding]) if reader is None else None
    if reader is None:
        cost_entry, _ = _metadata_input(cost_binding, "joint acquisition cost")
    else:
        cost_raw = reader(cost_binding, "joint acquisition cost")
        cost_entry = {**cost_binding, "bytes": len(cost_raw)}
    entries = [{**binding, "bytes": len(request_raw)}, cost_entry]
    if reader is None:
        fences = before_request + before_cost
        _require(_control_input_fence(entries) == fences,
                 "joint acquisition control inputs changed during declaration")
        _VERIFIED_CONTROL_INPUTS = ([dict(entry) for entry in entries], fences)
    return entries
