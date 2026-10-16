"""Pure ownership of layer quantum records and their campaign bindings.

Raw-file custody, Stage A proofs, row readiness and tensor execution belong
in their boundary adapters. This owner preserves the v1 record wire.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Literal

from .dev_mode import seal_check
from .digests import canonical_json_bytes as canonical_bytes
from .digests import canonical_json_sha256 as canonical_sha256, newline_utf8_sha256, is_sha256hex

LAYER_QUANTUM_SCHEMA = "prismaquant.joint_layer_quanta.v1"


class QuantumRecordRefused(ValueError):
    """A record-set refusal tied to one input record, before file adaptation."""

    def __init__(self, index: int, reason: str):
        super().__init__(reason)
        self.index = index


def require_quantum_digest(value: object, label: str) -> str:
    if type(value) is not str or not is_sha256hex(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def quantum_id(layer: int) -> str:
    if type(layer) is not int or layer < 0:
        raise ValueError(f"a quantum layer must be a nonnegative integer, not {layer!r}")
    return f"layer-{layer:03d}"


def roster_digest(qnames: Sequence[str]) -> str:
    """``sha256`` of the sorted qname roster, one per line (#768's construction)."""
    ordered = sorted(qnames)
    if any(type(name) is not str or not name for name in ordered):
        raise ValueError("a unit roster holds nonempty qname strings")
    if len(set(ordered)) != len(ordered):
        raise ValueError("a unit roster holds unique qnames")
    return newline_utf8_sha256(ordered)



def quantum_record_identity(record: Mapping) -> str:
    return canonical_sha256(
        {key: value for key, value in record.items() if key != "identity_sha256"},
        where="quantum record")


def require_quantum_identity(record: Mapping) -> None:
    if not isinstance(record, dict) or record.get("schema") != LAYER_QUANTUM_SCHEMA:
        raise ValueError("a quantum record has a foreign schema: refusing")
    if quantum_record_identity(record) != record.get("identity_sha256"):
        raise ValueError(
            f"quantum {record.get('quantum_id')!r} identity does not recompute: "
            "edited record, refusing")


def seal_quantum_record(body: Mapping) -> dict:
    record = copy.deepcopy(dict(body))
    record["identity_sha256"] = quantum_record_identity(record)
    return record


def new_quantum_record(*, layer: int, campaign: Mapping, read_set: Mapping,
                       chunks: Sequence[Mapping], windows: Sequence[Mapping],
                       adjoint: Mapping, output_space: Mapping,
                       catalog_extension: Mapping | None = None) -> dict:
    body = {"schema": LAYER_QUANTUM_SCHEMA, "quantum_id": quantum_id(layer),
            "layer": layer, "campaign": campaign, "read_set": read_set,
            "chunks": chunks, "windows": windows, "adjoint": adjoint,
            "output_space": output_space}
    if catalog_extension is not None:
        body["catalog_extension"] = dict(catalog_extension)
    return seal_quantum_record(body)


def replace_quantum_readset(record: Mapping, *, kind: Literal["boundary", "executable"],
                            block: Mapping) -> dict:
    require_quantum_identity(record)
    if kind not in ("boundary", "executable"):
        raise ValueError("a quantum readset is boundary or executable")
    return seal_quantum_record({**record, f"{kind}_readset": dict(block)})


def require_quantum_campaign(record: Mapping, campaign: Mapping) -> None:
    """Refuse a record built for any other campaign revision (fail closed)."""
    require_quantum_identity(record)
    if not isinstance(campaign, dict):
        raise ValueError("a campaign binding must be an object")
    bound = record.get("campaign", {})
    for field in ("plan_sha256", "prepared_sha256", "read_manifest_sha256",
                  "unit_roster_sha256"):
        expected = campaign.get(field)
        if expected is None:
            raise ValueError(f"the campaign binding names no {field}: refusing")
        refusal = ValueError(
            f"quantum {record.get('quantum_id')!r} was sealed for another "
            f"{field}: refusing")
        if field == "unit_roster_sha256":
            # The roster is what the record measured, not how it was run.
            if bound.get(field) != require_quantum_digest(expected, field):
                raise refusal
            continue
        # The plan, prepared and read-manifest digests are run seals
        # (PQ #1147): dev mode prints a mismatch and admits the record.
        seal_check(field, require_quantum_digest(expected, field), bound.get(field),
                   where=f"quantum {record.get('quantum_id')!r}", refusal=refusal)
    expected_scope = campaign.get("campaign_scope")
    scope_where = f"quantum {record.get('quantum_id')!r}"
    if not isinstance(expected_scope, dict):
        seal_check("campaign_scope", expected_scope, bound.get("campaign_scope"),
                   where=scope_where, same=False, refusal=ValueError(
                       "the campaign binding names no campaign_scope: refusing"))
    else:
        seal_check("campaign_scope", expected_scope, bound.get("campaign_scope"),
                   where=scope_where,
                   same=(canonical_bytes(bound.get("campaign_scope"), where="quantum scope")
                         == canonical_bytes(expected_scope, where="campaign scope")),
                   refusal=lambda: ValueError(
                       f"quantum {record.get('quantum_id')!r} was sealed for another scope: "
                       f"refusing"))
    adjoint = record.get("adjoint", {})
    if "receipt_sha256" in adjoint and adjoint["receipt_sha256"] is not None:
        raise ValueError(
            f"quantum {record.get('quantum_id')!r} binds a whole stage-A receipt; "
            f"records bind their slice (PQ #993): re-seal, refusing")
    if campaign.get("adjoint_receipt_sha256") is not None:
        # A caller still naming the whole receipt would otherwise be silently
        # ignored and admit an unbound record: fail closed.
        raise ValueError("a campaign binding names a whole stage-A receipt; records "
                         "bind their slice (PQ #993): pass adjoint_slice_sha256, refusing")
    expected_slice = campaign.get("adjoint_slice_sha256")
    if expected_slice is not None:
        actual = adjoint.get("slice_sha256")
        if actual is None:
            raise ValueError(
                f"quantum {record.get('quantum_id')!r} is unbound (pre-A): re-seal "
                f"against its stage-A slice before publishing, refusing")
        if actual != require_quantum_digest(expected_slice, "adjoint_slice_sha256"):
            raise ValueError(
                f"quantum {record.get('quantum_id')!r} binds another stage-A "
                f"slice: refusing")


def require_quantum_record_set(records: Sequence[Mapping]) -> None:
    if not records:
        raise ValueError("no quantum records to admit")
    for index, record in enumerate(records):
        try:
            require_quantum_identity(record)
            if not isinstance(record.get("campaign"), dict):
                raise ValueError("a quantum record carries no campaign binding")
            if not isinstance(record.get("adjoint"), dict):
                raise ValueError("a quantum record carries no adjoint binding")
            layer = record.get("layer")
            if record.get("quantum_id") != quantum_id(layer):
                raise ValueError("a quantum id does not name its layer")
        except ValueError as exc:
            raise QuantumRecordRefused(index, str(exc)) from exc
    ids = [record.get("quantum_id") for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("layer records carry duplicated quantum ids")
    first = records[0]["campaign"]
    for index, record in enumerate(records[1:], start=1):
        if record.get("campaign") != first:
            raise QuantumRecordRefused(index, "campaign block differs (mixed campaign)")
    for index, record in enumerate(records):
        if record.get("adjoint", {}).get("receipt_sha256") is not None:
            raise QuantumRecordRefused(
                index, "record binds a whole stage-A receipt, not its slice (PQ #993): regenerate")
    bindings = {record.get("adjoint", {}).get("slice_sha256") is not None for record in records}
    if len(bindings) != 1:
        raise ValueError("layer records mix unbound and slice-bound stage-A bindings")
