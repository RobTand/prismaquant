"""Check a recorded CAS receipt against its own digests.

The receipts of the original GLM capture were written by the retired execution system and are historical
evidence. A reader holding only the receipt can still decide whether it is the one its digests name; this
module decides that, from the receipt alone and with no import of the system that wrote it. It does not
re-derive the attestation from the sealed action, so it proves the receipt is internally consistent, not that
the action it names is the one the reader expects: a reader binds that itself (action key, inputs, result digest).
The schema strings are frozen wire identifiers of the recorded documents.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

CAS_RECEIPT_SCHEMA = "prismaquant.prismabuild.cas_receipt.v3"
WORKER_ATTESTATION_SCHEMA = "prismaquant.prismabuild.worker_attestation.v2"
POOL_OUTCOME_SCHEMA = "prismaquant.prismabuild.pool_outcome.v1"
#: The refusals, in the order they are decided.
RECEIPT_REFUSALS = ("cas-receipt-shape", "cas-receipt-digest", "worker-attestation-digest")
_RECEIPT_KEYS = frozenset({"schema", "action_key", "action_manifest_sha256", "producer", "result", "receipt_sha256"})


def _canonical_sha256(value: object) -> str | None:
    """SHA-256 of canonical JSON (sorted keys, compact separators, no ASCII escaping), or None if not finite JSON."""
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    except (TypeError, ValueError):
        return None


def cas_receipt_self_check(receipt: object) -> str | None:
    """``None`` when a v3 receipt agrees with its own digests, else the first refusal of :data:`RECEIPT_REFUSALS`."""
    if (not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_KEYS
            or receipt["schema"] != CAS_RECEIPT_SCHEMA):
        return RECEIPT_REFUSALS[0]
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if _canonical_sha256(body) != receipt["receipt_sha256"]:
        return RECEIPT_REFUSALS[1]
    producer = receipt["producer"]
    if not isinstance(producer, Mapping) or not (
            producer.get("schema") == WORKER_ATTESTATION_SCHEMA
            and producer.get("action_key") == receipt["action_key"]
            and _canonical_sha256({key: value for key, value in producer.items() if key != "attestation_sha256"})
            == producer.get("attestation_sha256")):
        return RECEIPT_REFUSALS[2]
    return None
