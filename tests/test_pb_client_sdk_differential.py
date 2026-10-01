"""The PrismaBuild client SDK decides what the old internal calls decided.

Decoupling step 2 moves five modules off PrismaBuild internals and onto
``prismabuild.client`` (PB #1254): ``staged_lease``,
``stage_a_produced_output``, ``glm_capture_compatibility``,
``residency_map`` and ``quality_prefill_pb_adapter``. Each test here runs the
OLD path and the NEW path on the same corpus and requires the same decision
and the same bytes.

The OLD path means one of two things:

- the PB internal the module used to call (``pool._read_json``,
  ``produced_output.safe_release_instance`` with ``reader_lease``, the
  ``core`` regexes), imported here directly, because tests are outside the
  boundary gate; or
- PQ's own restated validator, copied verbatim from the parent commit
  ``c36c02e1e76``. The copies are the evidence and are not used by any
  production path.

Refusal *reasons* may differ where PB's validator now words the refusal
(residency maps); the decision and the adopted state may not.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import pytest



@pytest.fixture
def client(installed_client_sdk):
    from prismaquant import staged_lease

    assert staged_lease.client_sdk() is installed_client_sdk
    return installed_client_sdk


def _outcome(call):
    """``("ok", value)`` or ``(exception type name, message)``."""

    try:
        return ("ok", call())
    except Exception as exc:  # noqa: BLE001 - the type IS the decision
        return (type(exc).__name__, str(exc))


# ---------------------------------------------------------------------------
# staged_lease + stage_a_produced_output: the calls are PB's own objects
# ---------------------------------------------------------------------------

#: Every PB object the two lease-side modules now reach through the client,
#: and the internal it used to be reached as.
_SAME_OBJECTS = {
    "acquire_for": ("reader_lease", "acquire_for"),
    "open_pinned": ("reader_lease", "open_pinned"),
    "release": ("reader_lease", "release"),
    "covers_for_keys": ("reader_lease", "covers_for_keys"),
    "leases_root": ("reader_lease", "leases_root"),
    "injected_context": ("reader_lease", "injected_context"),
    "READER_LEASE_TAG": ("reader_lease", "READER_LEASE_TAG"),
    "read_data_manifest": ("core", "read_data_manifest"),
    "DATA_MANIFEST_MAX_BYTES": ("core", "DATA_MANIFEST_MAX_BYTES"),
    "manifest_read_entries": ("storage_tiers", "manifest_read_entries"),
    "PoolQueue": ("pool", "PoolQueue"),
    "CLAIMED": ("pool", "CLAIMED"),
    "POOL_OUTCOME_SCHEMA_V1": ("pool", "POOL_OUTCOME_SCHEMA_V1"),
    "read_residency_fragments": ("residency_map", "read_fragments"),
    "compose_residency_map": ("residency_map", "compose"),
    "write_residency_map": ("residency_map", "write_map"),
    "validate_residency_map": ("residency_map", "validate_map"),
    "ResidencyMapError": ("residency_map", "ResidencyMapError"),
    "ID_PATTERN": ("core", "_ID_RE"),
    "ENV_NAME_PATTERN": ("core", "_ENV_RE"),
    "canonical_sha256": ("core", "canonical_sha256"),
}

#: The produced-output calls ``stage_a_produced_output`` makes, each the
#: same object under the client as under ``produced_output``.
_PRODUCED_OUTPUT = (
    "declared_template", "bind_declared_instance", "declare_instance",
    "admit_instance", "admit_funded_window", "require_prewrite",
    "abort_prewrite", "publish_prepaid_batch", "safe_release_instance",
    "recover_batches", "due_mover_rows",
)


def test_the_client_serves_the_same_pb_objects(client):
    import importlib

    for name, (module, attribute) in _SAME_OBJECTS.items():
        internal = getattr(importlib.import_module(f"prismabuild.{module}"), attribute)
        assert getattr(client, name) is internal, name
    produced_output = importlib.import_module("prismabuild.produced_output")
    for name in _PRODUCED_OUTPUT:
        assert getattr(client, name) is getattr(produced_output, name), name


def _claimed_corpus(action_key: str) -> dict[str, bytes | None]:
    row = {"action_key": action_key, "cas_root": "/cas/root",
           "residency": {"manifest_sha256": "f" * 64, "manifest_bytes": 812}}
    return {
        "row": json.dumps(row).encode(),
        "row-other-action": json.dumps(dict(row, action_key="0" * 64)).encode(),
        "row-no-cas-root": json.dumps(dict(row, cas_root="")).encode(),
        "row-no-residency": json.dumps(dict(row, residency=None)).encode(),
        "row-bad-size": json.dumps(dict(row, residency={
            "manifest_sha256": "f" * 64, "manifest_bytes": True})).encode(),
        "absent": None,
        "array": b"[1, 2]",
        "torn": b'{"action_key": "',
        "empty": b"",
        "not-utf8": b"\xff\xfe{}",
        "scalar": b"7",
    }


def _write_claim(queue_root: Path, action_key: str, body: bytes | None) -> None:
    path = queue_root / "claimed" / f"{action_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    if body is not None:
        path.write_bytes(body)


def test_claimed_record_reads_like_pool_read_json(client, tmp_path):
    """``stage_a_produced_output``'s owner-claim read, old call vs SDK call."""

    import prismabuild.pool as pool

    queue = pool.PoolQueue(tmp_path / "queue")
    key = "e" * 64
    for label, body in _claimed_corpus(key).items():
        _write_claim(tmp_path / "queue", key, body)
        assert queue.item_path(pool.CLAIMED, key) == tmp_path / "queue" / "claimed" / f"{key}.json"
        old = _outcome(lambda: pool._read_json(queue.item_path(pool.CLAIMED, key)))
        new = _outcome(lambda: client.read_claimed_record(queue, key))
        assert new == old, (label, old, new)


def _old_sealed_row(queue_root: str, action_key: str):
    """The claim-row half of ``staged_lease.resolve_sealed_readset`` at
    c36c02e1e76 (its own reader, with its 8 MiB bound)."""

    from prismaquant.staged_lease import ReadsetUnbound

    row_path = Path(queue_root) / "claimed" / f"{action_key}.json"
    try:
        with open(row_path, "rb") as handle:
            row = json.loads(handle.read(8 * 1024 * 1024 + 1).decode("utf-8"))
    except FileNotFoundError:
        raise ReadsetUnbound(f"no claim row at {row_path}") from None
    except (OSError, UnicodeError, ValueError) as error:
        raise ReadsetUnbound(f"claim row is unreadable: {error}") from None
    if not isinstance(row, dict) or row.get("action_key") != action_key:
        raise ReadsetUnbound("claim row names another action")
    cas_root = row.get("cas_root")
    residency = row.get("residency")
    if not isinstance(cas_root, str) or not cas_root:
        raise ReadsetUnbound("claim row publishes no cas_root")
    if not isinstance(residency, dict):
        raise ReadsetUnbound("claim row publishes no residency block")
    digest = residency.get("manifest_sha256")
    size = residency.get("manifest_bytes")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ReadsetUnbound("residency block names no manifest digest")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ReadsetUnbound("residency block names no manifest size")
    return cas_root, digest, size


def test_sealed_readset_row_decides_like_the_old_reader(client, tmp_path, monkeypatch):
    from prismaquant import staged_lease

    queue_root, key = tmp_path / "queue", "e" * 64
    ctx = {"queue_root": str(queue_root), "action_key": key}
    monkeypatch.setattr(staged_lease, "resolve_context", lambda env=None: (client, ctx))
    for label, body in _claimed_corpus(key).items():
        _write_claim(queue_root, key, body)
        old = _outcome(lambda: _old_sealed_row(str(queue_root), key))
        new = _outcome(lambda: staged_lease.resolve_sealed_readset())
        assert new[0] == old[0], (label, old, new)
        if old[0] == "ok":
            assert new == old, label
        else:
            assert old[0] == "ReadsetUnbound", (label, old)


def test_release_passes_the_same_arguments(client, monkeypatch):
    import prismabuild.produced_output as produced_output
    import prismabuild.reader_lease as reader_lease

    calls = []

    def recorder(queue, instance, template, *, lease_sdk=None):
        calls.append((queue, instance, template, lease_sdk))
        return {"released": len(calls)}

    monkeypatch.setattr(produced_output, "safe_release_instance", recorder)
    queue, instance, template = object(), {"id": "i"}, {"id": "t"}
    # Old: stage_a_produced_output imported reader_lease itself.
    old = produced_output.safe_release_instance(queue, instance, template,
                                                lease_sdk=reader_lease)
    new = client.release_produced_instance(queue, instance, template)
    assert calls[0] == calls[1] == (queue, instance, template, reader_lease)
    assert (old, new) == ({"released": 1}, {"released": 2})


def test_stage_a_release_goes_through_the_client(client, monkeypatch):
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication

    seen = []
    monkeypatch.setattr(client, "release_produced_instance",
                        lambda *args: seen.append(args) or {"ok": 1})
    publication = BoundaryProducedPublication.__new__(BoundaryProducedPublication)
    publication._po = client
    publication.queue, publication.instance, publication.template = "q", "i", "t"
    assert publication.release() == {"ok": 1}
    assert seen == [("q", "i", "t")]


# ---------------------------------------------------------------------------
# glm_capture_compatibility: the CAS receipt self-check
# ---------------------------------------------------------------------------

def _old_verify_cas_receipt(receipt, snapshot, output):
    """``glm_capture_compatibility._verify_cas_receipt`` at c36c02e1e76."""

    from prismaquant.digests import DIRECT_UTF8_STRICT
    from prismaquant.glm_source_derivative import _require
    _digest = DIRECT_UTF8_STRICT.sha256
    _require(set(receipt) == {'schema', 'action_key', 'action_manifest_sha256', 'producer', 'result', 'receipt_sha256'} and
             receipt['schema'] == 'prismaquant.prismabuild.cas_receipt.v3', 'canonical v3 CAS receipt required')
    body = {key: value for key, value in receipt.items() if key != 'receipt_sha256'}
    _require(_digest(body) == receipt['receipt_sha256'], 'CAS receipt body digest differs')
    producer = receipt['producer']
    _require(producer.get('schema') == 'prismaquant.prismabuild.worker_attestation.v2' and
             producer.get('action_key') == receipt['action_key'] and
             _digest({key: value for key, value in producer.items() if key != 'attestation_sha256'}) == producer.get('attestation_sha256'),
             'CAS producer attestation digest differs')
    _require(snapshot['input'] in producer.get('inputs', []), 'CAS producer does not bind the reviewed source snapshot input')
    _require(receipt['result'] == dict(sha256=hashlib.sha256(output).hexdigest(), bytes=len(output)),
             'original CAS output is not bound by its receipt')


def _sha(value) -> str:
    from prismaquant.digests import DIRECT_UTF8_STRICT
    return DIRECT_UTF8_STRICT.sha256(value)


def _receipt(output: bytes, snapshot_input: dict, *, action="c" * 64):
    producer = {"schema": "prismaquant.prismabuild.worker_attestation.v2",
                "action_key": action, "host": "sparky",
                "inputs": [snapshot_input]}
    producer["attestation_sha256"] = _sha(producer)
    receipt = {"schema": "prismaquant.prismabuild.cas_receipt.v3",
               "action_key": action, "action_manifest_sha256": "d" * 64,
               "producer": producer,
               "result": {"sha256": hashlib.sha256(output).hexdigest(),
                          "bytes": len(output)}}
    receipt["receipt_sha256"] = _sha(receipt)
    return receipt


def _reseal(receipt, *, producer=True, body=True):
    receipt = json.loads(json.dumps(receipt))
    if producer and isinstance(receipt.get("producer"), dict):
        receipt["producer"].pop("attestation_sha256", None)
        receipt["producer"]["attestation_sha256"] = _sha(receipt["producer"])
    if body:
        receipt.pop("receipt_sha256", None)
        receipt["receipt_sha256"] = _sha(receipt)
    return receipt


def _receipt_corpus():
    output = b"captured bytes\n"
    snapshot = {"input": {"id": "pbrun.checkout-snapshot", "sha256": "e" * 64,
                          "bytes": 12}}
    good = _receipt(output, snapshot["input"])

    def edit(fn, **reseal):
        receipt = json.loads(json.dumps(good))
        fn(receipt)
        return _reseal(receipt, **reseal) if reseal else receipt

    cases = {
        "valid": good,
        "extra-key": edit(lambda r: r.update(extra=1), producer=False),
        "missing-key": edit(lambda r: r.pop("action_manifest_sha256"), producer=False),
        "wrong-schema": edit(lambda r: r.update(schema="prismaquant.prismabuild.cas_receipt.v2"),
                             producer=False),
        "body-digest": edit(lambda r: r.update(receipt_sha256="0" * 64)),
        "body-edited": edit(lambda r: r.update(action_manifest_sha256="f" * 64)),
        "producer-schema": edit(lambda r: r["producer"].update(
            schema="prismaquant.prismabuild.worker_attestation.v1"), producer=True),
        "producer-action": edit(lambda r: r["producer"].update(action_key="9" * 64),
                                producer=True),
        "producer-digest": edit(lambda r: r["producer"].update(attestation_sha256="1" * 64),
                                producer=False, body=True),
        "producer-no-digest": edit(lambda r: r["producer"].pop("attestation_sha256"),
                                   producer=False, body=True),
        "snapshot-unbound": edit(lambda r: r["producer"].update(inputs=[]), producer=True),
        "result-differs": edit(lambda r: r["result"].update(bytes=1), producer=False),
        "not-an-object": ["schema"],
    }
    return cases, snapshot, output


def test_receipt_check_decides_like_the_restated_one(client):
    from prismaquant import glm_capture_compatibility as glm

    cases, snapshot, output = _receipt_corpus()
    decisions = {}
    for label, receipt in cases.items():
        old = _outcome(lambda: _old_verify_cas_receipt(receipt, snapshot, output))
        new = _outcome(lambda: glm._verify_cas_receipt(receipt, snapshot, output))
        assert new == old, (label, old, new)
        decisions[label] = old[0] if old[0] != "ValueError" else old[1]
    # The corpus reaches every refusal, not only the first one.
    assert decisions["valid"] == "ok"
    assert {decisions[k] for k in ("extra-key", "missing-key", "wrong-schema",
                                   "not-an-object")} == {
        "GLM source derivative: canonical v3 CAS receipt required"}
    assert decisions["body-digest"].endswith("CAS receipt body digest differs")
    assert decisions["body-edited"].endswith("CAS receipt body digest differs")
    for label in ("producer-schema", "producer-action", "producer-digest",
                  "producer-no-digest"):
        assert decisions[label].endswith("CAS producer attestation digest differs"), label
    assert decisions["snapshot-unbound"].endswith("reviewed source snapshot input")
    assert decisions["result-differs"].endswith("not bound by its receipt")


def test_pool_outcome_schema_is_the_same_string(client):
    assert client.POOL_OUTCOME_SCHEMA_V1 == "prismaquant.prismabuild.pool_outcome.v1"


# ---------------------------------------------------------------------------
# residency_map: PB's validate_map in place of the restated field rules
# ---------------------------------------------------------------------------

_OLD_SCHEMA = "prismaquant.prismabuild.residency_map.v1"
_OLD_ROOT_KEYS = {"schema", "tier_id", "stage_root", "manifest_sha256", "leads",
                  "generation", "entries"}
_OLD_RAM_ROOT_KEYS = {"ram_tier_id", "ram_root", "ram_epoch"}
_OLD_ENTRY_KEYS = {"stage_path", "bytes", "offset", "sha256", "ram_path"}
_HEX = frozenset("0123456789abcdef")


class _OldRefused(Exception):
    pass


def _is_hex64(value):
    return type(value) is str and len(value) == 64 and all(c in _HEX for c in value)


def _old_ram_header(payload):
    ram_root = payload.get("ram_root")
    if ram_root is not None and (
            type(ram_root) is not str or not ram_root.startswith("/")
            or os.path.normpath(ram_root) != ram_root):
        raise _OldRefused("residency map ram_root is not a normalized absolute path")
    ram_tier_id = payload.get("ram_tier_id")
    if ram_tier_id is not None and (
            type(ram_tier_id) is not str or not ram_tier_id or "/" in ram_tier_id):
        raise _OldRefused("residency map ram_tier_id is not a tier id")
    ram_epoch = payload.get("ram_epoch")
    if ram_epoch is not None and (
            type(ram_epoch) is not str or not ram_epoch or "/" in ram_epoch):
        raise _OldRefused("residency map ram_epoch is not an epoch")
    return ram_tier_id, ram_root, ram_epoch


def _old_entry(key, row, stage_root, prefix, ram_root=None):
    head, separator, path = key.partition(":")
    if not separator or not path or not head.isdigit():
        raise _OldRefused(f"malformed residency map key {key!r}")
    offset = int(head)
    if type(row) is not dict:
        raise _OldRefused(f"residency map entry {key!r} must be an object")
    unknown = sorted(set(row) - _OLD_ENTRY_KEYS)
    if unknown:
        raise _OldRefused(f"unknown residency map entry fields: {unknown}")
    stage_path = row.get("stage_path")
    if (type(stage_path) is not str or not stage_path.startswith("/")
            or os.path.normpath(stage_path) != stage_path):
        raise _OldRefused(f"residency map entry {key!r} stage_path is not a normalized absolute path")
    if not (stage_path == stage_root or stage_path.startswith(prefix)):
        raise _OldRefused(f"residency map entry {key!r} is staged outside {stage_root!r}")
    size = row.get("bytes")
    if type(size) is not int or isinstance(size, bool) or size <= 0:
        raise _OldRefused(f"residency map entry {key!r} has no positive size")
    declared = row.get("offset", offset)
    if type(declared) is not int or isinstance(declared, bool) or declared < 0:
        raise _OldRefused(f"residency map entry {key!r} offset is not a count")
    if declared != offset:
        raise _OldRefused(f"residency map entry {key!r} offset {declared} disagrees with its key")
    if not _is_hex64(row.get("sha256")):
        raise _OldRefused(f"residency map entry {key!r} has no SHA-256 digest")
    checked = {"stage_path": stage_path, "bytes": size, "offset": offset,
               "sha256": row["sha256"], "declared_path": path}
    ram_path = row.get("ram_path")
    if ram_path is not None:
        if ram_root is None:
            raise _OldRefused(f"residency map entry {key!r} names a ram_path, "
                              "but the map announces no ram root")
        if (type(ram_path) is not str or not ram_path.startswith("/")
                or os.path.normpath(ram_path) != ram_path):
            raise _OldRefused(f"residency map entry {key!r} ram_path is not a normalized absolute path")
        ram_prefix = ram_root.rstrip("/") + "/"
        if not (ram_path == ram_root or ram_path.startswith(ram_prefix)):
            raise _OldRefused(f"residency map entry {key!r} ram_path is outside "
                              f"the map's ram root {ram_root!r}")
        checked["ram_path"] = ram_path
    return checked


def _old_adopt(payload, manifest_sha256):
    """``ResidencyResolver._adopt`` at c36c02e1e76, returning what it set."""

    if type(payload) is not dict:
        raise _OldRefused(f"residency map must be an object, not {type(payload).__name__}")
    unknown = sorted(set(payload) - _OLD_ROOT_KEYS - _OLD_RAM_ROOT_KEYS)
    if unknown:
        raise _OldRefused(f"unknown residency map fields: {unknown}")
    missing = sorted(_OLD_ROOT_KEYS - set(payload))
    if missing:
        raise _OldRefused(f"residency map is missing {missing}")
    if payload["schema"] != _OLD_SCHEMA:
        raise _OldRefused(f"residency map declares schema {payload['schema']!r}, not {_OLD_SCHEMA}")
    if not _is_hex64(payload["manifest_sha256"]):
        raise _OldRefused("residency map manifest_sha256 is not a digest")
    if payload["manifest_sha256"] != manifest_sha256:
        raise _OldRefused("residency map names another data manifest")
    stage_root = payload["stage_root"]
    if (type(stage_root) is not str or not stage_root.startswith("/")
            or os.path.normpath(stage_root) != stage_root):
        raise _OldRefused("residency map stage_root is not a normalized absolute path")
    tier_id = payload["tier_id"]
    if type(tier_id) is not str or not tier_id or "/" in tier_id:
        raise _OldRefused("residency map tier_id is not a tier id")
    leads = payload["leads"]
    if type(leads) is not list or any(not _is_hex64(lead) for lead in leads):
        raise _OldRefused("residency map leads must be 64-character action keys")
    if len(set(leads)) != len(leads):
        raise _OldRefused("residency map leads repeat a key")
    generation = payload["generation"]
    if type(generation) is not int or isinstance(generation, bool) or generation < 0:
        raise _OldRefused("residency map generation must be a count")
    ram_tier_id, ram_root, ram_epoch = _old_ram_header(payload)
    entries = payload["entries"]
    if type(entries) is not dict:
        raise _OldRefused("residency map entries must be an object")
    prefix = stage_root.rstrip("/") + "/"
    adopted = {}
    for key, row in entries.items():
        adopted[str(key)] = _old_entry(str(key), row, stage_root, prefix, ram_root)
    if any("ram_path" in entry for entry in adopted.values()) and (
            ram_tier_id is None or ram_root is None or ram_epoch is None):
        raise _OldRefused("a residency map naming ram paths must announce its "
                          "ram tier, root and epoch")
    return {"entries": adopted, "tier_id": tier_id, "stage_root": stage_root,
            "leads": tuple(leads), "generation": generation,
            "ram_tier_id": ram_tier_id, "ram_root": ram_root, "ram_epoch": ram_epoch}


_MANIFEST = "a" * 64
_LEAD = "b" * 64
_DIGEST = "c" * 64


def _map(**changes):
    payload = {
        "schema": _OLD_SCHEMA, "tier_id": "nvme-stage", "stage_root": "/stage/root",
        "manifest_sha256": _MANIFEST, "leads": [_LEAD], "generation": 2,
        "entries": {
            "0:/mnt/shared/model/shard-1.safetensors": {
                "stage_path": "/stage/root/ab/shard-1", "bytes": 10, "offset": 0,
                "sha256": _DIGEST},
            "1048576:/mnt/shared/model/shard-2.safetensors": {
                "stage_path": "/stage/root/cd/shard-2", "bytes": 5, "sha256": _DIGEST},
        },
    }
    for key, value in changes.items():
        if value is _DROP:
            payload.pop(key)
        else:
            payload[key] = value
    return payload


_DROP = object()
_RAM = {"ram_tier_id": "ram-sparky", "ram_root": "/dev/shm/pb-ram", "ram_epoch": "e1"}


def _with_entry(key, row, **header):
    payload = _map(**header)
    payload["entries"] = dict(payload["entries"], **{key: row})
    return payload


def _row(**changes):
    row = {"stage_path": "/stage/root/ef/x", "bytes": 3, "offset": 0, "sha256": _DIGEST}
    for key, value in changes.items():
        if value is _DROP:
            row.pop(key)
        else:
            row[key] = value
    return row


def _map_corpus():
    corpus = {
        "valid": _map(),
        "valid-no-entries": _map(entries={}),
        "valid-no-leads": _map(leads=[]),
        "valid-generation-0": _map(generation=0),
        "valid-ram-header-only": _map(**_RAM),
        "valid-ram-entry": _with_entry(
            "0:/mnt/shared/r", _row(ram_path="/dev/shm/pb-ram/x"), **_RAM),
        "valid-stage-path-is-root": _with_entry("0:/mnt/shared/r", _row(stage_path="/stage/root")),
        "valid-ram-path-is-root": _with_entry(
            "0:/mnt/shared/r", _row(ram_path="/dev/shm/pb-ram"), **_RAM),
        "not-an-object": [1, 2],
        "unknown-field": _map(extra=1),
        "wrong-schema": _map(schema="prismaquant.prismabuild.residency_map.v2"),
        "manifest-uppercase": _map(manifest_sha256="A" * 64),
        "manifest-short": _map(manifest_sha256="a" * 63),
        "manifest-other": _map(manifest_sha256="d" * 64),
        "stage-root-relative": _map(stage_root="stage/root"),
        "stage-root-unnormalized": _map(stage_root="/stage//root"),
        "stage-root-trailing-slash": _map(stage_root="/stage/root/"),
        "tier-empty": _map(tier_id=""),
        "tier-slash": _map(tier_id="a/b"),
        "tier-int": _map(tier_id=3),
        "leads-not-list": _map(leads="x"),
        "leads-bad": _map(leads=["B" * 64]),
        "leads-repeat": _map(leads=[_LEAD, _LEAD]),
        "generation-negative": _map(generation=-1),
        "generation-bool": _map(generation=True),
        "generation-str": _map(generation="2"),
        "entries-list": _map(entries=[]),
        "key-no-colon": _with_entry("/mnt/shared/r", _row()),
        "key-no-path": _with_entry("0:", _row()),
        "key-not-digit": _with_entry("x:/mnt/shared/r", _row()),
        "key-negative": _with_entry("-1:/mnt/shared/r", _row()),
        "entry-not-object": _with_entry("0:/mnt/shared/r", [1]),
        "entry-unknown-field": _with_entry("0:/mnt/shared/r", _row(extra=1)),
        "stage-path-outside": _with_entry("0:/mnt/shared/r", _row(stage_path="/elsewhere/x")),
        "stage-path-sibling-prefix": _with_entry(
            "0:/mnt/shared/r", _row(stage_path="/stage/rootx/y")),
        "stage-path-relative": _with_entry("0:/mnt/shared/r", _row(stage_path="x")),
        "stage-path-missing": _with_entry("0:/mnt/shared/r", _row(stage_path=_DROP)),
        "bytes-zero": _with_entry("0:/mnt/shared/r", _row(bytes=0)),
        "bytes-bool": _with_entry("0:/mnt/shared/r", _row(bytes=True)),
        "bytes-missing": _with_entry("0:/mnt/shared/r", _row(bytes=_DROP)),
        "offset-disagrees": _with_entry("0:/mnt/shared/r", _row(offset=4)),
        "offset-negative": _with_entry("0:/mnt/shared/r", _row(offset=-1)),
        "offset-bool": _with_entry("1:/mnt/shared/r", _row(offset=True)),
        "valid-offset-from-key": _with_entry("7:/mnt/shared/r", _row(offset=_DROP)),
        "sha-bad": _with_entry("0:/mnt/shared/r", _row(sha256="C" * 64)),
        "sha-missing": _with_entry("0:/mnt/shared/r", _row(sha256=_DROP)),
        "ram-path-no-root": _with_entry("0:/mnt/shared/r", _row(ram_path="/dev/shm/pb-ram/x")),
        "ram-path-outside": _with_entry(
            "0:/mnt/shared/r", _row(ram_path="/dev/shm/other/x"), **_RAM),
        "ram-path-relative": _with_entry("0:/mnt/shared/r", _row(ram_path="x"), **_RAM),
        "ram-path-no-epoch": _with_entry(
            "0:/mnt/shared/r", _row(ram_path="/dev/shm/pb-ram/x"),
            ram_tier_id="ram-sparky", ram_root="/dev/shm/pb-ram"),
        "ram-path-no-tier": _with_entry(
            "0:/mnt/shared/r", _row(ram_path="/dev/shm/pb-ram/x"),
            ram_root="/dev/shm/pb-ram", ram_epoch="e1"),
        "ram-root-unnormalized": _map(ram_root="/dev/shm/../pb"),
        "ram-tier-slash": _map(ram_tier_id="a/b"),
        "ram-epoch-slash": _map(ram_epoch="e/1"),
        "ram-epoch-empty": _map(ram_epoch=""),
    }
    for key in ("schema", "tier_id", "stage_root", "manifest_sha256", "leads",
                "generation", "entries"):
        corpus[f"missing-{key}"] = _map(**{key: _DROP})
    return corpus


def _new_adopt(tmp_path, payload):
    from prismaquant.residency_map import ResidencyResolver

    resolver = ResidencyResolver(tmp_path / "ctx" / "map.json")
    resolver._manifest_sha256 = _MANIFEST
    resolver._adopt(payload, json.dumps(payload).encode())
    return {"entries": resolver._entries, "tier_id": resolver._tier_id,
            "stage_root": resolver._stage_root, "leads": resolver._leads,
            "generation": resolver._generation,
            "ram_tier_id": resolver._ram_tier_id, "ram_root": resolver._ram_root,
            "ram_epoch": resolver._ram_epoch}


def test_residency_map_adoption_decides_like_the_restated_rules(client, tmp_path):
    from prismaquant.residency_map import ResidencyMapRefused

    corpus = _map_corpus()
    accepted = set()
    for label, payload in corpus.items():
        old = _outcome(lambda: _old_adopt(payload, _MANIFEST))
        new = _outcome(lambda: _new_adopt(tmp_path, payload))
        if old[0] == "ok":
            assert new == old, (label, old, new)
            accepted.add(label)
        else:
            assert old[0] == "_OldRefused", (label, old)
            assert new[0] == ResidencyMapRefused.__name__, (label, old, new)
    assert accepted == {label for label in corpus if label.startswith("valid")}


# ---------------------------------------------------------------------------
# quality_prefill_pb_adapter: grammars, identity digest, capability probe
# ---------------------------------------------------------------------------

#: The patterns the adapter restated at c36c02e1e76.
_OLD_PB_ID_RE = re.compile(r"[a-z0-9][a-z0-9._/-]{0,255}\Z")
_OLD_PB_ENV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

_TEXTS = (
    "a", "0", "abc.def/ghi-jkl_mno", "A", "_x", "x" * 256, "x" * 257, "a/", "-a",
    ".a", "a b", "a\n", "é", "PATH", "PRISMABUILD_QUEUE_ROOT", "9LIVES", "x-y",
    "", "a" * 300, "gpu", "mem_gb", "a..b", "a//b", "ab\u0000",
)


def test_adapter_grammars_are_pbs(client):
    from prismaquant import quality_prefill_pb_adapter as adapter

    assert client.ID_PATTERN.pattern == _OLD_PB_ID_RE.pattern
    assert client.ENV_NAME_PATTERN.pattern == _OLD_PB_ENV_RE.pattern
    for text in _TEXTS:
        old = _outcome(lambda: adapter._text(text, where="w", pattern=_OLD_PB_ID_RE))
        new = _outcome(lambda: adapter._identifier(text, where="w"))
        assert new == old, (text, old, new)
        old_env = _OLD_PB_ENV_RE.fullmatch(text) is not None
        assert (client.ENV_NAME_PATTERN.fullmatch(text) is not None) == old_env, text


_DOCUMENTS = (
    {}, [], "text", 0, -1, 1.5, 1e300, True, None,
    {"b": 1, "a": [1, 2.0, "é", None, True]},
    {"nested": {"z": {"y": [{"x": "\u2603"}]}}, "emoji": "\U0001f600"},
    {"quote": "\"\\/\b\f\n\r\t", "ctl": "\u0001"},
    {"big": 2 ** 70, "neg": -(2 ** 70), "tiny": 5e-324},
    [{"k": v} for v in range(5)],
)


def test_adapter_identity_digest_is_the_old_bytes(client):
    from prismaquant import quality_prefill_pb_adapter as adapter
    from prismaquant.digests import canonical_json_sha256

    for document in _DOCUMENTS:
        old = canonical_json_sha256(document, where="quality-prefill document")
        assert adapter.canonical_sha256(document) == old, document
    for bad in (float("nan"), {"x": float("inf")}, {1, 2}, object()):
        old = _outcome(lambda: adapter._fail("document is not canonical JSON data: x"))
        new = _outcome(lambda: adapter.canonical_sha256(bad))
        assert new[0] == old[0], (bad, old, new)


def _runtime_tree(root: Path, *, old_halves: bool, client_tags):
    """A PB runtime tree: the files the old probe looked for, a client, both."""

    if old_halves:
        (root / "src/prismabuild").mkdir(parents=True, exist_ok=True)
        (root / "src/prismabuild/decomposition.py").write_text("")
        (root / "tools/fleet").mkdir(parents=True, exist_ok=True)
        (root / "tools/fleet/pbcampaign.py").write_text("def decompose(args):\n    pass\n")
    if client_tags is not None:
        from prismaquant.staged_lease import PB_CLIENT_SDK_VERSION
        package = root / "src/prismabuild"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("")
        (package / "client.py").write_text(textwrap.dedent(f"""\
            SDK_VERSION = {PB_CLIENT_SDK_VERSION}
            CAPABILITIES = frozenset({sorted(client_tags)!r})
        """))
    return root


def _old_decomposition_supported(root: Path) -> bool:
    """``decomposition_support(root)["supported"]`` at c36c02e1e76."""

    module = (root / "src/prismabuild/decomposition.py").is_file()
    tool = root / "tools/fleet/pbcampaign.py"
    entry = tool.is_file() and "def decompose(" in tool.read_text(
        encoding="utf-8", errors="replace")
    return bool(module and entry)


def test_decomposition_probe_agrees_on_every_tree_that_carries_the_client(tmp_path):
    from prismaquant import quality_prefill_pb_adapter as adapter

    trees = {
        "both": _runtime_tree(tmp_path / "both", old_halves=True,
                              client_tags={"reader-lease-v1", "progress-v1",
                                           "decomposition-v1"}),
        "neither": _runtime_tree(tmp_path / "neither", old_halves=False,
                                 client_tags={"reader-lease-v1", "progress-v1"}),
        "empty": tmp_path / "empty",
    }
    for label, root in trees.items():
        root.mkdir(parents=True, exist_ok=True)
        new = adapter.decomposition_support(root)
        assert new["supported"] == _old_decomposition_supported(root), (label, new)
    # The one tree the two probes read differently, recorded rather than
    # hidden: the files are there but the runtime publishes no client. The
    # capability is PB's to advertise, so the new probe says unsupported.
    transitional = _runtime_tree(tmp_path / "transitional", old_halves=True,
                                 client_tags=None)
    assert _old_decomposition_supported(transitional) is True
    answer = adapter.decomposition_support(transitional)
    assert answer["supported"] is False and answer["sdk_error"]
