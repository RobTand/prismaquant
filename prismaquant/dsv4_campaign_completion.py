"""Fail-closed terminal receipt for the DSv4 anchored-AURA campaign.

The live campaign is a transient user-systemd service.  Counting unit files is
not a completion signal: the last journal can land before the monolithic
payload, CPU tail, and service process have completed, and an inactive service
may have failed.  This module binds one already-running systemd invocation,
requires its exact main process to exit successfully, audits the complete AURA
journal/payload closure, and repeats the historical layer-42..38 payload
comparison before publishing one self-hashed receipt.

The activation-safe replay consumes that receipt as an admission token and
then independently performs its stronger scalar/source-identity reconstruction.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import stat
import sys
from typing import Any


RECEIPT_SCHEMA = "prismaquant.dsv4_aura_campaign_completion.v1"
PRODUCER_SCHEMA = "prismaquant.dsv4_aura_campaign_completion.producer.v1"
SERVICE_SCHEMA = "prismaquant.systemd_terminal_success.v1"
AURA_CHECKPOINT_IDENTITY_SCHEMA = "prismaquant.aura_checkpoint.identity.v1"
AURA_CHECKPOINT_MANIFEST_SCHEMA = "prismaquant.aura_checkpoint.manifest.v1"
AURA_CHECKPOINT_UNIT_SCHEMA = "prismaquant.aura_checkpoint.unit.v1"
STREAMED_PAYLOAD_SCHEMA = "prismaquant.aura_cost.v1"
DEFAULT_SERVICE_UNIT = "pq-aura-dsv4-streamed-cached.service"
DEFAULT_RECEIPT_NAME = "campaign_completion_receipt.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_INVOCATION = re.compile(r"[0-9a-f]{32}")


class CampaignCompletionError(RuntimeError):
    """The service or completed campaign cannot authorize replay."""


@dataclass(frozen=True)
class CompletionContract:
    expected_unit_count: int
    overlap_layers: tuple[int, ...]
    units_per_overlap_layer: int

    def to_dict(self) -> dict[str, object]:
        return {
            "expected_unit_count": self.expected_unit_count,
            "overlap_layers": list(self.overlap_layers),
            "units_per_overlap_layer": self.units_per_overlap_layer,
        }


DSV4_COMPLETION_CONTRACT = CompletionContract(
    expected_unit_count=33_325,
    overlap_layers=(42, 41, 40, 39, 38),
    units_per_overlap_layer=775,
)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CampaignCompletionError("value is not canonical JSON data") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _reject_duplicate_members(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    # Deliberate standalone twin of prismaquant.schemas.unique_json_object:
    # wait_dsv4_aura_campaign loads this file without the package so terminal
    # receipt publication stays off the GPU runtime dependency set, and
    # load_completion_receipt reaches _load_json there -- even a lazy package
    # import would couple it to torch. Same loop, same refusal.
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignCompletionError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_members,
            parse_constant=lambda item: (_ for _ in ()).throw(
                CampaignCompletionError(f"non-finite JSON value {item}")
            ),
        )
    except CampaignCompletionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignCompletionError(f"unreadable JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise CampaignCompletionError(f"JSON file is not an object: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_bytes(path: Path) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise CampaignCompletionError(f"required file is absent: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise CampaignCompletionError(f"required path is not one real file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CampaignCompletionError(f"required file is unreadable: {path}") from exc


def file_descriptor(path: Path) -> dict[str, object]:
    encoded = _file_bytes(path)
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": len(encoded),
        "sha256": _sha256_bytes(encoded),
    }


def completion_receipt_path(work_dir: str | os.PathLike[str]) -> Path:
    return Path(work_dir) / "artifacts" / DEFAULT_RECEIPT_NAME


def historical_checkpoint_root(work_dir: str | os.PathLike[str]) -> Path:
    return Path(work_dir).resolve(strict=False).parent / "aura-cb-reprice" / (
        "checkpoints/aura"
    )


def _manifest_scope(
    checkpoint_root: Path,
    *,
    expected_unit_count: int,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, object]]:
    if checkpoint_root.is_symlink():
        raise CampaignCompletionError(
            f"checkpoint root is not one real directory: {checkpoint_root}"
        )
    root = checkpoint_root.resolve(strict=True)
    if not root.is_dir():
        raise CampaignCompletionError(
            f"checkpoint root is not one real directory: {root}"
        )
    manifest_path = root / "manifest.json"
    manifest_bytes = _file_bytes(manifest_path)
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != AURA_CHECKPOINT_MANIFEST_SCHEMA:
        raise CampaignCompletionError("AURA checkpoint manifest schema differs")
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping) or (
        identity.get("schema") != AURA_CHECKPOINT_IDENTITY_SCHEMA
    ):
        raise CampaignCompletionError("AURA checkpoint identity is invalid")
    identity_sha256 = canonical_sha256(identity)
    if manifest.get("identity_sha256") != identity_sha256:
        raise CampaignCompletionError("AURA checkpoint identity checksum differs")
    rows = manifest.get("units")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise CampaignCompletionError("AURA checkpoint unit table is invalid")
    if len(rows) != expected_unit_count:
        raise CampaignCompletionError(
            f"AURA manifest has {len(rows)} units, expected {expected_unit_count}"
        )
    paths: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise CampaignCompletionError("AURA manifest unit is not an object")
        qname = str(row.get("qname", ""))
        expected_file = (
            "units/" + hashlib.sha256(qname.encode("utf-8")).hexdigest() + ".pkl"
        )
        if (
            not qname
            or qname in paths
            or row.get("file") != expected_file
        ):
            raise CampaignCompletionError(
                f"AURA manifest unit binding differs for {qname!r}"
            )
        paths[qname] = root / expected_file
    return manifest, paths, {
        "path": str(manifest_path.resolve()),
        "size_bytes": len(manifest_bytes),
        "sha256": _sha256_bytes(manifest_bytes),
        "identity_sha256": identity_sha256,
    }


def _unit_envelope(
    path: Path,
    *,
    qname: str,
    identity_sha256: str,
) -> tuple[bytes, dict[str, str]]:
    encoded = _file_bytes(path)
    try:
        envelope = pickle.loads(encoded)
    except Exception as exc:
        raise CampaignCompletionError(
            f"AURA unit envelope is corrupt for {qname}: {path}"
        ) from exc
    if not isinstance(envelope, Mapping):
        raise CampaignCompletionError(f"AURA unit is not an envelope for {qname}")
    payload = envelope.get("payload")
    if not isinstance(payload, bytes):
        raise CampaignCompletionError(f"AURA unit has no payload for {qname}")
    payload_sha256 = _sha256_bytes(payload)
    if (
        envelope.get("schema") != AURA_CHECKPOINT_UNIT_SCHEMA
        or envelope.get("qname") != qname
        or envelope.get("identity_sha256") != identity_sha256
        or envelope.get("payload_sha256") != payload_sha256
    ):
        raise CampaignCompletionError(
            f"AURA unit envelope identity/checksum differs for {qname}"
        )
    try:
        state = pickle.loads(payload)
    except Exception as exc:
        raise CampaignCompletionError(
            f"AURA unit payload is corrupt for {qname}"
        ) from exc
    if not isinstance(state, Mapping) or not isinstance(state.get("rows"), Mapping):
        raise CampaignCompletionError(f"AURA unit payload is invalid for {qname}")
    return payload, {
        "qname": qname,
        "envelope_sha256": _sha256_bytes(encoded),
        "payload_sha256": payload_sha256,
    }


def _overlap_qnames(
    paths: Mapping[str, Path], contract: CompletionContract,
) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for layer in contract.overlap_layers:
        prefix = f"model.layers.{layer}."
        names = sorted(name for name in paths if name.startswith(prefix))
        if len(names) != contract.units_per_overlap_layer:
            raise CampaignCompletionError(
                f"layer {layer} has {len(names)} AURA units, expected "
                f"{contract.units_per_overlap_layer}"
            )
        result[layer] = names
    return result


def audit_campaign_closure(
    work_dir: str | os.PathLike[str],
    historical_root: str | os.PathLike[str],
    *,
    contract: CompletionContract = DSV4_COMPLETION_CONTRACT,
) -> dict[str, object]:
    """Re-hash the completed live journal/payload and historical overlap."""
    work = Path(work_dir).resolve(strict=True)
    current_root = work / "checkpoints" / "aura"
    payload_path = work / "artifacts" / "streamed_anchor_aura.pkl"
    manifest, paths, manifest_descriptor = _manifest_scope(
        current_root, expected_unit_count=contract.expected_unit_count
    )
    units_dir = current_root / "units"
    if units_dir.is_symlink() or not units_dir.is_dir():
        raise CampaignCompletionError("AURA unit directory is absent or unsafe")
    actual_files = set(units_dir.iterdir())
    expected_files = set(paths.values())
    if actual_files != expected_files:
        raise CampaignCompletionError(
            "AURA unit directory does not contain exactly the manifest closure"
        )

    overlap_names = _overlap_qnames(paths, contract)
    overlap_name_set = {
        name for names in overlap_names.values() for name in names
    }
    current_overlap_payloads: dict[str, bytes] = {}
    unit_descriptors: list[dict[str, str]] = []
    identity_sha256 = str(manifest["identity_sha256"])
    for qname in sorted(paths):
        payload, descriptor = _unit_envelope(
            paths[qname], qname=qname, identity_sha256=identity_sha256
        )
        unit_descriptors.append(descriptor)
        if qname in overlap_name_set:
            current_overlap_payloads[qname] = payload

    monolithic_bytes = _file_bytes(payload_path)
    try:
        monolithic = pickle.loads(monolithic_bytes)
    except Exception as exc:
        raise CampaignCompletionError(
            f"completed streamed payload is corrupt: {payload_path}"
        ) from exc
    if not isinstance(monolithic, Mapping):
        raise CampaignCompletionError("completed streamed payload is not a mapping")
    costs = monolithic.get("costs")
    stats = monolithic.get("stats")
    provenance = monolithic.get("provenance")
    expected_names = set(paths)
    if (
        monolithic.get("schema") != STREAMED_PAYLOAD_SCHEMA
        or not isinstance(costs, Mapping)
        or not isinstance(stats, Mapping)
        or not isinstance(provenance, Mapping)
        or set(map(str, costs)) != expected_names
        or set(map(str, stats)) != expected_names
    ):
        raise CampaignCompletionError(
            "completed streamed payload does not exactly cover the journal scope"
        )

    old_manifest, old_paths, old_manifest_descriptor = _manifest_scope(
        Path(historical_root), expected_unit_count=contract.expected_unit_count
    )
    old_overlap_names = _overlap_qnames(old_paths, contract)
    if old_overlap_names != overlap_names:
        raise CampaignCompletionError("historical/live overlap qname scopes differ")
    old_identity_sha256 = str(old_manifest["identity_sha256"])
    overlap_descriptors: list[dict[str, str]] = []
    for layer in contract.overlap_layers:
        for qname in overlap_names[layer]:
            old_payload, old_descriptor = _unit_envelope(
                old_paths[qname],
                qname=qname,
                identity_sha256=old_identity_sha256,
            )
            current_payload = current_overlap_payloads[qname]
            if old_payload != current_payload:
                raise CampaignCompletionError(
                    f"historical payload differs for layer {layer}: {qname}"
                )
            overlap_descriptors.append({
                "qname": qname,
                "payload_sha256": old_descriptor["payload_sha256"],
            })

    payload_descriptor = {
        "path": str(payload_path.resolve()),
        "size_bytes": len(monolithic_bytes),
        "sha256": _sha256_bytes(monolithic_bytes),
    }
    qnames = sorted(paths)
    return {
        "work_dir": str(work),
        "checkpoint_root": str(current_root.resolve()),
        "checkpoint_manifest": manifest_descriptor,
        "streamed_payload": payload_descriptor,
        "unit_count": len(unit_descriptors),
        "qname_set_sha256": canonical_sha256(qnames),
        "unit_envelope_set_sha256": canonical_sha256(unit_descriptors),
        "unit_payload_set_sha256": canonical_sha256([
            {
                "qname": row["qname"],
                "payload_sha256": row["payload_sha256"],
            }
            for row in unit_descriptors
        ]),
        "historical_overlap": {
            "checkpoint_root": str(Path(historical_root).resolve(strict=True)),
            "checkpoint_manifest": old_manifest_descriptor,
            "layers": list(contract.overlap_layers),
            "units_per_layer": contract.units_per_overlap_layer,
            "unit_count": len(overlap_descriptors),
            "exact_payload_match_count": len(overlap_descriptors),
            "payload_pair_set_sha256": canonical_sha256(overlap_descriptors),
        },
    }


def validate_terminal_service_evidence(value: Mapping[str, object]) -> None:
    if value.get("schema") != SERVICE_SCHEMA:
        raise CampaignCompletionError("systemd terminal evidence schema differs")
    invocation = value.get("invocation_id")
    main_pid = value.get("main_pid")
    main_pid_start_ticks = value.get("main_pid_start_ticks")
    if not isinstance(invocation, str) or _INVOCATION.fullmatch(invocation) is None:
        raise CampaignCompletionError("systemd invocation id is invalid")
    if isinstance(main_pid, bool) or not isinstance(main_pid, int) or main_pid <= 1:
        raise CampaignCompletionError("systemd main PID is invalid")
    if (
        isinstance(main_pid_start_ticks, bool)
        or not isinstance(main_pid_start_ticks, int)
        or main_pid_start_ticks <= 0
    ):
        raise CampaignCompletionError("systemd main PID start time is invalid")
    for key in (
        "exec_main_code", "exec_main_status", "exec_main_pid", "n_restarts"
    ):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int):
            raise CampaignCompletionError(
                f"systemd terminal evidence {key} is not an integer"
            )
    required = {
        "unit": DEFAULT_SERVICE_UNIT,
        "initial_active_state": "active",
        "initial_sub_state": "running",
        "restart_policy": "no",
        "terminal_active_state": "inactive",
        "terminal_sub_state": "dead",
        "result": "success",
        "exec_main_code": 1,  # CLD_EXITED
        "exec_main_status": 0,
        "exec_main_pid": main_pid,
        "n_restarts": 0,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise CampaignCompletionError(
                f"systemd terminal evidence differs at {key}: "
                f"{value.get(key)!r} != {expected!r}"
            )
    if value.get("initial_invocation_id") != invocation:
        raise CampaignCompletionError("systemd invocation changed while waiting")


def validate_producer_identity(value: Mapping[str, object]) -> None:
    if value.get("schema") != PRODUCER_SCHEMA:
        raise CampaignCompletionError("completion producer schema differs")
    if not isinstance(value.get("commit"), str) or _COMMIT.fullmatch(
        str(value.get("commit"))
    ) is None:
        raise CampaignCompletionError("completion producer commit is invalid")
    for key in ("tree", "closure_sha256"):
        raw = value.get(key)
        expected = _COMMIT if key == "tree" else _SHA256
        if not isinstance(raw, str) or expected.fullmatch(raw) is None:
            raise CampaignCompletionError(
                f"completion producer {key} is invalid"
            )
    snapshot = value.get("snapshot")
    if not isinstance(snapshot, str) or not Path(snapshot).is_absolute():
        raise CampaignCompletionError("completion producer snapshot is invalid")


def build_completion_receipt(
    *,
    service: Mapping[str, object],
    producer: Mapping[str, object],
    campaign: Mapping[str, object],
    contract: CompletionContract = DSV4_COMPLETION_CONTRACT,
) -> dict[str, object]:
    validate_terminal_service_evidence(service)
    validate_producer_identity(producer)
    body: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": contract.to_dict(),
        "service": dict(service),
        "producer": dict(producer),
        "campaign": dict(campaign),
    }
    body["receipt_sha256"] = canonical_sha256(body)
    return body


def load_completion_receipt(
    path: str | os.PathLike[str],
    *,
    contract: CompletionContract = DSV4_COMPLETION_CONTRACT,
) -> dict[str, Any]:
    receipt = _load_json(Path(path))
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise CampaignCompletionError("campaign completion receipt schema differs")
    observed_hash = receipt.get("receipt_sha256")
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    expected_hash = canonical_sha256(body)
    if observed_hash != expected_hash:
        raise CampaignCompletionError("campaign completion receipt self-hash differs")
    if receipt.get("contract") != contract.to_dict():
        raise CampaignCompletionError("campaign completion contract differs")
    service = receipt.get("service")
    producer = receipt.get("producer")
    campaign = receipt.get("campaign")
    if not isinstance(service, Mapping) or not isinstance(producer, Mapping):
        raise CampaignCompletionError("completion receipt identity blocks are invalid")
    if not isinstance(campaign, Mapping):
        raise CampaignCompletionError("completion receipt campaign block is invalid")
    validate_terminal_service_evidence(service)
    validate_producer_identity(producer)
    return receipt


def _atomic_publish_new(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise CampaignCompletionError(
            f"completion receipt already exists and will not be replaced: {path}"
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CampaignCompletionError(
                f"completion receipt appeared concurrently: {path}"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish_completion_receipt(
    path: str | os.PathLike[str],
    receipt: Mapping[str, object],
    *,
    contract: CompletionContract = DSV4_COMPLETION_CONTRACT,
) -> Path:
    output = Path(path)
    encoded = json.dumps(
        dict(receipt), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    _atomic_publish_new(output, encoded)
    loaded = load_completion_receipt(output, contract=contract)
    if loaded != dict(receipt):
        raise CampaignCompletionError("published completion receipt differs")
    return output


def verify_receipt_against_current_campaign(
    receipt_path: str | os.PathLike[str],
    *,
    work_dir: str | os.PathLike[str],
    historical_root: str | os.PathLike[str],
    expected_producer_commit: str | None = None,
    contract: CompletionContract = DSV4_COMPLETION_CONTRACT,
) -> dict[str, Any]:
    receipt = load_completion_receipt(receipt_path, contract=contract)
    if expected_producer_commit is not None and (
        receipt["producer"].get("commit") != expected_producer_commit
    ):
        raise CampaignCompletionError(
            "completion receipt was produced by a different PrismaQuant commit"
        )
    observed = audit_campaign_closure(
        work_dir, historical_root, contract=contract
    )
    if receipt.get("campaign") != observed:
        raise CampaignCompletionError(
            "current campaign/journal/overlap closure differs from its receipt"
        )
    return receipt


def verify_receipt_for_replay(
    receipt_path: str | os.PathLike[str],
    *,
    work_dir: str | os.PathLike[str],
    historical_root: str | os.PathLike[str],
    expected_producer_commit: str | None = None,
    contract: CompletionContract = DSV4_COMPLETION_CONTRACT,
) -> dict[str, Any]:
    """Cheap pre-replay check; the replay re-hashes all unit payloads itself."""
    receipt = load_completion_receipt(receipt_path, contract=contract)
    if expected_producer_commit is not None and (
        receipt["producer"].get("commit") != expected_producer_commit
    ):
        raise CampaignCompletionError(
            "completion receipt was produced by a different PrismaQuant commit"
        )
    campaign = receipt["campaign"]
    work = Path(work_dir).resolve(strict=True)
    # The historical tree was re-hashed by the terminal waiter.  CPU replay
    # mounts the completed campaign/receipt read-only but intentionally does
    # not mount the historical campaign, so bind its canonical path without
    # requiring it to be visible in that container.
    old_root = Path(historical_root).resolve(strict=False)
    if (
        campaign.get("work_dir") != str(work)
        or campaign.get("checkpoint_root")
        != str((work / "checkpoints" / "aura").resolve(strict=True))
        or (campaign.get("historical_overlap") or {}).get("checkpoint_root")
        != str(old_root)
    ):
        raise CampaignCompletionError("completion receipt path binding differs")
    for key, path in (
        ("checkpoint_manifest", work / "checkpoints" / "aura" / "manifest.json"),
        ("streamed_payload", work / "artifacts" / "streamed_anchor_aura.pkl"),
    ):
        observed = file_descriptor(path)
        expected = campaign.get(key)
        if not isinstance(expected, Mapping):
            raise CampaignCompletionError(f"receipt lacks {key} descriptor")
        for field in ("path", "size_bytes", "sha256"):
            if expected.get(field) != observed.get(field):
                raise CampaignCompletionError(
                    f"completion receipt {key}.{field} differs"
                )
    return receipt


def assert_replay_matches_completion_receipt(
    replay: Mapping[str, object], receipt: Mapping[str, object]
) -> None:
    campaign = receipt.get("campaign")
    if not isinstance(campaign, Mapping):
        raise CampaignCompletionError("completion receipt campaign is invalid")
    expected = {
        "source_payload": campaign.get("streamed_payload"),
        "checkpoint_manifest": campaign.get("checkpoint_manifest"),
        "unit_checkpoint_count": campaign.get("unit_count"),
        "unit_payload_set_sha256": campaign.get("unit_payload_set_sha256"),
    }
    for key, value in expected.items():
        observed = replay.get(key)
        if key in {"source_payload", "checkpoint_manifest"}:
            if not isinstance(observed, Mapping) or not isinstance(value, Mapping):
                raise CampaignCompletionError(
                    f"replay/receipt descriptor {key} is invalid"
                )
            fields = ("path", "size_bytes", "sha256")
            if key == "checkpoint_manifest":
                fields = (*fields, "identity_sha256")
            if any(observed.get(field) != value.get(field) for field in fields):
                raise CampaignCompletionError(
                    f"activation-safe replay differs from completion receipt at {key}"
                )
        elif observed != value:
            raise CampaignCompletionError(
                f"activation-safe replay differs from completion receipt at {key}"
            )


def _invocation_hex(value: object) -> str:
    try:
        encoded = bytes(int(item) for item in value)  # type: ignore[arg-type]
    except Exception as exc:
        raise CampaignCompletionError("systemd invocation id is unreadable") from exc
    if len(encoded) != 16:
        raise CampaignCompletionError("systemd invocation id is not 128-bit")
    return encoded.hex()


def _pid_start_ticks(pid: int) -> int:
    try:
        # The command name is parenthesized and may contain spaces.  Split only
        # after the final ')'; field 22 is index 19 in the remaining fields.
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2 :].split()
        return int(tail[19])
    except Exception as exc:
        raise CampaignCompletionError(
            f"cannot bind systemd MainPID {pid} to its /proc start time"
        ) from exc


def _systemd_dbus_dependencies() -> tuple[object, object, object]:
    try:
        import dbus  # type: ignore[import-not-found]
        from dbus.mainloop.glib import DBusGMainLoop  # type: ignore[import-not-found]
        from gi.repository import GLib  # type: ignore[import-not-found]
    except Exception as exc:
        raise CampaignCompletionError(
            "python3-dbus and PyGObject are required for race-free systemd waiting"
        ) from exc
    return dbus, DBusGMainLoop, GLib


def _wait_for_referenced_systemd_service(
    *,
    manager: object,
    bus: object,
    dbus: object,
    GLib: object,
    unit: str,
    timeout_seconds: int,
) -> dict[str, object]:
    """Capture terminal evidence while the caller holds Manager.RefUnit."""
    try:
        object_path = str(manager.GetUnit(unit))
    except Exception as exc:
        raise CampaignCompletionError(
            f"campaign service is not loaded and active: {unit}"
        ) from exc
    service_object = bus.get_object("org.freedesktop.systemd1", object_path)
    properties = dbus.Interface(
        service_object, "org.freedesktop.DBus.Properties"
    )
    cache: dict[str, object] = {}
    terminal = {"seen": False, "quit_scheduled": False}
    failure: list[BaseException] = []
    loop = GLib.MainLoop()

    def finish_terminal() -> bool:
        # systemd can publish Unit and Service property changes as separate
        # signals.  Refresh once after the terminal Unit signal so signal
        # ordering cannot leave stale Result/ExecMain* evidence in the receipt.
        try:
            refreshed_unit = properties.GetAll("org.freedesktop.systemd1.Unit")
            refreshed_service = properties.GetAll("org.freedesktop.systemd1.Service")
        except Exception as exc:
            failure.append(CampaignCompletionError(
                "campaign terminal properties could not be refreshed while "
                "holding its systemd unit reference"
            ))
        else:
            cache.update({str(key): value for key, value in refreshed_unit.items()})
            cache.update({str(key): value for key, value in refreshed_service.items()})
        loop.quit()
        return False

    def changed(interface: object, values: Mapping[object, object], _invalid: object) -> None:
        for key, value in values.items():
            cache[str(key)] = value
        if str(interface) == "org.freedesktop.systemd1.Unit" and str(
            values.get("ActiveState", "")
        ) in {"inactive", "failed"}:
            terminal["seen"] = True
            if not terminal["quit_scheduled"]:
                terminal["quit_scheduled"] = True
                GLib.timeout_add(100, finish_terminal)

    def removed(raw_unit: object, _path: object) -> None:
        if str(raw_unit) == unit and not terminal["seen"]:
            failure.append(
                CampaignCompletionError(
                    "campaign unit disappeared before a terminal state was captured"
                )
            )
            loop.quit()

    bus.add_signal_receiver(
        changed,
        signal_name="PropertiesChanged",
        dbus_interface="org.freedesktop.DBus.Properties",
        path=object_path,
    )
    bus.add_signal_receiver(
        removed,
        signal_name="UnitRemoved",
        dbus_interface="org.freedesktop.systemd1.Manager",
        path="/org/freedesktop/systemd1",
    )
    try:
        unit_values = properties.GetAll("org.freedesktop.systemd1.Unit")
        service_values = properties.GetAll("org.freedesktop.systemd1.Service")
    except Exception as exc:
        raise CampaignCompletionError(
            "campaign service changed before its invocation could be bound"
        ) from exc
    cache.update({str(key): value for key, value in unit_values.items()})
    cache.update({str(key): value for key, value in service_values.items()})
    initial_active = str(cache.get("ActiveState", ""))
    initial_sub = str(cache.get("SubState", ""))
    main_pid = int(cache.get("MainPID", 0))
    invocation = _invocation_hex(cache.get("InvocationID"))
    restart = str(cache.get("Restart", ""))
    n_restarts = int(cache.get("NRestarts", -1))
    if (
        initial_active != "active"
        or initial_sub != "running"
        or main_pid <= 1
        or restart != "no"
        or n_restarts != 0
    ):
        raise CampaignCompletionError(
            "campaign service is not one active, non-restarting invocation"
        )
    start_ticks = _pid_start_ticks(main_pid)

    def timeout() -> bool:
        failure.append(CampaignCompletionError("campaign service wait timed out"))
        loop.quit()
        return False

    timeout_source = GLib.timeout_add_seconds(timeout_seconds, timeout)
    loop.run()
    if timeout_source and not failure:
        GLib.source_remove(timeout_source)
    if failure:
        raise failure[0]
    terminal_invocation = _invocation_hex(cache.get("InvocationID"))
    evidence = {
        "schema": SERVICE_SCHEMA,
        "unit": unit,
        "object_path": object_path,
        "invocation_id": terminal_invocation,
        "initial_invocation_id": invocation,
        "main_pid": main_pid,
        "main_pid_start_ticks": start_ticks,
        "initial_active_state": initial_active,
        "initial_sub_state": initial_sub,
        "restart_policy": restart,
        "terminal_active_state": str(cache.get("ActiveState", "")),
        "terminal_sub_state": str(cache.get("SubState", "")),
        "result": str(cache.get("Result", "")),
        "exec_main_code": int(cache.get("ExecMainCode", -1)),
        "exec_main_status": int(cache.get("ExecMainStatus", -1)),
        "exec_main_pid": int(cache.get("ExecMainPID", 0)),
        "n_restarts": int(cache.get("NRestarts", -1)),
    }
    validate_terminal_service_evidence(evidence)
    return evidence


def wait_for_bound_systemd_service(
    unit: str = DEFAULT_SERVICE_UNIT,
    *,
    timeout_seconds: int = 12 * 60 * 60,
) -> dict[str, object]:
    """Subscribe to and hold one active invocation through terminal audit."""
    if unit != DEFAULT_SERVICE_UNIT:
        raise CampaignCompletionError(
            f"release waiter accepts only {DEFAULT_SERVICE_UNIT}"
        )
    if timeout_seconds <= 0:
        raise CampaignCompletionError("systemd wait timeout must be positive")
    dbus, DBusGMainLoop, GLib = _systemd_dbus_dependencies()
    DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()
    manager_object = bus.get_object(
        "org.freedesktop.systemd1", "/org/freedesktop/systemd1"
    )
    manager = dbus.Interface(
        manager_object, "org.freedesktop.systemd1.Manager"
    )
    subscribed = False
    referenced = False
    cleanup_errors: list[str] = []
    try:
        manager.Subscribe()
        subscribed = True
        try:
            # RefUnit must precede even GetUnit: CollectMode may otherwise
            # unload this transient unit in the terminal signal/property gap.
            manager.RefUnit(unit)
        except Exception as exc:
            raise CampaignCompletionError(
                f"could not retain campaign service identity: {unit}"
            ) from exc
        referenced = True
        return _wait_for_referenced_systemd_service(
            manager=manager,
            bus=bus,
            dbus=dbus,
            GLib=GLib,
            unit=unit,
            timeout_seconds=timeout_seconds,
        )
    finally:
        if referenced:
            try:
                manager.UnrefUnit(unit)
            except Exception as exc:
                cleanup_errors.append(f"UnrefUnit: {exc}")
        if subscribed:
            try:
                manager.Unsubscribe()
            except Exception as exc:
                cleanup_errors.append(f"Unsubscribe: {exc}")
        if cleanup_errors and sys.exc_info()[0] is None:
            raise CampaignCompletionError(
                "systemd waiter cleanup failed: " + "; ".join(cleanup_errors)
            )


__all__ = [
    "CampaignCompletionError",
    "CompletionContract",
    "DEFAULT_RECEIPT_NAME",
    "DEFAULT_SERVICE_UNIT",
    "DSV4_COMPLETION_CONTRACT",
    "PRODUCER_SCHEMA",
    "RECEIPT_SCHEMA",
    "SERVICE_SCHEMA",
    "assert_replay_matches_completion_receipt",
    "audit_campaign_closure",
    "build_completion_receipt",
    "completion_receipt_path",
    "historical_checkpoint_root",
    "load_completion_receipt",
    "publish_completion_receipt",
    "verify_receipt_against_current_campaign",
    "verify_receipt_for_replay",
    "wait_for_bound_systemd_service",
]
