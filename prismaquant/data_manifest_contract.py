"""Scientific read-manifest shape and byte-accounting checks, without campaign planning."""
from __future__ import annotations

import os

class ReadsetRefused(RuntimeError):
    """A campaign input, read set or data manifest the producer refuses.

    A domain error: the dispatcher and the experiment entry points turn it
    into their exit status, and library callers catch it like any other
    ``RuntimeError``.
    """


SCHEMA = "prismaquant.prismabuild.data_manifest.v1"
SCHEMA_V2 = "prismaquant.prismabuild.data_manifest.v2"
SHARED_MOUNT = "/mnt/shared"

#: ``prismabuild.core._DATA_MANIFEST_KEYS`` and ``_DATA_MANIFEST_ENTRY_KEYS``,
#: restated because PrismaBuild is not importable from the environments that
#: build a manifest.  ``validate_data_manifest`` builds both through
#: ``_exact_mapping``: an unknown key is a refusal, not an ignored extra.
MANIFEST_KEYS = frozenset({"schema", "produced_by", "mount_prefix", "entries",
                           "entry_count", "total_bytes", "annotations"})
ENTRY_KEYS = frozenset({"path", "offset", "bytes", "sha256"})

#: ``prismabuild.core.DATA_MANIFEST_MAX_ENTRIES`` and
#: ``DATA_MANIFEST_MAX_BYTES``: ``validate_data_manifest`` refuses a longer
#: entry list and ``load_data_manifest`` refuses a larger file.  A routed GLM
#: row is about 870 capture and weight entries plus a 5,920-file seed wire, so
#: roughly 6,800 entries and 2 MB -- well inside both, but the ceilings belong
#: here so a future read set that crosses one is refused by the producer
#: rather than by the fleet.
MAX_ENTRIES = 1_000_000
MAX_READS = 4_000_000
MAX_MANIFEST_BYTES = 64 * 1024 * 1024


def check_manifest(manifest: dict, *, where: str = "data manifest") -> dict:
    """Validate the retained scientific manifest shape before publication.

    These checks preserve the historical wire contract. The native launch
    boundary independently validates its supported scientific read contract.
    """
    if set(manifest) != set(MANIFEST_KEYS):
        raise ReadsetRefused(
            f"{where}: fields differ: "
            f"missing={sorted(MANIFEST_KEYS - set(manifest))}, "
            f"extra={sorted(set(manifest) - MANIFEST_KEYS)}")
    if manifest["schema"] != SCHEMA:
        raise ReadsetRefused(f"{where}: schema must be {SCHEMA}")
    for field in ("produced_by", "annotations"):
        if not isinstance(manifest[field], dict):
            raise ReadsetRefused(f"{where}: {field} must be an object")
    prefix = manifest["mount_prefix"]
    if not prefix.startswith("/") or prefix != os.path.normpath(prefix):
        raise ReadsetRefused(f"{where}: mount_prefix must be a normalized absolute path")
    if prefix == "/":
        raise ReadsetRefused(f"{where}: mount_prefix must name a mount, not the root")
    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries:
        raise ReadsetRefused(f"{where}: entries must be a non-empty array")
    if len(entries) > MAX_ENTRIES:
        raise ReadsetRefused(f"{where}: entries exceed {MAX_ENTRIES}")
    seen: set[tuple[str, int]] = set()
    total = 0
    for index, entry in enumerate(entries):
        at = f"{where} entries[{index}]"
        if set(entry) != set(ENTRY_KEYS):
            raise ReadsetRefused(f"{at}: fields differ")
        path, offset, size = entry["path"], entry["offset"], entry["bytes"]
        if not path.startswith(prefix + "/"):
            raise ReadsetRefused(f"{at}: {path} is outside {prefix}")
        if os.path.normpath(path) != path:
            raise ReadsetRefused(f"{at}: {path} is not normalized")
        if not isinstance(offset, int) or offset < 0:
            raise ReadsetRefused(f"{at}: offset must be a non-negative integer")
        if not isinstance(size, int) or size <= 0:
            raise ReadsetRefused(f"{at}: bytes must be positive")
        if (path, offset) in seen:
            raise ReadsetRefused(f"{at}: repeats a (path, offset)")
        seen.add((path, offset))
        total += size
    if manifest["entry_count"] != len(entries):
        raise ReadsetRefused(f"{where}: entry_count disagrees with entries")
    if manifest["total_bytes"] != total:
        raise ReadsetRefused(f"{where}: total_bytes disagrees with entries")
    return manifest


def check_manifest_bytes(blob: bytes, *, where: str = "data manifest") -> bytes:
    """Refuse an oversized scientific manifest before publication.

    ``load_data_manifest`` stats the file before it parses it, so an oversized
    manifest fails at submission with nothing validated.  Checking the bytes
    the producer is about to write keeps that refusal here.
    """
    if len(blob) > MAX_MANIFEST_BYTES:
        raise ReadsetRefused(
            f"{where}: manifest file is {len(blob)} bytes, over the "
            f"{MAX_MANIFEST_BYTES}-byte limit")
    return blob


def check_manifest_v2(manifest: dict, *, where: str = "data manifest V2") -> dict:
    """Validate V2 unique entries and ordered read coverage."""
    if set(manifest) != MANIFEST_KEYS | {"read_plan"} or manifest.get("schema") != SCHEMA_V2:
        raise ReadsetRefused(f"{where}: complete {SCHEMA_V2} fields required")
    common = {key: value for key, value in manifest.items() if key != "read_plan"}
    common["schema"] = SCHEMA
    check_manifest(common, where=where)
    if "phases" in manifest["annotations"]:
        raise ReadsetRefused(f"{where}: V2 uses read_plan, not annotations.phases")
    plan = manifest["read_plan"]
    if not isinstance(plan, dict) or set(plan) != {"phases", "read_bytes"}:
        raise ReadsetRefused(f"{where}: complete read_plan required")
    phases = plan["phases"]
    if not isinstance(phases, list) or not phases:
        raise ReadsetRefused(f"{where}: read_plan needs nonempty phases")
    names, used, cumulative, reads = set(), set(), 0, 0
    for phase in phases:
        if not isinstance(phase, dict) or set(phase) != {
                "name", "entry_indices", "bytes", "cumulative_bytes"}:
            raise ReadsetRefused(f"{where}: malformed read phase")
        name, indices = phase["name"], phase["entry_indices"]
        if (not isinstance(name, str) or not name or name.strip() != name
                or "\x00" in name or name in names or not isinstance(indices, list)):
            raise ReadsetRefused(f"{where}: invalid or repeated read phase name/indices")
        names.add(name)
        reads += len(indices)
        if reads > MAX_READS:
            raise ReadsetRefused(f"{where}: read references exceed {MAX_READS}")
        local, size = set(), 0
        for index in indices:
            if (type(index) is not int or index < 0 or index >= len(manifest["entries"])
                    or index in local):
                raise ReadsetRefused(f"{where}: invalid or repeated phase entry index")
            local.add(index)
            used.add(index)
            size += manifest["entries"][index]["bytes"]
        cumulative += size
        if (type(phase["bytes"]) is not int or phase["bytes"] != size
                or type(phase["cumulative_bytes"]) is not int
                or phase["cumulative_bytes"] != cumulative):
            raise ReadsetRefused(f"{where}: phase byte accounting differs")
    if used != set(range(len(manifest["entries"]))) or type(plan["read_bytes"]) is not int or plan["read_bytes"] != cumulative:
        raise ReadsetRefused(f"{where}: read_plan coverage or read_bytes differs")
    return manifest
