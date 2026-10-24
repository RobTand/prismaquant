# Original source fragments from 6d8288adc94b3c4b624c143ed372fb6513f32784:tools/build_prismasnap_27b_campaign.py
# Domain dependencies are injected by the behavioral tests.
from __future__ import annotations
import json
import hashlib
from pathlib import Path

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
        raise CampaignBuildError("value is not finite canonical JSON") from exc

def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise CampaignBuildError(f"expected one regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def _rootfs_layers_sha256(layers: Sequence[str]) -> str:
    encoded = _canonical_bytes(list(layers)) + b"\n"
    return hashlib.sha256(encoded).hexdigest()


def stdout_json(payload):
    return json.dumps(payload, sort_keys=True, allow_nan=False)
