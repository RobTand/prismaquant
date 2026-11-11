"""No-clobber byte publication for Tessera artifacts."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile


class ClusterCampaignV2Error(RuntimeError):
    """Base error retained for artifact publication callers."""


class CampaignContractError(ClusterCampaignV2Error, ValueError):
    """An artifact publication conflicts with an existing output."""


def _atomic_write_new_bytes(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise CampaignContractError(f"output already exists: {path}")
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CampaignContractError(f"output appeared concurrently: {path}") from exc
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
