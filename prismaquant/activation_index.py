"""Disk-backed activation index shared by cost measurement and export."""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import re
import stat

import torch

from .tensor_digests import tensor_identity


class ActivationIndex:
    """Disk-backed activation cache.

    Building the index walks the cache dir once to map Linear name → path,
    but no tensor data is read until `load()` is called for a specific name.
    This keeps resident memory small even when the cache is 20 GB on disk.

    The probe writes files as `re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"`,
    so we apply the same forward transform to each candidate name from the
    probe stats and check for the file. This avoids ambiguity if a name ever
    contained characters that collapse under the substitution.
    """

    _FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")

    def __init__(
        self,
        cache_dir: Path,
        candidate_names,
        *,
        verification_contract: Mapping[str, object] | None = None,
    ):
        self.cache_dir = cache_dir
        self._paths: dict[str, Path] = {}
        self._aliases: dict[str, str] = {}
        self._verified_dir_fd: int | None = None
        self._verification_records: dict[str, Mapping[str, object]] | None = None
        self._verification_schema: str | None = None
        self._verification_priority_schema: str | None = None
        self._verification_cover_sha256: str | None = None

        if verification_contract is not None:
            records = verification_contract.get("records")
            schema = verification_contract.get("schema")
            priority_schema = verification_contract.get("priority_schema")
            cover_sha256 = verification_contract.get(
                "cover_identity_sha256"
            )
            if (
                not isinstance(records, Mapping)
                or not isinstance(schema, str)
                or not isinstance(priority_schema, str)
                or not isinstance(cover_sha256, str)
            ):
                raise ValueError(
                    "activation verification contract is malformed"
                )
            if any(
                not isinstance(name, str) or not isinstance(record, Mapping)
                for name, record in records.items()
            ):
                raise ValueError(
                    "activation verification records are malformed"
                )
            before = self.cache_dir.lstat()
            if (
                not stat.S_ISDIR(before.st_mode)
                or self.cache_dir.is_symlink()
            ):
                raise ValueError(
                    "verified activation cache must be a real directory"
                )
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(self.cache_dir, flags)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or int(opened.st_dev) != int(before.st_dev)
                or int(opened.st_ino) != int(before.st_ino)
            ):
                os.close(descriptor)
                raise ValueError(
                    "verified activation cache changed while opening"
                )
            self._verified_dir_fd = descriptor
            self._verification_records = {
                str(name): dict(record) for name, record in records.items()
            }
            self._verification_schema = schema
            self._verification_priority_schema = priority_schema
            self._verification_cover_sha256 = cover_sha256

        def _add_name(name: str):
            fname = self._FNAME_SUB.sub("__", name) + ".pt"
            fp = self.cache_dir / fname
            if self._verification_records is not None:
                if name in self._verification_records:
                    self._paths[name] = fp
            elif fp.is_file():
                self._paths[name] = fp

        if isinstance(candidate_names, dict):
            for name, meta in candidate_names.items():
                _add_name(str(name))
                if isinstance(meta, dict):
                    experts_qname = meta.get("_packed_experts_module")
                    if isinstance(experts_qname, str) and experts_qname:
                        _add_name(experts_qname)
                        if experts_qname in self._paths:
                            self._aliases[str(name)] = experts_qname
        else:
            for name in candidate_names:
                _add_name(str(name))

    def _path_for_name(self, name: str) -> Path | None:
        if name in self._paths:
            return self._paths[name]
        alias = self._aliases.get(name)
        if alias is not None and alias in self._paths:
            return self._paths[alias]
        if self._verification_records is not None:
            return None
        fname = self._FNAME_SUB.sub("__", name) + ".pt"
        fp = self.cache_dir / fname
        if fp.is_file():
            self._paths[name] = fp
            return fp
        return None

    def __contains__(self, name: str) -> bool:
        return self._path_for_name(name) is not None

    def __len__(self) -> int:
        return len(self._paths)

    def load(self, name: str) -> torch.Tensor:
        blob = self.load_blob(name)
        return blob["inputs"]

    def load_blob(self, name: str) -> dict:
        fp = self._path_for_name(name)
        if fp is None:
            raise KeyError(name)
        if self._verification_records is None:
            return torch.load(fp, map_location="cpu", weights_only=False)
        resolved_name = name
        if resolved_name not in self._verification_records:
            resolved_name = self._aliases.get(name, name)
        expected = self._verification_records.get(resolved_name)
        if expected is None or self._verified_dir_fd is None:
            raise ValueError(
                f"{name}: activation has no committed verification record"
            )
        fname = self._FNAME_SUB.sub("__", resolved_name) + ".pt"
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(fname, flags, dir_fd=self._verified_dir_fd)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(
                    f"{resolved_name}: verified activation is not regular"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                blob = torch.load(
                    handle, map_location="cpu", weights_only=False
                )
            after = os.fstat(descriptor)
            stable = lambda value: (
                int(value.st_dev), int(value.st_ino), int(value.st_mode),
                int(value.st_nlink), int(value.st_size),
                int(value.st_mtime_ns), int(value.st_ctime_ns),
            )
            if stable(after) != stable(before):
                raise ValueError(
                    f"{resolved_name}: activation changed while loading"
                )
        finally:
            os.close(descriptor)
        try:
            current = os.stat(
                fname,
                dir_fd=self._verified_dir_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ValueError(
                f"{resolved_name}: activation disappeared while loading"
            ) from exc
        if stable(current) != stable(after):
            raise ValueError(
                f"{resolved_name}: activation pathname changed while loading"
            )
        if not isinstance(blob, Mapping) or set(blob) != {
            "inputs", "name", "row_indices", "row_priorities",
            "sample_parallel_merge",
        }:
            raise ValueError(
                f"{resolved_name}: verified activation fields differ"
            )

        def _tensor_identity(value: object) -> dict[str, object]:
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"{resolved_name}: verified activation tensor differs"
                )
            # The same recipe the merge committed (#1384).
            return tensor_identity(value)

        identity = {
            "inputs": _tensor_identity(blob["inputs"]),
            "row_indices": _tensor_identity(blob["row_indices"]),
            "row_priorities": _tensor_identity(blob["row_priorities"]),
            "source_shards": expected.get("source_shards"),
            "priority_group": expected.get("priority_group"),
        }
        from prismaquant.cost_stage_checkpoint import canonical_json_sha256

        identity["identity_sha256"] = canonical_json_sha256(
            identity, where=f"consumed sample-parallel activation {resolved_name}"
        )
        expected_stamp = {
            "schema": self._verification_schema,
            "cover_identity_sha256": self._verification_cover_sha256,
            "identity_sha256": identity["identity_sha256"],
            "priority_schema": self._verification_priority_schema,
            "priority_group": expected.get("priority_group"),
        }
        if (
            dict(expected) != identity
            or blob.get("name") != resolved_name
            or blob.get("sample_parallel_merge") != expected_stamp
        ):
            raise ValueError(
                f"{resolved_name}: consumed activation differs from commit"
            )
        return dict(blob)

    def load_with_row_indices(self, name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        blob = self.load_blob(name)
        row_indices = blob.get("row_indices") if isinstance(blob, dict) else None
        if not isinstance(row_indices, torch.Tensor):
            row_indices = None
        return blob["inputs"], row_indices

    def prefetch_bytes(self, name: str) -> int:
        """Price the selected file without changing its loader or identity checks."""
        fp = self._path_for_name(name)
        if fp is None:
            raise KeyError(name)
        if self._verified_dir_fd is None:
            size = fp.stat().st_size
        else:
            # The verified loader uses this pinned directory, not a potentially
            # replaced cache_dir pathname. Price that same file.
            size = os.stat(
                fp.name, dir_fd=self._verified_dir_fd, follow_symlinks=False,
            ).st_size
        if size <= 0:
            raise ValueError(f"{name}: activation prefetch needs a nonempty file")
        return size

    def names(self):
        return self._paths.keys()

    def close(self) -> None:
        if self._verified_dir_fd is not None:
            os.close(self._verified_dir_fd)
            self._verified_dir_fd = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
