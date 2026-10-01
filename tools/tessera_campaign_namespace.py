"""Shared, package-free metadata boundary for opt-in campaign namespaces."""
import stat
from pathlib import Path


def refuse_path_symlinks(value: str, *, directory: bool = True) -> None:
    """Inspect existing ancestors without creating or resolving the destination."""
    path = Path(value)
    for ancestor in (*reversed(path.parents), path):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"cannot inspect scratch path {ancestor}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"scratch path contains a symlink: {ancestor}")
        if (ancestor != path or directory) and not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"scratch path is not a directory: {ancestor}")
