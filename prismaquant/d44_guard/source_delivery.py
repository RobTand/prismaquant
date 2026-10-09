"""Deliver the accepted D44 standalone source (PQ #2481).

The guard needs its standalone owners to run. This module stages the
vendored owner tree for a campaign run. It verifies each file digest
against the pin. It refuses a missing or changed file.

The staged tree keeps the standalone layout. The launcher subprocess
starts beside its sibling owners. The container entry refuses when
the host cannot supply g3_residency.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN_PATH = HERE / "standalone_source_pin.json"
VENDORED_DIR = HERE / "_vendored"
GUARD_PATH = HERE / "campaign_launch.py"

PINNED_HEAD = "fa3775151f77dc713fd78882daf6e147ab471243"
PINNED_BUNDLE_SHA256 = "7ea79d62000dc4a41d940a9c72eb925e6dbd2b752e49dccd25b22b3a78c998e0"
PINNED_GUARD_SHA256 = "957951a047ac86b2ae123a393ae5b4563f2d75f518052998cd35bf2cc153338a"
PINNED_FROZEN = {
    "campaign_launch.py": "38ca21ef9156eaf0f706643dd14dd3258db658aa652a98a2832c440f1d99d2e9",
    "d44_subsample.py": "78099bf99e593c6e7c7bc48cb7209e2eb4923f76fde6ef7033bda09e408df881",
    "d44_training.py": "58b62ba9e80431609cd4f49385a7d5b483a1699449b7a0b8d42bdc79f62725b4",
    "encode_launch.py": "120b6d79a064846a5ab8cc8cf2dfa3ed2e723b80ea953d0be297e88ee9f1cd8f",
    "stage1.py": "70b24cc00041aaf3d00fd0f46458132209042a11a620d37a521b417189c6c8b0",
    "v2_launch.py": "e42e4dc8c93d7f8f2fb8700d1e617b50336a0e254b76d59dc0d639d0c07f2cdc",
}

STAGED_OWNERS = ("stage1.py", "d44_training.py", "d44_subsample.py", "v2_launch.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pin() -> dict:
    """Read the pin. Refuse a pin that drifts from the constants."""
    pin = json.loads(PIN_PATH.read_text())
    if pin.get("accepted_head") != PINNED_HEAD:
        raise ValueError("D44 source pin head drifted")
    if pin.get("accepted_bundle_sha256") != PINNED_BUNDLE_SHA256:
        raise ValueError("D44 source pin bundle drifted")
    if pin.get("corrected_guard_sha256") != PINNED_GUARD_SHA256:
        raise ValueError("D44 source pin guard drifted")
    if pin.get("frozen_entries") != PINNED_FROZEN:
        raise ValueError("D44 source pin entries drifted")
    return pin


def verify_vendored() -> dict[str, str]:
    """Hash each vendored file. Refuse a missing or changed file."""
    pin = load_pin()
    digests = {}
    for name, want in pin["frozen_entries"].items():
        path = VENDORED_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"D44 vendored source is absent: {name}")
        found = _sha(path)
        if found != want:
            raise ValueError(f"D44 vendored source changed: {name}")
        digests[name] = found
    if _sha(GUARD_PATH) != pin["corrected_guard_sha256"]:
        raise ValueError("D44 corrected guard changed")
    return digests


def stage_tree(dest: Path) -> Path:
    """Stage the runnable owner tree. Return the guard entry path."""
    verify_vendored()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in STAGED_OWNERS:
        shutil.copyfile(VENDORED_DIR / name, dest / name)
    guard = dest / "campaign_launch.py"
    shutil.copyfile(GUARD_PATH, guard)
    shutil.copyfile(Path(__file__).resolve(), dest / "source_delivery.py")
    shutil.copyfile(PIN_PATH, dest / "standalone_source_pin.json")
    return guard


def load_staged_guard(dest: Path):
    """Stage the tree. Import the guard with its staged owners."""
    guard_path = stage_tree(dest)
    staged = str(Path(dest).resolve())
    sys.path.insert(0, staged)
    try:
        spec = importlib.util.spec_from_file_location(
            "d44_delivered_campaign_launch", guard_path)
        if spec is None or spec.loader is None:
            raise ImportError("D44 staged guard has no loader")
        module = importlib.util.module_from_spec(spec)
        sys.modules["d44_delivered_campaign_launch"] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(staged)
    return module


def require_container_residency() -> None:
    """Refuse the container path when g3_residency is absent."""
    found = sys.modules.get("g3_residency")
    if found is not None and not hasattr(found, "container_contract"):
        found = None
    if found is None:
        try:
            spec = importlib.util.find_spec("g3_residency")
        except (ImportError, AttributeError, ValueError):
            spec = None
        if spec is None:
            raise ImportError(
                "D44 container path needs g3_residency; the accepted tree "
                "lacks this owner, so the host must supply it")
