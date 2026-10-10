"""Deliver the accepted D44 standalone source (PQ #2481).

The repository supplies the frozen owners and the corrected guard.
External owners and the container template have immutable digest bindings.
Source, template, and module identities use the shared D32 seal.
Missing dependencies and stored-byte corruption refuse in both modes.

Native routes use the repository snapshot on each selected worker.
Explicit standalone delivery uses a fresh tree on the shared fleet mount.
The delivery never overwrites the active source tree.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN_PATH = HERE / "standalone_source_pin.json"
VENDORED_DIR = HERE / "_vendored"
GUARD_PATH = HERE / "campaign_launch.py"

_SCHEMA = "prismaquant.d44_standalone_source_pin.v2"

# Every vendored owner stages beside the guard entry. The vendored
# v2_score.py is the standalone argv adapter. It stages under a distinct
# name so the external numerical scorer keeps its import name.
STAGED_OWNERS = (
    "stage1.py",
    "d44_training.py",
    "d44_subsample.py",
    "d44.py",
    "d44_g3.py",
    "d44_weight_leg.py",
    "codec_factorial.py",
    "cd2_alignment.py",
    "source_file.py",
    "d44-frozen-method.json",
    "v2_launch.py",
    "encode_launch.py",
)
STAGED_RENAMED = {"v2_score.py": "d44_v2_score_adapter.py"}

# The guard entry imports these modules from its staged siblings.
REQUIRED_STAGED_IMPORTS = (
    "stage1",
    "d44_training",
    "d44_subsample",
    "d44",
    "d44_weight_leg",
    "codec_factorial",
    "cd2_alignment",
    "source_file",
)

# The staged guard imports these modules from the bound owners directory.
REQUIRED_EXTERNAL_IMPORTS = (
    "g3_residency",
    "v2_score",
    "g3_offline_decoded_kl",
    "g3_lib",
    "g3_readset",
    "exl3_torch",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_check(kind, expected, actual, *, where, refusal, environ=None):
    """Use the shared D32 helper without the numerical package initializer."""
    policy = HERE / "_policy"
    if not policy.is_dir():
        policy = HERE.parent
    package_name = "_d44_delivery_policy"
    name = package_name + ".dev_mode"
    found = sys.modules.get(name)
    if found is None:
        package = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(package_name, loader=None, is_package=True))
        package.__path__ = [str(policy)]
        sys.modules[package_name] = package
        spec = importlib.util.spec_from_file_location(name, policy / "dev_mode.py")
        if spec is None or spec.loader is None:
            raise ImportError("D44 delivery cannot load the D32 seal helper")
        found = importlib.util.module_from_spec(spec)
        sys.modules[name] = found
        try:
            spec.loader.exec_module(found)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return found.seal_check(kind, expected, actual, where=where,
                            refusal=refusal, environ=environ)


def load_pin() -> dict:
    """Read the pin. Check its schema and required bindings."""
    pin = json.loads(PIN_PATH.read_text())
    if pin.get("schema") != _SCHEMA:
        raise ValueError("D44 source pin has an unknown schema")
    for key in ("accepted_head", "accepted_bundle_sha256",
                "corrected_guard_sha256", "frozen_entries",
                "external_owner_bindings"):
        if key not in pin:
            raise ValueError(f"D44 source pin lacks {key}")
    return pin


def check_recorded_source(*, environ=None) -> bool:
    """Seal the recorded accepted source against the pin (D32).

    The head and bundle digest are provenance, not own bytes. A
    mismatch stamps in dev mode and refuses in certified mode.
    """
    pin = load_pin()
    ok = _seal_check(
        "D44 accepted source head",
        "fa3775151f77dc713fd78882daf6e147ab471243",
        pin.get("accepted_head"), where=str(PIN_PATH),
        refusal=ValueError("D44 accepted source head drifted"), environ=environ)
    ok = _seal_check(
        "D44 accepted bundle digest",
        "7ea79d62000dc4a41d940a9c72eb925e6dbd2b752e49dccd25b22b3a78c998e0",
        pin.get("accepted_bundle_sha256"), where=str(PIN_PATH),
        refusal=ValueError("D44 accepted bundle digest drifted"), environ=environ) and ok
    return ok


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


def verify_external_bindings(*, owners_dir: Path | None = None) -> dict[str, str]:
    """Compare external source identities through the shared D32 seal."""
    pin = load_pin()
    bindings = pin["external_owner_bindings"]
    root = Path(owners_dir) if owners_dir is not None else Path(
        bindings["owners_dir"])
    digests = {}
    for name, want in bindings["files"].items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"D44 external owner is absent: {path}")
        found = _sha(path)
        _seal_check("D44 external source", want, found, where=str(path),
                    refusal=ValueError(f"D44 external owner changed: {path}"))
        digests[name] = found
    template = bindings["container_template"]
    template_path = Path(template["path"])
    if not template_path.is_file():
        raise FileNotFoundError(
            f"D44 container template is absent: {template_path}")
    found = _sha(template_path)
    _seal_check("D44 container template", template["sha256"], found,
                where=str(template_path),
                refusal=ValueError("D44 container template changed"))
    digests["container_template"] = found
    return digests


def _check_staged_imports(staged: Path, owners_dir: Path) -> None:
    """Import every required owner from the staged tree. Refuse a gap."""
    staged = staged.resolve()
    owners = owners_dir.resolve()
    script = (
        "import importlib, sys\n"
        "import source_delivery as delivery\n"
        f"STAGED = {str(staged)!r}\n"
        f"OWNERS = {str(owners)!r}\n"
        "sys.path[:0] = [STAGED, OWNERS]\n"
        f"for root, names in [(STAGED, {list(REQUIRED_STAGED_IMPORTS)!r}),\n"
        f"                    (OWNERS, {list(REQUIRED_EXTERNAL_IMPORTS)!r})]:\n"
        "    for name in names:\n"
        "        module = importlib.import_module(name)\n"
        "        delivery._seal_check('D44 module path', root + '/' + name + '.py',\n"
        "            module.__file__, where=name,\n"
        "            refusal=ImportError('D44 module path changed: ' + name))\n"
    )
    import subprocess

    done = subprocess.run(
        [sys.executable, "-c", script], cwd=staged, capture_output=True, text=True,
        timeout=300)
    if done.stdout:
        print(done.stdout, end="", flush=True)
    if done.returncode != 0:
        raise ImportError(
            "D44 staged tree lacks a required owner: "
            f"{done.stderr.strip() or done.stdout.strip()}")


def stage_tree(dest: Path, *, owners_dir: Path | None = None) -> Path:
    """Stage the runnable owner tree. Return the guard entry path.

    The destination must be a fresh separate directory. This operation
    refuses an existing path, so it never overwrites the active source
    tree. It verifies every vendored and external owner first. It then
    checks that the staged tree imports each required owner.
    """
    check_recorded_source()
    verify_vendored()
    bindings = load_pin()["external_owner_bindings"]
    owners = Path(owners_dir) if owners_dir is not None else Path(
        bindings["owners_dir"])
    verify_external_bindings(owners_dir=owners)
    dest = Path(dest)
    try:
        dest.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"D44 delivery refuses an existing stage path: {dest}")
    staged: list[Path] = []
    try:
        for name in STAGED_OWNERS:
            target = dest / name
            shutil.copyfile(VENDORED_DIR / name, target)
            staged.append(target)
        for name, staged_name in STAGED_RENAMED.items():
            target = dest / staged_name
            shutil.copyfile(VENDORED_DIR / name, target)
            staged.append(target)
        guard = dest / "campaign_launch.py"
        shutil.copyfile(GUARD_PATH, guard)
        staged.append(guard)
        for name in ("source_delivery.py", "standalone_source_pin.json"):
            target = dest / name
            shutil.copyfile(HERE / name, target)
            staged.append(target)
        policy = dest / "_policy"
        policy.mkdir()
        for name in ("dev_mode.py", "digests.py"):
            target = policy / name
            shutil.copyfile(HERE.parent / name, target)
            staged.append(target)
        _check_staged_imports(dest, owners)
    except BaseException:
        for target in staged:
            try:
                target.unlink()
            except OSError:
                pass
        policy = dest / "_policy"
        if policy.exists():
            shutil.rmtree(policy)
        try:
            dest.rmdir()
        except OSError:
            pass
        raise
    return guard


def stage_fresh(*, parent: Path | None = None,
                owners_dir: Path | None = None) -> Path:
    """Stage the tree in a fresh directory. Return the guard entry."""
    selected = parent or os.environ.get("D44_DELIVERY_STAGE_PARENT")
    root = Path(selected) if selected else Path(
        "/mnt/shared/tessera-measurements/d44-delivered")
    root.mkdir(parents=True, exist_ok=True)
    claimed = Path(tempfile.mkdtemp(prefix="d44-delivered-", dir=str(root)))
    dest = claimed / "tree"
    try:
        return stage_tree(dest, owners_dir=owners_dir)
    except BaseException:
        try:
            shutil.rmtree(claimed, ignore_errors=True)
        finally:
            raise


def require_container_residency(*, owners_dir: Path | None = None) -> None:
    """Verify the bound g3_residency owner. Refuse drift or absence."""
    pin = load_pin()
    bindings = pin["external_owner_bindings"]
    root = Path(owners_dir) if owners_dir is not None else Path(
        bindings["owners_dir"])
    want = bindings["files"]["g3_residency.py"]
    path = root / "g3_residency.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"D44 bound g3_residency is absent: {path}")
    found = _sha(path)
    _seal_check("D44 residency source", want, found, where=str(path),
                refusal=ValueError("D44 bound g3_residency changed"))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = sys.modules.get("g3_residency")
    if module is not None:
        _seal_check("D44 residency module path", str(path),
                    getattr(module, "__file__", None), where="g3_residency",
                    refusal=ImportError("D44 residency module path changed"))
    if module is None:
        spec = importlib.util.spec_from_file_location(
            "g3_residency", path)
        if spec is None or spec.loader is None:
            raise ImportError("D44 bound g3_residency has no loader")
        module = importlib.util.module_from_spec(spec)
        sys.modules["g3_residency"] = module
        spec.loader.exec_module(module)
    if not hasattr(module, "container_contract"):
        raise ImportError(
            "D44 bound g3_residency lacks container_contract")


def check_d30_binding(*, owners_dir: Path | None = None) -> Path:
    """Verify the bound D30 launcher and template. Return the template."""
    pin = load_pin()
    bindings = pin["external_owner_bindings"]
    root = Path(owners_dir) if owners_dir is not None else Path(
        bindings["owners_dir"])
    want = bindings["files"]["v2_launch.py"]
    path = root / "v2_launch.py"
    if not path.is_file():
        raise FileNotFoundError(f"D44 bound D30 launcher is absent: {path}")
    _seal_check("D44 D30 source", want, _sha(path), where=str(path),
                refusal=ValueError("D44 bound D30 launcher changed"))
    template = bindings["container_template"]
    template_path = Path(template["path"])
    _seal_check("D44 container template", template["sha256"], _sha(template_path),
                where=str(template_path),
                refusal=ValueError("D44 container template changed"))
    text = path.read_text()
    suffix = template["path"]
    base = "/mnt/shared/tessera-measurements"
    if suffix.startswith(base):
        suffix = suffix[len(base):]
    # The launcher names the bound template through BASE plus a suffix.
    if suffix not in text and template["path"] not in text:
        _seal_check("D44 D30 template path", template["path"], "another template",
                    where=str(path),
                    refusal=ValueError("D44 bound D30 launcher names another template"))
    return template_path


def owners_dir(*, override: Path | None = None) -> Path:
    """Return the bound external owners directory."""
    if override is not None:
        return Path(override)
    return Path(load_pin()["external_owner_bindings"]["owners_dir"])


