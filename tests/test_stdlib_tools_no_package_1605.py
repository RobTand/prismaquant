"""Stdlib-only tools must load with the repo off ``sys.path`` (issue #1605).

``tools/serve_fingerprint.py`` and ``tools/prismaquant_runtime_snapshot.py``
run inside serving containers from a bootstrap with no installed package
(``prismaquant/tessera_runtime_contract.py:3477-3480`` and the snapshot
module docstring). ``prismaquant/container_runtime_identity.py`` is likewise
stdlib-only. A delegation to ``prismaquant.schemas`` inside any of their
loader paths dies with ``ModuleNotFoundError`` in the container while every
in-repo test stays green, because each test process can import the package.

Each case below rebuilds the container layout under a fresh directory -- the
source file at its real ``tools/`` or ``prismaquant/`` bootstrap path plus,
for the fingerprint, the pin data
file its module-level table reads through ``__file__`` -- then loads the
tool standalone in a subprocess whose ``sys.path`` has the real repo root
and ``tools/`` scrubbed (``cwd`` is the case root, ``PYTHONPATH`` is unset)
and whose import system refuses ``prismaquant`` outright (an editable
install resolves it through a meta-path finder, not ``sys.path``), and
exercises the loader. The fingerprint and snapshot cases fail on the
pre-revert head and pass after it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
PIN_RELATIVE = Path("prismaquant") / "tessera_runtime" / "tessera_serving_runtime_pin.json"

_CHILD = (
    "import importlib.util, os, sys\n"
    "from pathlib import Path\n"
    "repo, tool, payload = sys.argv[1], sys.argv[2], sys.argv[3]\n"
    "banned = {{os.path.normpath(repo), os.path.normpath(os.path.join(repo, 'tools'))}}\n"
    "sys.path = [p for p in sys.path\n"
    "            if os.path.normpath(os.path.abspath(p or os.getcwd())) not in banned]\n"
    # An editable install (CI's ``pip install -e .``) resolves the package
    # through a meta-path finder, not through ``sys.path``, so scrubbing the
    # path alone leaves it importable. Refuse it the way a container without
    # the package does.
    "class _NoPackage:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name == 'prismaquant' or name.startswith('prismaquant.'):\n"
    "            raise ModuleNotFoundError(f'No module named {{name!r}}', name=name)\n"
    "        return None\n"
    "sys.meta_path.insert(0, _NoPackage())\n"
    "for _name in [n for n in sys.modules\n"
    "              if n == 'prismaquant' or n.startswith('prismaquant.')]:\n"
    "    del sys.modules[_name]\n"
    "try:\n"
    "    import prismaquant.schemas\n"
    "except ImportError:\n"
    "    pass\n"
    "else:\n"
    "    raise SystemExit('package still importable')\n"
    "{before}\n"
    "spec = importlib.util.spec_from_file_location('standalone_tool', tool)\n"
    "mod = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(mod)\n"
    "{stmt}\n"
)

MODEL = "test-model"
CARD = {
    "object": "list",
    "data": [{
        "id": MODEL,
        "object": "model",
        "created": 123,
        "root": "/r",
        "owned_by": "o",
    }],
}

_PIN_ACCEPT = (
    "got = mod._read_tessera_serving_pin_payload(Path(payload))\n"
    "assert got == {'a': 1}, got\n"
    "print('PIN-OK')\n"
)
_PIN_REFUSE = (
    "try:\n"
    "    mod._read_tessera_serving_pin_payload(Path(payload))\n"
    "except ValueError as exc:\n"
    "    assert 'repeats JSON key' in str(exc), exc\n"
    "    print('PIN-DUP-OK')\n"
    "else:\n"
    "    raise SystemExit('duplicate pin key accepted')\n"
)
_MODELS_ACCEPT = (
    "got = mod.models_endpoint_binding_from_bytes(\n"
    "    Path(payload).read_bytes(), request_url='http://x/v1',\n"
    "    expected_served_model='test-model')\n"
    "assert got['model_count'] == 1, got\n"
    "print('MODELS-OK')\n"
)
_SNAP_ACCEPT = (
    "got = mod._load_manifest(Path(payload))\n"
    "assert got == {'a': 1}, got\n"
    "print('SNAP-OK')\n"
)
_SNAP_REFUSE = (
    "try:\n"
    "    mod._load_manifest(Path(payload))\n"
    "except mod.SnapshotError as exc:\n"
    "    assert 'duplicate manifest member' in str(exc), exc\n"
    "    print('SNAP-DUP-OK')\n"
    "else:\n"
    "    raise SystemExit('duplicate manifest member accepted')\n"
)
_CRI_ACCEPT = (
    "got = mod._load_json_object(Path(payload), where='probe')\n"
    "assert got == {'a': 1}, got\n"
    "print('CRI-OK')\n"
)
_CRI_REFUSE = (
    "try:\n"
    "    mod._load_json_object(Path(payload), where='probe')\n"
    "except mod.RuntimeIdentityError as exc:\n"
    "    assert 'duplicate JSON member' in str(exc), exc\n"
    "    print('CRI-DUP-OK')\n"
    "else:\n"
    "    raise SystemExit('duplicate member accepted')\n"
)


def _stage(workdir: Path, tool: str, with_pin: bool = False) -> Path:
    """Stage a standalone source file at its real bootstrap-relative path."""
    root = workdir / "container"
    destination = root / tool
    destination.parent.mkdir(parents=True)
    shutil.copy2(REPO / tool, destination)
    if with_pin:
        pin = root / PIN_RELATIVE
        pin.parent.mkdir(parents=True)
        shutil.copy2(REPO / PIN_RELATIVE, pin)
    return root


def _run_isolated(root: Path, tool: str, stmt: str, payload: Path, *, before: str = "") -> str:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD.format(stmt=stmt, before=before),
         str(REPO), str(root / tool), str(payload)],
        cwd=root, capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout


def _write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


def test_pin_loader_runs_without_the_package(tmp_path: Path) -> None:
    root = _stage(tmp_path, "tools/serve_fingerprint.py", with_pin=True)
    out = _run_isolated(root, "tools/serve_fingerprint.py", _PIN_ACCEPT,
                        _write(root, "pin.json", '{"a": 1}'))
    assert "PIN-OK" in out
    out = _run_isolated(root, "tools/serve_fingerprint.py", _PIN_REFUSE,
                        _write(root, "dup.json", '{"a": 1, "a": 2}'))
    assert "PIN-DUP-OK" in out


def test_models_endpoint_loader_runs_without_the_package(tmp_path: Path) -> None:
    root = _stage(tmp_path, "tools/serve_fingerprint.py", with_pin=True)
    out = _run_isolated(
        root, "tools/serve_fingerprint.py", _MODELS_ACCEPT,
        _write(root, "models.json", json.dumps(CARD)))
    assert "MODELS-OK" in out


def test_snapshot_manifest_loader_runs_without_the_package(tmp_path: Path) -> None:
    root = _stage(tmp_path, "tools/prismaquant_runtime_snapshot.py")
    out = _run_isolated(root, "tools/prismaquant_runtime_snapshot.py",
                        _SNAP_ACCEPT, _write(root, "m.json", '{"a": 1}'))
    assert "SNAP-OK" in out
    out = _run_isolated(root, "tools/prismaquant_runtime_snapshot.py",
                        _SNAP_REFUSE,
                        _write(root, "d.json", '{"a": 1, "a": 2}'))
    assert "SNAP-DUP-OK" in out


def test_container_identity_loader_runs_without_the_package(tmp_path: Path) -> None:
    root = _stage(tmp_path, "prismaquant/container_runtime_identity.py")
    out = _run_isolated(root, "prismaquant/container_runtime_identity.py",
                        _CRI_ACCEPT, _write(root, "o.json", '{"a": 1}'))
    assert "CRI-OK" in out
    out = _run_isolated(root, "prismaquant/container_runtime_identity.py",
                        _CRI_REFUSE,
                        _write(root, "d.json", '{"a": 1, "a": 2}'))
    assert "CRI-DUP-OK" in out


#: The served gold client's container layout (#2629): the client, the two
#: sibling tools it imports in script mode, the pin data file the fingerprint
#: module reads at import, and the digest owner it loads by file path.
_GOLD_TOOL = "tools/measure_served_gold.py"
_GOLD_SIBLINGS = ("tools/serve_fingerprint.py", "tools/spec_decode_guard.py")
_DIGEST_OWNER = "prismaquant/digests.py"
_GOLD_OWNER_MODULE = "_measure_served_gold_digests"

_GOLD_ACCEPT = (
    "import hashlib, json\n"
    "p = Path(payload)\n"
    "assert mod._sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()\n"
    "v = {'k': 'h\\u00e9llo\\u2713', 'astral': '\\U0001F600', 'n': {3: [1, 2.5, None], 1: 2}}\n"
    "want = hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'),\n"
    "                                 allow_nan=False).encode('utf-8')).hexdigest()\n"
    "assert mod._canonical_sha256(v) == want\n"
    "try:\n"
    "    mod._canonical_sha256({'x': float('nan')})\n"
    "except ValueError:\n"
    "    pass\n"
    "else:\n"
    "    raise SystemExit('nonfinite canonical JSON accepted')\n"
    "c = mod._calibration_contract(p, p.parent, [1, -2, 3], seqlen=2,\n"
    "                              n_tokens_requested=3)\n"
    "assert c['calib_ids_sha256'] == hashlib.sha256(b'1,-2,3').hexdigest(), c\n"
    "assert c['corpus_sha256'] == hashlib.sha256(p.read_bytes()).hexdigest(), c\n"
    "owner = Path(mod._DIGESTS.__file__).resolve()\n"
    "assert owner == (Path.cwd() / 'prismaquant' / 'digests.py').resolve(), owner\n"
    "loaded = [n for n in sys.modules if n == 'prismaquant' or n.startswith('prismaquant.')]\n"
    "assert not loaded, loaded\n"
    "assert 'torch' not in sys.modules\n"
    "print('GOLD-OK')\n"
)


def _stage_gold(workdir: Path, *, with_owner: bool) -> Path:
    """Stage the gold client's deployed layout; optionally omit the owner."""
    root = _stage(workdir, _GOLD_TOOL, with_pin=True)
    for name in _GOLD_SIBLINGS + ((_DIGEST_OWNER,) if with_owner else ()):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, destination)
    return root


def test_served_gold_digest_loader_runs_without_the_package(tmp_path: Path) -> None:
    root = _stage_gold(tmp_path, with_owner=True)
    corpus = root / "corpus.txt"
    corpus.write_bytes("h\u00e9llo \u2713 \U0001F600\x00tail\n".encode("utf-8"))
    out = _run_isolated(root, _GOLD_TOOL, _GOLD_ACCEPT, corpus)
    assert "GOLD-OK" in out


def test_served_gold_refuses_a_missing_digest_owner(tmp_path: Path) -> None:
    root = _stage_gold(tmp_path, with_owner=False)
    assert not (root / _DIGEST_OWNER).exists()
    load = "spec.loader.exec_module(mod)\n"
    assert load in _CHILD
    child = _CHILD.replace(load, (
        "try:\n"
        "    spec.loader.exec_module(mod)\n"
        "except RuntimeError as exc:\n"
        "    assert 'digests.py' in str(exc), exc\n"
        "    assert 'REFUSING' in str(exc), exc\n"
        f"    assert {_GOLD_OWNER_MODULE!r} not in sys.modules\n"
        "    print('GOLD-OWNER-REFUSED')\n"
        "else:\n"
        "    raise SystemExit('loaded without its digest owner')\n"
    ))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", child.format(stmt="", before=""),
         str(REPO), str(root / _GOLD_TOOL), str(root / "unused")],
        cwd=root, capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "GOLD-OWNER-REFUSED" in proc.stdout


@pytest.mark.parametrize("script,option", [
    ("stageb_a4_quantizer_launch.py", "--group-policy-sha256"),
    ("glm_native_moe_launch.py", "--spec-sha256"),
])
def test_host_image_launcher_help_before_package_or_device_access(tmp_path, script, option):
    """Only the deployed launcher/public stdlib owner exist; PQ is denied."""
    relative = "experiments/" + script
    root = _stage(tmp_path, relative)
    owner = root / "prismaquant/container_runtime_identity.py"
    owner.parent.mkdir(parents=True)
    shutil.copy2(REPO / "prismaquant/container_runtime_identity.py", owner)
    before = (
        "import subprocess\n"
        "def no_process(*args, **kwargs):\n"
        "    raise AssertionError('--help attempted process/device access')\n"
        "for name in ('run', 'call', 'check_output', 'Popen'):\n"
        "    setattr(subprocess, name, no_process)\n"
        "class NoDevicePackage:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('torch', 'tessera'):\n"
        "            raise AssertionError('--help imported device package: ' + name)\n"
        "sys.meta_path.insert(0, NoDevicePackage())\n"
        "sys.argv = [tool, '--help']\n"
    )
    out = _run_isolated(root, relative, "", _write(root, "unused", ""), before=before)
    assert "usage:" in out
    assert option in out
