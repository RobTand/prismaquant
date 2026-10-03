"""Stdlib-only tools must load with the repo off ``sys.path`` (issue #1605).

``tools/serve_fingerprint.py`` and ``tools/prismaquant_runtime_snapshot.py``
run inside serving containers from a bootstrap with no installed package
(``prismaquant/tessera_runtime_contract.py:3477-3480`` and the snapshot
module docstring). ``tools/container_runtime_identity.py`` is likewise
stdlib-only. A delegation to ``prismaquant.schemas`` inside any of their
loader paths dies with ``ModuleNotFoundError`` in the container while every
in-repo test stays green, because each test process can import the package.

Each case below rebuilds the container layout under a fresh directory -- the
tool file at ``<root>/tools/<name>`` plus, for the fingerprint, the pin data
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


def _run_isolated(root: Path, tool: str, stmt: str, payload: Path) -> str:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD.format(stmt=stmt),
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
