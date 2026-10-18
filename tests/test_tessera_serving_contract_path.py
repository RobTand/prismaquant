"""Installed contract resources belong to producer metadata, without serving imports."""
from __future__ import annotations

import subprocess
import sys


def test_contract_path_resolves_the_importable_serving_package(monkeypatch, tmp_path):
    from importlib import resources
    from prismaquant import tessera_runtime_contract as trc

    requested = []

    def installed_package(name):
        requested.append(name)
        return tmp_path

    monkeypatch.setattr(resources, "files", installed_package)
    resolved = trc.contract_path()
    assert requested == ["tessera"]
    assert resolved == tmp_path / "serving" / "runtime_contract.json"


def test_render_reader_uses_the_canonical_resolver(monkeypatch):
    from prismaquant import tessera_render as tr
    from prismaquant import tessera_runtime_contract as trc

    packaged_resource = object()
    monkeypatch.setattr(trc, "contract_path", lambda: packaged_resource)
    assert tr.tessera_serving_contract_path() is packaged_resource


def test_both_readers_resolve_the_same_table():
    """Both names for the file read the same bytes, from the same package."""
    from importlib.resources import as_file

    from prismaquant import tessera_render as tr
    from prismaquant import tessera_runtime_contract as trc

    with as_file(trc.contract_path()) as a, as_file(
        tr.tessera_serving_contract_path()
    ) as b:
        assert a.read_bytes() == b.read_bytes()
        assert a.read_bytes(), "the packaged contract must not be empty"


def test_locating_the_contract_registers_nothing_and_needs_no_gpu_stack():
    """The resource anchor imports only the producer package."""
    script = (
        "import sys\n"
        "from importlib import resources\n"
        "t = resources.files('tessera').joinpath('serving',"
        "'runtime_contract.json')\n"
        "assert 'tessera.serving' not in sys.modules\n"
        "assert 'torch' not in sys.modules, sorted(\n"
        "    m for m in sys.modules if m.startswith('torch'))\n"
        "assert 'vllm' not in sys.modules, sorted(\n"
        "    m for m in sys.modules if m.startswith('vllm'))\n"
        "assert len(t.read_bytes()) > 0\n"
        "print('OK')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "OK" in out.stdout
