"""One rule for locating the packaged Tessera contract, in one place (#144).

Two producer readers locate ``tessera/serving/runtime_contract.json``:
``tessera_runtime_contract.contract_path`` (the dev-pin answer reader) and
``tessera_render.tessera_serving_contract_path`` (the menu-level admission
lookup). They must share one resolver, and that resolver is
``importlib.resources.files("tessera.serving")``: it reads the table of the
``tessera.serving`` package that is actually importable, identically from a
wheel, an editable install or an in-repo checkout, never a copy.

``resources.files`` does import the ``tessera.serving`` package, but that
import is cheap by the runtime's own design: its ``__init__`` defines
``register()`` and calls nothing at module scope, importing neither torch
nor vLLM, so locating the contract registers nothing and needs no GPU.
What a producer must not need is the serving-side *code*
(``tessera.serving.contract``'s validator imports the plugin's dispatch
tables), so the JSON is read directly rather than through that module.
"""
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
    assert requested == ["tessera.serving"]
    assert resolved == tmp_path / "runtime_contract.json"


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
    """The true reason, mechanically: the anchor import is lazy and cheap.

    ``resources.files("tessera.serving")`` imports the package, but the
    package pulls in neither torch nor vLLM -- so a producer locates the
    contract without the serving stack. Derived from the runtime's own
    import behavior in a subprocess, not from prose maintained here.
    """
    script = (
        "import sys\n"
        "from importlib import resources\n"
        "t = resources.files('tessera.serving').joinpath("
        "'runtime_contract.json')\n"
        "assert 'tessera.serving' in sys.modules\n"
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
