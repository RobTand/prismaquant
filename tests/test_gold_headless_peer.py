"""The peer launcher that keeps rank 1 alive, and rank 0 from hanging forever.

vLLM spawns its workers, and a spawned worker re-imports the LAUNCHING FILE as
`__mp_main__`. A gold driver that ships callables through `apply_model` /
`collective_rpc` pickles them as `__main__.<name>`, so under the stock
`/usr/local/bin/vllm` peer those names are missing, rank 1 dies with
`AttributeError`, and rank 0 waits on it with no timeout (2026-09-14, GLM-5.3
4-layer stub, stopped by hand after 30 minutes).

These tests never import vLLM: the CLI entry point is replaced, so what is
checked is the launcher's contract -- the driver namespace really is copied
into the module that will become `__mp_main__`, and the stock argv is passed
through unchanged.
"""
from __future__ import annotations

import pathlib
import sys
import types

import pytest

if not (pathlib.Path(__file__).resolve().parents[1] / "tools").is_dir():
    pytest.skip("requires a repo checkout (tools/ scripts)",
                allow_module_level=True)

from tools import gold_headless_peer

#: A real repository module with public names and no heavy imports, so the
#: namespace copy is proven against something that actually exists rather than
#: against a stub built to pass.
DRIVER = "tools.gold_measurement_fidelity"


@pytest.fixture
def fake_vllm_cli(monkeypatch):
    """Replace the stock CLI entry point and record the argv it received."""
    seen: dict[str, list[str]] = {}

    def main() -> None:
        seen["argv"] = list(sys.argv)

    cli_main = types.ModuleType("vllm.entrypoints.cli.main")
    cli_main.main = main
    for name, module in (
        ("vllm", types.ModuleType("vllm")),
        ("vllm.entrypoints", types.ModuleType("vllm.entrypoints")),
        ("vllm.entrypoints.cli", types.ModuleType("vllm.entrypoints.cli")),
        ("vllm.entrypoints.cli.main", cli_main),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return seen


def test_the_drivers_public_names_reach_the_module_that_becomes_mp_main(
    monkeypatch, fake_vllm_cli,
):
    """The whole point: a spawned worker re-imports THIS file, so the driver's
    callables must be resolvable as globals of it."""
    monkeypatch.setenv(gold_headless_peer.DRIVER_ENV, DRIVER)
    gold_headless_peer.main(["serve", "/models/stub", "--node-rank", "1",
                             "--headless"])
    assert gold_headless_peer.full_kl_fidelity is not None
    assert fake_vllm_cli["argv"][0] == "vllm"


def test_stock_argv_is_passed_through_unchanged_under_a_vllm_argv0(
    monkeypatch, fake_vllm_cli,
):
    """The peer must be a stock serve in every respect except its namespace."""
    monkeypatch.setenv(gold_headless_peer.DRIVER_ENV, DRIVER)
    arguments = ["serve", "/models/stub", "--node-rank", "1", "--headless",
                 "--tensor-parallel-size", "2", "--enforce-eager"]
    gold_headless_peer.main(list(arguments))
    assert fake_vllm_cli["argv"] == ["vllm", *arguments]


def test_an_unnamed_driver_refuses_instead_of_launching_a_bare_serve(
    monkeypatch, fake_vllm_cli,
):
    """A default driver would be right for one gold tool and hang rank 0 for
    another, so absence refuses rather than guessing."""
    monkeypatch.delenv(gold_headless_peer.DRIVER_ENV, raising=False)
    with pytest.raises(SystemExit) as exit_info:
        gold_headless_peer.main(["serve", "/models/stub", "--headless"])
    assert "PQ_GOLD_PEER_DRIVER" in str(exit_info.value)
    assert "argv" not in fake_vllm_cli


def test_no_serve_arguments_refuses_rather_than_starting_an_empty_cli(
    monkeypatch, fake_vllm_cli,
):
    monkeypatch.setenv(gold_headless_peer.DRIVER_ENV, DRIVER)
    with pytest.raises(SystemExit):
        gold_headless_peer.main([])
    assert "argv" not in fake_vllm_cli


def test_the_namespace_copy_leaves_the_launching_modules_identity_alone():
    """Copying `__name__`/`__file__` from the driver would destroy exactly the
    `__mp_main__` identity this launcher exists to establish."""
    namespace = {"__name__": "__main__", "__file__": "/peer/launcher.py"}
    copied = gold_headless_peer.load_driver_namespace(
        DRIVER,
        root=pathlib.Path(__file__).resolve().parents[1],
        namespace=namespace,
    )
    assert "full_kl_fidelity" in copied
    assert not any(name.startswith("__") for name in copied)
    assert namespace["__name__"] == "__main__"
    assert namespace["__file__"] == "/peer/launcher.py"


def test_the_spawned_workers_reimport_gets_the_driver_namespace(
    monkeypatch, fake_vllm_cli,
):
    """The mechanism itself, exercised the way multiprocessing exercises it.

    A spawned vLLM worker re-imports the launching FILE with
    `run_name="__mp_main__"`. Top-level code runs; `main()` and anything under
    `if __name__ == "__main__"` do not. So the driver names must be present in
    the globals that re-import produces -- and the stock CLI must NOT have been
    started by it, or every worker would try to launch a server.

    Calling `main()` directly (as the tests above do) cannot catch this: it
    inspects the already-imported module object rather than the namespace the
    spawn actually builds.
    """
    import runpy

    monkeypatch.setenv(gold_headless_peer.DRIVER_ENV, DRIVER)
    produced = runpy.run_path(
        gold_headless_peer.__file__, run_name="__mp_main__")
    assert "full_kl_fidelity" in produced
    assert "argv" not in fake_vllm_cli


def test_importing_without_the_driver_variable_stays_inert(monkeypatch):
    """The module-level load is gated, so a doc tool or a test that merely
    imports this file does not drag a driver in behind it."""
    import runpy

    monkeypatch.delenv(gold_headless_peer.DRIVER_ENV, raising=False)
    produced = runpy.run_path(
        gold_headless_peer.__file__, run_name="__mp_main__")
    assert "full_kl_fidelity" not in produced
    assert produced["_DRIVER"] == ""


def test_a_relative_root_override_refuses():
    """A relative root resolves against the peer's working directory, which is
    not the coordinator's; that is how two ranks come to import two trees."""
    import os

    os.environ[gold_headless_peer.ROOT_ENV] = "relative/checkout"
    try:
        with pytest.raises(ValueError, match="absolute"):
            gold_headless_peer.repository_root()
    finally:
        del os.environ[gold_headless_peer.ROOT_ENV]
