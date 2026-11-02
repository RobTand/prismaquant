"""The per-test bound shared by ordinary and child pytest sessions."""
from pathlib import Path
import os

import pytest

#: PrismaBuild's per-test bound plugin, by the name ``-p`` loads it under.
PRISMABUILD_TEST_BOUND_PLUGIN = "prismabuild.pytest_test_bound"
#: The per-test bound in seconds, which ``pbtest`` exports to every shard and
#: the plugin reads. Unset or empty: no bound.
PRISMABUILD_TEST_BOUND_ENV = "PRISMABUILD_TEST_TIMEOUT_S"


def load_test_bound(config) -> None:
    """Register PrismaBuild's per-test bound when the shard sets one.

    Does nothing when ``PRISMABUILD_TEST_TIMEOUT_S`` is unset or empty, or
    when the plugin is already registered (``-p``). Otherwise the plugin's
    file is found without importing ``prismabuild`` (``find_spec`` of a
    top-level package does not run it), loaded under a private module name,
    and registered under the plugin's own name. It imports only the standard
    library and pytest. A bound the session asks for but cannot load is
    refused: running unbounded and green is what the bound exists to end.
    """
    import importlib.util

    if not os.environ.get(PRISMABUILD_TEST_BOUND_ENV, "").strip():
        return
    if config.pluginmanager.has_plugin(PRISMABUILD_TEST_BOUND_PLUGIN):
        return
    try:
        package = importlib.util.find_spec("prismabuild")
    except (ImportError, ValueError):
        package = None
    candidates = [Path(location) / "pytest_test_bound.py" for location in
                  (package.submodule_search_locations or ())] if package else []
    path = next((candidate for candidate in candidates if candidate.is_file()),
                None)
    if path is None:
        raise pytest.UsageError(
            f"{PRISMABUILD_TEST_BOUND_ENV} is set, but no "
            f"{PRISMABUILD_TEST_BOUND_PLUGIN} is installed to apply it: this "
            "session would run with no per-test bound")
    spec = importlib.util.spec_from_file_location(
        "_pq_prismabuild_pytest_test_bound", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Registered from pytest_configure, a historic hook: the plugin's own
    # pytest_configure runs now and reads the bound.
    config.pluginmanager.register(module, PRISMABUILD_TEST_BOUND_PLUGIN)


def test_bound_args(config) -> list[str]:
    """The child's arguments for the per-test bound the parent runs under.

    pytest-timeout's ``--timeout``, and PrismaBuild's plugin when the parent
    named it with ``-p``. A plugin the parent loaded from
    ``PRISMABUILD_TEST_TIMEOUT_S`` loads in the child the same way, since the
    child inherits the variable (PQ #1055); naming it with ``-p`` there
    would import the ``prismabuild`` package into the child. A bound the
    parent does not run under is not added.
    """
    argv = []
    timeout = getattr(config.option, "timeout", None)
    if timeout:
        argv.append(f"--timeout={timeout}")
    if PRISMABUILD_TEST_BOUND_PLUGIN in (getattr(config.option, "plugins", None)
                                        or ()):
        argv += ["-p", PRISMABUILD_TEST_BOUND_PLUGIN]
    return argv
