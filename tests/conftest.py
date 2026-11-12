"""Shared pytest fixtures and hooks: only what every test needs (#3215).

Every test's static inputs include this file and every module it imports, at
module level or inside a fixture, so Bazel and the gate's selector run every
test again when any of them changes. This file therefore imports nothing from
``prismaquant`` or ``tools``. An autouse fixture that resets a ``prismaquant``
module's process-global state registers the reset with ``_when_loaded``: it
applies at once to a module already loaded, and to one the test loads later as
soon as that module's body has run, so it still wins as it did when the fixture
imported the module itself. A test that never loads the module does not depend
on it. Fixtures that only some tests use live in helper modules those tests
import (``lane_table_fixtures``, ``graph_lane_fixtures``,
``legacy_capture_fixtures``).
``tests/test_conftest_reach.py`` pins all of this.

What is here: the autouse fixtures -- profile detection is process-global
(issue #197; see its docstring), and the suite runs in certified mode unless a
test asks for dev mode (PQ #1147) -- and the session hooks: collection-skip
exit status, the environment line, the ``fleet_data``/``task_suite`` opt-ins,
``own_process`` modules, and the golden-row check.
"""
from __future__ import annotations

from collections.abc import Callable
import fcntl
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType

import pytest


#: Resets the autouse fixtures below apply to a module only once a test has
#: loaded it, keyed by module name. This file imports none of these modules, so
#: a test that never loads one does not depend on it (#3215). ``_when_loaded`` applies a reset to a module already
#: in ``sys.modules`` at once, and ``_resets_follow_module_loads`` applies it to
#: one the test loads later. Each fixture registers through a ``MonkeyPatch``
#: of its own, so the registration ends with that fixture's teardown, and no
#: fixture newly depends on the test's ``monkeypatch``: each keeps the place
#: in setup and teardown order it had before.
_LOAD_PATCHES: dict[str, tuple[Callable[[ModuleType], None], ...]] = {}


def _when_loaded(registrations: pytest.MonkeyPatch, name: str,
                 patch: Callable[[ModuleType], None]) -> None:
    """Apply ``patch`` to module ``name`` now if it is loaded, and whenever it loads,
    until ``registrations`` is undone."""

    module = sys.modules.get(name)
    if module is not None:
        patch(module)
    registrations.setitem(_LOAD_PATCHES, name, (*_LOAD_PATCHES.get(name, ()), patch))


def _first_load(registrations: pytest.MonkeyPatch, name: str,
                setup: Callable[[ModuleType], None]) -> list[ModuleType]:
    """``[module]`` once this test has module ``name``, loaded now or later; else ``[]``.

    ``setup`` runs on that module when it is first seen, which is where the
    fixture used to run it: right after importing the module itself. A second
    load under the same name in the same test (a reload, or a fresh copy after
    a test dropped the entry) is left alone, as it was then.
    """

    first: list[ModuleType] = []

    def take(module: ModuleType) -> None:
        if not first:
            first.append(module)
            setup(module)

    _when_loaded(registrations, name, take)
    return first


@pytest.fixture(autouse=True, scope="session")
def _resets_follow_module_loads():
    """Apply each registered reset to a module as a test first loads it.

    Wraps ``SourceFileLoader.exec_module``, which runs every ``.py`` module's
    body, for the session: once the body has run, the patches registered for
    that module name apply in registration order. Session-scoped, so it is in
    place before any test's own fixtures register a patch or load a module.
    """

    load = importlib.machinery.SourceFileLoader.exec_module

    def exec_module(loader, module):
        load(loader, module)
        for patch in _LOAD_PATCHES.get(module.__name__, ()):
            patch(module)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(importlib.machinery.SourceFileLoader, "exec_module", exec_module)
        yield


@pytest.fixture(autouse=True)
def _restore_profile_detection_globals():
    """Snapshot and restore the process-global state ``detect_profile`` reads.

    Profile detection is not a pure function of its arguments. It consults five
    module-level mutables that any test can perturb and that no single test
    owns, and a perturbation that outlives its test silently changes what
    ``detect_profile`` answers for every test after it in that process:

    * ``prismaquant.vendored.OVERRIDE_ERRORS`` -- the dead-vendored-override
      record. ``registry._refuse_dead_vendored_override`` raises on a hit.
      Since issue #201 that raise escapes ``_resolve`` as a
      ``DeadVendoredOverrideError``, so a stray entry now fails the next test
      that detects that architecture, loudly and by name. It used to happen
      *inside* ``_resolve``'s ``except Exception: continue``, which merely
      DEMOTED the profile that matched and let detection fall through to
      ``DefaultProfile`` -- silent, and the reason #197 was so hard to see.
    * ``prismaquant.vendored._QWEN3_REGISTERED`` -- once True, ``register_qwen3``
      returns before the ``OVERRIDE_ERRORS.pop()`` that a successful override
      performs, so a stray ``"qwen3"`` entry can never self-heal.
    * ``registry._REGISTERED``, ``_REGISTRY_GENERATION`` and
      ``_DETECTION_ORDER_CACHE`` -- the registration list, and the derived
      detection order cached against its generation counter.

    Issue #197 was exactly the first two together:
    ``test_vendored_qwen3.py::test_register_qwen3_does_not_cache_a_failed_registration``
    forces ``register_qwen3()`` to fail, ``_fatal()`` records
    ``OVERRIDE_ERRORS["qwen3"]`` on the way past, and nothing puts it back --
    while ``_QWEN3_REGISTERED`` is restored to True, which is what makes the
    entry permanent. Every later ``detect_profile()`` on a qwen3 checkpoint in
    that process then answers ``DefaultProfile``, whose ``structure_spec()`` is
    None, so ``tessera_serving_scope.unit_structure_from_profile`` refuses with
    "explicit Tessera scope needs a declared model profile". That is how
    ``tests/test_tessera_export_scope.py`` came to pass alone and fail in
    company -- on whichever xdist worker happened to draw the two files in that
    order, which is why it was invisible in a serial run (``test_tessera_*``
    sorts before ``test_vendored_*``).

    Restoring is silent and unconditional rather than an assertion. Several
    tests perturb this state deliberately and are right to; the defect is never
    the perturbation, only its escape. Restoring at the one place that owns
    "process-global detection state" fixes the class, where guarding each
    polluter fixes an instance and waits for the next one.

    Cost is a dict copy and a short list copy per test, and the writes only
    happen when something actually changed.
    """
    saved = {}

    def snapshot_registry(registry):
        saved["registry"] = (list(registry._REGISTERED), registry._REGISTRY_GENERATION,
                             registry._DETECTION_ORDER_CACHE)

    def snapshot_vendored(vendored):
        saved["vendored"] = (dict(vendored.OVERRIDE_ERRORS), vendored._QWEN3_REGISTERED)

    with pytest.MonkeyPatch.context() as registrations:
        registries = _first_load(registrations, "prismaquant.model_profiles.registry",
                                 snapshot_registry)
        vendoreds = _first_load(registrations, "prismaquant.vendored", snapshot_vendored)
        try:
            yield
        finally:
            for vendored in vendoreds:
                saved_errors, saved_qwen3_registered = saved["vendored"]
                if vendored.OVERRIDE_ERRORS != saved_errors:
                    vendored.OVERRIDE_ERRORS.clear()
                    vendored.OVERRIDE_ERRORS.update(saved_errors)
                vendored._QWEN3_REGISTERED = saved_qwen3_registered
            for registry in registries:
                saved_registered, saved_generation, saved_order_cache = saved["registry"]
                if registry._REGISTERED != saved_registered:
                    # In place: importers bind this list object, not its name.
                    registry._REGISTERED[:] = saved_registered
                registry._REGISTRY_GENERATION = saved_generation
                registry._DETECTION_ORDER_CACHE = saved_order_cache


@pytest.fixture(autouse=True)
def _no_served_quantizer_identity_carried_between_tests():
    """No test inherits another test's served-quantizer identity binding.

    ``nvfp4_activation_contract._ACTIVE_SERVED_QUANTIZER`` is process-global,
    and ``bind_served_quantizer_identity`` refuses to quietly re-bind once
    something has priced under the first identity (the reuse guarantee behind
    RobTand/prismaquant#567) -- so once one test binds the registered
    operator, ``StaticActivationContract.quantize_dequantize`` takes the
    registered-operator branch for every later test in the same process, not
    only the one that bound it. A per-module reset does not cover every
    caller: ``tests/test_render_score_served_quantizer_identity.py`` resets
    *before* binding in most of its tests but never after, so a CPU test that
    exercises the contract's dynamic fallback
    (``test_stageb_one_pass_spill.py::
    test_row_local_qdq_admission_admits_the_nvfp4_activation_paths
    [served-static-scale]``) fails when it runs after that file in the same
    shard and passes alone (RobTand/prismaquant#1215) -- the same
    invisible-in-serial, shard-order-dependent shape as #197's
    profile-detection leak, on a different global.
    ``_reset_served_quantizer_identity_for_tests`` also drops the resolution
    cache (``_RESOLVED_SERVED_QUANTIZER``), so a test that monkeypatches
    registration or kernel loading to probe resolution
    (``tests/test_nvfp4_served_dequant_kernel.py``) cannot leave a stale
    resolved identity behind for the next test to read either.

    Resetting to the unbound state is safe as a suite-wide default: unbound is
    exactly the state production starts a process in, and it makes a served
    rung refuse rather than guess. It is not the same thing as
    ``priced_model_screen.py``'s MODEL default, which stays scoped per-module
    on purpose (a repository-wide default binding would mask a missing
    production bind); this fixture only clears what a prior test left behind,
    it never binds anything itself.
    """
    def reset(owner):
        owner._reset_served_quantizer_identity_for_tests()

    with pytest.MonkeyPatch.context() as registrations:
        owners = _first_load(registrations, "prismaquant.nvfp4_activation_contract", reset)
        yield
        for owner in owners:
            reset(owner)


@pytest.fixture(autouse=True)
def _no_staged_tier_policy_carried_between_tests():
    """No test inherits another test's strict staged-tier policy.

    The policy is process-global, and an entry point driven in-process
    activates it from its sealed args. A per-module reset (PQ #845) covers
    only the module that carries it; several modules drive the same entry
    points without one, and CI distributes tests with ``--dist worksteal``,
    so which tests run after a leaker is luck. On PR #901 it was nine tessera
    tests on one worker, each refused ``readset-not-staged`` (PQ #906).
    """
    def deactivate(policy):
        policy.deactivate_staged_tier_policy_for_tests()

    with pytest.MonkeyPatch.context() as registrations:
        policies = _first_load(registrations, "prismaquant.staged_tier_policy", deactivate)
        yield
        for policy in policies:
            deactivate(policy)


#: ``prismaquant.tessera_joint_aura.ACTIVATION_SCALE_ENV``, written out so this
#: file does not import the joint lease; ``tests/test_conftest_reach.py`` pins
#: that the two agree.
ACTIVATION_SCALE_ENV = "PRISMAQUANT_PROD_ACT_SCALES"


@pytest.fixture(autouse=True)
def _no_activation_scale_policy_carried_between_tests():
    """No test inherits another test's ``PRISMAQUANT_PROD_ACT_SCALES``.

    ``run_adjoint_capture`` writes the sealed ``production_act_scales`` of its
    config into the process environment, and the in-process stage-A tests feed
    it ``"0"``. The variable changes what the joint lease does (it skips the
    calibrated clamp), so a test that compares against a clamping oracle failed
    whenever ``--dist worksteal`` put it on a worker after such a test (PQ
    #2230's dense sidecar oracle failed on a Spark this way). Restore the
    incoming value, or its absence, after every test.
    """
    incoming = os.environ.get(ACTIVATION_SCALE_ENV)
    yield
    if incoming is None:
        os.environ.pop(ACTIVATION_SCALE_ENV, None)
    else:
        os.environ[ACTIVATION_SCALE_ENV] = incoming


@pytest.fixture(autouse=True)
def _certified_mode_unless_asked(monkeypatch):
    """Tests run in certified mode unless they ask for dev mode.

    Dev mode is the default outside the suite (PQ #1147): the gates refuse
    only under ``PRISMAQUANT_DEV_MODE=0``. Every refusal test in this suite
    was written against the certified default, so the suite-wide default is
    ``0`` and those tests keep their meaning. A dev-mode test sets ``1`` (or
    deletes the variable, to test the unset default) in its own body, which
    runs after this fixture and wins.
    """
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")


def pytest_sessionfinish(session, exitstatus):
    """A run whose only tests skipped at collection is green, not empty.

    pytest returns ``EXIT_NO_TESTS_COLLECTED`` (5) whenever
    ``session.testscollected`` is zero. A module that skips at *collection*
    time -- ``pytest.importorskip`` at module scope, directly or through an
    imported sibling test module -- reports its skip and collects nothing, so
    the run prints ``1 skipped`` and still exits 5.

    That is invisible in a full-suite run, because some other file collects.
    It matters where a shard gets its own pytest: a shard whose only file skips at collection exits 5, and the shard
    reads as failed even though pytest reported the skip and nothing went
    wrong (RobTand/prismaquant#915, seen on
    ``tests/test_tessera_stack_group_cli.py``, which imports
    ``test_glm_campaign_streaming`` -> ``test_glm5_next_streamed_forward_parity``
    -> ``pytest.importorskip("transformers.models.glm5_next")``).

    The conversion is deliberately narrow. It fires only when pytest itself
    reported at least one skip, so an empty collection with no skip -- a typo
    in a path, a ``-k`` that matched nothing, a deselect that emptied the run
    -- still exits 5 and still reads as a defect. Failures, errors and usage
    errors have their own exit codes and are never touched.
    """
    if exitstatus != pytest.ExitCode.NO_TESTS_COLLECTED:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None or not reporter.stats.get("skipped"):
        return
    session.exitstatus = pytest.ExitCode.OK


#: Distributions whose versions decide what this suite can run. The GLM
#: modules need transformers >= 5.16 (``test_glm5_next_streamed_forward_parity``
#: and the five modules that import from it), and Tessera is pinned
#: (``tools/resolve_*_dev_pin.py``).
ENVIRONMENT_DISTRIBUTIONS = ("torch", "transformers", "tessera-quant")


def environment_line() -> str:
    """The interpreter and the gating versions, as one summary line."""

    from importlib import metadata
    import platform

    parts = [f"python={platform.python_version()}"]
    for name in ENVIRONMENT_DISTRIBUTIONS:
        try:
            parts.append(f"{name}={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            parts.append(f"{name}=absent")
    return f"pq-test-environment: {' '.join(parts)} ({sys.executable})"


def pytest_terminal_summary(terminalreporter):
    """Name the environment in every session's output (PQ #1090).

    Six GLM modules skipped at collection in every run because
    the test interpreter had transformers 5.6.0, and nothing in a receipt
    said which transformers a run had. A run's output is its receipt, and a
    run with ``--no-header`` has no report header to carry this; the terminal
    summary is printed either way.
    """

    terminalreporter.write_line(environment_line())


# ---------------------------------------------------------------------------
# Modules that need a pytest process of their own (PQ #1008)
# ---------------------------------------------------------------------------
#
# Some harnesses cannot share a process: a module may need an import graph no
# other module in the session has touched, and in a session that has already
# imported something else it used to skip every test that needed it, so a
# multi-file shard reported them as skipped and stayed green (25 tests in the
# #996 run).
#
# A module marked ``own_process`` keeps its tests and their node ids. When
# the session collects only that module, nothing changes. When it collects
# other modules too, each of the module's items is replaced by a proxy with
# the same node id. The first proxy to run starts one child pytest over
# exactly the module's collected node ids, and every proxy then reports its
# own test's outcome from the child: a failure with the child's traceback, a
# skip with the child's reason, an expected failure as one. A test the child
# never reported fails and names the child's exit status. Under xdist the
# workers share one child per module through a lock in the run's common
# temporary root, so a module is never run twice in a session.
#
# The per-test bound applies to each test in the child, not to the module
# (PQ #1027). The child starts before any bound is armed for the first proxy,
# and runs under the bound the parent runs under: pytest-timeout's
# ``--timeout``, the only per-test bound this suite declares
# (``pytest-timeout`` is a test dependency in ``pyproject.toml``). A requested
# bound without the plugin is a failure, never a skip: pytest refuses an
# unrecognized ``--timeout``, and a child that cannot load the plugin exits
# non-zero and fails its proxies by exit status.

OWN_PROCESS_MARK = "own_process"
#: Set in a child session only: the file the child appends its reports to.
OWN_PROCESS_REPORT_ENV = "PQ_OWN_PROCESS_REPORT"
OWN_PROCESS_MEASURE_ENV = "PQ_OWN_PROCESS_MEASURE"


#: Marks a test that reads fleet-local campaign data that no native record
#: declares (PQ #1014): a gigabyte real-data read, a walk over a campaign
#: tree. Such reads bypass declared admission, so these tests skip by default,
#: with a reason that names the opt-in. They run when ``-m`` names the mark
#: (``-m fleet_data``) or ``PQ_FLEET_DATA_TESTS=1`` is set. A skip, not a deselection:
#: under xdist only the workers see a deselection, so a shard of nothing but
#: these tests would exit 5 ("no tests collected") and read as a failure.
FLEET_DATA_MARK = "fleet_data"
FLEET_DATA_ENV = "PQ_FLEET_DATA_TESTS"
FLEET_DATA_SKIP_REASON = (
    f"{FLEET_DATA_MARK}: reads fleet data no native record declares "
    f"(PQ #1014); run with -m {FLEET_DATA_MARK} or {FLEET_DATA_ENV}=1")
TASK_SUITE_MARK = "task_suite"
TASK_SUITE_ENV = "PQ_TASK_SUITE_TESTS"
#: The mark is interpolated, as in ``FLEET_DATA_SKIP_REASON``: a literal
#: "-m task_suite" reads to the dependency graph as a launch of
#: ``prismaquant/task_suite.py``, which this file does not run (#3215).
TASK_SUITE_SKIP_REASON = (
    f"{TASK_SUITE_MARK}: run the required layered gate with -m {TASK_SUITE_MARK} or "
    f"{TASK_SUITE_ENV}=1; recorded 76 passed, 0 skipped; "
    "action 20cb79a0504632a78f6412465012eb868f56f5a432b4717cba6ff3c619560bb6; "
    "receipt 0ed9d0134b60b0df181b15dd605b41e0ca5ce7e5e1fe0ef8d0ee14cb25440913")


class _ExplicitMarkerSelection:
    """Report a skip unless the run selects the required environment."""

    def __init__(self, mark, environment, reason):
        self.mark = mark
        self.environment = environment
        self.reason = reason

    def requested(self, config) -> bool:
        return (os.environ.get(self.environment) == "1"
                or self.mark in (config.getoption("markexpr", "") or ""))

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, session, config, items):
        if self.requested(config):
            return
        for item in items:
            if item.get_closest_marker(self.mark) is not None:
                item.add_marker(pytest.mark.skip(reason=self.reason))


def pytest_configure(config):
    config.pluginmanager.register(_ExplicitMarkerSelection(
        FLEET_DATA_MARK, FLEET_DATA_ENV, FLEET_DATA_SKIP_REASON), "pq-fleet-data-selection")
    config.pluginmanager.register(_ExplicitMarkerSelection(
        TASK_SUITE_MARK, TASK_SUITE_ENV, TASK_SUITE_SKIP_REASON), "pq-task-suite-selection")
    config.addinivalue_line(
        "markers",
        f"{OWN_PROCESS_MARK}: the module needs a pytest process of its own. In "
        "a session that collects other modules too, its tests run in one "
        "child pytest and report back under their own node ids (PQ #1008).")



class OwnProcessFailure(Exception):
    """A test failed, or never reported, in its module's own process."""


def _own_process_root(config) -> Path:
    base = Path(config._tmp_path_factory.getbasetemp())
    if os.environ.get("PYTEST_XDIST_WORKER"):
        # Every xdist worker's basetemp sits under the run's one root.
        base = base.parent
    return base / "own-process"


def _fold_own_process_reports(records, *, log):
    """Per node id, one outcome from a child's phase reports."""
    outcomes, collection = {}, []
    for record in records:
        if record["when"] == "collect":
            collection.append(record)
            continue
        folded = outcomes.setdefault(record["nodeid"],
                                     {"outcome": "passed", "phases": []})
        folded["phases"].append(record["when"])
        if record["outcome"] == "failed":
            message = f"[{record['when']}] {record['longrepr']}"
            if folded["outcome"] == "failed":
                folded["message"] += "\n" + message
            else:
                folded.update(outcome="failed", message=message)
        elif record["outcome"] == "skipped" and folded["outcome"] != "failed":
            if record.get("wasxfail") is not None:
                folded.update(outcome="xfailed", reason=record["wasxfail"])
            else:
                folded.update(outcome="skipped", reason=record["reason"])
    for folded in outcomes.values():
        # A test counts as passed only once its call phase reported: a child
        # that died inside a test leaves a setup report and nothing else.
        if folded["outcome"] == "passed" and "call" not in folded["phases"]:
            folded.update(outcome="failed", message=(
                "the test never reported its call phase in its own process "
                f"(phases: {folded['phases']}); see {log}"))
    return outcomes, collection


def _own_process_bound_args(config) -> list[str]:
    """The child's arguments for the per-test bound the parent runs under.

    pytest-timeout's ``--timeout``. The child names the plugin with ``-p`` only
    when the parent registered it under that explicit name (autoload disabled):
    a parent that autoloaded it registered it as ``timeout``, its child autoloads
    it too, and pytest refuses the same module under a second name ("Plugin
    already registered under a different name"). A bound the parent does not
    run under is not added.
    """
    timeout = getattr(config.option, "timeout", None)
    if not timeout:
        return []
    plugins = config.pluginmanager
    named = plugins.has_plugin("pytest_timeout") and not plugins.has_plugin("timeout")
    return [*(["-p", "pytest_timeout"] if named else []), f"--timeout={timeout}"]


def _own_process_measure_args(config, root, key):
    launcher = ["-m", "tests._own_process_profile", str(root / f"{key}.pstats")]
    argv, env = [], {}
    sources = getattr(config.option, "cov_source", [])
    if sources != [] and not getattr(config.option, "no_cov", False):
        argv = ["-p", "pytest_cov", "--cov-reset", "--cov-report=",
                "--cov-fail-under=0"]
        argv += [f"--cov={source}" for source in (sources or [""])]
        argv += ["--cov-config=" + str(config.option.cov_config)]
        if config.option.cov_branch:
            argv.append("--cov-branch")
        if config.option.cov_context:
            argv.append("--cov-context=" + config.option.cov_context)
        env["COVERAGE_FILE"] = str(root / f"{key}.coverage")
    return launcher, argv, env


def _write_own_process_measurement(root, key, nodeids, state, records,
                                   started, lock_wait, executed_child):
    phases = {phase: sum(row["duration"] for row in records
                        if row["when"] == phase)
              for phase in ("setup", "call", "teardown")}
    child = json.loads((root / f"{key}.child-measurement.json").read_text())
    record = {
        "schema": "prismaquant.own_process_measurement.v1",
        "nodeids": nodeids, "parent_pid": os.getpid(),
        "executed_child": executed_child, "lock_wait_s": lock_wait,
        "parent_elapsed_s": time.perf_counter() - started,
        "child_elapsed_s": child["elapsed_s"], "child_phase_s": phases,
        "child_unattributed_s": child["elapsed_s"] - sum(phases.values()),
        "child_profile": child["profile"], "child_coverage": child["coverage"],
        "child_returncode": state["returncode"], "child_error": state["error"],
    }
    (root / f"{key}.{os.getpid()}.measurement.json").write_text(
        json.dumps(record))


def _run_own_process(config, nodeids: tuple[str, ...]) -> dict:
    """Run ``nodeids`` in one child pytest, once per session, and read it."""
    measure = os.environ.get(OWN_PROCESS_MEASURE_ENV) == "1"
    started = time.perf_counter() if measure else None
    executed_child = False
    key = hashlib.sha256("\n".join(nodeids).encode()).hexdigest()[:16]
    root = _own_process_root(config)
    root.mkdir(parents=True, exist_ok=True)
    reports = root / f"{key}.jsonl"
    done = root / f"{key}.done.json"
    log = root / f"{key}.log"
    with open(root / f"{key}.lock", "a+") as lock:
        waiting = time.perf_counter() if measure else None
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock_wait = time.perf_counter() - waiting if measure else None
        try:
            if not done.exists():
                executed_child = True
                reports.unlink(missing_ok=True)
                env = {name: value for name, value in os.environ.items()
                       if not name.startswith("PYTEST_XDIST_")
                       and name not in ("PYTEST_ADDOPTS", "PYTEST_CURRENT_TEST")}
                env[OWN_PROCESS_REPORT_ENV] = str(reports)
                launcher, measure_args = ["-m", "pytest"], []
                if measure:
                    launcher, measure_args, measure_env = _own_process_measure_args(
                        config, root, key)
                    env.update(measure_env)
                argv = [sys.executable, "-s", *launcher, "-q", "--no-header",
                        "-p", "no:cacheprovider",
                        "--rootdir", str(config.rootpath),
                        "--basetemp", str(root / f"{key}.tmp")]
                if config.inipath is not None:
                    argv += ["-c", str(config.inipath)]
                argv += _own_process_bound_args(config)
                argv += measure_args
                argv += list(nodeids)
                returncode, error = None, None
                child_started = time.perf_counter() if measure else None
                try:
                    with open(log, "wb") as out:
                        returncode = subprocess.run(
                            argv, cwd=str(config.rootpath), env=env,
                            stdout=out, stderr=subprocess.STDOUT).returncode
                except BaseException as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    state = {"returncode": returncode, "error": error}
                    if measure:
                        child_measurement = {
                            "elapsed_s": time.perf_counter() - child_started,
                            "profile": str(root / f"{key}.pstats"),
                            "coverage": measure_env.get("COVERAGE_FILE"),
                        }
                    done.write_text(json.dumps(state))
                    if measure:
                        (root / f"{key}.child-measurement.json").write_text(
                            json.dumps(child_measurement))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    state = json.loads(done.read_text())
    lines = reports.read_text().splitlines() if reports.exists() else []
    records = [json.loads(line) for line in lines]
    if measure:
        _write_own_process_measurement(root, key, nodeids, state, records,
                                       started, lock_wait, executed_child)
    outcomes, collection = _fold_own_process_reports(records, log=log)
    reported_failure = any(item["outcome"] == "failed" for item in outcomes.values())
    collection_failure = any(item["outcome"] == "failed" for item in collection)
    collection_skip = (not outcomes and collection
                       and all(item["outcome"] == "skipped" for item in collection))
    expected_exit = (state["returncode"] == pytest.ExitCode.OK
                     or (state["returncode"] == pytest.ExitCode.TESTS_FAILED
                         and (reported_failure or collection_failure))
                     or (state["returncode"] == pytest.ExitCode.NO_TESTS_COLLECTED
                         and collection_skip))
    if state.get("error") or not expected_exit:
        # Keep ordinary failed/skip/xfail mappings. A failure after the last
        # report still needs one proxy to carry the child's terminal failure.
        target = next((nodeid for nodeid in nodeids
                       if outcomes.get(nodeid, {}).get("outcome") == "failed"), None)
        if target is None:
            target = next((nodeid for nodeid in nodeids
                           if outcomes.get(nodeid, {}).get("outcome") == "passed"),
                          nodeids[0])
        folded = outcomes.setdefault(target, {"outcome": "passed", "phases": []})
        message = (f"the module's own pytest process exited unexpectedly "
                   f"(exit {state['returncode']}"
                   + (f", {state['error']}" if state.get("error") else "")
                   + f"); see {log}")
        if folded["outcome"] == "failed":
            message = folded["message"] + "\n" + message
        elif collection_failure:
            message += "\n" + "\n".join(item["longrepr"] for item in collection
                                           if item["outcome"] == "failed")
        folded.update(outcome="failed", message=message)
    return {"state": state, "outcomes": outcomes, "collection": collection,
            "log": log}


class _OwnProcessGroup:
    """One own-process module's collected node ids, run at most once."""

    def __init__(self, nodeids):
        self.nodeids = tuple(nodeids)
        self.result = None
        self.error = None

    def start(self, config):
        """Run the child before any test's bound is armed.

        An error is kept for the proxies to report as test failures: raised
        here, outside any test, it would end the session instead.
        """
        if self.result is None and self.error is None:
            try:
                self.result = _run_own_process(config, self.nodeids)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"

    def run(self, config):
        if self.error is not None:
            raise OwnProcessFailure(
                f"the module's own pytest process did not run: {self.error}")
        if self.result is None:
            self.result = _run_own_process(config, self.nodeids)
        return self.result


class OwnProcessItem(pytest.Item):
    """Stands in for one test of an own-process module in a shared session."""

    def __init__(self, *, group, source_reportinfo, **kwargs):
        super().__init__(**kwargs)
        self.group = group
        self.source_reportinfo = source_reportinfo

    def runtest(self):
        result = self.group.run(self.config)
        outcome = result["outcomes"].get(self.nodeid)
        if outcome is None:
            for record in result["collection"]:
                if record["outcome"] == "skipped":
                    pytest.skip(record["reason"])
            try:
                tail = result["log"].read_text(errors="replace").splitlines()[-40:]
            except OSError:
                tail = []
            state = result["state"]
            raise OwnProcessFailure(
                f"{self.nodeid} reported nothing from its own pytest process "
                f"(exit {state['returncode']}"
                + (f", {state['error']}" if state.get("error") else "")
                + f"; log {result['log']}):\n" + "\n".join(
                    [record["longrepr"] for record in result["collection"]
                     if record["outcome"] == "failed"] + tail))
        if outcome["outcome"] == "failed":
            raise OwnProcessFailure(outcome["message"])
        if outcome["outcome"] == "xfailed":
            pytest.xfail(outcome["reason"])
        if outcome["outcome"] == "skipped":
            pytest.skip(outcome["reason"])

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, OwnProcessFailure):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        path, line, _description = self.source_reportinfo
        return path, line, f"{self.nodeid} (own process)"


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    """Give each ``own_process`` module its own process in a shared session.

    ``trylast`` so that ``-k``, ``-m`` and every other plugin have already
    chosen the items: the child runs exactly the node ids chosen here.
    """
    if os.environ.get(OWN_PROCESS_REPORT_ENV):
        return
    if len({item.path for item in items}) < 2:
        return
    modules: dict[Path, list[int]] = {}
    for index, item in enumerate(items):
        if item.get_closest_marker(OWN_PROCESS_MARK) is not None:
            modules.setdefault(item.path, []).append(index)
    for indices in modules.values():
        group = _OwnProcessGroup(items[index].nodeid for index in indices)
        for index in indices:
            original = items[index]
            items[index] = OwnProcessItem.from_parent(
                original.parent, name=original.name, group=group,
                source_reportinfo=original.reportinfo())


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Start an own-process module's child outside every per-test bound.

    The first proxy used to start the child inside its own call phase, so one
    test's bound covered the whole module (PQ #1027): a module could time out
    on its test count, and the proxy that failed was the first one, not the
    one that hung. ``tryfirst`` makes this the outermost wrapper, so the child
    runs before pytest-timeout's protocol timer
    is armed for the proxy, and each test in the child runs under a
    bound of its own (``_own_process_bound_args``). The child's collection
    and exit are bounded as the parent session's own are, by the run's
    ceiling. Its wall time is not attributed to any proxy in ``--durations``
    or the junit report.
    """
    if isinstance(item, OwnProcessItem):
        item.group.start(item.config)
    return (yield)


def _own_process_record(report) -> dict:
    reason = ""
    if report.skipped:
        longrepr = report.longrepr
        reason = (str(longrepr[2]) if isinstance(longrepr, tuple)
                  and len(longrepr) == 3 else str(longrepr or ""))
        if reason.startswith("Skipped: "):
            reason = reason[len("Skipped: "):]
    record = {"nodeid": report.nodeid, "when": getattr(report, "when", "collect"),
            "outcome": report.outcome,
            "longrepr": report.longreprtext if report.failed else "",
            "reason": reason, "wasxfail": getattr(report, "wasxfail", None)}
    if (os.environ.get(OWN_PROCESS_MEASURE_ENV) == "1"
            and isinstance(report, pytest.TestReport)):
        record["duration"] = report.duration
    return record


def _append_own_process_record(report) -> None:
    path = os.environ.get(OWN_PROCESS_REPORT_ENV)
    if path:
        with open(path, "a", encoding="utf-8") as out:
            out.write(json.dumps(_own_process_record(report)) + "\n")


def pytest_runtest_logreport(report):
    """In a child session, hand each phase's report to the parent."""
    _append_own_process_record(report)


def pytest_collectreport(report):
    """In a child session, hand a failed or skipped collection to the parent."""
    if not report.passed:
        _append_own_process_record(report)


# -- golden tables: every frozen row is used (tests/golden_table.py) ---------

_GOLDEN_OUTCOME = pytest.StashKey[str]()
_GOLDEN_DESELECTED = pytest.StashKey[set]()


def pytest_deselected(items):
    """Record each module ``-k``, ``-m`` or ``--deselect`` did not select whole."""
    for item in items:
        item.config.stash.setdefault(_GOLDEN_DESELECTED, set()).add(item.path)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Keep the outcome each test reached, for the golden-row check below."""
    report = yield
    if report.when == "call" or not report.passed:
        item.stash[_GOLDEN_OUTCOME] = report.outcome
    return report


@pytest.fixture(scope="module", autouse=True)
def _golden_rows_all_used(request):
    """Fail a module whose opted-in golden table kept rows no passing call used.

    Active only when this process ran the whole module; see
    ``golden_table.unfinished_tests_if_whole_module_ran``.
    """
    yield
    # A module holding a GoldenTable has imported tests.golden_table itself;
    # while that module is not loaded there is no table to check.
    golden = sys.modules.get("tests.golden_table")
    if golden is None:
        return
    tables = [table for table in vars(request.module).values()
              if isinstance(table, golden.GoldenTable) and table.every_row_used]
    if not tables:
        return
    module = request.node.path
    unfinished = golden.unfinished_tests_if_whole_module_ran(
        module, args=request.config.args,
        deselected=request.config.stash.get(_GOLDEN_DESELECTED, set()),
        outcomes={item.nodeid: item.stash.get(_GOLDEN_OUTCOME, None)
                  for item in request.session.items if item.path == module})
    if unfinished is None:
        return
    for table in tables:
        unused = table.unused_rows(unfinished=unfinished)
        if unused:
            pytest.fail(
                f"fixtures/{table.name}.json has {len(unused)} frozen row(s) that no "
                f"passing call used: {unused[:10]}. Restore the call, or archive the "
                "rows of a retired test with its archive wall.", pytrace=False)
