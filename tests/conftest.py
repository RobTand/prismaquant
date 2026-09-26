"""Shared pytest fixtures.

Formerly minimal: this repo had no conftest before 2026-08-30. It carried one
NON-autouse fixture, ``synthetic_cb_target``, which let a codebook export test
declare that its bodies were CPU fixtures rather than served artifacts, through
the route-status gate's own ``PQ_CB_NON_NATIVE_TARGET`` declaration. That gate,
that declaration and the export tests that used it all went into
``archive/gridbook_lane_2026-09-02/`` when the Gridbook codebook serving lane
was retired on 2026-09-02, so the fixture has no subject left.

What is here now is the autouse fixtures -- profile detection is
process-global (issue #197; see its docstring), and the suite runs in
certified mode unless a test asks for dev mode (PQ #1147) -- and the
legacy-lane-grammar fixtures (``legacy_v4_contract``, ``legacy_v5_contract``,
``down_convert_lane_table``), which exist so a test about an OLDER Tessera
lane grammar owns its fixture instead of asserting that the installed
contract is a version it no longer is.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _installed_contract() -> dict:
    """The Tessera contract the environment installs, parsed."""
    from importlib.resources import as_file

    from prismaquant import tessera_runtime_contract as contract

    with as_file(contract.contract_path()) as path:
        return json.loads(path.read_text(encoding="utf-8"))


def down_convert_lane_table(payload: dict, schema: str) -> dict:
    """The installed contract, expressed in an OLDER lane grammar.

    Legacy-grammar tests need a legacy table. They used to get one by reading
    the installed contract, which was v4 at the time -- so on the day Tessera
    shipped ``tessera.lane-eligibility.v6`` (its PR #176) every such test began
    asserting that the installed contract was a version it no longer is. That
    is a fixture problem, not a reader problem: a test about a legacy grammar
    must OWN its legacy fixture, exactly as ``test_tessera_contract_v5`` already
    did by rewriting ``runtime`` onto every cell.

    Down-converting rather than hand-writing keeps the fixture honest about
    everything the grammar did not change -- families, rungs, route statuses,
    launches -- so the test still exercises real cells. It is a FIXTURE and
    never an attestation: nothing derived from it is recorded anywhere.

    ``schema`` is ``tessera.lane-eligibility.v9`` (reduce each v10
    ``platforms`` entry back to a bare key set and drop the platforms that
    exist only to say what they do NOT execute), ``...v8`` (drop the v9
    ``smoke.record`` as well), ``...v7`` (drop the v8 ``evidence.artifact`` as well),
    ``...v6`` (drop v7's ``smoke.attribution`` and ``smoke.control`` too),
    ``...v5`` (drop the whole ``evidence`` block and the ``runtime`` version
    fields) or ``...v4`` (drop the per-cell ``runtime`` scope as well).

    Dropping v9's record is not only a key removal: v9 DERIVES
    ``smoke.attribution`` from the record, and v7/v8 derive it from the
    control.  A fixture that removed the record and kept the derived
    attribution would be a table no validator would accept, so the
    attribution is re-derived under the older rule -- through the reader's own
    ``derive_smoke_attribution``, which is that rule's home here.
    """
    from prismaquant.lane_eligibility import (
        EVIDENCE_ATTRIBUTION_UNATTRIBUTED, EVIDENCE_OUTCOME_IDENTICAL,
        EVIDENCE_ATTRIBUTION_SHARED, EVIDENCE_ATTRIBUTION_NOT_SHARED)

    payload = copy.deepcopy(payload)
    lane = payload["lane_eligibility"]
    lane["schema"] = schema
    version = int(schema.rsplit(".v", 1)[1])
    if version <= 9:
        # v10 made a platform entry an OBJECT that states what the platform
        # executes.  Under v9 the value was never read, and a platform with no
        # cell said nothing at all -- so a v9 fixture keeps only the platforms
        # that carry cells, with only the identity key v22 published.
        attested = {cell["platform"] for cell in lane["cells"]}
        lane["platforms"] = {
            key: {k: v for k, v in entry.items()
                  if k in ("compute_capability", "gcn_arch")}
            for key, entry in lane["platforms"].items()
            if key in attested
        }
    for cell in lane["cells"]:
        if version <= 5:
            cell.pop("evidence", None)
        else:
            evidence = cell["evidence"]
            if version <= 8:
                smoke = evidence["smoke"]
                if smoke.pop("record", None) is not None:
                    control = smoke.get("control")
                    smoke["attribution"] = (
                        EVIDENCE_ATTRIBUTION_UNATTRIBUTED if control is None
                        else EVIDENCE_ATTRIBUTION_SHARED
                        if control["outcome"] == EVIDENCE_OUTCOME_IDENTICAL
                        else EVIDENCE_ATTRIBUTION_NOT_SHARED)
            if version <= 7:
                evidence.pop("artifact", None)
            if version <= 6:
                evidence["smoke"].pop("attribution", None)
                evidence["smoke"].pop("control", None)
        if version <= 4:
            cell.pop("runtime", None)
        elif version <= 5:
            runtime = cell.get("runtime", {})
            cell["runtime"] = {"image": runtime["image"],
                               "execution_modes": runtime["execution_modes"]}
    return payload


def lane_cells_on_one_image(payload: dict, image: str | None = None) -> dict:
    """The installed table's cells measured on ONE runtime image.

    Several fixtures flatten every cell's ``runtime`` onto one fixture image
    (or down-convert below v6, where the image axis does not exist). That was
    safe while no two cells of one (platform, family, structure, regime)
    scope lived on different images. Tessera contract v38 broke it: the stock
    image's dense ``TESSERA_E4M3_K1`` pair (resident|streamed) and the GLM
    image's dense ``_resident`` pair are one scope on two images, and the
    flattened union is a table in which two cells claim the same scope --
    which the reader correctly refuses. A fixture therefore keeps one real
    image's roster and never relabels a union.

    ``image`` defaults to ``versions.default_serve_image``, whose roster
    (the dense E2M1 and E4M3 pairs) is what these fixtures were written
    against. A FIXTURE, never an attestation.
    """
    payload = copy.deepcopy(payload)
    if image is None:
        image = payload["versions"]["default_serve_image"]
    lane = payload["lane_eligibility"]
    kept = [cell for cell in lane["cells"] if cell["runtime"]["image"] == image]
    assert kept, f"the installed table publishes no cell on {image!r}"
    lane["cells"] = kept
    return payload


def project_lane_cells_onto_structures(payload: dict, structures) -> dict:
    """The installed contract's cell coverage, re-addressed to ``structures``.

    Contract v23 (lane schema v10) made family coverage structure-specific:
    the packaged dense cells publish TESSERA_E2M1_K2 only, while
    TESSERA_E4M3_K1 is published only as routed_moe. A fixture that
    synthesizes one structure by relabelling the OTHER structure's cells --
    how the allocator fixtures were built when the packaged contract carried
    no routed_moe cell at all -- therefore loses every family the source
    structure did not cover, and the admission gate (correctly) refuses the
    rungs those tests select. This helper instead keeps each (platform,
    family, regime) scope's own coverage and re-addresses it onto exactly
    ``structures``:

    * a cell whose structure is requested is kept as published;
    * every published cell is cloned into each requested structure its
      scope does not already cover, so a clone carries its family's own
      admission facts -- rungs, activation contract, route status, plugin
      requirement, launch set -- and only ``structure``, a pure lookup
      discriminator (``cell_matches_serving_context``), is synthesized;
    * cells whose structure is not requested are dropped after cloning, so
      ``project_lane_cells_onto_structures(payload, ("dense",))`` is a
      dense-only table whose dense coverage is still every scope's.

    A clone is added only when neither a kept cell nor an earlier clone in
    the same scope claims an overlapping residency, read off the published
    ``TESSERA_SERVE_MODE`` flag by the same grammar
    ``parse_v4_cell_contract`` reads. The overlap
    key deliberately ignores the runtime image/execution axes because the
    fixtures that call this then flatten every cell's runtime onto one
    fixture image, which would otherwise reintroduce the overlap the reader
    refuses. Like ``down_convert_lane_table`` this is a FIXTURE, never an
    attestation: nothing derived from it is recorded anywhere.
    """
    wanted = list(dict.fromkeys(str(structure) for structure in structures))
    payload = copy.deepcopy(payload)
    lane = payload["lane_eligibility"]
    head = "TESSERA_SERVE_MODE="

    def _residencies(cell: dict) -> frozenset[str]:
        for flag in cell.get("requires_serve_flags", ()):
            if str(flag).startswith(head):
                return frozenset(str(flag)[len(head):].split("|"))
        return frozenset()

    kept = [cell for cell in lane["cells"] if cell.get("structure") in wanted]
    claimed: dict[tuple[str, str, str, str], set[str]] = {}
    for cell in kept:
        key = (cell["platform"], cell["family"], cell["regime"], cell["structure"])
        claimed.setdefault(key, set()).update(_residencies(cell))
    clones = []
    # Clone from EVERY published cell, not only the kept ones: a scope whose
    # only cell lives in a dropped structure still owns that scope's family
    # facts, and the requested structures inherit them from it.
    for cell in lane["cells"]:
        for structure in wanted:
            if structure == cell.get("structure"):
                continue
            key = (cell["platform"], cell["family"], cell["regime"], structure)
            if _residencies(cell) & claimed.get(key, set()):
                continue
            clone = copy.deepcopy(cell)
            clone["id"] = f"{cell['id']}_{structure}_fixture"
            clone["structure"] = structure
            clones.append(clone)
            claimed.setdefault(key, set()).update(_residencies(cell))
    lane["cells"] = kept + clones
    published = [s for s in lane.get("structures", []) if s in wanted]
    lane["structures"] = published + [s for s in wanted if s not in published]
    return payload


@pytest.fixture
def legacy_v4_contract() -> dict:
    """The installed contract expressed as a v4 lane table.

    v4 has no runtime-image axis, so the default image's roster is kept
    first (``lane_cells_on_one_image``): since contract v38 two images
    publish the same dense E4M3 scopes, and without the axis they overlap.
    """
    return down_convert_lane_table(lane_cells_on_one_image(_installed_contract()),
                                   "tessera.lane-eligibility.v4")


@pytest.fixture
def legacy_v5_contract() -> dict:
    """The installed contract expressed as a v5 lane table."""
    return down_convert_lane_table(_installed_contract(),
                                   "tessera.lane-eligibility.v5")


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
    from prismaquant.model_profiles import registry
    import prismaquant.vendored as vendored

    saved_errors = dict(vendored.OVERRIDE_ERRORS)
    saved_qwen3_registered = vendored._QWEN3_REGISTERED
    saved_registered = list(registry._REGISTERED)
    saved_generation = registry._REGISTRY_GENERATION
    saved_order_cache = registry._DETECTION_ORDER_CACHE
    try:
        yield
    finally:
        if vendored.OVERRIDE_ERRORS != saved_errors:
            vendored.OVERRIDE_ERRORS.clear()
            vendored.OVERRIDE_ERRORS.update(saved_errors)
        vendored._QWEN3_REGISTERED = saved_qwen3_registered
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
    from prismaquant import nvfp4_activation_contract as owner

    owner._reset_served_quantizer_identity_for_tests()
    yield
    owner._reset_served_quantizer_identity_for_tests()


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
    from prismaquant.staged_tier_policy import (
        deactivate_staged_tier_policy_for_tests)
    deactivate_staged_tier_policy_for_tests()
    yield
    deactivate_staged_tier_policy_for_tests()


@pytest.fixture(autouse=True, scope="module")
def _no_prismabuild_import_carried_between_modules():
    """No module inherits another module's ``prismabuild`` imports (PQ #1281).

    The per-test restore below cannot see an import made by a module-scoped
    fixture: that fixture is set up before the test's own snapshot is taken.
    ``tests/test_fleet_acceptance_level1.py``'s and
    ``tests/test_fullstack_real_chain.py``'s ``pb`` fixtures import from a
    sealed generation that way. An autouse fixture is set up before the other
    fixtures of its scope, so this snapshot comes first and the module's
    teardown puts the imports back.
    """
    from fleet_sdk import prismabuild_imports_restored
    with prismabuild_imports_restored():
        yield


@pytest.fixture(autouse=True)
def _no_prismabuild_import_carried_between_tests():
    """No test inherits another test's ``prismabuild`` imports (PQ #1281).

    Where ``import prismabuild`` resolves is process-global: ``sys.path``
    and ``sys.modules``. A test that imports PrismaBuild from a sealed
    generation tree left both pointing at it, and every later test on that
    worker then ran against the generation instead of the installed SDK
    (``tests/test_stageb_one_pass_spill.py`` then
    ``tests/test_strict_reader_tier_enforcement.py`` refused
    ``no-claim-context``). ``fleet_sdk.prismabuild_imports_restored`` says
    what is put back and why.
    """
    from fleet_sdk import prismabuild_imports_restored
    with prismabuild_imports_restored():
        yield


@pytest.fixture(autouse=True)
def _no_staged_range_wait_unless_asked(monkeypatch):
    """Tests do not wait on a PrismaBuild fleet that is not running.

    ``StagedShardReader._range_for`` waits a bounded time for a range the
    residency map does not cover yet, because on the fleet an uncovered span
    is usually a mover that has not run rather than a range that will never
    arrive (PQ #874). Under pytest there are no movers: every uncovered span
    is permanent, so the default 300 s bound would turn each of the several
    existing tests that assert an immediate ``readset-not-staged`` refusal
    into a five-minute sleep that still passes.

    Setting the bound to 0 here restores the pre-#874 refuse-on-first-miss
    behaviour as the suite-wide default, which is what those tests were
    written against. The tests that are ABOUT the wait set the variable
    themselves; a test's own ``monkeypatch.setenv`` runs after this fixture
    and wins.

    This hides no cost. What waits on the fleet is
    ``layer_streaming._await_layer_readset``, and only it: the strict policy
    path, the layer read, and only spans PrismaBuild's sealed readset
    declares and no mover has written yet. A span the readset does not
    declare, an entry that covers a span and fails a check, and an unbound
    readset all refuse at once there, exactly as they do here. The one
    behaviour this default replaces is the 300 s a *declared* range that
    never arrives would cost before failing -- which needs movers to be
    meaningful, and there are none under pytest.
    """
    monkeypatch.setenv("PRISMAQUANT_STAGED_RANGE_WAIT_S", "0")


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
    It matters under PrismaBuild, where ``pbtest`` gives each shard its own
    pytest: a shard whose only file skips at collection exits 5, and the shard
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
#: and the five modules that import from it), and Tessera and PrismaBuild are
#: pinned (``tools/resolve_*_dev_pin.py``).
ENVIRONMENT_DISTRIBUTIONS = ("torch", "transformers", "tessera-quant", "prismabuild")


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

    Six GLM modules skipped at collection in every PrismaBuild run because
    the test interpreter had transformers 5.6.0, and nothing in a receipt
    said which transformers a run had. A shard's output is its receipt, and
    ``pbtest`` runs pytest with ``--no-header``, so the report header cannot
    carry this; the terminal summary is printed either way.
    """

    terminalreporter.write_line(environment_line())


# ---------------------------------------------------------------------------
# Modules that need a pytest process of their own (PQ #1008)
# ---------------------------------------------------------------------------
#
# Some harnesses cannot share a process. The Stage A produced-output harness
# (``test_stage_a_produced_boundary_chain``) resolves a pinned PrismaBuild
# bundle, and one process can hold only one ``prismabuild``: in a session
# that has already imported the deployed one, the harness used to skip every
# test that needed the bundle, so a multi-file shard reported them as
# skipped and stayed green (25 tests in the #996 run).
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
# ``--timeout``, PrismaBuild's ``prismabuild.pytest_test_bound``, or both. A
# test that hangs in the child fails alone, with the bound named in its
# failure, and the tests after it still run.
#
# PrismaBuild's bound is on whenever ``PRISMABUILD_TEST_TIMEOUT_S`` is set
# (PQ #1055). ``pbtest`` exports the variable to every shard, but no shard
# names the plugin with ``-p`` and ``prismabuild`` registers no entry point,
# so this conftest registers it. It loads the plugin from its file rather
# than importing the ``prismabuild`` package: an ``own_process`` harness
# refuses a process whose ``prismabuild`` is not its pinned candidate, and a
# child session inherits the variable and loads the plugin the same way.

OWN_PROCESS_MARK = "own_process"
#: Set in a child session only: the file the child appends its reports to.
OWN_PROCESS_REPORT_ENV = "PQ_OWN_PROCESS_REPORT"
#: PrismaBuild's per-test bound plugin, by the name ``-p`` loads it under.
PRISMABUILD_TEST_BOUND_PLUGIN = "prismabuild.pytest_test_bound"
#: The per-test bound in seconds, which ``pbtest`` exports to every shard and
#: the plugin reads. Unset or empty: no bound.
PRISMABUILD_TEST_BOUND_ENV = "PRISMABUILD_TEST_TIMEOUT_S"


def _load_prismabuild_test_bound(config) -> None:
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


#: Marks a test that reads fleet-local campaign data or live PrismaBuild
#: state that no PB action declares (PQ #1014): a gigabyte real-data read, a
#: walk over a campaign tree, a read of the live queue. Such reads bypass
#: PB's admission and tiered caching, so these tests skip by default, with a
#: reason that names the opt-in. They run when ``-m`` names the mark
#: (``pbtest --pytest-args "-m fleet_data"``) or ``PQ_FLEET_DATA_TESTS=1`` is
#: set; declaring their reads to PB is PB #915. A skip, not a deselection:
#: under xdist only the workers see a deselection, so a shard of nothing but
#: these tests would exit 5 ("no tests collected") and read as a failure.
FLEET_DATA_MARK = "fleet_data"
FLEET_DATA_ENV = "PQ_FLEET_DATA_TESTS"
FLEET_DATA_SKIP_REASON = (
    f"{FLEET_DATA_MARK}: reads fleet data PrismaBuild does not declare "
    f"(PQ #1014); run with -m {FLEET_DATA_MARK} or {FLEET_DATA_ENV}=1")


class _FleetDataSelection:
    """Skip ``fleet_data`` tests unless the run asks for them."""

    @staticmethod
    def requested(config) -> bool:
        return (os.environ.get(FLEET_DATA_ENV) == "1"
                or FLEET_DATA_MARK in (config.getoption("markexpr", "") or ""))

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, session, config, items):
        if self.requested(config):
            return
        for item in items:
            if item.get_closest_marker(FLEET_DATA_MARK) is not None:
                item.add_marker(pytest.mark.skip(reason=FLEET_DATA_SKIP_REASON))


def pytest_configure(config):
    config.pluginmanager.register(_FleetDataSelection(), "pq-fleet-data-selection")
    config.addinivalue_line(
        "markers",
        f"{OWN_PROCESS_MARK}: the module needs a pytest process of its own. In "
        "a session that collects other modules too, its tests run in one "
        "child pytest and report back under their own node ids (PQ #1008).")
    _load_prismabuild_test_bound(config)


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


def _run_own_process(config, nodeids: tuple[str, ...]) -> dict:
    """Run ``nodeids`` in one child pytest, once per session, and read it."""
    key = hashlib.sha256("\n".join(nodeids).encode()).hexdigest()[:16]
    root = _own_process_root(config)
    root.mkdir(parents=True, exist_ok=True)
    reports = root / f"{key}.jsonl"
    done = root / f"{key}.done.json"
    log = root / f"{key}.log"
    with open(root / f"{key}.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if not done.exists():
                reports.unlink(missing_ok=True)
                env = {name: value for name, value in os.environ.items()
                       if not name.startswith("PYTEST_XDIST_")
                       and name not in ("PYTEST_ADDOPTS", "PYTEST_CURRENT_TEST")}
                env[OWN_PROCESS_REPORT_ENV] = str(reports)
                argv = [sys.executable, "-m", "pytest", "-q", "--no-header",
                        "-p", "no:cacheprovider",
                        "--rootdir", str(config.rootpath),
                        "--basetemp", str(root / f"{key}.tmp")]
                if config.inipath is not None:
                    argv += ["-c", str(config.inipath)]
                argv += _own_process_bound_args(config)
                argv += list(nodeids)
                returncode, error = None, None
                try:
                    with open(log, "wb") as out:
                        returncode = subprocess.run(
                            argv, cwd=str(config.rootpath), env=env,
                            stdout=out, stderr=subprocess.STDOUT).returncode
                except BaseException as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    done.write_text(json.dumps(
                        {"returncode": returncode, "error": error}))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    state = json.loads(done.read_text())
    lines = reports.read_text().splitlines() if reports.exists() else []
    outcomes, collection = _fold_own_process_reports(
        [json.loads(line) for line in lines], log=log)
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

    def __init__(self, *, group, **kwargs):
        super().__init__(**kwargs)
        self.group = group

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
        return self.path, None, f"{self.nodeid} (own process)"


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
                original.parent, name=original.name, group=group)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Start an own-process module's child outside every per-test bound.

    The first proxy used to start the child inside its own call phase, so one
    test's bound covered the whole module (PQ #1027): a module could time out
    on its test count, and the proxy that failed was the first one, not the
    one that hung. ``tryfirst`` makes this the outermost wrapper, so the child
    runs before pytest-timeout's protocol timer or PrismaBuild's per-phase
    alarms are armed for the proxy, and each test in the child runs under a
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
    return {"nodeid": report.nodeid, "when": getattr(report, "when", "collect"),
            "outcome": report.outcome,
            "longrepr": report.longreprtext if report.failed else "",
            "reason": reason, "wasxfail": getattr(report, "wasxfail", None)}


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
