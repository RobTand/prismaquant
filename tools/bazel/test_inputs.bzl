"""Inputs a test reads that no static evidence names, each with its reason.

tools/bazel/build_files.py derives every other input from the source. A test
listed here reads files through a root it receives as a parameter, or runs a
child process the analysis can't follow. The remote executor makes such a read
fail rather than pass, which is how an entry is found. Keep each entry to the
files named, and the reason true.

data and deps are labels. modules are repository modules the test runs by a
name no evidence shows; the generator turns each into its library.
"""

TEST_INPUTS = {
    "//tests:test_conftest_names_the_test_environment": struct(
        deps = ["@pq_pip//tessera_quant", "@pq_pip//torch", "@pq_pip//transformers"],
        reason = "it checks the terminal summary names torch, transformers and tessera-quant versions, which conftest reads from installed metadata without importing them.",
    ),
    "//tests:test_cost_surrogate_calibration_anchors": struct(
        data = ["//:archive/grouped_kl_2026-05-28/docs/grouped_kl_allocator_results_2026-05-20.md"],
        reason = "tests/fixtures/cost_surrogate_calibration_anchors.json names the source_doc it checks exists.",
    ),
    "//tests:test_duplication_baseline": struct(
        data = ["//:files/prismaquant", "//:files/tools"],
        reason = "tools/duplication_inventory.py scans prismaquant and tools under the root its caller passes.",
    ),
    "//tests:test_gate_node_partition": struct(
        deps = ["@pq_pip//pytest_xdist"],
        modules = ["tools/ci/gate_collect.py"],
        reason = "it runs tools/ci/gate_run.py, whose child pytest loads the gate_collect plugin by name (-p gate_collect) and runs xdist (-n).",
    ),
    "//tests:test_gate_run": struct(
        deps = ["@pq_pip//pytest_xdist"],
        modules = ["tools/ci/gate_collect.py"],
        reason = "its child pytest loads the gate_collect plugin by name (-p gate_collect) and runs xdist (-n).",
    ),
    "//tests:test_import_fanout": struct(
        data = ["//:files", "//tests:files"],
        reason = "tools/ci/import_fanout.py scans tests, prismaquant, tools, experiments, scripts and pqtel under the root its caller passes.",
    ),
    "//tests:test_io_site_freeze": struct(
        data = ["//:files/prismaquant", "//:files/tools"],
        reason = "it scans every module under prismaquant and tools from a root it computes.",
    ),
    "//tests:test_model_profile_conformance": struct(
        data = ["//:files/prismaquant"],
        reason = "it scans every prismaquant module for readers of each serving spec field (_PACKAGE_ROOT.rglob).",
    ),
    "//tests:test_package_tools_boundary": struct(
        data = ["//:files/prismaquant"],
        reason = "it scans every prismaquant module for imports of tools.",
    ),
    "//tests:test_prismabuild_ban": struct(
        data = ["//:files/prismaquant", "//:files/tools", "//tests:files"],
        reason = "it scans the live source tree for retired execution-system mechanisms.",
    ),
    "//tests:test_serving_runtime_patch_set": struct(
        data = ["//:files/prismaquant/serving_runtime_patches"],
        reason = "it reads each patch set's MANIFEST.json through SERVING_RUNTIME_PATCH_SETS_DIR, a path imported from the module.",
    ),
    "//tests:test_tool_imports_1304": struct(
        data = ["//:files/scripts", "//:files/tools"],
        modules = ["scripts/", "tools/"],
        reason = "it parses every file under tools and scripts and imports each prismaquant and tools module they name.",
    ),
    "//tests:test_tessera_core_boundary": struct(
        data = ["//:files/prismaquant"],
        reason = "it scans every prismaquant module for Tessera lane references.",
    ),
    "//tests:test_tessera_export_lane": struct(
        data = ["//:prismaquant/run-pipeline.sh"],
        reason = "it reads run-pipeline.sh beside prismaquant.__file__, a path the analysis can't resolve.",
    ),
    "//tests:test_tessera_fleet_drivers": struct(
        modules = [
            "tools/render_identity.py",
            "tools/tessera_fleet/dispatch_ladder.py",
            "tools/tessera_fleet/dispatch_model.py",
            "tools/tessera_fleet/model_worker.py",
            "tools/tessera_fleet/status.py",
        ],
        reason = "it runs each driver's --help with python -m tools.<name> from a parametrized f-string.",
    ),
}
