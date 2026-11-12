"""Macros the generated BUILD files use: one library per module, one test per file.

tools/bazel/build_files.py writes the targets. These macros add what every
module or test shares, so the generated files hold only per-module evidence.
"""

load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("//tools/bazel:nonhermetic.bzl", "nonhermetic")
load("//tools/bazel:test_inputs.bzl", "TEST_INPUTS")

# What every hosted gate shard has: pytest.ini (testpaths, pythonpath, markers),
# the per-test budgets, and the pytest plugins the gate's command line uses.
# The conftest chain and each module's own pip packages come from the generator.
_TEST_DATA = [
    "//:pytest.ini",
    "//:tools/ci/gate_budget.json",
]
_TEST_DEPS = [
    "//tools/bazel:runner",
    "//:tools/ci/gate_run",
    "@pq_pip//pytest",
    "@pq_pip//pytest_timeout",
]

def _unique(labels):
    """A generated label and a shared one may name the same target once each."""
    unique = {}
    for label in labels:
        unique[str(native.package_relative_label(label))] = label
    return list(unique.values())

def pq_py_library(name, srcs, deps = [], data = [], **kwargs):
    py_library(name = name, srcs = srcs, deps = deps, data = data, **kwargs)

def pq_py_test(name, path, library, args = [], tags = [], deps = [], data = [], **kwargs):
    """Run the test file at repository path `path`, whose module is `library`.

    deps and data are the test's own inputs, which never propagate: the conftests
    above it, modules it names in strings, programs its closure launches, the
    tree reads of production modules it imports itself
    (tools/bazel/build_files.py), and test_inputs.bzl's entry.
    """
    extra = TEST_INPUTS.get("//%s:%s" % (native.package_name(), name))
    extra_tags, exec_properties, env = nonhermetic(name)
    # A hosted shard gives each xdist worker about one core and 4 GB; torch's
    # import alone needs most of a small executor's default.
    exec_properties = dict({"EstimatedCPU": "2", "EstimatedMemory": "4GB"}, **exec_properties)
    py_test(
        name = name,
        main = "//tools/bazel:python_runner.py",
        srcs = ["//tools/bazel:python_runner.py"],
        args = [path] + args,
        deps = _unique([library] + deps + _TEST_DEPS + (extra.deps if extra and hasattr(extra, "deps") else [])),
        data = _unique(_TEST_DATA + data + (extra.data if extra and hasattr(extra, "data") else [])),
        tags = tags + extra_tags,
        exec_properties = exec_properties,
        env = env,
        **kwargs
    )
