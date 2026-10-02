"""Metadata-only namespace admission; no model, Docker or historical inputs."""
import copy
import json
import os
import subprocess
import sys

import pytest

import tools.tessera_campaign_namespace as namespace
from tools import dispatch_tessera_campaign as dispatch
from tools import tessera_campaign_container as adapter
from tools.pq_profile_digest import profile_digest_owner

owner = profile_digest_owner()
COMMIT = "b" * 40
PROVENANCE = {"tessera_commit": "c" * 40, "prismabuild_commit": "d" * 40,
              "container_content_sha256": "e" * 64}


def request():
    env = {"TMPDIR": "/old/tmp", "HF_HOME": "/old/hf", "PYTHONPATH": "/workspace"}
    spec = {"container": {"image": "sha256:" + "f" * 64,
                          "content_sha256": "e" * 64, "mounts": []}, "env": env}
    return {"argv": ["python3", "-m", "tools.tessera_campaign_container", "--spec",
                     json.dumps(spec), "--", "python", "-m", "prismaquant.tessera_campaign",
                     "--out", "/old/cost.pkl", "--cache-dir", "/old/cache",
                     "--checkpoint", "/old/cost.anchors.json", "--model", "/readonly/model",
                     "--units", "/readonly/selection.json"],
            "env": env, "cwd": "/reviewed/checkout", "data_manifest": "/readonly/readset.json",
            "demand": {"cpu": 2, "mem_gb": 8, "gpu": 1}}


def arguments(tmp_path) -> dict:
    row = request()
    spec = json.loads(row["argv"][4])
    spec["container"]["mounts"] = [{"source": str(tmp_path), "target": str(tmp_path), "readonly": False}]
    row["argv"][4] = json.dumps(spec)
    rows = [row]
    key = owner.canonical_json_sha256(rows[0], where="fixture request")
    evidence = {"path": "/readonly/reconciliation.json", "sha256": "1" * 64}
    reconciliation = {"schema": "prismaquant.tessera_namespace_reconciliation.v1",
                      "requests_sha256": owner.canonical_json_sha256(rows, where="fixture roster"),
                      "evidence": evidence, "status": {key: "unfinished"}}
    return {"requests": rows, "selected": [key], "reconciliation": reconciliation,
            "expected_evidence": evidence,
            "readsets": {key: {"path": rows[0]["data_manifest"], "sha256": "2" * 64}},
            "provenance": PROVENANCE, "expected_provenance": PROVENANCE,
            "reviewed_commit": "a" * 40, "executed_commit": COMMIT,
            "checkout": "/reviewed/checkout", "root": str(tmp_path / "new")}


def test_adapter_refuses_unbound_namespace_before_docker(monkeypatch):
    row = request()
    spec = json.loads(row["argv"][4])
    spec["namespace_binding"] = {"schema": "prismaquant.tessera_campaign_namespace.v1"}
    monkeypatch.setattr(adapter, "inspect_or_load", lambda *args: pytest.fail("Docker reached"))
    with pytest.raises(RuntimeError, match="namespace"):
        adapter.main(["--spec", json.dumps(spec), "--", *row["argv"][6:]])


def test_preparation_is_deterministic_and_preserves_legacy(tmp_path):
    kwargs = arguments(tmp_path)
    original = copy.deepcopy(kwargs["requests"])
    first = dispatch.prepare_namespace_requests(**kwargs)
    assert first == dispatch.prepare_namespace_requests(**kwargs)
    assert kwargs["requests"] == original
    assert not (tmp_path / "new").exists()
    row = first[0]
    inner = dispatch._inner_campaign_argv(row)
    for flag in ("--out", "--cache-dir", "--checkpoint"):
        assert inner[inner.index(flag) + 1].startswith(str(tmp_path / "new") + "/")
    spec = json.loads(row["argv"][4])
    assert spec["env"] == row["env"]
    assert spec["namespace_binding"]["executed_commit"] == COMMIT
    assert spec["namespace_binding"]["reviewed_commit"] == "a" * 40


def test_completed_request_cannot_be_selected(tmp_path):
    kwargs = arguments(tmp_path)
    kwargs["reconciliation"]["status"][kwargs["selected"][0]] = "completed"
    with pytest.raises(RuntimeError, match="unfinished"):
        dispatch.prepare_namespace_requests(**kwargs)


@pytest.mark.parametrize("change, message", [
    ("duplicate", "duplicate"), ("selected_duplicate", "duplicate"),
    ("missing_status", "reconciliation"), ("missing_reconciliation", "reconciliation"),
    ("bad_roster", "roster digest"),
    ("evidence", "evidence differs"), ("provenance", "provenance differs"),
    ("readset", "readset path"), ("short_commit", "full source"),
    ("traversal", "canonical"), ("env", "environment differs"),
    ("duplicate_out", "one explicit"), ("equals_out", "separate value"),
    ("scratch", "scratch overrides"),
])
def test_preparation_refusals(tmp_path, change, message):
    kwargs = arguments(tmp_path)
    if change == "duplicate":
        kwargs["requests"].append(copy.deepcopy(kwargs["requests"][0]))
    elif change == "selected_duplicate":
        kwargs["selected"] *= 2
    elif change == "missing_status":
        kwargs["reconciliation"]["status"] = {}
    elif change == "missing_reconciliation":
        kwargs["reconciliation"] = None
    elif change == "bad_roster":
        kwargs["reconciliation"]["requests_sha256"] = "0" * 64
    elif change == "evidence":
        kwargs["expected_evidence"] = {"path": "/different.json", "sha256": "3" * 64}
    elif change == "provenance":
        kwargs["expected_provenance"] = {**PROVENANCE, "tessera_commit": "0" * 40}
    elif change == "readset":
        kwargs["readsets"][kwargs["selected"][0]]["path"] = "/different.json"
    elif change == "short_commit":
        kwargs["executed_commit"] = "bff19452"
    elif change == "traversal":
        kwargs["root"] += "/../escape"
    else:
        row = kwargs["requests"][0]
        if change == "env":
            row["env"] = {**row["env"], "TMPDIR": "/different"}
        elif change == "duplicate_out":
            row["argv"] += ["--out", "/different"]
        elif change == "equals_out":
            index = row["argv"].index("--out")
            row["argv"][index:index + 2] = ["--out=/different"]
        elif change == "scratch":
            spec = json.loads(row["argv"][4])
            row["env"]["PRISMAQUANT_CONTAINER_CACHE_ROOT"] = "/different"
            spec["env"] = row["env"]
            row["argv"][4] = json.dumps(spec)
        key = owner.canonical_json_sha256(row, where="fixture request")
        old = kwargs["selected"][0]
        kwargs["selected"] = [key]
        kwargs["reconciliation"]["status"] = {key: "unfinished"}
        kwargs["reconciliation"]["requests_sha256"] = owner.canonical_json_sha256(kwargs["requests"], where="fixture roster")
        kwargs["readsets"] = {key: kwargs["readsets"][old]}
    with pytest.raises(RuntimeError, match=message):
        dispatch.prepare_namespace_requests(**kwargs)


def prepared(tmp_path):
    return dispatch.prepare_namespace_requests(**arguments(tmp_path))


def test_publication_identical_resume_preserves_outputs(tmp_path):
    rows = prepared(tmp_path)
    dispatch.publish_namespace_requests(rows)
    binding = namespace.validate_namespace_request(rows[0])
    root = tmp_path / "new"
    before = {path: path.read_bytes() for path in root.rglob("*.json")}
    output = root / binding["request_key"] / "cost.pkl"
    output.write_bytes(b"durable completed anchors stand in")
    dispatch.publish_namespace_requests(rows)
    namespace.require_namespace_publication(rows[0])
    assert {path: path.read_bytes() for path in before} == before
    assert output.read_bytes() == b"durable completed anchors stand in"


@pytest.mark.parametrize("occupied", ["empty", "foreign", "conflicting"])
def test_publication_no_overwrite(tmp_path, occupied):
    rows = prepared(tmp_path)
    root = tmp_path / "new"
    root.mkdir()
    if occupied != "empty":
        (root / "namespace.json").write_bytes(b"foreign bytes")
    if occupied == "foreign":
        (root / "legacy.pkl").write_bytes(b"immutable legacy")
    before = {path: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(RuntimeError, match="occupied|bytes differ"):
        dispatch.publish_namespace_requests(rows)
    assert {path: path.read_bytes() for path in root.iterdir()} == before


@pytest.mark.parametrize("field", ["out", "source", "binding", "request", "env"])
def test_binding_refuses_tampering(tmp_path, field):
    row = prepared(tmp_path)[0]
    spec = json.loads(row["argv"][4])
    if field == "out":
        row["argv"][row["argv"].index("--out") + 1] = str(tmp_path / "new-sibling" / "cost.pkl")
    elif field == "binding":
        spec["namespace_binding"]["request_key"] = "0" * 64
    elif field == "request":
        spec["namespace_binding"]["request_sha256"] = "0" * 64
    elif field == "env":
        row["env"]["TMPDIR"] = "/foreign"
    row["argv"][4] = json.dumps(spec)
    with pytest.raises(RuntimeError, match="namespace"):
        namespace.validate_namespace_request(row, executed_commit="0" * 40 if field == "source" else COMMIT)


@pytest.mark.parametrize("location", ["root", "row", "out", "environment"])
def test_symlink_escape_refusal(tmp_path, location):
    rows = prepared(tmp_path)
    binding = namespace.validate_namespace_request(rows[0])
    root = tmp_path / "new"
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    if location == "root":
        root.symlink_to(foreign, target_is_directory=True)
    else:
        directory = root / binding["request_key"]
        if location == "row":
            root.mkdir()
            directory.symlink_to(foreign, target_is_directory=True)
        else:
            directory.mkdir(parents=True)
            (directory / ("cost.pkl" if location == "out" else "environment")).symlink_to(foreign)
    with pytest.raises(RuntimeError, match="symlink"):
        namespace.validate_namespace_request(rows[0])
    assert list(foreign.iterdir()) == []


def test_adapter_positive_and_source_refusal_before_docker(tmp_path, monkeypatch):
    row = prepared(tmp_path)[0]
    dispatch.publish_namespace_requests([row])
    spec = json.loads(row["argv"][4])
    for name, value in row["env"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(adapter, "checkout_commit", lambda cwd: COMMIT)
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **k: type("Result", (), {"returncode": 0, "stdout": ""})())
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "prismaquant" / "__init__.py").write_text("# synthetic checkout package\n")
    assert adapter.validate_namespace_launch(spec, row["argv"][6:], cwd=str(tmp_path), environ=row["env"]) == COMMIT
    monkeypatch.setattr(adapter, "checkout_commit", lambda cwd: "0" * 40)
    monkeypatch.setattr(adapter, "inspect_or_load", lambda *args: pytest.fail("Docker reached"))
    with pytest.raises(RuntimeError, match="source commit differs"):
        adapter.main(["--spec", json.dumps(spec), "--", *row["argv"][6:]])


def test_adapter_refuses_dirty_full_source_and_outer_env(tmp_path, monkeypatch):
    row = prepared(tmp_path)[0]
    dispatch.publish_namespace_requests([row])
    spec = json.loads(row["argv"][4])
    with pytest.raises(RuntimeError, match="outer/spec environment"):
        adapter.validate_namespace_launch(spec, row["argv"][6:], cwd=str(tmp_path), environ={})
    monkeypatch.setattr(adapter, "checkout_commit", lambda cwd: COMMIT)
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **k: type("Result", (), {"returncode": 1})())
    with pytest.raises(RuntimeError, match="full executed source"):
        adapter.validate_namespace_launch(spec, row["argv"][6:], cwd=str(tmp_path), environ=row["env"])


def test_unpublished_binding_cannot_launch_and_legacy_is_optional(tmp_path):
    row = prepared(tmp_path)[0]
    spec = json.loads(row["argv"][4])
    with pytest.raises(RuntimeError, match="ownership is unavailable"):
        adapter.validate_namespace_launch(spec, row["argv"][6:], cwd=str(tmp_path), environ=row["env"])
    old = request()
    assert adapter.validate_namespace_launch(json.loads(old["argv"][4]), old["argv"][6:], cwd=str(tmp_path), environ={}) is None


def test_existing_unbuffered_command_shape(tmp_path):
    kwargs = arguments(tmp_path)
    row = kwargs["requests"][0]
    row["argv"].insert(7, "-u")
    key = owner.canonical_json_sha256(row, where="fixture request")
    old = kwargs["selected"][0]
    kwargs["selected"] = [key]
    kwargs["reconciliation"]["status"] = {key: "unfinished"}
    kwargs["reconciliation"]["requests_sha256"] = owner.canonical_json_sha256(kwargs["requests"], where="fixture roster")
    kwargs["readsets"] = {key: kwargs["readsets"][old]}
    assert dispatch.prepare_namespace_requests(**kwargs)[0]["argv"][7] == "-u"


def test_complete_roster_does_not_retarget_completed_requests(tmp_path):
    kwargs = arguments(tmp_path)
    completed = request()
    completed["argv"] += ["--seed-checkpoint", "/readonly/old.anchors.json",
                           "--seed-wire-dir", "/readonly/old-wire"]
    complete_key = owner.canonical_json_sha256(completed, where="fixture completed request")
    kwargs["requests"].append(completed)
    kwargs["reconciliation"]["status"][complete_key] = "completed"
    kwargs["reconciliation"]["requests_sha256"] = owner.canonical_json_sha256(kwargs["requests"], where="fixture roster")
    original = copy.deepcopy(kwargs["requests"])
    rows = dispatch.prepare_namespace_requests(**kwargs)
    assert len(rows) == 1
    assert namespace.validate_namespace_request(rows[0])["original_request_sha256"] == kwargs["selected"][0]
    assert kwargs["requests"] == original


def test_malformed_embedded_spec_refuses_with_namespace_error():
    row = request()
    row["argv"][4] = "not JSON"
    with pytest.raises(RuntimeError, match="namespace container spec is not valid JSON"):
        namespace.namespace_request_parts(row)


def test_source_inspection_failure_is_explicit_refusal(tmp_path, monkeypatch):
    row = prepared(tmp_path)[0]
    dispatch.publish_namespace_requests([row])
    spec = json.loads(row["argv"][4])
    monkeypatch.setattr(adapter, "checkout_commit", lambda cwd: COMMIT)

    def unavailable(*args, **kwargs):
        raise adapter.subprocess.TimeoutExpired("git", 60)

    monkeypatch.setattr(adapter.subprocess, "run", unavailable)
    with pytest.raises(RuntimeError, match="namespace cannot inspect executed source"):
        adapter.validate_namespace_launch(spec, row["argv"][6:], cwd=str(tmp_path), environ=row["env"])


def test_untracked_executed_source_is_not_the_bound_commit(tmp_path, monkeypatch):
    row = prepared(tmp_path)[0]
    dispatch.publish_namespace_requests([row])
    spec = json.loads(row["argv"][4])
    monkeypatch.setattr(adapter, "checkout_commit", lambda cwd: COMMIT)
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **k: type("Result", (), {
        "returncode": 0, "stdout": "?? tools/foreign_module.py\n"})())
    with pytest.raises(RuntimeError, match="namespace full executed source"):
        adapter.validate_namespace_launch(spec, row["argv"][6:], cwd=str(tmp_path), environ=row["env"])


def replace_fixture_request(kwargs, row):
    old = kwargs["selected"][0]
    key = owner.canonical_json_sha256(row, where="fixture request")
    kwargs["requests"] = [row]
    kwargs["selected"] = [key]
    kwargs["reconciliation"]["status"] = {key: "unfinished"}
    kwargs["reconciliation"]["requests_sha256"] = owner.canonical_json_sha256([row], where="fixture roster")
    kwargs["readsets"] = {key: kwargs["readsets"][old]}


@pytest.mark.parametrize("pythonpath", [None, "/image-only", "/image-only:/workspace"])
def test_review_source_requires_known_guarded_checkout(tmp_path, monkeypatch, pythonpath):
    kwargs = arguments(tmp_path)
    row = kwargs["requests"][0]
    spec = json.loads(row["argv"][4])
    if pythonpath is None:
        row["env"].pop("PYTHONPATH")
    else:
        row["env"]["PYTHONPATH"] = pythonpath
    spec["env"] = row["env"]
    row["argv"][4] = json.dumps(spec)
    replace_fixture_request(kwargs, row)
    row = dispatch.prepare_namespace_requests(**kwargs)[0]
    dispatch.publish_namespace_requests([row])
    spec = json.loads(row["argv"][4])
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "prismaquant" / "__init__.py").write_text("# genuine fixture checkout\n")
    monkeypatch.setattr(adapter, "checkout_commit", lambda cwd: COMMIT)
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **k: type("Result", (), {"returncode": 0, "stdout": ""})())
    with pytest.raises(RuntimeError, match="guarded|checkout|import"):
        adapter.validate_namespace_launch(spec, row["argv"][6:], cwd=str(tmp_path), environ=row["env"])
    for name, value in row["env"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(adapter, "inspect_or_load", lambda *a: pytest.fail("Docker reached"))
    with pytest.raises(RuntimeError, match="guarded|checkout|import"):
        adapter.main(["--spec", json.dumps(spec), "--", *row["argv"][6:]])


@pytest.mark.parametrize("mount_case", ["absent", "readonly", "remapped", "hidden"])
def test_review_mount_coverage_refusal(tmp_path, mount_case):
    kwargs = arguments(tmp_path)
    row = kwargs["requests"][0]
    spec = json.loads(row["argv"][4])
    if mount_case == "absent":
        spec["container"]["mounts"] = []
    elif mount_case == "readonly":
        spec["container"]["mounts"][0]["readonly"] = True
    elif mount_case == "remapped":
        spec["container"]["mounts"][0]["source"] = "/foreign"
    else:
        spec["container"]["mounts"].append({"source": "/foreign", "target": kwargs["root"], "readonly": False})
    row["argv"][4] = json.dumps(spec)
    replace_fixture_request(kwargs, row)
    with pytest.raises(RuntimeError, match="mount|hidden"):
        dispatch.prepare_namespace_requests(**kwargs)
    assert not (tmp_path / "new").exists()


def test_review_fresh_and_identical_publication_establishes_writable_temps(tmp_path):
    rows = prepared(tmp_path)
    assert not (tmp_path / "new").exists()
    for attempt in range(2):
        dispatch.publish_namespace_requests(rows)
        namespace.require_namespace_publication(rows[0])
        for name in ("TMPDIR", "TMP", "TEMP"):
            directory = adapter.Path(rows[0]["env"][name])
            assert directory.is_dir(), name
            probe = directory / ("owned-probe-" + str(attempt))
            probe.write_bytes(b"writable")
            assert probe.read_bytes() == b"writable"
            probe.unlink()
        observed = subprocess.run([sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"],
                                  env={**os.environ, **rows[0]["env"]}, check=True,
                                  capture_output=True, text=True, timeout=30)
        assert observed.stdout.strip() == rows[0]["env"]["TMPDIR"]


@pytest.mark.parametrize("value", ["traversal", "relative", "double_slash", "leading_double_slash", "contained"])
def test_review_path_input_is_canonical_before_overlap(tmp_path, value):
    kwargs = arguments(tmp_path)
    row = kwargs["requests"][0]
    if value == "traversal":
        path = str(tmp_path / "existing") + "/../new"
    elif value == "relative":
        path = "../new"
    elif value == "double_slash":
        path = str(tmp_path) + "//outside"
    elif value == "leading_double_slash":
        path = "/" + str(tmp_path / "new" / "selection.json")
    else:
        path = str(tmp_path / "new" / "model")
    row["argv"][row["argv"].index("--units") + 1] = path
    replace_fixture_request(kwargs, row)
    with pytest.raises(RuntimeError, match="canonical|overlap"):
        dispatch.prepare_namespace_requests(**kwargs)
    assert not (tmp_path / "new").exists()


def test_review_guarded_checkout_positive_and_alternate_root_refusal(tmp_path):
    checkout = tmp_path / "checkout"
    alternate = tmp_path / "alternate"
    for root in (checkout, alternate):
        (root / "prismaquant").mkdir(parents=True)
        (root / "prismaquant" / "__init__.py").write_text("# synthetic package\n")
    spec = {"env": {"PYTHONPATH": "/workspace"}, "container": {"mounts": []}}
    assert adapter.guarded_import_root(spec, cwd=str(checkout), require_checkout=True) == ("/workspace", checkout)
    spec["container"]["mounts"] = [{"source": str(alternate), "target": "/alternate", "readonly": True}]
    spec["env"]["PYTHONPATH"] = "/alternate:/workspace"
    with pytest.raises(RuntimeError, match="guarded imports"):
        adapter.guarded_import_root(spec, cwd=str(checkout), require_checkout=True)


def import_shadow_launch_fixture(tmp_path, monkeypatch, shadow):
    """Real committed checkout and mounted foreign file; no selection mocks."""
    checkout = tmp_path / "import-checkout"
    alternate = tmp_path / "import-alternate"
    (checkout / "prismaquant").mkdir(parents=True)
    (checkout / "prismaquant" / "__init__.py").write_text("# checkout package\n")
    alternate.mkdir()
    foreign = alternate / shadow
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_text("raise AssertionError('foreign source must never execute')\n")
    for argv in (["git", "init", "--quiet", str(checkout)],
                 ["git", "-C", str(checkout), "add", "prismaquant"],
                 ["git", "-C", str(checkout), "-c", "user.name=Fixture",
                  "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "fixture"]):
        subprocess.run(argv, check=True, capture_output=True, timeout=30)
    commit = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    kwargs = arguments(tmp_path)
    kwargs.update(checkout=str(checkout), executed_commit=commit)
    row = kwargs["requests"][0]
    row["cwd"] = str(checkout)
    spec = json.loads(row["argv"][4])
    spec["container"]["mounts"].append({"source": str(alternate), "target": "/alternate", "readonly": True})
    row["env"]["PYTHONPATH"] = "/alternate:/workspace"
    spec["env"] = row["env"]
    row["argv"][4] = json.dumps(spec)
    replace_fixture_request(kwargs, row)
    row = dispatch.prepare_namespace_requests(**kwargs)[0]
    dispatch.publish_namespace_requests([row])
    spec = json.loads(row["argv"][4])
    monkeypatch.chdir(checkout)
    for name, value in row["env"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(adapter, "inspect_or_load", lambda *a: pytest.fail("Docker reached past mounted import shadow"))
    return row, spec, foreign


def test_import_shadow_source_module_refused_before_docker(tmp_path, monkeypatch):
    row, spec, foreign = import_shadow_launch_fixture(tmp_path, monkeypatch, "prismaquant.py")
    with pytest.raises(RuntimeError, match="namespace guarded imports.*shadow"):
        adapter.main(["--spec", json.dumps(spec), "--", *row["argv"][6:]])
    assert foreign.read_text() == "raise AssertionError('foreign source must never execute')\n"


@pytest.mark.parametrize("shadow", [
    "prismaquant.pyc", "prismaquant.so", "prismaquant.container-unknown.so",
    "prismaquant/__init__.pyc", "prismaquant/__init__.so",
    "prismaquant/__init__.container-unknown.so",
])
def test_import_shadow_bytecode_native_refused_before_docker(tmp_path, monkeypatch, shadow):
    row, spec, foreign = import_shadow_launch_fixture(tmp_path, monkeypatch, shadow)
    with pytest.raises(RuntimeError, match="namespace guarded imports.*shadow|namespace guarded imports.*native initializer") as error:
        adapter.main(["--spec", json.dumps(spec), "--", *row["argv"][6:]])
    assert str(foreign) in str(error.value)


@pytest.mark.parametrize("shadow", ["prismaquant.py", "prismaquant.pyc"])
def test_import_shadow_same_entry_source_package_precedence(tmp_path, shadow):
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "prismaquant" / "__init__.py").write_text("# supported package\n")
    (tmp_path / shadow).write_bytes(b"not executed")
    spec = {"env": {"PYTHONPATH": "/workspace"}, "container": {"mounts": []}}
    assert adapter.guarded_import_root(spec, cwd=str(tmp_path), require_checkout=True) == ("/workspace", tmp_path)


@pytest.mark.parametrize("initializer", ["__init__.so", "__init__.container-unknown.so"])
def test_import_shadow_native_initializer_supersedes_source_ambiguity(tmp_path, initializer):
    package = tmp_path / "prismaquant"
    package.mkdir()
    (package / "__init__.py").write_text("# supported package\n")
    (package / initializer).write_bytes(b"unknown ABI, must refuse conservatively")
    spec = {"env": {"PYTHONPATH": "/workspace"}, "container": {"mounts": []}}
    with pytest.raises(RuntimeError, match="ambiguous native initializer"):
        adapter.guarded_import_root(spec, cwd=str(tmp_path), require_checkout=True)


@pytest.mark.parametrize("prefix", ["/image-only", "$DYNAMIC", "relative-root"])
def test_import_shadow_unknown_dynamic_prefix_failclosed(tmp_path, prefix):
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "prismaquant" / "__init__.py").write_text("# package\n")
    spec = {"env": {"PYTHONPATH": prefix + ":/workspace"}, "container": {"mounts": []}}
    with pytest.raises(RuntimeError, match="unknown earlier root"):
        adapter.guarded_import_root(spec, cwd=str(tmp_path), require_checkout=True)


@pytest.mark.parametrize("layout", ["namespace", "empty", "later-module", "pycache", "unrelated-native"])
def test_import_shadow_supported_search_controls(tmp_path, layout):
    checkout, alternate = tmp_path / "checkout", tmp_path / "alternate"
    (checkout / "prismaquant").mkdir(parents=True)
    (checkout / "prismaquant" / "__init__.py").write_text("# package\n")
    alternate.mkdir()
    if layout == "namespace":
        (alternate / "prismaquant").mkdir()
    elif layout == "later-module":
        (alternate / "prismaquant.py").write_text("raise AssertionError('not executed')\n")
    elif layout == "pycache":
        (alternate / "__pycache__").mkdir()
        (alternate / "__pycache__" / "prismaquant.cpython-314.pyc").write_bytes(b"not a top-level candidate")
    elif layout == "unrelated-native":
        (alternate / "prismaquant_helpers.so").write_bytes(b"unrelated")
    spec = {"env": {"PYTHONPATH": "/workspace:/alternate" if layout == "later-module" else "/alternate:/workspace"},
            "container": {"mounts": [{"source": str(alternate), "target": "/alternate", "readonly": True}]}}
    assert adapter.guarded_import_root(spec, cwd=str(checkout), require_checkout=True) == ("/workspace", checkout)


def test_import_shadow_inspection_error_failclosed(tmp_path, monkeypatch):
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "prismaquant" / "__init__.py").write_text("# package\n")
    actual_iterdir = adapter.Path.iterdir

    def unreadable(path):
        if path == tmp_path / "prismaquant":
            raise PermissionError("fixture inaccessible mounted directory")
        return actual_iterdir(path)

    monkeypatch.setattr(adapter.Path, "iterdir", unreadable)
    spec = {"env": {"PYTHONPATH": "/workspace"}, "container": {"mounts": []}}
    with pytest.raises(RuntimeError, match="cannot inspect root"):
        adapter.guarded_import_root(spec, cwd=str(tmp_path), require_checkout=True)


def test_import_shadow_legacy_non_opt_in_selection_unchanged(tmp_path):
    checkout, alternate = tmp_path / "checkout", tmp_path / "alternate"
    (checkout / "prismaquant").mkdir(parents=True)
    (checkout / "prismaquant" / "__init__.py").write_text("# package\n")
    alternate.mkdir()
    (alternate / "prismaquant.py").write_text("raise AssertionError('not executed')\n")
    spec = {"env": {"PYTHONPATH": "/alternate:/workspace"},
            "container": {"mounts": [{"source": str(alternate), "target": "/alternate", "readonly": True}]}}
    assert adapter.guarded_import_root(spec, cwd=str(checkout)) == ("/workspace", checkout)
    assert adapter.validate_namespace_launch(spec, [], cwd=str(checkout), environ={}) is None


def test_review_temporary_symlink_race_cannot_escape_or_replace_bindings(tmp_path, monkeypatch):
    row = prepared(tmp_path)[0]
    dispatch.publish_namespace_requests([row])
    binding = namespace.require_namespace_publication(row)
    row_root = tmp_path / "new" / binding["request_key"]
    before = {name: (row_root / name).read_bytes() for name in ("binding.json", "request.json")}
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    destination = adapter.Path(row["env"]["TMPDIR"])
    actual_open = os.open

    def race(name, flags, *args, **kwargs):
        if name == "TMPDIR":
            destination.rmdir()
            destination.symlink_to(foreign, target_is_directory=True)
        return actual_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(namespace.os, "open", race)
    with pytest.raises(RuntimeError, match="cannot establish"):
        namespace.establish_namespace_temporaries(row)
    assert list(foreign.iterdir()) == []
    assert {name: (row_root / name).read_bytes() for name in before} == before


def test_review_temporary_writability_refuses_without_metadata_overwrite(tmp_path, monkeypatch):
    row = prepared(tmp_path)[0]
    dispatch.publish_namespace_requests([row])
    binding = namespace.require_namespace_publication(row)
    path = tmp_path / "new" / binding["request_key"] / "binding.json"
    before = path.read_bytes()
    monkeypatch.setattr(namespace.os, "access", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="not writable"):
        namespace.establish_namespace_temporaries(row)
    assert path.read_bytes() == before


def test_review_adapter_repairs_missing_owned_temps_before_entry(tmp_path, monkeypatch):
    row = prepared(tmp_path)[0]
    dispatch.publish_namespace_requests([row])
    spec = json.loads(row["argv"][4])
    directory = adapter.Path(row["env"]["TMPDIR"])
    directory.rmdir()
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "prismaquant" / "__init__.py").write_text("# fixture actual checkout\n")
    monkeypatch.setattr(adapter, "checkout_commit", lambda cwd: COMMIT)
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **k: type("Result", (), {"returncode": 0, "stdout": ""})())
    assert adapter.validate_namespace_launch(spec, row["argv"][6:], cwd=str(tmp_path), environ=row["env"]) == COMMIT
    assert directory.is_dir()


def mount_component_aliases(mounts, root, violation):
    result = copy.deepcopy(mounts)
    if violation == "source":
        result.append({"source": "//foreign", "target": "/external", "readonly": True})
    elif violation == "target":
        result.append({"source": "/foreign", "target": "//external", "readonly": True})
    else:
        result.append({"source": "/foreign", "target": "/" + root, "readonly": False})
    return result


@pytest.mark.parametrize("violation", ["source", "target", "shadow"])
def test_mount_component_preparation_refuses_leading_double_slash(tmp_path, violation):
    kwargs = arguments(tmp_path)
    row = kwargs["requests"][0]
    spec = json.loads(row["argv"][4])
    spec["container"]["mounts"] = mount_component_aliases(
        spec["container"]["mounts"], kwargs["root"], violation)
    row["argv"][4] = json.dumps(spec)
    replace_fixture_request(kwargs, row)
    with pytest.raises(RuntimeError, match="namespace paths must be canonical"):
        dispatch.prepare_namespace_requests(**kwargs)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("violation", ["source", "target", "shadow"])
def test_mount_component_adapter_refuses_self_consistent_alias_before_docker(tmp_path, monkeypatch, violation):
    row = prepared(tmp_path)[0]
    binding = json.loads(row["argv"][4])["namespace_binding"]
    normalized = binding["normalized_request"]
    spec = json.loads(normalized["argv"][4])
    spec["container"]["mounts"] = mount_component_aliases(
        spec["container"]["mounts"], binding["root"], violation)
    normalized["argv"][4] = owner.canonical_json_bytes(spec, where="fixture spec").decode()
    base = {key: value for key, value in binding.items() if key not in ("request_key", "request", "request_sha256")}
    request_key = owner.canonical_json_sha256(base, where="namespace row binding")
    request = namespace.namespace_retarget(normalized, binding["root"] + "/" + request_key)
    binding = {**base, "request_key": request_key, "request": copy.deepcopy(request),
               "request_sha256": owner.canonical_json_sha256(request, where="namespace request")}
    spec = json.loads(request["argv"][4])
    spec["namespace_binding"] = binding
    row = copy.deepcopy(request)
    row["argv"][4] = owner.canonical_json_bytes(spec, where="fixture bound spec").decode()
    # Model an already published, self-consistent old contract without asking the
    # current preparation/publication policy to admit it first.
    root = tmp_path / "new"
    directory = root / request_key
    directory.mkdir(parents=True)
    record = {"schema": namespace.SCHEMA, "root": str(root), "bindings": {
        request_key: owner.canonical_json_sha256(binding, where="namespace ownership")}}
    for path, value in ((root / "namespace.json", record),
                        (directory / "binding.json", binding), (directory / "request.json", row)):
        path.write_bytes(owner.canonical_json_bytes(value, where="namespace publication"))
    before = {path: path.read_bytes() for path in root.rglob("*.json")}
    for name, value in row["env"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(adapter, "checkout_commit", lambda cwd: COMMIT)
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **k: type("Result", (), {"returncode": 0, "stdout": ""})())
    monkeypatch.setattr(adapter, "inspect_or_load", lambda *a: pytest.fail("Docker reached"))
    with pytest.raises(RuntimeError, match="namespace paths must be canonical"):
        adapter.main(["--spec", json.dumps(spec), "--", *row["argv"][6:]])
    assert {path: path.read_bytes() for path in before} == before
    assert not (directory / "environment").exists()


def test_mount_component_valid_identity_mount_covers_outputs_and_owned_temps(tmp_path, monkeypatch):
    kwargs = arguments(tmp_path)
    rows = dispatch.prepare_namespace_requests(**kwargs)
    assert not (tmp_path / "new").exists()
    dispatch.publish_namespace_requests(rows)
    row = rows[0]
    spec = json.loads(row["argv"][4])
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "prismaquant" / "__init__.py").write_text("# mount component checkout\n")
    monkeypatch.setattr(adapter, "checkout_commit", lambda cwd: COMMIT)
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **k: type("Result", (), {"returncode": 0, "stdout": ""})())
    assert adapter.validate_namespace_launch(spec, row["argv"][6:], cwd=str(tmp_path), environ=row["env"]) == COMMIT
    for name in ("TMPDIR", "TMP", "TEMP"):
        path = adapter.Path(row["env"][name])
        assert path.is_relative_to(tmp_path / "new") and path.is_dir()


@pytest.mark.parametrize("violation", ["mount", "input"])
def test_review_adapter_rechecks_self_consistent_old_contract_gaps(tmp_path, violation):
    row = prepared(tmp_path)[0]
    binding = json.loads(row["argv"][4])["namespace_binding"]
    normalized = binding["normalized_request"]
    if violation == "mount":
        spec = json.loads(normalized["argv"][4])
        spec["container"]["mounts"] = []
        normalized["argv"][4] = owner.canonical_json_bytes(spec, where="fixture spec").decode()
    else:
        normalized["argv"][normalized["argv"].index("--units") + 1] = str(tmp_path) + "/existing/../new"
    base = {key: value for key, value in binding.items() if key not in ("request_key", "request", "request_sha256")}
    request_key = owner.canonical_json_sha256(base, where="namespace row binding")
    request = namespace.namespace_retarget(normalized, binding["root"] + "/" + request_key)
    binding = {**base, "request_key": request_key, "request": copy.deepcopy(request),
               "request_sha256": owner.canonical_json_sha256(request, where="namespace request")}
    spec = json.loads(request["argv"][4])
    spec["namespace_binding"] = binding
    with pytest.raises(RuntimeError, match="mount|canonical"):
        namespace.namespace_adapter_request(spec, request["argv"][6:], request["env"])
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("location", ["root", "row"])
def test_publication_parent_symlink_race_cannot_write_foreign_metadata(tmp_path, monkeypatch, location):
    from prismaquant import cost_stage_checkpoint as checkpoint

    rows = prepared(tmp_path)
    binding = namespace.validate_namespace_request(rows[0])
    root = tmp_path / "new"
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    swapped = False
    publish = checkpoint.publish_new_bytes

    def swap_parent_after_preflight(path, payload, **kwargs):
        nonlocal swapped
        parent = root if location == "root" else root / binding["request_key"]
        if path.parent == parent and not swapped:
            assert not parent.exists()
            parent.symlink_to(foreign, target_is_directory=True)
            swapped = True
        return publish(path, payload, **kwargs)

    monkeypatch.setattr(checkpoint, "publish_new_bytes", swap_parent_after_preflight)
    with pytest.raises((RuntimeError, OSError)):
        dispatch.publish_namespace_requests(rows)
    assert swapped, "the publication boundary was exercised"
    assert list(foreign.iterdir()) == [], "publication wrote through the raced parent symlink"


def test_publication_concurrent_identical_callers_preserve_no_clobber(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from prismaquant import cost_stage_checkpoint as checkpoint

    rows = prepared(tmp_path)
    first_publication = Barrier(2)
    publish = checkpoint.publish_new_bytes

    def both_preflights_complete(path, payload, **kwargs):
        if path.name == "namespace.json":
            first_publication.wait(timeout=10)
        return publish(path, payload, **kwargs)

    monkeypatch.setattr(checkpoint, "publish_new_bytes", both_preflights_complete)
    with ThreadPoolExecutor(max_workers=2) as callers:
        futures = [callers.submit(dispatch.publish_namespace_requests, rows) for _ in range(2)]
        for future in futures:
            future.result(timeout=20)
    namespace.require_namespace_publication(rows[0])
    assert not list((tmp_path / "new").rglob("*.tmp*"))
