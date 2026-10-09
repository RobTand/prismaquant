"""Row classes: placement is a property of the class, and it is attested.

Three properties, in the order they matter.

*   **The default class did not move.** ``submit`` re-run is the resume and a
    finished row is a CAS hit on its action key, so a row dict that gains a
    key, loses one or changes a value re-runs work that is already done. The
    first test pins the whole row dict for a spec with no ``classes`` block and
    for a spec whose ``classes`` block declares the default explicitly.
*   **Recorded host attestation follows the existing dev-mode policy.** A
    missing or retired interpreter stamps and continues by default; explicit
    certified mode refuses. Malformed inventory, unknown tags, mixed instruction
    sets and incompatible container device runtimes still refuse in either mode.
*   **A weights-only class cannot be handed Hessian-aware work.** That is the
    gfx1201/sm121 wire divergence (RobTand/tessera#472) expressed as a gate
    rather than a comment.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import dispatch_tessera_campaign as dispatch  # noqa: E402
import tessera_campaign_container as container_tool  # noqa: E402


ROCM_PYTHON = "/home/rob/ml-venvs/torch-rocm7/bin/python"
GB10_PYTHON = "/home/rob/gb10-venvs/example/bin/python"
SDK3_PYTHON = "/home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python"
SDK4_PYTHON = "/home/rob/venvs/pq-pbdc4803-tessera-b40c93cb/bin/python"
SDK5_PYTHON = "/home/rob/venvs/pq-task-suite-layer-sdk5-20261009/bin/python"
SDK5_BASELINE = "/home/rob/venvs/pq-pin-fca4c6ce0-pb027103d9"
SDK4_ATTESTED_BY = "204b35146fab0c7c135781d594c4c15890b83fe74c9ec3d7463f802863700209"
SDK4_PB_COMMIT = "dc4803daaf09b6426083d2d36bd2a2da3d6832fe"
SDK5_PB_COMMIT = "027103d9a8417e06c7f13356e58779a313cd7088"
TESSERA_COMMIT = "fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb"


def fleet() -> dict:
    """The tracked attestation table, plus one attested GB10 host interpreter.

    The tracked table attests the fleet's canonical SDK4 interpreter on
    ``gb10``; this fixture needs the spec's own example path instead, so a test
    that wants a non-containerised default class injects the entry it needs
    rather than borrowing the tracked one.
    """
    table = json.loads(json.dumps(dispatch.load_fleet_interpreters()))
    table["tags"]["gb10"]["interpreters"][GB10_PYTHON] = {
        "attested_by": "0" * 64, "observed": "fixture"}
    return table


def base_spec(**extra) -> dict:
    spec = {
        "model": "/models/example",
        "campaign_argv": ["--families", "tessera16"],
        "cwd": "/home/rob/example",
        "python": GB10_PYTHON,
        "env": {"OMP_NUM_THREADS": "1"},
    }
    spec.update(extra)
    return spec


def build(spec, **kwargs) -> dict:
    return dispatch._row(spec, ["--model", "/models/example"], mem_gb=34,
                         timeout_s=None, **kwargs)


# --------------------------------------------------------------------------
# The default class is today's row, byte for byte
# --------------------------------------------------------------------------

def test_default_class_row_is_unchanged_for_a_spec_with_no_classes():
    row = build(base_spec())
    assert row == {
        "argv": [GB10_PYTHON, "-u", "-m", "prismaquant.tessera_campaign",
                 "--model", "/models/example"],
        "cwd": "/home/rob/example",
        "demand": {"gpu": 1, "cpu": 4, "mem_gb": 34},
        "env": {"OMP_NUM_THREADS": "1"},
        "tags": ["gb10"],
        "retry_safe": True,
        "progress_phases": ["startup=3600", "pricing=900", "finalize=1800"],
    }


def test_declaring_classes_does_not_move_the_default_row():
    plain = base_spec(tags=["gb10"], cpus=6)
    classed = base_spec(tags=["gb10"], cpus=6, classes={
        "default": {"_why": "the campaign's own rows"},
        "rocm-encode": {"python": ROCM_PYTHON, "tags": ["gfx1201"],
                        "wire_shared": False, "weights_only": True},
    })
    assert build(classed) == build(plain)
    assert json.dumps(build(classed), sort_keys=True) == json.dumps(
        build(plain), sort_keys=True)


def test_a_container_spec_builds_the_same_argv_as_before():
    spec = base_spec(python="python3", container={
        "image": "prismaquant-example:pinned",
        "mounts": [{"source": "/mnt/shared", "target": "/mnt/shared"}]})
    row = build(spec)
    assert row["argv"][:4] == ["python3", "-m",
                               "tools.tessera_campaign_container", "--spec"]
    embedded = json.loads(row["argv"][4])
    assert embedded == {"container": spec["container"],
                        "env": {"OMP_NUM_THREADS": "1"}}


# --------------------------------------------------------------------------
# A class carries its own interpreter, environment, tags and CPU count
# --------------------------------------------------------------------------

def test_a_class_overrides_placement_and_merges_its_environment():
    spec = base_spec(classes={
        "rocm-encode": {"python": ROCM_PYTHON, "tags": ["gfx1201"], "cpus": 4,
                        "env": {"TESSERA_EXAMPLE": "1"},
                        "wire_shared": False, "weights_only": True}})
    row = build(spec, module="prismaquant.example_encode",
                row_class_name="rocm-encode")
    assert row["argv"][0] == ROCM_PYTHON
    assert row["tags"] == ["gfx1201"]
    assert row["env"] == {"OMP_NUM_THREADS": "1", "TESSERA_EXAMPLE": "1"}
    assert row["demand"] == {"gpu": 1, "cpu": 4, "mem_gb": 34}


def test_an_undeclared_class_is_refused():
    try:
        build(base_spec(), row_class_name="rocm-encode")
    except dispatch.RowClassRefused as error:
        assert "declares no row class" in str(error)
    else:
        raise AssertionError("an undeclared class built a row")


def test_a_class_may_not_declare_a_field_it_does_not_own():
    spec = base_spec(classes={"rocm-encode": {"model": "/models/other"}})
    try:
        dispatch.validate_row_classes(spec, fleet=fleet())
    except dispatch.RowClassRefused as error:
        assert "does not own" in str(error)
    else:
        raise AssertionError("a class redefined the campaign's model")


# --------------------------------------------------------------------------
# Placement is attested, not asserted
# --------------------------------------------------------------------------

def test_an_unattested_interpreter_on_a_tag_is_refused():
    spec = base_spec(classes={
        "rocm-encode": {"python": "/home/rob/no/such/python",
                        "tags": ["gfx1201"],
                        "wire_shared": False, "weights_only": True}})
    try:
        dispatch.validate_row_classes(spec, fleet=fleet())
    except dispatch.RowClassRefused as error:
        assert "not attested on tag 'gfx1201'" in str(error)
    else:
        raise AssertionError("an unattested interpreter was accepted")


def test_the_rocm_interpreter_is_refused_on_a_gb10_tag():
    """The interpreter exists -- on the other box. That is the whole failure."""
    spec = base_spec(classes={
        "rocm-encode": {"python": ROCM_PYTHON, "tags": ["gb10"],
                        "wire_shared": False, "weights_only": True}})
    try:
        dispatch.validate_row_classes(spec, fleet=fleet())
    except dispatch.RowClassRefused as error:
        assert "not attested on tag 'gb10'" in str(error)
    else:
        raise AssertionError("a ROCm interpreter was accepted on a GB10 tag")


def test_a_tag_the_fleet_does_not_attest_is_refused():
    spec = base_spec(classes={
        "rocm-encode": {"python": ROCM_PYTHON, "tags": ["some-new-box"],
                        "wire_shared": False, "weights_only": True}})
    try:
        dispatch.validate_row_classes(spec, fleet=fleet())
    except dispatch.RowClassRefused as error:
        assert "does not attest" in str(error)
    else:
        raise AssertionError("an unknown tag placed a row")


def test_a_class_spanning_two_instruction_sets_is_refused(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    spec = base_spec(classes={
        "rocm-encode": {"python": ROCM_PYTHON, "tags": ["gfx1201", "gb10"],
                        "wire_shared": False, "weights_only": True}})
    try:
        dispatch.validate_row_classes(spec, fleet=fleet())
    except dispatch.RowClassRefused as error:
        assert "instruction sets" in str(error)
    else:
        raise AssertionError("one class was placed on two ISAs")


def test_a_wire_sharing_class_may_not_leave_the_campaign_isa():
    spec = base_spec(classes={
        "rocm-encode": {"python": ROCM_PYTHON, "tags": ["gfx1201"]}})
    try:
        dispatch.validate_row_classes(spec, fleet=fleet())
    except dispatch.RowClassRefused as error:
        assert "not bit-comparable" in str(error)
    else:
        raise AssertionError("a wire-sharing class left the campaign's ISA")


def test_a_class_that_does_not_share_the_wire_must_be_weights_only():
    spec = base_spec(classes={
        "rocm-encode": {"python": ROCM_PYTHON, "tags": ["gfx1201"],
                        "wire_shared": False}})
    try:
        dispatch.validate_row_classes(spec, fleet=fleet())
    except dispatch.RowClassRefused as error:
        assert "weights_only" in str(error)
    else:
        raise AssertionError("a non-weights-only class was placed off-ISA")


def test_the_rocm_encode_class_this_issue_exists_for_is_accepted():
    spec = base_spec(classes={
        "default": {"_why": "the campaign's own rows"},
        "rocm-encode": {"python": ROCM_PYTHON, "tags": ["gfx1201"], "cpus": 4,
                        "wire_shared": False, "weights_only": True,
                        "_why": "weights-only encode quanta only"}})
    records = {record["class"]: record
               for record in dispatch.validate_row_classes(spec, fleet=fleet())}
    assert records["default"]["isa"] == "sm121"
    assert records["rocm-encode"]["isa"] == "gfx1201"
    assert records["rocm-encode"]["weights_only"] is True
    assert records["rocm-encode"]["wire_shared"] is False


def test_a_container_class_is_refused_the_wrong_gpu_runtime(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    spec = base_spec(python="python3", container={
        "image": "prismaquant-example:pinned"}, classes={
        "rocm-encode": {"tags": ["gfx1201"], "wire_shared": False,
                        "weights_only": True,
                        "container": {"image": "prismaquant-rocm:pinned"}}})
    try:
        dispatch.validate_row_classes(spec, fleet=fleet())
    except dispatch.RowClassRefused as error:
        assert "attaches its GPU with 'rocm-wsl'" in str(error)
    else:
        raise AssertionError("an NVIDIA container was placed on a ROCm box")


def test_a_container_class_with_the_box_s_gpu_runtime_is_accepted():
    spec = base_spec(python="python3", container={
        "image": "prismaquant-example:pinned"}, classes={
        "rocm-encode": {"tags": ["gfx1201"], "wire_shared": False,
                        "weights_only": True,
                        "container": {"image": "prismaquant-rocm:pinned",
                                      "gpu_runtime": "rocm-wsl"}}})
    records = {record["class"]: record
               for record in dispatch.validate_row_classes(spec, fleet=fleet())}
    assert records["rocm-encode"]["containerized"] is True


# --------------------------------------------------------------------------
# Weights-only means weights-only
# --------------------------------------------------------------------------

def rocm_spec() -> dict:
    return base_spec(classes={
        "rocm-encode": {"python": ROCM_PYTHON, "tags": ["gfx1201"],
                        "wire_shared": False, "weights_only": True}})


def test_a_weights_only_class_may_not_run_the_campaign_module():
    try:
        build(rocm_spec(), row_class_name="rocm-encode")
    except dispatch.RowClassRefused as error:
        assert "prismaquant.tessera_campaign" in str(error)
    else:
        raise AssertionError("a weights-only class ran the campaign module")


def test_a_weights_only_row_may_not_carry_a_hessian_aware_flag():
    for flag in dispatch.HESSIAN_AWARE_FLAGS:
        try:
            dispatch._row(rocm_spec(), ["--model", "/models/example",
                                        flag, "/some/path"],
                          mem_gb=4, timeout_s=None,
                          module="prismaquant.example_encode",
                          row_class_name="rocm-encode")
        except dispatch.RowClassRefused as error:
            assert flag in str(error)
        else:
            raise AssertionError(f"a weights-only row carried {flag}")


def test_a_weights_only_row_may_not_carry_an_attached_hessian_aware_value():
    """``--flag=value`` is the same flag as ``--flag value``."""
    for flag in dispatch.HESSIAN_AWARE_FLAGS:
        try:
            dispatch._row(rocm_spec(), ["--model", "/models/example",
                                        f"{flag}=/some/path"],
                          mem_gb=4, timeout_s=None,
                          module="prismaquant.example_encode",
                          row_class_name="rocm-encode")
        except dispatch.RowClassRefused as error:
            assert flag in str(error)
        else:
            raise AssertionError(f"a weights-only row carried {flag}=")


def test_the_campaign_s_own_rows_are_unaffected_by_the_weights_only_gate():
    """The gate is a property of the class, not of the flag list."""
    row = dispatch._row(base_spec(), ["--model", "/models/example",
                                      "--calibration-census", "/c.json"],
                        mem_gb=34, timeout_s=None)
    assert "--calibration-census" in row["argv"]


# --------------------------------------------------------------------------
# The container launcher emits the ROCm flags only for a ROCm class
# --------------------------------------------------------------------------

def docker_argv(container: dict) -> list[str]:
    return container_tool.docker_command(
        {"container": container, "env": {}}, ["python3", "-c", "pass"],
        cwd="/home/rob/example", uid=1000, gid=1000,
        image_id="sha256:" + "0" * 64, environ={})


def test_an_undeclared_gpu_runtime_still_emits_gpus_all():
    argv = docker_argv({"image": "prismaquant-example:pinned"})
    assert argv[:5] == ["docker", "run", "--rm", "--gpus", "all"]
    assert "/dev/dxg" not in argv


def test_the_rocm_runtime_emits_the_wsl_device_and_library_mount():
    argv = docker_argv({"image": "prismaquant-rocm:pinned",
                        "gpu_runtime": "rocm-wsl"})
    assert argv[:5] == ["docker", "run", "--rm", "--device", "/dev/dxg"]
    assert ("type=bind,src=/usr/lib/wsl/lib,dst=/usr/lib/wsl/lib,readonly"
            in argv)
    assert "--gpus" not in argv


def test_a_cpu_only_launch_attaches_nothing_under_either_runtime():
    for runtime in ("nvidia", "rocm-wsl"):
        argv = container_tool.docker_command(
            {"container": {"image": "example:pinned", "gpu_runtime": runtime},
             "env": {}}, ["python3", "-c", "pass"], cwd="/home/rob/example",
            uid=1000, gid=1000, image_id="sha256:" + "0" * 64,
            with_gpu=False, environ={})
        assert "--gpus" not in argv and "/dev/dxg" not in argv


def test_an_unknown_gpu_runtime_is_refused():
    try:
        container_tool.validate_container(
            {"container": {"image": "example:pinned", "gpu_runtime": "rocm"}})
    except RuntimeError as error:
        assert "gpu_runtime" in str(error)
    else:
        raise AssertionError("an unknown GPU runtime was accepted")


def test_the_runtime_s_own_mount_may_not_be_declared_twice():
    try:
        container_tool.validate_container({"container": {
            "image": "example:pinned", "gpu_runtime": "rocm-wsl",
            "mounts": [{"source": "/usr/lib/wsl/lib",
                        "target": "/usr/lib/wsl/lib", "readonly": True}]}})
    except RuntimeError as error:
        assert "duplicate bind" in str(error)
    else:
        raise AssertionError("the WSL library mount was declared twice")


# --------------------------------------------------------------------------
# The tracked attestation table
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tag", ["dl380g10", "sparky", "sparklina"])
@pytest.mark.parametrize("class_name", ["default", "retired"])
def test_retired_interpreters_refuse_in_certified_mode(tag, class_name, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    spec = base_spec(python=SDK4_PYTHON, tags=[tag], classes={
        class_name: {"python": SDK3_PYTHON}})
    with pytest.raises(dispatch.RowClassRefused):
        dispatch.validate_row_classes(spec)
    shape = dispatch.load_fleet_interpreters()["tags"][tag]
    assert SDK3_PYTHON not in shape["interpreters"]


@pytest.mark.parametrize("tag", ["dl380g10", "sparky", "sparklina"])
@pytest.mark.parametrize("class_name", ["default", "history-probe"])
def test_retired_interpreters_continue_in_default_dev_mode(
        tag, class_name, monkeypatch, capsys):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    spec = base_spec(python=SDK4_PYTHON, tags=[tag], classes={
        class_name: {"python": SDK3_PYTHON}})
    records = dispatch.validate_row_classes(spec)
    record = next(record for record in records if record["class"] == class_name)
    assert record["python"] == SDK3_PYTHON
    assert "[DEV-MODE]" in capsys.readouterr().out
    assert SDK3_PYTHON not in dispatch.load_fleet_interpreters()["tags"][tag][
        "interpreters"]


def test_unattested_interpreter_continues_in_default_dev_mode(monkeypatch, capsys):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    python = "/home/rob/no/such/python"
    spec = base_spec(python=SDK4_PYTHON, classes={"probe": {"python": python}})
    records = dispatch.validate_row_classes(spec)
    assert records[0]["python"] == python
    assert "[DEV-MODE]" in capsys.readouterr().out


@pytest.mark.parametrize("tag", ["dl380g10", "gb10", "sparky", "sparklina"])
def test_active_sdk4_interpreters_validate_explicit_classes(tag):
    spec = base_spec(python=SDK4_PYTHON, tags=[tag], classes={"default": {}})
    records = dispatch.validate_row_classes(spec)
    assert records == [{"class": "default", "isa": dispatch.load_fleet_interpreters()[
        "tags"][tag]["isa"], "tags": [tag], "python": SDK4_PYTHON, "cpus": 4,
        "containerized": False, "wire_shared": True, "weights_only": False}]

def test_merge_train_interpreters_cover_vehicles_with_and_without_pr2216():
    table = dispatch.load_fleet_interpreters()["tags"]["dl380g10"][
        "interpreters"]
    assert table[SDK4_PYTHON]["attested_by"] == SDK4_ATTESTED_BY
    assert SDK4_PB_COMMIT in table[SDK4_PYTHON]["observed"]
    assert len(table[SDK5_PYTHON]["attested_by"]) == 64
    observed = table[SDK5_PYTHON]["observed"]
    assert SDK5_BASELINE in observed
    assert SDK5_PB_COMMIT in observed
    assert TESSERA_COMMIT in observed
    for python in (SDK4_PYTHON, SDK5_PYTHON):
        records = dispatch.validate_row_classes(base_spec(
            python=python, tags=["dl380g10"], classes={"default": {}}))
        assert records[0]["python"] == python


def test_active_sdk5_interpreter_validates_on_dl380g10():
    record = dispatch.load_fleet_interpreters()["tags"]["dl380g10"][
        "interpreters"][SDK5_PYTHON]
    assert len(record["attested_by"]) == 64
    records = dispatch.validate_row_classes(base_spec(
        python=SDK5_PYTHON, tags=["dl380g10"], classes={"default": {}}))
    assert records[0]["python"] == SDK5_PYTHON


@pytest.mark.parametrize("dev_env", [None, "0"])
@pytest.mark.parametrize("history, message", [
    (None, "retired_interpreters mapping"),
    ({SDK3_PYTHON: {}}, "names no PrismaBuild action key"),
    ({SDK4_PYTHON: {"attested_by": "0" * 64}}, "both active and retired"),
])
def test_invalid_retired_inventory_is_refused(
        tmp_path, history, message, dev_env, monkeypatch):
    if dev_env is None:
        monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    else:
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev_env)
    table = dispatch.load_fleet_interpreters()
    table["tags"]["dl380g10"]["retired_interpreters"] = history
    source = tmp_path / "fleet.json"
    source.write_text(json.dumps(table))
    with pytest.raises(dispatch.RowClassRefused, match=message):
        dispatch.load_fleet_interpreters(source)


def test_the_tracked_table_attests_the_rocm_interpreter_with_a_receipt():
    table = dispatch.load_fleet_interpreters()
    for tag in ("wsl-gpu", "gfx1201"):
        shape = table["tags"][tag]
        assert shape["isa"] == "gfx1201"
        assert shape["gpu_runtime"] == "rocm-wsl"
        record = shape["interpreters"][ROCM_PYTHON]
        assert len(record["attested_by"]) == 64
        records = dispatch.validate_row_classes(base_spec(
            python=ROCM_PYTHON, tags=[tag], classes={"default": {}}))
        assert records[0]["isa"] == "gfx1201"
        assert records[0]["python"] == ROCM_PYTHON
    for tag in ("gb10", "sparky", "sparklina", "gx10-6b77"):
        assert table["tags"][tag]["isa"] == "sm121"
        assert table["tags"][tag]["gpu_runtime"] == "nvidia"
