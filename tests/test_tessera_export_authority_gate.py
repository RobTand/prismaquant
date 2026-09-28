"""The reuse authority reaches Tessera's exporter only when its pin attests it.

Tessera#599 step 2 gave the serving exporter a ``--producer-authority``
option, and Tessera contract v40 publishes that as data
(``producer_interface.reuse_authority``). An exporter from before v40
refuses the option as an unknown argument, and the live campaign pins one
(``tessera-a3e83875``), so every PrismaQuant export argv asks the checkout's
own contract first (principle 14: read what the pinned runtime attests, never
assert it). Since #1587 the named driver is the supported package entry
point ``src/tessera/export_serving.py`` (tessera#687); a contract whose block
does not list it refuses rather than silently dropping the option.

These tests drive the real builders:

* ``dispatch_tessera_campaign submit-export --dry-run`` end to end, through
  ``cmd_submit_export`` and the real ``_pbrun_argv``: an old-pin contract
  yields the argv the submitter built before this gate existed, byte for byte,
  and a v40 contract inserts the option after the exporter, naming the adopter
  in the PrismaQuant tree the container runs;
* ``run-pipeline.sh``'s export call, which carries the option only through the
  same helper.
"""
from __future__ import annotations

import json
import shlex
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import dispatch_tessera_campaign as campaign  # noqa: E402
from prismaquant import tessera_export_lane as lane  # noqa: E402

#: The block Tessera contract v40 publishes (src/tessera/serving/runtime_contract.json).
V40_PRODUCER_INTERFACE = {
    "schema": "tessera.producer-interface.v1",
    "reuse_authority": {
        "option": "--producer-authority",
        "attribute": "PRODUCER_AUTHORITY",
        "protocol": "tessera.cached_unit.ReuseAuthority",
        "canonical_capture_attribute": "canonical_hessian_capture",
        "drivers": ["experiments/bf16_reach_roster.py",
                    "experiments/export_glm53_tessera.py",
                    "experiments/glm_routed_owner_inputs.py",
                    "src/tessera/export_serving.py",
                    "tools/glm_cpu_cached_pack_probe.py"]}}

#: A pre-v40 contract: it publishes no producer interface at all.
OLD_PIN_CONTRACT = {"contract_version": 39, "formats": []}
NEW_PIN_CONTRACT = {"contract_version": 40, "formats": [],
                    "producer_interface": V40_PRODUCER_INTERFACE}

CONTAINER_TESSERA = "/tessera-pin"
EXPORTER_MODULE = "tessera.export_serving"
INNER = ["python3", "-m", EXPORTER_MODULE, "/models/glm", "/out/exported",
         "--plan-json", "/out/plan.json", "--device", "cuda"]


def _checkout(root: Path, contract: dict) -> Path:
    exporter = root / "src" / "tessera" / "export_serving.py"
    exporter.parent.mkdir(parents=True)
    exporter.write_text("# exporter\n")
    contract_path = root / "src" / "tessera" / "serving" / "runtime_contract.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(json.dumps(contract))
    return root


class _StubProducer(types.SimpleNamespace):
    """The manifest producer, reduced to what ``submit-export`` reads back."""

    def __init__(self):
        super().__init__(argv=None)

    def deterministic_entry_provenance(self, entry_point, **kwargs):
        return {"entry_point": entry_point, **kwargs}

    def build_export_manifest(self, plan, *, argv=None, **kwargs):
        self.argv = argv
        return {"entry_count": 0, "total_bytes": 0,
                "annotations": {"counts": {}, "bytes": {}, "phases": []}}

    def check_manifest_bytes(self, blob, *, where):
        return None


def _submit(tmp_path: Path, monkeypatch, contract: dict, *, gate: bool = True,
            inner=INNER) -> tuple[list, list]:
    """Run ``submit-export --dry-run``; return (pbrun argv, manifest argv)."""
    run = tmp_path / ("gated" if gate else "ungated")
    checkout = _checkout(run / "tessera", contract)
    workspace = run / "workspace"
    (workspace / "prismaquant").mkdir(parents=True)
    (workspace / "prismaquant" / "tessera_reuse_authority.py").write_text("# adopter\n")
    campaign_plan = run / "campaign" / "plan.json"
    campaign_plan.parent.mkdir(parents=True)
    campaign_plan.write_text("{}")
    plan = run / "joint-plan.json"
    plan.write_text(json.dumps({"inputs": {"campaign_plan": {"path": str(campaign_plan)}}}))
    spec = run / "spec.json"
    spec.write_text(json.dumps({"container": {"image": "example/tessera@sha256:" + "0" * 64,
                                              "mounts": [{"source": str(checkout),
                                                          "target": CONTAINER_TESSERA}]}}))
    for name in ("assignment.json", "cost.json"):
        (run / name).write_text("{}")
    producer = _StubProducer()
    monkeypatch.setattr(campaign, "_manifest_producer", lambda: producer)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    monkeypatch.chdir(workspace)
    if not gate:
        # The submitter as it was before this gate: the caller's inner, untouched.
        monkeypatch.setattr(campaign, "export_inner_with_authority",
                            lambda inner, spec, *, cwd: list(inner))
    seen = []
    real = campaign._pbrun_argv
    monkeypatch.setattr(campaign, "_pbrun_argv",
                        lambda *a, **k: seen.append(real(*a, **k)) or seen[-1])
    code = campaign.main([
        "submit-export", "--plan", str(plan),
        "--assignment", str(run / "assignment.json"),
        "--allocation-cost", str(run / "cost.json"),
        "--spec", str(spec), "--demand", "gpu=1,mem_gb=104",
        "--pbrun", "/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py",
        "--timeout-s", "7200", "--manifest-dir", str(run / "manifests"),
        "--dry-run", "--", *inner])
    assert code == 0 and len(seen) == 1
    return seen[0], producer.argv


def _relative(argv: list, root: Path) -> bytes:
    """The argv as bytes, with the run's tmp root folded out so two runs compare."""
    return "\0".join(argv).replace(str(root), "<run>").encode()


def test_an_old_pin_submits_the_argv_it_submitted_before_the_gate(tmp_path, monkeypatch):
    gated, gated_manifest = _submit(tmp_path, monkeypatch, OLD_PIN_CONTRACT)
    before, before_manifest = _submit(tmp_path, monkeypatch, OLD_PIN_CONTRACT, gate=False)
    assert "--producer-authority" not in gated
    assert _relative(gated, tmp_path / "gated") == _relative(before, tmp_path / "ungated")
    assert gated_manifest == before_manifest == INNER
    assert gated[-len(INNER):] == INNER
    print("old-pin argv:", shlex.join(gated))


def test_a_v40_pin_gets_the_authority_after_the_exporter(tmp_path, monkeypatch):
    gated, manifest_argv = _submit(tmp_path, monkeypatch, NEW_PIN_CONTRACT)
    expected = [INNER[0], "-m", EXPORTER_MODULE, "--producer-authority",
                "/workspace/prismaquant/tessera_reuse_authority.py", *INNER[3:]]
    assert gated[-len(expected):] == expected
    assert manifest_argv == expected
    before, _ = _submit(tmp_path, monkeypatch, NEW_PIN_CONTRACT, gate=False)
    # The two inserted tokens are the whole difference from the ungated argv.
    index = gated.index("--producer-authority")
    assert _relative(gated[:index] + gated[index + 2:], tmp_path / "gated") == \
        _relative(before, tmp_path / "ungated")


def test_a_caller_that_already_passes_the_option_is_left_alone(tmp_path, monkeypatch):
    inner = [*INNER, "--producer-authority", "/elsewhere/authority.py"]
    gated, manifest_argv = _submit(tmp_path, monkeypatch, NEW_PIN_CONTRACT, inner=inner)
    assert manifest_argv == inner and gated.count("--producer-authority") == 1


def test_a_v40_pin_with_no_adopter_in_the_run_tree_refuses(tmp_path):
    checkout = _checkout(tmp_path / "tessera", NEW_PIN_CONTRACT)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = {"container": {"mounts": [{"source": str(checkout), "target": CONTAINER_TESSERA}]}}
    with pytest.raises(RuntimeError, match="has no prismaquant/tessera_reuse_authority.py"):
        campaign.export_inner_with_authority(INNER, spec, cwd=str(workspace))


def test_an_exporter_behind_no_mount_refuses(tmp_path):
    with pytest.raises(RuntimeError, match="not a file behind any mount"):
        campaign.export_inner_with_authority(INNER, {"container": {"mounts": []}},
                                             cwd=str(tmp_path))


def test_the_helper_reads_the_contract_not_a_constant(tmp_path):
    old = _checkout(tmp_path / "old", OLD_PIN_CONTRACT)
    new = _checkout(tmp_path / "new", NEW_PIN_CONTRACT)
    assert lane.producer_authority_argv(old, "/a.py") == []
    assert lane.producer_authority_argv(new, "/a.py") == ["--producer-authority", "/a.py"]
    unlisted = json.loads(json.dumps(NEW_PIN_CONTRACT))
    unlisted["producer_interface"]["reuse_authority"]["drivers"].remove(lane.EXPORTER_DRIVER)
    # #1587 (tessera#691 review item 1): a block that does not list the
    # driver refuses -- silently dropping the option hands the exporter an
    # argv whose cached-unit bundle refuses downstream instead of here.
    with pytest.raises(lane.ProducerInterfaceError, match="does not include"):
        lane.producer_authority_argv(_checkout(tmp_path / "unlisted", unlisted), "/a.py")
    renamed = json.loads(json.dumps(NEW_PIN_CONTRACT))
    renamed["producer_interface"]["reuse_authority"]["option"] = "--authority"
    with pytest.raises(lane.ProducerInterfaceError, match="does not publish"):
        lane.producer_authority_argv(_checkout(tmp_path / "renamed", renamed), "/a.py")
    with pytest.raises(lane.ProducerInterfaceError, match="packages no"):
        lane.producer_authority_argv(tmp_path / "missing", "/a.py")


def _both_drivers_contract():
    """The v44 shape: the package exporter beside the legacy shim."""
    contract = json.loads(json.dumps(NEW_PIN_CONTRACT))
    drivers = contract["producer_interface"]["reuse_authority"]["drivers"]
    assert campaign.LEGACY_EXPORTER_SCRIPT not in drivers
    return contract


def test_a_legacy_shim_inner_keeps_the_authority(tmp_path):
    """No silent drop: the shim is still a listed driver at v44, so an
    inner naming ``experiments/export_tessera_serving.py`` gets the option
    (checked against the spelling it uses), not a downstream
    MISSING_REUSE_AUTHORITY."""
    contract = _both_drivers_contract()
    contract["producer_interface"]["reuse_authority"]["drivers"].append(
        campaign.LEGACY_EXPORTER_SCRIPT)
    checkout = _checkout(tmp_path / "tessera", contract)
    shim = checkout / campaign.LEGACY_EXPORTER_SCRIPT
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text("# shim\n")
    inner = ["python3", f"{CONTAINER_TESSERA}/{campaign.LEGACY_EXPORTER_SCRIPT}",
             "/models/glm", "/out/exported", "--plan-json", "/out/plan.json"]
    spec = {"container": {"mounts": [{"source": str(checkout),
                                          "target": CONTAINER_TESSERA}]}}
    # No declared mount holds a PrismaQuant package, so the sealed checkout
    # (cwd) is the tree to run: the adopter lives beside the test's cwd.
    adopter = tmp_path / "prismaquant" / "tessera_reuse_authority.py"
    adopter.parent.mkdir(parents=True)
    adopter.write_text("# adopter\n")
    got = campaign.export_inner_with_authority(inner, spec, cwd=str(tmp_path))
    assert got == [*inner[:2], "--producer-authority",
                   "/workspace/prismaquant/tessera_reuse_authority.py",
                   *inner[2:]]


def test_a_legacy_shim_inner_refuses_when_only_the_new_driver_is_listed(tmp_path):
    """The check names the spelling the inner uses: a contract attesting
    only the package exporter refuses the shim up front."""
    checkout = _checkout(tmp_path / "tessera", _both_drivers_contract())
    shim = checkout / campaign.LEGACY_EXPORTER_SCRIPT
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text("# shim\n")
    inner = ["python3", f"{CONTAINER_TESSERA}/{campaign.LEGACY_EXPORTER_SCRIPT}"]
    spec = {"container": {"mounts": [{"source": str(checkout),
                                          "target": CONTAINER_TESSERA}]}}
    with pytest.raises(lane.ProducerInterfaceError, match="does not include"):
        campaign.export_inner_with_authority(inner, spec, cwd=str(tmp_path))


def test_module_checkout_honours_a_relative_mount_source(tmp_path):
    """``_module_checkout`` takes ``cwd``: a relative mount source resolves
    against it instead of the process cwd."""
    checkout = tmp_path / "rel" / "tessera"
    exporter = checkout / "src" / "tessera" / "export_serving.py"
    exporter.parent.mkdir(parents=True)
    exporter.write_text("# exporter\n")
    mounts = [{"source": "rel/tessera", "target": CONTAINER_TESSERA}]
    assert campaign._module_checkout(mounts, cwd=str(tmp_path)) == checkout


def test_run_pipeline_passes_the_authority_only_through_the_helper():
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    call = script[script.index("python3 -m tessera.export_serving"):]
    call = call[:call.index("2>&1 | tee")]
    assert "--producer-authority" not in call
    assert '"${TESSERA_AUTHORITY_ARGS[@]}"' in call
    gate = script[:script.index("python3 -m tessera.export_serving")]
    gate = gate[gate.rindex("TESSERA_AUTHORITY_LINES=$("):]
    assert '"${PIPELINE_SCRIPT_DIR}/tessera_producer_interface.py"' in gate
    assert '"${TESSERA_REPO%/}" "${PIPELINE_SCRIPT_DIR}/tessera_reuse_authority.py"' in gate


def test_the_producer_interface_reader_stands_alone():
    """run-pipeline.sh runs it by path, so it imports only the standard library."""
    import ast
    from prismaquant import tessera_producer_interface as interface
    path = ROOT / "prismaquant" / "tessera_producer_interface.py"
    imported = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import cannot run by path"
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "collections", "json", "pathlib", "sys", "typing"}
    for name in ("EXPORTER_DRIVER", "PRODUCER_AUTHORITY_OPTION", "ProducerInterfaceError",
                 "advertises_producer_authority", "checkout_contract_path",
                 "producer_authority_argv"):
        assert getattr(lane, name) is getattr(interface, name), name


def _run_pipeline_authority_gate() -> str:
    """The ``run-pipeline.sh`` lines that decide the exporter's authority argv."""
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    gate = script[:script.index("python3 -m tessera.export_serving")]
    gate = gate[gate.rindex("  if ! TESSERA_AUTHORITY_LINES=$("):]
    return gate[:gate.index("\n  fi\n")] + "\n  fi\n"


@pytest.mark.parametrize("contract, expected", [
    (OLD_PIN_CONTRACT, []),
    (NEW_PIN_CONTRACT, ["--producer-authority",
                        str(ROOT / "prismaquant" / "tessera_reuse_authority.py")]),
])
def test_run_pipeline_reads_the_contract_without_importing_prismaquant(
        tmp_path, contract, expected):
    """The run-pipeline gate reads one JSON block, so it imports no torch.

    The gate runs once per Tessera export, beside the preflight. Importing the
    ``prismaquant`` package to read the contract costs the torch and
    transformers imports every time, which on a small runner pushed the
    tessera-arm tests past their 30 s bound. ``-X importtime`` names every
    module the gate's interpreter imports.
    """
    import subprocess
    checkout = _checkout(tmp_path / "tessera", contract)
    driver = (
        'set -euo pipefail\n'
        'python3() { "$PYTEST_PYTHON" -X importtime "$@"; }\n'
        + _run_pipeline_authority_gate()
        + 'printf "%s" "$TESSERA_AUTHORITY_LINES"\n')
    import os
    env = {**os.environ, "PYTEST_PYTHON": sys.executable, "PYTHONPATH": str(ROOT),
           "PIPELINE_SCRIPT_DIR": str(ROOT / "prismaquant"),
           "TESSERA_REPO": str(checkout)}
    done = subprocess.run(
        ["bash", "-c", driver], cwd=ROOT, capture_output=True, text=True,
        timeout=120, check=False, env=env)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == expected
    imported = {line.rsplit("|", 1)[-1].strip().split(".")[0]
                for line in done.stderr.splitlines() if line.startswith("import time:")}
    assert not imported & {"prismaquant", "torch", "transformers", "numpy"}, sorted(imported)


def test_the_packaged_pin_attests_the_new_driver(tmp_path):
    """Flipped by the PQ #1616 pin move to tessera#691: the admitted v44
    contract attests ``src/tessera/export_serving.py`` beside the shim, so
    the reader passes the authority through instead of refusing.  Before the
    bump this test asserted the refusal (the v42 contract attested the old
    shim path only); the refusal path itself is still pinned by
    ``test_the_helper_reads_the_contract_not_a_constant`` on fixture
    contracts."""
    from prismaquant import tessera_render as tr
    from importlib.resources import as_file
    with as_file(tr.tessera_serving_contract_path()) as path:
        contract = json.loads(Path(path).read_text())
    assert lane.advertises_producer_authority(contract)
    checkout = _checkout(tmp_path / "tessera", contract)
    assert lane.producer_authority_argv(checkout, "/a.py") == [
        "--producer-authority", "/a.py"]
