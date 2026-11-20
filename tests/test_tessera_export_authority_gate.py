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

These tests check the producer contract and the export call in ``run-pipeline.sh``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

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


def test_producer_cli_refuses_invalid_contract_json(tmp_path, capsys):
    from prismaquant import tessera_producer_interface

    checkout = _checkout(tmp_path / "tessera", NEW_PIN_CONTRACT)
    (checkout / "src/tessera/serving/runtime_contract.json").write_text("{")
    assert tessera_producer_interface.main([str(checkout), "/a.py"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Expecting property name" in captured.err


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
