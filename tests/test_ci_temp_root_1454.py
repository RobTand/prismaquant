"""Hosted CI's temporary fixtures must use its owned, non-overlay workspace."""
from pathlib import Path

import pytest
import yaml

from tests.test_container_cache_charge_1091 import _layout
from tests.test_container_cache_roots_1072 import _docker_env

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["suite", "scope"])
def test_ci_temp_root_can_hold_an_explicit_charged_cache(tmp_path, mode):
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["tests"]["steps"]
    runner_temp = tmp_path / "runner-temp"
    if mode == "suite":
        step = next(s for s in steps if s.get("name") == "Run tests")
        configured = step.get("env", {}).get("TMPDIR", "/tmp")
        temporary = Path(configured.replace("${{ runner.temp }}", str(runner_temp)))
    else:
        step = next(s for s in steps if s.get("name") == "Run cgroup-reading tests in a scope of their own")
        temporary = runner_temp if 'TMPDIR="$RUNNER_TEMP"' in step["run"] else Path("/tmp")
    assert temporary.is_absolute() and temporary.is_relative_to(runner_temp)
    spec, env = _layout(temporary / "case")
    inside = _docker_env(spec, env)
    assert inside["HF_HOME"] == env["PRISMAQUANT_CONTAINER_CACHE_ROOT"] + "/hf"
    assert not runner_temp.exists()
