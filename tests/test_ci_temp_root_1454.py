"""CI test fixtures must use the temporary workspace owned by their executor."""
from pathlib import Path

import pytest
import yaml

from tests.test_container_cache_charge_1091 import _layout
from tests.test_container_cache_roots_1072 import _docker_env
from tools.bazel.runtime import gate_environment

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["hosted", "bazel"])
def test_ci_temp_root_can_hold_an_explicit_charged_cache(tmp_path, mode):
    runner_temp = tmp_path / "runner temp"
    if mode == "hosted":
        workflow = yaml.safe_load((ROOT / ".github/workflows/bazel-hosted.yml").read_text())
        step = workflow["jobs"]["tests"]["steps"][0]
        assert step["name"] == "Configure owned temporary space"
        assert step["run"] == 'printf \'TMPDIR=%s\\n\' "$RUNNER_TEMP" >> "$GITHUB_ENV"\n'
        temporary = runner_temp
    else:
        environment = gate_environment({"TMPDIR": "/tmp", "HOME": "/tmp"}, ROOT, runner_temp)
        temporary = Path(environment["TMPDIR"])
        assert environment["HOME"] == str(runner_temp)
    assert temporary.is_absolute() and temporary.is_relative_to(runner_temp)
    spec, env = _layout(temporary / "case")
    inside = _docker_env(spec, env)
    assert inside["HF_HOME"] == env["PRISMAQUANT_CONTAINER_CACHE_ROOT"] + "/hf"
    assert not runner_temp.exists()
